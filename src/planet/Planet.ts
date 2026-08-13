import {
  BufferGeometry,
  Color,
  Frustum,
  Group,
  Mesh,
  MeshBasicMaterial,
  Points,
  PointsMaterial,
  Sphere,
  Vector3,
  Matrix4,
  Quaternion,
} from 'three'
import { MeshBasicNodeMaterial, MeshStandardNodeMaterial } from 'three/webgpu'
import {
  attribute,
  positionWorld,
  uniform,
  vec3,
  saturate,
  mix,
  clamp,
  dot,
  normalize,
  max,
  sqrt,
  cross,
  asin,
  atan2,
  sin,
  cos,
  fract,
  step,
} from 'three/tsl'
import { Tectonics, TectonicQuery } from './tectonics'
import { Climate, ClimateSample } from './climate'
import { Erosion } from './erosion'
import { Subsurface } from './subsurface'
import { makeTerrainSampler } from './terrainSampler'
import { makeCaveSampler, CaveField } from './caveField'
import { texelToDir, texelIndex, neighborTexel } from './cubemap'
import { InteriorParams, DerivedInterior, DEFAULT_INTERIOR, deriveInterior } from './interior'
import { buildChunkGeometry, arraysToGeometry, ChunkMeshArrays, ChunkMeshData } from './ChunkMesher'
import { MeshWorkerPool } from './MeshWorkerPool'
import { QuadtreeNode } from './QuadtreeNode'
import { TectonicsDebug } from './TectonicsDebug'
import { WindDebug } from './WindDebug'
import { WindFlow } from './WindFlow'
import { VolumetricClouds } from './VolumetricClouds'
import { Atmosphere } from './Atmosphere'
import { PlanetGizmos } from './PlanetGizmos'
import { RADIUS as WORLD_RADIUS, HEIGHT_SCALE_REF, deriveErosionRes } from './worldConstants'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PlanetOptions {
  seed: number
  radius: number
  heightScale: number
  resolution?: number
  maxDepth?: number
  splitFactor?: number
  plateCount?: number
  /** Target triangle pixel size for the SSE LOD metric (default 2.5). */
  targetTriPx?: number
  /** Mean surface temperature, °C-ish, for the climate model (default 15 = Earth).
   *  Ignored when interior params are active — use interior.surfaceTemp instead. */
  baseTemp?: number
  /** Atmospheric thickness 0..1 — thick shrinks gradients, thin makes extremes (default 0.6). */
  atmosphere?: number
  /** Rotation period in seconds — drives the circulation band count (default 600 = Earth-like → 3 cells). */
  rotationPeriodS?: number
  /** Axial tilt in degrees — climate uses it for the > 54° insolation inversion (default 23.4). */
  axialTiltDeg?: number
  /** Number of intraplate hotspot (mantle plume) volcanoes (default 6). */
  hotspotCount?: number
  /** Global intensity multiplier for hotspots (default 1). */
  hotspotIntensity?: number
  /** Interior physics parameters (new 5-root model). Merged over DEFAULT_INTERIOR.
   *  When present and overrideInterior is false (the default), these drive all
   *  derived fields (plateCount/arcDensity/hotspotCount/surfaceTemp/atmosphere/etc.)
   *  via deriveInterior(). */
  interior?: Partial<InteriorParams>
  /** When true, ignore the derived interior and use the manual mid-layer knob values
   *  (manualHeat/manualSurfaceTemp/manualAtmosphere) directly as internalHeat/surfaceTemp/
   *  atmosphere overrides to deriveInterior(). Defaults to false. */
  overrideInterior?: boolean
}

interface Stats {
  leaves: number
  cached: number
  minLevel: number
  maxLevel: number
  pendingBuilds: number
  lastBuildMs: number
  plates: number
  volcanoes: number
  hotspots: number
  bandCount: number
}

// ---------------------------------------------------------------------------
// LRU cache (capacity-capped, evicts oldest on overflow)
// ---------------------------------------------------------------------------

class LruCache<V extends { dispose(): void }> {
  private readonly map = new Map<string, V>()
  private readonly capacity: number

  constructor(capacity: number) {
    this.capacity = capacity
  }

  get(key: string): V | undefined {
    const val = this.map.get(key)
    if (val !== undefined) {
      // Refresh to "most recently used" by re-inserting
      this.map.delete(key)
      this.map.set(key, val)
    }
    return val
  }

  set(key: string, val: V): void {
    if (this.map.has(key)) {
      this.map.delete(key)
    }
    this.map.set(key, val)
    if (this.map.size > this.capacity) {
      // Evict LRU (first entry in insertion order)
      const firstKey = this.map.keys().next().value!
      const evicted = this.map.get(firstKey)!
      this.map.delete(firstKey)
      evicted.dispose()
    }
  }

  /** Return value to cache without promoting to MRU (geometry we're not actively displaying). */
  return(key: string, val: V): void {
    this.set(key, val)
  }

  delete(key: string): V | undefined {
    const val = this.map.get(key)
    this.map.delete(key)
    return val
  }

  get size(): number {
    return this.map.size
  }

  disposeAll(): void {
    for (const val of this.map.values()) val.dispose()
    this.map.clear()
  }
}

// ---------------------------------------------------------------------------
// Geometry cache wrapper (ChunkMeshData has a .geometry field)
// ---------------------------------------------------------------------------

class CachedMeshData {
  constructor(public readonly data: ChunkMeshData) {}
  dispose(): void {
    this.data.geometry.dispose()
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BUILD_BUDGET_PER_FRAME = 40 // chunks per frame — balanced between responsiveness and frame budget
const BUILD_BUDGET_MS = 16        // ms/frame ceiling on meshing
const LRU_CAPACITY = 8192         // res=32: ~38 KB/chunk × 8192 ≈ 311 MB geometry cache ceiling
const HYSTERESIS = 0.15 // 15% — SSE merge fires at threshold * (1 + HYSTERESIS)
const EPS_DIST = 0.1    // minimum camera-to-node distance (prevents div-by-zero at contact)
const DEBUG_TIMING = true         // set true to log bake timings (erosion etc.) to the console

// ---------------------------------------------------------------------------
// Planet
// ---------------------------------------------------------------------------

export class Planet extends Group {
  readonly seed: number

  private readonly radius: number
  private readonly heightScale: number
  private readonly resolution: number

  /** Maximum quadtree depth. Live-tuneable via setMaxDepth(). */
  maxDepth: number

  /** Legacy split-distance factor (kept for merge threshold computation only). */
  private readonly splitFactor: number

  /**
   * Target triangle pixel size for the SSE split metric.
   * Split when a chunk's projected edge spans more than (resolution * targetTriPx) pixels.
   * Live-tuneable. Default 2.5 → splitThresholdPx ≈ 80 px for resolution=32.
   */
  targetTriPx: number

  /**
   * Camera vertical FOV in radians — updated every update() call.
   * Initialised to 60° (typical default) so getSurfaceRadiusAt is usable before first update().
   */
  private _vFovRadians: number

  /**
   * Drawing-buffer height in pixels — updated every update() call.
   * Initialised to 1080 as a safe fallback.
   */
  private _screenHeightPx: number

  private plateCount: number
  private arcDensity = 1.0
  private hotspotCount: number
  private hotspotIntensity = 1

  // Interior physics parameters — merged over DEFAULT_INTERIOR at construction.
  // When overrideInterior is false, deriveInterior(this.interior) drives the tectonic knobs.
  private interior: InteriorParams
  private overrideInterior: boolean
  /** Cached result of the most recent deriveInterior() call — populated in buildHeightFn. */
  private _derived!: DerivedInterior

  // Climate knobs — stored, applied on the next regenerate() (which rebuilds the Climate).
  // baseTemp and atmosphere are sourced from d.surfaceTemp / d.atmosphere in derived mode;
  // these fields are kept as fallbacks for legacy callers and the manual override path.
  private baseTemp: number
  private atmosphere: number
  private rotationPeriodS: number
  /** Axial tilt in degrees — live, settable via setAxialTilt(). */
  private axialTiltDeg: number

  // Manual mid-layer overrides — active when overrideInterior is true.
  // These pin specific DerivedInterior inputs (passed as overrides to deriveInterior).
  private manualHeat: number        // 0..1, maps to internalHeat override
  private manualSurfaceTemp: number // °C, maps to surfaceTemp override
  private manualAtmosphere: number  // 0..1, maps to atmosphere override

  // Erosion bake parameters — applied on the next buildHeightFn call (triggered by regenerate).
  /** Cubemap resolution for the erosion bake (128 | 256 | 512). Default 256. */
  private erosionRes: 128 | 256 | 512
  /** Multiplier on the incision rate K0. 1.0 = default carve, 0 = no erosion, 3 = heavy. */
  private erosionStrength: number
  /** Multiplier on the deposition coefficient G. 1.0 = default, 0 = no deposition, 3 = heavy fans. */
  private erosionDeposition: number
  // B (iterated uplift+erode) parameters
  /** Number of B loop iterations (default 30). */
  private bSteps: number = 30
  /** Uplift magnitude per B step in metres.
   * Authored at HEIGHT_SCALE_REF=1200 as 40 m/step; scales proportionally with heightScale
   * so the normalized (/ B_HEIGHT_SCALE) uplift contribution is invariant under uniform scale.
   * Initialized lazily in constructor once heightScale is known. */
  private bUpliftRate: number
  private heightFn!: (dir: Vector3, level: number) => number
  /** Erosion instance — set by buildHeightFn, passed to sync-fallback meshing. */
  private _erosion: Erosion | null = null
  private _subsurface: Subsurface | null = null

  /** Cave field — rebuilt each buildHeightFn() call (main-thread only, not shipped to workers). */
  private _caveField: CaveField | null = null

  /**
   * Waterline elevation — normalized units matching heightFn output (~[-1,1]).
   * Computed once per buildHeightFn via Fibonacci-sphere hypsometry and the
   * derived oceanCoverage from interior.ts. Shipped to workers via InitMsg.
   */
  private _seaLevel = 0

  /** Tectonic simulation — rebuilt on regenerate(). */
  tectonics!: Tectonics

  /** Climate fields (temperature + moisture) — rebuilt on regenerate() after heightFn + tectonics. */
  private climateSim!: Climate
  /** Reused ClimateSample scratch for climateFn — safe because meshing is serial. */
  private readonly _climateScratch: ClimateSample = { temperature: 0, moisture: 0 }
  /** Bound climate sampler passed to the mesher — set by buildHeightFn. */
  private climateFn!: (dir: Vector3, height: number) => ClimateSample

  /** Debug overlay — rebuilt on regenerate(), always a child of this Group. */
  private tectonicsDebug!: TectonicsDebug

  /** Pole-axis + equator gizmos — built once in constructor, never rebuilt. */
  private gizmos!: PlanetGizmos

  /** Six root nodes, one per cube face. */
  private roots!: QuadtreeNode[]

  /** Currently visible leaf meshes keyed by node key. */
  private readonly visibleMeshes = new Map<string, Mesh>()

  /** Points overlays for visible chunks keyed by node key (only populated when _showVertices is on). */
  private readonly visiblePoints = new Map<string, Points>()

  /** Geometry cache (returns geometry on mesh removal, evicts on overflow). */
  private readonly geoCache = new LruCache<CachedMeshData>(LRU_CAPACITY)

  /**
   * Build queue for pending geometries. Each entry records:
   *  - key: node key
   *  - node: the QuadtreeNode to build
   *  - priority: projected-pixel size at enqueue time — higher = closer to camera, builds first
   *  - waitingFor: "split-ready" — parent key this belongs to (or null if it's
   *    a standalone leaf or a parent being built for merge)
   */
  private readonly buildQueue: Array<{ key: string; node: QuadtreeNode; priority: number }> = []
  private readonly buildQueueSet = new Set<string>()

  /**
   * Split-pending set: nodes that *want* to split but are waiting for all 4
   * child geometries to be ready. Keyed by parent key, value is the parent node.
   * The parent mesh stays visible until all 4 children are cached.
   */
  private readonly splitPending = new Map<string, QuadtreeNode>()

  /**
   * Merge-pending set: nodes whose children are being removed but the parent
   * geometry isn't cached yet. The children stay visible until the parent is
   * cached, then swapped in one frame.
   * Key = parent key, value = { parent node, child keys }.
   */
  private readonly mergePending = new Map<
    string,
    { node: QuadtreeNode; childKeys: string[] }
  >()

  // Materials
  /** Shared lit material for normal view — one instance, never disposed per-mesh. */
  private readonly normalMaterial: MeshStandardNodeMaterial
  private readonly debugMaterials: MeshBasicMaterial[]
  /** Plate-color node material (flat, unlit, reads 'plateColor' attribute). Created once. */
  private readonly plateColorMaterial: MeshBasicNodeMaterial
  /** Heightmap node material: grayscale by elevation (unlit), derived in-shader from vertex world-distance. */
  private readonly heightmapMaterial: MeshBasicNodeMaterial
  /** Pure unlit wireframe — white edges only, no fill, no lighting. */
  private readonly wireMaterial: MeshBasicMaterial
  /** Shared points material for vertex overlay — yellow constant-size screen-space dots. */
  private readonly pointsMaterial: PointsMaterial
  private debugColorsActive = false
  private tectonicsViewActive = false
  private heightmapViewActive = false
  private wireframeActive = false
  private _showVertices = false

  // Climate debug view
  /** Climate node material — built once in constructor, disposed in dispose(). */
  private readonly climateMaterial: MeshBasicNodeMaterial
  private climateViewActive = false
  private _climateField: 'temperature' | 'moisture' | 'insolation' = 'temperature'

  // Wind debug view
  /** Wind node material — built once in constructor, disposed in dispose(). */
  private readonly windMaterial: MeshBasicNodeMaterial
  private windViewActive = false
  private _windField: 'flow' | 'speed' | 'direction' = 'flow'

  // Materials debug view (rock hardness scalar)
  /** Hardness node material — built once in constructor, disposed in dispose(). */
  private readonly hardnessMaterial: MeshBasicNodeMaterial
  private materialsViewActive = false

  /** Wetness/aquifer node material — built once in constructor, disposed in dispose(). */
  private readonly wetnessMaterial: MeshBasicNodeMaterial
  private wetnessViewActive = false

  // Wind bake parameters — stored on Planet so regenerate() can rebuild Climate with them.
  // These affect only the baked wind field (not the sampler), so fromBaked stays unchanged.
  private _windSwirlStrength  = 0.6   // blend weight of vortex swirls vs zonal base (default 0.6)
  private _windNHigh          = 4     // HIGH-pressure centres per hemisphere (default 4)
  private _windNLow           = 3     // LOW-pressure centres per hemisphere (default 3)
  private _windCrossIsobarMax = 0.4   // max cross-isobar tilt angle in radians (default 0.4)
  private _windSigmaBase      = 0.18  // base Gaussian half-width in radians for pressure centres (default 0.18)
  private _windLatSpread      = 0.17  // ± latitude spread for pressure-centre placement (default 0.17)
  private _windRetrogradeBake = 1     // +1 prograde / -1 retrograde for baked field (default +1)
  private _windEquatorTaper   = 0.20  // equatorial taper half-width in radians (default 0.20)

  /** WindDebug overlay — built after climateSim exists, always a child of this Group. */
  private windDebug!: WindDebug

  /** WindFlow overlay — advected particle streaklines, coexists with arrows. */
  private windFlow!: WindFlow

  /** VolumetricClouds overlay — raymarched cloud layer, independent toggle. */
  cloudShell!: VolumetricClouds

  /** Atmosphere shell — analytic single-scatter glow, always visible. */
  atmosphereShell!: Atmosphere

  /** Whether wind arrows are shown when in wind view. */
  private _showWindArrows = true
  /** Whether wind flow streaklines are shown when in wind view. */
  private _showWindFlow = true

  /** Arrow density for WindDebug — live settable. */
  private _windArrowDensity = 1500

  // Wind material uniforms (all live-updatable via setters; _uPoleAxis is shared with climate)
  private readonly _uWindBands     = uniform(3)      // N band pairs, initialised from deriveBandCount()
  private readonly _uWindStrength  = uniform(1)      // global speed multiplier
  private readonly _uTurbulence    = uniform(0)      // 0 = off, >0 = perturbation strength
  private readonly _uRetrograde    = uniform(1)      // +1 prograde / -1 retrograde
  private readonly _uWindTime      = uniform(0)      // per-frame animation clock
  private readonly _uWindField     = uniform(0)      // 0=flow 1=speed 2=direction

  // Climate material uniforms (all world-frame; live-updatable via setters)
  private readonly _uSunDir   = uniform(new Vector3(0, 1, 0))   // normalised world sun direction
  private readonly _uPoleAxis = uniform(new Vector3(0, 1, 0))   // planet pole in world space
  private readonly _uRedistribution = uniform(0.5)              // 0=day/night 1=latitude field
  private readonly _uBaseTemp  = uniform(15)                     // °C
  private readonly _uGreenhouse = uniform(0)                     // °C offset
  private readonly _uTempRange  = uniform(55)                    // °C equator→pole spread
  private readonly _uLapseRate  = uniform(50)                    // °C over full height

  // Cached inverse world matrix + quaternion for rotation-safe transforms
  // Re-computed each update() call.
  private readonly _invWorldMatrix = new Matrix4()
  private readonly _invWorldQuat = new Quaternion()
  // Scratch vectors — preallocated, zero allocations in hot paths
  private readonly _camLocalScratch = new Vector3()
  private readonly _dirLocalScratch = new Vector3()
  // Camera direction in the planet-local frame, for WindFlow's visible-cap LOD.
  private readonly _windCamDir = new Vector3()
  // SSE scratch: holds (camPos - nodeCenter) for nearest-point distance computation.
  private readonly _sseScratch = new Vector3()
  // Frustum culling — built once per frame in update(), used in collect()
  private readonly _localFrustum = new Frustum()
  private readonly _frustumMatrix = new Matrix4()
  private readonly _frustumSphere = new Sphere()
  private _frustumActive = false

  // Stats
  private lastBuildMs = 0

  // Freeze flag
  private frozen = false

  // Worker pool for off-main-thread chunk meshing
  private pool: MeshWorkerPool | null = null
  private poolReady = false
  private poolGeneration = 0

  // Diagnostic overlay
  private _diagEnabled = false
  private _diagEl: HTMLDivElement | null = null
  private _diagFrame = 0

  // ---------------------------------------------------------------------------

  constructor(opts: PlanetOptions) {
    super()
    this.seed = opts.seed
    this.radius = opts.radius
    this.heightScale = opts.heightScale
    this.resolution = opts.resolution ?? 32
    this.maxDepth = opts.maxDepth ?? 10
    this.splitFactor = opts.splitFactor ?? 3.0
    this.plateCount = opts.plateCount ?? 16
    this.hotspotCount = opts.hotspotCount ?? 6
    this.hotspotIntensity = opts.hotspotIntensity ?? 1
    this.targetTriPx = opts.targetTriPx ?? 2.5
    this.baseTemp = opts.baseTemp ?? 15
    this.atmosphere = opts.atmosphere ?? 0.6
    this.rotationPeriodS = opts.rotationPeriodS ?? 600
    this.axialTiltDeg = opts.axialTiltDeg ?? 23.4
    this.interior = { ...DEFAULT_INTERIOR, ...opts.interior }
    this.overrideInterior = opts.overrideInterior ?? false
    // Manual mid-layer defaults — used when overrideInterior is true.
    this.manualHeat = 0.5
    this.manualSurfaceTemp = 15
    this.manualAtmosphere = 0.6
    // Erosion bake defaults.
    // deriveErosionRes(WORLD_RADIUS) = 256 — the proven ~1.5 s budget.
    // main.ts ui.erosionRes is initialised to the same value so slider and bake agree.
    this.erosionRes = deriveErosionRes(WORLD_RADIUS)
    this.erosionStrength = 1.0
    this.erosionDeposition = 1.0
    // bUpliftRate: authored at HEIGHT_SCALE_REF as 40 m/step; scales with heightScale so
    // the normalized (workH / B_HEIGHT_SCALE) uplift per step is invariant under uniform scale.
    this.bUpliftRate = 40 * (this.heightScale / HEIGHT_SCALE_REF)
    // Safe fallbacks — caller updates these on the first update() call.
    this._vFovRadians = Math.PI / 3  // 60°
    this._screenHeightPx = 1080

    // Shared normal-view material — one instance for all terrain chunks.
    // Node material so the per-vertex 'subsurfaceWet' scalar can drop roughness on
    // water (lakes/seeps) → PBR-Fresnel sun glint; dry land stays matte
    // (wet=0 → roughness 1, identical to before). wet=1 → roughness 0.15 (≈ ocean shell).
    this.normalMaterial = new MeshStandardNodeMaterial({ vertexColors: true, roughness: 1, metalness: 0 })
    this.normalMaterial.roughnessNode = saturate(attribute('subsurfaceWet', 'float')).mul(-0.85).add(1.0)

    // Pre-build per-level debug materials with a distinct hue per level
    this.debugMaterials = Array.from({ length: this.maxDepth + 1 }, (_, i) => {
      const hue = i / (this.maxDepth + 1)
      const mat = new MeshBasicMaterial({ wireframe: false, vertexColors: false })
      mat.color.setHSL(hue, 0.85, 0.55)
      return mat
    })

    // Plate-color node material: unlit flat shading reading the baked 'plateColor' attribute.
    // attribute('plateColor', 'vec3') creates an AttributeNode for our baked Float32×3 attribute.
    this.plateColorMaterial = new MeshBasicNodeMaterial()
    this.plateColorMaterial.colorNode = attribute('plateColor', 'vec3')
    this.plateColorMaterial.vertexColors = false

    // Heightmap node material: grayscale by elevation, computed in-shader so toggling needs no
    // rebake. |positionWorld| = planet radius (rotation-invariant under tilt/spin), so
    // e = (radius − RADIUS)/heightScale. Remap the ACTUAL terrain range [E_MIN, E_MAX] linearly
    // to black→white so relief reads with full contrast (the naïve [−1,1] map left all land in
    // faint mid-gray). Darkest = deepest ocean, brightest = highest peak.
    this.heightmapMaterial = new MeshBasicNodeMaterial()
    {
      const E_MIN = -0.5  // deepest ocean shown as black
      const E_MAX = 0.9   // highest peak shown as white
      const e = positionWorld.length().sub(uniform(this.radius)).mul(uniform(1 / this.heightScale))
      const g = saturate(e.sub(E_MIN).mul(1 / (E_MAX - E_MIN)))
      this.heightmapMaterial.colorNode = vec3(g, g, g)
      this.heightmapMaterial.vertexColors = false
    }

    // Climate debug material: unlit, reads per-vertex 'climateMoist' float attribute and
    // several world-frame uniforms to shade one of three climate fields. A single integer
    // uniform `uField` selects temperature(0)/moisture(1)/insolation(2) so we need only
    // one material instance; setClimateField flips `uField` and re-swaps visible meshes.
    this.climateMaterial = this._buildClimateMaterial()

    // Wind debug material: purely analytical in-shader, no per-vertex attributes.
    // Defaults: band count from rotation period; strength from equator-pole temp range.
    // Both are updated after buildHeightFn initialises the climate sim (see below).
    this.windMaterial = this._buildWindMaterial()

    // Hardness debug material: reads the baked per-vertex 'rockHardness' float attribute
    // and maps it to a clear dark-to-bright ramp for the materials view.
    this.hardnessMaterial = this._buildHardnessMaterial()
    this.wetnessMaterial = this._buildWetnessMaterial()

    // Pure unlit wireframe: white edges only, no fill, no lighting.
    // When wireframe mode is on, every chunk renders with this material regardless of view mode.
    this.wireMaterial = new MeshBasicMaterial({ color: 0xffffff, wireframe: true })

    // Vertex dot overlay: yellow constant screen-space dots (sizeAttenuation false keeps
    // them at a fixed pixel size at any zoom level).
    this.pointsMaterial = new PointsMaterial({ color: 0xffff00, size: 2.5, sizeAttenuation: false })

    // Sync climate uniforms with initial option values.
    this._uBaseTemp.value     = this.baseTemp
    this._uRedistribution.value = 0.5
    this._updatePoleAxis()

    this.buildHeightFn(opts.seed)

    // Sync wind uniforms from the rotation/climate systems now that buildHeightFn has run.
    // bandCount → how many E-W band pairs the shader draws (matches the climate cell count).
    // uWindStrength is a plain manual gain (default 1); the ΔT→speed coupling is
    // handled in-shader via effStrength = uWindStrength * (uTempRange / 55).
    this._uWindBands.value    = this.deriveBandCount()
    this._uWindStrength.value = 1.0
    this.buildTectonicsDebug(opts.seed)
    this.buildWindDebug(opts.seed)
    this.buildWindFlow()
    this.buildCloudShell()
    this.buildAtmosphere()

    // Gizmos are seed-independent — built once, never rebuilt on regenerate().
    this.gizmos = new PlanetGizmos({ radius: this.radius })
    this.gizmos.visible = true   // DEFAULT ON
    this.add(this.gizmos)

    this.buildRoots()
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * update() is rotation-safe: converts cameraWorldPos to planet-local space
   * before any LOD math. Call it with the camera's world-space position even
   * when the planet group is tilted + spinning.
   *
   * @param cameraWorldPos  World-space camera position (Three.js world coordinates).
   * @param vFovRadians     Camera vertical FOV in radians (camera.fov is degrees — convert before passing).
   * @param screenHeightPx  Drawing-buffer height in pixels (renderer.getDrawingBufferSize().y).
   */
  update(cameraWorldPos: Vector3, vFovRadians: number, screenHeightPx: number, viewProj?: Matrix4): void {
    // Update the diagnostic overlay before the frozen guard so it shows live values
    // (including frozen:true) even when LOD selection is paused.
    if (this._diagEnabled) {
      this._updateDiagOverlay(cameraWorldPos, vFovRadians, screenHeightPx, this.frozen)
    }

    if (this.frozen) return

    // Store for use in SSE metric throughout this frame.
    this._vFovRadians = vFovRadians
    this._screenHeightPx = screenHeightPx

    // Cache inverse world transform once per frame (rigid body: translate + rotate only).
    this.updateMatrixWorld()
    this._invWorldMatrix.copy(this.matrixWorld).invert()
    // Extract inverse rotation quaternion for direction transforms (no translation needed).
    this.getWorldQuaternion(this._invWorldQuat).invert()

    // Push fresh per-frame cloud uniforms: time (from the wind clock, set before update()),
    // camera world position, and inverse world rotation so density is sampled in planet-local frame.
    if (this.cloudShell.visible) {
      this.cloudShell.update(this._uWindTime.value, cameraWorldPos, this._invWorldQuat)
    }

    // Convert camera world position → planet-local position (zero alloc via scratch).
    this._camLocalScratch.copy(cameraWorldPos).applyMatrix4(this._invWorldMatrix)

    // Build a planet-LOCAL frustum so collect() can cull off-screen subtrees without
    // allocating per node. viewProj maps world→clip; multiplying by matrixWorld on the
    // right makes it local→clip, so Frustum.intersectsSphere works in local space.
    if (viewProj !== undefined) {
      this._frustumMatrix.multiplyMatrices(viewProj, this.matrixWorld)
      this._localFrustum.setFromProjectionMatrix(this._frustumMatrix)
      this._frustumActive = true
    } else {
      this._frustumActive = false
    }

    this.promoteReadySplits()
    this.promoteReadyMerges(this._camLocalScratch)
    this.selectLeaves(this._camLocalScratch)
    this.drainBuildQueue()
  }

  setWireframe(on: boolean): void {
    this.wireframeActive = on
    // When on: every visible chunk swaps to wireMaterial (pure unlit white edges, no fill).
    // When off: each chunk reverts to its normal materialFor(level) material.
    for (const [, mesh] of this.visibleMeshes) {
      mesh.material = on
        ? this.wireMaterial
        : this.materialFor(this.levelFromKey(mesh.userData.key as string))
    }
  }

  setShowVertices(on: boolean): void {
    this._showVertices = on
    if (on) {
      // Create Points overlays for all currently-visible chunks.
      for (const [key, mesh] of this.visibleMeshes) {
        if (!this.visiblePoints.has(key)) {
          this._addPoints(key, mesh)
        }
      }
    } else {
      // Remove all Points overlays.
      for (const [, pts] of this.visiblePoints) {
        this.remove(pts)
        // Do NOT dispose pts.geometry — it is the shared chunk geometry owned by the LRU cache.
        // Only dispose the Points object itself (no GPU resources beyond the shared geometry reference).
      }
      this.visiblePoints.clear()
    }
  }

  setDebugColors(on: boolean): void {
    this.debugColorsActive = on
    // Mutual exclusivity: turning on LOD colors turns off tectonics + heightmap + materials + wind views.
    if (on) {
      if (this.tectonicsViewActive) {
        this.tectonicsViewActive = false
        this.tectonicsDebug.visible = false
      }
      this.windViewActive  = false
      this.heightmapViewActive = false
      this.materialsViewActive = false
    }
    // Wireframe overrides view-mode material; only swap when wireframe is off.
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  setTectonicsView(on: boolean): void {
    this.tectonicsViewActive = on
    this.tectonicsDebug.visible = on
    // Mutual exclusivity: turning on tectonics turns off LOD debug colors + heightmap + materials + wind.
    if (on) {
      this.debugColorsActive = false
      this.heightmapViewActive = false
      this.materialsViewActive = false
      this.windViewActive  = false
    }
    // Wireframe overrides view-mode material; only swap when wireframe is off.
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  setHeightmapView(on: boolean): void {
    this.heightmapViewActive = on
    // Mutual exclusivity: heightmap is its own view; turn off LOD colors + tectonics + materials + wind.
    if (on) {
      this.debugColorsActive = false
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
      this.materialsViewActive = false
      this.windViewActive  = false
    }
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Climate debug view API
  // ---------------------------------------------------------------------------

  /**
   * Enable or disable the climate debug view.
   * Mutually exclusive with the other debug views.
   */
  setClimateView(on: boolean): void {
    this.climateViewActive = on
    if (on) {
      this.debugColorsActive   = false
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
      this.heightmapViewActive = false
      this.materialsViewActive = false
      this.windViewActive  = false
    }
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  /**
   * Switch which climate field is visualised.
   * Re-applies the material swap so the change is visible immediately.
   */
  setClimateField(f: 'temperature' | 'moisture' | 'insolation'): void {
    this._climateField = f
    // Flip the field integer uniform that _buildClimateMaterial references.
    // 0 = temperature, 1 = moisture, 2 = insolation
    const fieldIndex = f === 'temperature' ? 0 : f === 'moisture' ? 1 : 2
    ;(this.climateMaterial.userData as { uField?: { value: number } }).uField!.value = fieldIndex
    // Re-swap visible meshes if the climate view is active so the change shows immediately.
    if (this.climateViewActive && !this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.climateMaterial
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Wind debug view API
  // ---------------------------------------------------------------------------

  /**
   * Single source of truth for wind overlay visibility.
   * Applies the per-overlay flags (_showWindArrows / _showWindFlow) so the caller
   * never has to know about them. Call this instead of toggling windDebug/windFlow directly.
   */
  setWindOverlaysVisible(on: boolean): void {
    this.windDebug.setVisible(on && this._showWindArrows)
    this.windFlow.setVisible(on && this._showWindFlow)
  }

  /**
   * Single source of truth for cloud-shell visibility.
   * The shell should be shown in normal view and hidden in every debug view.
   */
  setCloudShellVisible(on: boolean): void {
    this.cloudShell.setVisible(on)
  }

  /**
   * Enable or disable the wind debug view.
   * Wind view shows the baked-field arrow overlay (WindDebug) over normal terrain.
   * Mutually exclusive with the other debug views (climate/tectonics/heightmap/lod).
   * Wind overlay visibility is managed externally via setWindOverlaysVisible().
   */
  setWindView(on: boolean): void {
    this.windViewActive = on
    if (on) {
      this.climateViewActive   = false
      this.debugColorsActive   = false
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
      this.heightmapViewActive = false
      this.materialsViewActive = false
    }
    // Terrain material reverts to normal when wind view is on (arrows/flow are the visualisation).
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  /**
   * Switch which wind sub-field is visualised.
   * 'flow' (0) = animated direction+speed streaks (default).
   * 'speed' (1) = heatmap of wind magnitude.
   * 'direction' (2) = pure hue-wheel by bearing.
   * Re-applies material swap so the change is visible immediately.
   */
  setWindField(f: 'flow' | 'speed' | 'direction'): void {
    this._windField = f
    const idx = f === 'flow' ? 0 : f === 'speed' ? 1 : 2
    this._uWindField.value = idx
    if (this.windViewActive && !this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.windMaterial
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Materials debug view API (rock hardness)
  // ---------------------------------------------------------------------------

  /**
   * Enable or disable the materials (rock hardness) debug view.
   * Shows the baked per-vertex rockHardness scalar on a dark→bright ramp.
   * Mutually exclusive with all other debug views.
   */
  setMaterialsView(on: boolean): void {
    this.materialsViewActive = on
    if (on) {
      this.windViewActive      = false
      this.climateViewActive   = false
      this.debugColorsActive   = false
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
      this.heightmapViewActive = false
      this.wetnessViewActive   = false
    }
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  /**
   * Enable or disable the soil-wetness / aquifer debug view.
   * Shows the per-vertex surface-wetness scalar (lakes + groundwater seeps)
   * on a dry-brown → teal → cyan ramp. Mutually exclusive with other debug views.
   */
  setWetnessView(on: boolean): void {
    this.wetnessViewActive = on
    if (on) {
      this.materialsViewActive = false
      this.windViewActive      = false
      this.climateViewActive   = false
      this.debugColorsActive   = false
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
      this.heightmapViewActive = false
    }
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
  }

  /** Soil-wetness ramp material: reads per-vertex 'subsurfaceWet' [0,1]. */
  private _buildWetnessMaterial(): MeshBasicNodeMaterial {
    const w = saturate(attribute('subsurfaceWet', 'float'))
    const dry = vec3(0x3a / 255, 0x2a / 255, 0x18 / 255) // dry brown
    const mid = vec3(0x1f / 255, 0x9e / 255, 0x8a / 255) // damp teal
    const wet = vec3(0x2f / 255, 0xd0 / 255, 0xff / 255) // saturated cyan
    const t0 = saturate(w.mul(2))
    const t1 = saturate(w.sub(0.5).mul(2))
    const mat = new MeshBasicNodeMaterial()
    mat.colorNode = mix(mix(dry, mid, t0), wet, t1)
    mat.vertexColors = false
    return mat
  }

  /** Global wind speed multiplier. Live — updates uniform immediately. */
  setWindStrength(n: number): void {
    this._uWindStrength.value = Math.max(0, n)
  }

  /** Zonal band-pair count N. Live — updates uniform immediately. */
  setWindBands(n: number): void {
    this._uWindBands.value = Math.max(1, n)
  }

  /** Turbulence perturbation strength. 0 = off. Live — updates uniform immediately. */
  setTurbulence(n: number): void {
    this._uTurbulence.value = Math.max(0, n)
  }

  /**
   * Rotation direction. +1 = prograde (Earth-like westerlies blow east).
   * -1 = retrograde (Venus-like; reverses E-W flow). Live — updates uniform immediately.
   */
  setRetrograde(b: boolean): void {
    this._uRetrograde.value = b ? -1 : 1
  }

  /** Per-frame animation clock for the 'flow' sub-mode. Live — updates uniform immediately. */
  setWindTime(t: number): void {
    this._uWindTime.value = t
  }

  // ---------------------------------------------------------------------------
  // Wind bake param setters — each stores the value and triggers a full regenerate
  // because these are baked into Climate's wind cubemap (not live uniforms).
  // ---------------------------------------------------------------------------

  /** Vortex swirl blend weight [0,1]. 0 = pure zonal belts, 1 = full vortex. Triggers rebake. */
  setWindSwirl(n: number): void {
    this._windSwirlStrength = Math.max(0, Math.min(2, n))
    this.regenerate(this.seed)
  }

  /** HIGH-pressure centres per hemisphere [0,12]. Triggers rebake. */
  setWindHighs(n: number): void {
    this._windNHigh = Math.max(0, Math.min(12, Math.round(n)))
    this.regenerate(this.seed)
  }

  /** LOW-pressure centres per hemisphere [0,12]. Triggers rebake. */
  setWindLows(n: number): void {
    this._windNLow = Math.max(0, Math.min(12, Math.round(n)))
    this.regenerate(this.seed)
  }

  /** Max cross-isobar tilt angle in radians [0, π/2]. Triggers rebake. */
  setWindCoriolis(n: number): void {
    this._windCrossIsobarMax = Math.max(0, Math.min(Math.PI / 2, n))
    this.regenerate(this.seed)
  }

  /** Base Gaussian half-width for pressure centres in radians [0.05, 1.0]. Triggers rebake. */
  setWindVortexSize(n: number): void {
    this._windSigmaBase = Math.max(0.05, Math.min(1.0, n))
    this.regenerate(this.seed)
  }

  /** ± latitude spread for pressure-centre placement in radians [0, π/4]. Triggers rebake. */
  setWindLatSpread(n: number): void {
    this._windLatSpread = Math.max(0, Math.min(Math.PI / 4, n))
    this.regenerate(this.seed)
  }

  /** Equatorial taper half-width in radians [0, 0.5]. Triggers rebake. */
  setWindEquatorTaper(v: number): void {
    this._windEquatorTaper = Math.max(0, Math.min(0.5, v))
    this.regenerate(this.seed)
  }

  // ---------------------------------------------------------------------------
  // Transient wind setters — LIVE, no rebake. Forward to climateSim.setWindTransient.
  // ---------------------------------------------------------------------------

  setWindDrift(v: number): void       { this.climateSim.setWindTransient({ driftSpeed: v }) }
  setWindPulseRate(v: number): void   { this.climateSim.setWindTransient({ pulseRate: v }) }
  setWindPulseDepth(v: number): void  { this.climateSim.setWindTransient({ pulseDepth: v }) }
  setWindEddyStrength(v: number): void { this.climateSim.setWindTransient({ eddyStrength: v }) }
  setWindEddyScale(v: number): void   { this.climateSim.setWindTransient({ eddyScale: v }) }
  setWindEddyTimeScale(v: number): void { this.climateSim.setWindTransient({ eddyTimeScale: v }) }

  /**
   * Animate the wind overlays at time `t` with frame delta `dt`. Self-gated:
   * each overlay returns immediately when not visible (zero cost off-screen).
   * Call once per frame after planet.setWindTime(t).
   */
  animateWind(t: number, dt: number, camWorldPos: Vector3): void {
    if (this.windDebug.visible)   this.windDebug.animate(t)
    if (this.windFlow.visible) {
      // Camera into the planet-local frame (the wind field's frame). _invWorldQuat is
      // from the previous update() — a 1-frame-stale rotation is invisible on an overlay.
      // Planet sits at the world origin (only spun), so altitude = |camPos| − radius.
      const altitude = camWorldPos.length() - this.radius
      this._windCamDir.copy(camWorldPos).applyQuaternion(this._invWorldQuat).normalize()
      this.windFlow.animate(dt, this._windCamDir, altitude)
    }
    // Cloud update (time + fresh _invWorldQuat + cameraWorldPos) is pushed in update(),
    // which runs after animateWind so the rotation quaternion is from the current frame.
  }

  /** Retrograde bake flag. +1 = prograde (Earth-like), -1 = retrograde (Venus-like). Triggers rebake. */
  setWindRetrogradeBake(r: number): void {
    this._windRetrogradeBake = r >= 0 ? 1 : -1
    this.regenerate(this.seed)
  }

  /**
   * Set the wind arrow density (Fibonacci-sphere sample count). Live rebuild — rebuilds
   * WindDebug immediately. No rebake of the baked field.
   */
  setWindArrowDensity(n: number): void {
    this._windArrowDensity = Math.max(1, Math.round(n))
    this.windDebug.setDensity(this._windArrowDensity)
  }

  /**
   * Scale the rendered arrow length. 1.0 = default (radius * 0.045 per arrow).
   * Live — scales the InstancedMesh objects directly without rebuilding. No rebake.
   */
  setWindArrowScale(s: number): void {
    this.windDebug.setArrowScale(Math.max(0.01, s))
  }

  /**
   * Show or hide the wind arrow overlay. Live — applies immediately when in wind view.
   */
  setWindArrowsVisible(b: boolean): void {
    this._showWindArrows = b
    if (this.windViewActive) this.windDebug.setVisible(b)
  }

  /**
   * Show or hide the wind flow streaklines. Live — applies immediately when in wind view.
   */
  setWindFlowVisible(b: boolean): void {
    this._showWindFlow = b
    if (this.windViewActive) this.windFlow.setVisible(b)
  }

  /** Set wind flow particle count. Triggers rebuild. */
  setWindFlowDensity(n: number): void {
    this.windFlow.setDensity(n)
  }

  /** Set wind flow advection speed. Live. */
  setWindFlowSpeed(v: number): void {
    this.windFlow.setFlowSpeed(v)
  }

  /** Set wind flow trail length. Triggers rebuild. */
  setWindFlowTrail(k: number): void {
    this.windFlow.setTrailLength(k)
  }

  /** Set wind flow particle lifetime. Live. */
  setWindFlowLifetime(s: number): void {
    this.windFlow.setLifetime(s)
  }

  // ---------------------------------------------------------------------------
  // Cloud shell public API
  // ---------------------------------------------------------------------------

  /** Live — mutates cloud coverage uniform, no rebuild. */
  setCloudCoverage(v: number): void    { this.cloudShell.setCoverage(v) }
  /** Live — mutates cloud scroll speed uniform, no rebuild. */
  setCloudScrollSpeed(v: number): void { this.cloudShell.setScrollSpeed(v) }
  /** Live — mutates cloud FBM frequency uniform, no rebuild. */
  setCloudScale(v: number): void       { this.cloudShell.setCloudScale(v) }
  /** Live — mutates cloud wind-warp strength uniform, no rebuild. */
  setCloudWarp(v: number): void        { this.cloudShell.setWindWarp(v) }
  /** Live — mutates cloud opacity uniform, no rebuild. */
  setCloudOpacity(v: number): void     { this.cloudShell.setOpacity(v) }
  /** Scale shell altitude. Scales the mesh to the new radius. */
  setCloudAltitude(mul: number): void  { this.cloudShell.setAltitude(mul) }
  /** Live — mutates cloud favorability weight uniform, no rebuild. */
  setCloudFavWeight(v: number): void   { this.cloudShell.setFavWeight(v) }
  /** Live — mutates cloud moisture weight uniform, no rebuild. */
  setCloudMoistWeight(v: number): void { this.cloudShell.setMoistWeight(v) }
  /** Live — mutates cloud convergence weight uniform, no rebuild. */
  setCloudConvWeight(v: number): void  { this.cloudShell.setConvWeight(v) }
  /** Live — mutates cloud convergence gain uniform, no rebuild. */
  setCloudConvGain(v: number): void    { this.cloudShell.setConvGain(v) }
  /** Live — mutates cloud ITCZ weight uniform, no rebuild. */
  setCloudItczWeight(v: number): void  { this.cloudShell.setItczWeight(v) }
  setCloudWeatherWeight(v: number): void { this.cloudShell.setWeatherWeight(v) }
  setCloudCellWeight(v: number): void  { this.cloudShell.setCellWeight(v) }
  /** Debug "cloud map" mode (flat coverage heatmap + contours) for the 'cloud' view. */
  setCloudDebugMode(on: boolean): void { this.cloudShell.setDebugMode(on) }
  /** Live — mutates cloud billow uniform, no rebuild. */
  setCloudBillow(v: number): void      { this.cloudShell.setBillow(v) }
  /** Live — mutates cloud detail uniform, no rebuild. */
  setCloudDetail(v: number): void      { this.cloudShell.setDetail(v) }
  /** Live — mutates cloud softness uniform, no rebuild. */
  setCloudSoftness(v: number): void    { this.cloudShell.setSoftness(v) }
  /** Live — mutates cloud volume uniform, no rebuild. */
  setCloudVolume(v: number): void      { this.cloudShell.setVolume(v) }
  /** Live — sets cloud layer base altitude (in heightScale multiples). Rescales mesh. */
  setCloudBase(v: number): void        { this.cloudShell.setCloudBase(v) }
  /** Live — sets cloud layer thickness (in heightScale multiples). Rescales mesh. */
  setCloudThick(v: number): void       { this.cloudShell.setCloudThick(v) }
  /** Live — mutates cloud extinction density (_uSigmaT) uniform, no rebuild. */
  setCloudDensity(v: number): void     { this.cloudShell.setDensity(v) }
  /** Live — sets raymarching step count. Clamps to >=1. */
  setCloudStepCount(v: number): void   { this.cloudShell.setStepCount(v) }
  /** Live — sets light-march step count per cloud sample. Clamps to >=1. */
  setCloudLightSteps(v: number): void  { this.cloudShell.setLightSteps(v) }
  /** Live — mutates Henyey-Greenstein anisotropy uniform, no rebuild. */
  setCloudHg(v: number): void          { this.cloudShell.setHgAnisotropy(v) }
  /** Live — mutates powder-effect strength uniform, no rebuild. */
  setCloudPowder(v: number): void      { this.cloudShell.setPowder(v) }
  /** Live — mutates detail-erosion noise scale uniform, no rebuild. */
  setCloudDetailScale(v: number): void { this.cloudShell.setDetailScale(v) }
  /** Live — mutates round-base profile exponent uniform, no rebuild. */
  setCloudRoundBase(v: number): void   { this.cloudShell.setRoundBase(v) }
  /** Live — mutates billow-top profile exponent uniform, no rebuild. */
  setCloudBillowTop(v: number): void   { this.cloudShell.setBillowTop(v) }
  /** Live — mutates ambient sky-light contribution uniform, no rebuild. */
  setCloudAmbient(v: number): void     { this.cloudShell.setAmbient(v) }
  /** Live — mutates convective cloud-type strength uniform, no rebuild. */
  setCloudType(v: number): void        { this.cloudShell.setTypeStrength(v) }
  /** The cloud shell mesh (for the half-res CloudCompositor's offscreen pass). Null if unbuilt. */
  getCloudMesh(): Mesh | null { return this.cloudShell.getMesh() }

  // ---------------------------------------------------------------------------
  // Atmosphere public API
  // ---------------------------------------------------------------------------

  /** Live — mutates atmosphere density uniform, no rebuild. */
  setAtmosphereDensity(v: number): void      { this.atmosphereShell.setDensity(v) }
  /** Live — mutates atmosphere tint uniform, no rebuild. */
  setAtmosphereTint(c: Color): void {
    this.atmosphereShell.setTint(c)
  }
  /** Live — mutates atmosphere sun intensity uniform, no rebuild. */
  setAtmosphereSunIntensity(v: number): void { this.atmosphereShell.setSunIntensity(v) }
  /** Live — mutates atmosphere horizon pow exponent uniform, no rebuild. */
  setAtmosphereScaleHeight(v: number): void  { this.atmosphereShell.setScaleHeight(v) }
  /** Live — mutates atmosphere zenith in-scatter floor uniform, no rebuild. */
  setAtmosphereSkyFloor(v: number): void     { this.atmosphereShell.setSkyFloor(v) }
  /** Scale atmosphere shell radius. Scales mesh without rebuild. */
  setAtmosphereHeight(mul: number): void     { this.atmosphereShell.setAtmHeight(mul) }
  /** Show or hide the atmosphere shell. */
  setAtmosphereVisible(v: boolean): void     { this.atmosphereShell.setVisible(v) }

  /**
   * Set the world-space sun direction (normalised). Live — no rebake.
   * This is the direction FROM the sun (i.e. towards which lit faces point),
   * stored in world frame so it can be passed directly without conversion.
   */
  setSunDir(worldSunDir: Vector3): void {
    this._uSunDir.value.copy(worldSunDir).normalize()
  }

  /**
   * Heat redistribution factor in [0,1].
   * 0 = pure day/night terminator; 1 = pure latitudinal field.
   * Live — updates uniform immediately, no rebake.
   */
  setRedistribution(n: number): void {
    this._uRedistribution.value = Math.max(0, Math.min(1, n))
  }

  /**
   * Greenhouse offset in °C.
   * Updates the live uniform immediately. Does NOT trigger regenerate — the caller
   * is responsible for calling regenerate(seed) via onFinishChange to rebake the
   * baked biome temperature.
   */
  setGreenhouse(n: number): void {
    this._uGreenhouse.value = n
  }

  /**
   * Lapse rate in °C over full normalised terrain height.
   * Updates the live uniform immediately. Does NOT trigger regenerate — the caller
   * is responsible for calling regenerate(seed) via onFinishChange to rebake the
   * baked biome temperature.
   */
  setLapseRate(n: number): void {
    this._uLapseRate.value = n
  }

  /**
   * Equator-to-pole temperature spread in °C.
   * Live — updates uniform immediately, no rebake.
   */
  setTempRange(n: number): void {
    this._uTempRange.value = n
  }

  setFrozen(on: boolean): void {
    this.frozen = on
  }

  /**
   * Show or hide the pole-axis + equator gizmos independently of the
   * tectonics debug overlay.
   */
  setGizmosVisible(on: boolean): void {
    this.gizmos.visible = on
  }

  /**
   * Enable or disable the LOD diagnostic overlay and console logging.
   * Off by default. When turned off at runtime, hides the overlay div if it exists.
   */
  setDiagEnabled(on: boolean): void {
    this._diagEnabled = on
    if (!on && this._diagEl !== null) {
      this._diagEl.style.display = 'none'
    } else if (on && this._diagEl !== null) {
      this._diagEl.style.display = ''
    }
  }

  /**
   * Set the desired plate count for the next regenerate() call.
   * Clamped to [0, 48]. Values 0 or 1 trigger the non-tectonic (stagnant-lid) regime.
   * Has no effect on the current terrain — takes effect on the next regenerate(seed) invocation.
   */
  setPlateCount(n: number): void {
    this.plateCount = Math.max(0, Math.min(48, n))
  }

  /**
   * Set the desired arc density for the next regenerate() call.
   * Clamped to [0.2, 3]. Has no effect on the current terrain — takes effect
   * on the next regenerate(seed) invocation.
   */
  setArcDensity(d: number): void {
    this.arcDensity = Math.max(0.2, Math.min(3, d))
  }

  /**
   * Set the desired hotspot count for the next regenerate() call.
   * Clamped to [0, 20]. Applied on the next regenerate(seed).
   */
  setHotspotCount(n: number): void {
    this.hotspotCount = Math.max(0, Math.min(20, Math.round(n)))
  }

  /**
   * Set the global hotspot intensity multiplier for the next regenerate() call.
   * Clamped to [0, 3]. Applied on the next regenerate(seed).
   */
  setHotspotIntensity(v: number): void {
    this.hotspotIntensity = Math.max(0, Math.min(3, v))
  }

  // ---------------------------------------------------------------------------
  // Interior physics API
  // ---------------------------------------------------------------------------

  /**
   * Merge additional interior parameters into the active InteriorParams and regenerate.
   * Accepts the new 5-root params: mass, composition, age, insolation, waterBudget.
   * surfaceTemp is now a derived output — use setManualSurfaceTemp() to pin it directly.
   *
   * Guard: if the merged result is identical to the current interior (e.g. startup call
   * with default values already set by the constructor), skip regenerate to avoid a
   * redundant bake.
   */
  setInteriorParams(p: Partial<InteriorParams>): void {
    const merged = { ...this.interior, ...p }
    if (JSON.stringify(merged) === JSON.stringify(this.interior)) return
    this.interior = merged
    this.regenerate(this.seed)
  }

  /**
   * Switch between derived-interior mode (false, default) and manual mid-layer override
   * mode (true). In override mode, manualHeat/manualSurfaceTemp/manualAtmosphere are
   * passed as internalHeat/surfaceTemp/atmosphere overrides into deriveInterior().
   * Triggers a full regenerate so the terrain updates immediately.
   */
  setOverrideInterior(on: boolean): void {
    this.overrideInterior = on
    this.regenerate(this.seed)
  }

  // ---------------------------------------------------------------------------
  // Manual mid-layer setters (active when overrideInterior is true)
  // ---------------------------------------------------------------------------

  /**
   * Set the manual internal heat override in [0..1].
   * Only active when overrideInterior is true. Triggers regenerate.
   */
  setManualHeat(n: number): void {
    this.manualHeat = Math.max(0, Math.min(1, n))
    this.regenerate(this.seed)
  }

  /**
   * Set the manual surface temperature override in °C.
   * Only active when overrideInterior is true. Triggers regenerate
   * and pushes the value to the live _uBaseTemp uniform immediately.
   */
  setManualSurfaceTemp(n: number): void {
    this.manualSurfaceTemp = n
    this._uBaseTemp.value = n
    this.regenerate(this.seed)
  }

  /**
   * Set the manual atmosphere override in [0..1].
   * Only active when overrideInterior is true. Triggers regenerate.
   */
  setManualAtmosphere(n: number): void {
    this.manualAtmosphere = Math.max(0, Math.min(1, n))
    this.regenerate(this.seed)
  }

  // ---------------------------------------------------------------------------
  // Erosion tuning API
  // ---------------------------------------------------------------------------

  /**
   * Set the erosion bake resolution (128 | 256 | 512).
   * Stored; takes effect on the next regenerate() call (full rebake).
   */
  setErosionRes(n: 128 | 256 | 512): void {
    this.erosionRes = n
    this.regenerate(this.seed)
  }

  /**
   * Set the erosion incision-rate multiplier (0..3, default 1).
   * Scales K0 by this factor: 0 = no erosion, 1 = default, 3 = heavy carve.
   * Stored; takes effect on the next regenerate() call (full rebake).
   */
  setErosionStrength(n: number): void {
    this.erosionStrength = Math.max(0, Math.min(3, n))
    this.regenerate(this.seed)
  }

  /**
   * Set the deposition coefficient multiplier (0..3, default 1).
   * Scales G by this factor: 0 = no deposition (pure incision), 1 = default, 3 = heavy fans/deltas.
   * Stored; takes effect on the next regenerate() call (full rebake).
   */
  setErosionDeposition(n: number): void {
    this.erosionDeposition = Math.max(0, Math.min(3, n))
    this.regenerate(this.seed)
  }

  /**
   * Set the number of B (iterated uplift+erode) loop steps.
   * Clamped to [1, 200]. Takes effect on the next regenerate() call.
   */
  setBSteps(n: number): void {
    this.bSteps = Math.max(1, Math.min(200, Math.round(n)))
    this.regenerate(this.seed)
  }

  /**
   * Set the B uplift rate in metres per step.
   * Upper bound scales with heightScale: 500 m/step at HEIGHT_SCALE_REF=1200,
   * proportionally larger at larger scales (5000 m/step at HEIGHT_SCALE=12000).
   */
  setBUpliftRate(n: number): void {
    const maxRate = 500 * (this.heightScale / HEIGHT_SCALE_REF)
    this.bUpliftRate = Math.max(0, Math.min(maxRate, n))
    this.regenerate(this.seed)
  }

  /**
   * Set the axial tilt in degrees. Updates planet.rotation.z, refreshes the pole-axis
   * uniform, and regenerates so the climate rebuilds with the new axialTiltRad.
   *
   * Guard: if the value is already the same (e.g. startup call with default 23.4 already
   * set by the constructor), skip regenerate to avoid a redundant bake.  rotation.z and
   * the pole-axis uniform are still updated so the visual state stays correct.
   */
  setAxialTilt(deg: number): void {
    this.rotation.z = (deg * Math.PI) / 180
    this.refreshPoleAxis()
    if (deg === this.axialTiltDeg) return
    this.axialTiltDeg = deg
    this.regenerate(this.seed)
  }

  /**
   * Return the derived interior computed during the most recent buildHeightFn call.
   * Includes the new Stage-A outputs: gravity, internalHeat, surfaceTemp, atmosphere,
   * equilibriumTemp (populated by the new interior.ts contract).
   */
  getDerivedInterior(): DerivedInterior {
    return this._derived
  }

  /**
   * Return the waterline elevation in normalized units (~[-1,1], same as heightFn output).
   * Computed once per regenerate() via Fibonacci-sphere hypsometry from oceanCoverage.
   * Use this to position the water shell or shade land vs. ocean on the GPU.
   */
  getSeaLevel(): number {
    return this._seaLevel
  }

  /**
   * Set the mean surface temperature (°C-ish) for the climate model.
   * Stored for rebake on the next regenerate(seed).
   * Also updates the climate material uniform live so the debug view responds instantly.
   * Note: surfaceTemp is now a derived field — in non-override mode use
   * setInteriorParams({ insolation: v }) or other roots to influence it indirectly.
   * In override mode use setManualSurfaceTemp() to pin it directly.
   */
  setBaseTemp(t: number): void {
    this.baseTemp = t
    this._uBaseTemp.value = t
  }

  /**
   * Set atmospheric thickness in [0,1] (thick → uniform/small gradients, thin → extremes).
   * Stored; applied on the next regenerate(seed).
   */
  setAtmosphere(a: number): void {
    this.atmosphere = Math.max(0, Math.min(1, a))
  }

  /**
   * Set rotation period in seconds. Drives the circulation band count via
   * deriveBandCount(). Stored; applied on the next regenerate(seed).
   */
  setRotationPeriod(s: number): void {
    this.rotationPeriodS = Math.max(1, s)
  }

  /**
   * Circulation cells per hemisphere derived from rotation period.
   * Faster rotation (shorter period) → stronger Coriolis → more, narrower cells.
   * bandCount = clamp(round(3 · sqrt(600 / period)), 1, 7). period=600 → 3 (Earth-like),
   * 150 → 6, 3000 → 1. Generic: any rotation maps to a band count, no hardcoded "3 cells".
   */
  private deriveBandCount(): number {
    const n = Math.round(3 * Math.sqrt(600 / this.rotationPeriodS))
    return Math.max(1, Math.min(7, n))
  }

  /**
   * Live-set the maximum quadtree depth.
   * Clamped to [4, 20]. Takes effect on the next update() call (no rebuild needed).
   */
  setMaxDepth(n: number): void {
    this.maxDepth = Math.max(4, Math.min(20, n))
  }

  /**
   * Live-set the SSE target triangle pixel size.
   * Clamped to [0.5, 32]. Takes effect on the next update() call.
   */
  setTargetTriPx(n: number): void {
    this.targetTriPx = Math.max(0.5, Math.min(32, n))
  }

  regenerate(seed: number): void {
    // Remove all Points overlays first (geometry is owned by the mesh, disposed below).
    for (const pts of this.visiblePoints.values()) {
      this.remove(pts)
    }
    this.visiblePoints.clear()
    // Remove all visible meshes from scene and dispose their geometries explicitly
    // (visible geometries are NOT in the cache — invariant: cache holds only non-displayed geometries)
    for (const mesh of this.visibleMeshes.values()) {
      this.remove(mesh)
      ;(mesh.geometry as BufferGeometry).dispose()
      // Shared materials (normalMaterial, debugMaterials, plateColorMaterial, wireMaterial) are NOT disposed per-mesh.
    }
    this.visibleMeshes.clear()

    // Dispose all cached geometry
    this.geoCache.disposeAll()

    // Clear pending state
    this.buildQueue.length = 0
    this.buildQueueSet.clear()
    this.splitPending.clear()
    this.mergePending.clear()

    // Preserve debug overlay visibility across regeneration.
    const debugWasVisible  = this.tectonicsDebug.visible
    const cloudsWasVisible = this.cloudShell.visible
    const atmWasVisible    = this.atmosphereShell.visible
    // Wind overlays are restored from the active view flags after rebuild (see below).

    // Dispose existing TectonicsDebug (remove from this Group, free GPU resources).
    this.tectonicsDebug.dispose()
    this.remove(this.tectonicsDebug)

    // Dispose existing WindDebug (removes its InstancedMeshes from scene + frees GPU resources).
    this.windDebug.dispose()

    // Dispose existing WindFlow (removes LineSegments from scene + frees GPU resources).
    this.windFlow.dispose()

    // Dispose existing VolumetricClouds (removes mesh from scene + frees GPU resources).
    this.cloudShell.dispose()

    // Dispose existing Atmosphere (removes mesh from scene + frees GPU resources).
    this.atmosphereShell.dispose()

    // Rebuild
    ;(this as { seed: number }).seed = seed
    this.buildHeightFn(seed)
    this.buildTectonicsDebug(seed)
    this.tectonicsDebug.visible = debugWasVisible
    this.buildWindDebug(seed)
    this.buildWindFlow()
    // Restore wind overlay visibility via the single source of truth.
    // windViewActive is still set from before the regen.
    this.setWindOverlaysVisible(this.windViewActive)
    this.buildCloudShell()
    // Restore cloud-shell visibility: shell was only on in normal view, so restoring the
    // saved flag is safe. applyView in main.ts will re-assert the correct state next frame.
    this.setCloudShellVisible(cloudsWasVisible)
    this.buildAtmosphere()
    // Restore atmosphere visibility (always-on, but respect any explicit toggle).
    this.setAtmosphereVisible(atmWasVisible)
    // Refresh pole-axis uniform in case the world matrix changed since construction.
    this._updatePoleAxis()

    this.buildRoots()
  }

  /**
   * getSurfaceRadiusAt(v): rotation-safe world-space query.
   * Converts the world-space direction v into planet-local space (rotation only,
   * no translation) before sampling heightFn. Preallocated scratch — zero allocs.
   */
  getSurfaceRadiusAt(v: Vector3): number {
    // Apply inverse world rotation to get a planet-local direction.
    // Sample at maxDepth for full-detail accuracy (altitude / HUD read at ground level).
    this._dirLocalScratch.copy(v).normalize().applyQuaternion(this._invWorldQuat)
    return this.radius + this.heightFn(this._dirLocalScratch, this.maxDepth) * this.heightScale
  }

  /** Expose the Tectonics instance (main.ts HUD may read plate count). */
  get tectonicsInstance(): Tectonics {
    return this.tectonics
  }

  /** Expose the Climate instance (rebuilt on regenerate). */
  get climate(): Climate {
    return this.climateSim
  }

  /** Expose the Subsurface instance (rebuilt on regenerate; null until first buildHeightFn). */
  getSubsurface(): Subsurface | null {
    return this._subsurface
  }

  /** Expose the CaveField (main-thread only; rebuilt on regenerate; null until first buildHeightFn). */
  getCaveField(): CaveField | null {
    return this._caveField
  }

  getStats(): Stats {
    let minLevel = Infinity
    let maxLevel = 0
    for (const [key] of this.visibleMeshes) {
      const lv = this.levelFromKey(key)
      if (lv < minLevel) minLevel = lv
      if (lv > maxLevel) maxLevel = lv
    }
    if (this.visibleMeshes.size === 0) minLevel = 0
    const poolBacklog = this.pool
      ? this.pool.pendingCount + this.pool.inFlightCount
      : 0
    return {
      leaves: this.visibleMeshes.size,
      cached: this.geoCache.size,
      minLevel,
      maxLevel,
      pendingBuilds: this.buildQueue.length + poolBacklog,
      lastBuildMs: this.lastBuildMs,
      plates: this.tectonics.plates.length,
      volcanoes: this.tectonics.volcanoes.length,
      hotspots: this.tectonics.hotspots.length,
      bandCount: this.deriveBandCount(),
    }
  }

  dispose(): void {
    // Remove all Points overlays (geometry shared with meshes — do NOT dispose it here).
    for (const pts of this.visiblePoints.values()) {
      this.remove(pts)
    }
    this.visiblePoints.clear()
    // Dispose visible mesh geometries explicitly
    // (not in cache — invariant: cache holds only non-displayed geometries)
    for (const mesh of this.visibleMeshes.values()) {
      this.remove(mesh)
      ;(mesh.geometry as BufferGeometry).dispose()
      // Shared materials are disposed once below — not per-mesh.
    }
    this.visibleMeshes.clear()
    this.geoCache.disposeAll()
    this.normalMaterial.dispose()
    for (const m of this.debugMaterials) m.dispose()
    this.plateColorMaterial.dispose()
    this.heightmapMaterial.dispose()
    this.climateMaterial.dispose()
    this.windMaterial.dispose()
    this.hardnessMaterial.dispose()
    this.wetnessMaterial.dispose()
    this.wireMaterial.dispose()
    this.pointsMaterial.dispose()
    this.tectonicsDebug.dispose()
    this.windDebug.dispose()
    this.windFlow.dispose()
    this.cloudShell.dispose()
    this.atmosphereShell.dispose()
    this.gizmos.dispose()
    this.climateSim.dispose()
    this.pool?.dispose()
    this.pool = null
  }

  // ---------------------------------------------------------------------------
  // Internal: initialisation
  // ---------------------------------------------------------------------------

  /**
   * Build heightFn, tectonics, and climate from seed via the shared factory.
   * The factory contains the single source of truth for all terrain closures;
   * this method is a thin caller that assigns the results to Planet fields.
   */
  private buildHeightFn(seed: number): void {
    // Derive all physics outputs from the interior root params.
    // In override mode, pin the three mid-layer fields (internalHeat/surfaceTemp/atmosphere)
    // to the manual knob values so deriveInterior still runs the full pipeline with those
    // inputs pinned — plate count, regime, etc. are still derived, not hardcoded.
    const ov = this.overrideInterior
      ? {
          internalHeat: this.manualHeat,
          surfaceTemp:  this.manualSurfaceTemp,
          atmosphere:   this.manualAtmosphere,
        }
      : undefined
    const d = deriveInterior(this.interior, ov)
    this._derived = d

    // Sync the legacy knob fields so external getters remain consistent.
    this.plateCount       = d.plateCount
    this.arcDensity       = d.arcDensity
    this.hotspotCount     = d.hotspotCount
    this.hotspotIntensity = d.hotspotIntensity

    // Keep the live climate uniform in sync with the derived surface temperature.
    this._uBaseTemp.value = d.surfaceTemp

    const preT0 = performance.now()
    const tectonics = new Tectonics({
      seed,
      plateCount: d.plateCount,
      arcDensity: d.arcDensity,
      hotspotCount: d.hotspotCount,
      hotspotIntensity: d.hotspotIntensity,
      driftScale: d.driftScale,
      composition: this.interior.composition,
    })

    // Shared sampler options (without tectonics/climate/erosion — added per step below)
    const samplerOpts = {
      seed,
      radius: this.radius,
      heightScale: this.heightScale,
      plateCount: d.plateCount,
      arcDensity: d.arcDensity,
      baseTemp: d.surfaceTemp,
      atmosphere: d.atmosphere,
      bandCount: this.deriveBandCount(),
      axialTiltRad: (this.axialTiltDeg * Math.PI) / 180,
      driftScale: d.driftScale,
      tectonics,
      // Wind bake params — forwarded to Climate so the baked field reflects slider values.
      swirlStrength:     this._windSwirlStrength,
      nHigh:             this._windNHigh,
      nLow:              this._windNLow,
      crossIsobarMax:    this._windCrossIsobarMax,
      sigmaBase:         this._windSigmaBase,
      latSpread:         this._windLatSpread,
      windRetrograde:    this._windRetrogradeBake,
      equatorTaperWidth: this._windEquatorTaper,
    }

    // --- Step 1: build base-only sampler (seed for B; omits orogenic stamps) ---
    const baseSampler0 = makeTerrainSampler({ ...samplerOpts, baseOnly: true })
    const climate = baseSampler0.climate

    const EROSION_DEFAULT_K0 = 0.35
    const EROSION_DEFAULT_G  = 3.0

    // --- Step 2: bake upliftForcing field at B's cubemap resolution ------------
    // upliftForcing[c] = max(0, tectonics.upliftAt(dirC)) * smoothstep(-0.05, 0.15, crustDist)
    // This concentrates uplift on continental-crust land cells and avoids ocean uplift.
    const B_RES   = this.erosionRes
    const bN      = 6 * B_RES * B_RES
    const upliftForcing = new Float32Array(bN)
    {
      const _bDir     = new Vector3()
      const _bScratch: TectonicQuery = { plateId:0, neighborId:0, boundaryDist:0, convergence:0, shear:0, crustDist:0, paleoDist:0, otherCrustDist:0, baseElevation:0, rockHardness:0 }
      for (let face = 0; face < 6; face++) {
        for (let y = 0; y < B_RES; y++) {
          for (let x = 0; x < B_RES; x++) {
            const c = texelIndex(face, x, y, B_RES)
            texelToDir(face, x, y, B_RES, _bDir)
            const u = Math.max(0, tectonics.upliftAt(_bDir))
            tectonics.query(_bDir, _bScratch)
            const cd = _bScratch.crustDist
            // smoothstep(-0.05, 0.15, cd): ramp land only (cd > 0 = land/crust)
            const t = Math.max(0, Math.min(1, (cd - (-0.05)) / (0.15 - (-0.05))))
            const gate = t * t * (3 - 2 * t)
            upliftForcing[c] = u * gate
          }
        }
      }
    }

    // --- Step 3: bake B (iterated uplift+erode loop) ---------------------------
    if (DEBUG_TIMING) console.log(`pre-erosion (tectonics+uplift+sampler) ${(performance.now() - preT0).toFixed(1)}ms`)
    const bakeT0 = performance.now()
    const erosion = new Erosion({
      seed,
      heightFn: baseSampler0.heightFn,
      climate,
      tectonics,    // rock hardness modulates per-cell incision rate + talus angle
      oceanCoverage: d.oceanCoverage,
      res: this.erosionRes,
      K0:           EROSION_DEFAULT_K0 * this.erosionStrength,
      depositionG:  EROSION_DEFAULT_G  * this.erosionDeposition,
      upliftForcing,
      bSteps:       this.bSteps,
      bUpliftRate:  this.bUpliftRate,
    })
    if (DEBUG_TIMING) console.log(`B bake ${(performance.now() - bakeT0).toFixed(1)}ms`)

    // --- Step 3.5: bake Subsurface (geology layers) ----------------------------
    // Baked after erosion (needs erosion.accAt/lakeMaskAt/deltaAt) and before the
    // final sampler so we avoid a circular dependency on this.heightFn.
    // heightFn: use the same base sampler that erosion used — pre-orogenic-stamp
    // base elevations are sufficient for geology bake accuracy at SUB_BAKE_LEVEL=5.
    // seaLevel (provisional): computed from baseSampler0.heightFn via the same
    // Fibonacci hypsometry approach used post-sampler, but on the pre-erosion base.
    // This is provisional — the final seaLevel (used for water rendering) is computed
    // below on the eroded heightFn. The difference is small enough that geology
    // layers (water table, hydrocarbon) need only approximate coastline placement.
    let provisionalSeaLevel = 0
    {
      const N_HYPS     = 8192
      const LOD_HYPS   = 5
      const goldenAngle = Math.PI * (3 - Math.sqrt(5))
      const hypsHeights = new Float32Array(N_HYPS)
      const hypsDir    = new Vector3()
      for (let i = 0; i < N_HYPS; i++) {
        const y = 1 - (2 * i + 1) / N_HYPS
        const r = Math.sqrt(Math.max(0, 1 - y * y))
        const theta = goldenAngle * i
        hypsDir.set(r * Math.cos(theta), y, r * Math.sin(theta))
        hypsHeights[i] = baseSampler0.heightFn(hypsDir, LOD_HYPS)
      }
      hypsHeights.sort()
      const idx = Math.min(N_HYPS - 1, Math.floor(d.oceanCoverage * N_HYPS))
      provisionalSeaLevel = hypsHeights[idx]
    }
    const subsurfaceBakeT0 = performance.now()
    const subsurface = new Subsurface({
      seed,
      heightFn: baseSampler0.heightFn,
      climate,
      erosion,
      tectonics,
      derived: d,
      composition: this.interior.composition,
      res: this.erosionRes,
      seaLevel: provisionalSeaLevel,
    })
    if (DEBUG_TIMING) console.log(`Subsurface bake ${(performance.now() - subsurfaceBakeT0).toFixed(1)}ms`)

    // --- Step 4: rebuild final sampler WITH erosion AND bActive -----------------
    // bActive = true: orogenic stamps are gated off so bDelta owns the elevation budget.
    const postT0 = performance.now()
    const sampler = makeTerrainSampler({ ...samplerOpts, tectonics, climate, erosion, subsurface, bActive: true })

    this.tectonics    = sampler.tectonics
    this.climateSim   = sampler.climate
    this.heightFn     = sampler.heightFn
    this._plateColorFn = sampler.plateColorFn
    this._erosion     = sampler.erosion
    this._subsurface  = sampler.subsurface
    this.climateFn    = (dir: Vector3, height: number): ClimateSample =>
      this.climateSim.sample(dir, height, this._climateScratch)

    // --- Cave field (main-thread only; not shipped to workers) --------------------
    // Built from the final eroded heightFn so cave mouths sit on the real terrain surface.
    // The cave field is experimental and main-thread-only. A fault in it must NOT
    // abort buildHeightFn — everything AFTER this (sea-level hypsometry, worker-pool
    // init + ship, mesh building) would be skipped, leaving the planet with NO terrain
    // meshes = a black screen. Isolate it: on error, disable caves and keep building.
    try {
      this._caveField = makeCaveSampler({
        seed,
        heightFn: sampler.heightFn,
        radius: this.radius,
        heightScale: this.heightScale,
      })
    } catch (err) {
      console.error('[cave] makeCaveSampler failed in buildHeightFn — disabling caves; terrain unaffected:', err)
      this._caveField = null
    }

    // --- Hypsometry: compute seaLevel from oceanCoverage ----------------------
    // Runs on the ERODED heightFn so the waterline correctly reflects erosion.
    // Sample heightFn at N Fibonacci-sphere points at a coarse LOD level (5),
    // sort ascending, then pick the percentile matching oceanCoverage so that
    // exactly that fraction of the surface sits at or below the waterline.
    // N=8192 gives ~0.5° angular spacing — adequate for continent/ocean structure.
    // This runs once per regenerate on the main thread; workers receive the scalar.
    {
      const N = 8192
      const LOD_LEVEL = 5
      const goldenAngle = Math.PI * (3 - Math.sqrt(5)) // ~2.399963…
      const heights = new Float32Array(N)
      const fibDir = new Vector3()
      for (let i = 0; i < N; i++) {
        // Fibonacci sphere: uniform point distribution on unit sphere
        const y = 1 - (2 * i + 1) / N          // [-1, 1], biased toward ±1 endpoints
        const r = Math.sqrt(Math.max(0, 1 - y * y))
        const theta = goldenAngle * i
        fibDir.set(r * Math.cos(theta), y, r * Math.sin(theta))
        heights[i] = this.heightFn(fibDir, LOD_LEVEL)
      }
      heights.sort()
      const idx = Math.min(N - 1, Math.floor(d.oceanCoverage * N))
      this._seaLevel = heights[idx]
    }
    if (DEBUG_TIMING) console.log(`post-erosion (sampler+cave+hypsometry) ${(performance.now() - postT0).toFixed(1)}ms`)

    // Tear down any existing pool (handles both constructor first-run and regenerate).
    if (this.pool) {
      this.pool.dispose()
      this.pool = null
    }
    this.poolReady = false

    if (MeshWorkerPool.isSupported()) {
      const gen = ++this.poolGeneration
      this.pool = new MeshWorkerPool({
        seed,
        radius: this.radius,
        heightScale: this.heightScale,
        resolution: this.resolution,
        plateCount: d.plateCount,
        arcDensity: d.arcDensity,
        baseTemp: d.surfaceTemp,
        atmosphere: d.atmosphere,
        bandCount: this.deriveBandCount(),
        axialTiltRad: (this.axialTiltDeg * Math.PI) / 180,
        driftScale: d.driftScale,
        tectonics: this.tectonics.toBaked(),
        climate: this.climateSim.toBaked(),
        erosion: erosion.toBaked(),
        subsurface: subsurface.toBaked(),
        seaLevel: this._seaLevel,
        bActive: true,
      })
      this.pool.onResult = (key: string, arrays: ChunkMeshArrays) =>
        this.onWorkerResult(key, arrays, gen)
      this.pool.ready.then(() => {
        if (this.poolGeneration === gen) this.poolReady = true
      })
    }
  }

  /** plateColorFn for passing to buildChunkGeometry — set by buildHeightFn. */
  private _plateColorFn!: (dir: Vector3) => readonly [number, number, number]

  /** Build (or rebuild) TectonicsDebug and add it as a child. */
  private buildTectonicsDebug(seed: number): void {
    // buildHeightFn must have been called first so this.tectonics is ready.
    // We pass a planet-LOCAL surface sampler (raw heightFn path, not the world-space getSurfaceRadiusAt).
    const localScratch = new Vector3()
    this.tectonicsDebug = new TectonicsDebug(this.tectonics, {
      radius: this.radius,
      heightScale: this.heightScale,
      // Level 6 is sufficient coarse accuracy for arrow placement; avoids the cost
      // of 14-octave eval for purely decorative gizmo positioning.
      surfaceRadiusAt: (localDir: Vector3) => {
        localScratch.copy(localDir).normalize()
        return this.radius + this.heightFn(localScratch, 6) * this.heightScale
      },
    })
    this.tectonicsDebug.visible = false  // starts hidden
    this.add(this.tectonicsDebug)
  }

  /** Build (or rebuild) WindDebug and add it as a child. */
  private buildWindDebug(_seed: number): void {
    // climateSim must have been set by buildHeightFn before this is called.
    const localScratch = new Vector3()
    this.windDebug = new WindDebug(this, this.climateSim, {
      radius: this.radius,
      heightScale: this.heightScale,
      // Level 6 matches TectonicsDebug — accurate enough for arrow lift positioning.
      surfaceRadiusAt: (localDir: Vector3) => {
        localScratch.copy(localDir).normalize()
        return this.radius + this.heightFn(localScratch, 6) * this.heightScale
      },
      density: this._windArrowDensity,
    })
    this.windDebug.setVisible(false)  // starts hidden
    // WindDebug adds its InstancedMeshes directly to the scene passed in constructor,
    // which is `this` (the Planet Group). No explicit this.add() needed — WindDebug
    // calls scene.add(instancedArrows) on every mesh it creates.
  }

  /** Build (or rebuild) WindFlow and add its LineSegments as a child. */
  private buildWindFlow(): void {
    // climateSim must have been set by buildHeightFn before this is called.
    this.windFlow = new WindFlow(this, this.climateSim, {
      radius:      this.radius,
      heightScale: this.heightScale,
      density:     8000,
    })
    this.windFlow.setVisible(false)  // starts hidden
    // WindFlow adds its LineSegments directly to the scene passed in constructor,
    // which is `this` (the Planet Group). No explicit this.add() needed.
  }

  /** Build (or rebuild) VolumetricClouds and add its sphere mesh as a child. */
  private buildCloudShell(): void {
    // climateSim must have been set by buildHeightFn before this is called.
    this.cloudShell = new VolumetricClouds(this, this.climateSim, {
      radius:      this.radius,
      heightScale: this.heightScale,
      sunDir:      this._uSunDir,   // passed BY REFERENCE — live sun GUI updates propagate
    })
    this.cloudShell.setVisible(false)  // starts hidden
    // VolumetricClouds adds its Mesh directly to the scene passed in constructor,
    // which is `this` (the Planet Group). No explicit this.add() needed.
  }

  /** Build (or rebuild) Atmosphere and add its sphere mesh as a child. */
  private buildAtmosphere(): void {
    this.atmosphereShell = new Atmosphere(this, {
      radius:      this.radius,
      heightScale: this.heightScale,
      sunDir:      this._uSunDir,   // passed BY REFERENCE — live sun GUI updates propagate
    })
    this.atmosphereShell.setVisible(true)  // default ON — atmosphere is always-on appearance
    // Atmosphere adds its Mesh directly to the scene passed in constructor,
    // which is `this` (the Planet Group). No explicit this.add() needed.
  }

  private buildRoots(): void {
    this.roots = Array.from({ length: 6 }, (_, i) => new QuadtreeNode(i, 0, 0, 0, this.radius))
  }

  // ---------------------------------------------------------------------------
  // Internal: LOD selection
  // ---------------------------------------------------------------------------

  /**
   * Walk all 6 quadtrees and collect the desired leaf set.
   *
   * Split/merge decisions use the screen-space-error (SSE) metric:
   *   projPx = nodeSize * screenHeightPx / (2 * nearDist * tan(vFov/2))
   *
   * where nearDist is the camera distance to the nearest point on the node's
   * bounding sphere (not just its center). This ensures near-horizon chunks
   * with far-away centers still subdivide correctly when the player is at eye level.
   *
   * Split threshold: projPx > resolution * targetTriPx
   * Merge threshold: projPx < resolution * targetTriPx / (1 + HYSTERESIS)
   */
  private selectLeaves(cam: Vector3): void {
    // Pixel thresholds derived from targetTriPx and chunk resolution.
    const { splitThreshPx, mergeThreshPx } = this.lodThresholds()

    const desired = new Set<string>()

    // Collect desired leaves via recursive descent
    const collect = (node: QuadtreeNode): void => {
      // Frustum cull: if the node's bounding sphere is completely outside the
      // dilated local-space frustum, treat it as a coarse leaf and stop descending.
      // Dilation = nodeSize (one chunk width) so just-off-screen chunks pre-build,
      // keeping skirts seamless on camera turns.
      // The MERGE path (desiredWithChildren) and getSurfaceRadiusAt are NOT culled.
      if (this._frustumActive) {
        const center = node.surfaceCenter ?? node.worldCenter
        const radius = node.nodeSize * 0.7071 + node.nodeSize // half-diagonal + 1-chunk pad
        this._frustumSphere.set(center, radius)
        if (!this._localFrustum.intersectsSphere(this._frustumSphere)) {
          desired.add(node.key)
          return
        }
      }

      const projPx = this.computeProjPx(cam, node)
      const wantSplit = node.level < this.maxDepth && projPx > splitThreshPx

      if (!wantSplit) {
        // This node is a desired leaf
        desired.add(node.key)
        // If it still has children from a previous split but we don't want them,
        // schedule a merge (handled below outside collect)
        return
      }

      // We want to split.
      // If children don't exist yet, create the QuadtreeNode objects.
      if (node.children === null) {
        node.split(this.radius)
      }

      // Are all 4 child geometries ready?
      const allChildrenCached = node.children!.every(
        (c) => this.geoCache.get(c.key) !== undefined || this.visibleMeshes.has(c.key),
      )

      if (allChildrenCached || this.splitPending.has(node.key)) {
        // Either ready to swap or already waiting (children are being built)
        if (!allChildrenCached) {
          // Still pending — keep parent as leaf for now
          desired.add(node.key)
          // Ensure missing children are queued; enqueue with their SSE as priority.
          // Workers build them fast; collect() recurses deeper each frame as children arrive.
          for (const child of node.children!) {
            if (
              !this.geoCache.get(child.key) &&
              !this.visibleMeshes.has(child.key) &&
              !this.buildQueueSet.has(child.key)
            ) {
              this.enqueueBuild(child, this.computeProjPx(cam, child))
            }
          }
          this.splitPending.set(node.key, node)
          return
        }
        // All 4 children ready — recurse into them, clear split-pending
        this.splitPending.delete(node.key)
        for (const child of node.children!) collect(child)
      } else {
        // Children just created — enqueue all 4 and keep parent visible.
        // Workers build them fast; collect() recurses deeper each frame as children arrive.
        for (const child of node.children!) {
          if (!this.buildQueueSet.has(child.key)) {
            this.enqueueBuild(child, this.computeProjPx(cam, child))
          }
        }
        this.splitPending.set(node.key, node)
        desired.add(node.key) // parent stays visible
      }
    }

    for (const root of this.roots) collect(root)

    // --- Merge: nodes in desired whose children exist and are all leaves visible
    //     but projPx < mergeThreshPx (caller no longer wants split).
    const desiredWithChildren = (node: QuadtreeNode): void => {
      if (node.children === null) return

      const projPx = this.computeProjPx(cam, node)
      if (projPx < mergeThreshPx) {
        // Node is in desired as a leaf (no longer splits) but children still exist.
        const childKeys = node.children.map((c) => c.key)
        const allChildrenVisible = childKeys.every((k) => this.visibleMeshes.has(k))

        if (allChildrenVisible && !this.mergePending.has(node.key)) {
          // All children are rendered — do the normal visible-children merge path.
          const parentReady =
            this.geoCache.get(node.key) !== undefined || this.visibleMeshes.has(node.key)
          if (parentReady) {
            // Immediate swap: show parent, remove children
            this.addMesh(node)
            for (const c of node.children!) this.removeMesh(c.key)
            node.merge()
            return // subtree gone — nothing left to recurse into
          } else {
            // Enqueue parent build; keep children visible
            this.mergePending.set(node.key, { node, childKeys })
            if (!this.buildQueueSet.has(node.key)) this.enqueueBuild(node, projPx)
          }
        } else if (!allChildrenVisible) {
          // Some/all children are not visible.  Two cases:
          //   A) Phantom subtree: enqueueDeepPath created node objects that were
          //      never built (camera moved away before builds completed).
          //      Safe to reclaim immediately — no visible tiles are affected.
          //   B) Legitimate split-in-progress: projPx was HIGH when collect() ran,
          //      builds are pending.  We must NOT reclaim those.
          //
          // Discriminator: if projPx < mergeThreshPx (we are already in this
          // branch) AND none of the direct children are in buildQueueSet, then
          // this subtree is unwanted (phantom).  A wanted split always has
          // projPx > splitThreshPx (well above mergeThreshPx) and children
          // in buildQueueSet or splitPending.
          const noneQueued = childKeys.every((k) => !this.buildQueueSet.has(k))
          const noneInSplitPending = !this.splitPending.has(node.key)
          if (noneQueued && noneInSplitPending) {
            // Phantom subtree — reclaim it so we stop walking it every frame.
            // Cancel any in-flight worker jobs for these phantom child keys.
            for (const ck of childKeys) this.cancelBuild(ck)
            node.merge()
            return // subtree gone — nothing left to recurse into
          }
        }
      }

      // Only recurse when children still exist (merge() may have cleared them above).
      if (node.children) {
        for (const c of node.children) desiredWithChildren(c)
      }
    }

    for (const root of this.roots) desiredWithChildren(root)

    // Apply desired set: add newly wanted, remove no-longer-wanted
    for (const [key] of this.visibleMeshes) {
      if (!desired.has(key) && !this.isChildOfSplitPending(key) && !this.isMergePendingChild(key)) {
        this.removeMesh(key)
      }
    }

    for (const key of desired) {
      if (!this.visibleMeshes.has(key)) {
        // Find node for this key and add its mesh if geometry is ready
        const node = this.findNode(key)
        if (node) {
          const cached = this.geoCache.get(node.key)
          if (cached) {
            this.addMeshFromData(node, cached.data)
          } else if (!this.buildQueueSet.has(node.key)) {
            this.enqueueBuild(node)
          }
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Internal: split/merge promotion after builds complete
  // ---------------------------------------------------------------------------

  /**
   * Check splitPending nodes: if all 4 children now have geometry, perform swap.
   */
  private promoteReadySplits(): void {
    for (const [parentKey, parentNode] of this.splitPending) {
      if (parentNode.children === null) {
        this.splitPending.delete(parentKey)
        continue
      }
      const allReady = parentNode.children.every(
        (c) => this.geoCache.get(c.key) !== undefined || this.visibleMeshes.has(c.key),
      )
      if (allReady) {
        // Re-check that the split is still wanted at the current camera distance.
        // If the camera pulled back while children were building, promoting would
        // flash over-refined tiles for one frame.  Deleting the entry lets
        // selectLeaves re-decide this frame; it will re-enqueue the split next
        // frame if it is still wanted (projPx > splitThreshPx), so this cannot
        // cause a permanently stuck-coarse state.
        const { splitThreshPx } = this.lodThresholds()
        const currentProjPx = this.computeProjPx(this._camLocalScratch, parentNode)
        if (currentProjPx <= splitThreshPx) {
          // Cancel in-flight worker jobs for the children — we're not splitting any more.
          for (const child of parentNode.children) this.cancelBuild(child.key)
          this.splitPending.delete(parentKey)
          continue // don't promote; selectLeaves will re-decide this frame
        }
        // Swap: add children, remove parent
        for (const child of parentNode.children) this.addMesh(child)
        this.removeMesh(parentKey)
        this.splitPending.delete(parentKey)
      }
    }
  }

  /**
   * Check mergePending nodes: if parent geometry is now cached, re-validate that
   * children are still all leaves and the merge-distance condition still holds,
   * then swap children→parent. If invalid, discard the pending merge so the
   * selection loop re-decides naturally next frame.
   */
  private promoteReadyMerges(cam: Vector3): void {
    const { mergeThreshPx } = this.lodThresholds()

    for (const [parentKey, { node, childKeys }] of this.mergePending) {
      const parentCached = this.geoCache.get(parentKey)
      if (!parentCached) continue

      // Re-validate: children must all still be leaves (no grandchildren)
      const childrenStillLeaves =
        node.children !== null &&
        node.children.every((c) => c.children === null && this.visibleMeshes.has(c.key))

      // Re-validate: merge SSE condition must still hold (projPx still below merge threshold)
      const projPx = this.computeProjPx(cam, node)
      const mergeConditionHolds = projPx < mergeThreshPx

      if (!childrenStillLeaves || !mergeConditionHolds) {
        // Stale merge — discard; selection will re-decide next frame
        this.mergePending.delete(parentKey)
        continue
      }

      // Swap: add parent, remove children
      this.addMeshFromData(node, parentCached.data)
      for (const ck of childKeys) this.removeMesh(ck)
      node.merge()
      this.mergePending.delete(parentKey)
    }
  }

  // ---------------------------------------------------------------------------
  // Internal: build queue
  // ---------------------------------------------------------------------------

  private enqueueBuild(node: QuadtreeNode, priority = 0): void {
    this.buildQueue.push({ key: node.key, node, priority })
    this.buildQueueSet.add(node.key)
  }

  /**
   * Recursively pre-enqueue all build targets along the descent path from `node`
   * down to the desired target depth, WITHOUT waiting for each level to be built
   * first. This breaks the one-level-per-frame gating that caused slow surface
   * refinement.
   *
   * Only enqueues nodes that:
   *   - are not already cached or visible
   *   - are not already in the build queue
   *   - are at a level where projPx still exceeds splitThreshPx (would split)
   *   - are within maxDepth
   *
   * QuadtreeNode.split() is called eagerly to create the node objects needed to
   * know their keys/worldCenters for priority computation. This is safe — nodes
   * that are created but never built simply stay as phantom QuadtreeNode objects
   * (no GPU cost). If the camera moves away, selectLeaves will call merge() on
   * ancestors and the unreachable sub-tree is abandoned.
   *
   * Zero new per-frame allocations in the hot path: the scratch vectors used by
   * computeProjPx are already preallocated on `this`.
   */
  private enqueueDeepPath(node: QuadtreeNode, cam: Vector3, splitThreshPx: number): void {
    if (node.level >= this.maxDepth) return
    const projPx = this.computeProjPx(cam, node)
    if (projPx <= splitThreshPx) return

    // Ensure children exist as QuadtreeNode objects (no geometry built yet — just node metadata)
    if (node.children === null) {
      node.split(this.radius)
    }

    for (const child of node.children!) {
      // Enqueue build for this child if not already handled
      if (
        this.geoCache.get(child.key) === undefined &&
        !this.visibleMeshes.has(child.key) &&
        !this.buildQueueSet.has(child.key)
      ) {
        this.enqueueBuild(child, this.computeProjPx(cam, child))
      }
      // Recurse: if this child would also want to split, pre-enqueue its descendants too
      this.enqueueDeepPath(child, cam, splitThreshPx)
    }
  }

  private onWorkerResult(key: string, arrays: ChunkMeshArrays, gen: number): void {
    // Drop stale results from a pre-regenerate pool.
    if (gen !== this.poolGeneration) return
    // Clear from the queue set — the chunk is no longer pending/in-flight.
    this.buildQueueSet.delete(key)
    // Already cached or already displayed — nothing to do.
    if (this.geoCache.get(key) !== undefined || this.visibleMeshes.has(key)) return
    const { geometry, origin } = arraysToGeometry(arrays)
    this.geoCache.set(key, new CachedMeshData({ geometry, origin }))
  }

  private drainBuildQueue(): void {
    if (this.buildQueue.length === 0) return

    if (this.pool && this.poolReady) {
      // Worker path: hand all queued items to the pool, no main-thread meshing.
      // buildQueueSet entries stay set while in-flight; onWorkerResult clears them on completion.
      // Cap at 256 submits per frame to bound postMessage volume; excess stays in buildQueue.
      const SUBMIT_CAP = 256
      let submitted = 0
      // Sort highest-priority to front so the pool sees the most urgent work first.
      this.buildQueue.sort((a, b) => b.priority - a.priority)
      while (this.buildQueue.length > 0 && submitted < SUBMIT_CAP) {
        const item = this.buildQueue.shift()!
        // Already cached or visible — clean up queue set and skip.
        if (this.geoCache.get(item.key) !== undefined || this.visibleMeshes.has(item.key)) {
          this.buildQueueSet.delete(item.key)
          continue
        }
        // Submit to pool; pool dedups internally. buildQueueSet stays set until onWorkerResult fires.
        this.pool.submit({
          key: item.key,
          faceIndex: item.node.faceIndex,
          level: item.node.level,
          ix: item.node.ix,
          iy: item.node.iy,
          priority: item.priority,
        })
        submitted++
      }
      // Dispatch queued pool work to idle workers.
      this.pool.pump()
      this.lastBuildMs = 0
    } else {
      // Sync fallback: pool unsupported or not yet ready.
      // Sort highest-priority (largest projPx / closest to camera) to the front.
      this.buildQueue.sort((a, b) => b.priority - a.priority)

      const t0 = performance.now()
      let built = 0

      while (built < BUILD_BUDGET_PER_FRAME && this.buildQueue.length > 0) {
        // Enforce a wall-clock cap so a heavy frame doesn't stall.
        if (built > 0 && performance.now() - t0 > BUILD_BUDGET_MS) break

        const item = this.buildQueue.shift()!
        this.buildQueueSet.delete(item.key)

        // Already cached or already visible — skip.
        if (this.geoCache.get(item.key) !== undefined || this.visibleMeshes.has(item.key)) continue

        const data = buildChunkGeometry({
          faceIndex: item.node.faceIndex,
          level: item.node.level,
          ix: item.node.ix,
          iy: item.node.iy,
          resolution: this.resolution,
          radius: this.radius,
          heightScale: this.heightScale,
          heightFn: this.heightFn,
          plateColorFn: this._plateColorFn,
          climateFn: this.climateFn,
          erosion: this._erosion,
          subsurface: this._subsurface,
          hardnessFn: (dir) => this.tectonics.hardnessAt(dir),
        }, this._seaLevel)

        this.geoCache.set(item.key, new CachedMeshData(data))
        built++
      }

      this.lastBuildMs = performance.now() - t0
    }
  }

  // ---------------------------------------------------------------------------
  // Internal: mesh management
  // ---------------------------------------------------------------------------

  private addMesh(node: QuadtreeNode): void {
    if (this.visibleMeshes.has(node.key)) return
    const cached = this.geoCache.get(node.key)
    if (!cached) return
    this.addMeshFromData(node, cached.data)  // addMeshFromData deletes from cache
  }

  private addMeshFromData(node: QuadtreeNode, data: ChunkMeshData): void {
    if (this.visibleMeshes.has(node.key)) return
    // Remove from cache when promoting to visible — cache holds only non-displayed geometries
    this.geoCache.delete(node.key)
    const mesh = new Mesh(data.geometry)
    mesh.position.copy(data.origin)
    mesh.frustumCulled = true
    mesh.userData.key = node.key
    // Wireframe mode overrides view-mode material: pure unlit white edges.
    mesh.material = this.wireframeActive ? this.wireMaterial : this.materialFor(node.level)
    this.add(mesh)
    this.visibleMeshes.set(node.key, mesh)
    // Add vertex dots overlay if showVertices is on.
    if (this._showVertices) {
      this._addPoints(node.key, mesh)
    }
  }

  private cancelBuild(key: string): void {
    this.buildQueueSet.delete(key)
    this.pool?.cancel(key)
  }

  private removeMesh(key: string): void {
    const mesh = this.visibleMeshes.get(key)
    if (!mesh) return
    // Remove Points overlay first if present.
    const pts = this.visiblePoints.get(key)
    if (pts) {
      this.remove(pts)
      // Do NOT dispose pts.geometry — it is the shared chunk geometry still referenced by mesh.
      this.visiblePoints.delete(key)
    }
    this.remove(mesh)
    this.visibleMeshes.delete(key)
    // Shared materials are never disposed per-mesh. Return geometry to cache only.
    this.geoCache.return(key, new CachedMeshData({ geometry: mesh.geometry as BufferGeometry, origin: mesh.position.clone() }))
    // Cancel any in-flight worker job for this key (it's now displayed from cache — we don't need it).
    this.cancelBuild(key)
  }

  /** Create a Points overlay for a chunk mesh and add it to the scene group. */
  private _addPoints(key: string, mesh: Mesh): void {
    const pts = new Points(mesh.geometry, this.pointsMaterial)
    // Mirror the chunk mesh's position exactly so dots align with wireframe vertices.
    pts.position.copy(mesh.position)
    pts.frustumCulled = true
    this.add(pts)
    this.visiblePoints.set(key, pts)
  }

  // ---------------------------------------------------------------------------
  // Internal: climate material
  // ---------------------------------------------------------------------------

  /**
   * Compute the planet pole direction in world space from the axial tilt and
   * current world matrix. The pole is local +Y rotated by the planet's world
   * quaternion. Stored into `_uPoleAxis` for the climate material.
   *
   * Called once at construction and once on regenerate(). If the planet world
   * matrix changes (e.g. spin), the caller should call this again to keep the
   * uniform in sync.
   */
  private _updatePoleAxis(): void {
    // Planet local +Y (the pole in local space — axialTiltDeg is applied to the
    // planet group itself by the caller, so local +Y is always the pole).
    const pole = new Vector3(0, 1, 0)
    this.updateMatrixWorld()
    pole.transformDirection(this.matrixWorld)
    this._uPoleAxis.value.copy(pole)
  }

  /**
   * Recompute the pole-axis world-space uniform after external transforms (e.g.
   * after the axial tilt is set in main.ts). Forces the planet root's world matrix
   * to be current before recomputing so the tilt is reflected immediately.
   *
   * Call once after planet.rotation is set. No per-frame call needed — spin
   * rotates about the pole axis and leaves its world direction invariant.
   */
  refreshPoleAxis(): void {
    this._updatePoleAxis()
  }

  /**
   * Build the single climate MeshBasicNodeMaterial.
   *
   * Field selection uses an integer uniform (uField) so the same material instance
   * handles all three modes. setClimateField() mutates uField.value and re-swaps.
   *
   * All computation is in world space via positionWorld (same pattern as heightmapMaterial).
   */
  private _buildClimateMaterial(): MeshBasicNodeMaterial {
    // Integer uniform selects the active field: 0=temp, 1=moist, 2=insol.
    // Stored in mat.userData.uField so setClimateField() can mutate .value at runtime.
    const uField = uniform(0)

    // -------------------------------------------------------------------------
    // TSL nodes: geometry-derived values
    // -------------------------------------------------------------------------
    const RADIUS_VAL      = this.radius
    const HEIGHT_SCALE_VAL = this.heightScale

    // dir = normalize(positionWorld), r = |positionWorld|
    const pWorld = positionWorld
    const dir    = normalize(pWorld)
    const r      = pWorld.length()

    // heightFactor in [-1, 1]: 0 at sea level, +1 at full peak height
    const heightFactor = clamp(
      r.sub(RADIUS_VAL).div(HEIGHT_SCALE_VAL),
      -1,
      1,
    )

    // climInsol: annual-average latitude insolation (sqrt of sin²lat = |cosLat|, but here
    // the contract says sqrt(max(0, 1 - dot(dir,poleAxis)²)) which is |sinLat| = cosLat_from_equator)
    const dotPole     = dot(dir, this._uPoleAxis)
    const sinLatSq    = max(0, dotPole.mul(dotPole).oneMinus())
    const climInsol   = sqrt(sinLatSq)

    // instant: day-side insolation
    const instant = max(0, dot(dir, this._uSunDir))

    // blend: mix(instant, climInsol, R)
    const blend = mix(instant, climInsol, this._uRedistribution)

    // -------------------------------------------------------------------------
    // Temperature field (field 0)
    // -------------------------------------------------------------------------
    // T = baseTemp + greenhouse + tempRange*(blend - 0.5) - lapseRate*heightFactor
    // Note: this debug view is illustrative — its gradient (uTempRange) is intentionally
    // decoupled from the baked biome gradient in climate.ts; they serve different purposes.
    const T = this._uBaseTemp
      .add(this._uGreenhouse)
      .add(this._uTempRange.mul(blend.sub(0.5)))
      .sub(this._uLapseRate.mul(heightFactor))

    // Map T from [-60, 60] → [0, 1] for ramp input
    const tNorm = saturate(T.add(60).div(120))

    // Blue→cyan→green→yellow→red ramp (5 stops at t=0,0.25,0.5,0.75,1)
    const blue   = vec3(0.0, 0.0, 1.0)
    const cyan   = vec3(0.0, 1.0, 1.0)
    const green  = vec3(0.0, 0.8, 0.0)
    const yellow = vec3(1.0, 1.0, 0.0)
    const red    = vec3(1.0, 0.0, 0.0)

    // Blend across the 4 segments [0,0.25] [0.25,0.5] [0.5,0.75] [0.75,1]
    const t0 = saturate(tNorm.mul(4))                           // 0..1 over [0,0.25]
    const t1 = saturate(tNorm.sub(0.25).mul(4))                 // 0..1 over [0.25,0.5]
    const t2 = saturate(tNorm.sub(0.5).mul(4))                  // 0..1 over [0.5,0.75]
    const t3 = saturate(tNorm.sub(0.75).mul(4))                 // 0..1 over [0.75,1]
    const tempColor = mix(
      mix(mix(blue, cyan, t0), mix(cyan, green, t1), saturate(tNorm.mul(4).sub(1))),
      mix(mix(green, yellow, t2), mix(yellow, red, t3), saturate(tNorm.mul(4).sub(2))),
      saturate(tNorm.mul(2).sub(0.5)),
    )

    // -------------------------------------------------------------------------
    // Moisture field (field 1) — reads baked per-vertex float attribute
    // -------------------------------------------------------------------------
    const moist = attribute('climateMoist', 'float')
    // Brown(dry) → green(moist) → blue(wet) ramp
    const brown = vec3(0.55, 0.35, 0.10)
    const mGreen = vec3(0.15, 0.65, 0.15)
    const mBlue  = vec3(0.10, 0.40, 0.80)
    const moistColor = mix(
      mix(brown, mGreen, saturate(moist.mul(2))),
      mBlue,
      saturate(moist.sub(0.5).mul(2)),
    )

    // -------------------------------------------------------------------------
    // Insolation field (field 2) — grayscale of blend
    // -------------------------------------------------------------------------
    const insolColor = vec3(blend, blend, blend)

    // -------------------------------------------------------------------------
    // Field selection via integer uniform
    // We use select() chains: if uField==0 → temp, elif 1 → moist, else insol.
    // TSL select(cond, a, b) = cond ? a : b
    // -------------------------------------------------------------------------
    const uFieldNode = uField
    const colorNode  = uFieldNode.equal(0)
      .select(tempColor, uFieldNode.equal(1).select(moistColor, insolColor))

    const mat = new MeshBasicNodeMaterial()
    mat.colorNode = colorNode
    mat.vertexColors = false

    // Store the uField uniform reference so setClimateField can mutate it.
    ;(mat.userData as Record<string, unknown>).uField = uField

    return mat
  }

  // ---------------------------------------------------------------------------
  // Internal: wind material
  // ---------------------------------------------------------------------------

  /**
   * Build the single wind MeshBasicNodeMaterial.
   *
   * All computation is purely analytical in WORLD frame via positionWorld — no
   * per-vertex attributes needed.  Field selection mirrors _buildClimateMaterial:
   * an integer uniform (uWindField) picks flow(0)/speed(1)/direction(2).
   *
   * Math summary (WORLD frame):
   *   dir       = normalize(positionWorld)
   *   sinPhi    = dot(dir, poleAxis)                     // sin(latitude)
   *   cosPhi    = sqrt(1 - sinPhi²)
   *   cv        = cross(poleAxis, dir)
   *   east      = cv / max(length(cv), EPS)              // guarded — no NaN at poles
   *   north     = cross(dir, east)
   *   phi       = asin(clamp(sinPhi, -1, 1))
   *   effStr    = uWindStrength * (uTempRange / 55)      // ΔT coupling live in shader
   *   u = retrograde * (-sin(2N·phi)) * cosPhi * effStr // zonal (E-W)
   *   v = 0.3 * cos(2N·phi) * sin(2·phi) * effStr       // meridional (N-S)
   *   W = u*east + v*north;  speed = length(W)
   *   windDir   = W / max(speed, EPS)                   // guarded unit wind
   *   flow phase = fract( dot(positionWorld, windDir)*FREQ - uWindTime*speed*K )
   *               — streaks advect along the wind spatially (not just pulse in time)
   */
  private _buildWindMaterial(): MeshBasicNodeMaterial {
    // Small epsilon for guarded divides — keeps everything finite at singularities.
    const EPS = 1e-5

    // -------------------------------------------------------------------------
    // TSL geometry nodes
    // -------------------------------------------------------------------------
    const dir = normalize(positionWorld)

    // -------------------------------------------------------------------------
    // Latitude-derived scalars
    // -------------------------------------------------------------------------
    // sinPhi = dot(dir, poleAxis).  Near poles cosPhi→0, which naturally zeroes
    // the east tangent contribution before it can degenerate.
    const sinPhi = dot(dir, this._uPoleAxis)
    // cosPhi = sqrt(max(0, 1 - sinPhi²))
    const cosPhi = sqrt(max(0, sinPhi.mul(sinPhi).oneMinus()))
    // phi = asin(clamp(sinPhi, -1, 1))
    const phi = asin(clamp(sinPhi, -1, 1))

    // -------------------------------------------------------------------------
    // Tangent basis: east + north vectors on the sphere surface.
    // W1 — Pole singularity guard: normalize(cross(poleAxis, dir)) is NaN at the
    // poles (cross → zero vector).  Use a guarded divide instead so we get a
    // finite (near-zero) tangent everywhere.
    // -------------------------------------------------------------------------
    const cv   = cross(this._uPoleAxis, dir)
    const east  = cv.div(max(cv.length(), EPS))
    const north = cross(dir, east)

    // -------------------------------------------------------------------------
    // N1 — In-shader ΔT→strength coupling.
    // uWindStrength is a plain manual gain (default 1.0); uTempRange/55 scales
    // with the equator-pole thermal contrast so warmer planets blow harder.
    // Changing the climate tempRange slider now updates wind strength live.
    // -------------------------------------------------------------------------
    const effStrength = this._uWindStrength.mul(this._uTempRange.div(55.0))

    // -------------------------------------------------------------------------
    // Zonal (E-W) and meridional (N-S) wind components
    //   N = uWindBands (float)
    //   u = retrograde * (-sin(2N·phi)) * cosPhi * effStrength
    //   v = 0.3 * cos(2N·phi) * sin(2·phi) * effStrength
    // -------------------------------------------------------------------------
    const twoNPhi  = phi.mul(this._uWindBands).mul(2)
    const twoPhi   = phi.mul(2)

    const u = this._uRetrograde
      .mul(sin(twoNPhi).negate())
      .mul(cosPhi)
      .mul(effStrength)

    const v = cos(twoNPhi)
      .mul(sin(twoPhi))
      .mul(0.3)
      .mul(effStrength)

    // -------------------------------------------------------------------------
    // Wind vector and scalar speed
    // -------------------------------------------------------------------------
    const Wx = east.x.mul(u).add(north.x.mul(v))
    const Wy = east.y.mul(u).add(north.y.mul(v))
    const Wz = east.z.mul(u).add(north.z.mul(v))
    const W  = vec3(Wx, Wy, Wz)
    const speed = W.length()

    // -------------------------------------------------------------------------
    // Wind angle theta in the east/north tangent frame (atan2(v, u))
    // -------------------------------------------------------------------------
    const theta = atan2(v, u)

    // -------------------------------------------------------------------------
    // Hue-wheel helper: maps angle in radians to an RGB colour.
    // Uses 3 overlapping cosine segments spanning 0-2π so the wheel is smooth
    // and fully saturated.
    //   R = sat(cos(theta))
    //   G = sat(cos(theta - 2π/3))
    //   B = sat(cos(theta - 4π/3))
    // -------------------------------------------------------------------------
    const TWO_PI_OVER_3  = 2.0943951023931953  // 2π/3
    const FOUR_PI_OVER_3 = 4.1887902047863905  // 4π/3
    const hR = saturate(cos(theta))
    const hG = saturate(cos(theta.sub(TWO_PI_OVER_3)))
    const hB = saturate(cos(theta.sub(FOUR_PI_OVER_3)))
    const hueWheel = vec3(hR, hG, hB)

    // Neutral color used where there is effectively no wind.
    const NEUTRAL = vec3(0.1, 0.1, 0.1)

    // -------------------------------------------------------------------------
    // W2 — Speed-gate the hue to suppress false color at u=v=0 nodes.
    // At the equator and cell-boundary latitudes u and v can both be zero, so
    // atan2(0,0) resolves to 0 → hueWheel(0) paints a false red ring.
    // Gate: blend to NEUTRAL when speed < EPS.
    // -------------------------------------------------------------------------
    const hueGated = mix(NEUTRAL, hueWheel, step(EPS, speed))

    // -------------------------------------------------------------------------
    // Sub-mode 0: 'flow' — direction hue modulated by animated advecting streaks.
    // W3 — Genuine spatial advection along the wind direction.
    //   windDir = W / max(speed, EPS)   (guarded unit wind)
    //   phase   = fract( dot(positionWorld, windDir)*FREQ - uWindTime*speed*SPEEDK )
    // dot(positionWorld, windDir) is a real spatial coordinate (positionWorld is
    // the full-length world position, NOT the unit dir), so the pattern translates
    // along the wind vector over time.  Streaks crawl eastward in trade winds and
    // westward in westerly belts.
    // -------------------------------------------------------------------------
    const FLOW_FREQ  = 4e-4   // spatial frequency relative to world-space coords
    const ADVECT_K   = 1.2    // time-advection rate multiplier
    const windDir    = W.div(max(speed, EPS))   // guarded unit wind; W1-style divide
    const spatialArg = positionWorld.dot(windDir).mul(FLOW_FREQ)
    const phaseRaw   = spatialArg.sub(this._uWindTime.mul(speed).mul(ADVECT_K))
    const phase      = fract(phaseRaw)
    const streak     = saturate(phase.oneMinus().mul(8).sub(4).oneMinus())  // narrow bright band
    const flowBrightness = saturate(speed.mul(2)).mul(streak.mul(0.7).add(0.3))
    const flowColor  = hueGated.mul(flowBrightness)

    // -------------------------------------------------------------------------
    // Sub-mode 1: 'speed' — heatmap calm(blue)→fast(red)
    // -------------------------------------------------------------------------
    const speedNorm = saturate(speed.mul(2))
    const sBlue   = vec3(0.1, 0.2, 0.9)
    const sCyan   = vec3(0.0, 0.8, 0.8)
    const sYellow = vec3(1.0, 0.9, 0.0)
    const sRed    = vec3(1.0, 0.1, 0.0)
    const speedColor = mix(
      mix(sBlue, sCyan, saturate(speedNorm.mul(3))),
      mix(sYellow, sRed, saturate(speedNorm.sub(0.667).mul(3))),
      saturate(speedNorm.sub(0.333).mul(3)),
    )

    // -------------------------------------------------------------------------
    // Sub-mode 2: 'direction' — pure hue wheel, gated to neutral at no-wind nodes
    // -------------------------------------------------------------------------
    const dirColor = hueGated

    // -------------------------------------------------------------------------
    // Field selection (mirrors _buildClimateMaterial pattern)
    // -------------------------------------------------------------------------
    const colorNode = this._uWindField.equal(0)
      .select(flowColor, this._uWindField.equal(1).select(speedColor, dirColor))

    const mat = new MeshBasicNodeMaterial()
    mat.colorNode = colorNode
    mat.vertexColors = false
    return mat
  }

  // ---------------------------------------------------------------------------
  // Internal: hardness material
  // ---------------------------------------------------------------------------

  /**
   * Build the single hardness MeshBasicNodeMaterial for the materials view.
   *
   * Reads the baked per-vertex 'rockHardness' float attribute (0 = soft sediment,
   * 1 = hard igneous/volcanic) and maps it to a vivid turbo-style ramp for
   * maximum perceptual discrimination:
   *   0.00 → deep indigo  #2b00d4
   *   0.25 → cyan         #00e5ff
   *   0.50 → lime green   #39ff14
   *   0.75 → orange       #ff7700
   *   1.00 → crimson red  #dd0000
   *
   * Four segments, each covering 0.25 of the [0,1] range.  saturate() clamps
   * the attribute so skirt-copied border vertices don't escape the ramp.
   */
  private _buildHardnessMaterial(): MeshBasicNodeMaterial {
    const h = attribute('rockHardness', 'float')

    // Five stops — vivid, high-contrast, spans luminance AND hue
    const c0 = vec3(0x2b / 255, 0x00 / 255, 0xd4 / 255) // deep indigo  #2b00d4
    const c1 = vec3(0x00 / 255, 0xe5 / 255, 0xff / 255) // cyan         #00e5ff
    const c2 = vec3(0x39 / 255, 0xff / 255, 0x14 / 255) // lime green   #39ff14
    const c3 = vec3(0xff / 255, 0x77 / 255, 0x00 / 255) // orange       #ff7700
    const c4 = vec3(0xdd / 255, 0x00 / 255, 0x00 / 255) // crimson red  #dd0000

    // Per-segment parameter: each t_n runs 0→1 over its quarter of [0,1]
    const hClamped = saturate(h)
    const t0 = saturate(hClamped.mul(4))                   // 0..1 over [0.00, 0.25]
    const t1 = saturate(hClamped.sub(0.25).mul(4))         // 0..1 over [0.25, 0.50]
    const t2 = saturate(hClamped.sub(0.50).mul(4))         // 0..1 over [0.50, 0.75]
    const t3 = saturate(hClamped.sub(0.75).mul(4))         // 0..1 over [0.75, 1.00]

    // Chain four mix() calls — each overwrites the previous blend once its
    // segment starts, keeping the mapping continuous end-to-end.
    const colorNode = mix(mix(mix(mix(c0, c1, t0), c2, t1), c3, t2), c4, t3)

    const mat = new MeshBasicNodeMaterial()
    mat.colorNode = colorNode
    mat.vertexColors = false
    return mat
  }

  // ---------------------------------------------------------------------------
  // Internal: helpers
  // ---------------------------------------------------------------------------

  /**
   * Returns the shared material for a chunk at this level.
   * View mode priority: materials → climate → heightmap → tectonics → lodColors → normal.
   * Wind view uses the windDebug arrow overlay over normal terrain — no special mesh material.
   */
  private materialFor(level: number): MeshStandardNodeMaterial | MeshBasicMaterial | MeshBasicNodeMaterial {
    if (this.materialsViewActive) {
      return this.hardnessMaterial
    }
    if (this.climateViewActive) {
      return this.climateMaterial
    }
    if (this.heightmapViewActive) {
      return this.heightmapMaterial
    }
    if (this.tectonicsViewActive) {
      return this.plateColorMaterial
    }
    if (this.debugColorsActive) {
      return this.debugMaterials[Math.min(level, this.debugMaterials.length - 1)]
    }
    if (this.wetnessViewActive) {
      return this.wetnessMaterial
    }
    return this.normalMaterial
  }

  private levelFromKey(key: string): number {
    const parts = key.split('/')
    return parseInt(parts[1], 10)
  }

  /** Find a node in the quadtree by key (brute-force walk; only called when adding). */
  private findNode(key: string): QuadtreeNode | null {
    const parts = key.split('/')
    const faceIndex = parseInt(parts[0], 10)
    const targetLevel = parseInt(parts[1], 10)
    const targetIx = parseInt(parts[2], 10)
    const targetIy = parseInt(parts[3], 10)

    const search = (node: QuadtreeNode): QuadtreeNode | null => {
      if (node.faceIndex !== faceIndex) return null
      if (node.level === targetLevel && node.ix === targetIx && node.iy === targetIy) return node
      if (node.children === null) return null
      for (const c of node.children) {
        const r = search(c)
        if (r) return r
      }
      return null
    }

    return search(this.roots[faceIndex])
  }

  /**
   * Compute the screen-space-error projected-pixel size for a node.
   *
   * Uses a terrain-height-adjusted bounding sphere so that elevated chunks
   * (hills) are measured at their ACTUAL surface radius rather than sea-level.
   * surfaceCenter is computed once per node and cached — no per-frame heightFn calls.
   *
   * surfaceCenter = centerDir * (radius + heightFn(centerDir, level) * heightScale)
   * This is in planet-local space, the same frame as the local camera.
   *
   * boundRadius is the geometric half-diagonal of the patch (no elevation padding —
   * the surfaceCenter already measures distance to the real displaced surface).
   *
   * projPx = nodeSize * screenHeightPx / (2 * nearDist * tan(vFov/2))
   *
   * Zero-alloc: uses this._sseScratch.
   */
  private computeProjPx(cam: Vector3, node: QuadtreeNode): number {
    // Lazy-init: compute surfaceCenter once and cache it on the node.
    // h is recovered from the cached vector afterwards to avoid a second heightFn call.
    if (node.surfaceCenter === null) {
      const h = this.heightFn(node.centerDir, node.level)
      node.surfaceCenter = node.centerDir.clone().multiplyScalar(this.radius + h * this.heightScale)
    }

    const camToCenter = this._sseScratch.copy(cam).distanceTo(node.surfaceCenter)
    // Bounding sphere radius: half-diagonal of the square patch. No absolute-elevation
    // padding — that over-refined high terrain within a ~heightScale radius and starved
    // everything else. The surfaceCenter fix already measures to the real surface.
    const boundRadius = node.nodeSize * 0.7071067811865476 // Math.SQRT2 / 2
    const nearDist = Math.max(node.nodeSize * 0.01, camToCenter - boundRadius)
    // Guard: tan(vFov/2) could be 0 if vFov is degenerate
    const tanHalfFov = Math.tan(this._vFovRadians * 0.5)
    if (tanHalfFov < 1e-6) return 0
    return (node.nodeSize * this._screenHeightPx) / (2 * nearDist * tanHalfFov)
  }

  /** Check if a key belongs to a child of a splitPending parent (must stay visible). */
  private isChildOfSplitPending(key: string): boolean {
    for (const [, parent] of this.splitPending) {
      if (parent.children && parent.children.some((c) => c.key === key)) return true
    }
    return false
  }

  /** Check if a key belongs to a child in a mergePending set (must stay visible until parent ready). */
  private isMergePendingChild(key: string): boolean {
    for (const [, { childKeys }] of this.mergePending) {
      if (childKeys.includes(key)) return true
    }
    return false
  }

  // ---------------------------------------------------------------------------
  // Internal: LOD threshold helpers
  // ---------------------------------------------------------------------------

  /**
   * Single source of truth for LOD pixel thresholds.
   * Both selectLeaves and promoteReadyMerges read from here so they agree on
   * the exact split/merge band.
   */
  private lodThresholds(): { splitThreshPx: number; mergeThreshPx: number } {
    const splitThreshPx = this.resolution * this.targetTriPx
    const mergeThreshPx = splitThreshPx / (1 + HYSTERESIS)
    return { splitThreshPx, mergeThreshPx }
  }

  // ---------------------------------------------------------------------------
  // Diagnostic overlay
  // ---------------------------------------------------------------------------

  /** Create the #lod-diag overlay once and append it to document.body. */
  private _createDiagOverlay(): HTMLDivElement {
    const el = document.createElement('div')
    el.id = 'lod-diag'
    el.style.cssText = [
      'position:fixed',
      'top:8px',
      'left:8px',
      'z-index:99999',
      'font:20px/1.4 monospace',
      'color:#0f0',
      'background:rgba(0,0,0,0.72)',
      'padding:10px 14px',
      'border-radius:4px',
      'pointer-events:none',
      'white-space:pre',
      'user-select:none',
    ].join(';')
    document.body.appendChild(el)
    return el
  }

  /**
   * Recompute and display the LOD diagnostic overlay.
   * Called every update() invocation regardless of frozen state.
   * Throttled to every 5 frames to keep string-formatting cost negligible.
   *
   * @param cameraWorldPos  World-space camera position (same arg as update()).
   * @param vFovRadians     Vertical FOV in radians.
   * @param screenHeightPx  Drawing-buffer height in pixels.
   * @param isFrozen        Whether the planet is currently frozen.
   */
  private _updateDiagOverlay(
    cameraWorldPos: Vector3,
    vFovRadians: number,
    screenHeightPx: number,
    isFrozen: boolean,
  ): void {
    // Create element on first call; restore display if it was hidden by setDiagEnabled(false).
    if (this._diagEl === null) {
      this._diagEl = this._createDiagOverlay()
    } else if (this._diagEl.style.display === 'none') {
      this._diagEl.style.display = ''
    }

    // Throttle: only reformat string every 5 frames.
    this._diagFrame++
    if (this._diagFrame % 5 !== 0) return

    // Camera altitude above sphere surface (world-space length minus radius).
    const altWorld = cameraWorldPos.length() - this.radius

    // Planet-local camera position (recomputed here so we can show it even when frozen).
    // Re-use a temporary vector rather than polluting _camLocalScratch (which may not be
    // populated yet when frozen).
    const camLocalLen = cameraWorldPos.clone().applyMatrix4(this._invWorldMatrix).length()

    const vFovDeg = (vFovRadians * 180) / Math.PI

    const { splitThreshPx } = this.lodThresholds()
    const mergeThreshPx = splitThreshPx / (1 + HYSTERESIS)
    // targetTriPx drives splitThreshPx
    const targetTriPx = this.targetTriPx

    const stats = this.getStats()

    // projPx for roots[0] — the root-level projection reveals whether camera distance
    // actually reaches the metric (it should scale with 1/altWorld).
    const rootProjPx = this.computeProjPx(
      cameraWorldPos.clone().applyMatrix4(this._invWorldMatrix),
      this.roots[0],
    )

    const lines = [
      `frozen: ${isFrozen}`,
      `altWorld: ${altWorld.toFixed(0)} m`,
      `camLocalLen: ${camLocalLen.toFixed(0)}`,
      `screenH: ${screenHeightPx} px`,
      `vFovDeg: ${vFovDeg.toFixed(1)}`,
      `maxDepth: ${this.maxDepth}  targetTriPx: ${targetTriPx.toFixed(2)}  splitThreshPx: ${splitThreshPx.toFixed(1)}  mergeThreshPx: ${mergeThreshPx.toFixed(1)}`,
      `lod: ${stats.minLevel}..${stats.maxLevel}  (${stats.leaves} leaves)`,
      `rootProjPx: ${rootProjPx.toFixed(1)}`,
      `buildQueue: ${stats.pendingBuilds}  cached: ${stats.cached}`,
    ]

    this._diagEl.textContent = lines.join('\n')
  }
}
