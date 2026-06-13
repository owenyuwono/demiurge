import {
  BufferGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Points,
  PointsMaterial,
  Vector3,
  Matrix4,
  Quaternion,
} from 'three'
import { MeshBasicNodeMaterial } from 'three/webgpu'
import { attribute } from 'three/tsl'
import { createNoise3D, ridged, fbm, erosionFbm } from './noise'
import { Tectonics, TectonicQuery, boundaryRelief } from './tectonics'
import { Climate, ClimateSample } from './climate'
import { buildChunkGeometry, ChunkMeshData } from './ChunkMesher'
import { QuadtreeNode } from './QuadtreeNode'
import { TectonicsDebug } from './TectonicsDebug'
import { PlanetGizmos } from './PlanetGizmos'

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
  /** Mean surface temperature, °C-ish, for the climate model (default 15 = Earth). */
  baseTemp?: number
  /** Atmospheric thickness 0..1 — thick shrinks gradients, thin makes extremes (default 0.6). */
  atmosphere?: number
  /** Rotation period in seconds — drives the circulation band count (default 600 = Earth-like → 3 cells). */
  rotationPeriodS?: number
  /** Axial tilt in degrees — climate uses it for the > 54° insolation inversion (default 23.4). */
  axialTiltDeg?: number
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
const LRU_CAPACITY = 4096         // hold a larger fine working set without evicting visible chunks
const HYSTERESIS = 0.15 // 15% — SSE merge fires at threshold * (1 + HYSTERESIS)
const EPS_DIST = 0.1    // minimum camera-to-node distance (prevents div-by-zero at contact)

// Tectonic heightFn tuning constants
// Note: boundary profile constants live in boundaryRelief() in tectonics.ts
// Crust-keyed base height constants
const TECT_LAND_BASE_0         = 0.06   // coastline base
const TECT_LAND_BASE_SS        = 0.17   // smoothstep gain inland (interiors climb to ~0.23+)
const TECT_LAND_PLATE_MOD      = 0.15   // plate.baseElevation modifier on land
const TECT_OCEAN_BASE_0        = -0.42
const TECT_OCEAN_DEPTH_AMP     = 0.10   // abyssal deepening
const TECT_OCEAN_DEPTH_SAT     = 0.45   // saturation distance (radians)
const TECT_OCEAN_PLATE_MOD     = 0.10   // plate.baseElevation modifier on ocean
const TECT_SHELF_W_PASSIVE     = 0.12   // wide passive shelf
const TECT_SHELF_W_ACTIVE      = 0.045  // active margin shelf — narrower than passive (2.7× contrast)
const TECT_SHELF_W_STAGNANT    = 0.045  // stagnant-lid shelf — narrow (no wide passive margin tectonics)
const TECT_COAST_LERP_HI       = 0.018  // crust→ocean transition edge hi [was 0.012 — gentler coast ramp trims the 120m base step where a breaching cone coincides]
// Highlands (signed fbm: plateaus AND basins)
const TECT_HIGHLANDS_AMP       = 0.12   // [was 0.42] de-saturation pass — flatter interiors raise mountain prominence
const TECT_HIGHLANDS_FREQ      = 1.9
const TECT_HIGHLANDS_OCT       = 5
const TECT_HIGHLANDS_SS_LO     = 0.012
const TECT_HIGHLANDS_SS_HI     = 0.10
// Paleo-orogens (fossil ranges)
const TECT_PALEO_AMP           = 0.16   // tectonic regime (16-plate path) — original value
const TECT_PALEO_AMP_STAGNANT  = 1.10   // stagnant-lid regime only (plateCount 0 or 1) — higher because boundaryRelief=0
const TECT_PALEO_SIGMA         = 0.10   // gaussian half-width in radians
const TECT_PALEO_FREQ          = 2.6
const TECT_PALEO_OCT           = 5
const TECT_PALEO_SS_LO         = 0.012
const TECT_PALEO_SS_HI         = 0.08
// Elevation-coupled interior detail
const TECT_HILL_AMP_BASE       = 0.10
const TECT_HILL_AMP_RANGE      = 0.10
const TECT_HILL_SS_LO          = 0.05
const TECT_HILL_SS_HI          = 0.30
const TECT_DETAIL_FBM_SCALE    = 3.2    // fbm frequency scale (unchanged)
const TECT_DETAIL_RIDGE        = 0.05   // ridged amplitude on land only [was 0.07 — trims coast-adjacent step for GATE X 120m]
const TECT_DETAIL_RIDGE_SCALE  = 7.0    // ridged frequency scale (unchanged)
// Offshore island / skerry band
const TECT_ISLAND_AMP          = 0.12
const TECT_ISLAND_FREQ         = 14.0
const TECT_ISLAND_OCT          = 4
const TECT_ISLAND_THRESH       = 0.55
const TECT_ISLAND_COAST_LO     = -0.05  // crustDist band lo
const TECT_ISLAND_COAST_HI     = -0.005 // crustDist band hi

// Arc-volcano headroom blend: volcano cone is attenuated by (1 - smoothstep(LO,HI,terrainH))
// so summits on already-high terrain stay well below the clamp (visible crater, no crease),
// while cones over low terrain (ocean → islands) keep full strength and still breach.
// HI lowered to ~0.50 (terrain never approaches 1.0 after de-saturation, so the old
// HI=1.0 barely engaged) — a cone on a 0.25-high platform is now ~70% strength, on a
// 0.40+ cordillera nearly gone, while abyssal/ocean cones (terrainH < LO) are untouched.
const TECT_VOLC_HEADROOM_LO    = 0.15
const TECT_VOLC_HEADROOM_HI    = 0.50
// Cap on the summed volcano-cone contribution. A single breaching cone (≤ ~0.64) is
// well under this; the cap only bites where multiple cones overlap, preventing stacked
// summits from blowing past the [-1,1] clamp (the H' max-saturation outlier).
const TECT_VOLC_SUM_MAX        = 0.64

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

  // Climate knobs — stored, applied on the next regenerate() (which rebuilds the Climate).
  private baseTemp: number
  private atmosphere: number
  private rotationPeriodS: number
  private readonly axialTiltDeg: number

  private heightFn!: (dir: Vector3, level: number) => number

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
  private readonly normalMaterial: MeshStandardMaterial
  private readonly debugMaterials: MeshBasicMaterial[]
  /** Plate-color node material (flat, unlit, reads 'plateColor' attribute). Created once. */
  private readonly plateColorMaterial: MeshBasicNodeMaterial
  /** Pure unlit wireframe — white edges only, no fill, no lighting. */
  private readonly wireMaterial: MeshBasicMaterial
  /** Shared points material for vertex overlay — yellow constant-size screen-space dots. */
  private readonly pointsMaterial: PointsMaterial
  private debugColorsActive = false
  private tectonicsViewActive = false
  private wireframeActive = false
  private _showVertices = false

  // Cached inverse world matrix + quaternion for rotation-safe transforms
  // Re-computed each update() call.
  private readonly _invWorldMatrix = new Matrix4()
  private readonly _invWorldQuat = new Quaternion()
  // Scratch vectors — preallocated, zero allocations in hot paths
  private readonly _camLocalScratch = new Vector3()
  private readonly _dirLocalScratch = new Vector3()
  // SSE scratch: holds (camPos - nodeCenter) for nearest-point distance computation.
  private readonly _sseScratch = new Vector3()

  // Stats
  private lastBuildMs = 0

  // Freeze flag
  private frozen = false

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
    this.targetTriPx = opts.targetTriPx ?? 2.5
    this.baseTemp = opts.baseTemp ?? 15
    this.atmosphere = opts.atmosphere ?? 0.6
    this.rotationPeriodS = opts.rotationPeriodS ?? 600
    this.axialTiltDeg = opts.axialTiltDeg ?? 23.4
    // Safe fallbacks — caller updates these on the first update() call.
    this._vFovRadians = Math.PI / 3  // 60°
    this._screenHeightPx = 1080

    // Shared normal-view material — one instance for all terrain chunks.
    this.normalMaterial = new MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 })

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

    // Pure unlit wireframe: white edges only, no fill, no lighting.
    // When wireframe mode is on, every chunk renders with this material regardless of view mode.
    this.wireMaterial = new MeshBasicMaterial({ color: 0xffffff, wireframe: true })

    // Vertex dot overlay: yellow constant screen-space dots (sizeAttenuation false keeps
    // them at a fixed pixel size at any zoom level).
    this.pointsMaterial = new PointsMaterial({ color: 0xffff00, size: 2.5, sizeAttenuation: false })

    this.buildHeightFn(opts.seed)
    this.buildTectonicsDebug(opts.seed)

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
  update(cameraWorldPos: Vector3, vFovRadians: number, screenHeightPx: number): void {
    if (this.frozen) return

    // Store for use in SSE metric throughout this frame.
    this._vFovRadians = vFovRadians
    this._screenHeightPx = screenHeightPx

    // Cache inverse world transform once per frame (rigid body: translate + rotate only).
    this.updateMatrixWorld()
    this._invWorldMatrix.copy(this.matrixWorld).invert()
    // Extract inverse rotation quaternion for direction transforms (no translation needed).
    this.getWorldQuaternion(this._invWorldQuat).invert()

    // Convert camera world position → planet-local position (zero alloc via scratch).
    this._camLocalScratch.copy(cameraWorldPos).applyMatrix4(this._invWorldMatrix)

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
    // Mutual exclusivity: turning on LOD colors turns off tectonics view.
    if (on && this.tectonicsViewActive) {
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
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
    // Mutual exclusivity: turning on tectonics turns off LOD debug colors.
    if (on && this.debugColorsActive) {
      this.debugColorsActive = false
    }
    // Wireframe overrides view-mode material; only swap when wireframe is off.
    if (!this.wireframeActive) {
      for (const [, mesh] of this.visibleMeshes) {
        mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
      }
    }
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
   * Set the mean surface temperature (°C-ish) for the climate model.
   * Stored; applied on the next regenerate(seed) (which rebuilds the Climate).
   */
  setBaseTemp(t: number): void {
    this.baseTemp = t
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

    // Preserve tectonics debug visibility across regeneration.
    const debugWasVisible = this.tectonicsDebug.visible

    // Dispose existing TectonicsDebug (remove from this Group, free GPU resources).
    this.tectonicsDebug.dispose()
    this.remove(this.tectonicsDebug)

    // Rebuild
    ;(this as { seed: number }).seed = seed
    this.buildHeightFn(seed)
    this.buildTectonicsDebug(seed)
    this.tectonicsDebug.visible = debugWasVisible

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

  getStats(): Stats {
    let minLevel = Infinity
    let maxLevel = 0
    for (const [key] of this.visibleMeshes) {
      const lv = this.levelFromKey(key)
      if (lv < minLevel) minLevel = lv
      if (lv > maxLevel) maxLevel = lv
    }
    if (this.visibleMeshes.size === 0) minLevel = 0
    return {
      leaves: this.visibleMeshes.size,
      cached: this.geoCache.size,
      minLevel,
      maxLevel,
      pendingBuilds: this.buildQueue.length,
      lastBuildMs: this.lastBuildMs,
      plates: this.tectonics.plates.length,
      volcanoes: this.tectonics.volcanoes.length,
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
    this.wireMaterial.dispose()
    this.pointsMaterial.dispose()
    this.tectonicsDebug.dispose()
    this.gizmos.dispose()
    this.climateSim.dispose()
  }

  // ---------------------------------------------------------------------------
  // Internal: initialisation
  // ---------------------------------------------------------------------------

  /**
   * Build heightFn and tectonics from seed. Both are rebuilt together so they
   * share a single source of truth: the same heightFn closure serves the mesher
   * and getSurfaceRadiusAt via the same path.
   */
  private buildHeightFn(seed: number): void {
    const noise = createNoise3D(seed)

    this.tectonics = new Tectonics({ seed, plateCount: this.plateCount, arcDensity: this.arcDensity })
    const plates = this.tectonics.plates

    // Regime selection — hoisted out of the per-vertex closure (branch-free in hot path).
    // plateCount===0 and ===1 both produce plates.length===1 (the non-tectonic stagnant-lid case).
    const isStagnantLid = plates.length === 1
    const paleoAmp    = isStagnantLid ? TECT_PALEO_AMP_STAGNANT  : TECT_PALEO_AMP
    const shelfWBase  = isStagnantLid ? TECT_SHELF_W_STAGNANT    : TECT_SHELF_W_PASSIVE

    // One scratch TectonicQuery per heightFn call — safe because heightFn is called serially.
    const scratch: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0 }

    // Separate scratch for plateColorFn (both called back-to-back on the same dir — see memo below).
    // 1-entry memoization: if dir.x/y/z identical to the previous heightFn call, reuse the
    // query result for plateColorFn, halving Voronoi cost under serial meshing.
    // The memo now stores the full query so plateColorFn can read boundaryDist/convergence/shear.
    let memoX = NaN, memoY = NaN, memoZ = NaN
    let memoBoundaryDist = 0, memoConvergence = 0, memoShear = 0, memoPlateId = 0
    const scratchColor: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0 }
    // Preallocated scratch tuple for plateColorFn — zero allocations in the hot path.
    const _colorScratch: [number, number, number] = [0, 0, 0]

    // Smoothstep helper (in-scope for heightFn closure)
    const _ss3 = (e0: number, e1: number, x: number): number => {
      const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)))
      return t * t * (3 - 2 * t)
    }

    // ridgedFn wraps ridged() for boundaryRelief's ridgedAt callback (captures the noise instance)
    const ridgedFn = (d: Vector3, freq: number, octaves: number): number =>
      ridged(noise, d.x * freq, d.y * freq, d.z * freq, { octaves })

    // erosionFbmFn wraps erosionFbm for belt/cordillera texture terms in boundaryRelief
    // erosionFbm outputs ≈[-1,1]; map to [0,1] like ridged does
    const erosionFbmFn = (d: Vector3, freq: number, octaves: number): number => {
      const ev = erosionFbm(noise, d.x * freq, d.y * freq, d.z * freq, { octaves, erosion: 8 })
      return ev * 0.5 + 0.5
    }

    // ---------------------------------------------------------------------------
    // Level-adaptive detail octave count:
    //
    //   LOD level → fbm octaves    → ridged octaves
    //   0–4       → 6              → 4
    //   5         → 7              → 4    (clamp(5-2,4,10)=4)
    //   6         → 8              → 4
    //   7         → 9              → 5
    //   8         → 10             → 6
    //   9         → 11             → 7
    //   10        → 12             → 8
    //   11        → 13             → 9
    //   12        → 14             → 10
    //
    // LOD-consistency guarantee (no low-frequency popping between levels):
    //
    //   fbm() divides by maxAmp = Σ_{i=0}^{N-1} 0.5^i — the normalization constant
    //   changes with octave count, which would shift low-frequency content between
    //   LOD levels and cause visible terrain popping. To prevent this, detail FBM
    //   is computed as:
    //
    //     detail_fbm = fbm6_base + Σ_{o=7..N} amp_o · noise_o(dir)
    //
    //   where fbm6_base uses the FIXED 6-octave normalization (maxAmp6 = 63/32),
    //   and each extra octave o contributes amp_o = (0.5^(o-1)) / maxAmp6 —
    //   exactly what fbm() would contribute if normalization were held constant.
    //   This ensures octaves 1-6 are byte-identical regardless of LOD level;
    //   only octaves 7..N are ADDITIVE on top.
    //
    //   Similarly ridged() is computed with a fixed 4-octave norm for the base
    //   plus additive extra octaves.
    // ---------------------------------------------------------------------------

    // Precompute fixed normalization constants (sum of gains for base octave counts)
    const FBM_BASE_OCTAVES    = 6
    const RIDGED_BASE_OCTAVES = 4
    const FBM_GAIN    = 0.5
    const FBM_LAC     = 2.0
    // maxAmp for a geometric series: Σ_{i=0}^{N-1} gain^i = (1 - gain^N) / (1 - gain)
    // For gain=0.5, N=6: 1+0.5+0.25+0.125+0.0625+0.03125 = 1.96875 = 63/32
    let _maxAmpFbm6 = 0; { let a = 1; for (let i = 0; i < FBM_BASE_OCTAVES; i++) { _maxAmpFbm6 += a; a *= FBM_GAIN; } }
    let _maxAmpRidged4 = 0; { let a = 1; for (let i = 0; i < RIDGED_BASE_OCTAVES; i++) { _maxAmpRidged4 += a; a *= FBM_GAIN; } }
    const MAX_AMP_FBM6    = _maxAmpFbm6     // ≈ 1.96875
    const MAX_AMP_RIDGED4 = _maxAmpRidged4  // = 0.9375

    this.heightFn = (dir: Vector3, level: number): number => {
      const x = dir.x, y = dir.y, z = dir.z

      this.tectonics.query(dir, scratch)
      const own = plates[scratch.plateId]
      const c = scratch.crustDist  // signed SDF: + = crust/land, - = ocean

      // Memoize last queried dir for plateColorFn re-use (full query state)
      memoX = x; memoY = y; memoZ = z
      memoPlateId = scratch.plateId
      memoBoundaryDist = scratch.boundaryDist
      memoConvergence = scratch.convergence
      memoShear = scratch.shear

      // --- Ocean base (unchanged) ---
      const oceanBase = TECT_OCEAN_BASE_0
                      - TECT_OCEAN_DEPTH_AMP * Math.min(1, Math.sqrt(Math.max(0, -c) / TECT_OCEAN_DEPTH_SAT))
                      + own.baseElevation * TECT_OCEAN_PLATE_MOD

      // --- Land base: interiors climb to ~0.19+ ---
      const landBase = TECT_LAND_BASE_0
                     + TECT_LAND_BASE_SS * _ss3(0, 0.30, c)
                     + own.baseElevation * TECT_LAND_PLATE_MOD

      // --- Highlands: signed fbm gives plateaus AND basins; inland only ---
      const highlands = TECT_HIGHLANDS_AMP
                      * fbm(noise, x * TECT_HIGHLANDS_FREQ, y * TECT_HIGHLANDS_FREQ, z * TECT_HIGHLANDS_FREQ,
                            { octaves: TECT_HIGHLANDS_OCT })
                      * _ss3(TECT_HIGHLANDS_SS_LO, TECT_HIGHLANDS_SS_HI, c)

      // --- Paleo-orogens: worn belts along fossil boundaries ---
      const paleoEnv  = paleoAmp
                      * Math.exp(-((scratch.paleoDist / TECT_PALEO_SIGMA) ** 2))
                      * (0.45 + 0.55 * erosionFbm(noise,
                            x * TECT_PALEO_FREQ, y * TECT_PALEO_FREQ, z * TECT_PALEO_FREQ,
                            { octaves: TECT_PALEO_OCT, erosion: 8 }))
                      * _ss3(TECT_PALEO_SS_LO, TECT_PALEO_SS_HI, c)

      // --- Shelf width narrows at active margins ---
      const activeness = (1 - _ss3(0.02, 0.06, scratch.boundaryDist))
                       * _ss3(0.08, 0.25, Math.max(Math.abs(scratch.convergence), Math.abs(scratch.shear)))
      const shelfW = shelfWBase * (1 - activeness) + TECT_SHELF_W_ACTIVE * activeness

      // --- Base elevation: blend ocean/land across the shelf ---
      const combinedLand = landBase + highlands + paleoEnv
      const base = combinedLand + (oceanBase - combinedLand) * (1 - _ss3(-shelfW, TECT_COAST_LERP_HI, c))

      // --- Boundary relief profile (asymmetric, all regimes) ---
      const relief = boundaryRelief(scratch, plates, dir, ridgedFn, erosionFbmFn)

      // --- Interior detail: elevation-coupled hills (replaces flat interiorDamp suppression) ---
      const hillAmp = TECT_HILL_AMP_BASE + TECT_HILL_AMP_RANGE * _ss3(TECT_HILL_SS_LO, TECT_HILL_SS_HI, combinedLand)

      // LOD-adaptive octave counts (clamped to [base, max])
      const fbmOctaves    = Math.min(Math.max(level + 2, FBM_BASE_OCTAVES),    14)
      const ridgedOctaves = Math.min(Math.max(level - 2, RIDGED_BASE_OCTAVES), 10)

      // ---- Detail FBM: additive-octave formulation for LOD consistency --------
      let fbmValue = 0; let fbmAmp = 1;
      const fx = x * TECT_DETAIL_FBM_SCALE, fy = y * TECT_DETAIL_FBM_SCALE, fz = z * TECT_DETAIL_FBM_SCALE;
      let fbmFreq = 1;
      for (let o = 0; o < fbmOctaves; o++) {
        fbmValue += fbmAmp * noise(fx * fbmFreq, fy * fbmFreq, fz * fbmFreq);
        fbmAmp   *= FBM_GAIN;
        fbmFreq  *= FBM_LAC;
      }
      const detailFbm = hillAmp * (fbmValue / MAX_AMP_FBM6)

      // ---- Detail ridged: near coasts only ----
      let ridgedValue = 0; let ridgedAmp = 1; let ridgedFreq = 1;
      const rx = x * TECT_DETAIL_RIDGE_SCALE, ry = y * TECT_DETAIL_RIDGE_SCALE, rz = z * TECT_DETAIL_RIDGE_SCALE;
      for (let o = 0; o < ridgedOctaves; o++) {
        const n = noise(rx * ridgedFreq, ry * ridgedFreq, rz * ridgedFreq);
        ridgedValue += ridgedAmp * (1.0 - Math.abs(n)) ** 2;
        ridgedAmp   *= FBM_GAIN;
        ridgedFreq  *= FBM_LAC;
      }
      const detailRidged = TECT_DETAIL_RIDGE * (ridgedValue / MAX_AMP_RIDGED4) * _ss3(0, 0.08, c) * 2
                         * (0.7 + 0.3 * Math.exp(-scratch.boundaryDist / 0.18))

      const detail = detailFbm + detailRidged

      // --- Offshore skerries / archipelagos ---
      const islandBand = _ss3(TECT_ISLAND_COAST_LO, TECT_ISLAND_COAST_LO * 0.5 + TECT_ISLAND_COAST_HI * 0.5, c)
                       * (1 - _ss3(TECT_ISLAND_COAST_HI * 0.5 + TECT_ISLAND_COAST_LO * 0.5, TECT_ISLAND_COAST_HI, c))
      let islandH = 0
      if (islandBand > 0.01) {
        const iRaw = fbm(noise, x * TECT_ISLAND_FREQ, y * TECT_ISLAND_FREQ, z * TECT_ISLAND_FREQ,
                         { octaves: TECT_ISLAND_OCT })
        const iExcess = Math.max(0, iRaw - TECT_ISLAND_THRESH) / (1 - TECT_ISLAND_THRESH)
        islandH = Math.min(TECT_ISLAND_AMP, TECT_ISLAND_AMP * iExcess) * islandBand
      }

      // --- Arc volcanoes: headroom-aware additive blend ---
      // The cone+crater term is ADDED on top of terrain that, on a continental
      // cordillera (BR_CORD_HEIGHT = 1.30), is already near or above the clamp.
      // A raw add-then-clamp flat-tops the summit (peak pinned at 1.0), erases the
      // crater rim/dip, and leaves a clamp-edge crease where the flat top meets the
      // cone flank (the source of the GATE X 120 m discontinuity on seed 1).
      //
      // Fix: attenuate the volcano by the terrain's remaining headroom. Over low
      // terrain (ocean → islands) the cone is full strength (islands still breach);
      // where terrain is already high it is faded out, keeping summits below the
      // clamp (~0.88–0.97) so the crater stays visible and no crease forms.
      const terrainH = base + relief + detail + islandH      // pre-clamp terrain
      const volcano  = Math.min(TECT_VOLC_SUM_MAX, this.tectonics.volcanoElevation(dir))
      const headroom = 1 - _ss3(TECT_VOLC_HEADROOM_LO, TECT_VOLC_HEADROOM_HI, terrainH)
      return Math.max(-1, Math.min(1, terrainH + volcano * headroom))
    }

    // plateColorFn: uses the 1-entry memo to avoid a redundant Voronoi query when
    // the mesher calls heightFn then plateColorFn for the same dir back-to-back.
    // Near tectonic boundaries, blends a regime tint (red=collision, blue=rift, yellow=shear).
    this._plateColorFn = (dir: Vector3): readonly [number, number, number] => {
      let pid: number
      let bd: number, conv: number, sh: number
      if (dir.x === memoX && dir.y === memoY && dir.z === memoZ) {
        pid = memoPlateId
        bd = memoBoundaryDist
        conv = memoConvergence
        sh = memoShear
      } else {
        this.tectonics.query(dir, scratchColor)
        pid = scratchColor.plateId
        bd = scratchColor.boundaryDist
        conv = scratchColor.convergence
        sh = scratchColor.shear
      }
      const base = plates[pid].color
      // Regime tint near boundaries
      if (bd < 0.035) {
        const t = 1 - _ss3(0, 0.035, bd)
        let tr: number, tg: number, tb: number
        if (conv > 0.12) {
          tr = 0.95; tg = 0.18; tb = 0.12
        } else if (conv < -0.12) {
          tr = 0.15; tg = 0.35; tb = 0.95
        } else if (Math.abs(sh) > 0.18) {
          tr = 0.95; tg = 0.85; tb = 0.15
        } else {
          // Quiet boundary — no tint
          _colorScratch[0] = base[0]; _colorScratch[1] = base[1]; _colorScratch[2] = base[2]
          return _colorScratch
        }
        const blend = 0.65 * t
        _colorScratch[0] = base[0] * (1 - blend) + tr * blend
        _colorScratch[1] = base[1] * (1 - blend) + tg * blend
        _colorScratch[2] = base[2] * (1 - blend) + tb * blend
        return _colorScratch
      }
      _colorScratch[0] = base[0]; _colorScratch[1] = base[1]; _colorScratch[2] = base[2]
      return _colorScratch
    }

    // ---- Climate ----------------------------------------------------------
    // Built AFTER heightFn + tectonics exist (it depends on both). Color-only:
    // climate never feeds back into heightFn, so terrain geometry is unchanged.
    // bandCount comes from rotation; axial tilt feeds the > 54° inversion.
    // crustDistAt reuses a dedicated scratch query so it never clobbers the
    // heightFn / plateColorFn scratch above.
    const climateQueryScratch: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0 }
    this.climateSim = new Climate({
      seed,
      baseTemp: this.baseTemp,
      atmosphere: this.atmosphere,
      bandCount: this.deriveBandCount(),
      axialTiltRad: (this.axialTiltDeg * Math.PI) / 180,
      heightFn: this.heightFn,
      crustDistAt: (dir: Vector3): number => this.tectonics.query(dir, climateQueryScratch).crustDist,
    })
    this.climateFn = (dir: Vector3, height: number): ClimateSample =>
      this.climateSim.sample(dir, height, this._climateScratch)
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
          // Also pre-enqueue the full descent path so we don't wait one level per frame.
          for (const child of node.children!) {
            if (
              !this.geoCache.get(child.key) &&
              !this.visibleMeshes.has(child.key) &&
              !this.buildQueueSet.has(child.key)
            ) {
              this.enqueueBuild(child, this.computeProjPx(cam, child))
            }
            // Pre-enqueue grandchildren and deeper levels along the hot path
            this.enqueueDeepPath(child, cam, splitThreshPx)
          }
          this.splitPending.set(node.key, node)
          return
        }
        // All 4 children ready — recurse into them, clear split-pending
        this.splitPending.delete(node.key)
        for (const child of node.children!) collect(child)
      } else {
        // Children just created — enqueue all 4 and keep parent visible.
        // Also pre-enqueue the full descent path so deeper levels build concurrently.
        for (const child of node.children!) {
          if (!this.buildQueueSet.has(child.key)) {
            this.enqueueBuild(child, this.computeProjPx(cam, child))
          }
          // Pre-enqueue grandchildren and deeper levels along the hot path
          this.enqueueDeepPath(child, cam, splitThreshPx)
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

  private drainBuildQueue(): void {
    if (this.buildQueue.length === 0) return

    // Sort highest-priority (largest projPx / closest to camera) to the front
    // so the most visually-urgent chunks build first. O(n log n) but queue is small.
    this.buildQueue.sort((a, b) => b.priority - a.priority)

    const t0 = performance.now()
    let built = 0

    while (built < BUILD_BUDGET_PER_FRAME && this.buildQueue.length > 0) {
      // Also enforce a wall-clock cap so a heavy frame (complex heightFn) doesn't stall
      if (built > 0 && performance.now() - t0 > BUILD_BUDGET_MS) break

      const item = this.buildQueue.shift()!
      this.buildQueueSet.delete(item.key)

      // Already cached or already visible — skip
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
      })

      this.geoCache.set(item.key, new CachedMeshData(data))
      built++
    }

    this.lastBuildMs = performance.now() - t0
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
  // Internal: helpers
  // ---------------------------------------------------------------------------

  /**
   * Returns the shared material for a chunk at this level.
   * View mode priority: tectonics → lodColors → normal.
   */
  private materialFor(level: number): MeshStandardMaterial | MeshBasicMaterial | MeshBasicNodeMaterial {
    if (this.tectonicsViewActive) {
      return this.plateColorMaterial
    }
    if (this.debugColorsActive) {
      return this.debugMaterials[Math.min(level, this.debugMaterials.length - 1)]
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
    const nearDist = Math.max(EPS_DIST, camToCenter - boundRadius)
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
}
