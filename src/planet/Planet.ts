import {
  BufferGeometry,
  Frustum,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Points,
  PointsMaterial,
  Sphere,
  Vector3,
  Matrix4,
  Quaternion,
} from 'three'
import { MeshBasicNodeMaterial } from 'three/webgpu'
import { attribute, positionWorld, uniform, vec3, saturate } from 'three/tsl'
import { Tectonics } from './tectonics'
import { Climate, ClimateSample } from './climate'
import { makeTerrainSampler } from './terrainSampler'
import { buildChunkGeometry, arraysToGeometry, ChunkMeshArrays, ChunkMeshData } from './ChunkMesher'
import { MeshWorkerPool } from './MeshWorkerPool'
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
const LRU_CAPACITY = 8192         // res=32: ~38 KB/chunk × 8192 ≈ 311 MB geometry cache ceiling
const HYSTERESIS = 0.15 // 15% — SSE merge fires at threshold * (1 + HYSTERESIS)
const EPS_DIST = 0.1    // minimum camera-to-node distance (prevents div-by-zero at contact)

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

  // Cached inverse world matrix + quaternion for rotation-safe transforms
  // Re-computed each update() call.
  private readonly _invWorldMatrix = new Matrix4()
  private readonly _invWorldQuat = new Quaternion()
  // Scratch vectors — preallocated, zero allocations in hot paths
  private readonly _camLocalScratch = new Vector3()
  private readonly _dirLocalScratch = new Vector3()
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
    // Mutual exclusivity: turning on LOD colors turns off tectonics + heightmap views.
    if (on) {
      if (this.tectonicsViewActive) {
        this.tectonicsViewActive = false
        this.tectonicsDebug.visible = false
      }
      this.heightmapViewActive = false
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
    // Mutual exclusivity: turning on tectonics turns off LOD debug colors + heightmap.
    if (on) {
      this.debugColorsActive = false
      this.heightmapViewActive = false
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
    // Mutual exclusivity: heightmap is its own view; turn off LOD colors + tectonics.
    if (on) {
      this.debugColorsActive = false
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
    }
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
    this.wireMaterial.dispose()
    this.pointsMaterial.dispose()
    this.tectonicsDebug.dispose()
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
    const sampler = makeTerrainSampler({
      seed,
      radius: this.radius,
      heightScale: this.heightScale,
      plateCount: this.plateCount,
      arcDensity: this.arcDensity,
      baseTemp: this.baseTemp,
      atmosphere: this.atmosphere,
      bandCount: this.deriveBandCount(),
      axialTiltRad: (this.axialTiltDeg * Math.PI) / 180,
    })
    this.tectonics    = sampler.tectonics
    this.climateSim   = sampler.climate
    this.heightFn     = sampler.heightFn
    this._plateColorFn = sampler.plateColorFn
    this.climateFn    = (dir: Vector3, height: number): ClimateSample =>
      this.climateSim.sample(dir, height, this._climateScratch)

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
        plateCount: this.plateCount,
        arcDensity: this.arcDensity,
        baseTemp: this.baseTemp,
        atmosphere: this.atmosphere,
        bandCount: this.deriveBandCount(),
        axialTiltRad: (this.axialTiltDeg * Math.PI) / 180,
        tectonics: this.tectonics.toBaked(),
        climate: this.climateSim.toBaked(),
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
        })

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
  // Internal: helpers
  // ---------------------------------------------------------------------------

  /**
   * Returns the shared material for a chunk at this level.
   * View mode priority: tectonics → lodColors → normal.
   */
  private materialFor(level: number): MeshStandardMaterial | MeshBasicMaterial | MeshBasicNodeMaterial {
    if (this.heightmapViewActive) {
      return this.heightmapMaterial
    }
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
