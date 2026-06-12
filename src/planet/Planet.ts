import {
  BufferGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Vector3,
  Matrix4,
  Quaternion,
} from 'three'
import { MeshBasicNodeMaterial } from 'three/webgpu'
import { attribute } from 'three/tsl'
import { createNoise3D, fbm, ridged } from './noise'
import { buildChunkGeometry, ChunkMeshData } from './ChunkMesher'
import { QuadtreeNode } from './QuadtreeNode'
import { Tectonics, TectonicQuery } from './tectonics'
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
}

interface Stats {
  leaves: number
  cached: number
  maxLevel: number
  pendingBuilds: number
  lastBuildMs: number
  plates: number
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

const BUILD_BUDGET_PER_FRAME = 4 // reduced from 6: tectonic heightFn is ~2-3× costlier per vertex
const LRU_CAPACITY = 700
const HYSTERESIS = 0.15 // 15% — merge threshold = splitFactor * (1 + HYSTERESIS)

// Tectonic heightFn tuning constants (at top of closure for easy tuning)
const TECT_BOUNDARY_FALLOFF    = 0.05  // boundary falloff in radians ≈ 2.9°
const TECT_SUBDUCTION_FACTOR   = 0.5   // oceanic subduction trench depth
const TECT_COLLISION_BASE      = 0.85  // continental collision mountain base
const TECT_COLLISION_RIDGE     = 0.45  // ridged texture amplitude on collision belts
const TECT_OCEANIC_ARC         = 0.3   // oceanic-oceanic island arc height
const TECT_MID_OCEAN_RIDGE     = 0.18  // mid-ocean ridge uplift
const TECT_RIFT_DEPTH          = 0.14  // continental rift depression
const TECT_DETAIL_FBM          = 0.16  // detail fbm amplitude
const TECT_DETAIL_FBM_SCALE    = 3.2   // fbm frequency scale
const TECT_DETAIL_RIDGE        = 0.08  // detail ridged amplitude
const TECT_DETAIL_RIDGE_SCALE  = 7.0   // ridged frequency scale

// ---------------------------------------------------------------------------
// Planet
// ---------------------------------------------------------------------------

export class Planet extends Group {
  readonly seed: number

  private readonly radius: number
  private readonly heightScale: number
  private readonly resolution: number
  private readonly maxDepth: number
  private readonly splitFactor: number
  private plateCount: number

  private heightFn!: (dir: Vector3) => number

  /** Tectonic simulation — rebuilt on regenerate(). */
  tectonics!: Tectonics

  /** Debug overlay — rebuilt on regenerate(), always a child of this Group. */
  private tectonicsDebug!: TectonicsDebug

  /** Pole-axis + equator gizmos — built once in constructor, never rebuilt. */
  private gizmos!: PlanetGizmos

  /** Six root nodes, one per cube face. */
  private roots!: QuadtreeNode[]

  /** Currently visible leaf meshes keyed by node key. */
  private readonly visibleMeshes = new Map<string, Mesh>()

  /** Geometry cache (returns geometry on mesh removal, evicts on overflow). */
  private readonly geoCache = new LruCache<CachedMeshData>(LRU_CAPACITY)

  /**
   * Build queue for pending geometries. Each entry records:
   *  - key: node key
   *  - node: the QuadtreeNode to build
   *  - waitingFor: "split-ready" — parent key this belongs to (or null if it's
   *    a standalone leaf or a parent being built for merge)
   */
  private readonly buildQueue: Array<{ key: string; node: QuadtreeNode }> = []
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
  private readonly normalMaterial: MeshStandardMaterial
  private readonly debugMaterials: MeshBasicMaterial[]
  /** Plate-color node material (flat, unlit, reads 'plateColor' attribute). Created once. */
  private readonly plateColorMaterial: MeshBasicNodeMaterial
  private debugColorsActive = false
  private tectonicsViewActive = false
  private wireframeActive = false

  // Cached inverse world matrix + quaternion for rotation-safe transforms
  // Re-computed each update() call.
  private readonly _invWorldMatrix = new Matrix4()
  private readonly _invWorldQuat = new Quaternion()
  // Scratch vectors — preallocated, zero allocations in hot paths
  private readonly _camLocalScratch = new Vector3()
  private readonly _dirLocalScratch = new Vector3()

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

    this.normalMaterial = new MeshStandardMaterial({
      vertexColors: true,
      roughness: 1,
      metalness: 0,
    })

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
   */
  update(cameraWorldPos: Vector3): void {
    if (this.frozen) return

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
    this.normalMaterial.wireframe = on
    for (const m of this.debugMaterials) m.wireframe = on
    this.plateColorMaterial.wireframe = on
  }

  setDebugColors(on: boolean): void {
    this.debugColorsActive = on
    // Mutual exclusivity: turning on LOD colors turns off tectonics view.
    if (on && this.tectonicsViewActive) {
      this.tectonicsViewActive = false
      this.tectonicsDebug.visible = false
    }
    for (const [, mesh] of this.visibleMeshes) {
      mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
    }
  }

  setTectonicsView(on: boolean): void {
    this.tectonicsViewActive = on
    this.tectonicsDebug.visible = on
    // Mutual exclusivity: turning on tectonics turns off LOD debug colors.
    if (on && this.debugColorsActive) {
      this.debugColorsActive = false
    }
    // Swap chunk materials to/from plate-color node material.
    for (const [, mesh] of this.visibleMeshes) {
      mesh.material = this.materialFor(this.levelFromKey(mesh.userData.key as string))
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
   * Clamped to [4, 48]. Has no effect on the current terrain — takes effect
   * on the next regenerate(seed) invocation.
   */
  setPlateCount(n: number): void {
    this.plateCount = Math.max(4, Math.min(48, n))
  }

  regenerate(seed: number): void {
    // Remove all visible meshes from scene and dispose their geometries explicitly
    // (visible geometries are NOT in the cache — invariant: cache holds only non-displayed geometries)
    for (const mesh of this.visibleMeshes.values()) {
      this.remove(mesh)
      ;(mesh.geometry as BufferGeometry).dispose()
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
    this._dirLocalScratch.copy(v).normalize().applyQuaternion(this._invWorldQuat)
    return this.radius + this.heightFn(this._dirLocalScratch) * this.heightScale
  }

  /** Expose the Tectonics instance (main.ts HUD may read plate count). */
  get tectonicsInstance(): Tectonics {
    return this.tectonics
  }

  getStats(): Stats {
    let maxLevel = 0
    for (const [key] of this.visibleMeshes) {
      maxLevel = Math.max(maxLevel, this.levelFromKey(key))
    }
    return {
      leaves: this.visibleMeshes.size,
      cached: this.geoCache.size,
      maxLevel,
      pendingBuilds: this.buildQueue.length,
      lastBuildMs: this.lastBuildMs,
      plates: this.tectonics.plates.length,
    }
  }

  dispose(): void {
    // Dispose visible mesh geometries explicitly (not in cache — invariant)
    for (const mesh of this.visibleMeshes.values()) {
      this.remove(mesh)
      ;(mesh.geometry as BufferGeometry).dispose()
    }
    this.visibleMeshes.clear()
    this.geoCache.disposeAll()
    this.normalMaterial.dispose()
    for (const m of this.debugMaterials) m.dispose()
    this.plateColorMaterial.dispose()
    this.tectonicsDebug.dispose()
    this.gizmos.dispose()
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

    this.tectonics = new Tectonics({ seed, plateCount: this.plateCount })
    const plates = this.tectonics.plates

    // One scratch TectonicQuery per heightFn call — safe because heightFn is called serially.
    const scratch: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0 }

    // Separate scratch for plateColorFn (both called back-to-back on the same dir — see memo below).
    // 1-entry memoization: if dir.x/y/z identical to the previous heightFn call, reuse the
    // query result for plateColorFn, halving Voronoi cost under serial meshing.
    let memoX = NaN, memoY = NaN, memoZ = NaN
    let memoPlateId = 0
    const scratchColor: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0 }

    this.heightFn = (dir: Vector3): number => {
      const x = dir.x, y = dir.y, z = dir.z

      this.tectonics.query(dir, scratch)
      const plate = plates[scratch.plateId]
      const other = plates[scratch.neighborId]

      // Memoize last queried dir for plateColorFn re-use
      memoX = x; memoY = y; memoZ = z
      memoPlateId = scratch.plateId

      const base = plate.baseElevation
      const w = Math.exp(-scratch.boundaryDist / TECT_BOUNDARY_FALLOFF)
      const conv = scratch.convergence

      let tect = 0
      if (conv > 0) {
        // Converging
        if (plate.type === 'oceanic' && other.type === 'continental') {
          // Subduction trench on the oceanic side
          tect = -conv * w * TECT_SUBDUCTION_FACTOR
        } else if (plate.type === 'continental' || other.type === 'continental') {
          // Collision mountain belts with ridged texture
          tect = conv * w * (TECT_COLLISION_BASE + TECT_COLLISION_RIDGE * ridged(noise, x * 6.0, y * 6.0, z * 6.0, { octaves: 4 }))
        } else {
          // Oceanic-oceanic island arcs
          tect = conv * w * TECT_OCEANIC_ARC
        }
      } else {
        // Diverging: use |conv| so the sign is controlled purely by the factor.
        const absConv = -conv // conv<=0 here so -conv >= 0 gives |conv|
        if (plate.type === 'oceanic') {
          tect = absConv * w * TECT_MID_OCEAN_RIDGE    // positive: mid-ocean ridge uplift
        } else {
          tect = absConv * w * (-TECT_RIFT_DEPTH)      // negative: continental rift depression
        }
      }

      const detail = TECT_DETAIL_FBM * fbm(noise, x * TECT_DETAIL_FBM_SCALE, y * TECT_DETAIL_FBM_SCALE, z * TECT_DETAIL_FBM_SCALE, { octaves: 6 })
                   + TECT_DETAIL_RIDGE * ridged(noise, x * TECT_DETAIL_RIDGE_SCALE, y * TECT_DETAIL_RIDGE_SCALE, z * TECT_DETAIL_RIDGE_SCALE, { octaves: 4 }) * Math.max(0, base) * 4

      return Math.max(-1, Math.min(1, base + tect + detail))
    }

    // plateColorFn: uses the 1-entry memo to avoid a redundant Voronoi query when
    // the mesher calls heightFn then plateColorFn for the same dir back-to-back.
    this._plateColorFn = (dir: Vector3): readonly [number, number, number] => {
      let pid: number
      if (dir.x === memoX && dir.y === memoY && dir.z === memoZ) {
        // Same dir as last heightFn call — reuse memoized plate id.
        pid = memoPlateId
      } else {
        // Different dir (e.g. the origin vertex queried separately) — do a fresh query.
        this.tectonics.query(dir, scratchColor)
        pid = scratchColor.plateId
      }
      return plates[pid].color
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
      surfaceRadiusAt: (localDir: Vector3) => {
        localScratch.copy(localDir).normalize()
        return this.radius + this.heightFn(localScratch) * this.heightScale
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
   * Split/merge decisions use split threshold and merge threshold (with hysteresis).
   * Results drive addMesh / removeMesh / schedule builds.
   */
  private selectLeaves(cam: Vector3): void {
    const splitDist = this.splitFactor
    const mergeDist = this.splitFactor * (1 + HYSTERESIS)

    const desired = new Set<string>()

    // Collect desired leaves via recursive descent
    const collect = (node: QuadtreeNode): void => {
      const dist = cam.distanceTo(node.worldCenter)
      const wantSplit = node.level < this.maxDepth && dist < splitDist * node.nodeSize

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
          // Ensure missing children are queued
          for (const child of node.children!) {
            if (
              !this.geoCache.get(child.key) &&
              !this.visibleMeshes.has(child.key) &&
              !this.buildQueueSet.has(child.key)
            ) {
              this.enqueueBuild(child)
            }
          }
          this.splitPending.set(node.key, node)
          return
        }
        // All 4 children ready — recurse into them, clear split-pending
        this.splitPending.delete(node.key)
        for (const child of node.children!) collect(child)
      } else {
        // Children just created — enqueue all 4 and keep parent visible
        for (const child of node.children!) {
          if (!this.buildQueueSet.has(child.key)) {
            this.enqueueBuild(child)
          }
        }
        this.splitPending.set(node.key, node)
        desired.add(node.key) // parent stays visible
      }
    }

    for (const root of this.roots) collect(root)

    // --- Merge: nodes in desired whose children exist and are all leaves visible
    //     but dist > mergeDist (caller no longer wants split).
    //     If we just want to check for unwanted children, we also walk once more.
    const desiredWithChildren = (node: QuadtreeNode): void => {
      if (node.children === null) return

      const dist = cam.distanceTo(node.worldCenter)
      if (dist > mergeDist * node.nodeSize) {
        // Node is in desired as a leaf (no longer splits) but children still exist
        // → merge: we keep children visible until parent geometry ready.
        const childKeys = node.children.map((c) => c.key)
        const allChildrenVisible = childKeys.every((k) => this.visibleMeshes.has(k))

        if (allChildrenVisible && !this.mergePending.has(node.key)) {
          const parentReady =
            this.geoCache.get(node.key) !== undefined || this.visibleMeshes.has(node.key)
          if (parentReady) {
            // Immediate swap: show parent, remove children
            this.addMesh(node)
            for (const c of node.children!) this.removeMesh(c.key)
            node.merge()
          } else {
            // Enqueue parent build; keep children visible
            this.mergePending.set(node.key, { node, childKeys })
            if (!this.buildQueueSet.has(node.key)) this.enqueueBuild(node)
          }
        }
      }

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
    const mergeDist = this.splitFactor * (1 + HYSTERESIS)

    for (const [parentKey, { node, childKeys }] of this.mergePending) {
      const parentCached = this.geoCache.get(parentKey)
      if (!parentCached) continue

      // Re-validate: children must all still be leaves (no grandchildren)
      const childrenStillLeaves =
        node.children !== null &&
        node.children.every((c) => c.children === null && this.visibleMeshes.has(c.key))

      // Re-validate: merge-distance condition must still hold
      const dist = cam.distanceTo(node.worldCenter)
      const mergeConditionHolds = dist > mergeDist * node.nodeSize

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

  private enqueueBuild(node: QuadtreeNode): void {
    this.buildQueue.push({ key: node.key, node })
    this.buildQueueSet.add(node.key)
  }

  private drainBuildQueue(): void {
    if (this.buildQueue.length === 0) return

    const t0 = performance.now()
    let built = 0

    while (built < BUILD_BUDGET_PER_FRAME && this.buildQueue.length > 0) {
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
    const mesh = new Mesh(data.geometry, this.materialFor(node.level))
    mesh.position.copy(data.origin)
    mesh.frustumCulled = true
    mesh.userData.key = node.key
    this.add(mesh)
    this.visibleMeshes.set(node.key, mesh)
  }

  private removeMesh(key: string): void {
    const mesh = this.visibleMeshes.get(key)
    if (!mesh) return
    this.remove(mesh)
    this.visibleMeshes.delete(key)
    // Return geometry to cache (don't dispose)
    this.geoCache.return(key, new CachedMeshData({ geometry: mesh.geometry as BufferGeometry, origin: mesh.position.clone() }))
  }

  // ---------------------------------------------------------------------------
  // Internal: helpers
  // ---------------------------------------------------------------------------

  /**
   * Returns the active material for a chunk at this level.
   * View mode priority: tectonics → lodColors → normal.
   * wireframe is applied on top of whichever material is selected.
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
}
