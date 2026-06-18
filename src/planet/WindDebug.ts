import {
  BufferGeometry,
  Color,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  Object3D,
  Quaternion,
  Vector3,
} from 'three'
import { Climate, WindSample } from './climate'
import { buildArrowGeometry, fibonacciSphere } from './arrowGeometry'

// ---------------------------------------------------------------------------
// WindDebug — render-only arrow overlay for the wind field.
//
// Mirrors TectonicsDebug's structure:
//   - constructor builds the InstancedMesh; geo/mat/mesh tracked for dispose
//   - dispose() frees geo + mat, deduplicated by reference
//   - setVisible / setDensity(n) for live toggling + density rebuild
//
// Arrow length: WIND_MAX_LEN = radius * 0.045.
// At RADIUS = 500 000 that is ~22 500 units — visible from orbit.
//
// Color: bearing-hue (HSL hue from the tangent-plane wind direction) with
// brightness/saturation scaled by wind speed. Arrows are lifted
// heightScale * 2.0 above the terrain surface, matching TectonicsDebug.
//
// animate(t) drives windAtTime() per frame — transient eddies + drifting
// vortices animate live without rebuild. Returns early when !visible.
//
// No determinism impact — sampling is a read-only cubemap/transient query.
// ---------------------------------------------------------------------------

export class WindDebug {
  private _scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown }
  private _climate: Climate
  private _surfaceRadiusAt: (dir: Vector3) => number
  private _radius: number
  private _heightScale: number
  private _density: number

  private readonly _geometries: BufferGeometry[] = []
  private readonly _materials:  MeshBasicMaterial[] = []
  private readonly _meshes:     InstancedMesh[] = []

  private _visible = false

  // Cached per-build state — computed once in _setup(), reused in _writeInstances().
  private _sampleDirs: Vector3[] = []
  private _surfaceR: Float32Array = new Float32Array(0)

  // Per-arrow scratch fields — promoted from _build() locals to avoid per-frame allocations.
  private readonly _q    = new Quaternion()
  private readonly _zHat = new Vector3(0, 0, 1)
  private readonly _mat4 = new Matrix4()
  private readonly _pos  = new Vector3()
  private readonly _col  = new Color()
  private readonly _wDir = new Vector3()
  private readonly _arrowScale = new Vector3()

  // Reused scratch for windAtTime — zero-alloc per arrow.
  private readonly _wind: WindSample = { x: 0, y: 0, z: 0, speed: 0 }

  private _lastT = 0

  constructor(
    scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown },
    climate: Climate,
    opts: {
      radius:          number
      heightScale:     number
      surfaceRadiusAt: (dir: Vector3) => number
      /** Initial arrow-sample count (default 1500). */
      density?:        number
    },
  ) {
    this._scene           = scene
    this._climate         = climate
    this._radius          = opts.radius
    this._heightScale     = opts.heightScale
    this._surfaceRadiusAt = opts.surfaceRadiusAt
    this._density         = opts.density ?? 1500

    this._build()
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  get visible(): boolean { return this._visible }

  setVisible(v: boolean): void {
    this._visible = v
    for (const m of this._meshes) m.visible = v
  }

  /** Replace the arrow density (Fibonacci-sphere sample count) and rebuild. */
  setDensity(n: number): void {
    this._density = Math.max(1, n)
    this._rebuild()
  }

  /** Full rebuild — call after climate is regenerated. */
  rebuild(): void {
    this._rebuild()
  }

  /**
   * Scale the rendered arrow length. 1.0 = default. Live — scales the InstancedMesh
   * objects directly without rebuilding or re-sampling the wind field.
   */
  setArrowScale(s: number): void {
    for (const m of this._meshes) {
      m.scale.setScalar(s)
    }
  }

  /**
   * Animate the wind arrows at time `t` (seconds). Returns immediately if not visible.
   * Call once per frame from Planet.animateWind(t).
   */
  animate(t: number): void {
    if (!this._visible) return
    this._lastT = t
    this._writeInstances(t)
  }

  dispose(): void {
    this._teardown()
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private _teardown(): void {
    for (const m of this._meshes) this._scene.remove(m)

    // Deduplicated by reference (mirrors TectonicsDebug).
    const seenGeos = new Set<BufferGeometry>()
    for (const geo of this._geometries) {
      if (!seenGeos.has(geo)) { seenGeos.add(geo); geo.dispose() }
    }
    for (const mat of this._materials) mat.dispose()

    this._geometries.length = 0
    this._materials.length  = 0
    this._meshes.length     = 0
  }

  private _rebuild(): void {
    this._teardown()
    this._build()
  }

  /** One-time setup: sample dirs + lifted radii + geometry/material/InstancedMesh. */
  private _build(): void {
    // --- (i) one-time setup ---
    this._sampleDirs = fibonacciSphere(this._density)
    const count = this._sampleDirs.length

    // Precompute lifted surface radii — terrain doesn't move between frames.
    this._surfaceR = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      this._surfaceR[i] = this._surfaceRadiusAt(this._sampleDirs[i])
    }

    // Arrow geometry: shaft 2.5% of length, head 7% base, 28% head length —
    // identical to TectonicsDebug's velocity-field arrows.
    const arrowGeo = buildArrowGeometry(0.025, 0.07, 0.28)
    this._geometries.push(arrowGeo)

    const arrowMat = new MeshBasicMaterial({ vertexColors: false })
    this._materials.push(arrowMat)

    const instancedArrows = new InstancedMesh(arrowGeo, arrowMat, count)
    instancedArrows.count   = 0
    instancedArrows.visible = this._visible

    this._scene.add(instancedArrows)
    this._meshes.push(instancedArrows)

    // --- (ii) initial fill at t=0 ---
    this._writeInstances(this._lastT)
  }

  /**
   * Write per-arrow matrices + colors using windAtTime(dir, t). Hot path — zero
   * alloc. Reassigns instancedArrows.count to compact out skipped (calm) arrows.
   */
  private _writeInstances(t: number): void {
    if (this._meshes.length === 0) return
    const instancedArrows = this._meshes[0]
    const WIND_MAX_LEN = this._radius * 0.045

    let instanceIdx = 0
    for (let i = 0; i < this._sampleDirs.length; i++) {
      const dir = this._sampleDirs[i]
      this._climate.windAtTime(dir, t, this._wind)

      const spd = this._wind.speed
      if (spd < 0.02) continue  // skip near-zero wind (poles, calm pockets)

      // Arrow length scales linearly with speed.
      const arrowLen = WIND_MAX_LEN * spd

      // Position: hover heightScale * 2.0 above terrain.
      const surfR = this._surfaceR[i]
      this._pos.copy(dir).multiplyScalar(surfR + this._heightScale * 2.0)

      // Wind direction is already a tangent-plane unit vector from windAtTime.
      this._wDir.set(this._wind.x, this._wind.y, this._wind.z)
      const wLen = this._wDir.length()
      if (wLen < 1e-6) continue
      this._wDir.multiplyScalar(1 / wLen)

      // Orient arrow: rotate +Z onto the wind direction.
      this._q.setFromUnitVectors(this._zHat, this._wDir)
      this._mat4.makeRotationFromQuaternion(this._q)
      this._arrowScale.set(arrowLen, arrowLen, arrowLen)
      this._mat4.scale(this._arrowScale)
      this._mat4.setPosition(this._pos)
      instancedArrows.setMatrixAt(instanceIdx, this._mat4)

      // Color: bearing-hue (HSL hue from wind direction in the tangent plane).
      //
      // east = polarAxis × dir = (0,1,0) × dir = (dir.z, 0, -dir.x)
      // |east| = sqrt(dir.z² + dir.x²) = cos(lat)
      // north = dir × east (normalised)
      //
      // bearing = atan2(dot(wDir, east), dot(wDir, north))  — CW from north
      // hue     = (bearing / 2π + 1) mod 1
      // L (HSL) = 0.45 + 0.25 * speed   (dim for light breeze → bright at peak)
      const ex = dir.z
      const ez = -dir.x
      const eLen = Math.sqrt(ex * ex + ez * ez)  // cos(lat)

      if (eLen < 1e-6) {
        // Pole — degenerate tangent basis; colour white.
        this._col.setRGB(1, 1, 1)
      } else {
        const eNx = ex / eLen
        const eNz = ez / eLen

        // north = dir × east (unit, ey = 0 so some cross terms vanish)
        const nx = dir.y * eNz            // dir.y * eNz - dir.z * 0
        const ny = dir.z * eNx - dir.x * eNz
        const nz = -dir.y * eNx           // dir.x * 0 - dir.y * eNx

        const eDot = this._wDir.x * eNx + this._wDir.z * eNz       // ey = 0
        const nDot = this._wDir.x * nx  + this._wDir.y * ny + this._wDir.z * nz

        const bearing = Math.atan2(eDot, nDot)
        const hue     = ((bearing / (2 * Math.PI)) % 1 + 1) % 1
        const light   = 0.45 + 0.25 * spd
        this._col.setHSL(hue, 0.95, light)
      }
      instancedArrows.setColorAt(instanceIdx, this._col)

      instanceIdx++
    }

    instancedArrows.count = instanceIdx
    instancedArrows.instanceMatrix.needsUpdate = true
    if (instancedArrows.instanceColor) instancedArrows.instanceColor.needsUpdate = true
  }
}
