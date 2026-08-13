import {
  AdditiveBlending,
  BufferGeometry,
  DynamicDrawUsage,
  Float32BufferAttribute,
  LineSegments,
  Object3D,
  Vector3,
} from 'three'
import { LineBasicNodeMaterial } from 'three/webgpu'
import { Climate, WindSample } from './climate'

// ---------------------------------------------------------------------------
// WindFlow — advected particle streakline overlay (earth.nullschool "living
// wind map" look). Coexists with WindDebug (arrow overlay).
//
// Dynamic-LOD via a visible-cap budget: every frame all particles are kept
// inside the camera-facing spherical cap (≈ the visible hemisphere, shrinking
// as you zoom in). Particles that advect past the cap edge (off-screen, behind
// a small horizon margin) are teleported back into the cap. So the whole fixed
// particle budget is always spent on what's on screen → density concentrates
// as you approach the surface, and the hidden hemisphere costs nothing.
//
// Field: the static climatological mean `windAt()` (≈10× cheaper than the
// transient `windAtTime` — that funds the higher density). Streaklines still
// curve through every zonal band and vortex; only the slow in-place eddy
// wobble is dropped (invisible in fast-moving streaks). The arrows (WindDebug)
// keep windAtTime for the transient read.
//
// Colour: uniform cyan; additive blending brightens where lines converge
// (the reference's bright knots) and a per-vertex head→tail fade gives the
// streak. Render: ONE batched LineSegments draw call.
//
// Hot path: zero allocations. Local mulberry32 PRNG (fixed seed) — no reserved
// RNG stream consumed.
// ---------------------------------------------------------------------------

/** Cyan streak colour (additive; convergence brightens toward white). */
const CYAN_R = 0.35
const CYAN_G = 0.75
const CYAN_B = 1.0

/** mulberry32 PRNG factory. Returns a () => number in [0,1). */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0
  return (): number => {
    s += 0x6D2B79F5
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000
  }
}

export class WindFlow {
  private readonly _scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown }
  private readonly _climate: Climate
  private readonly _radius: number
  private readonly _heightScale: number
  private _density: number
  private _flowSpeed: number
  private _trailLen: number
  private _lifeBase: number
  private _visible: boolean

  // State arrays — sized N (density) and N*K (trail ring buffer)
  /** Current particle unit directions on the sphere: (x,y,z) per particle. */
  private _dirs!: Float32Array
  /** Per-particle age in seconds. */
  private _age!: Float32Array
  /** Per-particle jittered lifetime in seconds. */
  private _life!: Float32Array
  /** Per-particle wind speed [0,1] from the last advect — drives streak brightness. */
  private _speed!: Float32Array
  /**
   * Trail ring buffer: K world positions per particle stored flat.
   * Layout: particle i, slot k → base index (i * K + k) * 3.
   * Head slot index for particle i: _head[i].
   */
  private _trail!: Float32Array
  /** Ring-buffer head index (0..K-1) per particle. */
  private _head!: Int32Array

  // Render primitive
  private _mesh: LineSegments | null = null
  private _geo: BufferGeometry | null = null
  private _mat: LineBasicNodeMaterial | null = null
  /** Flat position array for LineSegments: (N*(K-1)*2) segment endpoints * 3 floats. */
  private _posBuf!: Float32Array
  /** Flat color array matching _posBuf layout: RGB per vertex. */
  private _colBuf!: Float32Array

  // Scratch — promoted to instance fields (zero hot-path allocations)
  private readonly _p  = new Vector3()
  private readonly _w  = new Vector3()
  private readonly _pN = new Vector3()
  private readonly _wind: WindSample = { x: 0, y: 0, z: 0, speed: 0 }
  // Visible-cap frame state (set at the top of animate(), read by _respawnInCap)
  private readonly _capAxis = new Vector3()  // camera direction in planet-local frame
  private readonly _capE1   = new Vector3()  // orthonormal tangent basis ⟂ axis
  private readonly _capE2   = new Vector3()
  private _capCos = -1  // cos(cap half-angle): particles with dir·axis < this are off-screen

  // PRNG — local fixed seed, not a reserved tectonics/climate stream
  private _rng: () => number

  constructor(
    scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown },
    climate: Climate,
    opts: {
      radius:      number
      heightScale: number
      /** Initial particle count (default 8000). */
      density?:    number
    },
  ) {
    this._scene      = scene
    this._climate    = climate
    this._radius     = opts.radius
    this._heightScale = opts.heightScale
    this._density    = opts.density ?? 8000
    // flowSpeed is an angular advection rate (arc = flowSpeed·speed·dt). It must be
    // high enough that the head visibly translates and the trail spans a readable arc
    // — otherwise the streak degenerates to a point that just re-orients in place.
    this._flowSpeed  = 0.6
    this._trailLen   = 12
    this._lifeBase   = 4.0
    this._visible    = false
    this._rng        = mulberry32(0x9E3779B9)

    this._allocate()
    this._initMaterial()
    this._initGeometry()
    this._scatter()
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  get visible(): boolean { return this._visible }

  setVisible(v: boolean): void {
    this._visible = v
    if (this._mesh !== null) this._mesh.visible = v
  }

  /** Replace particle count and rebuild. */
  setDensity(n: number): void {
    this._density = Math.max(1, Math.round(n))
    this._rebuild()
  }

  /** Live — no rebuild needed. */
  setFlowSpeed(v: number): void {
    this._flowSpeed = Math.max(0, v)
  }

  /** Change trail length (K). Triggers rebuild because segment count changes. */
  setTrailLength(k: number): void {
    this._trailLen = Math.max(2, Math.round(k))
    this._rebuild()
  }

  /** Live — no rebuild needed. */
  setLifetime(s: number): void {
    this._lifeBase = Math.max(0.1, s)
  }

  /** Full rebuild — call after climate regenerate. */
  rebuild(): void {
    this._rebuild()
  }

  /**
   * Advance all particles by dt seconds, then rebuild the LineSegments buffers.
   * `camDir` is the camera direction in the PLANET-LOCAL frame (unit); `altitude`
   * is camera height above the surface (world units). Both drive the visible-cap
   * LOD. Returns immediately when not visible (zero cost).
   */
  animate(dt: number, camDir: Vector3, altitude: number): void {
    if (!this._visible) return

    const N   = this._density
    const K   = this._trailLen
    const shellR = this._radius + this._heightScale * 2.0

    // Visible spherical cap: a point is above the horizon when dir·camDir ≥ R/(R+h).
    // Extend the cap slightly past the horizon (−0.04) so the respawn churn at the
    // cap edge lands just off-screen. Clamp the half-angle to ≥ ~11° (capCos ≤ 0.98)
    // so the cap never collapses to a point at very low altitude.
    const R = this._radius
    let capCos = R / (R + Math.max(1, altitude)) - 0.04
    if (capCos > 0.98) capCos = 0.98
    else if (capCos < -0.999) capCos = -0.999
    this._capCos = capCos

    // Cap basis: axis = camDir, plus an orthonormal tangent pair (e1, e2).
    this._capAxis.copy(camDir)
    const ax = this._capAxis.x, ay = this._capAxis.y, az = this._capAxis.z
    // Reference up not parallel to axis.
    const ux = Math.abs(ay) < 0.99 ? 0 : 1
    const uy = Math.abs(ay) < 0.99 ? 1 : 0
    // e1 = normalize(up × axis)
    let e1x = uy * az - 0 * ay
    let e1y = 0 * ax - ux * az
    let e1z = ux * ay - uy * ax
    const e1Len = Math.hypot(e1x, e1y, e1z) || 1
    e1x /= e1Len; e1y /= e1Len; e1z /= e1Len
    this._capE1.set(e1x, e1y, e1z)
    // e2 = axis × e1
    this._capE2.set(
      ay * e1z - az * e1y,
      az * e1x - ax * e1z,
      ax * e1y - ay * e1x,
    )

    for (let i = 0; i < N; i++) {
      const di = i * 3
      this._p.set(this._dirs[di], this._dirs[di + 1], this._dirs[di + 2])

      // Off-screen (past the cap edge) → teleport back into the visible cap.
      const dotCam = this._p.x * ax + this._p.y * ay + this._p.z * az
      if (dotCam < capCos) {
        this._respawnInCap(i, shellR)
        continue
      }

      this._climate.windAt(this._p, this._wind)
      const s = this._wind.speed

      this._age[i] += dt
      if (this._age[i] > this._life[i] || s < 0.02) {
        this._respawnInCap(i, shellR)
        continue
      }

      // Advect along great circle: pNew = normalize(p*cos(arc) + w*sin(arc))
      this._w.set(this._wind.x, this._wind.y, this._wind.z)
      const wLen = this._w.length()
      if (wLen > 1e-6) this._w.multiplyScalar(1 / wLen)

      const arc = this._flowSpeed * s * dt
      const cosA = Math.cos(arc)
      const sinA = Math.sin(arc)
      this._pN.set(
        this._p.x * cosA + this._w.x * sinA,
        this._p.y * cosA + this._w.y * sinA,
        this._p.z * cosA + this._w.z * sinA,
      ).normalize()

      // Store new direction + speed (for brightness).
      this._dirs[di]     = this._pN.x
      this._dirs[di + 1] = this._pN.y
      this._dirs[di + 2] = this._pN.z
      this._speed[i] = s

      // Push lifted world position into ring buffer at new head
      const newHead = (this._head[i] + 1) % K
      this._head[i] = newHead
      const ti = (i * K + newHead) * 3
      this._trail[ti]     = this._pN.x * shellR
      this._trail[ti + 1] = this._pN.y * shellR
      this._trail[ti + 2] = this._pN.z * shellR
    }

    // Write LineSegments buffers
    this._writeBuffers()
  }

  dispose(): void {
    this._teardown()
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /**
   * Teleport particle i to a uniform-random direction inside the current visible
   * cap and collapse its trail to that point (so no streak is drawn across the
   * sphere). Uses the cap basis set at the top of animate(). Zero-alloc.
   */
  private _respawnInCap(i: number, shellR: number): void {
    const rng = this._rng
    const K = this._trailLen

    // Uniform in cap: cosθ ∈ [capCos, 1], azimuth ∈ [0, 2π).
    const z = this._capCos + rng() * (1 - this._capCos)
    const r = Math.sqrt(Math.max(0, 1 - z * z))
    const phi = 2 * Math.PI * rng()
    const cphi = Math.cos(phi), sphi = Math.sin(phi)
    const a = this._capAxis, e1 = this._capE1, e2 = this._capE2
    let dx = a.x * z + (e1.x * cphi + e2.x * sphi) * r
    let dy = a.y * z + (e1.y * cphi + e2.y * sphi) * r
    let dz = a.z * z + (e1.z * cphi + e2.z * sphi) * r
    const len = Math.hypot(dx, dy, dz) || 1
    dx /= len; dy /= len; dz /= len

    const di = i * 3
    this._dirs[di]     = dx
    this._dirs[di + 1] = dy
    this._dirs[di + 2] = dz
    this._age[i]   = 0
    this._life[i]  = this._lifeBase * (0.5 + rng())
    this._speed[i] = 0

    const wx = dx * shellR, wy = dy * shellR, wz = dz * shellR
    for (let k = 0; k < K; k++) {
      const ti = (i * K + k) * 3
      this._trail[ti]     = wx
      this._trail[ti + 1] = wy
      this._trail[ti + 2] = wz
    }
    this._head[i] = 0
  }

  private _allocate(): void {
    const N = this._density
    const K = this._trailLen
    this._dirs  = new Float32Array(N * 3)
    this._age   = new Float32Array(N)
    this._life  = new Float32Array(N)
    this._speed = new Float32Array(N)
    this._trail = new Float32Array(N * K * 3)
    this._head  = new Int32Array(N)
    // (K-1) segments per particle, 2 endpoints per segment, 3 floats per position/color
    const vertCount = N * (K - 1) * 2
    this._posBuf = new Float32Array(vertCount * 3)
    this._colBuf = new Float32Array(vertCount * 3)
  }

  private _initMaterial(): void {
    this._mat = new LineBasicNodeMaterial({
      vertexColors: true,
      transparent:  true,
      depthWrite:   false,
      blending:     AdditiveBlending,
    })
  }

  private _initGeometry(): void {
    const N = this._density
    const K = this._trailLen
    const vertCount = N * (K - 1) * 2

    this._geo = new BufferGeometry()
    this._geo.setAttribute('position', new Float32BufferAttribute(this._posBuf, 3).setUsage(DynamicDrawUsage))
    this._geo.setAttribute('color',    new Float32BufferAttribute(this._colBuf, 3).setUsage(DynamicDrawUsage))
    this._geo.setDrawRange(0, vertCount)

    this._mesh = new LineSegments(this._geo, this._mat!)
    this._mesh.visible = this._visible
    this._mesh.frustumCulled = false  // shell is always partially in view
    this._scene.add(this._mesh)
  }

  private _scatter(): void {
    const N    = this._density
    const K    = this._trailLen
    const rng  = this._rng
    const shellR = this._radius + this._heightScale * 2.0

    for (let i = 0; i < N; i++) {
      // Uniform random sphere (camDir unknown at construction; animate() teleports
      // any off-screen particles into the cap on the first visible frame).
      const u1 = rng()
      const u2 = rng()
      const z = 2 * u1 - 1
      const r = Math.sqrt(Math.max(0, 1 - z * z))
      const theta = 2 * Math.PI * u2
      const dx = r * Math.cos(theta)
      const dy = z
      const dz = r * Math.sin(theta)

      const di = i * 3
      this._dirs[di]     = dx
      this._dirs[di + 1] = dy
      this._dirs[di + 2] = dz

      // Stagger initial age across [0, life) so trails don't all blink together
      this._life[i]  = this._lifeBase * (0.5 + rng())
      this._age[i]   = rng() * this._life[i]
      this._speed[i] = 0

      // Init entire trail ring to the lifted start position
      const wx = dx * shellR
      const wy = dy * shellR
      const wz = dz * shellR
      for (let k = 0; k < K; k++) {
        const ti = (i * K + k) * 3
        this._trail[ti]     = wx
        this._trail[ti + 1] = wy
        this._trail[ti + 2] = wz
      }
      this._head[i] = 0
    }
  }

  private _teardown(): void {
    if (this._mesh !== null) {
      this._scene.remove(this._mesh)
      this._mesh = null
    }
    if (this._geo !== null) {
      this._geo.dispose()
      this._geo = null
    }
    if (this._mat !== null) {
      this._mat.dispose()
      this._mat = null
    }
    // Guard against post-dispose animate() calls: _visible=false causes early return
    // before any _geo!/this._mesh! dereference.
    this._visible = false
  }

  private _rebuild(): void {
    this._teardown()
    // Reseed PRNG for consistent scatter after rebuild
    this._rng = mulberry32(0x9E3779B9)
    this._allocate()
    this._initMaterial()
    this._initGeometry()
    this._scatter()
  }

  /**
   * Write position + color arrays into the LineSegments geometry attributes.
   * Colour = cyan × head→tail brightness × per-particle speed brightness (RGB,
   * no alpha; additive blending brightens convergent knots toward white).
   * Just-respawned particles have a collapsed trail → zero-length segments →
   * draw nothing. One pass over all particles, no allocations, no wind samples.
   */
  private _writeBuffers(): void {
    const N = this._density
    const K = this._trailLen
    const pos = this._posBuf
    const col = this._colBuf
    let vi = 0  // vertex index (each vertex = 3 floats)

    for (let i = 0; i < N; i++) {
      // Faster wind → brighter streak (the reference's bright jets).
      const sb = 0.35 + 0.65 * this._speed[i]
      const cr = CYAN_R * sb, cg = CYAN_G * sb, cb = CYAN_B * sb

      const head = this._head[i]

      // Write (K-1) segments = (K-1)*2 vertices.
      // Segment s (0 = newest near head, K-2 = oldest near tail):
      //   start vertex = ring[(head - s) mod K]
      //   end   vertex = ring[(head - s - 1) mod K]
      // Brightness: head pair = 1.0, tail pair = 0.0 (linear across K-1 pairs)
      for (let s = 0; s < K - 1; s++) {
        const brightness = 1 - s / (K - 1)
        const kA = ((head - s)     % K + K) % K
        const kB = ((head - s - 1) % K + K) % K
        const tA = (i * K + kA) * 3
        const tB = (i * K + kB) * 3

        const vBase = vi * 3
        // Vertex A (start of segment, nearer to head)
        pos[vBase]     = this._trail[tA]
        pos[vBase + 1] = this._trail[tA + 1]
        pos[vBase + 2] = this._trail[tA + 2]
        col[vBase]     = cr * brightness
        col[vBase + 1] = cg * brightness
        col[vBase + 2] = cb * brightness
        vi++

        const vBase2 = vi * 3
        // Vertex B (end of segment, nearer to tail)
        const brightnessB = 1 - (s + 1) / (K - 1)
        pos[vBase2]     = this._trail[tB]
        pos[vBase2 + 1] = this._trail[tB + 1]
        pos[vBase2 + 2] = this._trail[tB + 2]
        col[vBase2]     = cr * brightnessB
        col[vBase2 + 1] = cg * brightnessB
        col[vBase2 + 2] = cb * brightnessB
        vi++
      }
    }

    const geoPosAttr = this._geo!.attributes.position as Float32BufferAttribute
    const geoColAttr = this._geo!.attributes.color    as Float32BufferAttribute
    geoPosAttr.needsUpdate = true
    geoColAttr.needsUpdate = true
  }
}
