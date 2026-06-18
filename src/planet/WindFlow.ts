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
// WindFlow — advected particle streakline overlay for the wind field.
//
// Coexists with WindDebug (arrow overlay). Renders batched THREE.LineSegments
// with per-vertex RGB color fading from bearing-hue (head) to black (tail),
// producing a head-to-tail brightness fade WITHOUT relying on per-vertex alpha.
//
// Material: LineBasicNodeMaterial (three/webgpu), vertexColors true, additive
// blending, depthWrite false — glows over the planet surface.
//
// Shell radius: constant = radius + heightScale * 2.0 (clears tallest terrain;
// no per-frame surfaceRadiusAt calls).
//
// Hot path: zero allocations. All scratch promoted to instance fields.
// Local mulberry32 PRNG (fixed seed 0x9E3779B9) — no reserved stream consumed.
// ---------------------------------------------------------------------------

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
  /**
   * Trail ring buffer: K world positions per particle stored flat.
   * Layout: particle i, slot k → base index (i * K + k) * 3.
   * Head slot index for particle i: _head[i].
   */
  private _trail!: Float32Array
  /** Ring-buffer head index (0..K-1) per particle. */
  private _head!: Int32Array
  /**
   * Per-particle cached bearing-hue RGB, computed once in the advect loop from
   * the already-sampled wind vector. Layout: (r,g,b) per particle = 3*N floats.
   * Consumed by _writeBuffers without calling windAtTime a second time.
   */
  private _color!: Float32Array

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
  /** Reusable 3-float scratch for hslToRgb output — avoids per-call heap allocation. */
  private readonly _rgb = new Float32Array(3)

  // PRNG — local fixed seed, not a reserved tectonics/climate stream
  private _rng: () => number

  constructor(
    scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown },
    climate: Climate,
    opts: {
      radius:      number
      heightScale: number
      /** Initial particle count (default 2000). */
      density?:    number
    },
  ) {
    this._scene      = scene
    this._climate    = climate
    this._radius     = opts.radius
    this._heightScale = opts.heightScale
    this._density    = opts.density ?? 2000
    this._flowSpeed  = 0.15
    this._trailLen   = 8
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
   * Returns immediately when not visible (zero cost).
   */
  animate(t: number, dt: number): void {
    if (!this._visible) return

    const N   = this._density
    const K   = this._trailLen
    const rng = this._rng
    const shellR = this._radius + this._heightScale * 2.0

    for (let i = 0; i < N; i++) {
      const di = i * 3
      this._p.set(this._dirs[di], this._dirs[di + 1], this._dirs[di + 2])

      this._climate.windAtTime(this._p, t, this._wind)
      const s = this._wind.speed

      this._age[i] += dt
      const doRespawn = this._age[i] > this._life[i] || s < 0.02

      if (doRespawn) {
        // Uniform random sphere direction via PRNG
        const u1 = rng()
        const u2 = rng()
        const z = 2 * u1 - 1
        const r = Math.sqrt(Math.max(0, 1 - z * z))
        const theta = 2 * Math.PI * u2
        this._p.set(r * Math.cos(theta), z, r * Math.sin(theta))

        this._age[i]  = 0
        this._life[i] = this._lifeBase * (0.5 + rng())

        // Store new direction
        this._dirs[di]     = this._p.x
        this._dirs[di + 1] = this._p.y
        this._dirs[di + 2] = this._p.z

        // Reset entire trail ring to the lifted start position
        const wx = this._p.x * shellR
        const wy = this._p.y * shellR
        const wz = this._p.z * shellR
        for (let k = 0; k < K; k++) {
          const ti = (i * K + k) * 3
          this._trail[ti]     = wx
          this._trail[ti + 1] = wy
          this._trail[ti + 2] = wz
        }
        this._head[i] = 0
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

      // Store new direction
      this._dirs[di]     = this._pN.x
      this._dirs[di + 1] = this._pN.y
      this._dirs[di + 2] = this._pN.z

      // Compute bearing-hue RGB once from the wind already sampled above.
      // Written into _color[i*3..] so _writeBuffers can read it without
      // calling windAtTime a second time.
      {
        const px = this._pN.x, py = this._pN.y, pz = this._pN.z
        const ex = pz, ez = -px
        const eLen = Math.sqrt(ex * ex + ez * ez)
        const ci = i * 3
        if (eLen < 1e-6) {
          // Near-pole: white
          this._color[ci]     = 1
          this._color[ci + 1] = 1
          this._color[ci + 2] = 1
        } else {
          const eNx = ex / eLen, eNz = ez / eLen
          const nx  =  py * eNz
          const ny  =  pz * eNx - px * eNz
          const nz  = -py * eNx
          const wx = this._wind.x, wy = this._wind.y, wz = this._wind.z
          const eDot = wx * eNx + wz * eNz
          const nDot = wx * nx + wy * ny + wz * nz
          const bearing = Math.atan2(eDot, nDot)
          const hue     = ((bearing / (2 * Math.PI)) % 1 + 1) % 1
          hslToRgb(hue, 0.95, 0.55, this._rgb)
          this._color[ci]     = this._rgb[0]
          this._color[ci + 1] = this._rgb[1]
          this._color[ci + 2] = this._rgb[2]
        }
      }

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

  private _allocate(): void {
    const N = this._density
    const K = this._trailLen
    this._dirs  = new Float32Array(N * 3)
    this._age   = new Float32Array(N)
    this._life  = new Float32Array(N)
    this._trail = new Float32Array(N * K * 3)
    this._head  = new Int32Array(N)
    this._color = new Float32Array(N * 3)  // r,g,b per particle; zeroed by default (white=0 is invisible, fine for fresh respawn)
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
      // Uniform random sphere
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
      this._life[i] = this._lifeBase * (0.5 + rng())
      this._age[i]  = rng() * this._life[i]

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
   * Per-vertex brightness fades from 1.0 at the head pair to 0.0 at the tail pair.
   * Color = bearing-hue (cached in _color from the advect loop) * brightness (RGB, no alpha).
   *
   * Called once per animate() call — no allocations, no windAtTime calls.
   */
  private _writeBuffers(): void {
    const N = this._density
    const K = this._trailLen
    const pos = this._posBuf
    const col = this._colBuf
    let vi = 0  // vertex index (each vertex = 2 floats[3])

    for (let i = 0; i < N; i++) {
      // Read cached bearing-hue RGB computed during the advect loop.
      // windAtTime is NOT called here — zero additional wind samples per frame.
      const ci = i * 3
      const hr = this._color[ci]
      const hg = this._color[ci + 1]
      const hb = this._color[ci + 2]

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
        col[vBase]     = hr * brightness
        col[vBase + 1] = hg * brightness
        col[vBase + 2] = hb * brightness
        vi++

        const vBase2 = vi * 3
        // Vertex B (end of segment, nearer to tail)
        const brightnessB = 1 - (s + 1) / (K - 1)
        pos[vBase2]     = this._trail[tB]
        pos[vBase2 + 1] = this._trail[tB + 1]
        pos[vBase2 + 2] = this._trail[tB + 2]
        col[vBase2]     = hr * brightnessB
        col[vBase2 + 1] = hg * brightnessB
        col[vBase2 + 2] = hb * brightnessB
        vi++
      }
    }

    const geoPosAttr = this._geo!.attributes.position as Float32BufferAttribute
    const geoColAttr = this._geo!.attributes.color    as Float32BufferAttribute
    geoPosAttr.needsUpdate = true
    geoColAttr.needsUpdate = true
  }
}

// ---------------------------------------------------------------------------
// HSL → RGB helper (pure function, no allocations, inline-able)
// ---------------------------------------------------------------------------

/** Alloc-free HSL → RGB: writes result into out[0..2] instead of returning a new array. */
function hslToRgb(h: number, s: number, l: number, out: Float32Array): void {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs((h * 6) % 2 - 1))
  const m = l - c / 2
  let r = 0, g = 0, b = 0
  const hi = Math.floor(h * 6)
  switch (hi) {
    case 0: r = c; g = x; break
    case 1: r = x; g = c; break
    case 2: g = c; b = x; break
    case 3: g = x; b = c; break
    case 4: r = x; b = c; break
    default: r = c; b = x; break
  }
  out[0] = r + m
  out[1] = g + m
  out[2] = b + m
}
