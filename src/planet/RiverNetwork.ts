import {
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Mesh,
  Object3D,
  Vector3,
} from 'three'
import { MeshStandardNodeMaterial } from 'three/webgpu'
import { Erosion } from './erosion'
import { Subsurface } from './subsurface'
import { EROSION_VINCISION_AMP } from './terrainSampler'
import { texelToDir, texelIndex, texAng, neighborTexel } from './cubemap'

// Carve gate — MUST match terrainSampler's vIncision gate so the water fill lines
// up with the channel the heightFn actually carves.
const CARVE_LO = 0.05
const CARVE_HI = 0.45

// ---------------------------------------------------------------------------
// RiverNetwork — drainage-network WATER SURFACE traced from the erosion flow
// field. Rivers are an emergent byproduct of the stream-power erosion sim
// (E = K·Q^m·S^n); this renders water in the channels the sampler's V-incision
// carves (heightFn vIncision), because every vertex is draped through heightFn.
//
// Tracing = great-circle streamlines of erosion.flowAt() (downhill tangent),
// stepped ONCE per reach, cube-sphere `visited` grid so each reach traces once.
// Headwaters seed from channel HEADS (one step upstream is below threshold) and
// from AQUIFER/SPRING emergence (water table ≥ terrain) — NOT every channel cell
// (that fragments into dashes as adjacent traces collide on the visited grid).
//
// Geometry = a smooth WATER SURFACE per reach: the coarse traced centerline is
// Catmull-Rom subdivided (kills the blocky ~6 km step angularity) and swept into
// a ribbon whose width ∝ √discharge (hydraulic geometry). Material matches the
// ocean shell (translucent low-roughness blue) so rivers read as water, not lines.
//
// Visualization-only: built main-thread from baked Erosion/Subsurface + heightFn,
// added to the Planet group. Does NOT touch terrain meshing/workers/toBaked.
// ---------------------------------------------------------------------------

export interface RiverParams {
  gridRes:    number   // seed/visited grid resolution
  threshold:  number   // channel-initiation threshold on accAt [0,1]
  stepMul:    number   // step length as a multiple of one grid texel's pitch
  drapeLevel: number   // heightFn LOD used to drape vertices
  maxSteps:   number   // loop guard per reach
  widthBase:  number   // channel width (world m) at zero discharge
  widthScale: number   // extra width (world m) at full discharge (×√acc)
  smoothSub:  number   // Catmull-Rom substeps per traced segment (1 = off)
  fillFraction: number // fraction of the carved channel depth the water fills [0,1]
}

const DEFAULTS: RiverParams = {
  gridRes:    128,
  threshold:  0.08,
  stepMul:    1.0,
  drapeLevel: 8,
  maxSteps:   4000,
  widthBase:  120,
  widthScale: 4200,
  smoothSub:  4,
  fillFraction: 0.7,
}

// Module-level scratch — single-threaded build, never re-entrant.
const _f    = new Vector3()
const _up   = new Vector3()
const _nt   = { face: 0, x: 0, y: 0 }
const _cdir = new Vector3()
const _ndir = new Vector3()
// 8-neighbour offsets (excluding 0,0) for discrete downstream selection.
const NB_DX = [-1, 0, 1, -1, 1, -1, 0, 1]
const NB_DY = [-1, -1, -1, 0, 0, 1, 1, 1]

/** Uniform Catmull-Rom interpolation of one scalar component. */
function catmull(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t, t3 = t2 * t
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
}

/** Scalar smoothstep, clamped. */
function smooth01(lo: number, hi: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)))
  return t * t * (3 - 2 * t)
}

export class RiverNetwork {
  private readonly _scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown }
  private readonly _radius: number
  private readonly _heightScale: number
  private _params: RiverParams
  private _visible = true

  private _mesh: Mesh | null = null
  private _geo: BufferGeometry | null = null
  private _mat: MeshStandardNodeMaterial | null = null

  constructor(
    scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown },
    opts: { radius: number; heightScale: number; params?: Partial<RiverParams> },
  ) {
    this._scene = scene
    this._radius = opts.radius
    this._heightScale = opts.heightScale
    this._params = { ...DEFAULTS, ...(opts.params ?? {}) }
  }

  get visible(): boolean { return this._visible }

  setVisible(v: boolean): void {
    this._visible = v
    if (this._mesh !== null) this._mesh.visible = v
  }

  build(
    erosion: Erosion,
    heightFn: (dir: Vector3, level: number) => number,
    seaLevel: number,
    subsurface: Subsurface | null,
  ): void {
    this._teardown()

    const { gridRes, threshold, stepMul, drapeLevel, maxSteps, widthBase, widthScale, smoothSub, fillFraction } = this._params
    const visited = new Uint8Array(6 * gridRes * gridRes)
    const stepArc = texAng(gridRes) * stepMul
    const lift = this._heightScale * 0.0002  // ~2m: just above the channel floor; small so it doesn't hover at walk scale
    // Seed-scan only needs a coarse sea/spring check; draping (in trace) uses the full
    // drapeLevel so water height matches the high-LOD terrain in walk mode (no float).
    const seedLevel = Math.min(8, drapeLevel)

    const pos: number[] = []
    const nrm: number[] = []

    // Per-reach centerline (world position) + half-width, reused per trace.
    const cwx: number[] = [], cwy: number[] = [], cwz: number[] = [], chw: number[] = []
    // Smoothed centerline buffers (Catmull-Rom output), reused per trace.
    const sx: number[] = [], sy: number[] = [], sz: number[] = [], shw: number[] = []

    const halfWidthFor = (acc: number): number =>
      0.5 * (widthBase + widthScale * Math.sqrt(Math.max(0, acc)))

    // Sweep a smooth water ribbon from the smoothed centerline arrays.
    const emitRibbon = (): void => {
      // Catmull-Rom subdivide cw* → s* (skip if too short or smoothing off).
      sx.length = 0; sy.length = 0; sz.length = 0; shw.length = 0
      const c = cwx.length
      if (c < 2) return
      if (smoothSub <= 1 || c < 3) {
        for (let i = 0; i < c; i++) { sx.push(cwx[i]); sy.push(cwy[i]); sz.push(cwz[i]); shw.push(chw[i]) }
      } else {
        for (let i = 0; i < c - 1; i++) {
          const i0 = i > 0 ? i - 1 : 0, i1 = i, i2 = i + 1, i3 = i < c - 2 ? i + 2 : c - 1
          for (let j = 0; j < smoothSub; j++) {
            const t = j / smoothSub
            sx.push(catmull(cwx[i0], cwx[i1], cwx[i2], cwx[i3], t))
            sy.push(catmull(cwy[i0], cwy[i1], cwy[i2], cwy[i3], t))
            sz.push(catmull(cwz[i0], cwz[i1], cwz[i2], cwz[i3], t))
            shw.push(chw[i1] + (chw[i2] - chw[i1]) * t)
          }
        }
        sx.push(cwx[c - 1]); sy.push(cwy[c - 1]); sz.push(cwz[c - 1]); shw.push(chw[c - 1])
      }

      const m = sx.length
      if (m < 2) return
      const Lx = new Array<number>(m), Ly = new Array<number>(m), Lz = new Array<number>(m)
      const Rx = new Array<number>(m), Ry = new Array<number>(m), Rz = new Array<number>(m)
      const Nx = new Array<number>(m), Ny = new Array<number>(m), Nz = new Array<number>(m)
      for (let i = 0; i < m; i++) {
        // Radial (up) direction from the world position.
        const wx = sx[i], wy = sy[i], wz = sz[i]
        const wl = Math.hypot(wx, wy, wz) || 1
        const px = wx / wl, py = wy / wl, pz = wz / wl
        Nx[i] = px; Ny[i] = py; Nz[i] = pz
        // Path tangent from neighbours, projected into the tangent plane.
        const ia = i > 0 ? i - 1 : 0, ib = i < m - 1 ? i + 1 : m - 1
        let tx = sx[ib] - sx[ia], ty = sy[ib] - sy[ia], tz = sz[ib] - sz[ia]
        const tr = tx * px + ty * py + tz * pz
        tx -= tr * px; ty -= tr * py; tz -= tr * pz
        let tl = Math.hypot(tx, ty, tz)
        if (tl < 1e-9) { tx = py; ty = -px; tz = 0; tl = Math.hypot(tx, ty, tz) || 1 }
        tx /= tl; ty /= tl; tz /= tl
        // Cross-channel = p × tangent (unit, tangent to sphere).
        let ex = py * tz - pz * ty, ey = pz * tx - px * tz, ez = px * ty - py * tx
        const el = Math.hypot(ex, ey, ez) || 1
        ex /= el; ey /= el; ez /= el
        const hw = shw[i]
        Lx[i] = wx + ex * hw; Ly[i] = wy + ey * hw; Lz[i] = wz + ez * hw
        Rx[i] = wx - ex * hw; Ry[i] = wy - ey * hw; Rz[i] = wz - ez * hw
      }
      for (let i = 0; i < m - 1; i++) {
        pos.push(Lx[i], Ly[i], Lz[i],  Rx[i], Ry[i], Rz[i],  Lx[i + 1], Ly[i + 1], Lz[i + 1])
        nrm.push(Nx[i], Ny[i], Nz[i],  Nx[i], Ny[i], Nz[i],  Nx[i + 1], Ny[i + 1], Nz[i + 1])
        pos.push(Rx[i], Ry[i], Rz[i],  Rx[i + 1], Ry[i + 1], Rz[i + 1],  Lx[i + 1], Ly[i + 1], Lz[i + 1])
        nrm.push(Nx[i], Ny[i], Nz[i],  Nx[i + 1], Ny[i + 1], Nz[i + 1],  Nx[i + 1], Ny[i + 1], Nz[i + 1])
      }
    }

    let nTraces = 0, totalPts = 0
    const brk = { sea: 0, lake: 0, incoh: 0, visited: 0 }

    const trace = (sf: number, sx: number, sy: number): void => {
      let cf = sf, cx = sx, cy = sy
      cwx.length = 0; cwy.length = 0; cwz.length = 0; chw.length = 0
      let haveLast = false, ldx = 0, ldy = 0, ldz = 0

      for (let step = 0; step < maxSteps; step++) {
        const idx = texelIndex(cf, cx, cy, gridRes)
        const already = visited[idx] === 1

        texelToDir(cf, cx, cy, gridRes, _cdir)
        const acc = erosion.accAt(_cdir)
        const h = heightFn(_cdir, drapeLevel)
        // Water fills the carved channel ~fillFraction deep so its surface clears the
        // floor's fine-detail bumps but stays below the banks (which occlude its edges).
        const carveNorm = EROSION_VINCISION_AMP * smooth01(CARVE_LO, CARVE_HI, acc)
        const r = this._radius + (h + fillFraction * carveNorm) * this._heightScale + lift
        cwx.push(_cdir.x * r); cwy.push(_cdir.y * r); cwz.push(_cdir.z * r); chw.push(halfWidthFor(acc))

        if (already) { brk.visited++; break }   // merged into an already-traced reach here
        visited[idx] = 1
        if (h < seaLevel) { brk.sea++; break }
        if (erosion.lakeMaskAt(_cdir) > 0.5) { brk.lake++; break }

        // Discrete downstream step: move to the 8-neighbour the flow points toward.
        // Two reaches entering the same cell pick the SAME neighbour, so they merge into
        // one shared path -> a guaranteed converging tree. Incoherent flow (len ~ 0) is
        // bridged with the last good direction.
        erosion.flowAt(_cdir, _f)
        const rdl = _f.dot(_cdir)
        _f.x -= rdl * _cdir.x; _f.y -= rdl * _cdir.y; _f.z -= rdl * _cdir.z
        const fl = Math.hypot(_f.x, _f.y, _f.z)
        let gx: number, gy: number, gz: number
        if (fl > 1e-3) {
          const inv = 1 / fl; gx = _f.x * inv; gy = _f.y * inv; gz = _f.z * inv
          ldx = gx; ldy = gy; ldz = gz; haveLast = true
        } else if (haveLast) {
          gx = ldx; gy = ldy; gz = ldz
        } else { brk.incoh++; break }

        let bestDot = -Infinity, bf = cf, bx = cx, by = cy
        for (let k = 0; k < 8; k++) {
          neighborTexel(cf, cx, cy, NB_DX[k] as -1 | 0 | 1, NB_DY[k] as -1 | 0 | 1, gridRes, _nt)
          texelToDir(_nt.face, _nt.x, _nt.y, gridRes, _ndir)
          const d = (_ndir.x - _cdir.x) * gx + (_ndir.y - _cdir.y) * gy + (_ndir.z - _cdir.z) * gz
          if (d > bestDot) { bestDot = d; bf = _nt.face; bx = _nt.x; by = _nt.y }
        }
        if (bestDot <= 0 || (bf === cf && bx === cx && by === cy)) { brk.incoh++; break }
        cf = bf; cx = bx; cy = by
      }
      nTraces++; totalPts += cwx.length
      emitRibbon()
    }

    // Seed from channel HEADS + springs.
    const startDir = new Vector3()
    let channelSeeds = 0, springSeeds = 0, maxAcc = 0
    const cosArc = Math.cos(stepArc), sinArc = Math.sin(stepArc)
    for (let face = 0; face < 6; face++) {
      for (let y = 0; y < gridRes; y++) {
        for (let x = 0; x < gridRes; x++) {
          texelToDir(face, x, y, gridRes, startDir)
          const a = erosion.accAt(startDir)
          if (a > maxAcc) maxAcc = a
          if (visited[texelIndex(face, x, y, gridRes)] === 1) continue
          if (a < 0.02) continue
          const h = heightFn(startDir, seedLevel)
          if (h < seaLevel) continue
          if (erosion.lakeMaskAt(startDir) > 0.5) continue
          const isSpring = subsurface !== null && subsurface.waterTableAt(startDir) > h
          if (a < threshold && !isSpring) continue
          let isHead = isSpring
          if (!isHead) {
            erosion.flowAt(startDir, _f)
            const rdl = _f.dot(startDir)
            _f.x -= rdl * startDir.x; _f.y -= rdl * startDir.y; _f.z -= rdl * startDir.z
            const fl = Math.hypot(_f.x, _f.y, _f.z)
            if (fl > 1e-6) {
              const inv = 1 / fl
              _up.set(
                startDir.x * cosArc - _f.x * inv * sinArc,
                startDir.y * cosArc - _f.y * inv * sinArc,
                startDir.z * cosArc - _f.z * inv * sinArc,
              ).normalize()
              isHead = erosion.accAt(_up) < threshold
            } else {
              isHead = true
            }
          }
          if (!isHead) continue
          if (isSpring && a < threshold) springSeeds++; else channelSeeds++
          trace(face, x, y)
        }
      }
    }

    const brkMax = nTraces - brk.sea - brk.lake - brk.incoh - brk.visited
    // eslint-disable-next-line no-console
    console.log(`[RiverNetwork] thr=${threshold} maxAcc=${maxAcc.toFixed(3)} seeds(ch/spr)=${channelSeeds}/${springSeeds} traces=${nTraces} avgLen=${(totalPts / Math.max(1, nTraces)).toFixed(1)} breaks(sea/lake/incoh/visited/max)=${brk.sea}/${brk.lake}/${brk.incoh}/${brk.visited}/${brkMax} tris=${pos.length / 9}`)

    if (pos.length === 0) return

    this._geo = new BufferGeometry()
    this._geo.setAttribute('position', new Float32BufferAttribute(new Float32Array(pos), 3))
    this._geo.setAttribute('normal',   new Float32BufferAttribute(new Float32Array(nrm), 3))

    // Match the ocean shell's water look so rivers and sea read consistently.
    this._mat = new MeshStandardNodeMaterial({
      color: 0x2e6b8f, roughness: 0.15, metalness: 0,
      transparent: true, opacity: 0.8, side: DoubleSide, depthWrite: false,
    })
    this._mesh = new Mesh(this._geo, this._mat)
    this._mesh.visible = this._visible
    this._mesh.frustumCulled = false
    this._mesh.renderOrder = 6
    this._scene.add(this._mesh)
  }

  dispose(): void { this._teardown() }

  private _teardown(): void {
    if (this._mesh !== null) { this._scene.remove(this._mesh); this._mesh = null }
    if (this._geo !== null) { this._geo.dispose(); this._geo = null }
    if (this._mat !== null) { this._mat.dispose(); this._mat = null }
  }
}
