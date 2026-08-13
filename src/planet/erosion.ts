/**
 * Global stream-power erosion field.
 *
 * Bakes once on the main thread and ships to Web Workers via toBaked()/fromBaked(),
 * mirroring the climate.ts pattern exactly. Workers call fromBaked() to reconstruct
 * a sampler-only instance (no re-simulation).
 *
 * Pipeline:
 *   1. Sample height + rainfall on a 256² cubemap at LOD 5.
 *   2. Compute provisional sea level from ocean coverage fraction.
 *   3. Priority-flood (Barnes 2014) to fill depressions → lakeLevel.
 *   4. MFD flow routing (multiple-flow-direction, all strictly-lower neighbors).
 *   5. Discharge accumulation in topological order (upstream→downstream).
 *   6. ~60 iterations of erosion-deposition (Davy–Lague): sediment flux routed
 *      high→low per topological order; deposition dominates near sea/flats → deltas.
 *   7. ~80 iterations of thermal diffusion (talus collapse).
 *   8. Lake flatten.
 *   9. Compose erosionDelta = workH − H0, clamped.
 *  10. 3×3 cross-face seam blur on erosionDelta (3 passes) + flowAccum (2 passes).
 *
 * Determinism: all randomness derived from seed via deriveSeed (stream 9 = bake,
 * stream 10 = domain-warp noise). No Math.random / Date.now / performance.now.
 */

import { Vector3 } from 'three'
import {
  texelToDir,
  texelIndex,
  neighborTexel,
  sampleSmooth,
} from './cubemap'
import { Climate, ClimateSample } from './climate'
import { Tectonics } from './tectonics'
import { createNoise3D } from './noise'
import { HEIGHT_SCALE, HEIGHT_SCALE_REF, RADIUS, RADIUS_REF, RES_REF, deriveSlopeThresh, deriveErosionRes } from './worldConstants'

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

// B_HEIGHT_SCALE: re-exported alias for HEIGHT_SCALE from worldConstants.
// Kept for back-compat; drives 4 metre-space ops in the B-path — must equal HEIGHT_SCALE.
export const B_HEIGHT_SCALE = HEIGHT_SCALE

const EROSION_RES           = deriveErosionRes(RADIUS)
const EROSION_BAKE_LEVEL    = 5
const LAKE_SENTINEL         = -999

const DEFAULT_K0            = 0.35    // incision rate (bumped for visible carve)
const DEFAULT_M_EXP         = 0.45   // discharge exponent
const DEFAULT_N_EXP         = 1.0    // slope exponent
const DEFAULT_TALUS         = 0.03   // thermal talus angle (normalized height)
const DEFAULT_KW_MIN        = 0.05   // min rain weight

const DEFAULT_ITERS_INCISION  = 60
const DEFAULT_ITERS_THERMAL   = 80
const DEFAULT_DEPOSITION_G  = 1.5   // deposition rate coefficient (lowered from 3.0 — reduces deposit rings)
const EPS_Q                 = 1e-10 // guard divisor in deposition term

const MFD_P         = 6     // MFD slope exponent; higher = sharper channel concentration

const EROSION_MAX   = 0.15
const SEDIMENT_MAX  = 0.15

// ---------------------------------------------------------------------------
// Temperature-gated process constants (Part B)
// ---------------------------------------------------------------------------
// Climate base temperature (°C, deterministic path — no sun term) modulates:
//   1. Thermal-diffusion talus angle: cold → lower talus (freeze-thaw shatters debris
//      faster, slopes fail at shallower angles); warm → higher talus (chemical cementation).
//   2. Chemical-weathering factor on stream-power incision: warm+wet → stronger chemical
//      dissolution (gentler rounded relief); cold → reduced chemical action (mechanical only).
//
// Both modulations are CONTINUOUS functions of temperature (smoothstep blend between
// cold and warm poles) — never a hard threshold — so they cannot crack seams.
// They affect ONLY the bake; they never enter heightFn geometry.
//
// Transition zone: linearly interpolate between cold and warm behaviour from
//   TEMP_COLD_THRESH (below = fully cold) to TEMP_WARM_THRESH (above = fully warm).
//   Earth ~15°C sits in the warm half; high-elevation / polar cells fall cold.
const TEMP_COLD_THRESH = -10   // °C — fully cold process regime below this
const TEMP_WARM_THRESH =  20   // °C — fully warm process regime above this

// Talus multiplier: cold cells get TEMP_TALUS_COLD_MUL × talusScale; warm cells get 1.0.
// Freeze-thaw shatters material quickly, driving talus below the purely-hardness baseline.
// Range (0,1]: 1 = no modulation; 0.6 = 40% lower talus angle at pole/altitude.
const TEMP_TALUS_COLD_MUL = 0.65   // −35% talus in fully-cold regime
const TEMP_TALUS_WARM_MUL = 1.10   // +10% talus in fully-warm regime (cementation)

// Chemical weathering boost on incision: warm+wet cells have additional dissolution.
// Implemented as a multiplicative factor on effective K0 (only the wet + warm term).
// TEMP_CHEM_WARM_EXTRA: additional K0 fraction at max temperature AND max moisture (1).
// TEMP_CHEM_COLD_FACTOR: K0 scale at fully-cold regime (< 1 → mechanical weathering only).
const TEMP_CHEM_WARM_EXTRA  = 0.30   // +30% incision when warm and saturated
const TEMP_CHEM_COLD_FACTOR = 0.80   // −20% incision in fully-cold (chemical action suppressed)

// ---------------------------------------------------------------------------
// Rock-hardness erodibility constants
// ---------------------------------------------------------------------------
// HARD_ERODIBILITY_STR: how strongly hardness modulates the stream-power incision rate K0.
//   erodibility = 1 + HARD_ERODIBILITY_STR * (0.5 - hardness)
//   → hard rock (hardness=1) → erodibility = 1 - 0.5*STR  (erodes slower, ridges/escarpments)
//   → soft rock (hardness=0) → erodibility = 1 + 0.5*STR  (erodes faster, smooth basins)
//   Range [0, 2]: 0 = no modulation (pre-T2 behaviour), 1.0 = ±50% K0 contrast.
const HARD_ERODIBILITY_STR = 1.2   // ±60% K0 contrast between hard/soft substrates

// HARD_TALUS_STR: fraction by which hardness scales the thermal-talus angle.
//   talusLocal = talus * (1 + HARD_TALUS_STR * (hardness - 0.5))
//   → hard rock holds steeper slopes (cliffier); soft rock collapses to gentler angles.
//   Range [0, 1]: 0 = no modulation, 1.0 = ±50% talus contrast.
const HARD_TALUS_STR       = 0.6   // ±30% talus contrast between hard/soft substrates

// ---------------------------------------------------------------------------
// Depositional environment constants
// ---------------------------------------------------------------------------
// Each recognizable landform class gets its own deposition-rate coefficient (G)
// and sediment accumulation cap.  Values are conservative — all stay well inside
// the headroom that terrainSampler.ts allows after clamping erosionDelta at ~line 538.
//
// Environment codes stored in depositEnv (Uint8Array):
//   0 = DEFAULT   – generic / lake-floor (existing behaviour)
//   1 = FLOODPLAIN – low-gradient mid-reach with moderate discharge
//   2 = ALLUVIAL_FAN – steep-to-flat transition (fan apex)
//   3 = DELTA       – channel entering ocean / lake margin

const ENV_DEFAULT    = 0
const ENV_FLOODPLAIN = 1
const ENV_FAN        = 2
const ENV_DELTA      = 3

// Deposition rate coefficients — units: same as DEFAULT_DEPOSITION_G (dimensionless).
// ENV_DEFAULT uses opts.depositionG (= DEFAULT_DEPOSITION_G = 1.5) so callers can
// override the baseline without touching these per-class constants.
const DEP_G_FLOODPLAIN = 2.0   // moderate, wide spreading
const DEP_G_FAN        = 3.5   // rapid load-drop at grade break
const DEP_G_DELTA      = 2.8   // high at shoreline margin

// Per-class sediment accumulation caps (same units as SEDIMENT_MAX = 0.15)
const SEDIMENT_MAX_DEFAULT    = SEDIMENT_MAX    // 0.15
const SEDIMENT_MAX_FLOODPLAIN = 0.22            // genuine floodplain flats
const SEDIMENT_MAX_FAN        = 0.28            // alluvial fan cone relief
const SEDIMENT_MAX_DELTA      = 0.25            // delta lobes

// Threshold tuning: all dimensionless in the H0 normalised-height space [0..1].
// Slope metric = ΔH_normalized / 1_cell (no explicit denominator; implicit cell = 1).
//
// Derivation: for a fixed physical slope angle θ the per-cell normalised rise is
//   ΔH_normalized = tan(θ) × cell_arc_metres / HEIGHT_SCALE
// Cell arc-length scales as (radius / res), so the threshold scales by
//   (radius / RADIUS_REF) × (HEIGHT_SCALE_REF / HEIGHT_SCALE) × (RES_REF / res)
// Under a uniform scale (radius×k, HEIGHT_SCALE×k) the first two factors cancel,
// leaving the threshold unchanged — scale-invariant.
// This is exactly deriveSlopeThresh() from worldConstants.ts.
//
// Baseline (RADIUS_REF=50 000, RES_REF=256):
//   DEP_SLOPE_LOW_THRESH_BASE  = 0.002
//   DEP_SLOPE_HIGH_THRESH_BASE = 0.008
//
// NOTE: these thresholds scale with the runtime bake resolution (opts.res), NOT the
// module-level EROSION_RES constant.  They are computed as local variables inside the
// bake after `const res` is resolved, so the physical slope angle they represent
// remains correct regardless of which slider value the user has chosen.
const DEP_SLOPE_LOW_THRESH_BASE  = 0.002   // baseline flat threshold (at RADIUS_REF, RES_REF)
const DEP_SLOPE_HIGH_THRESH_BASE = 0.008   // baseline steep threshold (at RADIUS_REF, RES_REF)
const DEP_Q_FLOODPLAIN_MIN    = 0.05    // normalised discharge threshold for floodplain
const DEP_Q_DELTA_MIN         = 0.08    // min discharge to classify as delta source
const DEP_SHORE_HOPS          = 2       // BFS depth: cells within this many steps of ocean/lake = shoreline band
// Fraction of surplus Qs spread laterally in fan cells (perpendicular to downslope)
const FAN_LATERAL_SPREAD      = 0.20
// FILL_EPS: tiny per-pop increment for Barnes ε-fill; keyed to flood ORDER, not spatial index.
// FILL_EPS * N ≈ 1e-9 * 393216 ≈ 4e-4 — negligible vs [-1,1] height range but monotonic.
const FILL_EPS      = 1e-9
// Minimum meaningful flood raise to count as a real lake cell (not just ε-gradient)
const LAKE_MIN_DEPTH = 0.002

// ---------------------------------------------------------------------------
// Domain-warp constants for query-time grid-alignment suppression
// ---------------------------------------------------------------------------
// Amplitude in direction-space radians; 1 erosion cell ≈ 0.006 rad at res 256.
// WARP_AMP targets ~5 cells of warp at the module-level default res (256).
// If the user raises res to 512 via the GUI slider the warp covers ~10 cells —
// slightly more, but still within the "break grid alignment without smearing"
// window and not worth a dynamic adjustment (baked amplitude, not per-query).
const WARP_AMP   = 0.03
const WARP_FREQ  = 1.8
const WARP_OCT   = 2
type Noise3DFn = (x: number, y: number, z: number) => number

// ---------------------------------------------------------------------------
// Local deterministic PRNG (copied verbatim from climate.ts)
// ---------------------------------------------------------------------------

function splitmix32Step(a: number): number {
  a |= 0
  a = (a + 0x9e3779b9) | 0
  let t = a ^ (a >>> 16)
  t = Math.imul(t, 0x21f0aaad)
  t = t ^ (t >>> 15)
  t = Math.imul(t, 0x735a2d97)
  return (t ^ (t >>> 15)) >>> 0
}

/**
 * Derive a child seed from a master seed and a stream id.
 * Erosion uses:
 *   stream 9 = bake-time simulation noise (reserved; currently unused inside bake)
 *   stream 10 = query-time domain-warp noise (breaks grid alignment in field lookups)
 * Both chosen well clear of tectonics (0..15) and climate (200).
 */
function deriveSeed(masterSeed: number, stream: number): number {
  const s = (masterSeed ^ Math.imul(stream + 1, 0xdeadbeef)) >>> 0
  return splitmix32Step(s)
}

// ---------------------------------------------------------------------------
// MinHeap (priority queue keyed by (elev ASC, idx ASC))
// ---------------------------------------------------------------------------

interface HeapItem { elev: number; idx: number }

class MinHeap {
  private data: HeapItem[] = []

  get size(): number { return this.data.length }

  /** Reset the heap to empty without re-allocating the backing array. */
  clear(): void { this.data.length = 0 }

  push(item: HeapItem): void {
    this.data.push(item)
    this._siftUp(this.data.length - 1)
  }

  pop(): HeapItem | undefined {
    const top = this.data[0]
    const last = this.data.pop()!
    if (this.data.length > 0) {
      this.data[0] = last
      this._siftDown(0)
    }
    return top
  }

  private _lt(a: HeapItem, b: HeapItem): boolean {
    return a.elev < b.elev || (a.elev === b.elev && a.idx < b.idx)
  }

  private _siftUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1
      if (this._lt(this.data[i], this.data[parent])) {
        const tmp = this.data[i]; this.data[i] = this.data[parent]; this.data[parent] = tmp
        i = parent
      } else {
        break
      }
    }
  }

  private _siftDown(i: number): void {
    const n = this.data.length
    for (;;) {
      const l = 2 * i + 1
      const r = 2 * i + 2
      let min = i
      if (l < n && this._lt(this.data[l], this.data[min])) min = l
      if (r < n && this._lt(this.data[r], this.data[min])) min = r
      if (min === i) break
      const tmp = this.data[i]; this.data[i] = this.data[min]; this.data[min] = tmp
      i = min
    }
  }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ErosionBaked {
  res:          number
  erosionDelta: Float32Array  // 6*res*res; normalized height delta (<=0 incision, >=0 sediment)
  flowAccum:    Float32Array  // 6*res*res; normalized log discharge 0..1
  flowDirX:     Float32Array  // 6*res*res; world-space unit downhill tangent X
  flowDirY:     Float32Array  // 6*res*res; world-space unit downhill tangent Y
  flowDirZ:     Float32Array  // 6*res*res; world-space unit downhill tangent Z
  lakeLevel:    Float32Array  // 6*res*res; basin spill elevation; LAKE_SENTINEL where not a lake
  lakeMask:     Float32Array  // 6*res*res; 1 inside a real lake basin, 0 elsewhere (safe to bilinear)
  /**
   * Depositional environment code per texel (ENV_DEFAULT/FLOODPLAIN/FAN/DELTA).
   * DISCRETE — sample nearest-neighbor only (depositEnvAt); NEVER feed into heightFn.
   * Used exclusively for mesh color pass; color discontinuities are acceptable.
   */
  depositEnv:   Uint8Array    // 6*res*res; environment code ∈ {0,1,2,3}
  K0:       number
  mExp:     number
  nExp:     number
  talus:    number
  kwMin:    number
  warpSeed: number            // seed for query-time domain-warp noise (stream 10)
}

export interface ErosionOpts {
  seed:          number
  heightFn:      (dir: Vector3, level: number) => number
  climate:       Climate
  oceanCoverage: number
  res?:          number   // default 256
  // tuning overrides
  K0?:           number
  mExp?:         number
  nExp?:         number
  talus?:        number
  kwMin?:        number
  depositionG?:  number   // deposition rate coefficient (default DEFAULT_DEPOSITION_G)
  /**
   * Baked Tectonics instance for per-cell rock hardness sampling during bake.
   * When provided, incision rate (K0) is modulated by hardness (hard→slower,
   * soft→faster) and talus angle is biased so hard rock holds cliffier slopes.
   * Hardness sampled via tectonics.hardnessAt() — same C1 baked-grid lookup
   * used by T1 / query().rockHardness; no new worker field needed (bake-only).
   * When absent, behaviour is byte-identical to pre-T2 (no modulation).
   */
  tectonics?: Tectonics
  // B path — iterated uplift+erode loop (all optional; absent = existing frozen-flow path)
  /** 6*res*res uplift field; when present, runs the iterated B loop instead of the frozen-flow path. */
  upliftForcing?: Float32Array
  /** Number of B loop iterations (default 30). */
  bSteps?:        number
  /** Uplift magnitude per step in metres (default 60). */
  bUpliftRate?:   number
}

// ---------------------------------------------------------------------------
// Helper — decode flat cell index back to (face, x, y)
// ---------------------------------------------------------------------------

function cellToFaceXY(c: number, res: number): { face: number; x: number; y: number } {
  const rr    = res * res
  const face  = Math.floor(c / rr)
  const rem   = c % rr
  const y     = Math.floor(rem / res)
  const x     = rem % res
  return { face, x, y }
}

// ---------------------------------------------------------------------------
// Erosion
// ---------------------------------------------------------------------------

export class Erosion {
  private readonly res:          number
  private readonly erosionDelta: Float32Array
  private readonly flowAccum:    Float32Array
  private readonly flowDirX:     Float32Array
  private readonly flowDirY:     Float32Array
  private readonly flowDirZ:     Float32Array
  private readonly lakeLevel:    Float32Array
  private readonly lakeMask:     Float32Array
  /** Depositional environment code (ENV_*) per texel; nearest-neighbor query only. */
  private readonly depositEnv:   Uint8Array
  private readonly K0:           number
  private readonly mExp:         number
  private readonly nExp:         number
  private readonly talus:        number
  private readonly kwMin:        number
  private readonly depositionG:  number

  // Domain-warp noise (query-time; breaks cubemap grid alignment in field lookups).
  // Rebuilt in constructor, fromBaked, and identity — determinism is critical.
  private _warpSeed: number = 0
  private _warpNoise!: Noise3DFn

  // Zero-alloc scratch for query methods
  private readonly _scratch  = new Vector3()
  private readonly _scratch2 = new Vector3()

  constructor(opts: ErosionOpts) {
    const res         = opts.res         ?? EROSION_RES
    const K0          = opts.K0          ?? DEFAULT_K0
    const mExp        = opts.mExp        ?? DEFAULT_M_EXP
    const nExp        = opts.nExp        ?? DEFAULT_N_EXP
    const talus       = opts.talus       ?? DEFAULT_TALUS
    const kwMin       = opts.kwMin       ?? DEFAULT_KW_MIN
    const depositionG = opts.depositionG ?? DEFAULT_DEPOSITION_G

    this.res         = res
    this.K0          = K0
    this.mExp        = mExp
    this.nExp        = nExp
    this.talus       = talus
    this.kwMin       = kwMin
    this.depositionG = depositionG

    // Stream 9: reserved (was placeholder; kept for determinism of stream 10 derivation).
    void deriveSeed(opts.seed, 9)
    // Stream 10: domain-warp noise for query-time grid-alignment suppression.
    this._warpSeed  = deriveSeed(opts.seed, 10)
    this._warpNoise = createNoise3D(this._warpSeed)

    const N = 6 * res * res

    // Scratch direction/cell objects
    const _dir   = new Vector3()
    const _tmp1  = new Vector3()
    const _tmp2  = new Vector3()
    const nb     = { face: 0, x: 0, y: 0 }

    // -----------------------------------------------------------------------
    // Shared blur helper (used by both paths at the end)
    // -----------------------------------------------------------------------

    const blurField = (src: Float32Array<ArrayBuffer>, passes: number): Float32Array<ArrayBuffer> => {
      let cur: Float32Array<ArrayBuffer> = src
      const tmp = new Float32Array(N)
      for (let pass = 0; pass < passes; pass++) {
        for (let face = 0; face < 6; face++) {
          for (let y = 0; y < res; y++) {
            for (let x = 0; x < res; x++) {
              const c = texelIndex(face, x, y, res)
              let sum = 0
              let count = 0
              for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                  neighborTexel(face, x, y, dx as -1|0|1, dy as -1|0|1, res, nb)
                  sum += cur[texelIndex(nb.face, nb.x, nb.y, res)]
                  count++
                }
              }
              tmp[c] = sum / count
            }
          }
        }
        cur = new Float32Array(tmp)
      }
      return cur
    }

    if (opts.upliftForcing) {
      // =====================================================================
      // B PATH — iterated uplift+erode loop
      // =====================================================================
      // opts.heightFn is the base-only seed (no orogenic stamps).
      // upliftForcing[c] ∈ [0,1] is the per-cell uplift weight (pre-blurred).

      const upliftForcing = opts.upliftForcing
      const bSteps        = opts.bSteps    ?? 30
      // bUpliftRate authored at HEIGHT_SCALE_REF=1200 as 60 m/step; fallback scales with
      // B_HEIGHT_SCALE so normalized (/ B_HEIGHT_SCALE) contribution is scale-invariant.
      const bUpliftRate   = opts.bUpliftRate ?? (60 * (B_HEIGHT_SCALE / HEIGHT_SCALE_REF))

      // Constants in metre-space
      const DH_CLAMP_M   = 0.02 * B_HEIGHT_SCALE   // 24 m per-cell per-iteration cap
      const TALUS_M      = talus * B_HEIGHT_SCALE   // thermal talus in metres
      const LAKE_MIN_M   = LAKE_MIN_DEPTH * B_HEIGHT_SCALE  // 2.4 m minimum lake depth
      const B_DELTA_MAX  = 0.55

      // -----------------------------------------------------------------------
      // Step 1 — Sample H0 grid (in metres) + rainfall
      // -----------------------------------------------------------------------

      const H0   = new Float32Array(N)   // in metres
      const rain = new Float32Array(N)
      const climateSample: ClimateSample = { temperature: 0, moisture: 0 }

      for (let face = 0; face < 6; face++) {
        for (let y = 0; y < res; y++) {
          for (let x = 0; x < res; x++) {
            const c = texelIndex(face, x, y, res)
            texelToDir(face, x, y, res, _dir)
            const hNorm = opts.heightFn(_dir, EROSION_BAKE_LEVEL)
            H0[c] = hNorm * B_HEIGHT_SCALE   // convert to metres
            opts.climate.sample(_dir, hNorm, climateSample)
            rain[c] = Math.max(kwMin, climateSample.moisture)
          }
        }
      }

      // Rock hardness per cell — B path (in metre-space, same formula as existing path).
      // erodibilityB[c] modulates K0; talusScaleB[c] modulates thermal talus angle.
      // TALUS_M declared above (= talus * B_HEIGHT_SCALE).
      const erodibilityB = new Float32Array(N).fill(1.0)
      const talusScaleB  = new Float32Array(N).fill(TALUS_M)
      if (opts.tectonics) {
        for (let face = 0; face < 6; face++) {
          for (let y = 0; y < res; y++) {
            for (let x = 0; x < res; x++) {
              const c = texelIndex(face, x, y, res)
              texelToDir(face, x, y, res, _dir)
              const h = opts.tectonics.hardnessAt(_dir)
              erodibilityB[c] = 1.0 + HARD_ERODIBILITY_STR * (0.5 - h)
              talusScaleB[c]  = TALUS_M * (1.0 + HARD_TALUS_STR * (h - 0.5))
            }
          }
        }
      }

      // Fixed sea level from seed H0 (in metres); isOcean mask is FIXED for the loop.
      const sortedB = H0.slice().sort()
      const provSeaLevelM = sortedB[Math.min(N - 1, Math.floor(opts.oceanCoverage * N))]

      const isOcean = new Uint8Array(N)
      for (let c = 0; c < N; c++) {
        if (H0[c] < provSeaLevelM) isOcean[c] = 1
      }

      // -----------------------------------------------------------------------
      // Step 2 — Neighbor table (same cross-face logic)
      // -----------------------------------------------------------------------

      const neigh: number[][] = new Array(N)
      for (let c = 0; c < N; c++) {
        const { face, x, y } = cellToFaceXY(c, res)
        const ns: number[] = []
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue
            neighborTexel(face, x, y, dx as -1|0|1, dy as -1|0|1, res, nb)
            const nc = texelIndex(nb.face, nb.x, nb.y, res)
            if (nc !== c) ns.push(nc)
          }
        }
        neigh[c] = ns
      }

      // -----------------------------------------------------------------------
      // Working height array (in metres, modified each step)
      // -----------------------------------------------------------------------

      let workH = H0.slice()  // mutable working copy in metres

      // Last-step results (updated each iteration, finalized after loop)
      let lakeLevel    = new Float32Array(N).fill(LAKE_SENTINEL)
      let lakeMask     = new Float32Array(N)
      // Hoisted: allocated once, reset each B-step (avoids GB-scale GC churn).
      // Numerical output is byte-identical — reset to empty before each use.
      const mfdNeighbors: Array<Array<{ n: number; w: number }>> = new Array(N)
      const heapB = new MinHeap()
      let flowDirX     = new Float32Array(N)
      let flowDirY     = new Float32Array(N)
      let flowDirZ     = new Float32Array(N)
      let Q            = new Float64Array(N)
      let landCells: number[] = []
      let Hf           = new Float32Array(N)

      let bufA = new Float32Array(N)
      let bufB = new Float32Array(N)

      for (let step = 0; step < bSteps; step++) {
        // -----------------------------------------------------------------
        // a. Inject uplift on land cells
        // -----------------------------------------------------------------
        for (let c = 0; c < N; c++) {
          if (!isOcean[c]) {
            workH[c] += bUpliftRate * upliftForcing[c]
          }
        }

        // -----------------------------------------------------------------
        // b+c. Priority-flood on current workH
        // -----------------------------------------------------------------
        Hf = workH.slice()
        lakeLevel.fill(LAKE_SENTINEL)
        const inQueueB    = new Uint8Array(N)
        const processedB  = new Uint8Array(N)
        heapB.clear()

        for (let c = 0; c < N; c++) {
          if (isOcean[c]) {
            heapB.push({ elev: workH[c], idx: c })
            inQueueB[c] = 1
          }
        }

        let floodCounterB = 0
        while (heapB.size > 0) {
          const item = heapB.pop()!
          const { elev, idx } = item
          if (processedB[idx]) continue
          processedB[idx] = 1
          ++floodCounterB

          for (const n of neigh[idx]) {
            if (processedB[n]) continue
            const newElev = Math.max(Hf[n], elev + FILL_EPS * floodCounterB)
            if (newElev > Hf[n]) {
              if (newElev - workH[n] > LAKE_MIN_M) {
                lakeLevel[n] = newElev
              }
              Hf[n] = newElev
            }
            if (!inQueueB[n]) {
              heapB.push({ elev: Hf[n], idx: n })
              inQueueB[n] = 1
            }
          }
        }

        lakeMask = new Float32Array(N)
        for (let c = 0; c < N; c++) {
          lakeMask[c] = lakeLevel[c] !== LAKE_SENTINEL ? 1 : 0
        }

        // -----------------------------------------------------------------
        // d. Re-sort land cells by current Hf descending
        // -----------------------------------------------------------------
        landCells = []
        for (let c = 0; c < N; c++) {
          if (!isOcean[c]) landCells.push(c)
        }
        landCells.sort((a, b) => {
          const dh = Hf[b] - Hf[a]
          if (dh !== 0) return dh > 0 ? 1 : -1
          return b - a
        })

        // -----------------------------------------------------------------
        // e. Re-route MFD on current Hf
        // -----------------------------------------------------------------
        // mfdNeighbors is hoisted; each cell slot is cleared below in the loop
        // (mfdNeighbors[c] = []) before being written, matching prior behaviour.
        flowDirX = new Float32Array(N)
        flowDirY = new Float32Array(N)
        flowDirZ = new Float32Array(N)

        for (let c = 0; c < N; c++) {
          mfdNeighbors[c] = []
          if (isOcean[c]) continue

          const { face: fc, x: cx, y: cy } = cellToFaceXY(c, res)
          texelToDir(fc, cx, cy, res, _tmp1)

          const lowers: Array<{ n: number; rawW: number }> = []
          let wSum = 0
          for (const n of neigh[c]) {
            if (Hf[n] < Hf[c] || (Hf[n] === Hf[c] && n < c)) {
              const slope = Hf[c] - Hf[n]
              const rawW = Math.pow(slope, MFD_P)
              lowers.push({ n, rawW })
              wSum += rawW
            }
          }

          if (lowers.length === 0 || wSum < 1e-30) continue

          let fdx = 0, fdy = 0, fdz = 0
          const cellMfd: Array<{ n: number; w: number }> = []

          for (const { n, rawW } of lowers) {
            const w = rawW / wSum
            cellMfd.push({ n, w })

            const { face: nf, x: nx, y: ny } = cellToFaceXY(n, res)
            texelToDir(nf, nx, ny, res, _tmp2)
            const diffX = _tmp2.x - _tmp1.x
            const diffY = _tmp2.y - _tmp1.y
            const diffZ = _tmp2.z - _tmp1.z
            const dot = diffX * _tmp1.x + diffY * _tmp1.y + diffZ * _tmp1.z
            const tX = diffX - dot * _tmp1.x
            const tY = diffY - dot * _tmp1.y
            const tZ = diffZ - dot * _tmp1.z
            const tLen = Math.sqrt(tX * tX + tY * tY + tZ * tZ)
            if (tLen > 1e-8) {
              fdx += w * (tX / tLen)
              fdy += w * (tY / tLen)
              fdz += w * (tZ / tLen)
            }
          }

          mfdNeighbors[c] = cellMfd
          const fdLen = Math.sqrt(fdx * fdx + fdy * fdy + fdz * fdz)
          if (fdLen > 1e-8) {
            flowDirX[c] = fdx / fdLen
            flowDirY[c] = fdy / fdLen
            flowDirZ[c] = fdz / fdLen
          }
        }

        // -----------------------------------------------------------------
        // f. Accumulate discharge from rainfall
        // -----------------------------------------------------------------
        Q = new Float64Array(N)
        for (let c = 0; c < N; c++) Q[c] = rain[c]

        for (const c of landCells) {
          const mfd = mfdNeighbors[c]
          for (const { n, w } of mfd) {
            if (!isOcean[n]) {
              Q[n] += Q[c] * w
            }
          }
        }

        // -----------------------------------------------------------------
        // g. One incision+deposition sweep (Davy–Lague in metre space)
        // -----------------------------------------------------------------
        const G_B = depositionG
        const dt_B = 0.005

        bufA.set(Hf)
        bufB.set(Hf)
        const Qs_B = new Float64Array(N)

        bufB.set(bufA)
        Qs_B.fill(0)

        for (const c of landCells) {
          if (isOcean[c]) continue
          const mfd = mfdNeighbors[c]

          let slope = 0
          for (const { n, w } of mfd) {
            slope += w * Math.max(0, bufA[c] - bufA[n])
          }

          // Stream-power incision scaled by per-cell erodibility (hardness-derived).
          // The 1e-8 floor is in normalized-height units; scale by B_HEIGHT_SCALE so the
          // physical floor is invariant under uniform scale (negligible at both scales).
          const eros  = dt_B * K0 * erodibilityB[c] * Math.pow(Q[c], mExp) * Math.pow(Math.max(slope, 1e-8 * B_HEIGHT_SCALE), nExp)
          const dep   = dt_B * G_B * Qs_B[c] / Math.max(Q[c], EPS_Q)

          let dh = dep - eros
          if (dh >  DH_CLAMP_M) dh =  DH_CLAMP_M
          if (dh < -DH_CLAMP_M) dh = -DH_CLAMP_M
          bufB[c] = bufA[c] + dh

          const QsOut = Qs_B[c] + (eros - dep)
          const QsOutC = QsOut > 0 ? QsOut : 0
          for (const { n, w } of mfd) {
            if (!isOcean[n]) Qs_B[n] += QsOutC * w
          }
        }

        workH.set(bufB)

        // -----------------------------------------------------------------
        // h. Thermal diffusion — 3 passes per step
        // -----------------------------------------------------------------
        bufA.set(workH)
        for (let tPass = 0; tPass < 3; tPass++) {
          bufB.set(bufA)
          for (let c = 0; c < N; c++) {
            // Per-cell talus angle in metre-space (hard rock → steeper, cliffier).
            const talusCM = talusScaleB[c]
            for (const n of neigh[c]) {
              const slopeCN = bufA[c] - bufA[n]
              if (slopeCN > talusCM) {
                const transfer = 0.0005 * (slopeCN - talusCM) * 0.5
                bufB[c] -= transfer
                bufB[n] += transfer
              }
            }
          }
          const tmp = bufA; bufA = bufB; bufB = tmp
        }
        workH.set(bufA)
      }

      // -----------------------------------------------------------------------
      // i. Lake flatten (after loop)
      // -----------------------------------------------------------------------
      for (let c = 0; c < N; c++) {
        if (lakeLevel[c] !== LAKE_SENTINEL) {
          workH[c] = lakeLevel[c]
        }
      }

      // -----------------------------------------------------------------------
      // j. Compose bDelta = (workH - H0) / B_HEIGHT_SCALE, clamped to ±B_DELTA_MAX
      // -----------------------------------------------------------------------
      let erosionDelta = new Float32Array(N)
      for (let c = 0; c < N; c++) {
        let d = (workH[c] - H0[c]) / B_HEIGHT_SCALE
        if (d < -B_DELTA_MAX) d = -B_DELTA_MAX
        if (d >  B_DELTA_MAX) d =  B_DELTA_MAX
        erosionDelta[c] = d
      }

      // -----------------------------------------------------------------------
      // k. Compute flowAccum (log-normalized Q from last routing step)
      // -----------------------------------------------------------------------
      const Qmax_B = Q.reduce((m, v) => v > m ? v : m, 0)
      const flowAccum = new Float32Array(N)
      const logQmaxB = Math.log(1 + Qmax_B)
      for (let c = 0; c < N; c++) {
        const v = Math.log(1 + Q[c]) / (logQmaxB || 1)
        flowAccum[c] = v < 0 ? 0 : v > 1 ? 1 : v
      }

      // -----------------------------------------------------------------------
      // l. Seam blur (same passes as existing path)
      // -----------------------------------------------------------------------
      erosionDelta = blurField(erosionDelta, 3)
      const flowAccumBlurredB = blurField(flowAccum, 4)
      for (let c = 0; c < N; c++) flowAccum[c] = flowAccumBlurredB[c]

      // Blur flowDir components to de-grid warp direction (Fix 3).
      // Do NOT re-normalize — un-normalized magnitude is the coherence self-fade
      // signal that prevents swirl/pinwheel artifacts at convergence singularities.
      const flowDirXBlurredB = blurField(flowDirX, 4)
      const flowDirYBlurredB = blurField(flowDirY, 4)
      const flowDirZBlurredB = blurField(flowDirZ, 4)
      for (let c = 0; c < N; c++) {
        flowDirX[c] = flowDirXBlurredB[c]
        flowDirY[c] = flowDirYBlurredB[c]
        flowDirZ[c] = flowDirZBlurredB[c]
      }

      // Store results
      this.erosionDelta = erosionDelta
      this.flowAccum    = flowAccum
      this.flowDirX     = flowDirX
      this.flowDirY     = flowDirY
      this.flowDirZ     = flowDirZ
      this.lakeLevel    = lakeLevel
      this.lakeMask     = lakeMask
      // B path has no Step-5b classifier; depositEnv stays all-zero (ENV_DEFAULT).
      // Color pass will treat every cell as default landform — acceptable for B path.
      this.depositEnv   = new Uint8Array(N)

    } else {
      // =====================================================================
      // EXISTING PATH — frozen-flow pipeline (unchanged)
      // =====================================================================

      // Slope thresholds: must be derived from the RUNTIME res (opts.res ?? EROSION_RES),
      // not the module-level EROSION_RES constant. At RES_REF=256, RADIUS_REF=50 000,
      // HEIGHT_SCALE_REF=1200 the base values 0.002/0.008 represent ~0.13°/~0.51°
      // physical slopes. deriveSlopeThresh rescales them to whatever (RADIUS, HEIGHT_SCALE,
      // res) the current bake is using so the physical slope angle is invariant across
      // slider changes and uniform planet-scale changes.
      const depSlopeLow  = deriveSlopeThresh(DEP_SLOPE_LOW_THRESH_BASE,  RADIUS, HEIGHT_SCALE, res)
      const depSlopeHigh = deriveSlopeThresh(DEP_SLOPE_HIGH_THRESH_BASE, RADIUS, HEIGHT_SCALE, res)

      // -----------------------------------------------------------------------
      // Step 1 — Sample H0 grid + rainfall
      // -----------------------------------------------------------------------

      const H0       = new Float32Array(N)
      const rain     = new Float32Array(N)
      // Per-cell base temperature (°C, deterministic — no sun term) used for
      // temperature-gated process modulation (Part B).  Captured here alongside
      // moisture so we only call climate.sample() once per cell.
      const tempGrid = new Float32Array(N)
      const climateSample: ClimateSample = { temperature: 0, moisture: 0 }

      for (let face = 0; face < 6; face++) {
        for (let y = 0; y < res; y++) {
          for (let x = 0; x < res; x++) {
            const c = texelIndex(face, x, y, res)
            texelToDir(face, x, y, res, _dir)
            H0[c] = opts.heightFn(_dir, EROSION_BAKE_LEVEL)
            opts.climate.sample(_dir, H0[c], climateSample)
            rain[c]     = Math.max(kwMin, climateSample.moisture)
            tempGrid[c] = climateSample.temperature
          }
        }
      }

      // Rock hardness per cell [0,1]: sampled from baked tectonics grid (C1, domain-warped).
      // erodibility[c] = 1 + HARD_ERODIBILITY_STR * (0.5 - hardness)
      //   hard rock (1) → erodibility < 1 → incises slower → ridges/escarpments persist.
      //   soft rock (0) → erodibility > 1 → incises faster → smooth basins/badlands.
      // When opts.tectonics is absent all erodibilities stay 1.0 (pre-T2 behaviour).
      const erodibility = new Float32Array(N).fill(1.0)
      // talusScale[c] = talus * (1 + HARD_TALUS_STR * (hardness - 0.5))
      //   hard rock holds steeper talus angles (cliffier); soft rock collapses sooner.
      const talusScale  = new Float32Array(N).fill(talus)
      if (opts.tectonics) {
        for (let face = 0; face < 6; face++) {
          for (let y = 0; y < res; y++) {
            for (let x = 0; x < res; x++) {
              const c = texelIndex(face, x, y, res)
              texelToDir(face, x, y, res, _dir)
              const h = opts.tectonics.hardnessAt(_dir)  // [0,1] C1 baked-grid lookup
              erodibility[c] = 1.0 + HARD_ERODIBILITY_STR * (0.5 - h)
              talusScale[c]  = talus * (1.0 + HARD_TALUS_STR * (h - 0.5))
            }
          }
        }
      }

      // -----------------------------------------------------------------------
      // Temperature-gated process modulation (Part B)
      //
      // Two independent continuous adjustments derived from the per-cell base
      // temperature (°C, deterministic — sampled above with no sun term):
      //
      // 1. Talus-angle bias: cold cells (freeze-thaw / frost shattering) lower the
      //    talus angle so debris accumulates at shallower angles; warm cells add a
      //    small cement-hardening lift.  Applied multiplicatively on top of the
      //    hardness-derived talusScale so the two effects compose cleanly.
      //
      // 2. Chemical-weathering factor on erodibility: warm + wet conditions dissolve
      //    rock faster (gentle rounded valleys); cold conditions suppress chemical
      //    action (mechanical-dominated, leaving sharper forms).
      //
      // Both factors are CONTINUOUS smoothstep blends between TEMP_COLD_THRESH and
      // TEMP_WARM_THRESH — no hard thresholds so adjacent chunks on either side of
      // any isothermal contour remain byte-identical (no seam risk).
      //
      // Only the bake is affected; neither factor ever enters heightFn geometry.
      // -----------------------------------------------------------------------
      {
        const tempRange = TEMP_WARM_THRESH - TEMP_COLD_THRESH   // degrees spanning the transition
        for (let c = 0; c < N; c++) {
          const T = tempGrid[c]

          // t: continuous 0 (cold) → 1 (warm) with smoothstep easing
          const tLinear = (T - TEMP_COLD_THRESH) / tempRange
          const tClamp  = tLinear < 0 ? 0 : tLinear > 1 ? 1 : tLinear
          const tSmooth = tClamp * tClamp * (3 - 2 * tClamp)  // smoothstep(0,1,t)

          // 1. Talus bias: lerp between cold and warm multipliers
          const talusTempMul = TEMP_TALUS_COLD_MUL + (TEMP_TALUS_WARM_MUL - TEMP_TALUS_COLD_MUL) * tSmooth
          talusScale[c] *= talusTempMul

          // 2. Chemical weathering: cold suppresses incision; warm+wet boosts it.
          //    Warm bonus scales linearly with moisture (only warm AND wet dissolves rock).
          //    coldFactor: lerp from TEMP_CHEM_COLD_FACTOR (fully cold) → 1.0 (fully warm).
          //    warmBonus: tSmooth * rain[c] * TEMP_CHEM_WARM_EXTRA (additive over baseline 1.0).
          const coldFactor  = TEMP_CHEM_COLD_FACTOR + (1.0 - TEMP_CHEM_COLD_FACTOR) * tSmooth
          const warmBonus   = tSmooth * rain[c] * TEMP_CHEM_WARM_EXTRA
          erodibility[c] *= coldFactor * (1.0 + warmBonus)
        }
      }

      // Provisional sea level — sort a copy and pick the ocean-coverage percentile
      const sorted = H0.slice().sort()
      const provSeaLevel = sorted[Math.min(N - 1, Math.floor(opts.oceanCoverage * N))]

      const isOcean = new Uint8Array(N)
      for (let c = 0; c < N; c++) {
        if (H0[c] < provSeaLevel) isOcean[c] = 1
      }

      // -----------------------------------------------------------------------
      // Step 2 — Neighbor table
      // -----------------------------------------------------------------------

      const neigh: number[][] = new Array(N)
      for (let c = 0; c < N; c++) {
        const { face, x, y } = cellToFaceXY(c, res)
        const ns: number[] = []
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue
            neighborTexel(face, x, y, dx as -1|0|1, dy as -1|0|1, res, nb)
            const nc = texelIndex(nb.face, nb.x, nb.y, res)
            if (nc !== c) ns.push(nc)
          }
        }
        neigh[c] = ns
      }

      // -----------------------------------------------------------------------
      // Step 3 — Priority-flood (depression fill + lakes)
      // -----------------------------------------------------------------------

      const Hf        = H0.slice()
      const lakeLevel = new Float32Array(N).fill(LAKE_SENTINEL)
      const inQueue   = new Uint8Array(N)
      const processed = new Uint8Array(N)

      const heap = new MinHeap()

      // Seed with ocean cells
      for (let c = 0; c < N; c++) {
        if (isOcean[c]) {
          heap.push({ elev: H0[c], idx: c })
          inQueue[c] = 1
        }
      }

      // Monotonic flood-order counter for Barnes ε-fill.
      // Increments on each heap POP so the ε-gradient is determined by drainage order,
      // NOT by spatial index (which spanned ±N and corrupted heights).
      let floodCounter = 0

      while (heap.size > 0) {
        const item = heap.pop()!
        const { elev, idx } = item
        if (processed[idx]) continue
        processed[idx] = 1
        ++floodCounter

        for (const n of neigh[idx]) {
          if (processed[n]) continue
          // Standard Barnes ε-fill: flat-filled areas get a tiny monotonic gradient
          // in flood order so every cell has a valid drainage direction.
          const newElev = Math.max(Hf[n], elev + FILL_EPS * floodCounter)
          if (newElev > Hf[n]) {
            // Only record as a real lake if the cell was raised MEANINGFULLY above
            // its original terrain height — ε-bumped cells are not true lake beds.
            if (newElev - H0[n] > LAKE_MIN_DEPTH) {
              lakeLevel[n] = newElev
            }
            Hf[n] = newElev
          }
          if (!inQueue[n]) {
            heap.push({ elev: Hf[n], idx: n })
            inQueue[n] = 1
          }
        }
      }

      // Build the lake mask from the now-final lakeLevel array.
      // 1 for cells that are real lake basins (lakeLevel != LAKE_SENTINEL), 0 elsewhere.
      // This is a clean 0/1 field — safe to bilinear-sample without sentinel bleed.
      const lakeMask = new Float32Array(N)
      for (let c = 0; c < N; c++) {
        lakeMask[c] = lakeLevel[c] !== LAKE_SENTINEL ? 1 : 0
      }

      // -----------------------------------------------------------------------
      // Step 4 — MFD flow routing (Multiple-Flow-Direction, all strictly-lower neighbors)
      //
      // ACYCLICITY GUARANTEE: weights are assigned only to neighbors that are
      // strictly lower in (Hf, index) order. landCells is sorted Hf descending
      // (ties by index descending). So when we process cell c, every neighbor n
      // that received weight from c has (Hf[n], n) < (Hf[c], c) in lex order —
      // meaning n sorts AFTER c — and has not yet been processed. No cycles.
      // -----------------------------------------------------------------------

      // Per-cell MFD weights: mfdNeighbors[c] = array of {n, w} for all strictly-lower neighbors
      // (w already normalized to sum to 1 across that cell).
      const mfdNeighbors: Array<Array<{ n: number; w: number }>> = new Array(N)
      const flowDirX = new Float32Array(N)
      const flowDirY = new Float32Array(N)
      const flowDirZ = new Float32Array(N)

      for (let c = 0; c < N; c++) {
        mfdNeighbors[c] = []
        if (isOcean[c]) continue

        const { face: fc, x: cx, y: cy } = cellToFaceXY(c, res)
        texelToDir(fc, cx, cy, res, _tmp1)  // _tmp1 = dirC (unit vector at c)

        // Collect all strictly-lower neighbors and their raw weights (slope^MFD_P)
        const lowers: Array<{ n: number; rawW: number }> = []
        let wSum = 0
        for (const n of neigh[c]) {
          // Strictly lower in (Hf, index) lex order
          if (Hf[n] < Hf[c] || (Hf[n] === Hf[c] && n < c)) {
            const slope = Hf[c] - Hf[n]  // always positive
            const rawW = Math.pow(slope, MFD_P)
            lowers.push({ n, rawW })
            wSum += rawW
          }
        }

        if (lowers.length === 0 || wSum < 1e-30) continue

        // Normalize weights and store; simultaneously accumulate weighted flowDir
        let fdx = 0, fdy = 0, fdz = 0
        const cellMfd: Array<{ n: number; w: number }> = []

        for (const { n, rawW } of lowers) {
          const w = rawW / wSum
          cellMfd.push({ n, w })

          // Tangent direction from c to neighbor n: (dirN - dirC) projected perp to dirC, normalized
          const { face: nf, x: nx, y: ny } = cellToFaceXY(n, res)
          texelToDir(nf, nx, ny, res, _tmp2)  // _tmp2 = dirN
          // diff = dirN - dirC
          const diffX = _tmp2.x - _tmp1.x
          const diffY = _tmp2.y - _tmp1.y
          const diffZ = _tmp2.z - _tmp1.z
          // project perpendicular to dirC
          const dot = diffX * _tmp1.x + diffY * _tmp1.y + diffZ * _tmp1.z
          const tX = diffX - dot * _tmp1.x
          const tY = diffY - dot * _tmp1.y
          const tZ = diffZ - dot * _tmp1.z
          const tLen = Math.sqrt(tX * tX + tY * tY + tZ * tZ)
          if (tLen > 1e-8) {
            fdx += w * (tX / tLen)
            fdy += w * (tY / tLen)
            fdz += w * (tZ / tLen)
          }
        }

        mfdNeighbors[c] = cellMfd

        // Normalize the weighted-sum flow direction
        const fdLen = Math.sqrt(fdx * fdx + fdy * fdy + fdz * fdz)
        if (fdLen > 1e-8) {
          flowDirX[c] = fdx / fdLen
          flowDirY[c] = fdy / fdLen
          flowDirZ[c] = fdz / fdLen
        }
      }

      // -----------------------------------------------------------------------
      // Step 5 — MFD discharge accumulation
      // -----------------------------------------------------------------------

      const Q = new Float64Array(N)
      for (let c = 0; c < N; c++) Q[c] = rain[c]

      // Collect land cells and sort by Hf DESCENDING (process upstream first).
      // Ties broken by index DESCENDING (b - a) so that for equal Hf, higher
      // index cells are processed first. This matches the acyclicity guarantee:
      // a cell n receiving weight from c has n < c when Hf[n] === Hf[c], so n
      // sorts after c — n has not yet been processed when c distributes its Q.
      const landCells: number[] = []
      for (let c = 0; c < N; c++) {
        if (!isOcean[c]) landCells.push(c)
      }
      landCells.sort((a, b) => {
        const dh = Hf[b] - Hf[a]
        if (dh !== 0) return dh > 0 ? 1 : -1
        return b - a
      })

      for (const c of landCells) {
        const mfd = mfdNeighbors[c]
        for (const { n, w } of mfd) {
          if (!isOcean[n]) {
            Q[n] += Q[c] * w
          }
        }
      }

      const Qmax = Q.reduce((m, v) => v > m ? v : m, 0)
      const flowAccum = new Float32Array(N)
      const logQmax = Math.log(1 + Qmax)
      for (let c = 0; c < N; c++) {
        const v = Math.log(1 + Q[c]) / (logQmax || 1)
        flowAccum[c] = v < 0 ? 0 : v > 1 ? 1 : v
      }

      // -----------------------------------------------------------------------
      // Step 5b — Classify depositional environments
      //
      // Each land cell is assigned one of four environment codes (ENV_*) from
      // data already in scope: local slope, upstream slope, discharge Q,
      // and proximity to ocean/lake shore.  Classification is purely read-only —
      // no heights are modified here — and deterministic (fixed iteration order,
      // no randomness).
      //
      // Inputs (all computed above):
      //   Hf          – flood-filled heights
      //   Q           – discharge per cell
      //   isOcean     – ocean mask
      //   lakeLevel   – LAKE_SENTINEL or spill elevation
      //   mfdNeighbors – per-cell lower-neighbor MFD weights
      //   neigh        – full 8-connectivity adjacency table
      //
      // Output: depositEnv[c] ∈ {ENV_DEFAULT, ENV_FLOODPLAIN, ENV_FAN, ENV_DELTA}
      // -----------------------------------------------------------------------

      // --- Normalise Q to [0,1] for threshold comparisons ---
      const Qnorm = new Float32Array(N)
      const QlogMax = Math.log(1 + Qmax)
      for (let c = 0; c < N; c++) {
        const v = Math.log(1 + Q[c]) / (QlogMax || 1)
        Qnorm[c] = v < 0 ? 0 : v > 1 ? 1 : v
      }

      // --- Local downslope gradient at each cell (weighted-average over MFD) ---
      const localSlope = new Float32Array(N)
      for (let c = 0; c < N; c++) {
        if (isOcean[c]) continue
        let s = 0
        for (const { n, w } of mfdNeighbors[c]) {
          s += w * Math.max(0, Hf[c] - Hf[n])
        }
        localSlope[c] = s
      }

      // --- Max upstream slope arriving at each cell (steepest contributing channel) ---
      // For alluvial fan detection: the upstream slope was steep but c is gentle.
      const upstreamSlope = new Float32Array(N)
      // A land cell c receives an upstream-slope contribution from any neighbor n
      // where Hf[n] > Hf[c] (n drains into c in MFD sense).
      // We walk neigh[c] and check if c appears as a lower neighbor of n.
      // Simple proxy: max(Hf[n] - Hf[c]) over all higher neighbors n.
      for (let c = 0; c < N; c++) {
        if (isOcean[c]) continue
        let maxUpSlope = 0
        for (const n of neigh[c]) {
          if (Hf[n] > Hf[c]) {
            const s = Hf[n] - Hf[c]
            if (s > maxUpSlope) maxUpSlope = s
          }
        }
        upstreamSlope[c] = maxUpSlope
      }

      // --- Shoreline band: cells within DEP_SHORE_HOPS BFS hops of ocean/lake ---
      // Uses a simple BFS (deterministic: processes cells in index order each wave).
      const isShore = new Uint8Array(N)
      {
        let frontier: number[] = []
        for (let c = 0; c < N; c++) {
          if (isOcean[c] || lakeLevel[c] !== LAKE_SENTINEL) {
            isShore[c] = 1
            frontier.push(c)
          }
        }
        for (let hop = 0; hop < DEP_SHORE_HOPS; hop++) {
          const next: number[] = []
          for (const c of frontier) {
            for (const n of neigh[c]) {
              if (!isShore[n]) {
                isShore[n] = 1
                next.push(n)
              }
            }
          }
          frontier = next
        }
        // Clear ocean/lake cells from shore mask — only the LAND fringe is shore
        for (let c = 0; c < N; c++) {
          if (isOcean[c] || lakeLevel[c] !== LAKE_SENTINEL) isShore[c] = 0
        }
      }

      // --- Classify ---
      const depositEnv = new Uint8Array(N)  // ENV_DEFAULT everywhere to start
      for (let c = 0; c < N; c++) {
        if (isOcean[c]) continue

        const sl  = localSlope[c]
        const usl = upstreamSlope[c]
        const qn  = Qnorm[c]

        // DELTA: near shore with meaningful discharge
        if (isShore[c] && qn >= DEP_Q_DELTA_MIN) {
          depositEnv[c] = ENV_DELTA
          continue
        }

        // ALLUVIAL FAN: steep-to-flat transition — upstream was steep, here is gentle
        if (sl < depSlopeLow && usl > depSlopeHigh) {
          depositEnv[c] = ENV_FAN
          continue
        }

        // FLOODPLAIN: low gradient, moderate discharge, not near shore
        if (sl < depSlopeLow && qn >= DEP_Q_FLOODPLAIN_MIN) {
          depositEnv[c] = ENV_FLOODPLAIN
          continue
        }

        // DEFAULT: everything else (high-gradient incision, low Q flats, lake floor)
        depositEnv[c] = ENV_DEFAULT
      }

      // -----------------------------------------------------------------------
      // Step 6 — Erosion-deposition (Davy–Lague / Yuan stream-power extension)
      //
      // Per cell, per iteration (topological sweep high→low via landCells order):
      //   erosion    = dt · K0 · Q^mExp · S^nExp            (detachment — lowers bed)
      //   deposition = dt · G_env · QsIn / max(Q, eps)      (raises bed from incoming flux)
      //   dh         = deposition − erosion
      //   QsOut      = QsIn + erosion − deposition           (mass conservation)
      //
      // G_env and SEDIMENT_MAX_env vary per depositional environment (ENV_*) so
      // alluvial fans, floodplains and deltas build recognisable landforms while
      // high-gradient reaches retain the original behaviour.
      //
      // Alluvial fans additionally spread a fraction (FAN_LATERAL_SPREAD) of their
      // outgoing Qs sideways to non-downslope 8-connectivity neighbors at similar
      // elevation — building the characteristic perpendicular-to-flow cone shape.
      //
      // QsOut is routed to lower neighbors using the same MFD weights computed in
      // Step 4. Qs resets to zero at the start of each iteration so the flux is
      // recomputed from current topography each pass.
      //
      // Determinism: fixed landCells order + MFD weights + no randomness.
      // Stability: per-cell dh clamped to ±DH_CLAMP to prevent oscillation.
      // -----------------------------------------------------------------------

      const DH_CLAMP = 0.02   // per-iteration per-cell height-change cap (raised from 0.01 for deposition headroom)
      const dt = 0.005        // same as the old incision timestep

      // Lookup tables indexed by ENV_* for zero-branch inner loop.
      // ENV_DEFAULT respects opts.depositionG override so external callers that bump
      // the base G still affect uncategorised cells while landform classes keep their
      // own coefficients.
      const ENV_G: readonly number[] = [
        depositionG,       // 0 = ENV_DEFAULT (respects opts.depositionG override)
        DEP_G_FLOODPLAIN,  // 1 = ENV_FLOODPLAIN
        DEP_G_FAN,         // 2 = ENV_FAN
        DEP_G_DELTA,       // 3 = ENV_DELTA
      ]
      const ENV_SMAX: readonly number[] = [
        SEDIMENT_MAX_DEFAULT,     // 0 = ENV_DEFAULT
        SEDIMENT_MAX_FLOODPLAIN,  // 1 = ENV_FLOODPLAIN
        SEDIMENT_MAX_FAN,         // 2 = ENV_FAN
        SEDIMENT_MAX_DELTA,       // 3 = ENV_DELTA
      ]

      // Pre-compute lateral spread neighbor lists for fan cells — done once so the
      // inner loop allocates nothing.  Uses Hf (the pre-erosion flood-filled surface)
      // as the reference elevation; gentle deviations during iteration don't matter for
      // this structural classification.
      //
      // A lateral candidate is a neighbor of a fan cell that:
      //   • is not ocean
      //   • is not a downslope MFD target (which gets the primary flux already)
      //   • is at a similar elevation: within [−localSlope, 2*localSlope + depSlopeLow]
      //     of c, i.e. roughly iso-elevation or very gently lower — the lateral flank.
      const fanLateralNeighbors: number[][] = new Array(N).fill(null).map(() => [])
      for (let c = 0; c < N; c++) {
        if (depositEnv[c] !== ENV_FAN) continue
        const mfd = mfdNeighbors[c]
        const sl  = localSlope[c]
        // MFD target set for fast exclusion (small per cell, array scan acceptable)
        const mfdTargets: number[] = mfd.map(({ n }) => n)
        for (const n of neigh[c]) {
          if (isOcean[n]) continue
          if (mfdTargets.includes(n)) continue
          const dh_cn = Hf[c] - Hf[n]
          if (dh_cn >= -sl && dh_cn <= 2 * sl + depSlopeLow) {
            fanLateralNeighbors[c].push(n)
          }
        }
      }

      let bufA = Hf.slice()
      let bufB = new Float32Array(N)
      const Qs  = new Float64Array(N)  // sediment flux (recomputed each iteration)

      // Per-cell accumulated deposition (tracks against per-class SEDIMENT_MAX cap)
      const totalDep = new Float32Array(N)

      for (let iter = 0; iter < DEFAULT_ITERS_INCISION; iter++) {
        bufB.set(bufA)
        Qs.fill(0)

        // Single topological sweep in landCells order (high→low):
        // when cell c is visited, Qs[c] already holds all incoming flux from
        // higher upstream cells processed earlier this iteration.
        for (const c of landCells) {
          if (isOcean[c]) continue
          const mfd = mfdNeighbors[c]

          // Weighted-average slope over all lower MFD neighbors (using bufA heights)
          let slope = 0
          for (const { n, w } of mfd) {
            slope += w * Math.max(0, bufA[c] - bufA[n])
          }

          const env   = depositEnv[c]
          const G_env = ENV_G[env]
          const smax  = ENV_SMAX[env]

          // Stream-power incision: K0 scaled by per-cell erodibility (hardness-derived).
          // Hard rock (erodibility < 1) incises slower → ridges and escarpments persist.
          // Soft rock (erodibility > 1) incises faster → smooth basins and badlands form.
          const erosion    = dt * K0 * erodibility[c] * Math.pow(Q[c], mExp) * Math.pow(Math.max(slope, 1e-8), nExp)
          // Per-class deposition: soft-cap by remaining headroom so fast depositors
          // don't stack beyond their environment's geomorphic ceiling.
          const headroom   = Math.max(0, smax - totalDep[c])
          const rawDep     = dt * G_env * Qs[c] / Math.max(Q[c], EPS_Q)
          const deposition = rawDep < headroom ? rawDep : headroom

          // Height change: cap per-iteration delta to prevent runaway
          let dh = deposition - erosion
          if (dh >  DH_CLAMP) dh =  DH_CLAMP
          if (dh < -DH_CLAMP) dh = -DH_CLAMP
          bufB[c] = bufA[c] + dh
          if (dh > 0) totalDep[c] += dh

          // Route outgoing sediment flux to lower (downslope) neighbors.
          // NOTE: max(0, QsIn + erosion − deposition) is NOT strict mass conservation —
          // the floor discards flux that drains to the ocean or sinks (no downstream cell
          // to receive it). Negative flux would be unphysical, so it is clamped to zero.
          const QsOut = Qs[c] + (erosion - deposition)
          const QsOutClamped = QsOut > 0 ? QsOut : 0  // floor: sediment leaving to ocean/sink is discarded

          // Alluvial fan lateral spread: MOVE a fraction of outgoing Qs sideways to
          // pre-computed lateral neighbors (perpendicular-to-flow cone shape).
          // The lateral list is empty for non-fan cells so this is a no-op in those cases.
          //
          // Normal case (fan cell with downslope MFD targets):
          //   downslope gets (1 − FAN_LATERAL_SPREAD) of flux;
          //   lateral gets FAN_LATERAL_SPREAD → total = 1.0, mass conserved.
          //
          // Pit/sink case (fan cell with no MFD lower neighbors, mfd.length === 0):
          //   Routing all flux only to downslope (= nothing) would silently discard it.
          //   Instead, route ALL remaining flux to lateral neighbors — the cone still
          //   spreads, nothing is lost.  If there are no lateral neighbors either, the
          //   sink discards as normal (consistent with the ocean/lake-floor sink behaviour).
          let downSlopeFrac = 1.0
          if (env === ENV_FAN && QsOutClamped > 0) {
            const lateralNeighbors = fanLateralNeighbors[c]
            const nLateral = lateralNeighbors.length
            if (nLateral > 0) {
              // If no MFD downslope targets exist, the full flux goes laterally.
              const lateralFrac = mfd.length === 0 ? 1.0 : FAN_LATERAL_SPREAD
              downSlopeFrac = 1.0 - lateralFrac
              const share = (QsOutClamped * lateralFrac) / nLateral
              for (let li = 0; li < nLateral; li++) {
                Qs[lateralNeighbors[li]] += share
              }
            }
          }

          for (const { n, w } of mfd) {
            if (!isOcean[n]) {
              Qs[n] += QsOutClamped * downSlopeFrac * w
            }
          }
        }

        const tmp = bufA; bufA = bufB; bufB = tmp
      }

      // -----------------------------------------------------------------------
      // Step 7 — Thermal diffusion (~80 iterations, double-buffered)
      // -----------------------------------------------------------------------

      for (let iter = 0; iter < DEFAULT_ITERS_THERMAL; iter++) {
        bufB.set(bufA)
        for (let c = 0; c < N; c++) {
          // Per-cell talus angle: harder rock holds steeper slopes (cliffier escarpments).
          const talusC = talusScale[c]
          for (const n of neigh[c]) {
            const slopeCN = bufA[c] - bufA[n]
            if (slopeCN > talusC) {
              const transfer = 0.0005 * (slopeCN - talusC) * 0.5
              bufB[c] -= transfer
              bufB[n] += transfer
            }
          }
        }
        const tmp = bufA; bufA = bufB; bufB = tmp
      }

      const workH = bufA

      // -----------------------------------------------------------------------
      // Step 8 — Lake flatten
      // -----------------------------------------------------------------------

      for (let c = 0; c < N; c++) {
        if (lakeLevel[c] !== LAKE_SENTINEL) {
          workH[c] = lakeLevel[c]
        }
      }

      // -----------------------------------------------------------------------
      // Step 9 — Compose erosionDelta
      // -----------------------------------------------------------------------

      let erosionDelta = new Float32Array(N)
      for (let c = 0; c < N; c++) {
        const d = workH[c] - H0[c]
        // Upper clamp is per-environment so fans/floodplains/deltas keep their
        // extra relief while the erosion floor stays at the global EROSION_MAX.
        const smax = ENV_SMAX[depositEnv[c]]
        erosionDelta[c] = d < -EROSION_MAX ? -EROSION_MAX : d > smax ? smax : d
      }

      // -----------------------------------------------------------------------
      // Step 10 — Seam blur
      //   erosionDelta: 3 passes of 3×3 cross-face box blur (steep valley-edge
      //     ramps from a single pass bilinear-sample into terraces).
      //   flowAccum: 4 passes (raised from 2) — broadens the discharge gradient
      //     from ~600 m to ~1200 m radius so detailAmp varies smoothly and the
      //     256² grid iso-contour banding (Mechanism A) is suppressed.
      //   flowDirX/Y/Z: 4 passes to de-grid warp direction (Fix 3 / Mechanism B).
      //     Do NOT re-normalize — un-normalized magnitude is the coherence self-fade
      //     signal that prevents swirl/pinwheel artifacts at convergence singularities.
      //   Bake-time cost only — paid once.
      // -----------------------------------------------------------------------

      erosionDelta = blurField(erosionDelta, 3)
      const flowAccumBlurred = blurField(flowAccum, 4)
      for (let c = 0; c < N; c++) flowAccum[c] = flowAccumBlurred[c]

      const flowDirXBlurred = blurField(flowDirX, 4)
      const flowDirYBlurred = blurField(flowDirY, 4)
      const flowDirZBlurred = blurField(flowDirZ, 4)
      for (let c = 0; c < N; c++) {
        flowDirX[c] = flowDirXBlurred[c]
        flowDirY[c] = flowDirYBlurred[c]
        flowDirZ[c] = flowDirZBlurred[c]
      }

      // -----------------------------------------------------------------------
      // Store results
      // -----------------------------------------------------------------------

      this.erosionDelta = erosionDelta
      this.flowAccum    = flowAccum
      this.flowDirX     = flowDirX
      this.flowDirY     = flowDirY
      this.flowDirZ     = flowDirZ
      this.lakeLevel    = lakeLevel
      this.lakeMask     = lakeMask
      this.depositEnv   = depositEnv
    }
  }

  // -------------------------------------------------------------------------
  // Static constructors
  // -------------------------------------------------------------------------

  /**
   * Reconstruct a sampler-only Erosion from a baked snapshot.
   * Does NOT re-run the simulation — arrays from b are used directly.
   * Mirrors climate.ts fromBaked() exactly.
   */
  static fromBaked(b: ErosionBaked): Erosion {
    const e = Object.create(Erosion.prototype) as Erosion
    const R = e as unknown as Record<string, unknown>
    R['res']          = b.res
    R['erosionDelta'] = b.erosionDelta
    R['flowAccum']    = b.flowAccum
    R['flowDirX']     = b.flowDirX
    R['flowDirY']     = b.flowDirY
    R['flowDirZ']     = b.flowDirZ
    R['lakeLevel']    = b.lakeLevel
    R['lakeMask']     = b.lakeMask
    // depositEnv: MUST be assigned explicitly — Object.create does not run field
    // initializers, so workers would read undefined and deposit-env queries would crash.
    R['depositEnv']   = b.depositEnv
    R['K0']    = b.K0
    R['mExp']  = b.mExp
    R['nExp']  = b.nExp
    R['talus'] = b.talus
    R['kwMin'] = b.kwMin
    // Field initializers don't run under Object.create — must init manually.
    R['_scratch']  = new Vector3()
    R['_scratch2'] = new Vector3()
    // Rebuild domain-warp noise from the stored seed (byte-identical to constructor path).
    const warpSeed = b.warpSeed
    R['_warpSeed']  = warpSeed
    R['_warpNoise'] = createNoise3D(warpSeed)
    return e
  }

  /**
   * Return an Erosion with all-zero erosionDelta/flowAccum/flowDir and
   * lakeLevel filled with LAKE_SENTINEL. Bypasses the constructor.
   */
  static identity(res: number): Erosion {
    const N  = 6 * res * res
    const e  = Object.create(Erosion.prototype) as Erosion
    const R  = e as unknown as Record<string, unknown>
    R['res']          = res
    R['erosionDelta'] = new Float32Array(N)
    R['flowAccum']    = new Float32Array(N)
    R['flowDirX']     = new Float32Array(N)
    R['flowDirY']     = new Float32Array(N)
    R['flowDirZ']     = new Float32Array(N)
    const ll = new Float32Array(N); ll.fill(LAKE_SENTINEL); R['lakeLevel'] = ll
    R['lakeMask']     = new Float32Array(N)  // all zeros = no lakes
    R['depositEnv']   = new Uint8Array(N)    // all zeros = ENV_DEFAULT everywhere
    R['K0']    = DEFAULT_K0
    R['mExp']  = DEFAULT_M_EXP
    R['nExp']  = DEFAULT_N_EXP
    R['talus'] = DEFAULT_TALUS
    R['kwMin'] = DEFAULT_KW_MIN
    R['_scratch']  = new Vector3()
    R['_scratch2'] = new Vector3()
    // Domain-warp noise: identity has all-zero fields so warp is a no-op, but
    // the noise must still be initialized to a deterministic fixed seed so that
    // _warpNoise is never undefined (required for zero-alloc query methods).
    const identWarpSeed = deriveSeed(0, 10)  // fixed: masterSeed=0, stream=10
    R['_warpSeed']  = identWarpSeed
    R['_warpNoise'] = createNoise3D(identWarpSeed)
    return e
  }

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /**
   * Snapshot this Erosion's baked state so a Web Worker can reconstruct a
   * sampler-only instance without re-simulating.
   *
   * Arrays are returned BY REFERENCE; the worker pool structured-clones them
   * (the main thread keeps its copies for ongoing queries).
   * Do NOT transfer .buffer — that would detach the main-thread arrays.
   */
  toBaked(): ErosionBaked {
    return {
      res:          this.res,
      erosionDelta: this.erosionDelta,
      flowAccum:    this.flowAccum,
      flowDirX:     this.flowDirX,
      flowDirY:     this.flowDirY,
      flowDirZ:     this.flowDirZ,
      lakeLevel:    this.lakeLevel,
      lakeMask:     this.lakeMask,
      depositEnv:   this.depositEnv,
      K0:       this.K0,
      mExp:     this.mExp,
      nExp:     this.nExp,
      talus:    this.talus,
      kwMin:    this.kwMin,
      warpSeed: this._warpSeed,
    }
  }

  // -------------------------------------------------------------------------
  // Internal helper — zero-alloc domain warp
  // -------------------------------------------------------------------------

  /**
   * Compute a domain-warped direction from `dir`.
   * Samples a 2-octave fbm to produce a displacement tangent, then normalizes.
   * Result is written into `this._scratch2` and returned.
   *
   * The warp is deliberately low-frequency (whole features bend into organic
   * curves rather than adding high-frequency jitter) and small amplitude
   * (~2 erosion cells) — enough to break cubemap grid alignment without
   * smearing valley shapes.
   */
  private _warpedDir(dir: Vector3): Vector3 {
    const noise = this._warpNoise
    const fx = dir.x * WARP_FREQ
    const fy = dir.y * WARP_FREQ
    const fz = dir.z * WARP_FREQ

    // 2-octave fbm for each of two offset noise samples (x-offset pattern).
    // Using two spatially offset evaluations gives an independent warp per axis.
    let wx = 0, wy = 0, wz = 0
    let amp = 1, freq = 1
    for (let o = 0; o < WARP_OCT; o++) {
      // First sample: warp X and Y components
      wx += amp * noise(fx * freq,         fy * freq,         fz * freq)
      wy += amp * noise(fx * freq + 3.7,   fy * freq + 2.1,   fz * freq + 1.3)
      wz += amp * noise(fx * freq + 7.3,   fy * freq + 5.9,   fz * freq + 4.1)
      amp  *= 0.5
      freq *= 2.0
    }
    // Normalize fbm output (max amplitude = sum of gains = 1 + 0.5 = 1.5 for 2 oct)
    const maxAmp = 1.5
    wx = (wx / maxAmp) * WARP_AMP
    wy = (wy / maxAmp) * WARP_AMP
    wz = (wz / maxAmp) * WARP_AMP

    // Project warp vector perpendicular to dir (keep result on sphere surface)
    const dot = wx * dir.x + wy * dir.y + wz * dir.z
    const tx = wx - dot * dir.x
    const ty = wy - dot * dir.y
    const tz = wz - dot * dir.z

    // Warped direction: dir + tangent displacement, then normalize
    const rx = dir.x + tx
    const ry = dir.y + ty
    const rz = dir.z + tz
    const len = Math.sqrt(rx * rx + ry * ry + rz * rz)
    const inv = len > 1e-8 ? 1 / len : 1
    this._scratch2.set(rx * inv, ry * inv, rz * inv)
    return this._scratch2
  }

  // -------------------------------------------------------------------------
  // Internal helper — smoothstep bilinear sampler (C1 continuity)
  // -------------------------------------------------------------------------

  /**
   * Sample a per-texel Float32Array at direction `dir` with C1 continuity.
   *
   * Identical to sampleSmooth from cubemap.ts EXCEPT smoothstep is applied
   * to the fractional texel coordinates before the bilinear lerp:
   *   fx = fx*fx*(3 - 2*fx)
   *   fy = fy*fy*(3 - 2*fy)
   *
   * This eliminates the C0 kinks (visible terracing) that bilinear interpolation
   * produces at cell edges when the field has steep gradients.
   *
   * NOT safe for fields containing LAKE_SENTINEL (-999) sentinel values —
   * smoothstep would smear sentinel into adjacent real cells. Use sampleSmooth
   * (from cubemap.ts) for lakeLevel instead.
   *
   * Uses this._scratch for a cross-face anchor texel (same module-level scratch
   * vectors as sampleSmooth are NOT used — we write into local vars here to
   * avoid re-entrancy risk with the cubemap module-level state). The four
   * corner texel lookups go through neighborTexel (cross-face safe).
   *
   * Zero-alloc: all temporaries are local scalar or this._scratch (pre-allocated).
   * Pure & deterministic: same (data, dir, res) → same result.
   */
  private _sampleC1(data: Float32Array, dir: Vector3): number {
    const res = this.res
    const dx = dir.x, dy = dir.y, dz = dir.z
    const ax = Math.abs(dx)
    const ay = Math.abs(dy)
    const az = Math.abs(dz)

    let face: number
    let sc: number
    let tc: number
    let mc: number

    if (ax >= ay && ax >= az) {
      if (dx > 0) { face = 0; mc = ax; sc =  dz; tc = dy }
      else        { face = 1; mc = ax; sc = -dz; tc = dy }
    } else if (ay >= ax && ay >= az) {
      if (dy > 0) { face = 2; mc = ay; sc = dx; tc =  dz }
      else        { face = 3; mc = ay; sc = dx; tc = -dz }
    } else {
      if (dz > 0) { face = 4; mc = az; sc = -dx; tc = dy }
      else        { face = 5; mc = az; sc =  dx; tc = dy }
    }

    const half = res * 0.5
    const fx = (sc / mc + 1.0) * half - 0.5
    const fy = (tc / mc + 1.0) * half - 0.5

    const x0 = Math.floor(fx)
    const y0 = Math.floor(fy)
    // Apply smoothstep to fractional coords for C1 continuity
    let fu = fx - x0
    let fv = fy - y0
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)

    const x1 = x0 + 1
    const y1 = y0 + 1

    let t00: number, t10: number, t01: number, t11: number

    if (x0 >= 0 && y0 >= 0 && x1 < res && y1 < res) {
      // Interior fast path: all 4 texels on the same face
      const base = face * res * res
      const row0 = base + y0 * res
      const row1 = base + y1 * res
      t00 = data[row0 + x0]
      t10 = data[row0 + x1]
      t01 = data[row1 + x0]
      t11 = data[row1 + x1]
    } else {
      // Edge path: one or more neighbors cross a face boundary
      // Use a local object for the texel result (Vector3 scratch is occupied by caller context).
      const nb: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
      const axx = x0 < 0 ? 0 : (x0 >= res ? res - 1 : x0)
      const ayy = y0 < 0 ? 0 : (y0 >= res ? res - 1 : y0)
      const ox0 = (x0 - axx) as -1 | 0 | 1
      const ox1 = (x1 - axx) as -1 | 0 | 1
      const oy0 = (y0 - ayy) as -1 | 0 | 1
      const oy1 = (y1 - ayy) as -1 | 0 | 1

      neighborTexel(face, axx, ayy, ox0, oy0, res, nb)
      t00 = data[texelIndex(nb.face, nb.x, nb.y, res)]
      neighborTexel(face, axx, ayy, ox1, oy0, res, nb)
      t10 = data[texelIndex(nb.face, nb.x, nb.y, res)]
      neighborTexel(face, axx, ayy, ox0, oy1, res, nb)
      t01 = data[texelIndex(nb.face, nb.x, nb.y, res)]
      neighborTexel(face, axx, ayy, ox1, oy1, res, nb)
      t11 = data[texelIndex(nb.face, nb.x, nb.y, res)]
    }

    const a = t00 + (t10 - t00) * fu
    const c = t01 + (t11 - t01) * fu
    return a + (c - a) * fv
  }

  // -------------------------------------------------------------------------
  // Query methods
  // -------------------------------------------------------------------------

  /** Normalized height delta at dir; <0 = incision, >0 = sediment. */
  deltaAt(dir: Vector3): number {
    return this._sampleC1(this.erosionDelta, this._warpedDir(dir))
  }

  /** Normalized log discharge 0..1 at dir. */
  accAt(dir: Vector3): number {
    return this._sampleC1(this.flowAccum, this._warpedDir(dir))
  }

  /**
   * World-space downhill flow vector at dir, written into out.
   *
   * Returns the RAW bilinear-interpolated vector — NOT renormalized to unit length.
   * Magnitude carries directional coherence: ~1 in coherent channels, →0 at
   * convergence/pit singularities where per-cell unit flowDirs point different ways
   * and cancel. The caller (steered detail in terrainSampler) uses this magnitude
   * as an automatic fade so the warp collapses to ~0 at pinwheel singularities.
   *
   * Per-cell flowDir values stored in the baked arrays are still unit vectors
   * (computed in the constructor); it is only the query-time bilinear result that
   * is returned un-normalized.
   */
  flowAt(dir: Vector3, out: Vector3): void {
    const dw = this._warpedDir(dir)
    out.set(
      this._sampleC1(this.flowDirX, dw),
      this._sampleC1(this.flowDirY, dw),
      this._sampleC1(this.flowDirZ, dw),
    )
    // No renormalization: raw magnitude is the coherence signal.
    // Guard only against NaN (e.g. all-zero input dir — degenerate case).
    if (!isFinite(out.x) || !isFinite(out.y) || !isFinite(out.z)) out.set(0, 0, 0)
  }

  /**
   * Lake spill elevation at dir; returns LAKE_SENTINEL (-999) where not a lake.
   * Uses plain bilinear (sampleSmooth) — NOT _sampleC1 — because lakeLevel
   * contains LAKE_SENTINEL (-999) sentinel values that would corrupt smoothstep
   * interpolation if mixed with real elevation values near lake boundaries.
   */
  lakeAt(dir: Vector3): number {
    return sampleSmooth(this.lakeLevel, this._warpedDir(dir), this.res, this._scratch)
  }

  /**
   * Lake presence mask at dir; smoothstep-bilinear samples the 0/1 mask field.
   * Returns values in [0,1]; values > 0.5 indicate the 0.5-isocontour lake shoreline.
   * Safe to use _sampleC1 because sentinel values never enter this array.
   */
  lakeMaskAt(dir: Vector3): number {
    return this._sampleC1(this.lakeMask, this._warpedDir(dir))
  }

  /**
   * Depositional environment code at dir.
   * Returns one of ENV_DEFAULT (0), ENV_FLOODPLAIN (1), ENV_FAN (2), ENV_DELTA (3).
   *
   * Sampled NEAREST-NEIGHBOR — NOT C1 or bilinear — because depositEnv is a discrete
   * integer classification, not a continuous field.  Interpolating between environment
   * codes is meaningless and would produce fractional values without semantic content.
   *
   * The domain warp is still applied so query locations match the same distorted grid
   * used by all other Erosion samplers.
   *
   * IMPORTANT: this value must NEVER feed into heightFn geometry.  Color discontinuities
   * at environment boundaries are acceptable; geometry seams are not.
   *
   * Returns the environment code as a number (0–3); cast to number for use in lookups.
   */
  depositEnvAt(dir: Vector3): number {
    // Apply the same domain warp used by all other samplers so queries are consistent.
    const dw  = this._warpedDir(dir)
    const res = this.res

    // Nearest-neighbor cube-map lookup (no interpolation).
    const dx = dw.x, dy = dw.y, dz = dw.z
    const ax = Math.abs(dx)
    const ay = Math.abs(dy)
    const az = Math.abs(dz)

    let face: number, sc: number, tc: number, mc: number
    if (ax >= ay && ax >= az) {
      if (dx > 0) { face = 0; mc = ax; sc =  dz; tc = dy }
      else        { face = 1; mc = ax; sc = -dz; tc = dy }
    } else if (ay >= ax && ay >= az) {
      if (dy > 0) { face = 2; mc = ay; sc = dx; tc =  dz }
      else        { face = 3; mc = ay; sc = dx; tc = -dz }
    } else {
      if (dz > 0) { face = 4; mc = az; sc = -dx; tc = dy }
      else        { face = 5; mc = az; sc =  dx; tc = dy }
    }

    const half = res * 0.5
    const fx   = (sc / mc + 1.0) * half - 0.5
    const fy   = (tc / mc + 1.0) * half - 0.5
    // Nearest: round to nearest texel, clamped to [0, res-1]
    let x = Math.round(fx)
    let y = Math.round(fy)
    if (x < 0) x = 0; else if (x >= res) x = res - 1
    if (y < 0) y = 0; else if (y >= res) y = res - 1

    return this.depositEnv[face * res * res + y * res + x]
  }
}
