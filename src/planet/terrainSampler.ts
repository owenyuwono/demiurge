/**
 * Shared terrain-sampler factory.
 *
 * Contains the single source of truth for:
 *   - TECT_* tuning constants
 *   - The heightFn / plateColorFn / climateFn closure construction
 *
 * Callable from both the main thread (via Planet.buildHeightFn) and Web Workers
 * (which pass pre-built fromBaked() instances via opts.tectonics / opts.climate).
 * The output is byte-identical regardless of call site — that is the determinism
 * guarantee that makes worker meshing safe.
 */

import { Vector3 } from 'three'
import { createNoise3D, ridged, fbm } from './noise'
import { Tectonics, TectonicQuery, boundaryRelief } from './tectonics'
import { Climate, ClimateSample } from './climate'
import { Erosion } from './erosion'
import { Subsurface } from './subsurface'

// ---------------------------------------------------------------------------
// TECT_* tuning constants (moved here from Planet.ts — single source of truth)
// Note: boundary profile constants live in boundaryRelief() in tectonics.ts
// ---------------------------------------------------------------------------

// Crust-keyed base height constants
export const TECT_LAND_BASE_0         = 0.06   // coastline base
export const TECT_LAND_BASE_SS        = 0.17   // smoothstep gain inland (interiors climb to ~0.23+)
export const TECT_LAND_PLATE_MOD      = 0.15   // plate.baseElevation modifier on land
export const TECT_OCEAN_BASE_0        = -0.42
export const TECT_OCEAN_DEPTH_AMP     = 0.10   // abyssal deepening
export const TECT_OCEAN_DEPTH_SAT     = 0.45   // saturation distance (radians)
export const TECT_OCEAN_PLATE_MOD     = 0.10   // plate.baseElevation modifier on ocean
export const TECT_SHELF_W_PASSIVE     = 0.12   // wide passive shelf
export const TECT_SHELF_W_ACTIVE      = 0.045  // active margin shelf — narrower than passive (2.7× contrast)
export const TECT_SHELF_W_STAGNANT    = 0.045  // stagnant-lid shelf — narrow (no wide passive margin tectonics)
export const TECT_COAST_LERP_HI       = 0.018  // crust→ocean transition edge hi [was 0.012 — gentler coast ramp trims the 120m base step where a breaching cone coincides]
// Highlands (signed fbm) — KEPT as dead exports; no longer used in heightFn (replaced by broadUplift)
export const TECT_HIGHLANDS_AMP       = 0.12   // unused — see HIGHLAND_AMP
export const TECT_HIGHLANDS_FREQ      = 1.9    // unused
export const TECT_HIGHLANDS_OCT       = 5      // unused
export const TECT_HIGHLANDS_SS_LO     = 0.012  // unused
export const TECT_HIGHLANDS_SS_HI     = 0.10   // unused
// Paleo-orogens — KEPT as dead exports; role folded into RUGGED_PALEO* below
export const TECT_PALEO_AMP           = 0.16   // unused — see RUGGED_PALEO
export const TECT_PALEO_AMP_STAGNANT  = 1.10   // unused — see RUGGED_PALEO_STAGNANT
export const TECT_PALEO_SIGMA         = 0.10   // unused — see PALEO_W
export const TECT_PALEO_FREQ          = 2.6    // unused
export const TECT_PALEO_OCT           = 5      // unused
export const TECT_PALEO_SS_LO         = 0.012  // unused
export const TECT_PALEO_SS_HI         = 0.08   // unused
// Elevation-coupled interior detail — KEPT as dead exports; replaced by TECT_DETAIL_FBM_BASE + rugged scaling
export const TECT_HILL_AMP_BASE       = 0.10   // unused — see TECT_DETAIL_FBM_BASE
export const TECT_HILL_AMP_RANGE      = 0.10   // unused
export const TECT_HILL_SS_LO          = 0.05   // unused
export const TECT_HILL_SS_HI          = 0.30   // unused
// ---------------------------------------------------------------------------
// Tectonically-derived ruggedness cascade constants
// ---------------------------------------------------------------------------
// Ruggedness source weights
export const BROAD_W                  = 0.15   // rad — active-uplift falloff behind convergent boundaries
export const PALEO_W                  = 0.07   // rad — paleo-orogen uplift falloff
export const RUGGED_ACTIVE            = 1.0    // weight of active convergence in ruggedness
export const RUGGED_PALEO             = 0.7    // weight of fossil-orogen proximity (tectonic regime)
export const RUGGED_PALEO_STAGNANT    = 1.1    // stagnant-lid: no convergence, paleo carries all ruggedness
// Broad regional uplift (replaces old TECT_HIGHLANDS_* signed fbm)
export const HIGHLAND_AMP             = 0.18   // broad regional uplift amplitude [was 0.22 — Option C broad uplift takes over active-convergence elevation]
// CC-collision plateau lift (sums with boundaryRelief BR_PLATEAU on far side)
export const PLATEAU_AMP              = 0.20   // regional plateau height contribution
export const PLATEAU_LO               = 0.30   // rugged*ccCollision ramp lo
export const PLATEAU_HI               = 0.70   // rugged*ccCollision ramp hi
export const PLATEAU_HILL_DAMP        = 0.6    // hill-amplitude suppression on plateau tops (flat-topped highlands)
// Hill dissection (dendritic mid-frequency; rugged-gated; FIXED octaves → LOD-invariant)
export const HILL_AMP                 = 0.17   // hill-dissection amplitude
export const HILL_FREQ                = 8.0    // hill spatial frequency
export const HILL_OCT                 = 5      // FIXED octave count — level-independent, trivially LOD-invariant
// Fine detail amplitude (replaces elevation-keyed hillAmp)
export const CRATON_FLOOR             = 0.22   // min fine-detail amplitude fraction in stable interiors [was 0.18 — raised so bare flats get ≳0.22×0.22×HEIGHT_SCALE ≈ ±58m baseline FBM, visible vs smooth B uplands]
export const TECT_DETAIL_FBM_BASE     = 0.22   // base fine-detail amplitude (scaled by craton floor + rugged) [was 0.10 — raises max detail amplitude from ~120m to ~264m, meaningful vs ±660m bDelta]
export const TECT_DETAIL_FBM_SCALE    = 3.2    // fbm frequency scale (unchanged)
export const TECT_DETAIL_RIDGE        = 0.025  // ridged amplitude on land only [was 0.05 — halved to reduce spiky crease at convergent boundaries via boundaryRelief/relief; in B-mode relief is gated off so we use 0.05 there — see conditional below]
export const TECT_DETAIL_RIDGE_SCALE  = 7.0    // ridged frequency scale (unchanged)
// Broad undulation floor — gentle swells/basins on ALL land (not rugged-gated)
export const UNDULATION_AMP           = 0.17   // broad swells/basins on all land [raised — plains read too flat]
export const UNDULATION_FREQ          = 3.5    // mid-large features (was 1.8 = continent-scale, barely visible as relief)
export const UNDULATION_OCT           = 4
export const UNDULATION_VAR_FREQ      = 0.7    // very-low-freq flatter/hillier modulation
export const UNDULATION_VAR_OCT       = 3
export const CRATON_MASK_STR          = 0.70   // undulation tapers to 30% on full orogens (fills plains, spares orogen clamp headroom)
// Rolling-hills floor: plains get partial hills even where rugged≈0
export const HILL_FLOOR               = 0.80   // fraction of HILL_AMP always present on land [raised — plains need visible rolling hills]
// Erosion-steered fine detail (Phase 4)
// HARD_DETAIL_BOOST: how strongly substrate hardness modulates fine-detail amplitude
//   and ridged scale in heightFn.
//   hardnessTerm = 1 + HARD_DETAIL_BOOST * (hardness - 0.5)
//   → hard substrate (hardness=1.0) → hardnessTerm = 1 + 0.5*BOOST  (sharper ridges/cliffs)
//   → soft substrate (hardness=0.0) → hardnessTerm = 1 - 0.5*BOOST  (smoother basins)
//   Applied as a continuous multiplicative factor on detailAmp AND ridgedScale.
//   NOT applied to broadDeform/broadUplift/plateau — macro relief is tectonics-driven, not substrate.
//   Range [0, 1]: 0 = no modulation, 1.0 = ±50% detail amplitude contrast.
export const HARD_DETAIL_BOOST = 0.7  // ±35% detail amplitude contrast between hard/soft substrate

// EROSION_WARP_STR: domain-warp magnitude along downhill flow in noise-space units.
//   0 = isotropic (no steering), ~0.10 = visible gully elongation.
//   CAPPED AT 0.10 — was previously raised to 0.15 then LOWERED to 0.05 to kill
//   swirl/pinwheel artifacts at flow-convergence singularities. Raised back to 0.10
//   (not 0.15) because the un-normalized flowAt magnitude self-fades the warp at
//   singularities. DO NOT raise above 0.10 — pinwheel artifacts return at 0.15+.
export const EROSION_WARP_STR         = 0.10
// EROSION_RIDGED_FLOOR: minimum ridged amplitude fraction on low-discharge interfluves.
//   1.0 = no discharge deepening (pre-erosion behaviour), 0.0 = fully suppressed.
export const EROSION_RIDGED_FLOOR     = 0.35
// EROSION_VINCISION_AMP: peak V-channel depth at full discharge (acc=1, land).
//   [was 0.008 ≈ 10m — invisible vs ±660m bDelta; raised to 0.03 ≈ 36m for readable channel incision]
export const EROSION_VINCISION_AMP    = 0.03

// ---------------------------------------------------------------------------
// Process-palette noise stream IDs (reserved, clear of all existing streams)
//   tectonics: 0–19, climate: 200–201, erosion: 9–10, subsurface: 30–31
//   Process palette streams: 40 (glacial), 41 (aeolian), 42 (karst)
// ---------------------------------------------------------------------------
export const PROC_STREAM_GLACIAL = 40
export const PROC_STREAM_AEOLIAN = 41
export const PROC_STREAM_KARST   = 42

// ---------------------------------------------------------------------------
// Geomorphic process palette — climate-gated qualitative detail modes
//
// Each mode modulates only the fine-detail seasoning layer (detailAmp, ridged,
// domain-warp); macro relief (broadDeform/broadUplift/plateau) is untouched.
// All blend weights are continuous smoothstep functions → no seams.
//
// FLUVIAL (wet temperate) is the default; its weight = 1 - Σ other weights.
//
// GLACIAL (cold × high-elevation × snowy moisture):
//   Smoothed U-profile — dampens fine FBM roughness (scoured surface) and
//   broadens V-incision to a shallower trough. Optional cirque steepening is
//   additive near ridge crests at very high glacial weight.
//     PROC_GLACIAL_STR: max suppression of fine FBM amplitude (0 = off, 1 = fully smooth)
//     PROC_GLACIAL_ELEV_LO/HI: elevation band where glacier supply peaks (normalized height)
//     PROC_GLACIAL_TEMP_LO/HI: temperature window for ice supply (°C)
//     PROC_GLACIAL_CIRQUE_AMP: small-amplitude headwall steepening near ridges
//
// AEOLIAN (arid × high wind):
//   Wind-aligned ripple/dune FBM — FBM input shifted perpendicular to wind to
//   create elongated dune forms, plus a mild deflation smooth.
//     PROC_AEOLIAN_STR: max strength of dune anisotropy blend (0 = off, 1 = full dune)
//     PROC_AEOLIAN_FREQ: spatial frequency of the ripple/dune noise
//     PROC_AEOLIAN_WARP: domain-warp magnitude along wind-perpendicular in noise-space
//     PROC_AEOLIAN_SMOOTH: fraction of FBM amplitude suppressed (deflation flatness)
//
// KARST (wet × soluble substrate):
//   Small negative closed-depression pits. Amplitude is deliberately conservative;
//   seam-safety is the first concern — this mode touches height, so any artifact
//   at baked-grid cell boundaries is visible.
//     PROC_KARST_STR: max pit depth relative to detailAmp (0 = off, 1 = full pits)
//     PROC_KARST_FREQ: spatial frequency of dissolution pits
//     PROC_KARST_SOLUB_HI: rockHardness below which solubility peaks (limestones ≈ medium-hard)
//
// ---------------------------------------------------------------------------

// GLACIAL
export const PROC_GLACIAL_STR      = 0.55  // max fine-FBM smoothing fraction
export const PROC_GLACIAL_ELEV_LO  = 0.10  // normalized height: glacial supply starts ramping
export const PROC_GLACIAL_ELEV_HI  = 0.35  // normalized height: glacial supply at peak
export const PROC_GLACIAL_TEMP_LO  = -40   // °C: colder than this → full cold-weight
export const PROC_GLACIAL_TEMP_HI  = 5     // °C: warmer than this → no cold-weight
export const PROC_GLACIAL_MOIST_LO = 0.20  // moisture below this → no snow supply
export const PROC_GLACIAL_MOIST_HI = 0.60  // moisture above this → full snow supply
export const PROC_GLACIAL_CIRQUE_AMP = 0.004  // cirque headwall steepening amplitude (≈±5 m)

// AEOLIAN
export const PROC_AEOLIAN_STR      = 0.50  // max dune-anisotropy strength
export const PROC_AEOLIAN_WIND_LO  = 0.30  // windSpeed below → no dune
export const PROC_AEOLIAN_WIND_HI  = 0.75  // windSpeed above → full dune
export const PROC_AEOLIAN_ARID_LO  = 0.00  // aridity (1-moisture) below → no dune
export const PROC_AEOLIAN_ARID_HI  = 0.70  // aridity above → full dune
export const PROC_AEOLIAN_FREQ     = 22.0  // dune/ripple FBM spatial frequency
export const PROC_AEOLIAN_WARP     = 0.04  // domain-warp offset perpendicular to wind
export const PROC_AEOLIAN_SMOOTH   = 0.40  // deflation: fraction FBM amplitude removed

// KARST
export const PROC_KARST_STR        = 0.30  // max depression depth (fraction of detailAmp); conservative
export const PROC_KARST_MOIST_LO   = 0.55  // moisture below → no karst
export const PROC_KARST_MOIST_HI   = 0.90  // moisture above → full karst
export const PROC_KARST_SOLUB_LO   = 0.20  // rockHardness (0=soft) low → not soluble (too soft = clay)
export const PROC_KARST_SOLUB_HI   = 0.65  // rockHardness high → not soluble (too hard = granite)
export const PROC_KARST_FREQ       = 28.0  // dissolution pit spatial frequency
export const PROC_KARST_LAND_LO    = 0.02  // crustDist land gate (no karst at coast edge)

// Offshore island / skerry band
export const TECT_ISLAND_AMP          = 0.12
export const TECT_ISLAND_FREQ         = 14.0
export const TECT_ISLAND_OCT          = 4
export const TECT_ISLAND_THRESH       = 0.55
export const TECT_ISLAND_COAST_LO     = -0.05  // crustDist band lo
export const TECT_ISLAND_COAST_HI     = -0.005 // crustDist band hi

// Arc-volcano headroom blend: volcano cone is attenuated by (1 - smoothstep(LO,HI,terrainH))
// so summits on already-high terrain stay well below the clamp (visible crater, no crease),
// while cones over low terrain (ocean → islands) keep full strength and still breach.
// HI lowered to ~0.50 (terrain never approaches 1.0 after de-saturation, so the old
// HI=1.0 barely engaged) — a cone on a 0.25-high platform is now ~70% strength, on a
// 0.40+ cordillera nearly gone, while abyssal/ocean cones (terrainH < LO) are untouched.
export const TECT_VOLC_HEADROOM_LO    = 0.15
export const TECT_VOLC_HEADROOM_HI    = 0.50
// Cap on the summed volcano-cone contribution. A single breaching cone (≤ ~0.64) is
// well under this; the cap only bites where multiple cones overlap, preventing stacked
// summits from blowing past the [-1,1] clamp (the H' max-saturation outlier).
export const TECT_VOLC_SUM_MAX        = 0.64

// ---------------------------------------------------------------------------
// Wide motion-driven uplift (Option C) — broad deformation zones
// ---------------------------------------------------------------------------
export const UPLIFT_AMP           = 0.18   // positive = converging plates → broad mountain belt
export const SUBSIDENCE_AMP       = 0.06   // negative uplift → broad lowland basin
export const UPLIFT_ZONE_W        = 0.25   // half-width falloff (radians)

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

export interface TerrainSamplerOpts {
  seed: number
  radius: number
  heightScale: number
  plateCount: number
  arcDensity: number
  baseTemp: number
  atmosphere: number
  bandCount: number
  axialTiltRad: number
  /** Heat redistribution R ∈ [0,1] forwarded into Climate. Defaults to Climate's own default (1). */
  redistribution?: number
  /** Greenhouse offset in °C forwarded into Climate. Defaults to Climate's own default (0). */
  greenhouse?: number
  /** Lapse rate in °C over full height forwarded into Climate. Defaults to Climate's own default (50). */
  lapseRate?: number
  /** Plate angular velocity multiplier ∈ [0.3, 2.0] from interior vigor. Defaults to 1.0. */
  driftScale?: number
  /**
   * Inject a pre-built Tectonics instance (e.g. fromBaked() in a Web Worker).
   * If omitted, the factory builds one from seed/plateCount/arcDensity.
   */
  tectonics?: Tectonics
  /**
   * Inject a pre-built Climate instance (e.g. fromBaked() in a Web Worker).
   * If omitted, the factory builds one after constructing heightFn + tectonics.
   */
  climate?: Climate
  /**
   * Inject a pre-built Erosion instance (e.g. fromBaked() in a Web Worker).
   * If omitted, no erosion delta is applied (byte-identical to pre-erosion output).
   */
  erosion?: Erosion
  /**
   * Inject a pre-built Subsurface instance (e.g. fromBaked() in a Web Worker).
   * Read-only geology/visualization data — never fed into heightFn.
   * If omitted, subsurface is null.
   */
  subsurface?: Subsurface
  /**
   * When true, omit all orogenic stamps (broadUplift, plateau, broadDeform, relief)
   * so that heightFn returns only the unclipped lowland base + detail.
   * Used as the seed heightFn for the B iterated uplift+erode loop.
   * When false/undefined, behaviour is byte-identical to before (no change in control flow).
   */
  baseOnly?: boolean
  /**
   * When true, gate off the orogenic stamps (broadUplift, plateau, broadDeform, relief)
   * so that bDelta (erosion.deltaAt) owns the elevation budget without clipping.
   * When false/undefined, behaviour is byte-identical to before.
   */
  bActive?: boolean

  // ---- Wind bake params (forwarded to Climate; no effect when opts.climate is injected) ----
  /** Blend weight of vortex swirls vs base zonal flow. Default 0.6. */
  swirlStrength?: number
  /** HIGH-pressure centres per hemisphere. Default 4. */
  nHigh?: number
  /** LOW-pressure centres per hemisphere. Default 3. */
  nLow?: number
  /** Max cross-isobar tilt angle in radians. Default 0.4. */
  crossIsobarMax?: number
  /** Base Gaussian half-width in radians for pressure centres. Default 0.18. */
  sigmaBase?: number
  /** ± latitude spread for pressure-centre placement in radians. Default 0.17. */
  latSpread?: number
  /** +1 prograde / -1 retrograde for baked wind field. Default +1. */
  windRetrograde?: number
  /** Half-width (radians) of equatorial taper for vortex swirls. Default 0.20. */
  equatorTaperWidth?: number
}

export interface TerrainSampler {
  heightFn: (dir: Vector3, level: number) => number
  plateColorFn: (dir: Vector3) => readonly [number, number, number]
  climateFn: (dir: Vector3, height: number) => ClimateSample
  tectonics: Tectonics
  climate: Climate
  /** The Erosion instance passed via opts, or null if none was provided. */
  erosion: Erosion | null
  /** The Subsurface instance passed via opts, or null if none was provided. */
  subsurface: Subsurface | null
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

// Local deriveSeed — same hash function used by tectonics/climate/erosion/subsurface.
// Produces a fully determined child seed from (masterSeed, stream) so each noise
// instance is independent and never collides with another stream.
function _deriveSeed(masterSeed: number, stream: number): number {
  const s = (masterSeed ^ Math.imul(stream + 1, 0xdeadbeef)) >>> 0
  return s ^ (s >>> 16)
}

export function makeTerrainSampler(opts: TerrainSamplerOpts): TerrainSampler {
  const {
    seed, plateCount, arcDensity,
    baseTemp, atmosphere, bandCount, axialTiltRad,
    redistribution, greenhouse, lapseRate,
    driftScale,
  } = opts
  // When baseOnly or bActive is true, orogenic stamps are zeroed.
  // baseOnly: used to generate the seed heightFn for the B loop (no stamps, no relief).
  // bActive: used in the final sampler when B's bDelta owns the elevation budget.
  const _gateStamps = opts.baseOnly === true || opts.bActive === true

  const noise = createNoise3D(seed)

  const tectonics  = opts.tectonics ?? new Tectonics({ seed, plateCount, arcDensity, driftScale })
  const erosion    = opts.erosion ?? null
  const subsurface = opts.subsurface ?? null
  const plates = tectonics.plates
  // Preallocated scratch for erosion.flowAt — zero allocations in the hot path.
  const _flowScratch = new Vector3()

  // ---------------------------------------------------------------------------
  // Process-palette noise instances (streams 40/41/42 — see PROC_STREAM_* above)
  //
  // Each process uses an independent noise instance seeded from the master seed
  // via _deriveSeed(seed, streamId).  This guarantees:
  //   • Different noise field per process (no accidental correlation with FBM noise).
  //   • Same output on main thread and workers (same seed, same stream → same noise).
  //   • No collision with tectonics/climate/erosion/subsurface streams.
  // ---------------------------------------------------------------------------
  const _glacialNoise = createNoise3D(_deriveSeed(seed, PROC_STREAM_GLACIAL))
  const _aeolianNoise = createNoise3D(_deriveSeed(seed, PROC_STREAM_AEOLIAN))
  const _karstNoise   = createNoise3D(_deriveSeed(seed, PROC_STREAM_KARST))

  // ---------------------------------------------------------------------------
  // Climate reference for process-palette weights.
  //
  // Captured at factory construction time from opts.climate (always non-null on
  // workers and on main-thread second-pass builds where climate is injected).
  // Null during the first-pass (base-only) build that feeds Climate's own bake —
  // in that case all process weights default to 0 and FLUVIAL dominates, which
  // produces byte-identical output to pre-process-palette code.
  //
  // Why NOT reference the locally-built `climate` variable:
  //   The `const climate = opts.climate ?? new Climate(...)` line below comes
  //   AFTER this factory code and after `heightFn` is defined.  During Climate's
  //   own bake, it calls heightFn — at that point the outer `climate` const has
  //   not yet been assigned, so accessing it would throw.  Using opts.climate
  //   (already resolved) avoids the circular dependency entirely.
  // ---------------------------------------------------------------------------
  const _climateRef: Climate | null = opts.climate ?? null

  // Preallocated scratch objects for climate lookups inside heightFn.
  // Separate from climateScratch (which is used by climateFn) to avoid aliasing.
  const _climProcSample: ClimateSample = { temperature: 0, moisture: 0 }
  const _windProcSample = { x: 0, y: 0, z: 0, speed: 0 }

  // Inline base-temperature constants derived from the same opts as Climate.
  // Replicates Climate.temperatureAt() without calling into the climate object.
  // EQUATOR_POLE_DELTA = 55 (from climate.ts), LAT_MEAN = 0.5.
  // No sun direction, no redistribution → deterministic, base-only.
  const _procGreenhouse   = opts.greenhouse ?? 0
  const _procLapseRate    = opts.lapseRate ?? 50    // LAPSE_PER_HEIGHT = 50
  const _procAtm          = Math.max(0, Math.min(1, atmosphere))
  const _procGradient     = 55 * (1 - 0.7 * _procAtm)        // equator→pole ΔT
  const _procLapseFactor  = 0.3 + 0.7 * _procAtm
  // Tilt inversion (TILT_INVERT_LO = 54°, TILT_INVERT_HI = 75° — from climate.ts)
  const _tiltLo = (54 * Math.PI) / 180
  const _tiltHi = (75 * Math.PI) / 180
  const _tiltT  = Math.max(0, Math.min(1, (axialTiltRad - _tiltLo) / (_tiltHi - _tiltLo)))
  const _procInvertBlend  = _tiltT * _tiltT * (3 - 2 * _tiltT)

  // Regime selection — hoisted out of the per-vertex closure (branch-free in hot path).
  // plateCount===0 and ===1 both produce plates.length===1 (the non-tectonic stagnant-lid case).
  const isStagnantLid = plates.length === 1
  // paleoAmp removed — paleo weighting now folded into RUGGED_PALEO / RUGGED_PALEO_STAGNANT
  const shelfWBase  = isStagnantLid ? TECT_SHELF_W_STAGNANT    : TECT_SHELF_W_PASSIVE

  // One scratch TectonicQuery per heightFn call — safe because heightFn is called serially.
  const scratch: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0 }

  // Separate scratch for plateColorFn (both called back-to-back on the same dir — see memo below).
  // 1-entry memoization: if dir.x/y/z identical to the previous heightFn call, reuse the
  // query result for plateColorFn, halving Voronoi cost under serial meshing.
  // The memo now stores the full query so plateColorFn can read boundaryDist/convergence/shear.
  let memoX = NaN, memoY = NaN, memoZ = NaN
  let memoBoundaryDist = 0, memoConvergence = 0, memoShear = 0, memoPlateId = 0
  const scratchColor: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0 }
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

  const heightFn = (dir: Vector3, level: number): number => {
    const x = dir.x, y = dir.y, z = dir.z

    tectonics.query(dir, scratch)
    const c = scratch.crustDist  // signed SDF: + = crust/land, - = ocean

    // Memoize last queried dir for plateColorFn re-use (full query state)
    memoX = x; memoY = y; memoZ = z
    memoPlateId = scratch.plateId
    memoBoundaryDist = scratch.boundaryDist
    memoConvergence = scratch.convergence
    memoShear = scratch.shear

    // --- Ocean base (unchanged) ---
    // scratch.baseElevation is the blurred per-plate base elevation (smooth across Voronoi
    // boundaries) — replaces the nearest-neighbor plates[scratch.plateId].baseElevation lookup
    // that caused a hard ~30 m cliff at plate boundaries.
    const oceanBase = TECT_OCEAN_BASE_0
                    - TECT_OCEAN_DEPTH_AMP * Math.min(1, Math.sqrt(Math.max(0, -c) / TECT_OCEAN_DEPTH_SAT))
                    + scratch.baseElevation * TECT_OCEAN_PLATE_MOD

    // --- Land base: interiors climb to ~0.19+ ---
    const landBase = TECT_LAND_BASE_0
                   + TECT_LAND_BASE_SS * _ss3(0, 0.30, c)
                   + scratch.baseElevation * TECT_LAND_PLATE_MOD

    // --- Tectonically-derived ruggedness (master relief modulator; no new noise field) ---
    // activeUplift: convergent-boundary uplift decaying behind the front
    const activeUplift = Math.max(0, scratch.convergence) * Math.exp(-scratch.boundaryDist / BROAD_W)
    // paleoUplift: proximity to fossil orogen core
    const paleoUplift  = Math.exp(-scratch.paleoDist / PALEO_W)
    // rugged: ~0 deep craton, high near orogens/active boundaries
    // stagnant-lid: no active convergence, so paleo carries all ruggedness via higher weight
    const ruggedRaw    = RUGGED_ACTIVE * activeUplift
                       + (isStagnantLid ? RUGGED_PALEO_STAGNANT : RUGGED_PALEO) * paleoUplift
    const rugged       = Math.max(0, Math.min(1, ruggedRaw))
    // ruggedElevation: drives broad regional elevation — paleo arm only.
    // Active convergence elevation is now supplied by broadDeform (Option C).
    const ruggedElevation = Math.max(0, Math.min(1,
      (isStagnantLid ? RUGGED_PALEO_STAGNANT : RUGGED_PALEO) * paleoUplift
    ))
    // Relief suppressed on ocean; only lands get highlands/hills
    const landGate     = _ss3(0, 0.02, c)

    // --- Wide motion-driven deformation (Option C) ---
    // Zeroed when baseOnly or bActive — orogenic stamps are not needed for the seed
    // heightFn and must be absent when bDelta owns the elevation budget.
    let broadDeform = 0
    if (!_gateStamps) {
      const u = tectonics.upliftAt(dir)
      const upliftTerm     = Math.max(0,  u) * UPLIFT_AMP * landGate
      const subsidenceTerm = Math.max(0, -u) * SUBSIDENCE_AMP * landGate
      broadDeform = (upliftTerm - subsidenceTerm) * Math.exp(-scratch.boundaryDist / UPLIFT_ZONE_W)
    }

    // --- Broad undulation: gentle swells/basins on ALL land (not rugged-gated) ---
    // Signed fbm so it raises AND lowers (topographic variety, not just bumps).
    // Sub-province variety field remaps a very-low-freq fbm to [0.5,1.3] so some
    // plains read flatter, others hillier — without imposing a uniform bump offset.
    const undNoise = fbm(noise, x * UNDULATION_FREQ, y * UNDULATION_FREQ, z * UNDULATION_FREQ,
                         { octaves: UNDULATION_OCT })   // ≈[-1,1]
    const undVarRaw = fbm(noise, x * UNDULATION_VAR_FREQ, y * UNDULATION_VAR_FREQ, z * UNDULATION_VAR_FREQ,
                          { octaves: UNDULATION_VAR_OCT })  // ≈[-1,1]
    const undVar    = 0.5 + 0.8 * (undVarRaw * 0.5 + 0.5)  // [0.5, 1.3]
    const undulation = UNDULATION_AMP * undVar * undNoise * landGate * (1 - CRATON_MASK_STR * rugged)

    // CC collision weight: both plates must be continental (high crustDist on both sides)
    const wMine  = _ss3(-0.10, 0.10, c)
    const wOther = _ss3(-0.10, 0.10, scratch.otherCrustDist)
    const ccCollision = wMine * wOther

    // --- Broad uplift (replaces old TECT_HIGHLANDS signed fbm — tectonically gated) ---
    // Cratons (ruggedElevation≈0) → no broad uplift; paleo orogens → full uplift
    // Zeroed when baseOnly or bActive — B's bDelta supplies orogenic elevation instead.
    const broadUplift = _gateStamps ? 0 : HIGHLAND_AMP * ruggedElevation * landGate

    // --- CC-collision plateau: regional smooth lift on both sides of continent–continent zones ---
    // Sums with boundaryRelief BR_PLATEAU on near-boundary side — that's fine, they occupy
    // different spatial extent (boundaryRelief is narrow-front; this is far-field interior)
    // Zeroed when baseOnly or bActive.
    const plateauFrac = _gateStamps ? 0 : _ss3(PLATEAU_LO, PLATEAU_HI, rugged * ccCollision)
    const plateau     = _gateStamps ? 0 : PLATEAU_AMP * plateauFrac * landGate

    // --- Shelf width narrows at active margins ---
    const activeness = (1 - _ss3(0.02, 0.06, scratch.boundaryDist))
                     * _ss3(0.08, 0.25, Math.max(Math.abs(scratch.convergence), Math.abs(scratch.shear)))
    const shelfW = shelfWBase * (1 - activeness) + TECT_SHELF_W_ACTIVE * activeness

    // --- Base elevation: blend ocean/land across the shelf ---
    // combinedLand: landBase + tectonically-derived broad lifts (no generic noise highlands/paleo)
    const combinedLand = landBase + broadUplift + plateau
    const base = combinedLand + (oceanBase - combinedLand) * (1 - _ss3(-shelfW, TECT_COAST_LERP_HI, c))

    // --- Boundary relief profile (asymmetric, all regimes) ---
    // Zeroed when baseOnly or bActive — B's bDelta replaces all orogenic structure.
    const relief = _gateStamps ? 0 : boundaryRelief(scratch, plates, dir, ridgedFn, base)

    // LOD-adaptive octave counts (clamped to [base, max])
    // Caps match the original 50km planet values: fbmOctaves=18, ridgedOctaves=10.
    // Octaves are angular (unit-sphere frequency) — under uniform scale the same
    // caps give identical ANGULAR detail at any radius, so the planet looks the same
    // just bigger. Bumping the caps added detail not present in the original.
    const fbmOctaves    = Math.min(Math.max(level + 2, FBM_BASE_OCTAVES),    18)
    const ridgedOctaves = Math.min(Math.max(level - 2, RIDGED_BASE_OCTAVES), 10)

    // ---- Phase-4 Steered fine detail (erosion present only) ----------------
    // When an Erosion instance is available we warp the FBM input along the
    // downhill flow direction so ridges/gullies elongate into valleys, and we
    // deepen the incision proportionally to the drainage discharge (accAt).
    //
    // LOD-consistency argument:
    //   Both `acc` and `_flowScratch` are pure functions of `dir` only — they
    //   query fixed-resolution erosion fields that carry no level information.
    //   The warp offset added to (fx,fy,fz) and the V-incision scale are therefore
    //   constant across all LOD levels for the same `dir`. Because the additive-
    //   octave structure is preserved (same loop, same MAX_AMP_FBM6 normalizer),
    //   octaves 1-6 are byte-identical between level 3 and level 12; only the
    //   additive higher octaves grow — exactly the same guarantee as before.
    //   No steering quantity touches `level`. Popping-free.
    //
    // erosion=null → warpX/Y/Z = x/y/z, acc = 0, vIncision = 0: byte-identical
    //   to the pre-erosion path.
    let warpX = x * TECT_DETAIL_FBM_SCALE
    let warpY = y * TECT_DETAIL_FBM_SCALE
    let warpZ = z * TECT_DETAIL_FBM_SCALE
    let acc = 0
    if (erosion) {
      // Downhill flow tangent in world-space (unit vector, always perpendicular to dir).
      erosion.flowAt(dir, _flowScratch)
      // Anisotropic domain warp: shift the FBM sample point along the flow direction
      // by a small fraction of a noise-space unit. Ridges/gullies will elongate
      // downhill. Warp strength tuned conservative — dial EROSION_WARP_STR to taste.
      const warp = EROSION_WARP_STR * TECT_DETAIL_FBM_SCALE
      warpX += _flowScratch.x * warp
      warpY += _flowScratch.y * warp
      warpZ += _flowScratch.z * warp
      acc = erosion.accAt(dir)   // 0..1 discharge (pure fn of dir)
    }

    // ---- Detail FBM: additive-octave formulation for LOD consistency --------
    // Outer amplitude scales by a drainage proxy: max of tectonic ruggedness (convergent
    // relief) and erosion discharge development when erosion is active.
    // This re-keys detail to the EROSION/drainage signal so B-mode uplands (where rugged
    // is near zero because orogenic stamps are gated off) still get meaningful detail in
    // developed drainage areas.
    // Smoothstep bounds: 0.30..0.85 match the _ss3(0.30, 0.85, acc) gate used in chan
    // below, so detail peaks in the same zones. Reuses the already-computed `acc` —
    // no second accAt call. Pure fn of dir only (acc queries fixed-resolution baked fields).
    // drainProxy/detailAmp declared HERE — after both `rugged` (line ~329) and `acc`
    // (set above) — so declaration order is: rugged → acc → drainProxy → detailAmp → FBM.
    const drainProxy = erosion ? Math.max(rugged, _ss3(0.30, 0.85, acc)) : rugged
    // Smoothstep drainProxy before applying to detailAmp so the amplitude transition
    // is C1 across 256² erosion cell edges (Fix 2 / Mechanism A).
    const detailAmpBase = Number.isFinite(drainProxy)
      ? TECT_DETAIL_FBM_BASE * (CRATON_FLOOR + (1 - CRATON_FLOOR) * _ss3(0.10, 0.70, drainProxy))
      : TECT_DETAIL_FBM_BASE * CRATON_FLOOR

    // Rock-hardness fine-detail modulation (T2):
    // scratch.rockHardness [0,1] is already filled by tectonics.query() above (line ~315).
    // It is a pure C1 baked-grid lookup — same on main thread and workers (DETERMINISM OK).
    // hardnessTerm is a CONTINUOUS linear function of hardness — no hard threshold,
    // so it produces no seams between chunks sampling the baked grid at different offsets.
    // Applied ONLY to fine-detail seasoning (detailAmp, ridgedScale); macro relief is unchanged.
    const hardnessTerm = 1.0 + HARD_DETAIL_BOOST * (scratch.rockHardness - 0.5)
    const detailAmp    = detailAmpBase * hardnessTerm

    // ---- Climate-gated qualitative process palette (WI-3b) ------------------
    //
    // Derive FLUVIAL / GLACIAL / AEOLIAN / KARST blend weights from:
    //   - BASE temperature (analytic inline, no sun/time term)
    //   - Baked moisture + wind (from _climateRef when non-null)
    //   - rockHardness (already in scratch)
    //   - elevation proxy (c = crustDist, surfaceH not yet computed — use landGate + base)
    //
    // Weight formula (all continuous smoothstep blends):
    //   wGlacial = _ss3(TEMP_LO, TEMP_HI, −temp) × _ss3(ELEV_LO, ELEV_HI, baseH) × _ss3(MOIST_LO, MOIST_HI, moisture)
    //   wAeolian = _ss3(WIND_LO, WIND_HI, windSpeed) × _ss3(ARID_LO, ARID_HI, 1−moisture)
    //   wKarst   = _ss3(MOIST_LO, MOIST_HI, moisture) × solubility × landGate_strict
    //   wFluvial = 1 − wGlacial − wAeolian − wKarst  (clamped ≥ 0)
    //
    // Determinism guarantee:
    //   • BASE temperature: pure analytic function of dir.y + height. No baked field.
    //     Constants (_procGradient, _procLapseFactor, etc.) are precomputed from opts at
    //     factory time — same on main thread and workers for the same opts.
    //   • Moisture: sampleSmooth on the baked 128² moisture field via _climateRef.sample().
    //     Both main thread (injected climate) and workers (fromBaked) share the SAME baked
    //     arrays, so identical dir → identical moisture. Bilinear = sampleSmooth, not C1,
    //     because Climate.sample() uses sampleSmooth internally. The moisture field is
    //     smooth enough (blurred, 128² resolution) that sampleSmooth artifacts are negligible.
    //   • Wind: same baked field, same bilinear sampler.
    //   • When _climateRef is null (base-only first pass): moisture = 0.5, windSpeed = 0,
    //     wGlacial = 0, wAeolian = 0, wKarst = 0 → pure FLUVIAL.
    //     That pass is used only during Climate's own bake; the final mesh always has climate.
    //   • All weights are C0-continuous (smoothstep of smooth inputs) — no hard if/else on
    //     any baked value → no seams between adjacent chunks.
    //   • _glacialNoise / _aeolianNoise / _karstNoise are independent noise instances derived
    //     from (seed, streamId) at factory time → deterministic, no collision with other streams.
    //
    // Only the fine-detail FBM output is modified; macro relief is untouched.
    // ---------------------------------------------------------------------

    // ---- Base temperature (analytic, no baked field, no sun/time term) ----
    // Replicates Climate.temperatureAt() for a reference height of `base`
    // (pre-detail elevation), which is the best stable proxy available at this
    // point in heightFn before the full surface is assembled.
    const _baseH     = base  // pre-detail elevation proxy
    const _absY      = Math.abs(y)
    const _cosLat    = Math.sqrt(Math.max(0, 1 - y * y))
    const _insolation = _cosLat * (1 - _procInvertBlend) + _absY * _procInvertBlend
    const _lapse     = _procLapseRate * Math.max(0, _baseH) * _procLapseFactor
    const _baseTemp  = baseTemp + _procGreenhouse + _procGradient * (_insolation - 0.5) - _lapse

    // ---- Baked moisture + wind (from _climateRef; default to neutral if null) ----
    let _procMoisture  = 0.5
    let _procWindSpeed = 0.0
    let _procWindX     = 0.0
    let _procWindZ     = 0.0
    if (_climateRef !== null) {
      _climateRef.sample(dir, _baseH, _climProcSample)
      _procMoisture = _climProcSample.moisture
      _climateRef.windAt(dir, _windProcSample)
      _procWindSpeed = _windProcSample.speed
      _procWindX     = _windProcSample.x
      _procWindZ     = _windProcSample.z
    }
    const _procAridity = 1.0 - _procMoisture

    // ---- GLACIAL weight ----
    // coldness: inverted temperature ramp (cold → 1, warm → 0)
    const _coldness    = _ss3(PROC_GLACIAL_TEMP_HI, PROC_GLACIAL_TEMP_LO, _baseTemp)
    // elevation supply: high land only (crustDist > LO_ELEV, using base as proxy)
    const _elevSupply  = _ss3(PROC_GLACIAL_ELEV_LO, PROC_GLACIAL_ELEV_HI, _baseH)
    // snow supply: wet enough to bring snow (moisture > LO)
    const _snowSupply  = _ss3(PROC_GLACIAL_MOIST_LO, PROC_GLACIAL_MOIST_HI, _procMoisture)
    const wGlacial     = _coldness * _elevSupply * _snowSupply * landGate

    // ---- AEOLIAN weight ----
    const _windGate    = _ss3(PROC_AEOLIAN_WIND_LO, PROC_AEOLIAN_WIND_HI, _procWindSpeed)
    const _aridGate    = _ss3(PROC_AEOLIAN_ARID_LO, PROC_AEOLIAN_ARID_HI, _procAridity)
    // Ocean surface is not aeolian (sand dunes need land); apply landGate
    const wAeolian     = _windGate * _aridGate * landGate * PROC_AEOLIAN_STR

    // ---- KARST weight ----
    // Solubility proxy: peaks at medium-hard rock (limestone window); too soft = clay, too hard = granite
    const _solubility  = _ss3(PROC_KARST_SOLUB_LO, (PROC_KARST_SOLUB_LO + PROC_KARST_SOLUB_HI) * 0.5, scratch.rockHardness)
                       * (1 - _ss3((PROC_KARST_SOLUB_LO + PROC_KARST_SOLUB_HI) * 0.5, PROC_KARST_SOLUB_HI, scratch.rockHardness))
    const _karstMoist  = _ss3(PROC_KARST_MOIST_LO, PROC_KARST_MOIST_HI, _procMoisture)
    // Karst only on land, not at coast fringe (crustDist > PROC_KARST_LAND_LO)
    const _karstLand   = _ss3(PROC_KARST_LAND_LO, PROC_KARST_LAND_LO * 3, c)
    const wKarst       = _karstMoist * _solubility * _karstLand * PROC_KARST_STR

    // ---- FLUVIAL weight = remainder (clamped non-negative) ----
    // wFluvial documents the conceptual weight budget (wGlacial + wAeolian + wKarst ≤ 1)
    // so future modes can see how much headroom remains. Not used in arithmetic (FLUVIAL
    // is the implicit baseline and does not multiply anything).
    const wFluvial     = Math.max(0, 1 - wGlacial * PROC_GLACIAL_STR - wAeolian - wKarst)
    void wFluvial  // intentionally unused in arithmetic; see comment above

    // ---- Apply GLACIAL + AEOLIAN deflation: dampen fine FBM roughness ----
    // Glacially-scoured surfaces are smooth (U-valleys, ice-planed bedrock):
    //   attenuate FBM amplitude by (1 - PROC_GLACIAL_STR × wGlacial).
    // Aeolian deflation (wind-scoured lag gravel) removes fluvial FBM roughness
    //   proportional to wAeolian × PROC_AEOLIAN_SMOOTH, replaced by dune ripples below.
    // Combined: detailAmpProc = detailAmp × (1 − glacialSup) × (1 − aeolianSup).
    const _glacialSmooth  = 1.0 - PROC_GLACIAL_STR * wGlacial
    const _aeolianDeflate = 1.0 - wAeolian * PROC_AEOLIAN_SMOOTH
    const detailAmpProc   = detailAmp * _glacialSmooth * _aeolianDeflate

    // ---- Apply AEOLIAN: wind-aligned dune ripple + domain warp ----
    // Two complementary contributions, both scaled by wAeolian:
    //
    // (1) Domain warp: shift the FBM sample point PERPENDICULAR to wind (in the xz
    //     tangent plane) so the existing detail elongates into dune ridges aligned with wind.
    //     perpendicular direction: (-windZ, windX) in the xz plane.
    //
    // (2) Ripple texture: a separate high-frequency noise (_aeolianNoise, stream 41)
    //     sampled ALONG the wind direction, creating transverse dune crests / sand ripples.
    //     The ripple point is shifted ALONG wind so crests run perpendicular to flow.
    //     Amplitude scales with wAeolian × detailAmp × PROC_AEOLIAN_SMOOTH fraction.
    //
    // Both contributions are zero when wAeolian = 0 or _climateRef = null
    // (wind x/z = 0 → perpendicular = 0 → no warp, ripple amp = 0 → no ripple).
    const _aeolianWarpMag  = PROC_AEOLIAN_WARP * TECT_DETAIL_FBM_SCALE * wAeolian
    const _perpX = -_procWindZ  // perpendicular to wind in xz plane (dune elongation axis)
    const _perpZ =  _procWindX
    let procWarpX = warpX + _perpX * _aeolianWarpMag
    let procWarpY = warpY                             // y component unchanged (polar axis)
    let procWarpZ = warpZ + _perpZ * _aeolianWarpMag

    // Aeolian ripple: sample _aeolianNoise shifted along wind so ripples align across wind.
    // Uses a separate noise stream (41) independent of the main FBM.
    const _aeolianShift = PROC_AEOLIAN_FREQ * 0.5 * wAeolian  // collapse to 0 when wAeolian=0
    const _aeolianRipple = _aeolianNoise(
      x * PROC_AEOLIAN_FREQ + _procWindX * _aeolianShift,
      y * PROC_AEOLIAN_FREQ,
      z * PROC_AEOLIAN_FREQ + _procWindZ * _aeolianShift,
    )
    // Ripple adds positive relief (dune crests); amplitude fraction comes from PROC_AEOLIAN_SMOOTH
    // (deflation suppresses the fluvial FBM by this fraction; we reuse the constant to keep
    // dune amplitude matched to the suppressed detail budget).
    const aeolianRippleAdd = detailAmp * PROC_AEOLIAN_SMOOTH * wAeolian * _aeolianRipple

    // ---- Apply KARST: negative closed-depression pits ----
    // A ridged-like pit pattern: (1 - |karstNoise|)² gives sharp pit floors.
    // Negated and amplitude-scaled by wKarst × detailAmp.
    // Uses PROC_STREAM_KARST noise — independent of the main FBM stream.
    const _karstRaw = _karstNoise(x * PROC_KARST_FREQ, y * PROC_KARST_FREQ, z * PROC_KARST_FREQ)
    const _karstPit = -((1.0 - Math.abs(_karstRaw)) ** 2)  // ∈ [-1, 0], peaks at noise=0
    const karstAdd  = detailAmp * wKarst * _karstPit      // small negative depression term

    // ---- Glacial cirque headwall steepening ----
    // Near ridge crests at high glacial weight, add a small positive ridge-aligned
    // perturbation (the cirque headwall). Uses _glacialNoise to keep it independent.
    // Amplitude is deliberately small (PROC_GLACIAL_CIRQUE_AMP < 0.005 → ±6 m).
    const _cirqueRaw = _glacialNoise(x * TECT_DETAIL_RIDGE_SCALE * 1.5, y * TECT_DETAIL_RIDGE_SCALE * 1.5, z * TECT_DETAIL_RIDGE_SCALE * 1.5)
    const cirqueAdd  = PROC_GLACIAL_CIRQUE_AMP * wGlacial * _cirqueRaw

    let fbmValue = 0; let fbmAmp = 1;
    let fbmFreq = 1;
    for (let o = 0; o < fbmOctaves; o++) {
      fbmValue += fbmAmp * noise(procWarpX * fbmFreq, procWarpY * fbmFreq, procWarpZ * fbmFreq);
      fbmAmp   *= FBM_GAIN;
      fbmFreq  *= FBM_LAC;
    }
    // detailAmpProc is detailAmp attenuated by glacial smoothing (see process palette above).
    const detailFbm = detailAmpProc * (fbmValue / MAX_AMP_FBM6)

    // ---- Detail ridged: near coasts only ----
    // `chan`: smoothstepped discharge gate — suppresses the concentric iso-contour
    // terrace banding that raw acc*acc produced. Declared here so both ridgedScale
    // and vIncision can reference it without a forward-reference error.
    const chan = _ss3(0.30, 0.85, acc)

    let ridgedValue = 0; let ridgedAmp = 1; let ridgedFreq = 1;
    const rx = x * TECT_DETAIL_RIDGE_SCALE, ry = y * TECT_DETAIL_RIDGE_SCALE, rz = z * TECT_DETAIL_RIDGE_SCALE;
    // Discharge-gated ridged amplitude: high-acc cells get full sharp ridges in
    // gullies; low-acc interfluves get reduced amplitude for gentler detail.
    // ridgedScale is a pure fn of `acc` which is a pure fn of `dir` — no `level`.
    // Use the same smoothstepped discharge (chan) as vIncision to avoid axis-aligned banding.
    // Also multiplied by hardnessTerm so hard substrate reads sharper/cliffier even
    // outside channels, and soft substrate reads smoother on interfluves.
    const ridgedScaleBase = erosion ? (EROSION_RIDGED_FLOOR + (1 - EROSION_RIDGED_FLOOR) * chan) : 1.0
    const ridgedScale = ridgedScaleBase * hardnessTerm
    for (let o = 0; o < ridgedOctaves; o++) {
      const n = noise(rx * ridgedFreq, ry * ridgedFreq, rz * ridgedFreq);
      ridgedValue += ridgedAmp * (1.0 - Math.abs(n)) ** 2;
      ridgedAmp   *= FBM_GAIN;
      ridgedFreq  *= FBM_LAC;
    }
    // In B/erosion mode, relief (orogenic boundary stamps) is gated off, so the original
    // justification for halving TECT_DETAIL_RIDGE (reduce spiky crease at convergent boundaries)
    // no longer applies. Use 0.05 when erosion is active, TECT_DETAIL_RIDGE (0.025) otherwise.
    const ridgeAmp = erosion ? 0.05 : TECT_DETAIL_RIDGE
    // _glacialSmooth + _aeolianDeflate both attenuate ridged: scoured terrain loses sharp ridges.
    const detailRidged = ridgeAmp * ridgedScale * (ridgedValue / MAX_AMP_RIDGED4) * _ss3(0, 0.08, c)
                       * (0.7 + 0.3 * Math.exp(-scratch.boundaryDist / 0.18))
                       * _glacialSmooth * _aeolianDeflate

    // ---- V-shaped incision deepened by discharge ----------------------------
    // A small negative term that carves a sharp channel floor proportional to
    // drainage discharge. Only active where acc > 0 and land (landGate).
    // Pure fn of `dir` (acc is fixed-resolution, level-independent).
    const vIncision = erosion
      ? -EROSION_VINCISION_AMP * chan * landGate
      : 0

    // karstAdd: small negative depression pits (karst process; see process palette above)
    // cirqueAdd: small positive ridge headwall steepening (glacial cirques)
    // aeolianRippleAdd: high-frequency transverse dune crests (aeolian process)
    const detail = detailFbm + detailRidged + vIncision + karstAdd + cirqueAdd + aeolianRippleAdd

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
    // broadUplift+plateau already inside base via combinedLand+shelf blend;
    // hills+detail added on top of the tectonic base.
    const surfaceH = base + relief + detail + islandH + undulation + broadDeform  // pre-erosion surface
    const volcano  = Math.min(TECT_VOLC_SUM_MAX, tectonics.volcanoElevation(dir))
    const headroom = 1 - _ss3(TECT_VOLC_HEADROOM_LO, TECT_VOLC_HEADROOM_HI, surfaceH) // volcano keyed to PRE-erosion surface
    const erosionH = erosion ? erosion.deltaAt(dir) : 0                                // <=0 carves valleys, >=0 deposits

    return Math.max(-1, Math.min(1, surfaceH + erosionH + volcano * headroom))
  }

  // plateColorFn: uses the 1-entry memo to avoid a redundant Voronoi query when
  // the mesher calls heightFn then plateColorFn for the same dir back-to-back.
  // Near tectonic boundaries, blends a regime tint (red=collision, blue=rift, yellow=shear).
  const plateColorFn = (dir: Vector3): readonly [number, number, number] => {
    let pid: number
    let bd: number, conv: number, sh: number
    if (dir.x === memoX && dir.y === memoY && dir.z === memoZ) {
      pid = memoPlateId
      bd = memoBoundaryDist
      conv = memoConvergence
      sh = memoShear
    } else {
      tectonics.query(dir, scratchColor)
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
  const climateQueryScratch: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0 }
  const climate = opts.climate ?? new Climate({
    seed,
    baseTemp,
    atmosphere,
    bandCount,
    axialTiltRad,
    redistribution,
    greenhouse,
    lapseRate,
    // Wind bake params — forwarded when provided; Climate uses its own defaults when absent.
    swirlStrength:     opts.swirlStrength,
    nHigh:             opts.nHigh,
    nLow:              opts.nLow,
    crossIsobarMax:    opts.crossIsobarMax,
    sigmaBase:         opts.sigmaBase,
    latSpread:         opts.latSpread,
    retrograde:        opts.windRetrograde,
    equatorTaperWidth: opts.equatorTaperWidth,
    heightFn,
    crustDistAt: (dir: Vector3): number => tectonics.query(dir, climateQueryScratch).crustDist,
  })

  const climateScratch: ClimateSample = { temperature: 0, moisture: 0 }
  const climateFn = (dir: Vector3, height: number): ClimateSample =>
    climate.sample(dir, height, climateScratch)

  return { heightFn, plateColorFn, climateFn, tectonics, climate, erosion, subsurface }
}
