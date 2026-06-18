/**
 * Climate system — generic, parameter-driven temperature + moisture fields.
 *
 * Design philosophy: NOTHING here is Earth-hardcoded. Earth's familiar patterns
 * (≈3 circulation cells, dry deserts near ±30°, wet equator, polar ice) EMERGE
 * from the parameters. Feed a different `bandCount`, `baseTemp`, `atmosphere`, or
 * `axialTiltRad` and you get a different — but internally consistent — world.
 *
 *   • TEMPERATURE is analytic and evaluated per call (cheap, exact, zero-alloc):
 *       a latitude profile (warm equator → cold pole) scaled by an equator-pole
 *       delta that a thick atmosphere shrinks, minus an altitude lapse. At extreme
 *       axial tilt (> 54°) the profile inverts toward pole-favouring insolation.
 *
 *   • MOISTURE is BAKED once into a cube-map field at construction and sampled
 *       smoothly per vertex (keeps the per-vertex cost to a bilinear-ish lookup).
 *       It composes three generic ingredients:
 *         - bandWetness: Hadley-style alternating wet/dry bands. `bandCount`
 *           alternations from equator to pole. bandCount=3 → wet equator, dry
 *           ≈±30°, wet ≈±60°, dry poles (Earth-like). bandCount=1 → wet equator,
 *           dry pole.
 *         - seaProximity: exponential falloff with inland distance (drier inland).
 *         - rainShadow: march upwind along the band's zonal wind; tall terrain
 *           upwind dries the leeward side.
 *
 * Determinism: any stochastic offsets derive from `seed` via deriveSeed (a local
 * splitmix32 stream, stream id documented at the call site). No Math.random.
 *
 * Stream ids reserved:
 *   200 — moisture-field jitter (analytic band model currently; reserved for future use)
 *   201 — wind-field low-amplitude swirl jitter (bakeWind — see equator taper note below)
 *   202 — transient curl-noise eddy (visualization-only, windAtTime)
 *
 * bakeWind equator taper: an eqMask = smoothstep(0, sin(equatorTaperWidth), |dir.y|)
 * is applied to the vortex swirl contribution only. This damps pressure vortices at
 * the equator (where physical Coriolis → 0) without touching the zonal base or
 * Coriolis tilt (which are already physically correct near the equator).
 */

import { Vector3 } from 'three'
import {
  dirToTexel,
  texelToDir,
  texelIndex,
  neighborTexel,
  sampleSmooth,
} from './cubemap'
import { createNoise3D } from './noise'

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ClimateParams {
  /** Master seed. Stochastic offsets derive from this (deriveSeed). */
  seed: number
  /** Mean surface temperature, °C-ish (Earth ≈ 15). Planet-class knob. */
  baseTemp: number
  /** 0..1: thick (→1) gives uniform/small gradients, thin (→0) gives extremes. Default ≈0.6. */
  atmosphere: number
  /** Circulation cells per hemisphere (1 slow … ~7 fast). Planet derives from rotation. */
  bandCount: number
  /** Axial tilt in radians — drives the > 54° insolation inversion. */
  axialTiltRad: number
  /** Heat redistribution R ∈ [0,1]: 0 = instant day/night extremes, 1 = fully climatological (uniform). Used only by temperatureLiveAt. Default 1. */
  redistribution?: number
  /** Greenhouse offset in °C added to baseTemp across the whole surface. Default 0. */
  greenhouse?: number
  /** Lapse rate: °C lost over the full normalized terrain height (replaces LAPSE_PER_HEIGHT). Default 50. */
  lapseRate?: number

  // ---- Wind bake parameters (affect only bakeWind; fromBaked stays unchanged) ----

  /**
   * Blend weight of vortex swirls vs the base zonal flow. 0 = pure zonal belts,
   * 1 = full vortex contribution. Earth default ≈ 0.6.
   */
  swirlStrength?: number
  /** Number of HIGH-pressure centres per hemisphere. Earth default 4. */
  nHigh?: number
  /** Number of LOW-pressure centres per hemisphere. Earth default 3. */
  nLow?: number
  /**
   * Maximum cross-isobar tilt angle (radians). Wind deflects rightward (N) /
   * leftward (S) from the pressure-gradient direction by up to this much at the
   * equator, fading to 0 at the poles. Earth default ≈ 0.4 rad.
   */
  crossIsobarMax?: number
  /**
   * Base half-width (radians) of each pressure-system Gaussian. Scaled by
   * sqrt(3/bandCount) so swirls stay visible regardless of band density.
   * Default ≈ 0.18 rad — makes vortices clearly visible on a 500 km planet.
   */
  sigmaBase?: number
  /**
   * ± spread (radians) of pressure-centre latitudes around their target belt.
   * Default ≈ 0.17 rad.
   */
  latSpread?: number
  /**
   * Sign of the zonal base flow at the equator (+1 = prograde/eastward equatorial
   * trade winds — Earth-like; -1 = retrograde). Default +1.
   * Note: the sign convention in U_zonal uses -cos(…) so retrograde=+1 gives
   * westward trades (easterly) at the equator, matching Earth.
   */
  retrograde?: number
  /**
   * Half-width (radians) of the equatorial taper applied to the vortex swirl only.
   * Below this latitude the swirl is damped to zero (physical: Coriolis→0 at equator).
   * Default 0.20 rad. Range [0, 0.5].
   */
  equatorTaperWidth?: number
}

export interface ClimateSample {
  /** °C-ish. */
  temperature: number
  /** 0..1. */
  moisture: number
}

/**
 * Baked wind sample at a surface point. The vector (x/y/z) is a unit tangent-plane
 * wind direction in planet-local space (magnitude ≈ 1 when speed > 0, zero vector at
 * poles where zonal wind collapses). Speed is a non-negative dimensionless measure
 * with 1.0 ≈ peak trade-wind magnitude; it couples to band shear and ΔT strength.
 */
export interface WindSample {
  /** Planet-local wind direction (unit tangent-plane vector). */
  x: number
  y: number
  z: number
  /** Dimensionless wind speed ≥ 0 (1 = strongest band-boundary shear). */
  speed: number
}

/**
 * Parameters for the transient (visualization-only) wind animation layer.
 * All fields are live-updatable via setWindTransient() — no rebake needed.
 */
export interface WindTransientParams {
  /** Angular drift speed of vortex centres along their zonal belt (rad/s). Default 0.015. */
  driftSpeed: number
  /** Oscillation rate of vortex amplitude (rad/s). Default 0.25. */
  pulseRate: number
  /** Fractional depth of vortex amplitude pulsing [0,1]. Default 0.35. */
  pulseDepth: number
  /** Blend weight of curl-noise eddies onto the wind velocity. Default 0.35. */
  eddyStrength: number
  /** Spatial frequency scale for curl-noise eddies. Default 6. */
  eddyScale: number
  /** Temporal frequency scale for curl-noise eddies. Default 0.3. */
  eddyTimeScale: number
}

/** Internal representation of one pressure vortex for both bake and windAtTime. */
interface Vortex {
  px: number; py: number; pz: number
  sigma: number
  A: number
  /** Longitude of the vortex centre at t=0 (radians). */
  baseLon: number
  /** Latitude of the vortex centre (radians, signed). */
  latC: number
  /** cos(latC) — precomputed for drifting centre rebuilds. */
  cosLatC: number
  /** sin(latC) — precomputed for drifting centre rebuilds. */
  sinLatC: number
  /** Per-vortex phase offset = i * 2.39996 (golden angle) — deterministic, no RNG draw. */
  phase: number
}

/**
 * Serialisable snapshot of a baked Climate — contains everything needed to
 * reconstruct a sampler-only instance in a Web Worker WITHOUT re-baking.
 *
 * Transfer `moistureField` as a Transferable (ArrayBuffer) across the Worker
 * message boundary; pass the rest as plain JSON.
 */
export interface ClimateBaked {
  /** Resolution used during the bake (= MOIST_RES = 128). */
  moistRes: number
  /** Baked cube-map moisture field, length 6·moistRes². */
  moistureField: Float32Array
  /**
   * Baked wind direction (X/Y/Z) + speed cube-map fields, each length 6·moistRes².
   * windX/Y/Z form a unit tangent-plane vector (zero at poles); windSpeed ∈ [0,1].
   * Stream id 201 was consumed during the bake for low-amplitude swirl jitter.
   */
  windX: Float32Array
  windY: Float32Array
  windZ: Float32Array
  windSpeed: Float32Array
  /** Scalar params needed to recompute the sampler-side derived values. */
  baseTemp: number
  atmosphere: number
  bandCount: number
  axialTiltRad: number
  redistribution?: number
  greenhouse?: number
  lapseRate?: number
}

interface ClimateDeps {
  /** Terrain height in [-1,1] for a unit dir at a given LOD level (for rain shadow). */
  heightFn: (dir: Vector3, level: number) => number
  /** Signed distance to crust edge (≈ coast) in radians; + = inland. From tectonics. */
  crustDistAt: (dir: Vector3) => number
}

// ---------------------------------------------------------------------------
// Tunable constants (the "physics" of the model — generic, not Earth-specific)
// ---------------------------------------------------------------------------

/** Cube-map resolution for the baked moisture field. 128 → 6·128² = 98304 texels. */
const MOIST_RES = 128

/** Equator→pole temperature drop, °C, at THIN atmosphere (atmosphere→0). */
const EQUATOR_POLE_DELTA = 55
/** Centres the latitude profile so baseTemp ≈ area-mean. cosLat area-mean over a sphere is 0.5. */
const LAT_MEAN = 0.5
/** Lapse rate: °C lost over the full normalized height (=1), at full atmosphere. */
const LAPSE_PER_HEIGHT = 50

/** Tilt (rad) where the insolation inversion begins (≈54°) and completes (≈75°). */
const TILT_INVERT_LO = (54 * Math.PI) / 180
const TILT_INVERT_HI = (75 * Math.PI) / 180

/** Inland moisture falloff scale (radians of crustDist). Smaller → coasts dry out faster.
 *  Raised so moisture penetrates deep continental interiors (the ~9 km of 0.18 left
 *  interiors at ~0 → all-desert). ~0.6 rad ≈ 30 km penetration on the 50 km planet:
 *  coasts stay wetter than deep interiors (realistic gradient) without uniform desert. */
const MOIST_INLAND_SCALE = 0.7
/** Floor on sea-proximity so no land is at exactly 0 from inland distance alone —
 *  the deepest interiors still get a little moisture (only the band model can fully dry them). */
const MOIST_SEAPROX_FLOOR = 0.21
/** Floor on the raw band wetness before the bandContrast power — descending dry
 *  bands keep a little moisture (real dry bands aren't bone-dry; the deep-interior
 *  desert comes from seaProximity/rainShadow, not from a literal-0 band). */
const BAND_FLOOR = 0.075
/** Rain-shadow strength: leeward drying per unit upwind height excess. */
const SHADOW_K = 1.6
/** Floor for the rain-shadow multiplier (a desert behind a wall still keeps a little moisture). */
const MIN_SHADOW = 0.25
/** Upwind march: number of steps and per-step angular distance (radians). */
const UPWIND_STEPS = 3
const UPWIND_STEP_RAD = 0.02
/** Coarse LOD level at which heightFn is sampled for rain shadow (cheap, stable). */
const RAINSHADOW_LEVEL = 8
/** Overall moisture gain before clamp (keeps the wettest places near saturation).
 *  Raised so the overall moisture level lifts mid-latitudes above the desert threshold. */
const MOIST_GAIN = 1.23
/** Cold places hold less liquid moisture: below this temp (°C) moisture is attenuated. */
const FROZEN_TEMP = -5
/** Width (°C) over which the frozen attenuation ramps in below FROZEN_TEMP. */
const FROZEN_WIDTH = 12

// ---------------------------------------------------------------------------
// Local deterministic PRNG (mirrors tectonics' private splitmix32 — kept local
// because tectonics does not export deriveSeed/makeRng).
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
 * Climate uses stream id 200 (moisture-field jitter) — chosen well clear of the
 * tectonics stream ids (0..15) so the two systems never collide on a master seed.
 */
function deriveSeed(masterSeed: number, stream: number): number {
  const s = (masterSeed ^ Math.imul(stream + 1, 0xdeadbeef)) >>> 0
  return splitmix32Step(s)
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0))
  return t * t * (3 - 2 * t)
}

// ---------------------------------------------------------------------------
// Climate
// ---------------------------------------------------------------------------

export class Climate {
  private readonly seed: number
  private readonly baseTemp: number
  private readonly atmosphere: number
  private readonly bandCount: number
  private readonly axialTiltRad: number
  private readonly redistribution: number
  private readonly greenhouse: number
  private readonly lapseRate: number

  // Wind bake parameters — affect only bakeWind(); not serialised into ClimateBaked
  // (the baked arrays already encode their effect, so fromBaked stays unchanged).
  private readonly swirlStrength: number
  private readonly nHigh: number
  private readonly nLow: number
  private readonly crossIsobarMax: number
  private readonly sigmaBase: number
  private readonly latSpread: number
  private readonly retrograde: number
  private readonly equatorTaperWidth: number

  /** Precomputed tilt-inversion blend (0 = normal gradient, 1 = fully pole-favouring). */
  private readonly invertBlend: number
  /** Precomputed equator→pole temperature gradient after the atmosphere shrink. */
  private readonly gradient: number
  /** Precomputed atmosphere factor for the altitude lapse. */
  private readonly lapseFactor: number
  /** Precomputed overall moisture scale — thin atmosphere → drier (more deserts). */
  private readonly moistGain: number
  /** Precomputed band-contrast exponent — thin atmosphere → sharper/deeper dry bands. */
  private readonly bandContrast: number

  // Transient visualization state — main-thread only, NOT serialized to ClimateBaked.
  // Workers never call windAtTime; fromBaked does not need these fields.
  /** Vortex table persisted from bakeWind for use in windAtTime. */
  private _vortices: Vortex[] = []
  /** Curl-noise function for windAtTime eddies (stream 202). */
  private _eddyNoise!: (x: number, y: number, z: number) => number
  /** Live transient knobs — merge-updated via setWindTransient(). */
  // Calmed defaults — the previous rates/amplitudes read as nervous jitter on the wind
  // arrows. Lower pulseRate/eddyTimeScale = slower animation; lower pulseDepth/eddyStrength
  // = gentler swings. (All live sliders, so they can be pushed back up.)
  private _windTransient: WindTransientParams = {
    driftSpeed:    0.010,
    pulseRate:     0.10,
    pulseDepth:    0.18,
    eddyStrength:  0.20,
    eddyScale:     6,
    eddyTimeScale: 0.12,
  }

  // Scratch fields for windAtTime — zero-alloc in the hot path.
  private readonly _wtBaseScratch: WindSample = { x: 0, y: 0, z: 0, speed: 0 }
  private readonly _wtEastScratch  = new Vector3()
  private readonly _wtNorthScratch = new Vector3()
  private readonly _wtDirScratch   = new Vector3()

  /** Baked moisture field — Float32Array, length 6·MOIST_RES². sampleSmooth(this). */
  private readonly moistureField: Float32Array

  /**
   * Baked wind fields — each Float32Array length 6·MOIST_RES².
   * windX/Y/Z: planet-local tangent-plane direction (unit vector, zero at poles).
   * windSpeed: dimensionless speed ∈ [0,1] (1 = peak trade-wind band-boundary shear).
   * Stream id 201 consumed for swirl jitter during bake.
   */
  private readonly windX: Float32Array
  private readonly windY: Float32Array
  private readonly windZ: Float32Array
  private readonly windSpeed: Float32Array

  // Zero-alloc scratch for sample() and the bake.
  private readonly _scratch = new Vector3()
  private readonly _windScratch = new Vector3()
  private readonly _eastScratch = new Vector3()
  private readonly _marchScratch = new Vector3()
  private readonly _texelDir = new Vector3()
  private readonly _polarAxis = new Vector3(0, 1, 0)

  constructor(opts: ClimateParams & ClimateDeps) {
    this.seed = opts.seed
    this.baseTemp = opts.baseTemp
    this.atmosphere = clamp01(opts.atmosphere)
    this.bandCount = Math.max(1, Math.round(opts.bandCount))
    this.axialTiltRad = opts.axialTiltRad
    this.redistribution = clamp01(opts.redistribution ?? 1)
    this.greenhouse = opts.greenhouse ?? 0
    this.lapseRate = opts.lapseRate ?? LAPSE_PER_HEIGHT
    this.swirlStrength = opts.swirlStrength ?? 0.6
    this.nHigh = opts.nHigh ?? 4
    this.nLow = opts.nLow ?? 3
    this.crossIsobarMax = opts.crossIsobarMax ?? 0.4
    this.sigmaBase = opts.sigmaBase ?? 0.18
    this.latSpread = opts.latSpread ?? 0.17
    this.retrograde = opts.retrograde ?? 1
    this.equatorTaperWidth = opts.equatorTaperWidth ?? 0.20

    // Stream 202: curl-noise eddy for windAtTime (visualization-only).
    this._eddyNoise = createNoise3D(deriveSeed(this.seed, 202))

    // Tilt inversion: above ≈54° poles receive more annual insolation than the
    // equator, flipping the gradient. Blend in smoothly to 75°.
    this.invertBlend = smoothstep(TILT_INVERT_LO, TILT_INVERT_HI, this.axialTiltRad)
    // Thick atmosphere shrinks the equator-pole delta (efficient heat transport).
    this.gradient = EQUATOR_POLE_DELTA * (1 - 0.7 * this.atmosphere)
    // No atmosphere → little lapse (no convecting air column to cool with height).
    this.lapseFactor = 0.3 + 0.7 * this.atmosphere
    // A thick atmosphere transports moisture efficiently: wetter overall and
    // softer bands. A thin atmosphere is arid with sharp, deep dry bands.
    //   moistGain (×MOIST_GAIN): 0.56 (thin) … 1.12 (thick) — overall wetness
    //     scale; the steep thin end keeps arid worlds arid (≈0.69 at atm 0.2).
    //   bandContrast: a power applied to the raw band wetness.
    //     A power > 1 SHARPENS the bands — it crushes moderate (mid-latitude,
    //     between-peak) moisture toward 0, leaving only the equatorial/60° peaks
    //     wet (the old 2.2−1.2·atm ≈ 1.48 at default did exactly this → all-desert
    //     mid-latitudes). We want BROAD wet bands at a normal atmosphere, so the
    //     default lands ≤ 1 (a mild power that keeps some contrast — deserts still
    //     form at descending bands — without flattening the wet zones to nothing):
    //       1.19 (thin, sharp deep dry bands) … 0.75 (thick, broad gentle bands);
    //       ≈0.97 at the default atmosphere 0.6.
    this.moistGain = MOIST_GAIN * (0.34 + 1.1 * this.atmosphere)
    this.bandContrast = 1.30 - 0.55 * this.atmosphere

    this.moistureField = this.bakeMoisture(opts.heightFn, opts.crustDistAt)

    const wind = this.bakeWind()
    this.windX = wind.windX
    this.windY = wind.windY
    this.windZ = wind.windZ
    this.windSpeed = wind.windSpeed
  }

  // -------------------------------------------------------------------------
  // Temperature — analytic, per-call, zero-alloc.
  // -------------------------------------------------------------------------

  /**
   * Climatological (latitude-average) insolation in [0,1] (1 = warmest band, 0 = coldest band).
   * Normal tilt: cosLat (warm equator). Extreme tilt: blends toward |dir.y|
   * (warm poles). `cosLat` is passed in to avoid recomputing the sqrt.
   * This is the single source of truth for the baked/deterministic temperature path.
   */
  private climatologicalInsolation(absY: number, cosLat: number): number {
    // normal = warm equator (cosLat); inverted = warm pole (|sin lat| = |dir.y|).
    return cosLat * (1 - this.invertBlend) + absY * this.invertBlend
  }

  /**
   * Instantaneous insolation at `dir` given a sun direction — simple Lambertian
   * dot product clamped to [0,1]. Both vectors must be unit length.
   * Used only by temperatureLiveAt; never reaches the baked/deterministic path.
   */
  private instantaneousInsolation(dir: Vector3, sunDir: Vector3): number {
    const d = dir.x * sunDir.x + dir.y * sunDir.y + dir.z * sunDir.z
    return d < 0 ? 0 : d
  }

  /**
   * Climatological temperature at a unit dir and normalized terrain height.
   * baseTemp is the area-mean; equator ≈ baseTemp + ~half gradient, pole ≈ baseTemp − ~half gradient.
   * greenhouse shifts the whole surface up; lapseRate replaces the old LAPSE_PER_HEIGHT constant.
   * Does NOT read redistribution or sunDir — stays deterministic for baked moisture + biomes.
   */
  private temperatureAt(dir: Vector3, height: number): number {
    const absY = Math.abs(dir.y)
    const cosLat = Math.sqrt(Math.max(0, 1 - dir.y * dir.y))
    const lat = this.climatologicalInsolation(absY, cosLat)
    const lapse = this.lapseRate * Math.max(0, height) * this.lapseFactor
    return this.baseTemp + this.greenhouse + this.gradient * (lat - LAT_MEAN) - lapse
  }

  /**
   * Live temperature at a unit dir and normalized terrain height, blended toward
   * the instantaneous (time-of-day) insolation by `redistribution`.
   * redistribution=1 → pure climatological (same as temperatureAt).
   * redistribution=0 → pure instantaneous (day/night extremes).
   * This is the canonical live model for future wind/CPU use. Zero-alloc.
   */
  temperatureLiveAt(dir: Vector3, height: number, sunDir: Vector3): number {
    const absY = Math.abs(dir.y)
    const cosLat = Math.sqrt(Math.max(0, 1 - dir.y * dir.y))
    const climatoIns = this.climatologicalInsolation(absY, cosLat)
    const instantIns = this.instantaneousInsolation(dir, sunDir)
    const r = this.redistribution
    const blendedIns = instantIns * (1 - r) + climatoIns * r
    const lapse = this.lapseRate * Math.max(0, height) * this.lapseFactor
    return this.baseTemp + this.greenhouse + this.gradient * (blendedIns - LAT_MEAN) - lapse
  }

  // -------------------------------------------------------------------------
  // Moisture bake
  // -------------------------------------------------------------------------

  /**
   * Circulation band wetness in [0,1]. `bandCount` wet/dry alternations from
   * equator to pole. cos(2·bandCount·latAngle): equator (latAngle 0) → 1 (wet);
   * first descending-dry trough at latAngle = π/(2·bandCount). For bandCount=3
   * that puts dry bands near ±30° and the poles, wet near the equator and ±60°.
   */
  private bandWetness(dirY: number): number {
    const latAngle = Math.asin(dirY < -1 ? -1 : dirY > 1 ? 1 : dirY)
    const raw = 0.5 + 0.5 * Math.cos(2 * this.bandCount * latAngle)
    // Raise the floor so descending bands aren't literally 0 — real "dry" bands
    // still get a little moisture; only the driest deep interiors become true
    // desert (via seaProximity/rainShadow). Floor 0.12, range 0.88.
    return BAND_FLOOR + (1 - BAND_FLOOR) * raw
  }

  /**
   * Index of the circulation band a latitude falls in (0 at equator, increasing
   * poleward). Used to pick the alternating zonal wind direction (sign = (−1)^band).
   */
  private bandIndex(dirY: number): number {
    const latAngle = Math.abs(Math.asin(dirY < -1 ? -1 : dirY > 1 ? 1 : dirY))
    // Each band spans π/(2·bandCount) of latitude angle.
    return Math.floor(latAngle / (Math.PI / (2 * this.bandCount)))
  }

  /**
   * Bake the moisture cube-map. For each texel:
   *   moisture = clamp01(bandWetness · seaProximity · rainShadow · MOIST_GAIN) · frozenAtten
   * then one cross-face 3×3-ish blur pass (moisture is smooth; this just removes
   * texel-grid stair-stepping at band edges).
   */
  private bakeMoisture(
    heightFn: (dir: Vector3, level: number) => number,
    crustDistAt: (dir: Vector3) => number,
  ): Float32Array {
    const res = MOIST_RES
    const field = new Float32Array(6 * res * res)
    const dir = this._texelDir
    const east = this._eastScratch
    const march = this._marchScratch
    const polar = this._polarAxis

    // Seed reserved for future jitter; touched so the stream id is deterministic
    // and documented even though the current band model is fully analytic.
    void deriveSeed(this.seed, 200)

    for (let face = 0; face < 6; face++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          texelToDir(face, x, y, res, dir)

          // 1. Banded circulation wetness. bandContrast deepens the dry bands
          //    under a thin atmosphere (raising the [0,1] band value to a power > 1
          //    pushes mid/low values toward 0, sharpening the descending-band troughs).
          const band = Math.pow(this.bandWetness(dir.y), this.bandContrast)

          // 2. Sea proximity — drier inland, but with a floor so deep interiors
          //    aren't uniformly bone-dry (coasts still wetter than interiors).
          const cd = crustDistAt(dir)
          const seaProx = MOIST_SEAPROX_FLOOR
            + (1 - MOIST_SEAPROX_FLOOR) * Math.exp(-Math.max(0, cd) / MOIST_INLAND_SCALE)

          // 3. Rain shadow — march UPWIND along this band's zonal wind.
          //    east tangent = normalize(polarAxis × dir); wind sign alternates per band.
          east.copy(polar).cross(dir)
          const eLen = east.length()
          let shadow = 1
          if (eLen > 1e-6) {
            east.multiplyScalar(1 / eLen)
            const windSign = (this.bandIndex(dir.y) & 1) === 0 ? 1 : -1
            // Upwind is opposite the wind travel direction.
            const upX = -windSign * east.x
            const upY = -windSign * east.y
            const upZ = -windSign * east.z
            const thisH = heightFn(dir, RAINSHADOW_LEVEL)
            let maxUpwind = thisH
            for (let s = 1; s <= UPWIND_STEPS; s++) {
              const d = s * UPWIND_STEP_RAD
              march.set(
                dir.x + upX * d,
                dir.y + upY * d,
                dir.z + upZ * d,
              ).normalize()
              const hUp = heightFn(march, RAINSHADOW_LEVEL)
              if (hUp > maxUpwind) maxUpwind = hUp
            }
            shadow = 1 - SHADOW_K * Math.max(0, maxUpwind - thisH)
            if (shadow < MIN_SHADOW) shadow = MIN_SHADOW
            else if (shadow > 1) shadow = 1
          }

          let m = clamp01(band * seaProx * shadow * this.moistGain)

          // 4. Frozen attenuation — very cold regions hold less liquid moisture.
          //    Sample temperature at this dir using the texel's coarse terrain height.
          const tHeight = heightFn(dir, RAINSHADOW_LEVEL)
          const temp = this.temperatureAt(dir, tHeight)
          if (temp < FROZEN_TEMP) {
            // 1 at FROZEN_TEMP, → 0.15 well below it (never fully zero — sublimation/snow still reads as some moisture).
            const f = clamp01((FROZEN_TEMP - temp) / FROZEN_WIDTH)
            m *= 1 - 0.85 * f
          }

          field[texelIndex(face, x, y, res)] = m
        }
      }
    }

    return this.blurField(field, res)
  }

  /**
   * One cross-face box blur (centre weight 4, four cardinal neighbours weight 1).
   * Uses neighborTexel for seam-correct sampling. Single pass — moisture is
   * already smooth, this only removes texel-grid stair-stepping.
   */
  private blurField(src: Float32Array, res: number): Float32Array {
    const out = new Float32Array(src.length)
    const nb = { face: 0, x: 0, y: 0 }
    for (let face = 0; face < 6; face++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          let sum = src[texelIndex(face, x, y, res)] * 4
          let w = 4
          // 4-neighbour cross.
          const offsets: Array<[-1 | 0 | 1, -1 | 0 | 1]> = [
            [-1, 0], [1, 0], [0, -1], [0, 1],
          ]
          for (const [dx, dy] of offsets) {
            neighborTexel(face, x, y, dx, dy, res, nb)
            sum += src[texelIndex(nb.face, nb.x, nb.y, res)]
            w += 1
          }
          out[texelIndex(face, x, y, res)] = sum / w
        }
      }
    }
    return out
  }

  // -------------------------------------------------------------------------
  // Wind bake
  // -------------------------------------------------------------------------

  /**
   * Bake tangent-plane wind direction (windX/Y/Z unit vectors) and dimensionless
   * wind speed into cubemap Float32Arrays at MOIST_RES.
   *
   * Model (3-cell general circulation + divergence-free vortex swirls):
   *
   *   1. TANGENT BASIS per texel: east = polarAxis × dir (normalised); north = dir × east;
   *      polarFade = |east before normalise| (→0 at poles); lat = asin(dir.y); absLat = |lat|.
   *      Guard: if eLen < 1e-5 → zero vector, speed 0.
   *
   *   2. ZONAL BASE: U_zonal = retrograde * (-cos(2·bandCount·absLat)); windZonal = U_zonal * east.
   *      Sign verification: equator absLat=0 → -cos(0)=-1 → retrograde(+1) gives -east
   *      (westward trades); first reversal at absLat = π/(2·bandCount) → eastward westerlies;
   *      second at absLat = π/bandCount → westward polar easterlies. ✓ Earth 3-cell model.
   *
   *   3. SWIRL VORTICES: generate from deriveSeed(seed,201), fixed iteration order
   *      hemisphere(outer) → family(HIGH then LOW) → index(inner) so adding more vortices
   *      never permutes earlier draws. HIGHs target latHigh = π/(4·bandCount);
   *      LOWs target latLow = 3π/(8·bandCount). Per slot draw u0..u3 from splitmix32:
   *        centerLat = targetLat * hemiSign + (u0-0.5)*2*latSpread
   *        centerLon = u1 * 2π
   *        p_k = unit vector at (centerLat, centerLon) about polarAxis
   *        sigma_k = sigmaBase * (0.8 + 0.4·u2) * sqrt(3/bandCount)
   *        A_k = (LOW?+1:-1) * (0.7 + 0.6·u3) * sign(centerLat)
   *      Velocity from streamfunction (curl of Gaussian → divergence-free):
   *        d = acos(clamp(dir·p_k,-1,1)); G = exp(-d²/(2σ²))
   *        t = dir - (dir·p_k)*p_k; if |t|>eps: dHat=normalize(t)
   *        grad_k = -(d/σ²) * G * A_k * dHat
   *        vel_k = dir × grad_k; windVortex += vel_k
   *
   *   4. EQUATOR TAPER: eqMask = smoothstep(0, sin(equatorTaperWidth), |dir.y|).
   *      Applied ONLY to the swirl contribution (vortices physically can't exist at the
   *      equator where Coriolis→0). Zonal base and Coriolis tilt are unchanged.
   *
   *   5. COMBINE: windVec = windZonal + swirlStrength * eqMask * windVortex.
   *
   *   6. CORIOLIS cross-isobar tilt: ξ = crossIsobarMax * (1-|sin(lat)|) * sign(lat);
   *      rotate windVec about dir by ξ via Rodrigues. (Max at equator, 0 at poles;
   *      rightward N, leftward S — Ekman-layer cross-isobar flow.)
   *
   *   7. NORMALISE & SPEED: windX/Y/Z = w'/|w'|;
   *      speedZonal = |sin(2·bandCount·lat)| * SHEAR_AMP * polarFade;
   *      speed = clamp01(speedZonal + swirlStrength * eqMask * |windVortex|).
   *
   *   8. BLUR: same cross-face box blur as moisture.
   *
   * Determinism: fixed-order vortex generation from deriveSeed(seed,201); equator taper
   * is a pure function of dir.y — no RNG, no reordering. fromBaked unchanged.
   * Stream id 201 consumed here (200 = moisture; 202 = transient eddy visualization).
   */
  private bakeWind(): Pick<ClimateBaked, 'windX' | 'windY' | 'windZ' | 'windSpeed'> {
    const res = MOIST_RES
    const n = 6 * res * res
    const rawX = new Float32Array(n)
    const rawY = new Float32Array(n)
    const rawZ = new Float32Array(n)
    const rawS = new Float32Array(n)

    const dir = this._texelDir
    const east = this._eastScratch
    const polar = this._polarAxis

    const { bandCount, swirlStrength, nHigh, nLow,
            crossIsobarMax, sigmaBase, latSpread, retrograde } = this

    // Band-boundary shear amplitude couples to ΔT (thin atm → stronger jets).
    const gradNorm = this.gradient / EQUATOR_POLE_DELTA  // 0.3 … 1.0
    const SHEAR_AMP = 0.5 + 0.5 * gradNorm              // 0.65 … 1.0

    // -----------------------------------------------------------------------
    // Build vortex table ONCE (fixed order: hemisphere → family → index).
    // Iterating hemisphere outer and family/index inner means adding vortices
    // (larger nHigh/nLow) always appends draws — never reorders earlier ones.
    // -----------------------------------------------------------------------

    // Pre-derive the stream-201 seed then walk a splitmix32 sequence.
    let rngState = deriveSeed(this.seed, 201)
    function nextU01(): number {
      rngState = splitmix32Step(rngState)
      return rngState / 0x100000000  // [0, 1)
    }

    // Target latitudes for HIGH and LOW centres (radians, magnitude only).
    // HIGH: first descending branch of the zonal cosine → ≈15–30° latitude.
    // LOW: trough between HIGH and the next ascending branch → ≈30–60° latitude.
    const latHigh = Math.PI / (4 * bandCount)          // ≈ π/(4·3) ≈ 15° for bc=3
    const latLow  = (3 * Math.PI) / (8 * bandCount)   // ≈ 3π/(8·3) ≈ 22.5° for bc=3

    // Each vortex: centre unit vector + spread + amplitude + transient fields.
    const vortices: Vortex[] = []

    const sigmaScale = sigmaBase * Math.sqrt(3 / bandCount)

    let vortexIdx = 0  // global index for golden-angle phase
    for (const hemiSign of [-1, 1]) {
      // HIGH family first, LOW second — fixed order within each hemisphere.
      for (const family of ['HIGH', 'LOW'] as const) {
        const count   = family === 'HIGH' ? nHigh : nLow
        const latTgt  = family === 'HIGH' ? latHigh : latLow
        const aSign   = family === 'LOW' ? 1 : -1   // LOW=+1 (CCW N), HIGH=-1 (CW N)

        for (let i = 0; i < count; i++) {
          const u0 = nextU01()
          const u1 = nextU01()
          const u2 = nextU01()
          const u3 = nextU01()

          const centerLat = latTgt * hemiSign + (u0 - 0.5) * 2 * latSpread
          const centerLon = u1 * 2 * Math.PI
          const cosLat = Math.cos(centerLat)
          const sinLat = Math.sin(centerLat)
          const cosLon = Math.cos(centerLon)
          const sinLon = Math.sin(centerLon)
          // Unit vector about polar axis Y; lon=0 faces +Z.
          //   x = cos(lat)*sin(lon), y = sin(lat), z = cos(lat)*cos(lon)
          const px = cosLat * sinLon
          const py = sinLat
          const pz = cosLat * cosLon

          const sigma = sigmaScale * (0.8 + 0.4 * u2)
          // Amplitude sign: aSign * (0.7 + 0.6·u3) * sign(centerLat)
          const latSign = centerLat >= 0 ? 1 : -1
          const A = aSign * (0.7 + 0.6 * u3) * latSign

          // Phase uses golden angle (no new nextU01() draw — preserves stream-201 count).
          const phase = vortexIdx * 2.39996

          vortices.push({
            px, py, pz, sigma, A,
            baseLon: centerLon,
            latC: centerLat,
            cosLatC: cosLat,
            sinLatC: sinLat,
            phase,
          })
          vortexIdx++
        }
      }
    }

    // Persist vortex table for windAtTime (visualization layer).
    this._vortices = vortices

    // Equator taper: precompute sin(equatorTaperWidth) once.
    const eqSinTaper = Math.max(1e-3, Math.sin(this.equatorTaperWidth))

    // -----------------------------------------------------------------------
    // Per-texel evaluation.
    // -----------------------------------------------------------------------
    for (let face = 0; face < 6; face++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          texelToDir(face, x, y, res, dir)

          // 1. Tangent basis.
          east.copy(polar).cross(dir)
          const eLen = east.length()
          const idx = texelIndex(face, x, y, res)

          if (eLen < 1e-5) {
            rawX[idx] = 0; rawY[idx] = 0; rawZ[idx] = 0; rawS[idx] = 0
            continue
          }
          east.multiplyScalar(1 / eLen)

          const lat = Math.asin(dir.y < -1 ? -1 : dir.y > 1 ? 1 : dir.y)
          const absLat = Math.abs(lat)
          const polarFade = eLen   // ≈1 at equator, →0 at poles

          // 2. Zonal base: U = retrograde * (-cos(2·bandCount·absLat))
          //    equator(absLat=0) → -1 → westward trades (correct for +retrograde)
          //    first reversal at absLat=π/(2·bandCount) → eastward westerlies ✓
          const U_zonal = retrograde * (-Math.cos(2 * bandCount * absLat))
          const zonX = U_zonal * east.x
          const zonY = U_zonal * east.y
          const zonZ = U_zonal * east.z

          // 3. Swirl vortices (curl of Gaussian streamfunction — divergence-free).
          let vortX = 0, vortY = 0, vortZ = 0

          for (const v of vortices) {
            const dot = dir.x * v.px + dir.y * v.py + dir.z * v.pz
            const dotC = dot < -1 ? -1 : dot > 1 ? 1 : dot
            const d = Math.acos(dotC)
            const sig2 = v.sigma * v.sigma
            const G = Math.exp(-(d * d) / (2 * sig2))

            // Tangent toward dir from centre: t = dir - dot * p_k
            const tx = dir.x - dotC * v.px
            const ty = dir.y - dotC * v.py
            const tz = dir.z - dotC * v.pz
            const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz)
            if (tLen < 1e-9) continue

            const dHx = tx / tLen
            const dHy = ty / tLen
            const dHz = tz / tLen

            // Gradient of Gaussian streamfunction (in the direction toward centre).
            const gScale = -(d / sig2) * G * v.A
            const gx = gScale * dHx
            const gy = gScale * dHy
            const gz = gScale * dHz

            // vel = dir × grad (curl → tangent-plane, divergence-free).
            vortX += dir.y * gz - dir.z * gy
            vortY += dir.z * gx - dir.x * gz
            vortZ += dir.x * gy - dir.y * gx
          }

          // 4. Equator taper — damp vortex swirl near equator (Coriolis→0 there).
          const eqMask = smoothstep(0, eqSinTaper, Math.abs(dir.y))

          // 5. Combine.
          let wx = zonX + swirlStrength * eqMask * vortX
          let wy = zonY + swirlStrength * eqMask * vortY
          let wz = zonZ + swirlStrength * eqMask * vortZ

          // 6. Coriolis/friction cross-isobar tilt: rotate about dir by ξ (Rodrigues).
          //    INTENTIONALLY max at the equator, →0 at the poles — this is the cross-isobar
          //    *deflection angle*, not the Coriolis force magnitude. Per the Guldberg-Mohn /
          //    Ekman surface balance tan(α)=friction/Coriolis, so α grows where Coriolis (∝sinφ)
          //    is WEAKEST (the tropics) and →0 where geostrophic balance dominates (the poles).
          //    So ξ=0 at the poles = stays geostrophic there. (Not a bug — don't "fix" the sign.)
          const latSign = lat >= 0 ? 1 : -1
          const xi = crossIsobarMax * (1 - Math.abs(Math.sin(lat))) * latSign
          const cosXi = Math.cos(xi)
          const sinXi = Math.sin(xi)
          const dDotW = dir.x * wx + dir.y * wy + dir.z * wz
          // k × w (k = dir)
          const crossX = dir.y * wz - dir.z * wy
          const crossY = dir.z * wx - dir.x * wz
          const crossZ = dir.x * wy - dir.y * wx
          wx = wx * cosXi + crossX * sinXi + dir.x * dDotW * (1 - cosXi)
          wy = wy * cosXi + crossY * sinXi + dir.y * dDotW * (1 - cosXi)
          wz = wz * cosXi + crossZ * sinXi + dir.z * dDotW * (1 - cosXi)

          // 7. Normalise direction.
          const wLen = Math.sqrt(wx * wx + wy * wy + wz * wz)
          const inv = wLen > 1e-9 ? 1 / wLen : 0
          rawX[idx] = wx * inv
          rawY[idx] = wy * inv
          rawZ[idx] = wz * inv

          // Speed: shear from band boundaries + vortex contribution (tapered at equator).
          const speedZonal = Math.abs(Math.sin(2 * bandCount * lat)) * SHEAR_AMP * polarFade
          const vortMag = Math.sqrt(vortX * vortX + vortY * vortY + vortZ * vortZ)
          rawS[idx] = clamp01(speedZonal + swirlStrength * eqMask * vortMag)
        }
      }
    }

    // 8. Box-blur each component for seam-smooth sampling (same pass as moisture).
    return {
      windX: this.blurField(rawX, res),
      windY: this.blurField(rawY, res),
      windZ: this.blurField(rawZ, res),
      windSpeed: this.blurField(rawS, res),
    }
  }

  // -------------------------------------------------------------------------
  // Sampling
  // -------------------------------------------------------------------------

  /**
   * Sample temperature (analytic) + moisture (baked field lookup) at a unit
   * planet-local direction and normalized terrain height. Zero-alloc: writes into
   * `out` and returns it.
   */
  sample(dir: Vector3, height: number, out: ClimateSample): ClimateSample {
    out.temperature = this.temperatureAt(dir, height)
    out.moisture = sampleSmooth(this.moistureField, dir, MOIST_RES, this._scratch)
    if (out.moisture < 0) out.moisture = 0
    else if (out.moisture > 1) out.moisture = 1
    return out
  }

  /**
   * Sample the baked wind at a unit planet-local direction. Writes into `out`
   * and returns it. Zero-alloc (reuses `_windScratch`).
   *
   * `out.x/y/z` is the planet-local tangent-plane wind direction (approximately
   * unit length; may be a near-zero vector at poles). `out.speed` ∈ [0,1].
   *
   * Stream 201 was consumed during the bake for the swirl jitter; sampling is a
   * pure bilinear cubemap lookup — deterministic and worker-safe.
   */
  windAt(dir: Vector3, out: WindSample): WindSample {
    out.x = sampleSmooth(this.windX, dir, MOIST_RES, this._windScratch)
    out.y = sampleSmooth(this.windY, dir, MOIST_RES, this._windScratch)
    out.z = sampleSmooth(this.windZ, dir, MOIST_RES, this._windScratch)
    out.speed = sampleSmooth(this.windSpeed, dir, MOIST_RES, this._windScratch)
    if (out.speed < 0) out.speed = 0
    else if (out.speed > 1) out.speed = 1
    return out
  }

  /**
   * Sample the baked wind speed only at a unit planet-local direction.
   * Returns a value ∈ [0,1]. Use `windAt` when you also need the direction.
   */
  windSpeedAt(dir: Vector3): number {
    const s = sampleSmooth(this.windSpeed, dir, MOIST_RES, this._windScratch)
    return s < 0 ? 0 : s > 1 ? 1 : s
  }

  // -------------------------------------------------------------------------
  // Transient visualization layer
  // -------------------------------------------------------------------------

  /**
   * Merge `p` into the live transient knobs. Takes effect immediately on the next
   * windAtTime() call — no rebake needed.
   */
  setWindTransient(p: Partial<WindTransientParams>): void {
    Object.assign(this._windTransient, p)
  }

  /**
   * Animated wind sample at `dir` and time `t` (seconds). Visualization-only —
   * adds drifting/pulsing vortices and curl-noise eddies on top of the static
   * baked field. ZERO-alloc: reuses instance scratch fields.
   *
   * NOT called by terrainSampler/workers — determinism is unaffected.
   * windAt/windSpeedAt/toBaked/fromBaked are unchanged.
   */
  windAtTime(dir: Vector3, t: number, out: WindSample): WindSample {
    const { driftSpeed, pulseRate, pulseDepth, eddyStrength, eddyScale, eddyTimeScale } =
      this._windTransient

    // a. Static mean wind as velocity vector.
    this.windAt(dir, this._wtBaseScratch)
    const staticSpeed = this._wtBaseScratch.speed
    let wx = this._wtBaseScratch.x * staticSpeed
    let wy = this._wtBaseScratch.y * staticSpeed
    let wz = this._wtBaseScratch.z * staticSpeed

    // b. Drifting / pulsing vortices.
    // Equator taper (live read of equatorTaperWidth).
    const eqSinTaperLive = Math.max(1e-3, Math.sin(this.equatorTaperWidth))
    const eqMaskLive = smoothstep(0, eqSinTaperLive, Math.abs(dir.y))

    let vortWx = 0, vortWy = 0, vortWz = 0
    const { bandCount } = this
    for (const v of this._vortices) {
      // Drift: each vortex centre drifts along its local zonal belt
      // (westward in trade belts, eastward in the westerlies).
      const driftSign = -Math.cos(2 * bandCount * Math.abs(v.latC))
      const lon = v.baseLon + driftSpeed * driftSign * t
      const px = v.cosLatC * Math.sin(lon)
      const py = v.sinLatC
      const pz = v.cosLatC * Math.cos(lon)

      // Pulse: amplitude oscillates.
      const A = v.A * (1 + pulseDepth * Math.sin(pulseRate * t + v.phase))

      // Gaussian-streamfunction curl (same math as bakeWind).
      const dot = dir.x * px + dir.y * py + dir.z * pz
      const dotC = dot < -1 ? -1 : dot > 1 ? 1 : dot
      const d = Math.acos(dotC)
      const sig2 = v.sigma * v.sigma
      const G = Math.exp(-(d * d) / (2 * sig2))

      const tx = dir.x - dotC * px
      const ty = dir.y - dotC * py
      const tz = dir.z - dotC * pz
      const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz)
      if (tLen < 1e-9) continue

      const dHx = tx / tLen
      const dHy = ty / tLen
      const dHz = tz / tLen

      const gScale = -(d / sig2) * G * A
      const gx = gScale * dHx
      const gy = gScale * dHy
      const gz = gScale * dHz

      vortWx += dir.y * gz - dir.z * gy
      vortWy += dir.z * gx - dir.x * gz
      vortWz += dir.x * gy - dir.y * gx
    }

    wx += eqMaskLive * vortWx
    wy += eqMaskLive * vortWy
    wz += eqMaskLive * vortWz

    // c. Curl-noise eddies (divergence-free surface flow).
    // Tangent basis: east = normalize(polarAxis × dir), north = dir × east.
    const polar = this._polarAxis
    const east  = this._wtEastScratch
    east.set(polar.y * dir.z - polar.z * dir.y,
             polar.z * dir.x - polar.x * dir.z,
             polar.x * dir.y - polar.y * dir.x)
    const eLen = east.length()
    if (eLen > 1e-6) {
      east.multiplyScalar(1 / eLen)
      const north = this._wtNorthScratch
      // north = dir × east
      north.set(
        dir.y * east.z - dir.z * east.y,
        dir.z * east.x - dir.x * east.z,
        dir.x * east.y - dir.y * east.x,
      )

      // Streamfunction Ψ(dir, t): 2-octave curl noise evaluated at dir (unit sphere point).
      // Using central finite differences for the surface gradient (h = 1e-3 rad).
      const h = 1e-3
      const N = this._eddyNoise

      // Sample Ψ in east direction.
      const dEast = this._wtDirScratch
      dEast.set(dir.x + h * east.x, dir.y + h * east.y, dir.z + h * east.z)
      const psiEp =
        N(eddyScale * dEast.x + eddyTimeScale * t, eddyScale * dEast.y, eddyScale * dEast.z) +
        0.5 * N(2 * eddyScale * dEast.x - eddyTimeScale * t, 2 * eddyScale * dEast.y, 2 * eddyScale * dEast.z)
      dEast.set(dir.x - h * east.x, dir.y - h * east.y, dir.z - h * east.z)
      const psiEm =
        N(eddyScale * dEast.x + eddyTimeScale * t, eddyScale * dEast.y, eddyScale * dEast.z) +
        0.5 * N(2 * eddyScale * dEast.x - eddyTimeScale * t, 2 * eddyScale * dEast.y, 2 * eddyScale * dEast.z)
      const psiE = (psiEp - psiEm) / (2 * h)

      // Sample Ψ in north direction.
      const north0 = north
      dEast.set(dir.x + h * north0.x, dir.y + h * north0.y, dir.z + h * north0.z)
      const psiNp =
        N(eddyScale * dEast.x + eddyTimeScale * t, eddyScale * dEast.y, eddyScale * dEast.z) +
        0.5 * N(2 * eddyScale * dEast.x - eddyTimeScale * t, 2 * eddyScale * dEast.y, 2 * eddyScale * dEast.z)
      dEast.set(dir.x - h * north0.x, dir.y - h * north0.y, dir.z - h * north0.z)
      const psiNm =
        N(eddyScale * dEast.x + eddyTimeScale * t, eddyScale * dEast.y, eddyScale * dEast.z) +
        0.5 * N(2 * eddyScale * dEast.x - eddyTimeScale * t, 2 * eddyScale * dEast.y, 2 * eddyScale * dEast.z)
      const psiN = (psiNp - psiNm) / (2 * h)

      // gradPsi = psiE * east + psiN * north (tangent plane gradient).
      // eddyVel = dir × gradPsi (divergence-free + tangent).
      const gpx = psiE * east.x + psiN * north0.x
      const gpy = psiE * east.y + psiN * north0.y
      const gpz = psiE * east.z + psiN * north0.z

      const eddyVx = dir.y * gpz - dir.z * gpy
      const eddyVy = dir.z * gpx - dir.x * gpz
      const eddyVz = dir.x * gpy - dir.y * gpx

      // Latitude weighting: stronger at mid-latitudes, tapered at equator and poles.
      // eqMaskLive is bit-identical to the equator smoothstep above — reuse it.
      const stormWeight = eqMaskLive * (1 - dir.y * dir.y * dir.y * dir.y)

      wx += eddyStrength * stormWeight * eddyVx
      wy += eddyStrength * stormWeight * eddyVy
      wz += eddyStrength * stormWeight * eddyVz
    }

    // d. Project out radial component (keep tangential only).
    const radial = wx * dir.x + wy * dir.y + wz * dir.z
    wx -= radial * dir.x
    wy -= radial * dir.y
    wz -= radial * dir.z

    // Normalise and fill out.
    const wMag = Math.sqrt(wx * wx + wy * wy + wz * wz)
    if (wMag > 1e-9) {
      out.x = wx / wMag
      out.y = wy / wMag
      out.z = wz / wMag
    } else {
      // Fall back to static direction.
      out.x = this._wtBaseScratch.x
      out.y = this._wtBaseScratch.y
      out.z = this._wtBaseScratch.z
    }
    out.speed = clamp01(wMag)
    return out
  }

  // -------------------------------------------------------------------------
  // Serialisation helpers
  // -------------------------------------------------------------------------

  /**
   * Snapshot this Climate's baked state so a Web Worker can reconstruct a
   * sampler-only instance without re-baking.
   *
   * `moistureField` is returned BY REFERENCE — transfer its `.buffer` across
   * the Worker boundary instead of copying it when possible.
   */
  toBaked(): ClimateBaked {
    return {
      moistRes: MOIST_RES,
      moistureField: this.moistureField,
      windX: this.windX,
      windY: this.windY,
      windZ: this.windZ,
      windSpeed: this.windSpeed,
      baseTemp: this.baseTemp,
      atmosphere: this.atmosphere,
      bandCount: this.bandCount,
      axialTiltRad: this.axialTiltRad,
      redistribution: this.redistribution,
      greenhouse: this.greenhouse,
      lapseRate: this.lapseRate,
    }
  }

  /**
   * Reconstruct a sampler-only Climate from a baked snapshot.
   * Does NOT call `bakeMoisture` — the field from `b` is used directly.
   * `sample()` works identically to a normally constructed instance.
   */
  static fromBaked(b: ClimateBaked): Climate {
    // Bypass the constructor so bakeMoisture is never called.
    const c = Object.create(Climate.prototype) as Climate

    // Field initialisers do not run under Object.create, so we must set every
    // private field explicitly.

    // --- identity scalars (mirrors constructor lines 188-191) ---
    ;(c as unknown as Record<string, unknown>)['seed'] = 0 // not used post-bake
    ;(c as unknown as Record<string, unknown>)['baseTemp'] = b.baseTemp
    ;(c as unknown as Record<string, unknown>)['atmosphere'] = b.atmosphere
    ;(c as unknown as Record<string, unknown>)['bandCount'] = b.bandCount
    ;(c as unknown as Record<string, unknown>)['axialTiltRad'] = b.axialTiltRad
    ;(c as unknown as Record<string, unknown>)['redistribution'] = b.redistribution ?? 1
    ;(c as unknown as Record<string, unknown>)['greenhouse'] = b.greenhouse ?? 0
    ;(c as unknown as Record<string, unknown>)['lapseRate'] = b.lapseRate ?? LAPSE_PER_HEIGHT

    // --- sampler derived scalars (same formulas as constructor lines 195-214) ---
    const atm = b.atmosphere
    ;(c as unknown as Record<string, unknown>)['invertBlend'] =
      smoothstep(TILT_INVERT_LO, TILT_INVERT_HI, b.axialTiltRad)
    ;(c as unknown as Record<string, unknown>)['gradient'] =
      EQUATOR_POLE_DELTA * (1 - 0.7 * atm)
    ;(c as unknown as Record<string, unknown>)['lapseFactor'] =
      0.3 + 0.7 * atm
    ;(c as unknown as Record<string, unknown>)['moistGain'] =
      MOIST_GAIN * (0.34 + 1.1 * atm)
    ;(c as unknown as Record<string, unknown>)['bandContrast'] =
      1.30 - 0.55 * atm

    // --- baked fields ---
    ;(c as unknown as Record<string, unknown>)['moistureField'] = b.moistureField
    ;(c as unknown as Record<string, unknown>)['windX'] = b.windX
    ;(c as unknown as Record<string, unknown>)['windY'] = b.windY
    ;(c as unknown as Record<string, unknown>)['windZ'] = b.windZ
    ;(c as unknown as Record<string, unknown>)['windSpeed'] = b.windSpeed

    // --- zero-alloc scratch (field initialisers don't run; must init manually) ---
    ;(c as unknown as Record<string, unknown>)['_scratch'] = new Vector3()
    ;(c as unknown as Record<string, unknown>)['_windScratch'] = new Vector3()
    ;(c as unknown as Record<string, unknown>)['_eastScratch'] = new Vector3()
    ;(c as unknown as Record<string, unknown>)['_marchScratch'] = new Vector3()
    ;(c as unknown as Record<string, unknown>)['_texelDir'] = new Vector3()
    ;(c as unknown as Record<string, unknown>)['_polarAxis'] = new Vector3(0, 1, 0)

    return c
  }

  dispose(): void {
    // No GPU resources; the typed array is released with the instance.
  }
}
