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
 */

import { Vector3 } from 'three'
import {
  dirToTexel,
  texelToDir,
  texelIndex,
  neighborTexel,
  sampleSmooth,
} from './cubemap'

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
}

export interface ClimateSample {
  /** °C-ish. */
  temperature: number
  /** 0..1. */
  moisture: number
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
  /** Scalar params needed to recompute the sampler-side derived values. */
  baseTemp: number
  atmosphere: number
  bandCount: number
  axialTiltRad: number
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

  /** Baked moisture field — Float32Array, length 6·MOIST_RES². sampleSmooth(this). */
  private readonly moistureField: Float32Array

  // Zero-alloc scratch for sample() and the bake.
  private readonly _scratch = new Vector3()
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
  }

  // -------------------------------------------------------------------------
  // Temperature — analytic, per-call, zero-alloc.
  // -------------------------------------------------------------------------

  /**
   * Latitude insolation profile in [0,1] (1 = warmest band, 0 = coldest band).
   * Normal tilt: cosLat (warm equator). Extreme tilt: blends toward |dir.y|
   * (warm poles). `cosLat` is passed in to avoid recomputing the sqrt.
   */
  private latProfile(absY: number, cosLat: number): number {
    // normal = warm equator (cosLat); inverted = warm pole (|sin lat| = |dir.y|).
    return cosLat * (1 - this.invertBlend) + absY * this.invertBlend
  }

  /**
   * Temperature at a unit dir and normalized terrain height.
   * baseTemp is the area-mean; equator ≈ baseTemp + ~half gradient, pole ≈ baseTemp − ~half gradient.
   */
  private temperatureAt(dir: Vector3, height: number): number {
    const absY = Math.abs(dir.y)
    const cosLat = Math.sqrt(Math.max(0, 1 - dir.y * dir.y))
    const lat = this.latProfile(absY, cosLat)
    const lapse = LAPSE_PER_HEIGHT * Math.max(0, height) * this.lapseFactor
    return this.baseTemp + this.gradient * (lat - LAT_MEAN) - lapse
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
      baseTemp: this.baseTemp,
      atmosphere: this.atmosphere,
      bandCount: this.bandCount,
      axialTiltRad: this.axialTiltRad,
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

    // --- baked field ---
    ;(c as unknown as Record<string, unknown>)['moistureField'] = b.moistureField

    // --- zero-alloc scratch (field initialisers don't run; must init manually) ---
    ;(c as unknown as Record<string, unknown>)['_scratch'] = new Vector3()
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
