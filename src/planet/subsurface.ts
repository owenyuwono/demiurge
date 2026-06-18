/**
 * Subsurface geology system.
 *
 * Bakes once on the main thread and ships to Web Workers via toBaked()/fromBaked(),
 * mirroring the erosion.ts pattern exactly.
 *
 * Fields baked:
 *   waterTable      — elevation of the local water table (normalized height units)
 *   porosity        — rock porosity 0..1 (higher = more porous/sedimentary)
 *   magmaTop        — top of the magma/partial-melt zone (normalized height)
 *   geothermalGrad  — geothermal gradient intensity 0..1
 *   permafrostTop   — top of permafrost layer (SUB_SENTINEL where absent)
 *   permafrostBot   — bottom of permafrost layer (SUB_SENTINEL where absent)
 *   permafrostMask  — clean 0/1 presence mask (bilinear-safe; no sentinels)
 *   hydrocarbon     — hydrocarbon potential 0..1
 *
 * Determinism: all randomness derived from seed via deriveSeed (streams 30/31,
 * clear of tectonics 0–18, climate 200, erosion 9/10). No Math.random / Date.now.
 */

import { Vector3 } from 'three'
import {
  texelToDir,
  texelIndex,
  neighborTexel,
  sampleSmooth,
} from './cubemap'
import { Climate, ClimateSample } from './climate'
import { Erosion } from './erosion'
import { Tectonics, TectonicQuery } from './tectonics'
import { DerivedInterior } from './interior'
import { createNoise3D } from './noise'

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/** Cubemap resolution for the bake. */
const SUB_RES = 256

/** Sentinel value for absent permafrost layers. */
const SUB_SENTINEL = -999

/** LOD level at which heightFn is sampled during bake. */
const SUB_BAKE_LEVEL = 5

// Stream ids — clear of tectonics (0–18), climate (200), erosion (9/10)
const SUB_STREAM_WARP = 30
const SUB_STREAM_ORE  = 31

// ---------------------------------------------------------------------------
// Water table
// ---------------------------------------------------------------------------

/** Maximum depth from surface to water table (normalized height units). */
const WT_MAX_DEPTH = 0.04

// ---------------------------------------------------------------------------
// Magma / geothermal
// ---------------------------------------------------------------------------

/** Boundary proximity falloff length scale (radians). */
const MAGMA_BOUNDARY_SCALE = 0.15

/** Volcano reference elevation for hotspot proximity (normalized). */
const VOLC_REF = 0.3

/** Maximum normalized depth of magma top below surface. */
const MAGMA_MAX_DEPTH = 0.35

/** Minimum normalized depth of magma top (clamp floor). */
const MAGMA_MIN_DEPTH = 0.02

/**
 * Per-regime magma lift offset (added to regimeLift subtraction).
 * Maps the Regime string union from interior.ts.
 */
const REGIME_MAGMA: Record<string, number> = {
  'dead-lid':    0.00,
  'stagnant-lid': 0.02,
  'plated':      0.05,
  'heat-pipe':   0.25,
  'magma-ocean': 0.50,
}

/** Geothermal gradient range (stored as geoScale scalar). */
const GEO_LO = 15    // °C / km equivalent (conceptual; used as multiplier)
const GEO_HI = 120

// ---------------------------------------------------------------------------
// Permafrost
// ---------------------------------------------------------------------------

/** Temperature threshold below which permafrost can form (°C). */
const PERMA_TEMP  = 0.0

/** Temperature range over which permafrost depth ramps in (°C). */
const PERMA_WIDTH = 20.0

/** Shallowest permafrost top offset from surface. */
const PERMA_TOP_DEPTH = 0.003

/** Maximum permafrost thickness (normalized height). */
const PERMA_MAX_DEPTH = 0.12

// ---------------------------------------------------------------------------
// Hydrocarbon
// ---------------------------------------------------------------------------

/** Geothermal gradient window for oil generation (maturation window). */
const GEO_OIL_LO = 0.25
const GEO_OIL_HI = 0.55

// ---------------------------------------------------------------------------
// Domain-warp constants (same pattern as erosion.ts)
// ---------------------------------------------------------------------------

const WARP_AMP  = 0.025
const WARP_FREQ = 1.6
const WARP_OCT  = 2

// Ore analytic query constants
const ORE_BAND_W      = 0.06   // rad — fault proximity falloff
const ORE_FREQ        = 12.0   // 3D noise frequency for vein structure
const ORE_VEIN_THRESH = 0.55   // ridged noise threshold for discrete veins
// Bell window: ore peaks in a shallow band just above the magma top.
// aboveMagma = elev - magmaTopAt(dir); bell peaks at ORE_BAND_CENTER,
// half-width ORE_BAND_HW; evaluates to 0 below magma (aboveMagma <= 0).
const ORE_BAND_CENTER = 0.03   // normalized elev above magma top at peak ore density
const ORE_BAND_HW     = 0.05   // half-width of the bell window

type Noise3DFn = (x: number, y: number, z: number) => number

// ---------------------------------------------------------------------------
// Local deterministic PRNG (copied verbatim from erosion.ts / climate.ts)
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
 * Subsurface uses:
 *   stream 30 = query-time domain-warp noise
 *   stream 31 = ore-vein analytic noise
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

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Ridged noise — maps signed noise to [0,1] with a ridge at 0. */
function ridgedNoise(noise: Noise3DFn, x: number, y: number, z: number): number {
  return 1 - Math.abs(noise(x, y, z))
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** All data needed to reconstruct a sampler-only Subsurface in a Web Worker. */
export interface SubsurfaceBaked {
  res:            number
  // 8 fields × 6·res² elements
  waterTable:     Float32Array
  porosity:       Float32Array
  magmaTop:       Float32Array
  geothermalGrad: Float32Array
  permafrostTop:  Float32Array
  permafrostBot:  Float32Array
  permafrostMask: Float32Array
  hydrocarbon:    Float32Array
  // Scalars
  porosityBase: number
  geoScale:     number
  warpSeed:     number
  oreSeed:      number
}

/** Fully-populated stratigraphy column at a single surface point. */
export interface SubsurfaceColumn {
  waterTable:    number
  magmaTop:      number
  geothermal:    number
  permafrostTop: number  // SUB_SENTINEL where absent
  permafrostBot: number  // SUB_SENTINEL where absent
  porosity:      number
  hydrocarbon:   number
  /** True when waterTable >= terrainH (spring emerges at surface). */
  springEmerges: boolean
}

export interface SubsurfaceOpts {
  seed:       number
  heightFn:   (dir: Vector3, level: number) => number
  climate:    Climate
  erosion:    Erosion
  tectonics:  Tectonics
  derived:    DerivedInterior
  /** composition root 0..1 (icy/volatile → rocky/metallic); passed separately
   *  because DerivedInterior does not expose the InteriorParams root directly. */
  composition: number
  seaLevel:   number
  res?:       number
}

// ---------------------------------------------------------------------------
// Subsurface
// ---------------------------------------------------------------------------

export class Subsurface {
  private readonly res:            number
  private readonly waterTable:     Float32Array
  private readonly porosity:       Float32Array
  private readonly magmaTop:       Float32Array
  private readonly geothermalGrad: Float32Array
  private readonly permafrostTop:  Float32Array
  private readonly permafrostBot:  Float32Array
  private readonly permafrostMask: Float32Array
  private readonly hydrocarbon:    Float32Array

  private readonly porosityBase: number
  private readonly geoScale:     number

  // Domain-warp noise (query-time; breaks cubemap grid alignment)
  private _warpSeed: number = 0
  private _warpNoise!: Noise3DFn

  // Ore vein noise (query-time analytic)
  private _oreSeed: number = 0
  private _oreNoise!: Noise3DFn

  // Zero-alloc scratch for query methods
  private readonly _scratch  = new Vector3()
  private readonly _scratch2 = new Vector3()   // warp output

  // Tectonics handle (kept for oreAt analytic path)
  private _tectonics: Tectonics | null = null

  // Scratch for oreAt's tectonics query — must be manually re-inited in fromBaked/identity
  private _oreQueryScratch: TectonicQuery = {
    plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0,
    shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0,
  }

  constructor(opts: SubsurfaceOpts) {
    const res = opts.res ?? SUB_RES
    this.res = res

    // Derive noise seeds for this system (streams 30 and 31)
    this._warpSeed  = deriveSeed(opts.seed, SUB_STREAM_WARP)
    this._oreSeed   = deriveSeed(opts.seed, SUB_STREAM_ORE)
    this._warpNoise = createNoise3D(this._warpSeed)
    this._oreNoise  = createNoise3D(this._oreSeed)
    this._tectonics = opts.tectonics

    const N = 6 * res * res

    // Bake-time scratch objects (not needed at query time)
    const _dir           = new Vector3()
    const _climateSample: ClimateSample = { temperature: 0, moisture: 0 }
    const _tectScratch: TectonicQuery = {
      plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0,
      shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0,
    }

    // Derived interior parameters used in bake
    const { internalHeat, vigor, regime, oceanCoverage } = opts.derived
    const regimeLift = REGIME_MAGMA[regime] ?? 0

    // porosityBase: lerp between sedimentary (high porosity) and rocky/metallic (low porosity)
    // composition: 0 = icy/volatile (more porous sediment-like), 1 = rocky/metallic (denser)
    const porosityBase = lerp(0.30, 0.08, opts.composition)
    this.porosityBase = porosityBase

    // geoScale scalar: lerp between a cool and hot geothermal regime
    const geoScale = lerp(GEO_LO, GEO_HI, internalHeat)
    this.geoScale = geoScale

    // waterBudget proxy: use oceanCoverage as a stand-in for total water availability.
    // oceanCoverage (0..1) is the best available scalar for planetary water supply
    // because DerivedInterior does not retain the raw waterBudget root.
    const waterBudget = oceanCoverage

    // Allocate output arrays
    const waterTableArr     = new Float32Array(N)
    const porosityArr       = new Float32Array(N)
    const magmaTopArr       = new Float32Array(N)
    const geothermalGradArr = new Float32Array(N)
    const permafrostTopArr  = new Float32Array(N)
    const permafrostBotArr  = new Float32Array(N)
    const permafrostMaskArr = new Float32Array(N)
    const hydrocarbonArr    = new Float32Array(N)

    // -----------------------------------------------------------------------
    // Main bake loop
    // -----------------------------------------------------------------------

    for (let face = 0; face < 6; face++) {
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          const c = texelIndex(face, x, y, res)
          texelToDir(face, x, y, res, _dir)

          // Sample terrain height and climate at this direction
          const h    = opts.heightFn(_dir, SUB_BAKE_LEVEL)
          opts.climate.sample(_dir, h, _climateSample)
          const temp     = _climateSample.temperature
          const moisture = _climateSample.moisture

          // Tectonics query for boundary distances and crust info
          opts.tectonics.query(_dir, _tectScratch)
          const boundaryDist = _tectScratch.boundaryDist

          // ----------------------------------------------------------------
          // Porosity
          // ----------------------------------------------------------------
          // Crust sediment contribution: shallow crustDist → more sediment deposition
          const sedimentTerm = 0.08 * smoothstep(0.10, 0.0, Math.abs(_tectScratch.crustDist))
          const por = clamp01(porosityBase + sedimentTerm)
          porosityArr[c] = por

          // ----------------------------------------------------------------
          // Water table
          // ----------------------------------------------------------------
          // supply: scaled by moisture; more moisture → shallower water table
          const supply = waterBudget * (0.4 + 0.6 * moisture)
          const depthToWater = WT_MAX_DEPTH * (1 - supply) * (1 - 0.5 * por)
          let wt = Math.max(opts.seaLevel, h - depthToWater)

          // Valley/lake pull: streams and lake basins elevate the local water table
          const wetPull = Math.max(
            smoothstep(0.5, 0.9, opts.erosion.accAt(_dir)),
            opts.erosion.lakeMaskAt(_dir),
          )
          wt = lerp(wt, Math.max(wt, h), wetPull)
          waterTableArr[c] = wt

          // ----------------------------------------------------------------
          // Magma top / geothermal gradient
          // ----------------------------------------------------------------
          const boundary = Math.max(
            Math.exp(-boundaryDist / MAGMA_BOUNDARY_SCALE),
            0,  // volcano proxy is analytic — skip in bake to avoid per-cell cost
          )
          // volcanoElevation is expensive; in the bake we omit it and rely on
          // the boundary-distance term to capture arc-front heat. The full
          // analytic proxy is available at query time via oreAt if needed.
          const hotspot = 0.0  // set to 0 in bake; hotspot nodes are sparse

          // Magma depth decreases with internal heat, vigor, boundary proximity
          let magmaDepth = MAGMA_MAX_DEPTH
            * (1 - 0.6 * internalHeat - 0.3 * vigor)
            * (1 - 0.7 * boundary)
            * (1 - hotspot)
          // Clamp to minimum safe depth then subtract regime contribution
          if (magmaDepth < MAGMA_MIN_DEPTH) magmaDepth = MAGMA_MIN_DEPTH
          magmaDepth -= regimeLift
          if (magmaDepth < MAGMA_MIN_DEPTH) magmaDepth = MAGMA_MIN_DEPTH

          magmaTopArr[c] = h - magmaDepth

          const geothermal = clamp01(
            0.2 + 0.6 * internalHeat + 0.4 * boundary + 0.5 * hotspot,
          )
          geothermalGradArr[c] = geothermal

          // ----------------------------------------------------------------
          // Permafrost
          // ----------------------------------------------------------------
          if (temp < PERMA_TEMP) {
            const depthFactor = clamp01((PERMA_TEMP - temp) / PERMA_WIDTH)
            const pTop = h - PERMA_TOP_DEPTH * (1 - depthFactor)
            const pBot = Math.max(
              h - PERMA_MAX_DEPTH * depthFactor,
              magmaTopArr[c] + 0.02,
            )
            permafrostTopArr[c]  = pTop
            permafrostBotArr[c]  = pBot
            permafrostMaskArr[c] = 1
          } else {
            permafrostTopArr[c]  = SUB_SENTINEL
            permafrostBotArr[c]  = SUB_SENTINEL
            permafrostMaskArr[c] = 0
          }

          // ----------------------------------------------------------------
          // Hydrocarbon
          // ----------------------------------------------------------------
          // Basin proxy: deltas and lake basins accumulate organic sediment
          const basin = Math.max(
            smoothstep(0, 0.05, opts.erosion.deltaAt(_dir)),
            opts.erosion.lakeMaskAt(_dir),
          )
          // Lowness: below sea level is most prospective for burial
          const lowness = smoothstep(0.15, -0.05, h - opts.seaLevel)
          // Maturation window: triangular bell in geothermal gradient space
          const maturation = 1 - Math.abs(geothermal - 0.5 * (GEO_OIL_LO + GEO_OIL_HI))
            / (0.5 * (GEO_OIL_HI - GEO_OIL_LO))
          const maturationClamped = clamp01(maturation)
          // Seal quality: tight (low porosity) rock traps hydrocarbons better
          const seal = 1 - 0.5 * por

          hydrocarbonArr[c] = clamp01(basin * lowness * maturationClamped * seal * waterBudget)
        }
      }
    }

    // -----------------------------------------------------------------------
    // Post-bake blur
    // Smooth fields blurred 2 passes; permafrost sentinel fields NOT blurred.
    // -----------------------------------------------------------------------

    const blurred = [
      waterTableArr, magmaTopArr, geothermalGradArr, porosityArr, hydrocarbonArr, permafrostMaskArr,
    ].map(f => blurField(f, N, res, 2))

    this.waterTable     = blurred[0]
    this.magmaTop       = blurred[1]
    this.geothermalGrad = blurred[2]
    this.porosity       = blurred[3]
    this.hydrocarbon    = blurred[4]
    this.permafrostMask = blurred[5]
    // Sentinel arrays: stored as-is (blur would smear sentinels)
    this.permafrostTop  = permafrostTopArr
    this.permafrostBot  = permafrostBotArr
  }

  // -------------------------------------------------------------------------
  // Static constructors
  // -------------------------------------------------------------------------

  /**
   * Reconstruct a sampler-only Subsurface from a baked snapshot.
   * Does NOT re-run the simulation.
   * `tectonics` is passed separately so oreAt() remains functional.
   */
  static fromBaked(b: SubsurfaceBaked, tectonics: Tectonics | null): Subsurface {
    const s = Object.create(Subsurface.prototype) as Subsurface
    const R = s as unknown as Record<string, unknown>

    // Data arrays and scalars from baked snapshot
    R['res']            = b.res
    R['waterTable']     = b.waterTable
    R['porosity']       = b.porosity
    R['magmaTop']       = b.magmaTop
    R['geothermalGrad'] = b.geothermalGrad
    R['permafrostTop']  = b.permafrostTop
    R['permafrostBot']  = b.permafrostBot
    R['permafrostMask'] = b.permafrostMask
    R['hydrocarbon']    = b.hydrocarbon
    R['porosityBase']   = b.porosityBase
    R['geoScale']       = b.geoScale

    // Field initializers don't run under Object.create — must init manually.
    R['_scratch']  = new Vector3()
    R['_scratch2'] = new Vector3()

    // Rebuild noise from stored seeds (byte-identical to constructor path)
    R['_warpSeed']  = b.warpSeed
    R['_warpNoise'] = createNoise3D(b.warpSeed)
    R['_oreSeed']   = b.oreSeed
    R['_oreNoise']  = createNoise3D(b.oreSeed)

    // Tectonics handle (for oreAt analytic path)
    R['_tectonics'] = tectonics

    // Zeroed TectonicQuery scratch (field initializers don't run)
    R['_oreQueryScratch'] = {
      plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0,
      shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0,
    }

    return s
  }

  /**
   * Return a Subsurface with all-zero / all-sentinel fields.
   * Bypasses the constructor. Deterministically seeded from seed=0.
   */
  static identity(res: number): Subsurface {
    const N  = 6 * res * res
    const s  = Object.create(Subsurface.prototype) as Subsurface
    const R  = s as unknown as Record<string, unknown>

    R['res']            = res
    R['waterTable']     = new Float32Array(N)
    R['porosity']       = new Float32Array(N)
    R['magmaTop']       = new Float32Array(N)
    R['geothermalGrad'] = new Float32Array(N)

    const ptArr = new Float32Array(N); ptArr.fill(SUB_SENTINEL)
    R['permafrostTop']  = ptArr
    const pbArr = new Float32Array(N); pbArr.fill(SUB_SENTINEL)
    R['permafrostBot']  = pbArr
    R['permafrostMask'] = new Float32Array(N)
    R['hydrocarbon']    = new Float32Array(N)

    R['porosityBase'] = 0
    R['geoScale']     = GEO_LO

    R['_scratch']  = new Vector3()
    R['_scratch2'] = new Vector3()

    // Deterministic fixed seeds for identity (warp is a no-op on zero-fields
    // but noise must still be initialized to avoid undefined on query paths)
    const identWarpSeed = deriveSeed(0, SUB_STREAM_WARP)
    const identOreSeed  = deriveSeed(0, SUB_STREAM_ORE)
    R['_warpSeed']  = identWarpSeed
    R['_warpNoise'] = createNoise3D(identWarpSeed)
    R['_oreSeed']   = identOreSeed
    R['_oreNoise']  = createNoise3D(identOreSeed)

    R['_tectonics'] = null

    R['_oreQueryScratch'] = {
      plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0,
      shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0,
    }

    return s
  }

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /**
   * Snapshot this Subsurface's baked state so a Web Worker can reconstruct a
   * sampler-only instance without re-baking.
   *
   * Arrays are returned BY REFERENCE; the worker pool structured-clones them.
   * Do NOT transfer .buffer — that would detach the main-thread arrays.
   */
  toBaked(): SubsurfaceBaked {
    return {
      res:            this.res,
      waterTable:     this.waterTable,
      porosity:       this.porosity,
      magmaTop:       this.magmaTop,
      geothermalGrad: this.geothermalGrad,
      permafrostTop:  this.permafrostTop,
      permafrostBot:  this.permafrostBot,
      permafrostMask: this.permafrostMask,
      hydrocarbon:    this.hydrocarbon,
      porosityBase:   this.porosityBase,
      geoScale:       this.geoScale,
      warpSeed:       this._warpSeed,
      oreSeed:        this._oreSeed,
    }
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  /**
   * Compute a domain-warped direction from `dir`.
   * Writes result into `this._scratch2` and returns it.
   * Identical pattern to erosion.ts _warpedDir().
   */
  private _warpedDir(dir: Vector3): Vector3 {
    const noise = this._warpNoise
    const fx = dir.x * WARP_FREQ
    const fy = dir.y * WARP_FREQ
    const fz = dir.z * WARP_FREQ

    let wx = 0, wy = 0, wz = 0
    let amp = 1, freq = 1
    for (let o = 0; o < WARP_OCT; o++) {
      wx += amp * noise(fx * freq,         fy * freq,         fz * freq)
      wy += amp * noise(fx * freq + 3.7,   fy * freq + 2.1,   fz * freq + 1.3)
      wz += amp * noise(fx * freq + 7.3,   fy * freq + 5.9,   fz * freq + 4.1)
      amp  *= 0.5
      freq *= 2.0
    }
    const maxAmp = 1.5
    wx = (wx / maxAmp) * WARP_AMP
    wy = (wy / maxAmp) * WARP_AMP
    wz = (wz / maxAmp) * WARP_AMP

    const dot = wx * dir.x + wy * dir.y + wz * dir.z
    const tx = wx - dot * dir.x
    const ty = wy - dot * dir.y
    const tz = wz - dot * dir.z

    const rx = dir.x + tx
    const ry = dir.y + ty
    const rz = dir.z + tz
    const len = Math.sqrt(rx * rx + ry * ry + rz * rz)
    const inv = len > 1e-8 ? 1 / len : 1
    this._scratch2.set(rx * inv, ry * inv, rz * inv)
    return this._scratch2
  }

  /**
   * Sample a per-texel Float32Array at direction `dir` with C1 continuity.
   * Identical to erosion.ts _sampleC1 — smoothstep on fractional texel coords.
   * NOT safe for fields containing SUB_SENTINEL values.
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
    let fu = fx - x0
    let fv = fy - y0
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)

    const x1 = x0 + 1
    const y1 = y0 + 1

    let t00: number, t10: number, t01: number, t11: number

    if (x0 >= 0 && y0 >= 0 && x1 < res && y1 < res) {
      const base = face * res * res
      const row0 = base + y0 * res
      const row1 = base + y1 * res
      t00 = data[row0 + x0]
      t10 = data[row0 + x1]
      t01 = data[row1 + x0]
      t11 = data[row1 + x1]
    } else {
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
  // Query methods — zero-alloc hot path
  // -------------------------------------------------------------------------

  /** Water table elevation (normalized height) at dir. */
  waterTableAt(dir: Vector3): number {
    return this._sampleC1(this.waterTable, this._warpedDir(dir))
  }

  /** Rock porosity 0..1 at dir. */
  porosityAt(dir: Vector3): number {
    return this._sampleC1(this.porosity, this._warpedDir(dir))
  }

  /** Top of magma/partial-melt zone (normalized height) at dir. */
  magmaTopAt(dir: Vector3): number {
    return this._sampleC1(this.magmaTop, this._warpedDir(dir))
  }

  /** Geothermal gradient intensity 0..1 at dir. */
  geothermalAt(dir: Vector3): number {
    return this._sampleC1(this.geothermalGrad, this._warpedDir(dir))
  }

  /**
   * Permafrost presence mask at dir (0..1, bilinear-safe).
   * Values > 0.5 lie inside the permafrost zone.
   */
  permafrostMaskAt(dir: Vector3): number {
    return this._sampleC1(this.permafrostMask, this._warpedDir(dir))
  }

  /**
   * Top of permafrost layer (normalized height) at dir.
   * Uses plain bilinear (sampleSmooth) — NOT _sampleC1 — because permafrostTop
   * may contain SUB_SENTINEL (-999) which would corrupt smoothstep interpolation.
   * Valid only where permafrostMaskAt() > 0.5.
   */
  permafrostTopAt(dir: Vector3): number {
    return sampleSmooth(this.permafrostTop, this._warpedDir(dir), this.res, this._scratch)
  }

  /**
   * Bottom of permafrost layer (normalized height) at dir.
   * Uses plain bilinear (sampleSmooth) — sentinel-safe.
   * Valid only where permafrostMaskAt() > 0.5.
   */
  permafrostBotAt(dir: Vector3): number {
    return sampleSmooth(this.permafrostBot, this._warpedDir(dir), this.res, this._scratch)
  }

  /** Hydrocarbon potential 0..1 at dir. */
  hydrocarbonAt(dir: Vector3): number {
    return this._sampleC1(this.hydrocarbon, this._warpedDir(dir))
  }

  /**
   * Analytic ore presence at `dir` and absolute normalized elevation `elev`
   * (same units as heightFn output, ~[-1,1]). Pass the vertex terrain height h
   * for surface-outcrop detection.
   *
   * Combines:
   *   - fault/paleo proximity (from tectonics query)
   *   - bell window above the magma top (ore forms in a shallow band just above intrusions)
   *   - ridged vein noise (discrete, veiny structure)
   *
   * Returns 0 if `_tectonics` is null (identity instance).
   * Zero-alloc: reuses `_oreQueryScratch`.
   *
   * Note: tectonics is queried at the raw `dir` while baked fields use the
   * domain-warped dir (_warpedDir). This is intentional — ore veins should
   * track unwarped tectonic geometry.
   */
  oreAt(dir: Vector3, elev: number): number {
    if (this._tectonics === null) return 0

    // Note: tectonics queried at raw dir (not _warpedDir) — intentional; see JSDoc.
    this._tectonics.query(dir, this._oreQueryScratch)
    const q = this._oreQueryScratch

    // Fault and paleo proximity — ore concentrates near ancient and active boundaries
    const faultProx =
      Math.exp(-q.boundaryDist / ORE_BAND_W) +
      0.5 * Math.exp(-q.paleoDist / ORE_BAND_W)

    // Bell window above the magma top: ore concentrates in a shallow band just above
    // shallow magma. aboveMagma > 0 means the query point sits above the magma top.
    const aboveMagma = elev - this.magmaTopAt(dir)
    const t = (aboveMagma - ORE_BAND_CENTER) / ORE_BAND_HW
    const magmaProx = Math.max(0, 1 - t * t)

    // Ridged vein noise — discrete veins, not a diffuse gradient
    const veinNoise = ridgedNoise(
      this._oreNoise,
      dir.x * ORE_FREQ,
      dir.y * ORE_FREQ,
      dir.z * ORE_FREQ,
    )

    // Hard vein threshold (step function) — discrete veins only
    const veinMask = veinNoise >= ORE_VEIN_THRESH ? 1 : 0

    return clamp01(faultProx * magmaProx * veinMask)
  }

  /**
   * Fill `out` with the full stratigraphy column at `dir` and terrain height `terrainH`.
   * Reuses ONE _warpedDir call across all field samples.
   */
  columnAt(dir: Vector3, terrainH: number, out: SubsurfaceColumn): void {
    const dw = this._warpedDir(dir)

    out.waterTable    = this._sampleC1(this.waterTable, dw)
    out.magmaTop      = this._sampleC1(this.magmaTop, dw)
    out.geothermal    = this._sampleC1(this.geothermalGrad, dw)
    out.permafrostTop = sampleSmooth(this.permafrostTop, dw, this.res, this._scratch)
    out.permafrostBot = sampleSmooth(this.permafrostBot, dw, this.res, this._scratch)
    out.porosity      = this._sampleC1(this.porosity, dw)
    out.hydrocarbon   = this._sampleC1(this.hydrocarbon, dw)
    out.springEmerges = out.waterTable >= terrainH
  }
}

// ---------------------------------------------------------------------------
// Module-level blur helper — same 3×3 cross-face box blur as erosion.ts
// ---------------------------------------------------------------------------

function blurField(src: Float32Array, N: number, res: number, passes: number): Float32Array {
  let cur: Float32Array = src
  const tmp = new Float32Array(N)
  const nb = { face: 0, x: 0, y: 0 }
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
