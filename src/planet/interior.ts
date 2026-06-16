// interior.ts — mantle root → tectonic regime derivation (steps 1-2 of generation pipeline)
// Pure & deterministic: no Math.random, no Date.now, no global state.
//
// PIPELINE:
//   Stage A: roots (mass, composition, age, insolation, waterBudget)
//            → physical state (gravity, internalHeat Q, equilibriumTemp, atmosphere, surfaceTemp)
//   Stage B: physical state → regime engine (vigor, mobility, regime, plateCount, …)
//            (Stage B logic is unchanged; only its inputs changed from roots to Stage-A outputs)
//
// regime-corner table (verified — run node against these cases to confirm):
//   1. Earth      (1,   0.5, 0.5,  1,   0.5) → plated,      plateCount 12-16, T≈14°C, atm≈0.58
//   2. Mars       (0.5, 0.5, 0.7,  0.43,0.15) → stagnant-lid, atm<0.2, T<-20°C
//   3. Mercury    (0.4, 0.6, 0.85, 6.7, 0.05) → dead-lid,    atm≈0
//   4. Venus      (0.85,0.5, 0.5,  1.9, 0.2)  → stagnant-lid, atm>0.8, T>100°C
//   5. Io-ish     (1.5, 0.7, 0.1,  1,   0.1)  → heat-pipe
//   6. Lava world (1,   0.5, 0.3, 15,   0.3)  → magma-ocean

export interface InteriorParams {
  mass: number          // Earth masses, ~0.05..10, default 1
  composition: number   // 0..1 icy/volatile → rocky/metallic, default 0.5
  age: number           // 0..1 young→old, default 0.5
  insolation: number    // stellar flux, Earth = 1, ~0.1..20, default 1
  waterBudget: number   // 0..1, default 0.5
}

export type Regime = 'dead-lid' | 'stagnant-lid' | 'plated' | 'heat-pipe' | 'magma-ocean'

export interface DerivedInterior {
  // Stage-A physical state (readouts):
  gravity: number          // Earth g (=1 at mass 1)
  internalHeat: number     // Q, 0..1 — derived (was a root)
  surfaceTemp: number      // °C — derived (was a root); feeds climate baseTemp
  atmosphere: number       // 0..1 — derived; feeds climate atmosphere
  equilibriumTemp: number  // T_eq °C, pre-greenhouse

  // Stage-B regime outputs (unchanged logic):
  vigor: number            // V, 0..1 — convective vigor
  mobility: number         // M, 0..1 — lid breakability into plates
  spectrum: number         // 0..1 continuous position dead-lid(0)→magma-ocean(1)
  regime: Regime
  plateCount: number       // integer, feeds Tectonics
  driftScale: number       // multiplier on plate drift speed
  hotspotCount: number     // integer
  hotspotIntensity: number
  arcDensity: number
  oceanCoverage: number    // 0..1, target fraction of surface covered by ocean
}

export const DEFAULT_INTERIOR: InteriorParams = {
  mass: 1,
  composition: 0.5,
  age: 0.5,
  insolation: 1,
  waterBudget: 0.5,
}

// All thresholds and coefficients in one place — tune here, not in the formulas.
export const INTERIOR_TUNING = {
  // ---- Stage A: mass → radius/density ----------------------------------------
  // Relative density: 0=icy/volatile → low density, 1=rocky/metallic → high density
  // Anchored so density(comp=0.5)=1 → radius(mass=1, comp=0.5)=1 → gravity=1
  DENSITY_ICY:     0.6,   // relative density at comp=0
  DENSITY_ROCKY:   1.4,   // relative density at comp=1

  // ---- Stage A: internalHeat Q ------------------------------------------------
  // Q = clamp01( radiogenics(comp) * heatRetention(mass) * ageDecay(age) )
  RADIO_LO:        0.4,   // radiogenic production at comp=0 (icy/volatile)
  RADIO_HI:        1.0,   // radiogenic production at comp=1 (rocky/metallic)
  // heatRetention: larger mass → lower surface-area/volume → better retention
  RETAIN_SCALE:    0.5,   // growth rate with ln(mass)
  RETAIN_MID:      0.65,  // retention at mass=1 (Earth)
  // ageDecay: young planets are hot from formation + high radiogenic flux
  AGE_YOUNG:       1.0,
  AGE_OLD:         0.3,

  // ---- Stage A: equilibriumTemp (no greenhouse) ------------------------------
  // T_eq_K = STEFAN_K * insolation^0.25 * (1 - albedo)^0.25
  // Earth (insolation=1, albedo≈0.375) → ~255K / -18°C bare
  STEFAN_K:        278.5, // K effective solar constant
  ALBEDO_BASE:     0.30,  // baseline planetary albedo
  ALBEDO_ICY:      0.15,  // extra albedo for comp=0 (icy worlds)

  // ---- Stage A: atmosphere ---------------------------------------------------
  // Atmosphere = clamp01( sigmoid( ATM_MASS*ln(mass) + ATM_COOL*(1-T_eq_norm)
  //                               + ATM_VOL*water + ATM_AGE*(1-age)
  //                               + ATM_CO2*comp*(1-water)
  //                               + hotCO2 - ATM_K0 ) )
  // where T_eq_norm = (T_eq_K - 200) / 300
  ATM_MASS:        2.5,   // large mass → strong gravity → retains atmosphere
  ATM_COOL:        2.0,   // cold T_eq → low Jeans escape → retains atmosphere
  ATM_VOL:         1.2,   // volatiles/water → volcanic outgassing
  ATM_AGE:         0.8,   // young → active volcanism → more outgassing
  ATM_CO2:         1.4,   // rocky + dry → baseline CO2 buildup
  // hotCO2: Venus mechanism — rocky+dry+hot enough that oceans never condense,
  // so CO2 never gets locked into carbonates → runaway thick CO2 atmosphere.
  // Only applies above mass gate (too-small planets can't retain CO2 regardless).
  ATM_HOT_CO2:     20.0,  // max CO2-runaway contribution
  ATM_HOT_THRESH:  270,   // K — above this T_eq, hot-CO2 kicks in
  ATM_HOT_WIDTH:   50,    // K — transition width
  ATM_MASS_MIN:    0.5,   // mass gate: below this, not enough gravity to hold CO2
  ATM_K0:          2.7,   // sigmoid offset — calibrate so Earth atm ≈ 0.58

  // ---- Stage A: surfaceTemp (greenhouse) ------------------------------------
  // Two-component greenhouse:
  //   Linear term: thin-atm CO2/H2O contribution (Earth-like)
  //   Runaway term: thick-atm positive feedback (Venus-like)
  // surfaceTemp °C = T_eq_celsius + GH_LINEAR*atm + GH_RUNAWAY*(clamp01((atm-GH_THRESH)/GH_WIDTH))^GH_EXP
  GH_LINEAR:       60.0,  // °C per unit atm (thin-atm linear contribution)
  GH_RUNAWAY:      800.0, // max runaway boost (saturates at atm=1)
  GH_THRESH:       0.55,  // atm above which runaway kicks in
  GH_WIDTH:        0.45,  // transition width for runaway
  GH_EXP:          2.0,   // power-law exponent for runaway curve

  // ---- Stage B: Vigor V -------------------------------------------------------
  // V = clamp01( internalHeat * sizeFactor(mass) * compFactor(comp)
  //              + tempBoost(surfaceTemp) + insolationVigor )
  // sizeFactor: larger mass → more internal pressure → stronger convection
  SIZE_SMALL:      0.7,
  SIZE_LARGE:      1.3,
  // compFactor: rocky/metallic → lower viscosity, more vigorous convection
  COMP_LO:         0.85,
  COMP_HI:         1.15,
  // tempBoost: extremely hot surface (>600°C) → crustal melting contribution
  HOT_LO:          600,   // °C — below this, no tempBoost
  HOT_HI:          1500,  // °C — saturates here
  TEMP_BOOST:      0.3,
  // insolationVigor: extreme T_eq_K → direct surface/crustal melting (lava worlds)
  // This is separate from tempBoost — it uses T_eq_K before greenhouse to avoid
  // the greenhouse → high surfaceTemp → tempBoost → ... circular dependency issue
  INSOL_THRESH_K:  400,   // K — above this T_eq_K, insolation starts contributing to V
  INSOL_RANGE_K:   100,   // K — saturates over this range
  INSOL_BOOST:     0.70,  // max contribution to V from extreme insolation

  // ---- Stage B: Mobility M ---------------------------------------------------
  // sigmoid(K_WATER*water + K_VIG*V + K_TEMP*tempNorm - K0)
  TEMP_NORM_LO:    -100,
  TEMP_NORM_RANGE: 200,
  K_WATER:         3.5,
  K_VIG:           1.5,
  K_TEMP:          0.4,
  K0:              2.0,

  // ---- Stage B: Regime thresholds (on V and M) --------------------------------
  DEAD_V:          0.08,  // V < this → dead-lid
  HOTPIPE_V:       0.65,  // V >= this (and < MAGMA_V) → heat-pipe
  MAGMA_V:         0.90,  // V >= this → magma-ocean
  MOBILE_M:        0.5,   // M < this (mid V band) → stagnant-lid; else → plated

  // ---- Stage B: spectrum band centers ----------------------------------------
  SPECTRUM_DEAD:     0.0,
  SPECTRUM_STAGNANT: 0.25,
  SPECTRUM_PLATED:   0.5,
  SPECTRUM_HOTPIPE:  0.75,
  SPECTRUM_MAGMA:    1.0,

  // ---- Stage B: plateCount ---------------------------------------------------
  PLATE_MIN:       3,
  PLATE_MAX:       24,

  // ---- Stage B: driftScale ---------------------------------------------------
  DRIFT_LO:        0.3,
  DRIFT_HI:        2.0,

  // ---- Stage B: hotspotCount -------------------------------------------------
  HOTSPOT_MAX:              18,
  HOTSPOT_BOOST_STAGNANT:   1.4,
  HOTSPOT_BOOST_HOTPIPE:    1.6,

  // ---- Stage B: hotspotIntensity ---------------------------------------------
  HOTSPOT_INT_LO:  0.5,
  HOTSPOT_INT_HI:  2.5,

  // ---- Stage B: arcDensity ---------------------------------------------------
  ARC_LO:          0.2,
  ARC_HI:          2.5,

  // ---- oceanCoverage ---------------------------------------------------------
  // oceanCoverage = lerp(OC_LO, OC_HI, waterBudget^OC_EXP)
  // Calibrated: waterBudget 0→~0.02, 0.5→~0.68, 1→~0.97
  // OC_EXP ≈ 0.52 lifts the mid-range so Earth-like budgets (0.5) yield ~70% ocean.
  OC_LO:           0.02,  // coverage at waterBudget = 0 (almost all land)
  OC_HI:           0.97,  // coverage at waterBudget = 1 (almost all ocean)
  OC_EXP:          0.52,  // power-law exponent (< 1 = convex, lifts midpoint)
}

// ---- helpers ----------------------------------------------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

function smoothstep(lo: number, hi: number, x: number): number {
  const t = clamp01((x - lo) / (hi - lo))
  return t * t * (3 - 2 * t)
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

// ---- Stage A sub-factors ----------------------------------------------------

function ageDecay(age: number): number {
  return lerp(INTERIOR_TUNING.AGE_YOUNG, INTERIOR_TUNING.AGE_OLD, clamp01(age))
}

// radius in Earth-radii from mass (Earth-masses) and composition
// density = lerp(DENSITY_ICY, DENSITY_ROCKY, composition) [anchored so comp=0.5→density=1]
// volume ∝ mass/density → radius = (mass/density)^(1/3)
function deriveRadius(mass: number, composition: number): number {
  const t = INTERIOR_TUNING
  const density = lerp(t.DENSITY_ICY, t.DENSITY_ROCKY, clamp01(composition))
  return Math.pow(mass / density, 1 / 3)
}

// radiogenic heat production: higher for rocky/metallic composition
function radiogenics(composition: number): number {
  const t = INTERIOR_TUNING
  return lerp(t.RADIO_LO, t.RADIO_HI, clamp01(composition))
}

// heat retention: larger mass → lower surface-area/volume → better retention
function heatRetention(mass: number): number {
  const t = INTERIOR_TUNING
  return clamp01(t.RETAIN_SCALE * Math.log(Math.max(mass, 0.01)) + t.RETAIN_MID)
}

// equilibrium temperature in Kelvin (pre-greenhouse, bare rock)
function equilibriumTempK(insolation: number, composition: number): number {
  const t = INTERIOR_TUNING
  const albedo = t.ALBEDO_BASE + t.ALBEDO_ICY * (1 - clamp01(composition))
  return t.STEFAN_K * Math.pow(insolation, 0.25) * Math.pow(1 - albedo, 0.25)
}

// atmosphere thickness from all roots
function deriveAtmosphere(
  mass: number,
  T_eq_K: number,
  waterBudget: number,
  age: number,
  composition: number
): number {
  const t = INTERIOR_TUNING
  // T_eq_norm: 0 at 200K (cold/retained), 1 at 500K (hot/escaping via Jeans)
  const T_eq_norm = clamp01((T_eq_K - 200) / 300)
  // CO2 runaway (Venus mechanism): hot T_eq + rocky + dry + sufficient mass
  // → no ocean condensation → CO2 never locked into carbonates → thick CO2 atm
  const massGate = clamp01((mass - t.ATM_MASS_MIN) / (1 - t.ATM_MASS_MIN))
  const hotCO2 =
    t.ATM_HOT_CO2 *
    smoothstep(t.ATM_HOT_THRESH, t.ATM_HOT_THRESH + t.ATM_HOT_WIDTH, T_eq_K) *
    clamp01(composition) *
    (1 - waterBudget) *
    massGate
  const sigInput =
    t.ATM_MASS * Math.log(Math.max(mass, 0.01)) +
    t.ATM_COOL * (1 - T_eq_norm) +
    t.ATM_VOL * waterBudget +
    t.ATM_AGE * (1 - age) +
    t.ATM_CO2 * clamp01(composition) * (1 - waterBudget) +
    hotCO2 -
    t.ATM_K0
  return clamp01(sigmoid(sigInput))
}

// greenhouse boost in °C: linear thin-atm term + runaway thick-atm term
function greenhouse(atmosphere: number): number {
  const t = INTERIOR_TUNING
  const linear = t.GH_LINEAR * atmosphere
  const runaway =
    t.GH_RUNAWAY * Math.pow(clamp01((atmosphere - t.GH_THRESH) / t.GH_WIDTH), t.GH_EXP)
  return linear + runaway
}

// ---- Stage B sub-factors (unchanged from original, mass replaces planetSize) -----

function sizeFactor(mass: number): number {
  const t = INTERIOR_TUNING
  return lerp(t.SIZE_SMALL, t.SIZE_LARGE, clamp01(mass / 3))
}

function compFactor(composition: number): number {
  const t = INTERIOR_TUNING
  return lerp(t.COMP_LO, t.COMP_HI, clamp01(composition))
}

function tempBoost(surfaceTemp: number): number {
  return smoothstep(INTERIOR_TUNING.HOT_LO, INTERIOR_TUNING.HOT_HI, surfaceTemp) *
         INTERIOR_TUNING.TEMP_BOOST
}

// ---- spectrum ---------------------------------------------------------------

function computeSpectrum(V: number, M: number, regime: Regime): number {
  const t = INTERIOR_TUNING
  const vMidFrac = clamp01((V - t.DEAD_V) / (t.HOTPIPE_V - t.DEAD_V))

  switch (regime) {
    case 'dead-lid': {
      const frac = clamp01(V / t.DEAD_V)
      return lerp(t.SPECTRUM_DEAD, t.SPECTRUM_STAGNANT, frac) * 0.5
    }
    case 'stagnant-lid': {
      const mFrac = clamp01(M / t.MOBILE_M)
      return lerp(t.SPECTRUM_STAGNANT * 0.5, t.SPECTRUM_PLATED, vMidFrac * 0.5 + mFrac * 0.15)
    }
    case 'plated': {
      const mFrac = clamp01((M - t.MOBILE_M) / (1 - t.MOBILE_M))
      return lerp(t.SPECTRUM_STAGNANT, t.SPECTRUM_HOTPIPE, vMidFrac * 0.6 + mFrac * 0.2)
    }
    case 'heat-pipe': {
      const vFrac = clamp01((V - t.HOTPIPE_V) / (t.MAGMA_V - t.HOTPIPE_V))
      return lerp(t.SPECTRUM_HOTPIPE, t.SPECTRUM_MAGMA, vFrac * 0.6)
    }
    case 'magma-ocean': {
      const vFrac = clamp01((V - t.MAGMA_V) / (1 - t.MAGMA_V))
      return lerp(t.SPECTRUM_HOTPIPE, t.SPECTRUM_MAGMA, 0.6 + vFrac * 0.4)
    }
  }
}

// ---- main export ------------------------------------------------------------

export function deriveInterior(
  p: InteriorParams,
  overrides?: Partial<Pick<DerivedInterior, 'internalHeat' | 'surfaceTemp' | 'atmosphere'>>
): DerivedInterior {
  const T = INTERIOR_TUNING

  // ============================================================
  // Stage A: roots → physical state
  // ============================================================

  // Radius and gravity (Earth-relative)
  const radius = deriveRadius(p.mass, p.composition)
  const gravity = p.mass / (radius * radius)

  // Internal heat Q: radiogenic production × heat retention × age decay
  let internalHeat = clamp01(
    radiogenics(p.composition) * heatRetention(p.mass) * ageDecay(p.age)
  )

  // Equilibrium temperature (bare-rock, no greenhouse)
  const T_eq_K = equilibriumTempK(p.insolation, p.composition)
  const equilibriumTemp = T_eq_K - 273.15  // °C

  // Atmosphere thickness
  let atmosphere = deriveAtmosphere(p.mass, T_eq_K, p.waterBudget, p.age, p.composition)

  // Surface temperature: T_eq + greenhouse boost
  let surfaceTemp = equilibriumTemp + greenhouse(atmosphere)

  // Apply overrides — pin mid-layer values before Stage B
  if (overrides) {
    if (overrides.internalHeat !== undefined) internalHeat = overrides.internalHeat
    if (overrides.atmosphere !== undefined) atmosphere = overrides.atmosphere
    if (overrides.surfaceTemp !== undefined) surfaceTemp = overrides.surfaceTemp
  }

  // ============================================================
  // Stage B: physical state → regime (logic unchanged)
  // ============================================================

  // --- Vigor V ---
  // insolationVigor: extreme T_eq_K → direct insolation-driven surface melting
  // (separate path from tempBoost to avoid greenhouse→tempBoost coupling issues)
  const insolationVigor =
    T.INSOL_BOOST * clamp01((T_eq_K - T.INSOL_THRESH_K) / T.INSOL_RANGE_K)

  const rawV =
    internalHeat * sizeFactor(p.mass) * compFactor(p.composition) +
    tempBoost(surfaceTemp) +
    insolationVigor
  const V = clamp01(rawV)

  // --- Mobility M ---
  const tempNorm = clamp01((surfaceTemp - T.TEMP_NORM_LO) / T.TEMP_NORM_RANGE)
  const sigInput =
    T.K_WATER * p.waterBudget + T.K_VIG * V + T.K_TEMP * tempNorm - T.K0
  const M = sigmoid(sigInput)

  // --- Regime ---
  // Order matters: magma/heat-pipe are driven by V alone (convection overwhelms lid stability).
  // Stagnant vs plated is only discriminated in the intermediate V band.
  let regime: Regime
  if (V >= T.MAGMA_V) {
    regime = 'magma-ocean'
  } else if (V >= T.HOTPIPE_V) {
    regime = 'heat-pipe'
  } else if (V < T.DEAD_V) {
    regime = 'dead-lid'
  } else if (M < T.MOBILE_M) {
    regime = 'stagnant-lid'
  } else {
    regime = 'plated'
  }

  // --- Spectrum ---
  const spectrum = clamp01(computeSpectrum(V, M, regime))

  // --- plateCount ---
  let plateCount: number
  if (regime === 'dead-lid' || regime === 'stagnant-lid') {
    plateCount = 1
  } else if (regime === 'plated') {
    const vNorm = clamp01((V - T.DEAD_V) / (T.HOTPIPE_V - T.DEAD_V))
    const combined = clamp01(0.25 * vNorm + 0.75 * M)
    plateCount = Math.round(lerp(T.PLATE_MIN, T.PLATE_MAX, combined))
  } else {
    plateCount = 1
  }

  // --- driftScale ---
  const driftScale = lerp(T.DRIFT_LO, T.DRIFT_HI, V)

  // --- hotspotCount ---
  let baseHotspots = lerp(0, T.HOTSPOT_MAX, V)
  if (regime === 'stagnant-lid') baseHotspots *= T.HOTSPOT_BOOST_STAGNANT
  if (regime === 'heat-pipe') baseHotspots *= T.HOTSPOT_BOOST_HOTPIPE
  const hotspotCount = Math.round(clamp01(baseHotspots / T.HOTSPOT_MAX) * T.HOTSPOT_MAX)

  // --- hotspotIntensity ---
  const hotspotIntensity = lerp(T.HOTSPOT_INT_LO, T.HOTSPOT_INT_HI, V)

  // --- arcDensity ---
  const arcDensity = lerp(T.ARC_LO, T.ARC_HI, M)

  // --- oceanCoverage ---
  // Smooth monotonic curve: lerp(OC_LO, OC_HI, waterBudget^OC_EXP)
  // Special-case waterBudget=0 to avoid 0^fractional in strict environments (= 0 anyway).
  const ocT = p.waterBudget <= 0 ? 0 : Math.pow(p.waterBudget, T.OC_EXP)
  const oceanCoverage = lerp(T.OC_LO, T.OC_HI, ocT)

  return {
    gravity,
    internalHeat,
    surfaceTemp,
    atmosphere,
    equilibriumTemp,
    vigor: V,
    mobility: M,
    spectrum,
    regime,
    plateCount,
    driftScale,
    hotspotCount,
    hotspotIntensity,
    arcDensity,
    oceanCoverage,
  }
}
