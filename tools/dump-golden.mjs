/**
 * dump-golden.mjs — Read-only golden-vector harness for demiurge terrain.
 *
 * Imports the pure-compute planet modules (noise, tectonics, terrainSampler)
 * and dumps a deterministic golden JSON to:
 *   ../minos/minos-planet/tests/goldens/golden.json
 *
 * Run from the demiurge directory:
 *   cd ~/Work/Prototypes/demiurge && npx tsx tools/dump-golden.mjs
 *
 * Fixed params (documented — must match minos's determinism test construction):
 *   seed          = 42
 *   plate_count   = 12
 *   arc_density   = 1.0
 *   hotspot_count = 8
 *   hotspot_intensity = 1.0
 *   climate: base_temp=15, atmosphere=0.6, band_count=3, axial_tilt=23°
 *
 * DO NOT modify any existing demiurge source file.
 */

import { createNoise3D, fbm, ridged } from '../src/planet/noise.ts'
import { Tectonics } from '../src/planet/tectonics.ts'
import { makeTerrainSampler } from '../src/planet/terrainSampler.ts'
import { Vector3 } from 'three'
import { writeFileSync, mkdirSync } from 'fs'
import { dirname } from 'path'
import { fileURLToPath } from 'url'

// ---------------------------------------------------------------------------
// Fixed params — must match minos's determinism test exactly
// ---------------------------------------------------------------------------

const PARAMS = {
  seed: 42,
  plate_count: 12,
  arc_density: 1.0,
  hotspot_count: 8,
  hotspot_intensity: 1.0,
  base_temp: 15.0,
  atmosphere: 0.6,
  band_count: 3,
  axial_tilt_deg: 23.0,
}

const AXIAL_TILT_RAD = PARAMS.axial_tilt_deg * (Math.PI / 180.0)

// ---------------------------------------------------------------------------
// Fixed sample coords for noise (50 points on unit sphere via golden angle)
// ---------------------------------------------------------------------------

function goldenSpherePoints(n) {
  const pts = []
  const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0))
  for (let i = 0; i < n; i++) {
    const y = 1.0 - (i / (n - 1.0)) * 2.0
    const r = Math.sqrt(Math.max(0, 1.0 - y * y))
    const theta = goldenAngle * i
    pts.push([r * Math.cos(theta), y, r * Math.sin(theta)])
  }
  return pts
}

// ---------------------------------------------------------------------------
// A. Noise samples
// ---------------------------------------------------------------------------

console.error('[golden] Building noise...')
const noise = createNoise3D(PARAMS.seed)

const noisePts = goldenSpherePoints(50)

const noiseSamples = noisePts.map(([x, y, z]) => ({
  x, y, z,
  value: noise(x, y, z),
}))

// FBM and ridged at fixed coords × {4,6,8} octaves
const fbmRidgedPts = goldenSpherePoints(20)
const octaveSets = [4, 6, 8]

const fbmSamples = []
const ridgedSamples = []

for (const [x, y, z] of fbmRidgedPts) {
  for (const oct of octaveSets) {
    fbmSamples.push({
      x, y, z,
      octaves: oct,
      frequency: 1.0,
      lacunarity: 2.0,
      gain: 0.5,
      value: fbm(noise, x, y, z, { octaves: oct, frequency: 1.0, lacunarity: 2.0, gain: 0.5 }),
    })
    ridgedSamples.push({
      x, y, z,
      octaves: oct,
      frequency: 1.0,
      lacunarity: 2.0,
      gain: 0.5,
      value: ridged(noise, x, y, z, { octaves: oct, frequency: 1.0, lacunarity: 2.0, gain: 0.5 }),
    })
  }
}

// ---------------------------------------------------------------------------
// B. Tectonics
// ---------------------------------------------------------------------------

console.error('[golden] Building tectonics (may take ~10s)...')
const tect = new Tectonics({
  seed: PARAMS.seed,
  plateCount: PARAMS.plate_count,
  arcDensity: PARAMS.arc_density,
  hotspotCount: PARAMS.hotspot_count,
  hotspotIntensity: PARAMS.hotspot_intensity,
})

const tectonicsDirs = goldenSpherePoints(30)
const tectonicsScratch = {
  plateId: 0, neighborId: 0, boundaryDist: 0,
  convergence: 0, shear: 0, crustDist: 0,
  paleoDist: 0, otherCrustDist: 0,
}

const tectSamples = tectonicsDirs.map(([x, y, z]) => {
  const dir = new Vector3(x, y, z)
  tect.query(dir, tectonicsScratch)
  return {
    x, y, z,
    plate_id: tectonicsScratch.plateId,
    neighbor_id: tectonicsScratch.neighborId,
    boundary_dist: tectonicsScratch.boundaryDist,
    convergence: tectonicsScratch.convergence,
    shear: tectonicsScratch.shear,
    crust_dist: tectonicsScratch.crustDist,
    paleo_dist: tectonicsScratch.paleoDist,
    other_crust_dist: tectonicsScratch.otherCrustDist,
  }
})

// ---------------------------------------------------------------------------
// C. heightFn — via makeTerrainSampler (reuses existing tect instance)
// ---------------------------------------------------------------------------

console.error('[golden] Building terrain sampler...')
const sampler = makeTerrainSampler({
  seed: PARAMS.seed,
  radius: 1.0,
  heightScale: 1.0,
  plateCount: PARAMS.plate_count,
  arcDensity: PARAMS.arc_density,
  baseTemp: PARAMS.base_temp,
  atmosphere: PARAMS.atmosphere,
  bandCount: PARAMS.band_count,
  axialTiltRad: AXIAL_TILT_RAD,
  tectonics: tect,  // reuse the already-built instance
})

const heightDirs = goldenSpherePoints(30)
const levels = [0, 4, 8, 12]

const heightSamples = []
for (const [x, y, z] of heightDirs) {
  const dir = new Vector3(x, y, z)
  for (const level of levels) {
    heightSamples.push({
      x, y, z,
      level,
      value: sampler.heightFn(dir, level),
    })
  }
}

// ---------------------------------------------------------------------------
// Assemble golden JSON
// ---------------------------------------------------------------------------

const golden = {
  _meta: {
    description: 'Demiurge terrain determinism goldens — read-only dump from demiurge planet modules.',
    generated: new Date().toISOString(),
    params: PARAMS,
    command: 'cd ~/Work/Prototypes/demiurge && npx tsx tools/dump-golden.mjs',
  },
  params: PARAMS,
  noise_samples: noiseSamples,
  fbm_samples: fbmSamples,
  ridged_samples: ridgedSamples,
  tectonics_samples: tectSamples,
  height_samples: heightSamples,
}

// ---------------------------------------------------------------------------
// Write output
// ---------------------------------------------------------------------------

const __dirname = dirname(fileURLToPath(import.meta.url))
const outPath = `${__dirname}/../../minos/minos-planet/tests/goldens/golden.json`

mkdirSync(dirname(outPath), { recursive: true })
writeFileSync(outPath, JSON.stringify(golden, null, 2), 'utf8')

console.error(`[golden] Written ${outPath}`)
console.error(`[golden] noise_samples: ${noiseSamples.length}`)
console.error(`[golden] fbm_samples: ${fbmSamples.length}`)
console.error(`[golden] ridged_samples: ${ridgedSamples.length}`)
console.error(`[golden] tectonics_samples: ${tectSamples.length}`)
console.error(`[golden] height_samples: ${heightSamples.length}`)
console.error('[golden] Done.')
