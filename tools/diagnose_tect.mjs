import { Tectonics } from '../src/planet/tectonics.ts'
import { Vector3 } from 'three'
import { texelIndex, dirToTexel } from '../src/planet/cubemap.ts'

const tect = new Tectonics({ seed: 42, plateCount: 12, arcDensity: 1.0, hotspotCount: 8, hotspotIntensity: 1.0 })

// Access internal state via toBaked  
const baked = tect.toBaked()

const RES = 256
const tex = { face: 0, x: 0, y: 0 }

// Check dist_field at north pole
const northPole = new Vector3(0, 1, 0)
dirToTexel(northPole, RES, tex)
const northIdx = texelIndex(tex.face, tex.x, tex.y, RES)
console.log(`North pole -> face=${tex.face}, x=${tex.x}, y=${tex.y}, idx=${northIdx}`)
console.log(`comp_id[north_pole] = ${baked.compId[northIdx]}`)
console.log(`dist_field_raw_nearby...`)

// Show dist_field values around north pole center (face 2, center)
console.log('\nRaw dist_field at face 2 center area:')
for (let dy = -2; dy <= 2; dy++) {
  let row = ''
  for (let dx = -2; dx <= 2; dx++) {
    const x = 128 + dx, y = 128 + dy
    const i = texelIndex(2, x, y, RES)
    row += `(${baked.compId[i]},${baked.distField[i].toPrecision(4)},nb=${baked.neighborId[i]}) `
  }
  console.log(`  y=${128+dy}: ${row}`)
}

// Query north pole
const q = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0 }
tect.query(northPole, q)
console.log(`\nquery(0,1,0): plate=${q.plateId}, neighbor=${q.neighborId}, bd=${q.boundaryDist.toPrecision(17)}`)
