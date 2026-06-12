/**
 * Deterministic spherical crack-propagation plate-tectonics model.
 *
 * Pure function of seed: no Math.random / Date.now anywhere.
 * All randomness is derived from splitmix32 sub-seeds.
 *
 * Pipeline (all in constructor):
 *   1. Trace cracks  — sphere-walking walkers marking a crack bitmask.
 *   2. Component count loop — add cracks until we have >= target components.
 *   3. Flood fill    — BFS over non-crack texels → component ids.
 *   4. Merge to target — absorb small/excess components; re-index.
 *   5. Plate properties — reuse existing seeded distributions.
 *   6. Distance + neighbour fields — Dijkstra on the cube-map grid.
 *   7. Query (hot path) — table lookup + gradient-based convergence.
 */

import { Vector3 } from 'three'
import { createNoise3D, fbm } from './noise'
import {
  dirToTexel,
  texelToDir,
  texelIndex,
  texAng,
  neighborTexel,
  sampleSmooth,
} from './cubemap'

// ---------------------------------------------------------------------------
// PRNG helpers — same mixer as noise.ts (splitmix32)
// ---------------------------------------------------------------------------

function splitmix32Step(s: number): number {
  s = (s + 0x9e3779b9) >>> 0
  let z = s
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0
  return (z ^ (z >>> 16)) >>> 0
}

function deriveSeed(masterSeed: number, stream: number): number {
  const s = (masterSeed ^ Math.imul(stream + 1, 0xdeadbeef)) >>> 0
  return splitmix32Step(s)
}

function makeRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = splitmix32Step(s)
    return s / 0x100000000
  }
}

// ---------------------------------------------------------------------------
// HSL → RGB
// ---------------------------------------------------------------------------

function hue2rgb(p: number, q: number, t: number): number {
  if (t < 0) t += 1
  if (t > 1) t -= 1
  if (t < 1 / 6) return p + (q - p) * 6 * t
  if (t < 1 / 2) return q
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
  return p
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)]
}

// ---------------------------------------------------------------------------
// Public types (UNCHANGED — consumers depend on every detail)
// ---------------------------------------------------------------------------

export interface Plate {
  id: number
  seedDir: Vector3
  type: 'oceanic' | 'continental'
  baseElevation: number
  omega: Vector3
  color: [number, number, number]
}

export interface TectonicQuery {
  plateId: number
  neighborId: number
  boundaryDist: number  // radians
  convergence: number   // ≈[-1,1]
}

// ---------------------------------------------------------------------------
// Domain-warp constants (unchanged from old tectonics.ts)
// ---------------------------------------------------------------------------

const WARP_O1 = { x: 13.7, y: 0.0,  z: 0.0  }
const WARP_O2 = { x: 0.0,  y: 13.7, z: 0.0  }
const WARP_O3 = { x: 0.0,  y: 0.0,  z: 13.7 }
const WARP_AMP = 0.08
const WARP_FREQ = 2.2
const WARP_OCTAVES = 4

// ---------------------------------------------------------------------------
// Cube-map resolution (private const)
// ---------------------------------------------------------------------------

const RES = 256
const TOTAL_TEXELS = 6 * RES * RES

// Pre-compute texel-angle constant at RES
const TEX_ANG = texAng(RES)

// ---------------------------------------------------------------------------
// Walker mechanics — Step 1
// ---------------------------------------------------------------------------

const STEP = 0.005             // radians per crack step
const BRANCH_PROB = 0.0035     // probability of branching per step
const POLAR_COS = Math.cos((10 * Math.PI) / 180)  // cos(10°) ≈ 0.9848 — kill within 10° of pole: |y| > cos(10°)
const SEED_POLAR_COS = Math.cos((12 * Math.PI) / 180)  // cos(12°) — reject seed spawn inside polar caps
const SELF_GRACE = 16          // trail window size (circular buffer)
const MIN_BUDGET = 0.6         // radians (minimum walker budget)
const MAX_BUDGET = 3.0         // radians (maximum walker budget)
const MAX_STEPS_HARD = 1200    // hard step cap per walker

// Stream ids for seeded RNG streams (extending old scheme)
// 0-6 = plate properties (same as before)
// 7   = crack walker seeds
// 8   = additional crack round seeds (for the count loop)

/**
 * Run one round of crack tracing.
 * Returns the crack bitmask (mutates the passed-in Uint8Array).
 * seedCount = K = number of seed cracks (each spawns 2 walkers).
 * opts.spawnFilter: optional predicate on texel index; seed spawns are rejected
 *   if the texel does not pass the filter (bounded rejection-sampling, ~200 attempts).
 *   Walkers roam freely once spawned — only spawn-point selection is filtered.
 */
function traceCracks(
  crackMask: Uint8Array,
  rng: () => number,
  seedCount: number,
  maxWalkers: number,
  dirScratch: Vector3,
  tex: { face: number; x: number; y: number },
  opts?: { spawnFilter?: (texelIdx: number) => boolean },
): void {
  const spawnFilter = opts?.spawnFilter
  // Active walkers: [posX, posY, posZ, tanX, tanY, tanZ, budget, steps, bias, trail]
  // trail = circular buffer of texel indices, length SELF_GRACE
  interface Walker {
    pos: Vector3
    tan: Vector3   // heading tangent (unit, ⊥ pos)
    budget: number // remaining radians
    steps: number  // total steps taken
    bias: number   // per-walker curvature bias
    trail: Uint32Array  // circular buffer of recent texel indices
    trailHead: number   // write index
  }

  const walkers: Walker[] = []

  const spawnWalker = (pos: Vector3, heading: Vector3, parentTrail?: Uint32Array, parentTrailHead?: number): void => {
    if (walkers.length >= maxWalkers) return
    const budget = MIN_BUDGET + rng() * (MAX_BUDGET - MIN_BUDGET)
    const bias = (rng() - 0.5) * 0.08
    // Inherit parent trail to avoid immediate self-collision death at branch point.
    // A branch is born at the parent's current position which is already marked;
    // without the inherited trail the corrected self-collision check kills it instantly.
    const trail = new Uint32Array(SELF_GRACE).fill(0xFFFFFFFF)
    let trailHead = 0
    if (parentTrail !== undefined && parentTrailHead !== undefined) {
      trail.set(parentTrail)
      trailHead = parentTrailHead
    }
    walkers.push({
      pos: pos.clone(),
      tan: heading.clone(),
      budget,
      steps: 0,
      bias,
      trail,
      trailHead,
    })
  }

  // Seed K starting positions — one walker each (bidirectional twins removed;
  // budget is doubled instead to keep total crack length comparable).
  // Reject spawn positions inside polar caps (|y| > SEED_POLAR_COS).
  // If spawnFilter is provided, also reject positions that don't pass it
  // (bounded rejection-sampling: up to 200 attempts total per seed).
  for (let s = 0; s < seedCount; s++) {
    // Random point on sphere via rejection sampling from rng stream.
    // Also reject polar-cap positions (bounded attempts to preserve determinism).
    let px = 0, py = 0, pz = 0
    const maxAttempts = spawnFilter ? 200 : 50
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const rx = rng() * 2 - 1
      const ry = rng() * 2 - 1
      const rz = rng() * 2 - 1
      const len = Math.sqrt(rx * rx + ry * ry + rz * rz)
      if (len > 0.01 && len < 1.0) {
        const nx = rx / len, ny = ry / len, nz = rz / len
        // Reject polar caps
        if (Math.abs(ny) > SEED_POLAR_COS) { continue }
        // Reject if spawn filter rejects this texel
        if (spawnFilter) {
          const stex = { face: 0, x: 0, y: 0 }
          dirScratch.set(nx, ny, nz)
          dirToTexel(dirScratch, RES, stex)
          const sidx = texelIndex(stex.face, stex.x, stex.y, RES)
          if (!spawnFilter(sidx)) { continue }
        }
        px = nx; py = ny; pz = nz
        break
      }
    }
    if (px === 0 && py === 0 && pz === 0) { continue }  // failed to find valid position
    dirScratch.set(px, py, pz).normalize()

    // Random tangent ⊥ pos
    const arbX = Math.abs(px) < 0.9 ? 1 : 0
    const arbY = Math.abs(px) < 0.9 ? 0 : 1
    const arbZ = 0
    // Cross product: pos × arb
    let tx = py * arbZ - pz * arbY
    let ty = pz * arbX - px * arbZ
    let tz = px * arbY - py * arbX
    const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz) || 1
    tx /= tLen; ty /= tLen; tz /= tLen

    const heading = new Vector3(tx, ty, tz)

    spawnWalker(dirScratch, heading)
  }

  // Process walkers
  let wi = 0
  while (wi < walkers.length) {
    const w = walkers[wi]

    // Kill if polar cap
    if (Math.abs(w.pos.y) > POLAR_COS) {
      walkers.splice(wi, 1)
      continue
    }

    // Budget/step cap
    if (w.budget <= 0 || w.steps >= MAX_STEPS_HARD) {
      walkers.splice(wi, 1)
      continue
    }

    // Current texel
    dirToTexel(w.pos, RES, tex)
    const curIdx = texelIndex(tex.face, tex.x, tex.y, RES)

    // Check self-collision (BUG A fix): die if this texel is already crack-marked
    // by something OTHER than the walker's own recent writes (trail window).
    // This kills on: foreign cracks (T-junctions), and the walker's own path older
    // than the trail window (loop closure). It never kills on own fresh marks.
    if (crackMask[curIdx] === 1) {
      const trailLen = Math.min(w.trailHead, SELF_GRACE)
      let ownMark = false
      for (let ti = 0; ti < trailLen; ti++) {
        if (w.trail[ti] === curIdx) { ownMark = true; break }
      }
      if (!ownMark) {
        walkers.splice(wi, 1)
        continue
      }
    }

    // Step the walker
    // Save previous texel for 4-connectivity diagonal fix
    const prevTex = { face: tex.face, x: tex.x, y: tex.y }

    // Mark crack at current position
    crackMask[curIdx] = 1

    // Write current texel to trail buffer
    w.trail[w.trailHead % SELF_GRACE] = curIdx
    w.trailHead++

    // Advance position: pos' = normalize(pos + STEP * tan)
    const nx = w.pos.x + STEP * w.tan.x
    const ny = w.pos.y + STEP * w.tan.y
    const nz = w.pos.z + STEP * w.tan.z
    const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1
    w.pos.set(nx / nLen, ny / nLen, nz / nLen)

    // Re-orthonormalize tan against new pos
    const tdot = w.tan.x * w.pos.x + w.tan.y * w.pos.y + w.tan.z * w.pos.z
    let tox = w.tan.x - tdot * w.pos.x
    let toy = w.tan.y - tdot * w.pos.y
    let toz = w.tan.z - tdot * w.pos.z
    const toLen = Math.sqrt(tox * tox + toy * toy + toz * toz) || 1
    tox /= toLen; toy /= toLen; toz /= toLen
    w.tan.set(tox, toy, toz)

    // Rotate tan around pos by curvature angle
    const curv = (rng() - 0.5) * 0.18 + w.bias
    // Rodrigues rotation: tan' = tan*cos(curv) + (pos×tan)*sin(curv)
    const cosc = Math.cos(curv)
    const sinc = Math.sin(curv)
    const cx = w.pos.y * w.tan.z - w.pos.z * w.tan.y
    const cy = w.pos.z * w.tan.x - w.pos.x * w.tan.z
    const cz = w.pos.x * w.tan.y - w.pos.y * w.tan.x
    w.tan.set(
      w.tan.x * cosc + cx * sinc,
      w.tan.y * cosc + cy * sinc,
      w.tan.z * cosc + cz * sinc,
    )

    // 4-CONNECTIVITY DIAGONAL FIX:
    // If the new texel is diagonal to the previous texel (both dx≠0 AND dy≠0),
    // additionally mark the shared orthogonal neighbour (dx,0) to close the gap.
    // Without this, a 4-connected flood fill leaks through diagonal crack segments:
    //
    //   . C      C = crack texel (prev)
    //   C .      The lower-left crack is diagonal — the top-left and bottom-right
    //            corners share no 4-connected crack edge between them, so flood
    //            fill flows "between" the two crack cells.
    //
    // Marking the (dx, 0) shared neighbour bridges the gap:
    //
    //   X C      X = extra mark — now the top-left cell IS bounded on all sides.
    //   C .
    dirToTexel(w.pos, RES, tex)
    const newIdx = texelIndex(tex.face, tex.x, tex.y, RES)
    if (tex.face === prevTex.face) {
      const fdx = tex.x - prevTex.x
      const fdy = tex.y - prevTex.y
      if (Math.abs(fdx) === 1 && Math.abs(fdy) === 1) {
        // Diagonal step — mark the (dx, 0) neighbour to seal 4-connected leaks.
        // Also record the bridge texel in the walker's trail so the neighbor-collision
        // check does not mistake the walker's own bridge for a foreign crack.
        const bridgeTex = { face: tex.face, x: prevTex.x + fdx, y: prevTex.y }
        if (bridgeTex.x >= 0 && bridgeTex.x < RES && bridgeTex.y >= 0 && bridgeTex.y < RES) {
          const bridgeIdx = texelIndex(bridgeTex.face, bridgeTex.x, bridgeTex.y, RES)
          crackMask[bridgeIdx] = 1
          w.trail[w.trailHead % SELF_GRACE] = bridgeIdx
          w.trailHead++
        }
      }
    }

    void newIdx

    w.budget -= STEP
    w.steps++

    // Branching: spawn a new walker at current pos with heading rotated ±(100°−140°).
    // The branch inherits the parent's trail buffer so it doesn't immediately die
    // on the parent's just-marked position under the corrected self-collision check.
    if (rng() < BRANCH_PROB && walkers.length < maxWalkers) {
      // Angle: uniform [100°, 140°] = [5π/9, 7π/9]
      const branchAngle = (100 + rng() * 40) * (Math.PI / 180)
      const sign = rng() < 0.5 ? 1 : -1
      const ba = branchAngle * sign
      const bcosa = Math.cos(ba)
      const bsina = Math.sin(ba)
      // Rotate w.tan around w.pos by ba (Rodrigues)
      const bcx = w.pos.y * w.tan.z - w.pos.z * w.tan.y
      const bcy = w.pos.z * w.tan.x - w.pos.x * w.tan.z
      const bcz = w.pos.x * w.tan.y - w.pos.y * w.tan.x
      const branchTan = new Vector3(
        w.tan.x * bcosa + bcx * bsina,
        w.tan.y * bcosa + bcy * bsina,
        w.tan.z * bcosa + bcz * bsina,
      )
      spawnWalker(w.pos.clone(), branchTan, w.trail, w.trailHead)
    }

    // Sequential processing: do NOT advance wi. The current walker continues
    // to be re-processed until it dies (splice removes it, keeping wi pointing
    // at the next walker). This runs each walker to completion before the next.
  }
}

// ---------------------------------------------------------------------------
// Great-circle crack injection — used by fragmentation loop
// ---------------------------------------------------------------------------

/**
 * Mark a band of texels along the great circle defined by `normal` as cracks.
 * The great circle is the set of sphere directions `d` where |dot(d, normal)| < sinHalfWidth.
 * This directly rasters cracks into the mask without requiring walkers,
 * guaranteeing a full bisecting split of the sphere.
 *
 * To restrict injection to a specific component only (target the largest plate),
 * pass `compFilter`: texels outside the component are skipped.
 * The band width is ~3 texels wide to ensure 4-connectivity after dilation.
 */
function crackGreatCircle(
  crackMask: Uint8Array,
  normal: { x: number; y: number; z: number },
  compId: Uint16Array | null,
  targetComp: number,
  dirScratch: Vector3,
): void {
  // sinHalfWidth: half-angle of the crack band in radians.
  // ~3 texels at RES=256: texAng(256) ≈ 0.0123 rad; 2 texels ≈ 0.025 rad
  const SIN_HALF = 0.020  // ~1.1° — about 2 texels wide

  const nLen = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z)
  if (nLen < 1e-9) return
  const nx = normal.x / nLen
  const ny = normal.y / nLen
  const nz = normal.z / nLen

  for (let f = 0; f < 6; f++) {
    const faceBase = f * RES * RES
    for (let ty = 0; ty < RES; ty++) {
      for (let tx = 0; tx < RES; tx++) {
        texelToDir(f, tx, ty, RES, dirScratch)
        const dot = dirScratch.x * nx + dirScratch.y * ny + dirScratch.z * nz
        if (Math.abs(dot) < SIN_HALF) {
          const idx = faceBase + ty * RES + tx
          if (compId !== null && compId[idx] !== targetComp) continue
          crackMask[idx] = 1
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Flood fill — Step 3
// ---------------------------------------------------------------------------

/**
 * 4-neighbor dilation of the crack mask into a fresh buffer.
 * Walkers die when a 4-neighbor is already crack — BEFORE marking their own
 * texel — so every crack-on-crack termination stops one texel short of the
 * crack it hit, and flood fill leaks through that gap: a chord across a
 * region almost never actually splits it. Dilating the mask handed to flood
 * fill (the raw mask stays untouched for walker collision) closes every such
 * near-miss into a true 4-connected junction. Never dilate an already-dilated
 * mask — always derive from the raw mask, or cracks grow without bound.
 */
function dilate4(src: Uint8Array): Uint8Array {
  const dst = new Uint8Array(src)
  const nbr = { face: 0, x: 0, y: 0 }
  const rr = RES * RES
  for (let i = 0; i < TOTAL_TEXELS; i++) {
    if (src[i] !== 1) continue
    const face = (i / rr) | 0
    const rem = i - face * rr
    const y = (rem / RES) | 0
    const x = rem - y * RES
    neighborTexel(face, x, y, 1, 0, RES, nbr)
    dst[texelIndex(nbr.face, nbr.x, nbr.y, RES)] = 1
    neighborTexel(face, x, y, -1, 0, RES, nbr)
    dst[texelIndex(nbr.face, nbr.x, nbr.y, RES)] = 1
    neighborTexel(face, x, y, 0, 1, RES, nbr)
    dst[texelIndex(nbr.face, nbr.x, nbr.y, RES)] = 1
    neighborTexel(face, x, y, 0, -1, RES, nbr)
    dst[texelIndex(nbr.face, nbr.x, nbr.y, RES)] = 1
  }
  return dst
}

/**
 * 4-connected BFS over non-crack texels.
 * Returns component ID per texel (0 = unassigned, 1..N = components).
 * After BFS, crack texels are assigned to nearest component via multi-source BFS.
 */
function floodFill(crackMask: Uint8Array): { compId: Uint16Array; compCount: number } {
  const compId = new Uint16Array(TOTAL_TEXELS)  // 0 = unassigned
  const nbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

  let nextComp = 1

  // BFS queue — reuse a large pre-allocated buffer
  const queue = new Int32Array(TOTAL_TEXELS)

  // Phase 1: assign components to non-crack texels
  for (let seed = 0; seed < TOTAL_TEXELS; seed++) {
    if (crackMask[seed] !== 0 || compId[seed] !== 0) continue

    // Start a new component from this texel
    const comp = nextComp++
    compId[seed] = comp
    let head = 0, tail = 0
    queue[tail++] = seed

    while (head < tail) {
      const idx = queue[head++]
      const face = (idx / (RES * RES)) | 0
      const rem  = idx % (RES * RES)
      const y    = (rem / RES) | 0
      const x    = rem % RES

      // 4-connected neighbours
      const dx4 = [1, -1, 0, 0] as const
      const dy4 = [0, 0, 1, -1] as const
      for (let d = 0; d < 4; d++) {
        neighborTexel(face, x, y, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr)
        const ni = texelIndex(nbr.face, nbr.x, nbr.y, RES)
        if (crackMask[ni] === 0 && compId[ni] === 0) {
          compId[ni] = comp
          queue[tail++] = ni
        }
      }
    }
  }

  // Phase 2: assign crack texels to nearest component via multi-source BFS.
  // Seed the queue with all component-labelled texels adjacent to crack texels.
  {
    let head = 0, tail = 0

    // Seed: non-crack texels that have at least one crack 4-neighbour
    for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
      if (crackMask[idx] !== 0 || compId[idx] === 0) continue
      const face = (idx / (RES * RES)) | 0
      const rem  = idx % (RES * RES)
      const y    = (rem / RES) | 0
      const x    = rem % RES

      const dx4 = [1, -1, 0, 0] as const
      const dy4 = [0, 0, 1, -1] as const
      for (let d = 0; d < 4; d++) {
        neighborTexel(face, x, y, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr)
        const ni = texelIndex(nbr.face, nbr.x, nbr.y, RES)
        if (crackMask[ni] !== 0 && compId[ni] === 0) {
          // Mark crack texel with this component and enqueue
          compId[ni] = compId[idx]
          queue[tail++] = ni
        }
      }
    }

    // BFS over crack texels
    while (head < tail) {
      const idx = queue[head++]
      const face = (idx / (RES * RES)) | 0
      const rem  = idx % (RES * RES)
      const y    = (rem / RES) | 0
      const x    = rem % RES
      const comp = compId[idx]

      const dx4 = [1, -1, 0, 0] as const
      const dy4 = [0, 0, 1, -1] as const
      for (let d = 0; d < 4; d++) {
        neighborTexel(face, x, y, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr)
        const ni = texelIndex(nbr.face, nbr.x, nbr.y, RES)
        if (compId[ni] === 0) {
          compId[ni] = comp
          queue[tail++] = ni
        }
      }
    }
  }

  // Any remaining unassigned texel (degenerate — shouldn't happen) gets comp 1
  for (let i = 0; i < TOTAL_TEXELS; i++) {
    if (compId[i] === 0) compId[i] = 1
  }

  return { compId, compCount: nextComp - 1 }
}

// ---------------------------------------------------------------------------
// Merge to target — Step 4
// ---------------------------------------------------------------------------

/**
 * Compact component ids to 0..N-1 with pole plates first (id 0 = north, id 1 = south).
 * Returns: new Uint16Array with remapped ids, new compCount.
 */
function remapComponents(
  compId: Uint16Array,
  idMap: Map<number, number>,  // oldId → newId
): void {
  for (let i = 0; i < TOTAL_TEXELS; i++) {
    const mapped = idMap.get(compId[i])
    if (mapped !== undefined) compId[i] = mapped
  }
}

/**
 * Merge components to target count.
 *
 * Returns the new compCount and remaps compId in-place.
 * poleCompN / poleCompS are the component ids for north/south pole texels —
 * these two are merge-exempt as absorbees.
 *
 * Change 2: uses balanced, capped absorb-target selection.
 * When absorbing component S (smallest first), candidates = S's adjacent
 * components. Choose the candidate with the SMALLEST current area.
 * Hard cap: skip any candidate where (areaS + areaCand) / TOTAL_TEXELS > maxShare
 * — if ALL candidates exceed, choose the one minimizing the resulting area
 * (cap is soft only when unavoidable). Deterministic tie-break: lowest root id.
 */
function mergeToTarget(
  compId: Uint16Array,
  compCount: number,
  targetCount: number,
  poleCompN: number,
  poleCompS: number,
  maxShare: number,
): number {
  const nbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

  // (a) Absorb slivers: components < 0.1% of total texels
  const SLIVER_THRESH = TOTAL_TEXELS * 0.001

  let currentCount = compCount

  // Helper: compute areas (upper bound on id space = compCount)
  const computeAreas = (cc: number): Int32Array => {
    const areas = new Int32Array(cc + 1)  // 1-indexed
    for (let i = 0; i < TOTAL_TEXELS; i++) if (compId[i] <= cc) areas[compId[i]]++
    return areas
  }

  // Helper: build full adjacency map (compA → Set<compB> of all neighbors)
  // Returns Map<compA, Map<compB, boundaryCount>>
  const buildAdjacencyMap = (): Map<number, Map<number, number>> => {
    const adj = new Map<number, Map<number, number>>()
    for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
      const ca = compId[idx]
      const face = (idx / (RES * RES)) | 0
      const rem  = idx % (RES * RES)
      const y    = (rem / RES) | 0
      const x    = rem % RES
      const dx4 = [1, -1, 0, 0] as const
      const dy4 = [0, 0, 1, -1] as const
      for (let d = 0; d < 4; d++) {
        neighborTexel(face, x, y, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr)
        const ni = texelIndex(nbr.face, nbr.x, nbr.y, RES)
        const cb = compId[ni]
        if (cb !== ca) {
          let inner = adj.get(ca)
          if (!inner) { inner = new Map(); adj.set(ca, inner) }
          inner.set(cb, (inner.get(cb) ?? 0) + 1)
        }
      }
    }
    return adj
  }

  // Helper: build neighbour map (compA → LONGEST-boundary neighbour compB)
  // Used only for sliver pass (same as old behavior)
  const buildBoundaryMap = (): Map<number, { bestNeighbour: number; bestCount: number }> => {
    const adj = buildAdjacencyMap()
    const result = new Map<number, { bestNeighbour: number; bestCount: number }>()
    for (const [ca, nbrs] of adj) {
      let best = 0, bestNbr = 0
      for (const [cb, cnt] of nbrs) {
        if (cnt > best) { best = cnt; bestNbr = cb }
      }
      result.set(ca, { bestNeighbour: bestNbr, bestCount: best })
    }
    return result
  }

  // Absorb comp `src` into comp `dst`: replace all src texels with dst
  const absorb = (src: number, dst: number): void => {
    for (let i = 0; i < TOTAL_TEXELS; i++) {
      if (compId[i] === src) compId[i] = dst
    }
  }

  // (a) Sliver cleanup: absorb components < threshold into their best neighbour.
  // Slivers are tiny noise islands from the crack network — always remove them.
  // After each pass, recount currentCount from actual live IDs to avoid drift.
  for (let pass = 0; pass < 3; pass++) {
    const areas = computeAreas(compCount)  // use original compCount as upper bound for IDs
    const bmap  = buildBoundaryMap()
    let changed = false
    for (let c = 1; c <= compCount; c++) {
      if (areas[c] === 0) continue
      if (c === poleCompN || c === poleCompS) continue
      if (areas[c] < SLIVER_THRESH) {
        const nb = bmap.get(c)
        if (nb && nb.bestNeighbour > 0) {
          absorb(c, nb.bestNeighbour)
          changed = true
        }
      }
    }
    if (!changed) break
    // Recount after each pass — don't use decrements (too many slivers can make count go negative)
    const liveAfterPass = new Set<number>()
    for (let i = 0; i < TOTAL_TEXELS; i++) liveAfterPass.add(compId[i])
    liveAfterPass.delete(0)
    currentCount = liveAfterPass.size
  }

  // (b) Reduce to targetCount via union-find single-pass.
  // Balanced capped merge: when absorbing S, pick the SMALLEST adjacent candidate
  // (not the one with longest boundary). Hard cap maxShare to prevent rich-get-richer.
  // Declare pole roots here so they are accessible after the block closes.
  let poleRootN = poleCompN
  let poleRootS = poleCompS
  {
    // Build per-root area table and adjacency structure (neighbor sets updated on union)
    const areas = computeAreas(compCount)

    // Union-find: parent[i] = i initially (using compId space, 1-indexed)
    const parent = new Int32Array(compCount + 2)
    for (let i = 0; i <= compCount + 1; i++) parent[i] = i

    const find = (x: number): number => {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]]  // path compression (halving)
        x = parent[x]
      }
      return x
    }

    // Per-root area (updated on union): index by root id (1-based up to compCount)
    // areas[] already computed above; we use it as the live area table.
    // Union: attach src root into dst root; update dst area; merge neighbor sets.
    // rootNeighbors: Map<root, Set<root>> — adjacency between current roots
    const rootNeighbors = new Map<number, Set<number>>()

    // Initialize rootNeighbors from full adjacency map
    const adj = buildAdjacencyMap()
    for (const [ca, nbrs] of adj) {
      let setA = rootNeighbors.get(ca)
      if (!setA) { setA = new Set(); rootNeighbors.set(ca, setA) }
      for (const [cb] of nbrs) {
        setA.add(cb)
        let setB = rootNeighbors.get(cb)
        if (!setB) { setB = new Set(); rootNeighbors.set(cb, setB) }
        setB.add(ca)
      }
    }

    const unionRoots = (srcRoot: number, dstRoot: number): void => {
      if (srcRoot === dstRoot) return
      // Attach srcRoot into dstRoot
      parent[srcRoot] = dstRoot
      areas[dstRoot] += areas[srcRoot]
      areas[srcRoot] = 0
      // Merge neighbor sets: dstRoot gains all of srcRoot's neighbors (minus itself)
      const srcNbrs = rootNeighbors.get(srcRoot)
      if (srcNbrs) {
        let dstNbrs = rootNeighbors.get(dstRoot)
        if (!dstNbrs) { dstNbrs = new Set(); rootNeighbors.set(dstRoot, dstNbrs) }
        for (const n of srcNbrs) {
          if (n === dstRoot) continue
          const nRoot = find(n)
          if (nRoot === dstRoot) continue
          dstNbrs.add(nRoot)
          const nSet = rootNeighbors.get(nRoot)
          if (nSet) {
            nSet.delete(srcRoot)
            nSet.add(dstRoot)
          }
        }
        srcNbrs.clear()
      }
    }

    // Collect non-exempt live components sorted by area ascending
    const candidates: Array<{ id: number; area: number }> = []
    for (let c = 1; c <= compCount; c++) {
      if (areas[c] === 0) continue
      candidates.push({ id: c, area: areas[c] })
    }
    candidates.sort((a, b) => a.area - b.area)

    const capTexels = maxShare * TOTAL_TEXELS

    // Merge smallest-first until we reach targetCount.
    // For each candidate (smallest S): find adjacent roots, pick the one with
    // smallest area that keeps combined area under cap. If all exceed cap, pick
    // the one producing smallest combined area (soft cap fallback).
    for (const cand of candidates) {
      if (currentCount <= targetCount) break
      const rc = find(cand.id)
      if (areas[rc] === 0) continue  // already merged away

      // Skip exempt components as absorbers (they can receive but not be dissolved)
      if (rc === find(poleCompN) || rc === find(poleCompS)) continue

      const nbrs = rootNeighbors.get(rc)
      if (!nbrs || nbrs.size === 0) continue

      // Resolve current roots of neighbors, deduplicate
      const candidateRoots = new Set<number>()
      for (const n of nbrs) {
        const rn = find(n)
        if (rn !== rc) candidateRoots.add(rn)
      }
      if (candidateRoots.size === 0) continue

      const areaS = areas[rc]

      // Find best candidate: smallest area under cap; fallback to smallest overall
      let bestRoot = -1
      let bestArea = Infinity

      // First pass: find smallest candidate that stays under cap
      for (const rn of candidateRoots) {
        const combined = areaS + areas[rn]
        if (combined <= capTexels) {
          if (areas[rn] < bestArea || (areas[rn] === bestArea && rn < bestRoot)) {
            bestArea = areas[rn]
            bestRoot = rn
          }
        }
      }

      // Fallback: all candidates exceed cap — pick smallest combined (soft cap)
      if (bestRoot === -1) {
        let bestCombined = Infinity
        for (const rn of candidateRoots) {
          const combined = areaS + areas[rn]
          if (combined < bestCombined || (combined === bestCombined && rn < bestRoot)) {
            bestCombined = combined
            bestArea = areas[rn]
            bestRoot = rn
          }
        }
      }

      if (bestRoot === -1) continue

      // Determine which root absorbs which: src (rc) merges INTO dst (bestRoot).
      // Pole roots may absorb (receive) but never be dissolved.
      const poleN = find(poleCompN)
      const poleS = find(poleCompS)
      const srcRoot = rc
      const dstRoot = bestRoot

      // If srcRoot is a pole root, it can only receive — skip (already guarded above).
      // If dstRoot is a pole root, it absorbs srcRoot (fine — pole grows by receiving).
      void poleN; void poleS

      unionRoots(srcRoot, dstRoot)
      currentCount--
    }

    // Capture canonical roots for the pole components before find() goes out of scope.
    poleRootN = find(poleCompN)
    poleRootS = find(poleCompS)

    // Apply union-find: scan compId once, replace with find(root)
    for (let i = 0; i < TOTAL_TEXELS; i++) {
      compId[i] = find(compId[i])
    }

    // Recount
    const liveFinal = new Set<number>()
    for (let i = 0; i < TOTAL_TEXELS; i++) liveFinal.add(compId[i])
    liveFinal.delete(0)
    currentCount = liveFinal.size
  }

  // Compact ids: collect ALL surviving distinct component IDs from compId.
  // We cannot assume IDs are contiguous in [1, currentCount] — absorb() can
  // create non-contiguous ID sets when dest IDs exceed currentCount.
  // Instead, scan compId to find all live IDs, then build the idMap.
  const liveIds = new Set<number>()
  for (let i = 0; i < TOTAL_TEXELS; i++) liveIds.add(compId[i])
  liveIds.delete(0)  // 0 is a sentinel (unassigned), should not appear but guard anyway

  const idMap = new Map<number, number>()

  // Degenerate case: both poles share the same component.
  let nextId = 1
  if (poleRootN === poleRootS) {
    idMap.set(poleRootN, nextId++)
  } else {
    idMap.set(poleRootN, nextId++)
    idMap.set(poleRootS, nextId++)
  }

  // Add remaining live IDs in sorted order for determinism
  const sortedOthers = Array.from(liveIds)
    .filter(c => c !== poleRootN && c !== poleRootS)
    .sort((a, b) => a - b)
  for (const c of sortedOthers) {
    idMap.set(c, nextId++)
  }

  remapComponents(compId, idMap)

  // Now ids are 1-based; subtract 1 to get 0-based
  for (let i = 0; i < TOTAL_TEXELS; i++) compId[i]--

  return idMap.size
}

// ---------------------------------------------------------------------------
// Pole-of-inaccessibility helper
// ---------------------------------------------------------------------------

/**
 * Find the texel with maximum distToBoundary within a given plate.
 * Returns the direction of that texel.
 */
function poleOfInaccessibility(
  plateId: number,
  compId: Uint16Array,
  distField: Float32Array,
  out: Vector3,
): void {
  let maxDist = -1
  let bestIdx = 0

  for (let i = 0; i < TOTAL_TEXELS; i++) {
    if (compId[i] === plateId && distField[i] > maxDist) {
      maxDist = distField[i]
      bestIdx = i
    }
  }

  const face = (bestIdx / (RES * RES)) | 0
  const rem  = bestIdx % (RES * RES)
  const y    = (rem / RES) | 0
  const x    = rem % RES
  texelToDir(face, x, y, RES, out)
}

// ---------------------------------------------------------------------------
// Binary heap for Dijkstra
// ---------------------------------------------------------------------------

/** Minimal binary min-heap for (priority, index) pairs. */
class BinaryHeap {
  private keys: Float32Array
  private vals: Int32Array
  private size: number
  private capacity: number

  constructor(capacity: number) {
    this.capacity = capacity
    this.keys = new Float32Array(capacity)
    this.vals = new Int32Array(capacity)
    this.size = 0
  }

  isEmpty(): boolean { return this.size === 0 }

  push(key: number, val: number): void {
    if (this.size === this.capacity) {
      // Grow: reallocate at 1.5× capacity and copy existing data
      const newCap = Math.ceil(this.capacity * 1.5)
      const newKeys = new Float32Array(newCap)
      const newVals = new Int32Array(newCap)
      newKeys.set(this.keys)
      newVals.set(this.vals)
      this.keys = newKeys
      this.vals = newVals
      this.capacity = newCap
    }
    const i = this.size++
    this.keys[i] = key
    this.vals[i] = val
    this._bubbleUp(i)
  }

  popMin(): { key: number; val: number } {
    const key = this.keys[0]
    const val = this.vals[0]
    const last = --this.size
    if (last > 0) {
      this.keys[0] = this.keys[last]
      this.vals[0] = this.vals[last]
      this._siftDown(0)
    }
    return { key, val }
  }

  private _bubbleUp(i: number): void {
    while (i > 0) {
      const p = (i - 1) >> 1
      if (this.keys[p] <= this.keys[i]) break
      // Swap
      let tmp = this.keys[p]; this.keys[p] = this.keys[i]; this.keys[i] = tmp
      let tmpv = this.vals[p]; this.vals[p] = this.vals[i]; this.vals[i] = tmpv
      i = p
    }
  }

  private _siftDown(i: number): void {
    const n = this.size
    for (;;) {
      const l = 2 * i + 1
      const r = 2 * i + 2
      let smallest = i
      if (l < n && this.keys[l] < this.keys[smallest]) smallest = l
      if (r < n && this.keys[r] < this.keys[smallest]) smallest = r
      if (smallest === i) break
      let tmp = this.keys[i]; this.keys[i] = this.keys[smallest]; this.keys[smallest] = tmp
      let tmpv = this.vals[i]; this.vals[i] = this.vals[smallest]; this.vals[smallest] = tmpv
      i = smallest
    }
  }
}

// ---------------------------------------------------------------------------
// Distance field — Step 6
// ---------------------------------------------------------------------------

/**
 * Build distance-to-boundary field using Dijkstra's algorithm.
 *
 * Choice rationale: Dijkstra with a binary heap is deterministic (tie-breaking
 * by texel index), runs in O(N log N), and naturally handles cross-face geometry
 * without any special edge-exchange passes. A chamfer/two-pass approach would
 * need iterated face exchanges to converge across faces — more code, same
 * asymptotic cost, less obvious correctness. Dijkstra wins on simplicity and
 * determinism.
 *
 * Determinism: initial sources are processed in texel-index order (the boundary
 * detection loop is sequential). The heap always returns the minimum-key entry;
 * ties are broken by the texel index stored in `val` (smaller index wins since
 * equal-priority entries are inserted in scan order). The output is thus a pure
 * function of compId.
 *
 * 8-connected steps: orthogonal cost = TEX_ANG, diagonal cost = TEX_ANG * sqrt(2).
 * This gives a good approximation to geodesic distance to the nearest boundary.
 *
 * Also fills neighborId: for each texel, nearestBoundary gives the source texel
 * whose neighbourhood contains a different-plate texel; neighborId is the first
 * distinct plateId found there (in deterministic scan order).
 */
function buildDistanceField(
  compId: Uint16Array,
  plateCount: number,
): {
  distField: Float32Array
  neighborId: Uint16Array
  nearestBoundary: Uint32Array
} {
  const INF = 1e30

  const distField     = new Float32Array(TOTAL_TEXELS).fill(INF)
  const neighborId    = new Uint16Array(TOTAL_TEXELS)
  const nearestBoundary = new Uint32Array(TOTAL_TEXELS)
  const settled       = new Uint8Array(TOTAL_TEXELS)

  // Heap capacity: each texel can be pushed at most once per relaxation
  // (we use a lazy deletion pattern — settled[] guards re-processing)
  // Upper bound: TOTAL_TEXELS relaxations.
  const heap = new BinaryHeap(TOTAL_TEXELS * 2)

  const nbr:  { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
  const nbr2: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

  const DIAG_COST = TEX_ANG * Math.SQRT2
  const ORTH_COST = TEX_ANG

  // Seed: boundary texels — those with a 4-neighbour of different plateId
  for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
    const face = (idx / (RES * RES)) | 0
    const rem  = idx % (RES * RES)
    const y    = (rem / RES) | 0
    const x    = rem % RES
    const pid  = compId[idx]

    const dx4 = [1, -1, 0, 0] as const
    const dy4 = [0, 0, 1, -1] as const
    for (let d = 0; d < 4; d++) {
      neighborTexel(face, x, y, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr)
      const ni = texelIndex(nbr.face, nbr.x, nbr.y, RES)
      if (compId[ni] !== pid) {
        // Boundary texel: zero distance, source is itself
        if (distField[idx] > 0) {
          distField[idx] = 0
          nearestBoundary[idx] = idx
          heap.push(0, idx)
        }
        break
      }
    }
  }

  // Dijkstra — 8-connected expansion
  const dx8 = [ 1, -1,  0,  0,  1, -1,  1, -1] as const
  const dy8 = [ 0,  0,  1, -1,  1,  1, -1, -1] as const

  while (!heap.isEmpty()) {
    const { key: dist, val: idx } = heap.popMin()

    if (settled[idx]) continue
    settled[idx] = 1

    const face = (idx / (RES * RES)) | 0
    const rem  = idx % (RES * RES)
    const y    = (rem / RES) | 0
    const x    = rem % RES

    for (let d = 0; d < 8; d++) {
      neighborTexel(face, x, y, dx8[d] as -1|0|1, dy8[d] as -1|0|1, RES, nbr)
      const ni = texelIndex(nbr.face, nbr.x, nbr.y, RES)
      if (settled[ni]) continue

      const cost = (dx8[d] !== 0 && dy8[d] !== 0) ? DIAG_COST : ORTH_COST
      const newDist = dist + cost

      if (newDist < distField[ni]) {
        distField[ni] = newDist
        nearestBoundary[ni] = nearestBoundary[idx]
        heap.push(newDist, ni)
      }
    }
  }

  // Build neighborId: for each texel, inspect its nearestBoundary source and
  // its 4-neighbours to find the first distinct plateId (deterministic scan order).
  void plateCount  // used implicitly through compId
  for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
    const ownPlate = compId[idx]
    const src = nearestBoundary[idx]

    const srcFace = (src / (RES * RES)) | 0
    const srcRem  = src % (RES * RES)
    const srcY    = (srcRem / RES) | 0
    const srcX    = srcRem % RES

    let found = ownPlate  // fallback: own plate (degenerate, convergence ≈ 0)

    // Check the source boundary texel itself
    if (compId[src] !== ownPlate) {
      found = compId[src]
    } else {
      // Check its 4-neighbours in fixed order
      const dx4 = [1, -1, 0, 0] as const
      const dy4 = [0, 0, 1, -1] as const
      for (let d = 0; d < 4; d++) {
        neighborTexel(srcFace, srcX, srcY, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr2)
        const ni = texelIndex(nbr2.face, nbr2.x, nbr2.y, RES)
        if (compId[ni] !== ownPlate) {
          found = compId[ni]
          break
        }
      }
    }

    neighborId[idx] = found
  }

  return { distField, neighborId, nearestBoundary }
}

// ---------------------------------------------------------------------------
// Tectonics — public class
// ---------------------------------------------------------------------------

export class Tectonics {
  readonly plates: Plate[]

  // Cube-map tables (read-only after construction)
  private readonly _compId:     Uint16Array   // per-texel plate id (0-based)
  private readonly _distField:  Float32Array  // per-texel distance to boundary (radians)
  private readonly _neighborId: Uint16Array   // per-texel neighbour plate id

  // Domain-warp noise
  private readonly _warpNoise: (x: number, y: number, z: number) => number

  // Preallocated scratch — zero allocations in query() / velocityAt()
  private readonly _warpedDir  = new Vector3()
  private readonly _vA         = new Vector3()
  private readonly _vB         = new Vector3()
  private readonly _smoothScratch = new Vector3()

  // Texel lookup scratch (reused in query)
  private readonly _tex = { face: 0, x: 0, y: 0 }

  // 4 nudge-directions for gradient computation in query()
  private readonly _gDir0 = new Vector3()
  private readonly _gDir1 = new Vector3()
  private readonly _gDir2 = new Vector3()
  private readonly _gDir3 = new Vector3()
  private readonly _gTex0 = { face: 0, x: 0, y: 0 }
  private readonly _gTex1 = { face: 0, x: 0, y: 0 }
  private readonly _gTex2 = { face: 0, x: 0, y: 0 }
  private readonly _gTex3 = { face: 0, x: 0, y: 0 }
  private readonly _tHat  = new Vector3()

  constructor(opts: { seed: number; plateCount?: number }) {
    const { seed, plateCount = 16 } = opts
    const target = plateCount  // EXACT target plate count

    // Sub-seed assignments (preserving original 0-6 for plate properties):
    //   0 = (old) site jitter — now unused; kept to preserve stream layout
    //   1 = plate type assignment
    //   2 = plate elevation
    //   3 = plate omega axis
    //   4 = plate omega speed
    //   5 = warp noise
    //   6 = (old) plate size bias — now unused; kept to preserve stream layout
    //   7 = crack walker seeds (main round)
    //   8 = crack walker seeds (additional rounds)
    this._warpNoise = createNoise3D(deriveSeed(seed, 5))

    const typeRng  = makeRng(deriveSeed(seed, 1))
    const elevRng  = makeRng(deriveSeed(seed, 2))
    const axisRng  = makeRng(deriveSeed(seed, 3))
    const speedRng = makeRng(deriveSeed(seed, 4))

    // -------------------------------------------------------------------------
    // Step 1 — Trace cracks
    // -------------------------------------------------------------------------

    const crackMask  = new Uint8Array(TOTAL_TEXELS)  // 1 = crack
    const dirScratch = new Vector3()
    const texScratch = { face: 0, x: 0, y: 0 }

    const K = Math.max(8, Math.round(target * 2.0))
    const maxW = 8 * K

    const walkerRng = makeRng(deriveSeed(seed, 7))
    traceCracks(crackMask, walkerRng, K, maxW, dirScratch, texScratch)

    // Circumpolar ring cracks at ±75° latitude. Walkers never enter the 10°
    // polar caps, so no random closed loop can separate a cap from the
    // mid-latitudes — without these rings both poles land in one giant
    // component on most seeds (debugged: 29/36 longitude sectors crack-free
    // at lat 78-82°). The rings guarantee distinct north/south polar plates
    // and pre-fragment the dominant component so the count loop converges.
    // Query-time domain warp (±~4.6°) wiggles them, so they don't render as
    // perfect circles — cf. Earth's circum-Antarctic ridge ring.
    const RING_LO = Math.sin((74 * Math.PI) / 180)
    const RING_HI = Math.sin((76 * Math.PI) / 180)
    for (let f = 0; f < 6; f++) {
      for (let ty = 0; ty < RES; ty++) {
        for (let tx = 0; tx < RES; tx++) {
          texelToDir(f, tx, ty, RES, dirScratch)
          const ay = Math.abs(dirScratch.y)
          if (ay >= RING_LO && ay <= RING_HI) crackMask[texelIndex(f, tx, ty, RES)] = 1
        }
      }
    }

    // -------------------------------------------------------------------------
    // Step 2 — Size-driven fragmentation loop
    // -------------------------------------------------------------------------

    // maxShare: the largest single plate must not exceed this fraction of the sphere.
    // For very small targets (e.g. target=4), no configuration can keep all plates
    // under 22%, so we relax the cap proportionally.
    const MAX_SHARE_BASE = 0.22
    const maxShare = Math.max(MAX_SHARE_BASE, 1.6 / target)

    // Count only "significant" components (area > SLIVER_THRESH) to avoid
    // counting tiny noise islands from the crack network as separate plates.
    const SLIVER_THRESH_COUNT = TOTAL_TEXELS * 0.001  // 0.1% of sphere

    const computeComponentStats = (cId: Uint16Array, cc: number): {
      significantCount: number
      largestShare: number
      largestComp: number
    } => {
      const areas = new Int32Array(cc + 2)
      for (let i = 0; i < TOTAL_TEXELS; i++) if (cId[i] <= cc) areas[cId[i]]++
      let sig = 0
      let largestArea = 0
      let largestComp = 1
      for (let c = 1; c <= cc; c++) {
        if (areas[c] >= SLIVER_THRESH_COUNT) {
          sig++
          if (areas[c] > largestArea) {
            largestArea = areas[c]
            largestComp = c
          }
        }
      }
      return {
        significantCount: sig,
        largestShare: largestArea / TOTAL_TEXELS,
        largestComp,
      }
    }

    let fillResult = floodFill(dilate4(crackMask))
    let compCount  = fillResult.compCount
    let compId     = fillResult.compId

    const extraRng = makeRng(deriveSeed(seed, 8))

    for (let rounds = 0; rounds < 24; rounds++) {
      const { significantCount, largestShare, largestComp } = computeComponentStats(compId, compCount)

      // Done when both criteria met
      if (significantCount >= target && largestShare <= maxShare) break

      if (largestShare > maxShare) {
        // Dominant component too large: bisect it with great-circle crack lines
        // restricted to texels inside the component, then add targeted random cracks.
        // Great-circle cuts through the giant are guaranteed bisectors — unlike random
        // walkers that mostly create peninsulas.  We inject ceil(target*0.5) great
        // circles through the giant per round (one per seed), each with a random
        // normal direction sampled from extraRng (deterministic).
        const numCuts = Math.ceil(target * 0.5)
        for (let c = 0; c < numCuts; c++) {
          // Random great-circle normal via rejection sampling from extraRng
          let gx = 0, gy = 0, gz = 0
          for (let attempt = 0; attempt < 20; attempt++) {
            const rx = extraRng() * 2 - 1
            const ry = extraRng() * 2 - 1
            const rz = extraRng() * 2 - 1
            const rlen = Math.sqrt(rx * rx + ry * ry + rz * rz)
            if (rlen > 0.01 && rlen < 1.0) {
              gx = rx / rlen; gy = ry / rlen; gz = rz / rlen
              break
            }
          }
          if (gx === 0 && gy === 0 && gz === 0) continue
          crackGreatCircle(crackMask, { x: gx, y: gy, z: gz }, compId, largestComp, dirScratch)
        }
        // Also add walker cracks restricted to the giant for irregular boundaries
        const currentCompId = compId  // capture current round's table
        const extraK = Math.ceil(target * 0.5)
        traceCracks(crackMask, extraRng, extraK, maxW, dirScratch, texScratch, {
          spawnFilter: (idx: number) => currentCompId[idx] === largestComp,
        })
      } else {
        // Count deficit only: add cracks anywhere (no spawn restriction)
        const extraK = Math.ceil(target * 0.6)
        traceCracks(crackMask, extraRng, extraK, maxW, dirScratch, texScratch)
      }

      const refill = floodFill(dilate4(crackMask))
      compCount = refill.compCount
      compId    = refill.compId
    }

    // -------------------------------------------------------------------------
    // Step 3 — Identify pole components (before merge to protect them)
    // -------------------------------------------------------------------------

    // North pole: dir = (0, 1, 0)
    dirScratch.set(0, 1, 0)
    dirToTexel(dirScratch, RES, texScratch)
    const northIdx = texelIndex(texScratch.face, texScratch.x, texScratch.y, RES)
    const poleCompN = compId[northIdx]

    // South pole: dir = (0, -1, 0)
    dirScratch.set(0, -1, 0)
    dirToTexel(dirScratch, RES, texScratch)
    const southIdx = texelIndex(texScratch.face, texScratch.x, texScratch.y, RES)
    const poleCompS = compId[southIdx]

    // -------------------------------------------------------------------------
    // Step 4 — Merge to target
    // -------------------------------------------------------------------------

    const finalCount = mergeToTarget(compId, compCount, target, poleCompN, poleCompS, maxShare)

    // After mergeToTarget, ids are 0-based:
    //   id 0 = north pole plate
    //   id 1 = south pole plate
    //   ids 2..finalCount-1 = other plates

    // -------------------------------------------------------------------------
    // Step 6 — Distance + neighbour fields (before Step 5 so seedDir is available)
    // -------------------------------------------------------------------------

    const { distField, neighborId } = buildDistanceField(compId, finalCount)

    this._compId     = compId
    this._distField  = distField
    this._neighborId = neighborId

    // -------------------------------------------------------------------------
    // Step 5 — Plate properties
    // -------------------------------------------------------------------------

    const n = finalCount
    const PHI_GOLD = 0.61803398875

    // Type assignment: plate 0 forced oceanic, plate 1 forced continental, rest random
    const types: Array<'oceanic' | 'continental'> = []
    for (let i = 0; i < n; i++) {
      let type: 'oceanic' | 'continental'
      if (i === 0) {
        type = 'oceanic'
      } else if (i === 1) {
        type = 'continental'
      } else {
        type = typeRng() < 0.60 ? 'oceanic' : 'continental'
      }
      types.push(type)
    }

    const plates: Plate[] = []

    for (let i = 0; i < n; i++) {
      const type = types[i]

      // baseElevation: oceanic [-0.55, -0.3], continental [0.06, 0.28]
      const baseElevation = type === 'oceanic'
        ? -0.55 + elevRng() * 0.25
        :  0.06 + elevRng() * 0.22

      // omega: random unit axis × speed in [0.4, 1.0]
      const ax = axisRng() * 2 - 1
      const ay = axisRng() * 2 - 1
      const az = axisRng() * 2 - 1
      const axisLen = Math.sqrt(ax * ax + ay * ay + az * az) || 1
      const speed = 0.4 + speedRng() * 0.6
      const omega = new Vector3(
        ax / axisLen * speed,
        ay / axisLen * speed,
        az / axisLen * speed,
      )

      // Color: golden-ratio hue spacing
      const hue = ((i * PHI_GOLD) % 1 + 1) % 1
      const color = type === 'oceanic'
        ? hslToRgb(hue, 0.55, 0.34)
        : hslToRgb(hue, 0.60, 0.55)

      // seedDir = pole of inaccessibility (texel with max dist-to-boundary in this plate)
      const seedDir = new Vector3()
      poleOfInaccessibility(i, compId, distField, seedDir)

      plates.push({
        id: i,
        seedDir,
        type,
        baseElevation,
        omega,
        color,
      })
    }

    this.plates = plates
  }

  // ---------------------------------------------------------------------------
  // query — hot path, zero-alloc
  // ---------------------------------------------------------------------------

  /**
   * Query plate ownership and boundary info at surface point `dir` (unit vector).
   * Zero allocations: all scratch state is preallocated on `this`.
   *
   * Steps:
   *   1. Domain-warp dir → warped direction (same warp as before).
   *   2. Look up plateId + neighborId from cube-map tables.
   *   3. boundaryDist = sampleSmooth(distField, warped) — smooth radians.
   *   4. convergence = clamp(dot(vA−vB, tHat) / 2, −1, 1) where tHat is the
   *      direction of steepest descent of the distance field (toward boundary),
   *      computed via central-difference on _distField at ±1-texel neighbors
   *      along the face U and V axes (4 direct array reads, no interpolation).
   *
   * tHat sign convention: steepest DESCENT of distance = toward the boundary.
   * At the boundary, dist=0; toward plate interior, dist > 0. So the gradient
   * ∇dist points INTO the plate. tHat = −∇dist (normalised) points TOWARD the
   * boundary and through it toward the neighbour. convergence = dot(vA−vB, tHat):
   * positive means vA pushes toward the boundary (converging), negative means
   * diverging — exactly the same semantics as the old Voronoi code.
   */
  query(dir: Vector3, out: TectonicQuery): TectonicQuery {
    const wn = this._warpNoise
    const w  = this._warpedDir

    // --- Domain-warp the query direction ---
    const dx = fbm(wn, dir.x + WARP_O1.x, dir.y + WARP_O1.y, dir.z + WARP_O1.z,
      { octaves: WARP_OCTAVES, frequency: WARP_FREQ })
    const dy = fbm(wn, dir.x + WARP_O2.x, dir.y + WARP_O2.y, dir.z + WARP_O2.z,
      { octaves: WARP_OCTAVES, frequency: WARP_FREQ })
    const dz = fbm(wn, dir.x + WARP_O3.x, dir.y + WARP_O3.y, dir.z + WARP_O3.z,
      { octaves: WARP_OCTAVES, frequency: WARP_FREQ })
    w.set(
      dir.x + WARP_AMP * dx,
      dir.y + WARP_AMP * dy,
      dir.z + WARP_AMP * dz,
    ).normalize()

    // --- Table lookups ---
    dirToTexel(w, RES, this._tex)
    const idx       = texelIndex(this._tex.face, this._tex.x, this._tex.y, RES)
    const plateId   = this._compId[idx]
    // Clamp neighborId to valid plate range in case of degenerate Dijkstra output
    // (isolated texels with nearestBoundary=0 can produce out-of-range values).
    const neighborId = Math.min(this._neighborId[idx], this.plates.length - 1)

    // --- Smooth boundary distance ---
    const boundaryDist = sampleSmooth(this._distField, w, RES, this._smoothScratch)

    // --- Gradient-based tHat (unwarped dir) ---
    // Use UNWARPED dir so velocity computation lives in the same space as the gradient.
    //
    // Strategy: central-difference on distField over ±1 texel neighbours along
    // face U and V axes. Convert to world-space by subtracting neighbour directions
    // (finite-difference approximation of the tangent-basis vectors). Project the
    // result onto the tangent plane, negate → tHat points toward the boundary
    // (steepest DESCENT of distField, which increases toward the plate interior).
    dirToTexel(dir, RES, this._tex)
    const { face, x: tx, y: ty } = this._tex

    const _nt0 = this._gTex0
    const _nt1 = this._gTex1
    const _nt2 = this._gTex2
    const _nt3 = this._gTex3

    // Neighbour texels: ±U (±1, 0), ±V (0, ±1) — cross-face safe via neighborTexel
    neighborTexel(face, tx, ty,  1,  0, RES, _nt0)
    neighborTexel(face, tx, ty, -1,  0, RES, _nt1)
    neighborTexel(face, tx, ty,  0,  1, RES, _nt2)
    neighborTexel(face, tx, ty,  0, -1, RES, _nt3)

    const d0 = this._distField[texelIndex(_nt0.face, _nt0.x, _nt0.y, RES)]
    const d1 = this._distField[texelIndex(_nt1.face, _nt1.x, _nt1.y, RES)]
    const d2 = this._distField[texelIndex(_nt2.face, _nt2.x, _nt2.y, RES)]
    const d3 = this._distField[texelIndex(_nt3.face, _nt3.x, _nt3.y, RES)]

    // Central-difference scalar gradient along U and V axes
    const gradU = (d0 - d1) * 0.5  // d/dU * (1 texel)
    const gradV = (d2 - d3) * 0.5  // d/dV * (1 texel)

    // World-space axis vectors via finite-difference of texel-centre directions.
    // uAxis = dir(+1,0) - dir(-1,0) ≈ 2 * tangent_U (up to projection distortion).
    texelToDir(_nt0.face, _nt0.x, _nt0.y, RES, this._gDir0)
    texelToDir(_nt1.face, _nt1.x, _nt1.y, RES, this._gDir1)
    texelToDir(_nt2.face, _nt2.x, _nt2.y, RES, this._gDir2)
    texelToDir(_nt3.face, _nt3.x, _nt3.y, RES, this._gDir3)

    const uax = this._gDir0.x - this._gDir1.x
    const uay = this._gDir0.y - this._gDir1.y
    const uaz = this._gDir0.z - this._gDir1.z
    const vax = this._gDir2.x - this._gDir3.x
    const vay = this._gDir2.y - this._gDir3.y
    const vaz = this._gDir2.z - this._gDir3.z

    // World-space gradient vector (not yet normalised, not projected)
    const gwx = gradU * uax + gradV * vax
    const gwy = gradU * uay + gradV * vay
    const gwz = gradU * uaz + gradV * vaz

    // Project onto tangent plane at dir (remove radial component)
    const gdot = gwx * dir.x + gwy * dir.y + gwz * dir.z
    const ptx = gwx - gdot * dir.x
    const pty = gwy - gdot * dir.y
    const ptz = gwz - gdot * dir.z

    const gLen = Math.sqrt(ptx * ptx + pty * pty + ptz * ptz)

    let convergence = 0
    if (gLen > 1e-7) {
      // tHat = −∇dist / |∇dist|, i.e. toward boundary (steepest descent)
      this._tHat.set(-ptx / gLen, -pty / gLen, -ptz / gLen)

      // Velocities using UNWARPED dir (unchanged from old code)
      const omegaA = this.plates[plateId].omega
      const omegaB = this.plates[neighborId].omega
      const px = dir.x, py = dir.y, pz = dir.z

      this._vA.set(
        omegaA.y * pz - omegaA.z * py,
        omegaA.z * px - omegaA.x * pz,
        omegaA.x * py - omegaA.y * px,
      )
      this._vB.set(
        omegaB.y * pz - omegaB.z * py,
        omegaB.z * px - omegaB.x * pz,
        omegaB.x * py - omegaB.y * px,
      )

      const diffX = this._vA.x - this._vB.x
      const diffY = this._vA.y - this._vB.y
      const diffZ = this._vA.z - this._vB.z

      const raw = (diffX * this._tHat.x + diffY * this._tHat.y + diffZ * this._tHat.z) / 2
      convergence = Math.max(-1, Math.min(1, raw))
    }

    out.plateId      = plateId
    out.neighborId   = neighborId
    out.boundaryDist = boundaryDist
    out.convergence  = convergence
    return out
  }

  // ---------------------------------------------------------------------------
  // velocityAt — zero-alloc, UNWARPED dir
  // ---------------------------------------------------------------------------

  /**
   * Tangent surface velocity of the owning plate at surface point `dir`: ω × dir.
   * Zero allocations: result written directly into `out`.
   * Uses unwarped dir (unchanged from original).
   */
  velocityAt(dir: Vector3, out: Vector3): Vector3 {
    // Find owning plate by direct table lookup (warped direction for consistency)
    const wn = this._warpNoise
    const w  = this._warpedDir

    const dx = fbm(wn, dir.x + WARP_O1.x, dir.y + WARP_O1.y, dir.z + WARP_O1.z,
      { octaves: WARP_OCTAVES, frequency: WARP_FREQ })
    const dy = fbm(wn, dir.x + WARP_O2.x, dir.y + WARP_O2.y, dir.z + WARP_O2.z,
      { octaves: WARP_OCTAVES, frequency: WARP_FREQ })
    const dz = fbm(wn, dir.x + WARP_O3.x, dir.y + WARP_O3.y, dir.z + WARP_O3.z,
      { octaves: WARP_OCTAVES, frequency: WARP_FREQ })
    w.set(
      dir.x + WARP_AMP * dx,
      dir.y + WARP_AMP * dy,
      dir.z + WARP_AMP * dz,
    ).normalize()

    dirToTexel(w, RES, this._tex)
    const idx   = texelIndex(this._tex.face, this._tex.x, this._tex.y, RES)
    const bestIdx = this._compId[idx]

    const omega = this.plates[bestIdx].omega
    const ddx = dir.x, ddy = dir.y, ddz = dir.z
    out.set(
      omega.y * ddz - omega.z * ddy,
      omega.z * ddx - omega.x * ddz,
      omega.x * ddy - omega.y * ddx,
    )
    return out
  }
}
