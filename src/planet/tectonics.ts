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
} from './cubemap'
// (RADIUS / RADIUS_REF imports removed — volcano constants are angular, scale-invariant)

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
  shear: number         // tangential (along-boundary) relative velocity component, ≈[-1,1]
  crustDist: number     // signed distance: +inside crust, -outside crust (radians)
  paleoDist: number     // radians — distance to nearest fossil/paleo crack boundary
  otherCrustDist: number  // crustDist mirrored to neighbor side (coarse, no fine warp)
  baseElevation: number   // blurred plate base elevation (smooth across Voronoi boundaries)
  rockHardness: number    // [0,1] substrate hardness: 1=hard igneous/volcanic, 0=soft sediment/basin
}

export interface Volcano {
  pos: Vector3
  baseRadius: number        // angular radius in radians
  height: number            // normalized, added to terrain
  craterRadiusFrac: number  // fraction of baseRadius for caldera
  kind: 'arc' | 'hotspot'  // NEW — arc = subduction front, hotspot = mantle plume
  intensity: number         // NEW — [0,1] relative intensity
}

export interface Hotspot {
  pos: Vector3
  intensity: number
  isChain: boolean
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

const RES = 512
const TOTAL_TEXELS = 6 * RES * RES

// Pre-compute texel-angle constant at RES
const TEX_ANG = texAng(RES)

// --- arc volcano constants ---
// Planet RADIUS = 50 km (world units). Angular radius θ rad → arc length R·θ = 50·θ km.
// baseRadius range: 0.018 rad → 50 * 0.018 = 0.90 km; 0.045 rad → 50 * 0.045 = 2.25 km. ✓
const ARC_CONV_THRESH   = 0.14    // convergence threshold; effective = ARC_CONV_THRESH / arcDensity
const ARC_OFFSET        = 0.020   // rad — volcanic-front offset onto overriding plate
const ARC_BAND          = 0.006   // rad — half-width of front band
const ARC_CRUST_WIDTH   = 0.090   // rad — arc-crust band half-width (wider than ARC_BAND so volcanoes land on crust) [was 0.035 — bumped so arc-land strip survives blur smoothing; at RES=512: TEX_ANG≈0.003rad, blur σ∝√(passes)·cellSize; 0.090→30 texels half-width, comfortably positive after 12-pass blur]
const ARC_SPACING       = 0.045   // rad — true min spacing (dart-throwing); effective = ARC_SPACING / arcDensity
const ARC_MAX           = 2000    // safety cap on total volcanoes
const ARC_SIZE_SPAN     = 0.25    // convergence span above threshold for convStrength ramp

// --- hotspot volcano constants ---
const HOTSPOT_V_MOVING   = 0.15          // |ω×pos| above this → plate is moving → chain
const HOTSPOT_CHAIN_STEP = 0.035         // rad between chain cones
const HOTSPOT_POLAR_COS  = Math.cos((8 * Math.PI) / 180)  // reject within 8° of poles
const HOTSPOT_BASE_R     = 0.06          // baseline angular radius (comparable to arc baseline)
const HOTSPOT_BASE_H     = 0.6           // baseline height (tall enough to breach sea level)

// --- volcano roughening (break the perfect-cone look; fixed-frequency, LOD-invariant) ---
// These perturb the analytic cone with 3D noise to produce irregular flanks,
// non-circular base footprints, and erosion-style gullies (barrancos).
// All frequencies chosen so that adjacent terrain samples (~30-120m apart on a 50km-radius
// planet, i.e. angular delta ~0.0006-0.0024 rad) see smooth noise variation.
const VOLC_BASE_WARP        = 0.32   // ± fractional wobble of the cone footprint radius
const VOLC_BASE_WARP_FREQ   = 14.0   // spatial freq of the base-radius warp noise (on unit sphere)
const VOLC_FLANK_ROUGH      = 0.22   // flank height roughness amplitude (fraction of local cone height)
const VOLC_FLANK_FREQ       = 26.0   // spatial freq of flank roughness fbm (on unit sphere)

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
// 0-6  = plate properties (same as before)
// 7    = crack walker seeds
// 8    = additional crack round seeds (for the count loop)
// 9    = crustNoise (already used)
// 10   = rpRng (already used)
// 11   = microRng (already used)
// 12   = paleo crack walker seeds (NEW)
// 13   = fine warp noise sub-seed (NEW)
// 15   = non-tectonic continental mask noise (NEW)
// 16   = hotspot position RNG (NEW)
// 17   = hotspot intensity RNG (NEW)
// 18   = volcano roughening noise (NEW)
// 19   = rock hardness FBM break-up noise (NEW)

// ---------------------------------------------------------------------------
// Non-tectonic continental mask constants (tuned to hit land-fraction gate)
// ---------------------------------------------------------------------------
const CONT_FREQ   = 1.1    // frequency of continental noise fbm
const CONT_THRESH = 0.10   // signed fbm threshold: c > CONT_THRESH → land

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
// Bake-time field smoothing — box blur (4+4 weights: center×4, neighbor×1 each)
// ---------------------------------------------------------------------------

/**
 * One-pass 4-neighbor weighted box blur of a Float32Array cubemap field.
 * Center weight 4, each cardinal neighbor weight 1 → total weight 8.
 * Reduces C0 creases at texel edges introduced by raw Dijkstra output.
 */
function blurField(src: Float32Array): Float32Array {
  const dst = new Float32Array(TOTAL_TEXELS)
  const nbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
  const dx4 = [1, -1, 0, 0] as const
  const dy4 = [0, 0, 1, -1] as const
  for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
    const face = (idx / (RES * RES)) | 0
    const rem  = idx % (RES * RES)
    const y    = (rem / RES) | 0
    const x    = rem % RES
    let sum = src[idx] * 4  // center weight 4
    for (let d = 0; d < 4; d++) {
      neighborTexel(face, x, y, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, nbr)
      sum += src[texelIndex(nbr.face, nbr.x, nbr.y, RES)]  // each neighbor weight 1
    }
    dst[idx] = sum / 8  // total weight 4 + 4×1 = 8
  }
  return dst
}

/** Apply blurField n times. Used to smooth the Dijkstra crust SDF's TEX_ANG staircase. */
function blurFieldN(src: Float32Array, n: number): Float32Array {
  let r = src
  for (let i = 0; i < n; i++) r = blurField(r)
  return r
}

// ---------------------------------------------------------------------------
// bakeUpliftField — wide-smoothed signed convergence for broad deformation
// ---------------------------------------------------------------------------
/**
 * Produces a wide-smoothed copy of convField for broad-belt deformation zones.
 * Algorithm: downsample 512² → 64² per face, apply 3 box-blur passes at 64²,
 * then bilinear-upsample back to 512². This creates a deformation zone ~8x the
 * cell width cheaply. Pure arithmetic, fully deterministic.
 */
function bakeUpliftField(convField: Float32Array): Float32Array {
  // Direct full-resolution smooth: convField is already 12-pass blurred.
  // Apply 3 more passes at 512² via the cross-face-correct blurField.
  // Target sigma: ~3 cells ≈ 600 m — sharp belts, not 2.5 km blobs.
  return blurFieldN(convField, 3)
}

// ---------------------------------------------------------------------------
// bakeConvShear — pre-bake per-texel convergence and shear
// ---------------------------------------------------------------------------

/**
 * Compute per-texel convergence and shear from plate omegas and the distance-
 * field gradient. Called once at construction time; the resulting arrays are
 * blurred before being stored to eliminate the gradient-direction sign-flips
 * that cause cliff-like discontinuities at sub-texel scale.
 *
 * The gradient is computed with a ±1-texel central-difference stencil on the
 * already-blurred `distField`. Blurring the output (3 passes) further smooths
 * saddle-point reversal artefacts.
 */
function bakeConvShear(
  distField: Float32Array,
  compId: Uint16Array,
  neighborId: Uint16Array,
  omegaX: Float64Array,
  omegaY: Float64Array,
  omegaZ: Float64Array,
  plateCount: number,
): { convField: Float32Array; shearField: Float32Array } {
  const convField  = new Float32Array(TOTAL_TEXELS)
  const shearField = new Float32Array(TOTAL_TEXELS)

  const nt0: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
  const nt1: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
  const nt2: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
  const nt3: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
  const dir0  = new Vector3()
  const dir1  = new Vector3()
  const dir2  = new Vector3()
  const dir3  = new Vector3()
  const cDir  = new Vector3()

  for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
    const face = (idx / (RES * RES)) | 0
    const rem  = idx % (RES * RES)
    const ty   = (rem / RES) | 0
    const tx   = rem % RES

    // Texel-centre direction
    texelToDir(face, tx, ty, RES, cDir)

    // Central-difference stencil ±1 texel
    neighborTexel(face, tx, ty,  1,  0, RES, nt0)
    neighborTexel(face, tx, ty, -1,  0, RES, nt1)
    neighborTexel(face, tx, ty,  0,  1, RES, nt2)
    neighborTexel(face, tx, ty,  0, -1, RES, nt3)

    const d0 = distField[texelIndex(nt0.face, nt0.x, nt0.y, RES)]
    const d1 = distField[texelIndex(nt1.face, nt1.x, nt1.y, RES)]
    const d2 = distField[texelIndex(nt2.face, nt2.x, nt2.y, RES)]
    const d3 = distField[texelIndex(nt3.face, nt3.x, nt3.y, RES)]

    texelToDir(nt0.face, nt0.x, nt0.y, RES, dir0)
    texelToDir(nt1.face, nt1.x, nt1.y, RES, dir1)
    texelToDir(nt2.face, nt2.x, nt2.y, RES, dir2)
    texelToDir(nt3.face, nt3.x, nt3.y, RES, dir3)

    const gradU = (d0 - d1) * 0.5
    const gradV = (d2 - d3) * 0.5
    const uax = dir0.x - dir1.x, uay = dir0.y - dir1.y, uaz = dir0.z - dir1.z
    const vax = dir2.x - dir3.x, vay = dir2.y - dir3.y, vaz = dir2.z - dir3.z

    const gwx = gradU * uax + gradV * vax
    const gwy = gradU * uay + gradV * vay
    const gwz = gradU * uaz + gradV * vaz

    // Project onto tangent plane at cDir
    const gdot = gwx * cDir.x + gwy * cDir.y + gwz * cDir.z
    const ptx = gwx - gdot * cDir.x
    const pty = gwy - gdot * cDir.y
    const ptz = gwz - gdot * cDir.z
    const gLen = Math.sqrt(ptx * ptx + pty * pty + ptz * ptz)

    if (gLen < 1e-7) continue  // boundary texel or isolated island — leave at 0

    const tHatX = -ptx / gLen
    const tHatY = -pty / gLen
    const tHatZ = -ptz / gLen

    const pid = compId[idx]
    const nid = Math.min(neighborId[idx], plateCount - 1)
    const cx = cDir.x, cy = cDir.y, cz = cDir.z

    const vAx = omegaY[pid] * cz - omegaZ[pid] * cy
    const vAy = omegaZ[pid] * cx - omegaX[pid] * cz
    const vAz = omegaX[pid] * cy - omegaY[pid] * cx
    const vBx = omegaY[nid] * cz - omegaZ[nid] * cy
    const vBy = omegaZ[nid] * cx - omegaX[nid] * cz
    const vBz = omegaX[nid] * cy - omegaY[nid] * cx

    const diffX = vAx - vBx
    const diffY = vAy - vBy
    const diffZ = vAz - vBz

    const rawConv = (diffX * tHatX + diffY * tHatY + diffZ * tHatZ) / 2
    convField[idx] = Math.max(-1, Math.min(1, rawConv))

    const bx = tHatY * cDir.z - tHatZ * cDir.y
    const by = tHatZ * cDir.x - tHatX * cDir.z
    const bz = tHatX * cDir.y - tHatY * cDir.x
    const bLen = Math.sqrt(bx * bx + by * by + bz * bz)
    if (bLen > 1e-7) {
      const rawShear = (diffX * bx / bLen + diffY * by / bLen + diffZ * bz / bLen) / 2
      shearField[idx] = Math.max(-1, Math.min(1, rawShear))
    }
  }

  // 12 blur passes — eliminates sign-flip artefacts at gradient saddle points.
  // Blur radius ∝ √(passes)·cellSize. To preserve the same angular smoothing
  // when cell size halves (RES=512 vs RES=256), passes must scale 4×: 3→12.
  // (6 passes at half-cell gives only √(6/3)·0.5 ≈ 0.71× the prior σ — not enough.)
  return {
    convField:  blurFieldN(convField,  12),
    shearField: blurFieldN(shearField, 12),
  }
}

// ---------------------------------------------------------------------------
// Serialisation wire types
// ---------------------------------------------------------------------------

export interface PlateWire {
  id: number
  seedDir: [number, number, number]
  type: 'oceanic' | 'continental'
  baseElevation: number
  omega: [number, number, number]
  color: [number, number, number]
}

export interface TectonicsBaked {
  res: number
  seed: number
  arcDensity: number
  hotspotCount: number
  hotspotIntensity: number
  plates: PlateWire[]
  compId: Uint16Array
  distField: Float32Array
  neighborId: Uint16Array
  crustDist: Float32Array
  paleoDist: Float32Array
  convField: Float32Array
  shearField: Float32Array
  upliftField: Float32Array
  baseElevField: Float32Array
  volPos: Float32Array
  volBaseRadius: Float32Array
  volHeight: Float32Array
  volCraterFrac: Float32Array
  volKind: Uint8Array          // 0=arc, 1=hotspot per volcano
  volIntensity: Float32Array   // intensity per volcano
  volBucketRes: number
  volBucketStart: Int32Array
  volBucketIdx: Int32Array
  hotspotPos: Float32Array     // 3 per hotspot, xyz interleaved
  hotspotIntensityArr: Float32Array
  hotspotIsChain: Uint8Array
  rockHardness: Float32Array   // [0,1] per texel: substrate hardness (composition-contrast-scaled)
}

// ---------------------------------------------------------------------------
// Tectonics — public class
// ---------------------------------------------------------------------------

export class Tectonics {
  readonly plates: Plate[]

  // Seed and arcDensity stored for toBaked() / fromBaked() round-trip
  private readonly _seed: number
  private readonly _arcDensity: number
  private readonly _hotspotCount: number
  private readonly _hotspotIntensity: number
  private readonly _composition: number

  // Cube-map tables (read-only after construction)
  private readonly _compId:     Uint16Array   // per-texel plate id (0-based)
  private readonly _distField:  Float32Array  // per-texel distance to boundary (radians)
  private readonly _neighborId: Uint16Array   // per-texel neighbour plate id
  private readonly _crustDist:  Float32Array  // per-texel signed crust SDF (radians)
  private readonly _paleoDist:  Float32Array  // per-texel distance to paleo crack network (radians)
  private readonly _convField:  Float32Array  // per-texel baked convergence (stable, blurred)
  private readonly _shearField: Float32Array  // per-texel baked shear (stable, blurred)
  private readonly _upliftField: Float32Array  // per-texel baked wide uplift (blurred convField)
  private _baseElevField: Float32Array        // per-texel blurred plate base elevation (smooth across Voronoi boundaries)
  private _crustAge:      Float32Array        // per-texel [0,1] age proxy: 0=fresh divergent, 1=old craton
  private _rockHardness:  Float32Array        // per-texel [0,1] substrate hardness (composition-scaled contrast)

  // Domain-warp noise
  private readonly _warpNoise: (x: number, y: number, z: number) => number
  // Fine-warp noise for coastline complexity (stream 13)
  private readonly _fineWarpNoise: (x: number, y: number, z: number) => number
  // Volcano roughening noise (stream 18) — irregular base + flank + gullies
  private readonly _volNoise: (x: number, y: number, z: number) => number
  // Rock hardness FBM break-up noise (stream 19) — breaks up smooth hardness iso-contours
  private readonly _hardnessNoise: (x: number, y: number, z: number) => number

  // Preallocated scratch — zero allocations in query() / velocityAt()
  private readonly _warpedDir  = new Vector3()
  private readonly _smoothScratch = new Vector3()
  private readonly _crustScratch  = new Vector3()
  private readonly _fineWarpScratch = new Vector3()
  private readonly _fineCrustScratch = new Vector3()
  private readonly _fineDistScratch = new Vector3()

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
  private readonly _otherCrustDir = new Vector3()  // scratch for mirrored crust probe

  // Volcano bake products — populated last in constructor
  private _volcanoes: Volcano[] = []
  private _hotspots:  Hotspot[] = []
  // Typed flat arrays for zero-alloc hot path (volcanoElevation)
  private _volPos:        Float32Array = new Float32Array(0)  // 3*nVol, xyz interleaved
  private _volBaseRadius: Float32Array = new Float32Array(0)
  private _volHeight:     Float32Array = new Float32Array(0)
  private _volCraterFrac: Float32Array = new Float32Array(0)
  // Cube-map bucket grid for spatial index (CSR)
  private _volBucketRes: number = 0
  private _volBucketStart: Int32Array = new Int32Array(0)  // length nCells+1
  private _volBucketIdx:   Int32Array = new Int32Array(0)  // concatenated volcano indices
  // Timing exposed for harness diagnostics (set after _bakeVolcanoes completes)
  _volcanoBakeMs: number = 0

  // Scratch for volcanoElevation hot path (zero-alloc)
  private readonly _volScratchTexel = { face: 0, x: 0, y: 0 }
  // Scratch for _bakeVolcanoes (one-time; preallocated to avoid GC churn during bake)
  private readonly _bakeQueryScratch: TectonicQuery = {
    plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0,
    shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0,
  }
  private readonly _bakeDirScratch = new Vector3()

  /** Public accessor: the baked volcanic arc candidates. */
  get volcanoes(): readonly Volcano[] { return this._volcanoes }

  /** Public accessor: the baked mantle-plume hotspot sources. */
  get hotspots(): readonly Hotspot[] { return this._hotspots }

  constructor(opts: { seed: number; plateCount?: number; arcDensity?: number; hotspotCount?: number; hotspotIntensity?: number; driftScale?: number; composition?: number }) {
    const _tBakeStart = performance.now()
    const { seed, plateCount = 16, arcDensity: arcDensityRaw = 1.0, hotspotCount: hotspotCountRaw = 6, hotspotIntensity: hotspotIntensityRaw = 1, driftScale = 1.0, composition: compositionRaw = 0.5 } = opts
    // Clamp composition to [0,1]: 0 = icy/sediment (soft, low contrast), 1 = rocky/basaltic (hard, high contrast)
    const composition = Math.max(0, Math.min(1, compositionRaw))
    const arcDensity = Math.max(0.2, Math.min(3, arcDensityRaw))
    // Store seed and clamped arcDensity for toBaked() / fromBaked() round-trip
    this._seed             = seed
    this._arcDensity       = arcDensity
    this._hotspotCount     = Math.max(0, Math.min(20, Math.round(hotspotCountRaw)))
    this._hotspotIntensity = Math.max(0, Math.min(3, hotspotIntensityRaw))
    this._composition      = composition
    // Initialise hardness arrays to defaults; overwritten by tectonic path below
    this._crustAge     = new Float32Array(TOTAL_TEXELS).fill(0.5)
    this._rockHardness = new Float32Array(TOTAL_TEXELS).fill(0.5)
    // Clamp plateCount to [0, 48]; values ≤ 1 trigger the non-tectonic branch
    const plateCountClamped = Math.max(0, Math.min(48, plateCount))
    const target = plateCountClamped  // EXACT target plate count (for tectonic path)

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
    //  12 = paleo crack walker seeds
    //  13 = fine warp noise
    //  15 = non-tectonic continental mask noise (NEW)
    //  16 = hotspot position RNG (NEW)
    //  17 = hotspot intensity RNG (NEW)
    //  18 = volcano roughening noise (NEW)
    this._warpNoise = createNoise3D(deriveSeed(seed, 5))
    this._fineWarpNoise = createNoise3D(deriveSeed(seed, 13))
    this._volNoise = createNoise3D(deriveSeed(seed, 18))
    this._hardnessNoise = createNoise3D(deriveSeed(seed, 19))

    // Non-tectonic branch: single plate, noise-driven crust SDF, no boundaries
    if (plateCountClamped <= 1) {
      const nt = Tectonics._computeNonTectonic(seed)
      this.plates       = nt.plates
      this._compId      = nt.compId
      this._neighborId  = nt.neighborId
      this._distField   = nt.distField
      this._convField   = nt.convField
      this._shearField  = nt.shearField
      this._upliftField = new Float32Array(TOTAL_TEXELS)  // all zeros — no convergence
      this._baseElevField = new Float32Array(TOTAL_TEXELS)  // all zeros — single plate, no variation
      this._crustDist   = nt.crustDist
      this._paleoDist   = nt.paleoDist
      // Bake hardness field (no divergent boundaries → uniform age + FBM break-up)
      this._bakeHardness(composition)
      const _t0 = performance.now()
      this._bakeVolcanoes(seed, arcDensity, this._hotspotCount, this._hotspotIntensity)
      this._volcanoBakeMs = performance.now() - _t0
      console.log(`tectonic bake ${(performance.now() - _tBakeStart).toFixed(1)}ms`)
      return
    }

    // typeRng (stream 1): consumed only inside the crust bake via typeRng2 (parallel instance).
    // elevRng (stream 2): consumed in Step 5.
    // axisRng (stream 3) and speedRng (stream 4): consumed once in the omega pre-derivation
    // inside the crust bake section — that is the SINGLE authoritative site for both.
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
    this._distField  = blurField(blurField(distField))
    this._neighborId = neighborId

    // -------------------------------------------------------------------------
    // Step 6a — Omega pre-derivation + convergence/shear field bake (HOISTED)
    // -------------------------------------------------------------------------
    // Omega arrays are derived ONCE here (axisRng/speedRng consumed exactly once,
    // same order as before). bakeConvShear is called here instead of after Step 5
    // so that _convField is available during the crust-mask bake (arc-crust pass).
    const plateOmX = new Float64Array(finalCount)
    const plateOmY = new Float64Array(finalCount)
    const plateOmZ = new Float64Array(finalCount)
    for (let i = 0; i < finalCount; i++) {
      const ax = axisRng() * 2 - 1
      const ay = axisRng() * 2 - 1
      const az = axisRng() * 2 - 1
      const axisLen = Math.sqrt(ax * ax + ay * ay + az * az) || 1
      const speed = (0.4 + speedRng() * 0.6) * driftScale
      plateOmX[i] = ax / axisLen * speed
      plateOmY[i] = ay / axisLen * speed
      plateOmZ[i] = az / axisLen * speed
    }
    {
      const { convField, shearField } = bakeConvShear(
        this._distField, this._compId, this._neighborId,
        plateOmX, plateOmY, plateOmZ, finalCount,
      )
      this._convField  = convField
      this._shearField = shearField
    }
    this._upliftField = bakeUpliftField(this._convField)

    // -------------------------------------------------------------------------
    // Step 6b — Crust mask bake + crust SDF
    // -------------------------------------------------------------------------

    const CRUST_A    = 0.60  // angular falloff weight
    const CRUST_B    = 0.20  // boundary dist weight
    const CRUST_C    = 0.45  // noise weight
    const CRUST_T    = 0.44  // threshold
    const CRUST_LEAN = 0.25  // leading-edge lean factor
    const CRUST_RP_LO = 0.55 // R_P lower bound factor
    const CRUST_RP_HI = 0.95 // R_P upper bound factor

    const crustMask  = new Uint8Array(TOTAL_TEXELS)
    const crustNoise = createNoise3D(deriveSeed(seed, 9))
    const rpRng      = makeRng(deriveSeed(seed, 10))
    const microRng   = makeRng(deriveSeed(seed, 11))

    // --- Per-plate area share ---
    const plateAreaShare = new Float64Array(finalCount)
    for (let t = 0; t < TOTAL_TEXELS; t++) plateAreaShare[compId[t]]++
    for (let i = 0; i < finalCount; i++) plateAreaShare[i] /= TOTAL_TEXELS

    // --- Continental plates: compute R_P and anchorDir, then mask ---
    // We need type info: re-derive using the same typeRng stream order.
    // Instead, compute this after types[] is known (Step 5 runs after us now).
    // To avoid ordering chicken-and-egg, we replicate the type determination inline.
    // (typeRng is consumed before this code so we must NOT call typeRng here.)
    // We store the crust bake for-now in a local array, applied per-type below.
    // Plan: bake mask in two passes — first pass sets up per-plate data, second writes mask.
    // We derive types identically to Step 5 (same stream 1) using a parallel rng.
    const typeRng2 = makeRng(deriveSeed(seed, 1))
    const typesPre: Array<'oceanic' | 'continental'> = []
    for (let i = 0; i < finalCount; i++) {
      if (i === 0) { typesPre.push('oceanic') }
      else if (i === 1) { typesPre.push('continental') }
      else { typesPre.push(typeRng2() < 0.60 ? 'oceanic' : 'continental') }
    }

    // --- Per-plate R_P and anchorDir ---
    const plateRp = new Float64Array(finalCount)
    const plateAnchorX = new Float64Array(finalCount)
    const plateAnchorY = new Float64Array(finalCount)
    const plateAnchorZ = new Float64Array(finalCount)
    // Also collect microcontinent data per oceanic plate
    const plateMicroR = new Float64Array(finalCount)   // 0 = no micro
    const plateMicroX = new Float64Array(finalCount)
    const plateMicroY = new Float64Array(finalCount)
    const plateMicroZ = new Float64Array(finalCount)

    // Precompute seedDir (pole of inaccessibility) per plate using already-built distField
    const plateSeedX = new Float64Array(finalCount)
    const plateSeedY = new Float64Array(finalCount)
    const plateSeedZ = new Float64Array(finalCount)
    {
      const tmpDir = new Vector3()
      for (let i = 0; i < finalCount; i++) {
        poleOfInaccessibility(i, compId, distField, tmpDir)
        plateSeedX[i] = tmpDir.x
        plateSeedY[i] = tmpDir.y
        plateSeedZ[i] = tmpDir.z
      }
    }

    // plateOmX/Y/Z are derived earlier (Step 6a) and remain in scope here.
    for (let i = 0; i < finalCount; i++) {
      if (typesPre[i] === 'continental') {
        const area = plateAreaShare[i]
        const rp = Math.min(CRUST_RP_HI, Math.max(CRUST_RP_LO, CRUST_RP_LO * Math.sqrt(area / 0.0625))) * (CRUST_RP_LO + (CRUST_RP_HI - CRUST_RP_LO) * rpRng())
        plateRp[i] = rp
        // anchorDir = normalize(seedDir + LEAN * velDir) where velDir = omega × seedDir
        const sx = plateSeedX[i], sy = plateSeedY[i], sz = plateSeedZ[i]
        const ox = plateOmX[i], oy = plateOmY[i], oz = plateOmZ[i]
        let vx = oy * sz - oz * sy
        let vy = oz * sx - ox * sz
        let vz = ox * sy - oy * sx
        const vLen = Math.sqrt(vx * vx + vy * vy + vz * vz)
        if (vLen < 1e-6) {
          plateAnchorX[i] = sx; plateAnchorY[i] = sy; plateAnchorZ[i] = sz
        } else {
          vx /= vLen; vy /= vLen; vz /= vLen
          const ax2 = sx + CRUST_LEAN * vx
          const ay2 = sy + CRUST_LEAN * vy
          const az2 = sz + CRUST_LEAN * vz
          const aLen = Math.sqrt(ax2 * ax2 + ay2 * ay2 + az2 * az2) || 1
          plateAnchorX[i] = ax2 / aLen; plateAnchorY[i] = ay2 / aLen; plateAnchorZ[i] = az2 / aLen
        }
      } else {
        // Oceanic: no continental crust, but maybe a microcontinent
        // Consume rpRng consistently (even for oceanic)
        rpRng()
        if (microRng() < 0.22) {
          // Find interior seed point: scan all texels of this plate with distField[t] > 0.05
          const candidates: number[] = []
          for (let t = 0; t < TOTAL_TEXELS; t++) {
            if (compId[t] === i && distField[t] > 0.05) candidates.push(t)
          }
          if (candidates.length > 0) {
            const pick = candidates[Math.floor(microRng() * candidates.length)]
            const mFace = (pick / (RES * RES)) | 0
            const mRem  = pick % (RES * RES)
            const mY    = (mRem / RES) | 0
            const mX    = mRem % RES
            const mDir  = new Vector3()
            texelToDir(mFace, mX, mY, RES, mDir)
            plateMicroX[i] = mDir.x; plateMicroY[i] = mDir.y; plateMicroZ[i] = mDir.z
            plateMicroR[i] = 0.04 + microRng() * 0.06
          }
        }
      }
    }

    // --- Apply crust mask ---
    {
      const tmpDir    = new Vector3()
      const ARC_CONV_THRESH_EFF = ARC_CONV_THRESH / this._arcDensity
      for (let t = 0; t < TOTAL_TEXELS; t++) {
        const face = (t / (RES * RES)) | 0
        const rem  = t % (RES * RES)
        const ty2  = (rem / RES) | 0
        const tx2  = rem % RES
        texelToDir(face, tx2, ty2, RES, tmpDir)
        const tx = tmpDir.x, ty3 = tmpDir.y, tz = tmpDir.z
        const pid = compId[t]

        if (typesPre[pid] === 'continental') {
          const rp = plateRp[pid]
          const ax2 = plateAnchorX[pid], ay2 = plateAnchorY[pid], az2 = plateAnchorZ[pid]
          const dotA = Math.max(-1, Math.min(1, tx * ax2 + ty3 * ay2 + tz * az2))
          const angle = Math.acos(dotA)
          const falloff = 1 - _ss(0, rp, angle)
          const bdist = distField[t]
          const bdistSS = _ss(0, 0.10, bdist)
          const fbmVal = fbm(crustNoise, tx * 1.3, ty3 * 1.3, tz * 1.3, { octaves: 5 }) * 0.5 + 0.5
          const score = CRUST_A * falloff + CRUST_B * bdistSS + CRUST_C * fbmVal
          if (score > CRUST_T) crustMask[t] = 1
        } else {
          // Oceanic: microcontinent only
          const mr = plateMicroR[pid]
          if (mr > 0) {
            const mx = plateMicroX[pid], my = plateMicroY[pid], mz = plateMicroZ[pid]
            const dotM = Math.max(-1, Math.min(1, tx * mx + ty3 * my + tz * mz))
            const mAngle = Math.acos(dotM)
            const fbmVal3 = fbm(crustNoise, tx * 3, ty3 * 3, tz * 3, { octaves: 3 }) * 0.5 + 0.5
            const threshold = mr * (0.7 + 0.5 * fbmVal3)
            if (mAngle < threshold) crustMask[t] = 1
          }
        }

        // --- Arc-crust pass ---
        // Mark texels along oceanic-convergent boundaries (volcanic front, overriding
        // side) as continental-type crust so volcanoes land on solid ground and the
        // surrounding area becomes coastal plain/shelf.  Runs regardless of own plate
        // type so OO arcs are also covered.  Only adds texels — never clears them.
        if (!crustMask[t]) {
          const convArc = this._sampleC1(this._convField, tmpDir)
          if (convArc > ARC_CONV_THRESH_EFF) {
            const bd = this._distField[t]
            if (bd >= ARC_OFFSET - ARC_CRUST_WIDTH && bd <= ARC_OFFSET + ARC_CRUST_WIDTH) {
              const nid = Math.min(neighborId[t], finalCount - 1)
              // Overriding-side gate (mirrors _bakeVolcanoes polarity):
              //   OC: own = continental, neighbour = oceanic
              //   OO: both oceanic, own id > neighbour id (arc-side convention)
              const ownType = typesPre[pid]
              const nbrType = typesPre[nid]
              const isOC = ownType === 'continental' && nbrType === 'oceanic'
              const isOO = ownType === 'oceanic'     && nbrType === 'oceanic' && pid > nid
              if (isOC || isOO) {
                crustMask[t] = 1
              }
            }
          }
        }
      }
    }

    // --- Crust SDF via Dijkstra ---
    let totalCrust = 0
    for (let t = 0; t < TOTAL_TEXELS; t++) if (crustMask[t]) totalCrust++

    const crustDist = new Float32Array(TOTAL_TEXELS)
    if (totalCrust === 0) {
      crustDist.fill(-1.5)
    } else if (totalCrust === TOTAL_TEXELS) {
      crustDist.fill(1.5)
    } else {
      const INF2 = 1e30
      const cDist = new Float32Array(TOTAL_TEXELS).fill(INF2)
      const cSettled = new Uint8Array(TOTAL_TEXELS)
      const cHeap = new BinaryHeap(TOTAL_TEXELS * 2)
      const cNbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

      // Seed: crust edge texels (crustMask differs from any 4-neighbour)
      for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
        const face = (idx / (RES * RES)) | 0
        const rem  = idx % (RES * RES)
        const cy   = (rem / RES) | 0
        const cx   = rem % RES
        const mine = crustMask[idx]
        const dx4 = [1, -1, 0, 0] as const
        const dy4 = [0, 0, 1, -1] as const
        for (let d = 0; d < 4; d++) {
          neighborTexel(face, cx, cy, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, cNbr)
          const ni = texelIndex(cNbr.face, cNbr.x, cNbr.y, RES)
          if (crustMask[ni] !== mine) {
            if (cDist[idx] > 0) {
              cDist[idx] = 0
              cHeap.push(0, idx)
            }
            break
          }
        }
      }

      const CDIAG = TEX_ANG * Math.SQRT2
      const CORTH = TEX_ANG
      const dx8c = [ 1, -1,  0,  0,  1, -1,  1, -1] as const
      const dy8c = [ 0,  0,  1, -1,  1,  1, -1, -1] as const

      while (!cHeap.isEmpty()) {
        const { key: cdistVal, val: cidx } = cHeap.popMin()
        if (cSettled[cidx]) continue
        cSettled[cidx] = 1
        const face = (cidx / (RES * RES)) | 0
        const rem  = cidx % (RES * RES)
        const cy   = (rem / RES) | 0
        const cx   = rem % RES
        for (let d = 0; d < 8; d++) {
          neighborTexel(face, cx, cy, dx8c[d] as -1|0|1, dy8c[d] as -1|0|1, RES, cNbr)
          const ni = texelIndex(cNbr.face, cNbr.x, cNbr.y, RES)
          if (cSettled[ni]) continue
          const cost = (dx8c[d] !== 0 && dy8c[d] !== 0) ? CDIAG : CORTH
          const newDist2 = cdistVal + cost
          if (newDist2 < cDist[ni]) {
            cDist[ni] = newDist2
            cHeap.push(newDist2, ni)
          }
        }
      }

      for (let t = 0; t < TOTAL_TEXELS; t++) {
        crustDist[t] = crustMask[t] === 1 ? cDist[t] : -cDist[t]
      }
    }

    // Light blur removes the Dijkstra 8-connected diagonal artifacts. (The texel staircase
    // that caused blocky coastlines is fixed by BILINEAR sampling at query time, not blur.)
    // 3 passes intentional — staircase smoothing is delegated to _sampleC1 C1 interpolation,
    // not extra blur. Don't raise this to match conv/shear; the mechanisms differ.
    this._crustDist = blurFieldN(crustDist, 3)

    // -------------------------------------------------------------------------
    // Step 6c — Paleo-orogen network (fossil plate boundaries)
    // -------------------------------------------------------------------------
    // Trace a SECOND independent crack network using new seeds (streams 12+).
    // These represent ancient sutures / paleo-orogens (Urals, Appalachians, etc).
    // Walkers may roam freely including near poles — no polar kill in spawning,
    // but the walker loop's existing polar kill (POLAR_COS) still applies so
    // walkers that wander into polar caps still die. We achieve "allow polar
    // walkers" by using a spawn filter that skips the SEED_POLAR_COS rejection
    // (we don't pass spawnFilter — default allows polar seeds).
    // Actually: traceCracks already rejects polar SEEDS via SEED_POLAR_COS but
    // allows walkers to move freely. For paleo we want polar seeds too — so we
    // need a dedicated spawn attempt. Since traceCracks always rejects polar seeds
    // (hardcoded), we live with that limitation and rely on the broad seed count
    // (14 seeds × wide budgets) to get good polar coverage from mid-lat seeds.
    const paleoMask = new Uint8Array(TOTAL_TEXELS)
    const paleoRng  = makeRng(deriveSeed(seed, 12))
    const PALEO_SEED_COUNT = 14
    const PALEO_MAX_WALKERS = 8 * PALEO_SEED_COUNT
    traceCracks(paleoMask, paleoRng, PALEO_SEED_COUNT, PALEO_MAX_WALKERS, dirScratch, texScratch)

    // Dilate4 the paleo mask, then multi-source Dijkstra → _paleoDist
    const paleoDilated = dilate4(paleoMask)

    // Multi-source Dijkstra from all dilated paleo crack texels
    const paleoDist = new Float32Array(TOTAL_TEXELS)
    {
      const INF3 = 1e30
      const pDist    = new Float32Array(TOTAL_TEXELS).fill(INF3)
      const pSettled = new Uint8Array(TOTAL_TEXELS)
      const pHeap    = new BinaryHeap(TOTAL_TEXELS * 2)
      const pNbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

      // Seed: all dilated paleo crack texels
      for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
        if (paleoDilated[idx] === 1) {
          pDist[idx] = 0
          pHeap.push(0, idx)
        }
      }

      const PDIAG = TEX_ANG * Math.SQRT2
      const PORTH = TEX_ANG
      const pdx8 = [ 1, -1,  0,  0,  1, -1,  1, -1] as const
      const pdy8 = [ 0,  0,  1, -1,  1,  1, -1, -1] as const

      while (!pHeap.isEmpty()) {
        const { key: pd, val: pidx } = pHeap.popMin()
        if (pSettled[pidx]) continue
        pSettled[pidx] = 1
        const face = (pidx / (RES * RES)) | 0
        const rem  = pidx % (RES * RES)
        const py   = (rem / RES) | 0
        const px   = rem % RES
        for (let d = 0; d < 8; d++) {
          neighborTexel(face, px, py, pdx8[d] as -1|0|1, pdy8[d] as -1|0|1, RES, pNbr)
          const ni = texelIndex(pNbr.face, pNbr.x, pNbr.y, RES)
          if (pSettled[ni]) continue
          const cost = (pdx8[d] !== 0 && pdy8[d] !== 0) ? PDIAG : PORTH
          const nd = pd + cost
          if (nd < pDist[ni]) {
            pDist[ni] = nd
            pHeap.push(nd, ni)
          }
        }
      }

      // Replace INF with a large but finite cap so consumers can safely clamp
      for (let t = 0; t < TOTAL_TEXELS; t++) {
        paleoDist[t] = pDist[t] === INF3 ? Math.PI : pDist[t]
      }
    }

    // 1 pass (blurField) intentional — staircase smoothing delegated to _sampleC1 C1 sampling.
    this._paleoDist = blurField(paleoDist)

    // -------------------------------------------------------------------------
    // Step 5 — Plate properties
    // -------------------------------------------------------------------------
    // Types, omegas, and seedDirs are already derived (typesPre, plateOmX/Y/Z, plateSeedX/Y/Z).
    // We consume elevRng here (stream 2 — interleaved per-plate as before).

    const n = finalCount
    const PHI_GOLD = 0.61803398875

    const plates: Plate[] = []

    for (let i = 0; i < n; i++) {
      const type = typesPre[i]

      // baseElevation: compressed to ±0.12 modifier range (crust SDF drives land/ocean split)
      const baseElevation = -0.12 + elevRng() * 0.24   // → [-0.12, +0.12]

      // omega: reuse the single-source derivation computed before the crust bake
      const omega = new Vector3(plateOmX[i], plateOmY[i], plateOmZ[i])

      // Color: golden-ratio hue spacing
      const hue = ((i * PHI_GOLD) % 1 + 1) % 1
      const color = type === 'oceanic'
        ? hslToRgb(hue, 0.55, 0.34)
        : hslToRgb(hue, 0.60, 0.55)

      // seedDir: reuse pre-computed pole of inaccessibility
      const seedDir = new Vector3(plateSeedX[i], plateSeedY[i], plateSeedZ[i])

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

    // -------------------------------------------------------------------------
    // Step 5b — Bake smooth base elevation field
    // -------------------------------------------------------------------------
    // Nearest-neighbor baseElevation reads (plates[compId[t]].baseElevation) cause
    // a hard ~30 m cliff at Voronoi plate boundaries. Blurring the field converts
    // the pixel-sharp step into a smooth ~1 km gradient that eliminates the seam.
    // K=6 at 512²: each cell ≈ 177 m → 6 passes ramps transition over ~1 km,
    // far less than plate scale (tens of km) so plate-to-plate contrast is preserved.
    {
      const baseElevRaw = new Float32Array(TOTAL_TEXELS)
      for (let i = 0; i < TOTAL_TEXELS; i++) {
        baseElevRaw[i] = plates[this._compId[i]].baseElevation
      }
      this._baseElevField = blurFieldN(baseElevRaw, 6)
    }

    // -------------------------------------------------------------------------
    // Step 7 — convergence + shear fields (hoisted to Step 6a; already assigned)
    // -------------------------------------------------------------------------
    // bakeConvShear was called in Step 6a (before the crust-mask bake) so that
    // _convField is available for the arc-crust pass. Nothing to do here.

    // -------------------------------------------------------------------------
    // Step 7b — Bake substrate hardness field (reads _convField, _distField, _crustDist)
    // -------------------------------------------------------------------------
    this._bakeHardness(composition)

    // -------------------------------------------------------------------------
    // Step 8 — Bake arc volcanoes (LAST — reads _convField/_crustDist/_distField)
    // -------------------------------------------------------------------------
    const _t0 = performance.now();
    this._bakeVolcanoes(seed, arcDensity, this._hotspotCount, this._hotspotIntensity);
    this._volcanoBakeMs = performance.now() - _t0;
    console.log(`tectonic bake ${(performance.now() - _tBakeStart).toFixed(1)}ms`)
  }

  // ---------------------------------------------------------------------------
  // _buildNonTectonic — stagnant-lid / single-plate regime
  // Called from constructor when plateCount <= 1. Assigns all required fields.
  // ---------------------------------------------------------------------------

  /**
   * Compute all field arrays for the non-tectonic (stagnant-lid) regime.
   * Returns a plain object; the constructor assigns them to readonly fields.
   * Static because readonly fields can only be assigned in the constructor.
   */
  private static _computeNonTectonic(seed: number): {
    plates: Plate[]
    compId: Uint16Array
    neighborId: Uint16Array
    distField: Float32Array
    convField: Float32Array
    shearField: Float32Array
    crustDist: Float32Array
    paleoDist: Float32Array
  } {

    // ---- 1. Continental mask from noise (stream 15) ----
    const contNoise = createNoise3D(deriveSeed(seed, 15))
    const crustMask = new Uint8Array(TOTAL_TEXELS)
    const dirScratch = new Vector3()
    const texScratch = { face: 0, x: 0, y: 0 }

    for (let face = 0; face < 6; face++) {
      for (let ty = 0; ty < RES; ty++) {
        for (let tx = 0; tx < RES; tx++) {
          texelToDir(face, tx, ty, RES, dirScratch)
          const c = fbm(contNoise, dirScratch.x * CONT_FREQ, dirScratch.y * CONT_FREQ, dirScratch.z * CONT_FREQ, { octaves: 5 })
          const t = texelIndex(face, tx, ty, RES)
          crustMask[t] = c > CONT_THRESH ? 1 : 0
        }
      }
    }

    // ---- 2. Crust SDF via Dijkstra (same machinery as Step 6b) ----
    let totalCrust = 0
    for (let t = 0; t < TOTAL_TEXELS; t++) if (crustMask[t]) totalCrust++

    const crustDistRaw = new Float32Array(TOTAL_TEXELS)
    if (totalCrust === 0) {
      crustDistRaw.fill(-1.5)
    } else if (totalCrust === TOTAL_TEXELS) {
      crustDistRaw.fill(1.5)
    } else {
      const INF2 = 1e30
      const cDist = new Float32Array(TOTAL_TEXELS).fill(INF2)
      const cSettled = new Uint8Array(TOTAL_TEXELS)
      const cHeap = new BinaryHeap(TOTAL_TEXELS * 2)
      const cNbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

      // Seed: crust edge texels (crustMask differs from any 4-neighbour)
      for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
        const face = (idx / (RES * RES)) | 0
        const rem  = idx % (RES * RES)
        const cy   = (rem / RES) | 0
        const cx   = rem % RES
        const mine = crustMask[idx]
        const dx4 = [1, -1, 0, 0] as const
        const dy4 = [0, 0, 1, -1] as const
        for (let d = 0; d < 4; d++) {
          neighborTexel(face, cx, cy, dx4[d] as -1|0|1, dy4[d] as -1|0|1, RES, cNbr)
          const ni = texelIndex(cNbr.face, cNbr.x, cNbr.y, RES)
          if (crustMask[ni] !== mine) {
            if (cDist[idx] > 0) {
              cDist[idx] = 0
              cHeap.push(0, idx)
            }
            break
          }
        }
      }

      const CDIAG = TEX_ANG * Math.SQRT2
      const CORTH = TEX_ANG
      const dx8c = [ 1, -1,  0,  0,  1, -1,  1, -1] as const
      const dy8c = [ 0,  0,  1, -1,  1,  1, -1, -1] as const

      while (!cHeap.isEmpty()) {
        const { key: cdistVal, val: cidx } = cHeap.popMin()
        if (cSettled[cidx]) continue
        cSettled[cidx] = 1
        const face = (cidx / (RES * RES)) | 0
        const rem  = cidx % (RES * RES)
        const cy   = (rem / RES) | 0
        const cx   = rem % RES
        for (let d = 0; d < 8; d++) {
          neighborTexel(face, cx, cy, dx8c[d] as -1|0|1, dy8c[d] as -1|0|1, RES, cNbr)
          const ni = texelIndex(cNbr.face, cNbr.x, cNbr.y, RES)
          if (cSettled[ni]) continue
          const cost = (dx8c[d] !== 0 && dy8c[d] !== 0) ? CDIAG : CORTH
          const newDist2 = cdistVal + cost
          if (newDist2 < cDist[ni]) {
            cDist[ni] = newDist2
            cHeap.push(newDist2, ni)
          }
        }
      }

      for (let t = 0; t < TOTAL_TEXELS; t++) {
        crustDistRaw[t] = crustMask[t] === 1 ? cDist[t] : -cDist[t]
      }
    }

    // Light blur removes Dijkstra diagonal artifacts. 3 passes intentional — staircase smoothing
    // delegated to _sampleC1 C1 sampling, not extra blur. See tectonic path note above.
    const crustDist = blurFieldN(crustDistRaw, 3)

    // ---- 3. Single plate ----
    const plates: Plate[] = [{
      id: 0,
      seedDir: new Vector3(0, 1, 0),
      type: 'continental',
      baseElevation: 0,
      omega: new Vector3(0, 0, 0),
      color: hslToRgb(0.0, 0.0, 0.55),
    }]

    // ---- 4. No-boundary fields ----
    const compId     = new Uint16Array(TOTAL_TEXELS)           // all 0 → plateId 0 everywhere
    const neighborId = new Uint16Array(TOTAL_TEXELS)           // all 0 → neighborId 0
    const distField  = new Float32Array(TOTAL_TEXELS).fill(Math.PI)  // sentinel "far from any boundary"
    const convField  = new Float32Array(TOTAL_TEXELS)          // all 0 — no convergence
    const shearField = new Float32Array(TOTAL_TEXELS)          // all 0 — no shear

    // ---- 5. Paleo orogens (identical to Step 6c in tectonic path) ----
    const paleoMask = new Uint8Array(TOTAL_TEXELS)
    const paleoRng  = makeRng(deriveSeed(seed, 12))
    const PALEO_SEED_COUNT  = 14
    const PALEO_MAX_WALKERS = 8 * PALEO_SEED_COUNT
    traceCracks(paleoMask, paleoRng, PALEO_SEED_COUNT, PALEO_MAX_WALKERS, dirScratch, texScratch)

    const paleoDilated = dilate4(paleoMask)

    const paleoDistRaw = new Float32Array(TOTAL_TEXELS)
    {
      const INF3 = 1e30
      const pDist    = new Float32Array(TOTAL_TEXELS).fill(INF3)
      const pSettled = new Uint8Array(TOTAL_TEXELS)
      const pHeap    = new BinaryHeap(TOTAL_TEXELS * 2)
      const pNbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

      for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
        if (paleoDilated[idx] === 1) {
          pDist[idx] = 0
          pHeap.push(0, idx)
        }
      }

      const PDIAG = TEX_ANG * Math.SQRT2
      const PORTH = TEX_ANG
      const pdx8 = [ 1, -1,  0,  0,  1, -1,  1, -1] as const
      const pdy8 = [ 0,  0,  1, -1,  1,  1, -1, -1] as const

      while (!pHeap.isEmpty()) {
        const { key: pd, val: pidx } = pHeap.popMin()
        if (pSettled[pidx]) continue
        pSettled[pidx] = 1
        const face = (pidx / (RES * RES)) | 0
        const rem  = pidx % (RES * RES)
        const py   = (rem / RES) | 0
        const px   = rem % RES
        for (let d = 0; d < 8; d++) {
          neighborTexel(face, px, py, pdx8[d] as -1|0|1, pdy8[d] as -1|0|1, RES, pNbr)
          const ni = texelIndex(pNbr.face, pNbr.x, pNbr.y, RES)
          if (pSettled[ni]) continue
          const cost = (pdx8[d] !== 0 && pdy8[d] !== 0) ? PDIAG : PORTH
          const nd = pd + cost
          if (nd < pDist[ni]) {
            pDist[ni] = nd
            pHeap.push(nd, ni)
          }
        }
      }

      for (let t = 0; t < TOTAL_TEXELS; t++) {
        paleoDistRaw[t] = pDist[t] === INF3 ? Math.PI : pDist[t]
      }
    }

    const paleoDist = blurField(paleoDistRaw)

    return { plates, compId, neighborId, distField, convField, shearField, crustDist, paleoDist }
  }

  // ---------------------------------------------------------------------------
  // _bakeHardness — bake crustAge + rockHardness cubemap fields
  //
  // crustAge [0,1]: 0 = fresh/divergent/volcanic crust, 1 = old stable craton.
  // Algorithm:
  //   1. Identify "divergent source" texels: convField < DIV_THRESH and
  //      distField < DIV_DIST (near a divergent boundary). These represent
  //      actively-spreading mid-ocean ridges — the youngest crust.
  //   2. Multi-source Dijkstra from divergent sources → divergentDist (radians).
  //   3. Also identify "volcanic" texels (high convField near arcs / hotspot
  //      positions) → volcanicDist.
  //   4. crustAge = saturate(divergentDist / AGE_SCALE) — far from ridges = old.
  //   5. rockHardness: blend of
  //        (a) age term: oldest craton = medium-hard (sediment-covered), freshest
  //            divergent = hard basalt, mid = medium.
  //        (b) volcanic proximity: near arc/hotspot → very hard (fresh igneous).
  //        (c) crustDist sign: oceanic basalt harder than continental sediment cover.
  //        (d) FBM break-up (stream 19) at low amplitude to avoid perfectly smooth iso-contours.
  //      All terms weighted to produce result in [0,1] before composition scaling.
  //   6. Composition scales contrast: hardness = mean + (hardness - mean) * contrastScale
  //      where contrastScale = lerp(0.25, 1.0, composition) — basaltic worlds have
  //      full contrast, icy/sediment worlds have flat hardness near the mean (0.5).
  //   7. Box-blur 3 passes for C1-friendliness.
  //
  // Reads: _convField, _distField, _crustDist (all already assigned before this call).
  // Reads: _volcanoes (empty at call time since _bakeVolcanoes runs AFTER this).
  //   Instead of using volcano positions (not yet available), we use _convField > VOLC_THRESH
  //   as a proxy for volcanic arc proximity, plus hotspot positions from _hotspots (also
  //   not yet available). We use the raw _convField and _crustDist fields as proxies.
  // ---------------------------------------------------------------------------

  private _bakeHardness(composition: number): void {
    // --- Constants ---
    // Divergent boundary identification: convField < DIV_THRESH near a boundary
    const DIV_THRESH   = -0.05  // convField below this → divergent
    const DIV_DIST     = 0.08   // radians — must be near a plate boundary
    // Age scale: divergentDist / AGE_SCALE → [0,1], saturated at 1
    const AGE_SCALE    = 0.40   // radians — half-sphere worth of "age"
    // Volcanic arc proximity proxy threshold
    const VOLC_THRESH  = 0.15   // convField above this → arc/volcanic zone
    // Hardness FBM break-up amplitude and frequency
    const HARD_FBM_AMP  = 0.12  // ±0.12 hardness noise amplitude
    const HARD_FBM_FREQ = 2.8   // spatial frequency on unit sphere

    // ---- Step 1: Multi-source Dijkstra from divergent-boundary texels → divergentDist ----
    const INF = 1e30
    const divergentDist = new Float32Array(TOTAL_TEXELS).fill(INF)
    const divSettled    = new Uint8Array(TOTAL_TEXELS)
    const divHeap       = new BinaryHeap(TOTAL_TEXELS * 2)
    const divNbr: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

    // Seed: texels near divergent boundaries (spreading ridges)
    for (let idx = 0; idx < TOTAL_TEXELS; idx++) {
      const conv = this._convField[idx]
      const dist = this._distField[idx]
      if (conv < DIV_THRESH && dist < DIV_DIST) {
        divergentDist[idx] = 0
        divHeap.push(0, idx)
      }
    }

    // If no divergent texels (non-tectonic case), leave divergentDist at INF → crustAge = 1 everywhere
    if (!divHeap.isEmpty()) {
      const DDIAG = TEX_ANG * Math.SQRT2
      const DORTH = TEX_ANG
      const ddx8 = [ 1, -1,  0,  0,  1, -1,  1, -1] as const
      const ddy8 = [ 0,  0,  1, -1,  1,  1, -1, -1] as const

      while (!divHeap.isEmpty()) {
        const { key: dd, val: didx } = divHeap.popMin()
        if (divSettled[didx]) continue
        divSettled[didx] = 1
        const face = (didx / (RES * RES)) | 0
        const rem  = didx % (RES * RES)
        const dy   = (rem / RES) | 0
        const dx   = rem % RES
        for (let d = 0; d < 8; d++) {
          neighborTexel(face, dx, dy, ddx8[d] as -1|0|1, ddy8[d] as -1|0|1, RES, divNbr)
          const ni = texelIndex(divNbr.face, divNbr.x, divNbr.y, RES)
          if (divSettled[ni]) continue
          const cost = (ddx8[d] !== 0 && ddy8[d] !== 0) ? DDIAG : DORTH
          const nd = dd + cost
          if (nd < divergentDist[ni]) {
            divergentDist[ni] = nd
            divHeap.push(nd, ni)
          }
        }
      }
    }

    // ---- Step 2: Build crustAge field ----
    // crustAge: 0 = freshest (divergent zones), 1 = oldest (cratons far from ridges)
    const crustAge = new Float32Array(TOTAL_TEXELS)
    for (let i = 0; i < TOTAL_TEXELS; i++) {
      const d = divergentDist[i]
      crustAge[i] = d >= INF * 0.5 ? 1.0 : Math.min(1.0, d / AGE_SCALE)
    }
    // Light blur for C1-friendliness
    this._crustAge = blurFieldN(crustAge, 2)

    // ---- Step 3: Build rockHardness field ----
    // Blend four terms into a [0,1] hardness value per texel:
    //   (a) Age term: young divergent = hard basalt (0.85), stable craton = medium (0.55),
    //       very old sediment-covered = soft (0.35). Uses a tent: peaks at freshest.
    //       hardAge = lerp(0.55, 0.85, 1 - age) for age < 0.5;
    //                 lerp(0.55, 0.35, (age - 0.5) / 0.5) for age >= 0.5
    //   (b) Volcanic arc proximity: convField > VOLC_THRESH → high hardness (0.90)
    //       blended in smoothly. This covers arc crust and near-volcanic areas.
    //   (c) Crust type: oceanic basalt (crustDist < 0) → +0.10 offset; continental
    //       sediment cover (crustDist > 0.05) → -0.10 offset.
    //   (d) FBM break-up: ±HARD_FBM_AMP amplitude noise to break up iso-contours.
    const hardnessRaw = new Float32Array(TOTAL_TEXELS)
    const dirScratch  = new Vector3()

    for (let i = 0; i < TOTAL_TEXELS; i++) {
      const face = (i / (RES * RES)) | 0
      const rem  = i % (RES * RES)
      const ty   = (rem / RES) | 0
      const tx   = rem % RES
      texelToDir(face, tx, ty, RES, dirScratch)

      // (a) Age term
      const age = this._crustAge[i]
      let hardAge: number
      if (age < 0.5) {
        // Fresh crust (0=divergent) → hard basalt; towards mid-age → medium
        hardAge = 0.55 + (0.85 - 0.55) * (1 - age * 2)
      } else {
        // Old craton → medium; very old sediment-draped → softer
        hardAge = 0.55 - (0.55 - 0.35) * ((age - 0.5) * 2)
      }

      // (b) Volcanic arc proximity (using convField as proxy — high conv = subduction/arc)
      const conv = this._convField[i]
      const volcFrac = Math.max(0, Math.min(1, (conv - VOLC_THRESH) / (1.0 - VOLC_THRESH)))
      const hardVolc = 0.90  // hardness near volcanic arc (fresh igneous)
      const hardB = hardAge + volcFrac * (hardVolc - hardAge)

      // (c) Crust-type offset: oceanic basalt harder than continental sediment cover
      const cd = this._crustDist[i]
      // crustDist < 0 = oceanic side (basalt), > 0 = continental side (sediment)
      const crustOffset = cd < 0 ? 0.08 : (cd > 0.05 ? -0.08 : 0)
      const hardC = hardB + crustOffset

      // (d) FBM break-up
      const fbmVal = fbm(
        this._hardnessNoise,
        dirScratch.x * HARD_FBM_FREQ,
        dirScratch.y * HARD_FBM_FREQ,
        dirScratch.z * HARD_FBM_FREQ,
        { octaves: 4 },
      )
      const hardD = hardC + HARD_FBM_AMP * fbmVal

      hardnessRaw[i] = Math.max(0, Math.min(1, hardD))
    }

    // ---- Step 4: Apply box blur for C1-friendliness (3 passes) ----
    const hardnessBlurred = blurFieldN(hardnessRaw, 3)

    // ---- Step 5: Scale contrast by composition ----
    // contrastScale: 0=icy/sediment → flat near 0.5; 1=rocky/basaltic → full range.
    // lerp(0.25, 1.0, composition)
    const contrastScale = 0.25 + 0.75 * composition
    const HARD_MEAN = 0.5
    const rockHardness = new Float32Array(TOTAL_TEXELS)
    for (let i = 0; i < TOTAL_TEXELS; i++) {
      const h = HARD_MEAN + (hardnessBlurred[i] - HARD_MEAN) * contrastScale
      rockHardness[i] = Math.max(0, Math.min(1, h))
    }

    this._rockHardness = rockHardness
  }

  // ---------------------------------------------------------------------------
  // _bakeVolcanoes — one-time bake of subduction-arc volcano positions
  // ---------------------------------------------------------------------------

  private _bakeVolcanoes(seed: number, arcDensity: number, hotspotCount: number, hotspotIntensity: number): void {
    // Reset hotspots at top so early-exit paths always leave a clean state.
    this._hotspots = []
    // Volcano constants are ANGULAR (radians) — scale-invariant under uniform planet scale.
    // No volScaleFactor needed: same angular cone at 500 km looks identical to 50 km.
    const ARC_CONV_THRESH_EFF = ARC_CONV_THRESH / arcDensity
    const ARC_SPACING_EFF     = ARC_SPACING / arcDensity

    const q       = this._bakeQueryScratch
    const dir     = this._bakeDirScratch
    const plates  = this.plates

    // ---- a. FRONT CANDIDATES ----
    // For each texel, quick-reject via _convField, then full query() for exact gates.
    interface Candidate {
      px: number; py: number; pz: number  // unit direction at texel center
      conv: number
      frontStrength: number
      hash: number     // deterministic per-texel integer for tie-breaking
    }
    const candidates: Candidate[] = []

    for (let face = 0; face < 6; face++) {
      for (let ty = 0; ty < RES; ty++) {
        for (let tx = 0; tx < RES; tx++) {
          texelToDir(face, tx, ty, RES, dir)

          // Cheap pre-reject: sample _convField at UNWARPED dir (no warp overhead).
          // Threshold is 0.5× so the unwarped sample can never falsely reject a texel
          // whose WARPED sample (used by the authoritative gate at line ~1986) would
          // pass. WARP_AMP=0.08 rad can shift the sample point by up to ~0.08 rad on
          // the sphere; the convField is blurred so the field varies smoothly, but to
          // be a strict superset we need enough slack to absorb that displacement.
          // 0.5× guarantees: any texel accepted by the real gate (conv > thresh) has
          // unwarped conv > thresh×0.5 unless the field is pathologically non-monotone
          // over an 0.08 rad neighbourhood, which the triple-blur ensures it isn't.
          const convPre = this._sampleC1(this._convField, dir)
          if (convPre < ARC_CONV_THRESH_EFF * 0.5) continue

          // Full query at texel-center direction (same warp as all later queries)
          this.query(dir, q)
          // Re-establish dir after query() (query() writes to its own _warpedDir, not dir — safe)
          texelToDir(face, tx, ty, RES, dir)

          if (q.convergence <= ARC_CONV_THRESH_EFF) continue
          const bd = q.boundaryDist
          if (bd < ARC_OFFSET - ARC_BAND || bd > ARC_OFFSET + ARC_BAND) continue

          // Overriding-side / oceanic-subductor test (mirroring boundaryRelief polarity)
          const wMine  = _ss(-0.10, 0.10, q.crustDist)
          const wOther = _ss(-0.10, 0.10, q.otherCrustDist)
          const own    = plates[q.plateId]
          const other  = plates[q.neighborId]

          // OC: continental overriding (Andes) — I'm continental, neighbor oceanic
          const wCO = wMine * (1 - wOther)
          // OO: oceanic-oceanic arc — both oceanic, higher id = overriding arc side
          const wOO = (1 - wMine) * (1 - wOther)
          const sideRamp = _ss(0.002, 0.015, bd)
          const ooSubduct = 0.5 + ((own.id < other.id ? 1 : 0) - 0.5) * sideRamp

          let frontStrength = 0
          let isCandidate = false

          if (wMine > 0.5 && wOther < 0.5) {
            // OC regime, I'm the continental overriding side
            frontStrength = wCO
            isCandidate = true
          } else if (wMine < 0.5 && wOther < 0.5 && own.id > other.id) {
            // OO regime, I'm the higher-id arc side
            frontStrength = wOO * (1 - ooSubduct)
            isCandidate = true
          }
          if (!isCandidate) continue

          const texIdx = texelIndex(face, tx, ty, RES)
          const hash   = deriveSeed(texIdx, 14)  // stable per-texel hash

          candidates.push({
            px: dir.x, py: dir.y, pz: dir.z,
            conv: q.convergence,
            frontStrength,
            hash,
          })
        }
      }
    }

    // Combined output arrays (arc volcanoes first, hotspot cones appended after)
    const volcanoes: Volcano[] = []
    const volPos:        number[] = []
    const volBaseRadius: number[] = []
    const volHeight:     number[] = []
    const volCraterFrac: number[] = []

    // ---- b. DART-THROWING with min-distance + render-consistent validation ----
    // Only runs when there are arc candidates; falls through to hotspot generation regardless.
    //
    // Two correctness requirements drive this pass:
    //   1. SPACING — enforce a TRUE min-distance (ARC_SPACING_EFF) between accepted
    //      volcanoes. A grid "one-per-cell" decimation does NOT do this: two winners
    //      in adjacent cells can sit far closer than the cell size. Dart-throwing
    //      (sort candidates by a seeded key, accept only if no already-accepted
    //      volcano lies within ARC_SPACING_EFF) gives a hard min-distance guarantee.
    //   2. PLACEMENT — validate each candidate at its FINAL (jittered) position by
    //      calling the same query() the renderer / harness uses. The per-texel scan
    //      above only finds candidates cheaply at texel centers; the jitter then
    //      moves the volcano by up to ~ARC_SPACING_EFF*0.4, which can carry it out
    //      of the [ARC_OFFSET ± ARC_BAND] band. Re-querying at the final pos and
    //      re-checking the band/convergence/overriding-side gate makes placement
    //      consistent-by-construction with what gets rendered.
    //
    // Proximity check is O(1): each accepted volcano stamps every grid cell within
    // ARC_SPACING_EFF of it. A new candidate is rejected iff its own cell is already
    // stamped — so we only ever read one cell. Stamping uses the cube-map grid, so
    // it is correct across face seams (cells are addressed by (face,x,y)).

    if (candidates.length > 0) {
    // Sort ALL candidates deterministically: highest convergence first, hash tie-break.
    candidates.sort((a, b) => {
      if (b.conv !== a.conv) return b.conv - a.conv
      return b.hash - a.hash
    })

    // Min-distance occupancy grid (cube-map). Cell angular size ≲ ARC_SPACING_EFF so
    // that a candidate's own cell is a sufficient proximity witness after stamping.
    const occRes = Math.max(2, Math.ceil((Math.PI / 2) / ARC_SPACING_EFF))
    const nOccCells = 6 * occRes * occRes
    const occupied = new Uint8Array(nOccCells)           // 1 = within ARC_SPACING of an accepted volcano
    const occCellAng = (Math.PI / 2) / occRes
    const occReach = ARC_SPACING_EFF + occCellAng * Math.SQRT2  // stamp radius incl. cell diagonal
    const occSearchR = Math.ceil(occReach / occCellAng) + 1
    const occTexScratch = { face: 0, x: 0, y: 0 }
    const occCellDir = new Vector3()

    // query scratch for final-position re-validation (separate from q used elsewhere? q is free here)
    const vq = q

    for (const cand of candidates) {
      if (volcanoes.length >= ARC_MAX) break

      // --- Seeded jitter keyed on the candidate's stable hash (independent of
      //     acceptance order → deterministic regardless of which darts land). ---
      const jitterRng = makeRng(deriveSeed(cand.hash ^ seed, 14))
      const jitterAngle = jitterRng() * ARC_SPACING_EFF * 0.4
      const jitterAzimuth = jitterRng() * Math.PI * 2

      // Build orthonormal tangent frame from (cand.px, cand.py, cand.pz)
      const px = cand.px, py = cand.py, pz = cand.pz
      let ux = 0, uy = 1, uz = 0
      if (Math.abs(py) > 0.9) { ux = 1; uy = 0; uz = 0 }
      let t1x = py * uz - pz * uy
      let t1y = pz * ux - px * uz
      let t1z = px * uy - py * ux
      const t1len = Math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z) || 1
      t1x /= t1len; t1y /= t1len; t1z /= t1len
      const t2x = py * t1z - pz * t1y
      const t2y = pz * t1x - px * t1z
      const t2z = px * t1y - py * t1x
      const cosA = Math.cos(jitterAzimuth), sinA = Math.sin(jitterAzimuth)
      const tx2 = cosA * t1x + sinA * t2x
      const ty2 = cosA * t1y + sinA * t2y
      const tz2 = cosA * t1z + sinA * t2z
      const axX = py * tz2 - pz * ty2
      const axY = pz * tx2 - px * tz2
      const axZ = px * ty2 - py * tx2
      const axLen = Math.sqrt(axX * axX + axY * axY + axZ * axZ) || 1
      const kx = axX / axLen, ky = axY / axLen, kz = axZ / axLen
      const cosJ = Math.cos(jitterAngle), sinJ = Math.sin(jitterAngle)
      const kdot = kx * px + ky * py + kz * pz
      const jx = px * cosJ + (ky * pz - kz * py) * sinJ + kx * kdot * (1 - cosJ)
      const jy = py * cosJ + (kz * px - kx * pz) * sinJ + ky * kdot * (1 - cosJ)
      const jz = pz * cosJ + (kx * py - ky * px) * sinJ + kz * kdot * (1 - cosJ)
      const jLen = Math.sqrt(jx * jx + jy * jy + jz * jz) || 1
      const posX = jx / jLen, posY = jy / jLen, posZ = jz / jLen

      // --- MIN-DISTANCE reject (O(1)): is this position's cell already occupied? ---
      occCellDir.set(posX, posY, posZ)
      dirToTexel(occCellDir, occRes, occTexScratch)
      const myOccCell = texelIndex(occTexScratch.face, occTexScratch.x, occTexScratch.y, occRes)
      if (occupied[myOccCell]) continue

      // --- PLACEMENT re-validate at FINAL position via the real query() ---
      // (same path the renderer + harness use → consistent by construction)
      occCellDir.set(posX, posY, posZ)
      this.query(occCellDir, vq)
      if (vq.convergence <= ARC_CONV_THRESH_EFF) continue
      const fbd = vq.boundaryDist
      if (fbd < ARC_OFFSET - ARC_BAND || fbd > ARC_OFFSET + ARC_BAND) continue
      const fwMine  = _ss(-0.10, 0.10, vq.crustDist)
      const fwOther = _ss(-0.10, 0.10, vq.otherCrustDist)
      const fown    = plates[vq.plateId]
      const fother  = plates[vq.neighborId]
      const sideOk =
        (fwMine > 0.5 && fwOther < 0.5) ||
        (fwMine < 0.5 && fwOther < 0.5 && fown.id > fother.id)
      if (!sideOk) continue

      // --- ACCEPTED. Instance props (convStrength from the re-validated conv). ---
      const conv = vq.convergence
      const convStrength = Math.max(0, Math.min(1, (conv - ARC_CONV_THRESH_EFF) / ARC_SIZE_SPAN))
      const rJitter  = 1 + (jitterRng() * 2 - 1) * 0.15
      const hJitter  = 1 + (jitterRng() * 2 - 1) * 0.15
      const cfJitter = (jitterRng() * 2 - 1) * 0.03

      // Radius floor raised (0.024) so the smallest cones aren't too steep: the
      // cone flank's worst-case 120 m step scales as height/baseRadius, and a
      // 0.018-rad / 0.55-height cone peaks right at the GATE X 120 m bound (0.10).
      // Widening the floor + trimming the height top keeps the steepest flank's
      // 120 m delta comfortably under 0.10 with no exemption, and (with the
      // headroom blend) keeps cordillera summits ≤ ~0.95.
      // Angular constants — no RADIUS scaling needed (uniform-scale-invariant).
      const baseRadius      = (0.054 + (0.070 - 0.054) * convStrength) * rJitter  // floor raised 0.024->0.054: gentler cone flanks keep 120 m step < 0.10 even where a breaching cone sits on the coast shelf
      const height          = (0.37  + (0.62  - 0.37)  * convStrength) * hJitter  // [was 0.55..0.95] de-saturation — ocean cones still breach, land cones modest
      const craterRadiusFrac = 0.12 + cfJitter

      const pos = new Vector3(posX, posY, posZ)
      volcanoes.push({ pos, baseRadius, height, craterRadiusFrac, kind: 'arc', intensity: 1 })
      volPos.push(posX, posY, posZ)
      volBaseRadius.push(baseRadius)
      volHeight.push(height)
      volCraterFrac.push(Math.max(0.05, craterRadiusFrac))

      // --- STAMP min-distance footprint: mark every cell within ARC_SPACING_EFF. ---
      const cFace = occTexScratch.face, cX = occTexScratch.x, cY = occTexScratch.y
      for (let dFace = 0; dFace < 6; dFace++) {
        for (let dty = 0; dty < occRes; dty++) {
          for (let dtx = 0; dtx < occRes; dtx++) {
            if (dFace === cFace) {
              if (Math.abs(dtx - cX) > occSearchR || Math.abs(dty - cY) > occSearchR) continue
            }
            texelToDir(dFace, dtx, dty, occRes, occCellDir)
            const dotVC = posX * occCellDir.x + posY * occCellDir.y + posZ * occCellDir.z
            const angVC = Math.acos(Math.max(-1, Math.min(1, dotVC)))
            if (angVC <= ARC_SPACING_EFF) {
              occupied[texelIndex(dFace, dtx, dty, occRes)] = 1
            }
          }
        }
      }
    }

    } // end if (candidates.length > 0)

    // ---- c. HOTSPOT GENERATION ----
    // Generate mantle-plume hotspot cones and append them to the same arrays.
    // Done AFTER arc collection so arc count is stable, BEFORE flat-array build.
    this._generateHotspots(seed, hotspotCount, hotspotIntensity, volcanoes, volPos, volBaseRadius, volHeight, volCraterFrac)

    // If both arc and hotspot counts are zero, leave flat arrays empty and return.
    if (volcanoes.length === 0) {
      this._volcanoes     = []
      this._volPos        = new Float32Array(0)
      this._volBaseRadius = new Float32Array(0)
      this._volHeight     = new Float32Array(0)
      this._volCraterFrac = new Float32Array(0)
      this._volBucketRes  = 0
      this._volBucketStart = new Int32Array(0)
      this._volBucketIdx   = new Int32Array(0)
      return
    }

    const nVol = volcanoes.length
    this._volcanoes = volcanoes

    // Flat typed arrays for hot path
    this._volPos        = new Float32Array(volPos)
    this._volBaseRadius = new Float32Array(volBaseRadius)
    this._volHeight     = new Float32Array(volHeight)
    this._volCraterFrac = new Float32Array(volCraterFrac)

    // ---- d. BUILD SPATIAL INDEX ----
    // Bucket res such that cell angular size ≳ max baseRadius.
    // max baseRadius ≈ 0.045 * 1.15 ≈ 0.052 rad for arcs; hotspot shields up to ~0.11 rad.
    // texAng(bucketRes) = (π/2)/bucketRes ≈ cell half-size → want cell diagonal ≳ maxBaseR
    // cell angular size = (π/2)/bucketRes; want ≈ 0.065 rad → bucketRes=24.
    // The per-volcano searchRadius scales with reach=baseRadius+cellDiag, so larger hotspot
    // shields simply stamp more cells — no change to the bucket res needed.
    const bucketRes = 24
    this._volBucketRes = bucketRes

    const nBucketCells = 6 * bucketRes * bucketRes
    // Per-cell count pass
    const cellCount = new Int32Array(nBucketCells)
    const volBucketScratch = { face: 0, x: 0, y: 0 }
    const cellAngSize = (Math.PI / 2) / bucketRes
    // cell diagonal ≈ cellAngSize * sqrt(2)
    const cellDiag = cellAngSize * Math.SQRT2
    // Dedicated scratch for cell-center direction (separate from dir/ssScratch)
    const cellDir = new Vector3()

    // For each volcano, find all cells within baseRadius + cellDiag and count
    const volCells: Array<number[]> = new Array(nVol)
    for (let vi = 0; vi < nVol; vi++) {
      const vx = this._volPos[vi * 3]
      const vy = this._volPos[vi * 3 + 1]
      const vz = this._volPos[vi * 3 + 2]
      const reach = this._volBaseRadius[vi] + cellDiag

      // Find candidate cells by iterating all cells within the sphere cap
      // Center cell of this volcano
      dir.set(vx, vy, vz)
      dirToTexel(dir, bucketRes, volBucketScratch)
      const cFace = volBucketScratch.face
      const cX    = volBucketScratch.x
      const cY    = volBucketScratch.y

      // Angular half-span in cells to search
      const searchRadius = Math.ceil(reach / cellAngSize) + 1

      const myVols: number[] = []
      for (let dFace = 0; dFace < 6; dFace++) {
        for (let dty = 0; dty < bucketRes; dty++) {
          for (let dtx = 0; dtx < bucketRes; dtx++) {
            // Quick manhattan filter on same face before angular test
            if (dFace === cFace) {
              if (Math.abs(dtx - cX) > searchRadius || Math.abs(dty - cY) > searchRadius) continue
            }
            // Angular distance from volcano pos to cell center
            texelToDir(dFace, dtx, dty, bucketRes, cellDir)
            const dotVC = vx * cellDir.x + vy * cellDir.y + vz * cellDir.z
            const angVC = Math.acos(Math.max(-1, Math.min(1, dotVC)))
            if (angVC <= reach) {
              const bucketCell = texelIndex(dFace, dtx, dty, bucketRes)
              myVols.push(bucketCell)
              cellCount[bucketCell]++
            }
          }
        }
      }
      volCells[vi] = myVols
    }

    // Build CSR offsets (prefix sum)
    const bucketStart = new Int32Array(nBucketCells + 1)
    for (let c = 0; c < nBucketCells; c++) {
      bucketStart[c + 1] = bucketStart[c] + cellCount[c]
    }
    const totalInserts = bucketStart[nBucketCells]
    const bucketIdx    = new Int32Array(totalInserts)

    // Fill CSR (need write cursor per cell)
    const writeCursor = new Int32Array(nBucketCells)
    for (let vi = 0; vi < nVol; vi++) {
      for (const cell of volCells[vi]) {
        const writePos = bucketStart[cell] + writeCursor[cell]
        bucketIdx[writePos] = vi
        writeCursor[cell]++
      }
    }

    this._volBucketStart = bucketStart
    this._volBucketIdx   = bucketIdx
  }

  // ---------------------------------------------------------------------------
  // _generateHotspots — mantle-plume hotspot cone generation
  // Appends hotspot cones to the shared volcanoes / volPos / ... arrays,
  // and populates this._hotspots with plume-source records.
  // ---------------------------------------------------------------------------

  private _generateHotspots(
    seed: number,
    hotspotCount: number,
    hotspotIntensity: number,
    volcanoes: Volcano[],
    volPos: number[],
    volBaseRadius: number[],
    volHeight: number[],
    volCraterFrac: number[],
  ): void {
    if (hotspotCount === 0) return

    // Volcano constants are ANGULAR (radians) — scale-invariant under uniform planet scale.
    // No volScaleFactor needed.

    // Seeded RNG streams — created ONCE before the per-hotspot loop (determinism).
    const posRng = makeRng(deriveSeed(seed, 16))
    const intRng = makeRng(deriveSeed(seed, 17))

    // Scratch vector for velocityAt (avoids allocation inside the loop)
    const velScratch = new Vector3()
    const posScratch = new Vector3()

    for (let h = 0; h < hotspotCount; h++) {
      // --- 1. Position: seeded random unit vector, boundary-independent ---
      let hx = 0, hy = 0, hz = 0
      for (let attempt = 0; attempt < 50; attempt++) {
        const rx = posRng() * 2 - 1
        const ry = posRng() * 2 - 1
        const rz = posRng() * 2 - 1
        const len = Math.sqrt(rx * rx + ry * ry + rz * rz)
        if (len > 0.01 && len < 1.0) {
          const nx = rx / len, ny = ry / len, nz = rz / len
          // Reject within 8° of poles (bounded — fall back after 50 attempts)
          if (Math.abs(ny) > HOTSPOT_POLAR_COS) continue
          hx = nx; hy = ny; hz = nz
          break
        }
      }
      // If rejection sampling exhausted, hx/hy/hz remain 0 → skip this hotspot
      if (hx === 0 && hy === 0 && hz === 0) continue

      // --- 2. Intensity ---
      const u = intRng()
      const intensity = Math.max(0, Math.min(1, (0.45 + 0.55 * u * u) * hotspotIntensity))

      // --- 3. Plate velocity at hotspot position ---
      posScratch.set(hx, hy, hz)
      this.velocityAt(posScratch, velScratch)
      const speed = velScratch.length()

      // --- 4. Chain vs Shield branch ---
      if (speed > HOTSPOT_V_MOVING) {
        // CHAIN — Hawaii-style: multiple cones trailing in the plate-motion direction
        const vHatX = velScratch.x / speed
        const vHatY = velScratch.y / speed
        const vHatZ = velScratch.z / speed

        const N = Math.max(2, Math.min(9, Math.round(2 + 4 * intensity)))

        for (let i = 0; i < N; i++) {
          const age = i / Math.max(1, N - 1)          // 0=newest, 1=oldest
          const ageScale = 1 - 0.7 * age

          // Step along +vHat: normalize(pos + vHat * step) — small-angle approximation
          const stepDist = i * HOTSPOT_CHAIN_STEP
          const cx = hx + vHatX * stepDist
          const cy = hy + vHatY * stepDist
          const cz = hz + vHatZ * stepDist
          const cLen = Math.sqrt(cx * cx + cy * cy + cz * cz) || 1
          const conePosX = cx / cLen, conePosY = cy / cLen, conePosZ = cz / cLen

          const baseRadius      = HOTSPOT_BASE_R * (0.7 + 0.6 * intensity) * ageScale
          const height          = HOTSPOT_BASE_H * (0.6 + 0.8 * intensity) * ageScale
          const craterRadiusFrac = 0.12

          const conePos = new Vector3(conePosX, conePosY, conePosZ)
          volcanoes.push({ pos: conePos, baseRadius, height, craterRadiusFrac, kind: 'hotspot', intensity })
          volPos.push(conePosX, conePosY, conePosZ)
          volBaseRadius.push(baseRadius)
          volHeight.push(height)
          volCraterFrac.push(Math.max(0.05, craterRadiusFrac))
        }

        this._hotspots.push({ pos: new Vector3(hx, hy, hz), intensity, isChain: true })

      } else {
        // SHIELD — Olympus Mons-style: one large broad volcano at pos
        const baseRadius      = HOTSPOT_BASE_R * 1.6 * (0.7 + 0.6 * intensity)
        const height          = HOTSPOT_BASE_H * 1.3 * (0.6 + 0.8 * intensity)
        const craterRadiusFrac = 0.15

        const conePos = new Vector3(hx, hy, hz)
        volcanoes.push({ pos: conePos, baseRadius, height, craterRadiusFrac, kind: 'hotspot', intensity })
        volPos.push(hx, hy, hz)
        volBaseRadius.push(baseRadius)
        volHeight.push(height)
        volCraterFrac.push(Math.max(0.05, craterRadiusFrac))

        this._hotspots.push({ pos: new Vector3(hx, hy, hz), intensity, isChain: false })
      }
    }
  }

  // ---------------------------------------------------------------------------
  // volcanoElevation — zero-alloc hot path
  //
  // Cone profile: f(x) = (1 - x^2)^2, x = a / baseRadius ∈ [0,1).
  //   C∞: d/dx = -4x(1-x^2), value 1 and slope 0 at x=0 (rounded summit),
  //   value 0 and slope 0 at x=1 (smooth base).
  //
  // Crater dimple: craterDepth*(1-(x/craterFrac)^2)^2 subtracted for x < craterFrac.
  //   C∞: value craterDepth and slope 0 at x=0 (caldera floor),
  //   value 0 and slope 0 at x=craterFrac (seamless join onto cone — C1 satisfied).
  //   Net: center ≈ height - craterDepth; rim at x≈craterFrac ≈ height — rim is high point. ✓
  // ---------------------------------------------------------------------------

  volcanoElevation(dir: Vector3): number {
    const nVol = this._volcanoes.length
    if (nVol === 0) return 0

    const bucketRes = this._volBucketRes
    if (bucketRes === 0) return 0
    dirToTexel(dir, bucketRes, this._volScratchTexel)
    const cell = texelIndex(
      this._volScratchTexel.face,
      this._volScratchTexel.x,
      this._volScratchTexel.y,
      bucketRes,
    )

    const start = this._volBucketStart[cell]
    const end   = this._volBucketStart[cell + 1]
    if (start === end) return 0

    const dx = dir.x, dy = dir.y, dz = dir.z
    let sum = 0

    // Perturbation 1 (irregular base): single noise sample at dir
    // — evaluated once per volcanoElevation call, not per cone.
    // This is intentional: it warps the footprint in a direction-specific way
    // without introducing per-cone allocations.
    const nWarp = this._volNoise(
      dx * VOLC_BASE_WARP_FREQ,
      dy * VOLC_BASE_WARP_FREQ,
      dz * VOLC_BASE_WARP_FREQ,
    )

    for (let i = start; i < end; i++) {
      const vi = this._volBucketIdx[i]
      const vx = this._volPos[vi * 3]
      const vy = this._volPos[vi * 3 + 1]
      const vz = this._volPos[vi * 3 + 2]

      // Angular distance via dot product — alloc-free
      const dotVal = dx * vx + dy * vy + dz * vz
      const a = Math.acos(Math.max(-1, Math.min(1, dotVal)))
      const br = this._volBaseRadius[vi]

      // Perturbation 1: warp the effective base radius to make it non-circular.
      // VOLC_BASE_WARP < 1 guarantees brEff > 0; clamp defensively.
      const brEff = Math.max(br * 0.3, br * (1 + VOLC_BASE_WARP * nWarp))
      if (a >= brEff) continue

      const x = a / brEff
      const x2 = x * x
      let cone = this._volHeight[vi] * (1 - x2) * (1 - x2)

      // Perturbation 2: flank roughness — fbm modulated by a mid-flank mask.
      // Mask peaks at mid-flank (x≈0.5) and fades to 0 at summit (x=0) and base (x=1),
      // keeping the peak height and surrounding terrain joins clean.
      const flankMask = Math.sin(Math.PI * x)  // 0 at x=0 and x=1, peak 1.0 at x=0.5
      const rough = fbm(
        this._volNoise,
        dx * VOLC_FLANK_FREQ,
        dy * VOLC_FLANK_FREQ,
        dz * VOLC_FLANK_FREQ,
        { octaves: 3 },
      )
      cone *= (1 + VOLC_FLANK_ROUGH * flankMask * rough)


      const cf = this._volCraterFrac[vi]
      if (x < cf) {
        const xc = x / cf
        const xc2 = xc * xc
        const craterDepth = 0.18 * this._volHeight[vi]
        sum += cone - craterDepth * (1 - xc2) * (1 - xc2)
      } else {
        sum += cone
      }
    }

    return sum
  }

  // ---------------------------------------------------------------------------
  // _sampleC1 — C1 smoothstep bilinear sampler (continuous fields only)
  // ---------------------------------------------------------------------------

  /**
   * Sample a per-texel Float32Array at direction `dir` with C1 continuity.
   * Mirrors erosion.ts's _sampleC1: identical to sampleSmooth from cubemap.ts
   * except smoothstep is applied to fractional texel coords before the bilinear lerp:
   *   fu = fu*fu*(3-2*fu)   fv = fv*fv*(3-2*fv)
   * This eliminates C0 kinks at cell edges that produce visible boundary staircasing.
   * NOT safe for sentinel-valued fields (_compId, _neighborId — use direct lookup).
   * Uses a local scratch object for cross-face neighbors (no module-level state aliasing).
   */
  private _sampleC1(data: Float32Array, dir: Vector3): number {
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

    const half = RES * 0.5
    const fx = (sc / mc + 1.0) * half - 0.5
    const fy = (tc / mc + 1.0) * half - 0.5

    const x0 = Math.floor(fx)
    const y0 = Math.floor(fy)
    let fu = fx - x0
    let fv = fy - y0
    // smoothstep for C1 continuity — zero slope at cell edges, no kinks
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)

    const x1 = x0 + 1
    const y1 = y0 + 1

    let t00: number, t10: number, t01: number, t11: number

    if (x0 >= 0 && y0 >= 0 && x1 < RES && y1 < RES) {
      const base = face * RES * RES
      const row0 = base + y0 * RES
      const row1 = base + y1 * RES
      t00 = data[row0 + x0]
      t10 = data[row0 + x1]
      t01 = data[row1 + x0]
      t11 = data[row1 + x1]
    } else {
      const nb: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
      const axx = x0 < 0 ? 0 : (x0 >= RES ? RES - 1 : x0)
      const ayy = y0 < 0 ? 0 : (y0 >= RES ? RES - 1 : y0)
      const ox0 = (x0 - axx) as -1 | 0 | 1
      const ox1 = (x1 - axx) as -1 | 0 | 1
      const oy0 = (y0 - ayy) as -1 | 0 | 1
      const oy1 = (y1 - ayy) as -1 | 0 | 1
      neighborTexel(face, axx, ayy, ox0, oy0, RES, nb)
      t00 = data[texelIndex(nb.face, nb.x, nb.y, RES)]
      neighborTexel(face, axx, ayy, ox1, oy0, RES, nb)
      t10 = data[texelIndex(nb.face, nb.x, nb.y, RES)]
      neighborTexel(face, axx, ayy, ox0, oy1, RES, nb)
      t01 = data[texelIndex(nb.face, nb.x, nb.y, RES)]
      neighborTexel(face, axx, ayy, ox1, oy1, RES, nb)
      t11 = data[texelIndex(nb.face, nb.x, nb.y, RES)]
    }

    const a = t00 + (t10 - t00) * fu
    const c = t01 + (t11 - t01) * fu
    return a + (c - a) * fv
  }

  /**
   * Returns the wide-smoothed uplift field value at the given direction.
   * Positive = convergence zone (broad orogen), negative = divergence zone (broad basin).
   * Uses the same domain-warp as query() for spatial continuity.
   */
  upliftAt(dir: Vector3): number {
    const wn = this._warpNoise
    const w  = this._smoothScratch

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

    return this._sampleC1(this._upliftField, w)
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
    // Fine warp for distField: breaks 177m staircase isoline without large profile drift.
    // Amplitude 0.006 ≈ 3 cells; same noise basis as crustDist warp; frequency 9 = ~2° wavelength.
    // Seed offsets (3.7, 8.2, 12.9) differ from crustDist (5.1, 7.3, 11.7) so warps are uncorrelated.
    const _fdw = this._fineDistScratch
    _fdw.set(
      w.x + 0.006 * fbm(this._fineWarpNoise, dir.x + 3.7, dir.y + 3.7, dir.z + 3.7, { octaves: 3, frequency: 9 }),
      w.y + 0.006 * fbm(this._fineWarpNoise, dir.x + 8.2, dir.y + 8.2, dir.z + 8.2, { octaves: 3, frequency: 9 }),
      w.z + 0.006 * fbm(this._fineWarpNoise, dir.x + 12.9, dir.y + 12.9, dir.z + 12.9, { octaves: 3, frequency: 9 }),
    ).normalize()
    const boundaryDist = this._sampleC1(this._distField, _fdw)

    // --- Convergence and shear from pre-baked blurred fields ---
    // Sampling the blurred baked fields eliminates the gradient-direction sign-flips
    // that caused cliff artefacts when tHat reversed across saddle points in the
    // raw distField gradient. The baked fields are blurred 6× at construction time.
    const convergence = this._sampleC1(this._convField,  w)
    const shear       = this._sampleC1(this._shearField, w)

    // --- Gradient-based tHat (unwarped dir) — ONLY used for otherCrustDist mirror ---
    // The gradient is no longer used for convergence/shear; it is only needed to
    // compute the Rodrigues rotation axis for the crust-dist mirror probe below.
    dirToTexel(dir, RES, this._tex)
    const { face, x: tx, y: ty } = this._tex

    const _nt0 = this._gTex0
    const _nt1 = this._gTex1
    const _nt2 = this._gTex2
    const _nt3 = this._gTex3

    neighborTexel(face, tx, ty,  1,  0, RES, _nt0)
    neighborTexel(face, tx, ty, -1,  0, RES, _nt1)
    neighborTexel(face, tx, ty,  0,  1, RES, _nt2)
    neighborTexel(face, tx, ty,  0, -1, RES, _nt3)

    const d0 = this._distField[texelIndex(_nt0.face, _nt0.x, _nt0.y, RES)]
    const d1 = this._distField[texelIndex(_nt1.face, _nt1.x, _nt1.y, RES)]
    const d2 = this._distField[texelIndex(_nt2.face, _nt2.x, _nt2.y, RES)]
    const d3 = this._distField[texelIndex(_nt3.face, _nt3.x, _nt3.y, RES)]

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

    const gradU = (d0 - d1) * 0.5
    const gradV = (d2 - d3) * 0.5
    const gwx = gradU * uax + gradV * vax
    const gwy = gradU * uay + gradV * vay
    const gwz = gradU * uaz + gradV * vaz

    const gdot = gwx * dir.x + gwy * dir.y + gwz * dir.z
    const ptx = gwx - gdot * dir.x
    const pty = gwy - gdot * dir.y
    const ptz = gwz - gdot * dir.z
    const gLen = Math.sqrt(ptx * ptx + pty * pty + ptz * ptz)

    if (gLen > 1e-7) {
      this._tHat.set(-ptx / gLen, -pty / gLen, -ptz / gLen)
    }

    out.plateId      = plateId
    out.neighborId   = neighborId
    out.boundaryDist = boundaryDist
    out.convergence  = convergence
    out.shear        = shear

    // paleoDist: sample using the standard warped direction w
    out.paleoDist = this._sampleC1(this._paleoDist, w)

    // baseElevation: blurred per-plate base elevation sampled continuously (no Voronoi step)
    out.baseElevation = this._sampleC1(this._baseElevField, w)

    // Fine warp for coastline complexity — applied ONLY to crustDist sampling
    // Plate/boundary/paleo fields keep using w (plate ownership must not pick up fine warp)
    const fwn = this._fineWarpNoise
    const fwDir = this._fineCrustScratch
    const fw0x = fbm(fwn, dir.x + 5.1, dir.y + 5.1, dir.z + 5.1, { octaves: 3, frequency: 9 })
    const fw0y = fbm(fwn, dir.x + 7.3, dir.y + 7.3, dir.z + 7.3, { octaves: 3, frequency: 9 })
    const fw0z = fbm(fwn, dir.x + 11.7, dir.y + 11.7, dir.z + 11.7, { octaves: 3, frequency: 9 })
    fwDir.set(
      w.x + 0.025 * fw0x,
      w.y + 0.025 * fw0y,
      w.z + 0.025 * fw0z,
    ).normalize()
    out.crustDist = this._sampleC1(this._crustDist, fwDir)

    // --- otherCrustDist: mirror through boundary to the neighbor side ---
    // Rotate the UNWARPED dir by 2·boundaryDist about the axis (dir × tHat).
    // Clamp theta to avoid huge rotations for interior points; blend by gradient strength
    // so weak-gradient regions gracefully fall back to crustDist (no discontinuous branch).
    let mirroredCrustDist = out.crustDist
    if (gLen > 1e-7) {
      const ax2 = dir.y * this._tHat.z - dir.z * this._tHat.y
      const ay2 = dir.z * this._tHat.x - dir.x * this._tHat.z
      const az2 = dir.x * this._tHat.y - dir.y * this._tHat.x
      const aLen2 = Math.sqrt(ax2 * ax2 + ay2 * ay2 + az2 * az2)
      if (aLen2 > 1e-7) {
        const mirrorAxis = this._otherCrustDir  // reuse scratch
        mirrorAxis.set(ax2 / aLen2, ay2 / aLen2, az2 / aLen2)
        // Clamp rotation angle to avoid wild overshoots for interior points
        const theta = Math.min(2 * boundaryDist, 0.06)
        const cosT = Math.cos(theta)
        const sinT = Math.sin(theta)
        const vx = dir.x, vy = dir.y, vz = dir.z
        const kx = mirrorAxis.x, ky = mirrorAxis.y, kz = mirrorAxis.z
        const kdotv = kx * vx + ky * vy + kz * vz
        const crossX = ky * vz - kz * vy
        const crossY = kz * vx - kx * vz
        const crossZ = kx * vy - ky * vx
        mirrorAxis.set(
          vx * cosT + crossX * sinT + kx * kdotv * (1 - cosT),
          vy * cosT + crossY * sinT + ky * kdotv * (1 - cosT),
          vz * cosT + crossZ * sinT + kz * kdotv * (1 - cosT),
        ).normalize()
        mirroredCrustDist = this._sampleC1(this._crustDist, mirrorAxis)
      }
    }
    // Blend by gradient strength: weak gradient → mirror equals self (no discontinuous jump)
    const gradStrength = _ss(1e-6, 5e-6, gLen * gLen)
    out.otherCrustDist = out.crustDist + gradStrength * (mirroredCrustDist - out.crustDist)

    // rockHardness: C1 bilinear sample from baked grid using domain-warped direction w
    out.rockHardness = this._sampleC1(this._rockHardness, w)

    return out
  }

  /**
   * Query substrate rock hardness at direction `dir`.
   * Returns [0,1]: 1=hard fresh igneous/volcanic, 0=soft sediment/basin.
   * Pure baked-grid lookup — identical on main thread and workers (no live noise).
   * Uses the same domain-warp as query() for spatial continuity.
   */
  hardnessAt(dir: { x: number; y: number; z: number }): number {
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

    return this._sampleC1(this._rockHardness, w)
  }

  // ---------------------------------------------------------------------------
  // velocityAt — zero-alloc, UNWARPED dir
  // IMPORTANT: velocityAt is defined below; boundaryRelief is a module-level export
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

  // ---------------------------------------------------------------------------
  // Serialisation — toBaked / fromBaked
  //
  // toBaked()   : snapshot all baked state by reference (no copy; postMessage
  //               will Structured-Clone / transfer the typed arrays).
  // fromBaked() : reconstruct a query-only Tectonics WITHOUT running the bake.
  //               Uses Object.create so class-field initialisers are skipped,
  //               then manually assigns every field the hot paths need.
  // ---------------------------------------------------------------------------

  toBaked(): TectonicsBaked {
    const nVol = this._volcanoes.length
    const volKind      = new Uint8Array(nVol)
    const volIntensity = new Float32Array(nVol)
    for (let vi = 0; vi < nVol; vi++) {
      volKind[vi]      = this._volcanoes[vi].kind === 'hotspot' ? 1 : 0
      volIntensity[vi] = this._volcanoes[vi].intensity
    }

    const nHot = this._hotspots.length
    const hotspotPos          = new Float32Array(nHot * 3)
    const hotspotIntensityArr = new Float32Array(nHot)
    const hotspotIsChain      = new Uint8Array(nHot)
    for (let hi = 0; hi < nHot; hi++) {
      const hs = this._hotspots[hi]
      hotspotPos[hi * 3]     = hs.pos.x
      hotspotPos[hi * 3 + 1] = hs.pos.y
      hotspotPos[hi * 3 + 2] = hs.pos.z
      hotspotIntensityArr[hi] = hs.intensity
      hotspotIsChain[hi]      = hs.isChain ? 1 : 0
    }

    return {
      res:              RES,
      seed:             this._seed,
      arcDensity:       this._arcDensity,
      hotspotCount:     this._hotspotCount,
      hotspotIntensity: this._hotspotIntensity,
      plates:         this.plates.map(p => ({
        id:            p.id,
        seedDir:       [p.seedDir.x, p.seedDir.y, p.seedDir.z] as [number, number, number],
        type:          p.type,
        baseElevation: p.baseElevation,
        omega:         [p.omega.x, p.omega.y, p.omega.z] as [number, number, number],
        color:         [p.color[0], p.color[1], p.color[2]] as [number, number, number],
      })),
      compId:           this._compId,
      distField:        this._distField,
      neighborId:       this._neighborId,
      crustDist:        this._crustDist,
      paleoDist:        this._paleoDist,
      convField:        this._convField,
      shearField:       this._shearField,
      upliftField:      this._upliftField,
      baseElevField:    this._baseElevField,
      volPos:           this._volPos,
      volBaseRadius:    this._volBaseRadius,
      volHeight:        this._volHeight,
      volCraterFrac:    this._volCraterFrac,
      volKind,
      volIntensity,
      volBucketRes:     this._volBucketRes,
      volBucketStart:   this._volBucketStart,
      volBucketIdx:     this._volBucketIdx,
      hotspotPos,
      hotspotIntensityArr,
      hotspotIsChain,
      rockHardness:     this._rockHardness,
    }
  }

  static fromBaked(b: TectonicsBaked): Tectonics {
    // Create instance without running constructor (and therefore without the bake).
    // Class field initialisers with `= expression` are NOT run by Object.create,
    // so every field that the hot paths (query / volcanoElevation / velocityAt)
    // read must be assigned explicitly below.
    const t = Object.create(Tectonics.prototype) as Tectonics

    // ---- 0. Seed / arcDensity / hotspot params (needed to rebuild noise and for further toBaked calls) ---
    ;(t as any)._seed             = b.seed
    ;(t as any)._arcDensity       = b.arcDensity
    ;(t as any)._hotspotCount     = b.hotspotCount
    ;(t as any)._hotspotIntensity = b.hotspotIntensity
    // _composition: not in baked (not needed at query time); default to 0.5 for fromBaked instances
    ;(t as any)._composition      = 0.5

    // ---- 1. Baked typed-array fields ----------------------------------------
    ;(t as any)._compId        = b.compId
    ;(t as any)._distField     = b.distField
    ;(t as any)._neighborId    = b.neighborId
    ;(t as any)._crustDist     = b.crustDist
    ;(t as any)._paleoDist     = b.paleoDist
    ;(t as any)._convField     = b.convField
    ;(t as any)._shearField    = b.shearField
    ;(t as any)._upliftField   = b.upliftField
    ;(t as any)._baseElevField = b.baseElevField
    ;(t as any)._rockHardness  = b.rockHardness

    // ---- 2. Volcano flat arrays and spatial index ---------------------------
    ;(t as any)._volPos        = b.volPos
    ;(t as any)._volBaseRadius = b.volBaseRadius
    ;(t as any)._volHeight     = b.volHeight
    ;(t as any)._volCraterFrac = b.volCraterFrac
    ;(t as any)._volBucketRes  = b.volBucketRes
    ;(t as any)._volBucketStart = b.volBucketStart
    ;(t as any)._volBucketIdx  = b.volBucketIdx

    // _volcanoes: volcanoElevation only checks length; fromBaked reconstructs
    // the Volcano[] from flat arrays so the `volcanoes` getter works too.
    const nVol = b.volPos.length / 3
    const volcanoes: Volcano[] = []
    for (let vi = 0; vi < nVol; vi++) {
      volcanoes.push({
        pos:             new Vector3(b.volPos[vi * 3], b.volPos[vi * 3 + 1], b.volPos[vi * 3 + 2]),
        baseRadius:      b.volBaseRadius[vi],
        height:          b.volHeight[vi],
        craterRadiusFrac: b.volCraterFrac[vi],
        kind:            b.volKind[vi] === 1 ? 'hotspot' : 'arc',
        intensity:       b.volIntensity[vi],
      })
    }
    ;(t as any)._volcanoes = volcanoes
    ;(t as any)._volcanoBakeMs = 0

    // _hotspots: restore from flat arrays
    const nHot = b.hotspotPos.length / 3
    const hotspots: Hotspot[] = []
    for (let hi = 0; hi < nHot; hi++) {
      hotspots.push({
        pos:       new Vector3(b.hotspotPos[hi * 3], b.hotspotPos[hi * 3 + 1], b.hotspotPos[hi * 3 + 2]),
        intensity: b.hotspotIntensityArr[hi],
        isChain:   b.hotspotIsChain[hi] === 1,
      })
    }
    ;(t as any)._hotspots = hotspots

    // ---- 3. Rehydrate plates (Vector3 for seedDir and omega) ----------------
    ;(t as any).plates = b.plates.map(pw => ({
      id:            pw.id,
      seedDir:       new Vector3(pw.seedDir[0], pw.seedDir[1], pw.seedDir[2]),
      type:          pw.type,
      baseElevation: pw.baseElevation,
      omega:         new Vector3(pw.omega[0], pw.omega[1], pw.omega[2]),
      color:         [pw.color[0], pw.color[1], pw.color[2]] as [number, number, number],
    }))

    // ---- 4. Rebuild noise closures from seed (streams 5, 13, 18, 19) -----------
    // Exact derivation mirrors the constructor.
    ;(t as any)._warpNoise      = createNoise3D(deriveSeed(b.seed, 5))
    ;(t as any)._fineWarpNoise  = createNoise3D(deriveSeed(b.seed, 13))
    ;(t as any)._volNoise       = createNoise3D(deriveSeed(b.seed, 18))
    // _hardnessNoise is only needed during _bakeHardness (constructor-time),
    // not at query time (rockHardness is a pure baked-grid lookup). Assign a
    // stub so the field is defined; it will never be called on fromBaked instances.
    ;(t as any)._hardnessNoise  = createNoise3D(deriveSeed(b.seed, 19))

    // ---- 5. Preallocated scratch — class field initialisers were NOT run ----
    // query() uses: _warpedDir, _smoothScratch, _crustScratch,
    //               _fineWarpScratch, _fineCrustScratch, _tex,
    //               _gDir0-3, _gTex0-3, _tHat, _otherCrustDir
    // volcanoElevation() uses: _volScratchTexel
    // velocityAt() uses: _warpedDir, _tex (shared with query)
    // _bakeQueryScratch / _bakeDirScratch are only used by _bakeVolcanoes (not
    // called after construction), but they are declared `readonly` on the class
    // so we must still initialise them to avoid a potential undefined-property
    // crash if any future code path reaches them.
    ;(t as any)._warpedDir         = new Vector3()
    ;(t as any)._smoothScratch     = new Vector3()
    ;(t as any)._crustScratch      = new Vector3()
    ;(t as any)._fineWarpScratch   = new Vector3()
    ;(t as any)._fineCrustScratch  = new Vector3()
    ;(t as any)._fineDistScratch   = new Vector3()
    ;(t as any)._tex               = { face: 0, x: 0, y: 0 }
    ;(t as any)._gDir0             = new Vector3()
    ;(t as any)._gDir1             = new Vector3()
    ;(t as any)._gDir2             = new Vector3()
    ;(t as any)._gDir3             = new Vector3()
    ;(t as any)._gTex0             = { face: 0, x: 0, y: 0 }
    ;(t as any)._gTex1             = { face: 0, x: 0, y: 0 }
    ;(t as any)._gTex2             = { face: 0, x: 0, y: 0 }
    ;(t as any)._gTex3             = { face: 0, x: 0, y: 0 }
    ;(t as any)._tHat              = new Vector3()
    ;(t as any)._otherCrustDir     = new Vector3()
    ;(t as any)._volScratchTexel   = { face: 0, x: 0, y: 0 }
    ;(t as any)._bakeQueryScratch  = {
      plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0,
      shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0,
    }
    ;(t as any)._bakeDirScratch    = new Vector3()

    return t
  }
}

// ---------------------------------------------------------------------------
// Asymmetric boundary relief profiles — pure function, headless-testable
//
// Geology references per block:
//   Andes / Cascades   — continental overriding-side volcanic cordillera
//   Mariana Trench     — oceanic subducting-side deep-sea trench
//   Japan arc          — oceanic-oceanic island arc (discrete islands)
//   Himalaya / Alps    — continent-continent collision belt + plateau
//   Mid-Atlantic Ridge — mid-ocean spreading ridge with axial rift notch
//   East African Rift  — continental rift graben + shoulders
//   San Andreas Fault  — transform fault scarp + parallel ridging
// ---------------------------------------------------------------------------

// Gaussian helper: exp(−((x−center)/width)²)
function _g(x: number, center: number, width: number): number {
  const t = (x - center) / width
  return Math.exp(-(t * t))
}

// GLSL-style smoothstep
function _ss(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

// Profile shape constants (named for easy tuning)
// Convergent
const BR_TRENCH_DEPTH         = 0.60   // OC trench depth
const BR_FLEX_BULGE           = 0.05   // outer flexural bulge amplitude
const BR_FLEX_POS             = 0.030  // bulge center (rad)
const BR_FLEX_WIDTH           = 0.012  // bulge width (rad)
const BR_TRENCH_WIDTH         = 0.020  // trench width (rad)
const BR_CORD_HEIGHT          = 0.45   // cordillera peak amplitude (Andes) [was 0.72 — Option C broad uplift takes over]
const BR_CORD_POS             = 0.038  // cordillera center (rad) [was 0.030]
const BR_CORD_WIDTH           = 0.024  // cordillera width (rad) [was 0.018] — narrowed: taller crest, smaller footprint
const BR_CORD_RIDGE           = 0.60   // ridged texture amplitude on cordillera
const BR_CORD_RIDGE_BASE      = 0.55   // ridged texture floor
const BR_SHELF_DIP            = 0.06   // coastal shelf dip amplitude
const BR_SHELF_WIDTH          = 0.006  // shelf dip width (rad)
const BR_OO_TRENCH_DEPTH      = 0.50   // OO trench depth
const BR_OO_TRENCH_WIDTH      = 0.016  // OO trench width
const BR_ARC_HEIGHT           = 0.50   // island arc peak amplitude — high enough to reach the sea-level emergence clamp at gated cone peaks
const BR_ARC_POS              = 0.026  // arc center (rad)
const BR_ARC_WIDTH            = 0.014  // arc width (rad)
const BR_ARC_RIDGE_THRESH     = 0.45   // ridged threshold for discrete islands
const BR_ARC_RIDGE_SCALE      = 1 / 0.55 // normaliser for arc ridged
const BR_BELT_HEIGHT          = 0.52   // collision belt amplitude (Himalaya) [was 0.82 — Option C broad uplift takes over]
const BR_BELT_WIDTH           = 0.075  // belt width (rad) [was 0.110 — narrower footprint]
const BR_BELT_RIDGE           = 0.40   // ridged texture amplitude on belt [was 0.55 — trims belt peak to lower patchMax]
const BR_BELT_RIDGE_BASE      = 0.58   // ridged texture floor [was 0.60 — raised back to hold belt MEAN (prominence)]
const BR_PLATEAU_HEIGHT       = 0.24   // plateau amplitude (Tibetan Plateau) [was 0.30 — stacks on belt, trimmed for de-sat]
const BR_PLATEAU_INNER        = 0.05   // plateau inner ramp start
const BR_PLATEAU_OUTER        = 0.13   // plateau outer fade
const BR_PLATEAU_NEAR         = 0.015  // plateau near ramp start
const BR_PLATEAU_FAR          = 0.05   // plateau near ramp end
// Divergent
const BR_RIDGE_HEIGHT         = 0.22   // mid-ocean ridge amplitude
const BR_RIDGE_WIDTH          = 0.05   // ridge flank width (rad)
const BR_NOTCH_DEPTH          = 0.14   // axial rift notch depth
const BR_NOTCH_WIDTH          = 0.012  // notch half-width (rad)
const BR_RIFT_SHOULDER        = 0.07   // rift shoulder amplitude [was 0.18 — reduced for modest shoulder relief]
const BR_RIFT_SHOULDER_POS    = 0.022  // shoulder center (rad)
const BR_RIFT_SHOULDER_WIDTH  = 0.012  // shoulder width (rad)
const BR_GRABEN_DEPTH         = 0.24   // graben valley depth
const BR_GRABEN_WIDTH         = 0.020  // graben half-width (rad)
// Transform
const BR_SCARP_DEPTH          = 0.10   // fault scarp/valley depth
const BR_SCARP_WIDTH          = 0.024  // scarp half-width (rad) [was 0.008 — widened to ~1200m to hide 177m cell staircase]
const BR_RIDGE2_HEIGHT        = 0.05   // parallel ridging amplitude
const BR_RIDGE2_POS           = 0.012  // parallel ridge center (rad)
const BR_RIDGE2_WIDTH         = 0.008  // parallel ridge width (rad)
// Crust-aware regime thresholds
const CRUST_MINE_THRESH  = -0.020  // crustDist > this → we're on continental crust
const CRUST_CORD_SS_LO   = -0.02   // cordillera/belt smoothstep lo
const CRUST_CORD_SS_HI   =  0.01   // cordillera/belt smoothstep hi

/**
 * Asymmetric tectonic boundary relief profiles.
 *
 * Returns a height contribution in normalized units ≈[-1,1].
 * All terms are continuous in conv/shear (no hard regime branches on sign;
 * only smooth magnitude-scaled envelopes). The only permitted discontinuity
 * is at the exact plate boundary where plateId flips.
 *
 * @param q          TectonicQuery for the current sample point
 * @param plates     full Plate array from the Tectonics instance
 * @param dir        unit planet-local direction (used for noise sampling only)
 * @param ridgedAt   wraps a ridged noise call returning ≈[0,1]
 */
export function boundaryRelief(
  q: TectonicQuery,
  plates: Plate[],
  dir: Vector3,
  ridgedAt: (dir: Vector3, freq: number, octaves: number) => number,
  base: number,
): number {
  const d    = q.boundaryDist   // radians
  const sideRamp = _ss(0.002, 0.015, d)
  const own  = plates[q.plateId]
  const other = plates[q.neighborId]
  const conv  = q.convergence
  const sh    = q.shear

  // --- Convergent envelope ---
  const cp  = Math.max(0, conv)
  const cp2 = cp * _ss(0.04, 0.15, cp)

  // --- Divergent envelope ---
  const absConv = Math.abs(conv)
  const dp  = Math.max(0, -conv) * _ss(0.04, 0.15, absConv)

  // --- Transform envelope (engages when sliding dominates) ---
  const abssh = Math.abs(sh)
  const tw  = abssh * _ss(0.10, 0.30, abssh) * (1 - _ss(0.05, 0.15, Math.abs(conv)))

  // No cross-boundary ramp: now that conv/shear are sampled from pre-baked blurred
  // fields, cp2 and dp are smooth. The ramp was previously needed to suppress
  // the gradient-direction sign flip, but that source of discontinuity is gone.
  // Keeping the ramp would create artificial steep forearc drops.
  const cp2r = cp2
  const dpr  = dp

  // --- Continuous regime weights (0=oceanic, 1=continental) ---
  // Wide window (0.20 rad = ~16 texels on each side of zero) prevents rapid
  // per-texel weight changes that would cause cliff-like height discontinuities.
  const wMine  = _ss(-0.10, 0.10, q.crustDist)
  const wOther = _ss(-0.10, 0.10, q.otherCrustDist)
  // cordScaleFactor = wMine: reuse the same weight to avoid a second rapid
  // smoothstep on crustDist that would compound the gradient slope.
  const cordScaleFactor = wMine

  let relief = 0

  if (cp2r > 0) {
    // Continuous regime blending — weights 0=oceanic 1=continental
    const wOC = (1 - wMine) * wOther        // oceanic subducting under continent
    const wCO = wMine * (1 - wOther)         // continental overriding
    const wOO = (1 - wMine) * (1 - wOther)  // oceanic-oceanic
    const wCC = wMine * wOther               // continent-continent collision

    // OC: Mariana Trench (oceanic subducting side)
    relief += wOC * (-BR_TRENCH_DEPTH * cp2r * _g(d, 0, BR_TRENCH_WIDTH)
                    + BR_FLEX_BULGE  * cp2r * _g(d, BR_FLEX_POS, BR_FLEX_WIDTH))

    // CO: Andes / Cascades (continental overriding side)
    const ridgeFactor = BR_CORD_RIDGE_BASE + BR_CORD_RIDGE * ridgedAt(dir, 6.0, 4)
    relief += wCO * (BR_CORD_HEIGHT  * cp2r * _g(d, BR_CORD_POS, BR_CORD_WIDTH) * ridgeFactor * cordScaleFactor
                    - BR_SHELF_DIP  * cp2r * _g(d, 0.004, BR_SHELF_WIDTH))

    // OO: island arc (Japan-style) — polarity ramps from 0.5 at the boundary to asymmetric beyond 0.015 rad
    const ooSubduct = 0.5 + ((own.id < other.id ? 1 : 0) - 0.5) * sideRamp
    // 1a: low-freq along-strike envelope (~125 km features) gates discrete island clusters;
    //     smooth ramp keeps cluster edges soft and lets a faint submarine arc survive between clusters
    const arcActivity = ridgedAt(dir, 0.4, 2)  // feature size ≈ RADIUS/0.4 ≈ 125 km
    const arcClusterGate = _ss(0.45, 0.65, arcActivity)
    const arcRidged = Math.max(0, ridgedAt(dir, 9.0, 4) - BR_ARC_RIDGE_THRESH) * BR_ARC_RIDGE_SCALE
    // 1b: sea-level emergence clamp — BR_ARC_HEIGHT is large enough for arcRelief to hit the cap at
    //     strong cone peaks (base≈-0.42 → cap≈0.45), so `base + arcRelief ≈ +0.03` (low island);
    //     everywhere below the cap arcRelief stays submarine.  SEA_ISLAND_HEADROOM = 0.05 is the
    //     small extra lift that lets the very tip of a gated cone break the surface.
    const SEA_MARGIN = 0.02
    const SEA_ISLAND_HEADROOM = 0.05
    let arcRelief = BR_ARC_HEIGHT * cp2r * _g(d, BR_ARC_POS, BR_ARC_WIDTH) * arcRidged * arcClusterGate
    arcRelief = Math.min(arcRelief, Math.max(0, -base - SEA_MARGIN) + SEA_ISLAND_HEADROOM)
    relief += wOO * (
        ooSubduct * (-BR_OO_TRENCH_DEPTH * cp2r * _g(d, 0, BR_OO_TRENCH_WIDTH))
      + (1 - ooSubduct) * arcRelief
    )

    // CC: Himalaya / Alps collision belt
    const beltRidged = BR_BELT_RIDGE_BASE + BR_BELT_RIDGE * ridgedAt(dir, 6.0, 5)
    relief += wCC * (BR_BELT_HEIGHT * cp2r * _g(d, 0, BR_BELT_WIDTH) * beltRidged * cordScaleFactor)
    // Tibetan Plateau — polarity ramps from 0.5 at the boundary to asymmetric beyond 0.015 rad
    const ccPlateauSide = 0.5 + ((own.id > other.id ? 1 : 0) - 0.5) * sideRamp
    const plateauMask = (1 - _ss(BR_PLATEAU_INNER, BR_PLATEAU_OUTER, d))
                      * _ss(BR_PLATEAU_NEAR, BR_PLATEAU_FAR, d)
    relief += wCC * ccPlateauSide * (BR_PLATEAU_HEIGHT * cp2r * plateauMask * cordScaleFactor)
  }

  if (dpr > 0) {
    // Mid-ocean ridge (oceanic side) — stays submarine; ridgeCap suppresses emergence
    const ridgeCap = 1 - _ss(-0.10, -0.02, base)
    relief += (1 - wMine) * ridgeCap * (BR_RIDGE_HEIGHT * dpr * _g(d, 0, BR_RIDGE_WIDTH)
                                       - BR_NOTCH_DEPTH  * dpr * _g(d, 0, BR_NOTCH_WIDTH))
    // East African Rift (continental side)
    const shoulderSuppress = 1 - _ss(0.05, 0.25, base)
    relief += wMine * (BR_RIFT_SHOULDER * shoulderSuppress * dpr * _g(d, BR_RIFT_SHOULDER_POS, BR_RIFT_SHOULDER_WIDTH)
                     - BR_GRABEN_DEPTH  * dpr * _g(d, 0, BR_GRABEN_WIDTH))
  }

  if (tw > 0) {
    // San Andreas Fault — narrow fault scarp/valley + subtle parallel ridging
    relief += -BR_SCARP_DEPTH  * tw * _g(d, 0, BR_SCARP_WIDTH)
           + BR_RIDGE2_HEIGHT * tw * _g(d, BR_RIDGE2_POS, BR_RIDGE2_WIDTH)
                              * ridgedAt(dir, 11.0, 3)
  }

  return relief
}
