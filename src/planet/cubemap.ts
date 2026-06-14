/**
 * Cube-map utilities: pure, allocation-free, cross-face safe.
 *
 * Face order: 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z
 * Same convention as faceBases.ts (FACE_BASES[i] gives normal/tangents).
 *
 * Cross-face neighbours use the "dir-roundtrip" trick rather than hand-coded
 * adjacency tables:
 *   1. Compute the texel-centre direction.
 *   2. Nudge it by ~1.2 texel-angles along the appropriate face-tangent axis.
 *   3. Re-project with dirToTexel → correct face, correct texel, correct
 *      orientation automatically (the projection always picks the dominant axis).
 * No 12-edge table, no special cases.
 */

import { Vector3 } from 'three'
import { FACE_BASES } from './faceBases'

// ---------------------------------------------------------------------------
// dirToTexel — max-axis cube projection, nearest texel
// ---------------------------------------------------------------------------

/**
 * Project a unit direction onto the cube face whose normal is closest
 * (maximum dot product with direction = maximum absolute component).
 * Returns the face index and nearest texel coordinates (integer, 0..res-1).
 *
 * All operations are in-place; no allocations.
 */
export function dirToTexel(
  dir: Vector3,
  res: number,
  out: { face: number; x: number; y: number },
): void {
  const ax = Math.abs(dir.x)
  const ay = Math.abs(dir.y)
  const az = Math.abs(dir.z)

  let face: number
  let sc: number  // signed coord along U axis on this face
  let tc: number  // signed coord along V axis on this face
  let mc: number  // magnitude of the dominant axis (for division)

  if (ax >= ay && ax >= az) {
    // Dominant axis: X
    if (dir.x > 0) {
      // Face 0: +X  — basis u=+Z, v=+Y
      face = 0; mc = ax; sc = dir.z; tc = dir.y
    } else {
      // Face 1: −X  — basis u=−Z, v=+Y
      face = 1; mc = ax; sc = -dir.z; tc = dir.y
    }
  } else if (ay >= ax && ay >= az) {
    // Dominant axis: Y
    if (dir.y > 0) {
      // Face 2: +Y  — basis u=+X, v=+Z
      face = 2; mc = ay; sc = dir.x; tc = dir.z
    } else {
      // Face 3: −Y  — basis u=+X, v=−Z
      face = 3; mc = ay; sc = dir.x; tc = -dir.z
    }
  } else {
    // Dominant axis: Z
    if (dir.z > 0) {
      // Face 4: +Z  — basis u=−X, v=+Y
      face = 4; mc = az; sc = -dir.x; tc = dir.y
    } else {
      // Face 5: −Z  — basis u=+X, v=+Y
      face = 5; mc = az; sc = dir.x; tc = dir.y
    }
  }

  // Map sc/mc and tc/mc from [−1,1] → [0,res); clamp to valid range.
  const half = res * 0.5
  const xi = Math.floor((sc / mc + 1.0) * half)
  const yi = Math.floor((tc / mc + 1.0) * half)

  out.face = face
  out.x = Math.min(res - 1, Math.max(0, xi))
  out.y = Math.min(res - 1, Math.max(0, yi))
}

// ---------------------------------------------------------------------------
// texelToDir — texel-centre direction (unit vector)
// ---------------------------------------------------------------------------

/**
 * Compute the unit-sphere direction at the CENTRE of texel (face, x, y).
 * Uses the face's normal + tangent basis from FACE_BASES, then normalizes.
 * Writes result into `out` (no allocation).
 */
export function texelToDir(
  face: number,
  x: number,
  y: number,
  res: number,
  out: Vector3,
): void {
  const b = FACE_BASES[face]
  // Map texel centre to [−1, 1]² face coords
  const half = res * 0.5
  const u = (x + 0.5) / half - 1.0   // (x + 0.5) * 2/res - 1
  const v = (y + 0.5) / half - 1.0

  // Cube point = normal + u*tangentU + v*tangentV
  out.set(
    b.nx + u * b.ux + v * b.vx,
    b.ny + u * b.uy + v * b.vy,
    b.nz + u * b.uz + v * b.vz,
  ).normalize()
}

// ---------------------------------------------------------------------------
// texelIndex — flat array address
// ---------------------------------------------------------------------------

/** face*res*res + y*res + x */
export function texelIndex(face: number, x: number, y: number, res: number): number {
  return face * res * res + y * res + x
}

// ---------------------------------------------------------------------------
// texAng — per-texel angular pitch (radians)
// ---------------------------------------------------------------------------

/**
 * Angular pitch per texel at the face centre (uniform approximation).
 * Exact at the face centre; fine as a uniform metric for nudge distances.
 * (π/2) / res radians per texel — a quarter circle divided by res.
 */
export function texAng(res: number): number {
  return (Math.PI * 0.5) / res
}

// ---------------------------------------------------------------------------
// neighborTexel — cross-face neighbour via dir-roundtrip
// ---------------------------------------------------------------------------

// Scratch vectors used inside neighborTexel — module-level to avoid allocations.
// These are NOT re-entrant safe, but neighborTexel is only called in single-
// threaded BFS passes (never nested), so this is fine.
const _ntDir  = new Vector3()
const _ntNudge = new Vector3()

/**
 * Return the texel that is one grid step (dx, dy) from (face, x, y).
 *
 * In-bounds case: trivial integer offset.
 * Out-of-bounds (crossing a face edge): dir-roundtrip trick —
 *   1. texelToDir(face, x, y) → base direction d.
 *   2. Nudge d by 1.2·texAng along face tangent U (for dx≠0) and/or V (for dy≠0).
 *      The 1.2× factor ensures we clear the face edge by at least 0.2 texels.
 *   3. dirToTexel(nudged, res) → always lands on the correct neighbour face with
 *      correct orientation automatically (no hand-built adjacency table needed).
 *
 * For diagonal steps (dx≠0 && dy≠0) crossing an edge, both tangents are nudged
 * simultaneously. The result may land on either of the two adjacent faces — that
 * is geometrically correct and fine for our BFS/flood-fill purposes.
 *
 * The face's U/V basis vectors are read from FACE_BASES (same as faceBases.ts),
 * so orientation is consistent with the rest of the system.
 */
export function neighborTexel(
  face: number,
  x: number,
  y: number,
  dx: -1 | 0 | 1,
  dy: -1 | 0 | 1,
  res: number,
  out: { face: number; x: number; y: number },
): void {
  const nx = x + dx
  const ny = y + dy

  // Fast in-bounds path — no projection needed
  if (nx >= 0 && nx < res && ny >= 0 && ny < res) {
    out.face = face
    out.x = nx
    out.y = ny
    return
  }

  // Out-of-bounds: dir-roundtrip via tangent nudge.
  // Compute base direction at texel centre.
  texelToDir(face, x, y, res, _ntDir)

  const ang = texAng(res) * 1.2
  const b = FACE_BASES[face]

  // Nudge along U tangent when stepping in X direction (columns)
  if (dx !== 0) {
    // U-tangent of this face
    _ntNudge.set(b.ux, b.uy, b.uz).multiplyScalar(dx * ang)
    _ntDir.add(_ntNudge)
  }

  // Nudge along V tangent when stepping in Y direction (rows)
  if (dy !== 0) {
    // V-tangent of this face
    _ntNudge.set(b.vx, b.vy, b.vz).multiplyScalar(dy * ang)
    _ntDir.add(_ntNudge)
  }

  _ntDir.normalize()
  dirToTexel(_ntDir, res, out)
}

// ---------------------------------------------------------------------------
// sampleSmooth — true bilinear interpolation for smooth cross-face sampling
// ---------------------------------------------------------------------------

// Scratch state for sampleSmooth — module-level, no allocations in hot path.
// Single-threaded use only (never nested / re-entrant), same as neighborTexel.
const _ssTex0: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
const _ssTex1: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
const _ssTex2: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }
const _ssTex3: { face: number; x: number; y: number } = { face: 0, x: 0, y: 0 }

/**
 * Sample a per-texel Float32Array smoothly at direction `dir`.
 *
 * Method: TRUE BILINEAR interpolation. Project `dir` to continuous cube-face
 * coordinates (face + floating texel coords, where integer values are texel
 * centres — matching texelToDir's `(x+0.5)` convention). Find the 2×2 cell of
 * texels the point falls in plus the fractional weights (fu, fv), then bilinearly
 * blend the 4 values:  lerp(lerp(t00,t10,fu), lerp(t01,t11,fu), fv).
 *
 * Unlike the previous 4-tap box average (which was piecewise-constant within a
 * texel and jumped at texel boundaries — producing visible ~300m blocky cells
 * at coastlines), this varies continuously across the whole field.
 *
 * Interior fast path: all 4 texels in-bounds on the same face → direct fetch.
 * Edge path: any of the 4 texels crosses a face boundary → fetch via
 *   neighborTexel (the dir-roundtrip cross-face helper) so seams stay correct.
 * Corner approximation: at the 8 cube corners only 3 distinct texels meet; the
 *   diagonal neighbor there resolves (via neighborTexel) to one of the 3 real
 *   texels, so the corner texel value is effectively reused — an accepted, tiny
 *   approximation confined to the 8 corner cells.
 *
 * `scratch` is a caller-supplied Vector3 (unused now — kept for signature
 * compatibility; callers pass a non-aliased temporary). All state is either the
 * scratch or preallocated at module level — ZERO allocations in the hot path.
 * Pure & deterministic: same (data, dir, res) → same result.
 */
export function sampleSmooth(
  data: Float32Array,
  dir: Vector3,
  res: number,
  scratch: Vector3,
): number {
  // --- Project dir onto the dominant cube face, capturing CONTINUOUS coords ---
  const dx = dir.x, dy = dir.y, dz = dir.z
  const ax = Math.abs(dx)
  const ay = Math.abs(dy)
  const az = Math.abs(dz)

  let face: number
  let sc: number  // signed U-axis coord on this face
  let tc: number  // signed V-axis coord on this face
  let mc: number  // magnitude of dominant axis

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

  // Continuous texel coords: integer value == texel centre (matches texelToDir,
  // which places centres at (i + 0.5)). dirToTexel uses floor((sc/mc+1)*half);
  // a texel centre i sits at (sc/mc+1)*half = i + 0.5, so subtract 0.5 here.
  const half = res * 0.5
  const fx = (sc / mc + 1.0) * half - 0.5
  const fy = (tc / mc + 1.0) * half - 0.5

  // Cell the point falls in: integer corner (x0,y0) and fractional weights.
  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const fu = fx - x0
  const fv = fy - y0
  const x1 = x0 + 1
  const y1 = y0 + 1

  let t00: number, t10: number, t01: number, t11: number

  // --- Interior fast path: whole 2×2 cell on this face, in-bounds ---
  if (x0 >= 0 && y0 >= 0 && x1 < res && y1 < res) {
    const base = face * res * res
    const row0 = base + y0 * res
    const row1 = base + y1 * res
    t00 = data[row0 + x0]
    t10 = data[row0 + x1]
    t01 = data[row1 + x0]
    t11 = data[row1 + x1]
  } else {
    // --- Edge path: one or more neighbors cross a face boundary ---
    // Anchor texel = nearest in-bounds texel of the 2×2 cell (clamped). All four
    // corners are reached as integer (dx,dy) offsets from this anchor and routed
    // through neighborTexel, which returns the correct cross-face texel (and is
    // identity for in-bounds offsets).
    const axx = x0 < 0 ? 0 : (x0 >= res ? res - 1 : x0)
    const ayy = y0 < 0 ? 0 : (y0 >= res ? res - 1 : y0)
    // Offsets of each cell corner relative to the anchor (each in {-1,0,1}).
    const ox0 = (x0 - axx) as -1 | 0 | 1
    const ox1 = (x1 - axx) as -1 | 0 | 1
    const oy0 = (y0 - ayy) as -1 | 0 | 1
    const oy1 = (y1 - ayy) as -1 | 0 | 1

    neighborTexel(face, axx, ayy, ox0, oy0, res, _ssTex0)
    neighborTexel(face, axx, ayy, ox1, oy0, res, _ssTex1)
    neighborTexel(face, axx, ayy, ox0, oy1, res, _ssTex2)
    neighborTexel(face, axx, ayy, ox1, oy1, res, _ssTex3)

    t00 = data[texelIndex(_ssTex0.face, _ssTex0.x, _ssTex0.y, res)]
    t10 = data[texelIndex(_ssTex1.face, _ssTex1.x, _ssTex1.y, res)]
    t01 = data[texelIndex(_ssTex2.face, _ssTex2.x, _ssTex2.y, res)]
    t11 = data[texelIndex(_ssTex3.face, _ssTex3.x, _ssTex3.y, res)]
  }

  // touch scratch to keep TS happy about the (now-unused) param without a
  // separate eslint pragma; this is a no-op write the optimizer can drop.
  void scratch

  const a = t00 + (t10 - t00) * fu
  const c = t01 + (t11 - t01) * fu
  return a + (c - a) * fv
}
