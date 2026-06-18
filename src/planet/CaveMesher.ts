/**
 * CaveMesher.ts — Surface Nets cave geometry extractor.
 *
 * Consumes a CaveField SDF and outputs walkable cave geometry as typed arrays
 * (worker-safe compute core) or a BufferGeometry (main-thread upload path).
 *
 * Design mirrors ChunkMesher.ts:
 *   computeCaveArrays  — pure, typed-arrays-only, safe to run in a Web Worker.
 *   caveArraysToGeometry — builds the THREE BufferGeometry on the main thread.
 *
 * Surface Nets algorithm (Paul Bourke / Gibson 1998 variant):
 *   1. Fill a (VOX+1)³ density lattice using the cheap densityWithSurface path
 *      with a coarse-cached surfaceR (see §surfaceR cache below).
 *   2. For each of the VOX³ cells, test if all 8 corners share the same sign.
 *      Active cells (sign changes) get one vertex — the average of the
 *      linearly-interpolated zero-crossings on the up-to-12 crossing edges.
 *   3. For each axis-aligned grid EDGE with a sign change, emit a quad (2 tris)
 *      linking the vertices of the up-to-4 cells that share that edge.
 *
 * surfaceR cache:
 *   field.surfaceRadiusLocal(dir) calls heightFn — expensive (~14 noise octaves).
 *   Across one 32 m chunk the unit direction p.normalize() varies slowly, so we
 *   sample surfaceR on a coarse NxN grid of directions spanning the chunk's
 *   lattice and bilinearly interpolate for the dense (VOX+1)³ samples.
 *   We use N=SURF_CACHE_N=6, giving 36 heightFn evaluations instead of (VOX+1)³ = 15625.
 *   Accuracy: at RADIUS=50000 m a 32 m chunk subtends ~0.036°; over 6 samples the
 *   angular cell is ~0.007° — the interpolation error in surfaceR is negligible
 *   compared to the ~0.25 m NORMAL_H used inside the field itself.
 *
 * Normal orientation:
 *   field.normalAt returns ∇density, which points from air TOWARD rock (outward
 *   from the void). For a player inside the cave, inward-facing normals (pointing
 *   INTO the rock from the void) are what you want for lighting the cave walls.
 *   We therefore NEGATE the gradient → inward-facing normals.
 *
 * Winding:
 *   For each X-axis edge (ix→ix+1) at lattice point (ix, iy, iz) with a sign
 *   change, the four sharing cells are:
 *     c00 = cell(ix, iy-1, iz-1), c10 = cell(ix, iy,   iz-1),
 *     c01 = cell(ix, iy-1, iz  ), c11 = cell(ix, iy,   iz  )
 *   The quad orientation is chosen so the normal points in the -X direction when
 *   the density at ix is NEGATIVE (air) and at ix+1 is POSITIVE (rock), i.e.
 *   the face points back toward the air side.  We flip the winding when the sign
 *   direction reverses.  This ensures consistent inward-facing orientation with
 *   the negated gradient normals.
 *
 * Known correctness note on dual-face winding:
 *   The winding rule works reliably for simple topologies. In cells where the
 *   density isosurface is highly curved (multiple sign changes on one cell's 12
 *   edges), Surface Nets places one vertex per cell and can produce slightly
 *   inconsistent quads at sharp concavities — this is an inherent limitation of
 *   Surface Nets vs. Marching Cubes / Dual Contouring. For cave geometry at
 *   ~1.33 m voxels this is not expected to be visible in practice.
 */

import { BufferGeometry, BufferAttribute, Vector3 } from 'three'
import { CaveField } from './caveField'

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/** Meters per cave-chunk edge. */
export const CAVE_CHUNK_M = 32

/** Voxels per chunk edge (≈ 1.33 m voxels). */
export const CAVE_VOX = 24

// ---------------------------------------------------------------------------
// surfaceR cache resolution — small NxN grid of direction samples per chunk.
// 6×6 = 36 heightFn evals instead of (VOX+1)³ = 15625.
// ---------------------------------------------------------------------------
const SURF_CACHE_N = 6

// ---------------------------------------------------------------------------
// Cave rock color: dark warm rock #3a3530
// ---------------------------------------------------------------------------
const ROCK_R = 0x3a / 255
const ROCK_G = 0x35 / 255
const ROCK_B = 0x30 / 255

// ---------------------------------------------------------------------------
// Slightly lighter variation for AO-proxy blend target: #55504a
// ---------------------------------------------------------------------------
const ROCK_LT_R = 0x55 / 255
const ROCK_LT_G = 0x50 / 255
const ROCK_LT_B = 0x4a / 255

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

export interface CaveChunkParams {
  field: CaveField
  cx: number   // chunk grid X index
  cy: number   // chunk grid Y index
  cz: number   // chunk grid Z index
}

export interface CaveMeshArrays {
  positions:  Float32Array   // origin-relative vertex positions, stride 3
  normals:    Float32Array   // inward-facing (negated gradient), stride 3
  colors:     Float32Array   // dark rock palette, stride 3
  indices:    Uint32Array
  originX:    number
  originY:    number
  originZ:    number
  vertCount:  number
  indexCount: number
  empty:      boolean        // true when no iso-crossing was found
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

/** Flat index into a (VOX+1)³ lattice. */
function latIdx(x: number, y: number, z: number): number {
  const S = CAVE_VOX + 1
  return x + y * S + z * S * S
}

/** Flat index into a VOX³ cell grid. */
function cellIdx(cx: number, cy: number, cz: number): number {
  return cx + cy * CAVE_VOX + cz * CAVE_VOX * CAVE_VOX
}

// ---------------------------------------------------------------------------
// computeCaveArrays — worker-safe: only typed arrays + Vector3 math, no THREE GPU objects
// ---------------------------------------------------------------------------

export function computeCaveArrays(p: CaveChunkParams): CaveMeshArrays {
  const { field, cx, cy, cz } = p

  const S = CAVE_VOX + 1    // lattice edge size (samples per edge)
  const STEP = CAVE_CHUNK_M / CAVE_VOX  // meters per voxel ≈ 1.333 m

  // Chunk corner in planet-local space.
  const cornerX = cx * CAVE_CHUNK_M
  const cornerY = cy * CAVE_CHUNK_M
  const cornerZ = cz * CAVE_CHUNK_M

  // Chunk center = origin for origin-relative vertex output.
  const originX = cornerX + CAVE_CHUNK_M * 0.5
  const originY = cornerY + CAVE_CHUNK_M * 0.5
  const originZ = cornerZ + CAVE_CHUNK_M * 0.5

  // -------------------------------------------------------------------------
  // Step 0: build coarse surfaceR cache (SURF_CACHE_N × SURF_CACHE_N grid of
  // directions across the lattice, bilinearly interpolated for each sample).
  //
  // We span directions from the lattice's (0,0) corner to the (CAVE_VOX,CAVE_VOX)
  // corner along the XY face (using Z fixed at the chunk center Z, which is
  // sufficient since the angular span of the chunk is tiny).  In practice we
  // build a 2D cache in "lattice ix/iy" space and index by the normalized
  // (ix,iy) ∈ [0,1] of the sample point.  For a 32 m chunk at RADIUS=50000 the
  // directional variation along Z is also tiny, so a 2D angular grid is accurate.
  //
  // Cache layout: surfCache[a + b * SURF_CACHE_N] for (a in 0..N-1, b in 0..N-1).
  // The two cache axes correspond to the lattice X (a) and Y (b) axes.
  // -------------------------------------------------------------------------
  const surfCache = new Float32Array(SURF_CACHE_N * SURF_CACHE_N)
  const _cacheDir = new Vector3()
  const _cachePos = new Vector3()

  // Use chunk center Z for the cache directions (good enough — Z variation
  // across a 32 m chunk subtends a negligible angle at RADIUS=50000 m).
  const midZ = cornerZ + CAVE_CHUNK_M * 0.5

  for (let b = 0; b < SURF_CACHE_N; b++) {
    for (let a = 0; a < SURF_CACHE_N; a++) {
      // Lattice indices this cache cell maps to (evenly spaced 0..CAVE_VOX).
      const latX = (a / (SURF_CACHE_N - 1)) * CAVE_VOX
      const latY = (b / (SURF_CACHE_N - 1)) * CAVE_VOX
      const wx = cornerX + latX * STEP
      const wy = cornerY + latY * STEP
      _cachePos.set(wx, wy, midZ)
      _cacheDir.copy(_cachePos).normalize()
      surfCache[a + b * SURF_CACHE_N] = field.surfaceRadiusLocal(_cacheDir)
    }
  }

  /**
   * Bilinearly interpolate surfaceR for a lattice point (ix, iy, iz).
   * The cache is 2D (XY), valid because Z variation is negligible.
   */
  function cachedSurfaceR(ix: number, iy: number): number {
    // Normalised cache coordinates.
    const u = (ix / CAVE_VOX) * (SURF_CACHE_N - 1)
    const v = (iy / CAVE_VOX) * (SURF_CACHE_N - 1)
    const a0 = Math.min(Math.floor(u), SURF_CACHE_N - 2)
    const b0 = Math.min(Math.floor(v), SURF_CACHE_N - 2)
    const fu = u - a0
    const fv = v - b0
    const s00 = surfCache[a0     + b0 * SURF_CACHE_N]
    const s10 = surfCache[(a0+1) + b0 * SURF_CACHE_N]
    const s01 = surfCache[a0     + (b0+1) * SURF_CACHE_N]
    const s11 = surfCache[(a0+1) + (b0+1) * SURF_CACHE_N]
    return s00 * (1-fu)*(1-fv) + s10 * fu*(1-fv) + s01 * (1-fu)*fv + s11 * fu*fv
  }

  // -------------------------------------------------------------------------
  // Step 1: fill density lattice (S³ points).
  // -------------------------------------------------------------------------
  const density = new Float32Array(S * S * S)
  const _sampleP = new Vector3()

  for (let iz = 0; iz < S; iz++) {
    for (let iy = 0; iy < S; iy++) {
      for (let ix = 0; ix < S; ix++) {
        const wx = cornerX + ix * STEP
        const wy = cornerY + iy * STEP
        const wz = cornerZ + iz * STEP
        _sampleP.set(wx, wy, wz)
        // cachedSurfaceR uses only the XY axes (2D cache) — Z variation across
        // 32 m at RADIUS=50000 m contributes < 0.001% error in surfR.
        density[latIdx(ix, iy, iz)] = field.densityWithSurface(_sampleP, cachedSurfaceR(ix, iy))
      }
    }
  }

  // -------------------------------------------------------------------------
  // Step 2 + 3: Surface Nets — active-cell vertices + dual-face quads.
  //
  // We use a two-pass approach:
  //   Pass A: walk cells, compute vertices, fill cellVertIdx map.
  //   Pass B: walk edges, emit quads from the cellVertIdx map.
  //
  // Growable scratch arrays (Float64 for accumulation) are used in Pass A,
  // then trimmed to exact-size typed arrays at the end.
  // -------------------------------------------------------------------------

  // cellVertIdx[cellIdx] = vertex index (0-based), or -1 = inactive.
  const cellVertIdx = new Int32Array(CAVE_VOX * CAVE_VOX * CAVE_VOX).fill(-1)

  // Growable vertex + normal + color scratch (over-allocated, trimmed later).
  const maxVerts = CAVE_VOX * CAVE_VOX * CAVE_VOX   // upper bound
  const posScratch = new Float32Array(maxVerts * 3)
  const nrmScratch = new Float32Array(maxVerts * 3)
  const colScratch = new Float32Array(maxVerts * 3)

  let vertCount = 0

  // Scratch reuse across cells.
  const _cellPos  = new Vector3()  // iso-crossing accumulator (world-local)
  const _nrmOut   = new Vector3()
  const _edgePosA = new Vector3()
  const _edgePosB = new Vector3()

  // ---- Pass A: one vertex per active cell ---------------------------------
  for (let cz2 = 0; cz2 < CAVE_VOX; cz2++) {
    for (let cy2 = 0; cy2 < CAVE_VOX; cy2++) {
      for (let cx2 = 0; cx2 < CAVE_VOX; cx2++) {

        // Read the 8 corner densities of this cell.
        // Corner bit layout: bit0=+X, bit1=+Y, bit2=+Z (standard).
        const d000 = density[latIdx(cx2,   cy2,   cz2  )]
        const d100 = density[latIdx(cx2+1, cy2,   cz2  )]
        const d010 = density[latIdx(cx2,   cy2+1, cz2  )]
        const d110 = density[latIdx(cx2+1, cy2+1, cz2  )]
        const d001 = density[latIdx(cx2,   cy2,   cz2+1)]
        const d101 = density[latIdx(cx2+1, cy2,   cz2+1)]
        const d011 = density[latIdx(cx2,   cy2+1, cz2+1)]
        const d111 = density[latIdx(cx2+1, cy2+1, cz2+1)]

        // All-positive (solid) or all-negative (air) → inactive.
        const allPos = (d000 > 0 && d100 > 0 && d010 > 0 && d110 > 0 &&
                        d001 > 0 && d101 > 0 && d011 > 0 && d111 > 0)
        const allNeg = (d000 < 0 && d100 < 0 && d010 < 0 && d110 < 0 &&
                        d001 < 0 && d101 < 0 && d011 < 0 && d111 < 0)
        if (allPos || allNeg) continue

        // Active cell: accumulate iso-crossing positions for the 12 edges.
        // Edge list (local corner indices, dCornerA, dCornerB, frac axis):
        //   12 edges of the unit cube. We encode each as two world offsets
        //   along the lattice axes and their density values.
        let sumX = 0, sumY = 0, sumZ = 0
        let crossCount = 0

        // Helper: add crossing on segment (ax,ay,az)→(ax,ay,az)+axis
        function addCrossing(
          ax: number, ay: number, az: number,
          bx: number, by: number, bz: number,
          dA: number, dB: number,
        ): void {
          if ((dA < 0) === (dB < 0)) return  // no crossing
          const t = dA / (dA - dB)           // dA + t*(dB-dA) = 0
          sumX += ax + t * (bx - ax)
          sumY += ay + t * (by - ay)
          sumZ += az + t * (bz - az)
          crossCount++
        }

        // 8 corners in local-space coords (relative to cornerX/Y/Z)
        const lx0 = cx2 * STEP, lx1 = (cx2+1) * STEP
        const ly0 = cy2 * STEP, ly1 = (cy2+1) * STEP
        const lz0 = cz2 * STEP, lz1 = (cz2+1) * STEP

        // 12 edges
        addCrossing(lx0,ly0,lz0, lx1,ly0,lz0, d000,d100)  // edge 0: X at (0,0,0)
        addCrossing(lx0,ly1,lz0, lx1,ly1,lz0, d010,d110)  // edge 1: X at (0,1,0)
        addCrossing(lx0,ly0,lz1, lx1,ly0,lz1, d001,d101)  // edge 2: X at (0,0,1)
        addCrossing(lx0,ly1,lz1, lx1,ly1,lz1, d011,d111)  // edge 3: X at (0,1,1)
        addCrossing(lx0,ly0,lz0, lx0,ly1,lz0, d000,d010)  // edge 4: Y at (0,0,0)
        addCrossing(lx1,ly0,lz0, lx1,ly1,lz0, d100,d110)  // edge 5: Y at (1,0,0)
        addCrossing(lx0,ly0,lz1, lx0,ly1,lz1, d001,d011)  // edge 6: Y at (0,0,1)
        addCrossing(lx1,ly0,lz1, lx1,ly1,lz1, d101,d111)  // edge 7: Y at (1,0,1)
        addCrossing(lx0,ly0,lz0, lx0,ly0,lz1, d000,d001)  // edge 8: Z at (0,0,0)
        addCrossing(lx1,ly0,lz0, lx1,ly0,lz1, d100,d101)  // edge 9: Z at (1,0,0)
        addCrossing(lx0,ly1,lz0, lx0,ly1,lz1, d010,d011)  // edge 10: Z at (0,1,0)
        addCrossing(lx1,ly1,lz0, lx1,ly1,lz1, d110,d111)  // edge 11: Z at (1,1,0)

        if (crossCount === 0) continue  // degenerate (all exactly 0) → skip

        // Average of crossings → local-space vertex position.
        const invC = 1.0 / crossCount
        const vx = cornerX + sumX * invC   // world-local
        const vy = cornerY + sumY * invC
        const vz = cornerZ + sumZ * invC

        // Normal: field.normalAt returns outward-from-void gradient, so negate
        // it to get inward-facing (pointing into rock from player's perspective).
        _cellPos.set(vx, vy, vz)
        field.normalAt(_cellPos, _nrmOut)
        _nrmOut.negate()  // inward-facing

        // Cheap AO proxy: sample 6 neighbour cells' average density.
        // Higher average density around the vertex → darker (buried in rock).
        // This is a cheap approximation — not real AO — but adds visual variation.
        const dNeighbAvg = (d000 + d100 + d010 + d110 + d001 + d101 + d011 + d111) * 0.125
        // dNeighbAvg > 0: vertex near a lot of rock → darker. Range: roughly [-1..BIG].
        // Normalise to [0..1] using a soft saturate, then darken.
        const aoProxy = clamp01(dNeighbAvg * 0.5)  // 0 = air-side, 1 = deep rock
        const cr = ROCK_R + (ROCK_LT_R - ROCK_R) * (1 - aoProxy)
        const cg = ROCK_G + (ROCK_LT_G - ROCK_G) * (1 - aoProxy)
        const cb = ROCK_B + (ROCK_LT_B - ROCK_B) * (1 - aoProxy)

        // Store vertex (origin-relative).
        const vi = vertCount++
        posScratch[vi * 3    ] = vx - originX
        posScratch[vi * 3 + 1] = vy - originY
        posScratch[vi * 3 + 2] = vz - originZ
        nrmScratch[vi * 3    ] = _nrmOut.x
        nrmScratch[vi * 3 + 1] = _nrmOut.y
        nrmScratch[vi * 3 + 2] = _nrmOut.z
        colScratch[vi * 3    ] = cr
        colScratch[vi * 3 + 1] = cg
        colScratch[vi * 3 + 2] = cb

        cellVertIdx[cellIdx(cx2, cy2, cz2)] = vi
      }
    }
  }

  // Early out: nothing to render.
  if (vertCount === 0) {
    return {
      positions:  new Float32Array(0),
      normals:    new Float32Array(0),
      colors:     new Float32Array(0),
      indices:    new Uint32Array(0),
      originX, originY, originZ,
      vertCount: 0,
      indexCount: 0,
      empty: true,
    }
  }

  // ---- Pass B: emit quads for each sign-changing edge ---------------------
  //
  // For each axis direction, for each edge in that direction:
  //   edge goes from lattice point P to P + unit(axis).
  //   The four cells sharing that edge are the 4 cells in the plane perpendicular
  //   to the axis that touch both P and P+1.
  //
  // For a +X edge at (ix, iy, iz):
  //   cells: (ix, iy-1, iz-1), (ix, iy, iz-1), (ix, iy-1, iz), (ix, iy, iz)
  //
  // For a +Y edge at (ix, iy, iz):
  //   cells: (ix-1, iy, iz-1), (ix, iy, iz-1), (ix-1, iy, iz), (ix, iy, iz)
  //
  // For a +Z edge at (ix, iy, iz):
  //   cells: (ix-1, iy-1, iz), (ix, iy-1, iz), (ix-1, iy, iz), (ix, iy, iz)
  //
  // Winding: we want the quad to face toward the AIR side of the edge.
  //   If density[P] < 0 (air) and density[P+axis] > 0 (rock), the air is at P.
  //   For inward-facing normals (into rock from void), orient the quad so its
  //   geometric normal points in the -axis direction (toward air at P).
  //
  // The four cell-vertex quad is: (c00, c01, c10, c11) where cells are indexed
  // by their two perpendicular axes.  The CCW winding (from the air side) is:
  //   flipped = false (dA<0, i.e. air at lower index): c00, c10, c01 + c01, c10, c11
  //   flipped = true  (dA>0, i.e. air at upper index): c00, c01, c10 + c10, c01, c11
  //
  // (Flipping when density[P]>0 keeps the face pointing into the void regardless
  // of which side the rock is on.)

  // Upper bound on quads: 3 * VOX² * (VOX+1) / 2  ≈ 3 * 24² * 12 ≈ 20736
  // Over-allocate generously; trim at end.
  const maxQuads = 3 * CAVE_VOX * CAVE_VOX * (CAVE_VOX + 1)
  const idxScratch = new Uint32Array(maxQuads * 6)
  let indexCount = 0

  /**
   * Emit a dual quad for one sign-changing edge, given the 4 adjacent cell
   * indices (c00, c10, c01, c11) and the flip flag.
   *
   * Cells may be out of range (< 0 boundary) — those get vertex index -1 from
   * the cellVertIdx array initialisation and are skipped (incomplete quad).
   */
  function emitQuad(
    c00: number, c10: number,
    c01: number, c11: number,
    flipWinding: boolean,
  ): void {
    const v00 = c00 >= 0 ? cellVertIdx[c00] : -1
    const v10 = c10 >= 0 ? cellVertIdx[c10] : -1
    const v01 = c01 >= 0 ? cellVertIdx[c01] : -1
    const v11 = c11 >= 0 ? cellVertIdx[c11] : -1

    // Only emit if all 4 cells have vertices (fully interior shared edge).
    if (v00 < 0 || v10 < 0 || v01 < 0 || v11 < 0) return

    if (!flipWinding) {
      // Air at lower index: normal points toward -axis (air side)
      idxScratch[indexCount++] = v00
      idxScratch[indexCount++] = v10
      idxScratch[indexCount++] = v01
      idxScratch[indexCount++] = v01
      idxScratch[indexCount++] = v10
      idxScratch[indexCount++] = v11
    } else {
      // Air at upper index: flip to keep normal pointing toward air
      idxScratch[indexCount++] = v00
      idxScratch[indexCount++] = v01
      idxScratch[indexCount++] = v10
      idxScratch[indexCount++] = v10
      idxScratch[indexCount++] = v01
      idxScratch[indexCount++] = v11
    }
  }

  // Iterate lattice interior edges only (ix in 1..VOX-1 for the perpendicular
  // coords so all 4 cells exist within [0..VOX-1]).

  // --- X-axis edges: (ix, iy, iz) for ix=0..VOX-1, iy=1..VOX-1, iz=1..VOX-1
  for (let iz = 1; iz < CAVE_VOX; iz++) {
    for (let iy = 1; iy < CAVE_VOX; iy++) {
      for (let ix = 0; ix < CAVE_VOX; ix++) {
        const dA = density[latIdx(ix,   iy, iz)]
        const dB = density[latIdx(ix+1, iy, iz)]
        if ((dA < 0) === (dB < 0)) continue  // no crossing

        // 4 cells sharing this X-edge: differ in (cy, cz) ∈ {-1,0} rel to (iy,iz)
        const c00 = cellIdx(ix, iy-1, iz-1)
        const c10 = cellIdx(ix, iy,   iz-1)
        const c01 = cellIdx(ix, iy-1, iz  )
        const c11 = cellIdx(ix, iy,   iz  )
        emitQuad(c00, c10, c01, c11, dA > 0)
      }
    }
  }

  // --- Y-axis edges: (ix, iy, iz) for ix=1..VOX-1, iy=0..VOX-1, iz=1..VOX-1
  for (let iz = 1; iz < CAVE_VOX; iz++) {
    for (let iy = 0; iy < CAVE_VOX; iy++) {
      for (let ix = 1; ix < CAVE_VOX; ix++) {
        const dA = density[latIdx(ix, iy,   iz)]
        const dB = density[latIdx(ix, iy+1, iz)]
        if ((dA < 0) === (dB < 0)) continue

        const c00 = cellIdx(ix-1, iy, iz-1)
        const c10 = cellIdx(ix,   iy, iz-1)
        const c01 = cellIdx(ix-1, iy, iz  )
        const c11 = cellIdx(ix,   iy, iz  )
        emitQuad(c00, c10, c01, c11, dA > 0)
      }
    }
  }

  // --- Z-axis edges: (ix, iy, iz) for ix=1..VOX-1, iy=1..VOX-1, iz=0..VOX-1
  for (let iz = 0; iz < CAVE_VOX; iz++) {
    for (let iy = 1; iy < CAVE_VOX; iy++) {
      for (let ix = 1; ix < CAVE_VOX; ix++) {
        const dA = density[latIdx(ix, iy, iz  )]
        const dB = density[latIdx(ix, iy, iz+1)]
        if ((dA < 0) === (dB < 0)) continue

        const c00 = cellIdx(ix-1, iy-1, iz)
        const c10 = cellIdx(ix,   iy-1, iz)
        const c01 = cellIdx(ix-1, iy,   iz)
        const c11 = cellIdx(ix,   iy,   iz)
        emitQuad(c00, c10, c01, c11, dA > 0)
      }
    }
  }

  // -------------------------------------------------------------------------
  // Trim to exact size.
  // -------------------------------------------------------------------------
  const positions = new Float32Array(vertCount * 3)
  const normals   = new Float32Array(vertCount * 3)
  const colors    = new Float32Array(vertCount * 3)
  const indices   = new Uint32Array(indexCount)

  positions.set(posScratch.subarray(0, vertCount * 3))
  normals  .set(nrmScratch.subarray(0, vertCount * 3))
  colors   .set(colScratch.subarray(0, vertCount * 3))
  indices  .set(idxScratch.subarray(0, indexCount))

  return {
    positions, normals, colors, indices,
    originX, originY, originZ,
    vertCount, indexCount,
    empty: false,
  }
}

// ---------------------------------------------------------------------------
// caveArraysToGeometry — main-thread only: builds BufferGeometry from arrays.
// ---------------------------------------------------------------------------

export function caveArraysToGeometry(a: CaveMeshArrays): BufferGeometry {
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(a.positions, 3))
  geometry.setAttribute('normal',   new BufferAttribute(a.normals,   3))
  geometry.setAttribute('color',    new BufferAttribute(a.colors,    3))
  geometry.setIndex(new BufferAttribute(a.indices, 1))
  geometry.computeBoundingSphere()
  return geometry
}
