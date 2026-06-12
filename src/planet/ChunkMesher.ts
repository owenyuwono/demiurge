import { BufferGeometry, BufferAttribute, Vector3 } from 'three';
import { FACE_BASES, FaceBasis, cubeToSphere } from './faceBases';

// ---------------------------------------------------------------------------
// Public contract
// ---------------------------------------------------------------------------

export interface ChunkParams {
  faceIndex:  number;   // 0..5 → +X, −X, +Y, −Y, +Z, −Z
  level:      number;   // quadtree depth; root = 0
  ix:         number;   // tile column within face at this level, in [0, 2^level)
  iy:         number;   // tile row    within face at this level, in [0, 2^level)
  resolution: number;   // quads per side; vertex grid is (res+1)²
  radius:     number;
  heightScale: number;
  /**
   * height in [-1,1] given a UNIT sphere direction and LOD level.
   * The level drives the detail-octave count so fine noise is only computed
   * for chunks where it contributes visible geometry (≥ ~1 m quads at level 12).
   * multiplied by heightScale for world displacement.
   */
  heightFn:   (dir: Vector3, level: number) => number;
  /** optional plate color: returns [r,g,b] in [0,1] for a given unit sphere direction */
  plateColorFn?: (dir: Vector3) => readonly [number, number, number];
}

export interface ChunkMeshData {
  geometry: BufferGeometry;
  /** World-space chunk center (double precision). Set mesh.position = origin. */
  origin:   Vector3;
}

// FACE_BASES, FaceBasis, and cubeToSphere are imported from ./faceBases (single source of truth).

// ---------------------------------------------------------------------------
// Map tile (ix, iy) at depth `level` + grid coords (gi, gj) ∈ [0, res] to a
// cube-face point in [-1,1]², then to a world direction (unit sphere).
//
// Face [0,1]² parameterisation: u = (ix + gi/res) / 2^level, same for v.
// Mapped to cube [-1,1]²: cu = u*2 − 1, cv = v*2 − 1.
// Cube point: normal + cu*tangentU + cv*tangentV (already in [-1,1]³ by
//   construction since each face is a unit-cube face and cu,cv ∈ [-1,1]).
// ---------------------------------------------------------------------------

function gridToCubePoint(
  basis: FaceBasis,
  level: number,
  ix: number,
  iy: number,
  gi: number, // column index in [0, res]
  gj: number, // row    index in [0, res]
  res: number,
  out: { cx: number; cy: number; cz: number },
): void {
  const scale = 1.0 / (1 << level);
  const u = (ix + gi / res) * scale; // [0, 1] over the face
  const v = (iy + gj / res) * scale;
  const cu = u * 2 - 1;             // [-1, 1]
  const cv = v * 2 - 1;
  out.cx = basis.nx + cu * basis.ux + cv * basis.vx;
  out.cy = basis.ny + cu * basis.uy + cv * basis.vy;
  out.cz = basis.nz + cu * basis.uz + cv * basis.vz;
}

// ---------------------------------------------------------------------------
// Evaluate a vertex: cube coords → sphere dir → world position, return height
// ---------------------------------------------------------------------------

const _sphereDir = new Vector3();
const _cubePoint = { cx: 0, cy: 0, cz: 0 };
const _tempDir   = new Vector3();

function evalVertex(
  basis: FaceBasis,
  level: number,
  ix: number,
  iy: number,
  gi: number,
  gj: number,
  res: number,
  radius: number,
  heightScale: number,
  heightFn: (dir: Vector3) => number,  // pre-bound to the chunk's LOD level
  outDir: Vector3,   // receives unit sphere direction
  outWorld: Vector3, // receives world position
): number {
  gridToCubePoint(basis, level, ix, iy, gi, gj, res, _cubePoint);
  cubeToSphere(_cubePoint.cx, _cubePoint.cy, _cubePoint.cz, outDir);
  outDir.normalize(); // ensure unit length
  const h = heightFn(outDir);
  const r = radius + h * heightScale;
  outWorld.copy(outDir).multiplyScalar(r);
  return h;
}

// ---------------------------------------------------------------------------
// Vertex color from elevation and slope
//
// Band table (e = normalized elevation, 0 = sea level):
//
//  OCEAN (e < 0):
//   e < −0.55          abyssal       #16202e  very dark navy
//  −0.55..−0.18        deep ocean    →#1d3a52  mid blue
//  −0.18..−0.045       cont. slope   →#2e5a74  steel blue
//  −0.045..0           shelf         →#4d8a96  cyan-teal (waterline ≤0.004 window)
//
//  LAND (e ≥ 0):
//   0..0.012           sand          #b8a36e
//   0.012..0.18        lowland       →#5a7a4a  desaturated green
//   0.18..0.42         highland      →#8a7a55  brown-tan
//   0.42..0.62         bare rock     →#7a7060  grey-brown
//   e > 0.55 (blend)   snow          →#e6e8eb  near-white (full by 0.62)
//
//  SLOPE override (land only, slope > 0.22): blend toward rock #6e6a64
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function elevationColor(e: number, slope: number, out: Float32Array, base: number): void {
  let r: number, g: number, b: number;

  if (e < 0) {
    // ---- OCEAN -------------------------------------------------------
    // Abyssal #16202e
    const rAby = 0x16 / 255, gAby = 0x20 / 255, bAby = 0x2e / 255;
    // Deep   #1d3a52
    const rDep = 0x1d / 255, gDep = 0x3a / 255, bDep = 0x52 / 255;
    // Slope  #2e5a74
    const rSlp = 0x2e / 255, gSlp = 0x5a / 255, bSlp = 0x74 / 255;
    // Shelf  #4d8a96
    const rShf = 0x4d / 255, gShf = 0x8a / 255, bShf = 0x96 / 255;

    if (e < -0.55) {
      // pure abyssal
      r = rAby; g = gAby; b = bAby;
    } else if (e < -0.18) {
      // abyssal → deep  (window ~0.015 around −0.55)
      const t = clamp01((e + 0.55) / 0.015);
      r = lerp(rAby, rDep, t);
      g = lerp(gAby, gDep, t);
      b = lerp(bAby, bDep, t);
    } else if (e < -0.045) {
      // deep → cont. slope  (window 0.015 around −0.18)
      const t = clamp01((e + 0.18) / 0.015);
      r = lerp(rDep, rSlp, t);
      g = lerp(gDep, gSlp, t);
      b = lerp(bDep, bSlp, t);
    } else {
      // cont. slope → shelf  (window 0.015 around −0.045)
      // but shelf→waterline kept crisp: transition window ≤ 0.004 handled at e≥0
      const t = clamp01((e + 0.045) / 0.015);
      r = lerp(rSlp, rShf, t);
      g = lerp(gSlp, gShf, t);
      b = lerp(bSlp, bShf, t);
    }

  } else {
    // ---- LAND --------------------------------------------------------
    // Sand      #b8a36e
    const rSnd = 0xb8 / 255, gSnd = 0xa3 / 255, bSnd = 0x6e / 255;
    // Lowland   #5a7a4a
    const rLow = 0x5a / 255, gLow = 0x7a / 255, bLow = 0x4a / 255;
    // Highland  #8a7a55
    const rHig = 0x8a / 255, gHig = 0x7a / 255, bHig = 0x55 / 255;
    // Bare rock #7a7060
    const rRck = 0x7a / 255, gRck = 0x70 / 255, bRck = 0x60 / 255;
    // Snow      #e6e8eb
    const rSnw = 0xe6 / 255, gSnw = 0xe8 / 255, bSnw = 0xeb / 255;

    if (e < 0.012) {
      // shelf→sand: crisp waterline transition window = 0.004
      const t = clamp01(e / 0.004);
      // shelf color at e=0
      const rShf = 0x4d / 255, gShf = 0x8a / 255, bShf = 0x96 / 255;
      r = lerp(rShf, rSnd, t);
      g = lerp(gShf, gSnd, t);
      b = lerp(bShf, bSnd, t);
    } else if (e < 0.18) {
      // sand → lowland (window 0.015 around 0.012)
      const t = clamp01((e - 0.012) / 0.015);
      r = lerp(rSnd, rLow, t);
      g = lerp(gSnd, gLow, t);
      b = lerp(bSnd, bLow, t);
    } else if (e < 0.42) {
      // lowland → highland (window 0.015 around 0.18)
      const t = clamp01((e - 0.18) / 0.015);
      r = lerp(rLow, rHig, t);
      g = lerp(gLow, gHig, t);
      b = lerp(bLow, bHig, t);
    } else if (e < 0.62) {
      // highland → bare rock (window 0.015 around 0.42)
      const t = clamp01((e - 0.42) / 0.015);
      r = lerp(rHig, rRck, t);
      g = lerp(gHig, gRck, t);
      b = lerp(bHig, bRck, t);
    } else {
      r = rRck; g = gRck; b = bRck;
    }

    // Snow blend: start at 0.55, fully white by 0.62
    if (e > 0.55) {
      const snowT = clamp01((e - 0.55) / (0.62 - 0.55));
      r = lerp(r, rSnw, snowT);
      g = lerp(g, gSnw, snowT);
      b = lerp(b, bSnw, snowT);
    }

    // Slope override (land only): blend toward rock grey #6e6a64 on cliffs
    if (slope > 0.22) {
      const rockBlend = clamp01((slope - 0.22) / 0.20);
      r = lerp(r, 0x6e / 255, rockBlend);
      g = lerp(g, 0x6a / 255, rockBlend);
      b = lerp(b, 0x64 / 255, rockBlend);
    }
  }

  out[base    ] = r;
  out[base + 1] = g;
  out[base + 2] = b;
}

// ---------------------------------------------------------------------------
// Main builder
// ---------------------------------------------------------------------------

export function buildChunkGeometry(p: ChunkParams): ChunkMeshData {
  const { faceIndex, level, ix, iy, resolution: res, radius, heightScale, heightFn, plateColorFn } = p;
  // Convenience: heightFn with level pre-bound — avoids repeating `level` at every call site
  // inside this function (all vertices in a chunk share the same LOD level).
  const hFn = (dir: Vector3): number => heightFn(dir, level);
  const basis = FACE_BASES[faceIndex];
  const hasPlateColor = plateColorFn !== undefined;

  // Vertex counts
  const gridSize    = res + 1;              // vertices per side of interior grid
  const gridVerts   = gridSize * gridSize;  // interior grid vertex count
  // Skirt: one ring of duplicates at the 4 border edges.
  // 4 sides, each has (res+1) verts — but the 4 corners are each shared by
  // two sides, so total unique skirt verts = 4*(res+1) − 4 = 4*res.
  // We lay them out as 4 strips of (res+1) with shared corners deduplicated:
  //   bottom: indices 0..(res)  → (res+1) verts
  //   top:    indices (res)..(2*res)        — but corner res shared with bottom
  // For simplicity, lay them as 4 independent strips of res verts each
  // (dropping one repeated corner per strip to avoid duplication):
  //   bottom row (j=0):    i = 0..res-1  → res verts  (left corner kept, right skipped)
  //   right col (i=res):   j = 0..res-1  → res verts
  //   top row   (j=res):   i = res..1    → res verts  (right kept, left skipped)
  //   left col  (i=0):     j = res..1    → res verts
  // Each skirt quad pairs one border edge with its extruded skirt edge.
  // We instead use the simpler layout: 4*(res+1) skirt verts, indices are per-strip.
  // Each of 4 edges has (res+1) border verts → (res+1) skirt verts, res quads.
  const skirtVertsPerEdge = res + 1;
  const skirtVerts = 4 * skirtVertsPerEdge;
  const totalVerts = gridVerts + skirtVerts;

  // Index counts
  const gridIndexCount  = res * res * 6;   // 2 tris per quad
  const skirtIndexCount = 4 * res * 6;
  const totalIndices    = gridIndexCount + skirtIndexCount;

  // Allocate typed arrays
  const positions   = new Float32Array(totalVerts * 3);
  const normals     = new Float32Array(totalVerts * 3);
  const colors      = new Float32Array(totalVerts * 3);
  const plateColors = hasPlateColor ? new Float32Array(totalVerts * 3) : null;
  const indices     = new Uint32Array(totalIndices);

  // Scratch objects — reused, no per-vertex allocation
  const dir    = new Vector3();
  const world  = new Vector3();
  const dirL   = new Vector3();
  const worldL = new Vector3();
  const dirR   = new Vector3();
  const worldR = new Vector3();
  const dirD   = new Vector3();
  const worldD = new Vector3();
  const dirU   = new Vector3();
  const worldU = new Vector3();
  const tan1   = new Vector3();
  const tan2   = new Vector3();
  const nrm    = new Vector3();

  // -- Chunk center (for origin-relative positions) -------------------------
  // Center of tile = (ix+0.5)/2^level, (iy+0.5)/2^level on the face
  // Evaluated at grid center (gi=res/2, gj=res/2 in continuous terms = 0.5·res)
  // Use floating arithmetic directly:
  const scale = 1.0 / (1 << level);
  const uCenter = (ix + 0.5) * scale;
  const vCenter = (iy + 0.5) * scale;
  const cuC = uCenter * 2 - 1;
  const cvC = vCenter * 2 - 1;
  const cxC = basis.nx + cuC * basis.ux + cvC * basis.vx;
  const cyC = basis.ny + cuC * basis.uy + cvC * basis.vy;
  const czC = basis.nz + cuC * basis.uz + cvC * basis.vz;
  cubeToSphere(cxC, cyC, czC, _sphereDir);
  _sphereDir.normalize();
  const hCenter = hFn(_sphereDir);
  const rCenter = radius + hCenter * heightScale;
  const origin = new Vector3(
    _sphereDir.x * rCenter,
    _sphereDir.y * rCenter,
    _sphereDir.z * rCenter,
  );

  // -- Skirt depth ----------------------------------------------------------
  // chunkWorldSize ≈ arc length of the patch = (π/2 · radius) / 2^level
  const chunkWorldSize = (Math.PI / 2 * radius) / (1 << level);
  const skirtDepth = Math.max(heightScale * 1.5, chunkWorldSize * 0.05);

  // -- Central-difference step size: half a grid step in [0,1] face space ---
  // grid step in [0,1] face coords = 1 / (res * 2^level)
  const cdStep = 0.5 / (res * (1 << level));

  // Helper: evaluate world position for a face-space offset (du, dv) from
  // grid point (gi, gj), without touching the shared scratch objects.
  function evalOffset(
    gi: number,
    gj: number,
    du: number,
    dv: number,
    outDir: Vector3,
    outWorld: Vector3,
  ): void {
    // Continuous face [0,1] coords of this grid point
    const u0 = (ix + gi / res) * scale;
    const v0 = (iy + gj / res) * scale;
    const cu = (u0 + du) * 2 - 1;
    const cv = (v0 + dv) * 2 - 1;
    const cx = basis.nx + cu * basis.ux + cv * basis.vx;
    const cy = basis.ny + cu * basis.uy + cv * basis.vy;
    const cz = basis.nz + cu * basis.uz + cv * basis.vz;
    cubeToSphere(cx, cy, cz, outDir);
    outDir.normalize();
    const h = hFn(outDir);
    const r = radius + h * heightScale;
    outWorld.copy(outDir).multiplyScalar(r);
  }

  // -- Interior grid --------------------------------------------------------
  let vi = 0; // vertex write index

  for (let gj = 0; gj < gridSize; gj++) {
    for (let gi = 0; gi < gridSize; gi++) {
      evalVertex(basis, level, ix, iy, gi, gj, res, radius, heightScale, hFn, dir, world);

      // Position relative to chunk origin
      const px = world.x - origin.x;
      const py = world.y - origin.y;
      const pz = world.z - origin.z;
      positions[vi * 3    ] = px;
      positions[vi * 3 + 1] = py;
      positions[vi * 3 + 2] = pz;

      // Normal via central differences in face (u,v) space
      evalOffset(gi, gj, -cdStep,      0, dirL, worldL);
      evalOffset(gi, gj,  cdStep,      0, dirR, worldR);
      evalOffset(gi, gj,       0, -cdStep, dirD, worldD);
      evalOffset(gi, gj,       0,  cdStep, dirU, worldU);

      // Tangent vectors of the displaced surface
      tan1.subVectors(worldR, worldL); // ∂pos/∂u  (unnormalized)
      tan2.subVectors(worldU, worldD); // ∂pos/∂v

      nrm.crossVectors(tan1, tan2).normalize();

      // Ensure outward-facing normal (should agree with sphere dir)
      if (nrm.dot(dir) < 0) nrm.negate();

      normals[vi * 3    ] = nrm.x;
      normals[vi * 3 + 1] = nrm.y;
      normals[vi * 3 + 2] = nrm.z;

      // Vertex color — re-use the direction already set by evalVertex;
      // hFn is pre-bound to this chunk's level so the same octave count applies.
      const h = hFn(dir); // dir is set by evalVertex
      const slope = 1 - nrm.dot(dir);
      elevationColor(h, slope, colors, vi * 3);

      // Plate color (if provided)
      if (hasPlateColor && plateColors !== null) {
        const pc = plateColorFn!(dir);
        plateColors[vi * 3    ] = pc[0];
        plateColors[vi * 3 + 1] = pc[1];
        plateColors[vi * 3 + 2] = pc[2];
      }

      vi++;
    }
  }

  // -- Interior grid indices ------------------------------------------------
  let ii = 0; // index write pointer
  for (let gj = 0; gj < res; gj++) {
    for (let gi = 0; gi < res; gi++) {
      const a = gj * gridSize + gi;
      const b = a + 1;
      const c = a + gridSize;
      const d = c + 1;
      // Two CCW triangles (viewed from outside sphere)
      indices[ii++] = a;
      indices[ii++] = c;
      indices[ii++] = b;
      indices[ii++] = b;
      indices[ii++] = c;
      indices[ii++] = d;
    }
  }

  // -- Skirt vertices + indices ---------------------------------------------
  // For each of the 4 edges we emit (res+1) skirt verts = border vertex
  // pulled toward planet center by skirtDepth.
  // Skirt strip layout (skirt vertex index relative to skirtBase):
  //   edge 0 (bottom, gj=0):    si = 0..(res)
  //   edge 1 (right, gi=res):   si = (res+1)..(2res+1)
  //   edge 2 (top, gj=res):     si = (2res+2)..(3res+2)
  //   edge 3 (left, gi=0):      si = (3res+3)..(4res+3)

  const skirtBase = gridVerts; // first skirt vertex index

  // Helper: emit one skirt vertex (pullback toward center)
  // Also copies plateColor from border vertex when present.
  function emitSkirtVert(borderVI: number): void {
    // Read border vertex world position (origin-relative → add origin back)
    const bx = positions[borderVI * 3    ] + origin.x;
    const by = positions[borderVI * 3 + 1] + origin.y;
    const bz = positions[borderVI * 3 + 2] + origin.z;
    const len = Math.sqrt(bx * bx + by * by + bz * bz);
    const pullScale = (len - skirtDepth) / len;

    positions[vi * 3    ] = bx * pullScale - origin.x;
    positions[vi * 3 + 1] = by * pullScale - origin.y;
    positions[vi * 3 + 2] = bz * pullScale - origin.z;

    // Copy normal and color from border vertex
    normals[vi * 3    ] = normals[borderVI * 3    ];
    normals[vi * 3 + 1] = normals[borderVI * 3 + 1];
    normals[vi * 3 + 2] = normals[borderVI * 3 + 2];
    colors[vi * 3    ]  = colors[borderVI * 3    ];
    colors[vi * 3 + 1]  = colors[borderVI * 3 + 1];
    colors[vi * 3 + 2]  = colors[borderVI * 3 + 2];

    // Copy plateColor from border vertex (same vertex order as color)
    if (hasPlateColor && plateColors !== null) {
      plateColors[vi * 3    ] = plateColors[borderVI * 3    ];
      plateColors[vi * 3 + 1] = plateColors[borderVI * 3 + 1];
      plateColors[vi * 3 + 2] = plateColors[borderVI * 3 + 2];
    }

    vi++;
  }

  // Helper: emit quad indices for one skirt quad (border-edge quad facing out).
  // border0, border1 = two adjacent border verts traversed in edge order;
  // skirt0, skirt1 = their pullbacks toward planet center.
  // Winding: (border0,border1,skirt0) and (border1,skirt1,skirt0).
  // (b1−b0)×(sk0−b0) points away from the patch (outward skirt wall normal).
  function emitSkirtQuad(border0: number, border1: number, skirt0: number, skirt1: number): void {
    indices[ii++] = border0;
    indices[ii++] = border1;
    indices[ii++] = skirt0;
    indices[ii++] = border1;
    indices[ii++] = skirt1;
    indices[ii++] = skirt0;
  }

  // Edge 0: bottom row, gj=0, gi = 0..res (left to right)
  const e0Start = skirtBase;
  for (let gi = 0; gi <= res; gi++) {
    emitSkirtVert(/* borderVI = */ 0 * gridSize + gi);
  }
  for (let gi = 0; gi < res; gi++) {
    const border0 = 0 * gridSize + gi;
    const border1 = 0 * gridSize + gi + 1;
    const skirt0  = e0Start + gi;
    const skirt1  = e0Start + gi + 1;
    emitSkirtQuad(border0, border1, skirt0, skirt1);
  }

  // Edge 1: right column, gi=res, gj = 0..res (bottom to top)
  const e1Start = e0Start + (res + 1);
  for (let gj = 0; gj <= res; gj++) {
    emitSkirtVert(/* borderVI = */ gj * gridSize + res);
  }
  for (let gj = 0; gj < res; gj++) {
    const border0 = gj * gridSize + res;
    const border1 = (gj + 1) * gridSize + res;
    const skirt0  = e1Start + gj;
    const skirt1  = e1Start + gj + 1;
    emitSkirtQuad(border0, border1, skirt0, skirt1);
  }

  // Edge 2: top row, gj=res, gi = res..0 (right to left — reverse winding consistency)
  const e2Start = e1Start + (res + 1);
  for (let gi = res; gi >= 0; gi--) {
    emitSkirtVert(/* borderVI = */ res * gridSize + gi);
  }
  for (let qi = 0; qi < res; qi++) {
    // qi=0: gi=res→res-1, qi=1: gi=res-1→res-2 ...
    const gi0 = res - qi;
    const gi1 = res - qi - 1;
    const border0 = res * gridSize + gi0;
    const border1 = res * gridSize + gi1;
    const skirt0  = e2Start + qi;
    const skirt1  = e2Start + qi + 1;
    emitSkirtQuad(border0, border1, skirt0, skirt1);
  }

  // Edge 3: left column, gi=0, gj = res..0 (top to bottom — reverse)
  const e3Start = e2Start + (res + 1);
  for (let gj = res; gj >= 0; gj--) {
    emitSkirtVert(/* borderVI = */ gj * gridSize + 0);
  }
  for (let qi = 0; qi < res; qi++) {
    const gj0 = res - qi;
    const gj1 = res - qi - 1;
    const border0 = gj0 * gridSize + 0;
    const border1 = gj1 * gridSize + 0;
    const skirt0  = e3Start + qi;
    const skirt1  = e3Start + qi + 1;
    emitSkirtQuad(border0, border1, skirt0, skirt1);
  }

  // -- Assemble BufferGeometry ---------------------------------------------
  const geometry = new BufferGeometry();
  geometry.setAttribute('position',  new BufferAttribute(positions, 3));
  geometry.setAttribute('normal',    new BufferAttribute(normals,   3));
  geometry.setAttribute('color',     new BufferAttribute(colors,    3));
  if (hasPlateColor && plateColors !== null) {
    geometry.setAttribute('plateColor', new BufferAttribute(plateColors, 3));
  }
  geometry.setIndex(new BufferAttribute(indices, 1));
  geometry.computeBoundingSphere();

  return { geometry, origin };
}
