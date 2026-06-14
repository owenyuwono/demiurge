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
  /**
   * optional climate sampler: temperature (°C-ish) + moisture (0..1) for a unit
   * sphere direction and normalized terrain height. When provided, land is colored
   * by biome; when absent, the mesher falls back to the elevation palette so it
   * stays usable standalone.
   */
  climateFn?: (dir: Vector3, height: number) => { temperature: number; moisture: number };
}

export interface ChunkMeshData {
  geometry: BufferGeometry;
  /** World-space chunk center (double precision). Set mesh.position = origin. */
  origin:   Vector3;
}

export interface ChunkMeshArrays {
  positions:   Float32Array;
  normals:     Float32Array;
  colors:      Float32Array;
  plateColors: Float32Array | null;
  indices:     Uint32Array;
  originX: number;
  originY: number;
  originZ: number;
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
// Vertex color from elevation and slope.
//
// Colors the GROUND at every elevation — geology only, NO water color. The actual
// sea is a separate translucent shell (see main.ts), so the seabed is sediment/sand
// (dark in the deeps, sandy on the shelf), not blue. The water mesh tints whatever is
// below it — deep dark sediment → dark blue, shallow sand → turquoise — naturally, and
// when the water level is lowered the exposed seabed reads as honest sediment.
//
// Band table (e = normalized elevation, 0 = sea level):
//
//  SEABED (e < 0):
//   e < −0.55          abyssal sediment  #232220  near-black warm grey
//  −0.55..−0.18        deep sediment     →#3a352c  dark brown-grey
//  −0.18..−0.045       slope sediment    →#605442  medium brown
//  −0.045..0           sandy shelf       →#9c8a66  light tan
//
//  LAND (e ≥ 0):
//   0..0.012           sand          →#b8a36e  beach (blends up from shelf tan)
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
    // ---- SEABED (geology, not water — the blue comes from the water shell) -----
    // Abyssal sediment #232220
    const rAby = 0x23 / 255, gAby = 0x22 / 255, bAby = 0x20 / 255;
    // Deep sediment   #3a352c
    const rDep = 0x3a / 255, gDep = 0x35 / 255, bDep = 0x2c / 255;
    // Slope sediment  #605442
    const rSlp = 0x60 / 255, gSlp = 0x54 / 255, bSlp = 0x42 / 255;
    // Sandy shelf  #9c8a66
    const rShf = 0x9c / 255, gShf = 0x8a / 255, bShf = 0x66 / 255;

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
      // sandy shelf → beach sand, smooth (no crisp waterline — the water shell draws that now)
      const t = clamp01(e / 0.012);
      // sandy shelf color at e=0 (matches the seabed branch so the join is seamless)
      const rShf = 0x9c / 255, gShf = 0x8a / 255, bShf = 0x66 / 255;
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
// Biome coloring (climate-driven). Used when ChunkParams.climateFn is provided.
//   - Seabed (e < 0): identical geology palette to elevationColor (verbatim) —
//     the water shell draws the ocean, climate does not recolor the seabed.
//   - Land (e ≥ 0): biome chosen from temperature (°C) + moisture (0..1), blended
//     smoothly (no hard biome edges). Ice/snow where it's cold (caps poles AND
//     peaks via a temperature-driven snow line), desert where dry, forests/
//     grassland/tundra otherwise. High elevation trends rocky; cliffs → rock grey.
// ---------------------------------------------------------------------------

const SNOW_TEMP = -2;     // °C — below this, land trends to snow/ice (blended over a few °C)
const SNOW_BLEND = 5;     // °C window over which snow fades in below SNOW_TEMP

function smoothstepM(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

export function biomeColor(
  e: number,
  slope: number,
  temperature: number,
  moisture: number,
  out: Float32Array,
  base: number,
): void {
  let r: number, g: number, b: number;

  if (e < 0) {
    // ---- SEABED (verbatim copy of elevationColor's seabed branch) -----------
    const rAby = 0x23 / 255, gAby = 0x22 / 255, bAby = 0x20 / 255;
    const rDep = 0x3a / 255, gDep = 0x35 / 255, bDep = 0x2c / 255;
    const rSlp = 0x60 / 255, gSlp = 0x54 / 255, bSlp = 0x42 / 255;
    const rShf = 0x9c / 255, gShf = 0x8a / 255, bShf = 0x66 / 255;

    if (e < -0.55) {
      r = rAby; g = gAby; b = bAby;
    } else if (e < -0.18) {
      const t = clamp01((e + 0.55) / 0.015);
      r = lerp(rAby, rDep, t);
      g = lerp(gAby, gDep, t);
      b = lerp(bAby, bDep, t);
    } else if (e < -0.045) {
      const t = clamp01((e + 0.18) / 0.015);
      r = lerp(rDep, rSlp, t);
      g = lerp(gDep, gSlp, t);
      b = lerp(bDep, bSlp, t);
    } else {
      const t = clamp01((e + 0.045) / 0.015);
      r = lerp(rSlp, rShf, t);
      g = lerp(gSlp, gShf, t);
      b = lerp(bSlp, bShf, t);
    }

  } else {
    // ---- LAND: biome by (temperature, moisture) -----------------------------
    // Palette (all generic — Earth's look emerges from Earth-like params):
    const sandHot = [0xc2 / 255, 0xa8 / 255, 0x78 / 255];   // hot dry desert
    const sandCold = [0x9a / 255, 0x8c / 255, 0x70 / 255];  // cold dry / rocky steppe
    const grass = [0x8f / 255, 0x95 / 255, 0x55 / 255];     // grassland / steppe
    const tropical = [0x2f / 255, 0x6b / 255, 0x34 / 255];  // hot wet jungle
    const temperate = [0x4f / 255, 0x7a / 255, 0x45 / 255]; // temperate forest
    const tundra = [0x6b / 255, 0x6f / 255, 0x57 / 255];    // cold sparse (brown-grey)
    const rock = [0x7a / 255, 0x70 / 255, 0x60 / 255];      // bare rock (matches elevationColor)
    const snow = [0xe6 / 255, 0xe8 / 255, 0xeb / 255];      // ice / snow

    // --- 1. Beach sand at the waterline, seamless with the seabed shelf. -----
    // (kept so coastlines read as sand regardless of biome, like elevationColor)
    const rShf = 0x9c / 255, gShf = 0x8a / 255, bShf = 0x66 / 255;

    // --- 2. Dry vs vegetated axis -------------------------------------------
    // dryness: 1 when very dry (M→0), 0 once moisture clears the desert line.
    // Calibrated to the moisture field: desert only below ~0.20 (M≥~0.22 vegetates),
    // so deserts stay confined to the descending bands + driest deep interiors
    // instead of swallowing every mid-latitude / continental interior.
    const dry = 1 - smoothstepM(0.14, 0.22, moisture);
    // Within dry: hot deserts vs cold steppe (by temperature).
    const hotDry = smoothstepM(2, 14, temperature);
    const desertR = lerp(sandCold[0], sandHot[0], hotDry);
    const desertG = lerp(sandCold[1], sandHot[1], hotDry);
    const desertB = lerp(sandCold[2], sandHot[2], hotDry);

    // --- 3. Vegetated biome by temperature ----------------------------------
    // Build along the temperature axis: tundra (cold) → temperate forest (mild)
    // → tropical (hot). Tropical additionally requires moisture, else it stays
    // temperate. Smoothstep windows keep the transitions soft (no hard edges).
    const tWarm = smoothstepM(2, 8, temperature);        // 0 cold (tundra), 1 by 8°C (forest)
    const tHot = smoothstepM(18, 24, temperature);       // 0 mild, 1 by 24°C
    const tropWet = smoothstepM(0.42, 0.58, moisture);   // tropical needs moisture too (rainforest above ~0.45)
    const tropMix = tHot * tropWet;
    // mild-end color: temperate forest, pushed toward tropical when hot AND wet.
    const mildR = lerp(temperate[0], tropical[0], tropMix);
    const mildG = lerp(temperate[1], tropical[1], tropMix);
    const mildB = lerp(temperate[2], tropical[2], tropMix);
    // blend from tundra (cold) up to the mild/hot color as it warms.
    let vegR = lerp(tundra[0], mildR, tWarm);
    let vegG = lerp(tundra[1], mildG, tWarm);
    let vegB = lerp(tundra[2], mildB, tWarm);

    // --- 4. Grassland/steppe for middling moisture (between desert and forest) -
    // Peaks around M ≈ 0.22..0.40 (the steppe belt just above the desert line),
    // then fades out as forest takes over by ~0.46; suppressed in true tropical
    // (hot+wet) regions. Calibrated to the moisture field so a grassland belt sits
    // between the ±30° deserts and the wetter forested zones.
    const grassW = smoothstepM(0.20, 0.30, moisture) * (1 - smoothstepM(0.38, 0.48, moisture)) * (1 - tropMix);
    vegR = lerp(vegR, grass[0], grassW);
    vegG = lerp(vegG, grass[1], grassW);
    vegB = lerp(vegB, grass[2], grassW);

    // --- 5. Combine dry (desert) with vegetated along the dryness axis -------
    r = lerp(vegR, desertR, dry);
    g = lerp(vegG, desertG, dry);
    b = lerp(vegB, desertB, dry);

    // --- 6. Beach sand at the immediate waterline ---------------------------
    if (e < 0.012) {
      const t = clamp01(e / 0.012);
      // From shelf sand up to the chosen biome color (so beaches read sandy).
      r = lerp(rShf, r, t);
      g = lerp(gShf, g, t);
      b = lerp(bShf, b, t);
    }

    // --- 7. High elevation trends rocky -------------------------------------
    // Above e ≈ 0.5 blend toward bare rock (the T-driven snow below caps peaks).
    if (e > 0.5) {
      const rockT = smoothstepM(0.5, 0.72, e);
      r = lerp(r, rock[0], rockT);
      g = lerp(g, rock[1], rockT);
      b = lerp(b, rock[2], rockT);
    }

    // --- 8. Snow / ice by TEMPERATURE (latitude- AND altitude-dependent) -----
    // T already falls with both latitude and altitude, so this gives polar ice
    // caps at sea level AND a natural snow line on cold peaks. Blended over a few °C.
    if (temperature < SNOW_TEMP + SNOW_BLEND) {
      const snowT = 1 - smoothstepM(SNOW_TEMP, SNOW_TEMP + SNOW_BLEND, temperature);
      r = lerp(r, snow[0], snowT);
      g = lerp(g, snow[1], snowT);
      b = lerp(b, snow[2], snowT);
    }

    // --- 9. Slope override (land only): cliffs → rock grey #6e6a64 ----------
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
// Pure compute core — all meshing math, no THREE GPU objects (worker-safe).
// ---------------------------------------------------------------------------

export function computeChunkArrays(p: ChunkParams): ChunkMeshArrays {
  const { faceIndex, level, ix, iy, resolution: res, radius, heightScale, heightFn, plateColorFn, climateFn } = p;
  // Convenience: heightFn with level pre-bound — avoids repeating `level` at every call site
  // inside this function (all vertices in a chunk share the same LOD level).
  const hFn = (dir: Vector3): number => heightFn(dir, level);
  const basis = FACE_BASES[faceIndex];
  const hasPlateColor = plateColorFn !== undefined;
  const hasClimate = climateFn !== undefined;

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
  const positions    = new Float32Array(totalVerts * 3);
  const normals      = new Float32Array(totalVerts * 3);
  const colors       = new Float32Array(totalVerts * 3);
  const plateColors  = hasPlateColor ? new Float32Array(totalVerts * 3) : null;
  const indices      = new Uint32Array(totalIndices);

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
  // Scratch unit-dir for ghost (off-edge) border neighbours — evalVertex needs
  // a dir out-param but the ghost's normal only uses its world position.
  const _ghostDir = new Vector3();

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
  const originX = _sphereDir.x * rCenter;
  const originY = _sphereDir.y * rCenter;
  const originZ = _sphereDir.z * rCenter;

  // -- Skirt depth ----------------------------------------------------------
  // Proportional to the chunk's arc length so skirts scale naturally as the
  // quadtree deepens (depth 12→16 shrinks chunks ~16×, skirts shrink with them).
  // A coarser neighbour's quads are ~2× larger, so the seam displacement is at
  // most one quad-height step; SKIRT_FACTOR × chunkArcLen ≈ 1.6 quad-widths at
  // res=32, which is enough to close any T-junction gap without wasted fill.
  const SKIRT_FACTOR = 0.05; // tune here: 0.05 = skirt ≈ 5 % of chunk arc length
  const chunkArcLen = (Math.PI / 2 * radius) / (1 << level);
  const skirtDepth = SKIRT_FACTOR * chunkArcLen;

  // -- Ghost-edge position helper ---------------------------------------------
  // Computes the ORIGIN-RELATIVE world position of a vertex at grid index
  // (gi, gj) that may lie one step OUTSIDE the chunk grid (gi=-1, gi=res+1,
  // gj=-1, or gj=res+1). gridToCubePoint's parameterisation u=(ix+gi/res)*scale
  // is linear in gi, so an out-of-range index simply extrapolates the face
  // parameter — landing exactly where the adjacent same-level tile's interior
  // vertex sits. This lets the border ring use the SAME geometric normal method
  // as the interior (cross of neighbor positions), with the off-edge "ghost"
  // neighbor synthesised from the SAME continuous heightFn the neighbor chunk
  // samples → border-ring discontinuity and same-level seams both vanish.
  // 1 heightFn eval per call; only border vertices ever need it.
  function ghostPos(gi: number, gj: number, outWorld: Vector3): void {
    // _ghostDir is scratch; evalVertex writes the unit dir + world position.
    evalVertex(basis, level, ix, iy, gi, gj, res, radius, heightScale, hFn, _ghostDir, outWorld);
    outWorld.x -= originX;
    outWorld.y -= originY;
    outWorld.z -= originZ;
  }

  // -- Interior grid (two-pass) -----------------------------------------------
  //
  // Pass 1: compute every vertex's position (and cache height + sphere dir for Pass 2).
  //         1 heightFn eval per vertex — same as before.
  // Pass 2: compute normals.
  //   - Border ring (gi==0||gi==res||gj==0||gj==res): sphere-tangent central-diff
  //     (UNCHANGED — this is the seam-free path).
  //   - Interior (gi in [1,res-1], gj in [1,res-1]): normal from 4 grid-neighbor
  //     positions (0 extra heightFn evals).
  //         n = normalize( cross( P[gi+1,gj]-P[gi-1,gj],  P[gi,gj+1]-P[gi,gj-1] ) )
  //     then flipped outward (dot with radial direction > 0).
  //
  // This reduces heightFn evals from ~5/vertex to ~1/vertex for interior verts.
  // Border verts still pay the 4-eval cost, but they are only 4*(res+1)-4 of
  // the (res+1)^2 total — at res=32 that is 128 of 1089.

  // Scratch caches for the grid (allocated once here, not per-vertex).
  const hCache   = new Float32Array(gridVerts);       // height per grid vertex
  const dirCache = new Float32Array(gridVerts * 3);   // unit sphere dir per grid vertex

  // --- Pass 1: positions -------------------------------------------------------
  let vi = 0; // vertex write index
  for (let gj = 0; gj < gridSize; gj++) {
    for (let gi = 0; gi < gridSize; gi++) {
      const h = evalVertex(basis, level, ix, iy, gi, gj, res, radius, heightScale, hFn, dir, world);

      // Position relative to chunk origin
      positions[vi * 3    ] = world.x - originX;
      positions[vi * 3 + 1] = world.y - originY;
      positions[vi * 3 + 2] = world.z - originZ;

      // Cache height and sphere direction for Pass 2
      hCache[vi]          = h;
      dirCache[vi * 3    ] = dir.x;
      dirCache[vi * 3 + 1] = dir.y;
      dirCache[vi * 3 + 2] = dir.z;

      vi++;
    }
  }

  // --- Pass 2: normals + colors ------------------------------------------------
  vi = 0;
  for (let gj = 0; gj < gridSize; gj++) {
    for (let gi = 0; gi < gridSize; gi++) {
      // Restore cached sphere direction and height
      dir.x = dirCache[vi * 3    ];
      dir.y = dirCache[vi * 3 + 1];
      dir.z = dirCache[vi * 3 + 2];
      const h = hCache[vi];

      const onBorder = gi === 0 || gi === res || gj === 0 || gj === res;

      if (onBorder) {
        // --- Border: sphere-tangent central differences (UNCHANGED, seam-free) ---
        if (Math.abs(dir.y) < 0.9) {
          _tanUp.set(0, 1, 0);
        } else {
          _tanUp.set(1, 0, 0);
        }
        _tan1.crossVectors(dir, _tanUp).normalize(); // tangent 1 ⊥ dir
        _tan2.crossVectors(dir, _tan1);              // tangent 2 ⊥ dir ⊥ _tan1 (already unit)

        // Sample displaced surface at ±arcStep along each tangent direction.
        evalSphereOffset(dir, _tan1, -arcStep, hFn, dirL, worldL);
        evalSphereOffset(dir, _tan1,  arcStep, hFn, dirR, worldR);
        evalSphereOffset(dir, _tan2, -arcStep, hFn, dirD, worldD);
        evalSphereOffset(dir, _tan2,  arcStep, hFn, dirU, worldU);

        // Tangent vectors of the displaced surface
        tan1.subVectors(worldR, worldL); // ∂pos/∂_tan1 (unnormalized)
        tan2.subVectors(worldU, worldD); // ∂pos/∂_tan2

        nrm.crossVectors(tan1, tan2).normalize();

        // Ensure outward-facing normal (should agree with sphere dir)
        if (nrm.dot(dir) < 0) nrm.negate();
      } else {
        // --- Interior: cross product from 4 grid-neighbor positions ---------------
        // Neighbor vertex indices in the flat grid array
        const idxL = vi - 1;               // (gi-1, gj)
        const idxR = vi + 1;               // (gi+1, gj)
        const idxD = vi - gridSize;        // (gi,   gj-1)
        const idxU = vi + gridSize;        // (gi,   gj+1)

        // Read neighbor positions (origin-relative — offsets cancel in the cross product)
        worldL.set(positions[idxL * 3], positions[idxL * 3 + 1], positions[idxL * 3 + 2]);
        worldR.set(positions[idxR * 3], positions[idxR * 3 + 1], positions[idxR * 3 + 2]);
        worldD.set(positions[idxD * 3], positions[idxD * 3 + 1], positions[idxD * 3 + 2]);
        worldU.set(positions[idxU * 3], positions[idxU * 3 + 1], positions[idxU * 3 + 2]);

        // Central-difference tangent vectors
        tan1.subVectors(worldR, worldL); // ∂pos/∂gi direction
        tan2.subVectors(worldU, worldD); // ∂pos/∂gj direction

        nrm.crossVectors(tan1, tan2).normalize();

        // Ensure outward-facing (dot with radial direction of this vertex > 0)
        if (nrm.dot(dir) < 0) nrm.negate();
      }

      normals[vi * 3    ] = nrm.x;
      normals[vi * 3 + 1] = nrm.y;
      normals[vi * 3 + 2] = nrm.z;

      // Vertex color — slope from dot(normal, sphere dir) same as before
      const slope = 1 - nrm.dot(dir);
      if (hasClimate) {
        const cs = climateFn!(dir, h);
        biomeColor(h, slope, cs.temperature, cs.moisture, colors, vi * 3);
      } else {
        elevationColor(h, slope, colors, vi * 3);
      }

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
    const bx = positions[borderVI * 3    ] + originX;
    const by = positions[borderVI * 3 + 1] + originY;
    const bz = positions[borderVI * 3 + 2] + originZ;
    const len = Math.sqrt(bx * bx + by * by + bz * bz);
    const pullScale = (len - skirtDepth) / len;

    positions[vi * 3    ] = bx * pullScale - originX;
    positions[vi * 3 + 1] = by * pullScale - originY;
    positions[vi * 3 + 2] = bz * pullScale - originZ;

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

  return { positions, normals, colors, plateColors, indices, originX, originY, originZ };
}

// ---------------------------------------------------------------------------
// Main-thread wrapper: raw arrays → BufferGeometry + origin Vector3.
// ---------------------------------------------------------------------------

export function arraysToGeometry(a: ChunkMeshArrays): ChunkMeshData {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position',  new BufferAttribute(a.positions, 3));
  geometry.setAttribute('normal',    new BufferAttribute(a.normals,   3));
  geometry.setAttribute('color',     new BufferAttribute(a.colors,    3));
  if (a.plateColors !== null) {
    geometry.setAttribute('plateColor', new BufferAttribute(a.plateColors, 3));
  }
  geometry.setIndex(new BufferAttribute(a.indices, 1));
  geometry.computeBoundingSphere();

  const origin = new Vector3(a.originX, a.originY, a.originZ);
  return { geometry, origin };
}

// ---------------------------------------------------------------------------
// Public builder — thin composition of the two above.
// ---------------------------------------------------------------------------

export function buildChunkGeometry(p: ChunkParams): ChunkMeshData {
  return arraysToGeometry(computeChunkArrays(p));
}
