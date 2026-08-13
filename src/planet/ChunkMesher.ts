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
  /**
   * optional erosion sampler (pipeline step 9, Phase 3).
   * When present, land vertices inside lake basins receive a still-water tint
   * blended over the base biome/elevation color.
   */
  erosion?: {
    /**
     * Lake spill elevation for a unit sphere direction.
     * Returns a SENTINEL value (~-999) where the vertex is NOT inside a lake basin.
     */
    lakeAt(dir: Vector3): number;
    /**
     * Lake presence mask for a unit sphere direction; bilinear sample of a clean
     * 0/1 field (no sentinel mixing). Values > 0.5 are inside a lake basin.
     */
    lakeMaskAt(dir: Vector3): number;
    /**
     * Deposition environment code for a unit sphere direction.
     * Returns ENV_* code: 0=DEFAULT, 1=FLOODPLAIN, 2=FAN, 3=DELTA.
     * Nearest-neighbor sample — used for color only, never heightFn.
     */
    depositEnvAt(dir: Vector3): number;
  } | null;
  /**
   * optional subsurface sampler (pipeline step 11).
   * When present, land vertices receive emergence overlays: springs, permafrost
   * ice seeps, and ore outcrops blended over the base biome/erosion color.
   */
  subsurface?: {
    /** normalized water table elevation for a unit sphere direction */
    waterTableAt(dir: Vector3): number;
    /** ore density at a given unit sphere direction and absolute normalized elevation */
    oreAt(dir: Vector3, elev: number): number;
    /** permafrost presence mask (0/1) for a unit sphere direction */
    permafrostMaskAt(dir: Vector3): number;
    /** permafrost top elevation for a unit sphere direction */
    permafrostTopAt(dir: Vector3): number;
  } | null;
  /**
   * optional hardness sampler — returns C1 baked-grid rock hardness in [0,1]
   * for a unit sphere direction. When present, per-vertex hardness is baked as
   * a BufferAttribute ('rockHardness') for the materials view, AND used for
   * sediment/cap-rock color tinting in the normal view.
   */
  hardnessFn?: (dir: Vector3) => number;
}

export interface ChunkMeshData {
  geometry: BufferGeometry;
  /** World-space chunk center (double precision). Set mesh.position = origin. */
  origin:   Vector3;
}

export interface ChunkMeshArrays {
  positions:     Float32Array;
  normals:       Float32Array;
  colors:        Float32Array;
  plateColors:   Float32Array | null;
  climateMoist:  Float32Array | null;
  subsurfaceWet: Float32Array | null;
  /** Per-vertex rock hardness in [0,1]. null when hardnessFn not provided. */
  rockHardness:  Float32Array | null;
  indices:       Uint32Array;
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
// seaLevel is a normalized elevation value (same units as e, roughly [-1,1]) that
// is the dynamic waterline. All waterline decisions key off seaLevel, not e=0.
//
// Band table (depth = seaLevel - e when e < seaLevel; haw = e - seaLevel when e >= seaLevel):
//
//  SEABED (e < seaLevel):
//   depth > 0.55          abyssal sediment  #232220  near-black warm grey
//   0.18..0.55            deep sediment     →#3a352c  dark brown-grey
//   0.045..0.18           slope sediment    →#605442  medium brown
//   0..0.045              sandy shelf       →#9c8a66  light tan
//
//  LAND (e >= seaLevel):
//   0..0.012              exposed shelf / sediment →#8c7a58  barren beige (drained seabed)
//                         → transitions to sand beach at haw=0.012
//   0.012..0.18           sand beach        →#b8a36e
//   0.012..0.18           lowland           →#5a7a4a  desaturated green
//   0.18..0.42            highland          →#8a7a55  brown-tan
//   0.42..0.62            bare rock         →#7a7060  grey-brown
//   haw > 0.55 (blend)    snow              →#e6e8eb  near-white (full by 0.62)
//
//  SLOPE override (land only, slope > 0.22): blend toward rock #6e6a64
// ---------------------------------------------------------------------------

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function elevationColor(e: number, slope: number, seaLevel: number, out: Float32Array, base: number): void {
  let r: number, g: number, b: number;

  const belowWater = e < seaLevel;

  if (belowWater) {
    // ---- SEABED (geology, not water — the blue comes from the water shell) -----
    // depth increases as e drops further below seaLevel
    const depth = seaLevel - e;

    // Abyssal sediment #232220
    const rAby = 0x23 / 255, gAby = 0x22 / 255, bAby = 0x20 / 255;
    // Deep sediment   #3a352c
    const rDep = 0x3a / 255, gDep = 0x35 / 255, bDep = 0x2c / 255;
    // Slope sediment  #605442
    const rSlp = 0x60 / 255, gSlp = 0x54 / 255, bSlp = 0x42 / 255;
    // Sandy shelf  #9c8a66
    const rShf = 0x9c / 255, gShf = 0x8a / 255, bShf = 0x66 / 255;

    if (depth > 0.55) {
      // pure abyssal
      r = rAby; g = gAby; b = bAby;
    } else if (depth > 0.18) {
      // abyssal → deep  (window ~0.015 around depth=0.55)
      const t = clamp01((0.55 - depth) / 0.015);
      r = lerp(rAby, rDep, t);
      g = lerp(gAby, gDep, t);
      b = lerp(bAby, bDep, t);
    } else if (depth > 0.045) {
      // deep → cont. slope  (window 0.015 around depth=0.18)
      const t = clamp01((0.18 - depth) / 0.015);
      r = lerp(rDep, rSlp, t);
      g = lerp(gDep, gSlp, t);
      b = lerp(bDep, bSlp, t);
    } else {
      // cont. slope → shelf  (window 0.015 around depth=0.045)
      const t = clamp01((0.045 - depth) / 0.015);
      r = lerp(rSlp, rShf, t);
      g = lerp(gSlp, gShf, t);
      b = lerp(bSlp, bShf, t);
    }

  } else {
    // ---- LAND --------------------------------------------------------
    // heightAboveWater drives the land palette
    const haw = e - seaLevel;

    // Exposed shelf / drained seabed #8c7a58  (barren sediment for freshly-exposed basin)
    const rExp = 0x8c / 255, gExp = 0x7a / 255, bExp = 0x58 / 255;
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

    if (haw < 0.012) {
      // Exposed shelf → beach sand transition
      // Very small haw = freshly-drained seafloor (barren sediment/beige).
      // Transitions to beach sand at haw=0.012.
      const t = clamp01(haw / 0.012);
      r = lerp(rExp, rSnd, t);
      g = lerp(gExp, gSnd, t);
      b = lerp(bExp, bSnd, t);
    } else if (haw < 0.18) {
      // sand → lowland (window 0.015 around haw=0.012)
      const t = clamp01((haw - 0.012) / 0.015);
      r = lerp(rSnd, rLow, t);
      g = lerp(gSnd, gLow, t);
      b = lerp(bSnd, bLow, t);
    } else if (haw < 0.42) {
      // lowland → highland (window 0.015 around haw=0.18)
      const t = clamp01((haw - 0.18) / 0.015);
      r = lerp(rLow, rHig, t);
      g = lerp(gLow, gHig, t);
      b = lerp(bLow, bHig, t);
    } else if (haw < 0.62) {
      // highland → bare rock (window 0.015 around haw=0.42)
      const t = clamp01((haw - 0.42) / 0.015);
      r = lerp(rHig, rRck, t);
      g = lerp(gHig, gRck, t);
      b = lerp(bHig, bRck, t);
    } else {
      r = rRck; g = gRck; b = bRck;
    }

    // Snow blend: start at haw=0.55, fully white by haw=0.62
    if (haw > 0.55) {
      const snowT = clamp01((haw - 0.55) / (0.62 - 0.55));
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
//   - Seabed (e < seaLevel): identical geology palette to elevationColor (verbatim)
//     keyed on depth = seaLevel - e — the water shell draws the ocean color.
//     Cold seabed (temperature < SNOW_TEMP): rendered as sea ice (white), giving
//     polar ice caps over the ocean.
//   - Land (e >= seaLevel): biome chosen from temperature (°C) + moisture (0..1),
//     blended smoothly (no hard biome edges). Ice/snow where it's cold (caps poles
//     AND peaks via a temperature-driven snow line), desert where dry, forests/
//     grassland/tundra otherwise. High elevation trends rocky; cliffs → rock grey.
//     Very small heightAboveWater (= e - seaLevel) reads as barren exposed-shelf
//     sediment so a drained ocean basin looks like exposed seabed, not grassland.
// ---------------------------------------------------------------------------

const SNOW_TEMP = -2;     // °C — below this, land/ocean trends to snow/ice (blended over a few °C)
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
  seaLevel: number,
  out: Float32Array,
  base: number,
): void {
  let r: number, g: number, b: number;

  const belowWater = e < seaLevel;

  if (belowWater) {
    // ---- SEABED (geology, keyed on depth below seaLevel) --------------------
    const depth = seaLevel - e;

    const rAby = 0x23 / 255, gAby = 0x22 / 255, bAby = 0x20 / 255;
    const rDep = 0x3a / 255, gDep = 0x35 / 255, bDep = 0x2c / 255;
    const rSlp = 0x60 / 255, gSlp = 0x54 / 255, bSlp = 0x42 / 255;
    const rShf = 0x9c / 255, gShf = 0x8a / 255, bShf = 0x66 / 255;

    if (depth > 0.55) {
      r = rAby; g = gAby; b = bAby;
    } else if (depth > 0.18) {
      const t = clamp01((0.55 - depth) / 0.015);
      r = lerp(rAby, rDep, t);
      g = lerp(gAby, gDep, t);
      b = lerp(bAby, bDep, t);
    } else if (depth > 0.045) {
      const t = clamp01((0.18 - depth) / 0.015);
      r = lerp(rDep, rSlp, t);
      g = lerp(gDep, gSlp, t);
      b = lerp(bDep, bSlp, t);
    } else {
      const t = clamp01((0.045 - depth) / 0.015);
      r = lerp(rSlp, rShf, t);
      g = lerp(gSlp, gShf, t);
      b = lerp(bSlp, bShf, t);
    }

    // --- Sea ice: cold ocean surface → white --------------------------------
    // Where the ocean surface is cold enough, render it as polar ice.
    // Blended over the same SNOW_BLEND window as land snow so the transition
    // looks continuous at the coast.
    if (temperature < SNOW_TEMP + SNOW_BLEND) {
      const iceT = 1 - smoothstepM(SNOW_TEMP, SNOW_TEMP + SNOW_BLEND, temperature);
      const rIce = 0xe8 / 255, gIce = 0xee / 255, bIce = 0xf2 / 255; // slightly blue-white
      r = lerp(r, rIce, iceT);
      g = lerp(g, gIce, iceT);
      b = lerp(b, bIce, iceT);
    }

  } else {
    // ---- LAND: biome by (temperature, moisture), keyed on heightAboveWater --
    const haw = e - seaLevel;

    // Palette (all generic — Earth's look emerges from Earth-like params):
    const sandHot = [0xc2 / 255, 0xa8 / 255, 0x78 / 255];   // hot dry desert
    const sandCold = [0x9a / 255, 0x8c / 255, 0x70 / 255];  // cold dry / rocky steppe
    const grass = [0x8f / 255, 0x95 / 255, 0x55 / 255];     // grassland / steppe
    const tropical = [0x2f / 255, 0x6b / 255, 0x34 / 255];  // hot wet jungle
    const temperate = [0x4f / 255, 0x7a / 255, 0x45 / 255]; // temperate forest
    const tundra = [0x6b / 255, 0x6f / 255, 0x57 / 255];    // cold sparse (brown-grey)
    const rock = [0x7a / 255, 0x70 / 255, 0x60 / 255];      // bare rock (matches elevationColor)
    const snow = [0xe6 / 255, 0xe8 / 255, 0xeb / 255];      // ice / snow

    // Exposed shelf / drained seabed — barren sediment for freshly-exposed basin.
    // Blends in at very small haw so a drained ocean basin looks like exposed seabed.
    const rExp = 0x8c / 255, gExp = 0x7a / 255, bExp = 0x58 / 255;

    // --- 2. Dry vs vegetated axis -------------------------------------------
    // dryness: 1 when very dry (M→0), 0 once moisture clears the desert line.
    const dry = 1 - smoothstepM(0.14, 0.22, moisture);
    // Within dry: hot deserts vs cold steppe (by temperature).
    const hotDry = smoothstepM(2, 14, temperature);
    const desertR = lerp(sandCold[0], sandHot[0], hotDry);
    const desertG = lerp(sandCold[1], sandHot[1], hotDry);
    const desertB = lerp(sandCold[2], sandHot[2], hotDry);

    // --- 3. Vegetated biome by temperature ----------------------------------
    const tWarm = smoothstepM(2, 8, temperature);        // 0 cold (tundra), 1 by 8°C (forest)
    const tHot = smoothstepM(18, 24, temperature);       // 0 mild, 1 by 24°C
    const tropWet = smoothstepM(0.42, 0.58, moisture);   // tropical needs moisture too
    const tropMix = tHot * tropWet;
    const mildR = lerp(temperate[0], tropical[0], tropMix);
    const mildG = lerp(temperate[1], tropical[1], tropMix);
    const mildB = lerp(temperate[2], tropical[2], tropMix);
    let vegR = lerp(tundra[0], mildR, tWarm);
    let vegG = lerp(tundra[1], mildG, tWarm);
    let vegB = lerp(tundra[2], mildB, tWarm);

    // --- 4. Grassland/steppe for middling moisture --------------------------
    const grassW = smoothstepM(0.20, 0.30, moisture) * (1 - smoothstepM(0.38, 0.48, moisture)) * (1 - tropMix);
    vegR = lerp(vegR, grass[0], grassW);
    vegG = lerp(vegG, grass[1], grassW);
    vegB = lerp(vegB, grass[2], grassW);

    // --- 5. Combine dry (desert) with vegetated along the dryness axis -------
    r = lerp(vegR, desertR, dry);
    g = lerp(vegG, desertG, dry);
    b = lerp(vegB, desertB, dry);

    // --- 6. Exposed-shelf / beach band at the immediate waterline ----------
    // haw < 0.012: freshly-exposed seabed sediment (barren beige) blending up
    // to the biome color. This handles both normal beaches and drained basins.
    if (haw < 0.012) {
      const t = clamp01(haw / 0.012);
      r = lerp(rExp, r, t);
      g = lerp(gExp, g, t);
      b = lerp(bExp, b, t);
    }

    // --- 7. High elevation (keyed on haw) trends rocky ----------------------
    // Above haw ≈ 0.5 blend toward bare rock (the T-driven snow below caps peaks).
    if (haw > 0.5) {
      const rockT = smoothstepM(0.5, 0.72, haw);
      r = lerp(r, rock[0], rockT);
      g = lerp(g, rock[1], rockT);
      b = lerp(b, rock[2], rockT);
    }

    // --- 8. Snow / ice by TEMPERATURE (latitude- AND altitude-dependent) -----
    // T already falls with both latitude and altitude, so this gives polar ice
    // caps at sea level AND a natural snow line on cold peaks.
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

// ---------------------------------------------------------------------------
// Erosion tint constants (tunable by the user).
//
// Lake:  still-water blue #3a5e7a. Applied when lakeMaskAt > 0.5 (clean 0/1 mask,
//   bilinear-safe). Tint scales by mask value for a soft shoreline fade.
//
// There is deliberately NO river tint here. It was painted from `erosion.accAt`
// on the 256² bake grid (~3 km/texel at RADIUS=500 km), so a "river" read as a
// several-km-wide colour smear — a stripe of blue paint with no channel under it.
// The rivers feature was removed entirely; see CLAUDE.md "Decisions".
// ---------------------------------------------------------------------------

const LAKE_BLEND           = 0.72;
const LAKE_R = 0x3a / 255;   // #3a5e7a
const LAKE_G = 0x5e / 255;
const LAKE_B = 0x7a / 255;

// ---------------------------------------------------------------------------
// Subsurface emergence overlay constants.
//
// SPRING: water table at or above the terrain surface → spring / wet ground.
//   WT_WET_BAND      — normalized-height band over which spring emergence ramps
//                      to full blend (water table this far above surface = fully wet).
//   SPRING_BLEND     — max color-blend weight toward the spring tint.
//
// ICE SEEP: permafrost top at or above the terrain surface → pale blue film.
//   ICE_BLEND — blend weight toward the ice-seep tint.
//
// ORE OUTCROP: surface ore density above threshold → rust/ochre stain.
//   ORE_OUTCROP_THRESH — minimum oreAt(dir, h) to show a tint.
//   ORE_BLEND          — blend weight toward the ore tint.
// ---------------------------------------------------------------------------

const WT_WET_BAND        = 0.01;
const SPRING_BLEND       = 0.60;
const ICE_BLEND          = 0.50;
const ORE_OUTCROP_THRESH = 0.60;
const ORE_BLEND          = 0.45;

// Spring tint: dark green-blue #2f6e54
const SPRING_R = 0x2f / 255;
const SPRING_G = 0x6e / 255;
const SPRING_B = 0x54 / 255;

// Ice-seep tint: pale blue #bfe0ff
const ICE_R = 0xbf / 255;
const ICE_G = 0xe0 / 255;
const ICE_B = 0xff / 255;

// Ore-outcrop tint: rust/ochre #c08a3a
const ORE_R = 0xc0 / 255;
const ORE_G = 0x8a / 255;
const ORE_B = 0x3a / 255;

// ---------------------------------------------------------------------------
// Sediment / cap-rock tint constants (normal-view T7 coloring).
//
// These tints are applied AFTER biome/erosion coloring — final compositing step.
// All blends are smooth (no hard thresholds) so seams cannot result.
//
// HARD CAP-ROCK: high hardness (>0.65) on land → slight grey-brown shift on ridges.
//   Tint toward #7e7060 — desaturated dark brown-grey (harder than bare rock base).
//   Max blend 0.30 so biome color still reads through.
//
// SEDIMENT FANS/FLOODPLAINS: depositEnv == FAN (2) or FLOODPLAIN (1) → warm alluvial tone.
//   FLOODPLAIN tint: muted clay tan #a89270 — flat, silty, moist alluvial lowland.
//   FAN tint: dry alluvial fan sandy beige #c4aa7a.
//
// DELTA: salt-flat / tidal mudflat greyish-tan #9a9080.
//   Light, slightly desaturated; shows the marine-fluvial transition clearly.
// ---------------------------------------------------------------------------

const HARDCAP_THRESH     = 0.65;   // hardness above which cap-rock tint starts
const HARDCAP_BLEND_MAX  = 0.30;
const HARDCAP_R = 0x7e / 255;      // #7e7060
const HARDCAP_G = 0x70 / 255;
const HARDCAP_B = 0x60 / 255;

// depositEnv code constants (mirrors erosion.ts ENV_* values)
const ENV_DEFAULT    = 0;
const ENV_FLOODPLAIN = 1;
const ENV_FAN        = 2;
const ENV_DELTA      = 3;

const SEDIMENT_BLEND  = 0.28;       // max blend weight for sediment tints

// Floodplain: muted clay tan #a89270
const FLOODPLAIN_R = 0xa8 / 255;
const FLOODPLAIN_G = 0x92 / 255;
const FLOODPLAIN_B = 0x70 / 255;

// Alluvial fan: dry sandy beige #c4aa7a
const FAN_R = 0xc4 / 255;
const FAN_G = 0xaa / 255;
const FAN_B = 0x7a / 255;

// Delta: greyish-tan #9a9080
const DELTA_R = 0x9a / 255;
const DELTA_G = 0x90 / 255;
const DELTA_B = 0x80 / 255;

export function computeChunkArrays(p: ChunkParams, seaLevel: number): ChunkMeshArrays {
  const { faceIndex, level, ix, iy, resolution: res, radius, heightScale, heightFn, plateColorFn, climateFn, erosion, subsurface, hardnessFn } = p;
  // Convenience: heightFn with level pre-bound — avoids repeating `level` at every call site
  // inside this function (all vertices in a chunk share the same LOD level).
  const hFn = (dir: Vector3): number => heightFn(dir, level);
  const basis = FACE_BASES[faceIndex];
  const hasPlateColor  = plateColorFn !== undefined;
  const hasClimate     = climateFn    !== undefined;
  const hasSubsurface  = subsurface !== null && subsurface !== undefined;
  const hasHardness    = hardnessFn  !== undefined;

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
  const plateColors    = hasPlateColor  ? new Float32Array(totalVerts * 3) : null;
  const climateMoist   = hasClimate     ? new Float32Array(totalVerts)     : null;
  // Surface-wetness channel (legacy name): open water (lakes) + subsurface seep.
  // Drives the normal material's roughness so water reads as wet (specular). Allocated
  // whenever erosion OR subsurface is present.
  const subsurfaceWet  = (hasSubsurface || (erosion !== null && erosion !== undefined))
    ? new Float32Array(totalVerts) : null;
  const rockHardness   = hasHardness    ? new Float32Array(totalVerts)     : null;
  const indices        = new Uint32Array(totalIndices);

  // Scratch objects — reused, no per-vertex allocation
  const dir    = new Vector3();
  const world  = new Vector3();
  const worldL = new Vector3();
  const worldR = new Vector3();
  const worldD = new Vector3();
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

  // -- Skirt depth (computed after Pass 1, once the chunk's relief is known) --
  // A skirt only has to hide the LOD CRACK at a T-junction: the vertical gap
  // between this chunk's edge and a coarser neighbour's edge. That gap is the
  // height DISCONTINUITY between the two LOD samplings, which is bounded by the
  // terrain relief across the chunk — NOT by the chunk's width. Sizing the skirt
  // to arc length (the old bug) gave coarse distant chunks km-deep flanges that
  // showed up as giant white walls on flat plains. So we size it to the chunk's
  // own measured relief (maxH−minH over the interior grid) instead.
  //   skirtDepth = clamp(RELIEF_FACTOR · reliefWorld + FLOOR, FLOOR, CAP)
  const SKIRT_RELIEF_FACTOR = 1.5;  // cover up to ~1.5× the chunk's own relief
  const SKIRT_FLOOR = 2.0;          // m — hairline skirt even on dead-flat chunks (fp noise / tiny cracks); invisibly small
  const SKIRT_CAP = heightScale;    // m — never exceed total terrain relief (~1200 m); rugged chunks are fine-LOD & near anyway, so this rarely binds
  // Track the chunk's normalized height range over the interior grid (Pass 1).
  let minH = Infinity;
  let maxH = -Infinity;

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
  //         1 heightFn eval per vertex.
  // Pass 2: compute normals with ONE method for EVERY vertex —
  //         n = normalize( cross( P[gi+1,gj]-P[gi-1,gj],  P[gi,gj+1]-P[gi,gj-1] ) )
  //         flipped outward (dot with radial direction > 0).
  //   - Interior neighbours read the cached `positions` (0 extra heightFn evals).
  //   - Border vertices' off-edge neighbour(s) don't exist in the grid, so we
  //     synthesise a "ghost" position one grid step beyond the edge via ghostPos
  //     (1 heightFn eval each): a non-corner border vertex needs 1 ghost, a corner
  //     needs 2. Because every vertex uses the identical estimator there is no
  //     border↔interior discontinuity (the old seam ring), and because ghosts come
  //     from the SAME continuous heightFn the neighbour tile samples for its real
  //     edge-adjacent vertices, same-level chunk boundaries stay seam-free.
  //
  // Cost: only border vertices do extra heightFn evals — 1 per non-corner border
  // vertex + 2 per corner = 4*(res-1)+4*2 = 4*res+4 extra evals per chunk
  // (res=32 → 132), vs the old all-CD interior alternative's ~4 per interior
  // vertex (~res² extra). The interior stays free.

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

      // Track the chunk's relief (normalized height range) for the skirt depth.
      if (h < minH) minH = h;
      if (h > maxH) maxH = h;

      vi++;
    }
  }

  // -- Skirt depth from measured relief (see rationale above) ----------------
  // reliefWorld = the chunk's height span in WORLD units. The LOD T-junction gap
  // against a one-level-coarser neighbour cannot exceed the terrain deviation
  // across this chunk's edge, which is ≤ reliefWorld; RELIEF_FACTOR=1.5 adds
  // margin so the (now-small) skirt still fully closes any crack. Flat plains →
  // reliefWorld≈0 → skirtDepth≈SKIRT_FLOOR (~2 m), buried and invisible: no walls.
  const reliefWorld = (maxH - minH) * heightScale;
  let skirtDepth = SKIRT_RELIEF_FACTOR * reliefWorld + SKIRT_FLOOR;
  if (skirtDepth < SKIRT_FLOOR) skirtDepth = SKIRT_FLOOR;
  if (skirtDepth > SKIRT_CAP) skirtDepth = SKIRT_CAP;

  // --- Pass 2: normals + colors ------------------------------------------------
  vi = 0;
  for (let gj = 0; gj < gridSize; gj++) {
    for (let gi = 0; gi < gridSize; gi++) {
      // Restore cached sphere direction and height
      dir.x = dirCache[vi * 3    ];
      dir.y = dirCache[vi * 3 + 1];
      dir.z = dirCache[vi * 3 + 2];
      const h = hCache[vi];

      // --- Unified normal: cross product of 4 grid-neighbour positions ---------
      // Fetch each neighbour's ORIGIN-RELATIVE world position. In-bounds neighbours
      // read the cached `positions`; off-edge neighbours (border vertices only) are
      // synthesised as ghosts from the same continuous heightFn. The (gi,gj) grid
      // axes map to (column,row): +gi → +1 in the flat array, +gj → +gridSize.
      if (gi > 0) {
        const idxL = vi - 1;               // (gi-1, gj)
        worldL.set(positions[idxL * 3], positions[idxL * 3 + 1], positions[idxL * 3 + 2]);
      } else {
        ghostPos(gi - 1, gj, worldL);      // ghost at gi=-1
      }
      if (gi < res) {
        const idxR = vi + 1;               // (gi+1, gj)
        worldR.set(positions[idxR * 3], positions[idxR * 3 + 1], positions[idxR * 3 + 2]);
      } else {
        ghostPos(gi + 1, gj, worldR);      // ghost at gi=res+1
      }
      if (gj > 0) {
        const idxD = vi - gridSize;        // (gi, gj-1)
        worldD.set(positions[idxD * 3], positions[idxD * 3 + 1], positions[idxD * 3 + 2]);
      } else {
        ghostPos(gi, gj - 1, worldD);      // ghost at gj=-1
      }
      if (gj < res) {
        const idxU = vi + gridSize;        // (gi, gj+1)
        worldU.set(positions[idxU * 3], positions[idxU * 3 + 1], positions[idxU * 3 + 2]);
      } else {
        ghostPos(gi, gj + 1, worldU);      // ghost at gj=res+1
      }

      // Central-difference tangent vectors (origin offsets cancel in the subtraction)
      tan1.subVectors(worldR, worldL); // ∂pos/∂gi direction
      tan2.subVectors(worldU, worldD); // ∂pos/∂gj direction

      nrm.crossVectors(tan1, tan2).normalize();

      // Ensure outward-facing (dot with radial direction of this vertex > 0)
      if (nrm.dot(dir) < 0) nrm.negate();

      normals[vi * 3    ] = nrm.x;
      normals[vi * 3 + 1] = nrm.y;
      normals[vi * 3 + 2] = nrm.z;

      // Vertex color — slope from dot(normal, sphere dir) same as before
      const slope = 1 - nrm.dot(dir);
      if (hasClimate) {
        const cs = climateFn!(dir, h);
        biomeColor(h, slope, cs.temperature, cs.moisture, seaLevel, colors, vi * 3);
        climateMoist![vi] = cs.moisture;
      } else {
        elevationColor(h, slope, seaLevel, colors, vi * 3);
      }

      // -- Erosion + subsurface tinting (land only) --------------------------------
      // Compositing order:
      //   1. Lake
      //   2. Spring / ice-seep / ore (subsurface only — skipped when absent)
      if (erosion !== null && erosion !== undefined && h >= seaLevel) {
        const base3 = vi * 3;

        let springEmergence = 0;
        let surfWet = 0; // surface wetness [0,1] → roughness (lake / seep)
        if (hasSubsurface && subsurface !== null && subsurface !== undefined) {
          const wt = subsurface.waterTableAt(dir);
          springEmergence = clamp01((wt - h) / WT_WET_BAND);
        }

        // Step 1: Lake tint (still-water blue #3a5e7a).
        // Keys off the clean 0/1 lake mask; 0.5 isocontour is the shoreline.
        const lkMask = erosion.lakeMaskAt(dir);
        if (lkMask > 0.5) {
          const lakeT = LAKE_BLEND * lkMask;
          colors[base3    ] = lerp(colors[base3    ], LAKE_R, lakeT);
          colors[base3 + 1] = lerp(colors[base3 + 1], LAKE_G, lakeT);
          colors[base3 + 2] = lerp(colors[base3 + 2], LAKE_B, lakeT);
          surfWet = Math.max(surfWet, lkMask);
        }

        // Step 2: Subsurface-only overlays — spring, ice seep, ore outcrop.
        // Entire block is skipped when subsurface is absent.
        if (hasSubsurface && subsurface !== null && subsurface !== undefined) {
          let wet = 0;

          // --- SPRING / wet ground ------------------------------------------------
          // Water table at or above terrain → dark green-blue tint (#2f6e54).
          // springEmergence already computed above for the effAcc boost.
          if (springEmergence > 0) {
            const t = springEmergence * SPRING_BLEND;
            colors[base3    ] = lerp(colors[base3    ], SPRING_R, t);
            colors[base3 + 1] = lerp(colors[base3 + 1], SPRING_G, t);
            colors[base3 + 2] = lerp(colors[base3 + 2], SPRING_B, t);
            wet = Math.max(wet, t);
          }

          // --- ICE SEEP -----------------------------------------------------------
          // Permafrost top at or above terrain → pale blue film (#bfe0ff).
          if (subsurface.permafrostMaskAt(dir) > 0.5 && subsurface.permafrostTopAt(dir) >= h) {
            colors[base3    ] = lerp(colors[base3    ], ICE_R, ICE_BLEND);
            colors[base3 + 1] = lerp(colors[base3 + 1], ICE_G, ICE_BLEND);
            colors[base3 + 2] = lerp(colors[base3 + 2], ICE_B, ICE_BLEND);
            wet = Math.max(wet, ICE_BLEND);
          }

          // --- ORE OUTCROP --------------------------------------------------------
          // High surface-ore density at this elevation → rust/ochre stain (#c08a3a).
          // Second arg is the absolute normalized terrain elevation (same units as h).
          if (subsurface.oreAt(dir, h) > ORE_OUTCROP_THRESH) {
            colors[base3    ] = lerp(colors[base3    ], ORE_R, ORE_BLEND);
            colors[base3 + 1] = lerp(colors[base3 + 1], ORE_G, ORE_BLEND);
            colors[base3 + 2] = lerp(colors[base3 + 2], ORE_B, ORE_BLEND);
          }

          surfWet = Math.max(surfWet, wet);
        }

        if (subsurfaceWet !== null) subsurfaceWet[vi] = surfWet;
      } else if (hasSubsurface && subsurface !== null && subsurface !== undefined && h >= seaLevel) {
        // Subsurface present but no erosion: spring/ice/ore still apply.
        // Lake tint is skipped (no lake mask available).
        const base3 = vi * 3;
        let wet = 0;

        const wt = subsurface.waterTableAt(dir);
        const springEmergence = clamp01((wt - h) / WT_WET_BAND);
        if (springEmergence > 0) {
          const t = springEmergence * SPRING_BLEND;
          colors[base3    ] = lerp(colors[base3    ], SPRING_R, t);
          colors[base3 + 1] = lerp(colors[base3 + 1], SPRING_G, t);
          colors[base3 + 2] = lerp(colors[base3 + 2], SPRING_B, t);
          wet = Math.max(wet, t);
        }

        if (subsurface.permafrostMaskAt(dir) > 0.5 && subsurface.permafrostTopAt(dir) >= h) {
          colors[base3    ] = lerp(colors[base3    ], ICE_R, ICE_BLEND);
          colors[base3 + 1] = lerp(colors[base3 + 1], ICE_G, ICE_BLEND);
          colors[base3 + 2] = lerp(colors[base3 + 2], ICE_B, ICE_BLEND);
          wet = Math.max(wet, ICE_BLEND);
        }

        if (subsurface.oreAt(dir, h) > ORE_OUTCROP_THRESH) {
          colors[base3    ] = lerp(colors[base3    ], ORE_R, ORE_BLEND);
          colors[base3 + 1] = lerp(colors[base3 + 1], ORE_G, ORE_BLEND);
          colors[base3 + 2] = lerp(colors[base3 + 2], ORE_B, ORE_BLEND);
        }

        subsurfaceWet![vi] = wet;
      }

      // Plate color (if provided)
      if (hasPlateColor && plateColors !== null) {
        const pc = plateColorFn!(dir);
        plateColors[vi * 3    ] = pc[0];
        plateColors[vi * 3 + 1] = pc[1];
        plateColors[vi * 3 + 2] = pc[2];
      }

      // Rock hardness attribute (materials view) + sediment/cap-rock tint (normal view).
      // hardnessFn is the same C1 baked-grid lookup used by tectonics.hardnessAt — deterministic.
      if (hasHardness && hardnessFn !== undefined) {
        const hval = hardnessFn(dir);
        // Bake scalar attribute for the materials view.
        if (rockHardness !== null) rockHardness[vi] = hval;

        // Sediment / cap-rock tinting in the normal view (land only).
        // Applied last so it composites over biome + subsurface overlays.
        // All blends are continuous smoothsteps — no hard thresholds (seam-safe).
        if (h >= seaLevel) {
          const base3 = vi * 3;

          // --- CAP-ROCK: hard ridges → slight grey-brown shift ---
          // Smoothstep gate from HARDCAP_THRESH to 1.0 so the blend ramps in
          // gradually; fully hard rock = HARDCAP_BLEND_MAX mix toward #7e7060.
          if (hval > HARDCAP_THRESH) {
            const capT = clamp01((hval - HARDCAP_THRESH) / (1.0 - HARDCAP_THRESH)) * HARDCAP_BLEND_MAX;
            colors[base3    ] = lerp(colors[base3    ], HARDCAP_R, capT);
            colors[base3 + 1] = lerp(colors[base3 + 1], HARDCAP_G, capT);
            colors[base3 + 2] = lerp(colors[base3 + 2], HARDCAP_B, capT);
          }

          // --- SEDIMENT DEPOSITION: floodplain / fan / delta → warm alluvial tone ---
          // depositEnvAt is nearest-neighbor on erosion's discrete ENV grid.
          // Only active when erosion is present (depositEnvAt lives on erosion).
          if (erosion !== null && erosion !== undefined) {
            const env = erosion.depositEnvAt(dir);
            if (env !== ENV_DEFAULT) {
              let tR: number, tG: number, tB: number;
              if (env === ENV_FLOODPLAIN) {
                tR = FLOODPLAIN_R; tG = FLOODPLAIN_G; tB = FLOODPLAIN_B;
              } else if (env === ENV_FAN) {
                tR = FAN_R; tG = FAN_G; tB = FAN_B;
              } else { // ENV_DELTA
                tR = DELTA_R; tG = DELTA_G; tB = DELTA_B;
              }
              colors[base3    ] = lerp(colors[base3    ], tR, SEDIMENT_BLEND);
              colors[base3 + 1] = lerp(colors[base3 + 1], tG, SEDIMENT_BLEND);
              colors[base3 + 2] = lerp(colors[base3 + 2], tB, SEDIMENT_BLEND);
            }
          }
        }
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

    // Copy climateMoist from border vertex
    if (hasClimate && climateMoist !== null) {
      climateMoist[vi] = climateMoist[borderVI];
    }

    // Copy surface-wetness from border vertex
    if (subsurfaceWet !== null) {
      subsurfaceWet[vi] = subsurfaceWet[borderVI];
    }

    // Copy rockHardness from border vertex
    if (hasHardness && rockHardness !== null) {
      rockHardness[vi] = rockHardness[borderVI];
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

  return { positions, normals, colors, plateColors, climateMoist, subsurfaceWet, rockHardness, indices, originX, originY, originZ };
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
  if (a.climateMoist !== null) {
    geometry.setAttribute('climateMoist', new BufferAttribute(a.climateMoist, 1));
  }
  if (a.subsurfaceWet !== null) {
    geometry.setAttribute('subsurfaceWet', new BufferAttribute(a.subsurfaceWet, 1));
  }
  if (a.rockHardness !== null) {
    geometry.setAttribute('rockHardness', new BufferAttribute(a.rockHardness, 1));
  }
  geometry.setIndex(new BufferAttribute(a.indices, 1));
  geometry.computeBoundingSphere();

  const origin = new Vector3(a.originX, a.originY, a.originZ);
  return { geometry, origin };
}

// ---------------------------------------------------------------------------
// Public builder — thin composition of the two above.
// ---------------------------------------------------------------------------

export function buildChunkGeometry(p: ChunkParams, seaLevel = 0): ChunkMeshData {
  return arraysToGeometry(computeChunkArrays(p, seaLevel));
}
