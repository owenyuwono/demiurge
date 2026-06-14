/**
 * Dependency-free, fully deterministic 3D simplex noise + FBM utilities.
 *
 * Core rule: terrain is a pure function of (seed, position).
 * No Math.random. Same inputs → byte-identical outputs.
 */

// ---------------------------------------------------------------------------
// PRNG: splitmix32 — used only at construction time to fill perm table
// ---------------------------------------------------------------------------

function splitmix32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

// ---------------------------------------------------------------------------
// 3D simplex noise — Gustavson-style
// ---------------------------------------------------------------------------

// Skewing/unskewing factors for 3D
const F3 = 1.0 / 3.0;
const G3 = 1.0 / 6.0;

// 12 gradient directions (midpoints of edges of a cube)
const GRAD3: ReadonlyArray<readonly [number, number, number]> = [
  [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
  [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
  [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
];

function dot3(g: readonly [number, number, number], x: number, y: number, z: number): number {
  return g[0] * x + g[1] * y + g[2] * z;
}

/**
 * Creates a deterministic 3D simplex noise function seeded from `seed`.
 * Returns a closure that evaluates noise in [-1, 1] with no allocations.
 */
export function createNoise3D(seed: number): (x: number, y: number, z: number) => number {
  // Build permutation table via a seeded shuffle (Fisher-Yates)
  const rng = splitmix32(seed);
  const perm = new Uint8Array(512);
  const base = new Uint8Array(256);

  for (let i = 0; i < 256; i++) base[i] = i;

  // Fisher-Yates shuffle using the PRNG
  for (let i = 255; i > 0; i--) {
    const j = rng() % (i + 1);
    const tmp = base[i];
    base[i] = base[j];
    base[j] = tmp;
  }

  // Double the table to avoid index-wrapping arithmetic in the hot path
  for (let i = 0; i < 512; i++) perm[i] = base[i & 255];

  return function noise3D(x: number, y: number, z: number): number {
    // Skew the input space to determine which simplex cell we're in
    const s = (x + y + z) * F3;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    const k = Math.floor(z + s);

    const t = (i + j + k) * G3;
    // Unskew back to get distances from cell origin
    const X0 = i - t;
    const Y0 = j - t;
    const Z0 = k - t;
    const x0 = x - X0;
    const y0 = y - Y0;
    const z0 = z - Z0;

    // Determine which simplex we are in (6 simplices in 3D cube)
    let i1: number, j1: number, k1: number; // offsets for 2nd corner
    let i2: number, j2: number, k2: number; // offsets for 3rd corner
    if (x0 >= y0) {
      if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
      else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
      else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
      if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
      else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
      else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    // Offsets for corners 2, 3, 4
    const x1 = x0 - i1 + G3;
    const y1 = y0 - j1 + G3;
    const z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2.0 * G3;
    const y2 = y0 - j2 + 2.0 * G3;
    const z2 = z0 - k2 + 2.0 * G3;
    const x3 = x0 - 1.0 + 3.0 * G3;
    const y3 = y0 - 1.0 + 3.0 * G3;
    const z3 = z0 - 1.0 + 3.0 * G3;

    // Hash corner indices, looking up into perm table (doubled, so & 255 not needed)
    const ii = i & 255;
    const jj = j & 255;
    const kk = k & 255;

    const gi0 = perm[ii +     perm[jj +     perm[kk     ]]] % 12;
    const gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
    const gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
    const gi3 = perm[ii + 1  + perm[jj + 1  + perm[kk + 1 ]]] % 12;

    // Contribution from each corner
    let n0 = 0.0, n1 = 0.0, n2 = 0.0, n3 = 0.0;

    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 > 0) { t0 *= t0; n0 = t0 * t0 * dot3(GRAD3[gi0], x0, y0, z0); }

    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 > 0) { t1 *= t1; n1 = t1 * t1 * dot3(GRAD3[gi1], x1, y1, z1); }

    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 > 0) { t2 *= t2; n2 = t2 * t2 * dot3(GRAD3[gi2], x2, y2, z2); }

    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 > 0) { t3 *= t3; n3 = t3 * t3 * dot3(GRAD3[gi3], x3, y3, z3); }

    // Scale to [-1, 1]. Factor 32 from Gustavson's analysis.
    return 32.0 * (n0 + n1 + n2 + n3);
  };
}

// ---------------------------------------------------------------------------
// FBM options
// ---------------------------------------------------------------------------

export interface FbmOptions {
  octaves?:    number; // default 6
  frequency?:  number; // default 1
  lacunarity?: number; // default 2
  gain?:       number; // default 0.5
}

/**
 * Fractional Brownian Motion over a noise function.
 * Output ≈ [-1, 1] (normalized by the geometric series of amplitudes).
 */
export function fbm(
  noise: (x: number, y: number, z: number) => number,
  x: number,
  y: number,
  z: number,
  opts?: FbmOptions,
): number {
  const octaves    = opts?.octaves    ?? 6;
  const frequency  = opts?.frequency  ?? 1;
  const lacunarity = opts?.lacunarity ?? 2;
  const gain       = opts?.gain       ?? 0.5;

  let value     = 0.0;
  let amplitude = 1.0;
  let freq      = frequency;
  // Normaliser: sum of geometric series Σ gain^i for i=0..octaves-1
  let maxAmp    = 0.0;

  for (let i = 0; i < octaves; i++) {
    value  += amplitude * noise(x * freq, y * freq, z * freq);
    maxAmp += amplitude;
    amplitude *= gain;
    freq      *= lacunarity;
  }

  return value / maxAmp;
}


/**
 * Ridged multifractal variant.
 * Per octave: ridge = (1 − |n|)², accumulated with amplitude weighting.
 * Output ≈ [0, 1].
 */
export function ridged(
  noise: (x: number, y: number, z: number) => number,
  x: number,
  y: number,
  z: number,
  opts?: FbmOptions,
): number {
  const octaves    = opts?.octaves    ?? 6;
  const frequency  = opts?.frequency  ?? 1;
  const lacunarity = opts?.lacunarity ?? 2;
  const gain       = opts?.gain       ?? 0.5;

  let value     = 0.0;
  let amplitude = 1.0;
  let freq      = frequency;
  let maxAmp    = 0.0;

  for (let i = 0; i < octaves; i++) {
    const n     = noise(x * freq, y * freq, z * freq);
    const ridge = (1.0 - Math.abs(n)) ** 2; // squared for sharper ridges
    value  += amplitude * ridge;
    maxAmp += amplitude;
    amplitude *= gain;
    freq      *= lacunarity;
  }

  return value / maxAmp;
}
