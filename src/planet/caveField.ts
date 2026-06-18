/**
 * Cave field sampler — procedural 3D SDF-based cave network.
 *
 * All positions are planet-LOCAL 3D Vector3 (|p| ≈ RADIUS + elevation).
 * This module is standalone: no imports from Planet.ts or terrainSampler.ts.
 *
 * Scale math: all tuning constants are authored at RADIUS_REF = 50 000 m and
 * scaled at factory time by scaleFactor = radius / RADIUS_REF.
 *
 * At RADIUS_REF = 50 000 m (scaleFactor = 1):
 *   WORM_FREQ_BASE = 1.1e-3 → period = 1/1.1e-3 ≈ 909 m. Tunnel half-width
 *   ≈ WORM_RADIUS * period / (2π) ≈ 0.28 * 909 / 6.28 ≈ 40.5 m (walkable).
 *   SHAFT_RADIUS_BASE = 3.5 m → ~7 m wide opening.
 *   SHAFT_DEPTH_BASE  = 90 m → well inside the active worm shell.
 *   CAVE_SHELL_MAX_BASE = 350 m → top 350 m of crust, inside ±1200 m envelope.
 *
 * At RADIUS = 500 000 m (scaleFactor = 10):
 *   wormFreq = 1.1e-4 → period ≈ 9090 m (same proportional scale).
 *   shaftRadius ≈ 35 m, shaftDepth = 900 m, caveShellMax = 3500 m.
 *   Caves remain proportionally walkable on the NMS-giant world.
 */

import { Vector3 } from 'three'
import { createNoise3D } from './noise'
import { RADIUS, HEIGHT_SCALE, RADIUS_REF } from './worldConstants'

// ---------------------------------------------------------------------------
// Planet constants — imported from worldConstants (single source of truth)
// ---------------------------------------------------------------------------
// RADIUS, HEIGHT_SCALE, and RADIUS_REF are re-exported from worldConstants.ts;
// scale-dependent cave constants are derived at factory-call time from the
// runtime `radius` parameter so they stay proportionally correct at any scale.

// ---------------------------------------------------------------------------
// Cave tuning constants — scale-independent (dimensionless or absolute metres
// that do NOT depend on planet radius)
// ---------------------------------------------------------------------------

const CAVE_LOD        = 12    // fixed LOD for LOD-invariant cave geometry
const CAVE_SHELL_MIN  = 3     // meters below surface before cave can open
const WORM_RADIUS     = 0.28    // worm threshold (tunnels form where |n| < WORM_RADIUS)
const WORM_WARP_AMP   = 1.2     // domain-warp amplitude in noise-space units (scale-invariant)
const NORMAL_H        = 0.25    // central-difference step for normalAt (meters)
const BIG             = 1e6     // large positive = solid

// ---------------------------------------------------------------------------
// Cave tuning constants at baseline (RADIUS_REF = 50 000 m).
// Frequencies scale inversely with radius so the same physical tunnel size is
// preserved: wormFreq_actual = WORM_FREQ_BASE * (RADIUS_REF / radius).
// Linear dimensions (depth, shell, shaft) scale proportionally with radius.
// ---------------------------------------------------------------------------

const CAVE_SHELL_MAX_BASE = 350   // meters at RADIUS_REF — deepest cave wall
const WORM_FREQ_BASE      = 1.1e-3  // noise frequency at RADIUS_REF
const WORM_WARP_FREQ_BASE = 1.2e-4  // domain-warp frequency at RADIUS_REF
const SHAFT_RADIUS_BASE   = 3.5     // meters at RADIUS_REF — entrance shaft radius
const SHAFT_DEPTH_BASE    = 90      // meters at RADIUS_REF — shaft depth from surface

// ---------------------------------------------------------------------------
// Local deterministic PRNG (copied verbatim from erosion.ts)
// Cave uses streams 50, 51, 52 — clear of tectonics (0..15),
// climate (200), erosion (9, 10).
// ---------------------------------------------------------------------------

function splitmix32Step(a: number): number {
  a |= 0
  a = (a + 0x9e3779b9) | 0
  let t = a ^ (a >>> 16)
  t = Math.imul(t, 0x21f0aaad)
  t = t ^ (t >>> 15)
  t = Math.imul(t, 0x735a2d97)
  return (t ^ (t >>> 15)) >>> 0
}

function deriveSeed(masterSeed: number, stream: number): number {
  const s = (masterSeed ^ Math.imul(stream + 1, 0xdeadbeef)) >>> 0
  return splitmix32Step(s)
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)))
  return t * t * (3 - 2 * t)
}

/**
 * Capsule SDF. Returns negative inside, zero on surface, positive outside.
 * Standard: project p onto segment a→b, measure distance to nearest point,
 * subtract radius.
 */
function sdCapsule(p: Vector3, a: Vector3, b: Vector3, r: number): number {
  const pa = p.clone().sub(a)
  const ba = b.clone().sub(a)
  const h = Math.max(0, Math.min(1, pa.dot(ba) / ba.dot(ba)))
  return pa.clone().sub(ba.clone().multiplyScalar(h)).length() - r
}

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

export interface CaveMouth {
  pos: Vector3          // local-space point on the surface
  up: Vector3           // outward normal (= normalize(pos)), local space
  throatRadius: number  // radius of the entrance shaft in meters
}

export interface CaveSamplerOpts {
  seed: number
  heightFn: (dir: Vector3, level: number) => number
  radius?: number       // default RADIUS (50000)
  heightScale?: number  // default HEIGHT_SCALE (1200)
}

export interface CaveField {
  /** Raw SDF: positive = rock/solid, negative = air/void, 0 = cave wall. */
  densityAt(p: Vector3): number
  /** Cheaper when caller has already computed surfaceR for the sample point. */
  densityWithSurface(p: Vector3, surfaceR: number): number
  /** Surface radius in local space for a given unit-sphere direction. */
  surfaceRadiusLocal(dir: Vector3): number
  /**
   * Gradient of densityAt via central differences.
   * Points from air toward rock (i.e. the cave-wall inward normal from inside).
   * Writes into `out` and returns it.
   */
  normalAt(p: Vector3, out: Vector3): Vector3
  /** Single deterministic cave entrance for this seed. */
  mouth: CaveMouth
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function makeCaveSampler(opts: CaveSamplerOpts): CaveField {
  const { seed, heightFn } = opts
  const radius      = opts.radius      ?? RADIUS
  const heightScale = opts.heightScale ?? HEIGHT_SCALE

  // Scale factor relative to the baseline radius at which the tuning constants
  // were authored.  At RADIUS_REF = 50 000 m, scaleFactor = 1.0 (no change).
  // At RADIUS = 500 000 m, scaleFactor = 10.0 → caves scale proportionally.
  const scaleFactor = radius / RADIUS_REF

  // Scale-dependent constants derived at factory time.
  // Frequencies are INVERSELY proportional to radius so the physical tunnel
  // period (1 / freq) grows with the planet and caves remain walkable.
  // Linear dimensions (depth, shell, shaft) are DIRECTLY proportional.
  const wormFreq      = WORM_FREQ_BASE     / scaleFactor
  const wormWarpFreq  = WORM_WARP_FREQ_BASE / scaleFactor
  const shaftRadius   = SHAFT_RADIUS_BASE  * scaleFactor
  const shaftDepth    = SHAFT_DEPTH_BASE   * scaleFactor
  const caveShellMax  = CAVE_SHELL_MAX_BASE * scaleFactor

  // Create noise instances once — captured in closure, no per-call allocation.
  // Streams 50, 51, 52 stay clear of every other subsystem.
  const tunnelNoise  = createNoise3D(deriveSeed(seed, 50))
  const tunnelNoise2 = createNoise3D(deriveSeed(seed, 51))
  const warpNoise    = createNoise3D(deriveSeed(seed, 52))

  // -------------------------------------------------------------------------
  // Deterministic single mouth position — seeded direction
  // -------------------------------------------------------------------------

  // Derive a deterministic unit direction from the seed (stream 53).
  const ms1 = splitmix32Step(deriveSeed(seed, 53))
  const ms2 = splitmix32Step(ms1)
  const ms3 = splitmix32Step(ms2)
  const mouthDir = new Vector3(
    (ms1 / 0xFFFFFFFF) * 2 - 1,
    (ms2 / 0xFFFFFFFF) * 2 - 1,
    (ms3 / 0xFFFFFFFF) * 2 - 1,
  ).normalize()

  const mouthSurfR = radius + heightFn(mouthDir, CAVE_LOD) * heightScale
  const mouthPos   = mouthDir.clone().multiplyScalar(mouthSurfR)

  const mouth: CaveMouth = { pos: mouthPos, up: mouthDir.clone(), throatRadius: shaftRadius }

  // Precomputed shaft endpoints for the single mouth.
  const shaftTop    = mouthPos.clone()
  const shaftBottom = mouthPos.clone().sub(mouthDir.clone().multiplyScalar(shaftDepth))

  // -------------------------------------------------------------------------
  // Core field methods
  // -------------------------------------------------------------------------

  function surfaceRadiusLocal(dir: Vector3): number {
    return radius + heightFn(dir, CAVE_LOD) * heightScale
  }

  function densityWithSurface(p: Vector3, surfaceR: number): number {
    const r = p.length()
    const depthBelow = surfaceR - r

    // Shaft SDF must be evaluated BEFORE the above-surface early return so the
    // entrance opening is never capped solid.  Negative inside the shaft = air.
    const shaftSD = sdCapsule(p, shaftTop, shaftBottom, shaftRadius)

    // Above/at surface: air only inside the entrance shaft, solid everywhere else.
    // This punches the shaft opening through the surface without exposing the
    // terrain skin as a cave wall — the mesher sees solid on normal ground.
    if (depthBelow <= 0) return shaftSD < 0 ? shaftSD : BIG

    // Shell gate: smooth ramp in from CAVE_SHELL_MIN, ramp out before caveShellMax.
    // The fade-out band (60 m at baseline) scales with the planet so it stays a
    // visually smooth transition regardless of world size.
    const shellFadeOut = 60 * scaleFactor
    const shellGate =
      smoothstep(0, CAVE_SHELL_MIN, depthBelow) *
      (1 - smoothstep(caveShellMax - shellFadeOut, caveShellMax, depthBelow))

    // Domain-warp the sample point before tunnel noise evaluation.
    const wx = p.x + warpNoise(p.x * wormWarpFreq,       p.y * wormWarpFreq,       p.z * wormWarpFreq      ) * WORM_WARP_AMP
    const wy = p.y + warpNoise(p.y * wormWarpFreq + 1.7, p.z * wormWarpFreq,       p.x * wormWarpFreq      ) * WORM_WARP_AMP
    const wz = p.z + warpNoise(p.z * wormWarpFreq + 3.1, p.x * wormWarpFreq,       p.y * wormWarpFreq      ) * WORM_WARP_AMP

    // Two independent noise fields for worm tunnels.
    const n1 = tunnelNoise (wx * wormFreq,       wy * wormFreq,       wz * wormFreq      )
    const n2 = tunnelNoise2(wx * wormFreq + 5.3, wy * wormFreq + 2.7, wz * wormFreq + 8.1)

    // Worm SDF: negative inside the tunnel, positive outside.
    // max(|n1|, |n2|) < WORM_RADIUS → inside a tunnel (Chebyshev-like on noise).
    const wormSD = Math.max(Math.abs(n1), Math.abs(n2)) - WORM_RADIUS

    // Compose:
    //   - Worm is only carved within the valid depth shell.
    //   - Shaft cuts from surface to full shaftDepth WITHOUT the shellGate —
    //     it carves an opening even right at the surface where shellGate ≈ 0.
    const gatedWorm = shellGate > 0 ? wormSD : BIG
    return Math.min(gatedWorm, shaftSD)
  }

  function densityAt(p: Vector3): number {
    const dir    = p.clone().normalize()
    const surfR  = surfaceRadiusLocal(dir)
    return densityWithSurface(p, surfR)
  }

  function normalAt(p: Vector3, out: Vector3): Vector3 {
    const h = NORMAL_H

    const dx =
      densityAt(p.clone().add(new Vector3(h, 0, 0))) -
      densityAt(p.clone().sub(new Vector3(h, 0, 0)))
    const dy =
      densityAt(p.clone().add(new Vector3(0, h, 0))) -
      densityAt(p.clone().sub(new Vector3(0, h, 0)))
    const dz =
      densityAt(p.clone().add(new Vector3(0, 0, h))) -
      densityAt(p.clone().sub(new Vector3(0, 0, h)))

    out.set(dx, dy, dz)

    if (out.length() < 1e-9) {
      // Fallback: radial outward (won't happen in practice at valid cave walls).
      out.copy(p).normalize()
    } else {
      out.normalize()
    }

    return out
  }

  return {
    densityAt,
    densityWithSurface,
    surfaceRadiusLocal,
    normalAt,
    mouth,
  }
}
