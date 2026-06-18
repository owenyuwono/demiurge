/**
 * worldConstants.ts — single knob for planet scale.
 *
 * RADIUS is the one value to change when scaling the world.
 * All downstream params (camera.far, maxAltitude, water shell, etc.)
 * derive from RADIUS automatically — do NOT hardcode RADIUS elsewhere.
 *
 * RADIUS_REF / RES_REF are the baseline values that legacy slope thresholds
 * and tuning constants were authored at. Use deriveSlopeThresh() to rescale
 * any per-cell slope threshold to remain physically equivalent at the actual
 * (RADIUS, erosionRes) operating point.
 */

/** Planet radius in metres. 500 km = NMS-giant scale (was 50 km). */
export const RADIUS = 500_000

/**
 * Terrain relief amplitude in metres.
 * Set to HEIGHT_SCALE_REF × (RADIUS / RADIUS_REF) = 12 000 m → relief ratio
 * = 2.4 % of RADIUS, identical to the original 50 km planet's dramatic look.
 * Must scale proportionally with RADIUS to keep uniform-scale invariance.
 */
export const HEIGHT_SCALE = 12_000

/** Baseline radius (metres) that the legacy tuning constants were authored at. */
export const RADIUS_REF = 50_000

/**
 * Baseline relief amplitude (metres) at RADIUS_REF.
 * Used in deriveSlopeThresh to cancel out the HEIGHT_SCALE factor so that
 * the physical slope angle a threshold represents is invariant under a
 * uniform scale (RADIUS×k, HEIGHT_SCALE×k for any k).
 */
export const HEIGHT_SCALE_REF = 1_200

/** Baseline erosion-grid resolution that legacy slope thresholds were authored at. */
export const RES_REF = 256

/**
 * Rescale a slope threshold from the (RADIUS_REF, HEIGHT_SCALE_REF, RES_REF)
 * baseline to the actual (radius, heightScale, res) operating point so that
 * the same physical slope angle maps to the same threshold value regardless of
 * grid resolution, planet size, or relief amplitude.
 *
 * The erosion grid uses NORMALIZED heights in [0,1] (i.e. metres / HEIGHT_SCALE).
 * For a fixed physical slope angle θ the per-cell normalized rise is:
 *
 *   ΔH_normalized = tan(θ) × cell_arc_metres / HEIGHT_SCALE
 *
 * Cell arc length scales as (radius / res), so:
 *
 *   ΔH_normalized ∝ radius / (res × HEIGHT_SCALE)
 *
 * To keep the same physical angle at the new (radius, heightScale, res), the
 * threshold must scale by (radius / RADIUS_REF) × (HEIGHT_SCALE_REF / heightScale)
 * × (RES_REF / res).  Under a uniform scale (radius×k, heightScale×k) the two
 * radius/HEIGHT_SCALE factors cancel, yielding the original threshold — invariant.
 *
 *   scaledThresh = baseThresh
 *                × (radius       / RADIUS_REF      )   ← wider cells → smaller slope
 *                × (HEIGHT_SCALE_REF / heightScale  )   ← taller relief → larger slope number
 *                × (RES_REF      / res              )   ← finer grid → larger slope number
 *
 * Numeric check at uniform-scaled config (RADIUS=500000, HEIGHT_SCALE=12000, res=256):
 *   deriveSlopeThresh(0.002, 500000, 12000, 256)
 *   = 0.002 × (500000/50000) × (1200/12000) × (256/256)
 *   = 0.002 × 10 × 0.1 × 1 = 0.002  ✓ (same as the 50km/1200 baseline)
 *
 * @param baseThresh   Threshold value as authored at (RADIUS_REF, HEIGHT_SCALE_REF, RES_REF).
 * @param radius       Actual planet radius in metres.
 * @param heightScale  Actual terrain relief amplitude in metres.
 * @param res          Actual erosion-grid resolution (cells per side).
 * @returns            Threshold rescaled to be physically equivalent.
 */
export function deriveSlopeThresh(
  baseThresh:  number,
  radius:      number,
  heightScale: number,
  res:         number,
): number {
  return baseThresh * (radius / RADIUS_REF) * (HEIGHT_SCALE_REF / heightScale) * (RES_REF / res)
}

/**
 * Return the erosion-bake grid resolution for the given planet radius.
 *
 * The resolution is KEPT FIXED at 256 regardless of radius — bake cost scales as
 * res⁴ (grid area × iteration count via flow-routing), so doubling res to 512
 * already costs ~4× and is user-opt-in via the GUI slider.  Growing the grid with
 * radius would make large planets freeze the main thread before the first frame.
 *
 * The trade-off: erosion cells are physically coarser on bigger planets (each cell
 * covers a larger arc), giving smoother river networks at global scale.  The user
 * can raise resolution via the "bake res" GUI slider when they want crisp detail.
 *
 * Return type is the union {128|256|512} so callers are constrained to valid slider
 * values and the type flows cleanly into Planet.erosionRes.
 */
export function deriveErosionRes(_radius: number): 128 | 256 | 512 {
  // Fixed at 256 — the proven ~1.5 s budget at RADIUS_REF=50 000.
  // At RADIUS=500 000 (10× bigger) the cell arc is 10× coarser but cost is unchanged.
  return 256
}
