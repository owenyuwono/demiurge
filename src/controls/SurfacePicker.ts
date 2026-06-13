import { PerspectiveCamera, Raycaster, Vector2, Vector3 } from 'three'
import type { SurfacePlacement, SurfaceSampler } from './NavMode.js'

// ---------------------------------------------------------------------------
// Module-level scratch — reused across calls (SurfacePicker is click-rate, not
// per-frame, but keeping allocations out of the hot path is still good form).
// ---------------------------------------------------------------------------
const _raycaster = new Raycaster()
const _ndc       = new Vector2()
const _hit       = new Vector3()
const _up        = new Vector3()
const _position  = new Vector3()

export interface SurfacePickerOptions {
  sampler: SurfaceSampler
}

/**
 * Resolves a mouse position into a world-space surface placement on the planet.
 *
 * Strategy: analytic ray–sphere intersection against the ideal sphere
 * (radius = sampler.radius, centred at world origin), then snap the near-hit
 * direction to the real terrain via sampler.surfaceRadiusAt. No mesh involved.
 */
export class SurfacePicker {
  private readonly _camera:  PerspectiveCamera
  private readonly _dom:     HTMLElement
  private readonly _sampler: SurfaceSampler

  constructor(
    camera: PerspectiveCamera,
    dom:    HTMLElement,
    opts:   SurfacePickerOptions,
  ) {
    this._camera  = camera
    this._dom     = dom
    this._sampler = opts.sampler
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Convert a MouseEvent into a SurfacePlacement by mapping the cursor to NDC
   * then delegating to pickFromNDC. Returns null when the ray misses the planet.
   */
  pickFromEvent(e: MouseEvent): SurfacePlacement | null {
    const rect  = this._dom.getBoundingClientRect()
    const ndcX  =  ((e.clientX - rect.left)  / rect.width)  * 2 - 1
    const ndcY  = -((e.clientY - rect.top)   / rect.height) * 2 + 1  // Y flipped: CSS↓, NDC↑
    return this.pickFromNDC(ndcX, ndcY)
  }

  /**
   * Cast a ray from NDC coordinates (x ∈ [-1,1], y ∈ [-1,1]) and resolve it
   * to the planet surface.
   *
   * 1. Build ray via THREE.Raycaster (handles projection/view correctly).
   * 2. Analytic ray–sphere with sphere radius = sampler.radius, centre (0,0,0).
   * 3. Take the nearest positive t (near face of sphere).
   * 4. Snap the hit direction to real terrain via sampler.surfaceRadiusAt.
   */
  pickFromNDC(ndcX: number, ndcY: number): SurfacePlacement | null {
    _ndc.set(ndcX, ndcY)
    _raycaster.setFromCamera(_ndc, this._camera)

    const origin = _raycaster.ray.origin
    const dir    = _raycaster.ray.direction  // already normalised by Raycaster

    // ---- Analytic ray–sphere intersection -----------------------------------
    // Ray: P(t) = origin + t·dir
    // Sphere: |P|² = R²
    // Expand: |origin + t·dir|² = R²
    //   t²·(dir·dir) + 2t·(origin·dir) + (origin·origin - R²) = 0
    // Since dir is unit: a=1, b=2·(origin·dir), c=|origin|²-R²

    const R   = this._sampler.radius
    const b   = 2 * origin.dot(dir)
    const c   = origin.dot(origin) - R * R
    const dis = b * b - 4 * c   // discriminant (a=1 so 4ac = 4·1·c)

    if (dis < 0) return null  // ray misses the sphere entirely

    const sqrtDis = Math.sqrt(dis)
    const t0 = (-b - sqrtDis) * 0.5  // near intersection
    const t1 = (-b + sqrtDis) * 0.5  // far  intersection

    // Choose the nearest t that is in front of the ray origin (t > 0).
    // If both are negative the sphere is completely behind the camera.
    let t: number
    if (t0 > 0) {
      t = t0           // normal case: camera outside sphere, take near hit
    } else if (t1 > 0) {
      t = t1           // camera is inside the sphere, take far hit
    } else {
      return null      // both hits behind the ray origin
    }

    // ---- Hit point on ideal sphere ------------------------------------------
    _hit.copy(dir).multiplyScalar(t).add(origin)

    // ---- Radial up direction -------------------------------------------------
    _up.copy(_hit).normalize()

    // ---- Snap to real terrain via sampler -----------------------------------
    const surfaceR = this._sampler.surfaceRadiusAt(_up)
    _position.copy(_up).multiplyScalar(surfaceR)

    // Return new vectors so each result is independent (caller may store them).
    return {
      position: _position.clone(),
      up:       _up.clone(),
    }
  }
}
