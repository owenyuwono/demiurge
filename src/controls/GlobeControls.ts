import { PerspectiveCamera, Quaternion, Vector3 } from 'three'

// ---------------------------------------------------------------------------
// Zero-allocation scratch — one set of Vector3s preallocated at module level,
// reused every update() and every event handler. No `new` inside hot paths.
// ---------------------------------------------------------------------------
const _s = {
  a:    new Vector3(), // polar axis (normalized)
  e1:   new Vector3(), // axis-frame tangent 1
  e2:   new Vector3(), // axis-frame tangent 2
  p:    new Vector3(), // camera position scratch
  xRef: new Vector3(), // world-X reference for e1 construction
  zRef: new Vector3(), // world-Z fallback for e1 construction
  prev: new Vector3(), // previous-frame position for speed
  tmp:  new Vector3(), // generic scratch
}

// Scratch for right-drag free-look — allocated once, reused every frame.
const _fl = {
  q:    new Quaternion(), // scratch quaternion for free-look rotation
  axis: new Vector3(),    // camera right-axis for pitch
}

const _WORLD_X = new Vector3(1, 0, 0)
const _WORLD_Z = new Vector3(0, 0, 1)
const _DEFAULT_AXIS = new Vector3(0, 1, 0)
const _ORIGIN = new Vector3(0, 0, 0)

// Tunables -----------------------------------------------------------------------
const DEFAULT_PLANET_R  = 20_000
const DEFAULT_MIN_ALT   = 500
const DEFAULT_MAX_ALT   = 160_000  // 8 × default radius

const DRAG_RAD_PER_PX  = 0.0035  // base angular sensitivity
const FREELOOK_RAD_PER_PX = 0.003  // right-drag free-look sensitivity
const FREELOOK_PITCH_MAX  = Math.PI * 0.45  // ~81° clamp — keeps world upright
const DIST_FACTOR_MIN  = 0.004
const DIST_FACTOR_MAX  = 1.2

const INERTIA_K  = 4.5  // exponential decay constant for angular velocity coasting
const ZOOM_K     = 7.0  // exponential damp constant for r → targetR
const WHEEL_STEP = 0.13 // |ln(targetAlt multiplier)| per normalized notch

const PHI_MIN = 0.05           // ~2.9° — never cross poles; wider margin keeps lookAt roll stable near them
const PHI_MAX = Math.PI - 0.05

const KEY_ANGULAR_BASE = 0.9   // rad/s at distFactor=1
const KEY_SHIFT_MULT   = 4.0

const SPEED_SMOOTH_K = 6.0     // light smoothing for HUD currentSpeed readout

export class GlobeControls {
  // ---- Public contract -------------------------------------------------------
  readonly currentSpeed: number = 0

  // ---- Private state ---------------------------------------------------------
  private readonly _camera: PerspectiveCamera
  private readonly _dom:    HTMLElement

  private readonly _opts: {
    getAltitude?: (pos: Vector3) => number
    getPolarAxis?: () => Vector3
    radius:       number
    minAltitude:  number
    maxAltitude:  number
  }

  // Spherical coords about the polar axis
  private _r:   number  // distance from world origin
  private _theta: number  // azimuth
  private _phi:   number  // polar angle from axis [PHI_MIN, PHI_MAX]
  private _targetR: number  // smooth zoom target

  // Angular velocity (radians/s) for inertia coasting
  private _velTheta = 0
  private _velPhi   = 0

  // Drag tracking (left button — orbit)
  private _isDragging = false
  private _prevDragX  = 0
  private _prevDragY  = 0
  private _lastDragTime = 0

  // For instantaneous angular velocity estimation during drag
  // Use a short ring: track the last pointer event's delta/dt
  private _rawVelTheta = 0
  private _rawVelPhi   = 0

  // Right-drag free-look state
  private _isRightDragging = false
  private _prevRightDragX  = 0
  private _prevRightDragY  = 0
  // Accumulated yaw (around camera's local up) and pitch (around camera's right axis)
  // offsets applied on top of the base lookAt(_ORIGIN) each frame.
  private _freelookYaw   = 0
  private _freelookPitch = 0

  // Speed smoothing
  private _smoothedSpeed = 0

  // Keys
  private readonly _keys = new Set<string>()
  private _shiftHeld = false

  // Bound listener refs
  private readonly _onMouseDown:    (e: MouseEvent) => void
  private readonly _onMouseMove:    (e: MouseEvent) => void
  private readonly _onMouseUp:      (e: MouseEvent) => void
  private readonly _onWheel:        (e: WheelEvent) => void
  private readonly _onKeyDown:      (e: KeyboardEvent) => void
  private readonly _onKeyUp:        (e: KeyboardEvent) => void
  private readonly _onContextMenu:  (e: MouseEvent) => void

  constructor(
    camera: PerspectiveCamera,
    dom: HTMLElement,
    opts?: {
      getAltitude?: (pos: Vector3) => number
      getPolarAxis?: () => Vector3
      radius?: number
      minAltitude?: number
      maxAltitude?: number
    },
  ) {
    this._camera = camera
    this._dom    = dom
    this._opts   = {
      getAltitude:  opts?.getAltitude,
      getPolarAxis: opts?.getPolarAxis,
      radius:       opts?.radius       ?? DEFAULT_PLANET_R,
      minAltitude:  opts?.minAltitude  ?? DEFAULT_MIN_ALT,
      maxAltitude:  opts?.maxAltitude  ?? DEFAULT_MAX_ALT,
    }

    // ---- Initialize (r, θ, φ) from current camera position so construction
    //      never teleports the camera. ----------------------------------------
    const { r, theta, phi } = this._decompose(camera.position)
    this._r       = r
    this._theta   = theta
    this._phi     = phi
    this._targetR = r

    // ---- Bind listeners -------------------------------------------------------
    this._onMouseDown   = this._handleMouseDown.bind(this)
    this._onMouseMove   = this._handleMouseMove.bind(this)
    this._onMouseUp     = this._handleMouseUp.bind(this)
    this._onWheel       = this._handleWheel.bind(this)
    this._onKeyDown     = this._handleKeyDown.bind(this)
    this._onKeyUp       = this._handleKeyUp.bind(this)
    this._onContextMenu = this._handleContextMenu.bind(this)

    dom.addEventListener('mousedown',   this._onMouseDown)
    dom.addEventListener('wheel',       this._onWheel, { passive: false })
    dom.addEventListener('contextmenu', this._onContextMenu)
    document.addEventListener('mousemove', this._onMouseMove)
    document.addEventListener('mouseup',   this._onMouseUp)
    window.addEventListener('keydown', this._onKeyDown)
    window.addEventListener('keyup',   this._onKeyUp)
  }

  // ---------------------------------------------------------------------------
  // Main update — dt in seconds
  // ---------------------------------------------------------------------------
  update(dt: number): void {
    const cam = this._camera
    const safeDt = dt > 0 ? dt : 1e-4

    // ---- Build axis frame (a, e1, e2) from polar axis ----------------------
    // a = unit polar axis (constant direction per spec)
    const axisRaw = this._opts.getPolarAxis?.() ?? _DEFAULT_AXIS
    _s.a.copy(axisRaw).normalize()

    // e1 = world X projected onto the plane perpendicular to a
    _s.xRef.copy(_WORLD_X)
    const dotX = _s.xRef.dot(_s.a)
    _s.e1.copy(_s.xRef).addScaledVector(_s.a, -dotX)
    if (_s.e1.lengthSq() < 1e-8) {
      // Degenerate: a ≈ world X — fall back to world Z
      _s.zRef.copy(_WORLD_Z)
      const dotZ = _s.zRef.dot(_s.a)
      _s.e1.copy(_s.zRef).addScaledVector(_s.a, -dotZ)
    }
    _s.e1.normalize()
    _s.e2.crossVectors(_s.a, _s.e1) // a × e1, unit if a⊥e1

    // ---- Altitude for distFactor --------------------------------------------
    const rawAlt = this._opts.getAltitude
      ? this._opts.getAltitude(cam.position)
      : (this._r - this._opts.radius)
    const altitude   = Math.max(rawAlt, 1)
    const distFactor = Math.min(Math.max(altitude / this._opts.radius, DIST_FACTOR_MIN), DIST_FACTOR_MAX)

    // ---- Keyboard angular input ---------------------------------------------
    if (!this._isDragging) {
      const keyRate = KEY_ANGULAR_BASE * distFactor * (this._shiftHeld ? KEY_SHIFT_MULT : 1)
      if (this._keys.has('KeyA')) this._velTheta -= keyRate * safeDt
      if (this._keys.has('KeyD')) this._velTheta += keyRate * safeDt
      if (this._keys.has('KeyW')) this._velPhi   -= keyRate * safeDt
      if (this._keys.has('KeyS')) this._velPhi   += keyRate * safeDt

      // Note: keys set velocity directly each frame (no separate target needed);
      // when keys are released the inertia decay below coasts them to zero.
    }

    // ---- Inertia decay (frame-rate independent) -----------------------------
    // While dragging: _velTheta/_velPhi are driven by raw drag events (see
    // _handleMouseMove). When not dragging they decay exponentially.
    if (!this._isDragging) {
      const decay = Math.exp(-INERTIA_K * safeDt)
      this._velTheta *= decay
      this._velPhi   *= decay
    }

    // Apply angular velocity to spherical coords
    this._theta += this._velTheta * safeDt
    this._phi   += this._velPhi   * safeDt
    if (this._phi <= PHI_MIN || this._phi >= PHI_MAX) this._velPhi = 0 // don't push against the pole clamp
    this._phi    = Math.min(Math.max(this._phi, PHI_MIN), PHI_MAX)

    // ---- Smooth zoom (r → targetR) -----------------------------------------
    const zoomDecay = Math.exp(-ZOOM_K * safeDt)
    this._r = this._r * zoomDecay + this._targetR * (1 - zoomDecay)

    // Re-clamp altitude against live terrain so we don't clip mountains
    const liveAlt = this._opts.getAltitude
      ? this._opts.getAltitude(cam.position)
      : (this._r - this._opts.radius)
    if (liveAlt < this._opts.minAltitude) {
      // Push r out to honour minAltitude
      const surfR = this._r - liveAlt            // approx surface radius at camera column
      this._r       = surfR + this._opts.minAltitude
      this._targetR = this._r
    }

    // ---- Recompute camera position from (r, θ, φ) in axis frame -------------
    // p = r · (sinφ·(cosθ·e1 + sinθ·e2) + cosφ·a)
    _s.prev.copy(cam.position)

    const sinPhi = Math.sin(this._phi)
    const cosPhi = Math.cos(this._phi)
    const sinTh  = Math.sin(this._theta)
    const cosTh  = Math.cos(this._theta)

    cam.position
      .copy(_s.e1).multiplyScalar(sinPhi * cosTh)
      .addScaledVector(_s.e2, sinPhi * sinTh)
      .addScaledVector(_s.a,  cosPhi)
      .multiplyScalar(this._r)

    // ---- Orient camera: up = polar axis, look at origin --------------------
    cam.up.copy(_s.a)
    cam.lookAt(_ORIGIN)

    // ---- Free-look: apply right-drag yaw/pitch offset on top of base orientation --
    // Yaw rotates around the camera's local up (which is the polar axis after lookAt).
    // Pitch rotates around the camera's local right axis.
    if (this._freelookYaw !== 0 || this._freelookPitch !== 0) {
      // Build yaw quaternion around the polar axis (camera up after lookAt = _s.a)
      _fl.q.setFromAxisAngle(_s.a, this._freelookYaw)
      cam.quaternion.premultiply(_fl.q)

      // Derive the camera's right axis after the yaw, then apply pitch
      _fl.axis.set(1, 0, 0).applyQuaternion(cam.quaternion)
      _fl.q.setFromAxisAngle(_fl.axis, this._freelookPitch)
      cam.quaternion.premultiply(_fl.q)
    }

    // ---- currentSpeed (surface-tangential, smoothed for HUD) ---------------
    const rawSpeed = safeDt > 0 ? _s.prev.distanceTo(cam.position) / safeDt : 0
    const sDecay   = Math.exp(-SPEED_SMOOTH_K * safeDt)
    this._smoothedSpeed = this._smoothedSpeed * sDecay + rawSpeed * (1 - sDecay)
    // Assign through the cast since the property is `readonly` externally but
    // mutable internally via the same field.
    ;(this as { currentSpeed: number }).currentSpeed = this._smoothedSpeed
  }

  dispose(): void {
    this._dom.removeEventListener('mousedown',   this._onMouseDown)
    this._dom.removeEventListener('wheel',       this._onWheel)
    this._dom.removeEventListener('contextmenu', this._onContextMenu)
    document.removeEventListener('mousemove', this._onMouseMove)
    document.removeEventListener('mouseup',   this._onMouseUp)
    window.removeEventListener('keydown', this._onKeyDown)
    window.removeEventListener('keyup',   this._onKeyUp)
  }

  // ---------------------------------------------------------------------------
  // Decompose a world-space position into (r, θ, φ) against the current axis frame.
  // Called once at construction to initialize state without teleporting.
  // ---------------------------------------------------------------------------
  private _decompose(pos: Vector3): { r: number; theta: number; phi: number } {
    const axisRaw = this._opts.getPolarAxis?.() ?? _DEFAULT_AXIS
    _s.a.copy(axisRaw).normalize()

    _s.xRef.copy(_WORLD_X)
    const dotX = _s.xRef.dot(_s.a)
    _s.e1.copy(_s.xRef).addScaledVector(_s.a, -dotX)
    if (_s.e1.lengthSq() < 1e-8) {
      _s.zRef.copy(_WORLD_Z)
      const dotZ = _s.zRef.dot(_s.a)
      _s.e1.copy(_s.zRef).addScaledVector(_s.a, -dotZ)
    }
    _s.e1.normalize()
    _s.e2.crossVectors(_s.a, _s.e1)

    const r = pos.length()
    if (r < 1e-6) return { r: this._opts.radius + this._opts.minAltitude, theta: 0, phi: Math.PI / 2 }

    _s.p.copy(pos).divideScalar(r)  // unit direction p̂
    const cosPhiVal = Math.min(Math.max(_s.p.dot(_s.a), -1), 1)
    const phi   = Math.acos(cosPhiVal)
    const theta = Math.atan2(_s.p.dot(_s.e2), _s.p.dot(_s.e1))

    return { r, theta, phi: Math.min(Math.max(phi, PHI_MIN), PHI_MAX) }
  }

  // ---------------------------------------------------------------------------
  // Event handlers — zero allocations (all scratch is module-level)
  // ---------------------------------------------------------------------------

  private _handleMouseDown(e: MouseEvent): void {
    if (e.button === 0) {
      this._isDragging  = true
      this._prevDragX   = e.clientX
      this._prevDragY   = e.clientY
      this._lastDragTime = performance.now()
      // Kill inertia when grabbing the globe
      this._velTheta = 0
      this._velPhi   = 0
      this._rawVelTheta = 0
      this._rawVelPhi   = 0
    } else if (e.button === 2) {
      this._isRightDragging = true
      this._prevRightDragX  = e.clientX
      this._prevRightDragY  = e.clientY
    }
  }

  private _handleMouseMove(e: MouseEvent): void {
    if (this._isDragging) {
      const now = performance.now()
      const dtMs = now - this._lastDragTime
      const safeDtMs = dtMs > 0 ? dtMs : 1  // avoid /0

      const dxPx = e.clientX - this._prevDragX
      const dyPx = e.clientY - this._prevDragY
      this._prevDragX   = e.clientX
      this._prevDragY   = e.clientY
      this._lastDragTime = now

      // Altitude for distFactor — use current r; getAltitude may use cam.position
      // which was last set in update(). Approximate is fine here.
      const approxAlt  = this._opts.getAltitude
        ? this._opts.getAltitude(this._camera.position)
        : (this._r - this._opts.radius)
      const distFactor = Math.min(
        Math.max(Math.max(approxAlt, 1) / this._opts.radius, DIST_FACTOR_MIN),
        DIST_FACTOR_MAX,
      )

      // Dragging RIGHT → viewed surface moves right → azimuth DECREASES (Google Earth grab)
      const dTheta = -dxPx * DRAG_RAD_PER_PX * distFactor
      const dPhi   = -dyPx * DRAG_RAD_PER_PX * distFactor

      this._theta += dTheta
      this._phi   += dPhi
      this._phi    = Math.min(Math.max(this._phi, PHI_MIN), PHI_MAX)

      // Track instantaneous angular velocity for inertia hand-off on release
      const dtSec = safeDtMs / 1000
      // Smooth with a short exponential so a single jerky pointer event doesn't
      // dominate; k=12 converges in ~0.1 s, which is shorter than a typical flick.
      const smooth = Math.exp(-12 * dtSec)
      this._rawVelTheta = this._rawVelTheta * smooth + (dTheta / dtSec) * (1 - smooth)
      this._rawVelPhi   = this._rawVelPhi   * smooth + (dPhi   / dtSec) * (1 - smooth)
    }

    if (this._isRightDragging) {
      const dxPx = e.clientX - this._prevRightDragX
      const dyPx = e.clientY - this._prevRightDragY
      this._prevRightDragX = e.clientX
      this._prevRightDragY = e.clientY

      // Right = yaw left; down = pitch down. Negate X for natural look-around feel.
      this._freelookYaw   -= dxPx * FREELOOK_RAD_PER_PX
      this._freelookPitch -= dyPx * FREELOOK_RAD_PER_PX
      this._freelookPitch  = Math.min(Math.max(this._freelookPitch, -FREELOOK_PITCH_MAX), FREELOOK_PITCH_MAX)
    }
  }

  private _handleMouseUp(e: MouseEvent): void {
    if (e.button === 0 && this._isDragging) {
      this._isDragging = false
      // Hand instantaneous velocity off to the inertia coasting system
      this._velTheta = this._rawVelTheta
      this._velPhi   = this._rawVelPhi
      this._rawVelTheta = 0
      this._rawVelPhi   = 0
    } else if (e.button === 2) {
      this._isRightDragging = false
    }
  }

  private _handleContextMenu(e: MouseEvent): void {
    // Suppress the browser context menu so right-drag free-look works uninterrupted.
    e.preventDefault()
  }

  private _handleWheel(e: WheelEvent): void {
    e.preventDefault()

    // Normalise across deltaMode (0=px, 1=line, 2=page) — same convention as FlyCamera
    let delta = e.deltaY
    if (e.deltaMode === 1) delta *= 16
    if (e.deltaMode === 2) delta *= 400

    const notches = delta / 100  // 1 notch ≈ 100 px

    // Current altitude estimate (use last known r)
    const approxAlt = this._opts.getAltitude
      ? this._opts.getAltitude(this._camera.position)
      : (this._r - this._opts.radius)
    const clampedAlt = Math.max(approxAlt, 1)

    // Approximate local surface radius
    const surfR = this._r - clampedAlt

    // Exponential altitude scaling: scroll up (negative notches) → zoom in
    const factor = Math.exp(notches < 0 ? -WHEEL_STEP : WHEEL_STEP)
    const newTargetAlt = Math.min(
      Math.max(clampedAlt * factor, this._opts.minAltitude),
      this._opts.maxAltitude,
    )

    this._targetR = surfR + newTargetAlt
  }

  private _isInputTarget(e: KeyboardEvent): boolean {
    const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase()
    return tag === 'input' || tag === 'textarea'
  }

  private _handleKeyDown(e: KeyboardEvent): void {
    if (e.repeat || this._isInputTarget(e)) return
    this._shiftHeld = e.shiftKey
    switch (e.code) {
      case 'KeyW': case 'KeyS': case 'KeyA': case 'KeyD':
        this._keys.add(e.code)
        break
    }
  }

  private _handleKeyUp(e: KeyboardEvent): void {
    this._shiftHeld = e.shiftKey
    this._keys.delete(e.code)
  }
}
