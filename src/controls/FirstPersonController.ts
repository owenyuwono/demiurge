import { PerspectiveCamera, Vector3 } from 'three'
import type { SurfaceSampler, SurfacePlacement } from './NavMode.js'

// ---------------------------------------------------------------------------
// Zero-allocation scratch — module-level Vector3s reused every update() call.
// No `new` inside hot paths. Same pattern as GlobeControls.
// ---------------------------------------------------------------------------
const _s = {
  up:      new Vector3(), // current radial up (scratch)
  fwd:     new Vector3(), // forward scratch
  right:   new Vector3(), // right tangent scratch
  move:    new Vector3(), // movement vector scratch
  camPos:  new Vector3(), // camera position scratch
  lookDir: new Vector3(), // look direction (fwd tilted by pitch)
  tmp:     new Vector3(), // generic scratch
}

const _WORLD_X = new Vector3(1, 0, 0)
const _WORLD_Z = new Vector3(0, 0, 1)

// Clamp pitch to ±85° — never let the view flip upside-down
const PITCH_LIMIT = (85 * Math.PI) / 180

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface FirstPersonOptions {
  sampler:           SurfaceSampler
  eyeHeight?:        number   // metres above feet; default 1.7
  moveSpeed?:        number   // world units/sec; default 10
  runMultiplier?:    number   // shift multiplier; default 5
  lookSensitivity?:  number   // rad/pixel; default 0.0022
}

export class FirstPersonController {
  // ---- Public contract -------------------------------------------------------
  readonly position: Vector3   // world position of player feet
  get isLocked(): boolean { return this._isLocked }

  // ---- Private state ---------------------------------------------------------
  private readonly _camera:          PerspectiveCamera
  private readonly _dom:             HTMLElement
  private readonly _sampler:         SurfaceSampler
  private readonly _eyeHeight:       number
  private readonly _moveSpeed:       number
  private readonly _runMultiplier:   number
  private readonly _lookSensitivity: number

  // Local tangent-plane frame. _forward ⊥ _up, both normalized.
  // These are the source of truth for orientation; camera is derived each frame.
  private _up:      Vector3
  private _forward: Vector3

  // Pitch angle relative to the horizon (radians, clamped to ±PITCH_LIMIT).
  // Yaw is stored implicitly in _forward (rotated in-place each frame).
  private _pitch = 0

  // Accumulated mouse deltas — consumed and zeroed at the top of each update()
  private _dMouseX = 0
  private _dMouseY = 0

  // Keyboard state
  private readonly _keys  = new Set<string>()
  private _shiftHeld = false

  // Pointer lock
  private _isLocked   = false
  private _isEnabled  = false
  private _isDisposed = false

  // Bound listener refs (stored so removeEventListener gets the exact same fn)
  private readonly _onKeyDown:            (e: KeyboardEvent) => void
  private readonly _onKeyUp:              (e: KeyboardEvent) => void
  private readonly _onClick:              ()                 => void
  private readonly _onMouseMove:          (e: MouseEvent)    => void
  private readonly _onPointerLockChange:  ()                 => void

  // ---------------------------------------------------------------------------
  constructor(
    camera: PerspectiveCamera,
    dom:    HTMLElement,
    opts:   FirstPersonOptions,
  ) {
    this._camera          = camera
    this._dom             = dom
    this._sampler         = opts.sampler
    this._eyeHeight       = opts.eyeHeight       ?? 1.7
    this._moveSpeed       = opts.moveSpeed        ?? 10
    this._runMultiplier   = opts.runMultiplier    ?? 5
    this._lookSensitivity = opts.lookSensitivity  ?? 0.0022

    // One-time allocations for persistent state
    this.position = new Vector3()
    this._up      = new Vector3(0, 1, 0)
    this._forward = new Vector3(0, 0, -1)

    // Bind listener refs
    this._onKeyDown           = this._handleKeyDown.bind(this)
    this._onKeyUp             = this._handleKeyUp.bind(this)
    this._onClick             = this._handleClick.bind(this)
    this._onMouseMove         = this._handleMouseMove.bind(this)
    this._onPointerLockChange = this._handlePointerLockChange.bind(this)
  }

  // ---------------------------------------------------------------------------
  // spawn — place + orient; does NOT lock pointer
  // ---------------------------------------------------------------------------
  spawn(placement: SurfacePlacement): void {
    this._up.copy(placement.up).normalize()
    // Snap feet to the sampler-authoritative surface radius (same formula as update()).
    // This guarantees spawn and per-frame re-projection are consistent even if
    // placement.position has minor float drift from the analytic sphere intersection.
    const spawnSurfR = this._sampler.surfaceRadiusAt(this._up)
    this.position.copy(this._up).multiplyScalar(spawnSurfR)

    // Reset pitch; yaw is implicit in _forward (reset by building a fresh tangent)
    this._pitch = 0

    // Build initial forward: project +Z onto the tangent plane. Fall back to +X
    // if +Z is nearly parallel to up (player spawned at a pole).
    _s.tmp.copy(_WORLD_Z)
    const dotZ = _s.tmp.dot(this._up)
    _s.fwd.copy(_s.tmp).addScaledVector(this._up, -dotZ)
    if (_s.fwd.lengthSq() < 1e-8) {
      _s.tmp.copy(_WORLD_X)
      const dotX = _s.tmp.dot(this._up)
      _s.fwd.copy(_s.tmp).addScaledVector(this._up, -dotX)
    }
    _s.fwd.normalize()
    this._forward.copy(_s.fwd)

    // Set camera: eye sits at eyeHeight above feet along radial up.
    // camera radial distance = surfaceRadiusAt(up) + eyeHeight
    _s.camPos.copy(this._up).multiplyScalar(spawnSurfR + this._eyeHeight)
    this._camera.up.copy(this._up)
    this._camera.position.copy(_s.camPos)

    // Look straight ahead (pitch=0, so lookAt target is camPos + forward)
    _s.tmp.copy(_s.camPos).addScaledVector(this._forward, 1)
    this._camera.lookAt(_s.tmp)
  }

  // ---------------------------------------------------------------------------
  // enable — attach listeners; pointer-lock engages on next dom click
  // ---------------------------------------------------------------------------
  enable(): void {
    if (this._isDisposed || this._isEnabled) return
    this._isEnabled = true

    this._dom.addEventListener('click',             this._onClick)
    document.addEventListener('mousemove',          this._onMouseMove)
    document.addEventListener('pointerlockchange',  this._onPointerLockChange)
    window.addEventListener('keydown', this._onKeyDown)
    window.addEventListener('keyup',   this._onKeyUp)
  }

  // ---------------------------------------------------------------------------
  // disable — detach listeners; exit pointer lock
  // ---------------------------------------------------------------------------
  disable(): void {
    if (!this._isEnabled) return
    this._isEnabled = false

    this._dom.removeEventListener('click',             this._onClick)
    document.removeEventListener('mousemove',          this._onMouseMove)
    document.removeEventListener('pointerlockchange',  this._onPointerLockChange)
    window.removeEventListener('keydown', this._onKeyDown)
    window.removeEventListener('keyup',   this._onKeyUp)

    if (this._isLocked) document.exitPointerLock()

    // Clear transient input state so stale keys don't fire after re-enable
    this._keys.clear()
    this._shiftHeld = false
    this._dMouseX   = 0
    this._dMouseY   = 0
  }

  // ---------------------------------------------------------------------------
  // update — per-frame: look, WASD, re-project to surface. dt in seconds.
  // Zero allocations — all intermediates use module-level _s.* scratch.
  // ---------------------------------------------------------------------------
  update(dt: number): void {
    if (this._isDisposed) return
    const safeDt = dt > 0 ? dt : 1e-4

    // ---- 1. Consume accumulated mouse deltas -----------------------------------
    //    Capture then zero first so handler can accumulate while we compute.
    const dMouseX = this._dMouseX
    const dMouseY = this._dMouseY
    this._dMouseX = 0
    this._dMouseY = 0

    // ---- 2. Apply yaw delta: rotate _forward about _up ------------------------
    //    Yaw is stored implicitly in _forward. We rotate it in the tangent plane
    //    by the signed angle dYaw = dMouseX * sensitivity. right = forward × up,
    //    so a positive dYaw rotates _forward toward _right — i.e. moving the mouse
    //    right (dMouseX > 0) turns the view right, matching standard FPS feel.
    const dYaw = dMouseX * this._lookSensitivity
    if (dYaw !== 0) {
      // right = forward × up; rotating forward toward right is positive yaw
      _s.right.crossVectors(this._forward, this._up).normalize()
      const c = Math.cos(dYaw)
      const s = Math.sin(dYaw)
      // new_forward = c*forward + s*right
      _s.fwd
        .copy(this._forward).multiplyScalar(c)
        .addScaledVector(_s.right, s)
      this._forward.copy(_s.fwd).normalize()
    }

    // ---- 3. Apply pitch delta (clamped to ±PITCH_LIMIT) -----------------------
    this._pitch -= dMouseY * this._lookSensitivity
    this._pitch  = Math.min(Math.max(this._pitch, -PITCH_LIMIT), PITCH_LIMIT)

    // ---- 4. WASD movement in the tangent plane --------------------------------
    const fwdInput    = (this._keys.has('KeyW') ? 1 : 0) - (this._keys.has('KeyS') ? 1 : 0)
    const strafeInput = (this._keys.has('KeyD') ? 1 : 0) - (this._keys.has('KeyA') ? 1 : 0)

    if (fwdInput !== 0 || strafeInput !== 0) {
      const speed = this._moveSpeed * (this._shiftHeld ? this._runMultiplier : 1)

      // right = forward × up (fresh after possible yaw rotation)
      _s.right.crossVectors(this._forward, this._up).normalize()

      _s.move
        .copy(this._forward).multiplyScalar(fwdInput)
        .addScaledVector(_s.right, strafeInput)

      // Non-zero check guards the normalize; diagonals are normalized to 1
      if (_s.move.lengthSq() > 0) {
        _s.move.normalize()
        this.position.addScaledVector(_s.move, speed * safeDt)
      }
    }

    // ---- 5. Re-project feet to surface ----------------------------------------
    //    Critical: keeps the player glued to the sphere regardless of float error
    //    accumulation in position. Compute radial direction, query sampler, snap.
    _s.up.copy(this.position).normalize()
    const surfR = this._sampler.surfaceRadiusAt(_s.up)
    this.position.copy(_s.up).multiplyScalar(surfR)
    this._up.copy(_s.up)   // update stored radial up to match new position

    // Camera eye position
    _s.camPos.copy(_s.up).multiplyScalar(surfR + this._eyeHeight)
    this._camera.position.copy(_s.camPos)

    // ---- 6. Re-orthonormalize _forward against the new _up --------------------
    //    As the player walks around the planet's curvature, _up rotates so
    //    _forward (which was ⊥ the old _up) gradually gains an up-component.
    //    Project it out each frame to keep _forward in the tangent plane.
    const dot = this._forward.dot(this._up)
    _s.fwd.copy(this._forward).addScaledVector(this._up, -dot)
    if (_s.fwd.lengthSq() > 1e-10) {
      this._forward.copy(_s.fwd).normalize()
    }
    // If lengthSq is nearly 0, _forward collapsed to parallel with _up (extreme
    // edge case). Keep it as-is; the next mouse input will steer out of it.

    // ---- 7. Derive look direction: _forward tilted by _pitch about right ------
    _s.right.crossVectors(this._forward, this._up).normalize()
    // lookDir = cos(pitch)*forward + sin(pitch)*up
    // positive pitch = looking up; right-hand rule about _right axis
    const cp = Math.cos(this._pitch)
    const sp = Math.sin(this._pitch)
    _s.lookDir
      .copy(this._forward).multiplyScalar(cp)
      .addScaledVector(this._up, sp)

    // ---- 8. Apply to camera ---------------------------------------------------
    this._camera.up.copy(this._up)
    _s.tmp.copy(_s.camPos).addScaledVector(_s.lookDir, 1)
    this._camera.lookAt(_s.tmp)
  }

  // ---------------------------------------------------------------------------
  // dispose — full teardown; subsequent update() calls are safe no-ops
  // ---------------------------------------------------------------------------
  dispose(): void {
    if (this._isDisposed) return
    this._isDisposed = true
    this.disable()
  }

  // ---------------------------------------------------------------------------
  // Event handlers — zero allocations
  // ---------------------------------------------------------------------------

  private _handleClick(): void {
    if (!this._isEnabled || this._isDisposed) return
    this._dom.requestPointerLock()
  }

  private _handleMouseMove(e: MouseEvent): void {
    // Accumulate only while pointer is locked to this element
    if (!this._isLocked) return
    this._dMouseX += e.movementX
    this._dMouseY += e.movementY
  }

  private _handlePointerLockChange(): void {
    this._isLocked = document.pointerLockElement === this._dom
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
