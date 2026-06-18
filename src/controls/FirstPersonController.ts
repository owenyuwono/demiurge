import { PerspectiveCamera, Vector3 } from 'three'
import type { SurfaceSampler, SurfacePlacement, CaveCollider } from './NavMode.js'

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
  norm:    new Vector3(), // collision normal scratch
  feet:    new Vector3(), // feet-sphere sample point scratch
  head:    new Vector3(), // head-sphere sample point scratch
  torso:   new Vector3(), // torso-sphere sample point scratch
}

const _WORLD_X = new Vector3(1, 0, 0)
const _WORLD_Z = new Vector3(0, 0, 1)

// Clamp pitch to ±85° — never let the view flip upside-down
const PITCH_LIMIT = (85 * Math.PI) / 180

// ---------------------------------------------------------------------------
// Cave physics constants
// ---------------------------------------------------------------------------

/** Height from feet to top of head, metres. Matches default eyeHeight. */
const PLAYER_HEIGHT = 1.7
/** Radius of the feet collision sphere, metres. */
const FEET_R = 0.35
/** Radius of the head collision sphere, metres. */
const HEAD_R = 0.30
/** Radius of the torso collision sphere for wall slides, metres. */
const TORSO_R = 0.35
/** Gravitational acceleration toward planet centre, m/s². */
const G = 14
/** Maximum displacement per sub-step before sub-stepping kicks in, metres. */
const MAX_STEP_M = 0.3
/** How close to the surface radius (metres above) before snapping to SURFACE mode. */
const SURFACE_SNAP_M = 0.5
/** Small gap added when pushing out of rock, prevents re-penetration next query. */
const PUSHOUT_BIAS = 0.01
/** Hysteresis offset for SURFACE→CAVE transition: this many metres below surface. */
const CAVE_ENTER_DEPTH = 0.2

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface FirstPersonOptions {
  sampler:           SurfaceSampler
  eyeHeight?:        number   // metres above feet; default 1.7
  moveSpeed?:        number   // world units/sec; default 10
  runMultiplier?:    number   // shift multiplier; default 5
  lookSensitivity?:  number   // rad/pixel; default 0.0022
  collider?:         CaveCollider  // optional cave SDF; if absent, SURFACE only
}

export class FirstPersonController {
  // ---- Public contract -------------------------------------------------------
  readonly position: Vector3   // world position of player feet
  get isLocked(): boolean { return this._isLocked }
  get mode(): 'surface' | 'cave' { return this._mode }

  // ---- Private state ---------------------------------------------------------
  private readonly _camera:          PerspectiveCamera
  private readonly _dom:             HTMLElement
  private readonly _sampler:         SurfaceSampler
  private readonly _collider:        CaveCollider | undefined
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

  // Cave locomotion state
  private _mode: 'surface' | 'cave' = 'surface'
  private _vy = 0  // velocity along radial up, m/s (positive = outward)

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
    this._collider        = opts.collider
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
  // setCollider — inject a cave collider after construction
  // ---------------------------------------------------------------------------
  setCollider(collider: CaveCollider): void {
    // TypeScript: _collider is readonly but we need to allow a post-construction
    // injection for the cave-manager wiring in main.ts. Cast through unknown.
    ;(this as unknown as { _collider: CaveCollider })._collider = collider
  }

  // ---------------------------------------------------------------------------
  // setMode — externally request a mode switch (e.g. entering a cave mouth)
  // ---------------------------------------------------------------------------
  setMode(m: 'surface' | 'cave'): void {
    if (m === this._mode) return
    if (m === 'cave' && !this._collider) return  // no-op: can't enter cave without collider
    this._mode = m
    if (m === 'cave') {
      this._vy = 0  // start with zero vertical velocity
    }
  }

  // ---------------------------------------------------------------------------
  // spawn — place + orient; does NOT lock pointer
  // ---------------------------------------------------------------------------
  spawn(placement: SurfacePlacement, mode?: 'surface' | 'cave'): void {
    this._up.copy(placement.up).normalize()
    // Snap feet to the sampler-authoritative surface radius (same formula as update()).
    // This guarantees spawn and per-frame re-projection are consistent even if
    // placement.position has minor float drift from the analytic sphere intersection.
    const spawnSurfR = this._sampler.surfaceRadiusAt(this._up)
    this.position.copy(this._up).multiplyScalar(spawnSurfR)

    // Reset pitch; yaw is implicit in _forward (reset by building a fresh tangent)
    this._pitch = 0
    this._vy    = 0

    // Apply requested mode (default: surface; cave requires collider)
    if (mode !== undefined) {
      this._mode = (mode === 'cave' && this._collider) ? 'cave' : 'surface'
    } else {
      this._mode = 'surface'
    }

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
    //    Build the normalized move direction and compute per-frame speed.
    //    For SURFACE mode the displacement is applied immediately (original behavior).
    //    For CAVE mode the displacement is deferred into the sub-step loop below so
    //    both horizontal and vertical motion share the same tunneling guard.
    const fwdInput    = (this._keys.has('KeyW') ? 1 : 0) - (this._keys.has('KeyS') ? 1 : 0)
    const strafeInput = (this._keys.has('KeyD') ? 1 : 0) - (this._keys.has('KeyA') ? 1 : 0)

    const speed = this._moveSpeed * (this._shiftHeld ? this._runMultiplier : 1)

    if (fwdInput !== 0 || strafeInput !== 0) {
      // right = forward × up (fresh after possible yaw rotation)
      _s.right.crossVectors(this._forward, this._up).normalize()

      _s.move
        .copy(this._forward).multiplyScalar(fwdInput)
        .addScaledVector(_s.right, strafeInput)

      // Non-zero check guards the normalize; diagonals are normalized to 1
      if (_s.move.lengthSq() > 0) {
        _s.move.normalize()
      }
    } else {
      _s.move.set(0, 0, 0)
    }

    // ---- 5. Mode-dependent vertical resolution --------------------------------
    if (this._mode === 'surface' || !this._collider) {
      // ---- SURFACE mode: pin feet to terrain height (original behavior) --------
      // Apply horizontal move in one shot (no sub-stepping needed on the surface).
      if (_s.move.lengthSq() > 0) {
        this.position.addScaledVector(_s.move, speed * safeDt)
      }

      _s.up.copy(this.position).normalize()
      const surfR = this._sampler.surfaceRadiusAt(_s.up)
      this.position.copy(_s.up).multiplyScalar(surfR)
      this._up.copy(_s.up)

      // Camera eye position
      _s.camPos.copy(_s.up).multiplyScalar(surfR + this._eyeHeight)
      this._camera.position.copy(_s.camPos)

      // Auto-transition SURFACE→CAVE when the surface-pinned feet are inside void
      if (this._collider) {
        const density = this._collider.densityAt(this.position)
        // density ≤ 0 means the surface-pinned spot is in a void (cave void
        // extends above the terrain surface, so the player is at the cave mouth).
        // Apply hysteresis: only enter cave if we're in void by at least CAVE_ENTER_DEPTH.
        if (density <= -CAVE_ENTER_DEPTH) {
          this._mode = 'cave'
          this._vy   = 0
        }
      }
    } else {
      // ---- CAVE mode: gravity + SDF pushout ------------------------------------
      _s.up.copy(this.position).normalize()
      this._up.copy(_s.up)

      // Compute total displacement magnitude this frame for sub-step guard.
      // horzDisp uses the actual intended move distance (0 when no input).
      const horzDisp = _s.move.lengthSq() > 0 ? speed * safeDt : 0
      const gravDisp = Math.abs(this._vy * safeDt) + 0.5 * G * safeDt * safeDt
      const totalDisp = horzDisp + gravDisp
      const nSubSteps = Math.max(1, Math.ceil(totalDisp / MAX_STEP_M))
      const subDt = safeDt / nSubSteps
      // Per-step horizontal advance (metres along _s.move); 0 when no WASD input.
      const horzStep = horzDisp / nSubSteps

      for (let step = 0; step < nSubSteps; step++) {
        // 5a. Horizontal WASD: advance a fraction of the full-frame displacement.
        //     Doing this inside the loop is the tunneling fix — every sub-step
        //     advances at most MAX_STEP_M total, so fast horizontal moves are
        //     broken up the same way gravity steps are.
        if (horzStep > 0) {
          this.position.addScaledVector(_s.move, horzStep)
        }

        // 5b. Gravity: integrate vertical velocity and apply radial displacement
        this._vy -= G * subDt
        this.position.addScaledVector(this._up, this._vy * subDt)

        // 5c. Floor collision: feet sphere at `position + up * FEET_R`
        //     Iterate 2-3 times for robust convergence
        let grounded = false
        for (let iter = 0; iter < 3; iter++) {
          _s.feet.copy(this.position).addScaledVector(this._up, FEET_R)
          const fd = this._collider.densityAt(_s.feet)
          // Positive density means we're inside rock. Push out by (fd + bias) along outward normal.
          if (fd > 0) {
            this._collider.normalAt(_s.feet, _s.norm)
            const pushDist = fd + PUSHOUT_BIAS
            this.position.addScaledVector(_s.norm, pushDist)
            // If the normal has a positive up-component, correction moved us upward → grounded
            if (_s.norm.dot(this._up) > 0) {
              grounded = true
            }
          } else {
            break  // not penetrating, stop iteration
          }
        }
        if (grounded) {
          this._vy = 0
          // Recompute up after positional correction
          _s.up.copy(this.position).normalize()
          this._up.copy(_s.up)
        }

        // 5d. Ceiling collision: head sphere at `position + up * PLAYER_HEIGHT`
        _s.head.copy(this.position).addScaledVector(this._up, PLAYER_HEIGHT - HEAD_R)
        const cd = this._collider.densityAt(_s.head)
        if (cd > 0) {
          this._collider.normalAt(_s.head, _s.norm)
          this.position.addScaledVector(_s.norm, cd + PUSHOUT_BIAS)
          // Clamp upward velocity if ceiling pushes us down (normal has negative up component)
          if (_s.norm.dot(this._up) < 0 && this._vy > 0) {
            this._vy = 0
          }
          // Recompute up
          _s.up.copy(this.position).normalize()
          this._up.copy(_s.up)
        }

        // 5e. Wall collision: torso sphere along horizontal move direction.
        //     Only check if there was horizontal input this frame.
        if (_s.move.lengthSq() > 0) {
          _s.torso.copy(this.position)
            .addScaledVector(this._up, 1.0)       // mid-torso height
            .addScaledVector(_s.move, TORSO_R)    // displaced toward intended move dir
          const wd = this._collider.densityAt(_s.torso)
          if (wd > 0) {
            this._collider.normalAt(_s.torso, _s.norm)
            // Slide: project the horizontal move out of the wall normal
            // move -= norm * dot(move, norm)
            const proj = _s.move.dot(_s.norm)
            if (proj < 0) {  // only cancel the into-wall component
              _s.move.addScaledVector(_s.norm, -proj)
              // Re-normalize if still non-zero so slide speed is consistent
              if (_s.move.lengthSq() > 1e-10) _s.move.normalize()
            }
            // Push position out of wall as well
            this.position.addScaledVector(_s.norm, wd + PUSHOUT_BIAS)
          }
        }
      }

      // 5f. Camera eye for cave mode: feet + up * eyeHeight
      _s.up.copy(this.position).normalize()
      this._up.copy(_s.up)
      _s.camPos.copy(this.position).addScaledVector(this._up, this._eyeHeight)
      this._camera.position.copy(_s.camPos)

      // 5g. CAVE→SURFACE transition: if position radius nears surfaceRadius, snap back
      const r = this.position.length()
      const surfR = this._sampler.surfaceRadiusAt(this._up)
      if (r >= surfR - SURFACE_SNAP_M) {
        // Only snap if the surface-pin spot is open above (density ≤ 0 at surface feet)
        // — prevents snapping if the cave roof happens to graze the surface.
        _s.tmp.copy(this._up).multiplyScalar(surfR)  // where surface-mode would pin us
        const surfaceDensity = this._collider.densityAt(_s.tmp)
        if (surfaceDensity <= 0) {
          // Emerged onto open surface — snap to SURFACE mode
          this.position.copy(this._up).multiplyScalar(surfR)
          this._vy   = 0
          this._mode = 'surface'
          _s.camPos.copy(this._up).multiplyScalar(surfR + this._eyeHeight)
          this._camera.position.copy(_s.camPos)
        }
      }
    }

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
