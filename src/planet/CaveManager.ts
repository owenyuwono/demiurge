/**
 * CaveManager.ts — near-field cave meshing, scene management, and world-space
 * collision adapter for the first-person controller.
 *
 * Responsibilities:
 *  (a) Near-field cave meshing + scene management:
 *      Stream cave chunks in/out based on camera proximity. Chunks are parented
 *      under the Planet Group (which extends THREE.Group), so they inherit the
 *      planet's world quaternion and render correctly in all modes.
 *  (b) Dark cave material + headlamp PointLight.
 *  (c) CaveCollider adapter (world-space) that bridges the planet-LOCAL cave SDF
 *      to the world-space convention expected by the first-person controller.
 *
 * World↔local coordinate convention:
 *   Cave field positions are planet-LOCAL (|p| ≈ RADIUS + elevation).
 *   The Planet group may be rotated (spin mode). To convert:
 *     local = world.applyQuaternion( planet.getWorldQuaternion().invert() )
 *     world = local.applyQuaternion( planet.getWorldQuaternion() )
 *   We pre-compute the world quaternion once per update() and cache it in
 *   _worldQuat / _invWorldQuat to keep the hot collider path zero-alloc.
 */

import {
  DoubleSide,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  PointLight,
  Quaternion,
  Vector3,
} from 'three'
import { CaveField } from './caveField'
import { CaveCollider } from '../controls/NavMode'
import {
  CAVE_CHUNK_M,
  computeCaveArrays,
  caveArraysToGeometry,
} from './CaveMesher'

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/**
 * Near-field streaming radius in meters (local space).
 * Chunks whose center is within this radius of the camera are meshed.
 */
const CAVE_RADIUS_M = 128

/**
 * Eviction hysteresis: chunks whose center exceeds this radius are removed.
 * Slightly larger than CAVE_RADIUS_M to prevent thrashing at the boundary.
 */
const CAVE_EVICT_M = CAVE_RADIUS_M + 32

/**
 * Camera depth threshold (meters) for activating caves by depth alone. Caves
 * become active only when the camera is at/below the surface (depthBelow > 0).
 * Must stay BELOW the FP eye height (~1.7 m) so ordinary surface walking — where
 * the camera sits ~1.7 m above the terrain (depthBelow ≈ -1.7) — does NOT activate
 * the (synchronous, main-thread) cave mesher. Approaching an entrance is handled
 * independently by CAVE_NEARMOUTH_M.
 */
const CAVE_ACTIVATE_MARGIN = 0

/**
 * Proximity to the cave mouth (meters, local space) that also triggers
 * cave meshing even when the camera is slightly above the surface margin.
 */
const CAVE_NEARMOUTH_M = 80

/**
 * Maximum cave chunk shell depth for candidate chunk generation.
 * Mirrors the CaveField CAVE_SHELL_MAX constant.
 */
const CAVE_SHELL_MAX_M = 350

/**
 * Maximum chunks meshed per update() call.
 * Caps the synchronous meshing cost on first entry. Increase if acceptable.
 */
const CHUNKS_PER_FRAME = 6

/**
 * EPS for the gradient-magnitude estimation in the metric-distance converter.
 * Must be large enough to straddle one voxel of the noise field (~1.33 m voxels).
 */
const GRAD_EPS = 0.5   // meters

/**
 * Floor for gradient magnitude division — prevents divide-by-zero in flat regions.
 */
const GRAD_FLOOR = 1e-4

/**
 * Metric-distance clamp bounds for the world-space collider adapter.
 *
 * The cave field returns a large sentinel (~1e6) for points above the terrain
 * surface ("solid above ground").  With a near-zero gradMag the raw division
 * d0/gradMag can exceed 1e10 m, teleporting the player on cave exit.
 *
 * MAX_METRIC_ROCK: positive cap (pushout / penetration depth).  The controller
 *   sub-steps at ~player scale so penetrations stay well under 1 m; 8 m gives
 *   ample headroom without letting a BIG-sentinel produce a catastrophic pushout.
 *
 * MIN_METRIC_AIR: negative floor (air clearance).  Negative values mean "no
 *   collision" — a very negative value is harmless (produces no pushout), so the
 *   negative side is clamped loosely.
 */
const MAX_METRIC_ROCK =  8    // meters — cap on positive (rock-side) metric distance
const MIN_METRIC_AIR  = -64   // meters — floor on negative (air-side) metric distance

// ---------------------------------------------------------------------------
// Minimal Planet interface — CaveManager only needs these from Planet.
// The actual Planet class satisfies this (it extends Group and exposes
// getWorldQuaternion() inherited from Object3D).
// ---------------------------------------------------------------------------

export interface PlanetLike extends Object3D {
  /**
   * Inherited from THREE.Object3D. Returns the world-space quaternion into
   * the provided target and also returns it. Planet.ts calls this internally
   * already (line ~538). CaveManager calls it once per update().
   */
  getWorldQuaternion(target: Quaternion): Quaternion
}

// ---------------------------------------------------------------------------
// CaveManager options
// ---------------------------------------------------------------------------

export interface CaveManagerOpts {
  /** The cave SDF — produced by makeCaveSampler(). */
  field: CaveField
  /**
   * The planet Group. Cave meshes are added as children of this node, so they
   * inherit the planet's world rotation. Must implement PlanetLike (Planet
   * satisfies this as it extends THREE.Group).
   */
  planet: PlanetLike
  /** Optional: unused in MVP but kept for parity with potential scene-level
   *  lights that callers may inject.  Ignored internally. */
  scene?: { add: (o: Object3D) => void }
}

// ---------------------------------------------------------------------------
// Chunk key helper
// ---------------------------------------------------------------------------

function chunkKey(cx: number, cy: number, cz: number): string {
  return `${cx},${cy},${cz}`
}

// ---------------------------------------------------------------------------
// CaveManager
// ---------------------------------------------------------------------------

export class CaveManager {
  private _field: CaveField
  private readonly planet: PlanetLike

  // Cached quaternions — refreshed each update(). Zero-alloc on collider calls.
  private readonly _worldQuat    = new Quaternion()
  private readonly _invWorldQuat = new Quaternion()

  // Per-update scratch vectors (never hold references between calls).
  private readonly _camLocal    = new Vector3()
  private readonly _mouthLocal  = new Vector3()
  private readonly _chunkCenter = new Vector3()
  private readonly _pLocal      = new Vector3()
  private readonly _nLocal      = new Vector3()
  private readonly _nWorld      = new Vector3()
  private readonly _nTmp        = new Vector3()
  private readonly _pEps        = new Vector3()

  // Active cave chunk meshes keyed by chunkKey(cx,cy,cz).
  private readonly _caveMeshes = new Map<string, Mesh>()

  // Cave material — dark warm rock, vertex colors for AO.
  private readonly _material: MeshStandardMaterial

  // Headlamp.
  private readonly _headlamp: PointLight

  // Collider returned to callers — a stable closure object.
  private readonly _collider: CaveCollider

  constructor(opts: CaveManagerOpts) {
    this._field  = opts.field
    this.planet = opts.planet

    // ------------------------------------------------------------------
    // Cave material
    // Dark warm rock, vertexColors=true (CaveMesher bakes AO into colors),
    // double-sided for winding safety.
    // ------------------------------------------------------------------
    this._material = new MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.95,
      metalness: 0,
      color: 0x3a3530,  // base tint — vertex colors modulate over this
      side: DoubleSide,
    })

    // ------------------------------------------------------------------
    // Headlamp — warm yellow-white PointLight, modest 60 m range.
    // Starts invisible; enable with setHeadlampEnabled(true).
    // Position/orientation is inherited from the camera via attachHeadlamp().
    // ------------------------------------------------------------------
    this._headlamp = new PointLight(0xfff5e0, 3.0, 60, 2)
    this._headlamp.visible = false

    // ------------------------------------------------------------------
    // Collider — a stable object wrapping the adapter methods.
    // Bound to `this` so it can access _invWorldQuat/_worldQuat without
    // re-constructing closures.
    // ------------------------------------------------------------------
    this._collider = {
      densityAt:  (pWorld) => this._densityAtWorld(pWorld),
      normalAt:   (pWorld, out) => this._normalAtWorld(pWorld, out),
    }
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  /**
   * Per-frame update. Call once per frame with the camera's WORLD position.
   *
   * - Refreshes the cached world/inv-world quaternions.
   * - Decides if caves are active based on camera depth below surface or
   *   proximity to the cave mouth.
   * - Meshes missing near-field cave chunks (up to CHUNKS_PER_FRAME per call).
   * - Evicts far chunks from the scene.
   */
  update(cameraWorldPos: Vector3): void {
    // Refresh quaternion cache once per frame.
    this.planet.getWorldQuaternion(this._worldQuat)
    this._invWorldQuat.copy(this._worldQuat).invert()

    // Convert camera to planet-local space.
    this._camLocal.copy(cameraWorldPos).applyQuaternion(this._invWorldQuat)

    const active = this._isCaveActive(this._camLocal)

    if (!active) {
      this._evictAll()
      return
    }

    this._meshNearChunks(this._camLocal)
    this._evictFarChunks(this._camLocal)
  }

  /**
   * Returns the world-space CaveCollider adapter.
   * Pass this to the first-person controller when in cave mode.
   * The same object is returned on every call.
   */
  getCollider(): CaveCollider {
    return this._collider
  }

  /**
   * The single deterministic cave mouth in WORLD space.
   * Rotate the local mouth position by the planet's world quaternion.
   * main.ts uses this to spawn the player at the cave entrance.
   *
   * NOTE: _worldQuat is only valid after the first update() call.
   * Call getMouthWorld() only after the first update().
   */
  getMouthWorld(): { pos: Vector3; up: Vector3; throatRadius: number } {
    const { mouth } = this._field

    const pos = mouth.pos.clone().applyQuaternion(this._worldQuat)
    const up  = mouth.up.clone().applyQuaternion(this._worldQuat)

    return { pos, up, throatRadius: mouth.throatRadius }
  }

  /**
   * Toggle the headlamp light visibility.
   * Call with `true` when entering FirstPerson/cave mode, `false` on exit.
   */
  setHeadlampEnabled(on: boolean): void {
    this._headlamp.visible = on
  }

  /**
   * Attach the headlamp to the camera Object3D as a child.
   * The headlamp is positioned at the camera origin (0,0,0 in camera space)
   * and therefore moves/rotates with the camera automatically.
   * Safe to call multiple times — only the first call adds it as a child.
   */
  attachHeadlamp(camera: Object3D): void {
    if (this._headlamp.parent !== camera) {
      camera.add(this._headlamp)
    }
  }

  /**
   * Swap in a new cave field after a seed regenerate.
   *
   * Evicts all cached cave meshes and clears the empty-chunk memo so the new
   * seed rebuilds caves from scratch. The collider returned by getCollider()
   * remains valid — its methods read this._field at call time, so they
   * automatically use the new field after this swap.
   */
  setField(field: CaveField): void {
    this._field = field
    this._evictAll()
  }

  /**
   * Dispose all cave meshes and the material.
   * Call when the planet is destroyed or regenerated.
   */
  dispose(): void {
    this._evictAll()
    this._material.dispose()
    // Headlamp is a child of the camera; caller removes/detaches if needed.
  }

  // --------------------------------------------------------------------------
  // Activation check
  // --------------------------------------------------------------------------

  /**
   * Caves are active when the camera is at/near/below the surface OR within
   * CAVE_NEARMOUTH_M of the cave mouth (lets the player see the entrance approach).
   */
  private _isCaveActive(camLocal: Vector3): boolean {
    // Depth below surface: positive = below surface.
    const camDir = this._camLocal.clone().normalize()
    const surfR  = this._field.surfaceRadiusLocal(camDir)
    const camR   = camLocal.length()
    const depthBelow = surfR - camR   // positive when below surface

    if (depthBelow > -CAVE_ACTIVATE_MARGIN) {
      return true
    }

    // Also activate if within CAVE_NEARMOUTH_M of the mouth.
    this._mouthLocal.copy(this._field.mouth.pos)
    if (camLocal.distanceTo(this._mouthLocal) < CAVE_NEARMOUTH_M) {
      return true
    }

    return false
  }

  // --------------------------------------------------------------------------
  // Chunk streaming
  // --------------------------------------------------------------------------

  /**
   * Determine the set of candidate cave chunk cells that:
   *  1. Lie within CAVE_RADIUS_M of the camera in local space.
   *  2. Overlap the valid cave-shell depth range (0..CAVE_SHELL_MAX_M below surface).
   *
   * For each candidate not already in _caveMeshes, run the mesher. Caps work
   * at CHUNKS_PER_FRAME per call to avoid a synchronous hitch on first entry.
   *
   * NOTE: Meshing is synchronous on the main thread (MVP). For large CAVE_RADIUS_M
   * or very deep caves this may cause a brief hitch on first approach. Worker
   * offloading follows the same pattern as MeshWorkerPool and can be added later.
   */
  private _meshNearChunks(camLocal: Vector3): void {
    const cx0 = Math.floor((camLocal.x - CAVE_RADIUS_M) / CAVE_CHUNK_M)
    const cy0 = Math.floor((camLocal.y - CAVE_RADIUS_M) / CAVE_CHUNK_M)
    const cz0 = Math.floor((camLocal.z - CAVE_RADIUS_M) / CAVE_CHUNK_M)
    const cx1 = Math.floor((camLocal.x + CAVE_RADIUS_M) / CAVE_CHUNK_M)
    const cy1 = Math.floor((camLocal.y + CAVE_RADIUS_M) / CAVE_CHUNK_M)
    const cz1 = Math.floor((camLocal.z + CAVE_RADIUS_M) / CAVE_CHUNK_M)

    // Planet surface radius at camera direction — used to filter chunks too
    // far above or too far below the cave shell.
    const camDir = camLocal.clone().normalize()
    const surfR  = this._field.surfaceRadiusLocal(camDir)

    let meshedThisFrame = 0

    for (let cz = cz0; cz <= cz1 && meshedThisFrame < CHUNKS_PER_FRAME; cz++) {
      for (let cy = cy0; cy <= cy1 && meshedThisFrame < CHUNKS_PER_FRAME; cy++) {
        for (let cx = cx0; cx <= cx1 && meshedThisFrame < CHUNKS_PER_FRAME; cx++) {
          const key = chunkKey(cx, cy, cz)
          if (this._caveMeshes.has(key)) continue
          if (this._emptyChunks.has(key)) continue   // already known empty — don't re-mesh it every frame

          // Chunk center in local space.
          const ocx = (cx + 0.5) * CAVE_CHUNK_M
          const ocy = (cy + 0.5) * CAVE_CHUNK_M
          const ocz = (cz + 0.5) * CAVE_CHUNK_M

          this._chunkCenter.set(ocx, ocy, ocz)

          // Cull: chunk center must be within CAVE_RADIUS_M of camera.
          if (this._chunkCenter.distanceTo(camLocal) > CAVE_RADIUS_M) continue

          // Cull: chunk must overlap the cave shell (depth 0..CAVE_SHELL_MAX_M
          // below the local surface). Approximate by measuring the chunk center
          // depth below the surface at that direction.
          const chunkDir = this._chunkCenter.clone().normalize()
          const chunkSurfR = this._field.surfaceRadiusLocal(chunkDir)
          const chunkR = this._chunkCenter.length()
          const chunkDepth = chunkSurfR - chunkR  // positive = below surface

          // Skip if entirely above surface (with one chunk margin) or below max shell.
          if (chunkDepth < -(CAVE_CHUNK_M)) continue
          if (chunkDepth > CAVE_SHELL_MAX_M + CAVE_CHUNK_M) continue

          // Mesh this chunk.
          const arrays = computeCaveArrays({ field: this._field, cx, cy, cz })

          if (arrays.empty) {
            // No iso-crossing here → no geometry. Memoize in _emptyChunks (checked
            // by the guard at the top of the loop) so we don't recompute Surface Nets
            // on this chunk every frame. Cleared on eviction so re-entry re-evaluates.
            this._markEmpty(key)
            continue
          }

          const geometry = caveArraysToGeometry(arrays)
          const mesh = new Mesh(geometry, this._material)
          // Origin-relative local positions + local origin = local world position.
          // Parented under the Planet Group → inherits planet world transform.
          mesh.position.set(arrays.originX, arrays.originY, arrays.originZ)
          mesh.frustumCulled = false  // cave geometry is never frustum-culled independently

          this.planet.add(mesh)
          this._caveMeshes.set(key, mesh)
          meshedThisFrame++
        }
      }
    }
  }

  // Empty-chunk sentinel to avoid re-evaluating the same empty cell every frame.
  private readonly _emptyChunks = new Set<string>()

  private _markEmpty(key: string): void {
    this._emptyChunks.add(key)
  }

  private _evictFarChunks(camLocal: Vector3): void {
    for (const [key, mesh] of this._caveMeshes) {
      const ox = mesh.position.x
      const oy = mesh.position.y
      const oz = mesh.position.z
      this._chunkCenter.set(ox, oy, oz)
      if (this._chunkCenter.distanceTo(camLocal) > CAVE_EVICT_M) {
        this.planet.remove(mesh)
        mesh.geometry.dispose()
        this._caveMeshes.delete(key)
        // Allow re-meshing if camera returns.
        this._emptyChunks.delete(key)
      }
    }
  }

  private _evictAll(): void {
    for (const [, mesh] of this._caveMeshes) {
      this.planet.remove(mesh)
      mesh.geometry.dispose()
    }
    this._caveMeshes.clear()
    this._emptyChunks.clear()
  }

  // --------------------------------------------------------------------------
  // World-space CaveCollider adapter
  // --------------------------------------------------------------------------

  /**
   * Convert a WORLD-space position to a METRIC signed distance (meters).
   *
   * Conversion pipeline:
   *
   * 1. WORLD → LOCAL:
   *    pLocal = pWorld.applyQuaternion( invWorldQuat )
   *    (inverse of planet's world quaternion, refreshed once per update())
   *
   * 2. RAW DENSITY → METRIC DISTANCE:
   *    caveField.densityAt is a noise-based SDF with gradient magnitude ≪ 1/m.
   *    Returning it raw gives pushout forces ~100× too weak for the controller.
   *    We estimate the gradient magnitude at pLocal via a finite-difference
   *    directional sample along the gradient direction, then divide:
   *      d0      = field.densityAt(pLocal)
   *      gradDir = field.normalAt(pLocal, ...)   // unit vector ∇density / |∇density|
   *      d1      = field.densityAt(pLocal + gradDir * GRAD_EPS)
   *      gradMag = max(GRAD_FLOOR, (d1 - d0) / GRAD_EPS)
   *      metricDist = d0 / gradMag               // approximate signed distance, meters
   *    Sign is preserved: air < 0, rock > 0.
   *
   * Convention match:
   *    CaveField.densityAt: positive = rock, negative = air  ✓ matches CaveCollider.
   *    The metric scaling makes the pushout distance physically plausible.
   */
  private _densityAtWorld(pWorld: Vector3): number {
    // 1. World → local.
    this._pLocal.copy(pWorld).applyQuaternion(this._invWorldQuat)

    // 2. Raw density + gradient magnitude estimate.
    const d0 = this._field.densityAt(this._pLocal)

    // Get unit gradient direction (toward rock, i.e. toward increasing density).
    this._field.normalAt(this._pLocal, this._nTmp)

    // Sample d1 slightly along the gradient direction.
    this._pEps.copy(this._pLocal).addScaledVector(this._nTmp, GRAD_EPS)
    const d1 = this._field.densityAt(this._pEps)

    // Gradient magnitude along the unit direction.
    const gradMag = Math.max(GRAD_FLOOR, (d1 - d0) / GRAD_EPS)

    // Metric signed distance (meters). Sign preserved.
    // Clamp to [MIN_METRIC_AIR, MAX_METRIC_ROCK] so a BIG above-surface sentinel
    // (~1e6) divided by a near-zero gradMag (~1e-4) cannot produce a ~1e10 m
    // pushout that teleports the player on cave exit.
    const metric = d0 / gradMag
    return Math.max(MIN_METRIC_AIR, Math.min(MAX_METRIC_ROCK, metric))
  }

  /**
   * Return the OUTWARD unit normal at a world-space point.
   *
   * Convention mismatch:
   *   field.normalAt returns ∇density = points TOWARD ROCK (from air into solid).
   *   CaveCollider.normalAt must return OUTWARD = points TOWARD AIR (away from solid).
   *   Fix: negate the local gradient, then rotate to world space.
   *
   * 3. LOCAL OUTWARD NORMAL → WORLD OUTWARD NORMAL:
   *    localOutward = -field.normalAt(pLocal)
   *    worldOutward = localOutward.applyQuaternion( worldQuat )
   *    Write into `out` and return it.
   */
  private _normalAtWorld(pWorld: Vector3, out: Vector3): Vector3 {
    // 1. World → local.
    this._pLocal.copy(pWorld).applyQuaternion(this._invWorldQuat)

    // 2. Local gradient (points toward rock).
    this._field.normalAt(this._pLocal, this._nLocal)

    // 3. Negate → outward (toward air), then rotate to world space.
    //    Quaternion rotation preserves unit length, so the result is still unit.
    out.copy(this._nLocal).negate().applyQuaternion(this._worldQuat)

    return out
  }
}
