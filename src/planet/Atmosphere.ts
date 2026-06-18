import {
  AdditiveBlending,
  BackSide,
  Color,
  Mesh,
  Object3D,
  SphereGeometry,
  Vector3,
} from 'three'
import { MeshBasicNodeMaterial } from 'three/webgpu'
import {
  abs,
  cameraPosition,
  dot,
  exp,
  float,
  max,
  min,
  mix,
  normalize,
  positionWorld,
  pow,
  saturate,
  uniform,
  vec3,
} from 'three/tsl'

// ---------------------------------------------------------------------------
// Atmosphere — analytic single-scatter shell for the planet.
//
// Renders on BackSide with AdditiveBlending so it:
//   - glows blue at the limb when viewed from orbit
//   - fades to black in space (night side + non-grazing angles)
//   - shows blue sky tint from the surface looking up
//
// KEY DESIGN: the planet Group lives at the world origin and is only rotateY-
// spun (never translated). All lighting and camera math works directly in
// WORLD FRAME using three/tsl built-in nodes:
//   cameraPosition — world-space camera position (confirmed world-space in
//                    three@0.182; it is the camera's position in world coords)
//   positionWorld  — world-space fragment position
//   _uSunDir       — planet's existing _uSunDir uniform, passed BY REFERENCE
//
// Node graph — see inline comments.
// renderOrder=5 (behind CloudShell's 10; behind water's default 0 is fine
// because additive blending composites over whatever is behind it).
// ---------------------------------------------------------------------------

export class Atmosphere {
  private readonly _scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown }

  // Geometry constants
  private readonly _radius: number
  private readonly _heightScale: number
  private readonly _baseAtmHeightMul: number
  private _atmHeightMul: number

  // Uniform nodes — live setters mutate .value directly; no rebuild needed
  private readonly _uDensity       = uniform(1.0)
  // Overwritten at startup from the GUI hex (#2e6bff → linear ~0.027, 0.147, 1.0)
  private readonly _uTint          = uniform(new Vector3(0.027, 0.147, 1.0))
  private readonly _uSunIntensity  = uniform(1.3)
  private readonly _uScaleHeight   = uniform(3.0)
  // Zenith in-scatter floor. The base limb-glow term (pow(grazing, scaleHeight))
  // goes to EXACTLY zero at the zenith, so from the surface the sky overhead is
  // black and only a thin horizon band glows. This floor lifts the whole sunlit
  // dome so the sky reads blue overhead. 0 = old limb-only behaviour exactly.
  private readonly _uSkyFloor      = uniform(0.35)
  // _uSunDir is the PASSED reference from Planet — never create a new one
  private readonly _uSunDir: ReturnType<typeof uniform>

  // Render objects — tracked for dispose
  private _mesh:  Mesh | null = null
  private _geo:   SphereGeometry | null = null
  private _mat:   MeshBasicNodeMaterial | null = null

  private _visible = true  // default ON (unlike clouds which start hidden)

  constructor(
    scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown },
    opts: {
      radius:       number
      heightScale:  number
      /** Planet's own TSL uniform node — used directly in the node graph so sun lighting
       *  is always live regardless of how Planet updates the node's value. */
      sunDir:       ReturnType<typeof uniform>
      /** Shell outer radius as a multiple of heightScale above surface (default 2.5). */
      atmHeightMul?: number
    },
  ) {
    this._scene            = scene
    this._radius           = opts.radius
    this._heightScale      = opts.heightScale
    this._uSunDir          = opts.sunDir
    this._baseAtmHeightMul = opts.atmHeightMul ?? 2.5
    this._atmHeightMul     = this._baseAtmHeightMul

    this._buildMesh()
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  get visible(): boolean { return this._visible }

  setVisible(v: boolean): void {
    this._visible = v
    if (this._mesh !== null) this._mesh.visible = v
  }

  /** Live — mutates density uniform, no rebuild. */
  setDensity(v: number): void { this._uDensity.value = v }

  setTint(c: Color): void {
    this._uTint.value.set(c.r, c.g, c.b)
  }

  /** Live — mutates sun intensity uniform, no rebuild. */
  setSunIntensity(v: number): void { this._uSunIntensity.value = v }

  /** Live — mutates scale height (horizon pow exponent) uniform, no rebuild. */
  setScaleHeight(v: number): void { this._uScaleHeight.value = v }

  /** Live — mutates zenith in-scatter floor uniform, no rebuild. 0 = limb-only. */
  setSkyFloor(v: number): void { this._uSkyFloor.value = v }

  /**
   * Scale the atmosphere shell radius. Mirrors CloudShell.setAltitude:
   * geometry is built at the base radius; scale factor adjusts the mesh.
   */
  setAtmHeight(mul: number): void {
    this._atmHeightMul = mul
    if (this._mesh !== null) {
      const baseR = this._shellR(this._baseAtmHeightMul)
      const newR  = this._shellR(mul)
      this._mesh.scale.setScalar(newR / baseR)
    }
  }

  /**
   * No-op stub — cameraPosition is a built-in TSL node that updates each
   * frame automatically. Kept for API symmetry with CloudShell.update().
   */
  update(): void { /* cameraPosition is a built-in node — no per-frame uniform needed */ }

  /**
   * Dispose all GPU resources and remove the mesh from the scene.
   * Safe to call multiple times.
   */
  dispose(): void {
    this._teardown()
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private _shellR(mul: number): number {
    return this._radius + this._heightScale * mul
  }

  private _buildMesh(): void {
    const shellR     = this._shellR(this._atmHeightMul)
    const baseShellR = this._shellR(this._baseAtmHeightMul)

    // --- Geometry ---
    // 64×32 segments: smooth enough for a sphere at this scale; nothing
    // renders between vertices so higher counts are pure waste.
    this._geo = new SphereGeometry(baseShellR, 64, 32)

    // --- Scattering node graph ---
    //
    // All in WORLD FRAME — valid because the planet Group is at the origin
    // and only rotateY-spun (never translated).
    //
    // cameraPosition: built-in TSL node = world-space camera position.
    //   CONFIRMED exported from three/tsl (three@0.182) and world-space.
    //   Source: node_modules/three/src/nodes/accessors/Camera.js
    //   The node returns the camera's position in world coordinates, not
    //   eye/view space — no correction needed.
    //
    // positionWorld: built-in TSL node = world-space fragment position.
    //
    // _uSunDir: planet's own uniform node passed BY REFERENCE — no second sun uniform.

    const camPos  = cameraPosition
    const fragW   = positionWorld

    // viewDir: direction from camera to fragment (normalised)
    const viewDir = normalize(fragW.sub(camPos))

    // nFrag: outward surface normal at the fragment (world-space unit sphere direction)
    const nFrag   = normalize(fragW)

    // grazing: 1 at the limb (viewDir perpendicular to nFrag), 0 at zenith/nadir.
    // abs() makes it symmetric: works from both inside (surface) and outside (orbit).
    const grazing = saturate(float(1).sub(abs(dot(viewDir, nFrag))))

    // horizonGlow: pow sharpens toward the limb; higher _uScaleHeight → thinner band.
    // Guard the exponent: pow(0,0) is indeterminate in WGSL (NaN/speckle).
    // max(..., 0.1) ensures pow(0, >=0.1)=0 at the zenith.
    const horizonGlow = pow(grazing, max(this._uScaleHeight, float(0.1)))

    // airMassGlow: a physical air-mass curve that BROADENS the brightening across
    // the whole dome (not just a thin limb band). muUp = cos(view-zenith angle) =
    // dot(viewDir, surface normal): 1 looking straight up, 0 at the horizon. The
    // relative air mass 1/cos − 1 is 0 overhead and grows toward the horizon, and
    // brightness ~ 1 − exp(−k·airMass) rises gradually from the zenith — the real
    // Rayleigh shape — instead of the sharp pow() ramp. Combined (max) with the
    // pow() term so the _uScaleHeight slider still sharpens the very limb.
    const muUp        = saturate(dot(viewDir, nFrag))
    const relAirMass  = float(1).div(max(muUp, float(0.05))).sub(1)
    const airMassGlow = float(1).sub(exp(relAirMass.negate().mul(0.6)))
    const glow        = max(horizonGlow, airMassGlow)

    // density: lift the whole sunlit dome by the zenith floor so the sky reads blue
    // overhead from the surface instead of going black. Maps glow 0→skyFloor, 1→1.
    // skyFloor=0 reproduces a black zenith (so space stays black at the slider min).
    // This constant isotropic lift IS the cheap multiple-scattering ambient (a la Hillaire's
    // 1/(1-f) closed form): the dome away from the sun isn't pure single-scatter black.
    const density = mix(this._uSkyFloor, float(1), glow)

    // phase: Rayleigh phase function — brightens the side facing the sun.
    // mu = cos(angle between view and sun): +1 = looking toward sun, -1 = away.
    // phase = 0.75 * (1 + mu²) — classic Rayleigh, range [0.75, 1.5].
    const mu      = dot(viewDir, this._uSunDir)
    const phase   = float(0.75).mul(float(1).add(mu.mul(mu)))

    // sunUp: day/night fade — keeps the night-side limb black.
    // muSun = dot(nFrag, sunDir) > 0 = day side, < 0 = night side.
    // mul(2).add(0.3) gives a soft terminator: hits 1 at ~35° past noon,
    // fades to 0 ~9° into the night side. Tuned for a soft blue twilight arc.
    const muSun   = dot(nFrag, this._uSunDir)
    const sunUp   = saturate(muSun.mul(2).add(0.3))

    // amount: combined scalar driving both color and opacity.
    const amount  = density.mul(phase).mul(sunUp).mul(this._uDensity)

    // Per-channel Rayleigh reddening (β ∝ 1/λ⁴). The SUN's slant air mass (long when the sun
    // is low on that point's horizon) plus the VIEW's slant air mass remove blue first from
    // the in-scattered light, shifting the terminator limb and the sky toward a low sun
    // blue→orange→red (sunrise/sunset, horizon reddening). At high sun / overhead the slant
    // is ~0 so redden ≈ (1,1,1) and the tuned blue `tint` is preserved EXACTLY.
    const amView  = float(1).div(max(abs(dot(viewDir, nFrag)), float(0.05)))
    const amSun   = float(1).div(max(muSun, float(0.05)))
    const slant   = min(max(amView.add(amSun).sub(2), float(0)), float(40))
    const betaR   = vec3(0.175, 0.408, 1.0)   // red:green:blue ∝ 1/λ⁴, normalised to blue=1
    const redden  = exp(betaR.mul(this._uDensity).mul(0.06).mul(slant).negate())

    // scatterColor → colorNode: tint × sunIntensity × amount × wavelength reddening
    const scatterColor = this._uTint.mul(this._uSunIntensity).mul(amount).mul(redden)

    // scatterEnvelope → opacityNode: 0 at space / zenith / night, 1 at peak limb.
    // saturate clamps to [0,1] so additive blending never over-brightens.
    const scatterEnvelope = saturate(amount)

    // --- Material ---
    const mat = new MeshBasicNodeMaterial()
    mat.colorNode   = scatterColor
    mat.opacityNode = scatterEnvelope
    mat.transparent = true
    mat.depthWrite  = false
    mat.side        = BackSide
    mat.blending    = AdditiveBlending
    this._mat = mat

    // --- Mesh ---
    const mesh = new Mesh(this._geo, this._mat)
    // Scale to target altitude (baseAtmHeightMul baked into geo radius)
    mesh.scale.setScalar(shellR / baseShellR)
    mesh.renderOrder   = 5       // behind CloudShell (10); additive over the lit globe
    mesh.frustumCulled = false
    mesh.visible       = this._visible
    this._mesh = mesh

    this._scene.add(this._mesh)
  }

  /** Remove mesh from scene, dispose all GPU objects. Safe to call multiple times. */
  private _teardown(): void {
    if (this._mesh !== null) {
      this._scene.remove(this._mesh)
      this._mesh = null
    }
    if (this._geo !== null) {
      this._geo.dispose()
      this._geo = null
    }
    if (this._mat !== null) {
      this._mat.dispose()
      this._mat = null
    }
    this._visible = false
  }
}
