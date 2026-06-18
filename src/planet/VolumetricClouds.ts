import {
  ClampToEdgeWrapping,
  DataTexture,
  DoubleSide,
  FrontSide,
  LinearFilter,
  Matrix3,
  Mesh,
  NormalBlending,
  Object3D,
  Quaternion,
  RGBAFormat,
  RepeatWrapping,
  SphereGeometry,
  UnsignedByteType,
  Vector3,
} from 'three'
import { MeshBasicNodeMaterial } from 'three/webgpu'
import {
  Break,
  Fn,
  If,
  Loop,
  asin,
  atan,
  cameraPosition,
  clamp,
  cos,
  dot,
  exp,
  float,
  fwidth,
  interleavedGradientNoise,
  length,
  max,
  min,
  mix,
  mx_fractal_noise_float,
  mx_noise_float,
  mx_worley_noise_float,
  normalize,
  positionWorld,
  pow,
  saturate,
  screenCoordinate,
  select,
  sin,
  smoothstep,
  sqrt,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'
import { Climate, WindSample } from './climate'

// ---------------------------------------------------------------------------
// VolumetricClouds — full volumetric raymarched cloud layer for the planet.
//
// REPLACES CloudShell with the same external surface (field name cloudShell,
// buildCloudShell, get visible, setVisible, rebuild, dispose, and all
// setCloud* method names) so Planet.ts compiles unchanged.
//
// Architecture:
//   - DoubleSide SphereGeometry shell at outerR
//   - MeshBasicNodeMaterial fragment ray-marches a lit 3D cloud medium across
//     the spherical annulus innerR..outerR in world frame
//   - Equirect bakes (wind + fav) VERBATIM from CloudShell.ts
//   - Density shape sampled in planet-local frame via per-frame _uInvRot mat3
//   - Lit medium: HG phase + beer-powder + ambient + Beer extinction + early-out
//
// Depth occlusion:
//   The renderer is created with logarithmicDepthBuffer:true, so terrain writes
//   log-encoded depth values. viewportLinearDepth decodes via the standard
//   perspective inverse — the two don't compose, producing garbage tScene values
//   that erase most clouds. We therefore skip the depth clamp entirely and use
//   tExit0 directly (clouds-always-in-front).
//
//   This is correct: cloudBase >= heightScale*1.2 (~14.4 km) places the entire
//   cloud annulus well above peak terrain relief (~12 km). From orbit the clouds
//   are always above terrain; from the surface looking up, clouds are above the
//   camera. Clouds-always-in-front is the right behaviour in both cases.
//
//   Future: real log-depth occlusion would require logarithmicDepthToViewZ +
//   a dot(rd, cameraForward) ray-distance conversion to turn view-Z into t.
//
// renderOrder=10, NormalBlending, depthWrite=false, frustumCulled=false.
// Atmosphere stays renderOrder 5.
// ---------------------------------------------------------------------------

/** Scratch type for one-shot wind bake. Reused across all 131072 texels. */
interface _WindScratch {
  out: WindSample
  dir: Vector3
}

/** Extended scratch for the fav bake. */
interface _FavScratch {
  out:           WindSample
  dir:           Vector3
  climateSample: { temperature: number; moisture: number }
  dEast:  Vector3
  dWest:  Vector3
  dNorth: Vector3
  dSouth: Vector3
  east:   Vector3
  north:  Vector3
  wE_pos: WindSample
  wE_neg: WindSample
  wN_pos: WindSample
  wN_neg: WindSample
}

export class VolumetricClouds {
  private readonly _scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown }
  private readonly _climate: Climate

  // Geometry constants
  private readonly _radius:          number
  private readonly _heightScale:     number
  private readonly _baseAltitudeMul: number
  private readonly _uSunDir:         ReturnType<typeof uniform>
  private _altitudeMul:              number

  // ---------------------------------------------------------------------------
  // Kept uniforms (identical to CloudShell)
  // ---------------------------------------------------------------------------
  // NB: this is a THRESHOLD, not a fill fraction — HIGHER = LESS cloud. ~0.65 leaves big
  // CLEAR regions (deserts, subtropical oceans) so the organizing fields can concentrate
  // clouds into coherent systems — Earth has large cloud-free areas, not uniform cover.
  private readonly _uCoverage    = uniform(0.65)
  // Angular flow rate (rad/s ·windspeed) for great-circle advection — small = slow drift.
  private readonly _uScrollSpeed = uniform(0.001)
  private readonly _uScale       = uniform(6)
  // Wind-aligned domain warp — shears cloud shapes ALONG the wind flow for cyclonic swirls.
  // 0.2: visible swirl/flow; much above ~0.25 it smears clouds into thin streak filaments.
  private readonly _uWarp        = uniform(0.2)
  // Max cloud opacity — kept below 1 so even the densest clouds stay slightly translucent
  // (nothing reads as a flat opaque-white cutout).
  private readonly _uOpacity     = uniform(0.6)
  private readonly _uTime        = uniform(0)

  private readonly _uBillow      = uniform(0.45)
  private readonly _uDetail      = uniform(0.3)
  private readonly _uSoftness    = uniform(0.3)
  /** Kept for API compatibility. At 0: fully transparent; at 1: full raymarch. */
  private readonly _uVolume      = uniform(0.6)

  // Regional favorability strength (moisture + wind convergence) — secondary modulator now;
  // the warped weather map (uWeatherWeight) is the dominant organizer.
  private readonly _uFavWeight   = uniform(0.28)
  private readonly _uMoistWeight = uniform(0.6)
  // Convergence weight inside fav — raised so clouds gather strongly in the converging
  // (low-pressure / cyclone) zones → the swirling cloud systems.
  private readonly _uConvWeight  = uniform(0.8)
  private readonly _uConvGain    = uniform(2.0)
  // Latitude-band strength — Earth-like circulation (cloudy ITCZ + storm tracks, CLEAR
  // subtropical desert belt + poles). Carves the clear desert belts at ~30°.
  private readonly _uItczWeight  = uniform(0.12)
  // Weather-map strength — domain-warped low-freq organizer (big clear regions + flowing
  // systems). Moderate (0.3): enough organization without clumping everything into solid
  // grouped masses (higher read as "too grouped").
  private readonly _uWeatherWeight = uniform(0.3)
  // Voronoi "weather-cell" layer strength — tessellates cloudy regions into convection-cell
  // sized cells (cloudy near centres, clear lanes at boundaries). 0 = off (blobby weather
  // map), 1 = fully cellular. The structured-but-organic look (open/closed convection cells).
  private readonly _uCellWeight  = uniform(0.5)

  // ---------------------------------------------------------------------------
  // New volumetric uniforms
  // ---------------------------------------------------------------------------

  /** Cloud layer base height as a multiple of heightScale above surface. */
  private readonly _uCloudBase    = uniform(1.2)
  /** Cloud layer thickness as a multiple of heightScale. */
  private readonly _uCloudThick   = uniform(0.6)
  /** Extinction / density coefficient σ_T. */
  private readonly _uSigmaT       = uniform(4.0)
  /** Primary ray step count — set per-frame by distance LOD (max = _baseStepCount). */
  private readonly _uStepCount    = uniform(48, 'int')
  /** Fine-step cap as a fraction of annulus thickness — LOD raises it when far (coarse,
   *  cheap, complete) and lowers it when near (fine detail). */
  private readonly _uStepCap      = uniform(0.11)
  /** Distance-LOD scalars (CPU-side, applied in update()): the slider's step count is the
   *  CLOSE-UP max; far views drop to _minSteps so distant clouds barely raymarch. */
  private _baseStepCount = 48
  private readonly _minSteps = 8
  /** Light-march step count per primary sample. */
  private readonly _uLightSteps   = uniform(5, 'int')
  /** Henyey-Greenstein anisotropy (0=isotropic, 0.9=strong forward). */
  private readonly _uHgAnisotropy = uniform(0.35)
  /** Beer-powder dual-lobe strength. */
  private readonly _uPowder       = uniform(1.0)
  /** Detail noise frequency multiplier. */
  private readonly _uDetailScale  = uniform(4.0)
  /** Vertical profile: smoothstep rise width at base [0,1]. */
  private readonly _uRoundBase    = uniform(0.2)
  /** Vertical profile: smoothstep erosion start from billow top [0,1]. */
  private readonly _uBillowTop    = uniform(0.5)
  /** Ambient light contribution weight. Low (0.08) so the SUN drives light/dark colour
   *  contrast (3D shading) rather than a flat ambient fill that makes clouds read as
   *  uniform white whose only variation is opacity. */
  private readonly _uAmbient      = uniform(0.08)
  /** Convective cloud-TYPE strength [0,1]. 0 = uniform morphology everywhere (the exact
   *  pre-type behaviour — the slider's escape hatch); 1 = full stratus↔cumulonimbus variety
   *  driven by the baked convergence/moisture/latitude fields (towering, denser, anvil-topped
   *  cloud over ITCZ/convergence; flat thin stratus over calm dry subtropics). */
  private readonly _uTypeStrength = uniform(0.6)

  /**
   * Debug "cloud map" mode (0 = volumetric raymarch; 1 = flat coverage heatmap + contour
   * outlines). Used by the 'cloud' view to study cloud formation/topology cheaply (one
   * density sample per pixel, no march).
   */
  private readonly _uDebugMode = uniform(0.0)

  /**
   * Per-frame inverse-rotation mat3 uniform.
   * Transforms a world-frame point to planet-local frame.
   * Updated each frame via update(t, camWorldPos?, invWorldQuat?).
   */
  private readonly _uInvRot = uniform(new Matrix3(), 'mat3')

  // ---------------------------------------------------------------------------
  // Render objects
  // ---------------------------------------------------------------------------
  private _mesh:    Mesh | null = null
  private _geo:     SphereGeometry | null = null
  private _mat:     MeshBasicNodeMaterial | null = null
  private _windTex: DataTexture | null = null
  private _favTex:  DataTexture | null = null

  private _visible = false

  // ---------------------------------------------------------------------------
  // Bake scratch (zero per-texel allocation — verbatim from CloudShell)
  // ---------------------------------------------------------------------------
  private readonly _bakeScratch: _WindScratch = {
    out: { x: 0, y: 0, z: 0, speed: 0 },
    dir: new Vector3(),
  }

  private readonly _favScratch: _FavScratch = {
    out:           { x: 0, y: 0, z: 0, speed: 0 },
    dir:           new Vector3(),
    climateSample: { temperature: 0, moisture: 0 },
    dEast:  new Vector3(),
    dWest:  new Vector3(),
    dNorth: new Vector3(),
    dSouth: new Vector3(),
    east:   new Vector3(),
    north:  new Vector3(),
    wE_pos: { x: 0, y: 0, z: 0, speed: 0 },
    wE_neg: { x: 0, y: 0, z: 0, speed: 0 },
    wN_pos: { x: 0, y: 0, z: 0, speed: 0 },
    wN_neg: { x: 0, y: 0, z: 0, speed: 0 },
  }

  private readonly _polarAxis = new Vector3(0, 1, 0)

  constructor(
    scene: { add: (...o: Object3D[]) => unknown; remove: (...o: Object3D[]) => unknown },
    climate: Climate,
    opts: {
      radius:       number
      heightScale:  number
      /** Planet's own TSL uniform node — used by reference so sun updates propagate. */
      sunDir:       ReturnType<typeof uniform>
      /** Shell altitude above surface as a multiple of heightScale (default 1.5). */
      altitudeMul?: number
    },
  ) {
    this._scene           = scene
    this._climate         = climate
    this._radius          = opts.radius
    this._heightScale     = opts.heightScale
    this._uSunDir         = opts.sunDir
    this._baseAltitudeMul = opts.altitudeMul ?? 1.5
    this._altitudeMul     = this._baseAltitudeMul

    this._bakeWindTexture()
    this._bakeFavTexture()
    this._buildMesh()
  }

  // ---------------------------------------------------------------------------
  // Public API — identical surface to CloudShell
  // ---------------------------------------------------------------------------

  get visible(): boolean { return this._visible }

  setVisible(v: boolean): void {
    this._visible = v
    if (this._mesh !== null) this._mesh.visible = v
  }

  /** The cloud shell mesh (rebuilt on regenerate). Used by the half-res CloudCompositor to
   *  isolate clouds onto their own render layer for the offscreen pass. Null before first build. */
  getMesh(): Mesh | null { return this._mesh }

  /** Live — mutates uniform, no rebuild. */
  setCoverage(v: number):    void { this._uCoverage.value    = v }
  setScrollSpeed(v: number): void { this._uScrollSpeed.value = v }
  setCloudScale(v: number):  void { this._uScale.value       = v }
  setWindWarp(v: number):    void { this._uWarp.value        = v }
  setOpacity(v: number):     void { this._uOpacity.value     = v }

  setBillow(v: number):      void { this._uBillow.value      = v }
  setDetail(v: number):      void { this._uDetail.value      = v }
  setSoftness(v: number):    void { this._uSoftness.value    = v }
  setVolume(v: number):      void { this._uVolume.value      = v }

  setFavWeight(v: number):   void { this._uFavWeight.value   = v }
  setMoistWeight(v: number): void { this._uMoistWeight.value = v }
  setConvWeight(v: number):  void { this._uConvWeight.value  = v }
  setConvGain(v: number):    void { this._uConvGain.value    = v }
  setItczWeight(v: number):  void { this._uItczWeight.value  = v }
  setWeatherWeight(v: number): void { this._uWeatherWeight.value = v }
  setCellWeight(v: number): void { this._uCellWeight.value = v }

  /**
   * Debug "cloud map" mode. ON → flat coverage heatmap + contour outlines (study formation);
   * OFF → normal volumetric clouds. Also flips the material to opaque single-side depth-write
   * so the map renders cleanly (vs the translucent double-side volumetric shell).
   */
  setDebugMode(on: boolean): void {
    this._uDebugMode.value = on ? 1 : 0
    if (this._mat !== null) {
      this._mat.transparent = !on
      this._mat.depthWrite  = on
      this._mat.side        = on ? FrontSide : DoubleSide
      this._mat.needsUpdate = true
    }
  }

  // ---------------------------------------------------------------------------
  // New volumetric setters
  // ---------------------------------------------------------------------------

  /** Cloud layer base altitude. Triggers mesh rescale. Capped at 2.4 to stay below atmosphere. */
  setCloudBase(v: number):    void { this._uCloudBase.value = Math.min(v, 2.4);    this._rescaleMesh() }
  /** Cloud layer thickness. Triggers mesh rescale. Capped so base+thick <= 2.45. */
  setCloudThick(v: number):   void { this._uCloudThick.value = Math.min(v, 2.45 - this._uCloudBase.value);   this._rescaleMesh() }
  /** Extinction coefficient σ_T (density strength). */
  setDensity(v: number):      void { this._uSigmaT.value      = v }
  /** Primary ray step count (int). */
  setStepCount(v: number):    void { this._baseStepCount = Math.max(1, Math.round(v)) }
  /** Light-march step count per sample (int). */
  setLightSteps(v: number):   void { this._uLightSteps.value   = Math.max(1, Math.round(v)) }
  /** Henyey-Greenstein anisotropy g. Clamped [0, 0.95] to prevent infinite HG denominator. */
  setHgAnisotropy(v: number): void { this._uHgAnisotropy.value = Math.min(0.95, Math.max(0, v)) }
  /** Beer-powder dual-lobe strength. */
  setPowder(v: number):       void { this._uPowder.value       = v }
  /** Detail noise frequency. */
  setDetailScale(v: number):  void { this._uDetailScale.value  = v }
  /** Vertical profile base rounding. */
  setRoundBase(v: number):    void { this._uRoundBase.value    = v }
  /** Vertical profile billow top erosion. */
  setBillowTop(v: number):    void { this._uBillowTop.value    = v }
  /** Ambient light weight. */
  setAmbient(v: number):      void { this._uAmbient.value      = v }
  /** Convective cloud-type strength (0 = uniform; 1 = full stratus↔cumulonimbus variety). */
  setTypeStrength(v: number): void { this._uTypeStrength.value = v }

  /**
   * Thin alias for setCloudBase — kept for Planet API compatibility.
   * Previously also wrote _altitudeMul directly; now routes through setCloudBase
   * so _uCloudBase is always written via the single capped path.
   */
  setAltitude(mul: number): void {
    this._altitudeMul = mul
    this.setCloudBase(mul)
  }

  /**
   * Advance the cloud clock and push per-frame uniforms.
   *
   * @param t            Elapsed time in seconds.
   * @param camWorldPos  Camera world position (optional; cameraPosition built-in preferred).
   * @param invWorldQuat Inverse rotation of Planet Group as Quaternion or Matrix3.
   *                     When provided, sets _uInvRot so density is sampled in planet-local frame.
   */
  update(t: number, camWorldPos?: Vector3, invWorldQuat?: Quaternion | Matrix3): void {
    if (!this._visible) return
    this._uTime.value = t

    if (invWorldQuat !== undefined) {
      if (invWorldQuat instanceof Matrix3) {
        this._uInvRot.value.copy(invWorldQuat)
      } else {
        // Quaternion → rotation matrix 3×3 (inline, zero allocation)
        const q = invWorldQuat as Quaternion
        const { x: qx, y: qy, z: qz, w: qw } = q
        const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz
        const xx = qx * x2, xy = qx * y2, xz = qx * z2
        const yy = qy * y2, yz = qy * z2, zz = qz * z2
        const wx = qw * x2, wy = qw * y2, wz = qw * z2
        this._uInvRot.value.set(
          1 - (yy + zz),  xy - wz,         xz + wy,
          xy + wz,         1 - (xx + zz),  yz - wx,
          xz - wy,         yz + wx,         1 - (xx + yy),
        )
      }
    }

    // --- Distance LOD ---------------------------------------------------------
    // Raymarching is the expensive part, and from far away the volumetric depth isn't
    // perceptible — so scale the march down with distance: full steps + fine cap up close
    // (volumetric detail), few steps + coarse cap from orbit (cheap, still covers the thin
    // shell). The planet is at the world origin, so |camWorldPos| − radius = altitude.
    if (camWorldPos !== undefined) {
      const alt = camWorldPos.length() - this._radius
      // closeness: 1 below ~2·heightScale (near/in the cloud layer) → 0 above ~0.4·radius.
      const closeAlt = this._heightScale * 2
      const farAlt   = this._radius * 0.4
      const tRaw     = (alt - closeAlt) / (farAlt - closeAlt)
      const t        = tRaw < 0 ? 0 : tRaw > 1 ? 1 : tRaw
      const closeness = 1 - (t * t * (3 - 2 * t))   // smoothstep, inverted: 1 near, 0 far
      this._uStepCount.value = Math.max(
        this._minSteps,
        Math.round(this._minSteps + (this._baseStepCount - this._minSteps) * closeness),
      )
      // Cap: 0.11 (fine) up close → ~1.0 (coarse, whole-path) far, so the few far steps
      // still span the long grazing paths instead of being capped to tiny chunks.
      this._uStepCap.value = 0.11 + (1.0 - 0.11) * (1 - closeness)
    }
  }

  /**
   * Full rebuild: tears down mesh/geo/mat/textures, re-bakes from current climate.
   * Call after climate regenerate.
   */
  rebuild(): void {
    this._teardown()
    this._bakeWindTexture()
    this._bakeFavTexture()
    this._buildMesh()
  }

  /** Dispose all GPU resources and remove the mesh from the scene. */
  dispose(): void {
    this._teardown()
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /** Outer shell radius at current cloudBase + cloudThick, capped at 2.45× to stay below atmosphere. */
  private _outerR(): number {
    const outerMul = Math.min(this._uCloudBase.value + this._uCloudThick.value, 2.45)
    return this._radius + this._heightScale * outerMul
  }

  /** Base outer radius baked into geometry at construction defaults (base=1.2, thick=0.6). */
  private _baseOuterR(): number {
    return this._radius + this._heightScale * (1.2 + 0.6)
  }

  private _rescaleMesh(): void {
    if (this._mesh !== null) {
      this._mesh.scale.setScalar(this._outerR() / this._baseOuterR())
    }
  }

  // ---------------------------------------------------------------------------
  // Equirect bakes — VERBATIM from CloudShell.ts
  // ---------------------------------------------------------------------------

  /**
   * Bake wind equirect DataTexture from current climate.
   * 512×256 RGBA8. Zero per-texel allocation.
   */
  private _bakeWindTexture(): void {
    const W = 512
    const H = 256
    const data = new Uint8Array(W * H * 4)
    const TWO_PI = 2 * Math.PI
    const { out, dir } = this._bakeScratch

    for (let y = 0; y < H; y++) {
      const lat    = (y / H) * Math.PI - Math.PI / 2
      const cosLat = Math.cos(lat)
      const sinLat = Math.sin(lat)
      for (let x = 0; x < W; x++) {
        const lon = (x / W) * TWO_PI - Math.PI
        // Planet-local Cartesian: +Y = north pole
        dir.set(cosLat * Math.sin(lon), sinLat, cosLat * Math.cos(lon))

        this._climate.windAt(dir, out)

        const i = (y * W + x) * 4
        data[i]     = Math.round((out.x * 0.5 + 0.5) * 255)
        data[i + 1] = Math.round((out.y * 0.5 + 0.5) * 255)
        data[i + 2] = Math.round((out.z * 0.5 + 0.5) * 255)
        data[i + 3] = Math.round(out.speed * 255)
      }
    }

    const tex = new DataTexture(data, W, H, RGBAFormat, UnsignedByteType)
    tex.wrapS           = RepeatWrapping
    tex.wrapT           = ClampToEdgeWrapping
    tex.minFilter       = LinearFilter
    tex.magFilter       = LinearFilter
    tex.generateMipmaps = false
    tex.needsUpdate     = true
    this._windTex = tex
  }

  /**
   * Bake favorability equirect DataTexture from current climate.
   * 512×256 RGBA8, same lat/lon convention.
   *
   * Texel contract (verbatim from CloudShell):
   *   R = 0 (unused)
   *   G = clamp01(moisture)
   *   B = tanh(convergence_raw * 0.25) * 0.5 + 0.5   (CONV_NORM=0.25)
   *   A = 255
   */
  private _bakeFavTexture(): void {
    const W = 512
    const H = 256
    const data = new Uint8Array(W * H * 4)
    const TWO_PI = 2 * Math.PI
    const H_STEP = 0.01

    const { dir, climateSample,
            dEast, dWest, dNorth, dSouth,
            east, north,
            wE_pos, wE_neg, wN_pos, wN_neg } = this._favScratch
    const polar = this._polarAxis

    for (let y = 0; y < H; y++) {
      const lat    = (y / H) * Math.PI - Math.PI / 2
      const cosLat = Math.cos(lat)
      const sinLat = Math.sin(lat)
      for (let x = 0; x < W; x++) {
        const lon = (x / W) * TWO_PI - Math.PI
        dir.set(cosLat * Math.sin(lon), sinLat, cosLat * Math.cos(lon))

        this._climate.sample(dir, 0, climateSample)
        const moisture = climateSample.moisture

        east.copy(polar).cross(dir)
        const eLen = east.length()

        let convergence = 0
        if (eLen >= 1e-5) {
          east.multiplyScalar(1 / eLen)
          north.set(
            dir.y * east.z - dir.z * east.y,
            dir.z * east.x - dir.x * east.z,
            dir.x * east.y - dir.y * east.x,
          )

          dEast.set(
            dir.x + H_STEP * east.x,
            dir.y + H_STEP * east.y,
            dir.z + H_STEP * east.z,
          ).normalize()
          dWest.set(
            dir.x - H_STEP * east.x,
            dir.y - H_STEP * east.y,
            dir.z - H_STEP * east.z,
          ).normalize()
          dNorth.set(
            dir.x + H_STEP * north.x,
            dir.y + H_STEP * north.y,
            dir.z + H_STEP * north.z,
          ).normalize()
          dSouth.set(
            dir.x - H_STEP * north.x,
            dir.y - H_STEP * north.y,
            dir.z - H_STEP * north.z,
          ).normalize()

          this._climate.windAt(dEast,  wE_pos)
          this._climate.windAt(dWest,  wE_neg)
          this._climate.windAt(dNorth, wN_pos)
          this._climate.windAt(dSouth, wN_neg)

          const uEp = (wE_pos.x * east.x + wE_pos.y * east.y + wE_pos.z * east.z) * wE_pos.speed
          const uEm = (wE_neg.x * east.x + wE_neg.y * east.y + wE_neg.z * east.z) * wE_neg.speed
          const vNp = (wN_pos.x * north.x + wN_pos.y * north.y + wN_pos.z * north.z) * wN_pos.speed
          const vNm = (wN_neg.x * north.x + wN_neg.y * north.y + wN_neg.z * north.z) * wN_neg.speed

          const dU_dx = (uEp - uEm) / (2 * H_STEP)
          const dV_dy = (vNp - vNm) / (2 * H_STEP)
          convergence = -(dU_dx + dV_dy)
        }

        const CONV_NORM = 0.25
        const convPacked = Math.tanh(convergence * CONV_NORM) * 0.5 + 0.5

        const i = (y * W + x) * 4
        data[i]     = 0
        data[i + 1] = Math.round(moisture * 255)
        data[i + 2] = Math.round(convPacked * 255)
        data[i + 3] = 255
      }
    }

    const tex = new DataTexture(data, W, H, RGBAFormat, UnsignedByteType)
    tex.wrapS           = RepeatWrapping
    tex.wrapT           = ClampToEdgeWrapping
    tex.minFilter       = LinearFilter
    tex.magFilter       = LinearFilter
    tex.generateMipmaps = false
    tex.needsUpdate     = true
    this._favTex = tex
  }

  // ---------------------------------------------------------------------------
  // Mesh + volumetric raymarch material
  // ---------------------------------------------------------------------------

  private _buildMesh(): void {
    const baseOuterR = this._baseOuterR()
    const outerR     = this._outerR()

    // Sphere geometry at BASE outer radius; current outerR applied via mesh.scale
    this._geo = new SphereGeometry(baseOuterR, 96, 48)

    // ---------------------------------------------------------------------------
    // Capture uniform refs for TSL closures
    // ---------------------------------------------------------------------------
    const uInvRot       = this._uInvRot
    const uScale        = this._uScale
    const uScrollSpeed  = this._uScrollSpeed
    const uTime         = this._uTime
    const uWarp         = this._uWarp
    const uBillow       = this._uBillow
    const uDetail       = this._uDetail
    const uDetailScale  = this._uDetailScale
    const uSoftness     = this._uSoftness
    const uRoundBase    = this._uRoundBase
    const uBillowTop    = this._uBillowTop
    const uCoverage     = this._uCoverage
    const uFavWeight    = this._uFavWeight
    const uMoistWeight  = this._uMoistWeight
    const uConvWeight   = this._uConvWeight
    const uConvGain     = this._uConvGain
    const uItczWeight   = this._uItczWeight
    const uWeatherWeight = this._uWeatherWeight
    const uCellWeight   = this._uCellWeight
    const uSunDir       = this._uSunDir
    const uSigmaT       = this._uSigmaT
    const uLightSteps   = this._uLightSteps
    const uStepCount    = this._uStepCount
    const uStepCap      = this._uStepCap
    const uHgAnisotropy = this._uHgAnisotropy
    const uPowder       = this._uPowder
    const uAmbient      = this._uAmbient
    const uTypeStrength = this._uTypeStrength
    const uOpacity      = this._uOpacity
    const uCloudBase    = this._uCloudBase
    const uCloudThick   = this._uCloudThick
    const uDebugMode    = this._uDebugMode
    const windTex       = this._windTex!
    const favTex        = this._favTex!

    // Constant scalars baked into the graph (do not change after construction)
    const RADIUS_C       = float(this._radius)
    const HEIGHT_SCALE_C = float(this._heightScale)
    const INV_2PI        = float(1 / (2 * Math.PI))
    const INV_PI         = float(1 / Math.PI)
    const EPS            = float(1e-4)
    const PI_4           = float(Math.PI * 4)   // 4π for HG denominator
    // Vertical span (in noise units) the annulus covers — gives the clouds 3D thickness
    // instead of an extruded slab. Kept LOW (1.2): higher makes the density vary in regular
    // vertical SHELLS that, viewed edge-on from the surface horizon, read as stacked
    // lenticular "plates"/contour-lines (the stratified artifact).
    const VERT_NOISE     = float(1.2)

    // three's .d.ts under-types mx_worley_noise_float (caps args at 2), but its runtime
    // overload dispatches by EXACT arg count and the 3D form needs (p, jitter, metric).
    // Wrap with a cast so we can pass all three: jitter=1 (organic), metric=0 (euclidean).
    const worley = mx_worley_noise_float as unknown as
      (p: unknown, jitter: number, metric: number) => ReturnType<typeof float>

    // ---------------------------------------------------------------------------
    // density(p: vec3) -> float [0,1]
    // Returns the cloud fraction at world-frame point p.
    // ---------------------------------------------------------------------------
    type NodeLike = ReturnType<typeof vec3>
    type NodeF    = ReturnType<typeof float>
    // densityCore takes the planet-LOCAL unit direction and the radius |p| directly (no
    // uInvRot), so the compute bake can call it over reconstructed (lon, lat, altitude)
    // grid coordinates. World-space callers wrap it (see `density` below).
    const densityCore = Fn<readonly [NodeLike, NodeF]>(([dLocal, pLen]) => {

      // Equirect UV — must match bake convention exactly
      // u = atan2(d.x, d.z)/(2π)+0.5   v = asin(clamp(d.y,-1,1))/π+0.5
      const u = atan(dLocal.x, dLocal.z).mul(INV_2PI).add(0.5)
      const v = asin(clamp(dLocal.y, -1, 1)).mul(INV_PI).add(0.5)
      const uvCoord = vec2(u, v)

      // Wind texture: xyz = wind dir [-1,1] (UN-normalized), w = speed [0,1]
      const windSample = texture(windTex, uvCoord)
      const windVec    = windSample.xyz.mul(2).sub(1)   // UN-normalized → zero drift at calm
      const speed      = windSample.w

      // Fav texture: g = moisture, b = tanh-packed convergence
      const favSample = texture(favTex, uvCoord)

      // --- Cloud TYPE (convective morphology) -----------------------------------------
      // Continuous stratus↔cumulonimbus scalar from the already-baked climate fields: wind
      // convergence (low-pressure / ITCZ updraft) + moisture + the cloudy latitude bands →
      // towering, denser, anvil-topped convective cloud; calm dry subtropics → flat thin
      // stratus. Drives DENSITY + an anvil bump only (placement still comes from the weather
      // map below). uTypeStrength=0 → cloudType=0 everywhere = EXACTLY the pre-type behaviour.
      // Shared climate terms — also drive the coverage gate below, so computed once here.
      const moistTerm = favSample.g
      const conv      = favSample.b.mul(2).sub(1)
      const convTerm  = saturate(conv.mul(uConvGain))
      const absY      = dLocal.y.abs()                                   // |sin(lat)|
      const itcz      = smoothstep(0.0, 0.22, absY).oneMinus()           // cloudy equator
      const storm     = saturate(smoothstep(0.55, 0.74, absY).sub(smoothstep(0.86, 0.96, absY))) // ~45-60° tracks
      const cloudType = saturate(
        convTerm.mul(0.55).add(moistTerm.mul(0.2)).add(max(itcz, storm).mul(0.35)),
      ).mul(uTypeStrength)

      // Vertical profile: heightFrac = (|p| - innerR) / (outerR - innerR)
      // Clamp multipliers so the cloud shell always stays below the atmosphere (2.5×heightScale).
      const innerMul_v = min(uCloudBase, float(2.4))
      const outerMul_v = min(uCloudBase.add(uCloudThick), float(2.45))
      const innerR_v   = RADIUS_C.add(HEIGHT_SCALE_C.mul(innerMul_v))
      const outerR_v   = RADIUS_C.add(HEIGHT_SCALE_C.mul(outerMul_v))
      const heightFrac = clamp(pLen.sub(innerR_v).div(outerR_v.sub(innerR_v)), 0, 1)
      // profile = smoothstep(0, roundBase, hf) * (1 - smoothstep(1-billowTop, 1, hf))
      const profile = smoothstep(float(0), uRoundBase, heightFrac)
        .mul(smoothstep(float(1), float(1).sub(uBillowTop), heightFrac).oneMinus())

      // Great-circle advection of the sampling DIRECTION along the local wind.
      // The whole cloud field flows coherently over the sphere — masses drift and
      // visibly MERGE, rather than the noise morphing in place and blinking clouds
      // in/out against a static gate ("appearing out of thin air"). Because this is
      // a bounded rotation it never collapses over long sessions (unlike unbounded
      // linear drift in noise space). windDir is guarded against normalize(0) at calm
      // eyes/doldrums — there speed→0 so arc→0 and dAdv = dLocal (no motion), as wanted.
      // NB: coverage/favorability below stays sampled at the STATIC dLocal, so clouds
      // still cluster in climate-fixed humid zones and thin out as they drift into dry
      // air (natural formation/dissipation) instead of being rigidly painted on.
      const windDir = windVec.div(max(length(windVec), float(1e-4)))
      const arc     = uScrollSpeed.mul(uTime).mul(speed)
      const dAdv    = normalize(dLocal.mul(cos(arc)).add(windDir.mul(sin(arc))))

      // Noise sample point. The sample RADIUS grows with altitude through the annulus
      // (`uScale + heightFrac·VERT_NOISE`), so the noise is sampled across the shell's
      // thickness as a true 3D volume — clouds gain top-to-bottom structure instead of
      // being a 2D direction map extruded into flat-topped/flat-bottomed slabs (the
      // "blocky/cartoon" look). A small wind-aligned domain warp keeps the flow soft.
      const rNoise   = uScale.add(heightFrac.mul(VERT_NOISE))
      const p0       = dAdv.mul(rNoise)
      const poleFade = saturate(smoothstep(0.85, 1.0, dLocal.y.abs()).oneMinus())
      const warpN    = mx_noise_float(p0.mul(0.5))
      const warp     = windDir.mul(speed).mul(uWarp).mul(poleFade).mul(warpN)
      const pAdv     = p0.add(warp)

      // Worley (cellular) FBM — inverted so cell CENTRES are lumps. This is the real
      // "cauliflower" cloud structure (the Perlin-Worley basis used by Nubis / Blender
      // cloud materials), a big step up from the old sum-of-|gradient-noise| billow which
      // read as uniform popcorn. jitter=1 (organic placement), metric=0 (euclidean → round
      // blobs, not boxy). Worley is EXPENSIVE (27-cell loop/sample) and runs at every march
      // + light-march step, so only 2 octaves until the 3D density cache lands (which is
      // exactly how the reference affords rich Worley). Weights sum to 1 → output ~[0,1].
      const worleyFBM = worley(pAdv, 1.0, 0).oneMinus().mul(0.65)
        .add(worley(pAdv.mul(2.0), 1.0, 0).oneMinus().mul(0.35))

      // FBM base (Perlin-like connected structure)
      const fbmD = mx_fractal_noise_float(pAdv, 4, 2, 0.5).mul(0.5).add(0.5)

      // Perlin↔Worley blend — uBillow is now the Worley weight: the Worley term gives the
      // puffy cellular lumps, the FBM gives the connected wispy backbone.
      const baseShape = mix(fbmD, worleyFBM, uBillow)

      // Detail erosion — Nubis remap-from-edge instead of a flat subtract. This
      // DILATES dense cores (a fully-dense sample stays dense) and carves only the
      // low-density fringe, giving fluffy cauliflower bases and torn, wispy tops
      // rather than uniformly greying the whole field.
      // 2-octave high-frequency detail in ~[0,1].
      const hiFBM = mx_noise_float(pAdv.mul(uDetailScale)).abs().mul(0.625)
        .add(mx_noise_float(pAdv.mul(uDetailScale).mul(2)).abs().mul(0.375))
      // Height-graded: erode with hiFBM near the base (round billowy bottoms),
      // with its inverse near the top (wispy, sheared tops).
      const modifier = mix(hiFBM, hiFBM.oneMinus(), saturate(heightFrac.mul(5)))
      const erodeAmt = saturate(modifier.mul(uDetail))
      // remapClamp(baseShape, erodeAmt, 1) = saturate((baseShape - erodeAmt)/(1 - erodeAmt)).
      const shape = baseShape.remapClamp(erodeAmt, float(1))

      // ---------------------------------------------------------------------------
      // Coverage gate using favorability (verbatim from CloudShell)
      // ---------------------------------------------------------------------------
      // Regional favorability: moisture + wind convergence (moistTerm/convTerm from the shared
      // block above). Humid / converging air → cloud; dry / DIVERGING (sinking) air → clear —
      // the sinking branch dries continental interiors → deserts like the Sahara are cloud-free.
      const fav = saturate(uMoistWeight.mul(moistTerm).add(uConvWeight.mul(convTerm)))

      // Earth-like LATITUDE circulation bands (absY/itcz/storm from the shared block) — the
      // DOMINANT large-scale pattern: cloudy ITCZ + mid-lat storm tracks, CLEAR subtropical
      // desert belts (~30°) and poles. latSigned ∈ [-1,+1]: +1 in cloud bands, -1 in clear belts.
      const latBand   = max(itcz, storm)
      const latSigned = latBand.sub(0.5).mul(2.0)

      // === Domain-warped LOW-FREQUENCY weather map — THE large-scale organizer ===
      // The fix for "random scatter → continuous weather systems" (per Nubis / iq warping):
      // cloud PLACEMENT comes from a LOW-frequency field that is DOMAIN-WARPED — folded by a
      // noise vector (iq domain warping → flowing organic masses) and dragged by the wind (the
      // vortex wind is curl-like → coverage spirals into fronts/cyclones). The high-freq
      // `shape` above no longer decides placement; it only carves silhouette detail INSIDE the
      // covered zones. uWeatherWeight is the weather-map strength (now the dominant term).
      const W_SCALE = float(1.8)   // low freq → a few weather systems per hemisphere
      const warpVec = vec3(
        mx_noise_float(dLocal.mul(W_SCALE)),
        mx_noise_float(dLocal.mul(W_SCALE).add(19.1)),
        mx_noise_float(dLocal.mul(W_SCALE).add(43.7)),
      ).mul(0.28).add(windDir.mul(speed).mul(0.12))   // noise folding kept; directional wind
      const dWeather      = normalize(dLocal.add(warpVec))   // stretch cut hard (it smeared clouds into streaks)
      // Low-freq fbm at the warped direction → smooth flowing organized systems; sharpened
      // into big cloudy/clear REGIONS (clear deserts vs cloudy storm zones).
      const weatherRaw    = mx_fractal_noise_float(dWeather.mul(W_SCALE), 4, 2.0, 0.55).mul(0.5).add(0.5)
      const weatherShaped = smoothstep(float(0.42), float(0.58), weatherRaw)

      // Large-scale Voronoi WEATHER CELLS — tessellate the cloudy regions into convection-
      // cell-sized cells (cloudy near the cell CENTRE, clear lanes at the BOUNDARIES = fronts)
      // → the open/closed-convection-cell look. Sampled at a LIGHTLY-warped direction so the
      // cells flow with the wind but stay recognizably cellular (the strong dWeather warp
      // would scramble them). jitter=1 → irregular organic cells, not a visible grid.
      const dCell  = normalize(dLocal.add(warpVec.mul(0.3)))
      const cellF1 = worley(dCell.mul(1.5), 1.0, 0)                       // 0 at centres → high at edges (~25 cells)
      const cells  = smoothstep(float(0.3), float(0.55), cellF1).oneMinus()   // 1 over cell body, 0 in boundary lanes
      // Blend cells INTO the weather map (so cells only appear where the map is cloudy → variety
      // comes from the map, cellular structure from Voronoi). uCellWeight: 0 = blobby, 1 = cellular.
      const weatherCells = mix(weatherShaped, weatherShaped.mul(cells), uCellWeight)

      // Coverage threshold (lower = more cloud). The warped weather map DOMINATES (big clear
      // regions + continuous flowing systems); latitude bands add the desert belts; moisture /
      // convergence add regional detail.
      const t0raw    = uCoverage
        .sub(uWeatherWeight.mul(weatherCells.sub(0.5)).mul(2.0))
        .sub(uItczWeight.mul(latSigned))
        .sub(uFavWeight.mul(fav.sub(0.5)))
      // Cap the threshold so even the CLEAREST regions still let the highest shape peaks
      // poke through as small scattered SPECKS — fair-weather clouds dotting an otherwise
      // clear sky (Earth has these everywhere, not only in the big organized systems).
      const t0       = min(t0raw, float(0.93))
      const coverage = saturate(smoothstep(t0, t0.add(uSoftness), shape))

      // Fine density MOTTLE — 2-octave high-freq noise sampled at the advected point pAdv
      // (so it drifts WITH the cloud), strongly varying density within clouds → the
      // colour/opacity cycles between bright white (dense), grey (thinner) and translucent
      // (thinnest holes) so NOTHING is flat pure-white-opaque. Pairs with the density-tied
      // `texVar` colour term. Wide range (0.3–1.0) = lots of internal variation.
      const mottle = mx_noise_float(pAdv.mul(13.0)).mul(0.5).add(0.5).mul(0.65)
        .add(mx_noise_float(pAdv.mul(31.0)).mul(0.5).add(0.5).mul(0.35))   // [0,1]

      // Convective shaping (cloud TYPE): cumulonimbus pile higher (an anvil bump in the upper
      // band, ADDED proportional to the existing profile so it never paints cloud into clear
      // sky) and read optically thicker; stratus stays thin. Both terms vanish at cloudType=0.
      const anvil       = smoothstep(float(0.72), float(0.98), heightFrac).mul(cloudType).mul(0.4)
      const profileT    = saturate(profile.add(profile.mul(anvil)))
      const typeDensity = mix(float(1.0), float(1.5), cloudType)

      // Final density = coverage gate × vertical profile × mottle × convective thickness.
      return coverage.mul(profileT).mul(mix(float(0.3), float(1.0), mottle)).mul(typeDensity)
    })

    // World-space density used by the march: world point → planet-LOCAL dir + radius → cloud fraction.
    const density = Fn<readonly [NodeLike]>(([p]) =>
      densityCore(normalize(uInvRot.mul(p)), length(p)),
    )

    // ---------------------------------------------------------------------------
    // Ray-vs-sphere intersector — returns vec2(tNear, tFar) for sphere of radius R.
    // Discriminant < 0 → no hit → tNear = tFar = (-b) so caller checks tFar > tNear.
    // Packed: x = -b - sqrt(max(disc,0)), y = -b + sqrt(max(disc,0)).
    // When disc < 0, sqrt(0) yields x = y = -b (caller will detect tFar <= tEnter for miss).
    // ---------------------------------------------------------------------------
    type NodeLike3 = ReturnType<typeof vec3>
    type NodeLikeF = ReturnType<typeof float>
    const intersectSphereFn = Fn<readonly [NodeLike3, NodeLike3, NodeLikeF]>(([ro, rd, R]) => {
      const b     = dot(ro, rd)
      const c     = dot(ro, ro).sub(R.mul(R))
      const disc  = b.mul(b).sub(c)
      const sqrtD = sqrt(max(disc, float(0)))
      return vec2(b.negate().sub(sqrtD), b.negate().add(sqrtD))
    })

    // Discriminant helper — whether a sphere R is actually hit
    const discFn = Fn<readonly [NodeLike3, NodeLike3, NodeLikeF]>(([ro, rd, R]) => {
      const b = dot(ro, rd)
      const c = dot(ro, ro).sub(R.mul(R))
      return b.mul(b).sub(c)
    })

    // Henyey-Greenstein phase, factored so the raymarch can blend two lobes
    // (forward for the silver lining, weak backward so anti-sun faces aren't black).
    // p(μ,g) = (1 - g²) / (4π · (1 + g² - 2g·μ)^1.5)
    const hgFn = Fn<readonly [NodeLikeF, NodeLikeF]>(([mu, g]) => {
      const g2    = g.mul(g)
      const denom = pow(float(1).add(g2).sub(g.mul(2).mul(mu)), float(1.5)).mul(PI_4)
      return float(1).sub(g2).div(denom)
    })

    // Cornette-Shanks phase — energy-conserving Mie-like forward lobe. Sharper, more physical
    // silver lining than plain HG (it adds the (1+μ²) Rayleigh-style factor). Used for the
    // forward lobe; the weak back lobe stays HG. 3/(8π) = 1.5/(4π) so it reuses PI_4.
    // p(μ,g) = 3/(8π) · (1-g²)/(2+g²) · (1+μ²) / (1+g²-2gμ)^1.5
    const csFn = Fn<readonly [NodeLikeF, NodeLikeF]>(([mu, g]) => {
      const g2    = g.mul(g)
      const numer = float(1).sub(g2).mul(float(1).add(mu.mul(mu)))
      const denom = pow(float(1).add(g2).sub(g.mul(2).mul(mu)), float(1.5)).mul(float(2).add(g2))
      return float(1.5).div(PI_4).mul(numer).div(denom)
    })

    // ---------------------------------------------------------------------------
    // Main fragment Fn — full volumetric raymarch producing vec4(rgb, alpha)
    // ---------------------------------------------------------------------------
    const raymarchFn = Fn(() => {
      // Ray in world frame (Planet Group sits at world origin)
      const ro = vec3(cameraPosition)
      const rd = normalize(vec3(positionWorld).sub(ro))

      // Annulus bounds — multipliers capped to keep the shell below atmosphere (renderOrder 5).
      const innerMul_f = min(uCloudBase, float(2.4))
      const outerMul_f = min(uCloudBase.add(uCloudThick), float(2.45))
      const innerR_f   = RADIUS_C.add(HEIGHT_SCALE_C.mul(innerMul_f))
      const outerR_f   = RADIUS_C.add(HEIGHT_SCALE_C.mul(outerMul_f))

      // Discriminants — needed to detect valid hits
      const discOuter = discFn(ro, rd, outerR_f)
      const discInner = discFn(ro, rd, innerR_f)

      // Sphere intersections
      const outerHit = intersectSphereFn(ro, rd, outerR_f)
      const innerHit = intersectSphereFn(ro, rd, innerR_f)

      const outerT0 = outerHit.x
      const outerT1 = outerHit.y
      const innerT0 = innerHit.x
      const innerT1 = innerHit.y

      // Camera position relative to annulus
      const camDist      = length(ro)
      const inOuterSphere = camDist.lessThanEqual(outerR_f)
      const inInnerSphere = camDist.lessThan(innerR_f)

      // ---------------------------------------------------------------------------
      // Camera case selection (3 cases):
      //   (A) cam OUTSIDE outerR  → tEnter = outerT0
      //   (B) cam BETWEEN layers  → tEnter = 0
      //   (C) cam INSIDE innerR   → tEnter = innerT1 (skip inner sphere)
      //
      // tExit:
      //   default = outerT1 (ray exits outer sphere)
      //   if inner disc > 0 AND innerT0 > tEnter AND NOT inInner: use innerT0
      //   (stops at inner sphere surface from outside or between layers)
      // ---------------------------------------------------------------------------

      // tEnter: outside-outer=A, between=B, inside-inner=C
      // build as: if inOuterSphere → (if inInnerSphere → C else B) else A
      const tEnterRaw = select(
        inOuterSphere,
        select(inInnerSphere, innerT1, float(0)),
        outerT0,
      )
      const tEnter = max(tEnterRaw, float(0))

      // tExit: normally outerT1; if inner sphere is hit and innerT0 > tEnter (and not inside inner)
      const useInnerExit = discInner.greaterThan(float(0))
        .and(innerT0.greaterThan(tEnter))
        .and(inInnerSphere.not())
      // No scene-depth clamp: cloudBase ≥ heightScale·1.2 (~14.4 km) sits above all terrain
      // relief (~12 km), so clouds are always in front — terrain occlusion is a no-op here
      // (tried it: changed nothing visible) and not worth the log-depth decode.
      const tExit = select(useInnerExit, innerT0, outerT1)

      // No valid annulus segment: outer sphere not hit, or segment is degenerate
      const noHit = discOuter.lessThanEqual(float(0)).or(tExit.lessThanEqual(tEnter))

      // ---------------------------------------------------------------------------
      // Raymarch — dynamic Loop driven by _uStepCount.
      // Accumulators are declared here so they're in scope after the If block.
      // The entire march is skipped when noHit — rays that miss the annulus do
      // zero loop work (no texture samples, no light-march taps). When noHit the
      // accumulators stay at their init values (transmittance=1, scattered=0),
      // which produces opacity=0 (fully transparent) in the final output.
      // ---------------------------------------------------------------------------
      const transmittance = float(1.0).toVar()
      const scatteredR    = float(0.0).toVar()
      const scatteredG    = float(0.0).toVar()
      const scatteredB    = float(0.0).toVar()

      If(noHit.not(), () => {
        // Fine (in-cloud) step: path/count, CAPPED to ~10% of the annulus THICKNESS so
        // horizon-grazing rays (long near-tangent path through the thin shell) stay finely
        // sampled instead of divided into coarse chunks (that coarse sampling, dithered,
        // was the heavy grain). Grazing rays may not reach tExit within the budget; the
        // early-out covers dense cases and any far clip is graceful (distant clouds fade).
        const pathLen  = tExit.sub(tEnter)
        const fineStep = min(pathLen.div(float(uStepCount)), outerR_f.sub(innerR_f).mul(uStepCap))
        // Empty-space scout step. Skipping empty air is the MAIN perf lever — most of a ray
        // through a ~22%-coverage shell is empty; a full march of every step is far too
        // expensive (it overloads the GPU and hangs the page). Scout big-steps until it
        // finds cloud, then refines.
        const bigStep  = fineStep.mul(2.0)

        // Sun color: warm white
        const SUN_R = float(1.00)
        const SUN_G = float(0.95)
        const SUN_B = float(0.85)
        // Ambient sky colour, altitude-graded: cool blue-grey skylight up high (SKY_*) →
        // darker, earthier ground-bounce down low (GND_*), mixed by the sample's heightFrac
        // in the annulus. Cloud undersides read darker/warmer, tops bluer — tying the cloud
        // into the sky instead of a flat constant fill.
        const SKY_R = float(0.45)
        const SKY_G = float(0.52)
        const SKY_B = float(0.72)
        const GND_R = float(0.30)
        const GND_G = float(0.32)
        const GND_B = float(0.36)

        // Dual-lobe Henyey-Greenstein: a forward lobe (uHgAnisotropy) gives the
        // silver lining toward the sun; a fixed weak back lobe (g=-0.3) keeps faces
        // turned away from the sun gently lit instead of crushing to black.
        const cosTheta = dot(rd, uSunDir)
        const hgPhase  = mix(
          csFn(cosTheta, uHgAnisotropy),     // forward: energy-conserving Cornette-Shanks
          hgFn(cosTheta, float(-0.3)),       // weak back lobe (HG) so anti-sun faces aren't black
          float(0.5),
        )
        // Powder view-gate: 1 on the anti-sun hemisphere, 0 toward the sun, so the
        // dark-edge term stops fighting the silver lining (uPowder = its strength).
        const powderGate = saturate(cosTheta.mul(-0.5).add(0.5))

        // Light-march step: half annulus / nLightSteps
        const annulusHalf = outerR_f.sub(innerR_f).mul(0.5)
        const lightStep   = annulusHalf.div(float(uLightSteps))

        // Blue-noise (interleaved gradient noise) ray-start dither — breaks step banding.
        // Amplitude 0.3. NB: screen-locked/static — it redistributes undersampling error but
        // cannot reduce it, so a faint stable grain remains. The categorical fix is temporal
        // accumulation (offscreen history + reproject), which is NOT implemented yet; half-res
        // (CloudCompositor) softens it cheaply in the meantime.
        const jitter  = interleavedGradientNoise(screenCoordinate.xy)
        const tCur    = tEnter.add(jitter.mul(0.3).mul(fineStep)).toVar()
        const refined = float(0.0).toVar()   // 0 = scouting (big steps), 1 = fine march

        Loop(uStepCount, () => {
          If(tCur.greaterThanEqual(tExit), () => { Break() })

          const p = ro.add(rd.mul(tCur))

          If(refined.lessThan(0.5), () => {
            // Scouting: big-step through empty air. On the FIRST hit, back up one big step
            // (clamped to tEnter) and switch to fine steps so the near face isn't quantized
            // to the coarse scout grid (that read blocky).
            const dScout = density(p)
            If(dScout.greaterThan(EPS), () => {
              tCur.assign(max(tCur.sub(bigStep), tEnter))
              refined.assign(1)
            }).Else(() => {
              tCur.addAssign(bigStep)
            })
          }).Else(() => {
            // Fine march: full density; the expensive light-march only fires inside cloud.
            const d = density(p)

            If(d.greaterThan(EPS), () => {
              // Light march toward sun — accumulate optical depth
              const sumL = float(0.0).toVar()
              Loop(uLightSteps, ({ i: j }: { i: ReturnType<typeof float> }) => {
                const lp = p.add(uSunDir.mul(lightStep.mul(float(j).add(0.5))))
                sumL.addAssign(density(lp))
              })

              // Beer-powder dark-edge term, view-gated toward the anti-sun hemisphere.
              const powderRaw = float(1).sub(exp(d.negate().mul(2)))
              const powder    = mix(float(1), powderRaw, powderGate.mul(uPowder))

              // Multi-octave "multiscatter" Beer (Wrenninge/Hillaire): approximate multiple
              // scattering by summing the single-scatter term over a few octaves, each dimmer
              // (bⁿ), penetrating deeper (aⁿ softens extinction) and more isotropic (cⁿ
              // flattens the phase). Thick cloud interiors GLOW softly instead of crushing to
              // a flat value — the key to a voluminous, non-flat look — while thin edges stay
              // dark (powder). Costs 3 cheap iters (no extra density samples → no perf risk).
              const lum = float(0.0).toVar()
              Loop(3, ({ i: n }: { i: ReturnType<typeof float> }) => {
                const kN   = pow(float(0.5), float(n))                              // aⁿ = bⁿ = cⁿ
                const extN = exp(sumL.negate().mul(uSigmaT).mul(lightStep).mul(kN)) // aⁿ on extinction
                const phN  = mix(float(1).div(PI_4), hgPhase, kN)                   // cⁿ: phase → isotropic
                lum.addAssign(kN.mul(extN).mul(phN))                                // bⁿ contribution
              })
              const lit = lum.mul(powder)

              // Cloud TEXTURE — tie brightness to the local density so the noisy Worley
              // structure shows in the colour: thin/wispy samples render greyer/dimmer,
              // dense cores stay bright white. This breaks up the flat "opaque white" look
              // and reads as a textured volume. (Density varies with the cloud's own noise
              // and drifts with it, so the texture sticks to the cloud; works with the cache.)
              const texVar = mix(float(0.5), float(1.0), saturate(d.mul(2.2)))

              // Altitude-graded ambient colour: ground-tint below → sky-tint above.
              const hfA  = saturate(length(p).sub(innerR_f).div(outerR_f.sub(innerR_f)))
              const ambR = mix(GND_R, SKY_R, hfA)
              const ambG = mix(GND_G, SKY_G, hfA)
              const ambB = mix(GND_B, SKY_B, hfA)

              // Composite: lit sun + ambient sky, scaled by density × texture.
              const sR = lit.mul(SUN_R).add(uAmbient.mul(ambR)).mul(d).mul(texVar)
              const sG = lit.mul(SUN_G).add(uAmbient.mul(ambG)).mul(d).mul(texVar)
              const sB = lit.mul(SUN_B).add(uAmbient.mul(ambB)).mul(d).mul(texVar)

              scatteredR.addAssign(transmittance.mul(sR).mul(fineStep).mul(uSigmaT))
              scatteredG.addAssign(transmittance.mul(sG).mul(fineStep).mul(uSigmaT))
              scatteredB.addAssign(transmittance.mul(sB).mul(fineStep).mul(uSigmaT))

              // Beer extinction along view ray
              transmittance.mulAssign(exp(d.negate().mul(fineStep).mul(uSigmaT)))
            })

            tCur.addAssign(fineStep)
          })

          // Early-out when transmittance is effectively zero
          If(transmittance.lessThan(0.01), () => { Break() })
        })
      })

      // transmittance=1/scattered=0 when noHit → opacity=0 (transparent). No select needed.
      const opacity = saturate(float(1).sub(transmittance)).mul(uOpacity)

      return vec4(scatteredR, scatteredG, scatteredB, opacity)
    })

    // ---------------------------------------------------------------------------
    // Debug "cloud map" — flat coverage heatmap + contour outlines (the 'cloud' view).
    // One density sample at the annulus midpoint in the fragment's direction (no march).
    // Filled heatmap (clear = dark navy → cloudy = bright) + yellow iso-contour outlines
    // (cloud silhouette at 0.5 + faint 0.2/0.8 levels), so the formation/topology is legible.
    // ---------------------------------------------------------------------------
    const debugMapFn = Fn(() => {
      const dir      = normalize(vec3(positionWorld))
      const innerMul = min(uCloudBase, float(2.4))
      const outerMul = min(uCloudBase.add(uCloudThick), float(2.45))
      const innerR_d = RADIUS_C.add(HEIGHT_SCALE_C.mul(innerMul))
      const outerR_d = RADIUS_C.add(HEIGHT_SCALE_C.mul(outerMul))
      const rMid     = innerR_d.add(outerR_d).mul(0.5)
      const cov      = density(dir.mul(rMid))

      // Filled heatmap.
      const fill = mix(vec3(0.04, 0.07, 0.16), vec3(0.85, 0.90, 1.0), saturate(cov.mul(1.4)))

      // Iso-contour outlines — constant screen-space width via fwidth(cov).
      const g    = max(fwidth(cov), float(1e-5))
      const edge = smoothstep(float(0), g.mul(2.0), cov.sub(0.5).abs()).oneMinus()
      const lo   = smoothstep(float(0), g.mul(1.3), cov.sub(0.2).abs()).oneMinus().mul(0.45)
      const hi   = smoothstep(float(0), g.mul(1.3), cov.sub(0.8).abs()).oneMinus().mul(0.45)
      const contour = max(edge, max(lo, hi))

      return vec4(mix(fill, vec3(1.0, 0.88, 0.25), contour), float(1.0))
    })

    // Branch the fragment between the volumetric raymarch and the flat debug map (uniform
    // control flow → only one path runs at runtime).
    const fragResult = Fn(() => {
      const out = vec4(0.0, 0.0, 0.0, 0.0).toVar()
      If(uDebugMode.greaterThan(0.5), () => {
        out.assign(debugMapFn())
      }).Else(() => {
        out.assign(raymarchFn())
      })
      return out
    })()
    const colorNode   = fragResult.xyz
    const opacityNode = fragResult.w

    // --- Material ---
    const mat = new MeshBasicNodeMaterial()
    mat.colorNode   = colorNode
    mat.opacityNode = opacityNode
    mat.transparent  = true
    mat.depthWrite   = false
    mat.blending     = NormalBlending
    mat.side         = DoubleSide   // visible from inside the shell (walk mode)
    this._mat = mat

    // --- Mesh ---
    const mesh = new Mesh(this._geo, this._mat)
    mesh.scale.setScalar(outerR / baseOuterR)
    mesh.renderOrder   = 10
    mesh.frustumCulled = false
    mesh.visible       = this._visible
    this._mesh = mesh

    this._scene.add(this._mesh)
  }

  /** Remove mesh from scene and dispose all GPU resources. Safe to call multiple times. */
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
    if (this._windTex !== null) {
      this._windTex.dispose()
      this._windTex = null
    }
    if (this._favTex !== null) {
      this._favTex.dispose()
      this._favTex = null
    }
    this._visible = false
  }
}
