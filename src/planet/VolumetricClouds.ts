import {
  ClampToEdgeWrapping,
  DataTexture,
  DoubleSide,
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
  atan2,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  length,
  max,
  min,
  mix,
  mx_fractal_noise_float,
  mx_noise_float,
  normalize,
  positionWorld,
  pow,
  saturate,
  select,
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
  private readonly _uCoverage    = uniform(0.22)
  private readonly _uScrollSpeed = uniform(0.08)
  private readonly _uScale       = uniform(6)
  private readonly _uWarp        = uniform(0.0)
  private readonly _uOpacity     = uniform(0.9)
  private readonly _uTime        = uniform(0)

  private readonly _uBillow      = uniform(0.8)
  private readonly _uDetail      = uniform(0.2)
  private readonly _uSoftness    = uniform(0.3)
  /** Kept for API compatibility. At 0: fully transparent; at 1: full raymarch. */
  private readonly _uVolume      = uniform(0.6)

  private readonly _uFavWeight   = uniform(0.3)
  private readonly _uMoistWeight = uniform(0.6)
  private readonly _uConvWeight  = uniform(0.5)
  private readonly _uConvGain    = uniform(2.0)
  private readonly _uItczWeight  = uniform(0.05)

  // ---------------------------------------------------------------------------
  // New volumetric uniforms
  // ---------------------------------------------------------------------------

  /** Cloud layer base height as a multiple of heightScale above surface. */
  private readonly _uCloudBase    = uniform(1.2)
  /** Cloud layer thickness as a multiple of heightScale. */
  private readonly _uCloudThick   = uniform(0.6)
  /** Extinction / density coefficient σ_T. */
  private readonly _uSigmaT       = uniform(4.0)
  /** Primary ray step count (dynamic Loop — live slider). */
  private readonly _uStepCount    = uniform(40, 'int')
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
  /** Ambient light contribution weight. */
  private readonly _uAmbient      = uniform(0.25)

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
  setStepCount(v: number):    void { this._uStepCount.value    = Math.max(1, Math.round(v)) }
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

    // camWorldPos is intentionally unused — cameraPosition built-in is preferred.
    void camWorldPos
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
    const uSunDir       = this._uSunDir
    const uSigmaT       = this._uSigmaT
    const uLightSteps   = this._uLightSteps
    const uStepCount    = this._uStepCount
    const uHgAnisotropy = this._uHgAnisotropy
    const uPowder       = this._uPowder
    const uAmbient      = this._uAmbient
    const uOpacity      = this._uOpacity
    const uCloudBase    = this._uCloudBase
    const uCloudThick   = this._uCloudThick
    const windTex       = this._windTex!
    const favTex        = this._favTex!

    // Constant scalars baked into the graph (do not change after construction)
    const RADIUS_C       = float(this._radius)
    const HEIGHT_SCALE_C = float(this._heightScale)
    const INV_2PI        = float(1 / (2 * Math.PI))
    const INV_PI         = float(1 / Math.PI)
    const EPS            = float(1e-4)
    const PI_4           = float(Math.PI * 4)   // 4π for HG denominator
    const NORM_BILLOW    = float(1 / 1.875)     // normalization for 4-octave billow

    // ---------------------------------------------------------------------------
    // density(p: vec3) -> float [0,1]
    // Returns the cloud fraction at world-frame point p.
    // ---------------------------------------------------------------------------
    type NodeLike = ReturnType<typeof vec3>
    const densityFn = Fn<readonly [NodeLike]>(([p]) => {

      // Transform to planet-local for direction UV (planet spins with _uInvRot)
      const pLocal = uInvRot.mul(p)
      const dLocal = normalize(pLocal)

      // Equirect UV — must match bake convention exactly
      // u = atan2(d.x, d.z)/(2π)+0.5   v = asin(clamp(d.y,-1,1))/π+0.5
      const u = atan2(dLocal.x, dLocal.z).mul(INV_2PI).add(0.5)
      const v = asin(clamp(dLocal.y, -1, 1)).mul(INV_PI).add(0.5)
      const uvCoord = vec2(u, v)

      // Wind texture: xyz = wind dir [-1,1] (UN-normalized), w = speed [0,1]
      const windSample = texture(windTex, uvCoord)
      const windVec    = windSample.xyz.mul(2).sub(1)   // UN-normalized → zero drift at calm
      const speed      = windSample.w

      // Fav texture: g = moisture, b = tanh-packed convergence
      const favSample = texture(favTex, uvCoord)

      // Vertical profile: heightFrac = (|p| - innerR) / (outerR - innerR)
      // Clamp multipliers so the cloud shell always stays below the atmosphere (2.5×heightScale).
      const innerMul_v = min(uCloudBase, float(2.4))
      const outerMul_v = min(uCloudBase.add(uCloudThick), float(2.45))
      const innerR_v   = RADIUS_C.add(HEIGHT_SCALE_C.mul(innerMul_v))
      const outerR_v   = RADIUS_C.add(HEIGHT_SCALE_C.mul(outerMul_v))
      const pLen       = length(p)
      const heightFrac = clamp(pLen.sub(innerR_v).div(outerR_v.sub(innerR_v)), 0, 1)
      // profile = smoothstep(0, roundBase, hf) * (1 - smoothstep(1-billowTop, 1, hf))
      const profile = smoothstep(float(0), uRoundBase, heightFrac)
        .mul(smoothstep(float(1), float(1).sub(uBillowTop), heightFrac).oneMinus())

      // Wind-advected noise point
      const p0       = dLocal.mul(uScale)
      const drift    = windVec.mul(speed).mul(uScrollSpeed.mul(uTime))
      const poleFade = saturate(smoothstep(0.85, 1.0, dLocal.y.abs()).oneMinus())
      const warpN    = mx_noise_float(p0.mul(0.5))
      const warp     = windVec.mul(speed).mul(uWarp).mul(poleFade).mul(warpN)
      const pAdv     = p0.add(drift).add(warp)

      // Billow: 4-octave sum of |mx_noise_float| — puffy cauliflower
      const billow = mx_noise_float(pAdv).abs()
        .add(mx_noise_float(pAdv.mul(2)).abs().mul(0.5))
        .add(mx_noise_float(pAdv.mul(4)).abs().mul(0.25))
        .add(mx_noise_float(pAdv.mul(8)).abs().mul(0.125))
        .mul(NORM_BILLOW)

      // FBM base
      const fbmD = mx_fractal_noise_float(pAdv, 4, 2, 0.5).mul(0.5).add(0.5)

      // Blend billow vs FBM, then apply detail erosion
      const baseShape = mix(fbmD, billow, uBillow)
      const detailN   = mx_noise_float(pAdv.mul(uDetailScale)).abs()
      const eroded    = saturate(baseShape.sub(detailN.mul(uDetail)))

      // ---------------------------------------------------------------------------
      // Coverage gate using favorability (verbatim from CloudShell)
      // ---------------------------------------------------------------------------
      const moistTerm = favSample.g
      const conv      = favSample.b.mul(2).sub(1)
      const convTerm  = saturate(conv.mul(uConvGain))

      const absY      = dLocal.y.abs()
      const itczPart  = smoothstep(0.0, 0.25, absY).oneMinus().mul(0.6)
      const stormPart = smoothstep(0.45, 0.6, absY).sub(smoothstep(0.7, 0.85, absY)).mul(0.4)
      const itczTerm  = itczPart.add(stormPart)

      const fav = saturate(
        uMoistWeight.mul(moistTerm)
          .add(uConvWeight.mul(convTerm))
          .add(uItczWeight.mul(itczTerm))
      )
      const t0       = uCoverage.sub(uFavWeight.mul(fav.sub(0.5)))
      const coverage = saturate(smoothstep(t0, t0.add(uSoftness), eroded))

      // Final density = coverage gate × vertical profile
      return coverage.mul(profile)
    })

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
      const tExit0 = select(useInnerExit, innerT0, outerT1)

      // Depth occlusion: skip the viewportLinearDepth clamp.
      // The renderer uses logarithmicDepthBuffer:true, so terrain writes log-encoded
      // depth. viewportLinearDepth decodes via the standard perspective inverse —
      // the two don't compose, making tScene garbage and erasing most clouds.
      // We use tExit0 directly. This is correct: cloudBase >= heightScale*1.2 (~14.4 km)
      // sits above all terrain relief (~12 km), so clouds-always-in-front is right
      // from orbit and from the surface looking up.
      const tExit = tExit0

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
        const stepLen = tExit.sub(tEnter).div(float(uStepCount))

        // Sun color: warm white
        const SUN_R = float(1.00)
        const SUN_G = float(0.95)
        const SUN_B = float(0.85)
        // Sky ambient: cool blue-grey
        const SKY_R = float(0.45)
        const SKY_G = float(0.52)
        const SKY_B = float(0.72)

        // Henyey-Greenstein phase: (1 - g²) / (4π · (1 + g² - 2g·cosθ)^1.5)
        const cosTheta = dot(rd, uSunDir)
        const g        = uHgAnisotropy
        const g2       = g.mul(g)
        const hgBase   = float(1).add(g2).sub(g.mul(2).mul(cosTheta))
        const hgDenom  = pow(hgBase, float(1.5)).mul(PI_4)
        const hgPhase  = float(1).sub(g2).div(hgDenom)

        // Light-march step: half annulus / nLightSteps
        const annulusHalf = outerR_f.sub(innerR_f).mul(0.5)
        const lightStep   = annulusHalf.div(float(uLightSteps))

        Loop(uStepCount, ({ i }: { i: ReturnType<typeof float> }) => {
          const t = tEnter.add(float(i).add(0.5).mul(stepLen))
          const p = ro.add(rd.mul(t))

          const d = densityFn(p)

          If(d.greaterThan(EPS), () => {
            // Light march toward sun — accumulate optical depth
            const sumL = float(0.0).toVar()
            Loop(uLightSteps, ({ i: j }: { i: ReturnType<typeof float> }) => {
              const lp = p.add(uSunDir.mul(lightStep.mul(float(j).add(0.5))))
              sumL.addAssign(densityFn(lp))
            })

            // Beer extinction along light path
            const lightT = exp(sumL.negate().mul(uSigmaT).mul(lightStep))

            // Beer-powder dual lobe: powder = 1 - exp(-d * 2 * uPowder)
            const powder = float(1).sub(exp(d.negate().mul(2).mul(uPowder)))

            // Lit sample contribution = lightTransmittance × HG × powder
            const lit = lightT.mul(hgPhase).mul(powder)

            // Composite: lit sun + ambient sky, scaled by density
            const sR = lit.mul(SUN_R).add(uAmbient.mul(SKY_R)).mul(d)
            const sG = lit.mul(SUN_G).add(uAmbient.mul(SKY_G)).mul(d)
            const sB = lit.mul(SUN_B).add(uAmbient.mul(SKY_B)).mul(d)

            scatteredR.addAssign(transmittance.mul(sR).mul(stepLen).mul(uSigmaT))
            scatteredG.addAssign(transmittance.mul(sG).mul(stepLen).mul(uSigmaT))
            scatteredB.addAssign(transmittance.mul(sB).mul(stepLen).mul(uSigmaT))

            // Beer extinction along view ray
            transmittance.mulAssign(exp(d.negate().mul(stepLen).mul(uSigmaT)))
          })

          // Early-out when transmittance is effectively zero
          If(transmittance.lessThan(0.01), () => { Break() })
        })
      })

      // transmittance=1/scattered=0 when noHit → opacity=0 (transparent). No select needed.
      const opacity = saturate(float(1).sub(transmittance)).mul(uOpacity)

      return vec4(scatteredR, scatteredG, scatteredB, opacity)
    })

    // Split the vec4 output into colorNode + opacityNode
    const fragResult  = raymarchFn()
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
