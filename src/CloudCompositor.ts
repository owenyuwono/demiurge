import {
  AddEquation,
  Camera,
  CustomBlending,
  HalfFloatType,
  LinearFilter,
  Mesh,
  OneFactor,
  OneMinusSrcAlphaFactor,
  RGBAFormat,
  RenderTarget,
  SRGBColorSpace,
  Scene,
} from 'three'
import { MeshBasicNodeMaterial, QuadMesh, WebGPURenderer } from 'three/webgpu'
import { screenUV, texture } from 'three/tsl'

// ---------------------------------------------------------------------------
// CloudCompositor — half-resolution offscreen cloud pass (Part 2a).
//
// Renders the volumetric cloud shell to a HALF-RES render target in its own pass, then
// composites it over the main scene with a fullscreen quad. Quartering the cloud pixel count
// frees GPU budget so the per-ray step count can be raised (less grain), and the bilinear
// upsample softens residual grain. This is the offscreen FOUNDATION the temporal-accumulation
// pass (2b) builds on (history buffer + reproject).
//
// The DEFAULT render path is provably untouched: main.ts only calls render() here when the
// `cloud half-res` toggle is on. Cloud isolation uses a render LAYER set on the cloud mesh
// only DURING this method and restored to layer 0 before returning — so when the toggle is
// off the cloud renders normally on layer 0 in the single main pass. No global layer state.
// ---------------------------------------------------------------------------

const CLOUD_PASS_LAYER = 1

export class CloudCompositor {
  private readonly _rt: RenderTarget
  private readonly _mat: MeshBasicNodeMaterial
  private readonly _quad: QuadMesh
  private readonly _scale = 0.5

  constructor(width: number, height: number) {
    this._rt = new RenderTarget(
      Math.max(1, Math.floor(width * this._scale)),
      Math.max(1, Math.floor(height * this._scale)),
      {
        format:        RGBAFormat,
        type:          HalfFloatType,
        minFilter:     LinearFilter,   // bilinear upsample on composite
        magFilter:     LinearFilter,
        // Store display-encoded pixels; sampling auto-decodes so the canvas re-encode
        // round-trips → the composite blends in the same display space as the in-pass cloud.
        colorSpace:    SRGBColorSpace,
        depthBuffer:   true,
        stencilBuffer: false,
      },
    )

    // Composite quad: draw the (premultiplied) cloud RT over the already-rendered canvas.
    // The cloud renders NormalBlending over a transparent clear, so the RT holds premultiplied
    // RGB + coverage alpha → premultiplied "over": screen = rt.rgb + screen·(1 − rt.a).
    const sample = texture(this._rt.texture, screenUV)
    const mat = new MeshBasicNodeMaterial()
    mat.colorNode     = sample.rgb
    mat.opacityNode   = sample.a
    mat.transparent   = true
    mat.depthTest     = false
    mat.depthWrite    = false
    mat.toneMapped    = false          // RT is already tone-mapped; don't re-apply
    mat.blending      = CustomBlending
    mat.blendEquation = AddEquation
    mat.blendSrc      = OneFactor               // premultiplied src
    mat.blendDst      = OneMinusSrcAlphaFactor
    this._mat  = mat
    this._quad = new QuadMesh(mat)
  }

  /** Resize the half-res target. Call from the window resize handler. */
  setSize(width: number, height: number): void {
    this._rt.setSize(
      Math.max(1, Math.floor(width * this._scale)),
      Math.max(1, Math.floor(height * this._scale)),
    )
  }

  /**
   * Three-pass render: (1) clouds → half-res RT (cloud isolated on its own layer, transparent
   * clear), (2) the rest of the scene → canvas, (3) composite the cloud RT over the canvas.
   * If the cloud mesh is missing or hidden (non-normal view), falls back to a single normal
   * render — no 3-pass overhead and identical behaviour when clouds aren't showing.
   */
  render(renderer: WebGPURenderer, scene: Scene, camera: Camera, cloudMesh: Mesh | null): void {
    if (cloudMesh === null || cloudMesh.visible === false) {
      renderer.render(scene, camera)
      return
    }

    const prevAutoClear  = renderer.autoClear
    const prevAlpha       = renderer.getClearAlpha()
    const prevBackground  = scene.background

    // --- Pass 1: clouds only → half-res RT, cleared transparent ---
    // scene.background drives the clear, so null it out so the RT clears to the transparent
    // clear COLOR rather than the opaque space background (else the composite adds that tint).
    cloudMesh.layers.set(CLOUD_PASS_LAYER)
    camera.layers.set(CLOUD_PASS_LAYER)
    scene.background = null
    renderer.setClearColor(0x000000, 0)
    renderer.setRenderTarget(this._rt)
    renderer.render(scene, camera)

    // --- Pass 2: everything except clouds → canvas (cloud stays on layer 1 → excluded) ---
    scene.background = prevBackground
    renderer.setRenderTarget(null)
    camera.layers.set(0)
    renderer.render(scene, camera)

    // --- Pass 3: composite clouds over the canvas (no clear) ---
    renderer.autoClear = false
    this._quad.render(renderer)

    // Restore default state so the toggle-off single-pass path keeps working next frame.
    renderer.autoClear = prevAutoClear
    renderer.setClearAlpha(prevAlpha)
    cloudMesh.layers.set(0)
    camera.layers.set(0)
  }
}
