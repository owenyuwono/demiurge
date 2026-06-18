# Technical Briefing: Next-Generation Upgrades for the Demiurge Volumetric Cloud + Atmosphere System

> Research output (2026-06-18). Multi-source web research (Guerrilla "Nubis"/Schneider HZD SIGGRAPH 2015/2017,
> Hillaire/Frostbite, Inigo Quilez, Wrenninge, scratchapixel, three.js WebGPU/TSL) cross-checked and grounded
> against `src/planet/VolumetricClouds.ts` and `src/planet/Atmosphere.ts`. Feasibility: every high-priority item
> lives inside the existing `MeshBasicNodeMaterial` TSL fragment graph (Loop/If/Break already in use) — no compute
> shaders or 3D textures required.

**Audience:** senior graphics engineer with a working `VolumetricClouds.ts` raymarcher and `Atmosphere.ts` analytic shell.

---

## 1. What our implementation already does well

Our raymarcher is already a competent Nubis/HZD-lineage single-scatter renderer.

| Named technique | Where we already do it | Notes |
|---|---|---|
| **Perlin-Worley base shape** | `densityFn`: 4-octave `billow` (`Σ|mx_noise_float|·{1,.5,.25,.125}` × `NORM_BILLOW`) mixed with `mx_fractal_noise_float` FBM by `uBillow` | The sanctioned cheap stand-in for a 128³ Perlin-Worley texture (github.com/sebh/TileableVolumeNoise). Billow ≈ Worley/cellular character, FBM ≈ connected Perlin character. |
| **Detail erosion** | `eroded = saturate(baseShape - detailN·uDetail)` | Erode (subtract) — correct philosophy, but additive-subtract on the whole field, not Nubis remap-from-the-edge. See §2.2. |
| **Coverage remap/dilate** | `t0 = coverage - favWeight·(fav-0.5)`; `smoothstep(t0, t0+softness, eroded)` | Favorability-centered threshold = the dossier's centered coverage remap. Missing the post-threshold `*= coverage` multiply. |
| **Weather map (2D)** | `favTex` 512×256 RGBA8 (G=moisture, B=tanh convergence) + `windTex` | Infrastructure present; no **cloud-type** channel yet (R unused). |
| **Cloud-type height gradient** | `profile = smoothstep(0,roundBase,hf)·(1-smoothstep(1-billowTop,1,hf))` | One vertical profile, NOT type-selected — every region uses the same stratus-ish slab. |
| **Beer extinction** | `T *= exp(-d·stepLen·σT)`; `lightT = exp(-sumL·σT·lightStep)` | Standard Beer-Lambert, artistic σ_T. Correct. |
| **Beer-powder dark edge** | `powder = 1 - exp(-d·2·uPowder)` | Schneider powder term. Present, NOT view-gated (darkens edges everywhere incl. silver-lining side). |
| **HG phase (single lobe)** | `(1-g²)/(4π·(1+g²-2g·cosθ)^1.5)`, `g=0.35` | Single lobe — anti-sun faces under-lit. |
| **Light-march self-shadow** | inner `Loop(uLightSteps)` accumulating density toward `uSunDir` | Real 5-tap secondary march. Straight line, no cone jitter. |
| **Ambient in-scatter** | `+ uAmbient·SKY_*` | Flat constant. No height grading, no sky-color coupling. |
| **Spherical annulus march** | full 3-case ray/2-sphere intersect (A/B/C camera cases) | Correct & complete: orbit, in-shell, surface. |
| **Early ray termination** | `If(transmittance < 0.01) Break()` | Present. |

**Verdict:** the lighting backbone (Beer + powder + HG + true light-march + annulus + early-out) is production-shaped.
Biggest gaps: **shape realism** (remap erosion, cloud-type gradients), **multi-angle lighting** (dual-lobe + multiscatter),
**performance headroom** (empty-space skip, blue-noise dither), and a **sky-dome correctness issue** in `Atmosphere.ts` (§4).

---

## 2. Highest-impact upgrades, ranked

All feasible inside the existing fragment graph. (3D textures + `texture3D().sample()` + `RaymarchingBox` DO ship in
three WebGPU first-party — PR #30495 / `webgpu_volume_cloud` — if we later want a true 128³ base.)

| # | Technique | Payoff | Effort | Risk |
|---|---|---|---|---|
| 1 | **Cheap-vs-full empty-space skipping** (two-tier density) | **Largest perf win** — 2–4× fewer expensive samples in a mostly-empty shell | Medium: split `densityFn` → `densityCheap`/`densityFull`, big-step cheap, backtrack+small-step full on first hit | Low (output converges identical) |
| 2 | **Remap-based detail erosion** | **Biggest shape upgrade** — fluffy cauliflower bases + wispy tops, never carves holes in cores | Low: `remap(baseShape, modifier·erode, 1, 0, 1)` | Low |
| 3 | **Blue-noise ray-start dither** | Cut `uStepCount` 40→~24 with no banding | Low: 256² R8 texture, `tEnter += stepLen·fract(bn + frame·0.618)` | Low |
| 4 | **Dual-lobe HG** (forward+back) | Fixes dead-black anti-sun faces | Low: `mix(hg(mu,0.8), hg(mu,-0.3), 0.5)` | Very low |
| 5 | **Multi-octave "multiscatter" Beer** | Thick cores glow instead of going black | Medium: 2–3 octave loop with `aⁿ/bⁿ/cⁿ` scaling | Low |
| 6 | **Cone-jittered light march** | Softer self-shadows, kills light banding | Low: fixed 5-tap kernel offsetting each `lp` | Low |
| 7 | **Cloud-type channel** in `favTex.R` + type-blended height gradient | Real stratus/stratocumulus/cumulus sky regions | Medium | Low-med |
| 8 | **Post-threshold coverage multiply + anvil** | Shrinking clouds stay rounded; storm anvils | Low: `cov *= coverage` | Low |
| 9 | **Half-res render + bilateral upsample** | ~4× fewer marched pixels | High: offscreen RT + composite, breaks current in-scene shell simplicity | Medium — **defer** |

**Adoption order:** 1 → 2 → 3 → 4 → 5 → 6 → (7,8) → defer 9.

---

## 3. Multiple-scattering / lighting realism

### 3a. Dual-lobe HG (drop-in)
```ts
const hg = Fn(([mu, g]) => {
  const g2 = g.mul(g)
  const denom = pow(float(1).add(g2).sub(g.mul(2).mul(mu)), float(1.5)).mul(PI_4)
  return float(1).sub(g2).div(denom)
})
const mu      = dot(rd, uSunDir)
const hgPhase = mix(hg(mu, uHgForward), hg(mu, uHgBack), float(0.5)) // gF≈0.8, gB≈-0.3
```
`gBack` negative adds weak backscatter so away-from-sun faces keep a gentle glow. Cheaper Schlick variant:
`k = 1.55·g − 0.55·g³`, `p = (1−k²)/(4π·(1+k·mu)²)`.

### 3b. Multi-octave "multiscatter" Beer (Wrenninge/Hillaire; Skybolt `a=b=c=0.5, octaves=3`)
```ts
const lum = float(0).toVar()
Loop({ start: 0, end: 3 }, ({ i: n }) => {
  const k    = pow(float(0.5), float(n))                                  // aⁿ=bⁿ=cⁿ
  const extN = exp(sumL.negate().mul(uSigmaT).mul(lightStep).mul(k))      // aⁿ on extinction
  const phN  = mix(float(1).div(PI_4), hgPhase, k)                        // cⁿ: phase → isotropic
  lum.addAssign(k.mul(extN).mul(phN))                                     // bⁿ contribution
})
const lit = lum
```
**Multiscatter does NOT replace powder** — Nubis Cubed (2023) ships both. Powder darkens thin **edges**;
multiscatter brightens deep **interiors**. Keep both.

### 3c. View-gate the powder term
```ts
const powderGate = saturate(cosTheta.mul(-0.5).add(0.5))   // 1 anti-sun, 0 toward-sun
const powderEff  = mix(float(1), powder, powderGate.mul(uPowder))
```

> **Energy footnote:** swap point-sample-then-attenuate for Hillaire's analytic in-slice
> `Sint = (S − S·exp(−σ·ds))/σ; L += T·Sint; T *= exp(−σ·ds)` — removes step-count brightness drift.

---

## 4. The black-sky problem in `Atmosphere.ts`

### Why it goes black at zenith from the surface
`horizonGlow = pow(grazing, scaleHeight)`, `grazing = 1 − |dot(viewDir, nFrag)|`. Straight up, `viewDir ≈ nFrag`
→ `grazing → 0` → `pow(0,k) = 0`. That term encodes **only horizon air-mass thickening**, no zenith content —
backwards for a sky dome. Real Rayleigh: zenith is deep blue because in-scatter integrated along EVERY view ray is
non-zero. Brightness should be `(1 − exp(−β·opticalDepth))` (never zero), not bare `pow(grazing,k)`.

**Current state:** the code already has `_uSkyFloor` (def 0.35): `density = skyFloor + (1-skyFloor)·horizonGlow`.
This lifts the dome but is a **constant** — no air-mass shape (zenith-darker / horizon-brighter), no wavelength
dependence — so it reads as a flat blue wash, not a real deep-zenith→pale-horizon gradient. The fixes below add shape.

### Fix A — air-mass single scatter (proper fix, per-channel)
```ts
const muUp    = saturate(dot(viewDir, nFrag))                       // 1 zenith, 0 horizon
const airMass = float(1).div(                                       // Kasten-Young, bounded
  muUp.add(float(0.15).mul(pow(float(93.885).sub(acos(muUp).mul(57.29578)), float(-1.253)))))
const betaR   = vec3(0.0058, 0.0135, 0.0331)                        // per-channel Rayleigh
const viewOD  = betaR.mul(uZenithOD).mul(airMass)
const skyAmt  = vec3(1).sub(exp(viewOD.negate()))                   // NON-ZERO at zenith
```
Per-channel → automatic blue-zenith → whiter-horizon desaturation. Multiply by existing `phase`, `sunUp`, `uSunIntensity`.
Cheaper monotone proxy: `airMass = 1/max(muUp, 0.05)`.

### Fix B — air-mass-shaped floor (minimal, lowest-risk, two-line edit)
```ts
const muUp    = saturate(dot(viewDir, nFrag))
const zenithI = float(1).sub(exp(float(-1.0).div(max(muUp, float(0.05)))))  // 1/μ air mass, saturating
const density = mix(uSkyFloor, float(1), max(horizonGlow, zenithI))
```
Gives the zenith real depth while preserving the limb glow and the `skyFloor=0` "space stays black" slider min.

### Fix C — lower the `pow` exponent
`scaleHeight` ~0.5–1.0 widens the band, but alone does NOT fix zenith (`pow(0,0.5)=0`). Tuning aid only.

**Blending note:** shell is `AdditiveBlending`. For a true surface dome consider `NormalBlending` for the dome
contribution, additive only for the orbital limb. At minimum verify the additive floor reads correctly vs the black skybox in walk mode.

---

## 5. Cloud motion / animation realism

Current model is **pure advection** (`pAdv = p0 + drift + warp`), correctly using un-normalized wind (drift vanishes at
calm eyes/doldrums — the de-streaking fix). Gaps & planet-relevant upgrades:

- **Advection vs evolving density** — rigid translation = sliding decal; shapes never form/dissipate. Cheap fix: scroll the
  **detail octave** at a different rate/direction + a low-amplitude time term on the detail sample only → base drifts, edges boil.
- **Curl-noise wisp advection** — offset the detail sample by a divergence-free curl field, strongest at base:
  `pDetail = pAdv + curl·curlStrength·(1 - heightFrac)`. We already build a divergence-free curl field on **stream 202**
  for `WindFlow` — reuse it (keep visualization-only so determinism/worker boundary is untouched).
- **Spin coupling already correct** — density sampled in planet-local frame via `_uInvRot`; keep it. Wind/fav textures are
  the static climatological mean (consistent with project policy); keep transient `windAtTime` out of the cloud bake.
- **Avoid full temporal accumulation for now** — biggest theoretical speedup but most complex (history RT, motion vectors,
  ghosting). If blue-noise grain is objectionable, prefer a simple single-history reproject-and-lerp (`feedback≈0.85`).

---

## 6. Concrete next-steps checklist (impact-to-effort)

1. **[shape, low/high]** Remap erosion: `eroded = remap(baseShape, modifier·uErode, 1, 0, 1)`, `modifier = mix(hiFBM, 1−hiFBM, saturate(heightFrac·5))`. *(§2.2)*
2. **[lighting, low/high]** Dual-lobe HG: factor `hg` into an `Fn`, `mix(hg(mu,0.8), hg(mu,-0.3), 0.5)`. *(§3a)*
3. **[lighting, low]** View-gate powder by `cosTheta`. *(§3c)*
4. **[sky, low/high]** Fix `Atmosphere.ts` zenith — Fix B (2-line air-mass floor) or commit to Fix A (per-channel). Add `acos` import. *(§4)*
5. **[perf, medium/highest-perf]** Two-tier empty-space skipping; then lower `uStepCount`. *(§2.1)*
6. **[perf, low]** Blue-noise ray-start dither; add `_uFrame`; cut `uStepCount` 40→24. *(§2.3)*
7. **[lighting, medium]** Multi-octave multiscatter (3 octaves, a=b=c=0.5); keep powder. *(§3b)*
8. **[lighting, low]** Cone-jitter the light march (5-tap kernel). *(§2.6)*
9. **[shape, low]** Coverage multiply + anvil bias. *(§2.8)*
10. **[weather, medium]** Cloud-type channel in `favTex.R`; blend 3 height-stop profiles. *(§2.7)*
11. **[motion, low-med]** Curl-offset the detail sample (reuse stream-202 field) + independent detail scroll. *(§5)*
12. **[perf, high — defer]** Half-res RT + bilateral upsample, or simple TAA reproject. Only after 1–11. *(§2.9, §5)*

**Relevant files:** `src/planet/VolumetricClouds.ts` (1–3, 5–11), `src/planet/Atmosphere.ts` (4).
