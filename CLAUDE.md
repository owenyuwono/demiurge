# Demiurge — project notes for Claude

A No Man's Sky-style **procedural planet** prototype. Three.js **WebGPU** renderer. The vertical slice: seamless orbit→surface, terrain is a pure function of `(seed, position)`. Two camera modes: a globe/orbit camera and a first-person "walk" mode.

## Running & verifying
- **The user runs and verifies the app themselves.** Do NOT start the dev server, open a browser, use Playwright, or take screenshots to "check" rendering — write the code and hand it off. (WebGPU also won't render in a headless browser in this environment.)
- `npm run dev` = Vite dev server (the user runs it). `npx tsc --noEmit` = the typecheck gate (keep it green).

## World scale — single source of truth
`src/planet/worldConstants.ts` owns all scale constants. Do not hardcode these values elsewhere.

- `RADIUS = 500_000` (500 km), `HEIGHT_SCALE = 12_000` (12 km relief), `RADIUS_REF = 50_000`, `HEIGHT_SCALE_REF = 1_200`, `RES_REF = 256`
- The pipeline is angular/normalized; `RADIUS` and `HEIGHT_SCALE` enter **only** at final geometry (`r = RADIUS + hNorm * HEIGHT_SCALE`), camera maths, and absolute-physical quantities. fp32 holds at 500 km (origin-relative vertices; the 500k camera/chunk offset cancels in fp64 CPU-side modelView before the GPU).
- **Uniform-scale invariant:** this is a true 10× scale of the 50km/1200 baseline (relief ratio 2.4% preserved). When you add a parameter that depends on absolute height or distance, scale it as `param * (HEIGHT_SCALE / HEIGHT_SCALE_REF)` — otherwise it comes out 10× wrong. Example: `bUpliftRate` is authored at 40 m/step at the 1200 baseline, so Planet.ts initializes it as `40 * (heightScale / HEIGHT_SCALE_REF) = 400 m/step` at 500 km.
- `deriveSlopeThresh(baseThresh, radius, heightScale, res)` and `deriveErosionRes()` (returns 256, fixed, not scaled) live in `worldConstants.ts`.

## Terrain LOD
- Cube-sphere **quadtree** of heightfield chunks: `Planet.ts`, `QuadtreeNode.ts`, `ChunkMesher.ts`, `faceBases.ts`. Each chunk is a `resolution`×`resolution` grid (resolution = 32). Scale values come from `worldConstants.ts` (see above).
- **Screen-space-error selection** (`computeProjPx` / `selectLeaves`): a node splits when its projected size exceeds `resolution * targetTriPx` px. `targetTriPx` (default 2.0, range 0.5–8) and `maxDepth` (default 10, range 8–20) are live lil-gui sliders. Distance is nearest-point to a per-node bounding sphere whose center is the **terrain-height-adjusted `surfaceCenter`** — measuring to sea level under-refined elevated terrain (that was a bug; don't revert it).
- **Frustum culling** (`collect` vs `_localFrustum`): the build only refines/meshes chunks in the camera frustum (dilated ~1 chunk so turning doesn't pop). Off-screen stays coarse. This is a major perf win — keep it.

## Web Worker meshing — the key perf system
Chunk meshing runs on a **worker pool** OFF the main thread: `MeshWorkerPool.ts`, `meshWorker.ts`, `meshProtocol.ts` (N = `hardwareConcurrency-1`). The main thread no longer meshes — single-threaded meshing of the expensive ~14-octave tectonic heightFn was the bottleneck.

- **Determinism is critical.** Worker output must be **byte-identical** to the main thread or chunk seams crack. The terrain function is ONE shared factory — `terrainSampler.ts` `makeTerrainSampler` — used by both sides. Change terrain logic ONLY there; never fork it.
- The expensive tectonics/climate bake runs **once** on the main thread and is shipped to workers via `Tectonics.toBaked()/fromBaked()` and `Climate.toBaked()/fromBaked()` (workers reconstruct query-only instances, no re-bake). `fromBaked` uses `Object.create`, so it must explicitly init every hot-path scratch field (field initializers don't run).
- `ChunkMesher.computeChunkArrays` = worker-safe compute core (raw typed arrays, no THREE GPU objects); `arraysToGeometry` builds the `BufferGeometry` on the main thread.
- Build queue: `drainBuildQueue` submits to the pool; `onWorkerResult` caches results (with a generation counter to drop stale results across `regenerate()`); `selectLeaves` consumes the cache.
- **Do NOT pre-enqueue the whole descent tree.** `enqueueDeepPath` was removed — with fast workers it flooded the queue (~20k chunks) and thrashed the cache. Level-by-level descent + frustum culling is correct.

## Meshing gotchas
- **Normals:** interior vertex normals come from mesh geometry (neighbor-position cross products, cheap). **Border** vertices use **sphere-tangent central differences** — this is the cube-face SEAM FIX (face-local normals caused a lighting crease along the 12 cube edges). Don't change the border method.
- The heightFn (tectonic Voronoi query + ~14-octave FBM + ridged + climate + process palette) dominates per-chunk cost. Ridged-noise amplitude/octaves are tuned low to avoid jagged spikes at fine LOD.
- Skirts (sized proportional to chunk) hide LOD/seam cracks.

## Terrain diversity — substrate, process palette, deposition

### Substrate: rockHardness field
`tectonics.ts` bakes a per-texel `rockHardness` [0,1] field: a 4-term blend of crust age + volcanic-arc proximity + crust type + FBM (stream 19), with composition-scaled contrast (`contrastScale = 0.25 + 0.75 * composition`). Query: `hardnessAt(dir)` — C1 bilinear lookup. Shipped via `TectonicsBaked.rockHardness`.

**Differential erosion (active in B-path):** per-cell erodibility `K0` and talus angle are modulated by `hardness` in the erosion bake (`erodibilityB` / `talusScaleB`). Hard rock → higher talus, more resistant → ridges/mesas. Soft → lower resistance → smooth basins/badlands. Also modulates `detailAmp` in `heightFn` via `hardnessTerm = 1.0 + HARD_DETAIL_BOOST*0.7*(rockHardness - 0.5)`.

### Process palette (heightFn — `terrainSampler.ts`)
Continuous smoothstep-weighted blends; streams 40/41/42; gated on BASE temperature only (no sun term):
- **Fluvial** (baseline)
- **Glacial** (stream 40): `PROC_GLACIAL_STR=0.55`; elev band [0.10, 0.35], temp [-40, 5]°C, moisture [0.20, 0.60]
- **Aeolian** (stream 41): `PROC_AEOLIAN_STR=0.50`; windSpeed [0.30, 0.75], aridity [0.00, 0.70], dune freq 22.0; wind-aligned dunes
- **Karst** (stream 42): `PROC_KARST_STR=0.30`; moisture [0.55, 0.90], solubility window [0.20, 0.65] of rockHardness

### Deposition differentiation
Fan/floodplain/delta classifier (`depositEnv` Uint8, ENV_FLOODPLAIN/FAN/DELTA) with mass-conserved lateral fan spread lives in the **frozen-flow path** of `erosion.ts`. `ChunkMesher` applies tints post-biome: floodplain `#a89270`, fan `#c4aa7a`, delta `#9a9080`; cap-rock tint `#7e7060` when hardness > 0.65.

**Caveat — see Known gaps below.** The planet ships on the **B-path** (`bActive:true`; `Planet` always supplies `upliftForcing`), which leaves `depositEnv` all-zero. Fan/floodplain/delta color differentiation is therefore inactive in the rendered planet. Substrate hardness differential erosion IS active.

## Wind system (`climate.ts`)
`bakeWind()` produces a baked cube-map field (windX/Y/Z + windSpeed) consumed by `windAt(dir, out)` and `windSpeedAt(dir)`.

**Structure:** 3-cell zonal base (trades westward / westerlies eastward / polar easterlies westward, driven by `retrograde * -cos(2*bandCount*absLat)`) + Coriolis cross-isobar tilt (right N / left S, max at equator → geostrophic at poles) + streamfunction Gaussian pressure-vortex swirls (highs ~30°, lows ~60°, hemisphere-signed rotation, calm eye). Vortex placement is deterministic fixed-order from `deriveSeed(seed, 201)`. Stream 201.

**Equator taper (baked):** `equatorTaperWidth` param (radians, default 0.20, range 0–0.5) drives an equator mask `eqMask = smoothstep(0, sin(equatorTaperWidth), |dir.y|)` applied to the summed vortex swirl and its speed contribution only — zonal bands and Coriolis tilt are untouched. Effect: highs/lows are hemisphere-confined and a calm ITCZ/doldrums belt emerges at lat=0. Pure function of `dir.y`, no new RNG draws; stream-201 vortex placement is unchanged. Threaded `Planet._windEquatorTaper` → `terrainSampler` `SamplerOpts.equatorTaperWidth` → `Climate` ctor.

**Transient weather layer (visualization-only):** `windAtTime(dir, t, out)` returns the static climatological mean PLUS a transient delta: (i) the persisted vortex table re-evaluated with longitude-only drift and a sinusoidally pulsing amplitude (latitude preserved so equator confinement holds); (ii) a time-evolving divergence-free curl-noise eddy field — 2-octave streamfunction via `createNoise3D` on stream 202, finite-difference tangent gradient, `eddyVel = dir × gradΨ` — weighted by a `stormWeight` that is zero in the doldrums (shares `eqMask`) and fades at poles. Deterministic in `(dir, t)`; zero-alloc. **Determinism note:** terrain meshing and aeolian dunes read the static `windAt()` (millennia-mean) only; workers never call `windAtTime`; `toBaked/fromBaked/ClimateBaked` are unchanged — the transient layer cannot affect terrain or break worker determinism. `Planet.animateWind(t)` self-gates on `windDebug.visible`, so there is zero per-frame cost when the wind view is off.

Consumed by: static `windAt()` → aeolian dune alignment + terrain sampler; `windAtTime()` → `WindDebug` animated arrows; static `windAt()` baked to equirectangular texture → `CloudShell` domain warp (see below).

**Arrow visualization:** `WindDebug.ts` (mirrors `TectonicsDebug` structure); uses shared `arrowGeometry.ts` (`buildArrowGeometry`). Arrow length = `radius * 0.045`; bearing-hue coloring; positions sampled via Fibonacci sphere at `surfaceRadius + heightScale * 2.0`. Re-samples `windAtTime` each frame when visible. Shows the instantaneous direction field — each arrow re-points and resizes in place but conveys no flow continuity.

**Flow visualization:** `WindFlow.ts` — an advected-particle streakline overlay (earth.nullschool.net "living wind map" look) that **coexists** with the arrow overlay. Both live in the `wind` view (hotkey 6) and are independently toggleable via GUI checkboxes. Thousands of particles (default 2000) are seeded on the sphere and advected each frame along `windAtTime(dir, t)` via a great-circle step (`pNew = normalize(p·cos(arc) + w·sin(arc))`, `arc = flowSpeed·speed·dt`). Each particle keeps a short ring buffer of its last K positions (trail length, default 8), rendered as one batched `THREE.LineSegments`. Particles respawn on a jittered lifetime OR when speed drops below a threshold (so they don't pool in the doldrums); on respawn the trail ring collapses to zero-length so no streak is drawn across the sphere. Material: `LineBasicNodeMaterial` (from `three/webgpu`) with additive blending, `depthWrite` off, and a per-vertex RGB brightness fade head→tail — bright bearing-hue at the head, black at the tail. Deliberately avoids per-vertex alpha (WebGPU node-material alpha support was unconfirmed; additive + fade-to-black gives the same perceived fade robustly). Bearing-hue matches the arrow overlay for consistent reading. Particles render on a constant shell at `radius + heightScale·2.0` — no per-frame terrain-height query. Uses a LOCAL mulberry32 PRNG (fixed seed) for scatter/respawn — NOT a reserved RNG stream; does not touch terrain, meshing, workers, or `climate.ts`. `Planet.animateWind(t, dt)` drives both overlays and self-gates on visibility (zero per-frame cost when the wind view is off); `windAtTime` is sampled once per particle per frame, color path is allocation-free.

GUI sliders (rebake): swirl strength, # highs, # lows, Coriolis strength, vortex size, lat spread, equator taper. Live: storm drift (0–0.1, def 0.015), pulse rate (0–1.5, def 0.25), pulse depth (0–1, def 0.35), eddy strength (0–1, def 0.35), eddy scale (1–20, def 6), eddy time scale (0–1.5, def 0.3). Wind folder checkboxes: `wind arrows` (def on), `wind flow` (def on). Arrow live sliders: arrow density, arrow scale. Flow live sliders: flow density (200–6000, def 2000), flow speed (0–0.5, def 0.15), trail length (2–16, def 8), particle life (1–10 s, def 4).

## Cloud shell (`CloudShell.ts`)
A translucent procedural cloud layer wrapping the planet. The cloud shell renders **only in the normal view** and is hidden in every other view so it doesn't occlude data overlays. Visibility is centralized in `main.ts` `applyView` via `planet.setCloudShellVisible(ui.view === 'normal')`; wind overlays stay on their own gate (`setWindOverlaysVisible(ui.view === 'wind')`). The shell material is `DoubleSide`, so cloud cover is visible overhead from the surface (walk mode) as well as from orbit. Motivation: the arrow/streakline overlays convey wind direction and motion but not the massed gestalt of weather; the cloud shell gives broad cloud-formation shapes that follow the wind.

**Geometry and material:** a `SphereGeometry` shell at altitude `radius + heightScale * altitudeMul` (default mul 1.5, clears peaks). Rendered with a `MeshBasicNodeMaterial` TSL graph. `DoubleSide`, `NormalBlending`, transparent, `depthWrite` off, `renderOrder` 10.

**Shape — billow + FBM hybrid:** cloud density is a blend of (a) **billow noise** (sum-of-abs octaves — puffy cauliflower silhouettes) at a puff-scale feature size and (b) the classic `mx_fractal_noise_float` FBM, mixed by a live `billow` weight (0–1, def 0.8). A detail octave erodes cloud edges. The sample point is domain-warped by the local wind vector (see below) and scrolled over time. A soft `smoothstep` edge (`softness` 0.05–0.6, def 0.3) replaces the old hard coverage threshold.

**Fake-volume shading (`volume` 0–1, def 0.6):** a radial 2-tap density self-shadow gives lit tops and darker/bluish shadowed bases, combined with the world-frame sun term (ambient + N·L via `positionWorld` + shared `_uSunDir`). Clouds read as voluminous rather than a flat white film.

**Wind drift and domain warp — de-streaked:** drift and warp now use the **un-normalized** wind vector (`windDir * speed`), so both naturally vanish at calm vortex eyes and the equatorial doldrums — eliminating the `normalize(0)` singularity that produced radial pinwheel spokes. A `poleFade` additionally suppresses warp near the poles.

**Wind texture:** a 512×256 equirectangular RGBA `DataTexture` baked once (and on `regenerate()`) from `climate.windAt(dir)` — the static climatological mean. Packs tangent wind direction in RGB, speed in A. Sampled in the fragment shader via `dir → (lon, lat)` UV. The shell is a child of the Planet group so its local frame matches the bake and rides the planet's tilt/spin; lighting stays world-frame.

**Coverage favorability (atmospheric rules):** coverage threshold is shifted by a favorability field — a live-weighted blend of: (1) **moisture** from `climate.sample(dir, 0).moisture`; (2) **convergence** — negative divergence of the wind field, tanh-compressed (`tanh(conv * 0.25) * 0.5 + 0.5`) into an RGBA8 `DataTexture` (G = moisture, B = convergence; same UV convention); (3) a gentle **ITCZ/storm-belt latitude term**. Modulation is centered: `t0 = coverage - favWeight * (fav - 0.5)` — humid/converging zones gain cloud, dry/sinking zones lose it. `favWeight = 0` reproduces uniform coverage exactly.

**Determinism/scope:** visualization-only. `climate.ts` is read-only; no terrain, worker, or `toBaked/fromBaked` impact. `Planet.animateWind(t, dt)` advances the cloud time uniform; the shell self-gates on `setCloudShellVisible` (zero per-frame cost when hidden).

## Atmosphere shell (`Atmosphere.ts`)
A tunable atmospheric scattering shell — always-on appearance in both orbit and walk mode (not a view mode). From orbit it shows a blue glow at the planet limb; from the surface it gives a blue sky (in-scattering). Black space falls out naturally because zenith/deep-space rays accumulate near-zero scatter.

**Geometry and material:** a `BackSide` `SphereGeometry` at `radius + heightScale * atmHeightMul` (default mul ~2.5). `MeshBasicNodeMaterial`, `AdditiveBlending`, `depthWrite` off, `renderOrder` 5 — behind the cloud shell's `renderOrder` 10.

**TSL node graph (world frame):** computed entirely in world space using the built-in `cameraPosition` node + `positionWorld` + the shared `_uSunDir` uniform (same reference Planet uses for lighting — no per-frame camera upload needed; the planet sits at the world origin and is only spun, never translated). Terms: grazing/horizon thickening via `pow(grazing, scaleHeight)` (guarded against zero so the slider min can't break it); Rayleigh phase brightening toward the sun; a day/night fade so the night limb goes black.

**Ownership:** built in `Planet` constructor, rebuilt on `regenerate()`, disposed in `dispose()`. Reuses `Planet._uSunDir` by reference. Visualization-only — no terrain, worker, climate, or RNG impact; consumes no RNG stream.

## RNG stream registry
Never reuse a stream id. All streams are reserved permanently — add new ones at the end of each namespace.

| Range | Owner |
|-------|-------|
| 0–19  | Tectonics (1=typeRng, 2=elevRng, 3=axisRng, 4=speedRng, 5=warpNoise, 7=walkerRng, 8=extraRng, 9=crustNoise, 10=rpRng, 11=microRng, 12=paleoRng, 13=fineWarpNoise, 14=texelHash, 15=contNoise, 16=posRng, 17=intRng, 18=volNoise, 19=hardnessNoise) |
| 30–31 | Subsurface (30=query-time domain-warp, 31=ore-vein noise) |
| 40–42 | Process palette (40=glacial, 41=aeolian, 42=karst) |
| 200   | Climate moisture RNG |
| 201   | Climate wind vortex generation |
| 202   | Climate transient curl-noise eddy (visualization-only) |
| Erosion: 9=bake-time reserved, 10=query-time domain-warp (not in above table — erosion uses its own seed path) |

## Design principles
- **Everything is a slider** — no dropdowns, no selectable type lists. All parameters are continuous. Planet type emerges from physical dials (e.g. `composition` drives hardness contrast; `axialTilt` drives climate bands).
- **One physical dial per spectrum** — prefer a single param spanning Earth↔Venus↔Mercury rather than mode enums. This keeps the space smooth and avoids Earth-overfit.
- **Determinism contract** — one shared `makeTerrainSampler` factory, main + workers byte-identical. Bake-and-ship via `toBaked/fromBaked` (Object.create caveat: explicit scratch-field init required). C1 smoothstep sampling on all baked grids. Continuous blends not hard thresholds on baked fields.

## Decisions (settled — don't re-propose)
- **CDLOD vertex morphing:** tried, then reverted — ~doubled mesher cost for little gain; discrete LOD + skirts is used.
- **Full Nanite:** rejected — wrong tool for regenerable procedural heightfield terrain; WebGPU also lacks 64-bit atomics + multi-draw-indirect.
- **Simplex-noise-base terrain:** tried 2026-06-16, reverted same day — relief stays tectonic-simulation-driven.
- **Lake water-table landforms in heightFn:** tried (heightFn lake-flatten + marsh), reverted — produced a coarse 256²-grid staircase on small lakes. Lakes exist (smooth, blue) via bake-time erosion lake flatten only.
- **EROSION_RES:** fixed at 256, not scaled with RADIUS (`deriveErosionRes()` = 256). Scaling to 512+ caused main-thread freeze before first render.
- **B-path MinHeap / mfdNeighbors allocations:** hoisted out of the per-step loop (byte-identical output).
- **Wind field is intentionally a static climatological mean for terrain:** `windAt()` / the baked arrays are the millennia-average used by aeolian dunes and `terrainSampler`. The transient animated layer (`windAtTime`) is visualization-only — it never touches terrain, workers, or baked state.

## Known gaps / caveats
- **Deposition classifier (fan/floodplain/delta)** lives in the frozen-flow path. The planet always ships on the B-path → `depositEnv` all-zero → fan/floodplain/delta color tints inactive. Open follow-up to route classifier output into the B-path or merge paths.
- **Erosion `bUpliftRate` fallback:** `erosion.ts` has an internal fallback of `60 * (HEIGHT_SCALE / HEIGHT_SCALE_REF)` if `opts.bUpliftRate` is missing. `Planet.ts` always supplies it as `40 * (heightScale / HEIGHT_SCALE_REF)`, so the fallback is not hit in normal use — but don't silently drop `opts.bUpliftRate`.
- **HUD `KEYMAP_HELP`** in `Hud.ts` is stale — references only hotkeys 1–4, f, g; missing 5 (climate), 6 (wind), 7 (materials), c (cave). Update it when touching `Hud.ts`.

## Navigation
Globe/orbit camera (`GlobeControls.ts`) + first-person walk (`FirstPersonController.ts`, `controls/NavMode.ts`, `SurfacePicker.ts`, `debug/NavControlsBar.ts`). "Walk here" → click the planet → camera spawns at eye height (1.7 m) above the surface, gravity-aligned, mouse-look + WASD. Spin auto-pauses in walk mode. 'c' key spawns at a cave mouth.

## HUD / debug
HUD fields: `mode`, `lod` (visible level range), `lod cap`, `seed`, `regime`, `plates`, `volcanoes`, `hotspots`, `spin`, `view`, `lod cached`, `builds`, `build ms`.

**Hotkeys:**

| Key | Action |
|-----|--------|
| 1   | Toggle wireframe |
| 2   | Toggle LOD color view |
| 3   | Toggle tectonics view |
| 4   | Toggle spin |
| 5   | Toggle climate view |
| 6   | Toggle wind view (arrows + flow particles; each independently toggleable in GUI) |
| 7   | Toggle materials view (rockHardness; indigo→cyan→lime→orange→crimson = soft→hard) |
| f   | Freeze LOD |
| g   | New random seed |
| c   | Spawn at cave mouth |

View modes (7 total): `normal`, `lod`, `tectonics`, `heightmap`, `climate`, `wind`, `materials`. `heightmap` available via GUI dropdown only (no hotkey). The cloud shell is always-on in `normal` view and hidden in all others; there is no separate `clouds` view mode.

lil-gui panel: LOD tuning (`tri px target` 0.5–8 default 2.0, `max depth` 8–20 default 10), `wireframe`, `freeze LOD`, `lod diag` (toggleable on-screen LOD readout — OFF by default), view dropdown, spin, new seed, erosion sliders. **Atmosphere tab** contains two folders:

- **Wind folder** — rebake: swirl strength, # highs, # lows, Coriolis strength, vortex size, lat spread, equator taper (0–0.5 rad, def 0.20); checkboxes: `wind arrows` (def on), `wind flow` (def on) — both overlays show in the `wind` view per their checkboxes; arrow live: arrow density, arrow scale; flow live: flow density (200–6000, def 2000), flow speed (0–0.5, def 0.15), trail length (2–16, def 8), particle life (1–10 s, def 4); transient live: storm drift (0–0.1, def 0.015), pulse rate (0–1.5, def 0.25), pulse depth (0–1, def 0.35), eddy strength (0–1, def 0.35), eddy scale (1–20, def 6), eddy time scale (0–1.5, def 0.3).

- **Clouds folder** — sliders: coverage (def 0.22), scroll speed, cloud scale (1–16, def 6), wind warp (0–1, def 0.0), opacity (def 0.9), altitude; look sliders: billow (0–1, def 0.8), detail (0–1, def 0.2), softness (0.05–0.6, def 0.3), volume (0–1, def 0.6); favorability sliders (all live, no rebake): favorability weight (def 0.3), moisture wt, convergence wt, conv gain, ITCZ wt (def 0.05).

- **Atmosphere shell folder** — continuous controls: atmosphere density (0–2, def 1.0; 0 = airless planet), tint color picker (def `#2e6bff`), sun intensity (0–4, def 1.3), horizon thickness (0–8, def 4.0), shell height (1.0–6.0, def 2.5), enabled checkbox.
