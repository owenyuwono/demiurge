# Demiurge

A procedural planet prototype — seamless orbit to surface, on a 500 km world whose terrain is a pure function of `(seed, position)`. Three.js on **WebGPU**.

There is no heightmap and no authored terrain. Tectonic plates are simulated, they drive uplift, uplift drives erosion, erosion and latitude drive climate, and climate drives biomes, wind and weather. Every landform you can walk up to is a consequence of that chain.

---

## Running

Requires a WebGPU-capable browser (recent Chrome, Edge, or Safari 18+).

```bash
npm install
npm run dev        # Vite dev server
npm run typecheck  # tsc --noEmit — the gate, keep it green
npm run build      # production build
```

---

## Controls

Two camera modes. The globe camera orbits; **Walk here** drops you onto the surface at eye height (1.7 m), gravity-aligned, with mouse-look and WASD.

| Key | Action |
|-----|--------|
| `1` | Wireframe |
| `2` | LOD colour view |
| `3` | Tectonics view |
| `4` | Toggle spin |
| `5` | Climate view |
| `6` | Wind view (arrows + flow particles) |
| `7` | Materials view (rock hardness) |
| `8` | Wetness view |
| `f` | Freeze LOD (stop refining — inspect the current mesh) |
| `g` | New random seed |
| `c` | Spawn at a cave mouth |

Everything else lives in the lil-gui panel: LOD tuning, erosion, wind, clouds, atmosphere, and the view dropdown (which also carries `heightmap` and `cloud`, the two views with no hotkey).

---

## How it works

**Terrain LOD.** A cube-sphere quadtree of heightfield chunks, each a 32×32 grid. Nodes split on **screen-space error** — projected size beyond `resolution × targetTriPx` — measured to a bounding sphere centred on terrain-adjusted height, not sea level. Only chunks inside a slightly dilated camera frustum get refined or meshed; off-screen stays coarse. Skirts hide the cracks between LOD levels.

**Meshing runs on a worker pool.** The heightfield function is expensive (a tectonic Voronoi query, ~14 octaves of FBM, ridged noise, climate, and a process palette), and meshing it on the main thread was the bottleneck. Chunks are built across `hardwareConcurrency - 1` workers instead.

This makes **determinism a hard requirement**: worker output must be byte-identical to the main thread or chunk seams visibly crack. So the terrain function is one shared factory (`terrainSampler.ts`), used by both sides and never forked. The expensive tectonics and climate bakes run once on the main thread and ship to workers as plain typed arrays, which reconstruct query-only instances rather than re-baking.

**The simulation chain.**

- **Tectonics** — plates, boundaries, uplift forcing, volcanic arcs, hotspots, and a per-texel rock-hardness field blended from crust age, arc proximity, crust type and noise.
- **Erosion** — stream-power (`E = K·Q^m·S^n`) over a 256² cube grid, with lake filling and hardness-modulated erodibility. Hard rock resists and stands up as ridges and mesas; soft rock erodes into basins and badlands.
- **Climate** — temperature and moisture fields, plus a baked wind field: three-cell zonal circulation, Coriolis tilt, and pressure-vortex swirls, with a calm equatorial belt.
- **Process palette** — glacial, aeolian and karst weathering blended continuously over the fluvial baseline by local temperature, moisture, wind and rock solubility. No mode switches; everything is a weighted blend.

**Sky.** Clouds are a full volumetric raymarch through a spherical annulus — Perlin-Worley density, dual-lobe scattering with a light march toward the sun, multi-octave Beer extinction, and coverage driven by a domain-warped weather map plus circulation bands, so cloud systems form in humid convergent air and clear over the subtropical deserts. The whole field advects along the wind by great-circle rotation. Above it sits an atmospheric scattering shell that gives a blue limb from orbit, a blue sky from the ground, and reddening toward a low sun.

---

## World scale

`src/planet/worldConstants.ts` is the single source of truth. Radius 500 km, 12 km of relief — a true 10× scale of a 50 km / 1200 m baseline, with the relief ratio preserved.

The pipeline is angular and normalised throughout; absolute scale enters only at final geometry, camera maths, and genuinely physical quantities. fp32 holds at this size because vertices are origin-relative and the chunk offset cancels in fp64 on the CPU before reaching the GPU.

Anything depending on absolute height or distance must be scaled by `HEIGHT_SCALE / HEIGHT_SCALE_REF`, or it comes out 10× wrong.

---

## Design principles

- **Everything is a slider.** No dropdowns, no planet-type enums. Planet character emerges from continuous physical dials — `composition` drives hardness contrast, `axialTilt` drives climate bands.
- **One physical dial per spectrum.** Prefer a single parameter spanning Earth↔Venus↔Mercury over a mode list. Keeps the space smooth and avoids overfitting to Earth.
- **Continuous blends, never hard thresholds** on baked fields. Every baked grid is sampled with C1 smoothstep.
- **Determinism is the contract.** One shared sampler, bake-and-ship to workers, byte-identical output.

---

## Layout

```
src/
  planet/       terrain, tectonics, erosion, climate, clouds, atmosphere, LOD, workers
  controls/     globe camera, first-person walk, nav modes
  debug/        HUD, tabbed GUI, overlays
docs/           research notes
tools/          golden-vector determinism harness
```

`CLAUDE.md` carries the detailed engineering notes: subsystem internals, tuning history, the RNG stream registry, settled decisions, and known gaps.
