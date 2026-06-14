# Demiurge — project notes for Claude

A No Man's Sky-style **procedural planet** prototype. Three.js **WebGPU** renderer. The vertical slice: seamless orbit→surface, terrain is a pure function of `(seed, position)`. Two camera modes: a globe/orbit camera and a first-person "walk" mode.

## Running & verifying
- **The user runs and verifies the app themselves.** Do NOT start the dev server, open a browser, use Playwright, or take screenshots to "check" rendering — write the code and hand it off. (WebGPU also won't render in a headless browser in this environment.)
- `npm run dev` = Vite dev server (the user runs it). `npx tsc --noEmit` = the typecheck gate (keep it green).

## Terrain LOD
- Cube-sphere **quadtree** of heightfield chunks: `Planet.ts`, `QuadtreeNode.ts`, `ChunkMesher.ts`, `faceBases.ts`. Each chunk is a `resolution`×`resolution` grid (resolution = 32). `RADIUS = 50000`, `HEIGHT_SCALE = 1200`.
- **Screen-space-error selection** (`computeProjPx` / `selectLeaves`): a node splits when its projected size exceeds `resolution * targetTriPx` px. `targetTriPx` (~2) and `maxDepth` (18) are live lil-gui sliders. Distance is nearest-point to a per-node bounding sphere whose center is the **terrain-height-adjusted `surfaceCenter`** — measuring to sea level under-refined elevated terrain (that was a bug; don't revert it).
- **Frustum culling** (`collect` vs `_localFrustum`): the build only refines/meshes chunks in the camera frustum (dilated ~1 chunk so turning doesn't pop). Off-screen stays coarse. This is a major perf win — keep it.
- fp32 holds to ~maxDepth 19–20 (vertices are origin-relative; the ~50k camera/chunk coords cancel in fp64 CPU-side modelView before the GPU).

## Web Worker meshing — the key perf system
Chunk meshing runs on a **worker pool** OFF the main thread: `MeshWorkerPool.ts`, `meshWorker.ts`, `meshProtocol.ts` (N = `hardwareConcurrency-1`). The main thread no longer meshes — single-threaded meshing of the expensive ~14-octave tectonic heightFn was the bottleneck.

- **Determinism is critical.** Worker output must be **byte-identical** to the main thread or chunk seams crack. The terrain function is ONE shared factory — `terrainSampler.ts` `makeTerrainSampler` — used by both sides. Change terrain logic ONLY there; never fork it.
- The expensive tectonics/climate bake runs **once** on the main thread and is shipped to workers via `Tectonics.toBaked()/fromBaked()` and `Climate.toBaked()/fromBaked()` (workers reconstruct query-only instances, no re-bake). `fromBaked` uses `Object.create`, so it must explicitly init every hot-path scratch field (field initializers don't run).
- `ChunkMesher.computeChunkArrays` = worker-safe compute core (raw typed arrays, no THREE GPU objects); `arraysToGeometry` builds the `BufferGeometry` on the main thread.
- Build queue: `drainBuildQueue` submits to the pool; `onWorkerResult` caches results (with a generation counter to drop stale results across `regenerate()`); `selectLeaves` consumes the cache.
- **Do NOT pre-enqueue the whole descent tree.** `enqueueDeepPath` was removed — with fast workers it flooded the queue (~20k chunks) and thrashed the cache. Level-by-level descent + frustum culling is correct.

## Meshing gotchas
- **Normals:** interior vertex normals come from mesh geometry (neighbor-position cross products, cheap). **Border** vertices use **sphere-tangent central differences** — this is the cube-face SEAM FIX (face-local normals caused a lighting crease along the 12 cube edges). Don't change the border method.
- The heightFn (tectonic Voronoi query + ~14-octave FBM + ridged + climate) dominates per-chunk cost. Ridged-noise amplitude/octaves are tuned low to avoid jagged spikes at fine LOD.
- Skirts (sized proportional to chunk) hide LOD/seam cracks.

## Decisions (settled — don't re-propose)
- **CDLOD vertex morphing:** tried, then reverted — ~doubled mesher cost for little gain; discrete LOD + skirts is used.
- **Full Nanite:** rejected — wrong tool for regenerable procedural heightfield terrain; WebGPU also lacks 64-bit atomics + multi-draw-indirect.

## Navigation
Globe/orbit camera (`GlobeControls.ts`) + first-person walk (`FirstPersonController.ts`, `controls/NavMode.ts`, `SurfacePicker.ts`, `debug/NavControlsBar.ts`). "Walk here" → click the planet → camera spawns at eye height (1.7 m) above the surface, gravity-aligned, mouse-look + WASD. Spin auto-pauses in walk mode.

## HUD / debug
HUD shows `mode`, `alt`, `lod` (visible level range), `lod cap`. lil-gui panel: LOD tuning (`tri px target`, `max depth`), `wireframe`, `vertices`, `freeze LOD` ('f' key), `lod diag` (big on-screen LOD readout — OFF by default), view modes (1/2/3), spin (4), new seed ('g').
