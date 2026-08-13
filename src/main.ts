import * as THREE from 'three/webgpu'
import { Planet } from './planet/Planet.js'
import { GlobeControls } from './controls/GlobeControls.js'
import { FirstPersonController } from './controls/FirstPersonController.js'
import { SurfacePicker } from './controls/SurfacePicker.js'
import { NavMode, type SurfaceSampler } from './controls/NavMode.js'
import { NavControlsBar } from './debug/NavControlsBar.js'
import { Hud } from './debug/Hud.js'
import { DEFAULT_INTERIOR } from './planet/interior.js'
import { TabbedGui } from './debug/TabbedGui.js'
import { CaveManager } from './planet/CaveManager.js'
import { CloudCompositor } from './CloudCompositor.js'
import { RADIUS, HEIGHT_SCALE, HEIGHT_SCALE_REF, deriveErosionRes } from './planet/worldConstants.js'

function getSeedFromUrl(): number {
  const params = new URLSearchParams(window.location.search)
  const raw = params.get('seed')
  if (raw !== null) {
    const n = parseInt(raw, 10)
    if (!isNaN(n)) return n
  }
  return 1337
}

async function main(): Promise<void> {
  // --- Loading overlay (so boot is never a black screen) ---
  const loadingEl     = document.getElementById('loading')
  const loadingBar    = document.getElementById('loading-bar')
  const loadingTextEl = document.getElementById('loading-text')
  const setProgress = (pct: number, text?: string): void => {
    if (loadingBar !== null) loadingBar.style.width = `${pct}%`
    if (text !== undefined && loadingTextEl !== null) loadingTextEl.textContent = text
  }
  // Yield so the overlay/bar repaints before the next main-thread-blocking stage.
  const nextFrame = (): Promise<void> => new Promise<void>(r => requestAnimationFrame(() => r()))

  setProgress(8, 'Starting renderer…')
  await nextFrame()

  // --- Renderer ---
  // logarithmicDepthBuffer: true ensures good depth precision across the large
  // near=2 / far=RADIUS*15 range on both WebGPU and the WebGL2 fallback backend.
  const renderer = new THREE.WebGPURenderer({
    antialias: true,
    logarithmicDepthBuffer: true,
  })
  await renderer.init()
  setProgress(25, 'Generating planet…')
  await nextFrame()

  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.toneMapping = THREE.ACESFilmicToneMapping

  const app = document.getElementById('app')!
  app.appendChild(renderer.domElement)

  const backendName = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend === true ? 'WebGPU' : 'WebGL2'

  // --- Scene ---
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x06080c)

  // Surround lighting — the planet is lit from every side (no dark limb, no camera
  // tracking). Ambient + hemisphere give an even base so no surface is ever black;
  // two fixed directional fills from opposing angles keep terrain relief readable.
  // Dial: raise `ambient` toward flatter/more uniform, lower it for more relief contrast.
  const ambient = new THREE.AmbientLight(0xffffff, 0.35)
  scene.add(ambient)

  const hemi = new THREE.HemisphereLight(0x9ab4ff, 0x20180e, 0.6)
  scene.add(hemi)

  const fill1 = new THREE.DirectionalLight(0xffffff, 1.1)
  fill1.position.set(1, 0.8, 0.6)   // direction → origin; magnitude irrelevant for directional lights
  scene.add(fill1)

  const fill2 = new THREE.DirectionalLight(0xffffff, 0.7)
  fill2.position.set(-0.9, -0.3, -1)
  scene.add(fill2)

  // --- Planet ---
  let seed = getSeedFromUrl()
  // Pass initial interior params and axialTiltDeg so the constructor's single bake
  // uses the same values that main.ts startup code would otherwise set via
  // setInteriorParams / setAxialTilt — those calls become no-ops (identity guard)
  // when the stored values already match, preventing a second full erosion bake.
  console.time('new Planet (full bake)')
  const planet = new Planet({
    seed,
    radius: RADIUS,
    heightScale: HEIGHT_SCALE,
    maxDepth: 18,
    targetTriPx: 2.0,
    interior: {
      mass:        DEFAULT_INTERIOR.mass,
      composition: DEFAULT_INTERIOR.composition,
      age:         DEFAULT_INTERIOR.age,
      insolation:  DEFAULT_INTERIOR.insolation,
      waterBudget: DEFAULT_INTERIOR.waterBudget,
    },
    axialTiltDeg: 23.4,   // matches ui.obliquity default
  })
  console.timeEnd('new Planet (full bake)')
  scene.add(planet)
  setProgress(75, 'Compiling shaders…')
  await nextFrame()

  // Polar axis in world space — computed after setAxialTilt(ui.obliquity) is called during
  // startup init below, which sets rotation.z and calls refreshPoleAxis(). We capture it here
  // as a placeholder; it is overwritten before the first frame by the startup init block.
  // rotateY() post-multiplies about local Y, so this world axis never changes during spin.
  const POLAR_AXIS = new THREE.Vector3(0, 1, 0).applyQuaternion(planet.quaternion)

  // --- Water ---
  // Translucent water shell whose radius is driven by ui.waterLevel ∈ [0,1]: 0 sinks it
  // below the deepest terrain (no water), 1 lifts it above the highest peak (fully
  // submerged), 0.5 = sea level at terrain height 0 (the coastlines). Scene-level (not a
  // planet child) since a sphere is rotationally symmetric — tilt/spin don't affect it.
  const water = new THREE.Mesh(
    new THREE.SphereGeometry(RADIUS, 160, 80),
    new THREE.MeshStandardNodeMaterial({
      color: 0x2e6b8f,
      transparent: true,
      opacity: 0.72,
      roughness: 0.15,
      metalness: 0.0,
      depthWrite: false, // land in front still occludes it; seabed behind reads as blue-tinted
    }),
  )
  scene.add(water)

  // --- Camera ---
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.5, RADIUS * 15)
  camera.position.set(0, 0, RADIUS * 3)
  camera.lookAt(0, 0, 0)

  // --- Surface sampler (bridges Planet API → navigation modules) ---
  const sampler: SurfaceSampler = {
    surfaceRadiusAt: (dir: THREE.Vector3) => planet.getSurfaceRadiusAt(dir),
    radius: RADIUS,
  }

  // --- GlobeControls factory — single construction path used for initial create + reset ---
  function makeGlobeControls(): GlobeControls {
    return new GlobeControls(camera, renderer.domElement, {
      getAltitude: (pos) => pos.length() - planet.getSurfaceRadiusAt(pos),
      getPolarAxis: () => POLAR_AXIS,
      radius: RADIUS,
      minAltitude: 30,
      maxAltitude: RADIUS * 8,
    })
  }

  let globeControls = makeGlobeControls()

  // --- FirstPersonController + SurfacePicker (instantiated once, reused across entries) ---
  const firstPerson = new FirstPersonController(camera, renderer.domElement, { sampler })
  const picker = new SurfacePicker(camera, renderer.domElement, { sampler })

  // --- Navigation state ---
  let navMode: NavMode = NavMode.Globe

  // Saved Globe-mode camera state, restored by "Back to orbit"
  let savedCamPos = new THREE.Vector3()
  let savedCamUp = new THREE.Vector3()
  let savedSpin = true

  // Placement-mode click listener — kept as a named ref so we can remove it cleanly
  let placementClickListener: ((e: MouseEvent) => void) | null = null

  function removePlacementListener(): void {
    if (placementClickListener !== null) {
      renderer.domElement.removeEventListener('click', placementClickListener)
      placementClickListener = null
    }
  }

  // --- NavControlsBar callbacks ---
  function onEnterPlacement(): void {
    // Guard: only meaningful from Globe mode
    if (navMode !== NavMode.Globe) return

    // Save camera state so reset can restore it exactly
    savedCamPos = camera.position.clone()
    savedCamUp = camera.up.clone()

    // Dispose GlobeControls so it stops capturing mouse/keys and won't eat the placement click
    globeControls.dispose()

    navMode = NavMode.Placement
    bar.setMode(NavMode.Placement)

    // Attach placement-click listener
    placementClickListener = (e: MouseEvent) => {
      const placement = picker.pickFromEvent(e)
      if (placement === null) {
        // Ray missed the planet — leave listener attached for next click
        return
      }

      // Hit: save spin state and pause planet rotation (a world-fixed walker with
      // the planet rotating under them would slide across the ground)
      savedSpin = ui.spin
      ui.spin = false

      firstPerson.spawn(placement)
      navMode = NavMode.FirstPerson
      firstPerson.enable()

      // requestPointerLock is allowed here because we're inside a click gesture
      renderer.domElement.requestPointerLock()

      bar.setMode(NavMode.FirstPerson)

      // Remove this listener — only one successful pick per enter
      removePlacementListener()
    }

    renderer.domElement.addEventListener('click', placementClickListener)
  }

  function onReset(): void {
    // Clean up whichever mode we're in
    removePlacementListener()
    firstPerson.disable()

    // Restore spin to whatever it was before entering walk mode.
    // If we're resetting from pure Placement (user never picked a surface point),
    // savedSpin was not set in this session — but it holds its last assigned value
    // (initially true, or whatever the last FP entry saved), which is safe to restore.
    // Only restore spin if we actually paused it (i.e. we entered FP mode).
    if (navMode === NavMode.FirstPerson) {
      ui.spin = savedSpin
    }

    // Restore camera to its pre-placement position/orientation before constructing
    // GlobeControls — the constructor decomposes camera.position into spherical coords,
    // so restoring first means orbit view resumes exactly where it left off (no teleport).
    camera.position.copy(savedCamPos)
    camera.up.copy(savedCamUp)

    globeControls = makeGlobeControls()

    navMode = NavMode.Globe
    bar.setMode(NavMode.Globe)
  }

  // --- NavControlsBar (instantiate after callbacks are defined) ---
  const bar = new NavControlsBar({ onEnterPlacement, onReset })

  // --- HUD ---
  const hud = new Hud()

  // refreshDerived — reads planet.getDerivedInterior() and pushes values into the derived
  // readout controllers. Declared as a let so it can be assigned after the GUI is built
  // (the controllers don't exist yet when ui is constructed) but called at runtime by
  // ui.randomizeSeed / ui.regenerate which run after full initialisation.
  // eslint-disable-next-line prefer-const
  let refreshDerived: () => void = () => { /* assigned after GUI build */ }

  // caveManager — constructed after planet's first build (post init block below).
  // Declared here so ui.randomizeSeed / ui.regenerate can call setField() on new seeds.
  let caveManager: CaveManager | null = null

  // tabbed declared before ui so randomizeSeed can reference tabbed.controllersRecursive()
  let tabbed: TabbedGui

  // --- UI state (single source of truth for GUI + hotkeys) ---
  const ui = {
    view: 'normal' as 'normal' | 'lod' | 'tectonics' | 'heightmap' | 'climate' | 'wind' | 'cloud' | 'materials' | 'wetness',
    wireframe: false,
    vertices: false,
    freezeLod: false,
    gizmos: true,
    lodDiag: false,
    spin: false,
    spinPeriodS: 600,
    targetTriPx: 2.0,
    maxDepth: 18,
    water: true,
    clouds: false,
    // Interior parameters (primary tectonic roots)
    mass: DEFAULT_INTERIOR.mass,
    composition: DEFAULT_INTERIOR.composition,
    age: DEFAULT_INTERIOR.age,
    insolation: DEFAULT_INTERIOR.insolation,
    waterBudget: DEFAULT_INTERIOR.waterBudget,
    obliquity: 23.4,
    // Interior-derived readouts (read-only, updated by refreshDerived)
    derivedRegime: '—',
    derivedGravity: 0,
    derivedInternalHeat: 0,
    derivedSurfaceTemp: 0,
    derivedAtmosphere: 0,
    derivedEquilibriumTemp: 0,
    derivedVigor: 0,
    derivedMobility: 0,
    derivedPlateCount: 0,
    derivedOceanCoverage: '0%',
    derivedSeaLevel: 0,
    // Override (advanced) controls
    overrideInterior: false,
    manualHeat: 0.5,
    manualSurfaceTemp: 15,
    manualAtmosphere: 0.6,
    // Climate parameters
    climateField: 'temperature' as 'temperature' | 'moisture' | 'insolation',
    redistribution: 1,
    greenhouse: 0,
    lapseRate: 50,
    tempRange: 70,
    // Sun direction (azimuth 0..360°, elevation −90..90°)
    sunAzimuth: 45,
    sunElevation: 30,
    // Wind bake parameters (all trigger rebake on change)
    windSwirlStrength:  0.6,
    windHighs:          4,
    windLows:           3,
    windCoriolis:       0.4,   // radians
    windVortexSize:     0.18,  // radians (sigmaBase)
    windLatSpread:      0.17,  // radians
    windEquatorTaper:   0.20,  // radians — equatorial vortex-taper half-width
    // Wind transient parameters (live, no rebake) — calmed so the arrows don't jitter.
    windDriftSpeed:     0.010,
    windPulseRate:      0.10,
    windPulseDepth:     0.18,
    windEddyStrength:   0.20,
    windEddyScale:      6,
    windEddyTimeScale:  0.12,
    // Wind render parameters (live, no rebake)
    windArrowDensity:   1500,
    windArrowScale:     1.0,
    // Wind flow overlay parameters
    showWindArrows:     true,
    showWindFlow:       true,
    windFlowDensity:    8000,
    windFlowSpeed:      0.6,
    windFlowTrail:      12,
    windFlowLifetime:   4.0,
    // Cloud shell parameters (all live via setters — no rebake)
    cloudCoverage:     0.65,
    cloudScrollSpeed:  0.001,
    cloudScale:        6,
    cloudWarp:         0.2,
    cloudBillow:       0.45,
    cloudDetail:       0.3,
    cloudSoftness:     0.3,
    cloudVolume:       0.6,
    cloudOpacity:      0.6,
    cloudFavWeight:    0.28,
    cloudMoistWeight:  0.6,
    cloudConvWeight:   0.8,
    cloudConvGain:     2.0,
    cloudItczWeight:   0.12,
    cloudWeatherWeight: 0.3,
    cloudCellWeight:    0.5,
    // Volumetric cloud parameters (new raymarched layer)
    cloudBase:         1.2,
    cloudThick:        0.6,
    cloudDensity:      4.0,
    cloudStepCount:    48,
    cloudLightSteps:   5,
    cloudHg:           0.35,
    cloudPowder:       1.0,
    cloudDetailScale:  4.0,
    cloudRoundBase:    0.2,
    cloudBillowTop:    0.5,
    cloudAmbient:      0.08,
    cloudType:         0.6,
    // Half-res offscreen cloud pass (Part 2a). Off by default (experimental render-path change).
    cloudHalfRes:      false,
    // Erosion bake parameters — onFinishChange triggers rebake on release.
    // Default matches deriveErosionRes(RADIUS) = 256 so slider and first bake agree.
    erosionRes: deriveErosionRes(RADIUS) as 128 | 256 | 512,
    erosionStrength: 1.0,
    erosionDeposition: 1.0,
    bSteps: 30,
    // Authored at HEIGHT_SCALE_REF=1200 as 60 m/step; scale with HEIGHT_SCALE so the slider
    // default matches the Planet constructor default and is invariant under uniform scale.
    bUpliftRate: Math.round(60 * (HEIGHT_SCALE / HEIGHT_SCALE_REF)),
    // Atmosphere shell parameters (all live via setters — no rebake)
    atmosphereOn:           true,
    atmosphereDensity:      1.3,
    atmosphereTint:         '#2e6bff',
    atmosphereSunIntensity: 1.3,
    atmosphereScaleHeight:  3.0,
    atmosphereSkyFloor:     0.35,
    atmosphereHeight:       2.5,
    seed: String(seed),
    randomizeSeed: () => {
      const newSeed = (Math.random() * 2 ** 31) | 0
      seed = newSeed
      ui.seed = String(newSeed)
      planet.regenerate(newSeed)
      caveManager?.setField(planet.getCaveField()!)
      refreshDerived()
      const url = new URL(window.location.href)
      url.searchParams.set('seed', String(newSeed))
      history.replaceState(null, '', url.toString())
      tabbed.controllersRecursive().forEach((c) => c.updateDisplay())
    },
    regenerate: () => {
      planet.regenerate(seed)
      caveManager?.setField(planet.getCaveField()!)
      refreshDerived()
    },
  }

  // --- Apply functions (state → scene/loop) ---

  // Sun direction derived from azimuth/elevation sliders (world space, unit vector).
  const sunDir = new THREE.Vector3()

  function applySunDir(): void {
    const azRad = THREE.MathUtils.degToRad(ui.sunAzimuth)
    const elRad = THREE.MathUtils.degToRad(ui.sunElevation)
    sunDir.set(
      Math.cos(elRad) * Math.sin(azRad),
      Math.sin(elRad),
      Math.cos(elRad) * Math.cos(azRad),
    )
    planet.setSunDir(sunDir)
  }

  function applyView(): void {
    planet.setDebugColors(ui.view === 'lod')
    planet.setTectonicsView(ui.view === 'tectonics')
    planet.setHeightmapView(ui.view === 'heightmap')
    planet.setClimateView(ui.view === 'climate')
    planet.setWindView(ui.view === 'wind')
    planet.setMaterialsView(ui.view === 'materials')
    planet.setWetnessView(ui.view === 'wetness')
    planet.setWindOverlaysVisible(ui.view === 'wind')
    // Cloud shell shows in 'normal' (volumetric) and 'cloud' (flat coverage map + contours).
    planet.setCloudShellVisible((ui.view === 'normal' && ui.clouds) || ui.view === 'cloud')
    planet.setCloudDebugMode(ui.view === 'cloud')
  }

  function applyWireframe(): void {
    planet.setWireframe(ui.wireframe)
  }

  function applyVertices(): void {
    planet.setShowVertices(ui.vertices)
  }

  function applyFreeze(): void {
    planet.setFrozen(ui.freezeLod)
  }

  function applyGizmos(): void {
    planet.setGizmosVisible(ui.gizmos)
  }

  function applyWater(): void {
    // Sea level is derived from waterBudget via planet.getSeaLevel() (normalized elevation).
    // Radius = RADIUS + seaLevel * HEIGHT_SCALE. Hide shell entirely when water is toggled off
    // or when the planet has no meaningful water (seaLevel < 0, i.e. below lowest terrain).
    const seaLevel = planet.getSeaLevel()
    water.visible = ui.water && seaLevel > -1
    water.scale.setScalar((RADIUS + seaLevel * HEIGHT_SCALE) / RADIUS)
  }

  // --- Tabbed debug panel ---
  tabbed = new TabbedGui(['Planet', 'Atmosphere', 'View'])

  // === Tab 1: Planet — Presets folder (must be added first to appear at top) ===

  // Single source-of-truth preset table. Each entry maps to one button in the Presets folder.
  const PLANET_PRESETS = [
    { name: 'Earth',            mass: 1.0,  composition: 0.5, age: 0.5, insolation: 1.0,  waterBudget: 0.5,  obliquity: 23  },
    { name: 'Mars',             mass: 0.5,  composition: 0.5, age: 0.7, insolation: 0.43, waterBudget: 0.15, obliquity: 25  },
    { name: 'Venus',            mass: 0.85, composition: 0.5, age: 0.5, insolation: 1.9,  waterBudget: 0.2,  obliquity: 3   },
    { name: 'Mercury',          mass: 0.4,  composition: 0.6, age: 0.85,insolation: 6.7,  waterBudget: 0.05, obliquity: 0   },
    { name: 'Ocean World',      mass: 1.2,  composition: 0.4, age: 0.4, insolation: 0.9,  waterBudget: 0.95, obliquity: 23  },
    { name: 'Heat-pipe (Io)',   mass: 1.5,  composition: 0.7, age: 0.1, insolation: 1.0,  waterBudget: 0.1,  obliquity: 0   },
    { name: 'Lava World',       mass: 1.0,  composition: 0.5, age: 0.3, insolation: 15.0, waterBudget: 0.3,  obliquity: 10  },
  ] as const

  // Wrap button functions in an object so lil-gui's add(obj, fn) pattern works.
  const presetButtons: Record<string, () => void> = {}
  for (const preset of PLANET_PRESETS) {
    presetButtons[preset.name] = () => {
      // Step 1: update ui fields so Interior sliders reflect the preset immediately.
      ui.mass        = preset.mass
      ui.composition = preset.composition
      ui.age         = preset.age
      ui.insolation  = preset.insolation
      ui.waterBudget = preset.waterBudget
      ui.obliquity   = preset.obliquity

      // Step 2: apply obliquity first — sets rotation.z + pole-axis uniform, then
      // triggers an interim regenerate with the OLD interior values (invisible: both
      // calls are synchronous and the renderer hasn't painted yet).
      planet.setAxialTilt(preset.obliquity)

      // Step 3: apply the 5 interior roots — merges and triggers the final regenerate
      // with everything in its correct state. This is the rebuild that matters.
      planet.setInteriorParams({
        mass:        preset.mass,
        composition: preset.composition,
        age:         preset.age,
        insolation:  preset.insolation,
        waterBudget: preset.waterBudget,
      })

      // Step 4: sync all GUI displays (Interior sliders + derived readouts).
      refreshDerived()
      tabbed.controllersRecursive().forEach((c) => c.updateDisplay())
    }
  }

  const presetsFolder = tabbed.planetGui.addFolder('Presets')
  for (const preset of PLANET_PRESETS) {
    presetsFolder.add(presetButtons, preset.name).name(preset.name)
  }

  // === Tab 1: Planet ===
  const planetFolder = tabbed.planetGui.addFolder('Planet')
  planetFolder.add(ui, 'seed').name('seed').listen().disable()
  planetFolder.add(ui, 'randomizeSeed').name('new seed')
  planetFolder.add(ui, 'regenerate').name('regenerate (same seed)')

  // Interior folder — five fundamental root inputs that drive Stage A → regime derivation.
  // mass and insolation replace the old planetSize/internalHeat/surfaceTemp sliders.
  const interiorFolder = tabbed.planetGui.addFolder('Interior')
  interiorFolder.add(ui, 'mass', 0.05, 10, 0.01).name('mass (Earth=1)').onChange((v: number) => {
    planet.setInteriorParams({ mass: v })
    refreshDerived()
  })
  interiorFolder.add(ui, 'insolation', 0.1, 20, 0.01).name('insolation (Earth=1)').onChange((v: number) => {
    planet.setInteriorParams({ insolation: v })
    refreshDerived()
  })
  interiorFolder.add(ui, 'waterBudget', 0, 1, 0.01).name('water budget').onChange((v: number) => {
    planet.setInteriorParams({ waterBudget: v })
    refreshDerived()
  })
  interiorFolder.add(ui, 'age', 0, 1, 0.01).name('age').onChange((v: number) => {
    planet.setInteriorParams({ age: v })
    refreshDerived()
  })
  interiorFolder.add(ui, 'composition', 0, 1, 0.01).name('composition').onChange((v: number) => {
    planet.setInteriorParams({ composition: v })
    refreshDerived()
  })
  interiorFolder.add(ui, 'obliquity', 0, 90, 0.1).name('obliquity °').onChange((v: number) => {
    planet.setAxialTilt(v)
    refreshDerived()
  })

  // Derived readouts — disabled (read-only) controllers updated by refreshDerived().
  const derivedRegimeCtrl = interiorFolder.add(ui, 'derivedRegime').name('regime').disable()
  const derivedGravityCtrl = interiorFolder.add(ui, 'derivedGravity').name('gravity (g)').disable()
  const derivedInternalHeatCtrl = interiorFolder.add(ui, 'derivedInternalHeat').name('internal heat').disable()
  const derivedSurfaceTempCtrl = interiorFolder.add(ui, 'derivedSurfaceTemp').name('surface temp °C').disable()
  const derivedAtmosphereCtrl = interiorFolder.add(ui, 'derivedAtmosphere').name('atmosphere').disable()
  const derivedEquilibriumTempCtrl = interiorFolder.add(ui, 'derivedEquilibriumTemp').name('eq. temp °C').disable()
  const derivedVigorCtrl = interiorFolder.add(ui, 'derivedVigor').name('vigor').disable()
  const derivedMobilityCtrl = interiorFolder.add(ui, 'derivedMobility').name('mobility').disable()
  const derivedPlateCountCtrl = interiorFolder.add(ui, 'derivedPlateCount').name('plate count').disable()
  const derivedOceanCoverageCtrl = interiorFolder.add(ui, 'derivedOceanCoverage').name('ocean coverage').disable()
  const derivedSeaLevelCtrl = interiorFolder.add(ui, 'derivedSeaLevel').name('sea level').disable()

  // Assign the real refreshDerived now that the controllers exist.
  refreshDerived = () => {
    const d = planet.getDerivedInterior()
    ui.derivedRegime = d.regime
    ui.derivedGravity = parseFloat(d.gravity.toFixed(3))
    ui.derivedInternalHeat = parseFloat(d.internalHeat.toFixed(3))
    ui.derivedSurfaceTemp = parseFloat(d.surfaceTemp.toFixed(1))
    ui.derivedAtmosphere = parseFloat(d.atmosphere.toFixed(3))
    ui.derivedEquilibriumTemp = parseFloat(d.equilibriumTemp.toFixed(1))
    ui.derivedVigor = parseFloat(d.vigor.toFixed(3))
    ui.derivedMobility = parseFloat(d.mobility.toFixed(3))
    ui.derivedPlateCount = d.plateCount
    ui.derivedOceanCoverage = (d.oceanCoverage * 100).toFixed(1) + '%'
    ui.derivedSeaLevel = parseFloat(planet.getSeaLevel().toFixed(4))
    derivedRegimeCtrl.updateDisplay()
    derivedGravityCtrl.updateDisplay()
    derivedInternalHeatCtrl.updateDisplay()
    derivedSurfaceTempCtrl.updateDisplay()
    derivedAtmosphereCtrl.updateDisplay()
    derivedEquilibriumTempCtrl.updateDisplay()
    derivedVigorCtrl.updateDisplay()
    derivedMobilityCtrl.updateDisplay()
    derivedPlateCountCtrl.updateDisplay()
    derivedOceanCoverageCtrl.updateDisplay()
    derivedSeaLevelCtrl.updateDisplay()
    applyWater()
  }

  // Advanced (override) folder — pins the mid-layer physical state, bypassing the root→physical
  // derivation. Only active when overrideInterior is checked.
  const advancedFolder = tabbed.planetGui.addFolder('Advanced')
  advancedFolder.add(ui, 'overrideInterior').name('override interior').onChange((v: boolean) => {
    planet.setOverrideInterior(v)
    refreshDerived()
  })
  advancedFolder.add(ui, 'manualHeat', 0, 1, 0.01).name('manual heat (0–1)').onChange((v: number) => {
    planet.setManualHeat(v)
    refreshDerived()
  })
  advancedFolder.add(ui, 'manualSurfaceTemp', -90, 1500, 1).name('manual surface temp °C').onChange((v: number) => {
    planet.setManualSurfaceTemp(v)
    refreshDerived()
  })
  advancedFolder.add(ui, 'manualAtmosphere', 0, 1, 0.01).name('manual atmosphere (0–1)').onChange((v: number) => {
    planet.setManualAtmosphere(v)
    refreshDerived()
  })

  const motionFolder = tabbed.planetGui.addFolder('Motion')
  motionFolder.add(ui, 'spin').name('spin')
  // Rotation period drives both the visual spin (live) and the climate band count
  // (faster spin → more circulation cells); the band-count change applies on regenerate.
  motionFolder.add(ui, 'spinPeriodS', 60, 3600).name('period (s)').onChange((v: number) => {
    planet.setRotationPeriod(v)
  })

  // Erosion folder — triggers a synchronous bake (cost scales as res²·iters;
  // 512 ≈ 4× heavier than 256), so both controls use onFinishChange (apply on
  // release) rather than onChange (apply every tick).
  const erosionFolder = tabbed.planetGui.addFolder('Erosion')
  erosionFolder.add(ui, 'erosionRes', [128, 256, 512]).name('bake res').onFinishChange((v: number) => {
    planet.setErosionRes(v as 128 | 256 | 512)
  })
  erosionFolder.add(ui, 'erosionStrength', 0, 3, 0.01).name('strength (K0 ×)').onFinishChange((v: number) => {
    planet.setErosionStrength(v)
  })
  erosionFolder.add(ui, 'erosionDeposition', 0, 3, 0.01).name('deposition (G ×)').onFinishChange((v: number) => {
    planet.setErosionDeposition(v)
  })
  erosionFolder.add(ui, 'bSteps', 1, 100, 1).name('B steps').onFinishChange((v: number) => {
    planet.setBSteps(v)
  })
  // Range scales with HEIGHT_SCALE: 0–300 at 1200 m/step → 0–3000 at 12000 m/step.
  erosionFolder.add(ui, 'bUpliftRate', 0, Math.round(300 * (HEIGHT_SCALE / HEIGHT_SCALE_REF)), 1).name('B uplift rate (m/step)').onFinishChange((v: number) => {
    planet.setBUpliftRate(v)
  })

  // === Tab 2: Atmosphere ===

  // Climate: atmosphere = how uniform vs extreme the gradients are. baseTemp has moved to
  // Interior (surfaceTemp). Remaining sliders are all climate-specific.
  const climateFolder = tabbed.atmosphereGui.addFolder('Climate')
  climateFolder.add(ui, 'climateField', ['temperature', 'moisture', 'insolation']).name('field').onChange((v: 'temperature' | 'moisture' | 'insolation') => {
    planet.setClimateField(v)
  })
  climateFolder.add(ui, 'redistribution', 0, 1, 0.01).name('redistribution (day/night↔latitude)').onChange((v: number) => {
    planet.setRedistribution(v)
  })
  climateFolder.add(ui, 'greenhouse', 0, 100, 1).name('greenhouse °C')
    .onChange((v: number) => {
      planet.setGreenhouse(v)          // live: updates uniform immediately
    })
    .onFinishChange(() => {
      planet.regenerate(seed)          // rebake: biome temperature uses greenhouse offset
    })
  climateFolder.add(ui, 'lapseRate', 0, 80, 1).name('lapse rate °C')
    .onChange((v: number) => {
      planet.setLapseRate(v)           // live: updates uniform immediately
    })
    .onFinishChange(() => {
      planet.regenerate(seed)          // rebake: biome temperature uses lapse rate
    })
  climateFolder.add(ui, 'tempRange', 0, 120, 1).name('temp range °C (equator→pole)').onChange((v: number) => {
    planet.setTempRange(v)
  })

  // Wind: bake params trigger a full rebake; render params are live arrow rebuilds/scales.
  // The old analytic shader uniforms (windBands, windStrength, turbulence, retrograde)
  // are retired — the wind view now uses the baked-field WindDebug arrows over normal terrain.
  const windFolder = tabbed.atmosphereGui.addFolder('Wind')
  windFolder.add(ui, 'windSwirlStrength', 0, 2, 0.01).name('swirl strength').onFinishChange((v: number) => {
    planet.setWindSwirl(v)
  })
  windFolder.add(ui, 'windHighs', 0, 12, 1).name('highs / hemisphere').onFinishChange((v: number) => {
    planet.setWindHighs(v)
  })
  windFolder.add(ui, 'windLows', 0, 12, 1).name('lows / hemisphere').onFinishChange((v: number) => {
    planet.setWindLows(v)
  })
  windFolder.add(ui, 'windCoriolis', 0, 1.57, 0.01).name('Coriolis (rad)').onFinishChange((v: number) => {
    planet.setWindCoriolis(v)
  })
  windFolder.add(ui, 'windVortexSize', 0.05, 1.0, 0.01).name('vortex size (rad)').onFinishChange((v: number) => {
    planet.setWindVortexSize(v)
  })
  windFolder.add(ui, 'windLatSpread', 0, 0.785, 0.01).name('vortex lat spread (rad)').onFinishChange((v: number) => {
    planet.setWindLatSpread(v)
  })
  windFolder.add(ui, 'windEquatorTaper', 0, 0.5, 0.01).name('equator taper (rad)').onFinishChange((v: number) => {
    planet.setWindEquatorTaper(v)
  })
  windFolder.add(ui, 'windDriftSpeed', 0, 0.1, 0.001).name('drift speed').onChange((v: number) => {
    planet.setWindDrift(v)
  })
  windFolder.add(ui, 'windPulseRate', 0, 1.5, 0.01).name('pulse rate').onChange((v: number) => {
    planet.setWindPulseRate(v)
  })
  windFolder.add(ui, 'windPulseDepth', 0, 1, 0.01).name('pulse depth').onChange((v: number) => {
    planet.setWindPulseDepth(v)
  })
  windFolder.add(ui, 'windEddyStrength', 0, 1, 0.01).name('eddy strength').onChange((v: number) => {
    planet.setWindEddyStrength(v)
  })
  windFolder.add(ui, 'windEddyScale', 1, 20, 0.5).name('eddy scale').onChange((v: number) => {
    planet.setWindEddyScale(v)
  })
  windFolder.add(ui, 'windEddyTimeScale', 0, 1.5, 0.01).name('eddy time scale').onChange((v: number) => {
    planet.setWindEddyTimeScale(v)
  })
  windFolder.add(ui, 'windArrowDensity', 100, 5000, 50).name('arrow density').onChange((v: number) => {
    planet.setWindArrowDensity(v)
  })
  windFolder.add(ui, 'windArrowScale', 0.1, 5.0, 0.05).name('arrow length').onChange((v: number) => {
    planet.setWindArrowScale(v)
  })
  windFolder.add(ui, 'showWindArrows').name('wind arrows').onChange((v: boolean) => {
    planet.setWindArrowsVisible(v)
  })
  windFolder.add(ui, 'showWindFlow').name('wind flow').onChange((v: boolean) => {
    planet.setWindFlowVisible(v)
  })
  windFolder.add(ui, 'windFlowDensity', 1000, 30000, 500).name('flow density').onFinishChange((v: number) => {
    planet.setWindFlowDensity(v)
  })
  windFolder.add(ui, 'windFlowSpeed', 0, 3.0, 0.05).name('flow speed').onChange((v: number) => {
    planet.setWindFlowSpeed(v)
  })
  windFolder.add(ui, 'windFlowTrail', 2, 24, 1).name('trail length').onFinishChange((v: number) => {
    planet.setWindFlowTrail(v)
  })
  windFolder.add(ui, 'windFlowLifetime', 1, 10, 0.5).name('particle life').onChange((v: number) => {
    planet.setWindFlowLifetime(v)
  })

  // Cloud shell overlay — always visible in normal view; all sliders are live (no rebake).
  const cloudFolder = tabbed.atmosphereGui.addFolder('Clouds')
  cloudFolder.add(ui, 'clouds').name('visible (off by default)').onChange(() => applyView())
  cloudFolder.add(ui, 'cloudCoverage', 0, 1, 0.01).name('coverage').onChange((v: number) => {
    planet.setCloudCoverage(v)
  })
  cloudFolder.add(ui, 'cloudScrollSpeed', 0, 0.02, 0.0005).name('drift speed').onChange((v: number) => {
    planet.setCloudScrollSpeed(v)
  })
  cloudFolder.add(ui, 'cloudScale', 1, 16, 0.1).name('cloud scale').onChange((v: number) => {
    planet.setCloudScale(v)
  })
  cloudFolder.add(ui, 'cloudWarp', 0, 1, 0.01).name('wind warp').onChange((v: number) => {
    planet.setCloudWarp(v)
  })
  cloudFolder.add(ui, 'cloudBillow', 0, 1, 0.01).name('billow').onChange((v: number) => {
    planet.setCloudBillow(v)
  })
  cloudFolder.add(ui, 'cloudDetail', 0, 1, 0.01).name('detail').onChange((v: number) => {
    planet.setCloudDetail(v)
  })
  cloudFolder.add(ui, 'cloudSoftness', 0.05, 0.6, 0.01).name('softness').onChange((v: number) => {
    planet.setCloudSoftness(v)
  })
  cloudFolder.add(ui, 'cloudVolume', 0, 1, 0.01).name('volume').onChange((v: number) => {
    planet.setCloudVolume(v)
  })
  cloudFolder.add(ui, 'cloudOpacity', 0, 1, 0.01).name('opacity').onChange((v: number) => {
    planet.setCloudOpacity(v)
  })
  cloudFolder.add(ui, 'cloudFavWeight', 0, 1, 0.01).name('favorability').onChange((v: number) => {
    planet.setCloudFavWeight(v)
  })
  cloudFolder.add(ui, 'cloudMoistWeight', 0, 1, 0.01).name('moisture wt').onChange((v: number) => {
    planet.setCloudMoistWeight(v)
  })
  cloudFolder.add(ui, 'cloudConvWeight', 0, 1, 0.01).name('convergence wt').onChange((v: number) => {
    planet.setCloudConvWeight(v)
  })
  cloudFolder.add(ui, 'cloudConvGain', 0, 20, 0.5).name('conv gain').onChange((v: number) => {
    planet.setCloudConvGain(v)
  })
  cloudFolder.add(ui, 'cloudItczWeight', 0, 0.5, 0.01).name('latitude bands').onChange((v: number) => {
    planet.setCloudItczWeight(v)
  })
  cloudFolder.add(ui, 'cloudWeatherWeight', 0, 0.8, 0.01).name('weather systems').onChange((v: number) => {
    planet.setCloudWeatherWeight(v)
  })
  cloudFolder.add(ui, 'cloudCellWeight', 0, 1, 0.01).name('weather cells').onChange((v: number) => {
    planet.setCloudCellWeight(v)
  })
  cloudFolder.add(ui, 'cloudBase', 1.0, 2.0, 0.05).name('cloud base').onChange((v: number) => {
    planet.setCloudBase(v)
  })
  cloudFolder.add(ui, 'cloudThick', 0.2, 1.2, 0.05).name('cloud thickness').onChange((v: number) => {
    planet.setCloudThick(v)
  })
  cloudFolder.add(ui, 'cloudDensity', 0.5, 12, 0.1).name('density').onChange((v: number) => {
    planet.setCloudDensity(v)
  })
  cloudFolder.add(ui, 'cloudStepCount', 8, 96, 4).name('step count').onChange((v: number) => {
    planet.setCloudStepCount(v)
  })
  cloudFolder.add(ui, 'cloudLightSteps', 1, 8, 1).name('light steps').onChange((v: number) => {
    planet.setCloudLightSteps(v)
  })
  cloudFolder.add(ui, 'cloudHg', 0, 0.9, 0.01).name('HG anisotropy').onChange((v: number) => {
    planet.setCloudHg(v)
  })
  cloudFolder.add(ui, 'cloudPowder', 0, 3, 0.05).name('powder').onChange((v: number) => {
    planet.setCloudPowder(v)
  })
  cloudFolder.add(ui, 'cloudDetailScale', 1, 12, 0.5).name('detail scale').onChange((v: number) => {
    planet.setCloudDetailScale(v)
  })
  cloudFolder.add(ui, 'cloudRoundBase', 0.05, 0.6, 0.01).name('round base').onChange((v: number) => {
    planet.setCloudRoundBase(v)
  })
  cloudFolder.add(ui, 'cloudBillowTop', 0.1, 0.9, 0.01).name('billow top').onChange((v: number) => {
    planet.setCloudBillowTop(v)
  })
  cloudFolder.add(ui, 'cloudAmbient', 0, 0.6, 0.01).name('ambient').onChange((v: number) => {
    planet.setCloudAmbient(v)
  })
  cloudFolder.add(ui, 'cloudType', 0, 1, 0.01).name('cloud type').onChange((v: number) => {
    planet.setCloudType(v)
  })
  // Half-res offscreen cloud pass. Bound directly to ui.cloudHalfRes; read in the render loop.
  cloudFolder.add(ui, 'cloudHalfRes').name('half-res (offscreen, exp.)')

  // Atmosphere shell — always-on appearance (not a view mode); all sliders are live (no rebake).
  const atmosphereFolder = tabbed.atmosphereGui.addFolder('Atmosphere shell')
  atmosphereFolder.add(ui, 'atmosphereDensity', 0, 2, 0.01).name('atmosphere').onChange((v: number) => {
    planet.setAtmosphereDensity(v)
  })
  atmosphereFolder.addColor(ui, 'atmosphereTint').name('tint').onChange((v: string) => {
    // Convert '#rrggbb' hex to THREE.Color and pass to planet
    planet.setAtmosphereTint(new THREE.Color(v))
  })
  atmosphereFolder.add(ui, 'atmosphereSunIntensity', 0, 4, 0.05).name('sun intensity').onChange((v: number) => {
    planet.setAtmosphereSunIntensity(v)
  })
  atmosphereFolder.add(ui, 'atmosphereScaleHeight', 0, 8, 0.1).name('horizon thickness').onChange((v: number) => {
    planet.setAtmosphereScaleHeight(v)
  })
  atmosphereFolder.add(ui, 'atmosphereSkyFloor', 0, 1, 0.01).name('sky floor').onChange((v: number) => {
    planet.setAtmosphereSkyFloor(v)
  })
  atmosphereFolder.add(ui, 'atmosphereHeight', 1.0, 6.0, 0.05).name('shell height').onChange((v: number) => {
    planet.setAtmosphereHeight(v)
  })
  atmosphereFolder.add(ui, 'atmosphereOn').name('enabled').onChange((v: boolean) => {
    planet.setAtmosphereVisible(v)
  })

  // Sun direction controls — world-space azimuth + elevation converted to a unit vector.
  // The sun is used for climate (insolation/terminator), not scene lighting.
  const sunFolder = tabbed.atmosphereGui.addFolder('Sun')
  sunFolder.add(ui, 'sunAzimuth', 0, 360, 1).name('azimuth °').onChange(() => applySunDir())
  sunFolder.add(ui, 'sunElevation', -90, 90, 1).name('elevation °').onChange(() => applySunDir())

  // === Tab 3: View ===
  const viewFolder = tabbed.viewGui.addFolder('View')
  viewFolder.add(ui, 'view', ['normal', 'lod', 'tectonics', 'heightmap', 'climate', 'wind', 'cloud', 'materials', 'wetness']).name('mode').onChange(() => applyView())
  viewFolder.add(ui, 'wireframe').name('wireframe').onChange(() => applyWireframe())
  viewFolder.add(ui, 'vertices').name('vertices').onChange(() => applyVertices())
  viewFolder.add(ui, 'freezeLod').name('freeze LOD').onChange(() => applyFreeze())
  viewFolder.add(ui, 'gizmos').name('gizmos').onChange(() => applyGizmos())

  // LOD tuning: targetTriPx and maxDepth are live — no rebuild needed.
  const lodFolder = tabbed.viewGui.addFolder('LOD')
  lodFolder.add(ui, 'targetTriPx', 0.5, 8, 0.1).name('tri px target').onChange((v: number) => {
    planet.setTargetTriPx(v)
  })
  lodFolder.add(ui, 'maxDepth', 8, 20, 1).name('max depth').onChange((v: number) => {
    planet.setMaxDepth(v)
  })
  lodFolder.add(ui, 'lodDiag').name('lod diag').onChange((v: boolean) => {
    planet.setDiagEnabled(v)
  })

  const waterFolder = tabbed.viewGui.addFolder('Water')
  waterFolder.add(ui, 'water').name('visible').onChange(() => applyWater())

  // Apply initial gizmos + water state
  applyGizmos()
  applyWater()

  // Initialise interior params + axial tilt to defaults — sets them on the planet and
  // populates derived readouts. setAxialTilt also applies rotation.z and refreshes the
  // pole-axis uniform, replacing the manual rotation.z assignment that used to live here.
  planet.setInteriorParams({
    mass: ui.mass,
    composition: ui.composition,
    age: ui.age,
    insolation: ui.insolation,
    waterBudget: ui.waterBudget,
  })
  planet.setAxialTilt(ui.obliquity)
  refreshDerived()

  // Construct CaveManager after planet's first build (planet.setInteriorParams above triggers
  // regenerate, so getCaveField() is valid here). Attach headlamp to camera and give the
  // first-person controller the collider via setCollider() (it's already constructed above).
  // Guard against a null cave field (cave construction can fail/disable itself in
  // Planet.buildHeightFn without aborting the planet build). No cave field → no cave
  // system this session; the planet still renders and is fully walkable on the surface.
  const _caveField = planet.getCaveField()
  if (_caveField !== null) {
    caveManager = new CaveManager({ field: _caveField, planet })
    caveManager.attachHeadlamp(camera)
    firstPerson.setCollider(caveManager.getCollider())
  } else {
    console.warn('[cave] no cave field available — cave system disabled for this session')
  }

  // Initialise climate/sun uniforms to match slider defaults.
  // These calls only update uniforms/fields; no regenerate() is triggered here.
  // The constructor's buildHeightFn already baked with these default values.
  planet.setRedistribution(ui.redistribution)
  planet.setGreenhouse(ui.greenhouse)
  planet.setLapseRate(ui.lapseRate)
  planet.setTempRange(ui.tempRange)
  planet.setClimateField(ui.climateField)
  applySunDir()

  // Initialise atmosphere uniforms from ui defaults so the first frame renders correctly.
  planet.setAtmosphereDensity(ui.atmosphereDensity)
  planet.setAtmosphereTint(new THREE.Color(ui.atmosphereTint))
  planet.setAtmosphereSunIntensity(ui.atmosphereSunIntensity)
  planet.setAtmosphereScaleHeight(ui.atmosphereScaleHeight)
  planet.setAtmosphereSkyFloor(ui.atmosphereSkyFloor)
  planet.setAtmosphereHeight(ui.atmosphereHeight)
  planet.setAtmosphereVisible(ui.atmosphereOn)

  // Initialise cloud uniforms from ui defaults so the first frame renders correctly.
  // VolumetricClouds defaults already match these values, but explicit init keeps ui in sync
  // with any future default changes and follows the same pattern as atmosphere above.
  planet.setCloudBase(ui.cloudBase)
  planet.setCloudThick(ui.cloudThick)
  planet.setCloudDensity(ui.cloudDensity)
  planet.setCloudStepCount(ui.cloudStepCount)
  planet.setCloudLightSteps(ui.cloudLightSteps)
  planet.setCloudHg(ui.cloudHg)
  planet.setCloudPowder(ui.cloudPowder)
  planet.setCloudDetailScale(ui.cloudDetailScale)
  planet.setCloudRoundBase(ui.cloudRoundBase)
  planet.setCloudBillowTop(ui.cloudBillowTop)
  planet.setCloudAmbient(ui.cloudAmbient)
  planet.setCloudType(ui.cloudType)

  // Initialise view-gated visibility at startup (cloud shell, wind overlays, debug views).
  // Without this the cloud shell stays hidden on load until the view is changed, because
  // its visibility is only set inside applyView() / hotkey handlers.
  applyView()

  // Wind bake params are already at their defaults (Climate constructor defaults match ui defaults).
  // Arrow density + scale match Planet field defaults — no explicit init needed.

  // --- Hotkeys (route through ui state then sync GUI displays) ---
  window.addEventListener('keydown', (e) => {
    if (e.repeat) return

    switch (e.key) {
      case '1':
        ui.wireframe = !ui.wireframe
        applyWireframe()
        break
      case '2':
        ui.view = ui.view === 'lod' ? 'normal' : 'lod'
        applyView()
        break
      case '3':
        ui.view = ui.view === 'tectonics' ? 'normal' : 'tectonics'
        applyView()
        break
      case '4':
        ui.spin = !ui.spin
        break
      case '5':
        ui.view = ui.view === 'climate' ? 'normal' : 'climate'
        applyView()
        break
      case '6':
        ui.view = ui.view === 'wind' ? 'normal' : 'wind'
        applyView()
        break
      case '7':
        ui.view = ui.view === 'materials' ? 'normal' : 'materials'
        applyView()
        break
      case '8':
        ui.view = ui.view === 'wetness' ? 'normal' : 'wetness'
        applyView()
        break
      case 'f':
        ui.freezeLod = !ui.freezeLod
        applyFreeze()
        break
      case 'g':
        ui.randomizeSeed()
        return // randomizeSeed already calls controllersRecursive().updateDisplay()
      case 'c': {
        // Teleport to the cave mouth (surface spawn at entrance so player can walk in).
        // Only allowed from Globe mode — mirrors the "walk here" entry flow exactly.
        if (navMode !== NavMode.Globe || caveManager === null) break

        // Ensure world quaternion is current before reading getMouthWorld().
        caveManager.update(camera.position)
        const m = caveManager.getMouthWorld()

        // Save state + pause spin (planet rotating under a world-fixed walker is wrong).
        savedCamPos = camera.position.clone()
        savedCamUp  = camera.up.clone()
        savedSpin   = ui.spin
        ui.spin     = false

        // Dispose GlobeControls so it stops capturing input.
        globeControls.dispose()

        // Spawn the player standing at the cave mouth, facing inward (surface mode so they
        // can walk to the edge and fall in, auto-switching to cave mode on descent).
        firstPerson.spawn({ position: m.pos, up: m.up }, 'surface')
        navMode = NavMode.FirstPerson
        firstPerson.enable()

        renderer.domElement.requestPointerLock()
        bar.setMode(NavMode.FirstPerson)
        break
      }
    }
    tabbed.controllersRecursive().forEach((c) => c.updateDisplay())
  })

  // --- Half-resolution cloud compositor (Part 2a) — only used when `cloud half-res` is on. ---
  const cloudCompositor = new CloudCompositor(window.innerWidth, window.innerHeight)

  // --- Resize ---
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
    cloudCompositor.setSize(window.innerWidth, window.innerHeight)
  })

  // --- Loop ---
  const clock = new THREE.Clock()
  let elapsedS = 0
  let frameCount = 0
  let _loadingHidden = false
  let fpsAccum = 0
  let smoothFps = 0
  // Scratch for reading drawing-buffer size (zero-alloc per frame).
  const _drawingSize = new THREE.Vector2()
  // Scratch for the camera view-projection matrix (zero-alloc per frame).
  const _viewProj = new THREE.Matrix4()

  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1)
    elapsedS += dt
    planet.setWindTime(elapsedS)
    planet.animateWind(elapsedS, dt, camera.position)

    // Dispatch input update to whichever controller owns this frame.
    // Only ONE controller's update runs per frame — no double input on the camera.
    if (navMode === NavMode.Globe) {
      globeControls.update(dt)
    } else if (navMode === NavMode.FirstPerson) {
      firstPerson.update(dt)
    }
    // NavMode.Placement: no controller update — camera is frozen at saved position

    if (ui.spin) {
      // rotateY post-multiplies a local-axis quaternion, so with the axial tilt already
      // applied this spins about the tilted polar axis — correct Earth-like behaviour.
      planet.rotateY((2 * Math.PI / ui.spinPeriodS) * dt)
    }

    // Always update terrain LOD — must run every frame regardless of nav mode so
    // terrain keeps loading under the player in FP mode.
    // camera.fov is in degrees (Three.js convention) — convert to radians for the SSE metric.
    renderer.getDrawingBufferSize(_drawingSize)
    _viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
    planet.update(camera.position, THREE.MathUtils.degToRad(camera.fov), _drawingSize.y, _viewProj)

    // Update cave manager AFTER planet.update() so the planet world quaternion is fresh.
    // Headlamp follows the player into the cave and turns off on surface.
    if (caveManager !== null) {
      // The cave subsystem is experimental + near-field; a fault in it must NOT
      // black out the whole planet. Isolate it: on error, log the cause once and
      // disable it for the session so the main render keeps running.
      try {
        caveManager.update(camera.position)
        caveManager.setHeadlampEnabled(firstPerson.mode === 'cave')
      } catch (err) {
        console.error('[cave] update threw — disabling cave system to keep rendering:', err)
        caveManager = null
      }
    }

    // Lighting is fixed surround (set up once at init) — nothing to update per frame.

    // Smooth FPS over ~30 frames
    fpsAccum += 1 / dt
    frameCount++
    if (frameCount % 30 === 0) {
      smoothFps = fpsAccum / 30
      fpsAccum = 0
    }

    if (frameCount % 10 === 0) {
      const stats = planet.getStats()
      const altRaw = camera.position.length() - planet.getSurfaceRadiusAt(camera.position)
      const altStr = altRaw >= 1_000_000
        ? `${(altRaw / 1_000_000).toFixed(2)} Mm`
        : altRaw >= 1_000
          ? `${(altRaw / 1_000).toFixed(2)} km`
          : `${altRaw.toFixed(0)} m`
      const lodStr = stats.leaves > 0
        ? `${stats.minLevel}–${stats.maxLevel} (${stats.leaves} leaves)`
        : `— (0 leaves)`

      hud.update({
        backend: backendName,
        fps: smoothFps.toFixed(0),
        alt: altStr,
        speed: navMode === NavMode.Globe ? `${globeControls.currentSpeed.toFixed(0)} u/s` : '—',
        seed: String(seed),
        regime: planet.getDerivedInterior().regime,
        plates: String(stats.plates),
        volcanoes: String(stats.volcanoes),
        hotspots: String(stats.hotspots),
        spin: ui.spin ? 'on' : 'off',
        view: ui.view,
        mode: navMode,
        lod: lodStr,
        'lod cap': String(ui.maxDepth),
        'lod cached': String(stats.cached),
        'builds': String(stats.pendingBuilds),
        'build ms': stats.lastBuildMs.toFixed(1),
      })
    }

    // Half-res offscreen cloud path (Part 2a) when enabled; otherwise the normal single pass.
    // The compositor falls back to a plain render when clouds are hidden (non-normal view).
    if (ui.cloudHalfRes) {
      cloudCompositor.render(renderer, scene, camera, planet.getCloudMesh())
    } else {
      renderer.render(scene, camera)
    }

    // Fade out the loading overlay once a few frames have rendered (by then the WebGPU
    // shader pipelines are compiled and the planet is actually on screen, not black).
    if (!_loadingHidden && frameCount > 4) {
      _loadingHidden = true
      setProgress(100, 'Ready')
      loadingEl?.classList.add('hidden')
      window.setTimeout(() => loadingEl?.remove(), 700)
    }
  })
}

main()
