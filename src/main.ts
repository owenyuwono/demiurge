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

const RADIUS = 50_000
const HEIGHT_SCALE = 1_200

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
  // --- Renderer ---
  // logarithmicDepthBuffer: true ensures good depth precision across the large
  // near=2 / far=RADIUS*15 range on both WebGPU and the WebGL2 fallback backend.
  const renderer = new THREE.WebGPURenderer({
    antialias: true,
    logarithmicDepthBuffer: true,
  })
  await renderer.init()

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
  const planet = new Planet({ seed, radius: RADIUS, heightScale: HEIGHT_SCALE, maxDepth: 18, targetTriPx: 2.0 })
  scene.add(planet)

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

  // tabbed declared before ui so randomizeSeed can reference tabbed.controllersRecursive()
  let tabbed: TabbedGui

  // --- UI state (single source of truth for GUI + hotkeys) ---
  const ui = {
    view: 'normal' as 'normal' | 'lod' | 'tectonics' | 'heightmap' | 'climate' | 'wind',
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
    // Wind parameters
    windField: 'flow' as 'flow' | 'speed' | 'direction',
    windStrength: 1,
    windBands: 3,
    turbulence: 0.2,
    retrograde: false,
    // Erosion bake parameters — onFinishChange triggers rebake on release.
    erosionRes: 256 as 128 | 256 | 512,
    erosionStrength: 1.0,
    erosionDeposition: 1.0,
    bSteps: 30,
    bUpliftRate: 60,
    seed: String(seed),
    randomizeSeed: () => {
      const newSeed = (Math.random() * 2 ** 31) | 0
      seed = newSeed
      ui.seed = String(newSeed)
      planet.regenerate(newSeed)
      refreshDerived()
      const url = new URL(window.location.href)
      url.searchParams.set('seed', String(newSeed))
      history.replaceState(null, '', url.toString())
      tabbed.controllersRecursive().forEach((c) => c.updateDisplay())
    },
    regenerate: () => {
      planet.regenerate(seed)
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
  erosionFolder.add(ui, 'bUpliftRate', 0, 300, 1).name('B uplift rate (m/step)').onFinishChange((v: number) => {
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

  // Wind: all parameters are live uniform writes (no rebake).
  const windFolder = tabbed.atmosphereGui.addFolder('Wind')
  windFolder.add(ui, 'windField', ['flow', 'speed', 'direction']).name('field').onChange((v: 'flow' | 'speed' | 'direction') => {
    planet.setWindField(v)
  })
  windFolder.add(ui, 'windStrength', 0, 3, 0.01).name('wind strength').onChange((v: number) => {
    planet.setWindStrength(v)
  })
  windFolder.add(ui, 'windBands', 1, 8, 1).name('bands (N)').onChange((v: number) => {
    planet.setWindBands(v)
  })
  windFolder.add(ui, 'turbulence', 0, 1, 0.01).name('turbulence').onChange((v: number) => {
    planet.setTurbulence(v)
  })
  windFolder.add(ui, 'retrograde').name('retrograde').onChange((v: boolean) => {
    planet.setRetrograde(v)
  })

  // Sun direction controls — world-space azimuth + elevation converted to a unit vector.
  // The sun is used for climate (insolation/terminator), not scene lighting.
  const sunFolder = tabbed.atmosphereGui.addFolder('Sun')
  sunFolder.add(ui, 'sunAzimuth', 0, 360, 1).name('azimuth °').onChange(() => applySunDir())
  sunFolder.add(ui, 'sunElevation', -90, 90, 1).name('elevation °').onChange(() => applySunDir())

  // === Tab 3: View ===
  const viewFolder = tabbed.viewGui.addFolder('View')
  viewFolder.add(ui, 'view', ['normal', 'lod', 'tectonics', 'heightmap', 'climate', 'wind']).name('mode').onChange(() => applyView())
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

  // Initialise climate/sun uniforms to match slider defaults.
  // These calls only update uniforms/fields; no regenerate() is triggered here.
  // The constructor's buildHeightFn already baked with these default values.
  planet.setRedistribution(ui.redistribution)
  planet.setGreenhouse(ui.greenhouse)
  planet.setLapseRate(ui.lapseRate)
  planet.setTempRange(ui.tempRange)
  planet.setClimateField(ui.climateField)
  applySunDir()

  // Initialise wind uniforms to match slider defaults (uniform-only, no rebake).
  planet.setWindField(ui.windField)
  planet.setWindStrength(ui.windStrength)
  planet.setWindBands(ui.windBands)
  planet.setTurbulence(ui.turbulence)
  planet.setRetrograde(ui.retrograde)

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
      case 'f':
        ui.freezeLod = !ui.freezeLod
        applyFreeze()
        break
      case 'g':
        ui.randomizeSeed()
        return // randomizeSeed already calls controllersRecursive().updateDisplay()
    }
    tabbed.controllersRecursive().forEach((c) => c.updateDisplay())
  })

  // --- Resize ---
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  // --- Loop ---
  const clock = new THREE.Clock()
  let elapsedS = 0
  let frameCount = 0
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

    renderer.render(scene, camera)
  })
}

main()
