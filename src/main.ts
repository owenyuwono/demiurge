import * as THREE from 'three/webgpu'
import GUI from 'lil-gui'
import { Planet } from './planet/Planet.js'
import { GlobeControls } from './controls/GlobeControls.js'
import { FirstPersonController } from './controls/FirstPersonController.js'
import { SurfacePicker } from './controls/SurfacePicker.js'
import { NavMode, type SurfaceSampler } from './controls/NavMode.js'
import { NavControlsBar } from './debug/NavControlsBar.js'
import { Hud } from './debug/Hud.js'

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

  // Axial tilt: local Y is the spin axis; tilting Z by 23.4° makes the orbit Earth-like.
  planet.rotation.z = THREE.MathUtils.degToRad(23.4)

  // Polar axis in world space, computed once after the tilt is applied.
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

  // --- UI state (single source of truth for GUI + hotkeys) ---
  const ui = {
    view: 'normal' as 'normal' | 'lod' | 'tectonics',
    wireframe: false,
    vertices: false,
    freezeLod: false,
    gizmos: true,
    spin: true,
    spinPeriodS: 600,
    plateCount: 16,
    arcDensity: 1,
    targetTriPx: 2.0,
    maxDepth: 18,
    water: true,
    waterLevel: 0.5,
    seed: String(seed),
    randomizeSeed: () => {
      const newSeed = (Math.random() * 2 ** 31) | 0
      seed = newSeed
      ui.seed = String(newSeed)
      planet.regenerate(newSeed)
      const url = new URL(window.location.href)
      url.searchParams.set('seed', String(newSeed))
      history.replaceState(null, '', url.toString())
      gui.controllersRecursive().forEach((c) => c.updateDisplay())
    },
    regenerate: () => {
      planet.regenerate(seed)
    },
  }

  // --- Apply functions (state → scene/loop) ---
  function applyView(): void {
    planet.setDebugColors(ui.view === 'lod')
    planet.setTectonicsView(ui.view === 'tectonics')
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
    // waterLevel ∈ [0,1]: 0 = shell below the deepest terrain (dry), 1 = above the highest
    // peak (fully submerged), 0.5 = sea level at terrain height 0 (coastlines). The ±1.05
    // margin clears any clamped trench/peak so the extremes are truly dry / fully ocean.
    water.visible = ui.water && ui.waterLevel > 0
    const norm = (2 * ui.waterLevel - 1) * 1.05
    water.scale.setScalar((RADIUS + norm * HEIGHT_SCALE) / RADIUS)
  }

  // --- lil-gui panel (top-right, default placement) ---
  const gui = new GUI({ title: 'Demiurge' })

  const viewFolder = gui.addFolder('View')
  viewFolder.add(ui, 'view', ['normal', 'lod', 'tectonics']).name('mode').onChange(() => applyView())
  viewFolder.add(ui, 'wireframe').name('wireframe').onChange(() => applyWireframe())
  viewFolder.add(ui, 'vertices').name('vertices').onChange(() => applyVertices())
  viewFolder.add(ui, 'freezeLod').name('freeze LOD').onChange(() => applyFreeze())
  viewFolder.add(ui, 'gizmos').name('gizmos').onChange(() => applyGizmos())

  const planetFolder = gui.addFolder('Planet')
  planetFolder.add(ui, 'seed').name('seed').listen().disable()
  planetFolder.add(ui, 'randomizeSeed').name('new seed')
  planetFolder.add(ui, 'regenerate').name('regenerate (same seed)')

  // plateCount + regenerate grouped together: changing plateCount takes effect on next regenerate
  const tectonicsFolder = gui.addFolder('Tectonics')
  tectonicsFolder.add(ui, 'plateCount', 0, 48, 1).name('plate count (0/1 = no tectonics)').onChange((n: number) => {
    planet.setPlateCount(n)
  })
  tectonicsFolder.add(ui, 'arcDensity', 0.2, 3, 0.05).name('arc density').onChange((v: number) => {
    planet.setArcDensity(v)
  })

  const motionFolder = gui.addFolder('Motion')
  motionFolder.add(ui, 'spin').name('spin')
  motionFolder.add(ui, 'spinPeriodS', 60, 3600).name('period (s)')

  // LOD tuning: targetTriPx and maxDepth are live — no rebuild needed.
  const lodFolder = gui.addFolder('LOD')
  lodFolder.add(ui, 'targetTriPx', 0.5, 8, 0.1).name('tri px target').onChange((v: number) => {
    planet.setTargetTriPx(v)
  })
  lodFolder.add(ui, 'maxDepth', 8, 20, 1).name('max depth').onChange((v: number) => {
    planet.setMaxDepth(v)
  })

  const waterFolder = gui.addFolder('Water')
  waterFolder.add(ui, 'water').name('visible').onChange(() => applyWater())
  waterFolder.add(ui, 'waterLevel', 0, 1, 0.01).name('level (0=dry, 1=ocean)').onChange(() => applyWater())

  // Apply initial gizmos + water state
  applyGizmos()
  applyWater()

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
      case 'f':
        ui.freezeLod = !ui.freezeLod
        applyFreeze()
        break
      case 'g':
        ui.randomizeSeed()
        return // randomizeSeed already calls controllersRecursive().updateDisplay()
    }
    gui.controllersRecursive().forEach((c) => c.updateDisplay())
  })

  // --- Resize ---
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  // --- Loop ---
  const clock = new THREE.Clock()
  let frameCount = 0
  let fpsAccum = 0
  let smoothFps = 0
  // Scratch for reading drawing-buffer size (zero-alloc per frame).
  const _drawingSize = new THREE.Vector2()

  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1)

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
    planet.update(camera.position, THREE.MathUtils.degToRad(camera.fov), _drawingSize.y)

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
        plates: String(stats.plates),
        volcanoes: String(stats.volcanoes),
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
