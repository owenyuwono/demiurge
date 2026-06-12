import * as THREE from 'three/webgpu'
import GUI from 'lil-gui'
import { Planet } from './planet/Planet.js'
import { GlobeControls } from './controls/GlobeControls.js'
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

  // Headlight: moved to camera position each frame so the whole visible globe is lit.
  // The default target remains at the origin, so it always shines from camera → planet.
  // Slope normals still shade terrain relief; no dark side.
  const sun = new THREE.DirectionalLight(0xffffff, 2.5)
  scene.add(sun)

  // Raised to 0.55 so grazing terrain near the limb doesn't go fully black.
  const hemi = new THREE.HemisphereLight(0x88aaff, 0x1a140e, 0.55)
  scene.add(hemi)

  // --- Planet ---
  let seed = getSeedFromUrl()
  const planet = new Planet({ seed, radius: RADIUS, heightScale: HEIGHT_SCALE, maxDepth: 12 })
  scene.add(planet)

  // Axial tilt: local Y is the spin axis; tilting Z by 23.4° makes the orbit Earth-like.
  planet.rotation.z = THREE.MathUtils.degToRad(23.4)

  // Polar axis in world space, computed once after the tilt is applied.
  // rotateY() post-multiplies about local Y, so this world axis never changes during spin.
  const POLAR_AXIS = new THREE.Vector3(0, 1, 0).applyQuaternion(planet.quaternion)

  // --- Camera ---
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.5, RADIUS * 15)
  camera.position.set(0, 0, RADIUS * 3)
  camera.lookAt(0, 0, 0)

  // --- GlobeControls (replaces FlyCamera) ---
  const controls = new GlobeControls(camera, renderer.domElement, {
    getAltitude: (pos) => pos.length() - planet.getSurfaceRadiusAt(pos),
    getPolarAxis: () => POLAR_AXIS,
    radius: RADIUS,
    minAltitude: 30,
    maxAltitude: RADIUS * 8,
  })

  // --- HUD ---
  const hud = new Hud()

  // --- UI state (single source of truth for GUI + hotkeys) ---
  const ui = {
    view: 'normal' as 'normal' | 'lod' | 'tectonics',
    wireframe: false,
    freezeLod: false,
    gizmos: true,
    spin: true,
    spinPeriodS: 600,
    plateCount: 16,
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

  function applyFreeze(): void {
    planet.setFrozen(ui.freezeLod)
  }

  function applyGizmos(): void {
    planet.setGizmosVisible(ui.gizmos)
  }

  // --- lil-gui panel (top-right, default placement) ---
  const gui = new GUI({ title: 'Demiurge' })

  const viewFolder = gui.addFolder('View')
  viewFolder.add(ui, 'view', ['normal', 'lod', 'tectonics']).name('mode').onChange(() => applyView())
  viewFolder.add(ui, 'wireframe').name('wireframe').onChange(() => applyWireframe())
  viewFolder.add(ui, 'freezeLod').name('freeze LOD').onChange(() => applyFreeze())
  viewFolder.add(ui, 'gizmos').name('gizmos').onChange(() => applyGizmos())

  const planetFolder = gui.addFolder('Planet')
  planetFolder.add(ui, 'seed').name('seed').listen().disable()
  planetFolder.add(ui, 'randomizeSeed').name('new seed')
  planetFolder.add(ui, 'regenerate').name('regenerate (same seed)')

  // plateCount + regenerate grouped together: changing plateCount takes effect on next regenerate
  const tectonicsFolder = gui.addFolder('Tectonics')
  tectonicsFolder.add(ui, 'plateCount', 4, 48, 1).name('plate count').onChange((n: number) => {
    planet.setPlateCount(n)
  })

  const motionFolder = gui.addFolder('Motion')
  motionFolder.add(ui, 'spin').name('spin')
  motionFolder.add(ui, 'spinPeriodS', 60, 3600).name('period (s)')

  // Apply initial gizmos state
  applyGizmos()

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

  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1)

    controls.update(dt)

    if (ui.spin) {
      // rotateY post-multiplies a local-axis quaternion, so with the axial tilt already
      // applied this spins about the tilted polar axis — correct Earth-like behaviour.
      planet.rotateY((2 * Math.PI / ui.spinPeriodS) * dt)
    }

    planet.update(camera.position)

    // Headlight: copy camera world position so the sun always faces whatever the user sees.
    sun.position.copy(camera.position)

    // Smooth FPS over ~30 frames
    fpsAccum += 1 / dt
    frameCount++
    if (frameCount % 30 === 0) {
      smoothFps = fpsAccum / 30
      fpsAccum = 0
    }

    if (frameCount % 10 === 0) {
      const stats = planet.getStats()
      const altKm = (camera.position.length() - planet.getSurfaceRadiusAt(camera.position)) / 1000

      hud.update({
        backend: backendName,
        fps: smoothFps.toFixed(0),
        altitude: `${altKm.toFixed(2)} km`,
        speed: `${controls.currentSpeed.toFixed(0)} u/s`,
        seed: String(seed),
        plates: String(stats.plates),
        spin: ui.spin ? 'on' : 'off',
        view: ui.view,
        'lod leaves': String(stats.leaves),
        'lod cached': String(stats.cached),
        'lod depth': String(stats.maxLevel),
        'builds': String(stats.pendingBuilds),
        'build ms': stats.lastBuildMs.toFixed(1),
      })
    }

    renderer.render(scene, camera)
  })
}

main()
