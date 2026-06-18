import {
  BufferGeometry,
  Color,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  OctahedronGeometry,
  Quaternion,
  Vector3,
} from 'three'
import { Tectonics, TectonicQuery } from './tectonics'
import { buildArrowGeometry, fibonacciSphere } from './arrowGeometry'

// ---------------------------------------------------------------------------
// TectonicsDebug
// ---------------------------------------------------------------------------

export class TectonicsDebug extends Group {
  private readonly _geometries: BufferGeometry[] = []
  private readonly _materials:  MeshBasicMaterial[] = []
  private readonly _meshes: (Mesh | InstancedMesh)[] = []

  constructor(
    tectonics: Tectonics,
    opts: {
      radius:          number
      heightScale:     number
      surfaceRadiusAt: (dir: Vector3) => number
    }
  ) {
    super()
    this.visible = false

    const { radius, heightScale, surfaceRadiusAt } = opts

    // -----------------------------------------------------------------------
    // 1. VELOCITY FIELD — instanced arrows, ~240 samples
    //
    // Arrow size: max length = radius * 0.05 ≈ 1,000 units at radius=20,000.
    // At a 20,000-unit viewing distance: angular size = 2·arctan(500/20000) ≈ 2.86°.
    // Shaft radius ≈ 2.5% of length, head radius ≈ 7%, head length ≈ 28%.
    // -----------------------------------------------------------------------
    const SAMPLE_COUNT  = 240
    const FIELD_MAX_LEN = radius * 0.05   // ~1,000 units at r=20,000

    const sampleDirs = fibonacciSphere(SAMPLE_COUNT)

    // First pass: compute velocities and find maxSpeed — fix nit (a): call
    // velocityAt(dir, new Vector3()) directly rather than cloning _vel.
    const velocities: Vector3[] = []
    let maxSpeed = 1e-6 // avoid div-by-zero
    for (const dir of sampleDirs) {
      const v = tectonics.velocityAt(dir, new Vector3())
      velocities.push(v)
      const spd = v.length()
      if (spd > maxSpeed) maxSpeed = spd
    }

    // Field arrow geometry: shaft 2.5% of length, head 7% base, 28% length
    const fieldArrowGeo = buildArrowGeometry(0.025, 0.07, 0.28)
    this._geometries.push(fieldArrowGeo)

    const fieldArrowMat = new MeshBasicMaterial({ color: 0xffffff, vertexColors: false })
    this._materials.push(fieldArrowMat)

    const instancedArrows = new InstancedMesh(fieldArrowGeo, fieldArrowMat, SAMPLE_COUNT)
    instancedArrows.count = 0

    const _q       = new Quaternion()
    const _zHat    = new Vector3(0, 0, 1)
    const _mat4    = new Matrix4()
    const _pos     = new Vector3()
    const _col     = new Color()
    // Reused scratch for query() — avoids a fresh allocation per arrow.
    // Using the real warped+weighted ownership keeps field-arrow colors
    // consistent with the terrain's painted plate regions near boundaries.
    const _scratch: TectonicQuery = { plateId: 0, neighborId: 0, boundaryDist: 0, convergence: 0, shear: 0, crustDist: 0, paleoDist: 0, otherCrustDist: 0, baseElevation: 0, rockHardness: 0 }

    let instanceIdx = 0
    for (let i = 0; i < sampleDirs.length; i++) {
      const dir = sampleDirs[i]
      const vel = velocities[i]
      const spd = vel.length()
      if (spd < 1e-4) continue // skip near-zero velocity

      // Position: hover heightScale * 2.0 above terrain
      const surfR = surfaceRadiusAt(dir)
      _pos.copy(dir).multiplyScalar(surfR + heightScale * 2.0)

      // Orientation: rotate +Z onto normalized velocity direction
      const velDir = vel.clone().normalize()
      _q.setFromUnitVectors(_zHat, velDir)

      // Scale: length ≈ FIELD_MAX_LEN * (0.45 + 0.55 * speed/maxSpeed)
      const scale = FIELD_MAX_LEN * (0.45 + 0.55 * spd / maxSpeed)

      _mat4.makeRotationFromQuaternion(_q)
      _mat4.scale(new Vector3(scale, scale, scale))
      _mat4.setPosition(_pos)
      instancedArrows.setMatrixAt(instanceIdx, _mat4)

      // Color: owning plate color via real warped+weighted query, brightened 1.35×.
      // This matches the terrain's painted ownership exactly — raw dot-product
      // ownership near boundaries would mis-color arrows inside a big neighbor's cell.
      const plateIdx = tectonics.query(dir, _scratch).plateId
      const [r, g, b] = tectonics.plates[plateIdx].color
      _col.setRGB(
        Math.min(1, r * 1.35),
        Math.min(1, g * 1.35),
        Math.min(1, b * 1.35),
      )
      instancedArrows.setColorAt(instanceIdx, _col)

      instanceIdx++
    }

    instancedArrows.count = instanceIdx
    instancedArrows.instanceMatrix.needsUpdate = true
    if (instancedArrows.instanceColor) instancedArrows.instanceColor.needsUpdate = true

    this.add(instancedArrows)
    this._meshes.push(instancedArrows)

    // -----------------------------------------------------------------------
    // 2. PER-PLATE ARROWS — two meshes per plate for contrast
    //
    // Arrow length = radius * 0.09 ≈ 1,800 units at radius=20,000.
    // At a 20,000-unit viewing distance: angular size = 2·arctan(900/20000) ≈ 5.15°.
    //
    // Two-mesh approach: a slightly thicker white arrow underneath + the
    // plate-colored arrow on top at 92% scale. ~32 extra draw calls in a
    // debug view — negligible. Gives clear contrast against same-hue ground.
    // Shaft 3% of length, head 8% base, 28% length (proportionally thicker).
    // -----------------------------------------------------------------------
    const PLATE_ARROW_LEN = radius * 0.09  // ~1,800 units at r=20,000

    // White backing arrow: slightly wider (shaft 3.8%, head 9.5%)
    const plateArrowWhiteGeo = buildArrowGeometry(0.038, 0.095, 0.28)
    this._geometries.push(plateArrowWhiteGeo)

    // Colored foreground arrow: same proportions as above but will be scaled to 92%
    const plateArrowColorGeo = buildArrowGeometry(0.030, 0.080, 0.28)
    this._geometries.push(plateArrowColorGeo)

    for (const plate of tectonics.plates) {
      const seedDir = plate.seedDir.clone().normalize()
      const vel = tectonics.velocityAt(seedDir, new Vector3())
      const spd = vel.length()
      if (spd < 1e-4) continue

      const velDir = vel.clone().normalize()

      const surfR = surfaceRadiusAt(seedDir)
      // Hover heightScale * 3.0 above terrain
      _pos.copy(seedDir).multiplyScalar(surfR + heightScale * 3.0)

      // White backing mesh — drawn at full PLATE_ARROW_LEN scale
      const whiteMat = new MeshBasicMaterial({ color: 0xffffff })
      this._materials.push(whiteMat)

      const whiteMesh = new Mesh(plateArrowWhiteGeo, whiteMat)
      whiteMesh.position.copy(_pos)
      whiteMesh.quaternion.setFromUnitVectors(_zHat, velDir)
      whiteMesh.scale.setScalar(PLATE_ARROW_LEN)

      // Plate-colored foreground mesh — drawn at 92% scale so white shows around edges
      const [r, g, b] = plate.color
      const colorMat = new MeshBasicMaterial({ color: new Color(r, g, b) })
      this._materials.push(colorMat)

      const colorMesh = new Mesh(plateArrowColorGeo, colorMat)
      colorMesh.position.copy(_pos)
      colorMesh.quaternion.setFromUnitVectors(_zHat, velDir)
      colorMesh.scale.setScalar(PLATE_ARROW_LEN * 0.92)

      this.add(whiteMesh, colorMesh)
      this._meshes.push(whiteMesh, colorMesh)
    }

    // -----------------------------------------------------------------------
    // 3. VOLCANO MARKERS — instanced octahedra, one per arc volcano
    //
    // OctahedronGeometry is orientation-free so no axis-bake needed.
    // Scale = radius * 0.012 (uniform) — large enough to read from orbit.
    // Lift = heightScale * 2.0 above surfaceRadiusAt (which already includes
    // the baked terrain cone height), matching the velocity-arrow hover idiom.
    // Color: 0xff5a1e (hot orange), MeshBasicMaterial (unlit / full-bright).
    // Only arc volcanoes (kind === 'arc') are shown here; hotspot cones below.
    // -----------------------------------------------------------------------
    const arcVolcs = tectonics.volcanoes.filter(v => v.kind === 'arc')

    if (arcVolcs.length > 0) {
      const VOLCANO_SCALE = radius * 0.012

      const volcanoGeo = new OctahedronGeometry(1)
      this._geometries.push(volcanoGeo)

      const volcanoMat = new MeshBasicMaterial({ color: 0xff5a1e, vertexColors: false })
      this._materials.push(volcanoMat)

      const volcanoMesh = new InstancedMesh(volcanoGeo, volcanoMat, arcVolcs.length)
      volcanoMesh.count = 0

      const _scaleVec = new Vector3()
      const _identityQuat = new Quaternion()

      for (let i = 0; i < arcVolcs.length; i++) {
        const v = arcVolcs[i]
        const surfR = surfaceRadiusAt(v.pos)
        _pos.copy(v.pos).multiplyScalar(surfR + heightScale * 2.0)
        _scaleVec.setScalar(VOLCANO_SCALE)
        _mat4.compose(_pos, _identityQuat, _scaleVec)
        volcanoMesh.setMatrixAt(i, _mat4)
      }

      volcanoMesh.count = arcVolcs.length
      volcanoMesh.instanceMatrix.needsUpdate = true

      this.add(volcanoMesh)
      this._meshes.push(volcanoMesh)
    }

    // -----------------------------------------------------------------------
    // 3b. HOTSPOT CONE MARKERS — instanced octahedra, one per hotspot volcano
    //
    // Color: 0xff2a6d (bright magenta-red), visually distinct from arc orange.
    // Size: HOTSPOT_SCALE * (0.7 + 0.5 * clamp(intensity, 0, 3)) per instance,
    // so intensity=1 → scale*1.2, intensity=3 → scale*2.2 (bigger than arc pins).
    // -----------------------------------------------------------------------
    const hotspotVolcs = tectonics.volcanoes.filter(v => v.kind === 'hotspot')

    if (hotspotVolcs.length > 0) {
      const HOTSPOT_SCALE = radius * 0.012

      const hotspotGeo = new OctahedronGeometry(1)
      this._geometries.push(hotspotGeo)

      const hotspotMat = new MeshBasicMaterial({ color: 0xff2a6d, vertexColors: false })
      this._materials.push(hotspotMat)

      const hotspotMesh = new InstancedMesh(hotspotGeo, hotspotMat, hotspotVolcs.length)
      hotspotMesh.count = 0

      const _hsScaleVec = new Vector3()
      const _identityQuat2 = new Quaternion()

      for (let i = 0; i < hotspotVolcs.length; i++) {
        const v = hotspotVolcs[i]
        const scale = HOTSPOT_SCALE * (0.7 + 0.5 * Math.min(3, v.intensity))
        const surfR = surfaceRadiusAt(v.pos)
        _pos.copy(v.pos).multiplyScalar(surfR + heightScale * 2.0)
        _hsScaleVec.setScalar(scale)
        _mat4.compose(_pos, _identityQuat2, _hsScaleVec)
        hotspotMesh.setMatrixAt(i, _mat4)
      }

      hotspotMesh.count = hotspotVolcs.length
      hotspotMesh.instanceMatrix.needsUpdate = true

      this.add(hotspotMesh)
      this._meshes.push(hotspotMesh)
    }

    // -----------------------------------------------------------------------
    // 3c. PLUME-SOURCE MARKERS — instanced arrows, one per hotspot plume
    //
    // Arrow points radially outward (+Z oriented along hs.pos) from the surface.
    // Color: 0xff79c6 (bright pink) — distinct from both arc orange and hotspot red.
    // Length: PLUME_LEN = radius * 0.05 * (0.8 + 0.4 * clamp(intensity, 0, 3))
    // so a plume arrow is clearly larger than the ~radius*0.012..0.035 cone octahedra.
    // Arrow shape: chunky pin — shaft 4%, head 11%, head-length 30%.
    // -----------------------------------------------------------------------
    if (tectonics.hotspots.length > 0) {
      const plumeArrowGeo = buildArrowGeometry(0.04, 0.11, 0.30)
      this._geometries.push(plumeArrowGeo)

      const plumeArrowMat = new MeshBasicMaterial({ color: 0xff79c6, vertexColors: false })
      this._materials.push(plumeArrowMat)

      const plumeMesh = new InstancedMesh(plumeArrowGeo, plumeArrowMat, tectonics.hotspots.length)
      plumeMesh.count = 0

      for (let i = 0; i < tectonics.hotspots.length; i++) {
        const hs = tectonics.hotspots[i]
        const PLUME_LEN = radius * 0.05 * (0.8 + 0.4 * Math.min(3, hs.intensity))
        const surfR = surfaceRadiusAt(hs.pos)
        _pos.copy(hs.pos).multiplyScalar(surfR + heightScale * 2.0)
        _q.setFromUnitVectors(_zHat, hs.pos.clone().normalize())
        _mat4.makeRotationFromQuaternion(_q)
        _mat4.scale(new Vector3(PLUME_LEN, PLUME_LEN, PLUME_LEN))
        _mat4.setPosition(_pos)
        plumeMesh.setMatrixAt(i, _mat4)
      }

      plumeMesh.count = tectonics.hotspots.length
      plumeMesh.instanceMatrix.needsUpdate = true

      this.add(plumeMesh)
      this._meshes.push(plumeMesh)
    }

    // NOTE: Pole axis has been removed from TectonicsDebug.
    // It now lives in PlanetGizmos (always-visible, independent of tectonics view).
  }

  dispose(): void {
    // Geometries — deduplicated by reference (plate arrow geos are shared across meshes)
    const seenGeos = new Set<BufferGeometry>()
    for (const geo of this._geometries) {
      if (!seenGeos.has(geo)) { seenGeos.add(geo); geo.dispose() }
    }
    for (const mat of this._materials) mat.dispose()
    this._geometries.length = 0
    this._materials.length  = 0
    this._meshes.length     = 0
  }
}
