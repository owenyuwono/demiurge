import {
  BufferGeometry,
  ConeGeometry,
  CylinderGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  TorusGeometry,
} from 'three'

// ---------------------------------------------------------------------------
// PlanetGizmos — pole-axis lines + equator ring
//
// Lives in planet-LOCAL space (this Group is a child of the rotating/tilted
// planet). Local +Y IS the spin axis, so no additional rotation is needed for
// the pole cylinders. Intended to be always-visible regardless of the
// tectonics debug overlay.
// ---------------------------------------------------------------------------

export class PlanetGizmos extends Group {
  private readonly _geometries: BufferGeometry[] = []
  private readonly _materials:  MeshBasicMaterial[] = []
  private readonly _meshes: Mesh[] = []

  constructor(opts: { radius: number }) {
    super()

    const { radius } = opts

    const POLE_THICKNESS = radius * 0.0035
    const POLE_LENGTH    = radius * 1.35
    const POLE_SEGS      = 6
    const TIP_RADIUS     = radius * 0.012
    const TIP_LENGTH     = radius * 0.04

    // -----------------------------------------------------------------------
    // North shaft: y = 0 → +POLE_LENGTH  (color #44ddff)
    // -----------------------------------------------------------------------
    const northShaftGeo = new CylinderGeometry(POLE_THICKNESS, POLE_THICKNESS, POLE_LENGTH, POLE_SEGS)
    northShaftGeo.translate(0, POLE_LENGTH / 2, 0)
    this._geometries.push(northShaftGeo)

    // North arrowhead cone tip at the very top of the north shaft
    const northTipGeo = new ConeGeometry(TIP_RADIUS, TIP_LENGTH, POLE_SEGS)
    northTipGeo.translate(0, POLE_LENGTH + TIP_LENGTH / 2, 0)
    this._geometries.push(northTipGeo)

    // -----------------------------------------------------------------------
    // South shaft: y = 0 → −POLE_LENGTH  (color #ff6644)
    // -----------------------------------------------------------------------
    const southShaftGeo = new CylinderGeometry(POLE_THICKNESS, POLE_THICKNESS, POLE_LENGTH, POLE_SEGS)
    southShaftGeo.translate(0, -POLE_LENGTH / 2, 0)
    this._geometries.push(southShaftGeo)

    const northMat = new MeshBasicMaterial({ color: 0x44ddff })
    const southMat = new MeshBasicMaterial({ color: 0xff6644 })
    this._materials.push(northMat, southMat)

    const northShaftMesh = new Mesh(northShaftGeo, northMat)
    const northTipMesh   = new Mesh(northTipGeo,   northMat)
    const southShaftMesh = new Mesh(southShaftGeo,  southMat)

    this.add(northShaftMesh, northTipMesh, southShaftMesh)
    this._meshes.push(northShaftMesh, northTipMesh, southShaftMesh)

    // -----------------------------------------------------------------------
    // Equator ring — in the local XZ plane (y = 0).
    //
    // Max terrain surface = radius + heightScale * 1 = radius * 1.02 (at
    // radius=20,000 and heightScale=400). Use radius*1.025 so the ring floats
    // just above the highest possible terrain.
    //
    // TorusGeometry lies in the XY plane by default; rotateX(π/2) flips it
    // into the XZ plane.
    // -----------------------------------------------------------------------
    const EQUATOR_RADIUS = radius * 1.025
    const EQUATOR_TUBE   = radius * 0.0028
    const equatorGeo = new TorusGeometry(EQUATOR_RADIUS, EQUATOR_TUBE, 8, 256)
    equatorGeo.rotateX(Math.PI / 2)
    this._geometries.push(equatorGeo)

    const equatorMat = new MeshBasicMaterial({ color: 0xffd54a })
    this._materials.push(equatorMat)

    const equatorMesh = new Mesh(equatorGeo, equatorMat)
    this.add(equatorMesh)
    this._meshes.push(equatorMesh)
  }

  dispose(): void {
    for (const geo of this._geometries) geo.dispose()
    for (const mat of this._materials)  mat.dispose()
    this._geometries.length = 0
    this._materials.length  = 0
    this._meshes.length     = 0
  }
}
