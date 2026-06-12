import { Vector3 } from 'three'

// ---------------------------------------------------------------------------
// Face bases — 6 faces of the cube, each described by a normal and two tangent
// axes so that (u, v) ∈ [-1,1]² spans the face in a right-handed way.
//
//   cube point = normal + u * tangentU + v * tangentV,  before scale to [-1,1]³
//
// Face order: 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z
//
// These are the canonical conventions for this project. Both QuadtreeNode and
// ChunkMesher import from here so LOD distance checks and mesh geometry always
// agree on which patch a (faceIndex, level, ix, iy) tuple refers to.
// ---------------------------------------------------------------------------

export interface FaceBasis {
  nx: number; ny: number; nz: number; // unit normal (cube face direction)
  ux: number; uy: number; uz: number; // U tangent
  vx: number; vy: number; vz: number; // V tangent
}

export const FACE_BASES: readonly FaceBasis[] = [
  // +X: normal = +X, u = +Z, v = +Y
  { nx:  1, ny:  0, nz:  0,  ux:  0, uy:  0, uz:  1,  vx:  0, vy:  1, vz:  0 },
  // −X: normal = −X, u = −Z, v = +Y
  { nx: -1, ny:  0, nz:  0,  ux:  0, uy:  0, uz: -1,  vx:  0, vy:  1, vz:  0 },
  // +Y: normal = +Y, u = +X, v = +Z
  { nx:  0, ny:  1, nz:  0,  ux:  1, uy:  0, uz:  0,  vx:  0, vy:  0, vz:  1 },
  // −Y: normal = −Y, u = +X, v = −Z
  { nx:  0, ny: -1, nz:  0,  ux:  1, uy:  0, uz:  0,  vx:  0, vy:  0, vz: -1 },
  // +Z: normal = +Z, u = −X, v = +Y
  { nx:  0, ny:  0, nz:  1,  ux: -1, uy:  0, uz:  0,  vx:  0, vy:  1, vz:  0 },
  // −Z: normal = −Z, u = +X, v = +Y
  { nx:  0, ny:  0, nz: -1,  ux:  1, uy:  0, uz:  0,  vx:  0, vy:  1, vz:  0 },
]

// ---------------------------------------------------------------------------
// Improved cube-to-sphere mapping (Everitt/Zucker style).
// Eliminates the area-distortion of plain normalize by warping each axis via
// the other two:
//   sx = x · √(1 − y²/2 − z²/2 + y²z²/3)   (cyclic for sy, sz)
// Input cube point must be in [-1, 1]³.
// ---------------------------------------------------------------------------

export function cubeToSphere(cx: number, cy: number, cz: number, out: Vector3): void {
  const x2 = cx * cx
  const y2 = cy * cy
  const z2 = cz * cz
  out.x = cx * Math.sqrt(Math.max(0, 1 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3))
  out.y = cy * Math.sqrt(Math.max(0, 1 - z2 * 0.5 - x2 * 0.5 + z2 * x2 / 3))
  out.z = cz * Math.sqrt(Math.max(0, 1 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3))
}

// ---------------------------------------------------------------------------
// Compute the unit-sphere direction of the centre of a face patch.
// The patch at (level, ix, iy) covers a 1/2^level × 1/2^level tile in
// [−1,1]² face-local space.
// ---------------------------------------------------------------------------

const _patchScratch = new Vector3()

export function patchCenterDir(faceIndex: number, level: number, ix: number, iy: number): Vector3 {
  const scale = 1.0 / (1 << level)
  const u = (ix + 0.5) * scale  // [0, 1] face coords of tile centre
  const v = (iy + 0.5) * scale
  const cu = u * 2 - 1           // [-1, 1]
  const cv = v * 2 - 1

  const b = FACE_BASES[faceIndex]
  const cx = b.nx + cu * b.ux + cv * b.vx
  const cy = b.ny + cu * b.uy + cv * b.vy
  const cz = b.nz + cu * b.uz + cv * b.vz

  cubeToSphere(cx, cy, cz, _patchScratch)
  _patchScratch.normalize()
  return _patchScratch.clone()
}
