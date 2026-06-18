import {
  BufferAttribute,
  BufferGeometry,
  ConeGeometry,
  CylinderGeometry,
  Float32BufferAttribute,
  Vector3,
} from 'three'

// ---------------------------------------------------------------------------
// Shared helpers — reused by TectonicsDebug and WindDebug
// ---------------------------------------------------------------------------

/** Fibonacci-sphere lattice — deterministic, uniform-ish distribution. */
export function fibonacciSphere(count: number): Vector3[] {
  const dirs: Vector3[] = []
  const phi = Math.PI * (Math.sqrt(5) - 1) // golden angle
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const theta = phi * i
    dirs.push(new Vector3(r * Math.cos(theta), y, r * Math.sin(theta)))
  }
  return dirs
}

/**
 * Build a unit arrow geometry along +Z.
 *
 * Accepts proportional shaft/head parameters so callers can tune for size.
 *
 *   shaftRadiusFrac  — shaft radius as fraction of total arrow length (1.0)
 *   headRadiusFrac   — cone base radius as fraction of total length
 *   headLenFrac      — cone length as fraction of total length
 *
 * Shaft runs from z=0 to z=(1-headLenFrac). Cone tip is at z=1.
 * Returns one merged BufferGeometry with position/normal/uv + Uint32 index.
 */
export function buildArrowGeometry(
  shaftRadiusFrac = 0.025,
  headRadiusFrac  = 0.07,
  headLenFrac     = 0.28,
): BufferGeometry {
  const SHAFT_LENGTH = 1.0 - headLenFrac
  const HEAD_LENGTH  = headLenFrac
  const SHAFT_RADIUS = shaftRadiusFrac
  const HEAD_RADIUS  = headRadiusFrac
  const RADIAL_SEGS  = 6

  // Shaft: CylinderGeometry along +Y, translated so base at y=0
  const shaftGeo = new CylinderGeometry(SHAFT_RADIUS, SHAFT_RADIUS, SHAFT_LENGTH, RADIAL_SEGS, 1, true)
  shaftGeo.translate(0, SHAFT_LENGTH / 2, 0)

  // Head: ConeGeometry along +Y, base at y=SHAFT_LENGTH, tip at y=1
  const headGeo = new ConeGeometry(HEAD_RADIUS, HEAD_LENGTH, RADIAL_SEGS)
  headGeo.translate(0, SHAFT_LENGTH + HEAD_LENGTH / 2, 0)

  // Merge by concatenating attribute arrays and offsetting head indices
  const shaftPos = shaftGeo.attributes.position.array as Float32Array
  const shaftNor = shaftGeo.attributes.normal.array as Float32Array
  const shaftUv  = shaftGeo.attributes.uv.array as Float32Array
  const shaftIdx = shaftGeo.index!.array as Uint16Array | Uint32Array

  const headPos  = headGeo.attributes.position.array as Float32Array
  const headNor  = headGeo.attributes.normal.array as Float32Array
  const headUv   = headGeo.attributes.uv.array as Float32Array
  const headIdx  = headGeo.index!.array as Uint16Array | Uint32Array

  const shaftVertCount = shaftPos.length / 3

  const positions = new Float32Array(shaftPos.length + headPos.length)
  const normals   = new Float32Array(shaftNor.length + headNor.length)
  const uvs       = new Float32Array(shaftUv.length  + headUv.length)
  positions.set(shaftPos, 0);        positions.set(headPos, shaftPos.length)
  normals.set(shaftNor, 0);          normals.set(headNor,  shaftNor.length)
  uvs.set(shaftUv, 0);               uvs.set(headUv,  shaftUv.length)

  const indices = new Uint32Array(shaftIdx.length + headIdx.length)
  for (let i = 0; i < shaftIdx.length; i++) indices[i] = shaftIdx[i]
  for (let i = 0; i < headIdx.length;  i++) indices[shaftIdx.length + i] = headIdx[i] + shaftVertCount

  const geo = new BufferGeometry()
  geo.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geo.setAttribute('normal',   new Float32BufferAttribute(normals, 3))
  geo.setAttribute('uv',       new Float32BufferAttribute(uvs, 2))
  // Fix: use BufferAttribute directly instead of Array.from (Uint32BufferAttribute pattern)
  geo.setIndex(new BufferAttribute(indices, 1))

  // Bake +Y → +Z so instances only need to orient along +Z
  geo.rotateX(-Math.PI / 2)

  shaftGeo.dispose()
  headGeo.dispose()

  return geo
}
