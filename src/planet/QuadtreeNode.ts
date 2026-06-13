import { Vector3 } from 'three'
import { patchCenterDir } from './faceBases'

export class QuadtreeNode {
  readonly faceIndex: number
  readonly level: number
  readonly ix: number
  readonly iy: number

  /** Unique cache key, identical to the key used by Planet for geometry lookups. */
  readonly key: string

  /** Unit-sphere direction of this patch's centre. */
  readonly centerDir: Vector3

  /**
   * Approximate world-space centre of the patch (planet origin at 0,0,0).
   * Equals centerDir · radius; radius is injected at construction time.
   */
  readonly worldCenter: Vector3

  /**
   * Approximate arc-length side of this patch:
   *   nodeSize = (π/2 · radius) / 2^level
   */
  readonly nodeSize: number

  children: [QuadtreeNode, QuadtreeNode, QuadtreeNode, QuadtreeNode] | null = null

  /**
   * Terrain-height-adjusted centre, in planet-local space.
   * Null until first LOD metric use; computed and cached at that point by Planet.
   * Equals centerDir * (radius + heightFn(centerDir) * heightScale).
   */
  surfaceCenter: Vector3 | null = null

  constructor(faceIndex: number, level: number, ix: number, iy: number, radius: number) {
    this.faceIndex = faceIndex
    this.level = level
    this.ix = ix
    this.iy = iy
    this.key = `${faceIndex}/${level}/${ix}/${iy}`

    this.centerDir = patchCenterDir(faceIndex, level, ix, iy)
    this.worldCenter = this.centerDir.clone().multiplyScalar(radius)
    this.nodeSize = ((Math.PI / 2) * radius) / Math.pow(2, level)
  }

  /** Create the 4 child nodes (2ix+cx, 2iy+cy). */
  split(radius: number): void {
    if (this.children !== null) return
    const nl = this.level + 1
    const bx = this.ix * 2
    const by = this.iy * 2
    this.children = [
      new QuadtreeNode(this.faceIndex, nl, bx,     by,     radius),
      new QuadtreeNode(this.faceIndex, nl, bx + 1, by,     radius),
      new QuadtreeNode(this.faceIndex, nl, bx,     by + 1, radius),
      new QuadtreeNode(this.faceIndex, nl, bx + 1, by + 1, radius),
    ]
  }

  /** Drop children (they become garbage-collected when callers release refs). */
  merge(): void {
    this.children = null
  }
}
