// Shared message protocol between the main thread (MeshWorkerPool) and the
// mesh workers (meshWorker.ts). Kept in its own file so both sides import one
// source of truth and can be authored in parallel.

import type { TectonicsBaked } from './tectonics'
import type { ClimateBaked } from './climate'
import type { ErosionBaked } from './erosion'
import type { SubsurfaceBaked } from './subsurface'

/** main → worker, once on startup: everything needed to rebuild the terrain sampler. */
export interface InitMsg {
  type: 'init'
  seed: number
  radius: number
  heightScale: number
  resolution: number
  plateCount: number
  arcDensity: number
  baseTemp: number
  atmosphere: number
  bandCount: number
  axialTiltRad: number
  redistribution?: number
  greenhouse?: number
  lapseRate?: number
  /** Plate angular velocity multiplier from interior vigor (informational; workers use baked omega). */
  driftScale?: number
  /** Baked tectonic fields (transferred as a clone — worker reconstructs via Tectonics.fromBaked). */
  tectonics: TectonicsBaked
  /** Baked climate field (worker reconstructs via Climate.fromBaked). */
  climate: ClimateBaked
  /** Baked erosion field (worker reconstructs via Erosion.fromBaked). */
  erosion?: ErosionBaked
  /** Baked subsurface field (worker reconstructs via Subsurface.fromBaked). */
  subsurface?: SubsurfaceBaked
  /**
   * Waterline elevation in the same normalized units as heightFn output (~[-1,1]).
   * Computed once on the main thread via Fibonacci-sphere hypsometry and shipped
   * to workers so they never recompute it (determinism guarantee).
   */
  seaLevel: number
  /**
   * When true, orogenic stamps (broadUplift, plateau, broadDeform, relief) are gated
   * off so the baked bDelta field owns the elevation budget. Must mirror the flag used
   * to build the main-thread final sampler — omitting it causes seams.
   */
  bActive?: boolean
}

/** main → worker, per chunk to mesh. resolution is fixed at init, so not repeated here. */
export interface BuildMsg {
  type: 'build'
  id: number
  faceIndex: number
  level: number
  ix: number
  iy: number
}

/** main → worker, best-effort drop of a chunk no longer wanted. */
export interface CancelMsg {
  type: 'cancel'
  id: number
}

export type WorkerInMsg = InitMsg | BuildMsg | CancelMsg

/** worker → main, once init + sampler reconstruction completes. */
export interface ReadyMsg {
  type: 'ready'
}

/** worker → main, a finished chunk. The buffers are transferred (zero-copy). */
export interface DoneMsg {
  type: 'done'
  id: number
  positions: ArrayBuffer
  normals: ArrayBuffer
  colors: ArrayBuffer
  plateColors: ArrayBuffer | null
  climateMoist: ArrayBuffer | null
  subsurfaceWet: ArrayBuffer | null
  /** Per-vertex rock hardness scalar in [0,1]. null when tectonics not available. */
  rockHardness: ArrayBuffer | null
  indices: ArrayBuffer
  originX: number
  originY: number
  originZ: number
  /** Exact element counts (buffers are exactly sized, but pass these for robust view construction). */
  vertCount: number
  indexCount: number
}

export type WorkerOutMsg = ReadyMsg | DoneMsg
