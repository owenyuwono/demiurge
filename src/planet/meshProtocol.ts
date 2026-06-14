// Shared message protocol between the main thread (MeshWorkerPool) and the
// mesh workers (meshWorker.ts). Kept in its own file so both sides import one
// source of truth and can be authored in parallel.

import type { TectonicsBaked } from './tectonics'
import type { ClimateBaked } from './climate'

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
  /** Baked tectonic fields (transferred as a clone — worker reconstructs via Tectonics.fromBaked). */
  tectonics: TectonicsBaked
  /** Baked climate field (worker reconstructs via Climate.fromBaked). */
  climate: ClimateBaked
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

/** worker → main, a finished chunk. The five buffers are transferred (zero-copy). */
export interface DoneMsg {
  type: 'done'
  id: number
  positions: ArrayBuffer
  normals: ArrayBuffer
  colors: ArrayBuffer
  plateColors: ArrayBuffer | null
  indices: ArrayBuffer
  originX: number
  originY: number
  originZ: number
  /** Exact element counts (buffers are exactly sized, but pass these for robust view construction). */
  vertCount: number
  indexCount: number
}

export type WorkerOutMsg = ReadyMsg | DoneMsg
