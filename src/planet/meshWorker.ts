/**
 * Web Worker entry point — meshes terrain chunks off the main thread.
 *
 * Lifecycle:
 *   1. Main thread sends InitMsg once: worker reconstructs Tectonics + Climate
 *      from their baked fields and builds the shared terrain sampler. Posts ReadyMsg.
 *   2. Main thread sends BuildMsg per chunk: worker calls computeChunkArrays and
 *      posts DoneMsg with the five typed-array buffers as transferables (zero-copy).
 *   3. CancelMsg is a best-effort hint — the worker is serial so there is nothing
 *      to cancel; the message is silently ignored.
 *
 * Worker-safety: no DOM, no window, no document. All imports are pure compute.
 */

import { computeChunkArrays } from './ChunkMesher'
import { makeTerrainSampler } from './terrainSampler'
import { Tectonics } from './tectonics'
import { Climate } from './climate'
import { Erosion } from './erosion'
import { Subsurface } from './subsurface'
import type { WorkerInMsg, WorkerOutMsg, ReadyMsg, DoneMsg } from './meshProtocol'

// ---------------------------------------------------------------------------
// Minimal worker-scope typing — avoids DOM lib's Window overload for postMessage
// and scopes onmessage to the correct input message type.
// (tsconfig has no "WebWorker" lib, so we define what we need locally.)
// ---------------------------------------------------------------------------

interface MeshWorkerScope {
  onmessage: ((e: MessageEvent<WorkerInMsg>) => void) | null
  postMessage(message: WorkerOutMsg, transfer: Transferable[]): void
}

const ctx = self as unknown as MeshWorkerScope

// ---------------------------------------------------------------------------
// Worker state — set once on InitMsg, consumed on every BuildMsg.
// ---------------------------------------------------------------------------

let sampler: ReturnType<typeof makeTerrainSampler> | null = null
let RADIUS       = 0
let HEIGHT_SCALE = 0
let RES          = 0
let SEA_LEVEL    = 0

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

ctx.onmessage = (e: MessageEvent<WorkerInMsg>): void => {
  const msg = e.data

  if (msg.type === 'init') {
    const tectonics  = Tectonics.fromBaked(msg.tectonics)
    const climate    = Climate.fromBaked(msg.climate)
    const erosion    = msg.erosion    != null ? Erosion.fromBaked(msg.erosion)                          : undefined
    const subsurface = msg.subsurface != null ? Subsurface.fromBaked(msg.subsurface, tectonics) : undefined

    sampler = makeTerrainSampler({
      seed:           msg.seed,
      radius:         msg.radius,
      heightScale:    msg.heightScale,
      plateCount:     msg.plateCount,
      arcDensity:     msg.arcDensity,
      baseTemp:       msg.baseTemp,
      atmosphere:     msg.atmosphere,
      bandCount:      msg.bandCount,
      axialTiltRad:   msg.axialTiltRad,
      redistribution: msg.redistribution,
      greenhouse:     msg.greenhouse,
      lapseRate:      msg.lapseRate,
      tectonics,
      climate,
      erosion,
      subsurface,
      bActive:        msg.bActive,
    })

    RADIUS       = msg.radius
    HEIGHT_SCALE = msg.heightScale
    RES          = msg.resolution
    SEA_LEVEL    = msg.seaLevel

    const ready: ReadyMsg = { type: 'ready' }
    ctx.postMessage(ready, [])

  } else if (msg.type === 'build') {
    const a = computeChunkArrays({
      faceIndex:    msg.faceIndex,
      level:        msg.level,
      ix:           msg.ix,
      iy:           msg.iy,
      resolution:   RES,
      radius:       RADIUS,
      heightScale:  HEIGHT_SCALE,
      heightFn:     sampler!.heightFn,
      plateColorFn: sampler!.plateColorFn,
      climateFn:    sampler!.climateFn,
      erosion:      sampler!.erosion,
      subsurface:   sampler!.subsurface,
      hardnessFn:   (dir) => sampler!.tectonics.hardnessAt(dir),
    }, SEA_LEVEL)

    const done: DoneMsg = {
      type:         'done',
      id:           msg.id,
      positions:    a.positions.buffer as ArrayBuffer,
      normals:      a.normals.buffer as ArrayBuffer,
      colors:       a.colors.buffer as ArrayBuffer,
      plateColors:  a.plateColors   ? (a.plateColors.buffer   as ArrayBuffer) : null,
      climateMoist:   a.climateMoist   ? (a.climateMoist.buffer   as ArrayBuffer) : null,
      subsurfaceWet:  a.subsurfaceWet  ? (a.subsurfaceWet.buffer  as ArrayBuffer) : null,
      rockHardness:   a.rockHardness   ? (a.rockHardness.buffer   as ArrayBuffer) : null,
      indices:        a.indices.buffer as ArrayBuffer,
      originX:      a.originX,
      originY:      a.originY,
      originZ:      a.originZ,
      vertCount:    a.positions.length / 3,
      indexCount:   a.indices.length,
    }

    const transferList: Transferable[] = [
      a.positions.buffer,
      a.normals.buffer,
      a.colors.buffer,
      a.indices.buffer,
    ]
    if (a.plateColors) {
      transferList.push(a.plateColors.buffer)
    }
    if (a.climateMoist) {
      transferList.push(a.climateMoist.buffer)
    }
    if (a.subsurfaceWet) {
      transferList.push(a.subsurfaceWet.buffer)
    }
    if (a.rockHardness) {
      transferList.push(a.rockHardness.buffer)
    }

    ctx.postMessage(done, transferList)

  } // else msg.type === 'cancel' — serial worker, nothing in-flight to cancel
}
