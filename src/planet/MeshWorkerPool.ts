import type { InitMsg, BuildMsg, DoneMsg, ReadyMsg, WorkerOutMsg } from './meshProtocol'
import type { ChunkMeshArrays } from './ChunkMesher'

// ---------------------------------------------------------------------------
// Public request shape
// ---------------------------------------------------------------------------

export interface MeshRequest {
  key: string
  faceIndex: number
  level: number
  ix: number
  iy: number
  priority: number
}

// ---------------------------------------------------------------------------
// Internal worker slot
// ---------------------------------------------------------------------------

interface WorkerSlot {
  worker: Worker
  idle: boolean
}

// ---------------------------------------------------------------------------
// MeshWorkerPool
// ---------------------------------------------------------------------------

export class MeshWorkerPool {
  // -- Public contract -------------------------------------------------------
  readonly ready: Promise<void>
  onResult: ((key: string, arrays: ChunkMeshArrays) => void) | null = null

  // -- Internal state --------------------------------------------------------
  private readonly slots: WorkerSlot[]
  private nextId = 1                           // monotonic job id counter
  private readonly idToKey = new Map<number, string>()    // in-flight id → key
  private readonly keyToId = new Map<string, number>()    // in-flight key → id (for cancel)
  private readonly pending: MeshRequest[] = [] // FIFO queue, pumped priority-desc
  private readonly pendingKeys = new Set<string>()        // fast dedup for submit
  private readonly discardKeys = new Set<string>()        // cancelled in-flight keys
  private readyCount = 0
  private resolveReady!: () => void

  // ---------------------------------------------------------------------------

  constructor(init: Omit<InitMsg, 'type'>, size?: number) {
    const workerCount = size ?? Math.max(1, (navigator.hardwareConcurrency ?? 4) - 1)

    this.ready = new Promise<void>((resolve) => {
      this.resolveReady = resolve
    })

    // Build the init message. Buffers are NOT in a transfer list so they are
    // structured-cloned to each worker (every worker gets its own read-only copy
    // and the main thread retains its arrays for Tectonics/Climate queries).
    const initMsg: InitMsg = { type: 'init', ...init }

    this.slots = []
    for (let i = 0; i < workerCount; i++) {
      const worker = new Worker(
        new URL('./meshWorker.ts', import.meta.url),
        { type: 'module' },
      )
      const slot: WorkerSlot = { worker, idle: false } // idle after 'ready'
      worker.onmessage = (evt: MessageEvent<WorkerOutMsg>) => this.onWorkerMessage(slot, evt.data)
      // Clone to each worker — no transfer list.
      worker.postMessage(initMsg)
      this.slots.push(slot)
    }
  }

  // ---------------------------------------------------------------------------
  // Public getters
  // ---------------------------------------------------------------------------

  get idleCount(): number {
    return this.slots.filter((s) => s.idle).length
  }

  get inFlightCount(): number {
    return this.idToKey.size
  }

  get pendingCount(): number {
    return this.pending.length
  }

  // ---------------------------------------------------------------------------
  // submit / cancel / pump
  // ---------------------------------------------------------------------------

  submit(req: MeshRequest): void {
    // Dedup: if already pending or in-flight, ignore.
    // But if the key was in-flight and previously cancelled (in discardKeys), un-cancel it
    // so the in-flight result will be delivered to onResult when it arrives.
    if (this.pendingKeys.has(req.key) || this.keyToId.has(req.key)) {
      this.discardKeys.delete(req.key)
      return
    }
    this.pending.push(req)
    this.pendingKeys.add(req.key)
    this.pump()
  }

  cancel(key: string): void {
    // Drop from pending queue if not yet dispatched.
    const idx = this.pending.findIndex((r) => r.key === key)
    if (idx !== -1) {
      this.pending.splice(idx, 1)
      this.pendingKeys.delete(key)
      return
    }
    // If in-flight, mark for discard when the result comes back.
    if (this.keyToId.has(key)) {
      this.discardKeys.add(key)
    }
  }

  pump(): void {
    while (this.pending.length > 0) {
      // Find an idle slot.
      const slotIdx = this.slots.findIndex((s) => s.idle)
      if (slotIdx === -1) break

      // Pick highest-priority pending request.
      let bestIdx = 0
      for (let i = 1; i < this.pending.length; i++) {
        if (this.pending[i].priority > this.pending[bestIdx].priority) {
          bestIdx = i
        }
      }
      const req = this.pending[bestIdx]
      this.pending.splice(bestIdx, 1)
      this.pendingKeys.delete(req.key)

      // Assign a unique id and mark the slot busy.
      const id = this.nextId++
      this.idToKey.set(id, req.key)
      this.keyToId.set(req.key, id)
      const slot = this.slots[slotIdx]
      slot.idle = false

      const msg: BuildMsg = {
        type: 'build',
        id,
        faceIndex: req.faceIndex,
        level: req.level,
        ix: req.ix,
        iy: req.iy,
      }
      slot.worker.postMessage(msg)
    }
  }

  // ---------------------------------------------------------------------------
  // Worker message handler
  // ---------------------------------------------------------------------------

  private onWorkerMessage(slot: WorkerSlot, msg: WorkerOutMsg): void {
    if (msg.type === 'ready') {
      this.onReady(slot)
      return
    }
    if (msg.type === 'done') {
      this.onDone(slot, msg)
      return
    }
  }

  private onReady(slot: WorkerSlot): void {
    slot.idle = true
    this.readyCount++
    if (this.readyCount === this.slots.length) {
      this.resolveReady()
    }
  }

  private onDone(slot: WorkerSlot, msg: DoneMsg): void {
    slot.idle = true

    const key = this.idToKey.get(msg.id)
    this.idToKey.delete(msg.id)
    if (key !== undefined) {
      this.keyToId.delete(key)
    }

    // Reconstruct ChunkMeshArrays from the transferred buffers.
    const { vertCount, indexCount } = msg
    const arrays: ChunkMeshArrays = {
      positions:    new Float32Array(msg.positions,  0, vertCount * 3),
      normals:      new Float32Array(msg.normals,     0, vertCount * 3),
      colors:       new Float32Array(msg.colors,      0, vertCount * 3),
      plateColors:  msg.plateColors !== null
        ? new Float32Array(msg.plateColors, 0, vertCount * 3)
        : null,
      climateMoist:   msg.climateMoist !== null
        ? new Float32Array(msg.climateMoist,  0, vertCount)
        : null,
      subsurfaceWet:  msg.subsurfaceWet !== null
        ? new Float32Array(msg.subsurfaceWet, 0, vertCount)
        : null,
      rockHardness:   msg.rockHardness !== null
        ? new Float32Array(msg.rockHardness,  0, vertCount)
        : null,
      indices:        new Uint32Array(msg.indices,      0, indexCount),
      originX:      msg.originX,
      originY:      msg.originY,
      originZ:      msg.originZ,
    }

    // Deliver result unless the key was cancelled or no handler is set.
    if (key !== undefined && !this.discardKeys.has(key) && this.onResult !== null) {
      this.onResult(key, arrays)
    }

    // Clean up discard tracking.
    if (key !== undefined) {
      this.discardKeys.delete(key)
    }

    // A worker just freed up — dispatch any pending work.
    this.pump()
  }

  // ---------------------------------------------------------------------------
  // dispose
  // ---------------------------------------------------------------------------

  dispose(): void {
    for (const slot of this.slots) {
      slot.worker.terminate()
    }
    this.slots.length = 0
    this.idToKey.clear()
    this.keyToId.clear()
    this.pending.length = 0
    this.pendingKeys.clear()
    this.discardKeys.clear()
  }

  // ---------------------------------------------------------------------------
  // Static capability check
  // ---------------------------------------------------------------------------

  static isSupported(): boolean {
    return typeof Worker !== 'undefined'
  }
}
