import type * as THREE from 'three'

/** The active navigation / interaction mode. Owned by main.ts. */
export enum NavMode {
  Globe = 'globe',
  Placement = 'placement',
  FirstPerson = 'firstperson',
}

/**
 * A minimal surface-height query that the first-person controller and the
 * placement picker depend on, so neither needs to import Planet directly.
 * Implemented in main.ts by delegating to planet.getSurfaceRadiusAt.
 */
export interface SurfaceSampler {
  /**
   * World-space surface radius (units from the planet origin) along a
   * world-space direction. The direction need not be normalized.
   */
  surfaceRadiusAt(worldDir: THREE.Vector3): number
  /** Nominal planet sphere radius (no terrain displacement). */
  readonly radius: number
}

/** The resolved result of a surface placement pick. */
export interface SurfacePlacement {
  /** World-space point exactly on the terrain surface. */
  position: THREE.Vector3
  /** World-space local up (radial, normalized). */
  up: THREE.Vector3
}
