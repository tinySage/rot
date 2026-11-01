/** Shared surface both GrayScottSim (WebGL2) and WebGPUSim implement, so Figure 1's
 * driving code in main.ts doesn't need to know which backend is active. */
export interface SimParams {
  F: number;
  k: number;
  Du: number;
  Dv: number;
  dt: number;
  edgeBoost: number;
  faceReduction: number;
  anisotropic: boolean;
  anisoStrength: number;
}

export interface SimBackend {
  readonly simWidth: number;
  readonly simHeight: number;
  params: SimParams;

  reset(density?: number, seedData?: Float32Array): void;
  setBaseImage(source: TexImageSource): void;
  setGuidanceData(data: Uint8Array): void;
  setTensorData(data: Float32Array): void;
  step(): void;
  stepN(n: number): void;
  renderToCanvas(width: number, height: number): void;
  readStateData(): Promise<Float32Array>;
  readBiomass(): Promise<number>;
}
