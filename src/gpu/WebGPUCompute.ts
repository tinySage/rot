import stepWGSL from './step.wgsl?raw';
import { makeSeedData } from '../gl/glUtils';
import type { SimParams } from './SimBackend';

export const DEFAULT_GPU_PARAMS: SimParams = {
  F: 0.037,
  k: 0.06,
  Du: 0.16,
  Dv: 0.08,
  dt: 1.0,
  edgeBoost: 1.6,
  faceReduction: 0.75,
  anisotropic: false,
  anisoStrength: 0.85,
};

const TILE = 8;

/**
 * The Gray-Scott solver itself, ported to a WebGPU compute pipeline (WGSL,
 * workgroup shared memory for the isotropic Laplacian — see step.wgsl).
 * Holds no canvas or render pipeline, so it can be used standalone for the
 * benchmark (§4) as well as wrapped by WebGPUSim for Figure 1's display.
 */
export class WebGPUCompute {
  readonly device: GPUDevice;
  readonly width: number;
  readonly height: number;
  params: SimParams = { ...DEFAULT_GPU_PARAMS };

  private stateBufs: [GPUBuffer, GPUBuffer];
  private guidanceBuf: GPUBuffer;
  private tensorBuf: GPUBuffer;
  private paramsBuf: GPUBuffer;
  private computePipeline: GPUComputePipeline;
  private bindGroups: [GPUBindGroup, GPUBindGroup];
  private current = 0;
  private workgroupsX: number;
  private workgroupsY: number;
  private readbackBuf: GPUBuffer | null = null;

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.workgroupsX = Math.ceil(width / TILE);
    this.workgroupsY = Math.ceil(height / TILE);

    const n = width * height;
    const stateSize = n * 2 * 4; // vec2<f32>
    const guidanceSize = n * 2 * 4; // vec2<f32>
    const tensorSize = n * 4 * 4; // vec4<f32>

    const stateUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.stateBufs = [
      device.createBuffer({ size: stateSize, usage: stateUsage }),
      device.createBuffer({ size: stateSize, usage: stateUsage }),
    ];
    this.guidanceBuf = device.createBuffer({
      size: guidanceSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.tensorBuf = device.createBuffer({
      size: tensorSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.paramsBuf = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Neutral guidance/tensor so Flocal == F and anisotropic == isotropic
    // until real data lands.
    device.queue.writeBuffer(this.guidanceBuf, 0, new Float32Array(n * 2));
    const neutralTensor = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) neutralTensor[i * 4] = 1; // e1 = (1, 0)
    device.queue.writeBuffer(this.tensorBuf, 0, neutralTensor);

    const module = device.createShaderModule({ code: stepWGSL });
    this.computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });

    const layout = this.computePipeline.getBindGroupLayout(0);
    const makeBindGroup = (inBuf: GPUBuffer, outBuf: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: this.guidanceBuf } },
          { binding: 4, resource: { buffer: this.tensorBuf } },
        ],
      });
    this.bindGroups = [
      makeBindGroup(this.stateBufs[0], this.stateBufs[1]),
      makeBindGroup(this.stateBufs[1], this.stateBufs[0]),
    ];

    this.reset();
  }

  /** The buffer currently holding the latest state — for a render pipeline to bind. */
  get currentStateBuffer(): GPUBuffer {
    return this.stateBufs[this.current];
  }

  /** Index into stateBuffers() matching currentStateBuffer — for a render pipeline that
   * keeps one bind group per buffer and needs to pick the right one each frame. */
  get currentIndex(): number {
    return this.current;
  }

  /** Both ping-pong buffers, in fixed order — for building one render bind group per buffer. */
  get stateBuffers(): [GPUBuffer, GPUBuffer] {
    return this.stateBufs;
  }

  reset(density = 0.02, seedData?: Float32Array): void {
    const seed = seedData ?? makeSeedData(this.width, this.height, density);
    const n = this.width * this.height;
    const packed = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      packed[i * 2] = seed[i * 4];
      packed[i * 2 + 1] = seed[i * 4 + 1];
    }
    this.device.queue.writeBuffer(this.stateBufs[0], 0, packed);
    this.device.queue.writeBuffer(this.stateBufs[1], 0, packed);
    this.current = 0;
  }

  setGuidanceData(data: Uint8Array): void {
    const n = this.width * this.height;
    const packed = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      packed[i * 2] = data[i * 4] / 255;
      packed[i * 2 + 1] = data[i * 4 + 1] / 255;
    }
    this.device.queue.writeBuffer(this.guidanceBuf, 0, packed);
  }

  setTensorData(data: Float32Array): void {
    this.device.queue.writeBuffer(this.tensorBuf, 0, data as BufferSource);
  }

  private writeParams(): void {
    const buf = new ArrayBuffer(48);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = this.width;
    u32[1] = this.height;
    f32[2] = this.params.F;
    f32[3] = this.params.k;
    f32[4] = this.params.Du;
    f32[5] = this.params.Dv;
    f32[6] = this.params.dt;
    f32[7] = this.params.edgeBoost;
    f32[8] = this.params.faceReduction;
    u32[9] = this.params.anisotropic ? 1 : 0;
    f32[10] = this.params.anisoStrength;
    this.device.queue.writeBuffer(this.paramsBuf, 0, buf);
  }

  stepN(n: number): void {
    if (n <= 0) return;
    this.writeParams();
    const encoder = this.device.createCommandEncoder();
    for (let i = 0; i < n; i++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.bindGroups[this.current]);
      pass.dispatchWorkgroups(this.workgroupsX, this.workgroupsY, 1);
      pass.end();
      this.current = 1 - this.current;
    }
    this.device.queue.submit([encoder.finish()]);
  }

  step(): void {
    this.stepN(1);
  }

  /** Returns (u, v, 0, 1) per texel as RGBA floats — same layout GrayScottSim.readStateData()
   * uses, so callers (Figure 4, the biomass chart) don't need to know which backend ran. */
  async readStateData(): Promise<Float32Array> {
    const n = this.width * this.height;
    const byteSize = n * 2 * 4;
    if (!this.readbackBuf) {
      this.readbackBuf = this.device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.stateBufs[this.current], 0, this.readbackBuf, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const packed = new Float32Array(this.readbackBuf.getMappedRange().slice(0));
    this.readbackBuf.unmap();

    const rgba = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      rgba[i * 4] = packed[i * 2];
      rgba[i * 4 + 1] = packed[i * 2 + 1];
      rgba[i * 4 + 3] = 1;
    }
    return rgba;
  }

  async readBiomass(): Promise<number> {
    const state = await this.readStateData();
    let sum = 0;
    const n = this.width * this.height;
    for (let i = 0; i < n; i++) sum += state[i * 4 + 1];
    return sum;
  }

  destroy(): void {
    this.stateBufs.forEach((b) => b.destroy());
    this.guidanceBuf.destroy();
    this.tensorBuf.destroy();
    this.paramsBuf.destroy();
    this.readbackBuf?.destroy();
  }
}
