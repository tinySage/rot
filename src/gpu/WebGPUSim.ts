import compositeWGSL from './composite.wgsl?raw';
import { WebGPUCompute } from './WebGPUCompute';
import type { SimBackend, SimParams } from './SimBackend';

/**
 * Figure 1's WebGPU backend: wraps WebGPUCompute (the solver) with a render
 * pipeline that draws the same height-mapped Blinn-Phong composite as
 * src/shaders/composite-lit.frag.glsl, straight to the canvas's own
 * GPUCanvasContext.
 */
export class WebGPUSim implements SimBackend {
  readonly simWidth: number;
  readonly simHeight: number;

  private compute: WebGPUCompute;
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;

  private baseTexture: GPUTexture;
  private sampler: GPUSampler;
  private compositeUniformBuf: GPUBuffer;
  private renderPipeline: GPURenderPipeline;
  private renderBindGroups: [GPUBindGroup, GPUBindGroup];

  constructor(device: GPUDevice, canvas: HTMLCanvasElement, width: number, height: number) {
    this.device = device;
    this.simWidth = width;
    this.simHeight = height;
    this.compute = new WebGPUCompute(device, width, height);

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Failed to acquire a WebGPU canvas context');
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.compositeUniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.compositeUniformBuf, 0, new Uint32Array([width, height, 0, 0]));

    const module = device.createShaderModule({ code: compositeWGSL });
    this.renderPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.baseTexture = device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.baseTexture },
      new Uint8Array([128, 126, 124, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );

    this.renderBindGroups = this.buildRenderBindGroups();
  }

  get params(): SimParams {
    return this.compute.params;
  }
  set params(value: SimParams) {
    this.compute.params = value;
  }

  private buildRenderBindGroups(): [GPUBindGroup, GPUBindGroup] {
    const layout = this.renderPipeline.getBindGroupLayout(0);
    const makeBindGroup = (stateBuf: GPUBuffer): GPUBindGroup =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.compositeUniformBuf } },
          { binding: 1, resource: { buffer: stateBuf } },
          { binding: 2, resource: this.baseTexture.createView() },
          { binding: 3, resource: this.sampler },
        ],
      });
    // Mirrors WebGPUCompute's own two ping-pong buffers, in the same order.
    const [a, b] = this.compute.stateBuffers;
    return [makeBindGroup(a), makeBindGroup(b)];
  }

  reset(density = 0.02, seedData?: Float32Array): void {
    this.compute.reset(density, seedData);
  }

  setBaseImage(source: TexImageSource): void {
    const img = source as HTMLImageElement | HTMLCanvasElement;
    const width = 'naturalWidth' in img ? img.naturalWidth || img.width : img.width;
    const height = 'naturalHeight' in img ? img.naturalHeight || img.height : img.height;
    if (!width || !height) return;

    this.baseTexture.destroy();
    this.baseTexture = this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: img as HTMLImageElement | HTMLCanvasElement },
      { texture: this.baseTexture },
      [width, height],
    );
    this.renderBindGroups = this.buildRenderBindGroups();
  }

  setGuidanceData(data: Uint8Array): void {
    this.compute.setGuidanceData(data);
  }

  setTensorData(data: Float32Array): void {
    this.compute.setTensorData(data);
  }

  step(): void {
    this.compute.stepN(1);
  }

  stepN(n: number): void {
    this.compute.stepN(n);
  }

  renderToCanvas(_width: number, _height: number): void {
    const encoder = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pass.setPipeline(this.renderPipeline);
    // Index matches WebGPUCompute's internal ping-pong parity: after stepN(n),
    // the "current" (latest) buffer is bindGroup index compute.currentIndex.
    pass.setBindGroup(0, this.renderBindGroups[this.compute.currentIndex]);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async readStateData(): Promise<Float32Array> {
    return this.compute.readStateData();
  }

  async readBiomass(): Promise<number> {
    return this.compute.readBiomass();
  }
}
