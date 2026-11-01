import quadVert from '../shaders/quad.vert.glsl?raw';
import stepFrag from '../shaders/step.frag.glsl?raw';
import compositeFrag from '../shaders/composite.frag.glsl?raw';
import compositeLitFrag from '../shaders/composite-lit.frag.glsl?raw';
import {
  createFloatFbo,
  createProgram,
  createQuad,
  createRgba8TextureFromData,
  createRgba16fTextureFromData,
  makeSeedData,
  type FloatFbo,
} from './glUtils';
import type { SimBackend, SimParams } from '../gpu/SimBackend';

export type GrayScottParams = SimParams;

export const DEFAULT_PARAMS: GrayScottParams = {
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

const BIOMASS_SAMPLE_WIDTH = 40;
const BIOMASS_SAMPLE_HEIGHT = 18;

export class GrayScottSim implements SimBackend {
  readonly gl: WebGL2RenderingContext;
  readonly simWidth: number;
  readonly simHeight: number;
  params: GrayScottParams = { ...DEFAULT_PARAMS };

  private fboA: FloatFbo;
  private fboB: FloatFbo;
  private readFbo: FloatFbo;
  private writeFbo: FloatFbo;
  private biomassFbo: FloatFbo;
  private biomassBuffer: Float32Array;

  private baseTexture: WebGLTexture;
  private guidanceTexture: WebGLTexture;
  private tensorTexture: WebGLTexture;
  private quad: WebGLVertexArrayObject;
  private stepProgram: WebGLProgram;
  private compositeFlatProgram: WebGLProgram;
  private compositeLitProgram: WebGLProgram;

  private stepUniforms: Record<string, WebGLUniformLocation | null>;
  private compositeFlatUniforms: Record<string, WebGLUniformLocation | null>;
  private compositeLitUniforms: Record<string, WebGLUniformLocation | null>;

  constructor(gl: WebGL2RenderingContext, simWidth: number, simHeight: number) {
    this.gl = gl;
    this.simWidth = simWidth;
    this.simHeight = simHeight;

    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float is required for this simulation');
    }

    this.fboA = createFloatFbo(gl, simWidth, simHeight, makeSeedData(simWidth, simHeight));
    this.fboB = createFloatFbo(gl, simWidth, simHeight);
    this.readFbo = this.fboA;
    this.writeFbo = this.fboB;

    this.biomassFbo = createFloatFbo(gl, BIOMASS_SAMPLE_WIDTH, BIOMASS_SAMPLE_HEIGHT);
    this.biomassBuffer = new Float32Array(BIOMASS_SAMPLE_WIDTH * BIOMASS_SAMPLE_HEIGHT * 4);

    this.quad = createQuad(gl);
    this.stepProgram = createProgram(gl, quadVert, stepFrag);
    this.compositeFlatProgram = createProgram(gl, quadVert, compositeFrag);
    this.compositeLitProgram = createProgram(gl, quadVert, compositeLitFrag);

    this.stepUniforms = {
      uState: gl.getUniformLocation(this.stepProgram, 'uState'),
      uTexel: gl.getUniformLocation(this.stepProgram, 'uTexel'),
      uF: gl.getUniformLocation(this.stepProgram, 'uF'),
      uK: gl.getUniformLocation(this.stepProgram, 'uK'),
      uDu: gl.getUniformLocation(this.stepProgram, 'uDu'),
      uDv: gl.getUniformLocation(this.stepProgram, 'uDv'),
      uDt: gl.getUniformLocation(this.stepProgram, 'uDt'),
      uGuidance: gl.getUniformLocation(this.stepProgram, 'uGuidance'),
      uEdgeBoost: gl.getUniformLocation(this.stepProgram, 'uEdgeBoost'),
      uFaceReduction: gl.getUniformLocation(this.stepProgram, 'uFaceReduction'),
      uTensor: gl.getUniformLocation(this.stepProgram, 'uTensor'),
      uAnisotropic: gl.getUniformLocation(this.stepProgram, 'uAnisotropic'),
      uAnisoStrength: gl.getUniformLocation(this.stepProgram, 'uAnisoStrength'),
    };
    this.compositeFlatUniforms = {
      uBase: gl.getUniformLocation(this.compositeFlatProgram, 'uBase'),
      uState: gl.getUniformLocation(this.compositeFlatProgram, 'uState'),
    };
    this.compositeLitUniforms = {
      uBase: gl.getUniformLocation(this.compositeLitProgram, 'uBase'),
      uState: gl.getUniformLocation(this.compositeLitProgram, 'uState'),
      uTexel: gl.getUniformLocation(this.compositeLitProgram, 'uTexel'),
    };

    this.baseTexture = createRgba8Texture(gl, [128, 126, 124, 255]);
    // Neutral guidance (no edge, no face) so Flocal == F until real guidance lands.
    this.guidanceTexture = createRgba8Texture(gl, [0, 0, 0, 255]);
    // Neutral tensor (e1 = (1,0), coherence = 0): anisotropic mode behaves
    // like isotropic mode until a real structure tensor lands.
    this.tensorTexture = createRgba8Texture(gl, [255, 0, 0, 255]);
  }

  reset(density = 0.02, seedData?: Float32Array): void {
    const { gl } = this;
    const seed = seedData ?? makeSeedData(this.simWidth, this.simHeight, density);
    for (const fbo of [this.fboA, this.fboB]) {
      gl.bindTexture(gl.TEXTURE_2D, fbo.texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA16F,
        this.simWidth,
        this.simHeight,
        0,
        gl.RGBA,
        gl.FLOAT,
        fbo === this.fboA ? seed : null,
      );
    }
    this.readFbo = this.fboA;
    this.writeFbo = this.fboB;
  }

  setBaseImage(source: TexImageSource): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  /** Swaps in a precomputed guidance texture (R = edge strength, G = face mask). */
  setGuidanceTexture(texture: WebGLTexture): void {
    this.guidanceTexture = texture;
  }

  /** Swaps in a precomputed structure-tensor texture (RG = e1, B = coherence). */
  setTensorTexture(texture: WebGLTexture): void {
    this.tensorTexture = texture;
  }

  /** SimBackend-uniform entry point: builds a guidance texture in this context from raw
   * RGBA8 pixel data (e.g. read back from another context's Guidance instance). */
  setGuidanceData(data: Uint8Array): void {
    this.setGuidanceTexture(createRgba8TextureFromData(this.gl, this.simWidth, this.simHeight, data));
  }

  /** SimBackend-uniform entry point: builds a tensor texture in this context from raw
   * RGBA float data (e.g. read back from another context's TensorField instance). */
  setTensorData(data: Float32Array): void {
    this.setTensorTexture(createRgba16fTextureFromData(this.gl, this.simWidth, this.simHeight, data));
  }

  step(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.writeFbo.framebuffer);
    gl.viewport(0, 0, this.simWidth, this.simHeight);
    gl.useProgram(this.stepProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.readFbo.texture);
    gl.uniform1i(this.stepUniforms.uState, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.guidanceTexture);
    gl.uniform1i(this.stepUniforms.uGuidance, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tensorTexture);
    gl.uniform1i(this.stepUniforms.uTensor, 2);

    gl.uniform2f(this.stepUniforms.uTexel, 1 / this.simWidth, 1 / this.simHeight);
    gl.uniform1f(this.stepUniforms.uF, this.params.F);
    gl.uniform1f(this.stepUniforms.uK, this.params.k);
    gl.uniform1f(this.stepUniforms.uDu, this.params.Du);
    gl.uniform1f(this.stepUniforms.uDv, this.params.Dv);
    gl.uniform1f(this.stepUniforms.uDt, this.params.dt);
    gl.uniform1f(this.stepUniforms.uEdgeBoost, this.params.edgeBoost);
    gl.uniform1f(this.stepUniforms.uFaceReduction, this.params.faceReduction);
    gl.uniform1f(this.stepUniforms.uAnisotropic, this.params.anisotropic ? 1 : 0);
    gl.uniform1f(this.stepUniforms.uAnisoStrength, this.params.anisoStrength);

    gl.bindVertexArray(this.quad);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);

    const tmp = this.readFbo;
    this.readFbo = this.writeFbo;
    this.writeFbo = tmp;
  }

  stepN(n: number): void {
    for (let i = 0; i < n; i++) this.step();
  }

  private renderComposite(
    program: WebGLProgram,
    uniforms: Record<string, WebGLUniformLocation | null>,
    x: number,
    y: number,
    width: number,
    height: number,
  ): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(x, y, width, height);
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.uniform1i(uniforms.uBase, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.readFbo.texture);
    gl.uniform1i(uniforms.uState, 1);

    if (uniforms.uTexel) {
      gl.uniform2f(uniforms.uTexel, 1 / this.simWidth, 1 / this.simHeight);
    }

    gl.bindVertexArray(this.quad);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  /** Flat dark-green-to-black overlay (the Phase 1 look). */
  renderFlat(width: number, height: number, x = 0, y = 0): void {
    this.renderComposite(this.compositeFlatProgram, this.compositeFlatUniforms, x, y, width, height);
  }

  /** Height-mapped Blinn-Phong render with the full mould palette. */
  renderLit(width: number, height: number, x = 0, y = 0): void {
    this.renderComposite(this.compositeLitProgram, this.compositeLitUniforms, x, y, width, height);
  }

  /** Main display render — the lit look is the intended final rendering. */
  renderToCanvas(width: number, height: number): void {
    this.renderLit(width, height);
  }

  /** Reads back the current (u, v) state as RGBA floats, for sharing with another context.
   * Async to match WebGPUSim's (necessarily async) readback, even though WebGL2's
   * readPixels is itself synchronous. */
  async readStateData(): Promise<Float32Array> {
    const { gl } = this;
    const buffer = new Float32Array(this.simWidth * this.simHeight * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.readFbo.framebuffer);
    gl.readPixels(0, 0, this.simWidth, this.simHeight, gl.RGBA, gl.FLOAT, buffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return buffer;
  }

  /**
   * Estimates total biomass (sum of V) for Figure 5. A GPU blit downsamples
   * the full-resolution state into a tiny buffer first, so the periodic
   * readPixels this needs stays cheap even sampled every few frames.
   */
  async readBiomass(): Promise<number> {
    const { gl } = this;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.readFbo.framebuffer);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.biomassFbo.framebuffer);
    gl.blitFramebuffer(
      0,
      0,
      this.simWidth,
      this.simHeight,
      0,
      0,
      this.biomassFbo.width,
      this.biomassFbo.height,
      gl.COLOR_BUFFER_BIT,
      gl.LINEAR,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.biomassFbo.framebuffer);
    gl.readPixels(
      0,
      0,
      this.biomassFbo.width,
      this.biomassFbo.height,
      gl.RGBA,
      gl.FLOAT,
      this.biomassBuffer,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const n = this.biomassFbo.width * this.biomassFbo.height;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += this.biomassBuffer[i * 4 + 1];
    const avgV = sum / n;
    return avgV * this.simWidth * this.simHeight;
  }

  /** Frees GPU resources — used by the benchmark (§4), which creates one throwaway
   * instance per resolution and would otherwise leak framebuffers at 2048x2048. */
  destroy(): void {
    const { gl } = this;
    for (const fbo of [this.fboA, this.fboB, this.biomassFbo]) {
      gl.deleteFramebuffer(fbo.framebuffer);
      gl.deleteTexture(fbo.texture);
    }
    gl.deleteTexture(this.baseTexture);
    gl.deleteTexture(this.guidanceTexture);
    gl.deleteTexture(this.tensorTexture);
    gl.deleteProgram(this.stepProgram);
    gl.deleteProgram(this.compositeFlatProgram);
    gl.deleteProgram(this.compositeLitProgram);
    gl.deleteVertexArray(this.quad);
  }
}

function createRgba8Texture(gl: WebGL2RenderingContext, pixel: [number, number, number, number]): WebGLTexture {
  const texture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(pixel));
  return texture;
}
