import quadVert from '../shaders/quad.vert.glsl?raw';
import stepFrag from '../shaders/step.frag.glsl?raw';
import compositeFrag from '../shaders/composite.frag.glsl?raw';
import { createFloatFbo, createProgram, createQuad, type FloatFbo } from './glUtils';

export interface GrayScottParams {
  F: number;
  k: number;
  Du: number;
  Dv: number;
  dt: number;
}

export const DEFAULT_PARAMS: GrayScottParams = {
  F: 0.037,
  k: 0.06,
  Du: 0.16,
  Dv: 0.08,
  dt: 1.0,
};

export class GrayScottSim {
  readonly gl: WebGL2RenderingContext;
  readonly simWidth: number;
  readonly simHeight: number;
  params: GrayScottParams = { ...DEFAULT_PARAMS };

  private fboA: FloatFbo;
  private fboB: FloatFbo;
  private readFbo: FloatFbo;
  private writeFbo: FloatFbo;

  private baseTexture: WebGLTexture;
  private quad: WebGLVertexArrayObject;
  private stepProgram: WebGLProgram;
  private compositeProgram: WebGLProgram;

  private stepUniforms: Record<string, WebGLUniformLocation | null>;
  private compositeUniforms: Record<string, WebGLUniformLocation | null>;

  constructor(gl: WebGL2RenderingContext, simWidth: number, simHeight: number) {
    this.gl = gl;
    this.simWidth = simWidth;
    this.simHeight = simHeight;

    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float is required for this simulation');
    }

    this.fboA = createFloatFbo(gl, simWidth, simHeight, this.makeSeed());
    this.fboB = createFloatFbo(gl, simWidth, simHeight);
    this.readFbo = this.fboA;
    this.writeFbo = this.fboB;

    this.quad = createQuad(gl);
    this.stepProgram = createProgram(gl, quadVert, stepFrag);
    this.compositeProgram = createProgram(gl, quadVert, compositeFrag);

    this.stepUniforms = {
      uState: gl.getUniformLocation(this.stepProgram, 'uState'),
      uTexel: gl.getUniformLocation(this.stepProgram, 'uTexel'),
      uF: gl.getUniformLocation(this.stepProgram, 'uF'),
      uK: gl.getUniformLocation(this.stepProgram, 'uK'),
      uDu: gl.getUniformLocation(this.stepProgram, 'uDu'),
      uDv: gl.getUniformLocation(this.stepProgram, 'uDv'),
      uDt: gl.getUniformLocation(this.stepProgram, 'uDt'),
    };
    this.compositeUniforms = {
      uBase: gl.getUniformLocation(this.compositeProgram, 'uBase'),
      uState: gl.getUniformLocation(this.compositeProgram, 'uState'),
    };

    this.baseTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([128, 126, 124, 255]),
    );
  }

  private makeSeed(density = 0.02): Float32Array {
    const { simWidth: w, simHeight: h } = this;
    const data = new Float32Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      data[i * 4] = 1.0; // u
      data[i * 4 + 1] = 0.0; // v
      data[i * 4 + 2] = 0.0;
      data[i * 4 + 3] = 1.0;
    }

    const area = w * h;
    const blobCount = Math.max(4, Math.round(area * density / 120));
    for (let b = 0; b < blobCount; b++) {
      const cx = Math.floor(Math.random() * w);
      const cy = Math.floor(Math.random() * h);
      const radius = 2 + Math.floor(Math.random() * 3);
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > radius * radius) continue;
          const x = cx + dx;
          const y = cy + dy;
          if (x < 0 || x >= w || y < 0 || y >= h) continue;
          const idx = (y * w + x) * 4;
          data[idx] = 0.5;
          data[idx + 1] = 0.5 + Math.random() * 0.25;
        }
      }
    }
    return data;
  }

  reset(density = 0.02): void {
    const { gl } = this;
    const seed = this.makeSeed(density);
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

  step(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.writeFbo.framebuffer);
    gl.viewport(0, 0, this.simWidth, this.simHeight);
    gl.useProgram(this.stepProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.readFbo.texture);
    gl.uniform1i(this.stepUniforms.uState, 0);
    gl.uniform2f(this.stepUniforms.uTexel, 1 / this.simWidth, 1 / this.simHeight);
    gl.uniform1f(this.stepUniforms.uF, this.params.F);
    gl.uniform1f(this.stepUniforms.uK, this.params.k);
    gl.uniform1f(this.stepUniforms.uDu, this.params.Du);
    gl.uniform1f(this.stepUniforms.uDv, this.params.Dv);
    gl.uniform1f(this.stepUniforms.uDt, this.params.dt);

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

  renderToCanvas(width: number, height: number): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.compositeProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.uniform1i(this.compositeUniforms.uBase, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.readFbo.texture);
    gl.uniform1i(this.compositeUniforms.uState, 1);

    gl.bindVertexArray(this.quad);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }
}
