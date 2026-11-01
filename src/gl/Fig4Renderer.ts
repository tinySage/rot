import quadVert from '../shaders/quad.vert.glsl?raw';
import compositeFrag from '../shaders/composite.frag.glsl?raw';
import compositeLitFrag from '../shaders/composite-lit.frag.glsl?raw';
import { createProgram, createQuad } from './glUtils';

/**
 * Figure 4 needs the exact same simulation state as Figure 1, rendered two
 * ways, but WebGL2 contexts can't share objects across canvases. Rather than
 * re-running the (expensive) reaction-diffusion step a second time, the main
 * sim's cheap (u, v) state is copied into this context's own texture each
 * frame, and only the composite shaders re-run here.
 */
export class Fig4Renderer {
  readonly gl: WebGL2RenderingContext;
  readonly simWidth: number;
  readonly simHeight: number;

  private quad: WebGLVertexArrayObject;
  private flatProgram: WebGLProgram;
  private litProgram: WebGLProgram;
  private flatUniforms: Record<string, WebGLUniformLocation | null>;
  private litUniforms: Record<string, WebGLUniformLocation | null>;

  private baseTexture: WebGLTexture;
  private stateTexture: WebGLTexture;

  constructor(gl: WebGL2RenderingContext, simWidth: number, simHeight: number) {
    this.gl = gl;
    this.simWidth = simWidth;
    this.simHeight = simHeight;

    this.quad = createQuad(gl);
    this.flatProgram = createProgram(gl, quadVert, compositeFrag);
    this.litProgram = createProgram(gl, quadVert, compositeLitFrag);

    this.flatUniforms = {
      uBase: gl.getUniformLocation(this.flatProgram, 'uBase'),
      uState: gl.getUniformLocation(this.flatProgram, 'uState'),
    };
    this.litUniforms = {
      uBase: gl.getUniformLocation(this.litProgram, 'uBase'),
      uState: gl.getUniformLocation(this.litProgram, 'uState'),
      uTexel: gl.getUniformLocation(this.litProgram, 'uTexel'),
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

    this.stateTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.stateTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, simWidth, simHeight, 0, gl.RGBA, gl.FLOAT, null);
  }

  setBaseImage(source: TexImageSource): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  updateState(data: Float32Array): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.stateTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA16F,
      this.simWidth,
      this.simHeight,
      0,
      gl.RGBA,
      gl.FLOAT,
      data,
    );
  }

  /** Flat on the left half of the canvas, lit on the right. */
  render(canvasWidth: number, canvasHeight: number): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(this.quad);

    const halfWidth = Math.floor(canvasWidth / 2);

    gl.viewport(0, 0, halfWidth, canvasHeight);
    gl.useProgram(this.flatProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.uniform1i(this.flatUniforms.uBase, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTexture);
    gl.uniform1i(this.flatUniforms.uState, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.viewport(halfWidth, 0, canvasWidth - halfWidth, canvasHeight);
    gl.useProgram(this.litProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.uniform1i(this.litUniforms.uBase, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTexture);
    gl.uniform1i(this.litUniforms.uState, 1);
    gl.uniform2f(this.litUniforms.uTexel, 1 / this.simWidth, 1 / this.simHeight);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindVertexArray(null);
  }
}
