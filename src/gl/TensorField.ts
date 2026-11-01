import quadVert from '../shaders/quad.vert.glsl?raw';
import gradientTensorFrag from '../shaders/gradient-tensor.frag.glsl?raw';
import tensorEigenFrag from '../shaders/tensor-eigen.frag.glsl?raw';
import blurFrag from '../shaders/blur.frag.glsl?raw';
import { createFloatFbo, createProgram, createQuad, type FloatFbo } from './glUtils';

/**
 * Builds the structure-tensor field that drives anisotropic diffusion
 * (src/shaders/step.frag.glsl, §5). Runs once per uploaded image, not per
 * frame, in the same WebGL2 context as the simulation that consumes it.
 */
export class TensorField {
  readonly gl: WebGL2RenderingContext;
  readonly width: number;
  readonly height: number;

  private quad: WebGLVertexArrayObject;
  private gradientProgram: WebGLProgram;
  private blurProgram: WebGLProgram;
  private eigenProgram: WebGLProgram;

  private gradientUniforms: Record<string, WebGLUniformLocation | null>;
  private blurUniforms: Record<string, WebGLUniformLocation | null>;
  private eigenUniforms: Record<string, WebGLUniformLocation | null>;

  private baseTex: WebGLTexture;
  private rawFbo: FloatFbo;
  private blurTmpFbo: FloatFbo;
  private blurFinalFbo: FloatFbo;
  private eigenFbo: FloatFbo;

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl = gl;
    this.width = width;
    this.height = height;

    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float is required for the structure tensor');
    }

    this.quad = createQuad(gl);
    this.gradientProgram = createProgram(gl, quadVert, gradientTensorFrag);
    this.blurProgram = createProgram(gl, quadVert, blurFrag);
    this.eigenProgram = createProgram(gl, quadVert, tensorEigenFrag);

    this.gradientUniforms = {
      uImage: gl.getUniformLocation(this.gradientProgram, 'uImage'),
      uTexel: gl.getUniformLocation(this.gradientProgram, 'uTexel'),
    };
    this.blurUniforms = {
      uImage: gl.getUniformLocation(this.blurProgram, 'uImage'),
      uTexel: gl.getUniformLocation(this.blurProgram, 'uTexel'),
      uDirection: gl.getUniformLocation(this.blurProgram, 'uDirection'),
    };
    this.eigenUniforms = {
      uTensor: gl.getUniformLocation(this.eigenProgram, 'uTensor'),
    };

    this.baseTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

    this.rawFbo = createFloatFbo(gl, width, height);
    this.blurTmpFbo = createFloatFbo(gl, width, height);
    this.blurFinalFbo = createFloatFbo(gl, width, height);
    // e1 = (1, 0), coherence = 0: a neutral field equivalent to isotropic.
    this.eigenFbo = createFloatFbo(gl, width, height, neutralField(width, height));
  }

  get texture(): WebGLTexture {
    return this.eigenFbo.texture;
  }

  /** Reads back the final (e1, coherence) field as RGBA floats, for sharing with another context. */
  readPixelData(): Float32Array {
    const { gl } = this;
    const buffer = new Float32Array(this.width * this.height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.eigenFbo.framebuffer);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, buffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return buffer;
  }

  /** Recomputes the structure tensor from `source`, synchronously (pure GPU, no async detection step). */
  update(source: HTMLImageElement | HTMLCanvasElement, sourceWidth: number, sourceHeight: number): void {
    const { gl } = this;

    gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    gl.bindVertexArray(this.quad);
    gl.viewport(0, 0, this.width, this.height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rawFbo.framebuffer);
    gl.useProgram(this.gradientProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
    gl.uniform1i(this.gradientUniforms.uImage, 0);
    gl.uniform2f(this.gradientUniforms.uTexel, 1 / sourceWidth, 1 / sourceHeight);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.useProgram(this.blurProgram);
    gl.uniform2f(this.blurUniforms.uTexel, 1 / this.width, 1 / this.height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurTmpFbo.framebuffer);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.rawFbo.texture);
    gl.uniform1i(this.blurUniforms.uImage, 0);
    gl.uniform2f(this.blurUniforms.uDirection, 1, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFinalFbo.framebuffer);
    gl.bindTexture(gl.TEXTURE_2D, this.blurTmpFbo.texture);
    gl.uniform1i(this.blurUniforms.uImage, 0);
    gl.uniform2f(this.blurUniforms.uDirection, 0, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.eigenFbo.framebuffer);
    gl.useProgram(this.eigenProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.blurFinalFbo.texture);
    gl.uniform1i(this.eigenUniforms.uTensor, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}

function neutralField(width: number, height: number): Float32Array {
  const data = new Float32Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = 1; // e1.x
    data[i * 4 + 1] = 0; // e1.y
    data[i * 4 + 2] = 0; // coherence
    data[i * 4 + 3] = 1;
  }
  return data;
}
