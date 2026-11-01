import quadVert from '../shaders/quad.vert.glsl?raw';
import sobelFrag from '../shaders/sobel.frag.glsl?raw';
import combineFrag from '../shaders/guidance-combine.frag.glsl?raw';
import blurFrag from '../shaders/blur.frag.glsl?raw';
import { createProgram, createQuad, createRgba8Fbo, type Rgba8Fbo } from './glUtils';
import { detectFaces, type FaceBox } from '../guidance/faceDetector';

/**
 * Builds the (edge, face) guidance texture used both to steer the solver
 * (src/shaders/step.frag.glsl) and to draw Figure 3's three panels. Runs
 * once per uploaded image, not per frame.
 */
export class Guidance {
  readonly gl: WebGL2RenderingContext;
  readonly width: number;
  readonly height: number;

  private quad: WebGLVertexArrayObject;
  private sobelProgram: WebGLProgram;
  private combineProgram: WebGLProgram;
  private blurProgram: WebGLProgram;

  private sobelUniforms: Record<string, WebGLUniformLocation | null>;
  private combineUniforms: Record<string, WebGLUniformLocation | null>;
  private blurUniforms: Record<string, WebGLUniformLocation | null>;

  private baseTex: WebGLTexture;
  private faceTex: WebGLTexture;
  private edgeFbo: Rgba8Fbo;
  private combinedFbo: Rgba8Fbo;
  private blurTmpFbo: Rgba8Fbo;
  private blurFinalFbo: Rgba8Fbo;

  private data: Uint8Array;

  constructor(gl: WebGL2RenderingContext, width: number, height: number) {
    this.gl = gl;
    this.width = width;
    this.height = height;

    this.quad = createQuad(gl);
    this.sobelProgram = createProgram(gl, quadVert, sobelFrag);
    this.combineProgram = createProgram(gl, quadVert, combineFrag);
    this.blurProgram = createProgram(gl, quadVert, blurFrag);

    this.sobelUniforms = {
      uImage: gl.getUniformLocation(this.sobelProgram, 'uImage'),
      uTexel: gl.getUniformLocation(this.sobelProgram, 'uTexel'),
    };
    this.combineUniforms = {
      uEdge: gl.getUniformLocation(this.combineProgram, 'uEdge'),
      uFace: gl.getUniformLocation(this.combineProgram, 'uFace'),
    };
    this.blurUniforms = {
      uImage: gl.getUniformLocation(this.blurProgram, 'uImage'),
      uTexel: gl.getUniformLocation(this.blurProgram, 'uTexel'),
      uDirection: gl.getUniformLocation(this.blurProgram, 'uDirection'),
    };

    this.baseTex = createTexture(gl);
    this.faceTex = createTexture(gl);
    this.edgeFbo = createRgba8Fbo(gl, width, height);
    this.combinedFbo = createRgba8Fbo(gl, width, height);
    this.blurTmpFbo = createRgba8Fbo(gl, width, height);
    this.blurFinalFbo = createRgba8Fbo(gl, width, height);

    this.data = new Uint8Array(width * height * 4);
  }

  get texture(): WebGLTexture {
    return this.blurFinalFbo.texture;
  }

  /** R = edge strength, G = face mask, both in [0, 255], blurred. */
  get pixelData(): Uint8Array {
    return this.data;
  }

  async update(
    source: HTMLImageElement | HTMLCanvasElement,
    sourceWidth: number,
    sourceHeight: number,
  ): Promise<void> {
    const { gl } = this;

    gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    this.runSobel(sourceWidth, sourceHeight);

    const faces = await detectFaces(source, sourceWidth, sourceHeight);
    const faceCanvas = drawFaceMask(faces, this.width, this.height);
    gl.bindTexture(gl.TEXTURE_2D, this.faceTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, faceCanvas);

    this.runCombine();
    this.runBlur();
    this.readback();
  }

  private runSobel(sourceWidth: number, sourceHeight: number): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.edgeFbo.framebuffer);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.sobelProgram);
    gl.bindVertexArray(this.quad);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
    gl.uniform1i(this.sobelUniforms.uImage, 0);
    gl.uniform2f(this.sobelUniforms.uTexel, 1 / sourceWidth, 1 / sourceHeight);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  private runCombine(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.combinedFbo.framebuffer);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.combineProgram);
    gl.bindVertexArray(this.quad);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.edgeFbo.texture);
    gl.uniform1i(this.combineUniforms.uEdge, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.faceTex);
    gl.uniform1i(this.combineUniforms.uFace, 1);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  private runBlur(): void {
    const { gl } = this;
    gl.useProgram(this.blurProgram);
    gl.bindVertexArray(this.quad);
    gl.viewport(0, 0, this.width, this.height);
    gl.uniform2f(this.blurUniforms.uTexel, 1 / this.width, 1 / this.height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurTmpFbo.framebuffer);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.combinedFbo.texture);
    gl.uniform1i(this.blurUniforms.uImage, 0);
    gl.uniform2f(this.blurUniforms.uDirection, 1, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFinalFbo.framebuffer);
    gl.bindTexture(gl.TEXTURE_2D, this.blurTmpFbo.texture);
    gl.uniform1i(this.blurUniforms.uImage, 0);
    gl.uniform2f(this.blurUniforms.uDirection, 0, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindVertexArray(null);
  }

  private readback(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFinalFbo.framebuffer);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, this.data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}

function createTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const texture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  return texture;
}

function drawFaceMask(faces: FaceBox[], width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);
  ctx.globalCompositeOperation = 'lighter';

  for (const box of faces) {
    const cx = (box.x + box.width / 2) * width;
    const cy = (box.y + box.height / 2) * height;
    const rx = Math.max(box.width * width, 4) * 0.72;
    const ry = Math.max(box.height * height, 4) * 0.85;
    const r = Math.max(rx, ry);
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  return canvas;
}
