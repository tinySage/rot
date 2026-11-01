import quadVert from '../shaders/quad.vert.glsl?raw';
import stepFrag from '../shaders/step.frag.glsl?raw';
import tileFrag from '../shaders/tile.frag.glsl?raw';
import { createFloatFbo, createProgram, createQuad, makeSeedData, type FloatFbo } from './glUtils';
import { F_RANGE, K_RANGE, type Range } from '../state';

const DU = 0.16;
const DV = 0.08;
const DT = 1.0;

interface Tile {
  col: number;
  row: number;
  F: number;
  k: number;
  read: FloatFbo;
  write: FloatFbo;
}

export class PhaseMap {
  readonly gl: WebGL2RenderingContext;
  readonly cols: number;
  readonly rows: number;
  readonly tileRes: number;

  private tiles: Tile[] = [];
  private quad: WebGLVertexArrayObject;
  private stepProgram: WebGLProgram;
  private tileProgram: WebGLProgram;
  private stepUniforms: Record<string, WebGLUniformLocation | null>;
  private tileUniforms: Record<string, WebGLUniformLocation | null>;
  private neutralGuidance: WebGLTexture;

  constructor(
    gl: WebGL2RenderingContext,
    cols = 8,
    rows = 8,
    tileRes = 48,
    fRange: Range = F_RANGE,
    kRange: Range = K_RANGE,
  ) {
    this.gl = gl;
    this.cols = cols;
    this.rows = rows;
    this.tileRes = tileRes;

    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float is required for the phase map');
    }

    this.quad = createQuad(gl);
    this.stepProgram = createProgram(gl, quadVert, stepFrag);
    this.tileProgram = createProgram(gl, quadVert, tileFrag);

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
    };
    this.tileUniforms = {
      uState: gl.getUniformLocation(this.tileProgram, 'uState'),
    };

    // Tiles are pure (F, k) space, unaffected by image guidance: bind a
    // neutral 1x1 texture so the step shader's Flocal == F everywhere.
    this.neutralGuidance = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.neutralGuidance);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const F = fRange.min + ((col + 0.5) / cols) * (fRange.max - fRange.min);
        // Row 0 is the top of the grid; k increases upward, matching the
        // conventional Gray-Scott phase diagram orientation.
        const k = kRange.max - ((row + 0.5) / rows) * (kRange.max - kRange.min);
        const seed = makeSeedData(tileRes, tileRes, 0.05);
        const fboA = createFloatFbo(gl, tileRes, tileRes, seed);
        const fboB = createFloatFbo(gl, tileRes, tileRes);
        this.tiles.push({ col, row, F, k, read: fboA, write: fboB });
      }
    }
  }

  /** Maps a point in parameter space to its tile's (col, row) indices. */
  cellForParams(F: number, k: number, fRange: Range = F_RANGE, kRange: Range = K_RANGE) {
    const col = Math.min(
      this.cols - 1,
      Math.max(0, Math.floor(((F - fRange.min) / (fRange.max - fRange.min)) * this.cols)),
    );
    const row = Math.min(
      this.rows - 1,
      Math.max(
        0,
        Math.floor((1 - (k - kRange.min) / (kRange.max - kRange.min)) * this.rows),
      ),
    );
    return { col, row };
  }

  step(stepsPerTile: number): void {
    const { gl } = this;
    gl.useProgram(this.stepProgram);
    gl.bindVertexArray(this.quad);
    gl.viewport(0, 0, this.tileRes, this.tileRes);
    gl.uniform2f(this.stepUniforms.uTexel, 1 / this.tileRes, 1 / this.tileRes);
    gl.uniform1f(this.stepUniforms.uDu, DU);
    gl.uniform1f(this.stepUniforms.uDv, DV);
    gl.uniform1f(this.stepUniforms.uDt, DT);
    gl.uniform1f(this.stepUniforms.uEdgeBoost, 0);
    gl.uniform1f(this.stepUniforms.uFaceReduction, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.neutralGuidance);
    gl.uniform1i(this.stepUniforms.uGuidance, 1);

    for (const tile of this.tiles) {
      gl.uniform1f(this.stepUniforms.uF, tile.F);
      gl.uniform1f(this.stepUniforms.uK, tile.k);
      for (let i = 0; i < stepsPerTile; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, tile.write.framebuffer);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tile.read.texture);
        gl.uniform1i(this.stepUniforms.uState, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        const tmp = tile.read;
        tile.read = tile.write;
        tile.write = tmp;
      }
    }
    gl.bindVertexArray(null);
  }

  render(canvasWidth: number, canvasHeight: number): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.tileProgram);
    gl.bindVertexArray(this.quad);

    const tileW = Math.floor(canvasWidth / this.cols);
    const tileH = Math.floor(canvasHeight / this.rows);

    for (const tile of this.tiles) {
      const x = tile.col * tileW;
      const y = canvasHeight - (tile.row + 1) * tileH; // GL origin is bottom-left
      gl.viewport(x, y, tileW, tileH);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tile.read.texture);
      gl.uniform1i(this.tileUniforms.uState, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindVertexArray(null);
  }
}
