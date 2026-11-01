// Procedurally generated "neutral specimen" card: a plain grey ground with
// soft value noise and a gentle vignette. Stands in for a photograph until
// the viewer uploads one. No stock imagery.

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function valueNoiseField(width: number, height: number, rand: () => number): Float32Array {
  const field = new Float32Array(width * height);
  for (let i = 0; i < field.length; i++) field[i] = rand();

  // Box-blur a few times to turn white noise into soft value noise.
  const blurred = new Float32Array(width * height);
  const passes = 6;
  let src = field;
  let dst = blurred;
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            sum += src[ny * width + nx];
            count++;
          }
        }
        dst[y * width + x] = sum / count;
      }
    }
    const tmp = src;
    src = dst;
    dst = tmp;
  }
  return src;
}

export function generateSpecimen(width: number, height: number, seed = 1337): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D context unavailable');

  const rand = mulberry32(seed);
  const noise = valueNoiseField(width, height, rand);

  const image = ctx.createImageData(width, height);
  const baseGrey = 0.52;
  const cx = width / 2;
  const cy = height / 2;
  const maxDist = Math.hypot(cx, cy);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const n = noise[i] - 0.5; // roughly [-0.5, 0.5]
      const dist = Math.hypot(x - cx, y - cy) / maxDist;
      const vignette = 1.0 - 0.14 * Math.pow(dist, 1.6);
      const tone = (baseGrey + n * 0.08) * vignette;
      const g = Math.max(0, Math.min(1, tone));
      const p = i * 4;
      image.data[p] = Math.round(g * 255);
      image.data[p + 1] = Math.round(g * 255 * 0.99);
      image.data[p + 2] = Math.round(g * 255 * 0.97);
      image.data[p + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
  return canvas;
}
