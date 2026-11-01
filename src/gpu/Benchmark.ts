import { GrayScottSim } from '../gl/GrayScottSim';
import { WebGPUCompute } from './WebGPUCompute';
import { requestWebGPUDevice } from './detectWebGPU';

export interface BenchmarkResult {
  backend: 'WebGL2' | 'WebGPU';
  resolution: number;
  medianMs: number;
}

const RESOLUTIONS = [512, 1024, 2048];
const WARMUP_STEPS = 5;
const TIMED_STEPS = 24;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function benchmarkWebGL2(
  resolution: number,
  onProgress?: (msg: string) => void,
): Promise<number | null> {
  onProgress?.(`WebGL2 @ ${resolution}×${resolution}…`);
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { antialias: false });
  if (!gl) return null;

  let sim: GrayScottSim;
  try {
    sim = new GrayScottSim(gl, resolution, resolution);
  } catch {
    return null;
  }

  for (let i = 0; i < WARMUP_STEPS; i++) sim.step();
  gl.finish();

  const samples: number[] = [];
  for (let i = 0; i < TIMED_STEPS; i++) {
    const t0 = performance.now();
    sim.step();
    gl.finish();
    samples.push(performance.now() - t0);
  }
  sim.destroy();
  return median(samples);
}

async function benchmarkWebGPU(
  device: GPUDevice,
  resolution: number,
  onProgress?: (msg: string) => void,
): Promise<number | null> {
  onProgress?.(`WebGPU @ ${resolution}×${resolution}…`);
  let sim: WebGPUCompute;
  try {
    sim = new WebGPUCompute(device, resolution, resolution);
  } catch {
    return null;
  }

  sim.stepN(WARMUP_STEPS);
  await device.queue.onSubmittedWorkDone();

  const samples: number[] = [];
  for (let i = 0; i < TIMED_STEPS; i++) {
    const t0 = performance.now();
    sim.step();
    await device.queue.onSubmittedWorkDone();
    samples.push(performance.now() - t0);
  }
  sim.destroy();
  return median(samples);
}

/** Runs N timed steps at 512^2, 1024^2 and 2048^2 on WebGL2, and on WebGPU where
 * available, reporting the median ms/step for each (§4). */
export async function runBenchmark(onProgress?: (msg: string) => void): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  for (const resolution of RESOLUTIONS) {
    const ms = await benchmarkWebGL2(resolution, onProgress);
    if (ms !== null) results.push({ backend: 'WebGL2', resolution, medianMs: ms });
    await nextFrame();
  }

  const device = await requestWebGPUDevice();
  if (device) {
    for (const resolution of RESOLUTIONS) {
      const ms = await benchmarkWebGPU(device, resolution, onProgress);
      if (ms !== null) results.push({ backend: 'WebGPU', resolution, medianMs: ms });
      await nextFrame();
    }
  }

  return results;
}
