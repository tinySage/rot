import './style.css';
import { GrayScottSim, DEFAULT_PARAMS } from './gl/GrayScottSim';
import { generateSpecimen } from './specimen';

const SIM_WIDTH = 480;
const SIM_HEIGHT = 216;
const MAX_CANVAS_WIDTH = 1800;

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="masthead">
    <h1>Rot: content-aware reaction&ndash;diffusion on photographs</h1>
    <p class="byline">Muktanan Mohapatra</p>
  </header>
  <main>
    <figure class="fig bleed" id="fig1">
      <div class="fig-canvas-wrap" id="canvasWrap">
        <canvas id="simCanvas"></canvas>
        <div class="drop-hint"><span>drop an image, or use the button below</span></div>
      </div>
      <div class="fig-controls">
        <div class="control">
          <label for="fRange">F <span class="readout" id="fReadout"></span></label>
          <input type="range" id="fRange" min="0" max="0.08" step="0.001" />
        </div>
        <div class="control">
          <label for="kRange">k <span class="readout" id="kReadout"></span></label>
          <input type="range" id="kRange" min="0.03" max="0.07" step="0.0005" />
        </div>
        <div class="control">
          <label for="speedRange">steps/frame <span class="readout" id="speedReadout"></span></label>
          <input type="range" id="speedRange" min="1" max="40" step="1" />
        </div>
        <div class="buttons">
          <button type="button" id="pauseBtn"></button>
          <button type="button" id="resetBtn">reset</button>
          <button type="button" class="upload-btn" id="uploadBtn">upload image</button>
          <input type="file" id="fileInput" accept="image/*" />
        </div>
      </div>
      <figcaption>
        <span class="fig-label">Fig. 1</span>
        <span>
          A Gray&ndash;Scott reaction&ndash;diffusion system runs on the GPU and is composited
          over the image below as a flat dark overlay. Upload a photograph, or leave the
          procedurally generated specimen in place.
        </span>
      </figcaption>
    </figure>
  </main>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#simCanvas')!;
const canvasWrap = document.querySelector<HTMLDivElement>('#canvasWrap')!;
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
if (!gl) {
  throw new Error('WebGL2 is not available in this browser.');
}

const sim = new GrayScottSim(gl, SIM_WIDTH, SIM_HEIGHT);
sim.setBaseImage(generateSpecimen(SIM_WIDTH, SIM_HEIGHT));

let stepsPerFrame = 12;
let paused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const fRange = document.querySelector<HTMLInputElement>('#fRange')!;
const kRange = document.querySelector<HTMLInputElement>('#kRange')!;
const speedRange = document.querySelector<HTMLInputElement>('#speedRange')!;
const fReadout = document.querySelector<HTMLSpanElement>('#fReadout')!;
const kReadout = document.querySelector<HTMLSpanElement>('#kReadout')!;
const speedReadout = document.querySelector<HTMLSpanElement>('#speedReadout')!;
const pauseBtn = document.querySelector<HTMLButtonElement>('#pauseBtn')!;
const resetBtn = document.querySelector<HTMLButtonElement>('#resetBtn')!;
const uploadBtn = document.querySelector<HTMLButtonElement>('#uploadBtn')!;
const fileInput = document.querySelector<HTMLInputElement>('#fileInput')!;

function syncReadouts(): void {
  fReadout.textContent = sim.params.F.toFixed(3);
  kReadout.textContent = sim.params.k.toFixed(4);
  speedReadout.textContent = String(stepsPerFrame);
  pauseBtn.textContent = paused ? 'play' : 'pause';
}

fRange.value = String(DEFAULT_PARAMS.F);
kRange.value = String(DEFAULT_PARAMS.k);
speedRange.value = String(stepsPerFrame);
syncReadouts();

fRange.addEventListener('input', () => {
  sim.params.F = Number(fRange.value);
  syncReadouts();
});

kRange.addEventListener('input', () => {
  sim.params.k = Number(kRange.value);
  syncReadouts();
});

speedRange.addEventListener('input', () => {
  stepsPerFrame = Number(speedRange.value);
  syncReadouts();
});

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  syncReadouts();
});

resetBtn.addEventListener('click', () => {
  sim.reset();
});

uploadBtn.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) loadImageFile(file);
});

canvasWrap.addEventListener('dragover', (event) => {
  event.preventDefault();
  canvasWrap.classList.add('drag-over');
});

canvasWrap.addEventListener('dragleave', () => {
  canvasWrap.classList.remove('drag-over');
});

canvasWrap.addEventListener('drop', (event) => {
  event.preventDefault();
  canvasWrap.classList.remove('drag-over');
  const file = event.dataTransfer?.files?.[0];
  if (file) loadImageFile(file);
});

function loadImageFile(file: File): void {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    sim.setBaseImage(img);
    sim.reset();
    canvasWrap.classList.add('has-image');
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function resizeCanvas(): void {
  const rect = canvasWrap.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.min(Math.round(rect.width * dpr), MAX_CANVAS_WIDTH);
  const height = Math.round(rect.height * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

const resizeObserver = new ResizeObserver(resizeCanvas);
resizeObserver.observe(canvasWrap);
resizeCanvas();

function frame(): void {
  if (!paused) {
    sim.stepN(stepsPerFrame);
  }
  sim.renderToCanvas(canvas.width, canvas.height);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
