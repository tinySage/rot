import './style.css';
import 'katex/dist/katex.min.css';
import { GrayScottSim } from './gl/GrayScottSim';
import { PhaseMap } from './gl/PhaseMap';
import { generateSpecimen } from './specimen';
import { paramStore, F_RANGE, K_RANGE } from './state';
import { enhanceScrubNumbers } from './ui/scrubNumber';
import { renderEquation, wireEquationHover } from './ui/equations';

const SIM_WIDTH = 480;
const SIM_HEIGHT = 216;
const MAX_CANVAS_WIDTH = 1800;
const PHASE_COLS = 8;
const PHASE_ROWS = 8;
const PHASE_TILE_RES = 48;
const PHASE_STEPS_PER_TILE = 3;

const MODEL_EQUATION = String.raw`\begin{aligned}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u \;-\; u v^2 \;+\; \htmlClass{eqsym-F}{F}\,(1-u) \\[0.3em]
\frac{\partial v}{\partial t} &= D_v \nabla^2 v \;+\; u v^2 \;-\; (\htmlClass{eqsym-F}{F} + \htmlClass{eqsym-k}{k})\,v
\end{aligned}`;

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
        <div class="control" data-param="F">
          <label for="fRange">F <span class="readout" id="fReadout"></span></label>
          <input type="range" id="fRange" min="${F_RANGE.min}" max="${F_RANGE.max}" step="0.001" />
        </div>
        <div class="control" data-param="k">
          <label for="kRange">k <span class="readout" id="kReadout"></span></label>
          <input type="range" id="kRange" min="${K_RANGE.min}" max="${K_RANGE.max}" step="0.0005" />
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
          procedurally generated specimen in place. The parameters below are the same
          <em>F</em> and <em>k</em> discussed in &sect;1.
        </span>
      </figcaption>
    </figure>

    <article class="prose">
      <section class="abstract">
        <h2>Abstract</h2>
        <p>
          We render photographs as substrates for a simulated organism. A Gray&ndash;Scott
          reaction&ndash;diffusion system, stepped on the GPU, is seeded onto an image and
          left to run; its activator field is composited over the photograph as a spreading
          discoloration. This page is itself the apparatus: every figure below is a live
          simulation, not a rendering of one. &sect;1 states the model and its two governing
          parameters, feed rate <span class="scrub" data-scrub="F">0.037</span> and kill rate
          <span class="scrub" data-scrub="k">0.0600</span>, both of which are draggable
          wherever they appear in the text. &sect;2 and &sect;3 describe how image structure
          will guide, and currently does not yet guide, the diffusion; &sect;4 covers the
          WebGL2 implementation.
        </p>
      </section>

      <section id="sec-model">
        <h2>1&ensp;The model</h2>
        <p>
          Gray&ndash;Scott
          <a class="cite" href="#ref-2" aria-label="Gray and Scott 1984">[2]</a>
          is a two-species reaction&ndash;diffusion system, a special case of the general
          class of pattern-forming equations Turing proposed as a chemical account of
          morphogenesis
          <a class="cite" href="#ref-1" aria-label="Turing 1952">[1]</a>.
          A substrate <span class="eq-var">u</span> is continually replenished and consumed by
          an autocatalytic activator <span class="eq-var">v</span>, which in turn decays:
        </p>
        <div class="equation" id="eq-model"></div>
        <p>
          <span class="eq-var">D<sub>u</sub></span> and <span class="eq-var">D<sub>v</sub></span>
          are diffusion rates &mdash; the substrate diffuses roughly twice as fast as the
          activator, which is what allows stable spots and stripes to form at all. The feed
          rate, hovered above as <em>F</em>, sets how quickly substrate is added back to the
          system; the kill rate <em>k</em> sets how quickly activator is removed. Hover either
          symbol in the equation to see its control highlighted below and its axis picked out
          on the phase map in Figure&nbsp;2.
        </p>
        <aside class="sidenote">
          In most of the literature <span class="eq-var">u</span> and
          <span class="eq-var">v</span> are called the inhibitor and activator rather than
          substrate and activator; we use the chemical reading because it matches the
          feed/kill terminology directly.
        </aside>
        <p>
          With <em>F</em> near <span class="scrub" data-scrub="F">0.037</span> and
          <em>k</em> near <span class="scrub" data-scrub="k">0.0600</span>, the system sits in
          a narrow band that produces the self-replicating spots used as the default state of
          Figure&nbsp;1 above. Drag either number here &mdash; or the equation's symbols, or
          the sliders, or the crosshair in Figure&nbsp;2 &mdash; and all four stay in sync.
        </p>
      </section>

      <figure class="fig page" id="fig2">
        <div class="phase-map-wrap" id="phaseMapWrap">
          <canvas id="phaseCanvas"></canvas>
          <div class="phase-guide phase-guide-f" id="guideF"></div>
          <div class="phase-guide phase-guide-k" id="guideK"></div>
          <div class="phase-crosshair" id="phaseCrosshair"></div>
        </div>
        <div class="phase-axis-label phase-axis-f">F &rarr;</div>
        <div class="phase-axis-label phase-axis-k">k &uarr;</div>
        <figcaption>
          <span class="fig-label">Fig. 2</span>
          <span>
            Each of the 64 tiles is an independent, continuously running Gray&ndash;Scott
            simulation at a fixed (F, k), sharing one WebGL2 context with Figure&nbsp;1. The
            layout follows Pearson's classification of the patterns this parameter space
            produces
            <a class="cite" href="#ref-3" aria-label="Pearson 1993">[3]</a>.
            Drag the crosshair to drive Figure&nbsp;1 directly.
          </span>
        </figcaption>
      </figure>

      <section id="sec-structure">
        <h2>2&ensp;Guidance from image structure</h2>
        <p>
          The simulation above treats the photograph purely as a backdrop: <em>F</em>,
          <em>k</em>, <span class="eq-var">D<sub>u</sub></span> and
          <span class="eq-var">D<sub>v</sub></span> are uniform across the frame, so the mould
          spreads with no regard for what it is growing on. The next phase makes the image
          itself part of the reaction &mdash; local luminance and edge structure will modulate
          the diffusion tensor directly, so the organism grows faster across flat skies and
          skin, and is slowed or channelled by edges, so that decay reads as following the
          photograph's own contours rather than ignoring them.
        </p>
        <aside class="sidenote">
          The guidance field is planned to come from a luminance gradient and a face-region
          mask (via <code>@mediapipe/tasks-vision</code>), so skin is treated differently from
          background. Neither is wired up yet; this section describes intent, not behaviour.
        </aside>
        <p>
          Concretely, a per-pixel guidance texture computed once per uploaded image &mdash;
          not per frame &mdash; is expected to scale <span class="eq-var">D<sub>u</sub></span>
          and <span class="eq-var">D<sub>v</sub></span> locally, and possibly to perturb
          <em>F</em> itself in high-contrast regions. None of that is implemented in this
          phase: the composite you see is a flat overlay, described next.
        </p>
      </section>

      <section id="sec-rendering">
        <h2>3&ensp;Rendering</h2>
        <p>
          Simulation state lives in a pair of ping-ponged <code>RGBA16F</code> framebuffers:
          channel R holds <span class="eq-var">u</span>, channel G holds
          <span class="eq-var">v</span>. Each step samples a 3&times;3 neighbourhood with the
          classic nine-point weighted Laplacian (centre &minus;1, edge-neighbours 0.2,
          corner-neighbours 0.05) and integrates the two reaction&ndash;diffusion equations
          with a forward Euler step. Figure&nbsp;1 runs <span class="scrub" data-scrub-speed>12</span>
          of these steps per animation frame before compositing, trading a slightly coarser
          time step for visibly faster development.
        </p>
        <p>
          Compositing is presently a single pass: the activator field
          <span class="eq-var">v</span> is mapped onto a dark green, fading to black, and
          mixed over the base photograph by its own magnitude. There is no sampling of the
          image inside the reaction step yet &mdash; see &sect;2.
        </p>
        <aside class="sidenote">
          Figure&nbsp;2's 64 tiles are deliberately small (48&times;48 texels, three steps per
          tile per frame) &mdash; at this scale they exist to show where in parameter space a
          given (F, k) sits, not to look crisp individually.
        </aside>
      </section>

      <section id="sec-implementation">
        <h2>4&ensp;Implementation and performance</h2>
        <p>
          The page is a dependency-light Vite and TypeScript build with no UI framework:
          WebGL2 is driven directly, shaders live as separate <code>.glsl</code> files
          imported as raw text, and layout and type follow the rules in this repository's
          <code>CLAUDE.md</code> rather than a component library. KaTeX renders the one
          equation on this page; nothing else here depends on a math-typesetting pass at
          runtime.
        </p>
        <p>
          Figure&nbsp;1 and Figure&nbsp;2 run in separate WebGL2 contexts &mdash; one per
          <code>&lt;canvas&gt;</code>, as the API requires &mdash; but Figure&nbsp;2's 64
          tiles share a single context and a single pair of compiled programs between them,
          differing only in their per-tile framebuffers and the <em>F</em>, <em>k</em>
          uniforms sent before each tile's draw call. Half-float render targets
          (<code>RGBA16F</code>) are used throughout, which WebGL2 filters linearly without
          an extension; only <code>EXT_color_buffer_float</code> is required to render into
          them at all.
        </p>
      </section>

      <section id="references">
        <h2>References</h2>
        <ol class="refs">
          <li id="ref-1">
            Turing, A.&nbsp;M. (1952). The chemical basis of morphogenesis.
            <em>Philosophical Transactions of the Royal Society B</em>, 237, 37&ndash;72.
          </li>
          <li id="ref-2">
            Gray, P., &amp; Scott, S.&nbsp;K. (1984). Autocatalytic reactions in the
            isothermal, continuous stirred tank reactor.
            <em>Chemical Engineering Science</em>, 39, 1087&ndash;1097.
          </li>
          <li id="ref-3">
            Pearson, J.&nbsp;E. (1993). Complex patterns in a simple system.
            <em>Science</em>, 261, 189&ndash;192.
          </li>
        </ol>
      </section>
    </article>
  </main>
`;

renderEquation('eq-model', MODEL_EQUATION, true);

// --- Figure 1: the main interactive simulation -----------------------------

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

paramStore.subscribe(({ F, k }) => {
  sim.params.F = F;
  sim.params.k = k;
  if (fRange !== document.activeElement) fRange.value = String(F);
  if (kRange !== document.activeElement) kRange.value = String(k);
  fReadout.textContent = F.toFixed(3);
  kReadout.textContent = k.toFixed(4);
});

speedRange.value = String(stepsPerFrame);
speedReadout.textContent = String(stepsPerFrame);
pauseBtn.textContent = paused ? 'play' : 'pause';

fRange.addEventListener('input', () => paramStore.setF(Number(fRange.value)));
kRange.addEventListener('input', () => paramStore.setK(Number(kRange.value)));
speedRange.addEventListener('input', () => {
  stepsPerFrame = Number(speedRange.value);
  speedReadout.textContent = String(stepsPerFrame);
});

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  pauseBtn.textContent = paused ? 'play' : 'pause';
});

resetBtn.addEventListener('click', () => sim.reset());

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) loadImageFile(file);
});

canvasWrap.addEventListener('dragover', (event) => {
  event.preventDefault();
  canvasWrap.classList.add('drag-over');
});
canvasWrap.addEventListener('dragleave', () => canvasWrap.classList.remove('drag-over'));
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

new ResizeObserver(resizeCanvas).observe(canvasWrap);
resizeCanvas();

function frame(): void {
  if (!paused) sim.stepN(stepsPerFrame);
  sim.renderToCanvas(canvas.width, canvas.height);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// --- Figure 2: the (F, k) phase map -----------------------------------------

const phaseCanvas = document.querySelector<HTMLCanvasElement>('#phaseCanvas')!;
const phaseMapWrap = document.querySelector<HTMLDivElement>('#phaseMapWrap')!;
const phaseCrosshair = document.querySelector<HTMLDivElement>('#phaseCrosshair')!;
const guideF = document.querySelector<HTMLDivElement>('#guideF')!;
const guideK = document.querySelector<HTMLDivElement>('#guideK')!;

const phaseGl = phaseCanvas.getContext('webgl2', { antialias: false, alpha: false });
if (!phaseGl) {
  throw new Error('WebGL2 is not available in this browser.');
}
const phaseMap = new PhaseMap(phaseGl, PHASE_COLS, PHASE_ROWS, PHASE_TILE_RES);

function positionCrosshair(F: number, k: number): void {
  const fx = (F - F_RANGE.min) / (F_RANGE.max - F_RANGE.min);
  const ky = 1 - (k - K_RANGE.min) / (K_RANGE.max - K_RANGE.min);
  phaseCrosshair.style.left = `${fx * 100}%`;
  phaseCrosshair.style.top = `${ky * 100}%`;
  guideF.style.left = `${fx * 100}%`;
  guideK.style.top = `${ky * 100}%`;
}

paramStore.subscribe(({ F, k }) => positionCrosshair(F, k));

let draggingCrosshair = false;

function updateFromPointer(clientX: number, clientY: number): void {
  const rect = phaseMapWrap.getBoundingClientRect();
  const fx = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  const ky = Math.min(1, Math.max(0, (clientY - rect.top) / rect.height));
  paramStore.setF(F_RANGE.min + fx * (F_RANGE.max - F_RANGE.min));
  paramStore.setK(K_RANGE.max - ky * (K_RANGE.max - K_RANGE.min));
}

phaseMapWrap.addEventListener('pointerdown', (event) => {
  draggingCrosshair = true;
  phaseMapWrap.setPointerCapture(event.pointerId);
  updateFromPointer(event.clientX, event.clientY);
});
phaseMapWrap.addEventListener('pointermove', (event) => {
  if (!draggingCrosshair) return;
  updateFromPointer(event.clientX, event.clientY);
});
const endCrosshairDrag = (event: PointerEvent) => {
  draggingCrosshair = false;
  try {
    phaseMapWrap.releasePointerCapture(event.pointerId);
  } catch {
    // already released
  }
};
phaseMapWrap.addEventListener('pointerup', endCrosshairDrag);
phaseMapWrap.addEventListener('pointercancel', endCrosshairDrag);

function resizePhaseCanvas(): void {
  const rect = phaseMapWrap.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = Math.round(rect.width * dpr);
  if (phaseCanvas.width !== size || phaseCanvas.height !== size) {
    phaseCanvas.width = size;
    phaseCanvas.height = size;
  }
}
new ResizeObserver(resizePhaseCanvas).observe(phaseMapWrap);
resizePhaseCanvas();

function phaseFrame(): void {
  if (!paused) phaseMap.step(PHASE_STEPS_PER_TILE);
  phaseMap.render(phaseCanvas.width, phaseCanvas.height);
  requestAnimationFrame(phaseFrame);
}
requestAnimationFrame(phaseFrame);

// --- Signature interactions: scrubbable numbers and equation-hover linking -

enhanceScrubNumbers(document);
wireEquationHover(document);

document.querySelectorAll<HTMLElement>('[data-scrub-speed]').forEach((el) => {
  el.classList.add('scrub');
  el.textContent = String(stepsPerFrame);
  let dragging = false;
  let startX = 0;
  let startValue = stepsPerFrame;
  el.addEventListener('pointerdown', (event) => {
    dragging = true;
    startX = event.clientX;
    startValue = stepsPerFrame;
    el.setPointerCapture(event.pointerId);
  });
  el.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const delta = Math.round((event.clientX - startX) / 8);
    stepsPerFrame = Math.min(40, Math.max(1, startValue + delta));
    el.textContent = String(stepsPerFrame);
    speedRange.value = String(stepsPerFrame);
    speedReadout.textContent = String(stepsPerFrame);
  });
  el.addEventListener('pointerup', (event) => {
    dragging = false;
    try {
      el.releasePointerCapture(event.pointerId);
    } catch {
      // already released
    }
  });
});
