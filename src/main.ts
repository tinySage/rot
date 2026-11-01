import './style.css';
import 'katex/dist/katex.min.css';
import { GrayScottSim, DEFAULT_PARAMS } from './gl/GrayScottSim';
import { PhaseMap } from './gl/PhaseMap';
import { Guidance } from './gl/Guidance';
import { Fig4Renderer } from './gl/Fig4Renderer';
import { generateSpecimen } from './specimen';
import { paramStore, F_RANGE, K_RANGE } from './state';
import { enhanceScrubNumbers } from './ui/scrubNumber';
import { renderEquation, wireEquationHover } from './ui/equations';
import { BiomassChart, type BiomassSample } from './ui/BiomassChart';

const SIM_WIDTH = 480;
const SIM_HEIGHT = 216;
const MAX_CANVAS_WIDTH = 1800;
const PHASE_COLS = 8;
const PHASE_ROWS = 8;
const PHASE_TILE_RES = 48;
const PHASE_STEPS_PER_TILE = 3;
const FIG4_STATE_UPDATE_EVERY = 3;
const BIOMASS_SAMPLE_EVERY_FRAMES = 10;
const BIOMASS_MAX_SAMPLES = 300;

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
          A Gray&ndash;Scott reaction&ndash;diffusion system runs on the GPU and is lit and
          composited over the image below (see &sect;3). Upload a photograph &mdash; faces
          slow it down, edges speed it up, see &sect;2 &mdash; or leave the procedurally
          generated specimen in place. The parameters below are the same <em>F</em> and
          <em>k</em> discussed in &sect;1. You can also
          <a href="#" id="recordLink" class="caption-link">record a time-lapse (.webm)</a>
          of this figure.
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
          wherever they appear in the text. &sect;2 derives a guidance field from the uploaded
          photograph itself &mdash; edges and faces &mdash; so growth follows the image's own
          structure; &sect;3 covers how that state is lit and rendered; &sect;4 covers the
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
          On upload, the photograph is read twice before the solver ever touches it. A Sobel
          filter over luminance gives an edge-strength map; <code>@mediapipe/tasks-vision</code>'s
          <code>FaceDetector</code> gives face bounding boxes, which are painted as soft,
          feathered ellipses into a mask. Both are blurred and packed into one guidance
          texture &mdash; edge strength in R, face mask in G &mdash; computed once per upload,
          not per frame.
        </p>
        <aside class="sidenote">
          Face detection runs entirely client-side via a WebAssembly model fetched on first
          use. If it fails to load (offline, or blocked), the face channel stays at zero and
          growth is driven by edges alone &mdash; it degrades rather than breaking.
        </aside>
        <p>
          Inside the solver this texture scales the feed rate <em>F</em> per texel: faces
          multiply it down, so rot reaches skin last; edges multiply it up, so growth follows
          the photograph's own contours instead of ignoring them. <span class="eq-var">D<sub>u</sub></span>
          and <span class="eq-var">D<sub>v</sub></span> are still uniform &mdash; only the feed
          term is content-aware in this phase.
        </p>
      </section>

      <figure class="fig page" id="fig3">
        <div class="guidance-grid">
          <div class="guidance-panel">
            <canvas id="edgeCanvas" width="480" height="216"></canvas>
            <div class="guidance-panel-label">edge map</div>
          </div>
          <div class="guidance-panel">
            <canvas id="faceCanvas" width="480" height="216"></canvas>
            <div class="guidance-panel-label">face mask</div>
          </div>
          <div class="guidance-panel">
            <canvas id="rateCanvas" width="480" height="216"></canvas>
            <div class="guidance-panel-label">resulting rate field</div>
          </div>
        </div>
        <figcaption>
          <span class="fig-label">Fig. 3</span>
          <span>
            The guidance texture split into its two channels, and the feed-rate multiplier
            they produce together: blue where growth is suppressed (faces), orange where it is
            boosted (edges), grey where <em>F</em> is left alone. Computed from whatever image
            is currently loaded in Figure&nbsp;1.
          </span>
        </figcaption>
      </figure>

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
          Compositing now treats <span class="eq-var">v</span> as a height field rather than a
          flat mask. Central differences across the four neighbouring texels give a surface
          normal; a fixed key light is shaded with Blinn-Phong &mdash; diffuse plus a tight,
          high-exponent specular term standing in for a wet sheen on fresh growth. The albedo
          itself runs along a palette from pale grey-green at low concentration (bloom) to
          near-black as <span class="eq-var">v</span> saturates (necrosis), so colour and
          relief both track the same field.
        </p>
        <aside class="sidenote">
          Figure&nbsp;2's 64 tiles are deliberately small (48&times;48 texels, three steps per
          tile per frame) &mdash; at this scale they exist to show where in parameter space a
          given (F, k) sits, not to look crisp individually.
        </aside>
      </section>

      <figure class="fig page" id="fig4">
        <div class="fig4-wrap">
          <canvas id="fig4Canvas"></canvas>
        </div>
        <div class="fig4-labels">
          <span>flat</span>
          <span>lit</span>
        </div>
        <figcaption>
          <span class="fig-label">Fig. 4</span>
          <span>
            The same simulation state, Figure&nbsp;1's, rendered two ways: the Phase&nbsp;1
            flat overlay on the left, the height-mapped Blinn-Phong composite described above
            on the right. Figure&nbsp;1 itself now uses the lit version.
          </span>
        </figcaption>
      </figure>

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
          Right now this page is running Figure&nbsp;1 at
          <span class="readout" id="fpsReadout">&mdash;</span> fps, at a simulation
          resolution of <span class="readout" id="resReadout">&mdash;</span>, with
          <span class="readout" id="stepsReadout">&mdash;</span> reaction&ndash;diffusion
          steps per animation frame. Figure&nbsp;5 tracks one more live number: total
          biomass over time.
        </p>
      </section>

      <figure class="fig page" id="fig5">
        <div class="biomass-chart-wrap">
          <canvas id="biomassCanvas"></canvas>
        </div>
        <figcaption>
          <span class="fig-label">Fig. 5</span>
          <span>
            Total biomass (&sum;<span class="eq-var">v</span>) in Figure&nbsp;1 over time.
            Sampled every few frames from a GPU-downsampled copy of the state texture, not
            the full-resolution buffer, so the periodic <code>readPixels</code> this needs
            stays cheap. Resets whenever the simulation does.
          </span>
        </figcaption>
      </figure>

      <section id="sec-implementation-detail">
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
        <p>
          Figure&nbsp;4 poses the same context problem from the other direction: it needs
          Figure&nbsp;1's exact simulation state, but textures can't cross a context boundary.
          Re-running the reaction-diffusion step a second time would drift from Figure&nbsp;1
          and waste the GPU redoing real work, so instead only the cheap part is duplicated
          &mdash; Figure&nbsp;1's (<span class="eq-var">u</span>, <span class="eq-var">v</span>)
          state is read back and re-uploaded into Figure&nbsp;4's own context every third
          frame, where its two composite shaders run fresh.
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
const guidance = new Guidance(gl, SIM_WIDTH, SIM_HEIGHT);

const specimenCanvas = generateSpecimen(SIM_WIDTH, SIM_HEIGHT);
sim.setBaseImage(specimenCanvas);

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
const fpsReadout = document.querySelector<HTMLSpanElement>('#fpsReadout')!;
const resReadout = document.querySelector<HTMLSpanElement>('#resReadout')!;
const stepsReadout = document.querySelector<HTMLSpanElement>('#stepsReadout')!;

paramStore.subscribe(({ F, k }) => {
  sim.params.F = F;
  sim.params.k = k;
  if (fRange !== document.activeElement) fRange.value = String(F);
  if (kRange !== document.activeElement) kRange.value = String(k);
  fReadout.textContent = F.toFixed(3);
  kReadout.textContent = k.toFixed(4);
});

function setStepsPerFrame(n: number): void {
  stepsPerFrame = Math.min(40, Math.max(1, n));
  speedRange.value = String(stepsPerFrame);
  speedReadout.textContent = String(stepsPerFrame);
  stepsReadout.textContent = String(stepsPerFrame);
}

setStepsPerFrame(stepsPerFrame);
resReadout.textContent = `${SIM_WIDTH}×${SIM_HEIGHT}`;
pauseBtn.textContent = paused ? 'play' : 'pause';

fRange.addEventListener('input', () => paramStore.setF(Number(fRange.value)));
kRange.addEventListener('input', () => paramStore.setK(Number(kRange.value)));
speedRange.addEventListener('input', () => setStepsPerFrame(Number(speedRange.value)));

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  pauseBtn.textContent = paused ? 'play' : 'pause';
});

resetBtn.addEventListener('click', () => {
  sim.reset();
  resetBiomassHistory();
});

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
    resetBiomassHistory();
    canvasWrap.classList.add('has-image');
    fig4.setBaseImage(img);
    void updateGuidance(img, img.naturalWidth, img.naturalHeight);
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

// --- Figure 5: biomass over time --------------------------------------------

const biomassCanvas = document.querySelector<HTMLCanvasElement>('#biomassCanvas')!;
const biomassChart = new BiomassChart(biomassCanvas);
let biomassHistory: BiomassSample[] = [];
let biomassFrameCount = 0;
let simStartTime = performance.now();

function resetBiomassHistory(): void {
  biomassHistory = [];
  simStartTime = performance.now();
  biomassChart.setSamples(biomassHistory);
}

function recordBiomassSample(): void {
  const t = (performance.now() - simStartTime) / 1000;
  const v = sim.readBiomass();
  biomassHistory = [...biomassHistory, { t, v }];
  if (biomassHistory.length > BIOMASS_MAX_SAMPLES) {
    biomassHistory = biomassHistory.slice(biomassHistory.length - BIOMASS_MAX_SAMPLES);
  }
  biomassChart.setSamples(biomassHistory);
}

// --- Figure 1 caption: record a time-lapse ----------------------------------

const recordLink = document.querySelector<HTMLAnchorElement>('#recordLink')!;
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let recordStartTime = 0;
let recordTimerId: number | undefined;

recordLink.addEventListener('click', (event) => {
  event.preventDefault();
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  startRecording();
});

function startRecording(): void {
  if (typeof canvas.captureStream !== 'function' || typeof MediaRecorder === 'undefined') {
    recordLink.textContent = 'recording unsupported in this browser';
    return;
  }

  const stream = canvas.captureStream(30);
  recordedChunks = [];
  const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9'
    : 'video/webm';
  mediaRecorder = new MediaRecorder(stream, { mimeType });

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rot-timelapse-${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);

    recordLink.textContent = 'record a time-lapse (.webm)';
    recordLink.classList.remove('recording');
    if (recordTimerId !== undefined) window.clearInterval(recordTimerId);
  };

  mediaRecorder.start();
  recordStartTime = performance.now();
  recordLink.classList.add('recording');
  recordTimerId = window.setInterval(() => {
    const elapsed = Math.floor((performance.now() - recordStartTime) / 1000);
    recordLink.textContent = `stop recording (${elapsed}s)`;
  }, 250);
}

// --- Figure 3: edge map | face mask | resulting rate field -----------------

const edgeCanvas = document.querySelector<HTMLCanvasElement>('#edgeCanvas')!;
const faceCanvas = document.querySelector<HTMLCanvasElement>('#faceCanvas')!;
const rateCanvas = document.querySelector<HTMLCanvasElement>('#rateCanvas')!;
const edgeCtx = edgeCanvas.getContext('2d')!;
const faceCtx = faceCanvas.getContext('2d')!;
const rateCtx = rateCanvas.getContext('2d')!;

const RATE_LOW = [0.33, 0.49, 0.76]; // suppressed growth (faces) -> cool blue
const RATE_MID = [0.78, 0.76, 0.69]; // neutral, F unchanged -> paper grey
const RATE_HIGH = [0.69, 0.33, 0.14]; // boosted growth (edges) -> rust

function renderGuidancePanels(): void {
  const data = guidance.pixelData;
  const w = SIM_WIDTH;
  const h = SIM_HEIGHT;

  const edgeImg = edgeCtx.createImageData(w, h);
  const faceImg = faceCtx.createImageData(w, h);
  const rateImg = rateCtx.createImageData(w, h);

  const { edgeBoost, faceReduction } = DEFAULT_PARAMS;
  const minMul = 1 - faceReduction;
  const maxMul = 1 + edgeBoost;

  for (let i = 0; i < w * h; i++) {
    const src = i * 4;
    const edge = data[src] / 255;
    const face = data[src + 1] / 255;

    const e = Math.round(edge * 255);
    edgeImg.data[src] = e;
    edgeImg.data[src + 1] = e;
    edgeImg.data[src + 2] = e;
    edgeImg.data[src + 3] = 255;

    faceImg.data[src] = Math.round(255 * face);
    faceImg.data[src + 1] = Math.round(140 * face);
    faceImg.data[src + 2] = Math.round(60 * face);
    faceImg.data[src + 3] = 255;

    const multiplier = (1 + edgeBoost * edge) * (1 - faceReduction * face);
    let rgb: number[];
    if (multiplier < 1) {
      const t = (multiplier - minMul) / (1 - minMul);
      rgb = lerpColor(RATE_LOW, RATE_MID, t);
    } else {
      const t = (multiplier - 1) / (maxMul - 1);
      rgb = lerpColor(RATE_MID, RATE_HIGH, t);
    }
    rateImg.data[src] = Math.round(rgb[0] * 255);
    rateImg.data[src + 1] = Math.round(rgb[1] * 255);
    rateImg.data[src + 2] = Math.round(rgb[2] * 255);
    rateImg.data[src + 3] = 255;
  }

  edgeCtx.putImageData(edgeImg, 0, 0);
  faceCtx.putImageData(faceImg, 0, 0);
  rateCtx.putImageData(rateImg, 0, 0);
}

function lerpColor(a: number[], b: number[], t: number): number[] {
  const u = Math.min(1, Math.max(0, t));
  return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
}

async function updateGuidance(
  source: HTMLImageElement | HTMLCanvasElement,
  sourceWidth: number,
  sourceHeight: number,
): Promise<void> {
  await guidance.update(source, sourceWidth, sourceHeight);
  sim.setGuidanceTexture(guidance.texture);
  renderGuidancePanels();
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

let fpsFrameCount = 0;
let fpsLastTime = performance.now();

function frame(): void {
  if (!paused) {
    sim.stepN(stepsPerFrame);
    biomassFrameCount++;
    if (biomassFrameCount % BIOMASS_SAMPLE_EVERY_FRAMES === 0) recordBiomassSample();
  }
  sim.renderToCanvas(canvas.width, canvas.height);

  fpsFrameCount++;
  const now = performance.now();
  const elapsed = now - fpsLastTime;
  if (elapsed >= 500) {
    fpsReadout.textContent = ((fpsFrameCount * 1000) / elapsed).toFixed(0);
    fpsFrameCount = 0;
    fpsLastTime = now;
  }

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

void updateGuidance(specimenCanvas, SIM_WIDTH, SIM_HEIGHT);

// --- Figure 4: flat vs lit, same sim state ----------------------------------

const fig4Canvas = document.querySelector<HTMLCanvasElement>('#fig4Canvas')!;
const fig4Wrap = document.querySelector<HTMLDivElement>('.fig4-wrap')!;
const fig4Gl = fig4Canvas.getContext('webgl2', { antialias: false, alpha: false });
if (!fig4Gl) {
  throw new Error('WebGL2 is not available in this browser.');
}
const fig4 = new Fig4Renderer(fig4Gl, SIM_WIDTH, SIM_HEIGHT);
fig4.setBaseImage(specimenCanvas);

function resizeFig4Canvas(): void {
  const rect = fig4Wrap.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.min(Math.round(rect.width * dpr), MAX_CANVAS_WIDTH);
  const height = Math.round(rect.height * dpr);
  if (fig4Canvas.width !== width || fig4Canvas.height !== height) {
    fig4Canvas.width = width;
    fig4Canvas.height = height;
  }
}
new ResizeObserver(resizeFig4Canvas).observe(fig4Wrap);
resizeFig4Canvas();

let fig4FrameCount = 0;
function fig4Frame(): void {
  fig4FrameCount++;
  if (!paused && fig4FrameCount % FIG4_STATE_UPDATE_EVERY === 0) {
    fig4.updateState(sim.readStateData());
  }
  fig4.render(fig4Canvas.width, fig4Canvas.height);
  requestAnimationFrame(fig4Frame);
}
requestAnimationFrame(fig4Frame);

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
    setStepsPerFrame(startValue + delta);
    el.textContent = String(stepsPerFrame);
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
