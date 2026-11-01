// Builds the site, serves it, and captures the README media for Rot's
// /?demo mode: a full-page screenshot, three Figure 1 decay-sequence
// crops, a 20s mp4 and a 10s gif of Figure 1. Run with `npm run capture`.
//
// Requires Chromium (via Playwright) and ffmpeg on PATH.

import { chromium } from 'playwright';
import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const DOCS_DIR = path.join(ROOT, 'docs');
const PORT = 4799;
const BASE_URL = `http://localhost:${PORT}`;
const VIEWPORT = { width: 1440, height: 900 };

const CHROMIUM_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];

function run(cmd: string, args: string[], cwd: string): void {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  const result = spawnSync(cmd, args, { cwd, stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} exited with code ${result.status}`);
  }
}

function waitForServer(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      fetch(url)
        .then((res) => {
          if (res.ok) resolve();
          else if (Date.now() > deadline) reject(new Error(`${url} never returned 200`));
          else setTimeout(attempt, 300);
        })
        .catch(() => {
          if (Date.now() > deadline) reject(new Error(`${url} never came up`));
          else setTimeout(attempt, 300);
        });
    };
    attempt();
  });
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ffmpeg(args: string[]): void {
  console.log(`$ ffmpeg ${args.join(' ')}`);
  const result = spawnSync('ffmpeg', args, { stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`ffmpeg ${args.join(' ')} exited with code ${result.status}`);
  }
}

async function main(): Promise<void> {
  mkdirSync(DOCS_DIR, { recursive: true });

  console.log('Building...');
  run('npm', ['run', 'build'], ROOT);

  console.log('Starting vite preview...');
  const preview: ChildProcessWithoutNullStreams = spawn(
    'npx',
    ['vite', 'preview', '--port', String(PORT), '--strictPort'],
    { cwd: ROOT, stdio: 'pipe' },
  );
  preview.stdout.on('data', (d) => process.stdout.write(`[preview] ${d}`));
  preview.stderr.on('data', (d) => process.stderr.write(`[preview] ${d}`));

  const videoDir = mkdtempSync(path.join(tmpdir(), 'rot-capture-'));

  try {
    await waitForServer(BASE_URL, 30000);
    console.log('Preview server is up.');

    console.log(`Launching Chromium (${CHROMIUM_ARGS.join(' ')})...`);
    const browser = await chromium.launch({ headless: true, args: CHROMIUM_ARGS });
    const context = await browser.newContext({
      viewport: VIEWPORT,
      recordVideo: { dir: videoDir, size: VIEWPORT },
    });
    const page = await context.newPage();

    // Uncaught exceptions abort the capture; console-level errors (a blocked
    // third-party fetch for the face-detector model, say) are logged but
    // don't — Guidance already degrades gracefully when that happens.
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') console.warn(`[console] ${msg.text()}`);
    });

    console.log(`Opening ${BASE_URL}/?demo ...`);
    await page.goto(`${BASE_URL}/?demo`, { waitUntil: 'networkidle' });

    const canvasWrap = page.locator('#canvasWrap');
    await canvasWrap.waitFor({ state: 'visible' });
    // Let WebGL context setup and the first few frames settle.
    await wait(2000);

    const checkForPageErrors = () => {
      if (pageErrors.length > 0) {
        throw new Error(`Uncaught page errors:\n${pageErrors.join('\n')}`);
      }
    };
    checkForPageErrors();

    const box = await canvasWrap.boundingBox();
    if (!box) throw new Error('Could not read #canvasWrap bounding box');
    const crop = {
      x: Math.round(box.x),
      y: Math.round(box.y),
      w: Math.floor(box.width / 2) * 2,
      h: Math.floor(box.height / 2) * 2,
    };
    console.log('Figure 1 crop rect:', crop);

    // Plain page.screenshot({ clip }) rather than locator.screenshot(): the
    // latter's actionability "wait until stable" check can stall for a long
    // time under software rendering with six concurrent WebGL2 contexts
    // competing for the main thread — a clip rect just rasterizes the rect,
    // no stability wait.
    const fig1Clip = { x: crop.x, y: crop.y, width: crop.w, height: crop.h };
    const screenshotTimeout = 30000;

    console.log('t=0s: capturing fig1-00s.png');
    await page.screenshot({ path: path.join(DOCS_DIR, 'fig1-00s.png'), clip: fig1Clip, timeout: screenshotTimeout });

    await wait(18000);
    checkForPageErrors();
    console.log('t=20s: capturing fig1-20s.png');
    await page.screenshot({ path: path.join(DOCS_DIR, 'fig1-20s.png'), clip: fig1Clip, timeout: screenshotTimeout });

    await wait(20000);
    checkForPageErrors();
    console.log('t=40s: capturing screenshot.png (full page)');
    await page.screenshot({ path: path.join(DOCS_DIR, 'screenshot.png'), fullPage: true, timeout: screenshotTimeout });

    await wait(20000);
    checkForPageErrors();
    console.log('t=60s: capturing fig1-60s.png');
    await page.screenshot({ path: path.join(DOCS_DIR, 'fig1-60s.png'), clip: fig1Clip, timeout: screenshotTimeout });

    const video = page.video();
    await context.close();
    await browser.close();

    if (!video) throw new Error('No video was recorded');
    const rawVideoPath = await video.path();
    console.log('Raw recording:', rawVideoPath);

    const cropFilter = `crop=${crop.w}:${crop.h}:${crop.x}:${crop.y}`;

    console.log('Encoding docs/demo.mp4 (H.264, 20s, 960px wide)...');
    ffmpeg([
      '-y',
      '-i', rawVideoPath,
      '-t', '20',
      '-vf', `${cropFilter},scale=960:-2`,
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-crf', '23',
      '-pix_fmt', 'yuv420p',
      '-an',
      path.join(DOCS_DIR, 'demo.mp4'),
    ]);

    console.log('Encoding docs/demo.gif (10s, 720px, two-pass palette)...');
    const palettePath = path.join(videoDir, 'palette.png');
    ffmpeg([
      '-y',
      '-i', rawVideoPath,
      '-t', '10',
      '-vf', `${cropFilter},scale=720:-2:flags=lanczos,fps=15,palettegen`,
      palettePath,
    ]);
    ffmpeg([
      '-y',
      '-i', rawVideoPath,
      '-i', palettePath,
      '-t', '10',
      '-filter_complex',
      `[0:v]${cropFilter},scale=720:-2:flags=lanczos,fps=15[x];[x][1:v]paletteuse`,
      path.join(DOCS_DIR, 'demo.gif'),
    ]);

    console.log('\nDone. Wrote to docs/:');
    for (const name of ['screenshot.png', 'fig1-00s.png', 'fig1-20s.png', 'fig1-60s.png', 'demo.mp4', 'demo.gif']) {
      const p = path.join(DOCS_DIR, name);
      console.log(`  ${name}${existsSync(p) ? '' : '  <-- MISSING'}`);
    }
  } finally {
    preview.kill();
    rmSync(videoDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
