# Rot

**Rot: content-aware reaction–diffusion on photographs.** A living research
paper (in the Distill.pub tradition) where every figure is a GPU simulation,
not a static illustration. A Gray–Scott reaction–diffusion system is seeded
onto a photograph and left to run; its activator field is composited over
the image as a spreading, lit discoloration that is guided by the
photograph's own edges and faces.

Byline: Muktanan Mohapatra.

## Concept

The page treats a photograph as a substrate for a simulated organism. Rather
than illustrate reaction–diffusion with diagrams, the document *is* the
simulation: the equations are typeset once, and every figure that depends on
them is computed live, in the browser, on the GPU — including a parameter
space explorer, a guidance-field breakdown, a flat-vs-lit rendering
comparison, and a biomass-over-time readout.

## The equations

Gray–Scott is a two-species reaction–diffusion system. A substrate
<em>u</em> is continually replenished and consumed by an autocatalytic
activator <em>v</em>, which in turn decays:

$$
\begin{aligned}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u \;-\; u v^2 \;+\; F\,(1-u) \\
\frac{\partial v}{\partial t} &= D_v \nabla^2 v \;+\; u v^2 \;-\; (F + k)\,v
\end{aligned}
$$

$D_u$ and $D_v$ are diffusion rates (the substrate diffuses roughly twice as
fast as the activator — this asymmetry is what allows stable spots and
stripes to form at all). $F$, the feed rate, sets how quickly substrate is
replenished; $k$, the kill rate, sets how quickly activator is removed.
Different $(F, k)$ pairs produce qualitatively different patterns — spots,
stripes, self-replicating blobs, or nothing at all — which is what the
page's 8×8 phase-map figure is for.

The solver steps are nine-point weighted Laplacians on ping-ponged
`RGBA16F` framebuffers (channel R = $u$, channel G = $v$), integrated with a
forward Euler step, run several times per animation frame before
compositing.

## Guidance from image structure

On upload, the photograph is read twice before the solver ever touches it:

- A **Sobel filter** over luminance gives a per-pixel edge-strength map.
- `@mediapipe/tasks-vision`'s **`FaceDetector`** gives face bounding boxes,
  painted as soft, feathered ellipses into a mask.

Both are blurred and packed into one guidance texture (edge strength in R,
face mask in G), computed once per upload, not per frame. Face detection
runs entirely client-side via a WebAssembly model fetched on first use; if
it fails to load, the face channel simply stays at zero and growth is
driven by edges alone — the feature degrades rather than breaking.

Inside the solver, this texture scales the feed rate $F$ per texel: faces
multiply it down, so rot reaches skin last; edges multiply it up, so growth
follows the photograph's own contours instead of ignoring them.

## Rendering

The activator field $v$ is treated as a height map. Central differences
across the four neighbouring texels give a surface normal, which is shaded
with Blinn-Phong under a fixed key light, plus a tight, high-exponent
specular term standing in for a wet sheen on fresh growth. The mould's
albedo runs along a palette from pale grey-green at low concentration
(bloom) to near-black as $v$ saturates (necrosis), so colour and relief
both track the same field.

## Running locally

Requires Node.js 18+.

```bash
npm install
npm run dev
```

This starts a Vite dev server (default `http://localhost:5173`) with hot
module reload.

To produce a production build:

```bash
npm run build
npm run preview   # serve the built dist/ locally to sanity-check it
```

The build is a static site (`dist/`) — `vercel.json` is set up for a
zero-config static deploy on Vercel, but any static host works.

## Screenshot

<!-- TODO: add a screenshot, e.g. docs/screenshot.png -->
![Screenshot placeholder](docs/screenshot.png)

## Live

<!-- TODO: add the deployed URL once available -->
**Live demo:** _coming soon_
