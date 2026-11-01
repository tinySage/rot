# Rot — project rules

## Concept
The site is a living research paper (Distill.pub tradition). Title: "Rot: content-aware reaction–diffusion on photographs". Byline: Muktanan Mohapatra. Every figure is a live simulation. It must read as a publication, not a landing page.

## Stack
Vite + TypeScript, no framework. WebGL2 (WebGPU comes later). KaTeX for equations. @mediapipe/tasks-vision FaceDetector. Fonts self-hosted via @fontsource. Shaders in src/shaders/*.glsl. Static deploy on Vercel.

## Type
- Body: Newsreader, 19px/1.6, optical sizing on. Headings Newsreader 500, tight tracking.
- Captions, labels, UI: IBM Plex Sans 13px.
- Numbers, parameters, readouts: IBM Plex Mono 13px, tabular figures.
- Section heads numbered, sentence case: "2  Guidance from image structure". No Title Case.

## Colour tokens (CSS custom properties on :root)
--paper #F6F3EC  --ink #1C1B19  --muted #6B6760  --rule #D9D4C8  --accent #9A3412 (rust; interactive elements only)
Dark (prefers-color-scheme): --paper #141412 --ink #E8E4DA --muted #8F8A80 --rule #2E2C28, accent unchanged.
Mould render palette lives in the shader, independent of UI tokens.

## Grid
- Body column max 680px; right margin column 220px for sidenotes (they stack inline below 1000px).
- Figure widths: body, page (1080px) or bleed (100vw). Figure 1 is bleed.
- Figures carry a mono label "Fig. 2" plus a Plex Sans caption under a 1px --rule hairline.

## Signature interaction
Inline scrubbable numbers (Bret Victor's explorable-explanation style): parameter values written in the prose (F = 0.037) are draggable horizontally, update the simulation live, and stay synced with the sliders. Hovering a symbol in a KaTeX equation highlights its slider and its region on the phase map.

## Forbidden
Gradients, drop shadows, glassmorphism, border-radius > 2px, emoji, centred hero sections, Inter, decorative animation. The simulation is the only thing that moves. Respect prefers-reduced-motion (start paused).

## Working rules
Do not attempt visual verification; the user tests in the browser. At the end of every phase, run a production build, fix errors, then commit and push to the working branch.
