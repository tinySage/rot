import { defineConfig } from 'vite';

export default defineConfig({
  // Figure 1's WebGPU/WebGL2 backend selection uses top-level await.
  esbuild: { target: 'es2022' },
  build: { target: 'es2022' },
});
