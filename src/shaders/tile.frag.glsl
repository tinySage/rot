#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uState;

// Minimal visualisation for the phase-map tiles: V mapped onto a dark
// ground, independent of the UI palette (see CLAUDE.md).
void main() {
  float v = texture(uState, vUv).g;
  vec3 dark = vec3(0.05, 0.045, 0.04);
  vec3 green = vec3(0.33, 0.58, 0.27);
  vec3 color = mix(dark, green, smoothstep(0.0, 1.0, v));
  outColor = vec4(color, 1.0);
}
