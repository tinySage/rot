#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uBase;
uniform sampler2D uState;

// Flat dark-green-to-black overlay driven by V. This is a placeholder —
// content-aware guidance from image structure arrives in a later phase.
void main() {
  float v = texture(uState, vUv).g;
  vec3 base = texture(uBase, vUv).rgb;

  vec3 darkGreen = vec3(0.035, 0.11, 0.045);
  vec3 overlay = mix(vec3(0.0), darkGreen, smoothstep(0.0, 1.0, v));
  float alpha = clamp(v * 1.6, 0.0, 1.0);

  vec3 color = mix(base, overlay, alpha);
  outColor = vec4(color, 1.0);
}
