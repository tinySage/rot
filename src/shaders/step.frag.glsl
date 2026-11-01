#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uF;
uniform float uK;
uniform float uDu;
uniform float uDv;
uniform float uDt;

// 9-point weighted Laplacian:
//   0.05  0.2  0.05
//   0.2   -1   0.2
//   0.05  0.2  0.05
void main() {
  vec2 c  = texture(uState, vUv).rg;
  vec2 n  = texture(uState, vUv + vec2(0.0, uTexel.y)).rg;
  vec2 s  = texture(uState, vUv + vec2(0.0, -uTexel.y)).rg;
  vec2 e  = texture(uState, vUv + vec2(uTexel.x, 0.0)).rg;
  vec2 w  = texture(uState, vUv + vec2(-uTexel.x, 0.0)).rg;
  vec2 ne = texture(uState, vUv + vec2(uTexel.x, uTexel.y)).rg;
  vec2 nw = texture(uState, vUv + vec2(-uTexel.x, uTexel.y)).rg;
  vec2 se = texture(uState, vUv + vec2(uTexel.x, -uTexel.y)).rg;
  vec2 sw = texture(uState, vUv + vec2(-uTexel.x, -uTexel.y)).rg;

  vec2 lap = (n + s + e + w) * 0.2 + (ne + nw + se + sw) * 0.05 - c;

  float u = c.r;
  float v = c.g;
  float reaction = u * v * v;

  float du = uDu * lap.r - reaction + uF * (1.0 - u);
  float dv = uDv * lap.g + reaction - (uF + uK) * v;

  float newU = clamp(u + du * uDt, 0.0, 1.0);
  float newV = clamp(v + dv * uDt, 0.0, 1.0);

  outColor = vec4(newU, newV, 0.0, 1.0);
}
