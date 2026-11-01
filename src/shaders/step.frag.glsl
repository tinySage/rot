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

// R = edge strength, G = face mask, both blurred. See src/gl/Guidance.ts.
uniform sampler2D uGuidance;
uniform float uEdgeBoost;
uniform float uFaceReduction;

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

  // Faces get a reduced feed rate (rot reaches them last); edges get
  // increased growth, so decay reads as following the photograph's contours.
  vec2 guide = texture(uGuidance, vUv).rg;
  float rateMul = (1.0 + uEdgeBoost * guide.r) * (1.0 - uFaceReduction * guide.g);
  float fLocal = clamp(uF * rateMul, 0.0, 1.0);

  float du = uDu * lap.r - reaction + fLocal * (1.0 - u);
  float dv = uDv * lap.g + reaction - (fLocal + uK) * v;

  float newU = clamp(u + du * uDt, 0.0, 1.0);
  float newV = clamp(v + dv * uDt, 0.0, 1.0);

  outColor = vec4(newU, newV, 0.0, 1.0);
}
