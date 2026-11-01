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

// R,G = e1 (structure-tensor eigenvector, points across the local edge),
// B = coherence. See src/gl/TensorField.ts.
uniform sampler2D uTensor;
uniform float uAnisotropic;
uniform float uAnisoStrength;

// 9-point weighted Laplacian:
//   0.05  0.2  0.05
//   0.2   -1   0.2
//   0.05  0.2  0.05
vec2 isotropicLaplacian(vec2 c) {
  vec2 n  = texture(uState, vUv + vec2(0.0, uTexel.y)).rg;
  vec2 s  = texture(uState, vUv + vec2(0.0, -uTexel.y)).rg;
  vec2 e  = texture(uState, vUv + vec2(uTexel.x, 0.0)).rg;
  vec2 w  = texture(uState, vUv + vec2(-uTexel.x, 0.0)).rg;
  vec2 ne = texture(uState, vUv + vec2(uTexel.x, uTexel.y)).rg;
  vec2 nw = texture(uState, vUv + vec2(-uTexel.x, uTexel.y)).rg;
  vec2 se = texture(uState, vUv + vec2(uTexel.x, -uTexel.y)).rg;
  vec2 sw = texture(uState, vUv + vec2(-uTexel.x, -uTexel.y)).rg;
  return (n + s + e + w) * 0.2 + (ne + nw + se + sw) * 0.05 - c;
}

// Directional finite-difference approximation of div(D(x) grad): sample two
// steps along e1 (across the edge, slow) and two along e2 = perp(e1) (along
// the edge, fast), and combine their second differences by D(x)'s
// eigenvalues. 0.3 scales this to match isotropicLaplacian's magnitude, so
// the two modes are comparable at coherence 0 — see Fig. 6.
vec2 anisotropicLaplacian(vec2 c) {
  vec3 t = texture(uTensor, vUv).rgb;
  vec2 e1 = t.xy;
  vec2 e2 = vec2(-e1.y, e1.x);
  float coherence = t.z;

  vec2 p1 = texture(uState, vUv + e1 * uTexel).rg;
  vec2 m1 = texture(uState, vUv - e1 * uTexel).rg;
  vec2 p2 = texture(uState, vUv + e2 * uTexel).rg;
  vec2 m2 = texture(uState, vUv - e2 * uTexel).rg;

  vec2 lapAcross = p1 + m1 - 2.0 * c;
  vec2 lapAlong = p2 + m2 - 2.0 * c;

  float dMin = 1.0 - uAnisoStrength * coherence;
  float dMax = 1.0 + uAnisoStrength * coherence;
  return 0.3 * (lapAcross * dMin + lapAlong * dMax);
}

void main() {
  vec2 c = texture(uState, vUv).rg;
  vec2 lap = uAnisotropic > 0.5 ? anisotropicLaplacian(c) : isotropicLaplacian(c);

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
