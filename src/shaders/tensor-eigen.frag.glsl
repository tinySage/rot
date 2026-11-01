#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uTensor;

// Closed-form eigen-decomposition of the smoothed 2x2 structure tensor
// [[Sxx, Sxy], [Sxy, Syy]]. e1 (the larger eigenvalue) points across the
// local edge; its perpendicular e2 points along it. Output: e1.xy in RG,
// coherence (how reliable the orientation is) in B.
void main() {
  vec3 s = texture(uTensor, vUv).rgb;
  float sxx = s.r;
  float syy = s.g;
  float sxy = s.b;

  float theta = 0.5 * atan(2.0 * sxy, sxx - syy);
  vec2 e1 = vec2(cos(theta), sin(theta));

  float trace = sxx + syy;
  float diff = sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy);
  float lambda1 = (trace + diff) * 0.5;
  float lambda2 = (trace - diff) * 0.5;
  float coherence = (lambda1 + lambda2) > 1e-6 ? (lambda1 - lambda2) / (lambda1 + lambda2) : 0.0;

  outColor = vec4(e1, clamp(coherence, 0.0, 1.0), 1.0);
}
