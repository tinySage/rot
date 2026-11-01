#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uImage;
uniform vec2 uTexel;
uniform vec2 uDirection;

// Separable 9-tap Gaussian, run horizontally then vertically.
void main() {
  vec2 step = uTexel * uDirection * 2.0;
  vec4 sum = texture(uImage, vUv) * 0.227027;
  sum += texture(uImage, vUv + step * 1.0) * 0.1945946;
  sum += texture(uImage, vUv - step * 1.0) * 0.1945946;
  sum += texture(uImage, vUv + step * 2.0) * 0.1216216;
  sum += texture(uImage, vUv - step * 2.0) * 0.1216216;
  sum += texture(uImage, vUv + step * 3.0) * 0.054054;
  sum += texture(uImage, vUv - step * 3.0) * 0.054054;
  sum += texture(uImage, vUv + step * 4.0) * 0.016216;
  sum += texture(uImage, vUv - step * 4.0) * 0.016216;
  outColor = sum;
}
