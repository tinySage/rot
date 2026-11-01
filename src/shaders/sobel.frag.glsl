#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uImage;
uniform vec2 uTexel;

float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
  float tl = luma(texture(uImage, vUv + uTexel * vec2(-1.0, 1.0)).rgb);
  float t  = luma(texture(uImage, vUv + uTexel * vec2(0.0, 1.0)).rgb);
  float tr = luma(texture(uImage, vUv + uTexel * vec2(1.0, 1.0)).rgb);
  float l  = luma(texture(uImage, vUv + uTexel * vec2(-1.0, 0.0)).rgb);
  float r  = luma(texture(uImage, vUv + uTexel * vec2(1.0, 0.0)).rgb);
  float bl = luma(texture(uImage, vUv + uTexel * vec2(-1.0, -1.0)).rgb);
  float b  = luma(texture(uImage, vUv + uTexel * vec2(0.0, -1.0)).rgb);
  float br = luma(texture(uImage, vUv + uTexel * vec2(1.0, -1.0)).rgb);

  float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
  float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
  float mag = clamp(length(vec2(gx, gy)) * 0.6, 0.0, 1.0);

  outColor = vec4(mag, 0.0, 0.0, 1.0);
}
