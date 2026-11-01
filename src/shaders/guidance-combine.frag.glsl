#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uEdge;
uniform sampler2D uFace;

void main() {
  float edge = texture(uEdge, vUv).r;
  float face = texture(uFace, vUv).r;
  outColor = vec4(edge, face, 0.0, 1.0);
}
