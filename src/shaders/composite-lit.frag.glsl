#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uBase;
uniform sampler2D uState;
uniform vec2 uTexel;

// V as a height field: central-difference normals, a fixed key light,
// Blinn-Phong with a wet specular sheen, and a palette running from pale
// grey-green bloom to near-black necrosis by concentration.
void main() {
  float vC = texture(uState, vUv).g;
  float vL = texture(uState, vUv - vec2(uTexel.x, 0.0)).g;
  float vR = texture(uState, vUv + vec2(uTexel.x, 0.0)).g;
  float vD = texture(uState, vUv - vec2(0.0, uTexel.y)).g;
  float vU = texture(uState, vUv + vec2(0.0, uTexel.y)).g;

  const float heightScale = 18.0;
  vec3 normal = normalize(vec3((vL - vR) * heightScale, (vD - vU) * heightScale, 1.0));

  vec3 lightDir = normalize(vec3(0.45, 0.6, 0.66));
  vec3 viewDir = vec3(0.0, 0.0, 1.0);
  vec3 halfDir = normalize(lightDir + viewDir);

  float diff = max(dot(normal, lightDir), 0.0);
  float spec = pow(max(dot(normal, halfDir), 0.0), 48.0);

  vec3 bloom = vec3(0.78, 0.80, 0.74);
  vec3 necrosis = vec3(0.03, 0.03, 0.028);
  vec3 mouldAlbedo = mix(bloom, necrosis, smoothstep(0.0, 1.0, vC));

  vec3 ambient = mouldAlbedo * 0.45;
  vec3 diffuse = mouldAlbedo * diff * 0.75;
  vec3 specular = vec3(0.9, 0.95, 0.92) * spec * 0.6;

  vec3 lit = ambient + diffuse + specular;

  vec3 base = texture(uBase, vUv).rgb;
  float alpha = clamp(vC * 1.6, 0.0, 1.0);
  vec3 color = mix(base, lit, alpha);

  outColor = vec4(color, 1.0);
}
