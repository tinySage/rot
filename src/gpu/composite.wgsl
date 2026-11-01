// Height-mapped Blinn-Phong composite, ported from
// src/shaders/composite-lit.frag.glsl. The state buffer has no hardware
// filtering, so each fragment maps to its nearest simulation texel (the
// base photograph is still a real GPUTexture + sampler, so it stays
// smoothly filtered).

struct Uniforms {
  simWidth: u32,
  simHeight: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> state: array<vec2<f32>>;
@group(0) @binding(2) var baseTex: texture_2d<f32>;
@group(0) @binding(3) var baseSampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VertexOut {
  // Fullscreen triangle, no vertex buffer.
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var out: VertexOut;
  out.position = vec4<f32>(pos[i], 0.0, 1.0);
  out.uv = pos[i] * 0.5 + 0.5;
  out.uv.y = 1.0 - out.uv.y;
  return out;
}

fn clampedIndex(x: i32, y: i32) -> u32 {
  let cx = clamp(x, 0, i32(u.simWidth) - 1);
  let cy = clamp(y, 0, i32(u.simHeight) - 1);
  return u32(cy) * u.simWidth + u32(cx);
}

fn sampleV(x: i32, y: i32) -> f32 {
  return state[clampedIndex(x, y)].y;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
  let gx = i32(in.uv.x * f32(u.simWidth));
  let gy = i32(in.uv.y * f32(u.simHeight));

  let vC = sampleV(gx, gy);
  let vL = sampleV(gx - 1, gy);
  let vR = sampleV(gx + 1, gy);
  let vD = sampleV(gx, gy - 1);
  let vU = sampleV(gx, gy + 1);

  let heightScale = 18.0;
  let normal = normalize(vec3<f32>((vL - vR) * heightScale, (vD - vU) * heightScale, 1.0));

  let lightDir = normalize(vec3<f32>(0.45, 0.6, 0.66));
  let viewDir = vec3<f32>(0.0, 0.0, 1.0);
  let halfDir = normalize(lightDir + viewDir);

  let diff = max(dot(normal, lightDir), 0.0);
  let spec = pow(max(dot(normal, halfDir), 0.0), 48.0);

  let bloom = vec3<f32>(0.78, 0.80, 0.74);
  let necrosis = vec3<f32>(0.03, 0.03, 0.028);
  let mouldAlbedo = mix(bloom, necrosis, smoothstep(0.0, 1.0, vC));

  let ambient = mouldAlbedo * 0.45;
  let diffuse = mouldAlbedo * diff * 0.75;
  let specular = vec3<f32>(0.9, 0.95, 0.92) * spec * 0.6;
  let lit = ambient + diffuse + specular;

  let base = textureSample(baseTex, baseSampler, in.uv).rgb;
  let alpha = clamp(vC * 1.6, 0.0, 1.0);
  let color = mix(base, lit, alpha);

  return vec4<f32>(color, 1.0);
}
