// Gray-Scott reaction-diffusion step, ported from src/shaders/step.frag.glsl.
// State is a flat array<vec2<f32>> (u, v), ping-ponged between two storage
// buffers. The isotropic nine-point Laplacian is computed from a tile of
// state loaded into workgroup shared memory (with a one-texel halo) rather
// than re-reading global memory for every neighbour; the anisotropic path
// needs samples in an arbitrary direction, so it reads global memory
// directly (see the workgroup-shared-memory note in the prose).

struct Params {
  width: u32,
  height: u32,
  f: f32,
  k: f32,
  du: f32,
  dv: f32,
  dt: f32,
  edgeBoost: f32,
  faceReduction: f32,
  anisotropic: u32,
  anisoStrength: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stateIn: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> stateOut: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> guidance: array<vec2<f32>>; // edge, face
@group(0) @binding(4) var<storage, read> tensor: array<vec4<f32>>;   // e1.x, e1.y, coherence, _

const TILE: u32 = 8u;
var<workgroup> tile: array<array<vec2<f32>, 10>, 10>; // [y][x], 8x8 output + 1-texel halo

fn clampedIndex(x: i32, y: i32) -> u32 {
  let cx = clamp(x, 0, i32(params.width) - 1);
  let cy = clamp(y, 0, i32(params.height) - 1);
  return u32(cy) * params.width + u32(cx);
}

fn sampleState(x: i32, y: i32) -> vec2<f32> {
  return stateIn[clampedIndex(x, y)];
}

// Storage buffers have no hardware texture filtering, so the anisotropic
// path (which samples in an arbitrary, non-grid-aligned direction) does its
// own bilinear interpolation.
fn sampleBilinear(pos: vec2<f32>) -> vec2<f32> {
  let x0 = floor(pos.x);
  let y0 = floor(pos.y);
  let fx = pos.x - x0;
  let fy = pos.y - y0;
  let p00 = sampleState(i32(x0), i32(y0));
  let p10 = sampleState(i32(x0) + 1, i32(y0));
  let p01 = sampleState(i32(x0), i32(y0) + 1);
  let p11 = sampleState(i32(x0) + 1, i32(y0) + 1);
  return mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);
}

@compute @workgroup_size(10, 10, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tileOriginX = i32(wg.x * TILE) - 1;
  let tileOriginY = i32(wg.y * TILE) - 1;

  // Every thread in the 10x10 workgroup loads exactly one shared-memory
  // cell, covering an 8x8 output tile plus its one-texel halo.
  tile[lid.y][lid.x] = sampleState(tileOriginX + i32(lid.x), tileOriginY + i32(lid.y));
  workgroupBarrier();

  // Only the inner 8x8 threads (the halo border excluded) compute output.
  if (lid.x < 1u || lid.x > 8u || lid.y < 1u || lid.y > 8u) {
    return;
  }

  let outX = i32(wg.x * TILE) + i32(lid.x) - 1;
  let outY = i32(wg.y * TILE) + i32(lid.y) - 1;
  if (outX >= i32(params.width) || outY >= i32(params.height)) {
    return;
  }

  let c = tile[lid.y][lid.x];
  var lap: vec2<f32>;

  if (params.anisotropic == 0u) {
    // 9-point weighted Laplacian, entirely from shared memory.
    let n = tile[lid.y + 1u][lid.x];
    let s = tile[lid.y - 1u][lid.x];
    let e = tile[lid.y][lid.x + 1u];
    let w = tile[lid.y][lid.x - 1u];
    let ne = tile[lid.y + 1u][lid.x + 1u];
    let nw = tile[lid.y + 1u][lid.x - 1u];
    let se = tile[lid.y - 1u][lid.x + 1u];
    let sw = tile[lid.y - 1u][lid.x - 1u];
    lap = (n + s + e + w) * 0.2 + (ne + nw + se + sw) * 0.05 - c;
  } else {
    let t = tensor[clampedIndex(outX, outY)];
    let e1 = t.xy;
    let e2 = vec2<f32>(-e1.y, e1.x);
    let coherence = t.z;
    let origin = vec2<f32>(f32(outX), f32(outY));
    let p1 = sampleBilinear(origin + e1);
    let m1 = sampleBilinear(origin - e1);
    let p2 = sampleBilinear(origin + e2);
    let m2 = sampleBilinear(origin - e2);
    let lapAcross = p1 + m1 - 2.0 * c;
    let lapAlong = p2 + m2 - 2.0 * c;
    let dMin = 1.0 - params.anisoStrength * coherence;
    let dMax = 1.0 + params.anisoStrength * coherence;
    lap = 0.3 * (lapAcross * dMin + lapAlong * dMax);
  }

  let g = guidance[clampedIndex(outX, outY)];
  let rateMul = (1.0 + params.edgeBoost * g.x) * (1.0 - params.faceReduction * g.y);
  let fLocal = clamp(params.f * rateMul, 0.0, 1.0);

  let u = c.x;
  let v = c.y;
  let reaction = u * v * v;
  let du = params.du * lap.x - reaction + fLocal * (1.0 - u);
  let dv = params.dv * lap.y + reaction - (fLocal + params.k) * v;

  let newU = clamp(u + du * params.dt, 0.0, 1.0);
  let newV = clamp(v + dv * params.dt, 0.0, 1.0);

  stateOut[clampedIndex(outX, outY)] = vec2<f32>(newU, newV);
}
