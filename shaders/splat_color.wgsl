// Pass 2 - colour splat. Every particle on the front-most surface of its pixel
// adds its colour; the per-pixel count doubles as next frame's density feedback.
// The depth buffer it tests against was written by the simulation pass.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> depth: array<u32>;
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * ROW;
  if (i >= u.res.z) { return; }
  let part = particles[i];
  if (!drawable(part)) { return; }
  let pr = project(u.view_proj, u.res.xy, part.pos);
  if (!pr.ok) { return; }
  if (pr.z > key_depth(depth[pr.idx]) * (1.0 + u.tune.w)) { return; }
  // drop the low bits with random rounding, so the average stays exact
  let shift = u32(u.pace.y);
  let round = pcg(i ^ (u.res.w * 40503u)) & ((1u << shift) - 1u);
  let rgb = (vec3u(part.col, part.col >> 8u, part.col >> 16u) & vec3u(0xffu)) + round;
  let c = rgb >> vec3u(shift);
  let base = pr.idx * 2u;
  atomicAdd(&accum[base], c.x | (c.y << 16u));
  atomicAdd(&accum[base + 1u], c.z | 0x10000u);
}
