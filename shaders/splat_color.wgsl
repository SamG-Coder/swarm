// Pass 3 - colour splat. Every particle on the front-most surface of its pixel
// adds its colour; the per-pixel count doubles as next frame's density feedback.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> depth: array<u32>;
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.res.z) { return; }
  let part = particles[i];
  if (!drawable(part)) { return; }
  let pr = project(u.view_proj, u.res.xy, part.pos);
  if (!pr.ok) { return; }
  if (pr.z > key_depth(depth[pr.idx]) * (1.0 + u.tune.w)) { return; }
  let base = pr.idx * 4u;
  atomicAdd(&accum[base], part.col & 0xffu);
  atomicAdd(&accum[base + 1u], (part.col >> 8u) & 0xffu);
  atomicAdd(&accum[base + 2u], (part.col >> 16u) & 0xffu);
  atomicAdd(&accum[base + 3u], 1u);
}
