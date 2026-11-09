// Pass 2 - depth splat. Nearest particle per pixel wins via atomicMax.
//
// Depth uses a 2x2 footprint (the four pixels around the particle) while
// colour (pass 3) goes to one pixel only. The wider depth footprint makes
// surfaces opaque even where the swarm is thin; resolve then fills the pixels
// that have depth but no colour from their neighbours.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> depth: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u.res.z) { return; }
  let part = particles[i];
  if (!drawable(part)) { return; }
  let pr = project(u.view_proj, u.res.xy, part.pos);
  if (!pr.ok) { return; }
  let key = depth_key(pr.z);
  let hi = vec2i(u.res.xy) - 1;
  let a = clamp(vec2i(floor(pr.f - 0.5)), vec2i(0), hi);
  let b = min(a + 1, hi);
  let w = i32(u.res.x);
  atomicMax(&depth[u32(a.y * w + a.x)], key);
  atomicMax(&depth[u32(a.y * w + b.x)], key);
  atomicMax(&depth[u32(b.y * w + a.x)], key);
  atomicMax(&depth[u32(b.y * w + b.x)], key);
}
