// ---------------------------------------------------------------------------
// Swarm - shared prelude (prepended to every pass by the Rust core)
// ---------------------------------------------------------------------------

struct Uniforms {
  view_proj: mat4x4f,       // clip.w is view-space depth
  prev_view_proj: mat4x4f,  // last frame's matrix, for reading last frame's buffers
  cam_pos: vec4f,           // xyz, time (s)
  sun_dir: vec4f,           // xyz, focal length in pixels
  res: vec4u,               // width, height, active particles, frame index
  morph: vec4f,             // scene a, scene b, blend 0..1, jitter (px)
  cam_right: vec4f,         // xyz, tan(fov_y / 2)
  cam_up: vec4f,            // xyz, seconds since the previous frame
  cam_fwd: vec4f,           // xyz, debug mode
  tune: vec4f,              // turnover, respawn rate, restart rate, depth tolerance
};

struct Particle {
  pos: vec3f,
  col: u32,                 // rgb in the low 24 bits, flags in the high 8
};

const FLAG_SURFACE: u32 = 1u;   // sits on the surface (converged)
const FLAG_VISIBLE: u32 = 2u;   // survived last frame's depth test
const FLAG_BACK: u32 = 4u;      // faces away from the camera: never drawn
const NEAR: f32 = 0.05;
const BOUND: f32 = 4.0;         // particles live in [-BOUND, BOUND]^3

// --- random numbers ---------------------------------------------------------

var<private> rng_state: u32;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand() -> f32 {
  rng_state = pcg(rng_state);
  return f32(rng_state >> 8u) * (1.0 / 16777216.0);
}
fn rand_unit() -> vec3f {
  let z = rand() * 2.0 - 1.0;
  let a = rand() * 6.2831853;
  let r = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(r * cos(a), r * sin(a), z);
}

// --- projection -------------------------------------------------------------

struct Proj {
  ok: bool,
  idx: u32,   // pixel index
  f: vec2f,   // continuous pixel coordinates
  z: f32,     // view depth
};

fn project(vp: mat4x4f, res: vec2u, p: vec3f) -> Proj {
  var o: Proj;
  o.ok = false;
  let c = vp * vec4f(p, 1.0);
  if (c.w < NEAR) { return o; }
  let ndc = c.xy / c.w;
  if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0) { return o; }
  let fx = (ndc.x * 0.5 + 0.5) * f32(res.x);
  let fy = (0.5 - ndc.y * 0.5) * f32(res.y);
  let x = min(u32(fx), res.x - 1u);
  let y = min(u32(fy), res.y - 1u);
  o.ok = true;
  o.f = vec2f(fx, fy);
  o.idx = y * res.x + x;
  o.z = c.w;
  return o;
}

// Depth is stored as the bit pattern of 1/z: positive floats order like
// integers, so "nearest wins" is a single atomicMax and 0 means "empty".
// A particle is drawn if it sits on the surface and faces the camera.
fn drawable(p: Particle) -> bool {
  return ((p.col >> 24u) & (FLAG_SURFACE | FLAG_BACK)) == FLAG_SURFACE;
}

fn depth_key(z: f32) -> u32 { return bitcast<u32>(1.0 / z); }
fn key_depth(k: u32) -> f32 { return 1.0 / bitcast<f32>(k); }
