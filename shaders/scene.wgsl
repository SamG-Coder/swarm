// ---------------------------------------------------------------------------
// Swarm - the scene: signed distance fields + materials + lighting.
// Needs `u: Uniforms` to be declared by the including pass.
// ---------------------------------------------------------------------------

// Everything about the scene that depends on time but not on position. It is
// worked out once per shader invocation (`scene_at`) instead of inside every
// distance evaluation, of which a particle makes dozens per frame.
struct SceneFrame {
  t: f32,
  gyroid: vec2f,              // shell half thickness, core radius
  spin: vec2f,                // sponge: cos and sin of its rotation
  balls: array<vec4f, 6>,     // metaballs: centre, radius
};

// The frame the distance field is evaluated at. Passes set it with
// `sf = scene_at(time)`; the simulation also looks one frame into the past
// to track moving surfaces.
var<private> sf: SceneFrame;

fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}

fn sd_pedestal(p: vec3f) -> f32 {
  let q = p - vec3f(0.0, -1.9, 0.0);
  let d = vec2f(length(q.xz) - 2.94, abs(q.y) - 0.02);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0))) - 0.06;
}

// 0: a breathing gyroid shell carved out of a ball, around a small core
fn sd_gyroid_orb(p: vec3f) -> f32 {
  let q = p * 3.3 + vec3f(0.0, sf.t * 0.35, 0.0);
  let g = dot(sin(q), cos(q.yzx)) / 3.3;
  let shell = abs(g) - sf.gyroid.x;
  let r = length(p);
  return smin(max(shell * 0.62, r - 1.55), r - sf.gyroid.y, 0.22);
}

// 1: metaballs orbiting through a thin ring
fn sd_metaballs(p: vec3f) -> f32 {
  var d = 1e9;
  for (var i = 0; i < 6; i++) {
    let b = sf.balls[i];
    d = smin(d, length(p - b.xyz) - b.w, 0.5);
  }
  let ring = length(vec2f(length(p.xz) - 1.75, p.y)) - 0.07;
  return smin(d, ring, 0.3);
}

// 2: a three-half-twist ribbon around a rippling core
fn sd_ribbon(p: vec3f) -> f32 {
  let t = sf.t;
  let ang = atan2(p.z, p.x);
  let tw = ang * 1.5 + t * 0.6;
  let cs = cos(tw);
  let sn = sin(tw);
  let q0 = vec2f(length(p.xz) - 1.3, p.y);
  let q = vec2f(cs * q0.x - sn * q0.y, sn * q0.x + cs * q0.y);
  let b = abs(q) - vec2f(0.5, 0.07);
  let band = length(max(b, vec2f(0.0))) + min(max(b.x, b.y), 0.0) - 0.05;
  let core = length(p) - 0.5 - 0.05 * sin(7.0 * p.x + t) * sin(7.0 * p.y) * sin(7.0 * p.z - t);
  return min(band * 0.72, core * 0.85);
}

// 3: a turning Menger sponge, three levels deep, cut to a ball
fn sd_sponge(p: vec3f) -> f32 {
  const HALF = 1.2;
  let q = vec3f(sf.spin.x * p.x - sf.spin.y * p.z, p.y, sf.spin.y * p.x + sf.spin.x * p.z) / HALF;
  let b = abs(q) - vec3f(1.0);
  var d = length(max(b, vec3f(0.0))) + min(max(b.x, max(b.y, b.z)), 0.0);
  var s = 1.0;
  for (var m = 0; m < 3; m++) {
    let cell = q * s;
    let a = cell - 2.0 * floor(cell * 0.5) - 1.0;
    s *= 3.0;
    let r = abs(1.0 - 3.0 * abs(a));
    let bars = min(max(r.x, r.y), min(max(r.y, r.z), max(r.z, r.x)));
    d = max(d, (bars - 1.0) / s);
  }
  return max(d * HALF, length(p) - 1.6);
}

fn scene_sdf(id: u32, p: vec3f) -> f32 {
  switch id {
    case 0u { return sd_gyroid_orb(p); }
    case 1u { return sd_metaballs(p); }
    case 2u { return sd_ribbon(p); }
    default { return sd_sponge(p); }
  }
}


fn scene_in_use(id: u32) -> bool {
  return u32(u.morph.x) == id || (u.morph.z > 0.0 && u32(u.morph.y) == id);
}

fn scene_at(t: f32) -> SceneFrame {
  var f: SceneFrame;
  f.t = t;

  f.gyroid = vec2f(0.05 + 0.02 * sin(t * 0.7), 0.42 + 0.06 * sin(t * 1.3));
  f.spin = vec2f(cos(t * 0.25), sin(t * 0.25));
  if (scene_in_use(1u)) {
    for (var i = 0; i < 6; i++) {
      let k = f32(i);
      f.balls[i] = vec4f(
        sin(t * (0.50 + 0.11 * k) + k * 2.1) * 1.2,
        sin(t * (0.37 + 0.07 * k) + k * 1.3) * 0.75,
        cos(t * (0.43 + 0.09 * k) + k * 3.7) * 1.2,
        0.42 + 0.1 * sin(k * 1.7));
    }
  }
  return f;
}

fn map(p: vec3f) -> f32 {
  var d = scene_sdf(u32(u.morph.x), p);
  if (u.morph.z > 0.0) {
    d = mix(d, scene_sdf(u32(u.morph.y), p), u.morph.z);
  }
  return min(d, sd_pedestal(p));
}

// Tetrahedral finite differences: gradient in xyz, distance estimate in w.
fn map_grad(p: vec3f, h: f32) -> vec4f {
  let k = vec2f(1.0, -1.0);
  let a = map(p + k.xyy * h);
  let b = map(p + k.yyx * h);
  let c = map(p + k.yxy * h);
  let d = map(p + k.xxx * h);
  return vec4f((k.xyy * a + k.yyx * b + k.yxy * c + k.xxx * d) / (4.0 * h), (a + b + c + d) * 0.25);
}

// --- materials ----------------------------------------------------------------

fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(6.2831853 * (c * t + d));
}

fn scene_albedo(id: u32, p: vec3f) -> vec3f {
  switch id {
    case 0u {
      return palette(length(p) * 0.62, vec3f(0.55, 0.45, 0.38), vec3f(0.42, 0.34, 0.30),
                     vec3f(1.0, 1.0, 1.0), vec3f(0.02, 0.18, 0.36));
    }
    case 1u {
      return palette(p.y * 0.28 + p.x * 0.10 + 0.5, vec3f(0.60, 0.40, 0.50), vec3f(0.38, 0.30, 0.36),
                     vec3f(1.0, 0.8, 0.9), vec3f(0.95, 0.20, 0.55));
    }
    case 2u {
      let band = palette(atan2(p.z, p.x) / 6.2831853, vec3f(0.48, 0.52, 0.56), vec3f(0.40, 0.36, 0.38),
                         vec3f(1.0, 1.0, 2.0), vec3f(0.55, 0.30, 0.10));
      return mix(vec3f(0.92, 0.88, 0.82), band, smoothstep(0.55, 0.75, length(p)));
    }
    default {
      // terracotta outside, teal deep inside the holes
      return mix(vec3f(0.22, 0.50, 0.56), vec3f(0.80, 0.46, 0.32), smoothstep(0.45, 1.25, length(p)));
    }
  }
}

fn albedo(p: vec3f) -> vec3f {
  if (sd_pedestal(p) < 0.012) {
    let rings = step(0.5, fract(length(p.xz) * 1.5));
    return mix(vec3f(0.36, 0.37, 0.40), vec3f(0.42, 0.43, 0.46), rings);
  }
  var a = scene_albedo(u32(u.morph.x), p);
  if (u.morph.z > 0.0) { a = mix(a, scene_albedo(u32(u.morph.y), p), u.morph.z); }
  return a;
}

// --- lighting -------------------------------------------------------------------

const SUN_COLOR: vec3f = vec3f(1.00, 0.86, 0.68);

fn sky(dir: vec3f) -> vec3f {
  let sun = normalize(u.sun_dir.xyz);
  let up = clamp(dir.y, -1.0, 1.0);
  var c = mix(vec3f(0.50, 0.56, 0.68), vec3f(0.07, 0.13, 0.30), pow(max(up, 0.0), 0.55));
  c = mix(c, vec3f(0.10, 0.11, 0.14), smoothstep(0.0, -0.35, up));
  let s = max(dot(dir, sun), 0.0);
  c += SUN_COLOR * (0.35 * pow(s, 8.0) + 2.5 * pow(s, 600.0));
  return c;
}

fn soft_shadow(p: vec3f, dir: vec3f) -> f32 {
  var res = 1.0;
  var t = 0.04;
  for (var i = 0; i < 14; i++) {
    let h = map(p + dir * t);
    res = min(res, 9.0 * h / t);
    t += clamp(h, 0.03, 0.45);
    if (res < 0.01 || t > 4.5) { break; }
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

fn ambient_occlusion(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var w = 1.0;
  for (var i = 1; i <= 4; i++) {
    let h = 0.07 * f32(i);
    occ += (h - map(p + n * h)) * w;
    w *= 0.6;
  }
  return clamp(1.0 - 2.4 * occ, 0.0, 1.0);
}

fn tonemap(c: vec3f) -> vec3f {
  // ACES fit (Narkowicz), then display gamma
  let x = c * 0.9;
  let m = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(m, vec3f(1.0 / 2.2));
}

// Full shading of one surface point. Returns display-ready rgb in 0..1.
fn shade(p: vec3f, n: vec3f, view: vec3f) -> vec3f {
  let alb = albedo(p);
  let sun = normalize(u.sun_dir.xyz);
  let ndl = max(dot(n, sun), 0.0);
  var sh = 0.0;
  if (ndl > 0.0) { sh = soft_shadow(p + n * 0.02, sun); }
  let ao = ambient_occlusion(p, n);
  let amb = mix(vec3f(0.16, 0.15, 0.16), vec3f(0.34, 0.42, 0.60), n.y * 0.5 + 0.5);
  let spec = pow(max(dot(n, normalize(sun + view)), 0.0), 56.0) * sh * 0.55;
  let fres = pow(1.0 - max(dot(n, view), 0.0), 4.0);
  var c = alb * (SUN_COLOR * (2.6 * ndl * sh) + amb * ao);
  c += SUN_COLOR * spec + amb * (fres * ao * 0.35);
  let dist = distance(p, u.cam_pos.xyz);
  c = mix(c, sky(-view), 1.0 - exp(-max(dist - 4.0, 0.0) * 0.035));
  return tonemap(c);
}
