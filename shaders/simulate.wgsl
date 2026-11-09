// ---------------------------------------------------------------------------
// Pass 1 - simulate. One invocation per particle.
//
// Every particle (a) looks up what happened to it in LAST frame's framebuffer,
// (b) migrates if the image did not need it, (c) falls back onto the surface,
// and (d) shades itself. Geometry sampling is driven by image feedback.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dst: array<Particle>;
@group(0) @binding(3) var<storage, read> depth_prev: array<u32>;
@group(0) @binding(4) var<storage, read> accum_prev: array<u32>;

fn random_in_bounds() -> vec3f {
  return (vec3f(rand(), rand(), rand()) * 2.0 - 1.0) * vec3f(3.2, 2.2, 3.2);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n_active = u.res.z;
  if (i >= n_active) { return; }
  rng_state = pcg(i ^ pcg(u.res.w * 2654435761u + 17u));
  scene_time = u.cam_pos.w;

  let cam = u.cam_pos.xyz;
  let px = 1.0 / u.sun_dir.w;          // world size of one pixel at distance 1
  let tol = u.tune.w;

  let old = src[i];
  var p = old.pos;
  let was_drawn = drawable(old);

  // (a) feedback: did last frame's image use this particle, and how many
  // others landed in the same pixel?
  var visible = false;
  var crowd = 0.0;
  let pr = project(u.prev_view_proj, u.res.xy, p);
  if (was_drawn && pr.ok) {
    let key = depth_prev[pr.idx];
    if (key != 0u) {
      visible = pr.z <= key_depth(key) * (1.0 + tol);
      crowd = f32(accum_prev[pr.idx * 4u + 3u]);
    }
  }

  // (b) migrate. Hidden, back-facing and off-screen particles jump next to a
  // visible one, favouring thinly sampled pixels. Visible particles in crowded
  // pixels do the same, but only towards pixels emptier than their own. No
  // global target is needed: density simply flows downhill in screen space.
  var moved = false;
  var seeking = false;
  if (u.res.w == 0u) {
    p = random_in_bounds();
    moved = true;
  } else if (!visible) {
    let r = rand();
    if (r < u.tune.z) {
      p = random_in_bounds();            // rare restart: discover new geometry
      moved = true;
    } else {
      seeking = r < u.tune.y;
    }
  } else {
    seeking = crowd >= 3.0 && rand() < u.tune.x;
  }
  if (seeking) {
    for (var k = 0; k < 2; k++) {
      let j = pcg(rng_state + u32(k) * 7919u) % n_active;
      let donor = src[j];
      if (((donor.col >> 24u) & FLAG_VISIBLE) == 0u) { continue; }
      let dj = project(u.prev_view_proj, u.res.xy, donor.pos);
      if (!dj.ok) { continue; }
      let there = max(f32(accum_prev[dj.idx * 4u + 3u]), 1.0);
      let accept = select(1.0 / there, 1.0 - (there + 1.0) / crowd, visible);
      if (rand() < accept) {
        // land near the donor, in the emptiest of a few candidate spots, so
        // clones go straight into holes. A few reach further out to seed
        // regions that have no particles at all yet.
        let reach = select(5.0, 24.0, rand() < 0.1) * dj.z * px;
        var best = 1e9;
        for (var c = 0; c < 4; c++) {
          let cand = donor.pos + rand_unit() * (reach * (0.3 + 0.7 * rand()));
          let pc = project(u.prev_view_proj, u.res.xy, cand);
          var load = 0.5;                  // off-screen: rather not
          if (pc.ok) { load = f32(accum_prev[pc.idx * 4u + 3u]); }
          load += rand() * 0.25;
          if (load < best) { best = load; p = cand; }
        }
        moved = true;
        break;
      }
    }
  }

  // ride along with moving geometry. Falling back onto the surface (step c)
  // only follows motion along the normal, so on a moving shape the swarm
  // would slide to its trailing side. Compare the field now with the field a
  // frame ago and shift the particle to where the surface has the normal it
  // had before: exact for translation, a good guess for everything else.
  let dt = u.cam_up.w;
  if (was_drawn && !moved && dt > 0.0) {
    let hh = 0.02;
    scene_time = u.cam_pos.w - dt;
    let before = map_grad(p, hh);
    scene_time = u.cam_pos.w;
    let now = map_grad(p, hh);
    let glen = max(length(now.xyz), 1e-3);
    let n1 = now.xyz / glen;
    let curvature = (now.w - map(p)) / (hh * hh * glen);   // mean curvature from the Laplacian
    let n0 = normalize(before.xyz);
    var turn = n1 - n0;
    turn -= n1 * dot(turn, n1);
    let err = dot(turn, turn);
    if (abs(curvature) > 1.0 && err > 1e-8) {
      var slide = -turn / curvature;
      let len = length(slide);
      let reach = 2.5 * dt;                 // nothing here moves faster than this
      if (len > reach) { slide *= reach / len; }
      // the estimate assumes equal curvature in all directions, which is wrong
      // at saddles and creases: keep the move only if it really restores the normal
      let off = normalize(map_grad(p + slide, hh).xyz) - n0;
      if (dot(off, off) < 0.25 * err) { p += slide; }
    }
  }

  // sub-pixel drift keeps the swarm relaxed and supersamples over time
  var dist = max(distance(p, cam), NEAR);
  p += rand_unit() * (u.morph.w * dist * px);

  // (c) fall onto the surface: Newton steps along the distance gradient
  let h = clamp(0.5 * dist * px, 0.0008, 0.01);
  for (var it = 0; it < 2; it++) {
    let g = map_grad(p, h);
    var delta = g.xyz * (g.w / max(dot(g.xyz, g.xyz), 1e-4));
    let len = length(delta);
    if (len > 0.6) { delta *= 0.6 / len; }
    p -= delta;
  }
  let g = map_grad(p, h);
  let g2 = max(dot(g.xyz, g.xyz), 1e-4);
  dist = max(distance(p, cam), NEAR);
  let inside = all(abs(p) < vec3f(BOUND));
  let converged = inside && abs(g.w) < max(dist * px, 0.002) && g2 > 1e-3;

  // (d) shade in object space, once per particle, independent of resolution
  var flags = 0u;
  var rgb = old.col & 0x00ffffffu;
  if (converged) {
    p -= g.xyz * (g.w / g2);             // one more step, free: we have the gradient
    let n = g.xyz * inverseSqrt(g2);
    let view = normalize(cam - p);
    flags |= FLAG_SURFACE;
    if (dot(n, view) < 0.0) {
      flags |= FLAG_BACK;
    } else if (project(u.view_proj, u.res.xy, p).ok) {
      if (visible && !moved) { flags |= FLAG_VISIBLE; }
      let c = shade(p, n, view);
      let q = vec3u(clamp(c, vec3f(0.0), vec3f(1.0)) * 255.0 + 0.5);
      rgb = q.x | (q.y << 8u) | (q.z << 16u);
    }
  }
  if (!inside) { p = random_in_bounds(); }
  dst[i] = Particle(p, rgb | (flags << 24u));
}
