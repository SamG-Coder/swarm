// Pass 4 - resolve. Turns the splat buffers into the final image: averages the
// samples in each pixel, closes pinholes from neighbours at matching depth,
// and composites over the sky.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> depth: array<u32>;
@group(0) @binding(2) var<storage, read> accum: array<u32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn heat(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  return clamp(vec3f(1.5 * x, 1.6 * x * (1.4 - x), 1.2 * (1.0 - x) * (0.25 + x)) + 0.02, vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let res = vec2i(u.res.xy);
  let pix = clamp(vec2i(frag.xy), vec2i(0), res - 1);
  let tol = u.tune.w * 3.0;
  let debug = u32(u.cam_fwd.w);

  // background
  let ndc = vec2f((f32(pix.x) + 0.5) / f32(res.x) * 2.0 - 1.0, 1.0 - (f32(pix.y) + 0.5) / f32(res.y) * 2.0);
  let tanf = u.cam_right.w;
  let dir = normalize(u.cam_fwd.xyz + u.cam_right.xyz * (ndc.x * tanf * f32(res.x) / f32(res.y)) + u.cam_up.xyz * (ndc.y * tanf));
  let back = pow(clamp(sky(dir), vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

  let own_idx = u32(pix.y * res.x + pix.x);
  let own_key = depth[own_idx];
  let own_count = accum[own_idx * 4u + 3u];

  // reference depth: this pixel's surface, or the nearest one around it
  var zref = 1e30;
  if (own_key != 0u) {
    zref = key_depth(own_key);
  } else {
    for (var dy = -2; dy <= 2; dy++) {
      for (var dx = -2; dx <= 2; dx++) {
        let q = pix + vec2i(dx, dy);
        if (q.x < 0 || q.y < 0 || q.x >= res.x || q.y >= res.y) { continue; }
        let key = depth[u32(q.y * res.x + q.x)];
        if (key != 0u) { zref = min(zref, key_depth(key)); }
      }
    }
  }

  // gather coloured neighbours: on the same surface (3x3 and 5x5), and on a
  // surface clearly in front of this pixel (3x3)
  var same3 = vec4f(0.0);   // rgb * weight, weight
  var same5 = vec4f(0.0);
  var front = vec4f(0.0);
  var n5 = 0.0;
  var n_front = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let q = pix + vec2i(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= res.x || q.y >= res.y) { continue; }
      let idx = u32(q.y * res.x + q.x);
      let cnt = f32(accum[idx * 4u + 3u]);
      if (cnt == 0.0) { continue; }
      let z = key_depth(depth[idx]);
      let c = vec3f(f32(accum[idx * 4u]), f32(accum[idx * 4u + 1u]), f32(accum[idx * 4u + 2u])) / (cnt * 255.0);
      let w = exp(-0.45 * f32(dx * dx + dy * dy));
      let inner = abs(dx) <= 1 && abs(dy) <= 1;
      if (abs(z - zref) <= zref * tol) {
        same5 += vec4f(c * w, w);
        n5 += 1.0;
        if (inner) { same3 += vec4f(c * w, w); }
      } else if (inner && z < zref) {
        front += vec4f(c * w, w);
        n_front += 1.0;
      }
    }
  }

  var color = back;
  var z_out = 1e30;
  if (own_count > 0u) {
    if (n_front >= 5.0) {
      // a far surface peeking through a pinhole in a nearer one
      color = front.rgb / front.w;
    } else {
      // a real sample: keep it, lightly filtered with same-surface neighbours
      let own = vec3f(f32(accum[own_idx * 4u]), f32(accum[own_idx * 4u + 1u]), f32(accum[own_idx * 4u + 2u])) / (f32(own_count) * 255.0);
      color = (own * 2.0 + same3.rgb) / (2.0 + same3.w);
    }
    z_out = zref;
  } else if (own_key != 0u) {
    // covered by a depth footprint but not sampled: borrow from the surface
    if (same5.w > 0.0) { color = same5.rgb / same5.w; }
    z_out = zref;
  } else if (n5 >= 3.0) {
    // empty pixel hugging a surface: fade out towards the silhouette
    color = mix(back, same5.rgb / same5.w, smoothstep(2.0, 9.0, n5));
    z_out = zref;
  }

  if (debug == 1u) {
    color = heat(f32(own_count) / 6.0);
    if (own_count == 0u) { color = vec3f(0.0); }
  } else if (debug == 2u) {
    color = vec3f(select(0.0, clamp(1.6 - z_out * 0.16, 0.0, 1.0), z_out < 1e20));
  } else {
    let v = (vec2f(pix) + 0.5) / vec2f(res) - 0.5;
    color *= 1.0 - 0.55 * dot(v, v);
    color += (f32(pcg(own_idx ^ (u.res.w * 9781u)) & 255u) / 255.0 - 0.5) / 255.0;
  }
  return vec4f(color, 1.0);
}
