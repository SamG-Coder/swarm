//! The renderer: owns every GPU resource (by id), the camera and the scene
//! state, and turns each frame into a handful of commands.
//!
//! Frame graph (all compute except the last pass):
//!
//!   simulate     particles read LAST frame's depth + density buffers, migrate
//!                towards under-sampled visible surface, re-project onto the
//!                SDF, shade themselves and write THIS frame's depth
//!                (atomicMax of 1/z per pixel). Only one block of particles
//!                in `PERIOD` does all of that in a given frame; the others
//!                just follow their surface (see `simulate.wgsl`)
//!   splat color  atomicAdd of colour + count for particles on the front surface
//!   resolve      fullscreen pass: average, close pinholes, composite over sky
//!
//! There are two depth buffers, swapped every frame like the particle buffers,
//! so the simulation can read the old one while it fills the new one.

use crate::camera::Camera;
use crate::cmd::{usage, CommandStream, Id};
use crate::math::{vec3, Mat4};

const COMMON: &str = include_str!("../shaders/common.wgsl");
const SCENE: &str = include_str!("../shaders/scene.wgsl");
const SIMULATE: &str = include_str!("../shaders/simulate.wgsl");
const SPLAT_COLOR: &str = include_str!("../shaders/splat_color.wgsl");
const RESOLVE: &str = include_str!("../shaders/resolve.wgsl");

pub const UNIFORM_BYTES: usize = 288;
pub const PARTICLE_BYTES: u32 = 16;
pub const WORKGROUP: u32 = 256;
/// A particle gets a full turn (feedback, migration, tracking, shading) once
/// in this many frames, and a cheap surface-following step in the others.
/// A power of two.
pub const PERIOD: u32 = 4;
// the shader gets one age per class of blocks in a vec4
const _: () = assert!(PERIOD == 4);

/// How the swarm spends one frame (see `simulate.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pace {
    /// Seconds since the previous frame, at most 0.1.
    pub dt: f32,
    /// Newton steps for a particle that is only following its surface: more
    /// when frames are far apart. 0 means they are too far apart for that and
    /// every particle takes a full turn, every frame.
    pub light_steps: u32,
    /// Seconds since the last full turn of each class of blocks
    /// (block index modulo `PERIOD`).
    pub ages: [f32; PERIOD as usize],
}
pub const SCENE_COUNT: u32 = 4;
pub const SCENE_NAMES: [&str; SCENE_COUNT as usize] =
    ["gyroid orb", "metaballs", "ribbon", "sponge"];
const MIN_PARTICLES: u32 = 32 * 1024;
/// As many as one dispatch can cover: WebGPU guarantees 65535 workgroups.
const MAX_PARTICLES: u32 = 65_535 * WORKGROUP;

/// Tunables of the feedback loop (see `simulate.wgsl`). The rates are per
/// frame; the shader gets them as the chance per full turn.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// Chance per frame that a visible particle in a crowded pixel looks for
    /// an emptier one.
    pub turnover: f32,
    /// Chance per frame that a hidden particle migrates next to a visible one.
    pub respawn_rate: f32,
    /// Chance per frame that a hidden particle restarts at a random position.
    pub restart_rate: f32,
    /// Relative depth tolerance for "on the front-most surface".
    pub depth_tolerance: f32,
    /// Random drift per frame, in pixels.
    pub jitter_px: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            turnover: 0.25,
            respawn_rate: 0.35,
            restart_rate: 0.004,
            depth_tolerance: 0.01,
            jitter_px: 0.5,
        }
    }
}

struct SizeDependent {
    depth: [Id; 2],
    accum: Id,
    simulate: [Id; 2],
    splat_color: [Id; 2],
    resolve: [Id; 2],
}

pub struct Renderer {
    pub cmds: CommandStream,
    pub camera: Camera,
    pub tuning: Tuning,
    /// Scale the particle count to hold the frame rate.
    pub adaptive: bool,
    pub debug_mode: u32,

    width: u32,
    height: u32,
    max_particles: u32,
    active: u32,

    uniforms: Id,
    particles: [Id; 2],
    p_simulate: Id,
    p_splat_color: Id,
    p_resolve: Id,
    sized: SizeDependent,

    frame: u32,
    flip: usize,
    prev_view_proj: Option<Mat4>,

    time: f64,
    last_time: Option<f64>,
    /// When each class of blocks last had a full turn.
    turn_time: [f64; PERIOD as usize],
    frame_ms: f32,
    idle: f32,

    scene_a: u32,
    scene_b: u32,
    blend: f32,
    hold: f32,
    auto_cycle: bool,
}

fn round_particles(n: u32) -> u32 {
    (n.clamp(MIN_PARTICLES, MAX_PARTICLES) / WORKGROUP) * WORKGROUP
}

impl Renderer {
    pub fn new(width: u32, height: u32, max_particles: u32) -> Renderer {
        let mut cmds = CommandStream::default();
        let max_particles = round_particles(max_particles);
        let (width, height) = (width.max(1), height.max(1));

        let uniforms = cmds.create_buffer(UNIFORM_BYTES as u32, usage::UNIFORM | usage::COPY_DST);
        let particle_bytes = max_particles * PARTICLE_BYTES;
        let particles = [
            cmds.create_buffer(particle_bytes, usage::STORAGE | usage::COPY_DST),
            cmds.create_buffer(particle_bytes, usage::STORAGE | usage::COPY_DST),
        ];

        let period = format!("const PERIOD: u32 = {PERIOD}u;\n");
        let s_simulate = cmds.create_shader(&[&period, COMMON, SCENE, SIMULATE].concat());

        let s_splat_color = cmds.create_shader(&[COMMON, SPLAT_COLOR].concat());
        let s_resolve = cmds.create_shader(&[COMMON, SCENE, RESOLVE].concat());
        let p_simulate = cmds.create_compute_pipeline(s_simulate, "main");

        let p_splat_color = cmds.create_compute_pipeline(s_splat_color, "main");
        let p_resolve = cmds.create_render_pipeline(s_resolve, "vs", "fs");

        let mut r = Renderer {
            cmds,
            camera: Camera::default(),
            tuning: Tuning::default(),
            adaptive: true,
            debug_mode: 0,
            width,
            height,
            max_particles,
            active: max_particles,
            uniforms,
            particles,
            p_simulate,
            p_splat_color,
            p_resolve,
            sized: SizeDependent {
                depth: [Id(0); 2],
                accum: Id(0),
                simulate: [Id(0); 2],
                splat_color: [Id(0); 2],
                resolve: [Id(0); 2],
            },
            frame: 0,
            flip: 0,
            prev_view_proj: None,
            time: 0.0,
            last_time: None,
            turn_time: [0.0; PERIOD as usize],
            frame_ms: 16.7,
            idle: 1e9,
            scene_a: 0,
            scene_b: 0,
            blend: 0.0,
            hold: 0.0,
            auto_cycle: true,
        };
        r.sized = r.create_size_dependent();
        r
    }

    fn create_size_dependent(&mut self) -> SizeDependent {
        let pixels = self.width * self.height;
        let store = usage::STORAGE | usage::COPY_DST;
        let depth = [
            self.cmds.create_buffer(pixels * 4, store),
            self.cmds.create_buffer(pixels * 4, store),
        ];
        let accum = self.cmds.create_buffer(pixels * 16, store);
        let (u, p) = (self.uniforms, self.particles);
        let (ps, pc, pr) = (self.p_simulate, self.p_splat_color, self.p_resolve);
        let c = &mut self.cmds;
        // index = which particle and depth buffers simulate reads ("src");
        // it writes the other of each pair
        let simulate = [0, 1].map(|i| {
            let entries = [
                (0, u),
                (1, p[i]),
                (2, p[1 - i]),
                (3, depth[i]),
                (4, accum),
                (5, depth[1 - i]),
            ];
            c.create_bind_group(ps, 0, &entries)
        });
        // index = which particle and depth buffers hold the current frame
        let splat_color = [0, 1]
            .map(|i| c.create_bind_group(pc, 0, &[(0, u), (1, p[i]), (2, depth[i]), (3, accum)]));
        let resolve =
            [0, 1].map(|i| c.create_bind_group(pr, 0, &[(0, u), (1, depth[i]), (2, accum)]));
        SizeDependent {
            depth,
            accum,
            simulate,
            splat_color,
            resolve,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn active_particles(&self) -> u32 {
        self.active
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return;
        }
        let s = &self.sized;
        let mut stale = vec![s.accum];
        stale.extend(
            s.depth
                .iter()
                .chain(&s.simulate)
                .chain(&s.splat_color)
                .chain(&s.resolve),
        );
        for id in stale {
            self.cmds.destroy(id);
        }
        self.width = width;
        self.height = height;
        self.sized = self.create_size_dependent();
        // last frame's buffers are gone, so there is nothing to reproject into
        self.prev_view_proj = None;
    }

    // --- input ---------------------------------------------------------------

    pub fn pointer_drag(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
        self.idle = 0.0;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.camera.zoom(factor);
        self.idle = 0.0;
    }

    /// Show a scene immediately, without a morph, and stop the automatic tour.
    pub fn start_scene(&mut self, scene: u32) {
        self.auto_cycle = false;
        self.scene_a = scene % SCENE_COUNT;
        self.scene_b = self.scene_a;
        self.blend = 0.0;
    }

    /// Morph to a scene and stop the automatic tour.
    pub fn set_scene(&mut self, scene: u32) {
        self.auto_cycle = false;
        self.morph_to(scene % SCENE_COUNT);
    }

    pub fn toggle_auto_cycle(&mut self) {
        self.auto_cycle = !self.auto_cycle;
        self.hold = 0.0;
    }

    pub fn cycle_debug(&mut self) {
        self.debug_mode = (self.debug_mode + 1) % 3;
    }

    fn morph_to(&mut self, scene: u32) {
        if self.blend > 0.5 {
            self.scene_a = self.scene_b;
        }
        self.blend = 0.0;
        self.scene_b = scene;
        self.hold = 0.0;
    }

    // --- per frame -----------------------------------------------------------

    /// Moves time on and decides how the swarm spends this frame.
    fn advance(&mut self, time: f64) -> Pace {
        let elapsed = match self.last_time {
            Some(last) => ((time - last) as f32).max(0.0),
            None => 0.0,
        };
        let dt = elapsed.min(0.1);
        self.last_time = Some(time);
        self.time = time;
        if dt > 0.0 {
            self.frame_ms += (dt * 1000.0 - self.frame_ms) * 0.08;
        }

        // slow orbit whenever the user leaves the camera alone
        self.idle += dt;
        if self.idle > 3.0 {
            self.camera.yaw += dt * 0.12;
        }

        // scene tour: hold, then cross-fade the distance fields
        if self.scene_a != self.scene_b {
            self.blend += dt / 3.0;
            if self.blend >= 1.0 {
                self.scene_a = self.scene_b;
                self.blend = 0.0;
            }
        } else if self.auto_cycle {
            self.hold += dt;
            if self.hold > 9.0 {
                self.morph_to((self.scene_a + 1) % SCENE_COUNT);
            }
        }

        // spend exactly as many particles as the frame budget allows
        if self.adaptive && self.frame > 30 && dt > 0.0 {
            let factor = if self.frame_ms > 21.0 {
                (18.0 / self.frame_ms).clamp(0.9, 1.0)
            } else if self.frame_ms < 17.5 {
                1.01
            } else {
                1.0
            };
            let wanted = (self.active as f32 * factor) as u32;
            self.active = round_particles(wanted).min(self.max_particles);
        }

        // Following a surface with a single step only works while it moves a
        // little per frame. Below about 45 fps take two steps, and below 20
        // (or after a hitch) fall back to a full turn for everyone.
        let light_steps = if self.frame == 0 || elapsed > 0.05 {
            0
        } else if elapsed > 0.0225 {
            2
        } else {
            1
        };
        let mut ages = [0.0; PERIOD as usize];
        for (class, age) in ages.iter_mut().enumerate() {
            if self.frame > 0 {
                *age = ((time - self.turn_time[class]) as f32).clamp(0.0, 0.1 * PERIOD as f32);
            }
            if light_steps == 0 || (class as u32 + self.frame) % PERIOD == 0 {
                self.turn_time[class] = time;
            }
        }
        Pace {
            dt,
            light_steps,
            ages,
        }
    }

    pub fn uniform_bytes(&self, view_proj: &Mat4, prev: &Mat4, pace: &Pace) -> [u8; UNIFORM_BYTES] {
        let cam = &self.camera;
        let b = cam.basis();
        let tan = cam.tan_half_fov();
        let focal_px = self.height as f32 * 0.5 / tan;
        let sun = vec3(0.55, 0.62, 0.38).normalize();
        let smooth = self.blend * self.blend * (3.0 - 2.0 * self.blend);
        let t = &self.tuning;
        // a particle only rolls these dice on its full turn, which comes
        // round every `PERIOD` frames unless everyone has one every frame
        let turns = if pace.light_steps == 0 {
            1
        } else {
            PERIOD as i32
        };
        let per_turn = |rate: f32| 1.0 - (1.0 - rate.clamp(0.0, 1.0)).powi(turns);
        // the very first frame seeds every particle, not just the active ones
        let active = if self.frame == 0 {
            self.max_particles
        } else {
            self.active
        };

        let mut out = [0u8; UNIFORM_BYTES];
        let mut at = 0;
        let mut put = |bytes: [u8; 4]| {
            out[at..at + 4].copy_from_slice(&bytes);
            at += 4;
        };
        for v in view_proj.iter().chain(prev.iter()) {
            put(v.to_le_bytes());
        }
        for v in [
            b.pos.x,
            b.pos.y,
            b.pos.z,
            self.time as f32,
            sun.x,
            sun.y,
            sun.z,
            focal_px,
        ] {
            put(v.to_le_bytes());
        }
        for v in [self.width, self.height, active, self.frame] {
            put(v.to_le_bytes());
        }
        let floats = [
            [
                self.scene_a as f32,
                self.scene_b as f32,
                smooth,
                t.jitter_px,
            ],
            [b.right.x, b.right.y, b.right.z, tan],
            [b.up.x, b.up.y, b.up.z, pace.dt],
            [b.fwd.x, b.fwd.y, b.fwd.z, self.debug_mode as f32],
            [
                per_turn(t.turnover),
                per_turn(t.respawn_rate),
                per_turn(t.restart_rate),
                t.depth_tolerance,
            ],
            pace.ages,
            [pace.light_steps as f32, 0.0, 0.0, 0.0],
        ];
        for v in floats.into_iter().flatten() {
            put(v.to_le_bytes());
        }
        debug_assert_eq!(at, UNIFORM_BYTES);
        out
    }

    /// Emit the commands for one frame. `time` is in seconds.
    pub fn frame(&mut self, time: f64) {
        let pace = self.advance(time);

        let aspect = self.width as f32 / self.height as f32;
        let view_proj = self.camera.view_proj(aspect);
        let prev = self.prev_view_proj.unwrap_or(view_proj);
        let uniforms = self.uniform_bytes(&view_proj, &prev, &pace);
        let count = if self.frame == 0 {
            self.max_particles
        } else {
            self.active
        };
        let groups = count.div_ceil(WORKGROUP);
        let (src, dst) = (self.flip, 1 - self.flip);

        let s = &self.sized;
        let c = &mut self.cmds;
        c.write_buffer(self.uniforms, 0, &uniforms);
        c.clear_buffer(s.depth[dst]);
        c.compute_pass(self.p_simulate, s.simulate[src], groups);
        // simulate has read last frame's counts; now they can go
        c.clear_buffer(s.accum);
        c.compute_pass(self.p_splat_color, s.splat_color[dst], groups);
        c.render_pass(self.p_resolve, s.resolve[dst], 3);
        c.submit();

        if self.frame % 15 == 0 {
            let scene = if self.scene_a == self.scene_b {
                SCENE_NAMES[self.scene_a as usize].to_string()
            } else {
                format!(
                    "{} > {}",
                    SCENE_NAMES[self.scene_a as usize], SCENE_NAMES[self.scene_b as usize]
                )
            };
            let hud = format!(
                "{} points  |  {:.1} ms  |  {}x{}  |  {}",
                group_digits(self.active),
                self.frame_ms,
                self.width,
                self.height,
                scene
            );
            self.cmds.hud(&hud);
        }

        self.prev_view_proj = Some(view_proj);
        self.flip = dst;
        self.frame += 1;
    }
}

fn group_digits(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::*;
    use crate::math::transform_point;

    #[test]
    fn target_projects_to_screen_centre_with_view_depth_in_w() {
        let cam = Camera::default();
        let clip = transform_point(&cam.view_proj(16.0 / 9.0), cam.target);
        assert!((clip[0] / clip[3]).abs() < 1e-5 && (clip[1] / clip[3]).abs() < 1e-5);
        assert!((clip[3] - cam.radius).abs() < 1e-4);
    }

    #[test]
    fn points_right_and_up_of_target_land_right_and_up() {
        let cam = Camera::default();
        let b = cam.basis();
        let m = cam.view_proj(1.0);
        let r = transform_point(&m, cam.target + b.right);
        let u = transform_point(&m, cam.target + b.up);
        assert!(r[0] > 0.0 && r[1].abs() < 1e-5);
        assert!(u[1] > 0.0 && u[0].abs() < 1e-5);
    }

    #[test]
    fn uniform_block_matches_wgsl_layout() {
        let r = Renderer::new(640, 360, 100_000);
        let m = r.camera.view_proj(640.0 / 360.0);
        let mut pace = Pace {
            dt: 0.25,
            light_steps: 2,
            ages: [0.1, 0.2, 0.3, 0.4],
        };
        let bytes = r.uniform_bytes(&m, &m, &pace);
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert_eq!((u32_at(160), u32_at(164)), (640, 360)); // res
        assert_eq!(u32_at(168), r.max_particles);
        assert!((f32_at(204) - r.camera.tan_half_fov()).abs() < 1e-6); // cam_right.w
        assert_eq!(f32_at(220), 0.25); // cam_up.w = dt
        let turnover = 1.0 - (1.0 - r.tuning.turnover).powi(PERIOD as i32);
        assert!((f32_at(240) - turnover).abs() < 1e-6); // tune.x, per full turn
        assert!((f32_at(252) - r.tuning.depth_tolerance).abs() < 1e-6); // tune.w
        assert_eq!((f32_at(256), f32_at(268)), (0.1, 0.4)); // ages
        assert_eq!(f32_at(272), 2.0); // pace.x = light steps

        // with a full turn every frame the rates are per frame again
        pace.light_steps = 0;
        let bytes = r.uniform_bytes(&m, &m, &pace);
        let turnover = f32::from_le_bytes(bytes[240..244].try_into().unwrap());
        assert!((turnover - r.tuning.turnover).abs() < 1e-6);
    }

    /// Runs `frames` frames `step` seconds apart and returns the last pace.
    fn paced(r: &mut Renderer, start: f64, step: f64, frames: u32) -> (f64, Pace) {
        let mut time = start;
        let mut pace = None;
        for _ in 0..frames {
            time += step;
            pace = Some(r.advance(time));
            r.frame += 1;
        }
        (time, pace.unwrap())
    }

    #[test]
    fn swarm_takes_turns_at_speed_and_stops_when_frames_are_slow() {
        assert!(PERIOD.is_power_of_two());
        let mut r = Renderer::new(64, 64, 65_536);
        // the first frame seeds everything: a full turn for everyone
        assert_eq!(r.advance(0.0).light_steps, 0);
        r.frame += 1;

        // 60 fps: one step, and the class due now last had a turn PERIOD frames ago
        let (time, pace) = paced(&mut r, 0.0, 1.0 / 60.0, 3 * PERIOD);
        assert_eq!(pace.light_steps, 1);
        let due = ((PERIOD - (r.frame - 1) % PERIOD) % PERIOD) as usize;
        assert!((pace.ages[due] - PERIOD as f32 / 60.0).abs() < 1e-5);

        // 30 fps: two steps
        let (time, pace) = paced(&mut r, time, 1.0 / 30.0, 2 * PERIOD);
        assert_eq!(pace.light_steps, 2);

        // 10 fps: everyone has a full turn every frame, one frame apart
        let (time, pace) = paced(&mut r, time, 0.1, 3);
        assert_eq!(pace.light_steps, 0);
        assert!(pace.ages.iter().all(|age| (age - 0.1).abs() < 1e-5));

        // back to speed after a hitch
        let (_, pace) = paced(&mut r, time, 1.0 / 60.0, 2);
        assert_eq!(pace.light_steps, 1);
    }

    #[test]
    fn particle_counts_are_whole_workgroups() {
        let r = Renderer::new(64, 64, 100_001);
        assert_eq!(r.active_particles() % WORKGROUP, 0);
        assert!(r.active_particles() <= 100_001);
    }

    #[test]
    fn frame_emits_passes_in_order_and_ping_pongs() {
        let mut r = Renderer::new(64, 64, 65_536);
        r.cmds.clear();
        r.frame(0.0);
        let w = r.cmds.words().to_vec();
        assert_eq!(w[0], OP_WRITE_BUFFER);
        let after_uniforms = 4 + UNIFORM_BYTES / 4;
        let ops: Vec<u32> = [0, 2, 6, 8, 12, 16]
            .iter()
            .map(|o| w[after_uniforms + o])
            .collect();
        assert_eq!(
            ops,
            [
                OP_CLEAR_BUFFER,
                OP_COMPUTE_PASS,
                OP_CLEAR_BUFFER,
                OP_COMPUTE_PASS,
                OP_RENDER_PASS,
                OP_SUBMIT
            ]
        );
        // [cleared depth, simulate group, colour group, resolve group]
        let picks = |w: &[u32]| [1, 4, 10, 14].map(|o| w[after_uniforms + o]);
        let first = picks(&w);
        r.cmds.clear();
        r.frame(1.0 / 60.0);
        let second = picks(r.cmds.words());
        for (a, b) in first.iter().zip(&second) {
            assert_ne!(a, b);
        }
        r.cmds.clear();
        r.frame(2.0 / 60.0);
        assert_eq!(picks(r.cmds.words()), first);
    }

    #[test]
    fn the_largest_swarm_fits_one_dispatch_and_one_buffer() {
        let r = Renderer::new(64, 64, u32::MAX);
        assert_eq!(r.active_particles(), MAX_PARTICLES);
        assert!(MAX_PARTICLES / WORKGROUP <= 65_535);
        assert!(MAX_PARTICLES.checked_mul(PARTICLE_BYTES).is_some());
    }

    #[test]
    fn resize_replaces_size_dependent_resources() {
        let mut r = Renderer::new(64, 64, 65_536);
        r.cmds.clear();
        r.resize(64, 64);
        assert!(r.cmds.words().is_empty());
        r.resize(128, 96);
        let w = r.cmds.words();
        assert_eq!(
            w.iter()
                .step_by(2)
                .take(9)
                .filter(|&&op| op == OP_DESTROY)
                .count(),
            9
        );
        for at in [18, 22] {
            assert_eq!(
                &w[at..at + 4],
                &[
                    OP_CREATE_BUFFER,
                    w[at + 1],
                    128 * 96 * 4,
                    usage::STORAGE | usage::COPY_DST
                ]
            );
        }
    }

    #[test]
    fn digits_are_grouped() {
        assert_eq!(group_digits(1_048_576), "1,048,576");
        assert_eq!(group_digits(999), "999");
    }
}
