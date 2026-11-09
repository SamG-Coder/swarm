//! Orbit camera. Produces the view-projection matrix the shaders expect:
//! clip.xy / clip.w is NDC and clip.w is the view-space depth.

use crate::math::{vec3, Mat4, Vec3};

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub target: Vec3,
    pub fov_y: f32,
}

pub struct Basis {
    pub pos: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fwd: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            yaw: 0.6,
            pitch: 0.30,
            radius: 6.6,
            target: vec3(0.0, -0.35, 0.0),
            fov_y: 45f32.to_radians(),
        }
    }
}

impl Camera {
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-0.20, 1.45);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.radius = (self.radius * factor).clamp(2.2, 14.0);
    }

    pub fn tan_half_fov(&self) -> f32 {
        (self.fov_y * 0.5).tan()
    }

    pub fn basis(&self) -> Basis {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let pos = self.target + vec3(sy * cp, sp, cy * cp) * self.radius;
        let fwd = (self.target - pos).normalize();
        let right = fwd.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(fwd);
        Basis {
            pos,
            right,
            up,
            fwd,
        }
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let b = self.basis();
        let t = self.tan_half_fov();
        let (sx, sy) = (1.0 / (t * aspect), 1.0 / t);
        let col = |r: f32, u: f32, f: f32| [r * sx, u * sy, f, f];
        let cx = col(b.right.x, b.up.x, b.fwd.x);
        let cy = col(b.right.y, b.up.y, b.fwd.y);
        let cz = col(b.right.z, b.up.z, b.fwd.z);
        let cw = col(-b.right.dot(b.pos), -b.up.dot(b.pos), -b.fwd.dot(b.pos));
        let mut m = [0.0; 16];
        m[0..4].copy_from_slice(&cx);
        m[4..8].copy_from_slice(&cy);
        m[8..12].copy_from_slice(&cz);
        m[12..16].copy_from_slice(&cw);
        m
    }
}
