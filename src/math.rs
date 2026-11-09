//! Tiny vector math: just what an orbit camera needs.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        self * (1.0 / self.length())
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        vec3(self.x * s, self.y * s, self.z * s)
    }
}

/// Column-major 4x4 matrix, laid out exactly like WGSL's `mat4x4f`.
pub type Mat4 = [f32; 16];

/// Multiply a point by a column-major matrix, returning clip-space xyzw.
pub fn transform_point(m: &Mat4, p: Vec3) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (row, o) in out.iter_mut().enumerate() {
        *o = m[row] * p.x + m[4 + row] * p.y + m[8 + row] * p.z + m[12 + row];
    }
    out
}
