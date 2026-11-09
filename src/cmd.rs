//! The GPU command stream.
//!
//! Instead of binding to WebGPU call by call, the Rust core appends compact
//! commands to a `Vec<u32>`. The JS host decodes and executes them once per
//! frame, so there is exactly one WASM <-> JS crossing per frame and the core
//! needs no bindings crate at all. Rust allocates every object id itself,
//! which makes the stream write-only: nothing ever has to be read back.
//!
//! The same stream can be written to a file by a native build and replayed in
//! a browser (see `src/bin/trace.rs`), which is how the renderer is tested.
//!
//! Keep the opcodes in sync with `web/host.js`.

pub const OP_CREATE_BUFFER: u32 = 1; // id, size, usage
pub const OP_WRITE_BUFFER: u32 = 2; // id, offset, bytes
pub const OP_CREATE_SHADER: u32 = 3; // id, wgsl
pub const OP_CREATE_COMPUTE_PIPELINE: u32 = 4; // id, shader, entry
pub const OP_CREATE_RENDER_PIPELINE: u32 = 5; // id, shader, vs entry, fs entry
pub const OP_CREATE_BIND_GROUP: u32 = 6; // id, pipeline, group, n, n * (binding, buffer)
pub const OP_DESTROY: u32 = 7; // id
pub const OP_CLEAR_BUFFER: u32 = 8; // id
pub const OP_COMPUTE_PASS: u32 = 9; // pipeline, bind group, workgroups
pub const OP_RENDER_PASS: u32 = 10; // pipeline, bind group, vertex count (to the canvas)
pub const OP_SUBMIT: u32 = 11;
pub const OP_HUD: u32 = 12; // text

/// WebGPU `GPUBufferUsage` flags.
pub mod usage {
    pub const COPY_DST: u32 = 0x08;
    pub const UNIFORM: u32 = 0x40;
    pub const STORAGE: u32 = 0x80;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Id(pub u32);

#[derive(Default)]
pub struct CommandStream {
    words: Vec<u32>,
    next_id: u32,
}

impl CommandStream {
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn clear(&mut self) {
        self.words.clear();
    }

    fn alloc(&mut self) -> Id {
        self.next_id += 1;
        Id(self.next_id)
    }

    /// Length-prefixed bytes, zero-padded to a whole number of words.
    fn bytes(&mut self, data: &[u8]) {
        self.words.push(data.len() as u32);
        for chunk in data.chunks(4) {
            let mut w = [0u8; 4];
            w[..chunk.len()].copy_from_slice(chunk);
            self.words.push(u32::from_le_bytes(w));
        }
    }

    pub fn create_buffer(&mut self, size: u32, usage: u32) -> Id {
        let id = self.alloc();
        self.words.extend([OP_CREATE_BUFFER, id.0, size, usage]);
        id
    }

    pub fn write_buffer(&mut self, buffer: Id, offset: u32, data: &[u8]) {
        debug_assert!(
            data.len() % 4 == 0,
            "WebGPU buffer writes are 4-byte aligned"
        );
        self.words.extend([OP_WRITE_BUFFER, buffer.0, offset]);
        self.bytes(data);
    }

    pub fn create_shader(&mut self, wgsl: &str) -> Id {
        let id = self.alloc();
        self.words.extend([OP_CREATE_SHADER, id.0]);
        self.bytes(wgsl.as_bytes());
        id
    }

    pub fn create_compute_pipeline(&mut self, shader: Id, entry: &str) -> Id {
        let id = self.alloc();
        self.words
            .extend([OP_CREATE_COMPUTE_PIPELINE, id.0, shader.0]);
        self.bytes(entry.as_bytes());
        id
    }

    pub fn create_render_pipeline(&mut self, shader: Id, vs: &str, fs: &str) -> Id {
        let id = self.alloc();
        self.words
            .extend([OP_CREATE_RENDER_PIPELINE, id.0, shader.0]);
        self.bytes(vs.as_bytes());
        self.bytes(fs.as_bytes());
        id
    }

    /// Bind whole buffers to `@group(group)` of `pipeline`: `(binding, buffer)` pairs.
    pub fn create_bind_group(&mut self, pipeline: Id, group: u32, entries: &[(u32, Id)]) -> Id {
        let id = self.alloc();
        self.words.extend([
            OP_CREATE_BIND_GROUP,
            id.0,
            pipeline.0,
            group,
            entries.len() as u32,
        ]);
        for &(binding, buffer) in entries {
            self.words.extend([binding, buffer.0]);
        }
        id
    }

    pub fn destroy(&mut self, id: Id) {
        self.words.extend([OP_DESTROY, id.0]);
    }

    pub fn clear_buffer(&mut self, buffer: Id) {
        self.words.extend([OP_CLEAR_BUFFER, buffer.0]);
    }

    pub fn compute_pass(&mut self, pipeline: Id, bind_group: Id, workgroups: u32) {
        self.words
            .extend([OP_COMPUTE_PASS, pipeline.0, bind_group.0, workgroups]);
    }

    pub fn render_pass(&mut self, pipeline: Id, bind_group: Id, vertices: u32) {
        self.words
            .extend([OP_RENDER_PASS, pipeline.0, bind_group.0, vertices]);
    }

    pub fn submit(&mut self) {
        self.words.push(OP_SUBMIT);
    }

    pub fn hud(&mut self, text: &str) {
        self.words.push(OP_HUD);
        self.bytes(text.as_bytes());
    }
}
