//! Swarm - a feedback-driven point-swarm renderer for implicit surfaces.
//!
//! The crate has no dependencies. On `wasm32-unknown-unknown` it exports a
//! small C ABI (below) and needs no imports; `web/host.js` calls it once per
//! frame and executes the GPU command stream it leaves behind.

#![allow(clippy::manual_is_multiple_of)] // keep `%` so older toolchains build

pub mod camera;
pub mod cmd;
pub mod math;
pub mod renderer;

use renderer::Renderer;
use std::cell::RefCell;

thread_local! {
    static APP: RefCell<Option<Renderer>> = const { RefCell::new(None) };
}

fn with<R: Default>(f: impl FnOnce(&mut Renderer) -> R) -> R {
    APP.with(|app| app.borrow_mut().as_mut().map(f).unwrap_or_default())
}

/// Create the renderer. Leaves the setup commands in the command stream.
#[no_mangle]
pub extern "C" fn swarm_init(width: u32, height: u32, max_particles: u32) {
    APP.with(|app| *app.borrow_mut() = Some(Renderer::new(width, height, max_particles)));
}

/// Pointer to the commands produced by the last init / resize / frame call.
#[no_mangle]
pub extern "C" fn swarm_cmd_ptr() -> *const u32 {
    with(|r| r.cmds.words().as_ptr() as usize) as *const u32
}

/// Number of 32-bit words at `swarm_cmd_ptr`.
#[no_mangle]
pub extern "C" fn swarm_cmd_len() -> u32 {
    with(|r| r.cmds.words().len() as u32)
}

#[no_mangle]
pub extern "C" fn swarm_resize(width: u32, height: u32) {
    with(|r| {
        r.cmds.clear();
        r.resize(width, height);
    })
}

/// Produce one frame. `time` is in seconds.
#[no_mangle]
pub extern "C" fn swarm_frame(time: f64) {
    with(|r| {
        r.cmds.clear();
        r.frame(time);
    })
}

#[no_mangle]
pub extern "C" fn swarm_pointer_drag(dx: f32, dy: f32) {
    with(|r| r.pointer_drag(dx, dy))
}

#[no_mangle]
pub extern "C" fn swarm_zoom(factor: f32) {
    with(|r| r.zoom(factor))
}

#[no_mangle]
pub extern "C" fn swarm_set_scene(scene: u32) {
    with(|r| r.set_scene(scene))
}

#[no_mangle]
pub extern "C" fn swarm_toggle_auto_cycle() {
    with(|r| r.toggle_auto_cycle())
}

#[no_mangle]
pub extern "C" fn swarm_cycle_debug() {
    with(|r| r.cycle_debug())
}
