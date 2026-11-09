//! Native test harness: runs the renderer core without a browser and writes
//! the GPU command stream it produces to a file. `web/index.html?trace=<file>`
//! replays that file through the same JS host the WASM build uses, so the Rust
//! logic and the shaders can be verified end to end without compiling to WASM.
//!
//!   cargo run --release --bin trace -- web/trace.bin [width height particles frames scene debug fps]

use std::io::Write;
use swarm::renderer::Renderer;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize, default: u32| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    let path = args.get(1).map(String::as_str).unwrap_or("web/trace.bin");
    let (width, height) = (num(2, 640), num(3, 360));
    let (particles, frames) = (num(4, 262_144), num(5, 90));
    let (scene, debug, fps) = (num(6, 0), num(7, 0), num(8, 60));

    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut chunk = |r: &mut Renderer| -> std::io::Result<()> {
        let words = r.cmds.words();
        out.write_all(&(words.len() as u32).to_le_bytes())?;
        for w in words {
            out.write_all(&w.to_le_bytes())?;
        }
        r.cmds.clear();
        Ok(())
    };

    let mut r = Renderer::new(width, height, particles);
    r.adaptive = false;
    r.debug_mode = debug;
    // "12" = start at scene 1 and morph towards scene 2 halfway through
    let (first, second) = if scene >= 10 {
        (scene / 10, Some(scene % 10))
    } else {
        (scene, None)
    };
    r.start_scene(first);
    chunk(&mut r)?;
    for f in 0..frames {
        if let (Some(next), true) = (second, f == frames / 2) {
            r.set_scene(next);
        }
        r.frame(f64::from(f) / f64::from(fps));
        chunk(&mut r)?;
    }
    println!(
        "{path}: {frames} frames, {width}x{height}, {} particles",
        r.active_particles()
    );
    Ok(())
}
