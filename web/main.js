// Swarm bootstrap: sets up WebGPU, loads the Rust core (swarm.wasm) and pumps
// one frame of commands per animation frame.
//
//   ?n=8000000      particle budget (default 2M desktop, 600k touch devices,
//                   up to 64M where the GPU allows buffers that large)
//   ?trace=file.bin replay a command trace recorded by `cargo run --bin trace`

import { Host } from './host.js';

const canvas = document.getElementById('view');
const hud = document.getElementById('hud');
const params = new URLSearchParams(location.search);
const MAX_PARTICLE_BYTES = 1_073_741_824;   // 64M particles: the most the core will run

function fail(message) {
  const el = document.getElementById('error');
  el.textContent = message;
  el.style.display = 'grid';
  document.getElementById('bar').style.display = 'none';
  hud.textContent = 'stopped';
  window.__error = message;
  throw new Error(message);
}

function canvasPixels() {
  // Render at device resolution, capped at about 1080p worth of pixels.
  let scale = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  scale = Math.min(scale, Math.sqrt(2_100_000 / Math.max(1, w * h)));
  return [Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale))];
}

async function main() {
  if (!navigator.gpu) fail('This browser does not support WebGPU.\nTry a current Chrome, Edge, Safari or Firefox.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) fail('No WebGPU adapter is available on this device.');
  // Particles live in one storage buffer of 16 bytes each. The default limit
  // on a buffer binding (128 MB) would cap the swarm at 8M, so ask for as much
  // as the swarm that was asked for needs, or as much as this adapter has.
  const touch = matchMedia('(pointer: coarse)').matches;
  const wanted = Math.max(1, Math.floor(Number(params.get('n'))) || (touch ? 600_000 : 2_097_152));
  // a recorded trace can hold a swarm of any size
  const bytes = params.has('trace') ? MAX_PARTICLE_BYTES : Math.min(Math.max(wanted * 16, 134_217_728), MAX_PARTICLE_BYTES);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, bytes),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(bytes, 268_435_456)),
    },
  });
  device.addEventListener('uncapturederror', (e) => {
    console.error(e.error.message);
    window.__error = e.error.message;
    hud.textContent = 'GPU error: ' + e.error.message.split('\n')[0];
  });
  device.lost.then((info) => { hud.textContent = 'GPU device lost: ' + info.message; });

  const hudText = (text) => { hud.textContent = text; };
  if (params.has('trace')) return replay(device, hudText, params.get('trace'));

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  [canvas.width, canvas.height] = canvasPixels();
  context.configure({ device, format, alphaMode: 'opaque' });
  const host = new Host(device, context, format, hudText);

  const limit = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 16);
  const particles = Math.min(wanted, limit);

  let wasm;
  try {
    const response = await fetch('swarm.wasm');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    wasm = (await WebAssembly.instantiate(await response.arrayBuffer(), {})).instance.exports;
  } catch (e) {
    fail(`Could not load swarm.wasm (${e.message}).\nBuild it first: see README.md.`);
  }
  const run = () => host.exec(new Uint32Array(wasm.memory.buffer, wasm.swarm_cmd_ptr(), wasm.swarm_cmd_len()));

  wasm.swarm_init(canvas.width, canvas.height, particles);
  run();

  new ResizeObserver(() => {
    const [w, h] = canvasPixels();
    if (w === canvas.width && h === canvas.height) return;
    canvas.width = w;
    canvas.height = h;
    wasm.swarm_resize(w, h);
    run();
  }).observe(canvas);

  bindInput(wasm);

  const frame = (ms) => {
    wasm.swarm_frame(ms / 1000);
    run();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function bindInput(wasm) {
  const pointers = new Map();
  let pinch = 0;
  const spread = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) pinch = spread();
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (pointers.size === 1) {
      wasm.swarm_pointer_drag(dx, dy);
    } else if (pointers.size === 2) {
      const d = spread();
      if (pinch > 0 && d > 0) wasm.swarm_zoom(pinch / d);
      pinch = d;
    }
  });
  const release = (e) => { pointers.delete(e.pointerId); pinch = 0; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    wasm.swarm_zoom(Math.exp(Math.max(-1, Math.min(1, e.deltaY * 0.0015))));
  }, { passive: false });

  for (const b of document.querySelectorAll('button[data-scene]')) {
    b.addEventListener('click', () => wasm.swarm_set_scene(Number(b.dataset.scene)));
  }
  document.getElementById('tour').addEventListener('click', () => wasm.swarm_toggle_auto_cycle());
  document.getElementById('debug').addEventListener('click', () => wasm.swarm_cycle_debug());
  addEventListener('keydown', (e) => {
    if (e.key >= '1' && e.key <= '5') wasm.swarm_set_scene(Number(e.key) - 1);
    else if (e.key === ' ') { e.preventDefault(); wasm.swarm_toggle_auto_cycle(); }
    else if (e.key === 'd' || e.key === 'D') wasm.swarm_cycle_debug();
  });
}

// Replay a recorded command trace: a list of [word count, words...] chunks,
// the first from init and then one per frame. Frames are rendered into an
// offscreen texture and the last one is read back, so this also works in
// headless browsers that cannot present a WebGPU canvas.
async function replay(device, hudText, url) {
  document.getElementById('bar').style.display = 'none';
  const data = new Uint32Array(await (await fetch(url)).arrayBuffer());
  const chunks = [];
  for (let i = 0; i < data.length; i += 1 + data[i]) chunks.push(data.subarray(i + 1, i + 1 + data[i]));
  // the trace has a fixed resolution: read it from the first frame's uniforms
  const width = chunks[1][4 + 40], height = chunks[1][4 + 41];
  const texture = device.createTexture({
    size: [width, height], format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const host = new Host(device, { getCurrentTexture: () => texture }, 'rgba8unorm', hudText);
  host.exec(chunks[0]);
  for (let n = 1; n < chunks.length; n++) {
    host.exec(chunks[n]);
    await device.queue.onSubmittedWorkDone();
    window.__frame = n;
  }

  const stride = Math.ceil(width * 4 / 256) * 256;
  const readback = device.createBuffer({ size: stride * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: stride }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const rows = new Uint8Array(readback.getMappedRange());
  const image = new ImageData(width, height);
  for (let y = 0; y < height; y++) image.data.set(rows.subarray(y * stride, y * stride + width * 4), y * width * 4);
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').putImageData(image, 0, 0);
  window.__png = canvas.toDataURL('image/png');
  window.__done = true;
}

main();
