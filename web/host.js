// Swarm host: executes the GPU command stream produced by the Rust core.
// Keep the opcodes in sync with src/cmd.rs.

const OP = {
  CREATE_BUFFER: 1, WRITE_BUFFER: 2, CREATE_SHADER: 3, CREATE_COMPUTE_PIPELINE: 4,
  CREATE_RENDER_PIPELINE: 5, CREATE_BIND_GROUP: 6, DESTROY: 7, CLEAR_BUFFER: 8,
  COMPUTE_PASS: 9, RENDER_PASS: 10, SUBMIT: 11, HUD: 12, COMPUTE_PASS_ROWS: 13,
};

export class Host {
  /** `context` is a configured GPUCanvasContext, or anything with getCurrentTexture(). */
  constructor(device, context, format, onHud) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.onHud = onHud;
    this.objects = new Map();
    this.encoder = null;
    this.decoder = new TextDecoder();
  }

  /** Execute a stream of commands: a Uint32Array view (may alias WASM memory). */
  exec(words) {
    const { device, objects } = this;
    const bytes = new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
    let i = 0;
    const next = () => words[i++];
    const get = () => objects.get(next());
    const blob = () => {
      const len = next();
      const view = bytes.subarray(i * 4, i * 4 + len);
      i += (len + 3) >> 2;
      return view;
    };
    const text = () => this.decoder.decode(blob());
    const encoder = () => (this.encoder ??= device.createCommandEncoder());

    while (i < words.length) {
      const op = next();
      switch (op) {
        case OP.CREATE_BUFFER: {
          const id = next(), size = next(), usage = next();
          objects.set(id, device.createBuffer({ size, usage }));
          break;
        }
        case OP.WRITE_BUFFER: {
          const buffer = get(), offset = next(), data = blob();
          device.queue.writeBuffer(buffer, offset, data, 0, data.length);
          break;
        }
        case OP.CREATE_SHADER: {
          const id = next(), code = text();
          const module = device.createShaderModule({ code });
          module.getCompilationInfo().then((info) => {
            for (const m of info.messages) {
              if (m.type === 'error') console.error(`shader ${id}:${m.lineNum}:${m.linePos} ${m.message}`);
            }
          });
          objects.set(id, module);
          break;
        }
        case OP.CREATE_COMPUTE_PIPELINE: {
          const id = next(), module = get(), entryPoint = text();
          objects.set(id, device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } }));
          break;
        }
        case OP.CREATE_RENDER_PIPELINE: {
          const id = next(), module = get(), vs = text(), fs = text();
          objects.set(id, device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: vs },
            fragment: { module, entryPoint: fs, targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list' },
          }));
          break;
        }
        case OP.CREATE_BIND_GROUP: {
          const id = next(), pipeline = get(), group = next(), count = next();
          const entries = [];
          for (let k = 0; k < count; k++) {
            const binding = next();
            entries.push({ binding, resource: { buffer: get() } });
          }
          objects.set(id, device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries }));
          break;
        }
        case OP.DESTROY: {
          const id = next();
          objects.get(id)?.destroy?.();
          objects.delete(id);
          break;
        }
        case OP.CLEAR_BUFFER:
          encoder().clearBuffer(get());
          break;
        case OP.COMPUTE_PASS:
        case OP.COMPUTE_PASS_ROWS: {
          const pipeline = get(), bindGroup = get(), x = next(), y = op === OP.COMPUTE_PASS_ROWS ? next() : 1;
          const pass = encoder().beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(x, y);
          pass.end();
          break;
        }
        case OP.RENDER_PASS: {
          const pipeline = get(), bindGroup = get(), vertices = next();
          const pass = encoder().beginRenderPass({
            colorAttachments: [{
              view: this.context.getCurrentTexture().createView(),
              loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
          });
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.draw(vertices);
          pass.end();
          break;
        }
        case OP.SUBMIT:
          if (this.encoder) {
            device.queue.submit([this.encoder.finish()]);
            this.encoder = null;
          }
          break;
        case OP.HUD:
          this.onHud(text());
          break;
        default:
          throw new Error(`swarm: unknown opcode ${op} at word ${i - 1}`);
      }
    }
  }
}
