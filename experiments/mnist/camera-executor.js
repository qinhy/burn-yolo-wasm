import { Executor } from './executor.js';

export class CameraExecutor {
  static async create(weights, canvas, { allowTestCapture = false } = {}) {
    const engine = await Executor.create(weights, 1);
    const device = engine.device;
    try {
      const load = async name => {
        const response = await fetch(new URL(name, import.meta.url));
        if (!response.ok) throw new Error(`Cannot load ${name}`);
        const module = device.createShaderModule({ code: await response.text() });
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter(m => m.type === 'error');
        if (errors.length) throw new Error(errors.map(m => m.message).join('\n'));
        return module;
      };
      const [preprocessModule, renderModule] = await Promise.all([load('./camera-preprocess.wgsl'), load('./camera-render.wgsl')]);
      device.pushErrorScope('validation');
      const preprocess = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: preprocessModule, entryPoint: 'main' } });
      const format = navigator.gpu.getPreferredCanvasFormat();
      const render = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: renderModule, entryPoint: 'vs' }, fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
      const context = canvas.getContext('webgpu');
      if (!context) throw new Error('WebGPU canvas unavailable');
      context.configure({ device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | (allowTestCapture ? GPUTextureUsage.COPY_SRC : 0) });
      const options = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);
      return Object.assign(new CameraExecutor(), { engine, device, preprocess, render, context, options, sampler, canvas, inFlight: 0, dropped: 0, submitted: 0, lastOptions: '', destroyed: false });
    } catch (error) { engine.destroy(); throw error; }
  }

  // Import, bind, encode and submit in the SAME task: external video textures expire.
  // No pixel/logit readback, no awaited completion in the frame callback.
  frame(video, invert) {
    if (this.destroyed) return false;
    if (this.engine.lost) throw new Error(this.engine.lost);
    if (this.inFlight >= 2) { this.dropped++; return false; }
    const width = video.videoWidth, height = video.videoHeight;
    if (!width || !height || video.readyState < 2) return false;
    const key = `${width},${height},${invert}`;
    if (key !== this.lastOptions) {
      const scale = Math.min(1, 960 / width, 720 / height);
      this.canvas.width = Math.max(1, Math.round(width * scale));
      this.canvas.height = Math.max(1, Math.round(height * scale));
      this.device.queue.writeBuffer(this.options, 0, new Float32Array([width, height, +invert, 0]));
      this.lastOptions = key;
    }
    const external = this.device.importExternalTexture({ source: video });
    const resource = buffer => ({ buffer });
    const preBind = this.device.createBindGroup({ layout: this.preprocess.getBindGroupLayout(0), entries: [external, this.sampler, resource(this.engine.input), resource(this.options)].map((resource, binding) => ({ binding, resource })) });
    const renderBind = this.device.createBindGroup({ layout: this.render.getBindGroupLayout(0), entries: [external, this.sampler, resource(this.engine.input), resource(this.engine.output), resource(this.options)].map((resource, binding) => ({ binding, resource })) });
    const encoder = this.device.createCommandEncoder();
    const pre = encoder.beginComputePass();
    pre.setPipeline(this.preprocess);
    pre.setBindGroup(0, preBind);
    pre.dispatchWorkgroups(4, 4);
    pre.end();
    const inference = encoder.beginComputePass();
    inference.setPipeline(this.engine.pipeline);
    inference.setBindGroup(0, this.engine.bind);
    inference.dispatchWorkgroups(1);
    inference.end();
    const render = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    render.setPipeline(this.render);
    render.setBindGroup(0, renderBind);
    render.draw(3);
    render.end();
    this.device.queue.submit([encoder.finish()]);
    this.submitted++;
    this.inFlight++;
    // Asynchronous backpressure only. Never build an unbounded frame queue.
    this.device.queue.onSubmittedWorkDone().then(() => { this.inFlight--; }, error => {
      this.inFlight--;
      if (!this.destroyed) this.engine.lost = String(error);
    });
    return true;
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.context.unconfigure();
    this.engine.destroy();
  }
}
