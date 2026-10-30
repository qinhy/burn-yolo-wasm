// Direct browser WebGPU. Rust/WASM owns the embedded model and CPU reference.
export class Executor {
  static async create(weights, capacity = 100) {
    if (!navigator.gpu) throw new Error('WebGPU unavailable; use HTTPS or localhost in a supported browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No GPU adapter');
    const timestampSupported = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({ requiredFeatures: timestampSupported ? ['timestamp-query'] : [] });
    device.pushErrorScope('validation');
    const code = await (await fetch(new URL('./fused.wgsl', import.meta.url))).text();
    const module = device.createShaderModule({ code });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(errors.map(m => m.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const buffer = (size, usage) => device.createBuffer({ size, usage });
    const input = buffer(capacity * 784 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const params = buffer(weights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const output = buffer(capacity * 10 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const staging = buffer(capacity * 10 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(params, 0, weights);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [input, params, output].map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const querySet = timestampSupported ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    const queryResolve = timestampSupported ? buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC) : null;
    const queryRead = timestampSupported ? buffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) : null;
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
    const result = Object.assign(new Executor(), { device, pipeline, input, params, output, staging, bind, capacity, timestampSupported, querySet, queryResolve, queryRead, adapter: Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, adapter.info?.[key] ?? null])), busy: false, lost: null });
    device.lost.then(info => { result.lost = info.message || 'GPU device lost'; });
    device.addEventListener('uncapturederror', event => { result.lost = event.error.message; });
    return result;
  }

  encode(count, repeats, readback, timestamps = false) {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass(timestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    for (let i = 0; i < repeats; i++) pass.dispatchWorkgroups(count);
    pass.end();
    if (readback) encoder.copyBufferToBuffer(this.output, 0, this.staging, 0, count * 40);
    if (timestamps) {
      encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
      encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 16);
    }
    return encoder.finish();
  }

  async predict(pixels) {
    const count = pixels.length / 784;
    if (!Number.isInteger(count) || count < 1 || count > this.capacity || pixels.some(x => !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('Invalid normalized input batch');
    if (this.busy) throw new Error('Executor calls must be serialized');
    if (this.lost) throw new Error(this.lost);
    this.busy = true;
    let mapped = false;
    try {
      this.device.queue.writeBuffer(this.input, 0, pixels);
      this.device.queue.submit([this.encode(count, 1, true)]);
      await this.staging.mapAsync(GPUMapMode.READ, 0, count * 40);
      mapped = true;
      return new Float32Array(this.staging.getMappedRange(0, count * 40).slice(0));
    } finally {
      if (mapped) this.staging.unmap();
      this.busy = false;
    }
  }

  // GPU-resident repeated execution; host completion time, NOT GPU timestamp time.
  async resident(count, repeats = 200) {
    if (this.busy || this.lost) throw new Error(this.lost || 'Executor busy');
    if (!Number.isInteger(count) || count < 1 || count > this.capacity || !Number.isInteger(repeats) || repeats < 1 || repeats > 10000) throw new Error('Invalid benchmark dimensions');
    this.busy = true;
    try {
      const command = this.encode(count, repeats, false);
      const start = performance.now();
      this.device.queue.submit([command]);
      await this.device.queue.onSubmittedWorkDone();
      return (performance.now() - start) / repeats;
    } finally { this.busy = false; }
  }

  // Deliberately fenced diagnostics, separate from ordinary inference latency.
  async profile(pixels, repeats = 100) {
    const count = pixels.length / 784;
    if (!Number.isInteger(count) || count < 1 || count > this.capacity || pixels.some(x => !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('Invalid normalized input batch');
    if (!Number.isInteger(repeats) || repeats < 1 || repeats > 10000) throw new Error('Invalid repeat count');
    if (this.busy || this.lost) throw new Error(this.lost || 'Executor busy');
    this.busy = true;
    let mapped = false;
    let queryMapped = false;
    try {
      await this.device.queue.onSubmittedWorkDone(); // Exclude previous work.
      let start = performance.now();
      this.device.queue.writeBuffer(this.input, 0, pixels);
      const uploadWriteBufferHostMs = performance.now() - start;
      const waitStart = performance.now();
      await this.device.queue.onSubmittedWorkDone();
      const uploadCompletionWaitMs = performance.now() - waitStart;
      const uploadHostThroughCompletionMs = performance.now() - start;

      const compute = async iterations => {
        const encodeStart = performance.now();
        const command = this.encode(count, iterations, false, this.timestampSupported);
        const encodeHostMs = performance.now() - encodeStart;
        const submitStart = performance.now();
        this.device.queue.submit([command]);
        await this.device.queue.onSubmittedWorkDone();
        const submitThroughCompletionMs = performance.now() - submitStart;
        let gpuPassMs = null;
        if (this.timestampSupported) {
          await this.queryRead.mapAsync(GPUMapMode.READ, 0, 16);
          queryMapped = true;
          const ticks = new BigUint64Array(this.queryRead.getMappedRange(0, 16));
          if (ticks[1] < ticks[0]) throw new Error('Invalid GPU timestamp order');
          gpuPassMs = Number(ticks[1] - ticks[0]) / 1e6;
          this.queryRead.unmap();
          queryMapped = false;
        }
        return { encodeHostMs, submitThroughCompletionMs, gpuPassMs,
          gpuPassMsPerDispatch: gpuPassMs === null ? null : gpuPassMs / iterations };
      };
      const singleDispatch = await compute(1);
      const repeatedDispatches = await compute(repeats);

      // Compute is finished before this copy: map wait cannot include pending inference.
      start = performance.now();
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.output, 0, this.staging, 0, count * 40);
      const command = encoder.finish();
      const readbackEncodeHostMs = performance.now() - start;
      const submitStart = performance.now();
      this.device.queue.submit([command]);
      const readbackSubmitHostMs = performance.now() - submitStart;
      const mapStart = performance.now();
      await this.staging.mapAsync(GPUMapMode.READ, 0, count * 40);
      mapped = true;
      const readbackMapWaitMs = performance.now() - mapStart;
      const copyStart = performance.now();
      const logits = new Float32Array(this.staging.getMappedRange(0, count * 40).slice(0));
      this.staging.unmap();
      mapped = false;
      const readbackMappedCopyAndUnmapHostMs = performance.now() - copyStart;
      const readbackHostTotalMs = performance.now() - start;
      return { logits, metrics: { uploadWriteBufferHostMs, uploadCompletionWaitMs,
        uploadHostThroughCompletionMs, singleDispatch, repeatedDispatches,
        readbackEncodeHostMs, readbackSubmitHostMs, readbackMapWaitMs,
        readbackMappedCopyAndUnmapHostMs, readbackHostTotalMs } };
    } finally {
      if (mapped) this.staging.unmap();
      if (queryMapped) this.queryRead.unmap();
      this.busy = false;
    }
  }

  destroy() { this.device.destroy(); }
}
