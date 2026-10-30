# Three-layer MNIST execution experiment

Independent of YOLO/Burn exports. Runs in a browser, or can be served as the frontend of the existing Tauri shell. GPU access uses the browser's WebGPU API directly in `executor.js`; Rust/WASM embeds the weights and implements the CPU reference. This is a model-specific executor, not a replacement Burn backend.

Architecture (exactly three trainable layers, 17,898 parameters):

```
28×28 grayscale / 255
  → reshape/reorder into 16 nonoverlapping 7×7 patches
  → shared Linear(49,16), ReLU
  → flatten ordered tokens to 256 features
  → Linear(256,64), ReLU
  → Linear(64,10) logits
```

This uses ViT-style patch tokens, without attention or convolution. Flattening preserves patch position. Each linear includes a bias. No softmax is needed for argmax classification.

## Train and run

From the repository root:

```sh
python -m venv experiments/mnist/.venv
# Activate that environment, then:
pip install torch numpy
python experiments/mnist/train.py --epochs 15
cargo test --manifest-path experiments/mnist/Cargo.toml
wasm-pack build experiments/mnist --target web --release --out-dir pkg
python -m http.server 5174 --directory experiments/mnist
```

Open http://localhost:5174 and click **Validate and benchmark**. The separate page needs neither Node nor the YOLO export. On Windows the same commands work with installed Python/Rust/wasm-pack. The wasm32 target must be installed. Training uses CPU PyTorch, a seed of 42, all 60,000 training images and evaluates all 10,000 test images after each epoch. No test samples participate in gradient updates. The report records every epoch rather than selecting weights using the test set.

Included artifacts are the trained raw FP32 weights, training report, and 100 held-out pixel/logit fixtures. Rebuild WASM after retraining. The Rust tests compare all fixture logits to PyTorch. The browser checks CPU/PyTorch error ≤0.0002, GPU/CPU error ≤0.001 and matching CPU/GPU predictions before benchmarking. These bounds are numerical parity checks, not accuracy targets.

## Executor design and measurements

A 64-thread workgroup handles one image. It computes patch projection, synchronizes, computes the hidden layer, synchronizes, then computes the head. Workgroup barriers are sufficient because an image never spans workgroups. Intermediate tokens and hidden values use 1,280 bytes of workgroup memory. There is one compute dispatch per batch, three storage bindings, no intermediate GPU allocations or readbacks, no framework graph interpreter, and no softmax. Pipelines, weights, buffers and bind groups persist across calls. Calls must be serialized. Capacity defaults to 100 images.

The benchmark warms up 20 times, then reports median/p95 over 100 calls for single-image and 100-image batch inference. GPU end-to-end timing includes JS input upload, command encoding/submission, execution and mapped readback. CPU timing includes the JS/WASM boundary and output copy. It also reports 10 GPU-resident measurements, each amortized over 200 dispatches, using the last uploaded batch. Resident timing excludes command encoding, upload and readback but includes queue submission and host-observed completion. These are wall-clock measurements, **not isolated GPU kernel timestamps**. A separate CPU measurement amortizes 100 calls to reduce timer quantization for this small model. The JSON contains browser and actual selected adapter information.

### GPU timestamps and stage diagnostics (report schema version 2)

The page now reports `timestampQueries.supported` and `stageDiagnostics` for one image and the 100-image batch. The optional `timestamp-query` device feature is requested only when advertised. Unsupported devices still run all host measurements, with GPU timestamp fields set to `null`.

Each diagnostic case warms up once, then records 20 samples. Inputs are uploaded once per sample; there is a completion fence before compute. A compute pass is timestamped at its beginning and end, resolved to two unsigned 64-bit nanosecond values, and converted to milliseconds. Each sample measures both one dispatch and a pass containing 100 dispatches. Repeated-pass timing divided by 100 reduces quantization, but measures a different GPU workload. Zero single-pass durations may reflect timestamp quantization, not free execution.

| JSON field under `stages` | What it measures |
| --- | --- |
| `uploadWriteBufferHostMs` | CPU time in `queue.writeBuffer` |
| `uploadCompletionWaitMs` | Host wait for queued upload completion |
| `uploadHostThroughCompletionMs` | Write call plus completion wait |
| `singleDispatch.gpuPassMs` | GPU timestamp duration of a compute pass with one dispatch |
| `repeatedDispatches.gpuPassMsPerDispatch` | GPU duration of the 100-dispatch pass divided by 100 |
| `*.encodeHostMs` | CPU command encoding time |
| `*.submitThroughCompletionMs` | Host submission through queue completion, including timestamp resolution/copy if enabled |
| `readbackEncodeHostMs` | CPU encoding of output-buffer copy |
| `readbackSubmitHostMs` | CPU submission call |
| `readbackMapWaitMs` | Output copy and buffer mapping wait, with inference already completed |
| `readbackMappedCopyAndUnmapHostMs` | Copy mapped logits into JS memory and unmap |
| `readbackHostTotalMs` | Entire isolated readback path |

All stage fields report median and p95 milliseconds, or `null` if unavailable. Query-buffer mapping happens after compute host timing and before the isolated output readback measurement. Diagnostic logits are checked against the reference outputs as well.

These deliberately fenced stage measurements are **not additive** and do not reconstruct the ordinary prediction path. Upload and readback waits include browser scheduling and IPC; they are not pure PCIe bandwidth measurements. GPU pass timestamps exclude input upload, output readback, query resolution and host waiting. Compare the GPU duration with ordinary end-to-end latency to assess the scale of execution versus host/transfer overhead, without treating the difference as an exact stage breakdown.

On Windows, use the existing server command (`npm run mnist:serve`), hard refresh Chrome/Edge with **Ctrl+F5**, then click **Validate and benchmark**. Only JavaScript/HTML changed; retraining and rebuilding WASM are unnecessary. Download the JSON and inspect `stageDiagnostics`.

A tiny network can still be faster on CPU: GPU submission and readback costs may dominate. One workgroup per image prioritizes low dispatch overhead; it offers little GPU parallelism at batch size one. Batching improves occupancy but changes the workload. Measure on the intended Tauri WebView/browser and hardware before claiming a speedup. No GPU speedup is presumed.

For future optimization, compare workgroup sizes, output-major versus transposed weights, FP16 where supported, and dispatch strategies. Preserve parity checks and measure single-image latency separately from throughput.

References: [WebGPU specification](https://www.w3.org/TR/webgpu/) and [WGSL synchronization rules](https://www.w3.org/TR/WGSL/#sync-builtin-functions).

## Validation in this workspace

Training completed for 15 epochs: **97.30%** accuracy on all 10,000 MNIST test images. Native Rust fixture parity and the release WASM build passed. Chromium 156 executed the WASM and WGSL paths successfully using SwiftShader (software GPU): 100/100 CPU/GPU predictions agreed. See `artifacts/browser-validation.json` for the recorded validation run. These software-adapter timings do not predict hardware GPU performance. Run the page on your target device to measure that. Existing repository Node tests also passed.

The schema-version-2 timestamp/stage run is recorded in `artifacts/browser-validation-stages.json`. It passed with timestamp queries enabled; an additional browser test masked the optional feature and verified null GPU timestamps, finite host timings and matching outputs. Both tests used software rendering.

## Camera → GPU preprocessing → inference → rendering

Open **http://localhost:5174/camera.html** using the existing server (`npm run mnist:serve`). No retraining or WASM rebuild is needed if you already built MNIST. The benchmark page also links to this experiment.

- **Start MNIST video demo** creates a synthetic 30 FPS canvas video stream with a new fixture digit approximately once per second. The synthetic source is generated on the CPU; all downstream video preprocessing, inference and prediction rendering run on the GPU.
- **Start camera** requests browser camera permission. Show thick dark handwriting on light paper centered in the green square. Switch the ink checkbox for light handwriting on a dark background.
- **Stop** cancels frame scheduling, stops stream tracks and releases the GPU context/device.

Each frame imports the current video as a `GPUExternalTexture`, then submits one command buffer containing GPU center cropping, grayscale conversion, optional inversion and 20×20 fitting with a four-pixel MNIST margin; the existing fused NN compute pass; and a render pass. The render pass samples the video and reads the input tensor/logits directly to display a tensor preview, predicted digit and ten score bars. No CPU pixel extraction, tensor upload, output mapping or logit readback occurs in the streaming frame loop. The small preprocessing options buffer is updated only when dimensions or inversion change. Static weights are uploaded once at startup.

Import, bind group creation, encoding and submission happen synchronously in one video callback so the external texture is consumed before expiration. Bind groups referencing the imported texture are recreated each frame. Resources share the same GPU device. Up to two submissions may be in flight; asynchronous completion notifications provide backpressure and excess frames are dropped instead of building an unbounded queue. The callback does not await GPU completion. Those notifications still have browser overhead; this is not an overhead-free path.

The status reports submitted FPS, average host import/encoding/submission time, queue depth and dropped frames. FPS is not a presentation measurement and host time is not camera-to-display latency. There are no timestamp readbacks in this streaming mode. Browser video import may still require internal copies or color conversion; external textures do not guarantee zero-copy ingestion. See the [external texture specification](https://gpuweb.github.io/gpuweb/#gpuexternaltexture).

MNIST remains a handwriting model, not a general camera recognizer. Preprocessing does not segment arbitrary backgrounds or center handwriting by its mass. The video demo is the more controlled execution test. This experiment does not replace the YOLO/Burn camera path: migrating YOLO requires GPU preprocessing and output decoding/NMS to share its executor's GPU resources.

Validation: browser tests exercised continuous synthetic video, camera start/stop using Chromium's fake camera, both inversion modes, exact white/background preprocessing, Rust/GPU logit parity, and GPU-rendered digit segments versus CPU argmax. Diagnostic readbacks occurred only in the test harness. Results are in `artifacts/camera-validation.json`. Tests used headless Chromium/SwiftShader with consistent Vulkan video/WebGPU backends; the mixed ANGLE/software backend lost its GPU device during external video import. Windows NVIDIA camera performance has not been measured here.
