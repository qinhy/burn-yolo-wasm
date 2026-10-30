# Rust/WASM GPU video experiment

The `video` branch adds `/video.html`, alongside the original YOLO demo. The new path is:

```
HTML video / camera
  → GPU external texture
  → GPU RGB letterbox to [1,3,320,320]
  → Burn YOLO11 inference
  → GPU decode and class-aware greedy NMS
  → GPU video + colored box rendering
```

Rust owns GPU setup, resource binding, command encoding, frame scheduling, inference, postprocessing and rendering. GPU shaders are WGSL embedded in the Rust crate. TypeScript only handles DOM controls, camera/file acquisition, status polling and WASM initialization; it contains no WebGPU calls or pixel preprocessing. This uses the existing PyTorch-exported Burn model, not a rewritten YOLO network.

## Windows CMD

From the repository root, with the original generated model already present:

```cmd
cd /d C:\Users\ixs\Downloads\burn-yolo-wasm
rustc --version
rustup target add wasm32-unknown-unknown
npm run dev
```

Burn 0.22.1 needs Rust **1.95 or newer**. Update with `rustup update stable` if necessary. If dependencies are not installed, run `npm install` first (or use the existing Yarn setup). `npm run dev` builds WASM and starts Vite.

Open **http://localhost:5173/video.html** in Chrome or Edge. Choose **Start camera** or **Open video**. Chrome requests permission for the camera. Use **Stop** to release media tracks. A link on the original page opens this experiment.

If weights/source are missing, follow the existing export prerequisites and run:

```cmd
python scripts\prepare_model.py
```

Do not regenerate a working custom model merely to enable video. This experiment uses the same embedded model ZIP. No extra model download is needed when those artifacts already exist.

The build includes both HTML pages, so `npm run build` and the Tauri bundle include the experiment. Tauri's media CSP permits blob video sources; actual camera/WebGPU availability depends on the platform WebView. Chrome is the initial validation target, not a claim of tested Tauri camera support.

## What stays on the GPU

`Device::wgpu_options().init_with_setup_async()` supplies the exact wgpu setup used by Burn. Rust accesses the same browser GPU device and queue through wgpu's WebGPU handle. A persistent Burn input tensor is bound directly as the preprocessing shader's storage buffer. The generated model consumes that tensor. The output is made contiguous on the GPU, then its CubeCL-managed buffer is bound directly to GPU decode.

Burn's client is flushed before external consumers are submitted; this submits queued work without reading tensors or awaiting completion. Managed resources pin pooled input/output allocations while external GPU commands use them. Preprocessing, inference and rendering share a queue, so their submissions remain ordered.

The Rust video callback imports and submits all uses of the external video texture in one JavaScript task. It does not await inference. One inference may be in flight; incoming frames are skipped while it is busy. Asynchronous queue completion releases pinned output resources and allows the next frame. This limits latency growth and memory use, but completion notifications still have browser overhead.

Video import can involve internal browser copies and color conversion. The application does not extract CPU pixels or upload a CPU NCHW array each frame. It also does not read raw predictions, detections or canvas pixels during normal streaming. Browser/GPU presentation still has costs.

GPU preprocessing follows the original rounded letterbox size and floor padding, RGB 0–1 and padding value 114/255. GPU NMS uses the highest class per anchor, class-aware suppression, a maximum of 1,200 ranked candidates and 100 retained boxes. The initial NMS selection is serial and bounded; dense scenes may make it a bottleneck. Equal-score box selection can differ from Rust's unstable sort. Boxes are colored by class; text labels are not yet rendered.

## Metrics and correctness

The status includes submitted/completed counts, frames skipped while busy, host preprocessing/inference/postprocessing submission means (excluding the first three frames), and last completion wall time. Means are **host timings**, not GPU kernel durations. Submission FPS is not presentation FPS or camera-to-display latency. No readback cost is hidden inside a claimed GPU timestamp.

**Validate last frame** pauses streaming, waits for the pending frame, then deliberately reads back the input, YOLO head and GPU detections. It compares GPU decode/NMS with the existing CPU reference and reports `nmsMatchesCpu`. Its readbacks are counted separately in the metrics. Normal streaming never calls this diagnostic. Restart the source to resume. A failed GPU session requires a page reload.

For a repeatable real-image video test, save the standard Ultralytics `bus.jpg` fixture as `model/video-test-bus.jpg`, install optional browser test tooling, and run Vite in another terminal:

```cmd
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node scripts\test_gpu_video.mjs
python scripts\check_video_reference.py model\video-validation.json
```

The browser test synthesizes a video stream from the bus image, runs at least two Rust-scheduled frames, verifies zero normal-streaming readbacks, and compares GPU NMS with CPU NMS. It saves diagnostic tensors to the ignored `model/` directory. The Python check compares Burn's head with PyTorch on the **exact GPU-preprocessed input**, isolating inference correctness from resizing/color-conversion differences. It uses box-coordinate tolerances `atol=0.05, rtol=0.001`, and class-score tolerances `atol=0.002, rtol=0.01`.

Optional browser test environment variables: `VIDEO_TEST_URL`, `VIDEO_TEST_IMAGE`, `VIDEO_TEST_OUTPUT`, `CHROME_PATH`, and `VIDEO_TEST_SOFTWARE=1` for the headless Vulkan/SwiftShader configuration. Software timings are not hardware GPU benchmarks.

API references: [Burn GPU setup and resource-sharing contract](https://docs.rs/burn-tensor/0.22.1/burn_tensor/wgpu/index.html), [CubeCL managed WGPU resource](https://docs.rs/cubecl-wgpu/0.11.3/cubecl_wgpu/struct.WgpuResource.html), and [WebGPU external textures](https://gpuweb.github.io/gpuweb/#gpuexternaltexture).
