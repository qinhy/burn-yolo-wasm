# Burn YOLO WebAssembly performance diagnostics

This patch adds **measured timings**, a **repeatable CPU-vs-WebGPU benchmark**, and **JSON export** to the existing Vue/Burn/Tauri project. It does **not** include or overwrite your generated `yolo11n.rs`, `.bpk`, `model.zip`, or model metadata.

## Apply to an already prepared repo

Extract this patch archive **over the project root**, retaining the paths listed below. For example, extract into `C:\Users\ixs\Downloads\burn-yolo-wasm\`.

```powershell
cd C:\Users\ixs\Downloads\burn-yolo-wasm
yarn wasm
yarn dev
```

Use the same page on the web and in Tauri: `yarn desktop:dev` (requires Tauri prerequisites).

### How to collect comparable results

1. Start with a regular image (not the camera) and **run one detection**.
2. Expand **Performance diagnostics** in the right sidebar. You'll see last-frame stages and browser GPU adapter details.
3. Click **Benchmark CPU vs GPU**. The app initializes each backend, runs **3 warmups and 10 measured detections** on **the same preprocessed image**, and reports median and p90 latencies.
4. Click **Diagnostics JSON** to download `burn-yolo-performance-diagnostics.json`. Share that file with the developer to isolate the bottleneck.

The button is disabled while inference is in progress. The benchmark stops camera capture and tries to restore your previously selected backend afterward. If WebGPU fails, the CPU result is still retained and the error is visible.

## What is timed (all durations in milliseconds)

**Initialization**: `wasmLoadMs` (first WASM instantiation), `deviceInitMs` (Burn backend initialization), `unzipMs` (embedded ZIP decompression), and `modelLoadMs` (Burnpack deserialization/transfer to backend). Model re-use is flagged as `reusedModel`, with loading stages reported as zero for a re-use.

**One detection**:

| Metric | Meaning |
|---|---|
| `preprocessMs` | Canvas letterboxing + RGBA extraction in the frontend |
| `rgbaToNchwMs` | Rust normalization and NCHW reordering |
| `tensorAndForwardMs` | Tensor construction and `model.forward()` wall time; on GPU this is **not** total GPU execution |
| `readbackWaitMs` | `into_data_async()` wait; includes pending GPU execution, synchronization and output transfer |
| `dataToVecMs` | Burn TensorData to Rust `Vec<f32>` conversion |
| `decodeNmsMs` | CPU YOLO decode and class-aware NMS |
| `rustTotalMs` | Whole Rust `detect()` call (including above and overhead) |
| `hostRoundTripMs` | Frontend-to-Worker transfer, WASM call and reply |
| `renderMs` | 2D Canvas annotation drawing |
| `totalMs` | From frontend preprocessing start to completed canvas drawing |

`outputShape` and `outputBytes` help determine whether reading back the unfiltered YOLO head is significant. `nchwBytes` and `inputRgbaBytes` are also reported.

**Benchmark**: median and p90 of `hostRoundTripMs`, plus medians for Rust total, host forward dispatch, readback/wait, and NMS. Both backends use identical image data and thresholds; this excludes frontend preprocessing and rendering. The JS→Worker `Uint8Array` is copied before the timed request and then transferred by ownership.

## GPU selection caveat

The GPU metadata comes from **`navigator.gpu.requestAdapter()` in the Web Worker**. It describes the **browser's default adapter**, and might **not be the adapter chosen by Burn/wgpu**. Identification fields can be hidden or anonymized by the browser. On some browsers, Worker WebGPU is unavailable. No GPU device is explicitly created by this diagnostic query.

WebGPU `forward()` may only enqueue compute; the actual kernels can finish while the code awaits `into_data_async()`. For accurate **kernel-only** GPU profiling, you would need Burn/CubeCL GPU timestamp integration (if supported) or a browser GPU trace. Do not interpret `tensorAndForwardMs` as GPU kernel time, and do not interpret `readbackWaitMs` as transfer-only time.

The first frames may include shader compilation, pipeline setup, and caching. The benchmark drops 3 warmup samples but performance can still vary with power management, workload, thermal throttling, browser choice, and background tabs. Benchmark several rounds if results are close.

## Source files modified

- `crates/inference/Cargo.toml`: high resolution browser timing support (`js-sys`, `web-sys`).
- `crates/inference/src/lib.rs`: initialization and per-stage Rust metrics, preserving the cached WebGPU runtime fix.
- `src/inference.worker.ts`: Worker queue timing, browser adapter diagnostics, WASM instantiation timing.
- `src/lib/inference.ts`: typed reports and frontend↔Worker roundtrip timing.
- `src/lib/stats.mjs`, `src/lib/stats.d.mts`: benchmark summaries.
- `src/App.vue`: diagnostics sidebar, benchmark and downloadable JSON.
- `src/style.css`: diagnostics styling.
- `tests/stats.test.mjs`: metric summary tests.

## Test status

JavaScript geometry/statistics tests pass. Standalone TypeScript checking of the Worker and inference module passes. This environment lacks a Rust toolchain and generated model/weights, so the complete `wasm-pack` build, Vue production build and GPU results **are not locally verified**. Rebuild on your workstation, and share a runtime compiler error if one occurs.
