<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import meta from './model-meta.json';
import { InferenceClient, type BackendPreference, type Detection, type DetectReport, type InitReport } from './lib/inference';
import { summarize, type Summary } from './lib/stats.mjs';
import { drawDetections, preprocess, type Geometry } from './lib/geometry.ts';

const backend = ref<BackendPreference>('auto');
const activeBackend = ref('Not initialized');
const confidence = ref(0.25);
const iou = ref(0.45);
const busy = ref(false);
const ready = ref(false);
const camera = ref(false);
const dragging = ref(false);
const status = ref('Initializing Burn WebAssembly…');
const error = ref('');
const detections = ref<Detection[]>([]);
const elapsedMs = ref<number | null>(null);
const initReport = ref<InitReport | null>(null);
const lastReport = ref<DetectReport | null>(null);
type FrameTimings = { preprocessMs: number; renderMs: number; totalMs: number };
const frameTimings = ref<FrameTimings | null>(null);
type BenchmarkMetrics = 'roundTrip' | 'rustTotal' | 'forward' | 'readback' | 'nms';
type BenchmarkRow = {
  backend: string;
  preference: 'cpu' | 'gpu';
  status: 'ok' | 'error';
  error?: string;
  init?: InitReport;
  warmup: number;
  iterations: number;
  stats?: Record<BenchmarkMetrics, Summary | null>;
};
const benchmarkRows = ref<BenchmarkRow[]>([]);
const benchmarking = ref(false);
const benchmarkProgress = ref('');
const diagnosticsNotice = ref('');
const canvas = ref<HTMLCanvasElement | null>(null);
const video = ref<HTMLVideoElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const imageName = ref('');
const selectedImage = ref<HTMLImageElement | null>(null);
const hasFrame = ref(false);
const client = new InferenceClient();

let stream: MediaStream | null = null;
let cameraGeneration = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
const labelNames = meta.names;
const backendDescription = computed(() => activeBackend.value.includes('WebGPU') ? 'GPU accelerated' : 'CPU inference');
const sortedDetections = computed(() => detections.value.slice().sort((a, b) => b.score - a.score));
const comparison = computed(() => {
  const cpu = benchmarkRows.value.find((r) => r.preference === 'cpu')?.stats?.roundTrip?.median;
  const gpu = benchmarkRows.value.find((r) => r.preference === 'gpu')?.stats?.roundTrip?.median;
  if (!cpu || !gpu) return null;
  return cpu >= gpu
    ? `WebGPU ${ (cpu / gpu).toFixed(2) }× faster than CPU (median round-trip)`
    : `WebGPU ${ (gpu / cpu).toFixed(2) }× slower than CPU (median round-trip)`;
});

async function setBackend(): Promise<void> {
  if (busy.value) return;
  busy.value = true;
  ready.value = false;
  error.value = '';
  status.value = `Loading model (${backend.value})…`;
  try {
    const report = await client.initialize(backend.value);
    initReport.value = report;
    activeBackend.value = report.backend;
    console.info('[Burn YOLO] Initialization diagnostics', report);
    ready.value = true;
    status.value = `Model ready · ${activeBackend.value}`;
  } catch (cause) {
    error.value = `Model initialization failed: ${String(cause)}`;
    status.value = 'Initialization failed';
  } finally {
    busy.value = false;
  }
  if (ready.value && selectedImage.value && !camera.value) void runImage();
}

async function runFrame(source: CanvasImageSource, width: number, height: number): Promise<void> {
  if (!ready.value || busy.value || !canvas.value) return;
  busy.value = true;
  error.value = '';
  status.value = 'Detecting objects…';
  const started = performance.now();
  try {
    const { rgba, geometry }: { rgba: Uint8Array; geometry: Geometry } = preprocess(source, width, height);
    const preprocessMs = performance.now() - started;
    const report = await client.detect(rgba, confidence.value, iou.value);
    detections.value = report.detections;
    lastReport.value = report;
    const renderStart = performance.now();
    drawDetections(canvas.value, source, width, height, report.detections, geometry, labelNames);
    frameTimings.value = {
      preprocessMs,
      renderMs: performance.now() - renderStart,
      totalMs: performance.now() - started,
    };
    elapsedMs.value = frameTimings.value.totalMs;
    hasFrame.value = true;
    status.value = `${report.detections.length} detection${report.detections.length === 1 ? '' : 's'} · ${elapsedMs.value.toFixed(0)} ms`;
    if (!camera.value) console.info('[Burn YOLO] Frame diagnostics', { ...report, frameTimings: frameTimings.value });
  } catch (cause) {
    error.value = `Inference failed: ${String(cause)}`;
    status.value = 'Inference error';
  } finally {
    busy.value = false;
  }
}

async function runImage(): Promise<void> {
  const image = selectedImage.value;
  if (!image) return;
  await runFrame(image, image.naturalWidth, image.naturalHeight);
}

async function loadFile(file: File): Promise<void> {
  if (!file.type.startsWith('image/')) {
    error.value = 'Select a PNG, JPEG, WebP, or another browser-supported image.';
    return;
  }
  stopCamera();
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    selectedImage.value = image;
    imageName.value = file.name;
    detections.value = [];
    if (canvas.value) {
      const c = canvas.value;
      c.width = image.naturalWidth;
      c.height = image.naturalHeight;
      c.getContext('2d')?.drawImage(image, 0, 0);
      hasFrame.value = true;
    }
    if (ready.value) await runImage();
  } catch (cause) {
    error.value = `Could not open image: ${String(cause)}`;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function selectFile(event: Event): void {
  const input = event.target as HTMLInputElement;
  if (input.files?.[0]) void loadFile(input.files[0]);
  input.value = '';
}

function dropFile(event: DragEvent): void {
  event.preventDefault();
  dragging.value = false;
  if (event.dataTransfer?.files?.[0]) void loadFile(event.dataTransfer.files[0]);
}

function pasteFile(event: ClipboardEvent): void {
  const item = Array.from(event.clipboardData?.items ?? []).find((x) => x.type.startsWith('image/'));
  const file = item?.getAsFile();
  if (file) void loadFile(file);
}

function stopCamera(): void {
  cameraGeneration += 1;
  if (timer) clearTimeout(timer);
  timer = undefined;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  if (video.value) video.value.srcObject = null;
  camera.value = false;
}

async function cameraLoop(generation: number): Promise<void> {
  if (!camera.value || generation !== cameraGeneration) return;
  if (video.value && video.value.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !busy.value) {
    await runFrame(video.value, video.value.videoWidth, video.value.videoHeight);
  }
  if (camera.value && generation === cameraGeneration) {
    timer = setTimeout(() => void cameraLoop(generation), 80);
  }
}

async function toggleCamera(): Promise<void> {
  if (camera.value) {
    stopCamera();
    return;
  }
  if (!ready.value || !video.value) return;
  error.value = '';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.value.srcObject = stream;
    await video.value.play();
    selectedImage.value = null;
    imageName.value = 'Live camera';
    camera.value = true;
    const generation = ++cameraGeneration;
    void cameraLoop(generation);
  } catch (cause) {
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    error.value = `Camera permission or capture unavailable: ${String(cause)}`;
  }
}

const WARMUP = 3;
const RUNS = 10;

function formatMs(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(2)} ms`;
}
function formatMiB(bytes: number | null | undefined): string {
  return bytes == null ? '—' : `${(bytes / 1048576).toFixed(2)} MiB`;
}
function medianMs(value: Summary | null | undefined): string {
  return formatMs(value?.median);
}
function benchStage(rows: DetectReport[], selector: (row: DetectReport) => number): Summary | null {
  return summarize(rows.map(selector));
}

/** Same source pixels, thresholds, Worker, and WASM for each backend.
 * Excludes UI preprocessing and canvas drawing; includes Worker transfer and readback.
 * Keep this separate from the camera loop for reproducible results.
 */
async function runBenchmark(): Promise<void> {
  if (!selectedImage.value || !ready.value || busy.value) return;
  stopCamera();
  const image = selectedImage.value;
  const originalBackend = backend.value;
  busy.value = true;
  benchmarking.value = true;
  benchmarkRows.value = [];
  error.value = '';
  const start = performance.now();
  try {
    const { rgba } = preprocess(image, image.naturalWidth, image.naturalHeight);
    for (const preference of ['cpu', 'gpu'] as const) {
      benchmarkProgress.value = `Initializing ${preference.toUpperCase()}…`;
      status.value = benchmarkProgress.value;
      try {
        const init = await client.initialize(preference);
        const samples: DetectReport[] = [];
        for (let index = 0; index < WARMUP + RUNS; index += 1) {
          // detect transfers its input buffer, so provide a fresh copy each run.
          // The copy occurs before request timing and is excluded from round-trip.
          const input = new Uint8Array(rgba);
          benchmarkProgress.value = `${preference.toUpperCase()} ${index + 1}/${WARMUP + RUNS} ${index < WARMUP ? '(warmup)' : '(measured)'}`;
          status.value = benchmarkProgress.value;
          const output = await client.detect(input, confidence.value, iou.value);
          if (index >= WARMUP) samples.push(output);
          // Allow UI to paint progress between serial requests.
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        }
        const stats: Record<BenchmarkMetrics, Summary | null> = {
          roundTrip: benchStage(samples, (item) => item.hostRoundTripMs),
          rustTotal: benchStage(samples, (item) => item.metrics.rustTotalMs),
          forward: benchStage(samples, (item) => item.metrics.tensorAndForwardMs),
          readback: benchStage(samples, (item) => item.metrics.readbackWaitMs),
          nms: benchStage(samples, (item) => item.metrics.decodeNmsMs),
        };
        benchmarkRows.value.push({ preference, backend: init.backend, status: 'ok', init, warmup: WARMUP, iterations: RUNS, stats });
        console.table({ [preference]: Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, v?.median])) });
      } catch (cause) {
        benchmarkRows.value.push({ preference, backend: preference, status: 'error', error: String(cause), warmup: WARMUP, iterations: RUNS });
      }
    }
    console.info('[Burn YOLO] CPU / GPU benchmark summary', benchmarkRows.value);
    status.value = `Benchmark completed in ${((performance.now() - start) / 1000).toFixed(1)}s`;
  } catch (cause) {
    error.value = `Benchmark failed: ${String(cause)}`;
  } finally {
    // Model selection is restored even if GPU fails; original preference remains in UI.
    benchmarkProgress.value = 'Restoring original backend…';
    try {
      const restored = await client.initialize(originalBackend);
      activeBackend.value = restored.backend;
      initReport.value = restored;
      ready.value = true;
    } catch (cause) {
      ready.value = false;
      error.value = `Could not restore ${originalBackend} backend: ${String(cause)}`;
    }
    busy.value = false;
    benchmarking.value = false;
    benchmarkProgress.value = '';
  }
}

function exportDiagnostics(): void {
  const report = {
    capturedAt: new Date().toISOString(),
    app: 'Burn YOLO WASM diagnostics v1',
    model: meta,
    frontend: {
      isTauri: '__TAURI_INTERNALS__' in window,
      pageUrl: location.origin,
      devicePixelRatio: window.devicePixelRatio,
      selectedBackend: backend.value,
      activeBackend: activeBackend.value,
      confidence: confidence.value,
      iou: iou.value,
      sourceDimensions: selectedImage.value
        ? [selectedImage.value.naturalWidth, selectedImage.value.naturalHeight] : null,
    },
    initialization: initReport.value,
    lastFrame: frameTimings.value,
    lastInference: lastReport.value ? { ...lastReport.value, detections: undefined } : null,
    benchmark: benchmarkRows.value,
    timingNotes: [
      'tensorAndForwardMs is host tensor creation and forward dispatch, not isolated GPU execution.',
      'readbackWaitMs can include queued GPU work, synchronization and output transfer.',
      'Benchmark hostRoundTripMs includes Worker transfer and does not include image preprocessing or canvas rendering.',
      'The adapter query describes the browser default GPU adapter and may not identify Burn wgpu device.',
    ],
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'burn-yolo-performance-diagnostics.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  diagnosticsNotice.value = 'Diagnostics JSON downloaded.';
}

function saveImage(): void {
  if (!canvas.value || !hasFrame.value) return;
  const anchor = document.createElement('a');
  anchor.href = canvas.value.toDataURL('image/png');
  anchor.download = 'burn-yolo-detections.png';
  anchor.click();
}

onMounted(() => {
  void setBackend();
  window.addEventListener('paste', pasteFile);
});
onUnmounted(() => {
  stopCamera();
  client.dispose();
  window.removeEventListener('paste', pasteFile);
});
</script>

<template>
  <div class="app-shell" @dragover.prevent="dragging = true" @dragleave.prevent="dragging = false" @drop="dropFile">
    <header class="topbar">
      <div class="brand"><div class="brand-icon"><span>◈</span></div>
        <div><strong>BURN <span>YOLO</span></strong><small>WebAssembly object detection</small></div>
      </div>
      <div class="top-meta"><span class="pill"><span class="pulse" :class="{ idle: !ready }"></span>{{ ready ? 'Inference ready' : 'Not ready' }}</span><span class="meta-tag">WEB / TAURI · SAME ENGINE</span></div>
    </header>

    <main class="main-layout">
      <section class="work-area">
        <div class="section-heading"><div><p class="eyebrow">LOCAL INFERENCE</p><h1>See what your model sees.</h1><p>YOLO11n runs entirely on your device. No image upload to a server.</p></div></div>
        <div class="preview" :class="{ dragging }">
          <canvas ref="canvas" :class="{ hidden: !hasFrame }" aria-label="Image with YOLO detection overlays" />
          <div v-if="!hasFrame" class="empty-state" @click="fileInput?.click()">
            <div class="empty-icon">⌁</div>
            <h2>Drop an image to get started</h2>
            <p>JPEG, PNG, WebP · or paste an image</p>
            <button class="button button-primary" type="button" @click.stop="fileInput?.click()">Choose image</button>
          </div>
          <div v-if="dragging" class="drag-cover">Drop image here</div>
        </div>
        <video ref="video" autoplay muted playsinline class="hidden-video"></video>
        <input ref="fileInput" class="visually-hidden" type="file" accept="image/*" @change="selectFile" />
        <div class="actionbar">
          <button class="button button-primary" type="button" @click="fileInput?.click()">Upload image</button>
          <button class="button" type="button" :disabled="!ready" @click="toggleCamera">{{ camera ? 'Stop camera' : 'Use camera' }}</button>
          <button class="button" type="button" :disabled="!ready || busy || !selectedImage" @click="runImage">Run detection</button>
          <button class="button iconbutton" type="button" :disabled="!hasFrame" title="Download annotated PNG" @click="saveImage">↓ Save</button>
        </div>
        <div class="statusline"><span>{{ status }}</span><span v-if="imageName">{{ imageName }}</span></div>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </section>

      <aside class="sidebar">
        <section class="panel">
          <h3>Model settings</h3><p class="muted">{{ meta.model }} · {{ meta.size }} × {{ meta.size }} · {{ meta.names.length }} classes</p>
          <label for="backend">Compute backend</label>
          <select id="backend" v-model="backend" :disabled="busy" @change="setBackend"><option value="auto">Auto (GPU → CPU)</option><option value="gpu">WebGPU only</option><option value="cpu">WASM CPU only</option></select>
          <p class="backend-info"><span class="small-dot"></span>{{ activeBackend }} <small>· {{ backendDescription }}</small></p>
          <div class="range-title"><label for="conf">Confidence threshold</label><output>{{ confidence.toFixed(2) }}</output></div>
          <input id="conf" v-model.number="confidence" type="range" min="0.05" max="0.95" step="0.05" />
          <div class="range-title"><label for="iou">NMS IoU threshold</label><output>{{ iou.toFixed(2) }}</output></div>
          <input id="iou" v-model.number="iou" type="range" min="0.1" max="0.9" step="0.05" />
          <p class="hint">Change settings and select “Run detection” to update a loaded image. Camera mode updates automatically.</p>
        </section>
        <section class="panel">
          <div class="panel-title"><h3>Detections</h3><span class="count">{{ detections.length }}</span></div>
          <div class="stats"><div><b>{{ detections.length }}</b><span>objects found</span></div><div><b>{{ elapsedMs === null ? '—' : `${elapsedMs.toFixed(0)} ms` }}</b><span>last processing</span></div></div>
          <div v-if="sortedDetections.length" class="results">
            <div v-for="(d, index) in sortedDetections.slice(0, 25)" :key="index" class="result-row">
              <span class="result-indicator" :style="{ backgroundColor: `hsl(${(d.class_id * 47 + 187) % 360} 88% 57%)` }"></span>
              <span class="result-label">{{ labelNames[d.class_id] ?? `class ${d.class_id}` }}</span>
              <strong>{{ (d.score * 100).toFixed(1) }}%</strong>
            </div>
          </div>
          <p v-else class="results-empty">No results yet. Load an image or use the camera to begin.</p>
          <p v-if="sortedDetections.length > 25" class="hint">Showing the 25 highest-confidence results.</p>
        </section>
        <section class="panel performance-panel">
          <div class="panel-title"><h3>Performance diagnostics</h3><span class="count">BETA</span></div>
          <p class="muted">Stage timings and a repeatable warm CPU vs WebGPU comparison. All times are wall-clock, not hardware GPU timestamps.</p>
          <div class="diagnostic-actions">
            <button class="button button-primary" :disabled="busy || !ready || !selectedImage" @click="runBenchmark">{{ benchmarking ? 'Benchmark running…' : 'Benchmark CPU vs GPU' }}</button>
            <button class="button" @click="exportDiagnostics">↓ Diagnostics JSON</button>
          </div>
          <p v-if="benchmarkProgress" class="hint" role="status">{{ benchmarkProgress }}</p>
          <p v-if="diagnosticsNotice" class="hint">{{ diagnosticsNotice }}</p>
          <div v-if="frameTimings && lastReport" class="metrics">
            <h4>Last frame</h4>
            <div class="metric"><span>UI preprocessing</span><b>{{ formatMs(frameTimings.preprocessMs) }}</b></div>
            <div class="metric"><span>RGBA → NCHW (Rust)</span><b>{{ formatMs(lastReport.metrics.rgbaToNchwMs) }}</b></div>
            <div class="metric"><span>Tensor + forward dispatch*</span><b>{{ formatMs(lastReport.metrics.tensorAndForwardMs) }}</b></div>
            <div class="metric"><span>Readback + GPU wait*</span><b>{{ formatMs(lastReport.metrics.readbackWaitMs) }}</b></div>
            <div class="metric"><span>TensorData → Vec</span><b>{{ formatMs(lastReport.metrics.dataToVecMs) }}</b></div>
            <div class="metric"><span>Decode + NMS</span><b>{{ formatMs(lastReport.metrics.decodeNmsMs) }}</b></div>
            <div class="metric"><span>Rust total</span><b>{{ formatMs(lastReport.metrics.rustTotalMs) }}</b></div>
            <div class="metric"><span>Worker round-trip</span><b>{{ formatMs(lastReport.hostRoundTripMs) }}</b></div>
            <div class="metric"><span>Canvas drawing</span><b>{{ formatMs(frameTimings.renderMs) }}</b></div>
            <div class="metric metric-emphasis"><span>Complete user-facing frame</span><b>{{ formatMs(frameTimings.totalMs) }}</b></div>
            <div class="metric"><span>Output shape</span><b>{{ lastReport.metrics.outputShape.join(' × ') }}</b></div>
            <div class="metric"><span>Input RGBA</span><b>{{ formatMiB(lastReport.metrics.inputRgbaBytes) }}</b></div>
            <div class="metric"><span>GPU/CPU output data</span><b>{{ formatMiB(lastReport.metrics.outputBytes) }}</b></div>
          </div>
          <div v-if="benchmarkRows.length" class="benchmark-results">
            <h4>Warm benchmark ({{ WARMUP }} warmups + {{ RUNS }} samples each)</h4>
            <p v-if="comparison" class="comparison">{{ comparison }}</p>
            <div v-for="result in benchmarkRows" :key="result.preference" class="benchmark-result">
              <div class="metric metric-emphasis"><span>{{ result.preference.toUpperCase() }} · {{ result.backend }}</span><b>{{ result.status === 'ok' ? medianMs(result.stats?.roundTrip) : 'Failed' }}</b></div>
              <template v-if="result.stats">
                <div class="metric"><span>Round trip p90</span><b>{{ formatMs(result.stats.roundTrip?.p90) }}</b></div>
                <div class="metric"><span>Rust total median</span><b>{{ medianMs(result.stats.rustTotal) }}</b></div>
                <div class="metric"><span>Forward dispatch median*</span><b>{{ medianMs(result.stats.forward) }}</b></div>
                <div class="metric"><span>Readback/wait median*</span><b>{{ medianMs(result.stats.readback) }}</b></div>
                <div class="metric"><span>NMS median</span><b>{{ medianMs(result.stats.nms) }}</b></div>
                <div class="metric"><span>Device init</span><b>{{ formatMs(result.init?.deviceInitMs) }}</b></div>
              </template>
              <p v-else class="error">{{ result.error }}</p>
            </div>
          </div>
          <div v-if="initReport" class="metrics">
            <h4>Model startup / environment</h4>
            <div class="metric"><span>Model instance</span><b>{{ initReport.reusedModel ? 'Reused' : 'Cold load' }}</b></div>
            <div class="metric"><span>WASM loader (first load)</span><b>{{ formatMs(initReport.wasmLoadMs) }}</b></div>
            <div class="metric"><span>GPU/CPU device initialization</span><b>{{ formatMs(initReport.deviceInitMs) }}</b></div>
            <div class="metric"><span>ZIP decompression</span><b>{{ formatMs(initReport.unzipMs) }}</b></div>
            <div class="metric"><span>Burnpack model load</span><b>{{ formatMs(initReport.modelLoadMs) }}</b></div>
            <div class="metric"><span>ZIP / Burnpack bytes</span><b>{{ formatMiB(initReport.zipBytes) }} / {{ formatMiB(initReport.weightsBytes) }}</b></div>
            <div class="metric"><span>WebGPU in Worker</span><b>{{ initReport.environment.workerWebGpuAvailable ? 'Available' : 'Unavailable' }}</b></div>
            <div class="metric"><span>Logical CPU cores</span><b>{{ initReport.environment.hardwareConcurrency ?? 'Unknown' }}</b></div>
            <div class="metric"><span>Device memory (reported)</span><b>{{ initReport.environment.deviceMemoryGiB == null ? 'Not exposed' : `${initReport.environment.deviceMemoryGiB} GiB` }}</b></div>
            <div class="metric"><span>Adapter vendor*</span><b>{{ initReport.environment.adapter?.vendor || 'Unavailable' }}</b></div>
            <div class="metric"><span>Adapter device*</span><b>{{ initReport.environment.adapter?.device || initReport.environment.adapter?.description || 'Unavailable' }}</b></div>
            <p class="hint">{{ initReport.environment.adapterNote }}</p>
          </div>
          <p class="hint">*Forward is host-side submission, not isolated GPU compute. Readback may include queued GPU execution and synchronization. Browser GPU adapter diagnostics do not guarantee the exact GPU selected by Burn. Download JSON for full values, browser details and sample summaries.</p>
        </section>
        <section class="panel panel-small"><h3>Deployment</h3><p class="muted">PyTorch → ONNX → Burn → embedded WASM</p><div class="chips"><span>Offline-ready inference</span><span>Shared Vue UI</span><span>ZIP weights</span><span>Class-aware NMS</span></div></section>
      </aside>
    </main>
    <footer>Burn 0.22 · YOLO11n · This is a reference implementation. Check inference output and licenses before production deployment.</footer>
  </div>
</template>
