<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import meta from './model-meta.json';
import { InferenceClient, type BackendPreference, type Detection } from './lib/inference';
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

async function setBackend(): Promise<void> {
  if (busy.value) return;
  busy.value = true;
  ready.value = false;
  error.value = '';
  status.value = `Loading model (${backend.value})…`;
  try {
    activeBackend.value = await client.initialize(backend.value);
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
    const boxes = await client.detect(rgba, confidence.value, iou.value);
    detections.value = boxes;
    elapsedMs.value = performance.now() - started;
    drawDetections(canvas.value, source, width, height, boxes, geometry, labelNames);
    hasFrame.value = true;
    status.value = `${boxes.length} detection${boxes.length === 1 ? '' : 's'} · ${elapsedMs.value.toFixed(0)} ms`;
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
        <section class="panel panel-small"><h3>Deployment</h3><p class="muted">PyTorch → ONNX → Burn → embedded WASM</p><div class="chips"><span>Offline-ready inference</span><span>Shared Vue UI</span><span>ZIP weights</span><span>Class-aware NMS</span></div></section>
      </aside>
    </main>
    <footer>Burn 0.22 · YOLO11n · This is a reference implementation. Check inference output and licenses before production deployment.</footer>
  </div>
</template>
