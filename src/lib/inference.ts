export type BackendPreference = 'auto' | 'gpu' | 'cpu';

export interface Detection {
  x1: number; y1: number; x2: number; y2: number;
  score: number; class_id: number;
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean | null;
  features: string[];
  maxBufferSize: number | null;
  maxStorageBufferBindingSize: number | null;
  maxComputeWorkgroupSizeX: number | null;
}
export interface RuntimeEnvironment {
  userAgent: string;
  hardwareConcurrency: number | null;
  deviceMemoryGiB: number | null;
  isSecureContext: boolean;
  crossOriginIsolated: boolean;
  workerWebGpuAvailable: boolean;
  adapter: AdapterInfo | null;
  adapterNote: string;
}
export interface InitReport {
  backend: string;
  preference: BackendPreference;
  reusedModel: boolean;
  deviceInitMs: number;
  unzipMs: number;
  modelLoadMs: number;
  rustInitTotalMs: number;
  zipBytes: number;
  weightsBytes: number;
  wasmLoadMs: number;
  workerWallMs: number;
  queueWaitMs: number;
  hostRoundTripMs: number;
  environment: RuntimeEnvironment;
}
export interface DetectMetrics {
  backend: string;
  rgbaToNchwMs: number;
  tensorAndForwardMs: number;
  readbackWaitMs: number;
  dataToVecMs: number;
  decodeNmsMs: number;
  rustTotalMs: number;
  inputRgbaBytes: number;
  nchwBytes: number;
  outputBytes: number;
  outputShape: [number, number, number];
  detectionCount: number;
}
export interface DetectReport {
  detections: Detection[];
  metrics: DetectMetrics;
  wasmCallWallMs: number;
  workerWallMs: number;
  queueWaitMs: number;
  hostRoundTripMs: number;
}

type Response =
  | { id: number; ok: true; value: unknown }
  | { id: number; ok: false; error: string };
type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
};

/** Same Web Worker and WASM code in browser and Tauri. */
export class InferenceClient {
  private worker = new Worker(new URL('../inference.worker.ts', import.meta.url), { type: 'module' });
  private requests = new Map<number, Pending>();
  private nextId = 1;

  constructor() {
    this.worker.onmessage = (event: MessageEvent<Response>) => {
      const reply = event.data;
      const pending = this.requests.get(reply.id);
      if (!pending) return;
      this.requests.delete(reply.id);
      if (reply.ok) pending.resolve(reply.value);
      else pending.reject(new Error(reply.error));
    };
    this.worker.onerror = (event) => {
      for (const request of this.requests.values()) {
        request.reject(new Error(event.message || 'Inference worker crashed'));
      }
      this.requests.clear();
    };
  }

  private request<T>(data: object, transfer: Transferable[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.requests.set(id, { resolve: resolve as (value: unknown) => void, reject });
      try {
        this.worker.postMessage({ ...data, id }, transfer);
      } catch (cause) {
        this.requests.delete(id);
        reject(cause instanceof Error ? cause : new Error(String(cause)));
      }
    });
  }

  async initialize(preference: BackendPreference): Promise<InitReport> {
    const start = performance.now();
    const result = await this.request<Omit<InitReport, 'hostRoundTripMs'>>({ type: 'init', preference });
    return { ...result, hostRoundTripMs: performance.now() - start };
  }

  async detect(rgba: Uint8Array, confidence: number, iou: number): Promise<DetectReport> {
    const start = performance.now();
    const result = await this.request<Omit<DetectReport, 'hostRoundTripMs'>>(
      { type: 'detect', rgba, confidence, iou },
      [rgba.buffer as ArrayBuffer],
    );
    return { ...result, hostRoundTripMs: performance.now() - start };
  }

  dispose(): void {
    this.worker.terminate();
    for (const req of this.requests.values()) req.reject(new Error('Worker disposed'));
    this.requests.clear();
  }
}
