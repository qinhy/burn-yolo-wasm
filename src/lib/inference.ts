export type BackendPreference = 'auto' | 'gpu' | 'cpu';

export interface Detection {
  x1: number; y1: number; x2: number; y2: number;
  score: number; class_id: number;
}

type Response =
  | { id: number; ok: true; value: unknown }
  | { id: number; ok: false; error: string };

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
};

/** The worker lives inside the same frontend bundle in both Tauri and web. */
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
      this.worker.postMessage({ ...data, id }, transfer);
    });
  }

  initialize(preference: BackendPreference): Promise<string> {
    return this.request<string>({ type: 'init', preference });
  }

  detect(rgba: Uint8Array, confidence: number, iou: number): Promise<Detection[]> {
    // Transfer a copy to the worker without transferring the Canvas ImageData buffer.
    return this.request<Detection[]>(
      { type: 'detect', rgba, confidence, iou },
      [rgba.buffer as ArrayBuffer],
    );
  }

  dispose(): void {
    this.worker.terminate();
    for (const req of this.requests.values()) req.reject(new Error('Worker disposed'));
    this.requests.clear();
  }
}
