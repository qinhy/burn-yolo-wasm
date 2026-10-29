/// <reference lib="webworker" />

// Built by wasm-pack before Vite runs. The generated package is not committed.
// @ts-ignore generated module does not exist until after prepare_model + wasm-pack
import wasmInit, { initialize, detect } from './wasm/pkg/burn_yolo_inference.js';
// @ts-ignore wasm-pack generated WASM is resolved by Vite at build time
import wasmUrl from './wasm/pkg/burn_yolo_inference_bg.wasm?url';

let wasmReady: Promise<void> | undefined;
function loadWasm(): Promise<void> {
  if (!wasmReady) {
    wasmReady = wasmInit({ module_or_path: wasmUrl }).then(() => undefined);
  }
  return wasmReady;
}

type Message = {
  id: number;
  type: 'init' | 'detect';
  preference?: 'cpu' | 'gpu' | 'auto';
  rgba?: Uint8Array;
  confidence?: number;
  iou?: number;
};

// One serial queue avoids concurrent backend changes while a forward pass is running.
let chain: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<Message>) => {
  const msg = event.data;
  chain = chain.catch(() => undefined).then(async () => {
    try {
      await loadWasm();
      let value: unknown;
      if (msg.type === 'init') {
        value = await initialize(msg.preference ?? 'auto');
      } else if (msg.type === 'detect') {
        if (!msg.rgba) throw new Error('No RGBA pixels supplied');
        value = await detect(msg.rgba, msg.confidence ?? 0.25, msg.iou ?? 0.45);
      } else {
        throw new Error('Unknown worker request');
      }
      self.postMessage({ id: msg.id, ok: true, value });
    } catch (error) {
      self.postMessage({ id: msg.id, ok: false, error: String(error) });
    }
  });
};
