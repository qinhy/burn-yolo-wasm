// Optional integration test: npm install --no-save --package-lock=false playwright
// Build WASM, start Vite, then: node scripts/test_gpu_video.mjs
// VIDEO_TEST_IMAGE defaults to the public YOLO bus fixture saved under model/.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: process.env.VIDEO_TEST_SOFTWARE === '1'
    ? ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-angle=vulkan', '--use-vulkan=swiftshader', '--enable-features=Vulkan', '--disable-vulkan-surface']
    : ['--enable-unsafe-webgpu'],
});
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.error(message.text()); });
  await page.goto(`${process.env.VIDEO_TEST_URL || 'http://127.0.0.1:5173'}/video.html`);
  const result = await page.evaluate(async imageUrl => {
    const { default: init, VideoPipeline } = await import('/src/wasm/pkg/burn_yolo_inference.js');
    await init({ module_or_path: '/src/wasm/pkg/burn_yolo_inference_bg.wasm' });
    const image = new Image(); image.src = imageUrl; await image.decode();
    const source = document.createElement('canvas'); source.width = image.naturalWidth; source.height = image.naturalHeight;
    const ctx = source.getContext('2d'); ctx.drawImage(image, 0, 0);
    const stream = source.captureStream(10);
    const video = document.querySelector('#source'); video.srcObject = stream; await video.play();
    const timer = setInterval(() => ctx.drawImage(image, 0, 0), 100);
    let session;
    try {
      session = await VideoPipeline.create(document.querySelector('#output'), video);
      session.start();
      const deadline = performance.now() + 150000;
      while (session.metrics().completedFrames < 2) {
        if (performance.now() > deadline) throw new Error('Timed out waiting for video inference');
        if (session.metrics().error) throw new Error(session.metrics().error);
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      session.stop();
      while (session.metrics().inFlight) await new Promise(resolve => setTimeout(resolve, 20));
      const streamingMetrics = session.metrics();
      const report = await session.validate_last_frame();
      // Export the GPU-preprocessed tensor and head for an independent PyTorch comparison.
      return { streamingMetrics, nmsMatchesCpu: report.nmsMatchesCpu,
        gpuDetections: report.gpuDetections, cpuDetections: report.cpuDetections,
        headShape: report.headShape, input: Array.from(report.input), head: Array.from(report.head) };
    } finally {
      clearInterval(timer); session?.stop(); session?.free();
      video.pause(); video.srcObject = null; stream.getTracks().forEach(track => track.stop());
    }
  }, process.env.VIDEO_TEST_IMAGE || '/model/video-test-bus.jpg');
  assert.equal(result.streamingMetrics.error, undefined);
  assert.equal(result.streamingMetrics.pixelReadbacks, 0);
  assert.equal(result.streamingMetrics.predictionReadbacks, 0);
  assert.ok(result.streamingMetrics.completedFrames >= 2);
  assert.equal(result.nmsMatchesCpu, true);
  assert.ok(result.gpuDetections.length > 0, 'Expected detections on the bus fixture');
  assert.deepEqual(result.headShape, [1, 84, 2100]);
  assert.equal(result.input.length, 3 * 320 * 320);
  assert.ok(result.input.every(x => Number.isFinite(x) && x >= 0 && x <= 1));
  assert.ok(result.head.every(Number.isFinite));
  assert.deepEqual(errors, []);
  const target = process.env.VIDEO_TEST_OUTPUT || 'model/video-validation.json';
  await writeFile(target, JSON.stringify({ ...result, pageErrors: errors }));
  console.log(JSON.stringify({ streamingMetrics: result.streamingMetrics, nmsMatchesCpu: result.nmsMatchesCpu, detections: result.gpuDetections.length, artifact: target }, null, 2));
} finally { await browser.close(); }
