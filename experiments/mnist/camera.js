import init, { Mnist } from './pkg/mnist_tiny.js';
import { CameraExecutor } from './camera-executor.js';
const video = document.querySelector('#video');
const canvas = document.querySelector('#output');
const status = document.querySelector('#status');
const cameraButton = document.querySelector('#camera');
const demoButton = document.querySelector('#demo');
const stopButton = document.querySelector('#stop');
const invert = document.querySelector('#invert');
let renderer, stream, callback, demoTimer, generation = 0;
let starting = false;
const hasVideoCallback = typeof video.requestVideoFrameCallback === 'function';
function stop() {
  generation++;
  if (callback !== undefined) {
    if (hasVideoCallback) video.cancelVideoFrameCallback(callback);
    else cancelAnimationFrame(callback);
  }
  clearInterval(demoTimer);
  callback = demoTimer = undefined;
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.pause();
  video.srcObject = null;
  renderer?.destroy();
  renderer = undefined;
  stopButton.disabled = true;
  cameraButton.disabled = demoButton.disabled = starting;
}
async function start(demo) {
  if (starting) return;
  stop();
  starting = true;
  cameraButton.disabled = demoButton.disabled = true;
  status.textContent = 'Initializing video and GPU…';
  const token = generation;
  try {
    await init();
    if (token !== generation) return;
    const cpu = new Mnist();
    let weights;
    try { weights = cpu.weights(); } finally { cpu.free(); }
    const nextRenderer = await CameraExecutor.create(weights, canvas);
    if (token !== generation) { nextRenderer.destroy(); return; }
    renderer = nextRenderer;
    invert.checked = !demo;
    if (demo) {
      const fixtures = await (await fetch('./artifacts/fixtures.json')).json();
      if (token !== generation) return;
      const source = document.createElement('canvas');
      source.width = source.height = 280;
      const context = source.getContext('2d');
      const tile = document.createElement('canvas');
      tile.width = tile.height = 28;
      const tileContext = tile.getContext('2d');
      let index = 0;
      const draw = () => {
        const f = fixtures[Math.floor(index++ / 30) % fixtures.length];
        const image = new ImageData(28, 28);
        f.pixels.forEach((v,i) => image.data.set([v,v,v,255],i*4));
        tileContext.putImageData(image,0,0);
        context.imageSmoothingEnabled = false;
        // Inverse of GPU's 20x20 fit: place the MNIST inner 20x20 into source.
        context.drawImage(tile,4,4,20,20,0,0,280,280);
      };
      draw();
      stream = source.captureStream(30);
      demoTimer = setInterval(draw,1000/30);
    } else {
      const nextStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }, audio: false });
      if (token !== generation) { nextStream.getTracks().forEach(t=>t.stop()); return; }
      stream = nextStream;
    }
    video.srcObject = stream;
    await video.play();
    if (token !== generation) return;
    stopButton.disabled = false;
    let began = performance.now(), lastUpdate = began, hostTotal = 0, frames = 0;
    const schedule = () => { callback = hasVideoCallback ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick); };
    const tick = () => {
      if (token !== generation) return;
      try {
        const t = performance.now();
        if (renderer.frame(video, invert.checked)) { hostTotal += performance.now()-t; frames++; }
        if (t-lastUpdate > 500 || frames === 1) {
          status.textContent = JSON.stringify({ mode:demo?'MNIST video demo':'camera', adapter:renderer.engine.adapter,
            submittedFrames:renderer.submitted, droppedForBackpressure:renderer.dropped, inFlight:renderer.inFlight,
            submittedFps:+(frames/Math.max(0.001,(t-began)/1000)).toFixed(1),
            meanHostFrameSubmitMs:frames?+(hostTotal/frames).toFixed(3):0,
            frameScheduling:hasVideoCallback?'video frame callbacks':'animation frame callbacks (may repeat frames)',
            pixelReadbacks:0, logitReadbacks:0,
            note:'FPS counts submissions, not displayed frames; host time excludes GPU completion and display latency. Browser video import may involve internal copies.' },null,2);
          lastUpdate = t;
        }
        schedule();
      } catch (error) { console.error(error); stop(); status.textContent = String(error); }
    };
    schedule();
  } catch (error) { console.error(error); stop(); status.textContent = String(error); }
  finally {
    starting = false;
    cameraButton.disabled = demoButton.disabled = false;
    if (token !== generation) stop();
  }
}
cameraButton.onclick = () => start(false);
demoButton.onclick = () => start(true);
stopButton.onclick = () => {stop();status.textContent='Stopped. Camera tracks and GPU resources released.';};
window.addEventListener('pagehide',stop);
