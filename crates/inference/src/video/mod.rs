//! GPU-resident browser video pipeline. JS handles controls only; Rust owns execution.
mod gpu;
use super::{err, generated, now_ms, unzip_weights};
use burn::tensor::{Bytes, DType, Device, Tensor};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor, CubeBackend};
use cubecl_wgpu::{AutoCompiler, WgpuServer, WgpuSetup};
use gpu::*;
use js_sys::Uint8Array;
use serde::Serialize;
use std::{any::Any, cell::RefCell, rc::Rc};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{HtmlCanvasElement, HtmlVideoElement};

type Callback = Closure<dyn FnMut(f64, JsValue)>;
thread_local! { static SETUP: RefCell<Option<(Device,WgpuSetup)>> = const { RefCell::new(None) }; }

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Metrics {
    submitted_frames: u32,
    measured_frames: u32,
    warmup_frames: u32,
    completed_frames: u32,
    dropped_for_backpressure: u32,
    in_flight: bool,
    elapsed_ms: f64,
    submitted_fps: f64,
    mean_host_preprocess_submit_ms: f64,
    mean_host_inference_submit_ms: f64,
    mean_host_postprocess_render_submit_ms: f64,
    last_completion_wall_ms: f64,
    pixel_readbacks: u32,
    prediction_readbacks: u32,
    adapter_name: String,
    adapter_vendor: u32,
    error: Option<String>,
}
struct State {
    // Keep runtime and pooled allocations alive through external consumers.
    _setup: WgpuSetup,
    _device: Device,
    model: generated::Model,
    input: Tensor<4>,
    last_output: Option<Tensor<3>>,
    _input_guard: Box<dyn Any>,
    input_buffer: JsValue,
    gpu: JsValue,
    queue: JsValue,
    context: JsValue,
    canvas: HtmlCanvasElement,
    video: HtmlVideoElement,
    preprocess: JsValue,
    decode: JsValue,
    nms: JsValue,
    background: JsValue,
    boxes: JsValue,
    render_layout: JsValue,
    box_layout: JsValue,
    sampler: JsValue,
    config: JsValue,
    candidates: JsValue,
    detections: JsValue,
    active: bool,
    has_video_callback: bool,
    callback_id: Option<JsValue>,
    callback: Option<Callback>,
    error_handler: Option<Closure<dyn FnMut(JsValue)>>,
    metrics: Metrics,
    started: f64,
    confidence: f32,
    iou: f32,
    dimensions: (u32, u32),
}

fn tensor_buffer(tensor: &CubeTensor) -> Result<(JsValue, Box<dyn Any>), JsValue> {
    let guard = tensor
        .client
        .get_resource::<WgpuServer<AutoCompiler>>(tensor.handle.clone())
        .map_err(err)?;
    let resource = guard.resource();
    let buffer = resource
        .buffer
        .as_webgpu()
        .ok_or_else(|| err("Expected browser WebGPU buffer"))?;
    let binding = object([
        ("buffer", buffer.clone().into()),
        ("offset", (resource.offset as f64).into()),
        ("size", (resource.size as f64).into()),
    ]);
    Ok((binding, Box::new(guard)))
}
fn config_bytes(
    width: u32,
    height: u32,
    confidence: f32,
    iou: f32,
    shape: [usize; 3],
) -> Result<Vec<u8>, JsValue> {
    if width == 0 || height == 0 {
        return Err(err("Video has no decoded frame"));
    }
    let channels_first = shape[1] >= 5 && shape[1] <= 204 && shape[2] >= shape[1];
    let (channels, anchors) = if channels_first {
        (shape[1], shape[2])
    } else {
        (shape[2], shape[1])
    };
    if shape[0] != 1 || !(5..=204).contains(&channels) || anchors > 10000 {
        return Err(err(format!("Unsupported YOLO head {shape:?}")));
    }
    let scale = (320.0 / width as f32).min(320.0 / height as f32);
    let rw = (width as f32 * scale).round();
    let rh = (height as f32 * scale).round();
    let floats = [
        width as f32,
        height as f32,
        rw,
        rh,
        ((320.0 - rw) / 2.0).floor(),
        ((320.0 - rh) / 2.0).floor(),
        confidence,
        iou,
    ];
    let mut bytes = Vec::with_capacity(48);
    for v in floats {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in [
        anchors as u32,
        (channels - 4) as u32,
        u32::from(channels_first),
        0,
    ] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}
async fn render_pipelines(
    gpu: &JsValue,
    format: JsValue,
) -> Result<(JsValue, JsValue, JsValue, JsValue), JsValue> {
    let group0 = call(
        gpu,
        "createBindGroupLayout",
        &[object([(
            "entries",
            array(&[
                object([
                    ("binding", 0.into()),
                    ("visibility", 2.into()),
                    ("externalTexture", object([])),
                ]),
                object([
                    ("binding", 1.into()),
                    ("visibility", 2.into()),
                    ("sampler", object([("type", "filtering".into())])),
                ]),
                object([
                    ("binding", 2.into()),
                    ("visibility", 3.into()),
                    ("buffer", object([("type", "uniform".into())])),
                ]),
            ]),
        )])],
    )?;
    let group1 = call(
        gpu,
        "createBindGroupLayout",
        &[object([(
            "entries",
            array(&[object([
                ("binding", 0.into()),
                ("visibility", 1.into()),
                ("buffer", object([("type", "read-only-storage".into())])),
            ])]),
        )])],
    )?;
    let layout = call(
        gpu,
        "createPipelineLayout",
        &[object([(
            "bindGroupLayouts",
            array(&[group0.clone(), group1.clone()]),
        )])],
    )?;
    let module = module(gpu, include_str!("render.wgsl")).await?;
    let make = |vs: &str, fs: &str| {
        object([
            ("layout", layout.clone()),
            (
                "vertex",
                object([("module", module.clone()), ("entryPoint", vs.into())]),
            ),
            (
                "fragment",
                object([
                    ("module", module.clone()),
                    ("entryPoint", fs.into()),
                    ("targets", array(&[object([("format", format.clone())])])),
                ]),
            ),
            ("primitive", object([("topology", "triangle-list".into())])),
        ])
    };
    let background = promise(call(
        gpu,
        "createRenderPipelineAsync",
        &[make("background_vs", "background_fs")],
    )?)
    .await?;
    let boxes = promise(call(
        gpu,
        "createRenderPipelineAsync",
        &[make("boxes_vs", "boxes_fs")],
    )?)
    .await?;
    Ok((background, boxes, group0, group1))
}
impl State {
    fn schedule(&mut self) -> Result<(), JsValue> {
        if !self.active {
            return Ok(());
        }
        let cb: JsValue = self
            .callback
            .as_ref()
            .ok_or_else(|| err("Missing callback"))?
            .as_ref()
            .clone();
        self.callback_id = Some(if self.has_video_callback {
            call(self.video.as_ref(), "requestVideoFrameCallback", &[cb])?
        } else {
            call(
                &web_sys::window()
                    .ok_or_else(|| err("Window unavailable"))?
                    .into(),
                "requestAnimationFrame",
                &[cb],
            )?
        });
        Ok(())
    }
    fn frame(&mut self) -> Result<Option<Box<dyn Any>>, JsValue> {
        if self.metrics.in_flight {
            self.metrics.dropped_for_backpressure += 1;
            return Ok(None);
        }
        if self.video.ready_state() < 2 {
            return Ok(None);
        }
        let width = self.video.video_width();
        let height = self.video.video_height();
        if width == 0 || height == 0 {
            return Ok(None);
        }
        let t = now_ms();
        if self.dimensions != (width, height) {
            self.canvas.set_width(width.min(1280));
            self.canvas.set_height(
                ((height as f64 * self.canvas.width() as f64 / width as f64).round() as u32).max(1),
            );
            self.dimensions = (width, height);
        }
        // Source texture is imported and all uses submitted within this same JS task.
        let external = call(
            &self.gpu,
            "importExternalTexture",
            &[object([("source", self.video.clone().into())])],
        )?;
        let config = config_bytes(width, height, self.confidence, self.iou, [1, 84, 2100])?;
        call(
            &self.queue,
            "writeBuffer",
            &[
                self.config.clone(),
                0.into(),
                Uint8Array::from(config.as_slice()).into(),
            ],
        )?;
        let pre_bind = bind(
            &self.gpu,
            layout(&self.preprocess, 0)?,
            &[
                binding(0, external.clone()),
                binding(1, self.sampler.clone()),
                binding(2, self.input_buffer.clone()),
                binding(3, buffer_binding(self.config.clone())),
            ],
        )?;
        let encoder = call(&self.gpu, "createCommandEncoder", &[])?;
        dispatch(&encoder, &self.preprocess, &pre_bind, 40, 40)?;
        submit(&self.queue, encoder)?;
        let prep_ms = now_ms() - t;
        let forward_start = now_ms();
        let output = self.model.forward(self.input.clone());
        let shape = output.dims();
        self.last_output = Some(output.clone());
        let primitive = into_contiguous(
            output
                .try_into_primitive::<CubeBackend>()
                .map_err(|e| err(format!("{e:?}")))?,
        );
        if primitive.dtype != DType::F32 {
            return Err(err("GPU rendering requires an FP32 YOLO head"));
        }
        let (head, guard) = tensor_buffer(&primitive)?;
        // Submit pending CubeCL work before the external postprocess/render consumers.
        // flush submits work; it does not wait for GPU completion or read any tensor data.
        primitive.client.flush().map_err(err)?;
        let inference_ms = now_ms() - forward_start;
        let post_start = now_ms();
        let config = config_bytes(width, height, self.confidence, self.iou, shape)?;
        call(
            &self.queue,
            "writeBuffer",
            &[
                self.config.clone(),
                0.into(),
                Uint8Array::from(config.as_slice()).into(),
            ],
        )?;
        let channels_first = shape[1] >= 5 && shape[1] <= 204 && shape[2] >= shape[1];
        let anchors = if channels_first { shape[2] } else { shape[1] };
        let decode_bind = bind(
            &self.gpu,
            layout(&self.decode, 0)?,
            &[
                binding(0, head),
                binding(1, buffer_binding(self.candidates.clone())),
                binding(2, buffer_binding(self.config.clone())),
            ],
        )?;
        let nms_bind = bind(
            &self.gpu,
            layout(&self.nms, 0)?,
            &[
                binding(0, buffer_binding(self.candidates.clone())),
                binding(1, buffer_binding(self.detections.clone())),
                binding(2, buffer_binding(self.config.clone())),
            ],
        )?;
        let video_bind = bind(
            &self.gpu,
            self.render_layout.clone(),
            &[
                binding(0, external),
                binding(1, self.sampler.clone()),
                binding(2, buffer_binding(self.config.clone())),
            ],
        )?;
        let box_bind = bind(
            &self.gpu,
            self.box_layout.clone(),
            &[binding(0, buffer_binding(self.detections.clone()))],
        )?;
        let encoder = call(&self.gpu, "createCommandEncoder", &[])?;
        dispatch(
            &encoder,
            &self.decode,
            &decode_bind,
            ((anchors + 63) / 64) as u32,
            1,
        )?;
        dispatch(&encoder, &self.nms, &nms_bind, 1, 1)?;
        let texture = call(&self.context, "getCurrentTexture", &[])?;
        let view = call(&texture, "createView", &[])?;
        let pass = call(
            &encoder,
            "beginRenderPass",
            &[object([(
                "colorAttachments",
                array(&[object([
                    ("view", view),
                    ("loadOp", "clear".into()),
                    ("storeOp", "store".into()),
                    (
                        "clearValue",
                        object([
                            ("r", 0.into()),
                            ("g", 0.into()),
                            ("b", 0.into()),
                            ("a", 1.into()),
                        ]),
                    ),
                ])]),
            )])],
        )?;
        call(&pass, "setBindGroup", &[0.into(), video_bind])?;
        call(&pass, "setBindGroup", &[1.into(), box_bind])?;
        call(&pass, "setPipeline", &[self.background.clone()])?;
        call(&pass, "draw", &[3.into()])?;
        call(&pass, "setPipeline", &[self.boxes.clone()])?;
        call(&pass, "draw", &[24.into(), 100.into()])?;
        call(&pass, "end", &[])?;
        submit(&self.queue, encoder)?;
        if self.metrics.submitted_frames >= 3 {
            let count = self.metrics.measured_frames as f64;
            self.metrics.mean_host_preprocess_submit_ms =
                (self.metrics.mean_host_preprocess_submit_ms * count + prep_ms) / (count + 1.0);
            self.metrics.mean_host_inference_submit_ms =
                (self.metrics.mean_host_inference_submit_ms * count + inference_ms) / (count + 1.0);
            self.metrics.mean_host_postprocess_render_submit_ms =
                (self.metrics.mean_host_postprocess_render_submit_ms * count + now_ms()
                    - post_start)
                    / (count + 1.0);
            self.metrics.measured_frames += 1;
        } else {
            self.metrics.warmup_frames += 1;
        }
        self.metrics.submitted_frames += 1;
        self.metrics.in_flight = true;
        Ok(Some(Box::new((primitive, guard))))
    }
}

/// Rust-owned GPU streaming session. JS supplies DOM video/canvas handles only.
#[wasm_bindgen]
pub struct VideoPipeline {
    state: Rc<RefCell<State>>,
}
#[wasm_bindgen]
impl VideoPipeline {
    #[wasm_bindgen(js_name=create)]
    pub async fn create(
        canvas: HtmlCanvasElement,
        video: HtmlVideoElement,
    ) -> Result<VideoPipeline, JsValue> {
        let cached = SETUP.with(|slot| slot.borrow().clone());
        let (device, setup) = if let Some(pair) = cached {
            pair
        } else {
            let pair = Device::wgpu_options()
                .graphics_api(burn::tensor::wgpu::WgpuBackend::WebGpu)
                .init_with_setup_async()
                .await
                .map_err(err)?;
            SETUP.with(|slot| *slot.borrow_mut() = Some(pair.clone()));
            pair
        };
        let gpu: JsValue = setup
            .device
            .as_webgpu()
            .ok_or_else(|| err("Browser WebGPU required"))?
            .clone()
            .into();
        let queue = get(&gpu, "queue")?;
        call(&gpu, "pushErrorScope", &["validation".into()])?;
        let model = generated::Model::from_bytes(Bytes::from_bytes_vec(unzip_weights()?), &device);
        let input = Tensor::<4>::empty([1, 3, 320, 320], &device);
        let primitive = input
            .clone()
            .try_into_primitive::<CubeBackend>()
            .map_err(|e| err(format!("{e:?}")))?;
        let (input_buffer, input_guard) = tensor_buffer(&primitive)?;
        primitive.client.flush().map_err(err)?;
        let preprocess = compute(&gpu, include_str!("preprocess.wgsl")).await?;
        let decode = compute(&gpu, include_str!("decode.wgsl")).await?;
        let nms = compute(&gpu, include_str!("nms.wgsl")).await?;
        let context = canvas
            .get_context("webgpu")?
            .ok_or_else(|| err("WebGPU canvas unavailable"))?
            .into();
        let browser_gpu = get(&get(&js_sys::global(), "navigator")?, "gpu")?;
        let format = call(&browser_gpu, "getPreferredCanvasFormat", &[])?;
        call(
            &context,
            "configure",
            &[object([
                ("device", gpu.clone()),
                ("format", format.clone()),
                ("alphaMode", "opaque".into()),
            ])],
        )?;
        let (background, boxes, render_layout, box_layout) = render_pipelines(&gpu, format).await?;
        let sampler = call(
            &gpu,
            "createSampler",
            &[object([
                ("magFilter", "linear".into()),
                ("minFilter", "linear".into()),
            ])],
        )?;
        let config = buffer(&gpu, 48, 64 | 8)?; // UNIFORM | COPY_DST
        let candidates = buffer(&gpu, 10000 * 32, 128 | 4 | 8)?;
        let detections = buffer(&gpu, 32 + 100 * 32, 128 | 4 | 8)?;
        let error = promise(call(&gpu, "popErrorScope", &[])?).await?;
        if !error.is_null() {
            return Err(get(&error, "message")?);
        }
        let has_video_callback = get(video.as_ref(), "requestVideoFrameCallback")?.is_function();
        let info = setup.adapter.get_info();
        let metrics = Metrics {
            adapter_name: info.name,
            adapter_vendor: info.vendor,
            ..Metrics::default()
        };
        let state = Rc::new(RefCell::new(State {
            _setup: setup,
            _device: device,
            model,
            input,
            last_output: None,
            _input_guard: input_guard,
            input_buffer,
            gpu,
            queue,
            context,
            canvas,
            video,
            preprocess,
            decode,
            nms,
            background,
            boxes,
            render_layout,
            box_layout,
            sampler,
            config,
            candidates,
            detections,
            active: false,
            has_video_callback,
            callback_id: None,
            callback: None,
            error_handler: None,
            metrics,
            started: now_ms(),
            confidence: 0.25,
            iou: 0.45,
            dimensions: (0, 0),
        }));
        let weak = Rc::downgrade(&state);
        let error_handler = Closure::wrap(Box::new(move |event: JsValue| {
            if let Some(state) = weak.upgrade() {
                let message = get(&event, "error")
                    .and_then(|e| get(&e, "message"))
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_else(|| "WebGPU validation error".into());
                let mut s = state.borrow_mut();
                s.metrics.error = Some(message);
                s.active = false;
            }
        }) as Box<dyn FnMut(JsValue)>);
        call(
            &state.borrow().gpu,
            "addEventListener",
            &["uncapturederror".into(), error_handler.as_ref().clone()],
        )?;
        state.borrow_mut().error_handler = Some(error_handler);
        let weak = Rc::downgrade(&state);
        let callback = Closure::wrap(Box::new(move |_: f64, _: JsValue| {
            let Some(state) = weak.upgrade() else {
                return;
            };
            if !state.borrow().active {
                return;
            }
            let result = state.borrow_mut().frame();
            match result {
                Ok(Some(guard)) => {
                    let queue = state.borrow().queue.clone();
                    let start = now_ms();
                    let pending_state = state.clone();
                    spawn_local(async move {
                        let result = match call(&queue, "onSubmittedWorkDone", &[]) {
                            Ok(p) => promise(p).await,
                            Err(e) => Err(e),
                        };
                        {
                            let mut s = pending_state.borrow_mut();
                            s.metrics.in_flight = false;
                            s.metrics.completed_frames += 1;
                            s.metrics.last_completion_wall_ms = now_ms() - start;
                            if let Err(e) = result {
                                s.metrics.error = Some(format!("{e:?}"));
                                s.active = false;
                            }
                        }
                        drop(guard);
                    });
                }
                Ok(None) => {}
                Err(error) => {
                    let mut s = state.borrow_mut();
                    s.active = false;
                    s.metrics.error = Some(format!("{error:?}"));
                }
            }
            let result = state.borrow_mut().schedule();
            if let Err(e) = result {
                let mut s = state.borrow_mut();
                s.active = false;
                s.metrics.error = Some(format!("{e:?}"));
            }
        }) as Box<dyn FnMut(f64, JsValue)>);
        state.borrow_mut().callback = Some(callback);
        Ok(VideoPipeline { state })
    }
    pub fn start(&self) -> Result<(), JsValue> {
        let mut s = self.state.borrow_mut();
        if s.active {
            return Ok(());
        }
        if s.metrics.error.is_some() {
            return Err(err("Create a new session after a GPU error"));
        }
        s.active = true;
        s.started = now_ms();
        s.schedule()
    }
    pub fn stop(&self) -> Result<(), JsValue> {
        let mut s = self.state.borrow_mut();
        s.active = false;
        if let Some(id) = s.callback_id.take() {
            if s.has_video_callback {
                call(s.video.as_ref(), "cancelVideoFrameCallback", &[id])?;
            } else {
                call(
                    &web_sys::window()
                        .ok_or_else(|| err("Window unavailable"))?
                        .into(),
                    "cancelAnimationFrame",
                    &[id],
                )?;
            }
        }
        Ok(())
    }
    pub fn thresholds(&self, confidence: f32, iou: f32) -> Result<(), JsValue> {
        if !(0.0..=1.0).contains(&confidence) || !(0.0..=1.0).contains(&iou) {
            return Err(err("Thresholds must be in [0,1]"));
        }
        let mut s = self.state.borrow_mut();
        s.confidence = confidence;
        s.iou = iou;
        Ok(())
    }
    /// Explicit stopped-frame diagnostic only. Streaming never calls this readback path.
    pub async fn validate_last_frame(&self) -> Result<JsValue, JsValue> {
        let (input, head, gpu, queue, detections, confidence, iou) = {
            let s = self.state.borrow();
            if s.active || s.metrics.in_flight {
                return Err(err(
                    "Stop and wait for the in-flight frame before validating",
                ));
            }
            (
                s.input.clone(),
                s.last_output
                    .clone()
                    .ok_or_else(|| err("No inferred frame"))?,
                s.gpu.clone(),
                s.queue.clone(),
                s.detections.clone(),
                s.confidence,
                s.iou,
            )
        };
        let shape = head.dims();
        let input_values = input
            .into_data_async()
            .await
            .map_err(err)?
            .try_into_vec::<f32>()
            .map_err(err)?;
        let values = head
            .into_data_async()
            .await
            .map_err(err)?
            .try_into_vec::<f32>()
            .map_err(err)?;
        let staging = buffer(&gpu, 3232, 1 | 8)?;
        let encoder = call(&gpu, "createCommandEncoder", &[])?;
        call(
            &encoder,
            "copyBufferToBuffer",
            &[detections, 0.into(), staging.clone(), 0.into(), 3232.into()],
        )?;
        submit(&queue, encoder)?;
        promise(call(&staging, "mapAsync", &[1.into()])?).await?;
        let raw = Uint8Array::new(&call(&staging, "getMappedRange", &[])?).to_vec();
        call(&staging, "unmap", &[])?;
        call(&staging, "destroy", &[])?;
        let count = u32::from_le_bytes(raw[0..4].try_into().unwrap()).min(100) as usize;
        let mut gpu_boxes = Vec::new();
        for i in 0..count {
            let base = 32 + i * 32;
            let f = |offset: usize| {
                f32::from_le_bytes(raw[base + offset..base + offset + 4].try_into().unwrap())
            };
            gpu_boxes.push(super::postprocess::Detection {
                x1: f(0),
                y1: f(4),
                x2: f(8),
                y2: f(12),
                score: f(16),
                class_id: u32::from_le_bytes(raw[base + 20..base + 24].try_into().unwrap())
                    as usize,
            });
        }
        let cpu_boxes =
            super::postprocess::decode_and_nms(&values, shape, confidence, iou).map_err(err)?;
        let matches = gpu_boxes.len() == cpu_boxes.len()
            && gpu_boxes.iter().all(|a| {
                cpu_boxes.iter().any(|b| {
                    a.class_id == b.class_id
                        && (a.x1 - b.x1).abs() < 0.001
                        && (a.y1 - b.y1).abs() < 0.001
                        && (a.x2 - b.x2).abs() < 0.001
                        && (a.y2 - b.y2).abs() < 0.001
                        && (a.score - b.score).abs() < 0.00001
                })
            });
        {
            let mut s = self.state.borrow_mut();
            s.metrics.pixel_readbacks += 1;
            s.metrics.prediction_readbacks += 2;
        }
        let report = object([
            ("nmsMatchesCpu", matches.into()),
            (
                "gpuDetections",
                serde_wasm_bindgen::to_value(&gpu_boxes).map_err(err)?,
            ),
            (
                "cpuDetections",
                serde_wasm_bindgen::to_value(&cpu_boxes).map_err(err)?,
            ),
            (
                "headShape",
                serde_wasm_bindgen::to_value(&shape).map_err(err)?,
            ),
            (
                "input",
                js_sys::Float32Array::from(input_values.as_slice()).into(),
            ),
            ("head", js_sys::Float32Array::from(values.as_slice()).into()),
        ]);
        Ok(report)
    }
    pub fn metrics(&self) -> Result<JsValue, JsValue> {
        let mut s = self.state.borrow_mut();
        s.metrics.elapsed_ms = now_ms() - s.started;
        s.metrics.submitted_fps =
            s.metrics.submitted_frames as f64 / (s.metrics.elapsed_ms / 1000.0).max(0.001);
        serde_wasm_bindgen::to_value(&s.metrics).map_err(err)
    }
}
impl Drop for VideoPipeline {
    fn drop(&mut self) {
        let _ = self.stop();
        let s = self.state.borrow();
        if let Some(handler) = &s.error_handler {
            let _ = call(
                &s.gpu,
                "removeEventListener",
                &["uncapturederror".into(), handler.as_ref().clone()],
            );
        }
        let _ = call(&s.context, "unconfigure", &[]);
    }
}
