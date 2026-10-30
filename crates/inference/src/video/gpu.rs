//! Thin browser API bindings. All WebGPU orchestration stays in Rust.
use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

pub fn get(target: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(target, &key.into())
}
pub fn object<const N: usize>(fields: [(&str, JsValue); N]) -> JsValue {
    let out = Object::new();
    for (key, value) in fields {
        Reflect::set(&out, &key.into(), &value).expect("plain object property");
    }
    out.into()
}
pub fn array(values: &[JsValue]) -> JsValue {
    values.iter().cloned().collect::<Array>().into()
}
pub fn call(target: &JsValue, method: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    get(target, method)?
        .dyn_into::<Function>()?
        .apply(target, &Array::from_iter(args.iter().cloned()))
}
pub async fn promise(value: JsValue) -> Result<JsValue, JsValue> {
    JsFuture::from(value.dyn_into::<Promise>()?).await
}
pub fn buffer(device: &JsValue, bytes: u32, usage: u32) -> Result<JsValue, JsValue> {
    call(
        device,
        "createBuffer",
        &[object([("size", bytes.into()), ("usage", usage.into())])],
    )
}
pub fn binding(binding: u32, resource: JsValue) -> JsValue {
    object([("binding", binding.into()), ("resource", resource)])
}
pub fn buffer_binding(buffer: JsValue) -> JsValue {
    object([("buffer", buffer)])
}
pub fn bind(device: &JsValue, layout: JsValue, entries: &[JsValue]) -> Result<JsValue, JsValue> {
    call(
        device,
        "createBindGroup",
        &[object([("layout", layout), ("entries", array(entries))])],
    )
}
pub async fn module(device: &JsValue, source: &str) -> Result<JsValue, JsValue> {
    let module = call(
        device,
        "createShaderModule",
        &[object([("code", source.into())])],
    )?;
    let info = promise(call(&module, "getCompilationInfo", &[])?).await?;
    for message in Array::from(&get(&info, "messages")?).iter() {
        if get(&message, "type")?.as_string().as_deref() == Some("error") {
            return Err(get(&message, "message")?);
        }
    }
    Ok(module)
}
pub async fn compute(device: &JsValue, source: &str) -> Result<JsValue, JsValue> {
    let module = module(device, source).await?;
    promise(call(
        device,
        "createComputePipelineAsync",
        &[object([
            ("layout", "auto".into()),
            (
                "compute",
                object([("module", module), ("entryPoint", "main".into())]),
            ),
        ])],
    )?)
    .await
}
pub fn layout(pipeline: &JsValue, group: u32) -> Result<JsValue, JsValue> {
    call(pipeline, "getBindGroupLayout", &[group.into()])
}
pub fn dispatch(
    encoder: &JsValue,
    pipeline: &JsValue,
    bind: &JsValue,
    x: u32,
    y: u32,
) -> Result<(), JsValue> {
    let pass = call(encoder, "beginComputePass", &[])?;
    call(&pass, "setPipeline", &[pipeline.clone()])?;
    call(&pass, "setBindGroup", &[0.into(), bind.clone()])?;
    call(&pass, "dispatchWorkgroups", &[x.into(), y.into()])?;
    call(&pass, "end", &[])?;
    Ok(())
}
pub fn submit(queue: &JsValue, encoder: JsValue) -> Result<(), JsValue> {
    let command = call(&encoder, "finish", &[])?;
    call(queue, "submit", &[array(&[command])])?;
    Ok(())
}
