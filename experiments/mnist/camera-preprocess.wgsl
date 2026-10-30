@group(0) @binding(0) var frame: texture_external;
@group(0) @binding(1) var frameSampler: sampler;
@group(0) @binding(2) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(3) var<uniform> options: vec4<f32>; // width, height, invert, unused

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 28u || id.y >= 28u) { return; }
    // MNIST-like margin: fit the center camera square inside 20x20 pixels.
    var gray = 0.0;
    if (id.x >= 4u && id.x < 24u && id.y >= 4u && id.y < 24u) {
        let local = (vec2<f32>(id.xy) - vec2<f32>(4.0) + vec2<f32>(0.5)) / 20.0;
        let crop = min(options.x, options.y) / options.xy;
        let uv = vec2<f32>(0.5) + (local - vec2<f32>(0.5)) * crop;
        let rgb = textureSampleBaseClampToEdge(frame, frameSampler, uv).rgb;
        gray = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
        if (options.z > 0.5) { gray = 1.0 - gray; }
    }
    pixels[id.y * 28u + id.x] = clamp(gray, 0.0, 1.0);
}
