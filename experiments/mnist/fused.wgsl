// Exactly one workgroup per image: all dependencies stay within that workgroup.
@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
var<workgroup> tokens: array<f32, 256>;
var<workgroup> hidden: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    for (var i = lane; i < 256u; i += 64u) {
        let t = i / 16u;
        let c = i % 16u;
        var sum = weights[784u + c];
        for (var k = 0u; k < 49u; k++) {
            let pixel = (t / 4u * 7u + k / 7u) * 28u + t % 4u * 7u + k % 7u;
            sum += pixels[group.x * 784u + pixel] * weights[c * 49u + k];
        }
        tokens[i] = max(sum, 0.0);
    }
    workgroupBarrier();
    var sum = weights[17184u + lane];
    for (var k = 0u; k < 256u; k++) {
        sum += tokens[k] * weights[800u + lane * 256u + k];
    }
    hidden[lane] = max(sum, 0.0);
    workgroupBarrier();
    if (lane < 10u) {
        var result = weights[17888u + lane];
        for (var k = 0u; k < 64u; k++) {
            result += hidden[k] * weights[17248u + lane * 64u + k];
        }
        logits[group.x * 10u + lane] = result;
    }
}
