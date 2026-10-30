@group(0) @binding(0) var frame: texture_external;
@group(0) @binding(1) var frameSampler: sampler;
@group(0) @binding(2) var<storage, read> pixels: array<f32>;
@group(0) @binding(3) var<storage, read> logits: array<f32>;
@group(0) @binding(4) var<uniform> options: vec4<f32>;
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Vertex {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    var out: Vertex;
    out.position = vec4<f32>(p[i], 0, 1);
    out.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
    return out;
}
fn digit(uv: vec2<f32>, value: u32) -> bool {
    let masks = array<u32,10>(63u,6u,91u,79u,102u,109u,125u,7u,127u,111u);
    let boxes = array<vec4<f32>,7>(
        vec4<f32>(0.2,0.05,0.8,0.15), vec4<f32>(0.75,0.1,0.85,0.5),
        vec4<f32>(0.75,0.5,0.85,0.9), vec4<f32>(0.2,0.85,0.8,0.95),
        vec4<f32>(0.15,0.5,0.25,0.9), vec4<f32>(0.15,0.1,0.25,0.5),
        vec4<f32>(0.2,0.45,0.8,0.55));
    for (var i=0u; i<7u; i++) {
        let b=boxes[i];
        if ((masks[value] & (1u<<i)) != 0u && all(uv>=b.xy) && all(uv<=b.zw)) { return true; }
    }
    return false;
}
@fragment fn fs(in: Vertex) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0), vec2<f32>(0.99999));
    var color = textureSampleBaseClampToEdge(frame, frameSampler, uv).rgb;
    var best=0u;
    for(var i=1u;i<10u;i++) { if(logits[i]>logits[best]) {best=i;} }
    let crop=min(options.x,options.y)/options.xy;
    let local=(uv-vec2<f32>(0.5))/crop+vec2<f32>(0.5);
    if(all(local>=vec2<f32>(0)) && all(local<=vec2<f32>(1)) &&
       (any(local<vec2<f32>(0.006)) || any(local>vec2<f32>(0.994)))) {color=vec3<f32>(0.2,1,0.5);}
    // Bottom-left: exact 28x28 tensor consumed by inference.
    if(uv.x<0.24 && uv.y>0.68) {
        let p=vec2<u32>(clamp(vec2<f32>(uv.x/0.24,(uv.y-0.68)/0.32),vec2<f32>(0),vec2<f32>(0.99999))*28.0);
        color=vec3<f32>(pixels[p.y*28u+p.x]);
    }
    // Top-right: predicted digit, rendered directly from GPU logits.
    if(uv.x>0.82 && uv.y<0.28) {
        color=vec3<f32>(0.03,0.05,0.08);
        if(digit(vec2<f32>((uv.x-0.82)/0.18,uv.y/0.28),best)) {color=vec3<f32>(0.2,1,0.5);}
    }
    // Ten bottom bars show relative softmax scores. Argmax remains on GPU.
    if(uv.x>0.28 && uv.y>0.72) {
        let cell=(uv.x-0.28)/0.72*10.0;
        let c=min(u32(cell),9u);
        var total=0.0;
        for(var i=0u;i<10u;i++) {total+=exp(logits[i]-logits[best]);}
        let probability=exp(logits[c]-logits[best])/total;
        color=mix(color,vec3<f32>(0.02),0.75);
        if(fract(cell)>0.1 && fract(cell)<0.9 && uv.y>0.98-probability*0.24 && uv.y<0.98) {
            color=select(vec3<f32>(0.3,0.5,0.8),vec3<f32>(0.2,1,0.5),c==best);
        }
    }
    return vec4<f32>(color,1);
}
