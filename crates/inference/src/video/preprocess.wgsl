struct Config { source: vec2<f32>, resized: vec2<f32>, pad: vec2<f32>, confidence: f32, iou: f32, anchors: u32, classes: u32, channels_first: u32, unused: u32 }
@group(0) @binding(0) var frame: texture_external;
@group(0) @binding(1) var frame_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> input: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 if(id.x>=320u || id.y>=320u) {return;}
 let p=vec2<f32>(id.xy)+vec2<f32>(0.5);
 let local=p-config.pad;
 var rgb=vec3<f32>(114.0/255.0);
 if(all(local>=vec2<f32>(0)) && all(local<config.resized)) {
  rgb=textureSampleBaseClampToEdge(frame,frame_sampler,local/config.resized).rgb;
 }
 let index=id.y*320u+id.x;
 input[index]=rgb.r;input[102400u+index]=rgb.g;input[204800u+index]=rgb.b;
}
