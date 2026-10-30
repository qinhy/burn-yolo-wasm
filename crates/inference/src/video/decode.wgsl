struct Config { source: vec2<f32>, resized: vec2<f32>, pad: vec2<f32>, confidence: f32, iou: f32, anchors: u32, classes: u32, channels_first: u32, unused: u32 }
struct Candidate { bounds: vec4<f32>, score: f32, class_id: u32, suppressed: u32, unused: u32 }
@group(0) @binding(0) var<storage, read> head: array<f32>;
@group(0) @binding(1) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(2) var<uniform> config: Config;
fn at(c:u32,i:u32)->f32 {return head[select(i*(config.classes+4u)+c,c*config.anchors+i,config.channels_first!=0u)];}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;if(i>=config.anchors){return;}
 var score=config.confidence;var found=false;var class_id=0u;
 for(var c=0u;c<config.classes;c++) {let s=at(c+4u,i);if(s>=score && s<=1.0){score=s;class_id=c;found=true;}}
 let center=vec2<f32>(at(0u,i),at(1u,i));let size=vec2<f32>(at(2u,i),at(3u,i));
 let bounds=clamp(vec4<f32>(center-size*0.5,center+size*0.5),vec4<f32>(0),vec4<f32>(320));
 let valid=found&&all(size>vec2<f32>(0))&&all(bounds.zw>bounds.xy)&&all(abs(center)<vec2<f32>(1e10))&&all(size<vec2<f32>(1e10));
 candidates[i]=Candidate(bounds,select(-1.0,score,valid),class_id,0u,0u);
}
