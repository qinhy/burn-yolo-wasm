struct Config { source: vec2<f32>, resized: vec2<f32>, pad: vec2<f32>, confidence: f32, iou: f32, anchors: u32, classes: u32, channels_first: u32, unused: u32 }
struct Candidate { bounds: vec4<f32>, score: f32, class_id: u32, suppressed: u32, unused: u32 }
struct Detections { count: u32, padding: vec3<u32>, boxes: array<Candidate,100> }
@group(0) @binding(0) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(1) var<storage, read_write> detections: Detections;
@group(0) @binding(2) var<uniform> config: Config;
fn overlap(a:vec4<f32>,b:vec4<f32>)->f32 {
 let xy=max(a.xy,b.xy);let zw=min(a.zw,b.zw);let size=max(zw-xy,vec2<f32>(0));
 let intersection=size.x*size.y;let aa=a.zw-a.xy;let bb=b.zw-b.xy;
 return intersection/max(aa.x*aa.y+bb.x*bb.y-intersection,1e-10);
}
// Exact class-aware greedy NMS, bounded like the CPU reference: top 1200, at most 100 kept.
// Serial selection is intentionally simple; profile this stage before optimizing.
@compute @workgroup_size(1)
fn main() {
 detections.count=0u;
 for(var step=0u;step<min(config.anchors,1200u);step++) {
  var best=0u;var score=-1.0;
  for(var i=0u;i<config.anchors;i++){if(candidates[i].score>score){best=i;score=candidates[i].score;}}
  if(score<0.0){break;}
  let chosen=candidates[best];candidates[best].score=-1.0;
  if(chosen.suppressed!=0u){continue;}
  detections.boxes[detections.count]=chosen;detections.count++;
  if(detections.count>=100u){break;}
  for(var i=0u;i<config.anchors;i++) {
   if(candidates[i].score>=0.0 && candidates[i].class_id==chosen.class_id && overlap(chosen.bounds,candidates[i].bounds)>config.iou){candidates[i].suppressed=1u;}
  }
 }
}
