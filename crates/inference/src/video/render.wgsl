struct Config { source: vec2<f32>, resized: vec2<f32>, pad: vec2<f32>, confidence: f32, iou: f32, anchors: u32, classes: u32, channels_first: u32, unused: u32 }
struct Candidate { bounds: vec4<f32>, score: f32, class_id: u32, suppressed: u32, unused: u32 }
struct Detections { count: u32, padding: vec3<u32>, boxes: array<Candidate,100> }
@group(0) @binding(0) var frame: texture_external;
@group(0) @binding(1) var frame_sampler: sampler;
@group(0) @binding(2) var<uniform> config: Config;
struct Vertex { @builtin(position) position:vec4<f32>, @location(0) uv:vec2<f32> }
@vertex fn background_vs(@builtin(vertex_index) i:u32)->Vertex {
 let p=array<vec2<f32>,3>(vec2<f32>(-1,-1),vec2<f32>(3,-1),vec2<f32>(-1,3));
 var out:Vertex;out.position=vec4<f32>(p[i],0,1);out.uv=vec2<f32>(p[i].x*.5+.5,.5-p[i].y*.5);return out;
}
@fragment fn background_fs(in:Vertex)->@location(0) vec4<f32> {return textureSampleBaseClampToEdge(frame,frame_sampler,in.uv);}
@group(1) @binding(0) var<storage,read> detections:Detections;
struct BoxVertex { @builtin(position) position:vec4<f32>, @location(0) @interpolate(flat) color:vec4<f32> }
@vertex fn boxes_vs(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->BoxVertex {
 var out:BoxVertex;out.position=vec4<f32>(2,2,0,1);out.color=vec4<f32>(0);
 if(instance>=detections.count){return out;}
 let item=detections.boxes[instance];let bounds=item.bounds;
 let corners=array<vec2<f32>,4>(bounds.xy,vec2<f32>(bounds.z,bounds.y),bounds.zw,vec2<f32>(bounds.x,bounds.w));
 let edge=vertex/6u;let index=vertex%6u;
 let a=(corners[edge]-config.pad)/config.resized;
 let b=(corners[(edge+1u)%4u]-config.pad)/config.resized;
 let direction=normalize((b-a)*config.source);
 let normal=vec2<f32>(-direction.y,direction.x)*1.5/config.source;
 let points=array<vec2<f32>,6>(a-normal,b-normal,a+normal,a+normal,b-normal,b+normal);
 let p=points[index];out.position=vec4<f32>(p.x*2.0-1.0,1.0-p.y*2.0,0,1);
 let hue=fract(f32(item.class_id)*0.618034);
 out.color=vec4<f32>(0.25+0.75*abs(sin(vec3<f32>(hue,hue+.333,hue+.666)*6.283)),1);
 return out;
}
@fragment fn boxes_fs(in:BoxVertex)->@location(0) vec4<f32>{return in.color;}
