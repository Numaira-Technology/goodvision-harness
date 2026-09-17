"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
const {test}=require("node:test"),{loadTrace,replayTrace,resizeGray}=require("../tools/replay_camera_motion.cjs");
class History{
  constructor(){this.anchors=new Map();this.rejected=0;this.revision=0;this.calls=0;}
  appendFrame(frame){this.lastFrame=frame;return true;}
  acceptMetadata(data){for(const anchor of data.anchors)this.anchors.set(anchor.source_sequence,anchor);this.revision++;return true;}
  select(sequence,timestamp){this.calls++;return result(sequence,this.calls);}
}
const person={track_id:"p",polygons:[[[0,0],[1,0],[1,1]]]};
function result(sequence,calls){return {subjects:[{...person,complete:true,target_sequence:sequence}],coverage:"anchored",calls};}
function fixture(){
  const inputs=[{type:"frame",frame:{sequence:0,timestamp:0,width:2,height:2,gray_offset:0,gray_length:4}},
    {type:"metadata",data:{anchors:[{source_sequence:0,media_timestamp:0,epoch:"a",subjects:[person]}]}},
    {type:"present",id:0,sequence:0,timestamp:0},
    {type:"present",id:0,sequence:0,timestamp:0},
    {type:"frame",frame:{sequence:1,timestamp:.1,width:2,height:2,gray_offset:4,gray_length:4}},
    {type:"present",id:1,sequence:1,timestamp:.1}].map((input,trace_id)=>({...input,trace_id}));
  const messages=[{type:"frame_processed",sequence:0,accepted:true},
    {type:"metadata_processed",accepted:true,rejected:0,changed:true,latest_anchor_timestamp:0},
    {type:"present",id:0,sequence:0,expired:true,result:{subjects:[],coverage:"presentation_expired"}},
    {type:"present",id:0,sequence:0,result:result(0,1)},
    {type:"frame_processed",sequence:1,accepted:true},
    {type:"present",id:1,sequence:1,result:result(1,2)}];
  return {summary:{config:{}},inputs,outputs:new Map(messages.map((message,id)=>[id,{...message,trace_id:id}])),
    pixels:Buffer.from([1,2,3,4,5,6,7,8]),prefix:6,provenance:{}};
}
test("range selection rebuilds the complete prefix and joins repeated source queries by unique trace ID",()=>{
  const trace=fixture(),report=replayTrace(trace,{History,adapter:{},range:{fromSource:.1},includeResults:true});
  assert.equal(report.summary.replayed_inputs,6);assert.equal(report.summary.matching_responses,6);
  assert.equal(report.rows.length,1);assert.equal(report.rows[0].trace_id,5);
  assert.equal(report.rows[0].replayed_result.calls,2);assert.equal(report.summary.all_people_complete_ratio,1);
});
test("recorded expired queries never execute selection or alter subsequent cache state",()=>{
  const report=replayTrace(fixture(),{History,adapter:{}});
  assert.equal(report.summary.expired_queries,1);assert.equal(report.summary.selected_queries,3);
  assert.equal(report.rows[0].trace_id,2);assert.equal(report.rows[0].expired,true);
  assert.equal(report.summary.matching_responses,6);assert.equal(report.mismatches.length,0);
});
test("an incomplete trace replays only the paired prefix and reports actual response divergence",()=>{
  const trace=fixture();trace.prefix=4;trace.outputs.delete(4);trace.outputs.delete(5);
  trace.outputs.get(3).result={...result(0,1),coverage:"different"};
  const report=replayTrace(trace,{History,adapter:{}});assert.equal(report.summary.replayed_inputs,4);
  assert.equal(report.summary.mismatching_responses,1);assert.equal(report.mismatches[0].trace_id,3);
  assert.equal(report.mismatches[0].difference.path,"$.result.coverage");
});
function savedFixture(){
  const folder=fs.mkdtempSync(path.resolve(__dirname,"../tmp/motion-replay-test-")),trace=fixture(),assets={};
  for(const name of ["camera_display.js","camera_display_dense.js","opencv-4.13.0.js"]){
    const bytes=Buffer.from("// saved test asset "+name),file="motion-trace-"+name;fs.writeFileSync(path.join(folder,file),bytes);
    assets[name]={file,sha256:crypto.createHash("sha256").update(bytes).digest("hex")};
  }
  const rows=[];
  // Independent Workers can deliver their output before the tracing side-channel.
  for(const input of trace.inputs){rows.push({type:"motion_trace_output",message:trace.outputs.get(input.trace_id)});
    rows.push({type:"motion_trace_input",message:input});}
  const summary={schema:1,config:{},complete:true,files:{events:"events.jsonl",gray:"gray.bin"},
    provenance:{source_pixels:"browser_preprocess_exact",ordered_worker_input:true,source_assets:assets}};
  fs.writeFileSync(path.join(folder,"motion-trace-summary.json"),JSON.stringify(summary));
  fs.writeFileSync(path.join(folder,"events.jsonl"),rows.map(row=>JSON.stringify(row)).join("\n"));
  fs.writeFileSync(path.join(folder,"gray.bin"),trace.pixels);return {folder,summary,rows};
}
test("saved deployment hashes and binary offsets are checked before loading executable assets",()=>{
  const view=savedFixture(),trace=loadTrace(view.folder);assert.equal(trace.prefix,6);
  const file=path.join(view.folder,view.summary.provenance.source_assets["camera_display.js"].file);
  fs.appendFileSync(file,"modified");assert.throws(()=>loadTrace(view.folder),/differs from the recorded deployment/);
  const overridden=loadTrace(view.folder,{source:file});assert.equal(overridden.assets.source.override,true);
  fs.writeFileSync(path.join(view.folder,"gray.bin"),Buffer.alloc(1));
  assert.throws(()=>loadTrace(view.folder,{source:file}),/pixels are truncated/);
});
test("a missing input identity is an explicit gap, while an unpaired tail remains a usable prefix",()=>{
  const view=savedFixture();view.rows=view.rows.filter(row=>!(row.type==="motion_trace_output"&&row.message.trace_id===5));
  fs.writeFileSync(path.join(view.folder,"events.jsonl"),view.rows.map(row=>JSON.stringify(row)).join("\n"));
  assert.equal(loadTrace(view.folder).prefix,5);
  view.rows=view.rows.filter(row=>!(row.type==="motion_trace_input"&&row.message.trace_id===2));
  fs.writeFileSync(path.join(view.folder,"events.jsonl"),view.rows.map(row=>JSON.stringify(row)).join("\n"));
  assert.throws(()=>loadTrace(view.folder),/Missing, duplicate or reordered input/);
});

test("counterfactual resize preserves source identity and labels its inability to replay browser pixels exactly",()=>{
  const pixels=Uint8Array.from({length:32*32},(_,at)=>at%32),source={sequence:7,timestamp:.3,width:32,height:32,pixels};
  const small=resizeGray(source,16);assert.equal(small.sequence,7);assert.equal(small.timestamp,.3);
  assert.deepEqual([small.width,small.height],[16,16]);assert.equal(small.pixels[0],1);assert.equal(small.pixels[15],31);
  assert.equal(pixels.length,1024);assert.notEqual(small.pixels,pixels);
  const report=replayTrace(fixture(),{History,adapter:{},motionWidth:16});
  assert.equal(report.provenance.replay_mode,"counterfactual_resized_gray");
  assert.match(report.provenance.transformation.limitation,/Original expired queries stay expired/);
  assert.equal(report.summary.expired_queries,1);
});
