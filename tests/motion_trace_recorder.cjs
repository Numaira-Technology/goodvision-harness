const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const {test}=require("node:test"),{installMotionTraceCollector,createMotionTraceWriter}=require("../tools/motion_trace_recorder.cjs");
function collector(options={}){
  let args;
  const context=vm.createContext({TextEncoder,btoa,Uint8Array,GoodvisionCameraDisplay:{BrowserController:class{
    constructor(options){args=options;}
  }}});
  vm.runInContext(`(${installMotionTraceCollector.toString()})(${JSON.stringify({maxBytes:100000,maxSeconds:20,...options})})`,context);
  new context.GoodvisionCameraDisplay.BrowserController({});
  return {context,collect:args.onMotionTrace,options:args.motionTraceOptions};
}
function frame(id=0){return {type:"motion_trace_input",message:{type:"frame",trace_id:id,
  frame:{sequence:0,timestamp:0,width:2,height:2,pixels:new Uint8Array([1,2,3,4])}}};}
test("trace collector owns a separate private queue and makes exactly recoverable gray payloads",()=>{
  const view=collector();assert.equal(view.options.maxSeconds,20);
  const original=frame();assert.equal(view.collect(original),true);
  assert.equal(original.message.frame.pixels.byteLength,4);
  const batch=view.context.__drainMotionTrace();assert.equal(batch.records.length,1);
  assert.deepEqual([...Buffer.from(batch.records[0].message.frame.gray_base64,"base64")],[1,2,3,4]);
  assert.equal(view.context.__drainMotionTrace().records.length,0);assert.equal(batch.status.complete,true);
});
test("collector byte limits and stalled draining explicitly stop tracing without throwing",()=>{
  for(const options of [{maxBytes:10},{maxPendingBytes:10}]){
    const view=collector(options);assert.equal(view.collect(frame()),false);
    const batch=view.context.__drainMotionTrace();assert.equal(batch.status.complete,false);
    assert.equal(batch.status.reason,options.maxBytes?"collector_byte_limit":"collector_backpressure");
    assert.equal(batch.records.length,0);assert.equal(view.collect(frame()),false);
  }
  const view=collector();assert.equal(view.collect({type:"motion_trace_stopped",reason:"duration_limit"}),false);
  assert.equal(view.context.__drainMotionTrace().status.reason,"duration_limit");
});
test("binary writer records offsets, joins out-of-order responses, and marks missing responses incomplete",()=>{
  const folder=fs.mkdtempSync(path.resolve(__dirname,"../tmp/motion-trace-test-")),writer=createMotionTraceWriter(folder);
  const record=frame();record.message.frame.gray_base64=Buffer.from(record.message.frame.pixels).toString("base64");
  delete record.message.frame.pixels;
  writer.append({records:[{type:"motion_trace_output",message:{type:"frame_processed",trace_id:0}},record]});
  let summary=writer.finish({status:{complete:true},config:{}});assert.equal(summary.complete,true);
  assert.deepEqual([...fs.readFileSync(path.join(folder,"motion-trace-gray.bin"))],[1,2,3,4]);
  const rows=fs.readFileSync(path.join(folder,"motion-trace.jsonl"),"utf8").trim().split("\n").map(JSON.parse);
  assert.equal(rows[1].message.frame.gray_offset,0);assert.equal(rows[1].message.frame.gray_length,4);
  writer.append({records:[{type:"motion_trace_input",message:{type:"present",trace_id:1,sequence:0}}]});
  summary=writer.finish({status:{complete:true}});assert.equal(summary.complete,false);
  assert.deepEqual(summary.missing_output_ids,[1]);assert.equal(summary.reason,"unpaired_worker_messages");
});
test("writer never exceeds its disk budget and reports an incomplete diagnostic",()=>{
  const folder=fs.mkdtempSync(path.resolve(__dirname,"../tmp/motion-trace-test-")),writer=createMotionTraceWriter(folder,{maxBytes:10});
  assert.equal(writer.append({records:[{type:"motion_trace_output",message:{trace_id:0}}]}),"writer_byte_limit");
  const summary=writer.finish({status:{complete:true}});assert.equal(summary.complete,false);
  assert.equal(summary.bytes,0);assert.equal(summary.reason,"writer_byte_limit");
});

test("an older deployment without trace support cannot produce a falsely complete empty fixture",()=>{
  const folder=fs.mkdtempSync(path.resolve(__dirname,"../tmp/motion-trace-test-")),writer=createMotionTraceWriter(folder);
  const result=writer.finish({status:{complete:true}});
  assert.equal(result.complete,false);assert.equal(result.reason,"no_trace_messages");
});
