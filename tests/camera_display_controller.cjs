const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const {test}=require("node:test");
const code=fs.readFileSync(require.resolve("../src/goodvision_harness/camera_display.js"),"utf8");
const settle=()=>new Promise(resolve=>setImmediate(resolve));

test("simultaneous displays have distinct stable telemetry identities",()=>{
  const view=browser(),second=view.createController();
  const firstId=view.controller.snapshot().display_instance_id;
  assert.match(firstId,/^[A-Za-z0-9_-]{1,96}$/);
  assert.notEqual(second.snapshot().display_instance_id,firstId);
  view.render(1);assert.equal(view.controller.snapshot().display_instance_id,firstId);
  view.controller.stop();second.stop();
});

test("network source preserves server frame IDs, native pixels and playback cadence without uploading analysis",()=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:.5}});
  for(let i=0;i<15;i++){
    view.setClock(i/15);
    view.controller.capture(i/15*1000,{mediaTime:50+i/15,presentedFrames:100+i},
      {width:1920,height:1080,origin:i});
    view.render(i/15+.5);
  }
  assert.equal(view.controller.snapshot().distinct_presented_frames,15);
  assert.equal(view.canvas.dataset.frameSequence,"114");
  assert.equal(view.canvas.width,1920);assert.equal(view.canvas.height,1080);
  assert.equal(view.controller.snapshot().analysis_admitted,0);
  assert.equal(view.packets.length,0);
  assert.equal(view.workerMessages.filter(x=>x.type==="frame_bitmap").length,2);
  assert.equal(view.controller.snapshot().motion_frames_skipped,13);
  assert.ok(Math.abs(view.controller.snapshot().last_source_to_present_seconds-.5)<1e-6);
  view.controller.stop();
});

test("network batch framing rejects truncated or ambiguous source identities",()=>{
  const view=browser();
  const header={protocol:1,connection_id:"connection",config:{stream_session_id:"run",generation:2,crop_version:0},
    frames:[{source_sequence:33,media_timestamp:12.4,width:640,height:480,bytes:4}]};
  const json=new TextEncoder().encode(JSON.stringify(header)),bytes=new Uint8Array(8+json.length+4);
  bytes.set(new TextEncoder().encode("GVD1"));new DataView(bytes.buffer).setUint32(4,json.length);bytes.set(json,8);
  const decoded=view.api.decodeDisplayBatch(bytes.buffer);
  assert.equal(decoded.frames[0].source_sequence,33);assert.equal(decoded.frames[0].jpeg.length,4);
  assert.throws(()=>view.api.decodeDisplayBatch(bytes.buffer.slice(0,-1)),/invalid display frame/);
  bytes[0]=0;assert.throws(()=>view.api.decodeDisplayBatch(bytes.buffer),/invalid display header/);
  view.controller.stop();
});

test("network run and crop changes dispose prior overlays and metadata cannot cross runs",async()=>{
  const view=browser({globals:{createImageBitmap:async()=>({width:640,height:480,close(){}})}});
  const feed=new view.api.NetworkController({canvas:view.canvas,url:"/frames",metricsUrl:"/metrics"});
  const packet=(run,seq,crop=0)=>({connection_id:"connection",config:{stream_session_id:run,generation:2,crop_version:crop,
    enabled:true,delay_seconds:.5,target_fps:15},frames:[{source_sequence:seq,media_timestamp:seq/15,
      width:640,height:480,jpeg:new Uint8Array([1])}]});
  await feed.accept(packet("run-one",1));const first=feed.controller;
  first.overlays.set(1,{subjects:[{track_id:"old"}]});
  await feed.accept(packet("run-two",2));
  assert.equal(first.closed,true);assert.equal(first.overlays.size,0);
  assert.equal(feed.controller.receiveMetadata({stream_session_id:"run-one",generation:2,crop_version:0,anchors:[]}),false);
  const second=feed.controller;await feed.accept(packet("run-two",3,1));assert.equal(second.closed,true);
  feed.stop();view.controller.stop();
});
test("raw network frames report their actual canvas presentation without tracker telemetry",async()=>{
  const view=browser({globals:{createImageBitmap:async()=>({width:640,height:480,origin:42,close(){}})}});
  const observed=[];const feed=new view.api.NetworkController({canvas:view.canvas,url:"/frames",
    onFrame:value=>observed.push({value,origin:view.canvas.origin})});
  await feed.accept({connection_id:"connection",config:{stream_session_id:"raw-run",generation:1,crop_version:0,enabled:false,delay_seconds:0},
    frames:[{source_sequence:1,media_timestamp:1,width:640,height:480,jpeg:new Uint8Array([1])}]});
  assert.equal(observed.length,1);assert.equal(observed[0].value.presented,true);
  assert.equal(observed[0].value.buffering,false);assert.equal(observed[0].origin,42);
  assert.equal(feed.controller,null);feed.stop();view.controller.stop();
});

function browser({autoAck=true,metadataResponse,config={},globals={},settings:providedSettings,onMotionTrace=null,motionTraceOptions={}}={}){
  let clock=0,next=0;const encodings=[],packets=[],draws=[],labels=[],labelBoxes=[],workerMessages=[],workers=[],raf=new Map(),errors=[],reports=[];
  class Canvas{
    constructor(width,height){this.width=width;this.height=height;this.dataset={};}
    getContext(){return {drawImage:source=>{draws.push(clock);this.origin=source===video?++video.reads:source.origin;},
      beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){},clearRect:()=>{this.cleared=true;},
      save(){},restore(){},measureText:text=>({width:Array.from(text).length*14}),
      fillText:(text,x,y,maxWidth)=>labels.push({text,x,y,maxWidth}),
      fillRect:(x,y,width,height)=>labelBoxes.push({x,y,width,height}),
      getImageData:()=>({data:new Uint8Array(this.width*this.height*4).fill(100)})};}
    transferToImageBitmap(){return {width:this.width,height:this.height,origin:this.origin,close(){this.closed=true;}};}
    convertToBlob(){return new Promise(resolve=>encodings.push(()=>resolve({arrayBuffer:async()=>new Uint8Array([255,216,255,217]).buffer})));}
  }
  const video={videoWidth:640,videoHeight:480,reads:0,requestVideoFrameCallback:()=>1,cancelVideoFrameCallback(){}};
  const socket={readyState:1,bufferedAmount:0,send:packet=>{packets.push(packet);
    if(autoAck&&typeof packet!=="string")queueMicrotask(()=>controller.acknowledgeUpload({...unpack(packet),type:"camera_frame_ack",accepted:true}));}};
  const context=vm.createContext({module:{exports:{}},OffscreenCanvas:Canvas,TextEncoder,TextDecoder,ArrayBuffer,Blob,URLSearchParams,Uint8Array,DataView,
    MessageChannel:class{constructor(){this.port1={};this.port2={};}},
    Worker:class{constructor(){workers.push(this);this.messages=[];}
      postMessage(message){workerMessages.push(message);this.messages.push(message);
      if(message.type==="encode_jpeg")encodings.push(()=>this.onmessage({data:{type:"encoded_jpeg",
        sequence:message.sequence,bytes:new Uint8Array([255,216,255,217]).buffer}}));
    }terminate(){this.terminated=true;}},
    Date:class extends Date{static now(){return 1700000000000+clock;}},
    performance:{now:()=>clock},requestAnimationFrame:callback=>{raf.set(++next,callback);return next;},
    cancelAnimationFrame:id=>raf.delete(id),setTimeout,clearTimeout,structuredClone,AbortController,
    fetch:async(url,options)=>{reports.push({url,options});
      return metadataResponse&&url.endsWith("/metadata")?metadataResponse(options):{ok:true};},...globals});
  context.GoodvisionEvents=require("../src/goodvision_harness/assets/event_catalog.js");
  context.GoodvisionEvents.accept(require("./helpers/event_catalog.cjs").payload());
  vm.runInContext(code,context);
  const canvas=new Canvas(1,1);
  const settings=providedSettings||{sample_fps:2.5,max_image_side:640,jpeg_quality:85};
  const api=context.module.exports;
  const createController=(extra={})=>new api.BrowserController({video,canvas,socket,
    config:{stream_session_id:"run",generation:2,crop_version:0,delay_seconds:.5,target_fps:15,analysis_sample_fps:3,...config,...extra},
    settings,onMotionTrace,motionTraceOptions,onFailure:error=>errors.push(error)});
  const controller=createController();
  return {controller,createController,api,workers,video,socket,canvas,packets,encodings,workerMessages,errors,reports,settings,labels,labelBoxes,
    capture:time=>{clock=time*1000;controller.capture(clock,{mediaTime:time,presentedFrames:Math.round(time*1000)+1});},
    render:time=>{clock=time*1000;controller.animate();},
    finish:async()=>{encodings.shift()();await settle();},
    setClock:time=>{clock=time*1000;}};
}
function unpack(packet){const bytes=new Uint8Array(packet),size=new DataView(packet).getUint32(4);
  return JSON.parse(new TextDecoder().decode(bytes.slice(8,size+8)));}

test("original-resolution playback draws one readable ID and Chinese event label per person inside the image",()=>{
  const view=browser();view.video.videoWidth=1920;view.video.videoHeight=1080;view.canvas.clientWidth=960;
  view.capture(0);
  view.controller.overlays.set(0,{coverage:"anchored",subjects:[{target_sequence:0,track_id:"0",
    tracking_confirmation:"confirmed",alert:true,event_labels:["smoking","fall"],complete:true,
    polygons:[[[.98,0],[1,.1],[.95,.2]],[[.95,.3],[1,.3],[1,.4]]]}]});
  view.render(.5);
  assert.deepEqual([view.canvas.width,view.canvas.height],[1920,1080]);
  assert.deepEqual(view.labels.map(label=>label.text),["ID 0","吸烟 · 跌倒"]);
  assert.match(view.controller.context.font,/28px/);
  for(const box of view.labelBoxes){assert.ok(box.x>=0&&box.y>=0);
    assert.ok(box.x+box.width<=1920&&box.y+box.height<=1080);}
  view.capture(.6);view.controller.overlays.set(1,{subjects:[{target_sequence:1,track_id:"0",
    tracking_confirmation:"confirmed",alert:false,event_labels:[],polygons:[[[.5,.5],[.6,.5],[.6,.6]]]}]});
  view.render(1.1);assert.deepEqual(view.labels.map(label=>label.text),["ID 0","吸烟 · 跌倒","ID 0"]);
  assert.equal(view.errors.length,0);view.controller.stop();
});

test("open contours keep IDs, unconfirmed or absent contours have no event text, and long labels fit",()=>{
  const view=browser();view.capture(0);
  view.controller.overlays.set(0,{subjects:[{target_sequence:0,track_id:"p".repeat(100),
    tracking_confirmation:"provisional",alert:true,event_labels:["weapon"],polygons:[],
    open_segments:[[[.99,.95],[1,1]]]},{target_sequence:0,track_id:"invisible",polygons:[],open_segments:[]},
    {target_sequence:99,track_id:"stale",polygons:[[[0,0],[1,0],[1,1]]]}]});
  view.render(.5);assert.equal(view.labels.length,1);assert.ok(view.labels[0].text.startsWith("ID p"));
  assert.ok(view.labels[0].text.endsWith("…"));
  for(const box of view.labelBoxes)assert.ok(box.x>=0&&box.x+box.width<=640&&box.y>=0&&box.y+box.height<=480);
  view.controller.stop();
});

test("Canvas prefers a confirmed source-aligned gallery name and preserves Chinese event labels",t=>{
  const view=browser();t.after(()=>view.controller.stop());view.capture(0);view.capture(.34);
  view.controller.receiveMetadata({...view.controller.config,anchors:[],identity_snapshot_revision:1,
    identity_snapshot_timestamp:.34,identities:[{epoch:"epoch",track_id:"0",person_id:"gallery-person",
      name:"测试姓名",valid_from_timestamp:.2,valid_until_timestamp:null,revision:1}]});
  for(const sequence of [0,1])view.controller.overlays.set(sequence,{coverage:"anchored",subjects:[{
    target_sequence:sequence,epoch:"epoch",track_id:"0",tracking_confirmation:"confirmed",complete:true,
    polygons:[[[.2,.2],[.5,.2],[.5,.5]]],display_name:"stale worker name",alert:true,event_labels:["smoking"],
  }]});
  view.render(.5);assert.deepEqual(view.labels.map(item=>item.text),["ID 0","吸烟"]);
  view.render(.85);assert.deepEqual(view.labels.map(item=>item.text),["ID 0","吸烟","测试姓名","吸烟"]);
});

test("gallery revocation clears names from an already prepared overlay and stale metadata cannot restore them",t=>{
  const view=browser();t.after(()=>view.controller.stop());view.capture(0);
  const known={...view.controller.config,anchors:[],identity_snapshot_revision:1,identity_snapshot_timestamp:0,
    identities:[{epoch:"epoch",track_id:"0",person_id:"gallery-person",name:"测试姓名",
      valid_from_timestamp:0,valid_until_timestamp:null,revision:1}]};
  view.controller.receiveMetadata(known);
  view.controller.overlays.set(0,{coverage:"anchored",subjects:[{target_sequence:0,epoch:"epoch",track_id:"0",
    tracking_confirmation:"confirmed",complete:true,display_name:"测试姓名",polygons:[[[.2,.2],[.5,.2],[.5,.5]]]}]});
  view.controller.receiveMetadata({...known,identity_snapshot_revision:2,identities:[]});
  view.controller.receiveMetadata(known);
  view.render(.5);assert.deepEqual(view.labels.map(item=>item.text),["ID 0"]);
  assert.equal(view.controller.snapshot().complete_contour_frames,1);
  view.controller.stop();assert.equal(view.controller.identities.records.length,0);
});

test("preview owns at most one unused worker and a dense run can claim it while it is still warming",async()=>{
  const view=browser({config:{motion_backend:"dense"}});view.controller.stop();
  const {BrowserController}=view.api,before=view.workers.length;
  const warmed=BrowserController.warmupMotion(),worker=view.workers.at(-1);
  assert.equal(BrowserController.warmupMotion(),warmed);assert.equal(view.workers.length,before+1);
  assert.deepEqual(worker.messages.map(message=>message.type),["warmup_motion"]);
  view.setClock(2);const run=view.createController({stream_session_id:"new",generation:3});
  assert.equal(run.worker,worker);assert.equal(view.workers.length,before+3);
  assert.equal(run.snapshot().motion_warmup_reused,1);
  assert.deepEqual(worker.messages.map(message=>message.type),["warmup_motion","motion_input"]);
  assert.equal(worker.messages[1].config.stream_session_id,"new");
  BrowserController.discardWarmup();assert.notEqual(worker.terminated,true);
  worker.onmessage({data:{type:"motion_warmup_ready",runtime_load_ms:3000}});assert.equal(await warmed,true);
  view.setClock(2.5);worker.onmessage({data:{type:"motion_runtime_ready",runtime_load_ms:3000}});
  assert.equal(run.snapshot().motion_runtime_ready_elapsed_seconds,.5);
  assert.equal(run.snapshot().motion_runtime_load_seconds,3);assert.equal(run.snapshot().dense_motion_ready,0);
  view.setClock(3);worker.onmessage({data:{type:"motion_ready",backend:"dense",startup_drain_ms:450}});
  assert.equal(run.snapshot().motion_ready_elapsed_seconds,1);assert.equal(run.snapshot().motion_startup_drain_seconds,.45);
  assert.equal(run.startedAt,2000);assert.equal(run.lastMetricsAt,2000);
  run.stop();assert.equal(worker.terminated,true);
  const next=view.createController({stream_session_id:"next",generation:4});
  assert.notEqual(next.worker,worker);assert.equal(next.snapshot().motion_warmup_reused,0);next.stop();
});

test("completed preview warmup is claimed once and a stopped pending claim resolves without reuse",async()=>{
  const view=browser({config:{motion_backend:"dense"}});view.controller.stop();
  const {BrowserController}=view.api,warmed=BrowserController.warmupMotion(),worker=view.workers.at(-1);
  worker.onmessage({data:{type:"motion_warmup_ready"}});assert.equal(await warmed,true);
  const run=view.createController();assert.equal(run.worker,worker);assert.equal(run.overlays.size,0);
  assert.equal(run.uploads.length,0);assert.equal(run.frames.frames.length,0);run.stop();
  const pending=BrowserController.warmupMotion(),pendingWorker=view.workers.at(-1),second=view.createController();
  assert.equal(second.worker,pendingWorker);second.stop();assert.equal(await pending,false);
  assert.equal(pendingWorker.terminated,true);
});

test("discarded preview warmup releases its worker, ignores late failures and can start fresh",async()=>{
  const view=browser();view.controller.stop();const {BrowserController}=view.api,failures=[];
  const warmed=BrowserController.warmupMotion({onFailure:error=>failures.push(error)}),worker=view.workers.at(-1);
  BrowserController.discardWarmup();BrowserController.discardWarmup();assert.equal(await warmed,false);
  assert.equal(worker.terminated,true);worker.onerror();worker.onmessage({data:{type:"motion_warmup_ready"}});
  assert.equal(failures.length,0);
  const retry=BrowserController.warmupMotion();assert.notEqual(view.workers.at(-1),worker);
  BrowserController.discardWarmup();assert.equal(await retry,false);
});

test("preview warmup failure is visible once and a retry or run uses a new worker",async()=>{
  const view=browser({config:{motion_backend:"dense"}});view.controller.stop();
  const {BrowserController}=view.api,failures=[];
  const warmed=BrowserController.warmupMotion({onFailure:error=>failures.push(error)}),worker=view.workers.at(-1);
  worker.onmessage({data:{type:"failure",stage:"motion_warmup"}});worker.onerror();
  assert.equal(await warmed,false);assert.equal(failures.length,1);assert.match(failures[0],/预热失败/);
  assert.equal(worker.terminated,true);
  const run=view.createController();assert.notEqual(run.worker,worker);run.stop();
  const retry=BrowserController.warmupMotion();assert.notEqual(view.workers.at(-1),worker);
  view.workers.at(-1).onmessage({data:{type:"motion_warmup_ready"}});assert.equal(await retry,true);
  BrowserController.discardWarmup();
});

test("a claimed warmup failure is reported by run initialization and stopping settles the promise",async()=>{
  const view=browser({config:{motion_backend:"dense"}});view.controller.stop();
  const {BrowserController}=view.api,warmed=BrowserController.warmupMotion();
  const run=view.createController();run.worker.onmessage({data:{type:"failure",stage:"motion_warmup"}});
  assert.equal(await warmed,false);assert.equal(view.errors.length,0);
  run.worker.onmessage({data:{type:"failure",stage:"motion_initialization"}});
  assert.equal(view.errors.length,1);assert.equal(run.snapshot().motion_initialization_failures,1);run.stop();
});

test("motion trace is opt-in, and a stopped or failing collector cannot stop real capture",()=>{
  const ordinary=browser();assert.equal(Object.hasOwn(ordinary.workerMessages.find(message=>message.type==="preprocess_output"),"trace_enabled"),false);
  ordinary.controller.stop();
  for(const throws of [false,true]){
    const received=[],view=browser({onMotionTrace:record=>{received.push(record);if(throws)throw new Error("private storage failed");return false;},
      motionTraceOptions:{maxBytes:10000,maxSeconds:3}});
    const init=view.workerMessages.find(message=>message.type==="preprocess_output");
    assert.equal(init.trace_enabled,true);assert.equal(init.trace_max_bytes,10000);assert.equal(init.trace_max_seconds,3);
    view.controller.worker.onmessage({data:{type:"frame_processed",trace_id:4,sequence:0,accepted:true,compute_ms:1}});
    assert.equal(received[0].type,"motion_trace_output");assert.equal(received[0].message.trace_id,4);
    assert.equal(view.controller.snapshot().worker_frames_completed,1);
    assert.equal(view.workerMessages.at(-1).type,"stop_motion_trace");
    if(throws)assert.equal(view.controller.motionTraceFailure,"consumer_failure");
    view.capture(0);assert.equal(view.controller.snapshot().captured_frames,1);assert.equal(view.errors.length,0);
    view.controller.preprocessWorker.onmessage({data:{type:"motion_trace_stopped",reason:"requested"}});
    assert.equal(received.length,1);view.controller.stop();
  }
});

test("canvas presentation keeps 15 unique frames while all model metadata is absent",()=>{
  const view=browser(),seen=[];
  for(let i=0;i<40;i++){
    view.capture(i/15);view.render(i/15);
    if(view.canvas.dataset.frameSequence!==undefined&&seen.at(-1)!==view.canvas.dataset.frameSequence)
      seen.push(view.canvas.dataset.frameSequence);
  }
  assert.ok(seen.length>=32);assert.equal(new Set(seen).size,seen.length);
  assert.ok(Math.abs(view.controller.snapshot().distinct_presented_fps-15)<1e-6);
  assert.equal(view.controller.snapshot().propagated_contour_frames,0);
  assert.equal(view.controller.snapshot().frames_without_contours,seen.length);
  view.controller.stop();
});
test("display and analysis share one frozen native source even when the decoder advances on every read",()=>{
  const view=browser();view.video.videoWidth=1920;view.video.videoHeight=1080;
  view.controller.settings.max_image_side=1280;view.capture(0);
  const display=view.controller.frames.frames[0].bitmap,analysis=view.controller.uploads[0].image;
  assert.equal(view.video.reads,1);assert.equal(display.origin,analysis.origin);
  assert.deepEqual([display.width,display.height],[1920,1080]);
  assert.deepEqual([analysis.width,analysis.height],[1280,720]);
  view.capture(1/15);view.capture(.34);
  assert.equal(view.video.reads,3);assert.equal(analysis.origin,1);
  assert.equal(view.controller.uploads[1].image.origin,3);
  view.controller.stop();
});
test("slow JPEG work keeps every admitted analysis source in FIFO and drain sends them before stop",async()=>{
  const view=browser();view.capture(0);view.capture(.34);view.capture(.68);
  assert.equal(view.controller.snapshot().analysis_admitted,3);assert.equal(view.packets.length,0);
  let completed=false;const drain=view.controller.drain().then(()=>completed=true);
  await settle();assert.equal(completed,false);
  await view.finish();await view.finish();await view.finish();await drain;
  assert.deepEqual(view.packets.map(packet=>unpack(packet).source_sequence),[0,1,2]);
  assert.deepEqual(view.packets.map(packet=>unpack(packet).media_timestamp),[0,.34,.68]);
  assert.equal(view.controller.snapshot().analysis_pending,0);assert.equal(completed,true);
  view.controller.stop();
});
test("only a contour for the displayed frame is drawn, never a late different source",()=>{
  const view=browser();view.capture(0);
  const polygon=[[[.1,.1],[.2,.1],[.2,.2]]];
  view.controller.overlays.set(0,{subjects:[{target_sequence:90,polygons:polygon}],exact:false,coverage:"anchored"});
  let strokes=0;view.controller.context.stroke=()=>strokes++;
  view.render(.5);assert.equal(strokes,0);assert.equal(view.canvas.dataset.frameSequence,"0");
  view.controller.stop();
});

test("partial boundaries stay open and cannot inflate complete contour coverage",()=>{
  const view=browser();view.capture(0);let closes=0,strokes=0;
  view.controller.context.closePath=()=>closes++;view.controller.context.stroke=()=>strokes++;
  view.controller.overlays.set(0,{subjects:[{target_sequence:0,polygons:[],
    open_segments:[[[.1,.1],[.2,.1],[.2,.2]]],complete:false}],exact:false,coverage:"motion_unreliable"});
  view.render(.5);assert.equal(strokes,1);assert.equal(closes,0);
  assert.equal(view.controller.snapshot().partial_contour_frames,1);
  assert.equal(view.controller.snapshot().complete_contour_frames,0);
  assert.equal(view.controller.snapshot().frames_without_contours,0);view.controller.stop();
});

test("closed interpolation is displayed and counted separately from strict and all-person coverage",()=>{
  for(const allPresent of [false,true]){
    const view=browser();view.capture(0);let closes=0;view.controller.context.closePath=()=>closes++;
    view.controller.overlays.set(0,{subjects:[{target_sequence:0,polygons:[[[.1,.1],[.2,.1],[.2,.2]]],
      complete:false,approximate:true}],exact:false,coverage:allPresent?"interpolated":"motion_unreliable",
      all_subjects_present:allPresent});
    view.render(.5);const stats=view.controller.snapshot();assert.equal(closes,1);
    assert.equal(stats.approximate_contour_frames,1);assert.equal(stats.approximate_subjects,1);
    assert.equal(stats.continuous_contour_frames,allPresent?1:0);assert.equal(stats.complete_contour_frames,0);
    assert.equal(stats.frames_without_contours,0);view.controller.stop();
  }
});

test("metadata arriving before presentation refreshes an already queried empty overlay",()=>{
  const view=browser();view.capture(0);view.render(.4);
  const worker=view.controller.worker,polygon=[[[.1,.1],[.2,.1],[.2,.2]]];
  assert.equal(view.workerMessages.filter(message=>message.type==="present").length,1);
  worker.onmessage({data:{type:"present",sequence:0,result:{subjects:[],coverage:"no_anchor"}}});
  worker.onmessage({data:{type:"metadata_processed",accepted:true,rejected:0,changed:true,geometry_changed:true}});
  view.render(.45);
  assert.equal(view.workerMessages.filter(message=>message.type==="present").length,2);
  worker.onmessage({data:{type:"present",sequence:0,result:{subjects:[{
    target_sequence:0,tracking_confirmation:"confirmed",polygons:polygon,
  }],coverage:"anchored",exact:true}}});
  let strokes=0;view.controller.context.stroke=()=>strokes++;
  view.render(.5);assert.equal(strokes,1);
  assert.equal(view.controller.snapshot().frames_without_contours,0);
  view.controller.stop();
});

test("identical metadata polls do not enqueue duplicate pending display work",()=>{
  const view=browser();view.capture(0);view.render(.2);
  assert.equal(view.workerMessages.filter(message=>message.type==="present").length,1);
  for(let i=0;i<5;i++){
    view.controller.worker.onmessage({data:{type:"metadata_processed",accepted:true,changed:false,geometry_changed:false,rejected:0}});
    view.render(.21+i*.01);
  }
  assert.equal(view.workerMessages.filter(message=>message.type==="present").length,1);
  view.controller.stop();
});

test("long display buffering prepares exact-source contours before a 600ms worker delay and keeps the real playback deadline",t=>{
  const view=browser({config:{delay_seconds:6,target_fps:20}});t.after(()=>view.controller.stop());
  const queries=()=>view.workerMessages.filter(message=>message.type==="present");
  view.capture(0);view.capture(.05);view.render(4.49);assert.equal(queries().length,0);
  view.render(4.5);assert.deepEqual(queries().map(message=>message.sequence),[0]);
  view.render(4.56);assert.deepEqual(queries().map(message=>message.sequence),[0,1]);
  assert.deepEqual(queries().map(message=>message.timestamp),[0,.05]);
  for(const query of queries())assert.ok(Math.abs(query.deadline_epoch_ms-
    (1700000000000+(query.timestamp+6)*1000+100))<1e-6);
  assert.equal(view.controller.snapshot().distinct_presented_frames,0);
  const polygon=[[[.1,.1],[.2,.1],[.2,.2]]];let strokes=0;
  view.controller.context.stroke=()=>strokes++;
  view.setClock(5.16);
  for(const query of queries())view.controller.worker.onmessage({data:{...query,type:"present",result:{subjects:[{
    target_sequence:query.sequence,track_id:"0",tracking_confirmation:"confirmed",polygons:polygon,
    complete:true,alert:true,event_labels:["smoking"],
  }],coverage:"anchored",exact:true}}});
  view.render(5.9);assert.equal(strokes,0);assert.equal(view.labels.length,0);
  view.render(6);assert.equal(strokes,1);assert.equal(view.canvas.dataset.frameSequence,"0");
  view.render(6.06);assert.equal(strokes,2);assert.equal(view.canvas.dataset.frameSequence,"1");
  assert.equal(view.controller.snapshot().frames_without_contours,0);
  assert.equal(view.controller.snapshot().complete_contour_frames,2);
});

test("geometry updates coalesce while a source query is in flight and refresh once after its stale result",t=>{
  const view=browser({config:{delay_seconds:6,target_fps:20}});t.after(()=>view.controller.stop());
  const queries=()=>view.workerMessages.filter(message=>message.type==="present"),worker=view.controller.worker;
  view.capture(0);view.render(5.7);assert.equal(queries().length,1);
  const initial=queries()[0];
  for(let index=0;index<8;index++){
    worker.onmessage({data:{type:"metadata_processed",accepted:true,rejected:0,changed:true,geometry_changed:true}});
    view.render(5.71+index*.01);
  }
  assert.equal(queries().length,1,"new geometry must not enqueue a duplicate in-flight source");
  worker.onmessage({data:{...initial,type:"present",result:{subjects:[],coverage:"no_anchor"}}});
  view.render(5.85);assert.equal(queries().length,2,"one refresh must use the latest coalesced geometry");
  for(let index=0;index<8;index++)view.render(5.86+index*.01);
  assert.equal(queries().length,2,"the refresh remains single-flight until its own result");
  worker.onmessage({data:{...queries()[1],type:"present",result:{subjects:[{
    target_sequence:0,track_id:"0",polygons:[[[.1,.1],[.2,.1],[.2,.2]]],complete:true,
  }],coverage:"anchored",exact:true}}});
  view.render(5.98);assert.equal(queries().length,2,"a current result does not renew itself");
  view.render(6);assert.equal(view.controller.snapshot().complete_contour_frames,1);
  assert.equal(view.controller.snapshot().frames_without_contours,0);
});

test("failed presentation releases only its own query and waits for new geometry before retrying",t=>{
  const view=browser();t.after(()=>view.controller.stop());const {controller}=view,worker=controller.worker;
  const queries=()=>view.workerMessages.filter(message=>message.type==="present");
  view.capture(0);view.render(.2);assert.equal(queries().length,1);
  worker.onmessage({data:{type:"failure",request_type:"frame",sequence:0}});
  assert.equal(controller.pendingPresent.has(0),true,"a frame failure cannot release a presentation request");
  worker.onmessage({data:{type:"failure",request_type:"present",sequence:0}});
  assert.equal(controller.pendingPresent.has(0),false);assert.equal(controller.requestedRevision.has(0),false);
  assert.equal(controller.requested.has(0),true);
  for(let index=0;index<5;index++)view.render(.21+index*.01);
  assert.equal(queries().length,1,"a failed query must not loop on every animation tick");
  worker.onmessage({data:{type:"metadata_processed",accepted:true,rejected:0,changed:true,geometry_changed:true}});
  view.render(.3);assert.equal(queries().length,2,"new geometry enables one retry");
  worker.onmessage({data:{...queries()[1],type:"present",result:{subjects:[{
    target_sequence:0,polygons:[[[.1,.1],[.2,.1],[.2,.2]]],complete:true,
  }],coverage:"anchored",exact:true}}});
  view.render(.5);assert.equal(controller.snapshot().complete_contour_frames,1);
  assert.equal(controller.snapshot().frames_without_contours,0);
});

test("retired and stopped frames release unanswered presentation bookkeeping and late replies cannot revive them",t=>{
  const view=browser({config:{delay_seconds:6,target_fps:20}});t.after(()=>view.controller.stop());
  const {controller}=view,worker=controller.worker;
  view.capture(0);view.capture(.05);view.render(4.56);
  assert.deepEqual([...controller.pendingPresent],[0,1]);assert.equal(controller.requestedRevision.size,2);
  view.render(6);
  assert.deepEqual([...controller.pendingPresent],[1]);assert.equal(controller.requestedRevision.has(0),false);
  assert.equal(controller.requestedRevision.has(1),true);assert.equal(controller.requested.has(0),false);
  worker.onmessage({data:{type:"present",sequence:0,result:{subjects:[],coverage:"no_anchor"}}});
  assert.equal(controller.overlays.has(0),false);assert.deepEqual([...controller.pendingPresent],[1]);
  assert.equal(controller.snapshot().worker_results_late,1);
  controller.stop();assert.equal(controller.pendingPresent.size,0);assert.equal(controller.requestedRevision.size,0);
  assert.equal(controller.requested.size,0);
  worker.onmessage({data:{type:"present",sequence:1,result:{subjects:[],coverage:"no_anchor"}}});
  assert.equal(controller.overlays.size,0);assert.equal(controller.pendingPresent.size,0);
  assert.equal(controller.requestedRevision.size,0);
});

test("prefetch scales within the available buffer without moving presentation or its deadline",t=>{
  for(const [delay,lead] of [[.5,.35],[1,.75],[1.7,1.45],[6,1.5],[8,1.5]]){
    const view=browser({config:{delay_seconds:delay}});t.after(()=>view.controller.stop());view.capture(0);
    const starts=delay-lead,queries=()=>view.workerMessages.filter(message=>message.type==="present");
    view.render(starts-.001);assert.equal(queries().length,0,`no early query for ${delay}s buffer`);
    view.render(starts+.001);assert.equal(queries().length,1,`one prefetched query for ${delay}s buffer`);
    assert.ok(Math.abs(queries()[0].deadline_epoch_ms-(1700000000000+delay*1000+100))<1e-6);
    assert.equal(view.controller.snapshot().distinct_presented_frames,0);
  }
});

test("event-only metadata preserves queued prefetch while later frames naturally receive the new color",()=>{
  const view=browser(),worker=view.controller.worker,colors=[],polygon=[[[.1,.1],[.2,.1],[.2,.2]]];
  view.capture(0);view.capture(.1);view.render(.2);
  const present=(sequence,alert)=>worker.onmessage({data:{type:"present",sequence,result:{subjects:[{
    target_sequence:sequence,tracking_confirmation:"confirmed",polygons:polygon,complete:true,alert,
  }],coverage:"anchored",exact:true}}});
  present(0,false);
  for(let i=0;i<4;i++){
    worker.onmessage({data:{type:"metadata_processed",accepted:true,changed:true,geometry_changed:false,rejected:0}});
    view.render(.21+i*.01);
  }
  assert.deepEqual(view.workerMessages.filter(message=>message.type==="present").map(message=>message.sequence),[0]);
  view.render(.3);
  assert.deepEqual(view.workerMessages.filter(message=>message.type==="present").map(message=>message.sequence),[0,1]);
  present(1,true);view.controller.context.stroke=()=>colors.push(view.controller.context.strokeStyle);
  view.render(.5);view.render(.61);
  assert.deepEqual(colors,["#22c55e","#ef4444"]);view.controller.stop();
});
test("pending identity draws yellow and a confirmed event takes red priority",()=>{
  const view=browser(),worker=view.controller.worker,colors=[],polygon=[[[.1,.1],[.2,.1],[.2,.2]]];
  view.capture(0);view.capture(.1);view.render(.2);
  const present=(sequence,alert)=>worker.onmessage({data:{type:"present",sequence,result:{subjects:[{
    target_sequence:sequence,tracking_confirmation:"confirmed",polygons:polygon,complete:true,alert,warning:true,
  }],coverage:"anchored",exact:true}}});
  present(0,false);view.render(.3);present(1,true);
  view.controller.context.stroke=()=>colors.push(view.controller.context.strokeStyle);
  view.render(.5);view.render(.61);
  assert.deepEqual(colors,["#facc15","#ef4444"]);view.controller.stop();
});

test("capacity exhaustion is visible and never replaces an admitted analysis sample",()=>{
  const view=browser();
  for(let i=0;i<13;i++)view.capture(i*.34);
  assert.equal(view.controller.snapshot().analysis_admitted,12);
  assert.equal(view.controller.snapshot().analysis_pending,12);
  assert.equal(view.controller.snapshot().analysis_capture_pauses,1);
  assert.equal(view.controller.uploads[0].metadata.source_sequence,0);
  assert.equal(view.controller.snapshot().captured_frames,13);
  assert.ok(view.errors.length);view.controller.stop();
});

test("uploads remain in FIFO until the exact server acknowledgement and drain waits for it",async()=>{
  const view=browser({autoAck:false});view.capture(0);view.capture(.34);await view.finish();
  const first=unpack(view.packets[0]);
  assert.equal(view.controller.snapshot().analysis_sent,0);assert.equal(view.controller.snapshot().analysis_pending,2);
  assert.equal(view.packets.length,1);assert.equal(view.encodings.length,1);
  assert.equal(view.controller.acknowledgeUpload({...first,type:"camera_frame_ack",accepted:true,generation:999}),false);
  assert.equal(view.controller.acknowledgeUpload({...first,type:"camera_frame_ack",accepted:true,source_sequence:1}),false);
  let drained=false;const drain=view.controller.drain().then(()=>drained=true);await settle();assert.equal(drained,false);
  assert.equal(view.controller.acknowledgeUpload({...first,type:"camera_frame_ack",accepted:true}),true);
  await settle();await view.finish();assert.equal(view.packets.length,2);assert.equal(drained,false);
  view.controller.acknowledgeUpload({...unpack(view.packets[1]),type:"camera_frame_ack",accepted:true});
  await drain;assert.equal(drained,true);assert.equal(view.controller.snapshot().analysis_sent,2);
  view.controller.stop();
});

test("obsolete source rejection preserves admitted frames and pauses analysis without stopping real video",async()=>{
  const view=browser({autoAck:false});view.capture(0);await view.finish();
  view.controller.acknowledgeUpload({...unpack(view.packets[0]),type:"camera_frame_ack",accepted:false,reason:"obsolete_source"});
  await settle();assert.equal(view.controller.snapshot().analysis_pending,1);
  assert.equal(view.controller.snapshot().analysis_sent,0);assert.equal(view.controller.capturing,true);
  for(let i=1;i<=30;i++){view.capture(i/15);view.render(i/15);}
  assert.equal(view.controller.snapshot().captured_frames,31);assert.equal(view.controller.snapshot().analysis_admitted,1);
  assert.ok(view.controller.snapshot().distinct_presented_frames>20);assert.ok(view.errors.some(error=>error.includes("采集会话")));
  view.controller.stop();
});

test("capacity refusal retries the same admitted packet and successful acknowledgement resumes analysis",async()=>{
  const view=browser({autoAck:false});view.capture(0);await view.finish();
  const first=view.packets[0],header=unpack(first);
  view.controller.acknowledgeUpload({...header,type:"camera_frame_ack",accepted:false,reason:"capacity"});
  await settle();
  for(let i=1;i<=15;i++){view.capture(i/15);view.render(i/15);}
  assert.equal(view.controller.snapshot().analysis_admitted,1);assert.equal(view.controller.snapshot().captured_frames,16);
  await new Promise(resolve=>setTimeout(resolve,270));
  assert.equal(view.packets.length,2);assert.equal(view.packets[1],first);
  assert.equal(view.controller.snapshot().analysis_pending,1);assert.equal(view.controller.snapshot().analysis_sent,0);
  view.controller.acknowledgeUpload({...header,type:"camera_frame_ack",accepted:true});await settle();
  view.capture(1.1);assert.equal(view.controller.snapshot().analysis_admitted,2);
  assert.equal(view.controller.snapshot().analysis_sent,1);view.controller.stop();
});

test("provisional true-source masks are painted grey even if an invalid alert bit is set",()=>{
  const view=browser();view.capture(0);
  view.controller.overlays.set(0,{subjects:[{target_sequence:0,tracking_confirmation:"provisional",alert:true,
    polygons:[[[.1,.1],[.2,.1],[.2,.2]]]}],exact:true,coverage:"anchored"});
  let color;view.controller.context.stroke=()=>color=view.controller.context.strokeStyle;
  view.render(.5);assert.equal(color,"#9ca3af");view.controller.stop();
});
test("resolution changes do not reuse the old crop alignment",()=>{
  const view=browser();view.capture(0);view.video.videoWidth=1280;view.capture(.1);
  assert.equal(view.controller.snapshot().captured_frames,1);assert.equal(view.controller.capturing,false);
  assert.ok(view.errors.some(error=>error.includes("尺寸")));view.controller.stop();
});
test("duplicate video frame callbacks cannot inflate the real source or display frame count",()=>{
  const view=browser();view.capture(0);view.capture(0);
  assert.equal(view.controller.snapshot().captured_frames,1);assert.equal(view.controller.snapshot().analysis_admitted,1);
  view.controller.stop();
});

test("display telemetry reports real presentations and capture age without image or identity data",async()=>{
  const view=browser();
  for(let i=0;i<=75;i++){
    view.capture(i/15);view.render(i/15);
    if(view.encodings.length)await view.finish();
  }
  await settle();assert.equal(view.reports.length,1);
  const report=JSON.parse(view.reports[0].options.body),stats=view.controller.snapshot();
  assert.equal(view.reports[0].url,"/v1/video/display/metrics");
  assert.equal(report.stream_session_id,"run");assert.equal(report.generation,2);assert.equal(report.crop_version,0);
  assert.equal(report.report_sequence,1);assert.equal(report.elapsed_seconds,5);assert.equal(report.window_seconds,5);
  assert.equal(report.distinct_presented_frames,stats.distinct_presented_frames);
  assert.equal(report.window_presented_frames,stats.distinct_presented_frames);
  assert.equal(report.window_presented_fps,stats.distinct_presented_frames/5);
  assert.equal(report.contour_frame_ratio,0);assert.ok(report.source_to_present_p95_seconds>=.5);
  assert.ok(report.source_to_present_max_seconds<.57);assert.ok(report.max_present_gap_seconds<.07);
  assert.match(report.display_instance_id,/^[A-Za-z0-9_-]{1,96}$/);
  assert.ok(Object.entries(report).every(([key,value])=>key==="display_instance_id"||typeof value==="number"||value===null||value==="run"));
  assert.doesNotMatch(view.reports[0].options.body,/polygons|bitmap|subject_id|image|jpeg|identity/);
  view.controller.stop();await settle();
  assert.equal(view.reports.length,2);assert.equal(view.reports[1].options.keepalive,true);
  assert.equal(JSON.parse(view.reports[1].options.body).report_sequence,2);
});

test("metadata diagnostics separate valid payload, HTTP failure, JSON failure and obsolete run",async()=>{
  const valid={stream_session_id:"run",generation:2,crop_version:0,anchors:[],events:[]};
  for(const [response,counter] of [
    [()=>({ok:true,json:async()=>valid}),"metadata_received"],
    [()=>({ok:false,status:503}),"metadata_http_errors"],
    [()=>({ok:true,json:async()=>{throw new Error("invalid json");}}),"metadata_decode_errors"],
    [()=>({ok:true,json:async()=>({...valid,generation:3})}),"metadata_run_mismatches"],
  ]){
    const view=browser({metadataResponse:response});
    await view.controller.pollMetadata();
    assert.equal(view.controller.snapshot()[counter],1);
    assert.equal(view.controller.snapshot().metadata_requests,1);
    assert.equal(view.controller.snapshot().metadata_failures,counter==="metadata_received"?0:1);
    assert.equal(view.workerMessages.filter(message=>message.type==="metadata").length,counter==="metadata_received"?1:0);
    view.controller.stop();
  }
});

test("worker timing and presentation diagnostics distinguish late results from unsafe motion",()=>{
  const view=browser();view.capture(0);view.setClock(.02);
  view.controller.worker.onmessage({data:{type:"frame_processed",sequence:0,queued_at:0,accepted:true}});
  view.controller.worker.onmessage({data:{type:"metadata_processed",accepted:true,rejected:2}});
  view.controller.worker.onmessage({data:{type:"present",sequence:0,queued_at:10,result:{subjects:[],coverage:"motion_unreliable"}}});
  view.render(.5);
  view.controller.worker.onmessage({data:{type:"present",sequence:0,queued_at:10,result:{subjects:[],coverage:"no_anchor"}}});
  const stats=view.controller.snapshot();
  assert.equal(stats.worker_pending_frames,0);assert.equal(stats.worker_frames_completed,1);
  assert.equal(stats.worker_metadata_accepted,1);assert.equal(stats.worker_anchor_rejections,2);
  assert.equal(stats.worker_results_received,2);assert.equal(stats.worker_results_late,1);
  assert.equal(stats.coverage_motion_unreliable,1);assert.equal(stats.coverage_no_anchor,0);
  assert.equal(stats.worker_frame_latency_max_seconds,.02);assert.equal(stats.worker_result_latency_max_seconds,.49);
  view.controller.stop();
});

test("WebSocket metadata bypasses HTTP, preserves run barriers and rejects replayed transport revisions",async()=>{
  const view=browser({config:{transport:"websocket"}}),controller=view.controller;
  controller.start();await controller.pollMetadata();
  assert.equal(view.reports.length,0);assert.equal(controller.snapshot().metadata_requests,0);
  const data={type:"camera_display_metadata",stream_session_id:"run",generation:2,crop_version:0,
    metadata_revision:9,anchors:[],events:[],event_snapshot_revision:4,event_snapshot_timestamp:1};
  assert.equal(controller.receiveMetadata(data,{transport:"websocket"}),true);
  assert.equal(controller.receiveMetadata(data,{transport:"websocket"}),false);
  assert.equal(controller.receiveMetadata({...data,metadata_revision:8},{transport:"websocket"}),false);
  assert.equal(controller.receiveMetadata({...data,metadata_revision:10,generation:3},{transport:"websocket"}),false);
  assert.equal(controller.receiveMetadata({...data,metadata_revision:10,crop_version:1},{transport:"websocket"}),false);
  assert.equal(controller.receiveMetadata({...data,metadata_revision:undefined},{transport:"websocket"}),false);
  assert.equal(controller.receiveMetadata({...data,metadata_revision:10},{transport:"websocket"}),true);
  assert.equal(view.workerMessages.filter(message=>message.type==="metadata").length,1);
  assert.equal(controller.snapshot().worker_pending_metadata,2);
  controller.worker.onmessage({data:{type:"metadata_processed",accepted:true}});
  const messages=view.workerMessages.filter(message=>message.type==="metadata");
  assert.equal(messages.length,2);assert.equal(messages[1].data.event_snapshot_revision,4);
  assert.equal(controller.snapshot().metadata_duplicate_revisions,2);
  assert.equal(controller.snapshot().metadata_run_mismatches,2);
  assert.equal(controller.snapshot().metadata_invalid_revisions,1);
  assert.equal(controller.snapshot().metadata_ws_received,2);
  controller.stop();assert.equal(view.reports.length,0);
  const metrics=JSON.parse(view.packets.at(-1));
  assert.equal(metrics.type,"camera_display_metrics");assert.equal(metrics.metadata_ws_received,2);
});

test("metadata age distinguishes source staleness from receive-to-worker delay",async()=>{
  const view=browser({config:{transport:"websocket"}}),controller=view.controller;
  view.capture(0);view.capture(1);view.setClock(1.4);
  const data={stream_session_id:"run",generation:2,crop_version:0,metadata_revision:1,
    anchors:[{source_sequence:1,media_timestamp:.9}],events:[]};
  assert.equal(controller.receiveMetadata(data,{transport:"websocket"}),true);
  assert.ok(Math.abs(controller.snapshot().latest_received_anchor_age_seconds-.5)<1e-9);
  assert.equal(controller.snapshot().latest_applied_anchor_age_seconds,null);
  view.setClock(1.5);controller.worker.onmessage({data:{type:"metadata_processed",accepted:true,
    changed:true,queued_at:1400,latest_anchor_timestamp:.9}});
  assert.ok(Math.abs(controller.snapshot().latest_applied_anchor_age_seconds-.6)<1e-9);
  await controller.reportMetrics();
  const report=JSON.parse(view.packets.at(-1));
  assert.equal(report.window_metadata_apply_count,1);assert.equal(report.window_metadata_apply_max_seconds,.1);
  assert.equal(report.window_capture_total_count,2);assert.equal(report.window_motion_capture_count,2);
  assert.equal(report.window_source_freeze_count,2);assert.equal(report.window_analysis_capture_count,2);
  controller.stop();
});

test("bounded numeric telemetry records foreground changes and long tasks and removes observers on stop",async()=>{
  const listeners=new Map();let observer,observed,disconnected=false;
  const document={hidden:false,addEventListener:(name,callback)=>listeners.set(name,callback),
    removeEventListener:(name,callback)=>{if(listeners.get(name)===callback)listeners.delete(name);}};
  const view=browser({config:{transport:"websocket"},globals:{document,
    PerformanceObserver:class{
      static supportedEntryTypes=["longtask"];
      constructor(callback){observer=callback;}
      observe(options){observed=options.type;}
      disconnect(){disconnected=true;}
    }}});
  assert.equal(observed,"longtask");view.setClock(.2);document.hidden=true;listeners.get("visibilitychange")();
  view.setClock(.4);assert.equal(view.controller.snapshot().page_hidden,1);
  assert.equal(view.controller.snapshot().hidden_seconds,.2);
  view.setClock(.5);document.hidden=false;listeners.get("visibilitychange")();
  observer({getEntries:()=>[{startTime:100,duration:64,name:"unrecorded-private-page"},{startTime:-1,duration:1000}]});
  for(let i=0;i<1000;i++)view.controller.noteTiming("capture_total",i);
  assert.equal(view.controller.timingWindows.capture_total.samples.length,256);
  await view.controller.reportMetrics();const report=JSON.parse(view.packets.at(-1));
  assert.equal(report.visibility_changes,2);assert.equal(report.hidden_seconds,.3);assert.equal(report.page_hidden,0);
  assert.equal(report.longtask_count,1);assert.equal(report.longtask_total_seconds,.064);
  assert.equal(report.window_longtask_max_seconds,.064);assert.equal(report.window_capture_total_count,1000);
  assert.equal(report.window_capture_total_sample_count,256);
  assert.equal(report.window_capture_total_max_seconds,.999);
  assert.doesNotMatch(view.packets.at(-1),/unrecorded-private-page|polygons|bitmap|subject_id|jpeg|identity/);
  view.controller.stop();assert.equal(listeners.size,0);assert.equal(disconnected,true);
});

test("motion capture sends a separate small bitmap in source order without any main-thread pixel readback",()=>{
  const view=browser();view.controller.grayContext.getImageData=()=>{throw new Error("main-thread readback forbidden");};
  for(let i=0;i<3;i++){
    view.capture(i/15);
    view.controller.worker.onmessage({data:{type:"frame_processed",sequence:i,accepted:true}});
  }
  const frames=view.workerMessages.filter(message=>message.type==="frame_bitmap");
  assert.deepEqual(frames.map(message=>message.frame.sequence),[0,1,2]);
  assert.deepEqual(frames.map(message=>message.frame.timestamp),[0,1/15,2/15]);
  for(let i=0;i<3;i++){
    const sent=frames[i].frame,display=view.controller.frames.frames[i].bitmap;
    assert.equal(sent.width,138);assert.equal(sent.height,103);
    assert.equal(sent.bitmap.origin,display.origin);assert.notEqual(sent.bitmap,display);
    assert.equal(display.closed,undefined);assert.equal(sent.pixels,undefined);
  }
  assert.equal(view.controller.snapshot().motion_failures,0);view.controller.stop();
});

test("motion area stays bounded across camera aspect ratios without resizing display or changing admitted analysis FIFO",async t=>{
  for(const [width,height,motionWidth,expected] of [
    [640,480,160,[138,103]],
    [480,640,160,[103,138]],
    [1920,1080,320,[320,180]],
    [480,640,320,[180,240]],
  ]){
    const view=browser({config:{motion_width:motionWidth}});t.after(()=>view.controller.stop());
    view.video.videoWidth=width;view.video.videoHeight=height;view.controller.settings.max_image_side=320;
    for(const [sequence,time] of [0,.34,.68].entries()){
      view.capture(time);view.controller.worker.onmessage({data:{type:"frame_processed",sequence,accepted:true}});
    }
    const motions=view.workerMessages.filter(message=>message.type==="frame_bitmap");
    assert.deepEqual(motions.map(message=>message.frame.sequence),[0,1,2]);
    assert.deepEqual(motions.map(message=>message.frame.timestamp),[0,.34,.68]);
    assert.equal(view.controller.uploads.length,3);assert.equal(view.packets.length,0);
    for(let index=0;index<3;index++){
      const motion=motions[index].frame,display=view.controller.frames.frames[index].bitmap,
        analysis=view.controller.uploads[index].image;
      assert.deepEqual([motion.width,motion.height],expected);
      assert.ok(motion.width*motion.height<=motionWidth*motionWidth*9/16);
      assert.ok(motion.height<=240&&motion.width<=motionWidth);
      assert.ok(Math.abs(motion.width/motion.height-width/height)<.025);
      assert.deepEqual([display.width,display.height],[width,height]);
      assert.equal(Math.max(analysis.width,analysis.height),320);
      assert.equal(display.origin,motion.bitmap.origin);assert.equal(display.origin,analysis.origin);
    }
    await view.finish();await view.finish();await view.finish();
    assert.deepEqual(view.packets.map(packet=>unpack(packet).source_sequence),[0,1,2]);
    assert.deepEqual(view.packets.map(packet=>unpack(packet).media_timestamp),[0,.34,.68]);
    assert.equal(view.controller.snapshot().analysis_pending,0);
    assert.equal(view.controller.snapshot().analysis_sent,3);view.controller.stop();
  }
});

test("encoding failures retain the exact admitted image and do not freeze independent video capture",async()=>{
  const view=browser();view.capture(0);const source=view.controller.uploads[0].image;
  view.controller.encodeWorker.onmessage({data:{type:"encoding_failure",sequence:0}});await settle();
  assert.equal(view.controller.uploads[0].image,source);assert.equal(source.closed,undefined);
  assert.equal(view.controller.snapshot().analysis_pending,1);assert.equal(view.packets.length,0);
  view.capture(.1);assert.equal(view.controller.snapshot().captured_frames,2);
  assert.equal(view.controller.snapshot().analysis_admitted,1);assert.equal(view.controller.snapshot().analysis_upload_failures,1);
  view.controller.stop();assert.equal(source.closed,true);
  assert.equal(view.controller.encodeWorker.terminated,true);
});

test("a successful encoded response closes only its retained analysis copy and native display pixels survive",async()=>{
  const view=browser();view.capture(0);const source=view.controller.uploads[0].image,
    display=view.controller.frames.frames[0].bitmap;
  view.controller.encodeWorker.onmessage({data:{type:"encoded_jpeg",sequence:999,bytes:new ArrayBuffer(4)}});
  assert.equal(view.controller.pendingEncode.sequence,0);assert.equal(source.closed,undefined);
  await view.finish();assert.equal(source.closed,true);assert.equal(display.closed,undefined);
  assert.equal(view.controller.snapshot().analysis_sent,1);view.controller.stop();
});

test("an encoding Worker crash rejects the current wait and retains every admitted source in FIFO",async()=>{
  const view=browser();view.capture(0);view.capture(.34);
  view.controller.encodeWorker.onmessageerror();await settle();
  assert.equal(view.controller.pendingEncode,null);assert.equal(view.controller.uploading,false);
  assert.deepEqual(Array.from(view.controller.uploads,task=>task.metadata.source_sequence),[0,1]);
  assert.ok(view.controller.uploads.every(task=>!task.image.closed));
  assert.equal(view.controller.snapshot().analysis_upload_failures,1);view.controller.stop();
});

test("original camera resolution is presented independently of tiny model inputs and motion working size",()=>{
  const view=browser();view.video.videoWidth=1808;view.video.videoHeight=1030;
  view.controller.settings.max_image_side=320;view.capture(0);
  const display=view.controller.frames.frames[0].bitmap,analysis=view.controller.uploads[0].image,
    motion=view.workerMessages.find(message=>message.type==="frame_bitmap").frame.bitmap;
  assert.deepEqual([display.width,display.height],[1808,1030]);
  assert.deepEqual([analysis.width,analysis.height],[320,182]);
  assert.ok(motion.width*motion.height<=160*160*9/16);
  assert.ok(Math.abs(motion.width/motion.height-1808/1030)<.025);
  assert.equal(display.origin,analysis.origin);assert.equal(display.origin,motion.origin);
  view.render(.5);assert.deepEqual([view.canvas.width,view.canvas.height],[1808,1030]);
  assert.equal(display.closed,true);assert.equal(analysis.closed,undefined);view.controller.stop();
});

test("native-resolution display history remains bounded and closes retired and stopped frame resources",()=>{
  const view=browser();view.video.videoWidth=1808;view.video.videoHeight=1030;
  view.capture(0);const first=view.controller.frames.frames[0].bitmap;
  for(let i=1;i<=65;i++)view.capture(i/15);
  assert.equal(view.controller.frames.frames.length,17);assert.equal(first.closed,true);
  assert.equal(view.controller.snapshot().display_buffer_expired,49);
  const retained=Array.from(view.controller.frames.frames,frame=>frame.bitmap);
  view.controller.stop();assert.ok(retained.every(bitmap=>bitmap.closed));
});

test("six-second native buffer sustains ordered real video beyond 120 frames and reports the full startup wait",async()=>{
  const view=browser({config:{delay_seconds:6,target_fps:20,analysis_sample_fps:2.5,transport:"websocket"}});
  view.video.videoWidth=1808;view.video.videoHeight=1030;
  const shown=[],resources=[];let maxRetained=0;
  assert.equal(view.canvas.cleared,true);assert.equal(view.controller.frames.maxFrames,132);
  for(let tick=0;tick<=1200;tick++){
    const time=tick/60;
    if(tick%3===0){view.capture(time);resources.push(view.controller.frames.frames.at(-1).bitmap);
      if(view.encodings.length)await view.finish();}
    maxRetained=Math.max(maxRetained,view.controller.frames.frames.length);view.render(time);
    if(view.canvas.dataset.frameSequence!==undefined&&shown.at(-1)?.sequence!==Number(view.canvas.dataset.frameSequence))
      shown.push({sequence:Number(view.canvas.dataset.frameSequence),timestamp:Number(view.canvas.dataset.mediaTimestamp),at:time});
    if(time<6)assert.equal(view.controller.snapshot().distinct_presented_frames,0);
  }
  assert.ok(maxRetained>=120&&maxRetained<=132);assert.ok(shown.length>=280);
  assert.deepEqual(shown.map(frame=>frame.sequence),Array.from({length:shown.length},(_,index)=>index));
  assert.ok(shown.every((frame,index)=>index===0||frame.timestamp>shown[index-1].timestamp));
  assert.ok(shown.every(frame=>frame.at-frame.timestamp>=6-1e-9&&frame.at-frame.timestamp<6.04));
  const metrics=view.packets.filter(packet=>typeof packet==="string").map(packet=>JSON.parse(packet));
  assert.equal(metrics[0].elapsed_seconds,5);assert.equal(metrics[0].window_presented_fps,0);
  assert.equal(metrics[0].distinct_presented_frames,0);assert.equal(metrics[0].configured_display_delay_seconds,6);
  assert.ok(metrics.filter(item=>item.elapsed_seconds>=15).every(item=>item.window_presented_fps>=19.8));
  const stats=view.controller.snapshot();assert.equal(stats.display_buffer_expired,0);
  assert.equal(stats.required_display_buffer_bytes,1808*1030*4*132);
  assert.equal(stats.estimated_display_buffer_bytes,1808*1030*4*stats.buffered_frames);
  assert.equal(stats.analysis_pending,0);assert.equal(stats.analysis_sent,stats.analysis_admitted);
  const uploads=view.packets.filter(packet=>typeof packet!=="string").map(unpack);
  assert.ok(uploads.every((packet,index)=>Math.abs(packet.media_timestamp-index*.4)<1e-9));
  assert.ok(uploads.every((packet,index)=>packet.source_sequence===index*8));
  assert.ok(resources.slice(0,shown.length).every(bitmap=>bitmap.closed));
  assert.ok(resources.slice(shown.length).every(bitmap=>!bitmap.closed));
  view.controller.stop();assert.ok(resources.every(bitmap=>bitmap.closed));
  assert.equal(view.controller.snapshot().estimated_display_buffer_bytes,0);
});

test("six-second presentation keeps source-matched contours and ignores results arriving after their frame",()=>{
  const view=browser({config:{delay_seconds:6,target_fps:20,transport:"websocket"}});
  view.capture(0);view.capture(.05);view.render(5.7);
  const queries=view.workerMessages.filter(message=>message.type==="present");
  assert.deepEqual(queries.map(message=>message.sequence),[0,1]);
  assert.deepEqual(queries.map(message=>message.timestamp),[0,.05]);
  assert.equal(view.controller.snapshot().distinct_presented_frames,0);
  const polygon=[[[.1,.1],[.2,.1],[.2,.2]]];let strokes=0;view.controller.context.stroke=()=>strokes++;
  view.controller.worker.onmessage({data:{type:"present",sequence:0,result:{subjects:[{
    target_sequence:0,polygons:polygon,complete:true}],coverage:"anchored",exact:true}}});
  view.render(6);assert.equal(strokes,1);assert.equal(view.canvas.dataset.frameSequence,"0");
  view.controller.worker.onmessage({data:{type:"present",sequence:0,result:{subjects:[{
    target_sequence:0,polygons:polygon,complete:true}],coverage:"anchored",exact:true}}});
  view.controller.worker.onmessage({data:{type:"present",sequence:1,result:{subjects:[{
    target_sequence:0,polygons:polygon,complete:true}],coverage:"anchored",exact:false}}});
  view.render(6.1);assert.equal(strokes,1);assert.equal(view.canvas.dataset.frameSequence,"1");
  assert.equal(view.controller.snapshot().complete_contour_frames,1);
  assert.equal(view.controller.snapshot().worker_results_late,1);view.controller.stop();
});

test("oversized native buffer is rejected before pixel capture or analysis admission without resizing",()=>{
  const view=browser({config:{delay_seconds:6,target_fps:20}});
  view.video.videoWidth=3840;view.video.videoHeight=2160;view.capture(0);
  const stats=view.controller.snapshot();assert.equal(view.video.reads,0);assert.equal(stats.captured_frames,0);
  assert.equal(stats.analysis_admitted,0);assert.equal(stats.analysis_pending,0);assert.equal(stats.buffered_frames,0);
  assert.equal(stats.display_buffer_capacity_failures,1);assert.equal(stats.estimated_display_buffer_bytes,0);
  assert.equal(stats.required_display_buffer_bytes,3840*2160*4*132);assert.equal(view.controller.capturing,false);
  assert.ok(view.errors.some(error=>error.includes("缓冲预算")&&error.includes("画面未缩小")));
  view.capture(.1);assert.equal(view.errors.length,1);assert.equal(view.video.videoWidth,3840);
  assert.equal(view.workerMessages.some(message=>message.type==="frame_bitmap"),false);view.controller.stop();
});

test("display delay preserves previous defaults and caps longer requests at eight seconds",()=>{
  const ordinary=browser();assert.equal(ordinary.controller.frames.delay,.5);ordinary.controller.stop();
  const live=browser({config:{delay_seconds:1.7,target_fps:20}});
  assert.equal(live.controller.frames.delay,1.7);assert.equal(live.controller.frames.maxFrames,46);live.controller.stop();
  const fallback=browser({config:{delay_seconds:undefined}});
  assert.equal(fallback.controller.frames.delay,1);fallback.controller.stop();
  const immediate=browser({config:{delay_seconds:0}});immediate.capture(0);immediate.render(0);
  assert.equal(immediate.controller.frames.delay,0);assert.equal(immediate.controller.snapshot().distinct_presented_frames,0);
  immediate.render(.15);assert.equal(immediate.controller.snapshot().distinct_presented_frames,1);
  immediate.controller.stop();
  const capped=browser({config:{delay_seconds:60,target_fps:20}});
  assert.equal(capped.controller.frames.delay,8);assert.equal(capped.controller.frames.maxFrames,172);capped.controller.stop();
});

test("zero-buffer video waits for asynchronous per-frame contours, with a bounded worker-failure fallback",t=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:0}});
  t.after(()=>view.controller.stop());
  view.controller.capture(0,{mediaTime:50,presentedFrames:100},{width:640,height:480});
  view.render(0);assert.equal(view.controller.snapshot().distinct_presented_frames,0);
  const query=view.workerMessages.find(message=>message.type==="present");
  assert.equal(query.sequence,100);
  view.setClock(.02);
  view.controller.worker.onmessage({data:{type:"present",sequence:100,result:{coverage:"anchored",subjects:[{
    target_sequence:100,track_id:"person",polygons:[[[.2,.2],[.5,.2],[.5,.5]]],complete:true,
  }]}}});
  view.render(.02);
  assert.equal(view.controller.snapshot().propagated_contour_frames,1);
  assert.equal(view.controller.snapshot().worker_results_late,0);
  view.setClock(.1);
  view.controller.capture(100,{mediaTime:50.1,presentedFrames:101},{width:640,height:480});
  view.render(.1);view.render(.249);
  assert.equal(view.canvas.dataset.frameSequence,"100");
  view.render(.251);
  assert.equal(view.canvas.dataset.frameSequence,"101");
  assert.equal(view.controller.snapshot().frames_without_contours,1);
});

test("two SAM anchors per second supply moving contours on every 15 FPS display frame",t=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:2,anchor_policy:"nearest_arrived"}});
  t.after(()=>view.controller.stop());
  const adapter={estimateMotion:(a,b)=>({step:b[0]-a[0]}),
    reverseMotion:motion=>({step:-motion.step}),
    warpPolygons:(polygons,motion)=>polygons.map(p=>p.map(([x,y])=>[x+motion.step/1000,y]))};
  const history=new view.api.ContourHistory(view.controller.config,adapter),pendingAnchors=[],pendingReplies=[];
  let nextSample=0,queryIndex=0,anchors=0;
  for(let tick=0;tick<375;tick++){
    const now=tick/60;view.setClock(now);
    if(tick%4===0&&tick<240){
      const sequence=tick/4;
      history.appendFrame({sequence,timestamp:now,width:16,height:16,pixels:new Uint8Array(256).fill(sequence)});
      view.controller.capture(now*1000,{mediaTime:now,presentedFrames:sequence},{width:640,height:480});
      view.controller.worker.onmessage({data:{type:"frame_processed",sequence,accepted:true}});
      if(now+1e-6>=nextSample){
        pendingAnchors.push({at:now+.8,anchor:{source_sequence:sequence,media_timestamp:now,epoch:"epoch",
          subjects:[{track_id:"person",tracking_confirmation:"confirmed",
            polygons:[[[.2+sequence/1000,.2],[.5+sequence/1000,.2],[.5+sequence/1000,.5]]]}]}});
        nextSample+=.5;anchors++;
      }
    }
    while(pendingAnchors[0]?.at<=now){
      history.acceptMetadata({...view.controller.config,anchors:[pendingAnchors.shift().anchor],events:[]});
      view.controller.worker.onmessage({data:{type:"metadata_processed",accepted:true,geometry_changed:true}});
    }
    while(pendingReplies[0]?.at<=now){
      const {query}=pendingReplies.shift(),result=history.select(query.sequence,query.timestamp);
      if(result.subjects.length)assert.ok(Math.abs(result.subjects[0].polygons[0][0][0]-(.2+query.sequence/1000))<1e-6);
      view.controller.worker.onmessage({data:{type:"present",sequence:query.sequence,result}});
    }
    view.render(now);
    for(;queryIndex<view.workerMessages.length;queryIndex++){
      const query=view.workerMessages[queryIndex];
      if(query.type==="present")pendingReplies.push({at:now+.02,query});
    }
  }
  const stats=view.controller.snapshot();
  assert.equal(anchors,8);assert.equal(stats.distinct_presented_frames,60);
  assert.equal(stats.frames_without_contours,0);assert.equal(stats.observed_contour_frames,8);
  assert.equal(stats.propagated_contour_frames,52);assert.equal(stats.display_buffer_expired,0);
});

test("a slow camera cannot inflate its distinct FPS behind a longer presentation buffer",()=>{
  const view=browser({config:{delay_seconds:6,target_fps:20,transport:"websocket"}});
  for(let tick=0;tick<=900;tick++){
    if(tick%6===0)view.capture(tick/60);view.render(tick/60);
  }
  const metrics=view.packets.filter(packet=>typeof packet==="string").map(packet=>JSON.parse(packet));
  const last=metrics.at(-1);assert.equal(last.elapsed_seconds,15);assert.ok(last.window_presented_fps<=10.2);
  assert.ok(last.window_presented_fps>=9.8);assert.equal(last.source_callback_fps,10);
  assert.ok(last.distinct_presented_frames<=last.captured_frames);view.controller.stop();
});

test("a new buffering run clears the preceding run's canvas and displayed-frame identity",()=>{
  const view=browser();view.capture(0);view.render(.5);view.controller.stop();
  assert.equal(view.canvas.dataset.frameSequence,"0");view.canvas.cleared=false;
  const next=view.createController({delay_seconds:6,stream_session_id:"new",generation:3});
  assert.equal(view.canvas.cleared,true);assert.equal(view.canvas.dataset.frameSequence,undefined);
  assert.equal(view.canvas.dataset.mediaTimestamp,undefined);assert.equal(next.snapshot().distinct_presented_frames,0);next.stop();
});

test("three packets overlap ACK latency but a fourth cannot encode or send until a slot frees",async t=>{
  const view=browser({autoAck:false});t.after(()=>view.controller.stop());
  for(let i=0;i<5;i++)view.capture(i*.34);
  const sources=Array.from(view.controller.uploads,task=>task.image);
  await view.finish();await view.finish();await view.finish();
  assert.equal(view.packets.length,3);assert.equal(view.encodings.length,0);
  assert.equal(view.controller.snapshot().analysis_in_flight,3);
  assert.equal(view.controller.snapshot().analysis_pending,5);
  assert.equal(sources[2].closed,true);assert.equal(sources[3].closed,undefined);
  // Out-of-order receipts remove only the exact source, never the FIFO head.
  view.controller.acknowledgeUpload({...unpack(view.packets[1]),type:"camera_frame_ack",accepted:true});
  await settle();await view.finish();
  assert.deepEqual(view.packets.map(packet=>unpack(packet).source_sequence),[0,1,2,3]);
  assert.deepEqual(Array.from(view.controller.uploads,task=>task.metadata.source_sequence),[0,2,3,4]);
  assert.equal(view.controller.snapshot().analysis_sent,1);
  assert.equal(view.controller.snapshot().analysis_in_flight_peak,3);
});

test("a frame admitted during ACK wait sends immediately into the bounded window",async t=>{
  const view=browser({autoAck:false});t.after(()=>view.controller.stop());view.capture(0);await view.finish();
  assert.equal(view.encodings.length,0);view.capture(.34);assert.equal(view.encodings.length,1);
  await view.finish();assert.equal(view.packets.length,2);
  view.controller.acknowledgeUpload({...unpack(view.packets[0]),type:"camera_frame_ack",accepted:true});
  await settle();assert.equal(view.controller.snapshot().analysis_in_flight,1);
});

function uploadClock(){
  let next=0;const timers=new Map();
  return {timers,globals:{setTimeout:(callback,ms)=>{const id=++next;timers.set(id,{callback,ms});return id;},
    clearTimeout:id=>timers.delete(id)},
    fire:async ms=>{const entry=[...timers].find(([,value])=>value.ms===ms);assert.ok(entry,`missing ${ms}ms timer`);
      timers.delete(entry[0]);entry[1].callback();await settle();}};
}
function ack(view,index,accepted=true,reason=null){return view.controller.acknowledgeUpload({
  ...unpack(view.packets[index]),type:"camera_frame_ack",accepted,reason});}

test("a lost early ACK recovers automatically after later receipts without re-encoding or duplicate accounting",async t=>{
  const clock=uploadClock(),view=browser({autoAck:false,globals:clock.globals});t.after(()=>view.controller.stop());
  view.capture(0);view.capture(.34);await view.finish();await view.finish();
  ack(view,1);await clock.fire(5000);
  assert.equal(view.controller.snapshot().analysis_ack_timeouts,1);
  view.capture(1);assert.equal(view.controller.snapshot().analysis_admitted,2);
  await clock.fire(250);
  assert.equal(view.packets[2],view.packets[0]);assert.equal(view.encodings.length,0);
  assert.equal(ack(view,0),true);assert.equal(ack(view,2),false);
  assert.equal(view.controller.snapshot().analysis_sent,2);
  assert.equal(view.controller.snapshot().analysis_pending,0);
  assert.equal(view.controller.snapshot().analysis_upload_retries,1);
  assert.equal(view.controller.snapshot().analysis_upload_failures,0);
  view.capture(1.4);assert.equal(view.controller.snapshot().analysis_admitted,3);
});

test("a positive ACK between timeout and retry cancels the scheduled retransmission",async t=>{
  const clock=uploadClock(),view=browser({autoAck:false,globals:clock.globals});t.after(()=>view.controller.stop());
  view.capture(0);await view.finish();await clock.fire(5000);
  assert.equal(ack(view,0),true);await settle();
  assert.equal(clock.timers.size,0);assert.equal(view.packets.length,1);
  assert.equal(view.controller.snapshot().analysis_pending,0);
});

test("capacity refusals serialize retries across the whole window and resume admission only after recovery",async t=>{
  const clock=uploadClock(),view=browser({autoAck:false,globals:clock.globals});t.after(()=>view.controller.stop());
  for(let i=0;i<4;i++)view.capture(i*.34);
  await view.finish();await view.finish();await view.finish();
  for(let i=0;i<3;i++)ack(view,i,false,"capacity");
  for(let i=0;i<3;i++)await clock.fire(250);
  assert.equal(view.packets.length,4);assert.equal(view.packets[3],view.packets[0]);
  assert.equal(view.encodings.length,0);
  ack(view,3);await settle();assert.equal(view.packets[4],view.packets[1]);
  ack(view,4);await settle();assert.equal(view.packets[5],view.packets[2]);
  assert.equal(view.controller.analysisPause,"recovering");
  ack(view,5);await settle();await view.finish();
  assert.equal(unpack(view.packets[6]).source_sequence,3);ack(view,6);
  assert.equal(view.controller.snapshot().analysis_sent,4);assert.equal(view.controller.analysisPause,null);
});

test("permanent ACK loss stops bounded retries, retains evidence and clears owned timers on stop",async()=>{
  const clock=uploadClock(),view=browser({autoAck:false,globals:clock.globals});
  try{
    view.capture(0);await view.finish();
    await clock.fire(5000);await clock.fire(250);await clock.fire(5000);await clock.fire(500);await clock.fire(5000);
    assert.equal(view.packets.length,3);assert.equal(view.controller.snapshot().analysis_pending,1);
    assert.equal(view.controller.snapshot().analysis_sent,0);assert.equal(view.controller.snapshot().analysis_upload_failures,1);
    await assert.rejects(view.controller.drain(),/recovery stopped/);
    assert.equal(clock.timers.size,0);
  }finally{view.controller.stop();}
});

test("600ms receipts sustain 2.5 FPS without queue growth or playback capture pauses",async t=>{
  let now=0,next=0;const timers=new Map(),globals={
    setTimeout:(callback,ms)=>{const id=++next;timers.set(id,{callback,at:now+ms});return id;},clearTimeout:id=>timers.delete(id)};
  const view=browser({autoAck:false,globals,config:{analysis_sample_fps:2.5}});t.after(()=>view.controller.stop());
  view.socket.send=packet=>{view.packets.push(packet);globals.setTimeout(()=>ack(view,view.packets.indexOf(packet)),600);};
  let peak=0;
  for(now=0;now<=17000;now+=20){
    view.setClock(now/1000);
    for(const [id,timer] of [...timers])if(timer.at<=now){timers.delete(id);timer.callback();}
    if(now<16000&&now%400===0)view.capture(now/1000);
    await settle();while(view.encodings.length)await view.finish();
    peak=Math.max(peak,view.controller.snapshot().analysis_pending);
  }
  assert.equal(view.controller.snapshot().analysis_admitted,40);
  assert.equal(view.controller.snapshot().analysis_sent,40);assert.equal(peak,2);
  assert.equal(view.controller.snapshot().analysis_capture_pauses,0);
  assert.equal(view.controller.snapshot().analysis_upload_failures,0);
});

test("lookahead encoding failure cannot resend the acknowledged packet or discard the failing source",async()=>{
  const view=browser({autoAck:false});view.capture(0);view.capture(.34);view.capture(.68);
  const next=view.controller.uploads[1].image,third=view.controller.uploads[2].image;
  await view.finish();view.controller.encodeWorker.onmessage({data:{type:"encoding_failure",sequence:1}});
  await settle();assert.equal(next.closed,undefined);assert.equal(third.closed,undefined);
  assert.equal(view.packets.length,1);assert.equal(view.controller.snapshot().analysis_sent,0);
  view.controller.acknowledgeUpload({...unpack(view.packets[0]),type:"camera_frame_ack",accepted:true});
  await settle();assert.equal(view.packets.length,1);assert.equal(view.controller.snapshot().analysis_sent,1);
  assert.deepEqual(Array.from(view.controller.uploads,task=>task.metadata.source_sequence),[1,2]);
  assert.equal(view.controller.uploads[0].image,next);assert.equal(next.closed,undefined);
  assert.equal(view.controller.snapshot().analysis_upload_failures,1);view.controller.stop();
});

test("split preprocessing and motion timing counters expose both queues and expired query work",()=>{
  const view=browser();view.capture(0);view.capture(.1);
  view.controller.preprocessWorker.onmessage({data:{type:"frame_preprocessed",sequence:0,compute_ms:16,queue_ms:30}});
  let stats=view.controller.snapshot();assert.equal(stats.preprocess_pending_frames,1);assert.equal(stats.motion_pending_frames,1);
  view.controller.worker.onmessage({data:{type:"frame_processed",sequence:0,accepted:true,compute_ms:18,motion_queue_ms:40}});
  view.controller.worker.onmessage({data:{type:"present",sequence:0,expired:true,compute_ms:0,motion_queue_ms:500}});
  stats=view.controller.snapshot();assert.equal(stats.worker_pending_frames,1);assert.equal(stats.motion_pending_frames,0);
  assert.equal(stats.worker_present_expired,1);
  assert.equal(view.controller.timingWindows.motion_preprocess.total,16);
  assert.equal(view.controller.timingWindows.motion_preprocess_queue.total,30);
  assert.equal(view.controller.timingWindows.motion_compute.total,18);
  assert.equal(view.controller.timingWindows.motion_frame_queue.total,40);
  assert.equal(view.controller.timingWindows.motion_present_queue.max,500);view.controller.stop();
});

test("optional analysis compression caps preserve original display resolution and never mutate shared camera settings",()=>{
  const settings={sample_fps:2.5,max_image_side:1280,jpeg_quality:90};
  const view=browser({settings,config:{analysis_max_image_side:720,analysis_jpeg_quality:80}});
  view.video.videoWidth=1808;view.video.videoHeight=1030;view.capture(0);
  const display=view.controller.frames.frames[0].bitmap,analysis=view.controller.uploads[0].image;
  assert.deepEqual([display.width,display.height],[1808,1030]);
  assert.deepEqual([analysis.width,analysis.height],[720,410]);
  assert.deepEqual(settings,{sample_fps:2.5,max_image_side:1280,jpeg_quality:90});
  assert.notEqual(view.controller.settings,settings);assert.equal(view.controller.settings.sample_fps,2.5);
  assert.equal(view.workerMessages.find(message=>message.type==="encode_jpeg").quality,.8);
  view.controller.stop();
  const unchanged=browser({settings});
  assert.equal(unchanged.controller.settings.max_image_side,1280);assert.equal(unchanged.controller.settings.jpeg_quality,90);
  unchanged.controller.stop();
  const smaller=browser({settings:{...settings,max_image_side:320,jpeg_quality:70},
    config:{analysis_max_image_side:720,analysis_jpeg_quality:80}});
  assert.equal(smaller.controller.settings.max_image_side,320);assert.equal(smaller.controller.settings.jpeg_quality,70);
  smaller.controller.stop();
});

test("Canvas draws 人员追踪模块 pixel centers at image centers including both image edges",()=>{
  const view=browser(),path=[];
  view.controller.context.moveTo=(x,y)=>path.push(["move",x,y]);
  view.controller.context.lineTo=(x,y)=>path.push(["line",x,y]);
  view.controller.context.closePath=()=>path.push(["close"]);
  view.controller.context.stroke=()=>path.push(["stroke"]);
  view.capture(0);
  view.controller.overlays.set(0,{subjects:[{target_sequence:0,complete:true,
    polygons:[[[.5/640,.5/480],[639.5/640,.5/480],[639.5/640,479.5/480]]]}],
    exact:true,coverage:"anchored"});
  view.render(.5);
  assert.deepEqual(path,[["move",.5,.5],["line",639.5,.5],["line",639.5,479.5],["close"],["stroke"]]);
  view.controller.stop();
});

test("Canvas scales dense grid pixel centers and keeps partial segments open",()=>{
  const view=browser(),path=[];
  view.controller.context.moveTo=(x,y)=>path.push(["move",x,y]);
  view.controller.context.lineTo=(x,y)=>path.push(["line",x,y]);
  view.controller.context.closePath=()=>path.push(["close"]);
  view.controller.context.stroke=()=>path.push(["stroke"]);
  view.capture(0);
  view.controller.overlays.set(0,{subjects:[{target_sequence:0,complete:false,polygons:[],
    open_segments:[[[.5/160,.5/120],[150.5/160,110.5/120],[159.5/160,119.5/120]]]}],
    exact:false,coverage:"motion_unreliable"});
  view.render(.5);
  assert.deepEqual(path,[["move",2,2],["line",602,442],["line",638,478],["stroke"]]);
  assert.equal(view.controller.snapshot().complete_contour_frames,0);view.controller.stop();
});

test("geometry cache counters exclude expired work and results that never performed a lookup",()=>{
  const view=browser(),send=message=>view.controller.worker.onmessage({data:{type:"present",sequence:0,
    result:{subjects:[],coverage:"anchored"},...message}});
  send({geometry_cache_hit:false});send({geometry_cache_hit:true});send({geometry_cache_hit:true});
  send({result:{subjects:[],coverage:"no_anchor"}});
  send({expired:true,geometry_cache_hit:false,compute_ms:0});
  const stats=view.controller.snapshot();assert.equal(stats.geometry_cache_hits,2);
  assert.equal(stats.geometry_cache_misses,1);assert.equal(stats.worker_present_expired,1);
  assert.equal(stats.worker_results_received,5);view.controller.stop();
});


test("late playback discards expired display copies but retains the full four-second buffer",()=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:4}});
  for(let i=0;i<13;i++){
    view.setClock(i);
    view.controller.capture(i*1000,{mediaTime:i,presentedFrames:i+1},
      {width:640,height:480,origin:i});
  }
  view.controller.pendingPresent.add(1);view.controller.requested.add(1);
  view.controller.requestedRevision.set(1,0);view.controller.overlays.set(1,{subjects:[]});
  const oldBitmap=view.controller.frames.frames[0].bitmap;
  view.render(13);
  assert.equal(view.canvas.dataset.frameSequence,"10"); // external sequence is presentedFrames
  assert.equal(view.canvas.dataset.mediaTimestamp,"9");
  assert.equal(view.controller.frames.delay,4);
  assert.equal(view.controller.snapshot().display_catchup_expired,9);
  assert.equal(oldBitmap.closed,true);
  assert.equal(view.controller.pendingPresent.has(1),false);
  assert.equal(view.controller.overlays.has(1),false);
  const presented=view.controller.snapshot().distinct_presented_frames;
  view.render(13.2);assert.equal(view.controller.snapshot().distinct_presented_frames,presented);
  view.controller.worker.onmessage({data:{type:"present",sequence:1,result:{subjects:[]}}});
  assert.equal(view.controller.overlays.has(1),false);
  assert.equal(view.controller.snapshot().analysis_admitted,0);
  view.controller.stop();
});


function streamPacket(sequence,config={}){
  const data={protocol:1,connection_id:"connection",config:{stream_session_id:"run",generation:2,crop_version:0,
    enabled:true,delay_seconds:4,target_fps:15,transport:"stream",...config},
    frames:[{source_sequence:sequence,media_timestamp:sequence/15,width:640,height:480,bytes:4}],metadata:null};
  const header=new TextEncoder().encode(JSON.stringify(data)),body=new Uint8Array(8+header.length+4);
  body.set(new TextEncoder().encode("GVD1"));new DataView(body.buffer).setUint32(4,header.length);body.set(header,8);
  const packet=new Uint8Array(body.length+4);new DataView(packet.buffer).setUint32(0,body.length);packet.set(body,4);
  return packet;
}

test("continuous framing accepts arbitrary network splits and multiple pictures without truncation",()=>{
  const view=browser(),decoder=new view.api.DisplayStreamDecoder(),packets=[];
  const a=streamPacket(1),b=streamPacket(2),both=new Uint8Array(a.length+b.length);both.set(a);both.set(b,a.length);
  for(let i=0;i<both.length;i+=7)packets.push(...decoder.feed(both.subarray(i,i+7)));
  decoder.finish();assert.deepEqual(packets.map(p=>p.frames[0].source_sequence),[1,2]);
  assert.throws(()=>new view.api.DisplayStreamDecoder().feed(new Uint8Array([127,255,255,255])),/length/);
  const partial=new view.api.DisplayStreamDecoder();partial.feed(a.subarray(0,-1));assert.throws(()=>partial.finish(),/truncated/);
  view.controller.stop();
});

test("continuous video reaches all 15 frames despite an unanswered metadata request",async()=>{
  let requested=0;
  const view=browser({globals:{createImageBitmap:async()=>({width:640,height:480,close(){}}),
    fetch:async()=>{requested++;return new Promise(()=>{});}}});
  const feed=new view.api.NetworkController({canvas:view.canvas,url:"/v1/fleet/cameras/camera02/video/display/frames",metricsUrl:"/metrics"});
  for(let i=0;i<15;i++){
    view.setClock(i/15);
    const decoder=new view.api.DisplayStreamDecoder();feed.enqueue(decoder.feed(streamPacket(i))[0]);await settle();
  }
  assert.equal(requested,1);assert.equal(feed.controller.stats.network_frames_received,15);
  assert.equal(feed.controller.config.metadata_url,"/v1/fleet/cameras/camera02/video/display/metadata");
  for(let i=0;i<15;i++){view.setClock(4+i/15+.0001);feed.controller.animate();}
  assert.equal(feed.controller.snapshot().distinct_presented_frames,15);
  assert.equal(feed.controller.stats.frames_without_contours,15);
  feed.stop();view.controller.stop();
});

test("a short JPEG decode stall retains useful pictures and bounds an expired backlog",async()=>{
  let release;
  const view=browser({globals:{createImageBitmap:()=>new Promise(resolve=>{release=()=>resolve({width:640,height:480,close(){}});})}});
  const feed=new view.api.NetworkController({canvas:view.canvas,url:"/frames",metricsUrl:"/metrics"}),decoder=new view.api.DisplayStreamDecoder();
  for(let i=0;i<31;i++)feed.enqueue(decoder.feed(streamPacket(i,{transport:"frames"}))[0]);
  assert.equal(feed.queueFrames,30); // Two seconds survive inside the four-second buffer.
  for(let i=31;i<300;i++)feed.enqueue(decoder.feed(streamPacket(i,{transport:"frames"}))[0]);
  assert.ok(feed.queueFrames<=69);assert.ok(feed.queueBytes<=32*1024**2);
  assert.ok(feed.controller.stats.network_queue_dropped_frames>0);
  feed.stop();release();await settle();assert.equal(feed.queueFrames,0);view.controller.stop();
});

test("independent metadata accepts 204 and rejects stale delta revisions",async()=>{
  const view=browser({config:{transport:"stream"},globals:{fetch:async()=>({ok:true,status:204})}});
  await view.controller.pollMetadata();assert.equal(view.controller.stats.metadata_failures,0);
  const data={stream_session_id:"run",generation:2,crop_version:0,metadata_revision:9,anchors:[]};
  assert.equal(view.controller.receiveMetadata(data,{transport:"http_delta"}),true);
  assert.equal(view.controller.receiveMetadata({...data,metadata_revision:8},{transport:"http_delta"}),false);
  assert.equal(view.controller.stats.metadata_ws_received,0);view.controller.stop();
});


test("continuous fetch parses split chunks without waiting for metadata and cancels its reader",async()=>{
  const packet=streamPacket(1),next=streamPacket(2);let requests=0,cancelled=false,finish;
  const chunks=[packet.subarray(0,3),packet.subarray(3),next];
  const view=browser({globals:{createImageBitmap:async()=>({width:640,height:480,close(){}}),
    fetch:async(url,options)=>{
      if(url.includes("/metadata"))return new Promise(()=>{});
      requests++;options.signal.addEventListener("abort",()=>finish?.({done:true}));
      return {ok:true,headers:{get:()=>"application/vnd.goodvision.display-stream"},body:{getReader:()=>({
        read:async()=>chunks.length?{value:chunks.shift(),done:false}:new Promise(resolve=>{finish=resolve;}),
        cancel:async()=>{cancelled=true;finish?.({done:true});},
      })}};
    }}});
  const feed=new view.api.NetworkController({canvas:view.canvas,url:"/frames",metricsUrl:"/metrics"});
  const running=feed.pull();await settle();await settle();
  assert.equal(requests,1);assert.equal(feed.controller.stats.network_requests,1);
  assert.equal(feed.controller.stats.network_frames_received,2);
  feed.stop();await running;assert.equal(cancelled,true);view.controller.stop();
});


test("three minutes of stalled motion retain 15 FPS native video and a four-second playback buffer",t=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:4}});
  t.after(()=>view.controller.stop());let grayDraws=0;
  const draw=view.controller.grayContext.drawImage;
  view.controller.grayContext.drawImage=(...args)=>{grayDraws++;draw(...args);};
  for(let tick=0;tick<184*60;tick++){
    const now=tick/60;view.setClock(now);
    if(tick%4===0&&tick<180*60)view.controller.capture(now*1000,
      {mediaTime:now,presentedFrames:tick/4},{width:1920,height:1080});
    view.render(now);
    assert.ok(view.controller.snapshot().worker_pending_frames<=2);
    assert.ok(view.controller.snapshot().worker_pending_present_queries<=2);
    if(now<4)assert.equal(view.controller.snapshot().distinct_presented_frames,0);
  }
  const stats=view.controller.snapshot();
  assert.equal(stats.captured_frames,2700);assert.equal(stats.distinct_presented_frames,2700);
  assert.equal(view.canvas.dataset.frameSequence,"2699");
  assert.equal(stats.configured_display_delay_seconds,4);
  assert.ok(Math.abs(stats.last_source_to_present_seconds-4)<1e-6);
  assert.equal(stats.display_buffer_expired,0);assert.equal(stats.display_catchup_expired,0);
  assert.equal(stats.analysis_admitted,0);assert.equal(stats.motion_failures,0);
  assert.equal(stats.motion_frames_admitted,2);assert.equal(stats.motion_frames_skipped,2698);
  assert.equal(grayDraws,2,"skip before graphics readback or bitmap allocation");
  assert.equal(view.workerMessages.filter(message=>message.type==="present").length,2,
    "retiring the displayed frame cannot pretend its worker query has completed");
  assert.deepEqual([view.canvas.width,view.canvas.height],[1920,1080]);
});

test("frame admission waits for motion completion and resumes with current source pixels",t=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:4}});
  t.after(()=>view.controller.stop());
  const capture=seq=>{view.setClock(seq/15);view.controller.capture(seq/15*1000,
    {mediaTime:seq/15,presentedFrames:seq},{width:640,height:480,origin:seq});};
  capture(0);capture(1);
  view.controller.preprocessWorker.onmessage({data:{type:"frame_preprocessed",sequence:0}});
  capture(2);
  assert.equal(view.controller.snapshot().worker_pending_frames,2);
  assert.equal(view.controller.snapshot().preprocess_pending_frames,1);
  assert.equal(view.controller.snapshot().motion_pending_frames,1);
  view.controller.worker.onmessage({data:{type:"frame_processed",sequence:0,accepted:true}});
  capture(3);
  assert.deepEqual(view.workerMessages.filter(m=>m.type==="frame_bitmap").map(m=>m.frame.sequence),[0,1,3]);
  view.controller.worker.onmessage({data:{type:"frame_processed",sequence:1,accepted:false,expired:true}});
  // A preprocessor ACK from another worker can arrive after the terminal ACK.
  view.controller.preprocessWorker.onmessage({data:{type:"frame_preprocessed",sequence:1}});
  assert.equal(view.controller.snapshot().worker_pending_frames,1);
  assert.equal(view.controller.snapshot().motion_pending_frames,0);
  assert.equal(view.controller.snapshot().motion_frames_expired,1);
  assert.equal(view.controller.snapshot().motion_failures,0);
  view.controller.worker.onmessage({data:{type:"failure",request_type:"frame",sequence:3}});
  capture(20);
  assert.equal(view.workerMessages.filter(m=>m.type==="frame_bitmap").at(-1).frame.bitmap.origin,20);
  assert.equal(view.controller.snapshot().worker_pending_frames,1);
});

test("metadata bursts preserve bounded delta anchors and the latest authoritative event snapshot",t=>{
  const view=browser({config:{transport:"websocket"}});t.after(()=>view.controller.stop());
  for(let revision=0;revision<1000;revision++){
    view.setClock(revision/1000);
    assert.equal(view.controller.receiveMetadata({stream_session_id:"run",generation:2,crop_version:0,
      metadata_revision:revision,event_snapshot_revision:revision,event_snapshot_timestamp:revision/1000,
      events:[],anchors:[{source_sequence:revision,media_timestamp:revision/15,epoch:"epoch",subjects:[]}]},
      {transport:"websocket"}),true);
  }
  assert.equal(view.workerMessages.filter(m=>m.type==="metadata").length,1);
  assert.equal(view.controller.snapshot().worker_pending_metadata,2);
  assert.equal(view.controller.snapshot().motion_metadata_coalesced,998);
  view.controller.worker.onmessage({data:{type:"metadata_processed",accepted:true}});
  const sent=view.workerMessages.filter(m=>m.type==="metadata");
  assert.equal(sent.length,2);assert.equal(sent[1].data.event_snapshot_revision,999);
  assert.equal(sent[1].data.anchors.length,256);
  assert.equal(sent[1].data.anchors[0].source_sequence,744);
  assert.equal(sent[1].data.anchors.at(-1).source_sequence,999);
  assert.equal(sent[1].queued_at,1,"coalescing retains the oldest queued receipt time for honest latency");
  view.controller.worker.onmessage({data:{type:"metadata_processed",accepted:true}});
  assert.equal(view.controller.snapshot().worker_pending_metadata,0);
});


test("terminal worker failure retires pending work while buffered originals continue",t=>{
  const view=browser({config:{source_mode:"network",transport:"frames",delay_seconds:4}});
  t.after(()=>view.controller.stop());
  view.setClock(0);view.controller.capture(0,{mediaTime:0,presentedFrames:0},{width:640,height:480});
  assert.equal(view.controller.snapshot().worker_pending_frames,1);
  view.controller.preprocessWorker.onerror();
  assert.equal(view.controller.snapshot().worker_pending_frames,0);
  assert.equal(view.controller.snapshot().worker_pending_present_queries,0);
  assert.equal(view.controller.worker.terminated,true);assert.equal(view.controller.preprocessWorker.terminated,true);
  view.setClock(1);view.controller.capture(1000,{mediaTime:1,presentedFrames:15},{width:640,height:480});
  view.render(4);assert.equal(view.canvas.dataset.frameSequence,"0");
  view.render(5);assert.equal(view.canvas.dataset.frameSequence,"15");
  assert.equal(view.workerMessages.filter(m=>m.type==="frame_bitmap").length,1);
  assert.equal(view.controller.snapshot().worker_pending_frames,0);
});


test("coalescing a delayed HTTP response cannot resurrect an older event snapshot",t=>{
  const view=browser();t.after(()=>view.controller.stop());
  const data=revision=>({stream_session_id:"run",generation:2,crop_version:0,anchors:[],
    event_snapshot_revision:revision,event_snapshot_timestamp:revision,events:[]});
  view.controller.receiveMetadata(data(1));
  view.controller.receiveMetadata(data(3));
  view.controller.receiveMetadata({...data(2),events:[{event_label:"stale"}]});
  view.controller.worker.onmessage({data:{type:"metadata_processed",accepted:true}});
  const latest=view.workerMessages.filter(m=>m.type==="metadata").at(-1).data;
  assert.equal(latest.event_snapshot_revision,3);assert.equal(latest.event_snapshot_timestamp,3);
  assert.equal(latest.events.length,0);
});
