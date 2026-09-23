const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const {test}=require("node:test");
const source=fs.readFileSync(require.resolve("../src/goodvision_harness/camera_display.js"),"utf8");
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const config={stream_session_id:"run",generation:1,crop_version:0,motion_backend:"dense"};
function worker({fail=false,readbackFail=false}={}){
  const replies=[],imports=[],readbacks=[],encodings=[],adapters=[];
  const context=vm.createContext({Uint8Array,TextEncoder,DataView,structuredClone,performance,setTimeout,clearTimeout,
    OffscreenCanvas:class{
      constructor(width,height){this.width=width;this.height=height;}
      getContext(){return {drawImage:bitmap=>{this.origin=bitmap.origin;},getImageData:()=>{
        if(readbackFail)throw new Error("readback failed");
        readbacks.push(this.origin);return {data:new Uint8Array(this.width*this.height*4).fill(80)};
      }}};
      convertToBlob(){return new Promise(resolve=>encodings.push(()=>resolve({
        arrayBuffer:async()=>new Uint8Array([255,216,255,217]).buffer,
      })));}
    },
    postMessage:message=>replies.push(message),importScripts:url=>{
      imports.push(url);if(fail)throw new Error("asset unavailable");
      if(url.endsWith("camera_display_dense.js"))context.GoodvisionDenseMotion={createDenseMotionAdapter:()=>{
        const adapter={estimateMotion:()=>({last_warp_rejection:null}),warpPolygons:polygons=>polygons};
        adapters.push(adapter);return adapter;
      }};
      else context.cv={};
    }});
  vm.runInContext(source,context);
  return {context,replies,imports,readbacks,encodings,adapters,send:data=>context.onmessage({data}),ready:()=>{
    context.cv.Mat=function Mat(){};context.cv.onRuntimeInitialized();
  }};
}
function frame(sequence,timestamp){return {type:"frame",queued_at:0,
  frame:{sequence,timestamp,width:16,height:16,pixels:new Uint8Array(256)}};}
function bitmapFrame(sequence,timestamp){return {type:"frame_bitmap",queued_at:0,
  frame:{sequence,timestamp,width:16,height:16,bitmap:{width:16,height:16,origin:sequence,
    close(){this.closed=true;}}}};}
function metadata(session="run"){return {type:"metadata",data:{...config,stream_session_id:session,
  anchors:[{source_sequence:0,media_timestamp:0,epoch:"epoch",subjects:[{track_id:"p",
    polygons:[[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]]}]}],events:[]}};}

test("WASM startup retains source frames, metadata and presentation requests in their original order",async()=>{
  const view=worker();view.send({type:"reset",config});
  view.send(frame(0,0));view.send(metadata());view.send(frame(1,.1));
  view.send({type:"present",sequence:1,timestamp:.1,queued_at:0});
  await settle();assert.equal(view.replies.length,0);view.ready();await settle();
  assert.deepEqual(view.replies.map(reply=>reply.type),[
    "motion_runtime_ready","frame_processed","metadata_processed","frame_processed","present","motion_ready",
  ]);
  assert.deepEqual(view.replies.filter(reply=>reply.type==="frame_processed").map(reply=>reply.sequence),[0,1]);
  const shown=view.replies.find(reply=>reply.type==="present");
  assert.equal(shown.result.subjects[0].target_sequence,1);assert.ok(shown.compute_ms>=0);
  assert.equal(view.replies.at(-1).backend,"dense");
});

test("a new run during WASM initialization cannot replay the previous run's buffered pixels",async()=>{
  const view=worker();view.send({type:"reset",config});view.send(frame(0,0));await settle();
  view.send({type:"reset",config:{...config,motion_backend:"legacy",stream_session_id:"new"}});
  view.send(frame(10,0));view.ready();await settle();
  assert.deepEqual(view.replies.filter(reply=>reply.type==="frame_processed").map(reply=>reply.sequence),[10]);
  assert.equal(view.replies.some(reply=>reply.type==="motion_ready"),false);
});

test("a missing motion asset reports explicit initialization failure without silently using another estimator",async()=>{
  const view=worker({fail:true});view.send({type:"reset",config});view.send(frame(0,0));await settle();
  assert.equal(view.replies.length,1);assert.equal(view.replies[0].type,"failure");
  assert.equal(view.replies[0].stage,"motion_initialization");
  view.send(frame(1,.1));assert.equal(view.replies.length,1);
});

test("an expired presentation query skips only drawing work while its source remains available to later frames",async()=>{
  const view=worker();view.send({type:"reset",config});view.send(frame(0,0));view.send(metadata());
  view.send({type:"present",sequence:0,timestamp:0,deadline_epoch_ms:1});
  view.send(frame(1,.1));view.send({type:"present",sequence:1,timestamp:.1});
  await settle();view.ready();await settle();
  const shown=view.replies.filter(reply=>reply.type==="present");
  assert.equal(shown[0].expired,true);assert.equal(shown[0].compute_ms,0);
  assert.equal(shown[1].result.subjects[0].anchor_sequence,0);
  assert.equal(shown[1].result.subjects[0].target_sequence,1);
  assert.deepEqual(view.replies.filter(reply=>reply.type==="frame_processed").map(reply=>reply.sequence),[0,1]);
});

test("WASM startup converts transferred bitmaps to ordered gray history and promptly releases graphics resources",async()=>{
  const view=worker(),first=bitmapFrame(0,0),second=bitmapFrame(1,.1);
  view.send({type:"reset",config});view.send(first);view.send(metadata());view.send(second);
  view.send({type:"present",sequence:1,timestamp:.1});
  assert.equal(first.frame.bitmap.closed,true);assert.equal(second.frame.bitmap.closed,true);
  assert.deepEqual(view.readbacks,[0,1]);assert.equal(view.replies.length,0);
  await settle();view.ready();await settle();
  assert.deepEqual(view.replies.slice(1,5).map(reply=>reply.type),[
    "frame_processed","metadata_processed","frame_processed","present",
  ]);
  const frames=view.replies.filter(reply=>reply.type==="frame_processed");
  assert.deepEqual(frames.map(reply=>reply.sequence),[0,1]);
  assert.ok(frames.every(reply=>reply.accepted&&reply.preprocess_ms>=0));
  assert.equal(view.replies.find(reply=>reply.type==="present").result.subjects[0].anchor_sequence,0);
});

test("bitmap readback failure closes its resource and cannot report a fabricated source frame",()=>{
  const view=worker({readbackFail:true}),input=bitmapFrame(0,0);
  view.send({type:"reset",config:{...config,motion_backend:"legacy"}});view.send(input);
  assert.equal(input.frame.bitmap.closed,true);assert.equal(view.replies.length,1);
  assert.equal(view.replies[0].stage,"motion_preprocess");assert.equal(view.replies[0].sequence,0);
});

test("frames arriving after motion initialization failure are released without retaining stale graphics",async()=>{
  const view=worker({fail:true}),input=bitmapFrame(0,0);
  view.send({type:"reset",config});await settle();view.send(input);
  assert.equal(input.frame.bitmap.closed,true);assert.equal(view.readbacks.length,0);
  assert.equal(view.replies.length,1);assert.equal(view.replies[0].stage,"motion_initialization");
});

test("independent JPEG Worker returns the matching encoded source and closes its own bitmap",async()=>{
  const view=worker(),bitmap={width:640,height:480,origin:71,close(){this.closed=true;}};
  view.send({type:"encode_jpeg",sequence:71,bitmap,quality:.85});
  assert.equal(view.encodings.length,1);assert.equal(bitmap.closed,undefined);assert.equal(view.imports.length,0);
  view.encodings.shift()();await settle();
  assert.equal(view.replies.length,1);assert.equal(view.replies[0].type,"encoded_jpeg");
  assert.equal(view.replies[0].sequence,71);assert.equal(bitmap.closed,true);
  assert.deepEqual([...new Uint8Array(view.replies[0].bytes)],[255,216,255,217]);
});

test("independent grayscale Worker forwards pixels, metadata and presentations through one ordered port",()=>{
  const preprocess=worker(),motion=worker(),forwarded=[],input={onmessage:null};
  const configLegacy={...config,motion_backend:"legacy"};
  motion.send({type:"motion_input",config:configLegacy,port:input});
  preprocess.send({type:"preprocess_output",port:{postMessage:(message,transfers)=>{
    forwarded.push(message);
    if(message.type==="frame")assert.equal(transfers[0],message.frame.pixels.buffer);
  }}});
  const first=bitmapFrame(0,0),second=bitmapFrame(1,.1);
  preprocess.send(first);preprocess.send(metadata());preprocess.send(second);
  preprocess.send({type:"present",sequence:0,timestamp:0});
  assert.deepEqual(forwarded.map(message=>message.type),["frame","metadata","frame","present"]);
  assert.equal(first.frame.bitmap.closed,true);assert.equal(second.frame.bitmap.closed,true);
  assert.equal(forwarded[0].frame.pixels[0],80);assert.equal(forwarded[0].frame.bitmap,undefined);
  assert.deepEqual(preprocess.replies.map(message=>message.type),["frame_preprocessed","frame_preprocessed"]);
  assert.equal(preprocess.imports.length,0);assert.equal(motion.replies.length,0);
  for(const message of forwarded)input.onmessage({data:message});
  assert.deepEqual(motion.replies.map(message=>message.type),["frame_processed","metadata_processed","frame_processed","present"]);
  const shown=motion.replies.at(-1);assert.equal(shown.result.subjects[0].anchor_sequence,0);
  assert.equal(shown.result.subjects[0].target_sequence,0);
  assert.ok(motion.replies[0].motion_queue_ms>=0);
});

test("expired presentation deadlines survive grayscale forwarding and still count as expired in motion Worker",()=>{
  const preprocess=worker(),motion=worker(),input={onmessage:null};
  motion.send({type:"motion_input",config:{...config,motion_backend:"legacy"},port:input});
  preprocess.send({type:"preprocess_output",port:{postMessage:message=>input.onmessage({data:message})}});
  preprocess.send(bitmapFrame(0,0));preprocess.send({type:"present",sequence:0,timestamp:0,deadline_epoch_ms:1});
  assert.equal(motion.replies.at(-1).expired,true);assert.equal(motion.replies.at(-1).compute_ms,0);
  assert.equal(motion.replies[0].accepted,true);
});

test("preview warmup loads only the runtime and a pending run shares the same initialization",async()=>{
  const view=worker();view.send({type:"warmup_motion"});await settle();
  assert.equal(view.imports.length,2);assert.equal(view.adapters.length,0);assert.equal(view.replies.length,0);
  view.send({type:"motion_input",config,port:{}});
  view.send(frame(0,0));view.send(metadata());view.send({type:"present",sequence:0,timestamp:0});
  await settle();assert.equal(view.imports.length,2);assert.equal(view.adapters.length,0);
  view.ready();await settle();
  assert.equal(view.adapters.length,1);assert.equal(view.imports.length,2);
  assert.deepEqual(view.replies.map(reply=>reply.type),[
    "motion_warmup_ready","motion_runtime_ready","frame_processed","metadata_processed","present","motion_ready",
  ]);
  assert.equal(view.replies.find(reply=>reply.type==="present").result.subjects[0].anchor_sequence,0);
  assert.ok(view.replies[0].runtime_load_ms>=0);assert.ok(view.replies.at(-1).startup_drain_ms>=0);
});

test("a warmed runtime creates fresh adapters and clears run pixels, anchors and metadata",async()=>{
  const view=worker();view.send({type:"warmup_motion"});await settle();view.ready();await settle();
  assert.equal(view.adapters.length,0);
  view.send({type:"reset",config});await settle();
  view.send(frame(0,0));view.send(metadata());view.send({type:"present",sequence:0,timestamp:0});
  assert.equal(view.replies.at(-1).result.subjects.length,1);
  view.send({type:"reset",config:{...config,stream_session_id:"new",generation:2}});await settle();
  view.send(frame(0,0));view.send(metadata());view.send({type:"present",sequence:0,timestamp:0});
  assert.equal(view.adapters.length,2);assert.notEqual(view.adapters[0],view.adapters[1]);
  assert.equal(view.imports.length,2);assert.equal(view.replies.at(-1).result.subjects.length,0);
  assert.equal(view.replies.at(-1).result.coverage,"no_anchor");
  assert.equal(view.replies.filter(reply=>reply.type==="metadata_processed").at(-1).accepted,false);
});

test("preview runtime failures have an explicit stage and do not construct motion history",async()=>{
  const view=worker({fail:true});view.send({type:"warmup_motion"});await settle();
  assert.equal(view.replies.length,1);assert.equal(view.replies[0].stage,"motion_warmup");
  assert.equal(view.adapters.length,0);
  view.send(frame(0,0));view.send(metadata());assert.equal(view.replies.length,1);
});

test("runtime warmup does not assimilate OpenCV's legacy self-returning thenable",async()=>{
  const view=worker();view.send({type:"warmup_motion"});await settle();
  view.context.cv.then=()=>{throw new Error("legacy thenable must not be awaited");};
  view.ready();await settle();
  assert.equal(view.replies.length,1);assert.equal(view.replies[0].type,"motion_warmup_ready");
});

test("diagnostic motion trace preserves ordered inputs and transfers a separate exact gray copy",()=>{
  const preprocess=worker(),motion=worker(),input={},forwarded=[];
  motion.send({type:"motion_input",config:{...config,motion_backend:"legacy"},port:input});
  preprocess.send({type:"preprocess_output",trace_enabled:true,port:{postMessage:(message,transfer)=>{
    const copy=structuredClone(message,{transfer});forwarded.push(copy);input.onmessage({data:copy});
  }}});
  const first=bitmapFrame(0,0);preprocess.send(first);preprocess.send(metadata());
  preprocess.send({type:"present",id:0,sequence:0,timestamp:0,deadline_epoch_ms:1});
  preprocess.send({type:"present",id:0,sequence:0,timestamp:0});
  const traces=preprocess.replies.filter(reply=>reply.type==="motion_trace_input").map(reply=>reply.message);
  assert.deepEqual(traces.map(message=>message.trace_id),[0,1,2,3]);
  assert.deepEqual(traces.map(message=>message.type),["frame","metadata","present","present"]);
  assert.equal(traces[0].frame.pixels.byteLength,256);assert.equal(traces[0].frame.pixels[0],80);
  assert.notEqual(traces[0].frame.pixels.buffer,forwarded[0].frame.pixels.buffer);
  assert.equal(forwarded[0].frame.pixels[0],80);assert.equal(first.frame.bitmap.closed,true);
  assert.deepEqual(motion.replies.map(reply=>reply.trace_id),[0,1,2,3]);
  assert.equal(motion.replies[2].expired,true);assert.equal(motion.replies[3].result.subjects.length,1);
});

test("trace byte and duration limits stop only tracing and never omit an original motion frame",()=>{
  for(const limit of ["bytes","duration"]){
    const preprocess=worker(),forwarded=[];let now=0;preprocess.context.performance={now:()=>now};
    preprocess.send({type:"preprocess_output",trace_enabled:true,trace_max_bytes:limit==="bytes"?1024:100000,
      trace_max_seconds:1,port:{postMessage:message=>forwarded.push(message)}});
    preprocess.send(bitmapFrame(0,0));if(limit==="duration")now=1001;
    preprocess.send(bitmapFrame(1,.1));preprocess.send(bitmapFrame(2,.2));
    assert.equal(forwarded.length,3);assert.deepEqual(forwarded.map(message=>message.frame.sequence),[0,1,2]);
    assert.equal(preprocess.replies.filter(reply=>reply.type==="motion_trace_input").length,1);
    const stopped=preprocess.replies.find(reply=>reply.type==="motion_trace_stopped");
    assert.equal(stopped.complete,false);assert.equal(stopped.reason,limit==="bytes"?"byte_limit":"duration_limit");
    assert.equal(preprocess.replies.filter(reply=>reply.type==="failure").length,0);
  }
});

test("normal preprocessing does not emit trace records or add trace identities",()=>{
  const preprocess=worker(),forwarded=[];
  preprocess.send({type:"preprocess_output",port:{postMessage:message=>forwarded.push(message)}});
  preprocess.send(bitmapFrame(0,0));preprocess.send(metadata());
  assert.ok(forwarded.every(message=>!Object.hasOwn(message,"trace_id")));
  assert.deepEqual(preprocess.replies.map(reply=>reply.type),["frame_preprocessed"]);
});

test("Worker reports geometry memo lookup only after valid target and anchor selection",async()=>{
  const view=worker();view.send({type:"reset",config});
  view.send(frame(0,0));view.send({type:"present",sequence:0,timestamp:0});
  view.send(metadata());view.send(frame(1,.1));
  view.send({type:"present",sequence:1,timestamp:.1});view.send({type:"present",sequence:1,timestamp:.1});
  view.send({type:"present",sequence:1,timestamp:.1,deadline_epoch_ms:0});
  await settle();view.ready();await settle();
  const results=view.replies.filter(reply=>reply.type==="present");
  assert.equal(results[0].result.coverage,"no_anchor");assert.equal(results[0].geometry_cache_hit,undefined);
  assert.equal(results[1].geometry_cache_hit,false);assert.equal(results[1].result.geometry_cache_hit,false);
  assert.equal(results[2].geometry_cache_hit,true);assert.equal(results[2].result.geometry_cache_hit,true);
  assert.equal(results[3].expired,true);assert.equal(results[3].geometry_cache_hit,undefined);
});

test("metadata geometry changes distinguish new or corrected 人员追踪模块 anchors from event-only revisions",async()=>{
  const view=worker();view.send({type:"reset",config});view.send(frame(0,0));view.send(metadata());
  await settle();view.ready();await settle();
  const last=()=>view.replies.filter(reply=>reply.type==="metadata_processed").at(-1);
  assert.equal(last().changed,true);assert.equal(last().geometry_changed,true);
  const event={epoch:"epoch",track_id:"p",event_label:"smoking",event_id:"event",revision:1,
    phase:"confirmed",valid_from_timestamp:0,valid_until_timestamp:null};
  const events={type:"metadata",data:{...config,anchors:[],events:[event]}};
  view.send(events);assert.equal(last().changed,true);assert.equal(last().geometry_changed,false);
  view.send(events);assert.equal(last().changed,false);assert.equal(last().geometry_changed,false);
  const corrected=metadata();corrected.data.anchors[0].subjects[0].polygons[0][0]=[.25,.25];
  view.send(corrected);assert.equal(last().changed,true);assert.equal(last().geometry_changed,true);
  view.send(corrected);assert.equal(last().geometry_changed,false);
  const changedEpoch=structuredClone(corrected);changedEpoch.data.anchors[0].epoch="new-epoch";
  view.send(changedEpoch);assert.equal(last().geometry_changed,true);
  view.send(frame(1,.1));const newer=metadata();newer.data.anchors[0].source_sequence=1;
  newer.data.anchors[0].media_timestamp=.1;view.send(newer);assert.equal(last().geometry_changed,true);
  view.send(metadata("old-run"));assert.equal(last().accepted,false);assert.equal(last().geometry_changed,false);
  const invalid=metadata();invalid.data.anchors[0].source_sequence=999;
  view.send(invalid);assert.equal(last().accepted,true);assert.equal(last().rejected,1);
  assert.equal(last().changed,false);assert.equal(last().geometry_changed,false);
});


test("expired source bitmaps retire without readback or transferring ownership to motion",()=>{
  const view=worker(),forwarded=[],input=bitmapFrame(3,.2);
  input.motion_deadline_epoch_ms=1;
  view.send({type:"preprocess_output",port:{postMessage:message=>forwarded.push(message)}});
  view.send(input);
  assert.equal(input.frame.bitmap.closed,true);assert.equal(view.readbacks.length,0);
  assert.equal(forwarded.length,0);
  assert.equal(view.replies[0].type,"frame_processed");assert.equal(view.replies[0].sequence,3);
  assert.equal(view.replies[0].expired,true);assert.equal(view.replies[0].accepted,false);
});

test("frames that expire after grayscale or WASM startup never enter motion history",async()=>{
  const view=worker();view.send({type:"reset",config});
  const expired=frame(0,0);expired.motion_deadline_epoch_ms=1;view.send(expired);
  view.send({type:"present",id:0,sequence:0,timestamp:0});
  await settle();view.ready();await settle();
  const retired=view.replies.find(reply=>reply.type==="frame_processed");
  assert.equal(retired.expired,true);assert.equal(retired.accepted,false);
  assert.equal(view.replies.find(reply=>reply.type==="present").result.coverage,"missing_source");
  view.send(frame(20,20/15));
  const fresh=metadata();fresh.data.anchors[0].source_sequence=20;fresh.data.anchors[0].media_timestamp=20/15;
  view.send(fresh);view.send({type:"present",id:20,sequence:20,timestamp:20/15});
  assert.equal(view.replies.at(-1).result.subjects[0].target_sequence,20);
});

test("readback failures identify the exact frame admission to retire",()=>{
  const view=worker({readbackFail:true}),input=bitmapFrame(7,.5);
  view.send({type:"preprocess_output",port:{postMessage(){assert.fail("must not forward failed readback");}}});
  view.send(input);
  assert.equal(input.frame.bitmap.closed,true);
  assert.equal(view.replies[0].request_type,"frame_bitmap");assert.equal(view.replies[0].sequence,7);
});
