const assert=require("node:assert/strict");
const vm=require("node:vm");
const {test}=require("node:test");
const {html, script} = require("./helpers/dashboard.cjs");

function player({browserCamera=false}={}){
  let now=0,id=0;
  let networkStarts=0;
  const timers=new Map(),requests=[],revoked=[],rawFeed={dataset:{},removeAttribute(name){delete this[name]}};
  const state={running:true,sessionId:"one",info:{annotated_output:{output_fps:15}}};
  if(browserCamera)Object.assign(state,{mode:"camera",sessionMode:"stream",cameraStream:{}});
  const context=vm.createContext({
    state,
    ui:{rawFeed},document:{hidden:false},performance:{now:()=>now},AbortController,
    networkVideoDisplay:null,
    GoodvisionCameraDisplay:browserCamera?{NetworkController:class{
      static supported(){return true} constructor(){networkStarts++} start(){networkStarts++} stop(){}
    }}:undefined,
    Image:class{async decode(){}},
    URL:{createObjectURL:blob=>"blob:"+blob.sequence,revokeObjectURL:url=>revoked.push(url)},
    setTimeout:(callback,delay)=>{timers.set(++id,{callback,at:now+delay});return id},
    clearTimeout:key=>timers.delete(key),surfaces:()=>{},resetAnnotations:()=>{},
    fetch:(url,options)=>new Promise((resolve,reject)=>{
      const request={url,options,resolve,reject};requests.push(request);
      options.signal.addEventListener("abort",()=>reject(new Error("aborted")))
    })
  });
  vm.runInContext(script.slice(script.indexOf("const latestFeed="),script.indexOf("function settingsSourceLabel"))+`
    globalThis.player={openFeed,closeFeed};
  `,context);
  return {context,requests,revoked,rawFeed,timers,...context.player,networkStarts:()=>networkStarts,
    elapse:ms=>{now+=ms},
    tick:()=>{
      const [key,timer]=[...timers.entries()].sort((a,b)=>a[1].at-b[1].at)[0];
      timers.delete(key);now=Math.max(now,timer.at);timer.callback()
    },
    respond:(request,sequence)=>request.resolve({ok:true,blob:async()=>({sequence}),
      headers:{get:name=>name==="X-Frame-Sequence"?String(sequence):""}})
  };
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));

test("a hidden diagnostic console releases an existing continuous display",()=>{
  const view=player();let stopped=0;
  view.context.networkVideoDisplay={stop(){stopped++;}};
  view.context.GoodvisionFleet={detailHidden:()=>true};
  view.openFeed();view.openFeed();
  assert.equal(stopped,1);assert.equal(view.context.networkVideoDisplay,null);
  assert.equal(view.requests.length,0);assert.equal(view.timers.size,0);
});

test("embedded diagnostics reuse the parent display without opening another video stream",()=>{
  const view=player();let started=0,stopped=0,options;
  view.context.GoodvisionDiagnosticCameraId="camera02";
  view.context.GoodvisionCameraDisplay={SharedController:class{
    constructor(value){options=value;}start(){started++;}stop(){stopped++;}
  }};
  const row={canvas:{hidden:false},networkDisplay:{controller:{snapshot:()=>({display_instance_id:'parent'})}}};
  view.context.window={parent:{GoodvisionFleet:{app:{rows:new Map([["camera02",row]])}}}};
  view.openFeed();view.openFeed();assert.equal(started,1);
  assert.equal(options.source().media,row.canvas);assert.equal(view.requests.length,0);
  view.closeFeed();assert.equal(stopped,1);
});

test("slow delivery keeps one request and resumes from the newest frame",async()=>{
  const view=player();view.openFeed();view.openFeed();
  assert.equal(view.requests.length,1);
  assert.equal(view.requests[0].options.cache,"no-store");
  view.elapse(500);view.respond(view.requests[0],10);await settle();
  assert.equal(view.rawFeed.dataset.frameSequence,"10");
  view.tick();assert.equal(view.requests.length,2);
  view.respond(view.requests[1],30);await settle();
  assert.equal(view.rawFeed.dataset.frameSequence,"30");
  assert.deepEqual(view.revoked,["blob:10"]);
  view.closeFeed();assert.equal(view.timers.size,0);
  assert.equal(view.rawFeed.src,undefined);assert.deepEqual(view.revoked,["blob:10","blob:30"])
});

test("a session change cancels old requests without overwriting the new frame",async()=>{
  const view=player();view.openFeed();const old=view.requests[0];
  view.context.state.sessionId="two";view.openFeed();
  assert.equal(old.options.signal.aborted,true);
  view.respond(view.requests[1],1);view.respond(old,99);await settle();
  assert.equal(view.rawFeed.dataset.frameSequence,"1");
  assert.equal(view.timers.size,1);view.closeFeed()
});

test("hidden pages pause fetching and request a fresh frame on return",async()=>{
  const view=player();view.context.document.hidden=true;view.openFeed();
  assert.equal(view.requests.length,0);
  view.context.document.hidden=false;view.tick();
  assert.equal(view.requests.length,1);
  view.closeFeed();await settle();assert.equal(view.timers.size,0)
});

test("unavailable frames retry without leaving concurrent requests or URLs",async()=>{
  const view=player();view.openFeed();view.requests[0].resolve({ok:false});await settle();
  assert.equal(view.timers.size,1);view.tick();assert.equal(view.requests.length,2);
  view.respond(view.requests[1],5);await settle();assert.equal(view.rawFeed.src,"blob:5");
  view.closeFeed();assert.deepEqual(view.revoked,["blob:5"])
});

test("a browser camera never enters the server-owned network display transport",()=>{
  const view=player({browserCamera:true});view.openFeed();
  assert.equal(view.networkStarts(),0);assert.equal(view.requests.length,1);
  assert.equal(view.requests[0].url,"/v1/video/annotated/latest.jpg");view.closeFeed()
});

function cameraSender(){
  const encodings=[],packets=[],intervals=new Map();let intervalId=0,captures=0;
  const socket={readyState:1,bufferedAmount:0,send:packet=>packets.push(packet)};
  const state={cameraStream:{},cameraGeneration:0,cameraBusy:false,ws:socket,
    info:{annotated_output:{output_fps:15}}};
  const context=vm.createContext({state,ui:{cameraPreview:{videoWidth:640,videoHeight:480}},
    cameraConfig:()=>true,cameraSettings:()=>({sample_fps:2,max_image_side:640,jpeg_quality:75}),
    document:{createElement:()=>({getContext:()=>({drawImage:()=>captures++}),
      toBlob:resolve=>encodings.push(resolve)})},
    setInterval:(callback,delay)=>{const id=++intervalId;intervals.set(id,{callback,delay});return id},
    clearInterval:id=>intervals.delete(id),
  });
  vm.runInContext(html.slice(html.indexOf("function startSampler()"),html.indexOf("function clearAnnotation()"))+`
    globalThis.sampler={startSampler,stopSampler};
  `,context);
  return {state,socket,encodings,packets,...context.sampler,
    captures:()=>captures,
    tick:()=>[...intervals.values()][0].callback(),
    finish:()=>encodings.shift()({arrayBuffer:async()=>new Uint8Array([1,2,3]).buffer}),
  }
}

test("camera upload only pauses new captures above the existing 1 MiB queue bound",async()=>{
  const view=cameraSender();view.startSampler();view.socket.bufferedAmount=1024*1024+1;
  await view.tick();assert.equal(view.captures(),0);assert.equal(view.packets.length,0);
  view.socket.bufferedAmount=1024*1024;const pending=view.tick();
  assert.equal(view.captures(),1);
  await view.tick();assert.equal(view.captures(),1);
  view.finish();await pending;
  assert.equal(view.packets.length,1);
  assert.equal(new DataView(view.packets[0]).getUint16(0),640);
  view.stopSampler()
});

test("camera upload preserves an admitted frame when the socket backs up during encoding",async()=>{
  const view=cameraSender();view.startSampler();const pending=view.tick();
  view.socket.bufferedAmount=2*1024*1024;view.finish();await pending;
  assert.equal(view.packets.length,1);assert.equal(view.state.cameraBusy,false);
  view.stopSampler()
});

test("an old camera encoding cannot enter a restarted capture session",async()=>{
  const view=cameraSender();view.startSampler();const old=view.tick();
  view.stopSampler();view.startSampler();const fresh=view.tick();
  view.finish();await old;
  assert.equal(view.packets.length,0);assert.equal(view.state.cameraBusy,true);
  view.finish();await fresh;
  assert.equal(view.packets.length,1);assert.equal(view.state.cameraBusy,false);
  view.stopSampler()
});

function sessionStatus(){
  const requests=[],sent=[],effects={starts:0,stops:0,summaries:0,connections:0};
  const state={running:false,pending:false,stopping:false,sessionId:null,sessionRevision:0,
    mode:"camera",info:{api_key_present:true},runtimeErrorMessage:""};
  const socket={send:packet=>sent.push(JSON.parse(packet))};
  const ui={dryRun:{checked:true},statusJson:{},service:{},sourceChip:{},readinessChip:{},lastUpdate:{}};
  const noop=()=>{};
  const context=vm.createContext({state,ui,AbortController,
    detectionEvents:[],sessionBusyStates:new Set(["starting","running","stopping","draining"]),
    setTimeout:()=>1,clearTimeout:noop,
    fetch:()=>new Promise((resolve,reject)=>requests.push({resolve,reject})),
    resetOverallDescription:noop,reflectSessionSensitivities:noop,reset:noop,openFeed:noop,closeFeed:noop,
    startSampler:()=>effects.starts++,stopSampler:()=>effects.stops++,
    say:noop,controls:noop,renderScene:noop,updateAnalysisUi:noop,
    setConnection:()=>effects.connections++,setChannel:noop,
    runtimeErrorText:()=>"",updateAnnotatedOutput:noop,updateProcessingDiagnostics:noop,
    updateRuntimeSummary:()=>effects.summaries++,
    trackingConfigurationRequired:()=>false,processingMethods:()=>[],savedProcessingMethod:()=>"person_tracking",
    selectedFeatures:()=>[],selectedSensitivities:()=>({}),connect:async()=>socket,
    localCameraActive:()=>false,localCameraDisplay:{controller:null},
  });
  vm.runInContext(html.slice(html.indexOf("function message(data)"),html.indexOf("function reset(){"))+
    html.slice(html.indexOf("async function status(){"),html.indexOf("function resetOverallDescription(){"))+
    "globalThis.handlers={message,start,stop,status};",context);
  return {state,context,ui,requests,sent,effects,...context.handlers,
    respond:(request,data)=>request.resolve({ok:true,json:async()=>data})};
}
const activeSession=id=>({pipeline_state:"running",stream_session_id:id,input_mode:"camera",
  ready:true,features:[]});

test("an old idle status cannot stop a newly started camera session",async()=>{
  const view=sessionStatus(),pending=view.status();
  view.message({type:"session",stream_session_id:"new",mode:"camera"});
  assert.equal(view.effects.starts,1);
  view.respond(view.requests[0],{pipeline_state:"idle"});await pending;
  assert.equal(view.state.running,true);assert.equal(view.state.sessionId,"new");
  assert.equal(view.effects.stops,0);assert.equal(view.effects.summaries,0);
  assert.equal(view.state.statusLoading,false);
  const fresh=view.status();view.respond(view.requests[1],activeSession("new"));await fresh;
  assert.equal(view.effects.starts,1);assert.equal(view.effects.stops,0);
  assert.equal(view.effects.summaries,1);
});

test("an old active status cannot revive the same session after its ended message",async()=>{
  const view=sessionStatus();view.message({type:"session",stream_session_id:"same",mode:"camera"});
  const pending=view.status();view.message({type:"ended",reason:"stopped"});
  view.respond(view.requests[0],activeSession("same"));await pending;
  assert.equal(view.state.sessionId,"same");assert.equal(view.state.running,false);
  assert.equal(view.effects.stops,1);assert.equal(view.effects.summaries,0);
  assert.equal(view.state.statusLoading,false);
});

test("start intent invalidates old state even before a new session id arrives",async()=>{
  const view=sessionStatus(),pending=view.status();await view.start();
  view.respond(view.requests[0],activeSession("old"));await pending;
  assert.equal(view.state.pending,true);assert.equal(view.state.running,false);
  assert.equal(view.state.sessionId,null);assert.equal(view.effects.summaries,0);
  assert.equal(view.sent[0].type,"start");
});

test("stop intent and its completed drain both invalidate preceding active status",async()=>{
  const view=sessionStatus();view.message({type:"session",stream_session_id:"same",mode:"camera"});
  let finishDrain;view.context.localCameraActive=()=>true;
  view.context.localCameraDisplay.controller={drain:()=>new Promise(resolve=>{finishDrain=resolve})};
  const before=view.status(),stopping=view.stop();await settle();
  view.respond(view.requests[0],activeSession("same"));await before;
  assert.equal(view.effects.summaries,0);
  const during=view.status();finishDrain();await stopping;
  view.respond(view.requests[1],activeSession("same"));await during;
  assert.equal(view.state.stopping,true);assert.equal(view.effects.stops,1);
  assert.equal(view.effects.summaries,0);assert.equal(view.sent[0].type,"stop");
});

test("an obsolete status failure does not overwrite a new session's connection UI",async()=>{
  const view=sessionStatus(),pending=view.status();
  view.message({type:"session",stream_session_id:"new",mode:"camera"});
  view.requests[0].reject(new Error("old request timed out"));await pending;
  assert.equal(view.ui.service.textContent,undefined);assert.equal(view.effects.connections,0);
  assert.equal(view.state.running,true);assert.equal(view.state.statusLoading,false);
});

test("current status still recovers an existing session and stops it on genuine idle",async()=>{
  const view=sessionStatus(),active=view.status();
  view.respond(view.requests[0],activeSession("existing"));await active;
  assert.equal(view.state.running,true);assert.equal(view.state.sessionId,"existing");
  const idle=view.status();view.respond(view.requests[1],{pipeline_state:"idle"});await idle;
  assert.equal(view.state.running,false);assert.equal(view.effects.stops,1);
  assert.equal(view.effects.summaries,2);assert.equal(view.state.statusLoading,false);
});

function cameraSampler(outputFps){
  let tick,period;
  const sent=[],canvas={getContext:()=>({drawImage(){}}),
    toBlob:callback=>callback({arrayBuffer:async()=>new ArrayBuffer(8)})};
  const context=vm.createContext({
    state:{cameraStream:{},cameraGeneration:0,info:{annotated_output:{output_fps:outputFps}},
      ws:{readyState:1,bufferedAmount:0,send:packet=>sent.push(packet)}},
    ui:{cameraPreview:{videoWidth:640,videoHeight:480}},
    cameraConfig:()=>true,cameraSettings:()=>({sample_fps:2.5,max_image_side:320,jpeg_quality:75}),
    document:{createElement:()=>canvas},
    setInterval:(callback,delay)=>{tick=callback;period=delay;return 1},clearInterval:()=>{},
    Uint8Array,DataView,
  });
  vm.runInContext(script.slice(script.indexOf("function startSampler()"),script.indexOf("function clearAnnotation()")),context);
  context.startSampler();
  return {context,sent,period,tick,canvas}
}

test("camera capture schedules 30 FPS independently of 2.5 FPS analysis settings",async()=>{
  for(const outputFps of [undefined,30,60]){
    const sampler=cameraSampler(outputFps);
    assert.equal(sampler.period,1000/30);
    for(let index=0;index<30;index++)await sampler.tick();
    assert.equal(sampler.sent.length,30);
    assert.equal(sampler.context.cameraSettings().sample_fps,2.5)
  }
});

test("30 FPS camera capture pauses while encoding or above the upload bound",async()=>{
  const sampler=cameraSampler(30);
  sampler.context.state.cameraBusy=true;await sampler.tick();
  sampler.context.state.cameraBusy=false;
  sampler.context.state.ws.bufferedAmount=1024*1024+1;await sampler.tick();
  assert.equal(sampler.sent.length,0);
  sampler.context.state.ws.bufferedAmount=0;await sampler.tick();
  assert.equal(sampler.sent.length,1)
});

test("camera delivers an admitted frame if socket delivery stalls during encoding",async()=>{
  const sampler=cameraSampler(30);
  sampler.canvas.toBlob=callback=>{
    sampler.context.state.ws.bufferedAmount=1024*1024+1;
    callback({arrayBuffer:async()=>new ArrayBuffer(8)})
  };
  await sampler.tick();
  assert.equal(sampler.sent.length,1);
  assert.equal(sampler.context.state.cameraBusy,false)
});
