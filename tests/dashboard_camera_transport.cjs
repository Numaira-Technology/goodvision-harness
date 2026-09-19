const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const {test}=require("node:test");
const html=fs.readFileSync(require.resolve("../src/goodvision_harness/dashboard.html"),"utf8");

test("camera WebSocket routes independent metadata without consuming or altering upload acknowledgements",()=>{
  const delivered=[],acknowledged=[];
  const context=vm.createContext({localCameraDisplay:{controller:{
    receiveMetadata:(data,options)=>delivered.push({data,options}),
    acknowledgeUpload:data=>acknowledged.push(data),
  }}});
  vm.runInContext(html.slice(html.indexOf("function message(data){"),html.indexOf("async function start(){")),context);
  const metadata={type:"camera_display_metadata",metadata_revision:10,anchors:[]},
    ack={type:"camera_frame_ack",source_sequence:20,accepted:true};
  context.message(metadata);context.message(ack);
  assert.equal(delivered.length,1);assert.equal(delivered[0].data,metadata);
  assert.equal(delivered[0].options.transport,"websocket");
  assert.deepEqual(acknowledged,[ack]);
  context.localCameraDisplay.controller=null;context.message(metadata);context.message(ack);
  assert.equal(delivered.length,1);assert.equal(acknowledged.length,1);
});

function cameraPage({allowed=true,playError=false,captureError=false,displayEnabled=true,supported=true}={}){
  const calls=[],warmups=[],stream={getTracks:()=>[{stop:()=>calls.push("track_stop")}]};
  const state={cameraStream:null,info:{local_camera_display:{enabled:displayEnabled,motion_backend:"dense"}}};
  const ui={cameraPreview:{srcObject:null,play:async()=>{
    calls.push("play");if(playError)throw new Error("preview failed");}},cameraButton:{},cameraNote:{},start:{textContent:"Start"}};
  const context=vm.createContext({state,ui,cameraAllowed:allowed,
    navigator:{mediaDevices:{getUserMedia:async()=>{
      calls.push("camera");if(captureError)throw new Error("capture failed");return stream;}}},
    GoodvisionCameraDisplay:{BrowserController:{supported:()=>supported,
      warmupMotion:options=>{calls.push("warmup");warmups.push(options);return Promise.resolve(true);},
      discardWarmup:()=>calls.push("discard")}},
    stopSampler:()=>calls.push("stop_sampler"),surfaces(){},controls(){},say:message=>calls.push(message)});
  vm.runInContext(html.slice(html.indexOf("async function camera(){"),html.indexOf("function startSampler(){")),context);
  return {context,calls,warmups,stream,state,ui};
}

test("motion prewarm starts only after preview succeeds, and disabling preview disposes the unused worker",async()=>{
  const page=cameraPage();await page.context.camera();
  assert.equal(page.warmups.length,1);assert.ok(page.calls.indexOf("warmup")>page.calls.indexOf("play"));
  assert.equal(page.state.cameraStream,page.stream);assert.equal(page.ui.cameraNote.className,"note");
  await page.context.camera();assert.equal(page.state.cameraStream,null);
  assert.deepEqual(page.calls.slice(-3),["stop_sampler","discard","track_stop"]);
  assert.equal(page.ui.cameraPreview.srcObject,null);assert.equal(page.warmups.length,1);
});

test("preview failures, denied camera access and unsupported display never launch warmup",async()=>{
  for(const options of [{allowed:false},{captureError:true},{playError:true},{displayEnabled:false},{supported:false}]){
    const page=cameraPage(options);await page.context.camera();assert.equal(page.warmups.length,0);
    if(page.state.cameraStream)page.context.disableCamera();
  }
});

test("warmup errors remain visible only for their matching preview and do not poison a later preview",async()=>{
  const page=cameraPage();await page.context.camera();const failure=page.warmups[0].onFailure;
  failure("预热失败，将重试");assert.equal(page.ui.cameraNote.className,"note error");
  assert.match(page.ui.cameraNote.textContent,/预热失败/);
  page.context.disableCamera();failure("late failure");assert.equal(page.ui.cameraNote.textContent,"camera off");
  await page.context.camera();assert.equal(page.ui.cameraNote.className,"note");
});

test("page unload stops the active run and disposes any unclaimed motion warmup",()=>{
  const calls=[],handlers={},context=vm.createContext({
    window:{addEventListener:(name,handler)=>handlers[name]=handler},sceneView:{},archiveView:{stream:null},events:null,
    stopScenePolling:()=>calls.push("scene_stop"),stopSampler:()=>calls.push("run_stop"),resetAnnotations(){},
    GoodvisionCameraDisplay:{BrowserController:{discardWarmup:()=>calls.push("discard")}},
    state:{cameraStream:{getTracks:()=>[{stop:()=>calls.push("track_stop")}]},ws:{close:()=>calls.push("socket_close")}},
  });
  const start=html.indexOf('window.addEventListener("beforeunload",');
  vm.runInContext(html.slice(start,html.indexOf('document.addEventListener("visibilitychange",',start)),context);
  handlers.beforeunload();assert.equal(context.sceneView.closed,true);
  assert.deepEqual(calls,["scene_stop","run_stop","discard","track_stop","socket_close"]);
});

test("local display explains its startup buffer and never labels an unseen frame as 15 FPS",()=>{
  let options;
  const ui={cameraPreview:{},localCameraCanvas:{dataset:{}},rawBadgeText:{},annotationNote:{},cameraNote:{}};
  const context=vm.createContext({ui,state:{sessionId:"run",sessionMode:"camera",cameraGeneration:1,
    info:{local_camera_display:{enabled:true,protocol:1,delay_seconds:6}}},
    GoodvisionCameraDisplay:{BrowserController:class{
      static supported(){return true;}
      constructor(value){options=value;this.frames={delay:6};}
      start(){}
    }},cameraSettings:()=>({}),stopLatestFeed(){},resetAnnotations(){},surfaces(){},
  });
  vm.runInContext(html.slice(html.indexOf("const localCameraDisplay="),html.indexOf("function stopLocalCameraDisplay(){")),context);
  assert.equal(context.startLocalCameraDisplay(),true);
  assert.equal(ui.rawBadgeText.textContent,"正在缓冲画面以等待完整轮廓");
  assert.match(ui.annotationNote.textContent,/本机描边缓冲 6 秒/);
  options.onStatus({distinct_presented_frames:0,distinct_presented_fps:null});
  assert.equal(ui.rawBadgeText.textContent,"正在缓冲画面以等待完整轮廓");
  options.onStatus({distinct_presented_frames:1,distinct_presented_fps:null});
  assert.match(ui.rawBadgeText.textContent,/首帧已显示/);assert.doesNotMatch(ui.rawBadgeText.textContent,/15/);
  options.onStatus({distinct_presented_frames:20,distinct_presented_fps:19.5,observed_contour_frames:10});
  assert.match(ui.rawBadgeText.textContent,/19.5 帧\/秒/);
  options.onStatus({distinct_presented_frames:0,distinct_presented_fps:null,display_buffer_capacity_failures:1});
  assert.equal(ui.rawBadgeText.textContent,"画面缓冲配置不可用");
});
