"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

function fixture() {
  let clock=0;
  const requests=[];
  const canvas=()=>{
    const context={boxes:[],colors:[],images:0,clearRect(){this.boxes=[];this.colors=[];},
      drawImage(image){this.images++;this.lastImage=image;},setLineDash(){},strokeRect(...box){this.boxes.push(box);this.colors.push(this.strokeStyle);},fillText(){}};
    return {width:640,height:360,getContext:()=>context,toDataURL:()=>"data:image/jpeg;base64,ZmFrZQ==",context};
  };
  const context={window:{currentStage:"test-stage",trackingClient:"client",previewDetector:true},
    document:{createElement:()=>canvas()},performance:{now:()=>clock},AbortController,
    setTimeout:()=>1,clearTimeout:()=>{},
    fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject,body:JSON.parse(options.body)}))};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../tools/vlm_benchmark/preview_tracking.js"),"utf8"),context);
  const video={seeking:false,videoWidth:320,videoHeight:180,listeners:{},
    addEventListener(event,fn){this.listeners[event]=fn;},requestVideoFrameCallback:()=>1,cancelVideoFrameCallback(){}};
  const overlay=canvas(),note={textContent:""},evidence={canvas:canvas(),note:{textContent:""},container:{hidden:true}};
  const tracker=new context.window.PreviewTracking(video,overlay,note,0,{duration:6},evidence);
  tracker.playing=true;
  return {tracker,overlay,note,evidence,requests,advance:(ms,time)=>{clock=ms;return tracker.frame(ms,{mediaTime:time});}};
}
const box={track_id:"p1",bbox:{x1:.1,y1:.1,x2:.3,y2:.8},event_state:"alert"};
function reply(request,objects=[box]) {
  request.resolve({ok:true,json:async()=>({detector:true,source_time:request.body.time,
    epoch:request.body.epoch,status:"tracking",objects})});
}

test("current detector boxes are neutral and never replace the live video JPEG",async()=>{
  const f=fixture(),pending=f.advance(0,1.0);
  reply(f.requests[0]);await pending;
  assert.equal(f.overlay.context.boxes.length,1);
  assert.deepEqual(f.overlay.context.colors,["#ffd68b"]);
  assert.equal(f.overlay.context.images,0);
  assert.match(f.note.textContent,/框不表示事件判断/);
  assert.equal(f.requests[0].body.time,1);
});
test("slow valid detector results remain source-aligned evidence without boxes on newer frames",async()=>{
  const f=fixture(),first=f.advance(0,0);
  reply(f.requests[0]);await first;
  const pending=f.advance(67,.067);
  await f.advance(800,.8);
  assert.equal(f.overlay.context.boxes.length,0);
  reply(f.requests[1]);await pending;
  assert.equal(f.tracker.stats.staleResponses,1);
  assert.equal(f.tracker.stats.roundTripSeconds.at(-1),.733);
  assert.equal(f.tracker.lastObjects.length,0);
  assert.equal(f.overlay.context.boxes.length,0);
  assert.equal(f.evidence.container.hidden,false);
  assert.equal(f.evidence.canvas.context.boxes.length,1);
  assert.equal(f.evidence.canvas.context.colors[0],"#ffd68b");
  assert.notEqual(f.evidence.canvas.context.lastImage,f.tracker.captureCanvas);
  assert.match(f.evidence.note.textContent,/证据 0.07 s/);
  assert.match(f.evidence.note.textContent,/0.73 s 前/);
  const next=f.advance(1000,1.0);
  assert.match(f.evidence.note.textContent,/0.93 s 前/);
  f.tracker.playing=false;
  reply(f.requests[2]);await next;
  assert.equal(f.evidence.container.hidden,true);
  assert.equal(f.evidence.canvas.context.boxes.length,0);
});
test("a loop wrap invalidates an in-flight detector frame and uses the new epoch",async()=>{
  const f=fixture(),pending=f.advance(0,5.9);
  await f.advance(100,.01);
  assert.equal(f.tracker.epoch,1);
  assert.equal(f.requests[0].options.signal.aborted,true);
  reply(f.requests[0]);await pending;
  assert.equal(f.overlay.context.boxes.length,0);
  assert.equal(f.tracker.stats.staleResponses,1);
  assert.equal(f.evidence.container.hidden,true);
  const fresh=f.advance(150,.08);
  assert.equal(f.requests[1].body.epoch,1);
  assert.equal(f.requests[1].body.time,6.08);
  reply(f.requests[1]);await fresh;
  assert.equal(f.overlay.context.boxes.length,1);
});
test("stop clears boxes immediately and ignores responses arriving after stop",async()=>{
  const f=fixture(),pending=f.advance(0,0);
  f.tracker.playing=false;
  assert.equal(f.requests[0].options.signal.aborted,true);
  assert.equal(f.overlay.context.boxes.length,0);
  reply(f.requests[0]);await pending;
  assert.equal(f.tracker.lastObjects.length,0);
  assert.equal(f.tracker.stats.lastStatus,"stopped");
  assert.match(f.note.textContent,/播放已结束/);
  assert.equal(f.evidence.container.hidden,true);
});
test("seek, close and detector failures immediately clear retained boxes",async()=>{
  for(const action of ["seek","close","failure"]){
    const f=fixture(),first=f.advance(0,0);
    reply(f.requests[0]);await first;
    if(action==="seek") f.tracker.video.listeners.seeking();
    else if(action==="close") f.tracker.close();
    else {const pending=f.advance(67,.067);f.requests[1].reject(new Error("offline"));await pending;}
    assert.equal(f.tracker.lastObjects.length,0);
    assert.equal(f.overlay.context.boxes.length,0);
    assert.equal(f.evidence.container.hidden,true);
  }
});
test("wrong echoed source metadata cannot label current frames",async()=>{
  const f=fixture(),pending=f.advance(0,0);
  f.requests[0].resolve({ok:true,json:async()=>({detector:true,source_time:2,epoch:0,status:"tracking",objects:[box]})});
  await pending;
  assert.equal(f.overlay.context.boxes.length,0);
  assert.equal(f.tracker.stats.staleResponses,1);
});

test("a three-second serialized GPU reply remains inspectable in its matched evidence view",async()=>{
  const f=fixture(),pending=f.advance(0,1.0);
  await f.advance(3100,4.1);
  reply(f.requests[0]);await pending;
  assert.equal(f.overlay.context.boxes.length,0);
  assert.equal(f.evidence.container.hidden,false);
  assert.equal(f.evidence.canvas.context.boxes.length,1);
  assert.match(f.evidence.note.textContent,/证据 1.00 s/);
  assert.match(f.evidence.note.textContent,/3.10 s 前/);
  assert.equal(f.tracker.stats.evidenceFramesWithBoxes,1);
  assert.equal(f.tracker.stats.lastStatus,"evidence_only");
});
