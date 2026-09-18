const assert=require("node:assert/strict");
const vm=require("node:vm");
const {test}=require("node:test");
const {script} = require("./helpers/dashboard.cjs");

function player(){
  let now=0,id=0;
  const timers=new Map(),requests=[],revoked=[],rawFeed={dataset:{},removeAttribute(name){delete this[name]}};
  const context=vm.createContext({
    state:{running:true,sessionId:"one",info:{annotated_output:{output_fps:15}}},
    ui:{rawFeed},document:{hidden:false},performance:{now:()=>now},AbortController,
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
  return {context,requests,revoked,rawFeed,timers,...context.player,
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

function cameraSampler(outputFps){
  let tick,period;
  const sent=[],canvas={getContext:()=>({drawImage(){}}),
    toBlob:callback=>callback({arrayBuffer:async()=>new ArrayBuffer(8)})};
  const context=vm.createContext({
    state:{cameraStream:{},info:{annotated_output:{output_fps:outputFps}},
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

test("30 FPS camera capture drops work while encoding or socket delivery is busy",async()=>{
  const sampler=cameraSampler(30);
  sampler.context.state.cameraBusy=true;await sampler.tick();
  sampler.context.state.cameraBusy=false;
  sampler.context.state.ws.bufferedAmount=1;await sampler.tick();
  assert.equal(sampler.sent.length,0);
  sampler.context.state.ws.bufferedAmount=0;await sampler.tick();
  assert.equal(sampler.sent.length,1)
});

test("camera skips a captured frame if socket delivery stalls during encoding",async()=>{
  const sampler=cameraSampler(30);
  sampler.canvas.toBlob=callback=>{
    sampler.context.state.ws.bufferedAmount=1;
    callback({arrayBuffer:async()=>new ArrayBuffer(8)})
  };
  await sampler.tick();
  assert.equal(sampler.sent.length,0);
  assert.equal(sampler.context.state.cameraBusy,false)
});
