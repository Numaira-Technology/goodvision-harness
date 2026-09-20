const assert=require("node:assert/strict");
const vm=require("node:vm");
const {test}=require("node:test");
const {createDocument, runDashboard} = require("./helpers/dashboard.cjs");

function dashboard(){
  const {document, element} = createDocument();
  const requests=[];
  const context=vm.createContext({
    document,
    window:{isSecureContext:true},location:{hostname:"localhost"},
    fetch:async(url,options)=>{
      requests.push({url,options});
      if(context.fail)return {ok:false,json:async()=>({detail:"storage unavailable"})};
      const saved=JSON.parse(options.body).buffer_delay_seconds;
      return {ok:true,json:async()=>({persistence_available:true,buffer_delay_seconds:saved})}
    },
  });
  runDashboard(context, `
    state.running=true;state.gallery={ready:true};
    state.info={annotated_output:{output_fps:15,buffer_delay_seconds:2.5}};
    state.deployment={persistence_available:true,buffer_delay_seconds:2.5};
    status=async()=>updateAnnotatedOutput({...state.info.annotated_output,state:"running"});
    controls();updateAnnotatedOutput(state.info.annotated_output);
    globalThis.app={state,saveBufferDelay,controls,updateAnnotatedOutput};
  `);
  return {context,requests,element,...context.app}
}

test("a running session can save fractional buffering without submitting other edits",async()=>{
  const view=dashboard();
  assert.equal(view.element("buffer-delay-seconds").value,"2.5");
  assert.equal(view.element("save-buffer-delay-button").disabled,false);
  assert.equal(view.element("save-deployment-button").disabled,true);
  view.element("default-stream-url").value="rtsp://unsaved/live";
  view.element("default-api-key").value="unsaved-key";
  view.element("buffer-delay-seconds").value="3.75";view.state.bufferDelayDirty=true;
  await view.saveBufferDelay();
  assert.equal(view.requests.length,1);
  assert.equal(view.requests[0].url,"/v1/deployment/settings");
  assert.deepEqual(JSON.parse(view.requests[0].options.body),{buffer_delay_seconds:3.75});
  assert.equal(view.state.info.annotated_output.buffer_delay_seconds,3.75);
  assert.equal(view.element("default-stream-url").value,"rtsp://unsaved/live");
  assert.equal(view.element("default-api-key").value,"unsaved-key");
  assert.match(view.element("buffer-delay-note").textContent,/已应用并保存为 3.75 秒/);
  assert.equal(view.element("save-buffer-delay-button").disabled,false)
});

test("zero buffering is valid and invalid input never makes a request",async()=>{
  const view=dashboard();
  for(const raw of [""," ","-1","NaN","Infinity","30.1"]){
    view.element("buffer-delay-seconds").value=raw;
    await view.saveBufferDelay();
    assert.equal(view.requests.length,0);
    assert.match(view.element("buffer-delay-note").textContent,/请输入/)
  }
  view.element("buffer-delay-seconds").value="0";
  await view.saveBufferDelay();
  assert.deepEqual(JSON.parse(view.requests[0].options.body),{buffer_delay_seconds:0})
});

test("polling preserves edits and a failed save preserves the effective buffer",async()=>{
  const view=dashboard();
  view.element("buffer-delay-seconds").value="4";view.state.bufferDelayDirty=true;
  view.updateAnnotatedOutput({state:"running",buffer_delay_seconds:2.5});
  assert.equal(view.element("buffer-delay-seconds").value,"4");
  view.context.fail=true;
  await view.saveBufferDelay();
  assert.equal(view.state.info.annotated_output.buffer_delay_seconds,2.5);
  assert.equal(view.state.bufferDelayDirty,true);
  assert.equal(view.element("buffer-delay-seconds").value,"4");
  assert.match(view.element("buffer-delay-note").textContent,/保存失败/);
  assert.equal(view.element("save-buffer-delay-button").disabled,false)
});

test("buffer controls honor unavailable storage and settings operations",()=>{
  const view=dashboard();
  for(const name of ["deploymentLoading","deploymentSaving","pending"]){
    view.state[name]=true;view.controls();
    assert.equal(view.element("save-buffer-delay-button").disabled,true);
    view.state[name]=false
  }
  view.state.deployment.persistence_available=false;view.controls();
  assert.equal(view.element("buffer-delay-seconds").disabled,true)
});

test("zero extra buffer is distinct from measured output-frame delay",()=>{
  const view=dashboard();
  view.updateAnnotatedOutput({state:"running",buffer_delay_seconds:0,stream_delay_seconds:1.25,
    output_fps:15,annotation_mode:"scene"});
  assert.equal(view.element("buffer-delay-seconds").value,"0");
  assert.match(view.element("stream-meta").textContent,/额外缓存 0.0 秒/);
  assert.match(view.element("raw-badge-text").textContent,/输出帧延迟 1.3 秒/);
  assert.doesNotMatch(view.element("raw-badge-text").textContent,/实际延迟|零延迟/)
});
