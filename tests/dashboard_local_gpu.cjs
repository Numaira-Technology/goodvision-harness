const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");

function dashboard(settings={}){
  const elements=new Map(),requests=[],messages=[];
  function element(id){
    if(!elements.has(id))elements.set(id,{
      value:"",textContent:"",className:"",style:{},checked:false,
      querySelectorAll:()=>[],classList:{toggle(){}},replaceChildren(){},
      scrollIntoView(){},focus(){this.focused=true},
    });
    return elements.get(id)
  }
  const context=vm.createContext({
    document:{getElementById:element,querySelectorAll:()=>[{checked:true,dataset:{feature:"smoking"}}]},
    window:{isSecureContext:true},location:{hostname:"localhost"},
    initial:{persistence_available:true,configured_stream:true,dry_run:false,
      local_inference_enabled:false,local_tracking_enabled:false,
      local_gpu:{available:true,gpu_count:2,inference_device:"cuda:0",tracking_device:"cuda:1"},...settings},
    capture:raw=>messages.push(JSON.parse(raw)),
    fetch:async(url,options)=>{
      const payload=options?.body?JSON.parse(options.body):null;requests.push({url,payload});
      if(context.fail)return {ok:false,json:async()=>({detail:"save failed"})};
      if(payload)context.saved={...context.saved,...payload};
      return {ok:true,json:async()=>({...context.saved})}
    },
  });
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script.slice(0,script.indexOf("ui.defaultStreamUrl.oninput="))+`
    state.info={api_key_present:false};state.gallery={ready:true};state.deployment={...initial};
    state.routes={can_probe:true,can_select:true,state:"idle"};globalThis.saved={...initial};
    reset=()=>{};connect=async()=>({send:capture});
    globalThis.originalInfo=info;info=async()=>{};
    configureScene=()=>{};renderSensitivities=()=>{};updateAnalysisUi=()=>{};
    renderDeploymentSettings(true,true);
    globalThis.app={state,start,controls,renderDeploymentSettings,renderLocalGpuSettings,
      saveDeploymentSettings,loadDeploymentSettings,checkRoutes,selectRoute,loadRoutes,
      localInferenceEnabled,originalInfo};
  `,context);
  return {html,element,requests,messages,context,...context.app}
}

test("local compute switches save independently and discard restores their saved values",async()=>{
  const view=dashboard();
  assert.match(view.html,/id="local-inference-enabled"[^>]*aria-describedby/);
  assert.match(view.html,/id="local-tracking-enabled"[^>]*aria-describedby/);
  for(const [inference,tracking] of [[true,false],[false,true],[true,true],[false,false]]){
    view.element("local-inference-enabled").checked=inference;
    view.element("local-tracking-enabled").checked=tracking;
    await view.saveDeploymentSettings();
    const request=view.requests.at(-1);
    assert.equal(request.url,"/v1/deployment/settings");
    assert.equal(request.payload.local_inference_enabled,inference);
    assert.equal(request.payload.local_tracking_enabled,tracking);
    assert.equal(view.state.deployment.local_inference_enabled,inference);
    assert.equal(view.state.deployment.local_tracking_enabled,tracking)
  }
  view.element("local-inference-enabled").checked=true;
  view.element("local-tracking-enabled").checked=true;
  view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/待保存/);
  await view.loadDeploymentSettings(true);
  assert.equal(view.element("local-inference-enabled").checked,false);
  assert.equal(view.element("local-tracking-enabled").checked,false)
});

test("one GPU is shared and two GPUs reserve the second for tracking and other work",()=>{
  const view=dashboard({local_inference_enabled:true,local_tracking_enabled:true});
  assert.match(view.element("local-gpu-status").textContent,/2 张.*大模型 → GPU 0.*人物追踪及其他本地 GPU 任务 → GPU 1/);
  view.state.deployment.local_gpu={available:true,gpu_count:1,inference_device:"cuda:0",tracking_device:"cuda:0"};
  view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/1 张.*大模型 → GPU 0.*人物追踪及其他本地 GPU 任务 → GPU 0/);
  view.state.deployment.local_gpu={available:false,gpu_count:0};view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/本地 GPU 服务尚未就绪/);
  assert.match(view.element("local-gpu-status").className,/error/)
});

test("changing the tracking route does not carry the previous route's 人员追踪模块 method into the save",async()=>{
  const view=dashboard({local_tracking_enabled:true});
  vm.runInContext(`processingMethods=()=>[{id:"person_tracking",available:true}];
    renderProcessingSettings=()=>{};`,view.context);
  view.element("local-tracking-enabled").checked=false;
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].payload.local_tracking_enabled,false);
  assert.equal(Object.hasOwn(view.requests[0].payload,"processing_method"),false);
  view.element("local-tracking-enabled").checked=true;
  await view.saveDeploymentSettings();
  assert.equal(view.requests.at(-1).payload.local_tracking_enabled,true);
  assert.equal(Object.hasOwn(view.requests.at(-1).payload,"processing_method"),false)
});

test("switching to local inference does not persist a staged cloud API key",async()=>{
  const view=dashboard();
  view.element("default-api-key").value="staged-cloud-key";
  view.element("local-inference-enabled").checked=true;
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].payload.local_inference_enabled,true);
  assert.equal(Object.hasOwn(view.requests[0].payload,"api_key"),false);
  assert.equal(view.element("default-api-key").value,"");
  assert.match(view.element("deployment-note").textContent,/本次输入未保存/)
});

test("GPU changes are locked while sessions and settings operations are busy",()=>{
  const view=dashboard();
  for(const flag of ["running","pending","stopping","deploymentLoading","deploymentSaving","routeSubmitting"]){
    view.state[flag]=true;view.controls();
    assert.equal(view.element("local-inference-enabled").disabled,true,flag);
    assert.equal(view.element("local-tracking-enabled").disabled,true,flag);
    view.state[flag]=false
  }
  for(const status of [{pipeline_state:"draining"},{processing_cleanup:"closing"}]){
    view.state.processingStatus=status;view.controls();
    assert.equal(view.element("local-inference-enabled").disabled,true);
    assert.equal(view.element("local-tracking-enabled").disabled,true)
  }
  view.state.processingStatus=null;view.controls();
  assert.equal(view.element("local-inference-enabled").disabled,false);
  assert.equal(view.element("local-tracking-enabled").disabled,false)
});

test("saved local inference starts without a cloud key and hides or prevents cloud route actions",async()=>{
  const view=dashboard({local_inference_enabled:true});
  assert.equal(view.element("dry-run").checked,false);
  assert.equal(view.element("cloud-model-settings").hidden,true);
  assert.equal(view.element("cloud-route-panel").hidden,true);
  assert.equal(view.element("route-check-button").disabled,true);
  assert.equal(view.element("clear-api-key-button").hidden,true);
  await view.checkRoutes();await view.selectRoute("A");await view.loadRoutes();
  assert.equal(view.requests.length,0);
  await view.start();
  assert.equal(view.messages.length,1);
  assert.equal(view.messages[0].dry_run,false)
});

test("a staged local switch does not bypass the active remote model key requirement",async()=>{
  const view=dashboard({local_tracking_enabled:true});
  view.element("local-inference-enabled").checked=true;
  view.renderLocalGpuSettings();view.controls();
  assert.equal(view.element("cloud-model-settings").hidden,false);
  assert.equal(view.element("cloud-route-panel").hidden,false);
  await view.start();
  assert.equal(view.messages.length,0);
  assert.equal(view.element("default-api-key").focused,true);
  view.context.fail=true;await view.saveDeploymentSettings();
  assert.equal(view.localInferenceEnabled(),false);
  assert.equal(view.element("local-inference-enabled").checked,true);
  assert.match(view.element("deployment-note").textContent,/Could not save/)
});

test("switching back from local inference restores cloud credentials and route controls",async()=>{
  const view=dashboard({local_inference_enabled:true});
  view.element("local-inference-enabled").checked=false;
  await view.saveDeploymentSettings();
  assert.equal(view.element("cloud-model-settings").hidden,false);
  assert.equal(view.element("cloud-route-panel").hidden,false);
  assert.equal(view.element("route-check-button").disabled,false);
  await view.start();assert.equal(view.messages.length,0);
  assert.match(view.element("status-line").textContent,/API key/)
});

test("initial info does not force Test mode on when the saved local model needs no cloud key",async()=>{
  const view=dashboard({local_inference_enabled:true});
  view.state.deployment=null;
  await view.originalInfo();
  assert.equal(view.element("dry-run").checked,false);
  assert.equal(view.element("cloud-model-settings").hidden,true);
  view.state.info.local_inference_enabled=true;
  view.state.deployment={local_inference_enabled:false};
  assert.equal(view.localInferenceEnabled(),false,"saved settings take precedence over stale info")
});
