const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");
const {createDocument,runDashboard}=require("./helpers/dashboard.cjs");
const {payload:catalogPayload}=require("./helpers/event_catalog.cjs");

test("analysis protocol defaults to compatible API and persists GoodVision independently",async()=>{
  const view=dashboard({inference_mode:"server",tracking_mode:"legacy",
    inference_server_url:"http://analysis:8094",inference_server_model:"vision"});
  const protocol=view.element("inference-server-protocol");
  assert.equal(protocol.value,"compatible");
  protocol.value="goodvision";view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/待保存/);
  await view.saveDeploymentSettings();
  const payload=view.requests.at(-1).payload;
  assert.equal(payload.inference_server_protocol,"goodvision");
  assert.equal(payload.inference_server_url,"http://analysis:8094");
  assert.equal(payload.inference_server_model,"vision");
  assert.equal(payload.tracking_mode,"legacy");
  assert.equal(Object.hasOwn(payload,"api_key"),false);
  assert.equal(view.state.deployment.inference_server_protocol,"goodvision");
  protocol.value="compatible";await view.loadDeploymentSettings(true);
  assert.equal(protocol.value,"goodvision");
});

test("invalid analysis protocol is rejected and failed save preserves the draft",async()=>{
  const view=dashboard({inference_mode:"server",inference_server_url:"http://analysis:8094"});
  const protocol=view.element("inference-server-protocol");
  protocol.value="invalid";await view.saveDeploymentSettings();
  assert.equal(view.requests.length,0);assert.equal(protocol.focused,true);
  protocol.value="goodvision";view.context.fail=true;await view.saveDeploymentSettings();
  assert.equal(view.state.deployment.inference_server_protocol,undefined);
  assert.equal(protocol.value,"goodvision");
});

test("analysis protocol locks with shared settings and survives route saves",async()=>{
  const view=dashboard({inference_mode:"server",inference_server_protocol:"goodvision",
    inference_server_url:"http://analysis:8094"});
  view.context.GoodvisionFleet={extraBusy:()=>true};view.controls();
  assert.equal(view.element("inference-server-protocol").disabled,true);
  view.context.GoodvisionFleet={extraBusy:()=>false};
  for(const mode of ["local","cloud"]){
    view.element("inference-mode").value=mode;view.renderLocalGpuSettings();view.controls();
    assert.equal(view.element("inference-server-fields").hidden,true);
    await view.saveDeploymentSettings();
    assert.equal(Object.hasOwn(view.requests.at(-1).payload,"inference_server_protocol"),false);
    assert.equal(view.state.deployment.inference_server_protocol,"goodvision");
  }
});

function dashboard(settings={}){
  const requests=[],messages=[];
  const {document,element}=createDocument(["smoking"]);
  const context=vm.createContext({
    document,
    window:{isSecureContext:true},location:{hostname:"localhost"},URL,
    initial:{persistence_available:true,configured_stream:true,dry_run:false,
      local_inference_enabled:false,local_tracking_enabled:false,
      local_gpu:{available:true,gpu_count:2,inference_device:"cuda:0",tracking_device:"cuda:1"},...settings},
    capture:raw=>messages.push(JSON.parse(raw)),
    fetch:async(url,options)=>{
      const payload=options?.body?JSON.parse(options.body):null;requests.push({url,payload});
      if(context.fail)return {ok:false,json:async()=>({detail:"save failed"})};
      if(payload){
        const {tracking_api_key,clear_tracking_api_key,...publicFields}=payload;
        context.saved={...context.saved,...publicFields};
        if(tracking_api_key)context.saved.tracking_key_present=true;
        if(clear_tracking_api_key)context.saved.tracking_key_present=false
      }
      return {ok:true,json:async()=>({...context.saved,...catalogPayload()})}
    },
  });
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  runDashboard(context,`
    state.info={api_key_present:false};state.gallery={ready:true};state.deployment={...initial};
    state.routes={can_probe:true,can_select:true,state:"idle"};globalThis.saved={...initial};
    reset=()=>{};connect=async()=>({send:capture});
    globalThis.originalInfo=info;info=async()=>{};
    configureScene=()=>{};renderSensitivities=()=>{};updateAnalysisUi=()=>{};
    renderDeploymentSettings(true,true);
    globalThis.app={state,start,controls,renderDeploymentSettings,renderLocalGpuSettings,
      saveDeploymentSettings,loadDeploymentSettings,checkRoutes,selectRoute,loadRoutes,
      localInferenceEnabled,savedExecutionMode,originalInfo,saveApiKey};
  `,context);
  return {html,element,requests,messages,context,...context.app}
}

test("compute mode selectors save independently and discard restores their saved values",async()=>{
  const view=dashboard();
  assert.match(view.html,/id="local-inference-enabled"[^>]*aria-describedby/);
  assert.match(view.html,/id="local-tracking-enabled"[^>]*aria-describedby/);
  assert.match(view.html,/<select id="inference-mode"/);
  assert.match(view.html,/<select id="tracking-mode"/);
  for(const [inference,tracking] of [[true,false],[false,true],[true,true],[false,false]]){
    view.element("inference-mode").value=inference?"local":"cloud";
    view.element("tracking-mode").value=tracking?"local":"legacy";
    await view.saveDeploymentSettings();
    const request=view.requests.at(-1);
    assert.equal(request.url,"/v1/deployment/settings");
    assert.equal(request.payload.local_inference_enabled,inference);
    assert.equal(request.payload.local_tracking_enabled,tracking);
    assert.equal(request.payload.inference_mode,inference?"local":"cloud");
    assert.equal(request.payload.tracking_mode,tracking?"local":"legacy");
    assert.equal(view.state.deployment.local_inference_enabled,inference);
    assert.equal(view.state.deployment.local_tracking_enabled,tracking)
  }
  view.element("inference-mode").value="local";
  view.element("tracking-mode").value="local";
  view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/待保存/);
  await view.loadDeploymentSettings(true);
  assert.equal(view.element("local-inference-enabled").checked,false);
  assert.equal(view.element("local-tracking-enabled").checked,false)
});

test("sampling FPS validates, saves and reloads independently of video buffering",async()=>{
  const view=dashboard({tracking_sample_fps:2,buffer_delay_seconds:2});
  const input=view.element("tracking-sample-fps");
  assert.equal(input.value,"2");
  input.value="1.5";view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/待保存/);
  await view.saveDeploymentSettings();
  assert.equal(view.requests.at(-1).payload.tracking_sample_fps,1.5);
  assert.equal(Object.hasOwn(view.requests.at(-1).payload,"buffer_delay_seconds"),false);
  assert.equal(view.state.deployment.tracking_sample_fps,1.5);
  input.value="4";await view.loadDeploymentSettings(true);assert.equal(input.value,"1.5");
  for(const value of ["","0","NaN","Infinity","30.1"]){
    input.value=value;const count=view.requests.length;await view.saveDeploymentSettings();
    assert.equal(view.requests.length,count);assert.equal(input.focused,true);
    assert.match(view.element("deployment-note").textContent,/采样帧率/);
  }
  view.state.running=true;view.controls();assert.equal(input.disabled,true);
});

test("cloud GPU selection saves without a local GPU and shows editable rental controls",async()=>{
  const view=dashboard({local_gpu:{available:false,gpu_count:0}});
  view.element("tracking-mode").value="cloud";view.renderLocalGpuSettings();
  assert.equal(view.element("cloud-gpu-panel").hidden,false);
  assert.equal(view.element("tracking-server-fields").hidden,true);
  await view.saveDeploymentSettings();
  assert.equal(view.requests.at(-1).payload.tracking_mode,"cloud");
  assert.equal(view.requests.at(-1).payload.local_tracking_enabled,false);
  assert.match(view.html,/id="cloud-gpu-count"/);
  assert.match(view.html,/id="cloud-gpu-script"/);
});

test("SAM-only deployment allows local tracking while disabling the absent local VLM",async()=>{
  const view=dashboard({local_gpu:{available:true,gpu_count:2,
    tracking_device:0,tracking_available:true,inference_available:false}});
  view.controls();
  assert.equal(view.element("inference-mode-local").disabled,true);
  assert.equal(view.element("tracking-mode-local").disabled,false);
  view.element("tracking-mode").value="local";
  view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/人物追踪 → GPU 0/);
  assert.match(view.element("local-gpu-status").textContent,/大模型可使用云端 API/);
  await view.saveDeploymentSettings();
  assert.equal(view.requests.at(-1).payload.tracking_mode,"local");
  assert.equal(view.requests.at(-1).payload.inference_mode,"cloud");
  const count=view.requests.length;
  view.element("inference-mode").value="local";
  await view.saveDeploymentSettings();
  assert.equal(view.requests.length,count);
  assert.match(view.element("deployment-note").textContent,/服务未部署或不可用/);
});

test("one GPU is shared and two GPUs reserve the second for tracking and other work",()=>{
  const view=dashboard({local_inference_enabled:true,local_tracking_enabled:true});
  assert.match(view.element("local-gpu-status").textContent,/2 张.*大模型 → GPU 0.*人物追踪及其他本地 GPU 任务 → GPU 1/);
  view.state.deployment.local_gpu={available:true,gpu_count:1,inference_device:"cuda:0",tracking_device:"cuda:0"};
  view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/1 张.*大模型 → GPU 0.*人物追踪及其他本地 GPU 任务 → GPU 0/);
  view.state.deployment.local_gpu={available:false,gpu_count:0};view.renderLocalGpuSettings();
  assert.match(view.element("local-gpu-status").textContent,/本机未检测到可用 GPU/);
  assert.match(view.element("local-gpu-status").className,/error/)
});

test("changing the tracking route does not carry the previous route's 人员追踪模块 method into the save",async()=>{
  const view=dashboard({local_tracking_enabled:true});
  vm.runInContext(`processingMethods=()=>[{id:"person_tracking",available:true}];
    renderProcessingSettings=()=>{};`,view.context);
  view.element("tracking-mode").value="legacy";
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].payload.local_tracking_enabled,false);
  assert.equal(Object.hasOwn(view.requests[0].payload,"processing_method"),false);
  view.element("tracking-mode").value="local";
  await view.saveDeploymentSettings();
  assert.equal(view.requests.at(-1).payload.local_tracking_enabled,true);
  assert.equal(Object.hasOwn(view.requests.at(-1).payload,"processing_method"),false)
});

test("switching to local inference does not persist a staged cloud API key",async()=>{
  const view=dashboard();
  view.element("default-api-key").value="staged-cloud-key";
  view.element("inference-mode").value="local";
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
    for(const id of ["inference-mode","tracking-mode","inference-server-url","inference-server-model","tracking-server-url","tracking-server-capacity","tracking-server-protocol","tracking-node-key","clear-tracking-node-key"])
      assert.equal(view.element(id).disabled,true,flag+": "+id);
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
  view.element("inference-mode").value="local";
  view.renderLocalGpuSettings();view.controls();
  assert.equal(view.element("cloud-model-settings").hidden,false);
  assert.equal(view.element("cloud-route-panel").hidden,false);
  await view.start();
  assert.equal(view.messages.length,0);
  assert.equal(view.element("default-api-key").focused,true);
  view.context.fail=true;await view.saveDeploymentSettings();
  assert.equal(view.localInferenceEnabled(),false);
  assert.equal(view.element("inference-mode").value,"local");
  assert.match(view.element("deployment-note").textContent,/Could not save/)
});

test("switching back from local inference restores cloud credentials and route controls",async()=>{
  const view=dashboard({local_inference_enabled:true});
  view.element("inference-mode").value="cloud";
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

test("independent server modes save on a controller with no local GPU",async()=>{
  for(const inference of ["cloud","server"]){
    const view=dashboard({local_gpu:{available:false,gpu_count:0}});
    view.element("inference-mode").value=inference;
    view.element("tracking-mode").value="server";
    view.element("inference-server-url").value="http://model-box:8000/";
    view.element("inference-server-model").value="customer-vision";
    view.element("tracking-server-url").value="http://tracking-box:8092/";
    view.element("tracking-server-capacity").value="24";
    view.element("default-api-key").value="staged-cloud-key";
    view.renderLocalGpuSettings();view.controls();
    assert.equal(view.element("inference-mode").disabled,false);
    assert.equal(view.element("tracking-mode").disabled,false);
    assert.equal(view.element("inference-mode-local").disabled,true);
    assert.equal(view.element("tracking-mode-local").disabled,true);
    assert.equal(view.element("tracking-server-fields").hidden,false);
    assert.equal(view.element("inference-server-fields").hidden,inference!=="server");
    assert.doesNotMatch(view.element("local-gpu-status").className,/error/);
    await view.saveDeploymentSettings();
    const payload=view.requests[0].payload;
    assert.equal(payload.inference_mode,inference);
    assert.equal(payload.tracking_mode,"server");
    assert.equal(payload.local_inference_enabled,false);
    assert.equal(payload.local_tracking_enabled,false);
    assert.equal(payload.tracking_server_url,"http://tracking-box:8092");
    assert.equal(payload.tracking_server_capacity,24);
    if(inference==="server"){
      assert.equal(payload.inference_server_url,"http://model-box:8000");
      assert.equal(payload.inference_server_model,"customer-vision");
      assert.equal(Object.hasOwn(payload,"api_key"),false);
      assert.match(view.element("deployment-note").textContent,/本次输入未保存/)
    }else assert.equal(payload.api_key,"staged-cloud-key");
    assert.equal(view.element("default-api-key").value,"")
  }
});

test("all nine independently selected execution combinations persist",async()=>{
  for(const inference of ["cloud","local","server"])for(const tracking of ["legacy","local","server"]){
    const view=dashboard();
    view.element("inference-mode").value=inference;view.element("tracking-mode").value=tracking;
    view.element("inference-server-url").value="https://vlm.example:8443";
    view.element("tracking-server-url").value="http://sam.example:8092";
    view.element("tracking-server-capacity").value="100";
    await view.saveDeploymentSettings();
    assert.equal(view.state.deployment.inference_mode,inference);
    assert.equal(view.state.deployment.tracking_mode,tracking);
    assert.equal(view.localInferenceEnabled(),inference!=="cloud");
    assert.equal(view.element("local-inference-enabled").checked,inference==="local");
    assert.equal(view.element("local-tracking-enabled").checked,tracking==="local")
  }
});

test("server credentials and routing use the saved mode even though legacy local booleans are false",async()=>{
  const view=dashboard({inference_mode:"server",tracking_mode:"server",
    inference_server_url:"http://model-box:8000",tracking_server_url:"http://tracking-box:8092",tracking_server_capacity:16,
    local_gpu:{available:false,gpu_count:0}});
  assert.equal(view.localInferenceEnabled(),true);
  assert.equal(view.element("cloud-model-settings").hidden,true);
  assert.equal(view.element("cloud-route-panel").hidden,true);
  assert.equal(view.element("clear-api-key-button").hidden,true);
  assert.equal(view.element("dry-run").checked,false);
  await view.checkRoutes();await view.selectRoute("A");await view.loadRoutes();
  view.element("default-api-key").value="must-not-save";await view.saveApiKey();
  assert.equal(view.requests.length,0);
  await view.start();assert.equal(view.messages.length,1);assert.equal(view.messages[0].dry_run,false);
  view.state.pending=false;
  view.element("inference-mode").value="cloud";view.renderLocalGpuSettings();view.controls();
  assert.equal(view.element("cloud-model-settings").hidden,true,"unsaved cloud choice does not change effective routing");
  await view.saveDeploymentSettings();
  assert.equal(view.element("cloud-model-settings").hidden,false)
});

test("a staged server mode cannot bypass the saved cloud key requirement and survives failed save",async()=>{
  const view=dashboard();
  view.element("inference-mode").value="server";
  view.element("inference-server-url").value="http://model-box:8000";
  view.renderLocalGpuSettings();view.controls();
  assert.equal(view.element("cloud-model-settings").hidden,false);
  await view.start();assert.equal(view.messages.length,0);
  view.context.fail=true;await view.saveDeploymentSettings();
  assert.equal(view.localInferenceEnabled(),false);
  assert.equal(view.element("inference-mode").value,"server");
  assert.equal(view.element("inference-server-url").value,"http://model-box:8000")
});

test("discard restores every saved mode and server field",async()=>{
  const view=dashboard({inference_mode:"server",tracking_mode:"server",
    inference_server_url:"http://saved-model:8000",inference_server_model:"saved-model",
    tracking_server_url:"http://saved-tracker:8092",tracking_server_capacity:37});
  view.element("inference-mode").value="cloud";view.element("tracking-mode").value="legacy";
  for(const id of ["inference-server-url","inference-server-model","tracking-server-url","tracking-server-capacity"])
    view.element(id).value="unsaved";
  await view.loadDeploymentSettings(true);
  assert.equal(view.element("inference-mode").value,"server");
  assert.equal(view.element("tracking-mode").value,"server");
  assert.equal(view.element("inference-server-url").value,"http://saved-model:8000");
  assert.equal(view.element("inference-server-model").value,"saved-model");
  assert.equal(view.element("tracking-server-url").value,"http://saved-tracker:8092");
  assert.equal(view.element("tracking-server-capacity").value,"37");
  assert.equal(view.element("inference-server-fields").hidden,false);
  assert.equal(view.element("tracking-server-fields").hidden,false)
});

test("server URLs must be origins and invalid input is never sent",async()=>{
  const invalid=["","server:8000","ftp://server:8000","http://user:pass@server:8000",
    "http://server:8000/v1/chat/completions","http://server:8000?token=x","http://server:8000#fragment","http://server:8000?",
    "http:server:8000","http://server:","http://server:0","http://server/.","http://server\\path","http://ser%76er","http://ser\tver"];
  for(const mode of ["inference","tracking"])for(const url of invalid){
    const view=dashboard();view.element(mode+"-mode").value="server";
    view.element(mode+"-server-url").value=url;
    await view.saveDeploymentSettings();
    assert.equal(view.requests.length,0,mode+": "+url);
    assert.equal(view.element(mode+"-server-url").focused,true);
    assert.match(view.element("deployment-note").className,/error/)
  }
  const valid=dashboard();valid.element("tracking-mode").value="server";
  valid.element("tracking-server-url").value="https://[2001:db8::1]:8092/";
  await valid.saveDeploymentSettings();
  assert.equal(valid.requests[0].payload.tracking_server_url,"https://[2001:db8::1]:8092")
});

test("server model and positive integer capacity are validated without a twelve-camera ceiling",async()=>{
  for(const raw of [""," ","model\nname","x".repeat(513)]){
    const view=dashboard();view.element("inference-mode").value="server";
    view.element("inference-server-url").value="http://model:8000";view.element("inference-server-model").value=raw;
    await view.saveDeploymentSettings();assert.equal(view.requests.length,0);assert.equal(view.element("inference-server-model").focused,true)
  }
  for(const raw of ["","0","-1","1.5","1e3","abc","9007199254740992"]){
    const view=dashboard();view.element("tracking-mode").value="server";
    view.element("tracking-server-url").value="http://tracker:8092";view.element("tracking-server-capacity").value=raw;
    await view.saveDeploymentSettings();assert.equal(view.requests.length,0);assert.equal(view.element("tracking-server-capacity").focused,true)
  }
  const view=dashboard();view.element("tracking-mode").value="server";
  view.element("tracking-server-url").value="http://tracker:8092";view.element("tracking-server-capacity").value="1000";
  await view.saveDeploymentSettings();assert.equal(view.requests[0].payload.tracking_server_capacity,1000)
});

test("changing tracking mode, server address, or capacity omits the old processing method",async()=>{
  for(const change of ["mode","address","capacity","none"]){
    const view=dashboard({tracking_mode:"server",tracking_server_url:"http://tracker:8092",tracking_server_capacity:16});
    vm.runInContext(`processingMethods=()=>[{id:"person_tracking",available:true}];renderProcessingSettings=()=>{};status=async()=>{};`,view.context);
    if(change==="mode")view.element("tracking-mode").value="local";
    if(change==="address")view.element("tracking-server-url").value="http://replacement:8092";
    if(change==="capacity")view.element("tracking-server-capacity").value="24";
    await view.saveDeploymentSettings();
    assert.equal(Object.hasOwn(view.requests[0].payload,"processing_method"),change==="none",change)
  }
});

test("shared settings cannot be saved while another camera is active",async()=>{
  const view=dashboard();view.context.GoodvisionFleet={extraBusy:()=>true};view.controls();
  for(const id of ["inference-mode","tracking-mode","inference-server-url","inference-server-model","tracking-server-url","tracking-server-capacity","save-deployment-button"])
    assert.equal(view.element(id).disabled,true,id);
  await view.saveDeploymentSettings();assert.equal(view.requests.length,0);
  assert.match(view.element("deployment-note").textContent,/停止全部分析/)
});

test("no GPU blocks only local selection and hidden legacy values never override authoritative selectors",async()=>{
  const view=dashboard({local_gpu:{available:false,gpu_count:0}});
  view.element("inference-mode").value="local";await view.saveDeploymentSettings();assert.equal(view.requests.length,0);
  view.element("inference-mode").value="cloud";
  view.element("local-inference-enabled").checked=true;view.element("local-tracking-enabled").checked=true;
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].payload.inference_mode,"cloud");
  assert.equal(view.requests[0].payload.tracking_mode,"legacy");
  assert.equal(view.requests[0].payload.local_inference_enabled,false);
  assert.equal(view.requests[0].payload.local_tracking_enabled,false)
});

test("initial info with server mode does not force test mode or require a cloud key",async()=>{
  const view=dashboard({inference_mode:"server",local_inference_enabled:false});view.state.deployment=null;
  await view.originalInfo();assert.equal(view.localInferenceEnabled(),true);assert.equal(view.element("dry-run").checked,false);
  view.state.deployment={inference_mode:"cloud",local_inference_enabled:false};
  assert.equal(view.localInferenceEnabled(),false)
});

test("Contour saves a separate write-only node key and blank input preserves it",async()=>{
  const view=dashboard({tracking_mode:"server",tracking_server_protocol:"contour",
    tracking_server_url:"http://192.168.20.93:8093",tracking_server_capacity:1});
  assert.equal(view.element("tracking-node-key-fields").hidden,false);
  view.element("tracking-node-key").value="node-test-key";
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].payload.tracking_api_key,"node-test-key");
  assert.equal(view.requests[0].payload.tracking_server_protocol,"contour");
  assert.equal(Object.hasOwn(view.requests[0].payload,"api_key"),false);
  assert.equal(view.element("tracking-node-key").value,"");
  assert.equal(view.state.deployment.tracking_key_present,true);
  assert.equal(JSON.stringify(view.state.deployment).includes("node-test-key"),false);
  await view.saveDeploymentSettings();
  assert.equal(Object.hasOwn(view.requests.at(-1).payload,"tracking_api_key"),false);
  assert.equal(view.state.deployment.tracking_key_present,true);
  view.element("clear-tracking-node-key").checked=true;
  await view.saveDeploymentSettings();
  assert.equal(view.requests.at(-1).payload.clear_tracking_api_key,true);
  assert.equal(view.state.deployment.tracking_key_present,false)
});

test("Contour saves multiple cameras but rejects conflicting key edits before saving",async()=>{
  const view=dashboard({tracking_mode:"server",tracking_server_protocol:"contour",
    tracking_server_url:"http://node:8093",tracking_server_capacity:1});
  view.element("tracking-server-capacity").value="2";
  await view.saveDeploymentSettings();assert.equal(view.requests.length,1);
  assert.equal(view.requests[0].payload.tracking_server_capacity,2);
  view.element("tracking-node-key").value="node-test-key";
  view.element("clear-tracking-node-key").checked=true;
  await view.saveDeploymentSettings();assert.equal(view.requests.length,1);
  assert.match(view.element("deployment-note").textContent,/不能同时/)
});

test("legacy tracking hides and never sends a staged Contour credential",async()=>{
  const view=dashboard({tracking_mode:"server",tracking_server_protocol:"legacy",
    tracking_server_url:"http://tracker:8092",tracking_server_capacity:4});
  assert.equal(view.element("tracking-node-key-fields").hidden,true);
  view.element("tracking-node-key").value="node-test-key";
  view.element("clear-tracking-node-key").checked=true;
  await view.saveDeploymentSettings();
  assert.equal(Object.hasOwn(view.requests[0].payload,"tracking_api_key"),false);
  assert.equal(Object.hasOwn(view.requests[0].payload,"clear_tracking_api_key"),false)
});
