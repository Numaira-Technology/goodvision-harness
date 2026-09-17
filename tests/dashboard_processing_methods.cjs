const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");

function dashboard(){
  const elements=new Map(),requests=[],messages=[];
  function node(tag="div"){
    return {tag,children:[],dataset:{},style:{},value:"",textContent:"",className:"",checked:false,
      classList:{toggle(){}},setAttribute(key,value){this[key]=value},
      appendChild(child){this.children.push(child)},replaceChildren(){this.children=[]},
      querySelectorAll(tag){return this.children.flatMap(child=>[
        ...(child.tag===tag?[child]:[]),...child.querySelectorAll(tag),
      ])},scrollIntoView(){},focus(){},
    }
  }
  function element(id){if(!elements.has(id))elements.set(id,node());return elements.get(id)}
  const context=vm.createContext({
    document:{getElementById:element,createElement:node,
      querySelectorAll:()=>[{checked:true,dataset:{feature:"smoking"}}]},
    window:{isSecureContext:true},location:{hostname:"localhost"},AbortController,setTimeout,clearTimeout,
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
    state.info={api_key_present:true};state.gallery={ready:true};
    state.deployment={persistence_available:true,configured_stream:true,processing_method:"person_tracking",
      processing_methods:[{id:"person_tracking",label:"人员追踪模块 连续分析",description:"持续更新人物",available:true},
        {id:"hybrid",label:"快速跟踪 + 人员追踪模块 校准",description:"快速更新，随后校准",available:true}]};
    globalThis.saved={...state.deployment};
    reset=()=>{};connect=async()=>({send:capture});info=async()=>{};
    renderDeploymentSettings(true,true);
    globalThis.app={state,start,controls,renderDeploymentSettings,renderProcessingSettings,
      saveDeploymentSettings,saveApiKey,loadDeploymentSettings,savedProcessingMethod,
      updateProcessingDiagnostics};
  `,context);
  return {html,element,requests,messages,context,...context.app}
}

test("人员追踪模块 is fixed, obsolete route options stay hidden, and saved defaults select 人员追踪模块",async()=>{
  const view=dashboard();
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析");
  assert.doesNotMatch(view.html,/<select id="processing-method"/);
  assert.doesNotMatch(view.html,/快速跟踪|A \/ B 对比|两种方式/);
  assert.equal(view.element("processing-comparison-body").children.length,1);
  view.state.deployment.processing_method="hybrid";
  view.renderProcessingSettings();
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析 · 待启用");
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].url,"/v1/deployment/settings");
  assert.equal(view.requests[0].payload.processing_method,"person_tracking");
  assert.equal(Object.hasOwn(view.requests[0].payload,"route_id"),false);
  assert.equal(view.state.deployment.processing_method,"person_tracking");
  assert.match(view.element("deployment-note").textContent,/新会话使用 人员追踪模块 连续分析/);
  await view.start();assert.equal(view.messages.length,1);
  assert.equal(Object.hasOwn(view.messages[0],"processing_method"),false);
});

test("legacy and retired methods cannot start under the 人员追踪模块 label before saving",async()=>{
  for(const method of [null,"hybrid"]){
    const view=dashboard();view.state.deployment.processing_method=method;
    view.renderProcessingSettings();view.controls();
    assert.equal(view.savedProcessingMethod(),method);
    assert.equal(view.element("start-button").disabled,true);
    assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析 · 待启用");
    assert.match(view.element("processing-method-note").textContent,/请先保存默认设置，以启用 人员追踪模块 连续分析/);
    assert.doesNotMatch(view.element("processing-method-selected").textContent,/固定使用|hybrid/);
    await view.start();assert.equal(view.messages.length,0);assert.equal(view.requests.length,0);
    assert.match(view.element("status-line").textContent,/请先保存默认设置，以启用 人员追踪模块 连续分析/);
    await view.saveDeploymentSettings();
    assert.equal(view.requests[0].payload.processing_method,"person_tracking");
    assert.equal(view.savedProcessingMethod(),"person_tracking");
    assert.equal(view.element("start-button").disabled,false);
    await view.start();assert.equal(view.messages.length,1)
  }
});

test("actual runtime null overrides stale 人员追踪模块 defaults and info",async()=>{
  const view=dashboard();view.state.info.processing_method="person_tracking";
  view.updateProcessingDiagnostics({pipeline_state:"idle",processing_method:null});view.controls();
  assert.equal(view.savedProcessingMethod(),null);
  assert.equal(view.element("start-button").disabled,true);
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析 · 待启用");
  await view.start();assert.equal(view.messages.length,0);
  view.updateProcessingDiagnostics({pipeline_state:"idle",processing_method:"person_tracking"});view.controls();
  assert.equal(view.element("start-button").disabled,false);
  await view.start();assert.equal(view.messages.length,1)
});

test("unavailable 人员追踪模块 is not written while saving unrelated defaults",async()=>{
  const view=dashboard();view.state.deployment.processing_methods[0].available=false;
  await view.saveDeploymentSettings();
  assert.equal(Object.hasOwn(view.requests[0].payload,"processing_method"),false)
});

test("unsupported 人员追踪模块 with a null method preserves legacy diagnostic starts",async()=>{
  const view=dashboard();view.state.deployment.processing_method=null;
  view.state.deployment.processing_methods[0].available=false;
  view.state.deployment.processing_methods[0].reason="此部署需要启用持续人物与事件分析。";
  view.updateProcessingDiagnostics({pipeline_state:"idle",processing_method:null});
  view.renderProcessingSettings();view.controls();
  assert.equal(view.element("start-button").disabled,false);
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析 · 未启用");
  assert.doesNotMatch(view.element("processing-method-selected").textContent,/固定使用/);
  assert.match(view.element("processing-method-note").textContent,/此部署需要启用持续人物与事件分析/);
  assert.doesNotMatch(view.element("processing-method-note").textContent,/请先保存默认设置/);
  await view.start();assert.equal(view.messages.length,1);
  assert.equal(Object.hasOwn(view.messages[0],"processing_method"),false)
});

test("active and starting sessions still lock deployment changes",()=>{
  const view=dashboard();
  for(const field of ["running","pending"]){
    view.state[field]=true;view.controls();
    assert.equal(view.element("save-deployment-button").disabled,true);
    view.state[field]=false;view.controls();
    assert.equal(view.element("save-deployment-button").disabled,false)
  }
});

test("failed save keeps 人员追踪模块 and reports failure; discard reloads saved defaults",async()=>{
  const view=dashboard();view.context.fail=true;
  await view.saveDeploymentSettings();
  assert.equal(view.state.deployment.processing_method,"person_tracking");
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析");
  assert.match(view.element("deployment-note").textContent,/Could not save/);
  view.context.fail=false;await view.loadDeploymentSettings(true);
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析");
  assert.equal(view.element("start-button").disabled,false)
});

test("credential saves stay independent of the fixed 人员追踪模块 configuration",async()=>{
  const view=dashboard();
  view.element("default-api-key").value="test-key";await view.saveApiKey();
  assert.deepEqual(view.requests[0].payload,{api_key:"test-key"});
  view.renderProcessingSettings();
  assert.equal(view.element("processing-method").textContent,"人员追踪模块 连续分析")
});

test("unavailable plugins explain the reason and cannot start; legacy deployments have no processing payload",async()=>{
  const view=dashboard();
  view.state.deployment.processing_methods[0].available=false;
  view.state.deployment.processing_methods[0].reason="模型尚未加载";
  view.renderProcessingSettings();view.controls();
  assert.equal(view.element("start-button").disabled,true);
  assert.match(view.element("processing-method-note").textContent,/模型尚未加载/);
  await view.start();assert.equal(view.messages.length,0);
  delete view.state.deployment.processing_methods;delete view.state.deployment.processing_method;
  view.renderDeploymentSettings();
  assert.equal(view.element("processing-method-field").hidden,true);
  await view.saveDeploymentSettings();
  assert.equal(Object.hasOwn(view.requests[0].payload,"processing_method"),false)
});

test("人员追踪模块 observations use measured values and do not carry results across sessions",()=>{
  const view=dashboard();
  view.updateProcessingDiagnostics({processing_method:"person_tracking",stream_session_id:"one",pipeline_state:"running",
    processing:{method:"person_tracking",source_to_result_seconds:1.234,processed_frames:20,dropped_frames:0},
    annotated_output:{stream_delay_seconds:2.5}});
  let rows=view.element("processing-comparison-body").children;
  assert.deepEqual(rows[0].children.map(cell=>cell.textContent),["人员追踪模块 连续分析","运行中","1.23 秒","2.50 秒","20","0"]);
  assert.equal(rows.length,1);
  view.updateProcessingDiagnostics({processing_method:"hybrid",stream_session_id:"two",pipeline_state:"running",
    processing:{method:"hybrid"},annotated_output:{}});
  rows=view.element("processing-comparison-body").children;
  assert.equal(rows[0].children[2].textContent,"1.23 秒");
  assert.equal(rows.length,1);
  assert.equal(view.state.processingRuns.has("hybrid"),false);
  view.updateProcessingDiagnostics({processing_method:"person_tracking",stream_session_id:"three",pipeline_state:"starting",
    processing:{method:"person_tracking"},annotated_output:{}});
  rows=view.element("processing-comparison-body").children;
  assert.equal(rows[0].children[2].textContent,"—");
  assert.equal(rows[0].children[4].textContent,"—")
});

test("source drain and 人员追踪模块 cleanup keep new starts locked and explain shutdown",()=>{
  const view=dashboard();
  for(const pipeline of ["draining","stopping","idle"]){
    view.updateProcessingDiagnostics({pipeline_state:pipeline,processing_method:"person_tracking",
      processing_cleanup:"closing",processing_retiring_method:"person_tracking",processing:{method:"person_tracking"}});
    view.controls();
    assert.equal(view.element("start-button").disabled,true);
    if(pipeline!=="idle")assert.match(view.element("processing-method-selected").textContent,/正在停止：人员追踪模块 连续分析/);
    assert.match(view.element("processing-method-note").textContent,/正在收尾/)
  }
  view.updateProcessingDiagnostics({pipeline_state:"idle",processing_method:"person_tracking",
    processing_cleanup:"remote_check_pending",processing:{method:"person_tracking"}});
  view.controls();
  assert.equal(view.element("start-button").disabled,false);
  assert.match(view.element("processing-method-note").textContent,/会确认服务器已释放旧会话/)
});
