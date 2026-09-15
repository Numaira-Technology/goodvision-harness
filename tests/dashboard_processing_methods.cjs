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
    window:{isSecureContext:true},location:{hostname:"localhost"},
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
    state.deployment={persistence_available:true,configured_stream:true,processing_method:"sam_live",
      processing_methods:[{id:"sam_live",label:"SAM 连续分析",description:"持续更新人物",available:true},
        {id:"hybrid",label:"快速跟踪 + SAM 校准",description:"快速更新，随后校准",available:true}]};
    globalThis.saved={...state.deployment};
    reset=()=>{};connect=async()=>({send:capture});info=async()=>{};
    renderDeploymentSettings(true,true);
    globalThis.app={state,start,controls,renderDeploymentSettings,renderProcessingSettings,
      saveDeploymentSettings,saveApiKey,loadDeploymentSettings,processingMethodEdited,
      updateProcessingDiagnostics};
  `,context);
  return {element,requests,messages,context,...context.app}
}

test("a processing choice must be saved before a fresh session starts and is independent of model routing",async()=>{
  const view=dashboard();
  assert.equal(view.element("processing-method").value,"sam_live");
  view.element("processing-method").value="hybrid";view.processingMethodEdited();
  assert.equal(view.element("start-button").disabled,true);
  await view.start();assert.equal(view.messages.length,0);
  assert.equal(view.state.deployment.processing_method,"sam_live");
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].url,"/v1/deployment/settings");
  assert.equal(view.requests[0].payload.processing_method,"hybrid");
  assert.equal(Object.hasOwn(view.requests[0].payload,"route_id"),false);
  assert.equal(view.state.deployment.processing_method,"hybrid");
  assert.equal(view.state.processingMethodDirty,false);
  assert.match(view.element("deployment-note").textContent,/重新开始检测后生效/);
  await view.start();assert.equal(view.messages.length,1);
  assert.equal(Object.hasOwn(view.messages[0],"processing_method"),false);
});

test("active and starting sessions lock the switch; stopping permits a new choice",()=>{
  const view=dashboard();
  for(const field of ["running","pending"]){
    view.state[field]=true;view.controls();
    assert.equal(view.element("processing-method").disabled,true);
    assert.equal(view.element("save-deployment-button").disabled,true);
    view.state[field]=false;view.controls();
    assert.equal(view.element("processing-method").disabled,false)
  }
});

test("failed save keeps both the active saved route and the unsaved choice; discard restores the saved route",async()=>{
  const view=dashboard();view.context.fail=true;
  view.element("processing-method").value="hybrid";view.processingMethodEdited();
  await view.saveDeploymentSettings();
  assert.equal(view.state.deployment.processing_method,"sam_live");
  assert.equal(view.element("processing-method").value,"hybrid");
  assert.equal(view.state.processingMethodDirty,true);
  assert.equal(view.element("start-button").disabled,true);
  view.context.fail=false;await view.loadDeploymentSettings(true);
  assert.equal(view.element("processing-method").value,"sam_live");
  assert.equal(view.state.processingMethodDirty,false);
  assert.equal(view.element("start-button").disabled,false)
});

test("credential saves and info refreshes do not discard an unsaved processing choice",async()=>{
  const view=dashboard();
  view.element("processing-method").value="hybrid";view.processingMethodEdited();
  view.element("default-api-key").value="test-key";await view.saveApiKey();
  assert.deepEqual(view.requests[0].payload,{api_key:"test-key"});
  assert.equal(view.element("processing-method").value,"hybrid");
  assert.equal(view.state.processingMethodDirty,true);
  view.renderProcessingSettings();
  assert.equal(view.element("processing-method").value,"hybrid")
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

test("comparison observations use measured values and stay scoped to method and session",()=>{
  const view=dashboard();
  view.updateProcessingDiagnostics({processing_method:"sam_live",stream_session_id:"one",pipeline_state:"running",
    processing:{method:"sam_live",source_to_result_seconds:1.234,processed_frames:20,dropped_frames:0},
    annotated_output:{stream_delay_seconds:2.5}});
  let rows=view.element("processing-comparison-body").children;
  assert.deepEqual(rows[0].children.map(cell=>cell.textContent),["SAM 连续分析","运行中","1.23 秒","2.50 秒","20","0"]);
  assert.equal(rows[1].children[1].textContent,"尚未运行");
  view.updateProcessingDiagnostics({processing_method:"hybrid",stream_session_id:"two",pipeline_state:"running",
    processing:{method:"hybrid"},annotated_output:{}});
  rows=view.element("processing-comparison-body").children;
  assert.equal(rows[0].children[2].textContent,"1.23 秒");
  assert.equal(rows[1].children[2].textContent,"—");
  view.updateProcessingDiagnostics({processing_method:"sam_live",stream_session_id:"three",pipeline_state:"starting",
    processing:{method:"sam_live"},annotated_output:{}});
  rows=view.element("processing-comparison-body").children;
  assert.equal(rows[0].children[2].textContent,"—");
  assert.equal(rows[0].children[4].textContent,"—")
});

test("source drain and plugin cleanup keep the switch locked and explain which method is stopping",()=>{
  const view=dashboard();
  for(const pipeline of ["draining","stopping","idle"]){
    view.updateProcessingDiagnostics({pipeline_state:pipeline,processing_method:"hybrid",
      processing_cleanup:"closing",processing_retiring_method:"sam_live",processing:{method:"hybrid"}});
    view.controls();
    assert.equal(view.element("processing-method").disabled,true);
    assert.equal(view.element("start-button").disabled,true);
    if(pipeline!=="idle")assert.match(view.element("processing-method-selected").textContent,/正在停止：SAM 连续分析/);
    assert.match(view.element("processing-method-note").textContent,/正在收尾/)
  }
  view.updateProcessingDiagnostics({pipeline_state:"idle",processing_method:"hybrid",
    processing_cleanup:"remote_check_pending",processing:{method:"hybrid"}});
  view.controls();
  assert.equal(view.element("processing-method").disabled,false);
  assert.match(view.element("processing-method-note").textContent,/会确认服务器已释放旧会话/)
});
