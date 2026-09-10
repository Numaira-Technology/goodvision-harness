const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");

function dashboard(){
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
    capture:raw=>messages.push(JSON.parse(raw)),
    fetch:async(url,options)=>{
      const payload=JSON.parse(options.body);requests.push({url,payload});
      return {ok:true,json:async()=>({persistence_available:true,...payload})}
    },
  });
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script.slice(0,script.indexOf("ui.defaultStreamUrl.oninput="))+`
    state.info={api_key_present:true};state.gallery={ready:true};
    state.deployment={persistence_available:true,camera_layout:"mono"};
    reset=()=>{};connect=async()=>({send:capture});info=async()=>{};
    renderDeploymentSettings(true,true);
    globalThis.app={state,start,controls,renderDeploymentSettings,saveDeploymentSettings,saveApiKey};
  `,context);
  return {element,requests,messages,...context.app}
}

test("network starts send the chosen camera type; browser camera starts omit it",async()=>{
  const view=dashboard();
  for(const layout of ["mono","stereo"]){
    view.element("camera-layout").value=layout;
    await view.start();
    assert.equal(view.messages.at(-1).camera_layout,layout);
    assert.equal(view.element("camera-layout").disabled,true);
    view.state.pending=false
  }
  view.state.mode="camera";
  await view.start();
  assert.equal(Object.hasOwn(view.messages.at(-1),"camera_layout"),false)
});

test("saving the default restores it while a non-resetting refresh preserves a one-run choice",async()=>{
  const view=dashboard();
  assert.equal(view.element("camera-layout").value,"mono");
  view.element("default-camera-layout").value="stereo";
  await view.saveDeploymentSettings();
  assert.equal(view.requests[0].payload.camera_layout,"stereo");
  assert.equal(view.element("camera-layout").value,"stereo");
  view.element("camera-layout").value="mono";view.state.runOverridesDirty=true;
  view.renderDeploymentSettings(true,false);
  assert.equal(view.element("camera-layout").value,"mono");
  assert.equal(view.element("default-camera-layout").value,"stereo");
  view.renderDeploymentSettings(true,true);
  assert.equal(view.element("camera-layout").value,"stereo")
});

test("camera type is locked for an active session and unlocked after stopping",()=>{
  const view=dashboard();
  for(const busy of [true,false]){
    view.state.running=busy;view.controls();
    assert.equal(view.element("camera-layout").disabled,busy);
    assert.equal(view.element("save-deployment-button").disabled,busy)
  }
});

test("missing live credentials prompt for the console key without starting a session",async()=>{
  const view=dashboard();view.state.info.api_key_present=false;
  await view.start();
  assert.equal(view.messages.length,0);
  assert.equal(view.state.pending,false);
  assert.equal(view.element("default-api-key").focused,true);
  assert.match(view.element("status-line").textContent,/API key/);
  view.element("dry-run").checked=true;
  await view.start();
  assert.equal(view.messages.length,1)
});

test("the dedicated key button saves only the credential and clears the secret input",async()=>{
  const view=dashboard();
  view.element("default-api-key").value="test-only-key";
  view.element("default-stream-url").value="rtsp://unsaved/live";
  view.controls();assert.equal(view.element("save-api-key-button").disabled,false);
  await view.saveApiKey();
  assert.deepEqual(view.requests[0].payload,{api_key:"test-only-key"});
  assert.equal(view.element("default-api-key").value,"");
  assert.match(view.element("deployment-note").textContent,/无需 .env/)
});
