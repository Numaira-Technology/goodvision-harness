/* Execute the dashboard's own handlers against a small DOM implementation.
 * No production logic is copied: assertions inspect the rendered panel and log.
 */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");

class Element{
  constructor(){
    this.textContent="";this.className="";this.children=[];this.dataset={};this.style={};
    this.value="";this.checked=false;
    this.classList={
      add:name=>{if(!this.className.split(" ").includes(name))this.className+=" "+name},
      remove:name=>{this.className=this.className.split(" ").filter(item=>item!==name).join(" ")},
      toggle:(name,active)=>active?this.classList.add(name):this.classList.remove(name)
    }
  }
  append(...items){for(const item of items){item.parent=this;this.children.push(item)}}
  appendChild(item){this.append(item)}
  prepend(item){item.parent=this;this.children.unshift(item)}
  replaceChildren(...items){this.children=[];this.append(...items)}
  remove(){this.parent.children=this.parent.children.filter(item=>item!==this)}
  get lastChild(){return this.children.at(-1)}
  querySelectorAll(){return []}
  getAttribute(name){return this[name]||null}
  setAttribute(name,value){this[name]=value}
  removeAttribute(name){delete this[name]}
}

function dashboard(){
  const elements=new Map(),features=["fight","smoking","fall","weapon","stranger"];
  const featureInputs=features.map(feature=>Object.assign(new Element(),{checked:true,dataset:{feature}}));
  const document={
    getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},
    querySelectorAll:()=>featureInputs,createElement:()=>new Element()
  };
  const context=vm.createContext({document,window:{isSecureContext:true},
    location:{hostname:"localhost"},setTimeout,clearTimeout,clearInterval,
    requestAnimationFrame:()=>{},fetch:async()=>({json:async()=>context.statusData})});
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  new vm.Script(script); // Parse the complete shipped script, including browser startup.
  vm.runInContext(script.slice(0,script.indexOf("ui.defaultStreamUrl.oninput="))+`
    state.running=true;state.sessionId="session-1";state.sessionFeatures=detectionEvents;
    state.gallery={ready:true};state.galleryLoading=false;
    globalThis.app={analysis,message,status,reset,renderClassification,renderFaceAnnotation,summarizeFaces,state};
  `,context);
  return {app:context.app,context,element:id=>document.getElementById(id),
    status:event=>document.getElementById("event-statuses").children
      .find(item=>item.dataset.eventLabel===event)?.children[1].textContent};
}

function result(timestamp,values,{session="session-1",frame=Math.round(timestamp*10)}={}){
  return {type:"analysis_result",stream_session_id:session,analysis_kind:"multi_event",
    message_id:`${session}-${timestamp}-${frame}`,sequence:frame,
    frame:{timestamp_seconds:timestamp,source_frame_index:frame},model:{latency_seconds:0.1},
    judgments:Object.entries(values).map(([event_label,detected])=>({
      event_label,detected,confidence:0.9,description:`${event_label} observation`,
      object_ids:detected?["person-1"]:[]
    }))};
}
const clear={fight:false,smoking:false,fall:false,weapon:false};

test("overall description displays the frame's text rather than held event alerts",()=>{
  const view=dashboard();
  const description="画面中发现陌生人。 / An unknown person is visible in the frame.";
  view.app.analysis({...result(1,{...clear,stranger:true}),description});
  assert.equal(view.element("overall-description").textContent,description);
  view.app.analysis({...result(1.4,clear),description:"无异常 / No anomalies detected"});
  assert.equal(view.element("verdict").textContent,"STRANGER");
  assert.equal(view.element("overall-description").textContent,"无异常 / No anomalies detected");
  assert.equal(view.element("overall-description-meta").textContent,"Frame 14 · 1.400s");
});

test("overall description ignores older frames and results from other sessions",()=>{
  const view=dashboard();
  view.app.analysis({...result(2,{...clear,stranger:false}),description:"无异常 / No anomalies detected"});
  for(const old of [result(1,{smoking:true}),result(2,{smoking:true},{frame:19}),
    result(3,{smoking:true},{session:"other"})]){
    view.app.analysis({...old,description:"旧画面异常"});
    assert.equal(view.element("overall-description").textContent,"无异常 / No anomalies detected");
    assert.equal(view.element("overall-description-meta").textContent,"Frame 20 · 2.000s");
  }
});

test("overall description renders face results and treats model text as plain text",()=>{
  const view=dashboard(),description='<img src=x onerror="alert(1)">';
  view.app.analysis({...result(1,{stranger:true}),analysis_kind:"face_recognition",
    objects:[],description});
  assert.equal(view.element("overall-description").textContent,description);
  assert.equal(view.element("overall-description").children.length,0);
  assert.equal(view.element("overall-description-meta").textContent,"Frame 10 · 1.000s");
});

test("known face display includes the enrolled ID even when a name is present and preserves JSON",()=>{
  const view=dashboard(),personId="employee-007",name="Alex";
  const face={object_id:"face-1",label:name,confidence:0.98,
    bbox:{x1:0.1,y1:0.2,x2:0.3,y2:0.5},
    attributes:{status:"known",person_id:personId,name,similarity:0.91}};
  const observation={...result(1,{stranger:false}),analysis_kind:"face_recognition",objects:[face]};
  view.app.analysis(observation);
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.equal(view.element("decision").className,"decision clear");
  assert.equal(view.element("description").textContent,"1 known (ID: employee-007 | Alex)");
  assert.deepEqual(JSON.parse(view.element("result-json").textContent).objects,[face]);
  view.app.renderFaceAnnotation(observation,view.app.summarizeFaces(observation));
  const box=view.element("bbox-layer").children.find(item=>item.className==="bbox-box face-known");
  assert.ok(box,"a recognized face uses the green face-known style");
  assert.equal(box.children[0].textContent,"ID: employee-007 | Alex | 0.910");
  view.app.reset();
});

test("unknown faces stay red and never display an enrolled candidate ID",()=>{
  const view=dashboard();
  const observation={...result(1,{stranger:true}),analysis_kind:"face_recognition",objects:[{
    object_id:"face-1",label:"Alex",bbox:{x1:0.1,y1:0.2,x2:0.3,y2:0.5},
    attributes:{status:"unknown",candidate_person_id:"employee-007",similarity:0.2}
  }]};
  view.app.analysis(observation);
  assert.equal(view.element("verdict").textContent,"UNKNOWN");
  assert.equal(view.element("description").textContent,"1 unknown");
  view.app.renderFaceAnnotation(observation,view.app.summarizeFaces(observation));
  const box=view.element("bbox-layer").children.find(item=>item.className==="bbox-box face-unknown");
  assert.ok(box,"an unknown face keeps the red face-unknown style");
  assert.equal(box.children[0].textContent,"UNKNOWN | 0.200");
  assert.doesNotMatch(box.children[0].textContent,/employee-007|Alex/);
  view.app.reset();
});

test("known face IDs remain plain text and identities with the same name stay distinct",()=>{
  const view=dashboard(),personId='<img src=x onerror="alert(1)">';
  const observation={...result(1,{stranger:false}),analysis_kind:"face_recognition",objects:[
    {object_id:"face-1",attributes:{status:"known",person_id:personId,name:"Alex"}},
    {object_id:"face-2",attributes:{status:"known",person_id:"employee-008",name:"Alex"}}
  ]};
  view.app.analysis(observation);
  assert.equal(view.element("description").textContent,
    '2 known (ID: <img src=x onerror="alert(1)"> | Alex, ID: employee-008 | Alex)');
  assert.equal(view.element("description").children.length,0);
});

test("multi-event known identities persist across skipped face samples and expire with face evidence",()=>{
  const view=dashboard(),face={object_id:"face-1",label:"known_person",
    bbox:{x1:0.1,y1:0.2,x2:0.3,y2:0.5},
    attributes:{status:"known",person_id:"employee-007",name:"Alex"}};
  const observation={...result(1,{...clear,stranger:false}),objects:[face]};
  view.app.analysis(observation);
  for(const id of ["description","face-diagnostic-summary"])
    assert.match(view.element(id).textContent,/Known faces: ID: employee-007 \| Alex/);
  assert.deepEqual(JSON.parse(view.element("result-json").textContent).objects,[face]);
  assert.match(view.element("decision-meta").textContent,/\| 1 bbox \|/);
  view.app.analysis(result(1.4,clear));
  for(const id of ["description","face-diagnostic-summary"])
    assert.match(view.element(id).textContent,/ID: employee-007/);
  assert.equal(JSON.parse(view.element("result-json").textContent).objects,undefined);
  assert.match(view.element("decision-meta").textContent,/\| 1 bbox \|/);
  view.app.analysis(result(4.1,clear));
  for(const id of ["description","face-diagnostic-summary"])
    assert.doesNotMatch(view.element(id).textContent,/employee-007|Alex/);
  assert.equal(view.status("stranger"),"WAITING");
  assert.match(view.element("decision-meta").textContent,/\| 0 bboxes \|/);
});

test("multi-event box count includes known faces once and excludes unusable or missing boxes",()=>{
  const view=dashboard(),bbox={x1:0.1,y1:0.2,x2:0.3,y2:0.5};
  const observation={...result(1,{...clear,smoking:true,stranger:true}),objects:[
    {object_id:"face-known",bbox,attributes:{status:"known",person_id:"employee-007"}},
    {object_id:"face-unknown",bbox,attributes:{status:"unknown"}},
    {object_id:"face-unusable",bbox,attributes:{status:"unusable"}},
    {object_id:"face-no-box",attributes:{status:"known",person_id:"employee-008"}}
  ]};
  observation.judgments.find(item=>item.event_label==="smoking").object_ids=["face-known"];
  observation.judgments.find(item=>item.event_label==="stranger").object_ids=["face-unknown"];
  view.app.analysis(observation);
  assert.match(view.element("decision-meta").textContent,/\| 2 bboxes \|/);
});

test("a fresh unknown face observation clears the previous known identity",()=>{
  const view=dashboard();
  view.app.analysis({...result(1,{...clear,stranger:false}),objects:[{
    object_id:"face-1",attributes:{status:"known",person_id:"employee-007",name:"Alex"}
  }]});
  view.app.analysis({...result(1.4,{...clear,stranger:true}),objects:[{
    object_id:"face-2",attributes:{status:"unknown",candidate_person_id:"employee-007"}
  }]});
  assert.equal(view.element("verdict").textContent,"STRANGER");
  for(const id of ["description","face-diagnostic-summary"])
    assert.doesNotMatch(view.element(id).textContent,/employee-007|Alex|Known faces/);
});

test("missing overall description is unavailable and session reset clears the old frame",()=>{
  const view=dashboard();
  view.app.analysis({...result(1,{smoking:true}),description:"吸烟行为"});
  view.app.analysis(result(2,clear));
  assert.equal(view.element("overall-description").textContent,"本帧未提供整体描述 / No overall description for this frame");
  view.app.message({type:"session",stream_session_id:"session-2",mode:"stream",features:Object.keys(clear)});
  assert.equal(view.element("overall-description").textContent,"等待当前帧分析… / Waiting for frame analysis…");
  assert.equal(view.element("overall-description-meta").textContent,"等待分析帧");
  view.app.analysis({...result(1,{smoking:true},{session:"session-2"}),description:"吸烟行为"});
  view.app.message({type:"ended",reason:"stopped"});
  assert.equal(view.element("overall-description").textContent,"等待当前帧分析… / Waiting for frame analysis…");
  assert.equal(view.app.state.overallResult,null);
});

test("skipped face samples preserve panel alerts and face details without rewriting evidence",()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  const rows=[...view.element("event-statuses").children];
  for(const time of [1.4,1.8,2.2]){
    view.app.analysis(result(time,clear));
    assert.equal(view.element("verdict").textContent,"STRANGER");
    assert.equal(view.status("stranger"),"ALERT 90.0%");
    assert.match(view.element("face-diagnostic-summary").textContent,/STRANGER ALERT/);
    assert.equal(view.element("event-chip").textContent,"5 selected");
    rows.forEach((row,index)=>assert.equal(view.element("event-statuses").children[index],row));
    assert.equal(JSON.parse(view.element("result-json").textContent).judgments.length,4);
    const logRow=view.element("log-body").children[0];
    assert.equal(logRow.children[2].children[0].textContent,"PARTIAL");
    assert.equal(logRow.children[2].children[0].className,"pill neutral");
    assert.match(logRow.children[6].textContent,/Not analyzed in this frame: stranger/);
  }
});

test("explicit negative observations clear held alerts on the next valid keyframe",()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  view.app.analysis(result(1.4,clear));
  view.app.analysis(result(1.8,{...clear,stranger:false}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.equal(view.status("stranger"),"CLEAR 90.0%");
  assert.doesNotMatch(view.element("face-diagnostic-summary").className,/error/);
  assert.equal(view.element("log-body").children[0].children[2].children[0].textContent,"CLEAR");
});

test("a face-only clear observation does not claim the unobserved safety features are clear",()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,smoking:true,stranger:true}));
  view.app.analysis(result(1.4,{stranger:false}));
  assert.equal(view.element("verdict").textContent,"SMOKING");
  const logRow=view.element("log-body").children[0];
  assert.equal(logRow.children[2].children[0].textContent,"PARTIAL");
  assert.match(logRow.children[6].textContent,/Clear in this frame: stranger/);
  assert.match(logRow.children[6].textContent,/Not analyzed in this frame: fight, smoking, fall, weapon/);
});

test("positive partial results keep the event alert and disclose unobserved features",()=>{
  const view=dashboard();
  view.app.analysis(result(1,{smoking:true}));
  const logRow=view.element("log-body").children[0];
  assert.equal(logRow.children[2].children[0].textContent,"SMOKING");
  assert.equal(logRow.children[2].children[0].className,"pill yes");
  assert.match(logRow.children[6].textContent,/Not analyzed in this frame: fight, fall, weapon, stranger/);
  assert.equal(JSON.parse(view.element("result-json").textContent).judgments.length,1);
});

test("clear logs are scoped to the session's selected features and canonical event names",()=>{
  const view=dashboard();
  view.app.state.sessionFeatures=["smoking"];
  view.app.analysis(result(1,{smoke:false}));
  const logRow=view.element("log-body").children[0];
  assert.equal(logRow.children[2].children[0].textContent,"CLEAR");
  assert.doesNotMatch(logRow.children[6].textContent,/Not analyzed/);
});

test("late results cannot roll back a newer observation for the same event",()=>{
  const view=dashboard();
  view.app.analysis(result(2,{...clear,stranger:false}));
  view.app.analysis(result(1,{...clear,stranger:true}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.match(view.element("decision-meta").textContent,/video 2\.000s/);
  view.app.analysis(result(2,{stranger:true},{frame:19}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
});

test("late branch results can fill an omitted feature without rolling back newer features",()=>{
  const view=dashboard();
  view.app.analysis(result(2,{...clear,smoking:true}));
  view.app.analysis(result(1.8,{...clear,stranger:true}));
  assert.equal(view.element("verdict").textContent,"SMOKING + STRANGER");
  assert.match(view.element("decision-meta").textContent,/video 2\.000s/);
});

test("face carry expires only across newer keyframes and never reports unobserved features clear",()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  view.app.analysis(result(4,clear));
  assert.equal(view.element("verdict").textContent,"STRANGER");
  view.app.analysis(result(4.1,clear));
  assert.equal(view.element("verdict").textContent,"WAITING");
  assert.equal(view.status("stranger"),"WAITING");
  assert.equal(view.element("confidence").textContent,"4 events clear | 1 pending");
  view.app.analysis(result(4.2,{stranger:true}));
  view.app.renderClassification();
  assert.equal(view.element("verdict").textContent,"STRANGER");
});

test("empty or malformed branches cannot erase a valid observation",()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  view.app.analysis(result(1.4,{}));
  view.app.analysis(result(1.8,{stranger:null}));
  assert.equal(view.element("verdict").textContent,"STRANGER");
});

test("session changes, stop, and reset discard held classification",async()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  view.app.message({type:"session",stream_session_id:"session-2",mode:"stream",features:Object.keys(clear)});
  assert.equal(view.element("verdict").textContent,"WARMING");
  view.app.analysis(result(2,{...clear,stranger:true}));
  assert.equal(view.element("verdict").textContent,"WARMING");
  view.app.analysis(result(1,{smoking:true},{session:"session-2"}));
  assert.equal(view.element("verdict").textContent,"SMOKING");
  view.app.message({type:"ended",reason:"stopped"});
  assert.equal(view.element("verdict").textContent,"READY");
  assert.equal(view.app.state.classificationJudgments.size,0);
  view.app.reset();
  assert.equal(view.element("verdict").textContent,"WARMING");
});

test("polling a new server session resets the panel before fresh results",async()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  view.context.statusData={pipeline_state:"running",stream_session_id:"session-2",
    features:[...Object.keys(clear),"stranger"],input_mode:"stream",annotated_output:{}};
  await view.app.status();
  assert.equal(view.app.state.sessionId,"session-2");
  assert.equal(view.element("verdict").textContent,"WARMING");
  assert.equal(view.app.state.classificationJudgments.size,0);
});

test("runtime branch errors explain failures and clear all error UI after recovery",async()=>{
  const view=dashboard();
  const running={pipeline_state:"running",stream_session_id:"session-1",input_mode:"stream",
    features:[...Object.keys(clear),"stranger"],annotated_output:{}};
  view.context.statusData={...running,last_error:"analysis request failed",
    pipeline:{branch_errors:{safety:"timeout",stranger:"connection"}}};
  await view.app.status();
  const detail="Event analysis timed out | Face analysis connection failed";
  assert.equal(view.element("runtime-error").hidden,false);
  assert.equal(view.element("runtime-error").textContent,"Runtime error · "+detail);
  assert.equal(view.element("connection-detail").textContent,detail);
  assert.equal(view.element("status-line").textContent,detail);
  assert.equal(view.element("status-line").className,"status error");
  view.context.statusData={...running,ready:true,last_error:null,pipeline:{branch_errors:{}}};
  await view.app.status();
  assert.equal(view.element("runtime-error").hidden,true);
  assert.equal(view.element("runtime-error").textContent,"");
  assert.equal(view.element("connection-state").textContent,"Ready and delivering");
  assert.equal(view.element("status-line").className,"status");
  assert.doesNotMatch(view.element("status-line").textContent,/timed out|failed/);
});

test("every public branch failure code has a readable message and unknown values stay generic",async()=>{
  const view=dashboard();
  const reasons={request_failed:"request failed",timeout:"timed out",connection:"connection failed",
    http_error:"service returned an error",invalid_response:"returned an invalid result",empty_response:"returned no result"};
  for(const [code,message] of Object.entries(reasons)){
    view.context.statusData={pipeline_state:"running",stream_session_id:"session-1",
      last_error:"analysis request failed",pipeline:{branch_errors:{analysis:code}}};
    await view.app.status();
    assert.equal(view.element("runtime-error").textContent,"Runtime error · Analysis "+message);
  }
  view.context.statusData={pipeline_state:"running",stream_session_id:"session-1",
    last_error:"analysis request failed",pipeline:{branch_errors:{private_provider_detail:"secret_unknown_code",safety:"constructor"}}};
  await view.app.status();
  assert.equal(view.element("runtime-error").textContent,"Runtime error · Analysis request failed | Event analysis request failed");
  for(const id of ["runtime-error","connection-detail","status-line"])
    assert.doesNotMatch(view.element(id).textContent,/private_provider_detail|secret_unknown_code|constructor/);
});

test("legacy generic errors still display and stopping after recovery clears the status message",async()=>{
  const view=dashboard();
  view.context.statusData={pipeline_state:"running",stream_session_id:"session-1",last_error:"analysis request failed"};
  await view.app.status();
  assert.equal(view.element("runtime-error").textContent,"Runtime error · analysis request failed");
  view.context.statusData={pipeline_state:"stopped",last_error:"",pipeline:{branch_errors:{safety:"",stranger:null}}};
  await view.app.status();
  assert.equal(view.element("runtime-error").hidden,true);
  assert.equal(view.element("status-line").className,"status");
  assert.equal(view.element("status-line").textContent,"No active session. Choose a source to start detection.");
});

test("GPU-required rich disabling is explained without interrupting normal annotation or classification",async()=>{
  const view=dashboard();
  view.app.analysis(result(1,{...clear,stranger:true}));
  const running={pipeline_state:"running",stream_session_id:"session-1",input_mode:"stream",ready:true,
    features:[...Object.keys(clear),"stranger"],annotated_output:{state:"running",annotation_mode:"held"},last_error:null};
  view.context.statusData={...running,rich_labeling:{state:"disabled",disabled_reason:"gpu_required"}};
  await view.app.status();
  assert.equal(view.element("rich-annotation-status").hidden,false);
  assert.equal(view.element("rich-annotation-status").textContent,"Rich annotation disabled: GPU required");
  assert.equal(view.element("annotated-state").textContent,"HELD");
  assert.equal(view.element("output-channel").textContent,"Annotated output · held");
  assert.equal(view.element("connection-state").textContent,"Ready and delivering");
  assert.equal(view.element("runtime-error").hidden,true);
  assert.equal(view.element("verdict").textContent,"STRANGER");
  view.context.statusData={...running,rich_labeling:{state:"running"}};
  await view.app.status();
  assert.equal(view.element("rich-annotation-status").hidden,true);
  assert.equal(view.element("rich-annotation-status").textContent,"");
});

test("rich status does not invent GPU restrictions or display unknown internal reasons",async()=>{
  const view=dashboard();
  for(const rich of [undefined,{state:"disabled"},{state:"disabled",disabled_reason:"internal_worker_detail"},
    {state:"buffering"},{state:"running"}]){
    view.context.statusData={pipeline_state:"running",stream_session_id:"session-1",rich_labeling:rich};
    await view.app.status();
    assert.equal(view.element("rich-annotation-status").hidden,true);
    assert.equal(view.element("rich-annotation-status").textContent,"");
  }
});
