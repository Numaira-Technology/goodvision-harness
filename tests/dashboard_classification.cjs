/* Execute the dashboard's own handlers against a small DOM implementation.
 * No production logic is copied: assertions inspect the rendered panel and log.
 */
const assert=require("node:assert/strict");
const vm=require("node:vm");
const {test}=require("node:test");
const {createDocument, runDashboard} = require("./helpers/dashboard.cjs");

function dashboard(){
  const {document, element} = createDocument(["fight", "smoking", "fall", "weapon", "stranger"]);
  const context=vm.createContext({document,window:{isSecureContext:true},
    location:{hostname:"localhost"},setTimeout,clearTimeout,clearInterval,AbortController,
    requestAnimationFrame:()=>{},fetch:async()=>({ok:true,json:async()=>context.statusData})});
  runDashboard(context, `
    // These tests exercise classification; video transport has separate coverage.
    openFeed=()=>surfaces();
    state.running=true;state.sessionId="session-1";state.sessionFeatures=["fight","smoking","fall","weapon","stranger"];
    state.gallery={ready:true};state.galleryLoading=false;
    globalThis.app={analysis,message,status,reset,renderClassification,state,
      renderSensitivities,selectedSensitivities,sensitivityInputs,controls,start};
  `);
  return {app:context.app,context,element,
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

test("behavior alerts clear only after two distinct negative frames and raw evidence stays negative",()=>{
  const view=dashboard();view.app.state.sessionFeatures=["smoking"];
  view.app.analysis(result(0,{smoking:true}));
  for(let frame=1;frame<=1;frame++){
    view.app.analysis(result(frame,{smoking:false}));
    view.app.analysis(result(frame,{smoking:false}));
    view.app.analysis(result(frame+0.1,{fight:false}));
    assert.equal(view.element("verdict").textContent,"SMOKING");
    assert.match(view.status("smoking"),new RegExp("待解除 "+frame+"/2"));
  }
  view.app.analysis(result(2,{smoking:false}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.equal(JSON.parse(view.element("result-json").textContent).judgments[0].detected,false);
});

test("a new positive resets the streak independently for each event",()=>{
  const view=dashboard();view.app.state.sessionFeatures=["smoking","fight"];
  view.app.analysis(result(0,{smoking:true,fight:true}));
  view.app.analysis(result(1,{smoking:false,fight:false}));
  view.app.analysis(result(2,{smoking:true,fight:false}));
  assert.equal(view.element("verdict").textContent,"SMOKING");
  assert.equal(view.status("smoking"),"ALERT 90.0%");
  view.app.analysis(result(3,{smoking:false}));
  assert.equal(view.element("verdict").textContent,"SMOKING");
  view.app.analysis(result(4,{smoking:false}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
});

test("late positives notify without rolling back confirmed clear, current JSON or description",()=>{
  const view=dashboard();view.app.state.sessionFeatures=["smoking"];
  for(let frame=1;frame<=2;frame++)view.app.analysis({...result(frame,{smoking:false}),description:"current "+frame});
  const currentJson=view.element("result-json").textContent;
  const summary=view.element("overall-description").textContent;
  view.app.analysis({...result(0,{smoking:true}),description:"old smoking frame"});
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.equal(view.element("result-json").textContent,currentJson);
  assert.equal(view.element("overall-description").textContent,summary,"Event descriptions cannot replace the periodic scene summary");
  assert.match(view.element("decision-meta").textContent,/video 2\.000s/);
  assert.equal(view.element("event-notice").hidden,false);
  assert.match(view.element("event-notice").textContent,/迟到事件通知.*SMOKING.*0\.000s/);
  assert.equal(view.element("log-body").children[0].children[2].children[0].textContent,"SMOKING");
});

test("source-order replay and bounded history retain two-negative semantics",()=>{
  const view=dashboard();view.app.state.sessionFeatures=["smoking"];
  view.app.analysis(result(0,{smoking:true}));
  view.app.analysis(result(2,{smoking:false}));
  assert.equal(view.element("verdict").textContent,"SMOKING");
  view.app.analysis(result(1,{smoking:false}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
  for(let frame=3;frame<=300;frame++)view.app.analysis(result(frame,{smoking:false}));
  assert.equal(view.app.state.classificationHistory.get("smoking").entries.length,256);
  view.app.analysis(result(0,{smoking:true}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
});

test("per-event sensitivity is selectable before start, sent with the request, and frozen during a run",async()=>{
  const view=dashboard(),sent=[];
  view.app.state.running=false;view.app.state.info={api_key_present:true,
    event_sensitivity_defaults:{fight:"standard",smoking:"high",fall:"standard",weapon:"standard"},
    event_sensitivity_options:Object.fromEntries(Object.keys(clear).map(key=>[key,[
      {value:"standard",label:"标准",description:"Clear visual evidence"},
      {value:"high",label:"高",description:"Contextual evidence"}]]))};
  view.app.renderSensitivities();view.app.controls();
  assert.equal(view.app.sensitivityInputs.size,6);
  assert.equal(view.app.sensitivityInputs.get("smoking").value,"high");
  assert.equal(view.app.sensitivityInputs.get("smoking").disabled,false);
  view.app.sensitivityInputs.get("smoking").value="standard";
  view.app.sensitivityInputs.get("fall").value="high";
  view.app.state.ws={readyState:1,send:value=>sent.push(JSON.parse(value))};
  await view.app.start();
  assert.deepEqual(sent[0].event_sensitivities,{fight:"standard",smoking:"standard",fall:"high",weapon:"standard"});
  assert.ok([...view.app.sensitivityInputs.values()].every(input=>input.disabled));
  view.app.message({type:"session",stream_session_id:"session-2",mode:"stream",features:["smoking"],
    event_sensitivities:{smoking:"high"}});
  assert.equal(view.app.sensitivityInputs.get("smoking").value,"high");
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

test("v2 partial frame evidence stays incomplete while the capability overview uses recent branches",()=>{
  for(const stranger of [true,false]){
    const view=dashboard();
    view.app.analysis({...result(1,{...clear,stranger}),analysis_complete:true,missing_event_labels:[]});
    const description="分析未完成。部分能力暂无结果。 / Analysis incomplete; some capabilities have no result.";
    view.app.analysis({...result(1.4,clear),analysis_complete:false,missing_event_labels:["stranger"],description});
    assert.equal(view.element("verdict").textContent,stranger?"STRANGER":"CLEAR");
    assert.doesNotMatch(view.element("description").textContent,/Analysis incomplete/);
    assert.notEqual(view.element("overall-description").textContent,description);
    assert.equal(JSON.parse(view.element("result-json").textContent).judgments.length,4);
    assert.equal(view.element("log-body").children[0].children[2].children[0].textContent,"PARTIAL");
  }
});

test("event descriptions cannot replace a periodic global scene description",()=>{
  const view=dashboard();
  view.element("overall-description").textContent="房间内有两人，桌边摆着椅子。";
  for(const description of ["画面中发现陌生人。","无异常 / No anomalies detected"])
    view.app.analysis({...result(1,{...clear,stranger:true}),description});
  assert.equal(view.element("overall-description").textContent,"房间内有两人，桌边摆着椅子。");
});

test("incomplete negative results render pending and complete results restore clear",()=>{
  const view=dashboard();
  view.app.analysis({...result(1,clear),analysis_complete:false,
    missing_event_labels:["stranger"],description:"分析未完成。 / Analysis incomplete."});
  assert.equal(view.element("verdict").textContent,"INCOMPLETE");
  assert.equal(view.element("decision").className,"decision warming");
  assert.doesNotMatch(view.element("confidence").textContent,/events clear/);
  assert.doesNotMatch(view.element("description").textContent,/No target events detected/);
  assert.equal(view.status("stranger"),"WAITING");
  view.app.analysis({...result(2,{...clear,stranger:false}),analysis_complete:true,
    missing_event_labels:[],description:"无异常 / No anomalies detected"});
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.equal(view.element("decision").className,"decision clear");
});

test("incomplete results retain positive alerts",()=>{
  const view=dashboard();
  view.app.analysis({...result(1,{...clear,smoking:true}),analysis_complete:false,
    missing_event_labels:["stranger"],description:"吸烟。 / Smoking. Analysis incomplete."});
  assert.equal(view.element("verdict").textContent,"SMOKING");
  assert.equal(view.element("decision").className,"decision detected");
  assert.match(view.element("description").textContent,/Analysis incomplete/);
});

test("older frames and other sessions cannot overwrite a periodic summary",()=>{
  const view=dashboard();
  view.element("overall-description").textContent="定时全画面描述";
  view.element("overall-description-meta").textContent="Canvas 定时 Summary";
  for(const old of [result(1,{smoking:true}),result(2,{smoking:true},{frame:19}),
    result(3,{smoking:true},{session:"other"})])view.app.analysis({...old,description:"旧画面异常"});
  assert.equal(view.element("overall-description").textContent,"定时全画面描述");
  assert.equal(view.element("overall-description-meta").textContent,"Canvas 定时 Summary");
});

test("face descriptions remain separate from global descriptions",()=>{
  const view=dashboard();
  view.element("overall-description").textContent="周期性的场景概述";
  view.app.analysis({...result(1,{stranger:true}),analysis_kind:"face_recognition",
    objects:[],description:'<img src=x onerror="alert(1)">'});
  assert.equal(view.element("overall-description").textContent,"周期性的场景概述");
  assert.equal(view.element("overall-description").children.length,0);
});

test("session reset clears the old periodic description",()=>{
  const view=dashboard();
  view.element("overall-description").textContent="上个任务的场景概述";
  view.app.message({type:"session",stream_session_id:"session-2",mode:"stream",features:Object.keys(clear)});
  assert.equal(view.element("overall-description").textContent,"等待定时全画面描述");
  assert.equal(view.element("overall-description-meta").textContent,"Canvas 定时 Summary");
  view.app.analysis({...result(1,{smoking:true},{session:"session-2"}),description:"吸烟行为"});
  view.app.message({type:"ended",reason:"stopped"});
  assert.equal(view.element("overall-description").textContent,"等待定时全画面描述");
  assert.equal(view.app.state.overallResult,null);
});

test("independent partial branches make the capability overview ready without changing JSON",()=>{
  const view=dashboard();
  const safety={...result(2,clear),analysis_complete:false,missing_event_labels:["stranger"]};
  const face={...result(1.8,{stranger:false}),analysis_complete:false,
    missing_event_labels:Object.keys(clear)};
  view.app.analysis(safety);
  assert.equal(view.element("verdict").textContent,"INCOMPLETE");
  view.app.analysis(face);
  assert.equal(view.element("verdict").textContent,"CLEAR");
  assert.equal(view.element("confidence").textContent,"All 5 events clear");
  assert.doesNotMatch(view.element("description").textContent,/incomplete/i);
  assert.match(view.element("decision-meta").textContent,/video 2\.000s/);
  assert.deepEqual(JSON.parse(view.element("result-json").textContent),safety);
  assert.equal(view.element("log-body").children[0].children[2].children[0].textContent,"PARTIAL");
});

test("a newly delivered face branch alerts behind a newer safety frame immediately",()=>{
  const view=dashboard();
  const safety={...result(2,{...clear,smoking:true}),analysis_complete:false,
    missing_event_labels:["stranger"]};
  view.app.analysis(safety);
  const face={...result(1.8,{stranger:true}),analysis_complete:false,
    missing_event_labels:Object.keys(clear)};
  view.app.analysis(face);
  assert.equal(view.element("verdict").textContent,"SMOKING + STRANGER");
  assert.doesNotMatch(view.element("description").textContent,/incomplete/i);
  assert.deepEqual(JSON.parse(view.element("result-json").textContent),safety);
  view.app.analysis({...result(2.4,clear),analysis_complete:false,missing_event_labels:["stranger"]});
  assert.equal(view.element("verdict").textContent,"SMOKING + STRANGER");
  view.app.analysis({...result(2.2,{...clear,smoking:true}),analysis_complete:false,
    missing_event_labels:["stranger"]});
  assert.equal(view.element("verdict").textContent,"SMOKING + STRANGER");
  assert.equal(view.status("smoking"),"ALERT 90.0% · 待解除 1/2");
  assert.equal(view.status("stranger"),"ALERT 90.0%");
});

test("independent branch overview becomes pending again after face expiry or session reset",()=>{
  const view=dashboard();
  view.app.analysis({...result(1,{stranger:false}),analysis_complete:false,
    missing_event_labels:Object.keys(clear)});
  view.app.analysis({...result(2,clear),analysis_complete:false,missing_event_labels:["stranger"]});
  assert.equal(view.element("verdict").textContent,"CLEAR");
  view.app.analysis({...result(4.1,clear),analysis_complete:false,missing_event_labels:["stranger"]});
  assert.equal(view.element("verdict").textContent,"INCOMPLETE");
  assert.equal(view.status("stranger"),"WAITING");
  view.app.message({type:"session",stream_session_id:"session-2",mode:"stream",
    features:[...Object.keys(clear),"stranger"]});
  view.app.analysis({...result(0,clear,{session:"session-2"}),analysis_complete:false,
    missing_event_labels:["stranger"]});
  assert.equal(view.element("verdict").textContent,"INCOMPLETE");
  view.app.analysis(result(4.2,{stranger:true}));
  assert.equal(view.status("stranger"),"WAITING");
});


test("new safety events render independently and clear after repeated negatives",()=>{
  const view=dashboard();
  view.app.state.sessionFeatures=["ppe_violation","ground_litter"];
  view.app.analysis(result(1,{ppe_violation:true,ground_litter:true}));
  assert.equal(view.element("verdict").textContent,"PPE_VIOLATION + GROUND_LITTER");
  view.app.analysis(result(2,{ppe_violation:false,ground_litter:false}));
  assert.equal(view.element("decision").className,"decision detected");
  view.app.analysis(result(3,{ppe_violation:false,ground_litter:false}));
  assert.equal(view.element("verdict").textContent,"CLEAR");
});
