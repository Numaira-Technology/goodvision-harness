/* Exercise the shipped global-description binding against Canvas summary data. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");
const filename=path.join(__dirname,"../src/goodvision_harness/dashboard.html");
const html=fs.readFileSync(filename,"utf8");
const start=html.indexOf("function resetOverallDescription(){");
const end=html.indexOf("\nfunction compareClassificationFrames",start);
assert.ok(start>=0&&end>start);
const helpers=new vm.Script(html.slice(start,end)+"\nglobalThis.poll=pollOverallDescription;globalThis.render=renderOverallDescription;",{filename});
function summary(overrides={}){
  return {type:"workflow_summary",node_id:"scene-summary",source_id:"camera-source",stream_session_id:"summary-session",generation:8,
    source_frame_sequence:12,timestamp_seconds:4.8,created_at:"2026-10-18T12:00:00Z",status:"completed",test:false,
    summary:"A person is standing beside a table in a bright room.",summary_translations:{en:overrides.summary||"A person is standing beside a table in a bright room.",zh:"一人站在明亮房间的桌子旁。"},...overrides};
}
function fixture({language="en",items=[summary()],components=[{node_id:"scene-summary",state:"waiting",next_in_seconds:30}],response}={}){
  const ui={overallDescription:{dataset:{},textContent:""},overallDescriptionMeta:{textContent:""}};
  // Summary-only workflows intentionally retain a different native session id.
  const manager={state:"running",running:true,source_id:"camera-source",stream_session_id:"summary-session",generation:8,components};
  const state={running:true,sessionId:"old-native-session",diagnosticStatus:{pipeline_state:"running",stream_session_id:"old-native-session",
    workflow:{enabled:true,stream_session_id:"summary-session",generation:8,summaries:manager}}};
  const requests=[],timers=new Map(),listeners={};let timer=0;
  const locale={value:language};
  const context=vm.createContext({state,ui,sceneView:{closed:false},AbortController,
    GoodvisionI18n:{language:()=>locale.value},GoodvisionDiagnosticView:"insights",window:{addEventListener:(name,handler)=>{listeners[name]=handler;}},
    setTimeout(callback,delay){const id=++timer;timers.set(id,{callback,delay});return id;},clearTimeout(id){timers.delete(id);},
    async fetch(url,options){requests.push({url,options});const payload=response?await response(state):{status:manager,items};
      if(payload?.httpStatus)return {ok:false,status:payload.httpStatus};
      return {ok:true,status:200,json:async()=>payload};}});
  helpers.runInContext(context);
  const begin=html.indexOf('window.addEventListener("goodvision:language"');
  const end=html.indexOf('window.addEventListener("beforeunload"',begin);
  new vm.Script(html.slice(begin,end)).runInContext(context);
  return {context,state,ui,requests,timers,manager,poll:()=>context.poll(),setLanguage:value=>{locale.value=value;listeners["goodvision:language"]();}};
}
test("global scene description reads Canvas summaries using the workflow session, independently of native events",async()=>{
  const view=fixture();await view.poll();
  assert.deepEqual(view.requests.map(item=>item.url),["/v1/workflow/summaries?limit=100"]);
  assert.equal(view.ui.overallDescription.textContent,summary().summary);
  assert.equal(view.ui.overallDescription.dataset.interfaceCopy,"false");
  assert.match(view.ui.overallDescriptionMeta.textContent,/Canvas 定时 Summary.*视频 4.8 秒.*30 秒后/);
  assert.equal(view.timers.size,0,"Timeout is released after the finite request");
});
test("event explanations and face result descriptions cannot become the full-scene summary",async()=>{
  const view=fixture({items:[{...summary(),type:"analysis_result",description:"No smoking detected."},{...summary(),type:"overall_description",description:"Known person."}]});
  await view.poll();assert.match(view.ui.overallDescription.textContent,/等待首次定时/);
  assert.doesNotMatch(view.ui.overallDescription.textContent,/No smoking|Known person/);
});
test("old sessions, generations, sources, unselected nodes and one-shot tests are ignored",async()=>{
  const view=fixture({items:[summary({stream_session_id:"old"}),summary({generation:7}),summary({source_id:"other-camera"}),
    summary({node_id:"removed-node"}),summary({test:true})]});
  await view.poll();assert.match(view.ui.overallDescription.textContent,/等待首次定时/);
});
test("multiple periodic summary nodes keep their newest successful observations separately",async()=>{
  const view=fixture({components:[{node_id:"scene-summary",state:"waiting"},{node_id:"door-summary",state:"running"}],items:[
    summary({summary:"First scene",source_frame_sequence:10}),summary({summary:"Latest scene",source_frame_sequence:20}),
    summary({node_id:"door-summary",summary:"The door is open."}),summary({summary:"Late stale scene",source_frame_sequence:5})]});
  await view.poll();assert.match(view.ui.overallDescription.textContent,/scene-summary\nLatest scene/);
  assert.match(view.ui.overallDescription.textContent,/door-summary\nThe door is open/);
  assert.doesNotMatch(view.ui.overallDescription.textContent,/First scene|Late stale/);
  assert.match(view.ui.overallDescriptionMeta.textContent,/正在生成/);
});
test("a failed refresh retains the last successful summary and identifies the retry",async()=>{
  let fail=false;const view=fixture({response:()=>fail?{httpStatus:503}:{status:view.manager,items:[summary()]}});
  await view.poll();fail=true;await view.poll();
  assert.equal(view.ui.overallDescription.textContent,summary().summary);
  assert.match(view.ui.overallDescriptionMeta.textContent,/更新延迟.*重试/);
});
test("a failed scheduled inference does not replace the last successful global description",async()=>{
  const view=fixture({components:[{node_id:"scene-summary",state:"error",last_error:"timeout"}],items:[summary(),
    summary({source_frame_sequence:13,status:"error",summary:null,error:{code:"timeout"}})]});
  await view.poll();assert.equal(view.ui.overallDescription.textContent,summary().summary);
  assert.match(view.ui.overallDescriptionMeta.textContent,/描述更新失败/);
});
test("events-only tasks show that periodic description is disabled without reading the detection fallback",async()=>{
  const view=fixture();view.state.diagnosticStatus.workflow.summaries=null;await view.poll();
  assert.equal(view.requests.length,0);assert.match(view.ui.overallDescription.textContent,/未启用定时全画面描述/);
});
test("stop and one-shot workflow tests clear successful descriptions and do not poll the retained history",async()=>{
  const view=fixture();await view.poll();view.state.running=false;await view.poll();
  assert.equal(view.requests.length,1);assert.equal(view.ui.overallDescription.textContent,"任务已停止");
  view.state.running=true;view.state.diagnosticStatus.workflow_test_running=true;await view.poll();
  assert.equal(view.requests.length,1);assert.match(view.ui.overallDescription.textContent,/工作流测试中/);
});
test("a new workflow cannot inherit successful descriptions from the previous workflow",async()=>{
  const view=fixture();await view.poll();view.state.diagnosticStatus.workflow.stream_session_id="new-session";
  view.state.diagnosticStatus.workflow.generation=9;await view.poll();
  assert.match(view.ui.overallDescription.textContent,/等待首次/);assert.doesNotMatch(view.ui.overallDescription.textContent,/bright room/);
});
test("a summary request finishing after task replacement cannot paint the previous session",async()=>{
  const view=fixture({response:state=>{state.diagnosticStatus.workflow.stream_session_id="new-session";return {status:view.manager,items:[summary()]};}});
  await view.poll();assert.doesNotMatch(view.ui.overallDescription.textContent,/bright room/);
});
test("frame analysis preserves latest-result ordering without updating the Canvas summary",()=>{
  const body=html.slice(html.indexOf("function compareClassificationFrames"),html.indexOf("\nfunction resetClassification"));
  const ui={resultJson:{dataset:{},textContent:""},overallDescription:{textContent:"Periodic global scene description"}};
  const state={running:true,sessionId:"s",overallResult:null};const shown=[],logged=[];
  const context=vm.createContext({state,ui,consoleJson:JSON.stringify,summarizeFaces:()=>({}),analysisFace:item=>shown.push(item),
    logFace:item=>logged.push(item)});new vm.Script(body+"\nglobalThis.analyze=analysis;").runInContext(context);
  const current={type:"analysis_result",analysis_kind:"face_recognition",stream_session_id:"s",frame:{timestamp_seconds:5},description:"Face narration"};
  context.analyze(current);context.analyze({...current,frame:{timestamp_seconds:3},description:"Late face narration"});
  assert.equal(ui.overallDescription.textContent,"Periodic global scene description");
  assert.equal(shown.length,1);assert.equal(logged.length,1);assert.match(ui.resultJson.textContent,/Face narration/);
  assert.doesNotMatch(ui.resultJson.textContent,/Late face/);
});


test("scene content follows the initial Chinese interface and switches instantly without another request",async()=>{
  const view=fixture({language:"zh"});await view.poll();
  assert.equal(view.ui.overallDescription.textContent,summary().summary_translations.zh);
  assert.equal(view.ui.overallDescription.lang,"zh");
  view.setLanguage("en");
  assert.equal(view.ui.overallDescription.textContent,summary().summary_translations.en);
  assert.equal(view.ui.overallDescription.lang,"en");
  view.setLanguage("zh");
  assert.equal(view.ui.overallDescription.textContent,summary().summary_translations.zh);
  assert.equal(view.requests.length,1,"Switching uses the same observation, without model or API requests");
  assert.equal(view.ui.overallDescription.dataset.interfaceCopy,"false","Generated prose bypasses interface dictionary substitution");
});
test("a summary arriving after a language switch renders in the new interface language",async()=>{
  const view=fixture({language:"en",response:()=>{view.setLanguage("zh");return {status:view.manager,items:[summary()]};}});
  await view.poll();assert.equal(view.ui.overallDescription.textContent,summary().summary_translations.zh);
});
test("legacy unlabelled prose waits for localized content instead of leaking the wrong language",async()=>{
  const view=fixture({language:"zh",items:[summary({summary_translations:undefined})]});await view.poll();
  assert.match(view.ui.overallDescription.textContent,/等待当前语言/);
  assert.doesNotMatch(view.ui.overallDescription.textContent,/bright room/);
});
test("an explicitly labelled legacy description is shown only in its matching locale",async()=>{
  const view=fixture({items:[summary({summary_translations:undefined,summary_language:"en"})]});await view.poll();
  assert.equal(view.ui.overallDescription.textContent,summary().summary);
  view.setLanguage("zh");assert.match(view.ui.overallDescription.textContent,/等待当前语言/);
});
test("missing or malformed localized content never falls back to the opposite-language prose",async()=>{
  for(const zh of [null,"",42,{text:"bad"}]){
    const view=fixture({language:"zh",items:[summary({summary_translations:{en:"English text only",zh}})]});await view.poll();
    assert.match(view.ui.overallDescription.textContent,/等待当前语言/);
    view.setLanguage("en");assert.equal(view.ui.overallDescription.textContent,"English text only");
  }
});
test("language switching after stop cannot revive a cached translated description",async()=>{
  const view=fixture();await view.poll();view.state.running=false;view.setLanguage("zh");
  assert.equal(view.ui.overallDescription.textContent,"任务已停止");
  assert.equal(view.requests.length,1);
});
