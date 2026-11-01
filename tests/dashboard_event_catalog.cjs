const assert=require("node:assert/strict"),{test}=require("node:test"),vm=require("node:vm");
const {createDocument,runDashboard}=require("./helpers/dashboard.cjs");
function preset(id,{name=id,enabled=true,sensitivity=null,scope="person"}={}){
  return {id,labels:{zh:name,en:"English "+id},aliases:[],default_enabled:enabled,kind:"detection",scope,sensitivity};
}
const payload=events=>({event_catalog:{version:1,events}});
function fixture(){
  const {document,element}=createDocument([],{rejectInnerHTML:true});
  const context=vm.createContext({document,window:{isSecureContext:true},location:{hostname:"localhost"},
    setTimeout,clearTimeout,AbortController,fetch:async()=>({ok:false,status:503})});
  runDashboard(context,`globalThis.app={renderEventCatalog,selectedFeatures,selectedSensitivities,sensitivityInputs,state,info,
    inputs:()=>featureInputs,sceneName,canonicalEventLabel};`);
  return {app:context.app,context,element};
}
test("new backend presets drive selectors, overview, archive, scope and nonstandard options",()=>{
  const view=fixture(),events=[preset("spill",{name:"<img src=x onerror=alert(1)>",scope:"scene",sensitivity:{default:"strict",options:[
    {value:"strict",label:"严格",description:"可见积水"},{value:"relaxed",label:"宽松",description:"更多迹象"} ]}}),preset("door_open",{enabled:false})];
  view.app.renderEventCatalog(payload(events));
  assert.deepEqual([...view.app.selectedFeatures()],["spill"]);
  assert.deepEqual({...view.app.selectedSensitivities()},{spill:"strict"});
  assert.equal(view.element("feature-select").children[0].children[1].textContent,events[0].labels.zh);
  assert.deepEqual(view.element("archive-feature").children.map(option=>option.value),["","spill","door_open"]);
  assert.equal(view.element("test-presets").children.length,2);
  assert.equal(view.context.GoodvisionEvents.get("spill").scope,"scene");
  assert.equal(view.app.sceneName("old_event"),"old_event","historical unknown IDs remain readable");
});
test("refresh preserves choices and valid parameter drafts while dropping removed presets",()=>{
  const view=fixture(),sensitivity={default:"strict",options:[{value:"strict",label:"严格"},{value:"relaxed",label:"宽松"}]};
  view.app.renderEventCatalog(payload([preset("spill",{sensitivity}),preset("door_open")]));
  view.app.inputs()[0].checked=false;view.app.sensitivityInputs.get("spill").value="relaxed";
  view.element("archive-feature").value="door_open";
  view.app.renderEventCatalog(payload([preset("spill",{sensitivity}),preset("new_event",{enabled:false})]));
  assert.deepEqual([...view.app.selectedFeatures()],[]);
  assert.equal(view.app.sensitivityInputs.get("spill").value,"relaxed");
  assert.equal(view.element("archive-feature").value,"");
  view.app.renderEventCatalog(payload([preset("spill",{sensitivity:{default:"strict",options:[{value:"strict",label:"严格"}]}})]));
  assert.equal(view.app.sensitivityInputs.get("spill").value,"strict");
});
test("empty catalog renders no options and malformed versions do not replace valid metadata",async()=>{
  const view=fixture();view.app.renderEventCatalog(payload([]));
  assert.equal(view.app.inputs().length,0);
  assert.match(view.element("feature-select").textContent,/暂无/);
  assert.throws(()=>view.app.renderEventCatalog({event_catalog:{version:2,events:[]}}),/目录/);
  view.app.state.info={configured:true};await view.app.info();
  assert.equal(view.app.state.info,null);
  assert.equal(view.element("start-button").disabled,true);
});
test("event metadata supplies aliases and translations without modifying frontend dictionaries",()=>{
  const catalog=require("../src/goodvision_harness/assets/event_catalog.js");
  const i18n=require("../src/goodvision_harness/assets/console_i18n.js");
  const previous=globalThis.GoodvisionI18n;globalThis.GoodvisionI18n=i18n;
  try{
    const event=preset("spill",{name:"地面积水"});event.aliases=["water_spill"];
    catalog.accept(payload([event]));
    assert.equal(catalog.canonical("water-spill"),"spill");
    assert.equal(i18n.translate("地面积水","en"),"English spill");
    assert.equal(i18n.translate("English spill","zh"),"地面积水");
    const display=require("../src/goodvision_harness/camera_display.js");
    assert.match(display.eventObservationSummary({state:"completed",judgments:[{event_label:"spill",detected:true},{event_label:"retired_event",detected:false}]}),/地面积水.*retired_event/);
  }finally{globalThis.GoodvisionI18n=previous;}
});
