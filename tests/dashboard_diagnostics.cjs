const assert = require("node:assert/strict");
const test = require("node:test");
const {resultText, diagnosticsBase} = require("../src/goodvision_harness/assets/scene_diagnostics.js");

test("missing weapon verdict is not displayed as negative", () => {
  const text = resultText({response_outcome:"validated",result:"partial",
    detected_labels:[],missing_event_labels:["weapon"],unknown_event_labels:[]});
  assert.match(text, /部分完成/); assert.match(text, /weapon/); assert.doesNotMatch(text, /阴性/);
});
test("valid positive survives missing other label", () => {
  assert.match(resultText({response_outcome:"validated",result:"partial",
    detected_labels:["fight"],missing_event_labels:["weapon"]}), /阳性：fight/);
});
test("null and request failures cannot become negatives", () => {
  assert.match(resultText({response_outcome:"validated",unknown_event_labels:["fall"]}), /无法判断/);
  assert.match(resultText({response_outcome:"failed",failure_category:"timeout"}), /请求失败/);
});

test("camera diagnostics use the same scoped base for metadata and JPEGs", () => {
  assert.equal(diagnosticsBase("camera02"), "/v1/fleet/cameras/camera02/scene/diagnostics");
  assert.equal(diagnosticsBase(undefined), "/v1/scene/diagnostics");
  assert.equal(diagnosticsBase("../other"), "/v1/scene/diagnostics");
});

test("scoped panel fetches its camera and never claims unobserved confirmation", async () => {
  const vm = require("node:vm");
  const fs = require("node:fs");
  function element() {
    return {children:[], append(child){this.children.push(child)},
      replaceChildren(){this.children=[]}, style:{}};
  }
  let toggle;
  const panel = {open:true, addEventListener(_name, fn){toggle=fn}};
  const content = element(), calls=[];
  const context = {
    GoodvisionDiagnosticCameraId:"camera02", AbortController,
    setTimeout:()=>1, clearTimeout:()=>{}, addEventListener:()=>{},
    document:{hidden:false, addEventListener:()=>{}, createElement:element,
      getElementById:id=>id==="scene-diagnostics"?panel:content},
    fetch:async url=>{
      calls.push(url);
      return {ok:true,json:async()=>({records:[{
        request_id:"window:full", response_outcome:"validated", detected_labels:["smoking"],
        evidence_outcome:"not_observed", confirmed_event_labels:null, output_mode:"full",
        images_available:true,frames:[{frame_sequence:12,timestamp_seconds:.8,width:24,height:18}]
      }]})};
    }
  };
  vm.runInNewContext(fs.readFileSync(require.resolve(
    "../src/goodvision_harness/assets/scene_diagnostics.js"),"utf8"),context);
  toggle();
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(calls,["/v1/fleet/cameras/camera02/scene/diagnostics"]);
  const item=content.children[1];
  assert.match(item.children[0].textContent,/事件确认：此诊断未采集/);
  assert.doesNotMatch(item.children[0].textContent,/已确认事件：无/);
  assert.equal(item.children[1].children[0].src,
    "/v1/fleet/cameras/camera02/scene/diagnostics/window%3Afull/12.jpg");
});


test("disabled diagnostics explain state and stop polling until reopened", async () => {
  const vm = require("node:vm");
  const fs = require("node:fs");
  let toggle;
  const calls = [], delays = [];
  const panel = {open:true, addEventListener(_name, fn){toggle=fn}};
  const content = {replaceChildren(){}, textContent:""};
  const context = {
    AbortController, addEventListener:()=>{}, clearTimeout:()=>{},
    setTimeout:(_fn, delay)=>{delays.push(delay); return delays.length},
    document:{hidden:false, addEventListener:()=>{},
      createElement:()=>{throw new Error("disabled panel must not create image elements")},
      getElementById:id=>id==="scene-diagnostics"?panel:content},
    fetch:async url=>{
      calls.push(url);
      return {ok:true,json:async()=>({enabled:false,reason:"disabled",records:[]})};
    }
  };
  vm.runInNewContext(fs.readFileSync(require.resolve(
    "../src/goodvision_harness/assets/scene_diagnostics.js"),"utf8"),context);
  toggle();
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(content.textContent,/诊断未启用/);
  assert.deepEqual(calls,["/v1/scene/diagnostics"]);
  assert.deepEqual(delays,[3000]); // Only the request timeout, no poll timer.
  toggle();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.length,2);
});


test("expanded diagnostics pause hidden parent requests and images, then resume",async()=>{
  const vm=require("node:vm"),fs=require("node:fs");
  let active=false,toggle,resolve;const calls=[],timers=new Map(),images=[];let nextTimer=0;
  function element(tag){return {children:[],style:{},append(...items){this.children.push(...items);},replaceChildren(){this.children=[];},
    set src(value){images.push(value);}};}
  const panel={open:true,addEventListener(_name,fn){toggle=fn;}},content=element("div");
  const context={AbortController,channelSurfaceActive:()=>active,addEventListener(){},
    GoodvisionDiagnosticCameraId:"camera02",
    setTimeout(fn,delay){const id=++nextTimer;timers.set(id,{fn,delay});return id;},clearTimeout:id=>timers.delete(id),
    document:{hidden:false,addEventListener(){},createElement:element,getElementById:id=>id==="scene-diagnostics"?panel:content},
    fetch:async url=>{calls.push(url);return new Promise(done=>{resolve=done;});}};
  vm.runInNewContext(fs.readFileSync(require.resolve("../src/goodvision_harness/assets/scene_diagnostics.js"),"utf8"),context);
  const wake=()=>{const [id,timer]=[...timers].find(([,value])=>value.delay===2000);timers.delete(id);timer.fn();};
  const finish=()=>resolve({ok:true,json:async()=>({records:[{images_available:true,frames:[{frame_sequence:1}],request_id:"one"}]})});
  toggle();assert.deepEqual(calls,[]);
  active=true;wake();assert.equal(calls.length,1);
  active=false;finish();await new Promise(done=>setImmediate(done));
  assert.deepEqual(images,[],"A response that finished after hiding cannot start image requests");
  wake();assert.equal(calls.length,1);
  active=true;wake();assert.equal(calls.length,2);finish();await new Promise(done=>setImmediate(done));
  assert.deepEqual(images,["/v1/fleet/cameras/camera02/scene/diagnostics/one/1.jpg"]);
});
