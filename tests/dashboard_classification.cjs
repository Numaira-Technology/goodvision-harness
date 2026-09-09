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
    globalThis.app={analysis,message,status,reset,state};
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

test("overall description displays the latest frame text",()=>{
  const view=dashboard();
  const description="画面中发现陌生人。 / An unknown person is visible in the frame.";
  view.app.analysis({...result(1,{...clear,stranger:true}),description});
  assert.equal(view.element("overall-description").textContent,description);
  view.app.analysis({...result(1.4,clear),description:"无异常 / No anomalies detected"});
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
