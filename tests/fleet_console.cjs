const assert=require("node:assert/strict");
const {test}=require("node:test");
const fs=require("node:fs");
const path=require("node:path");
const {FleetConsole,normalizeFleet}=require("../src/goodvision_harness/fleet_console.js");

class Element{
  constructor(tag="div"){this.tagName=tag;this.children=[];this.hidden=false;this.disabled=false;this.value="";this.dataset={};this.attributes={};}
  append(...children){for(const child of children){child.parent=this;this.children.push(child);}}
  replaceChildren(...children){this.children=[];this.append(...children);}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);}
  setAttribute(key,value){this.attributes[key]=value;}
  removeAttribute(key){delete this.attributes[key];delete this[key];}
  focus(){this.focused=true;}
  scrollIntoView(){}
}
function fixture({count=3,allowed=true,mode=allowed?"local":"legacy",usable=allowed?count:1}={}){
  const elements=new Map();const document={hidden:false,createElement:tag=>new Element(tag),getElementById:id=>{
    if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
  }};
  const data={tracking_mode:mode,capacity:count,usable_capacity:usable,multi_camera_allowed:allowed,reason:allowed?"":"原部署追踪服务仅支持单路",
    analysis_start_url:"/v1/fleet/analysis/start",analysis_stop_url:"/v1/fleet/analysis/stop",analysis:{state:"idle",camera_ids:[],errors:{}},cameras:Array.from({length:count},(_,i)=>{
    const id="camera"+String(i+1).padStart(2,"0"),base="/v1/fleet/cameras/"+id;
    return {id,label:"摄像头 "+(i+1),own:i===0,enabled:i<usable,status_url:base+"/status",connect_url:base+"/connect",disconnect_url:base+"/disconnect",settings_url:base+"/settings",detail_url:base+"/analysis/settings",diagnostics_url:base+"/diagnostics",results_url:base+"/results/latest",scene_url:base+"/scene",preview_url:base+"/video/annotated/latest.jpg"};
  })};
  const requests=[],statuses=new Map(data.cameras.map(camera=>[camera.id,{pipeline_state:"idle",connection_state:"disconnected",analysis_state:"idle",configured_stream:false}]));
  const prefs=new Map(data.cameras.map(camera=>[camera.id,{features:["fight","smoking","fall","weapon","stranger"],event_sensitivities:{fight:"standard",smoking:"high",fall:"standard",weapon:"standard"},event_sensitivity_options:{smoking:[{value:"standard",label:"标准敏感度",description:"必须明确识别烟具"},{value:"high",label:"高敏感度",description:"允许结合动作和上下文判断"}]}}]));
  const errors=new Map(),revoked=[],scheduled=[];let blobId=0;
  const fetch=async(url,options={})=>{
    requests.push({url,method:options.method||"GET",body:options.body&&JSON.parse(options.body)});
    if(errors.has(url))throw new Error(errors.get(url));
    if(url==="/v1/fleet")return {ok:true,json:async()=>data};
    if(url===data.analysis_start_url||url===data.analysis_stop_url){
      const start=url===data.analysis_start_url;
      data.analysis={state:start?"running":"idle",camera_ids:start?[...statuses].filter(([,status])=>status.connection_state==="connected").map(([id])=>id):[],errors:{}};
      for(const [id,status] of statuses)statuses.set(id,{...status,pipeline_state:start&&data.analysis.camera_ids.includes(id)?"running":"idle",analysis_state:start&&data.analysis.camera_ids.includes(id)?"running":"idle"});
      return {ok:true,json:async()=>({analysis:data.analysis})};
    }
    const camera=data.cameras.find(camera=>url.startsWith("/v1/fleet/cameras/"+camera.id+"/"));
    if(!camera)throw new Error("Unknown URL "+url);
    if(url===camera.preview_url)return {ok:true,blob:async()=>({camera:camera.id})};
    if(url===camera.detail_url){if(options.method==="PUT")prefs.set(camera.id,{...prefs.get(camera.id),...JSON.parse(options.body)});return {ok:true,json:async()=>prefs.get(camera.id)};}
    if(url===camera.results_url)return {ok:true,json:async()=>({camera:camera.id,description:"结果"})};
    if(url===camera.scene_url)return {ok:true,json:async()=>({camera:camera.id,events:[]})};
    if(options.method==="POST")statuses.set(camera.id,{...statuses.get(camera.id),pipeline_state:"idle",analysis_state:"idle",connection_state:url.endsWith("/disconnect")?"disconnected":"connected",stream_session_id:camera.id,ready:false});
    return {ok:true,json:async()=>({...statuses.get(camera.id)})};
  };
  const app=new FleetConsole({document,fetch,clock:()=>100,setTimeout:fn=>{scheduled.push(fn);return scheduled.length;},clearTimeout:()=>{},
    AbortController,URL:{createObjectURL:()=>"blob:frame-"+(++blobId),revokeObjectURL:value=>revoked.push(value)}});
  return {app,data,document,requests,statuses,prefs,errors,revoked,scheduled};
}

test("status request failure does not label a still-playing video offline",async()=>{
  const view=fixture({count:1});await view.app.refresh();
  const row=view.app.rows.get("camera01");
  row.status={connection_state:"connected",analysis_state:"running"};row.lastPresentedAt=100;
  view.errors.set(row.camera.status_url,"timeout");await view.app.pollStatus();
  assert.equal(row.unreachable,true);assert.match(row.state.textContent,/视频播放中.*状态更新延迟/);
  row.lastPresentedAt=-5000;view.app.updateRow(row);
  assert.match(row.state.textContent,/暂时无法确认/);assert.doesNotMatch(row.state.textContent,/离线/);
  view.errors.delete(row.camera.status_url);await view.app.pollStatus();
  assert.equal(row.unreachable,false);view.app.close();
});

test("VLM health distinguishes recent success, incomplete labels and a failed attempt",async()=>{
  const view=fixture({count:1});await view.app.refresh();const row=view.app.rows.get("camera01");
  row.status={connection_state:"connected",analysis_state:"running",vlm:{independent:true,running:true,ready:true,last_success_age_seconds:1.2,analysis_complete:false}};
  view.app.updateRow(row);assert.match(row.vlmNote.textContent,/识别持续返回.*1.2.*部分检测项待确认/);
  row.status.vlm.last_attempt_state="failed";view.app.updateRow(row);
  assert.match(row.vlmNote.textContent,/最近一次失败/);assert.doesNotMatch(row.vlmNote.textContent,/离线/);
  Object.assign(row.status.vlm,{ready:false,last_success_age_seconds:20});view.app.updateRow(row);
  assert.match(row.vlmNote.textContent,/结果延迟.*20.0/);
  row.status.vlm.running=false;view.app.updateRow(row);assert.match(row.vlmNote.textContent,/暂不可用.*视频继续/);
  view.app.close();
});

test("each lane owns its diagnostic document; closing one leaves the others open",async()=>{
  const view=fixture({count:37});await view.app.refresh();
  for(const id of ["camera01","camera02","camera37"]){
    const row=view.app.rows.get(id);row.diagnostics.onclick();
    assert.equal(row.diagnosticsFrame.src,"/v1/fleet/cameras/"+id+"/diagnostics");
    assert.equal(row.diagnosticsPanel.hidden,false);
    assert.equal(row.diagnostics.attributes["aria-expanded"],"true");
  }
  const second=view.app.rows.get("camera02"),first=view.app.rows.get("camera01");
  const original=first.diagnosticsFrame;second.diagnostics.onclick();
  assert.equal(second.diagnosticsFrame,null);assert.equal(second.diagnosticsPanel.children.length,0);
  assert.equal(first.diagnosticsFrame,original);
  second.camera.diagnostics_url=null;second.diagnostics.onclick();
  assert.equal(second.diagnosticsFrame,null);assert.match(second.error,/诊断入口/);
  view.app.close();assert.equal(first.diagnosticsFrame,null);
});

test("legacy mode permits one camera while local tracking exposes arbitrary provisioned capacity",async()=>{
  const legacy=fixture({count:16,allowed:false});await legacy.app.refresh();
  assert.equal(legacy.app.visible.size,1);assert.equal(legacy.app.ui.add.hidden,false);assert.equal(legacy.app.ui.add.disabled,true);
  assert.equal(legacy.app.ui.mode.textContent,"原部署追踪服务 · 单路模式");
  assert.match(legacy.app.ui.capacity.textContent,/已部署 16 路通道，当前可用 1 路/);
  legacy.app.add();assert.equal(legacy.app.visible.size,1);
  for(const count of [1,2,5,12,16,24,37]){
    const view=fixture({count});await view.app.refresh();
    for(let i=1;i<count;i++)view.app.add();
    assert.equal(view.app.visible.size,count);assert.equal(view.app.ui.add.disabled,true);
    assert.equal(view.app.ui.mode.textContent,"本机 GPU 人物追踪 · 多路模式");
    assert.match(view.app.ui.capacity.textContent,new RegExp("已部署 "+count+" 路"));
    assert.match(view.app.ui.capacity.textContent,new RegExp("当前可用 "+count+" 路"));
    assert.doesNotMatch(view.app.ui.capacity.textContent,/12 路为/);
  }
});

test("cloud GPU exposes all provisioned cameras without a local GPU or a one-stream cap",async()=>{
  const view=fixture({count:37,usable:37,mode:"cloud"});
  await view.app.refresh();for(let i=1;i<37;i++)view.app.add();
  assert.equal(view.app.visible.size,37);
  assert.equal(view.app.ui.mode.textContent,"云 GPU 人物追踪 · 多路模式");
  assert.doesNotMatch(view.app.ui.capacity.textContent,/仅支持单路/);
});

test("independent tracking server supports multiple cameras without a local GPU and reports its usable capacity",async()=>{
  const view=fixture({count:24,usable:16,mode:"server"});
  view.data.local_gpu_available=false;
  await view.app.refresh();for(let i=0;i<24;i++)view.app.add();
  assert.equal(view.app.ui.mode.textContent,"独立 GPU 追踪服务器 · 多路模式");
  assert.equal(view.app.visible.size,16);assert.equal(view.app.ui.add.disabled,true);
  assert.match(view.app.ui.capacity.textContent,/已部署 24 路通道，当前可用 16 路/);
  assert.doesNotMatch(view.app.ui.capacity.textContent,/启用本地|启用本机|仅可连接 1 路/);
  const row=view.app.rows.get("camera16");row.input.value="rtsp://camera16/live";
  assert.equal(await view.app.start(row),true);
  assert.equal(view.app.rows.get("camera17").start.disabled,true);
});

test("backend camera flags are authoritative even for the primary camera and do not depend on the mode",async()=>{
  const view=fixture({count:3,mode:"local",usable:0,allowed:false});
  await view.app.refresh();
  assert.equal(view.app.visible.size,0);assert.equal(view.app.rows.get("camera01").start.disabled,true);
  assert.match(view.app.ui.capacity.textContent,/当前可用 0 路/);
  const normalized=normalizeFleet({tracking_mode:"server",capacity:3,usable_capacity:1,multi_camera_allowed:false,cameras:[
    {id:"primary",own:true,enabled:false},{id:"usable-peer",enabled:true},{id:"unspecified"}
  ]});
  assert.deepEqual(normalized.cameras.map(camera=>camera.enabled),[false,true,false]);
  assert.equal(normalized.trackingMode,"server");
  view.data.tracking_mode="server";view.data.multi_camera_allowed=true;view.data.usable_capacity=1;view.data.cameras[1].enabled=true;
  await view.app.refresh();
  assert.deepEqual([...view.app.visible],["camera02"]);assert.equal(view.app.rows.get("camera02").start.disabled,false);
});

test("one camera connects with its own URL and settings without touching another camera",async()=>{
  const view=fixture();await view.app.refresh();view.app.add();
  const row=view.app.rows.get("camera02"),first=view.app.rows.get("camera01");
  row.input.value="rtsp://private-user:private-password@camera-two/live";row.layout.value="stereo";
  assert.equal(await view.app.start(row),true);
  const posts=view.requests.filter(request=>request.method==="POST");
  assert.equal(posts.length,1);assert.match(posts[0].url,/camera02\/connect$/);
  assert.equal(posts[0].body.stream_url,row.input.value);assert.equal(posts[0].body.camera_layout,"stereo");
  assert.equal(Object.hasOwn(posts[0].body,"dry_run"),false,"saved no-model test mode must not be overridden");assert.equal(Object.hasOwn(posts[0].body,"features"),false,"connection must not start or configure analysis");
  assert.equal(first.status.connection_state,"disconnected");assert.equal(row.status.connection_state,"connected");assert.equal(row.status.analysis_state,"idle");
  assert.equal(row.remove.disabled,true);view.app.remove(row);assert.equal(view.app.visible.has("camera02"),true);
  assert.equal(await view.app.stop(row),true);assert.equal(row.status.connection_state,"disconnected");
  view.app.remove(row);assert.equal(view.app.visible.has("camera02"),false);
});

test("saved sources are write-only, can start without a URL and newly saved values are cleared",async()=>{
  const view=fixture();view.statuses.set("camera01",{pipeline_state:"idle",configured_stream:true,camera_layout:"stereo"});await view.app.refresh();
  const row=view.app.rows.get("camera01");assert.equal(row.input.value,"");assert.match(row.input.placeholder,/不回显/);
  assert.equal(await view.app.start(row),true);
  assert.equal(Object.hasOwn(view.requests.find(request=>request.method==="POST").body,"stream_url"),false);
  await view.app.stop(row);row.input.value="rtsp://camera-one/live";view.app.updateRow(row);
  assert.equal(await view.app.save(row),true);
  assert.match(view.requests.at(-1).url,/camera01\/settings$/);assert.equal(view.requests.at(-1).method,"PUT");
  assert.equal(row.input.value,"");assert.equal(row.status.configured_stream,true);
});

test("missing URLs fail only their own row and bulk start uses bounded concurrency",async()=>{
  const view=fixture({count:16});await view.app.refresh();for(let i=1;i<16;i++)view.app.add();
  const missing=view.app.rows.get("camera01");assert.equal(await view.app.start(missing),false);assert.match(missing.message.textContent,/填写/);
  let active=0,peak=0;const actual=view.app.fetch;
  view.app.fetch=async(url,options)=>{
    if(options.method==="POST"){
      active++;peak=Math.max(peak,active);await new Promise(resolve=>setTimeout(resolve,2));
      const result=await actual(url,options);active--;return result;
    }
    return actual(url,options);
  };
  for(const row of view.app.rows.values())if(row!==missing)row.input.value="rtsp://"+row.camera.id+"/live";
  await view.app.all("start");
  assert.equal(view.requests.filter(request=>request.method==="POST").length,15);
  assert.equal(peak,2);assert.match(view.app.ui.note.textContent,/15 路成功，1 路未完成/);
  assert.equal(missing.status.pipeline_state,"idle");
});

test("live cameras reappear after reload and stopped cameras never consume preview requests",async()=>{
  const view=fixture({count:16});view.statuses.set("camera16",{pipeline_state:"idle",connection_state:"connected",analysis_state:"idle",stream_session_id:"active16"});await view.app.refresh();
  assert.equal(view.app.visible.has("camera16"),true);assert.equal(view.app.visible.size,2);
  view.requests.length=0;await view.app.previews();
  assert.equal(view.requests.length,1);assert.match(view.requests[0].url,/camera16\/video\/annotated\/latest.jpg$/);
  const row=view.app.rows.get("camera16");assert.equal(row.image.hidden,false);
  view.statuses.set("camera16",{pipeline_state:"idle"});await view.app.pollStatus();
  assert.equal(row.image.hidden,true);assert.equal(row.previewUrl,null);assert.equal(view.revoked.length,1);
});

test("16 preview tiles use finite JPEG responses with only three in flight and pause in hidden tabs",async()=>{
  const view=fixture({count:16});for(const camera of view.data.cameras)view.statuses.set(camera.id,{pipeline_state:"idle",connection_state:"connected",analysis_state:"idle"});await view.app.refresh();
  let active=0,peak=0;const actual=view.app.fetch;
  view.app.fetch=async(url,options)=>{active++;peak=Math.max(peak,active);await new Promise(resolve=>setTimeout(resolve,2));const result=await actual(url,options);active--;return result;};
  view.requests.length=0;await view.app.previews();
  assert.equal(view.requests.length,16);assert.equal(peak,3);assert.ok(view.requests.every(request=>request.url.endsWith("/latest.jpg")));
  view.document.hidden=true;await view.app.previews();assert.equal(view.requests.length,16);
  view.app.close();assert.ok([...view.app.rows.values()].every(row=>row.previewUrl===null));
});

test("metadata failure disables starts, successful refresh restores controls, and status failures recover independently",async()=>{
  const view=fixture();await view.app.refresh();const row=view.app.rows.get("camera01");
  view.errors.set("/v1/fleet","offline");await view.app.refresh();
  assert.equal(view.app.metadataOk,false);assert.equal(view.app.ui.add.disabled,true);assert.equal(row.start.disabled,true);
  view.errors.delete("/v1/fleet");await view.app.refresh();assert.equal(row.start.disabled,false);
  view.errors.set(row.camera.status_url,"offline");await view.app.pollStatus();assert.equal(row.unreachable,true);assert.match(row.message.textContent,/offline/);
  view.errors.delete(row.camera.status_url);await view.app.pollStatus();assert.equal(row.unreachable,false);assert.equal(row.error,"");
});

test("server capability changes take effect only from saved backend state and unsafe endpoints are rejected",async()=>{
  const view=fixture();await view.app.refresh();view.app.add();assert.equal(view.app.visible.size,2);
  view.data.tracking_mode="legacy";view.data.multi_camera_allowed=false;view.data.usable_capacity=1;view.data.cameras[1].enabled=false;view.data.cameras[2].enabled=false;
  await view.app.refresh();assert.equal(view.app.visible.size,1);assert.equal(view.app.ui.add.hidden,false);assert.equal(view.app.ui.add.disabled,true);
  const normalized=normalizeFleet({multi_camera_allowed:true,cameras:[{id:"unsafe",preview_url:"//evil.example/video",connect_url:"https://evil.example/connect",status_url:"/\\evil.example/status"}]});
  assert.equal(normalized.cameras[0].preview_url,null);assert.equal(normalized.cameras[0].connect_url,null);assert.equal(normalized.cameras[0].status_url,null);
});

test("dashboard retains original controls and loads the standalone fleet controller",()=>{
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  assert.match(html,/id="fleet-detail-console" hidden/);assert.match(html,/id="start-button"/);
  assert.match(html,/<script src="\/assets\/fleet_console.js"><\/script>/);
  assert.match(html,/GoodvisionFleet\?\.refresh\?\.\(\)/);
  assert.match(html,/GoodvisionFleet\?\.detailHidden\?\.\(\)/);
});

test("legacy mode discovers surviving peer sessions and keeps stop controls available after primary restart",async()=>{
  const view=fixture({count:16,allowed:false});
  view.statuses.set("camera16",{pipeline_state:"idle",connection_state:"connected",analysis_state:"idle",stream_session_id:"surviving-peer",ready:true});
  await view.app.refresh();
  const row=view.app.rows.get("camera16");
  assert.equal(view.app.visible.has("camera16"),true);
  assert.equal(row.start.disabled,true);assert.equal(row.stop.disabled,false);
  assert.equal(view.app.ui["stop-all"].hidden,false);assert.equal(view.app.extraBusy(),false);
  assert.match(view.app.ui.capacity.textContent,/历史会话/);
  view.requests.length=0;await view.app.pollStatus();
  assert.equal(view.requests.length,2,"routine polls cover primary plus surviving peer, not all idle peers");
  await view.app.all("stop");assert.equal(row.status.pipeline_state,"idle");
  assert.equal(view.app.visible.has("camera16"),false);assert.equal(view.app.ui["stop-all"].disabled,true);
});

test("server capacity reductions keep disabled live channels visible, polled and stoppable",async()=>{
  const view=fixture({count:24,usable:16,mode:"server"});
  view.statuses.set("camera24",{pipeline_state:"idle",connection_state:"connected",analysis_state:"idle",stream_session_id:"existing-server-session",ready:true});
  await view.app.refresh();const row=view.app.rows.get("camera24");
  assert.equal(view.app.visible.has("camera24"),true);assert.equal(row.start.disabled,true);assert.equal(row.stop.disabled,false);
  assert.match(view.app.ui.capacity.textContent,/历史会话/);
  view.errors.set(row.camera.status_url,"offline");await view.app.pollStatus();
  assert.equal(row.unreachable,true);assert.equal(row.stop.disabled,false);
  view.errors.delete(row.camera.status_url);await view.app.pollStatus();assert.equal(row.unreachable,false);
  view.requests.length=0;await view.app.previews();
  assert.equal(view.requests.length,1);assert.match(view.requests[0].url,/camera24\/video\/annotated\/latest.jpg$/);
  await view.app.stop(row);assert.equal(row.status.connection_state,"disconnected");assert.equal(view.app.visible.has("camera24"),false);
});

test("compact monitoring view can be expanded and adding a channel reveals its connection fields",async()=>{
  const view=fixture({count:5});
  for(const camera of view.data.cameras.slice(0,4))view.statuses.set(camera.id,{pipeline_state:"idle",connection_state:"connected",analysis_state:"idle"});
  await view.app.refresh();assert.equal(view.app.compact,true);assert.match(view.app.ui.grid.className,/compact/);
  view.app.ui["view-toggle"].onclick();assert.equal(view.app.compact,false);
  view.app.ui["view-toggle"].onclick();assert.equal(view.app.compact,true);
  view.app.add();assert.equal(view.app.compact,false);assert.equal(view.app.visible.has("camera05"),true);
});

test("global analysis starts every connected camera, locks the cohort, and stopping preserves video previews",async()=>{
  const view=fixture({count:3});await view.app.refresh();view.app.add();
  const first=view.app.rows.get("camera01"),second=view.app.rows.get("camera02"),third=view.app.rows.get("camera03");
  for(const row of [first,second]){row.input.value="rtsp://"+row.camera.id+"/live";await view.app.start(row);}
  assert.equal(view.app.ui["analysis-start"].disabled,false);assert.equal(view.app.ui["analysis-stop"].disabled,true);
  view.requests.length=0;assert.equal(await view.app.analysis("start"),true);
  const posts=view.requests.filter(request=>request.method==="POST");
  assert.deepEqual(posts,[{url:"/v1/fleet/analysis/start",method:"POST",body:{}}]);
  assert.deepEqual(view.data.analysis.camera_ids,["camera01","camera02"]);
  for(const row of [first,second,third]){assert.equal(row.start.disabled,true);assert.equal(row.stop.disabled,true);assert.equal(row.remove.disabled,true);}
  assert.equal(view.app.ui.add.disabled,true);assert.equal(view.app.ui["stop-all"].disabled,true);
  assert.match(view.app.ui.count.textContent,/视频已连接 2 路 · 正在分析 2 路/);
  assert.equal(await view.app.stop(first),false);view.app.add();assert.equal(view.app.visible.size,2);
  assert.equal(await view.app.analysis("stop"),true);
  assert.equal(first.status.connection_state,"connected");assert.equal(first.status.analysis_state,"idle");
  assert.equal(second.stop.disabled,false);assert.equal(view.app.ui["analysis-start"].disabled,false);
  view.requests.length=0;await view.app.previews();assert.equal(view.requests.length,2);
  await view.app.stop(second);assert.equal(first.status.connection_state,"connected");assert.equal(second.status.connection_state,"disconnected");
});

test("single-camera mode still uses explicit global analysis controls",async()=>{
  const view=fixture({count:1,allowed:false});await view.app.refresh();
  const row=view.app.rows.get("camera01");row.input.value="rtsp://camera/live";await view.app.start(row);
  assert.equal(view.app.ui["analysis-start"].hidden,false);assert.equal(view.app.ui["analysis-stop"].hidden,false);
  assert.equal(row.start.textContent,"连接视频");assert.equal(row.stop.textContent,"断开视频");
  assert.equal(await view.app.analysis("start"),true);assert.equal(row.stop.disabled,true);
  assert.equal(await view.app.analysis("stop"),true);assert.equal(row.stop.disabled,false);
});

test("each card loads and saves its own preferences, catalog, status, results and scene",async()=>{
  const view=fixture();view.prefs.get("camera02").features=["smoking","stranger"];
  await view.app.refresh();view.app.add();const first=view.app.rows.get("camera01"),second=view.app.rows.get("camera02");
  assert.equal(second.detail.hidden,false);await view.app.toggleDetail(second);
  assert.equal(second.detailPanel.hidden,false);assert.equal(second.prefsLoaded,true);
  assert.equal(second.features.get("fight").checked,false);assert.equal(second.features.get("stranger").checked,true);
  assert.equal(second.sensitivities.get("smoking").select.value,"high");
  assert.match(second.sensitivities.get("smoking").note.textContent,/动作和上下文/);
  assert.match(second.resultJson.textContent,/camera02/);assert.match(second.sceneJson.textContent,/camera02/);
  assert.match(second.detailSummary.textContent,/视频未连接/);assert.equal(first.prefsLoaded,false);
  second.features.get("fall").checked=true;second.features.get("fall").onchange();
  second.sensitivities.get("smoking").select.value="standard";second.sensitivities.get("smoking").select.onchange();
  assert.equal(await view.app.savePreferences(second),true);
  assert.deepEqual(view.prefs.get("camera02").features,["smoking","fall","stranger"]);
  assert.equal(view.prefs.get("camera02").event_sensitivities.smoking,"standard");
  assert.equal(view.prefs.get("camera01").features.length,5);assert.equal(second.prefsDirty,false);
  second.features.get("fight").checked=true;second.features.get("fight").onchange();await view.app.loadPreferences(second,true);
  assert.equal(second.features.get("fight").checked,false);assert.equal(second.prefsDirty,false);
});

test("unsaved camera preferences persist before the one global start request and are locked while analyzing",async()=>{
  const view=fixture();await view.app.refresh();const row=view.app.rows.get("camera01");
  row.input.value="rtsp://camera/live";await view.app.start(row);await view.app.toggleDetail(row);
  row.features.get("weapon").checked=false;row.features.get("weapon").onchange();
  view.requests.length=0;assert.equal(await view.app.analysis("start"),true);
  const mutations=view.requests.filter(request=>request.method!=="GET");
  assert.equal(mutations.length,2);assert.equal(mutations[0].url,row.camera.detail_url);assert.equal(mutations[0].method,"PUT");
  assert.equal(mutations[1].url,view.data.analysis_start_url);assert.equal(view.prefs.get("camera01").features.includes("weapon"),false);
  assert.ok([...row.features.values()].every(input=>input.disabled));assert.ok([...row.sensitivities.values()].every(({select})=>select.disabled));
  assert.equal(row.prefsReload.disabled,true);assert.equal(await view.app.savePreferences(row),false);
});

test("failed or empty preference saves prevent global analysis without losing edits",async()=>{
  const view=fixture();await view.app.refresh();const row=view.app.rows.get("camera01");
  row.input.value="rtsp://camera/live";await view.app.start(row);await view.app.toggleDetail(row);
  row.features.get("fight").checked=false;row.features.get("fight").onchange();view.errors.set(row.camera.detail_url,"disk unavailable");
  view.requests.length=0;assert.equal(await view.app.analysis("start"),false);
  assert.equal(view.requests.some(request=>request.url===view.data.analysis_start_url),false);assert.equal(row.prefsDirty,true);
  assert.match(row.prefsNote.textContent,/disk unavailable/);view.errors.delete(row.camera.detail_url);
  for(const input of row.features.values())input.checked=false;
  assert.equal(await view.app.analysis("start"),false);assert.match(row.prefsNote.textContent,/至少选择一个/);
});

test("global requests immediately lock connection controls, and unknown peer states block starting",async()=>{
  const view=fixture();await view.app.refresh();const row=view.app.rows.get("camera01");row.input.value="rtsp://camera/live";await view.app.start(row);
  const actual=view.app.fetch;let release,enteredResolve;const entered=new Promise(resolve=>{enteredResolve=resolve;});
  view.app.fetch=async(url,options)=>{if(url===view.data.analysis_start_url)await new Promise(resolve=>{release=resolve;enteredResolve();});return actual(url,options);};
  const starting=view.app.analysis("start");
  assert.equal(row.stop.disabled,true);assert.equal(view.app.ui.add.disabled,true);assert.equal(await view.app.stop(row),false);
  await entered;release();await starting;await view.app.analysis("stop");
  view.errors.set(view.data.cameras[1].status_url,"offline");await view.app.pollStatus();
  assert.equal(view.app.ui["analysis-start"].disabled,true);
});

test("expanded camera details poll only their own result endpoints and sanitize arbitrary text",async()=>{
  const view=fixture({count:24});await view.app.refresh();const row=view.app.rows.get("camera16");
  await view.app.toggleDetail(row);view.requests.length=0;await view.app.pollStatus();
  const detailRequests=view.requests.filter(request=>request.url.endsWith("/results/latest")||request.url.endsWith("/scene"));
  assert.equal(detailRequests.length,2);assert.ok(detailRequests.every(request=>request.url.includes("camera16")));
  const normalized=normalizeFleet({analysis_start_url:"//evil/start",analysis_stop_url:"https://evil/stop",cameras:[{id:"x",detail_url:"//evil/detail",results_url:"https://evil/results",scene_url:"/\\evil/scene"}]});
  assert.equal(normalized.analysisStartUrl,null);assert.equal(normalized.analysisStopUrl,null);
  assert.equal(normalized.cameras[0].detail_url,null);assert.equal(normalized.cameras[0].results_url,null);assert.equal(normalized.cameras[0].scene_url,null);
});

test("diagnostics cannot expose a first-camera-only network analysis action",()=>{
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  assert.match(html,/id="fleet-analysis-start"[^>]*>开始全部分析/);assert.match(html,/id="fleet-analysis-stop"[^>]*>停止全部分析/);
  assert.match(html,/ui\.start\.hidden=fleetManaged;ui\.stop\.hidden=fleetManaged/);
  assert.match(html,/ui\.eventControls\.hidden=true/);assert.match(html,/GoodvisionFleet\.startAnalysis\(\)/);assert.match(html,/GoodvisionFleet\.stopAnalysis\(\)/);
  assert.match(html,/id="default-autostart"[^>]*> 启动时自动连接视频（分析需手动统一开始）/);
});

test("lost global response keeps an unknown cohort locked until authoritative recovery",async()=>{
  const view=fixture();await view.app.refresh();const row=view.app.rows.get("camera01");row.input.value="rtsp://camera/live";await view.app.start(row);
  view.errors.set(view.data.analysis_start_url,"response lost");view.errors.set("/v1/fleet","offline");
  assert.equal(await view.app.analysis("start"),false);
  assert.equal(view.app.fleet.analysis.state,"unknown");assert.equal(row.stop.disabled,true);assert.equal(view.app.ui["analysis-stop"].disabled,false);
  view.errors.delete("/v1/fleet");view.errors.delete(view.data.analysis_start_url);await view.app.refresh();
  assert.equal(view.app.fleet.analysis.state,"idle");assert.equal(row.stop.disabled,false);
});

test("compact view retains per-camera detail access and the global analysis toolbar",async()=>{
  const view=fixture({count:4});for(const camera of view.data.cameras)view.statuses.set(camera.id,{connection_state:"connected",analysis_state:"idle",pipeline_state:"idle"});
  await view.app.refresh();assert.equal(view.app.compact,true);
  assert.equal(view.app.ui["analysis-start"].hidden,false);assert.equal(view.app.ui["analysis-start"].disabled,false);
  const row=view.app.rows.get("camera04");assert.equal(row.detail.hidden,false);await view.app.toggleDetail(row);assert.equal(row.detailPanel.hidden,false);
  assert.equal(row.features.size,5);
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  assert.doesNotMatch(html,/\.fleet-grid\.compact \.fleet-card-fields\{display:none\}/);
});

test("fleet error arrays show camera labels and keep global stop available for unknown lanes",async()=>{
  const view=fixture();await view.app.refresh();
  view.data.analysis={state:"error",camera_ids:[],errors:[{camera_id:"camera02",message:"无法确认此路分析状态"}]};
  view.errors.set(view.data.cameras[1].status_url,"offline");await view.app.refresh();
  assert.match(view.app.ui["analysis-state"].textContent,/摄像头 2：无法确认此路分析状态/);
  assert.doesNotMatch(view.app.ui["analysis-state"].textContent,/\[object Object\]/);
  assert.equal(view.app.ui["analysis-stop"].disabled,false);assert.equal(view.app.ui["analysis-start"].disabled,true);
  assert.equal(view.app.rows.get("camera01").start.disabled,true);
  view.errors.delete(view.data.cameras[1].status_url);assert.equal(await view.app.analysis("stop"),true);
  assert.equal(view.app.fleet.analysis.state,"idle");assert.equal(view.app.rows.get("camera01").start.disabled,false);
});

test("prepare failure with known idle previews permits key and preference corrections while retaining stop recovery",async()=>{
  const view=fixture();await view.app.refresh();const row=view.app.rows.get("camera01");
  row.input.value="rtsp://camera/live";await view.app.start(row);await view.app.toggleDetail(row);
  assert.equal(view.app.extraBusy(),false,"preview-only video must not lock shared model/key settings");
  view.data.analysis={state:"error",camera_ids:[],errors:[{camera_id:"camera01",message:"请先保存 API key"}]};
  await view.app.refresh();assert.equal(view.app.analysisLocked(),false);assert.equal(view.app.extraBusy(),false);
  assert.equal(view.app.ui["analysis-stop"].disabled,false);assert.equal(view.app.ui["analysis-start"].disabled,false);
  assert.equal(row.features.get("fight").disabled,false);assert.equal(row.stop.disabled,false);
  row.features.get("fight").checked=false;row.features.get("fight").onchange();assert.equal(await view.app.savePreferences(row),true);
  view.data.analysis.camera_ids=["camera01"];await view.app.refresh();assert.equal(view.app.analysisLocked(),true);
  assert.equal(row.features.get("fight").disabled,true);assert.equal(view.app.ui["analysis-stop"].disabled,false);
});
