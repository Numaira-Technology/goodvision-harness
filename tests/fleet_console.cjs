const assert=require("node:assert/strict");
const {test}=require("node:test");
const fs=require("node:fs");
const path=require("node:path");
const {FleetConsole,normalizeFleet}=require("../src/goodvision_harness/fleet_console.js");

class Element{
  constructor(tag="div"){this.tagName=tag;this.children=[];this.hidden=false;this.disabled=false;this.value="";this.dataset={};this.attributes={};}
  append(...children){for(const child of children){child.parent=this;this.children.push(child);}}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);}
  setAttribute(key,value){this.attributes[key]=value;}
  removeAttribute(key){delete this.attributes[key];delete this[key];}
  focus(){this.focused=true;}
  scrollIntoView(){}
}
function fixture({count=3,allowed=true}={}){
  const elements=new Map();const document={hidden:false,createElement:tag=>new Element(tag),getElementById:id=>{
    if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
  }};
  const data={capacity:count,multi_camera_allowed:allowed,reason:allowed?"":"未启用本地人物追踪",cameras:Array.from({length:count},(_,i)=>{
    const id="camera"+String(i+1).padStart(2,"0"),base="/v1/fleet/cameras/"+id;
    return {id,label:"摄像头 "+(i+1),own:i===0,enabled:allowed||i===0,status_url:base+"/status",start_url:base+"/start",stop_url:base+"/stop",settings_url:base+"/settings",preview_url:base+"/video/annotated/latest.jpg"};
  })};
  const requests=[],statuses=new Map(data.cameras.map(camera=>[camera.id,{pipeline_state:"idle",configured_stream:false}]));
  const errors=new Map(),revoked=[],scheduled=[];let blobId=0;
  const fetch=async(url,options={})=>{
    requests.push({url,method:options.method||"GET",body:options.body&&JSON.parse(options.body)});
    if(errors.has(url))throw new Error(errors.get(url));
    if(url==="/v1/fleet")return {ok:true,json:async()=>data};
    const camera=data.cameras.find(camera=>url.startsWith("/v1/fleet/cameras/"+camera.id+"/"));
    if(!camera)throw new Error("Unknown URL "+url);
    if(url===camera.preview_url)return {ok:true,blob:async()=>({camera:camera.id})};
    if(options.method==="POST")statuses.set(camera.id,{...statuses.get(camera.id),pipeline_state:url.endsWith("/stop")?"idle":"running",stream_session_id:camera.id,ready:true});
    return {ok:true,json:async()=>({...statuses.get(camera.id)})};
  };
  const app=new FleetConsole({document,fetch,clock:()=>100,setTimeout:fn=>{scheduled.push(fn);return scheduled.length;},clearTimeout:()=>{},
    AbortController,URL:{createObjectURL:()=>"blob:frame-"+(++blobId),revokeObjectURL:value=>revoked.push(value)}});
  return {app,data,document,requests,statuses,errors,revoked,scheduled};
}

test("cloud mode permits one camera while local tracking exposes arbitrary provisioned capacity",async()=>{
  const cloud=fixture({count:16,allowed:false});await cloud.app.refresh();
  assert.equal(cloud.app.visible.size,1);assert.equal(cloud.app.ui.add.hidden,true);
  cloud.app.add();assert.equal(cloud.app.visible.size,1);
  for(const count of [1,2,5,12,16,24]){
    const view=fixture({count});await view.app.refresh();
    for(let i=1;i<count;i++)view.app.add();
    assert.equal(view.app.visible.size,count);assert.equal(view.app.ui.add.disabled,true);
    assert.match(view.app.ui.capacity.textContent,new RegExp("已部署 "+count+" 路"));
  }
});

test("one camera connects with its own URL and settings without touching another camera",async()=>{
  const view=fixture();await view.app.refresh();view.app.add();
  const row=view.app.rows.get("camera02"),first=view.app.rows.get("camera01");
  row.input.value="rtsp://private-user:private-password@camera-two/live";row.layout.value="stereo";
  assert.equal(await view.app.start(row),true);
  const posts=view.requests.filter(request=>request.method==="POST");
  assert.equal(posts.length,1);assert.match(posts[0].url,/camera02\/start$/);
  assert.equal(posts[0].body.stream_url,row.input.value);assert.equal(posts[0].body.camera_layout,"stereo");
  assert.equal(Object.hasOwn(posts[0].body,"dry_run"),false,"saved no-model test mode must not be overridden");assert.equal(posts[0].body.features.length,5);
  assert.equal(first.status.pipeline_state,"idle");assert.equal(row.status.pipeline_state,"running");
  assert.equal(row.remove.disabled,true);view.app.remove(row);assert.equal(view.app.visible.has("camera02"),true);
  assert.equal(await view.app.stop(row),true);assert.equal(row.status.pipeline_state,"idle");
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
  const view=fixture({count:16});view.statuses.set("camera16",{pipeline_state:"running",stream_session_id:"active16"});await view.app.refresh();
  assert.equal(view.app.visible.has("camera16"),true);assert.equal(view.app.visible.size,2);
  view.requests.length=0;await view.app.previews();
  assert.equal(view.requests.length,1);assert.match(view.requests[0].url,/camera16\/video\/annotated\/latest.jpg$/);
  const row=view.app.rows.get("camera16");assert.equal(row.image.hidden,false);
  view.statuses.set("camera16",{pipeline_state:"idle"});await view.app.pollStatus();
  assert.equal(row.image.hidden,true);assert.equal(row.previewUrl,null);assert.equal(view.revoked.length,1);
});

test("16 preview tiles use finite JPEG responses with only three in flight and pause in hidden tabs",async()=>{
  const view=fixture({count:16});for(const camera of view.data.cameras)view.statuses.set(camera.id,{pipeline_state:"running"});await view.app.refresh();
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
  view.data.multi_camera_allowed=false;view.data.cameras[1].enabled=false;view.data.cameras[2].enabled=false;
  await view.app.refresh();assert.equal(view.app.visible.size,1);assert.equal(view.app.ui.add.hidden,true);
  const normalized=normalizeFleet({multi_camera_allowed:true,cameras:[{id:"unsafe",preview_url:"//evil.example/video",start_url:"https://evil.example/start",status_url:"/\\evil.example/status"}]});
  assert.equal(normalized.cameras[0].preview_url,null);assert.equal(normalized.cameras[0].start_url,null);assert.equal(normalized.cameras[0].status_url,null);
});

test("dashboard retains original controls and loads the standalone fleet controller",()=>{
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  assert.match(html,/id="fleet-detail-console" hidden/);assert.match(html,/id="start-button"/);
  assert.match(html,/<script src="\/assets\/fleet_console.js"><\/script>/);
  assert.match(html,/GoodvisionFleet\?\.refresh\?\.\(\)/);
  assert.match(html,/GoodvisionFleet\?\.detailHidden\?\.\(\)/);
});

test("remote mode discovers surviving peer sessions and keeps stop controls available after primary restart",async()=>{
  const view=fixture({count:16,allowed:false});
  view.statuses.set("camera16",{pipeline_state:"running",stream_session_id:"surviving-peer",ready:true});
  await view.app.refresh();
  const row=view.app.rows.get("camera16");
  assert.equal(view.app.visible.has("camera16"),true);
  assert.equal(row.start.disabled,true);assert.equal(row.stop.disabled,false);
  assert.equal(view.app.ui["stop-all"].hidden,false);assert.equal(view.app.extraBusy(),true);
  assert.match(view.app.ui.capacity.textContent,/历史会话/);
  view.requests.length=0;await view.app.pollStatus();
  assert.equal(view.requests.length,2,"routine polls cover primary plus surviving peer, not all idle peers");
  await view.app.all("stop");assert.equal(row.status.pipeline_state,"idle");
  assert.equal(view.app.visible.has("camera16"),false);assert.equal(view.app.ui["stop-all"].hidden,true);
});

test("compact monitoring view can be expanded and adding a channel reveals its connection fields",async()=>{
  const view=fixture({count:5});
  for(const camera of view.data.cameras.slice(0,4))view.statuses.set(camera.id,{pipeline_state:"running"});
  await view.app.refresh();assert.equal(view.app.compact,true);assert.match(view.app.ui.grid.className,/compact/);
  view.app.ui["view-toggle"].onclick();assert.equal(view.app.compact,false);
  view.app.ui["view-toggle"].onclick();assert.equal(view.app.compact,true);
  view.app.add();assert.equal(view.app.compact,false);assert.equal(view.app.visible.has("camera05"),true);
});
