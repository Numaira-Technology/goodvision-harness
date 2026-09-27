const assert=require("node:assert/strict");
const {test}=require("node:test");
const {CloudGPUConsole}=require("../src/goodvision_harness/cloud_gpu_console.js");

function fixture(){
  const elements=new Map(),requests=[];
  const document={getElementById(id){if(!elements.has(id))elements.set(id,{value:"",textContent:""});return elements.get(id);}};
  let data={available:true,capacity:37,state:"stopped",settings:{gpu_name:"RTX_4090",gpu_count:1,streams:37,disk_gb:50}};
  const fetch=async(url,options)=>{
    const payload=options.body&&JSON.parse(options.body);requests.push({url,method:options.method,payload});
    if(options.method==="PUT")data={...data,settings:payload};
    if(url.endsWith("/start"))data={...data,state:"starting"};
    if(url.endsWith("/stop"))data={...data,state:"stopped"};
    return {ok:true,json:async()=>data};
  };
  const app=new CloudGPUConsole({document,fetch,setTimeout:()=>1,clearTimeout:()=>{},AbortController});
  return {app,requests,setData:value=>{data=value;}};
}

test("five A100 cards and N lanes are saved before explicit rental start",async()=>{
  const {app,requests}=fixture();await app.refresh();
  assert.equal(requests.length,1);assert.equal(app.ui.streams.value,"37");
  app.ui.name.value="A100";app.ui.count.value="5";app.ui.count.oninput();
  await app.action("start");
  assert.deepEqual(requests.slice(1),[
    {url:"/v1/cloud-gpu",method:"PUT",payload:{gpu_name:"A100",gpu_count:5,streams:37,disk_gb:50}},
    {url:"/v1/cloud-gpu/start",method:"POST",payload:{}}
  ]);
  assert.equal(app.ui.start.disabled,true);assert.equal(app.ui.stop.disabled,false);
  assert.equal(app.ui.script.disabled,true);
  await app.action("stop");assert.equal(requests.at(-1).url,"/v1/cloud-gpu/stop");
});

test("editable rental parameters survive polling; invalid JSON cannot rent",async()=>{
  const {app,requests}=fixture();await app.refresh();
  app.ui.script.value=JSON.stringify({gpu_name:"A100_SXM4",gpu_count:5,streams:37,disk_gb:100});
  app.ui.script.oninput();await app.refresh();assert.match(app.ui.script.value,/A100_SXM4/);
  app.ui.script.onchange();assert.equal(app.ui.count.value,"5");
  await app.action("save");assert.equal(requests.at(-1).payload.disk_gb,100);
  assert.equal(requests.some(r=>r.url.endsWith("/start")),false);
  for(const value of ["invalid",'{"gpu_count":5}',JSON.stringify({gpu_name:"A100",gpu_count:0,streams:37,disk_gb:50})]){
    app.ui.script.value=value;const before=requests.length;await app.action("start");assert.equal(requests.length,before);
  }
});

test("uncertain cleanup and unavailable controllers block the start button",async()=>{
  const {app,setData}=fixture();await app.refresh();
  setData({...app.data,state:"cleanup_required",instance_id:42});await app.refresh();
  assert.equal(app.ui.start.disabled,true);assert.match(app.ui.status.textContent,/42/);
  setData({available:false,capacity:37});await app.refresh();
  assert.equal(app.ui.start.disabled,true);assert.match(app.ui.status.textContent,/start_cloud_gpu/);
});
