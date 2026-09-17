/* Real dashboard + real models, with a user-supplied video replacing only the
 * camera device. Keep reports and screenshots in an ignored private directory.
 * node tools/evaluate_camera_live.cjs --video path.mp4 --output private-dir
 *   --seconds 50 --url http://127.0.0.1:8080
 * Optional private motion replay trace: --trace-motion true --trace-motion-seconds 90
 *   --trace-motion-max-mib 128 (disabled by default; never ordinary service logs).
 */
"use strict";
const fs=require("node:fs"),path=require("node:path"),http=require("node:http"),crypto=require("node:crypto"),assert=require("node:assert/strict");
const {chromium}=require("playwright");
const {installMotionTraceCollector,createMotionTraceWriter}=require("./motion_trace_recorder.cjs");
const args={};for(let i=2;i<process.argv.length;i+=2){assert.ok(process.argv[i+1]);args[process.argv[i].replace(/^--/,"")]=process.argv[i+1];}
assert.ok(args.video&&args.output,"--video and --output required");
const videoPath=path.resolve(args.video),out=path.resolve(args.output),base=args.url||"http://127.0.0.1:8080";
const duration=Number(args.seconds||50);assert.ok(duration>=10&&duration<=900);
const traceMotion=args["trace-motion"]==="true",traceSeconds=Number(args["trace-motion-seconds"]||Math.min(duration,120)),
  traceBytes=Number(args["trace-motion-max-mib"]||128)*1024**2;
if(traceMotion){assert.ok(traceSeconds>=1&&traceSeconds<=120);assert.ok(traceBytes>=1024&&traceBytes<=128*1024**2);}
fs.mkdirSync(out,{recursive:true});const size=fs.statSync(videoPath).size;
const report={kind:"recorded_video_through_real_dashboard_and_models",video:videoPath,duration_seconds:duration,
  limitation:"Replaces camera getUserMedia only; not sensor exposure timing or manually labelled contour truth.",started_at:new Date().toISOString(),reports:[],status:[],errors:[],ws:[],http_failures:[]};
const save=()=>fs.writeFileSync(path.join(out,"report.json"),JSON.stringify(report,null,2));
const server=http.createServer((req,res)=>{
  res.setHeader("Access-Control-Allow-Origin","*");
  if(req.url!=="/video"){res.writeHead(404).end();return;}
  const match=/^bytes=(\d+)-(\d*)$/.exec(req.headers.range||"");
  const start=match?Number(match[1]):0,end=match&&match[2]?Math.min(size-1,Number(match[2])):size-1;
  if(start>=size||end<start){res.writeHead(416).end();return;}
  res.writeHead(match?206:200,{"Content-Type":"video/mp4","Content-Length":end-start+1,"Accept-Ranges":"bytes",...(match?{"Content-Range":`bytes ${start}-${end}/${size}`}:{})});
  const stream=fs.createReadStream(videoPath,{start,end});stream.pipe(res);res.on("close",()=>stream.destroy());
});
async function status(){try{const response=await fetch(base+"/v1/status",{signal:AbortSignal.timeout(3500)});const s=await response.json();
  const {pipeline_state,stream_session_id,source_frames_received,analysis_frames_published,last_error,scene_drain,pipeline,tracking,rich_labeling,processing,scene,face,local_camera_display}=s;
  report.status.push({at:Date.now(),pipeline_state,stream_session_id,source_frames_received,analysis_frames_published,last_error,scene_drain,pipeline,tracking,rich_labeling,processing,scene,face,local_camera_display});
  return s;
}catch(error){report.errors.push({kind:"status",at:Date.now(),message:error.message});return null;}}
(async()=>{
 await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
 const videoUrl=`http://127.0.0.1:${server.address().port}/video`;
 const browser=await chromium.launch({executablePath:args.chrome||"C:/Program Files/Google/Chrome/Application/chrome.exe",headless:true,
   args:["--autoplay-policy=no-user-gesture-required","--disable-background-timer-throttling","--disable-renderer-backgrounding","--disable-backgrounding-occluded-windows"]});
 let page,active=false;
 let traceWriter=null,traceStatus=null,traceFailed=false,traceAssets={};
 async function collectTrace(){
   if(!traceWriter||!page||traceFailed)return;
   try{
     const batch=await page.evaluate(()=>globalThis.__drainMotionTrace());
     const failure=traceWriter.append(batch);traceStatus=batch.status;
     if(failure)throw new Error(failure);
   }catch(_){
     traceFailed=true;traceStatus={complete:false,reason:"trace_storage_failure"};
     await page.evaluate(()=>globalThis.__stopMotionTrace("trace_storage_failure")).catch(()=>{});
   }
 }
 try{
  page=await browser.newPage({viewport:{width:1440,height:1050}});
  page.on("pageerror",e=>report.errors.push({kind:"page",at:Date.now(),message:e.message}));
  page.on("requestfailed",r=>{if(!r.url().includes("/video"))report.http_failures.push({at:Date.now(),path:new URL(r.url()).pathname,error:r.failure()?.errorText});});
  page.on("websocket",ws=>ws.on("framereceived",({payload})=>{
    if(typeof payload!=="string")return;let msg;try{msg=JSON.parse(payload);}catch{return;}
    if(msg.type==="camera_display_metadata"){
      report.ws.push({at:Date.now(),...msg});
    }else if(["session","error","ended"].includes(msg.type)){
      report.ws.push({at:Date.now(),type:msg.type,stream_session_id:msg.stream_session_id,message:msg.type==="error"?msg.message:undefined,local_camera_display:msg.local_camera_display});
    }
  }));
  await page.addInitScript(({videoUrl})=>{
    Object.defineProperty(navigator.mediaDevices,"getUserMedia",{value:async()=>{
      const video=document.createElement("video");video.crossOrigin="anonymous";video.muted=true;video.playsInline=true;video.loop=true;video.src=videoUrl;
      Object.assign(video.style,{position:"fixed",width:"2px",height:"2px",bottom:"0",right:"0",pointerEvents:"none"});document.body.append(video);
      await video.play();window.__liveEvaluationSource=video;
      const stream=video.captureStream();if(!stream.getVideoTracks().length)await new Promise(resolve=>stream.addEventListener("addtrack",resolve,{once:true}));
      return stream;
    }});
  },{videoUrl});
  await page.goto(base,{waitUntil:"domcontentloaded"});
  await page.waitForFunction(()=>typeof state!=="undefined"&&state.info&&document.querySelector("#camera-mode"),{},{timeout:30000});
  if(traceMotion){
    let assetBytes=0;
    for(const asset of ["camera_display.js","camera_display_dense.js","opencv-4.13.0.js"]){
      try{
        const response=await fetch(base+"/assets/"+asset,{signal:AbortSignal.timeout(15000)});
        if(!response.ok)throw new Error("asset unavailable");
        const bytes=Buffer.from(await response.arrayBuffer()),file="motion-trace-"+asset;
        if(assetBytes+bytes.length>traceBytes-65536)throw new Error("asset budget exceeded");
        fs.writeFileSync(path.join(out,file),bytes);assetBytes+=bytes.length;
        traceAssets[asset]={file,bytes:bytes.length,sha256:crypto.createHash("sha256").update(bytes).digest("hex")};
      }catch(_){traceAssets[asset]={error:"asset_snapshot_unavailable"};}
    }
    const remainingBytes=Math.max(1024,traceBytes-assetBytes-65536);
    traceWriter=createMotionTraceWriter(out,{maxBytes:remainingBytes});
    await page.evaluate(installMotionTraceCollector,{maxBytes:remainingBytes,maxSeconds:traceSeconds});
  }
  await page.locator("#camera-mode").click();await page.locator("#camera-button").click();
  await page.waitForFunction(()=>document.querySelector("#camera-preview").videoWidth>0,{},{timeout:30000});
  await page.locator("#dry-run").uncheck();
  // Keep all configured event features and Face enabled; no stored defaults are changed.
  await page.locator("#feature-select input[type=checkbox]").evaluateAll(inputs=>inputs.forEach(input=>{input.checked=true;input.dispatchEvent(new Event("change",{bubbles:true}));}));
  await page.locator("#start-button").click();active=true;
  await page.waitForFunction(()=>typeof localCameraDisplay!=="undefined"&&localCameraDisplay.controller,{},{timeout:60000});
  report.config=await page.evaluate(()=>({config:localCameraDisplay.controller.config,settings:localCameraDisplay.controller.settings,features:selectedFeatures(),dry_run:state.dryRun,source:{width:ui.cameraPreview.videoWidth,height:ui.cameraPreview.videoHeight},assets:[...document.scripts].map(s=>s.src).filter(Boolean)}));
  const started=Date.now();report.capture_started_at=new Date(started).toISOString();let lastStatus=-10000,lastShot=-15000,lastContourShot=-15000;
  while(Date.now()-started<duration*1000){
    await new Promise(resolve=>setTimeout(resolve,1000));
    const snapshot=await page.evaluate(()=>{const c=ui.localCameraCanvas;return {stats:localCameraDisplay.controller?.snapshot()||localCameraDisplay.lastStats,note:ui.cameraNote.textContent,visibility:document.visibilityState,sourceTime:window.__liveEvaluationSource.currentTime,display:{width:c.width,height:c.height,frameSequence:c.dataset.frameSequence,mediaTimestamp:c.dataset.mediaTimestamp,contourCoverage:c.dataset.contourCoverage}};});
    report.reports.push({at:Date.now(),...snapshot});
    await collectTrace();
    if(Date.now()-lastStatus>=10000){await status();lastStatus=Date.now();}
    if(Date.now()-lastShot>=15000){await page.locator("#camera-display").screenshot({path:path.join(out,`display-${Math.round((Date.now()-started)/1000)}s.png`),timeout:5000}).catch(e=>report.errors.push({kind:"screenshot",message:e.message}));lastShot=Date.now();}
    if(Date.now()-lastContourShot>=15000&&["anchored","interpolated"].includes(snapshot.display.contourCoverage)){
      const painted=await page.evaluate(()=>{const c=ui.localCameraCanvas;return ["anchored","interpolated"].includes(c.dataset.contourCoverage)?{sequence:c.dataset.frameSequence,timestamp:c.dataset.mediaTimestamp,jpeg:c.toDataURL("image/jpeg",.92)}:null;});
      if(painted){fs.writeFileSync(path.join(out,`contour-source-${painted.sequence}.jpg`),Buffer.from(painted.jpeg.split(",")[1],"base64"));lastContourShot=Date.now();}
    }
    if(report.reports.length%5===0){save();const s=snapshot.stats;console.log(JSON.stringify({elapsed:Math.round((Date.now()-started)/1000),fps:s?.distinct_presented_fps,contours:s?.complete_contour_frames,continuous:s?.continuous_contour_frames,interpolated:s?.approximate_contour_frames,metadata:s?.metadata_received,pending:s?.analysis_pending,latency:s?.last_source_to_present_seconds}));}
  }
  report.before_stop=await page.evaluate(()=>localCameraDisplay.controller.snapshot());
  report.capture_stop_requested_at=new Date().toISOString();
  await page.locator("#stop-button").click();
  await page.waitForFunction(()=>!localCameraDisplay.controller,{},{timeout:30000});active=false;
  report.capture_stopped_at=new Date().toISOString();
  report.after_stop=await page.evaluate(()=>localCameraDisplay.lastStats);
  await collectTrace();
  const drainStart=Date.now();while(Date.now()-drainStart<120000){const s=await status();save();if(s&&s.pipeline_state==="idle"){report.drain_seconds=(Date.now()-drainStart)/1000;break;}await new Promise(resolve=>setTimeout(resolve,2000));}
  report.finished_at=new Date().toISOString();save();console.log(JSON.stringify({report:path.join(out,"report.json"),complete:true}));
 }catch(error){report.errors.push({kind:"fatal",message:error.stack});save();throw error;}
 finally{
   if(active&&page)await page.evaluate(()=>typeof stop==="function"?stop():undefined).catch(()=>{});
   if(traceWriter){
     await collectTrace().catch(()=>{traceStatus={complete:false,reason:"collector_unavailable"};});
     try{report.motion_trace=traceWriter.finish({status:traceStatus,config:report.config?.config,
       provenance:{source_pixels:"browser_preprocess_exact",ordered_worker_input:true,source_assets:traceAssets,
         note:"Optional diagnostic run; trace overhead is included in its normal FPS measurements."}});}
     catch(_){report.motion_trace={complete:false,reason:"trace_storage_failure"};}
   }
   await browser.close();server.close();save();
 }
})().catch(e=>{console.error(e);process.exitCode=1;server.close();});
