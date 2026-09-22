/* One complete real-time video through the production browser renderer and models.
 * A same-origin diagnostic page replaces camera acquisition with a local video.
 * Keep output private: metadata may contain names and model observations.
 * NODE_PATH=<playwright modules> node tools/evaluate_pipeline_latency.cjs
 *   --video sample.mp4 --video-url http://127.0.0.1:18094/experiment-video/original
 *   --output private-dir --url http://127.0.0.1:18094
 * Use an isolated service: this starts/stops a real analysis session and calls paid models.
 * The video URL must serve the given file with byte-range support on the service origin.
 */
"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
const {chromium}=require("playwright");
const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].slice(2)]=process.argv[i+1];
assert.ok(args.video&&args.output&&args.url&&args["video-url"],"Required: --video --video-url --output --url");
assert.equal(new URL(args["video-url"]).origin,new URL(args.url).origin,"Video must be served on the service origin");
const video=path.resolve(args.video),out=path.resolve(args.output);
assert.ok(fs.statSync(video).isFile(),"Video file must exist");
assert.ok(!fs.existsSync(path.join(out,"browser-report.json")),"Use a fresh output directory for each trial");
fs.mkdirSync(out,{recursive:true});
const log=(name,row)=>fs.appendFileSync(path.join(out,name+".jsonl"),JSON.stringify(row)+"\n");
const report={kind:"full_video_production_browser_controller_real_models",started_at:new Date().toISOString(),errors:[],snapshots:[]};
const save=()=>fs.writeFileSync(path.join(out,"browser-report.json"),JSON.stringify(report,null,2));
(async()=>{
  const browser=await chromium.launch({executablePath:args.chrome||"C:/Program Files/Google/Chrome/Application/chrome.exe",headless:true,
    args:["--autoplay-policy=no-user-gesture-required","--disable-background-timer-throttling","--disable-renderer-backgrounding","--disable-backgrounding-occluded-windows"]});
  let page,active=false,readers=[],abort=new AbortController();
  async function receive(name,route){
    try{
      const res=await fetch(args.url+route,{signal:abort.signal});assert.ok(res.ok);
      let buffer="";const decoder=new TextDecoder();for await(const chunk of res.body){
        buffer+=decoder.decode(chunk,{stream:true});
        let end;while((end=buffer.indexOf("\n\n"))>=0){
          const block=buffer.slice(0,end);buffer=buffer.slice(end+2);
          const raw=block.split("\n").filter(s=>s.startsWith("data:")).map(s=>s.slice(5).trim()).join("\n");
          if(raw)log(name,{at:Date.now(),payload:JSON.parse(raw)});
        }
      }
    }catch(e){if(!abort.signal.aborted)report.errors.push({kind:name,message:e.message});}
  }
  try{
    page=await browser.newPage({viewport:{width:1440,height:1050}});
    await page.context().grantPermissions(["local-network-access"],{origin:args.url});
    page.on("pageerror",e=>report.errors.push({kind:"page",message:e.message}));
    page.on("console",e=>{if(e.type()==="error")report.errors.push({kind:"console",message:e.text()});});
    page.on("requestfailed",r=>{if(r.failure()?.errorText!=="net::ERR_ABORTED")report.errors.push({kind:"request",path:new URL(r.url()).pathname,message:r.failure()?.errorText});});
    page.on("websocket",ws=>ws.on("framereceived",({payload})=>{
      if(typeof payload!=="string")return;
      try{const m=JSON.parse(payload);if(["camera_display_metadata","session","error","ended"].includes(m.type))log("ws",{at:Date.now(),payload:m});}catch{}
    }));
    await page.route("**/pipeline-latency-test",route=>route.fulfill({contentType:"text/html",body:
      '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#101827;color:white"><h2>Full pipeline latency test</h2><video id="source" muted playsinline crossorigin="anonymous" style="position:fixed;width:2px;height:2px"></video><canvas id="camera-display" style="width:1280px;max-width:100%"></canvas><script src="/assets/camera_display.js"></script></body></html>'}));
    // Keep the diagnostic video same-origin, including modern loopback access rules.
    await page.goto(args.url+"/pipeline-latency-test",{waitUntil:"load"});
    await page.evaluate(async url=>{
      window.__experimentVideo=document.querySelector("#source");
      __experimentVideo.src=url;
      await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Video readiness timed out; state='+__experimentVideo.readyState)),20000);
        __experimentVideo.onloadeddata=()=>{clearTimeout(timer);resolve();};__experimentVideo.onerror=()=>{clearTimeout(timer);reject(new Error("Video decode failed: "+__experimentVideo.error?.message));};});
    },args["video-url"]);
    // Instrument the unmodified presentation callbacks; no model/geometry replacement.
    await page.evaluate(()=>{
      const Original=GoodvisionCameraDisplay.BrowserController;window.__experimentRows=[];
      GoodvisionCameraDisplay.BrowserController=class extends Original{
        constructor(options){super(options);window.__experimentController=this;
          const push=this.frames.push.bind(this.frames),present=this.frames.notePresented.bind(this.frames);
          this.frames.push=frame=>{window.__experimentRows.push({kind:"capture",at:Date.now(),sequence:frame.sequence,pts:frame.timestamp,
            video_pts:window.__experimentVideo.currentTime});return push(frame);};
          this.frames.notePresented=(frame,now)=>{const o=this.overlays.get(frame.sequence),subjects=o?.subjects||[];
            window.__experimentRows.push({kind:"present",at:Date.now(),sequence:frame.sequence,pts:frame.timestamp,
              latency_seconds:(now-frame.capturedAt)/1000,coverage:o?.coverage||"not_ready",all_subjects_present:o?.all_subjects_present===true,
              exact:o?.exact===true,anchor_age_seconds:o?.anchor_age_seconds,subjects:subjects.map(s=>({track_id:s.track_id,
                complete:s.complete,approximate:s.approximate,method:s.method,alert:s.alert,warning:s.warning,
                display_name:this.identities.nameAt(s,frame.timestamp),event_labels:s.event_labels,anchor_sequence:s.anchor_sequence}))});
            return present(frame,now);
          };
        }
      };
    });
    if(args.preflight==="true"){
      report.camera_preflight=await page.evaluate(()=>({duration:__experimentVideo.duration,currentTime:__experimentVideo.currentTime,
        paused:__experimentVideo.paused,width:__experimentVideo.videoWidth,height:__experimentVideo.videoHeight}));
      console.log(JSON.stringify(report.camera_preflight));return;
    }
    active=true;
    await page.evaluate(async()=>{
      const info=await fetch('/v1/ui/info').then(r=>r.json());
      const ws=window.__experimentSocket=new WebSocket(location.origin.replace('http','ws')+'/ws');
      await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
      const session=await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(new Error('Session start timeout')),120000);
        ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='session'){clearTimeout(timer);resolve(m);}
          else if(m.type==='error'){clearTimeout(timer);reject(new Error(m.message));}};
        ws.send(JSON.stringify({type:'start',mode:'camera',analysis_mode:'multi',features:['fight','smoking','fall','weapon','stranger'],
          event_sensitivities:{fight:'standard',smoking:'high',fall:'standard',weapon:'standard'},dry_run:false}));
      });
      const config={...info.local_camera_display,stream_session_id:session.stream_session_id,generation:session.generation,crop_version:0};
      if(!config.enabled)throw new Error('Production browser contour path is disabled');
      const c=new GoodvisionCameraDisplay.BrowserController({video:__experimentVideo,canvas:document.querySelector('#camera-display'),config,
        settings:{max_image_side:config.analysis_max_image_side,jpeg_quality:config.analysis_jpeg_quality,sample_fps:config.analysis_sample_fps},socket:ws,
        onFailure:message=>__experimentRows.push({kind:'failure',at:Date.now(),message})});
      ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='camera_display_metadata')c.receiveMetadata(m,{transport:'websocket'});
        else if(m.type==='camera_frame_ack')c.acknowledgeUpload(m);
        else if(m.type==='error')__experimentRows.push({kind:'failure',at:Date.now(),message:m.message});};
      c.start();
    });
    report.config=await page.evaluate(()=>({config:__experimentController.config,settings:__experimentController.settings,video_duration:__experimentVideo.duration}));
    readers=[receive("vlm","/v1/vlm/results/events"),receive("results","/v1/results/events")];
    report.play_started_at=Date.now();
    await page.evaluate(async()=>{__experimentVideo.currentTime=0;await __experimentVideo.play();});
    let endedAt=null,lastStatus=0,lastShot=-20;
    const deadline=Date.now()+(report.config.video_duration+report.config.config.delay_seconds+45)*1000;
    while(Date.now()<deadline){
      await new Promise(r=>setTimeout(r,1000));
      const sample=await page.evaluate(()=>({at:Date.now(),source:__experimentVideo.currentTime,ended:__experimentVideo.ended,
        stats:__experimentController.snapshot(),rows:__experimentRows.splice(0)}));
      for(const row of sample.rows){log("frames",row);if(row.kind==="failure")report.errors.push(row);}delete sample.rows;report.snapshots.push(sample);
      if(sample.source-lastShot>=15){
        await page.locator("#camera-display").screenshot({path:path.join(out,`display-${Math.floor(sample.source)}s.png`),timeout:8000});lastShot=sample.source;
      }
      if(Date.now()-lastStatus>5000){
        const s=await fetch(args.url+"/v1/status?lightweight=true",{signal:AbortSignal.timeout(6000)}).then(r=>r.json());log("status",{at:Date.now(),payload:s});
        const f=await fetch(args.url+"/v1/faces/fast",{signal:AbortSignal.timeout(6000)}).then(r=>r.json());log("faces",{at:Date.now(),payload:f});
        console.log(JSON.stringify({source:sample.source,ended:sample.ended,presented:sample.stats.distinct_presented_frames,
          complete:sample.stats.complete_contour_frames,continuous:sample.stats.continuous_contour_frames,metadata:sample.stats.metadata_received}));lastStatus=Date.now();save();
      }
      if(sample.ended&&!endedAt){endedAt=Date.now();report.source_ended_at=endedAt;}
      if(endedAt&&Date.now()-endedAt>(report.config.config.delay_seconds+3)*1000&&sample.stats.buffered_frames===0)break;
    }
    assert.ok(endedAt,"The complete video must reach EOF");
    report.final=await page.evaluate(()=>__experimentController.snapshot());
    assert.equal(report.final.buffered_frames,0,"All buffered video frames must be presented");
    await page.locator("#camera-display").screenshot({path:path.join(out,"final-display.png"),timeout:10000});
    await page.evaluate(async()=>{await __experimentController.drain();__experimentSocket.send(JSON.stringify({type:'stop'}));__experimentController.stop();});active=false;
    for(let i=0;i<70;i++){
      const s=await fetch(args.url+"/v1/status?lightweight=true",{signal:AbortSignal.timeout(6000)}).then(r=>r.json());
      log("status",{at:Date.now(),payload:s});
      if(["idle","stopped"].includes(s.analysis_state||s.pipeline_state)&&!["running","pending"].includes(s.scene_drain?.state)){report.drained=true;break;}
      await new Promise(r=>setTimeout(r,2000));
    }
    report.completed=true;
  }catch(e){report.errors.push({kind:"fatal",message:e.stack});throw e;}
  finally{
    if(active&&page)await page.evaluate(()=>{window.__experimentController?.stop();if(window.__experimentSocket?.readyState===1)__experimentSocket.send(JSON.stringify({type:'stop'}));}).catch(()=>{});
    abort.abort();await Promise.allSettled(readers);await browser.close();report.finished_at=new Date().toISOString();save();
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
