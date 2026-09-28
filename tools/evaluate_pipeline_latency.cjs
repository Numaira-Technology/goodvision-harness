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
const crypto=require("node:crypto");
const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].slice(2)]=process.argv[i+1];
assert.ok(args.video&&args.output&&args.url&&args["video-url"],"Required: --video --video-url --output --url");
const analysisFeatures=args["analysis-features"]?args["analysis-features"].split(","):['fight','smoking','fall','weapon','stranger'];
const uploadWindow=Number(args["upload-window"]||1);
assert.ok([1,2,3,4].includes(uploadWindow),"--upload-window must be between 1 and 4");
const ackTimeoutMs=Number(args["ack-timeout-ms"]||5000);
const drainTimeoutMs=Number(args["drain-timeout-ms"]||10000);
assert.ok(Number.isInteger(ackTimeoutMs)&&ackTimeoutMs>=5000&&ackTimeoutMs<=30000,"--ack-timeout-ms must be 5000..30000");
assert.ok(Number.isInteger(drainTimeoutMs)&&drainTimeoutMs>=10000&&drainTimeoutMs<=60000,"--drain-timeout-ms must be 10000..60000");
assert.ok(analysisFeatures.length&&analysisFeatures.every(feature=>['fight','smoking','fall','weapon','stranger'].includes(feature))&&new Set(analysisFeatures).size===analysisFeatures.length,
  "--analysis-features must list unique supported features");
assert.equal(new URL(args["video-url"]).origin,new URL(args.url).origin,"Video must be served on the service origin");
const video=path.resolve(args.video),out=path.resolve(args.output);
assert.ok(fs.statSync(video).isFile(),"Video file must exist");
assert.ok(!fs.existsSync(path.join(out,"browser-report.json")),"Use a fresh output directory for each trial");
fs.mkdirSync(out,{recursive:true});
const log=(name,row)=>fs.appendFileSync(path.join(out,name+".jsonl"),JSON.stringify(row)+"\n");
const report={kind:args["camera-display-asset"]?"full_video_experimental_browser_controller_real_models":"full_video_production_browser_controller_real_models",started_at:new Date().toISOString(),errors:[],snapshots:[],
  video_sha256:crypto.createHash("sha256").update(fs.readFileSync(video)).digest("hex"),analysis_features:analysisFeatures};
if(args["camera-display-asset"]){
  const asset=fs.readFileSync(path.resolve(args["camera-display-asset"]));
  report.experimental_camera_display_sha256=crypto.createHash("sha256").update(asset).digest("hex");
}
const save=()=>fs.writeFileSync(path.join(out,"browser-report.json"),JSON.stringify(report,null,2));
const probe=async(name,url)=>{
  try{
    const response=await fetch(url,{signal:AbortSignal.timeout(6000)});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const payload=await response.json();log(name,{at:Date.now(),payload});return payload;
  }catch(error){
    const warning={at:Date.now(),probe:name,error_type:error.name||"Error"};
    (report.probe_warnings||(report.probe_warnings=[])).push(warning);
    log("probe_warnings",warning);return null;
  }
};
(async()=>{
  const platformChrome=process.platform==="darwin"?"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome":process.platform==="win32"?"C:/Program Files/Google/Chrome/Application/chrome.exe":undefined;
  const browser=await chromium.launch({executablePath:args.chrome||(platformChrome&&fs.existsSync(platformChrome)?platformChrome:undefined),headless:args.headful!=="true",
    args:["--autoplay-policy=no-user-gesture-required","--disable-background-timer-throttling","--disable-renderer-backgrounding","--disable-backgrounding-occluded-windows",
      ...(args["window-position"]?["--window-position="+args["window-position"]]:[])]});
  report.browser_version=browser.version();
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
    if(args["record-canvas"]==="true"){
      await page.exposeFunction("saveRecordingChunk",base64=>{
        fs.appendFileSync(path.join(out,"canvas-recording.webm"),Buffer.from(base64,"base64"));
      });
    }
    if(args["local-video"]==="true"){
      // Keep source playback local while all Harness HTTP/WS traffic uses the
      // forwarded cloud service. Support byte ranges without transcoding.
      await page.route(args["video-url"],async route=>{
        const size=fs.statSync(video).size,range=route.request().headers().range;
        // A full 160 MB response exceeds Chromium's DevTools pipe limit. Serve
        // bounded byte ranges so local playback never buffers the whole file in
        // a single Playwright route fulfillment.
        const maxChunk=8*1024*1024;
        let start=0,end=Math.min(size-1,maxChunk-1),status=206;
        if(range){
          const match=/^bytes=(\d+)-(\d*)$/.exec(range);
          if(!match||Number(match[1])>=size){await route.fulfill({status:416,headers:{"Content-Range":`bytes */${size}`}});return;}
          start=Number(match[1]);end=Math.min(match[2]?Number(match[2]):size-1,size-1,start+maxChunk-1);
          if(end<start){await route.fulfill({status:416});return;}
        }
        const handle=await fs.promises.open(video,"r"),body=Buffer.alloc(end-start+1);
        try{let read=0;while(read<body.length){const result=await handle.read(body,read,body.length-read,start+read);assert.ok(result.bytesRead);read+=result.bytesRead;}}
        finally{await handle.close();}
        await route.fulfill({status,body,headers:{"Content-Type":"video/mp4","Accept-Ranges":"bytes",
          "Content-Length":String(body.length),"Content-Range":`bytes ${start}-${end}/${size}`}});
      });
    }
    await page.context().grantPermissions(["local-network-access"],{origin:args.url});
    page.on("pageerror",e=>report.errors.push({kind:"page",message:e.message}));
    page.on("console",e=>{if(e.type()==="error")report.errors.push({kind:"console",message:e.text()});});
    page.on("requestfailed",r=>{if(r.failure()?.errorText!=="net::ERR_ABORTED")report.errors.push({kind:"request",path:new URL(r.url()).pathname,message:r.failure()?.errorText});});
    page.on("websocket",ws=>ws.on("framereceived",({payload})=>{
      if(typeof payload!=="string")return;
      try{const m=JSON.parse(payload);if(["camera_display_metadata","session","error","ended"].includes(m.type))log("ws",{at:Date.now(),payload:m});}catch{}
    }));
    if(args["camera-display-asset"]){
      const asset=fs.readFileSync(path.resolve(args["camera-display-asset"]));
      await page.route("**/assets/camera_display.js",route=>route.fulfill({contentType:"text/javascript",body:asset}));
    }
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
        noteTiming(name,milliseconds){
          super.noteTiming(name,milliseconds);
          if(name.startsWith('analysis_ack')||name.startsWith('analysis_server_')){
            const samples=window.__experimentTransportTimings||(window.__experimentTransportTimings=[]);
            if(samples.length<5000)samples.push({name,milliseconds,at:Date.now()});
          }
        }
        constructor(options){super(options);window.__experimentController=this;
          const push=this.frames.push.bind(this.frames),present=this.frames.notePresented.bind(this.frames);
          this.frames.push=frame=>{window.__experimentRows.push({kind:"capture",at:Date.now(),sequence:frame.sequence,pts:frame.timestamp,
            video_pts:window.__experimentVideo.currentTime});return push(frame);};
          this.frames.notePresented=(frame,now)=>{const o=this.overlays.get(frame.sequence),subjects=(o?.subjects||[]).filter(s=>s.target_sequence===frame.sequence);
            window.__experimentRows.push({kind:"present",at:Date.now(),sequence:frame.sequence,pts:frame.timestamp,
              latency_seconds:(now-frame.capturedAt)/1000,coverage:o?.coverage||"not_ready",all_subjects_present:o?.all_subjects_present===true,
              counted_complete:subjects.length>0&&o?.coverage==="anchored"&&subjects.every(s=>s.complete!==false),
              counted_continuous:o?.all_subjects_present===true&&subjects.length===o.subjects.length&&subjects.every(s=>s.polygons?.length&&(s.complete||s.approximate)),
              exact:o?.exact===true,anchor_age_seconds:o?.anchor_age_seconds,subjects:subjects.map(s=>({track_id:s.track_id,
                complete:s.complete,approximate:s.approximate,method:s.method,alert:s.alert,warning:s.warning,
                display_name:this.identities.nameAt(s,frame.timestamp),event_labels:s.event_labels,anchor_sequence:s.anchor_sequence}))});
            return present(frame,now);
          };
        }
      };
    });
    // Check the actual server policy before opening a real analysis session.
    // Playback alone cannot detect a disabled route or a silently changed buffer.
    const displayConfig=await page.evaluate(async()=>{
      const response=await fetch('/v1/ui/info');
      if(!response.ok)throw new Error('Cannot read production browser configuration');
      const config=(await response.json()).local_camera_display;
      if(!config?.enabled)throw new Error('Production browser contour path is disabled');
      if(!GoodvisionCameraDisplay.BrowserController.supported(__experimentVideo))
        throw new Error('Production browser contour APIs are unavailable');
      return config;
    });
    const benchmarkResponse=await fetch(args.url+"/benchmark/config");
    if(benchmarkResponse.ok){
      report.benchmark=await benchmarkResponse.json();
      const expectedDelay=report.benchmark.delay??report.benchmark.fixed_main?.effective_display_delay_seconds;
      assert.equal(displayConfig.delay_seconds,expectedDelay,"Browser display delay differs from the benchmark job");
    }
    report.display_preflight=displayConfig;
    if(args["warmup-motion"]==="true"){
      const started=Date.now();
      const ready=await page.evaluate(()=>Promise.race([
        GoodvisionCameraDisplay.BrowserController.warmupMotion(),
        new Promise(resolve=>setTimeout(()=>resolve(false),60000))
      ]));
      assert.ok(ready,"Browser contour motion worker did not warm up before playback");
      report.motion_pre_warm_seconds=(Date.now()-started)/1000;
    }
    if(args.preflight==="true"){
      await page.evaluate(async config=>{
        const Controller=GoodvisionCameraDisplay.BrowserController;
        const ready=await Promise.race([Controller.warmupMotion(),new Promise(resolve=>setTimeout(()=>resolve(false),20000))]);
        if(!ready)throw new Error('Production contour motion worker failed to initialize');
        // Open the real transport but never send "start" or upload video frames.
        const socket=new WebSocket(location.origin.replace('http','ws')+'/ws');
        try{
          await new Promise((resolve,reject)=>{
            const timer=setTimeout(()=>reject(new Error('Preflight WebSocket timeout')),10000);
            socket.onerror=()=>{clearTimeout(timer);reject(new Error('Preflight WebSocket failed'));};
            socket.onmessage=e=>{if(JSON.parse(e.data).type==='hello'){clearTimeout(timer);resolve();}};
          });
          const controller=new Controller({video:__experimentVideo,canvas:document.querySelector('#camera-display'),config,
            settings:{max_image_side:config.analysis_max_image_side,jpeg_quality:config.analysis_jpeg_quality,sample_fps:config.analysis_sample_fps},socket});
          controller.stop();
        }finally{
          await new Promise(resolve=>{socket.onclose=resolve;socket.close();setTimeout(resolve,2000);});
          Controller.discardWarmup();
        }
      },displayConfig);
      report.camera_preflight=await page.evaluate(()=>({duration:__experimentVideo.duration,currentTime:__experimentVideo.currentTime,
        paused:__experimentVideo.paused,width:__experimentVideo.videoWidth,height:__experimentVideo.videoHeight}));
      report.kind="video_and_contour_startup_preflight_no_analysis";
      assert.equal(report.errors.length,0,"Browser startup recorded errors; inspect browser-report.json");
      report.preflight=true;console.log(JSON.stringify({video:report.camera_preflight,display:displayConfig}));return;
    }
    active=true;
    await page.evaluate(async ({displayConfig,analysisFeatures,uploadWindow,ackTimeoutMs,drainTimeoutMs})=>{
      const ws=window.__experimentSocket=new WebSocket(location.origin.replace('http','ws')+'/ws');
      await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
      const session=await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(new Error('Session start timeout')),120000);
        ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='session'){clearTimeout(timer);resolve(m);}
          else if(m.type==='error'){clearTimeout(timer);reject(new Error(m.message));}};
        const sensitivities={fight:'standard',smoking:'high',fall:'standard',weapon:'standard'};
        ws.send(JSON.stringify({type:'start',mode:'camera',analysis_mode:'multi',features:analysisFeatures,
          event_sensitivities:Object.fromEntries(Object.entries(sensitivities).filter(([feature])=>analysisFeatures.includes(feature))),dry_run:false}));
      });
      const config={...displayConfig,stream_session_id:session.stream_session_id,generation:session.generation,
        crop_version:0,analysis_upload_window:uploadWindow,
        analysis_ack_timeout_ms:ackTimeoutMs,analysis_drain_timeout_ms:drainTimeoutMs};
      const c=new GoodvisionCameraDisplay.BrowserController({video:__experimentVideo,canvas:document.querySelector('#camera-display'),config,
        settings:{max_image_side:config.analysis_max_image_side,jpeg_quality:config.analysis_jpeg_quality,sample_fps:config.analysis_sample_fps},socket:ws,
        onFailure:message=>__experimentRows.push({kind:'failure',at:Date.now(),message})});
      ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type==='camera_display_metadata')c.receiveMetadata(m,{transport:'websocket'});
        else if(m.type==='camera_frame_ack')c.acknowledgeUpload(m);
        else if(m.type==='error')__experimentRows.push({kind:'failure',at:Date.now(),message:m.message});};
    },{displayConfig,analysisFeatures,uploadWindow,ackTimeoutMs,drainTimeoutMs});
    report.config=await page.evaluate(()=>({config:__experimentController.config,settings:__experimentController.settings,video_duration:__experimentVideo.duration}));
    readers=[receive("results","/v1/results/events")];
    if(analysisFeatures.some(feature=>['fight','smoking','fall','weapon'].includes(feature)))
      readers.push(receive("vlm","/v1/vlm/results/events"));
    if(args["sync-dir"]){
      const sync=path.resolve(args["sync-dir"]),lane=args.lane||"1";
      fs.writeFileSync(path.join(sync,`ready-${lane}.json`),JSON.stringify({duration:report.config.video_duration,at:Date.now()}));
      const gate=path.join(sync,"start.json"),deadline=Date.now()+180000;
      while(!fs.existsSync(gate)){assert.ok(Date.now()<deadline,"Playback barrier timed out");await new Promise(r=>setTimeout(r,50));}
      const {start_at}=JSON.parse(fs.readFileSync(gate,"utf8"));
      await new Promise(r=>setTimeout(r,Math.max(0,start_at-Date.now())));
    }
    report.play_started_at=Date.now();
    await page.evaluate(async record=>{
      if(record){
        const canvas=document.querySelector('#camera-display');
        canvas.width=__experimentVideo.videoWidth;canvas.height=__experimentVideo.videoHeight;
        const stream=canvas.captureStream(30);
        const mime='video/webm;codecs=vp8';
        if(!MediaRecorder.isTypeSupported(mime))throw new Error('Canvas recording codec unavailable');
        const recorder=window.__experimentRecorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:5000000});
        window.__recordWrites=Promise.resolve();
        recorder.ondataavailable=e=>{if(e.data.size)__recordWrites=__recordWrites.then(async()=>{
          const buffer=new Uint8Array(await e.data.arrayBuffer());let binary='';
          for(let i=0;i<buffer.length;i+=32768)binary+=String.fromCharCode(...buffer.subarray(i,i+32768));
          await saveRecordingChunk(btoa(binary));
        });};
        recorder.start(2000);
      }
      __experimentController.start();__experimentVideo.currentTime=0;await __experimentVideo.play();
    },args["record-canvas"]==="true");
    let endedAt=null,lastStatus=0,lastShot=-20;
    const deadline=Date.now()+(report.config.video_duration+report.config.config.delay_seconds+45)*1000;
    while(Date.now()<deadline){
      await new Promise(r=>setTimeout(r,1000));
      const sample=await page.evaluate(()=>({at:Date.now(),source:__experimentVideo.currentTime,ended:__experimentVideo.ended,
        stats:__experimentController.snapshot(),rows:__experimentRows.splice(0)}));
      for(const row of sample.rows){log("frames",row);if(row.kind==="failure")report.errors.push(row);}delete sample.rows;report.snapshots.push(sample);
      if(args["screenshots"]!=="false"&&sample.source-lastShot>=15){
        await page.locator("#camera-display").screenshot({path:path.join(out,`display-${Math.floor(sample.source)}s.png`),timeout:8000});lastShot=sample.source;
      }
      if(Date.now()-lastStatus>5000){
        await probe("status",args.url+"/v1/status?lightweight=true");
        await probe("faces",args.url+"/v1/faces/fast");
        console.log(JSON.stringify({source:sample.source,ended:sample.ended,presented:sample.stats.distinct_presented_frames,
          complete:sample.stats.complete_contour_frames,continuous:sample.stats.continuous_contour_frames,metadata:sample.stats.metadata_received}));lastStatus=Date.now();save();
      }
      if(sample.ended&&!endedAt){endedAt=Date.now();report.source_ended_at=endedAt;}
      if(endedAt&&Date.now()-endedAt>(report.config.config.delay_seconds+3)*1000&&sample.stats.buffered_frames===0)break;
    }
    assert.ok(endedAt,"The complete video must reach EOF");
    report.final=await page.evaluate(()=>__experimentController.snapshot());
    assert.equal(report.final.buffered_frames,0,"All buffered video frames must be presented");
    if(args["record-canvas"]==="true")await page.evaluate(async()=>{
      const recorder=__experimentRecorder;
      await new Promise(resolve=>{recorder.onstop=resolve;recorder.stop();});
      await __recordWrites;for(const track of recorder.stream.getTracks())track.stop();
    });
    if(args["screenshots"]!=="false")await page.locator("#camera-display").screenshot({path:path.join(out,"final-display.png"),timeout:10000});
    const drainStarted=Date.now();
    try{
      await page.evaluate(()=>__experimentController.drain());
      report.upload_drain={completed:true,elapsed_seconds:(Date.now()-drainStarted)/1000};
    }catch(error){
      report.upload_drain={completed:false,elapsed_seconds:(Date.now()-drainStarted)/1000,error:error.message};
      throw error;
    }finally{
      // A failed drain is itself important benchmark evidence. Preserve both
      // the outstanding queue and ACK timings before the browser is closed.
      const diagnostics=await page.evaluate(()=>({
        post_drain:window.__experimentController?.snapshot()||null,
        transport_timings:window.__experimentTransportTimings||[],
      })).catch(error=>({diagnostic_error:error.message}));
      report.post_drain=diagnostics.post_drain||null;
      report.transport_timings=diagnostics.transport_timings||[];
      if(diagnostics.diagnostic_error)report.upload_drain.diagnostic_error=diagnostics.diagnostic_error;
      save();
    }
    await page.evaluate(()=>{__experimentSocket.send(JSON.stringify({type:'stop'}));__experimentController.stop();});active=false;
    let drainProbeFailures=0;
    for(let i=0;i<70;i++){
      const s=await probe("status",args.url+"/v1/status?lightweight=true");
      if(!s){if(++drainProbeFailures>=3)throw new Error("Cloud status unavailable during drain");
        await new Promise(r=>setTimeout(r,2000));continue;}
      drainProbeFailures=0;
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
