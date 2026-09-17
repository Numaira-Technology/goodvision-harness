#!/usr/bin/env node
/* Offline real-Chrome/Worker replay. No camera permission, model calls or external network.
 * NODE_PATH=<Playwright packages> node tools/evaluate_camera_contours.cjs
 *   --fixtures .goodvision/diagnostics/<private-run> --source <camera_display.js>
 *   --output .goodvision/diagnostics/<private-run>/quality.json [--synthetic-seconds 12]
 * Use --quality false for throughput only; --target-fps 20 changes synthetic display
 * sampling only, keeping its 30 fps input and 3 fps analysis admission unchanged.
 * Input JPEGs and per-frame results are private: keep both fixtures/output out of git.
 */
"use strict";
const fs=require("node:fs"),path=require("node:path"),http=require("node:http");
const crypto=require("node:crypto"),assert=require("node:assert/strict");
const {chromium}=require("playwright");
const args={};for(let i=2;i<process.argv.length;i+=2){
  assert.ok(process.argv[i].startsWith("--")&&process.argv[i+1],"Expected --name value");
  args[process.argv[i].slice(2)]=process.argv[i+1];
}
assert.ok((args.fixtures||args.manifest)&&args.output,"--fixtures or --manifest, and --output are required");
if(args["synthetic-seconds"]){
  assert.ok(Number(args["synthetic-seconds"])>=10&&Number(args["synthetic-seconds"])<=120,"Synthetic duration must be 10–120 seconds");
  assert.ok(Number(args["target-fps"]||15)>=15&&Number(args["target-fps"]||15)<=30,"Synthetic display target must be 15–30 fps");
}
const root=path.resolve(__dirname,".."),folder=path.resolve(args.fixtures||path.dirname(args.manifest)),output=path.resolve(args.output);
const source=path.resolve(args.source||path.join(root,"src/goodvision_harness/camera_display.js"));
const motionBackend=args["motion-backend"]||"legacy";
assert.ok(["legacy","dense"].includes(motionBackend),"Motion backend must be legacy or dense");
const auxiliarySources=motionBackend==="dense"?[
  {route:"/assets/camera_display_dense.js",path:path.resolve(args["dense-source"]||path.join(root,"src/goodvision_harness/camera_display_dense.js")),suffix:"-dense.js"},
  {route:"/assets/opencv-4.13.0.js",path:path.resolve(args["opencv-source"]||path.join(root,"src/goodvision_harness/assets/opencv-4.13.0.js")),suffix:"-opencv.js"}
].map(asset=>({...asset,bytes:fs.readFileSync(asset.path)})):[];
const read=name=>JSON.parse(fs.readFileSync(path.join(folder,name),"utf8").replace(/^\uFEFF/,""));
let run,sets,fixtureProvenance=null;
if(args.manifest){
  const manifest=JSON.parse(fs.readFileSync(path.resolve(args.manifest),"utf8").replace(/^\uFEFF/,""));
  run=manifest.run;sets=manifest.sets;fixtureProvenance=manifest.provenance||null;
}else{
  const metadata=read("display-metadata.json"),earlyRows=read("early-sam-results.json");
  run={stream_session_id:metadata.stream_session_id,generation:metadata.generation,crop_version:metadata.crop_version};
  const earlyAnchors=earlyRows.flatMap(row=>row.result.frames.map(frame=>({
  source_sequence:frame.frame_sequence,media_timestamp:frame.timestamp_seconds,epoch:row.result.predictor_epoch,
  subjects:frame.objects.filter(object=>object.label==="person"&&object.attributes?.mask_contours?.polygons).map(object=>({
    track_id:object.attributes.native_track.object_id,polygons:object.attributes.mask_contours.polygons,
    tracking_confirmation:object.attributes.tracking_confirmation||"confirmed"}))
})));
  sets=[{name:"early",sources:read("early-source-images.json"),anchors:earlyAnchors},
    {name:"tail",sources:read("tail-source-images.json"),anchors:metadata.anchors}];
}
sets=sets.map(set=>{
  assert.match(set.name,/^[A-Za-z0-9_-]+$/,"Fixture names must be safe URL path segments");
  set.sources.sort((a,b)=>a.timestamp-b.timestamp);
  const bySequence=new Map(set.sources.map(frame=>[frame.sequence,frame]));
  set.anchors=set.anchors.filter(anchor=>bySequence.has(anchor.source_sequence));
  if(set.reference_anchors)set.reference_anchors=set.reference_anchors.filter(anchor=>bySequence.has(anchor.source_sequence));
  for(const anchor of [...set.anchors,...(set.reference_anchors||[])])assert.ok(Math.abs(bySequence.get(anchor.source_sequence).timestamp-anchor.media_timestamp)<1e-6);
  return set;
});
const sourceBytes=fs.readFileSync(source),metricsBytes=fs.readFileSync(path.join(__dirname,"contour_quality_metrics.cjs")),evaluatorBytes=fs.readFileSync(__filename);
const syntheticOptions={anchor_hz:Number(args["synthetic-anchor-hz"]||2),anchor_policy:args["anchor-policy"]||"past_only",trace:null};
assert.ok(Number.isFinite(syntheticOptions.anchor_hz)&&syntheticOptions.anchor_hz>0&&syntheticOptions.anchor_hz<=5);
assert.ok(["past_only","nearest_arrived"].includes(syntheticOptions.anchor_policy));
if(args["synthetic-anchor-trace"]){
  const tracePath=path.resolve(args["synthetic-anchor-trace"]),bytes=fs.readFileSync(tracePath),trace=JSON.parse(bytes.toString("utf8").replace(/^\uFEFF/,""));
  assert.ok(Array.isArray(trace.observations)&&trace.observations.length>1,"Synthetic trace requires actual SAM observations");
  const origin=trace.observations[0].timestamp;
  syntheticOptions.trace={path:tracePath,sha256:crypto.createHash("sha256").update(bytes).digest("hex"),scope:trace.scope,
    frames:trace.observations.map(observation=>({timestamp:observation.timestamp-origin,source_to_result_seconds:observation.source_to_result_seconds}))};
  assert.ok(syntheticOptions.trace.frames.every((frame,index)=>Number.isFinite(frame.timestamp)&&Number.isFinite(frame.source_to_result_seconds)&&frame.source_to_result_seconds>=0&&
    (!index||frame.timestamp>syntheticOptions.trace.frames[index-1].timestamp)),"Trace source times must be strictly ordered and ages nonnegative");
}
// Test-server-only timing wrapper. The production handler and messages are unchanged.
const workerTimingWrapper=`\nif(typeof document==="undefined"&&typeof importScripts==="function"){
  const originalHandler=globalThis.onmessage;
  globalThis.onmessage=function(event){const started=performance.now();
    try{return originalHandler.call(this,event);}finally{
      globalThis.postMessage({type:"evaluation_compute",operation:event.data?.type,duration_ms:performance.now()-started});
    }
  };
}\n`;
const failureCaptureWrapper=args["capture-failures"]==="true"?`\nif(typeof document==="undefined"&&typeof importScripts==="function"){
  const handle=globalThis.onmessage,send=globalThis.postMessage.bind(globalThis),frames=new Map(),anchors=new Map(),seen=new Set();
  globalThis.__evaluationFrameIdentity=new WeakMap();
  globalThis.postMessage=function(message,...rest){
    send(message,...rest);
    if(message.type!=="present")return;
    const result=message.result||{},anchor=anchors.get(result.anchor_sequence);
    if(!anchor)return;
    for(const [reason,count] of Object.entries(result.motion_rejections||{})){
      if(!count||seen.has(reason)||seen.size>=4)continue;seen.add(reason);
      const chain=[...frames.values()].filter(frame=>frame.sequence>=anchor.source_sequence&&frame.sequence<=message.sequence);
      send({type:"evaluation_failure",reason,anchor,result,frames:chain.map(frame=>({...frame,pixels:Array.from(frame.pixels)}))});
    }
  };
  globalThis.onmessage=function(event){const message=event.data;
    if(message.type==="frame"){globalThis.__evaluationFrameIdentity.set(message.frame.pixels,{sequence:message.frame.sequence,timestamp:message.frame.timestamp});
      frames.set(message.frame.sequence,{...message.frame,pixels:message.frame.pixels.slice()});
      while(frames.size>150)frames.delete(frames.keys().next().value);}
    if(message.type==="metadata")for(const anchor of message.data.anchors||[])anchors.set(anchor.source_sequence,anchor);
    return handle.call(this,event);
  };
}\n`:"";
// Diagnostic-only observation around the unchanged adapter APIs/OpenCV calls.
// Copies one failed raster and its inputs; never changes masks, thresholds or flow.
const denseFailureWrapper=args["capture-failures"]==="true"?`\n{
  const factory=globalThis.GoodvisionDenseMotion.createDenseMotionAdapter;
  globalThis.GoodvisionDenseMotion.createDenseMotionAdapter=function(cv,options){
    const adapter=factory(cv,options),threshold=cv.threshold,remap=cv.remap,seen=new Map();let observation=null;
    cv.threshold=function(...args){const result=threshold.apply(cv,args);
      if(observation&&args[1].rows===observation.height&&args[1].cols===observation.width)observation.binary=new Uint8Array(args[1].data);
      return result;
    };
    Object.assign(cv.threshold,threshold);
    cv.remap=function(...args){const result=remap.apply(cv,args);
      if(observation&&args[1].rows===observation.height&&args[1].cols===observation.width){
        observation.baseMask=new Uint8Array(args[0].data);observation.map=new Float32Array(args[2].data32F);
      }return result;
    };
    Object.assign(cv.remap,remap);
    for(const method of ["warpPolygons","warpChain"]){
      const original=adapter[method];if(!original)continue;
      adapter[method]=function(polygons,input){
        const motions=method==="warpChain"?input:[input],first=motions[0],last=motions.at(-1);
        if(!first||!last)return original.call(adapter,polygons,input);
        const target=globalThis.__evaluationFrameIdentity?.get(last.current)||null;
        observation={width:last.width,height:last.height};let result;
        try{
          result=original.call(adapter,polygons,input);
          const reason=last.last_warp_rejection,prior=seen.get(reason);
          if(!result&&["boundary_occluded","component_appearance"].includes(reason)&&
            (!prior||(prior.count<2&&target&&target.timestamp-prior.timestamp>=2))){
            seen.set(reason,{count:(prior?.count||0)+1,timestamp:target?.timestamp??0});
            globalThis.postMessage({type:"evaluation_dense_failure",method,reason,
              source:globalThis.__evaluationFrameIdentity?.get(first.previous)||null,target,
              width:last.width,height:last.height,input_polygons:polygons,
              source_pixels:Array.from(first.previous),target_pixels:Array.from(last.current),
              predicted_mask:observation.binary?Array.from(observation.binary):null,
              base_mask:observation.baseMask?Array.from(observation.baseMask):null,
              source_map:observation.map?Array.from(observation.map):null,
              diagnostics:last.last_warp_diagnostics,
              chain:motions.map(motion=>({source:globalThis.__evaluationFrameIdentity?.get(motion.previous)||null,
                target:globalThis.__evaluationFrameIdentity?.get(motion.current)||null,chain_index:motion.chain_index}))});
          }return result;
        }catch(error){
          if(!seen.has("debug_error")){seen.set("debug_error",{count:1});globalThis.postMessage({type:"evaluation_dense_debug_error",message:String(error.message),stack:String(error.stack)});}
          throw error;
        }finally{observation=null;}
      };
    }
    return adapter;
  };
}\n`:"";
const server=http.createServer((request,response)=>{
  let body,type="application/json";
  if(request.url==="/"){body='<video id="video" muted playsinline></video><canvas id="display"></canvas><script src="/metrics.js"></script><script src="/assets/camera_display.js"></script>';type="text/html";}
  else if(request.url==="/assets/camera_display.js"){body=Buffer.concat([sourceBytes,Buffer.from(workerTimingWrapper+failureCaptureWrapper)]);type="application/javascript";}
  else if(auxiliarySources.some(asset=>asset.route===request.url)){
    body=auxiliarySources.find(asset=>asset.route===request.url).bytes;
    if(request.url==="/assets/camera_display_dense.js")body=Buffer.concat([body,Buffer.from(denseFailureWrapper)]);
    type="application/javascript";
  }
  else if(request.url==="/metrics.js"){body=metricsBytes;type="application/javascript";}
  else if(request.url==="/fixture")body=JSON.stringify({run,sets:sets.map(set=>({...set,sources:set.sources.map(({sequence,timestamp})=>({sequence,timestamp}))}))});
  else if(/^\/image\/[A-Za-z0-9_-]+\/\d+$/.test(request.url)){
    const [, ,name,sequence]=request.url.split("/"),frame=sets.find(set=>set.name===name)?.sources.find(frame=>frame.sequence===Number(sequence));
    if(frame){body=Buffer.from(frame.jpeg,"base64");type="image/jpeg";}
  }
  if(body===undefined){response.statusCode=404;body="{}";}
  response.setHeader("Content-Type",type);response.end(body);
});
async function runReplay(page,maxPropagationSeconds,scenarioMode,motionWidth,delays,historySeconds,motionBackend,bufferedConfig){
  return page.evaluate(async({maxPropagationSeconds,scenarioMode,motionWidth,delays,historySeconds,motionBackend,bufferedConfig})=>{
    const {run,sets}=await(await fetch("/fixture")).json(),metrics=ContourQualityMetrics;
    if(typeof metrics.bestIoUMatching!=="function")throw new Error("Geometric matching helper is unavailable");
    const summaries=[],rows=[],workerTimings=[],computeTimings=[],backendReady=[];
    function raster(polygons,width,height){
      const canvas=new OffscreenCanvas(width,height),ctx=canvas.getContext("2d",{willReadFrequently:true});
      ctx.fillStyle="#fff";
      for(const polygon of polygons){ctx.beginPath();polygon.forEach(([x,y],i)=>i?ctx.lineTo(x*(width-1),y*(height-1)):ctx.moveTo(x*(width-1),y*(height-1)));ctx.closePath();ctx.fill();}
      const rgba=ctx.getImageData(0,0,width,height).data;
      return Uint8Array.from({length:width*height},(_,i)=>rgba[i*4+3]>=128?1:0);
    }
    function maskSummary(polygons,width,height){
      const mask=raster(polygons,width,height),points=polygons.flat();let area=0;
      for(const value of mask)area+=value;
      return {mask,area,minX:points.length?Math.max(0,Math.floor(Math.min(...points.map(point=>point[0]))*(width-1))-1):0,
        maxX:points.length?Math.min(width-1,Math.ceil(Math.max(...points.map(point=>point[0]))*(width-1))+1):-1,
        minY:points.length?Math.max(0,Math.floor(Math.min(...points.map(point=>point[1]))*(height-1))-1):0,
        maxY:points.length?Math.min(height-1,Math.ceil(Math.max(...points.map(point=>point[1]))*(height-1))+1):-1};
    }
    function rasterIoU(a,b,width){
      let intersection=0;
      for(let y=Math.max(a.minY,b.minY);y<=Math.min(a.maxY,b.maxY);y++)
        for(let x=Math.max(a.minX,b.minX);x<=Math.min(a.maxX,b.maxX);x++)
          if(a.mask[y*width+x]&&b.mask[y*width+x])intersection++;
      return a.area+b.area?intersection/(a.area+b.area-intersection):0;
    }
    function worker(config){
      const instance=new Worker("/assets/camera_display.js");let requestId=0,pending=null;
      let readyResolve,readyReject;const started=performance.now();
      const ready=motionBackend==="dense"?new Promise((resolve,reject)=>{readyResolve=resolve;readyReject=reject;}):Promise.resolve();
      const readyTimer=motionBackend==="dense"?setTimeout(()=>readyReject(new Error("Dense Worker initialization timeout")),15000):null;
      instance.onmessage=({data})=>{
        if(data.type==="motion_ready"){
          if(data.backend!==motionBackend){readyReject?.(new Error("Unexpected Worker backend"));return;}
          backendReady.push({backend:data.backend,initialization_ms:performance.now()-started});clearTimeout(readyTimer);readyResolve?.();return;
        }
        if(data.type==="failure"){clearTimeout(readyTimer);readyReject?.(new Error(`Real Worker failure: ${data.stage||"operation"}`));}
        if(data.type==="evaluation_compute"){computeTimings.push({...data,measurement:"handler_wall"});return;}
        if(Number.isFinite(data.compute_ms))computeTimings.push({operation:data.type==="present"&&data.expired?"present_expired":data.type,
          duration_ms:data.compute_ms,expired:!!data.expired,measurement:"formal_operation"});
        if(!pending)return;
        if(data.type==="failure"){pending.reject(new Error("Real Worker failure"));pending=null;return;}
        if(data.type!==pending.type)return;
        workerTimings.push({type:data.type,ms:performance.now()-pending.started});
        clearTimeout(pending.timer);pending.resolve(data);pending=null;
      };
      instance.onerror=error=>{clearTimeout(readyTimer);readyReject?.(new Error(error.message));if(pending)pending.reject(new Error(error.message));};
      instance.postMessage({type:"reset",config});
      return {async send(message,type,transfer=[]){
        await ready;
        if(pending)throw new Error("Replay only issues one operation per Worker at once");
        return new Promise((resolve,reject)=>{
          pending={type,resolve,reject,started:performance.now(),timer:setTimeout(()=>reject(new Error("Worker timeout")),10000)};
          instance.postMessage({...message,id:++requestId},transfer);
        });
      },close(){clearTimeout(readyTimer);instance.terminate();}};
    }
    function summarize(set,scenario,selected){
      const people=selected.flatMap(row=>row.people),visible=people.filter(person=>person.predicted);
      const afterAnchor=selected.filter(row=>row.coverage!=="no_anchor").flatMap(row=>row.people);
      const geometric=people.filter(person=>person.geometry_matched_iou>0);
      const lines=people.filter(person=>person.line_pixels>0);
      const mean=(records,key)=>records.length?records.reduce((sum,person)=>sum+(person[key]||0),0)/records.length:null;
      const counts={};for(const row of selected)counts[row.coverage]=(counts[row.coverage]||0)+1;
      return {set,scenario,frames_with_reference:selected.length,reference_people:people.length,
        predicted_people:visible.length,person_availability:people.length?visible.length/people.length:null,
        partial_people:people.filter(person=>person.partial).length,
        people_with_any_lines:lines.length,mean_line_precision_3px_visible:mean(lines,"line_precision_3px"),
        mean_line_precision_6px_visible:mean(lines,"line_precision_6px"),mean_boundary_recall_3px_all:mean(people,"boundary_recall_3px"),
        mean_boundary_recall_6px_all:mean(people,"boundary_recall_6px"),
        p95_line_boundary_error_px_visible:metrics.percentile(lines.map(person=>person.line_boundary_p95_px),.95),
        warmup_reference_people:people.length-afterAnchor.length,after_anchor_reference_people:afterAnchor.length,
        after_anchor_person_availability:afterAnchor.length?afterAnchor.filter(person=>person.predicted).length/afterAnchor.length:null,
        frames_without_anchor:selected.filter(row=>row.coverage==="no_anchor").length,
        failure_breakdown:{no_anchor:counts.no_anchor||0,expired:counts.anchor_expired||0,
          motion_gap:counts.motion_gap||0,matching_rejected:counts.motion_unreliable||0,
          missing_source:counts.missing_source||0,late:0,
          note:"Replay RPCs are awaited in order, so late display results are measured only by the realtime synthetic Controller run."},
        unexpected_people:selected.reduce((sum,row)=>sum+row.unmatched_prediction_count,0),
        mean_mask_iou_all:mean(people,"mask_iou"),mean_reference_recall_all:mean(people,"reference_recall"),
        mean_mask_iou_visible:mean(visible,"mask_iou"),mean_precision_visible:mean(visible,"prediction_precision"),
        geometry_matched_people:geometric.length,mean_geometry_matched_iou_all:mean(people,"geometry_matched_iou"),
        mean_geometry_matched_iou_matched:mean(geometric,"geometry_matched_iou"),
        geometry_matches_different_native_id:geometric.filter(person=>person.geometry_matched_prediction_track_id!==person.track_id).length,
        mean_boundary_chamfer_px_visible:mean(visible,"boundary_mean_chamfer_px"),
        p95_boundary_chamfer_px_visible:metrics.percentile(visible.map(person=>person.boundary_p95_chamfer_px),.95),
        complete_people_recall90:people.filter(person=>person.reference_recall>=.9).length,
        good_people_iou80_recall90:people.filter(person=>person.reference_recall>=.9&&person.mask_iou>=.8).length,
        mean_frozen_anchor_iou_all:mean(people.map(person=>person.frozen_anchor),"mask_iou"),coverage_counts:counts};
    }
    for(const set of sets){
      const images=new Map(),anchors=new Map(set.anchors.map(anchor=>[anchor.source_sequence,anchor])),
        references=new Map((set.reference_anchors||set.anchors).map(anchor=>[anchor.source_sequence,anchor]));
      for(const frame of set.sources){
        const image=await createImageBitmap(await(await fetch(`/image/${set.name}/${frame.sequence}`)).blob());
        const gray=new OffscreenCanvas(motionWidth,Math.min(240,Math.round(image.height/image.width*motionWidth)));
        const ctx=gray.getContext("2d",{willReadFrequently:true});ctx.drawImage(image,0,0,gray.width,gray.height);
        const rgba=ctx.getImageData(0,0,gray.width,gray.height).data;
        const pixels=Uint8Array.from({length:gray.width*gray.height},(_,i)=>(rgba[i*4]*77+rgba[i*4+1]*150+rgba[i*4+2]*29)>>8);
        images.set(frame.sequence,{width:gray.width,height:gray.height,pixels,originalWidth:image.width,originalHeight:image.height});image.close();
      }
      // Default gap retains actual production rejection. Sparse cases explicitly override only
      // that gap for a 3 fps image-motion stress test; they do not manufacture missing 15 fps frames.
      const sparseGap=scenarioMode==="production"?.2:.5;
      const scenarios=bufferedConfig?[
        {name:"exact_reference_control",lag:0,gap:.2,control:true},
        ...(bufferedConfig.mode==="trace"?[0]:bufferedConfig.model_delays).flatMap(modelDelay=>["past_only","nearest_arrived"].map(policy=>({
          name:bufferedConfig.mode==="trace"?`trace_${scenarioMode}_${policy}`:`buffered_${scenarioMode}_${policy}_model_${modelDelay}s`,buffered:true,policy,
          trace:bufferedConfig.mode==="trace",modelDelay,displayDelay:bufferedConfig.display_delay_seconds,cadenceHz:bufferedConfig.cadence_hz,gap:sparseGap})))
      ]:scenarioMode==="production"?[{name:"exact_reference_control",lag:0,gap:.2},
        {name:"production_gap_previous_anchor",lag:0,gap:.2,previous:true},
        ...(delays||[.4,1,1.5,2.5,3.5]).map(lag=>({name:`production_gap_delayed_${lag}s`,lag,gap:.2}))]:[
        {name:"exact_reference_control",lag:0,gap:.2},
        {name:"production_gap_delayed_1s",lag:1,gap:.2},
        {name:"sparse_gap_previous_anchor",lag:0,gap:.5,previous:true},
        ...(delays||[.4,1,1.5,2,2.5,3.5]).map(lag=>({name:`sparse_gap_delayed_${lag}s`,lag,gap:.5}))];
      for(const scenario of scenarios){
        const replay=worker({...run,max_frame_gap_seconds:scenario.gap,max_propagation_seconds:maxPropagationSeconds,
          max_history_seconds:historySeconds,motion_backend:motionBackend,anchor_policy:scenario.policy||"past_only"}),delivered=new Set(),selected=[];
        const suppliedAnchors=scenario.control?(set.reference_anchors||set.anchors):scenario.trace?set.anchors:scenario.buffered?metrics.sampleAnchorsAtCadence(set.anchors,scenario.cadenceHz):set.anchors;
        const activeAnchors=scenario.control?references:anchors;
        if(scenario.trace&&suppliedAnchors.some(anchor=>!Number.isFinite(anchor.availability_clock_seconds)))throw new Error("Actual trace requires every anchor arrival time");
        const arrivals=new Map(suppliedAnchors.map(anchor=>[anchor.source_sequence,anchor.availability_clock_seconds??anchor.media_timestamp+(scenario.modelDelay||0)]));
        const events=scenario.buffered?metrics.bufferedReplayEvents(set.sources,suppliedAnchors,scenario.modelDelay,scenario.displayDelay):
          set.sources.flatMap(frame=>[{type:"capture",frame,time:frame.timestamp},{type:"deliver",frame,time:frame.timestamp},{type:"present",frame,time:frame.timestamp}]);
        try{
          for(const event of events){
            const frame=event.frame;
            if(event.type==="capture"){
              const image=images.get(frame.sequence),pixels=image.pixels.slice();
              const ack=await replay.send({type:"frame",frame:{sequence:frame.sequence,timestamp:frame.timestamp,width:image.width,height:image.height,pixels}},"frame_processed",[pixels.buffer]);
              if(!ack.accepted)throw new Error("Fixture source frame rejected");continue;
            }
            if(event.type==="deliver"){
              const eligible=event.anchors||suppliedAnchors.filter(anchor=>!delivered.has(anchor.source_sequence)&&
                anchor.media_timestamp+scenario.lag<=frame.timestamp+1e-6&&
                (!scenario.previous||anchor.source_sequence<frame.sequence));
              if(eligible.length){
                const ack=await replay.send({type:"metadata",data:{...run,anchors:eligible,events:[]}},"metadata_processed");
                if(!ack.accepted||ack.rejected)throw new Error("Fixture metadata rejected");
                for(const anchor of eligible)delivered.add(anchor.source_sequence);
              }continue;
            }
            const {result}=await replay.send({type:"present",sequence:frame.sequence,timestamp:frame.timestamp},"present");
            if(result.subjects.some(person=>person.target_sequence!==frame.sequence))throw new Error("Worker returned contours for a different displayed source");
            if(scenario.buffered)for(const subject of result.subjects){
              const sequence=subject.anchor_sequence??result.anchor_sequence;
              if(!delivered.has(sequence)||arrivals.get(sequence)>event.time+1e-6)throw new Error("Worker used a SAM anchor before its result arrived");
              if(scenario.policy==="past_only"&&activeAnchors.get(sequence).media_timestamp>frame.timestamp+1e-6)throw new Error("Past-only Worker used a future source anchor");
            }
            const target=references.get(frame.sequence);
            if(!target||!target.subjects.length)continue;
            const image=images.get(frame.sequence);
            const latestDelivered=suppliedAnchors.filter(anchor=>delivered.has(anchor.source_sequence)&&anchor.media_timestamp<=frame.timestamp)
              .sort((a,b)=>b.media_timestamp-a.media_timestamp)[0];
            const firstFuture=scenario.buffered&&scenario.policy==="nearest_arrived"?suppliedAnchors.filter(anchor=>delivered.has(anchor.source_sequence)&&anchor.media_timestamp>frame.timestamp)
              .sort((a,b)=>a.media_timestamp-b.media_timestamp)[0]:null;
            const anchor=activeAnchors.get(result.anchor_sequence)||latestDelivered,width=image.originalWidth,height=image.originalHeight;
            const referenceMasks=target.subjects.map(reference=>maskSummary(reference.polygons,width,height));
            const predictionMasks=result.subjects.map(predicted=>maskSummary(predicted.polygons,width,height));
            const pairwise=referenceMasks.map(reference=>predictionMasks.map(predicted=>rasterIoU(reference,predicted,width)));
            const geometric=metrics.bestIoUMatching(pairwise);
            const people=target.subjects.map((reference,referenceIndex)=>{
              const predicted=result.subjects.find(person=>String(person.track_id)===String(reference.track_id)&&person.epoch===target.epoch);
              const complete=!!predicted&&predicted.complete!==false&&Array.isArray(predicted.polygons)&&predicted.polygons.length>0;
              const actualAnchor=activeAnchors.get(predicted?.anchor_sequence??result.anchor_sequence)||anchor;
              const pastSubject=latestDelivered?.subjects.find(person=>String(person.track_id)===String(reference.track_id));
              const futureSubject=firstFuture?.epoch===latestDelivered?.epoch&&firstFuture?.subjects.find(person=>String(person.track_id)===String(reference.track_id));
              const policyCandidate=pastSubject?(futureSubject&&firstFuture.media_timestamp-frame.timestamp<frame.timestamp-latestDelivered.media_timestamp?firstFuture:latestDelivered):null;
              const frozen=(predicted?actualAnchor:policyCandidate||anchor)?.subjects.find(person=>String(person.track_id)===String(reference.track_id));
              const expected=referenceMasks[referenceIndex].mask,matched=geometric.find(pair=>pair.reference_index===referenceIndex);
              const outputLines=complete?predicted.polygons.map(polygon=>[...polygon,polygon[0]]):predicted?.open_segments||[];
              return {track_id:String(reference.track_id),predicted:complete,partial:!!predicted&&!complete&&!!predicted.open_segments?.length,
                open_segments:predicted?.open_segments||[],
                actual_anchor_sequence:predicted?actualAnchor?.source_sequence??null:null,
                actual_anchor_signed_age_seconds:predicted&&actualAnchor?frame.timestamp-actualAnchor.media_timestamp:null,
                actual_anchor_arrival_clock_seconds:scenario.buffered&&predicted&&actualAnchor?arrivals.get(actualAnchor.source_sequence)??null:null,
                policy_candidate_anchor_sequence:scenario.buffered?policyCandidate?.source_sequence??null:undefined,
                policy_candidate_direction:scenario.buffered?(policyCandidate?(Math.abs(policyCandidate.media_timestamp-frame.timestamp)<1e-6?"exact":policyCandidate.media_timestamp>frame.timestamp?"future":"past"):"unobserved"):undefined,
                predicted_method:predicted?.method||null,
                relocalized:!!predicted?.relocalized,
                geometry_matched_iou:matched?.iou||0,
                geometry_matched_prediction_track_id:matched?String(result.subjects[matched.predicted_index].track_id):null,
                reference_polygons:reference.polygons,predicted_polygons:complete?predicted.polygons:[],
                ...metrics.compareMasks(raster(complete?predicted.polygons:[],width,height),expected,width,height),
                ...metrics.compareContourLines(outputLines,expected,width,height),
                frozen_anchor:metrics.compareMasks(raster(frozen?.polygons||[],width,height),expected,width,height)};
            });
            const row={set:set.name,scenario:scenario.name,sequence:frame.sequence,timestamp:frame.timestamp,
              presentation_clock_seconds:event.time,anchor_policy:scenario.policy||"past_only",
              available_anchor_sequences:scenario.buffered?[...delivered]:undefined,
              geometry_anchor_sequences:result.geometry_anchor_sequences||[],
              raster_width:width,raster_height:height,coverage:result.coverage,anchor_sequence:result.anchor_sequence??null,
              anchor_age_seconds:result.anchor_age_seconds??null,motion_rejections:result.motion_rejections||{},unmatched_prediction_count:result.subjects.filter(person=>
                !target.subjects.some(reference=>String(reference.track_id)===String(person.track_id)&&person.epoch===target.epoch)).length,people};
            row.geometry_matching=geometric;
            selected.push(row);rows.push(row);
          }
        }finally{replay.close();}
        const directionPeople=selected.flatMap(row=>row.people);
        summaries.push({...summarize(set.name,scenario.name,selected),...(scenario.buffered?{
          anchor_policy:scenario.policy,model_delay_seconds:scenario.trace?null:scenario.modelDelay,display_delay_seconds:scenario.displayDelay,
          actual_arrival_trace:!!scenario.trace,planned_anchor_cadence_hz:scenario.trace?null:scenario.cadenceHz,
          supplied_anchors:suppliedAnchors.map(anchor=>({sequence:anchor.source_sequence,timestamp:anchor.media_timestamp,arrival_clock_seconds:arrivals.get(anchor.source_sequence)})),
          actual_source_anchor_cadence_hz:suppliedAnchors.length>1?(suppliedAnchors.length-1)/(suppliedAnchors.at(-1).media_timestamp-suppliedAnchors[0].media_timestamp):null,
          future_geometry_people:directionPeople.filter(person=>person.actual_anchor_signed_age_seconds<0).length,
          by_geometry_direction:Object.fromEntries(["exact","past","future","unobserved"].map(direction=>{
            const records=directionPeople.filter(person=>person.policy_candidate_direction===direction),visible=records.filter(person=>person.predicted);
            return [direction,{reference_people:records.length,complete_people:visible.length,
              complete_availability:records.length?visible.length/records.length:null,
              mean_iou_all:records.length?records.reduce((sum,person)=>sum+person.mask_iou,0)/records.length:null,
              mean_iou_complete:visible.length?visible.reduce((sum,person)=>sum+person.mask_iou,0)/visible.length:null,
              p95_boundary_error_complete_px:metrics.percentile(visible.map(person=>person.boundary_p95_chamfer_px),.95)}];
          }))
        }:{})});
      }
    }
    return {summaries,rows,backend_ready:backendReady,worker_compute:computeTimings,worker_timings:{operations:workerTimings.length,
      note:"Sequential RPC wall time includes messaging and Worker compute; not realtime throughput.",
      by_type:Object.fromEntries([...new Set(workerTimings.map(row=>row.type))].map(type=>{
        const values=workerTimings.filter(row=>row.type===type).map(row=>row.ms);
        return [type,{count:values.length,mean_ms:values.reduce((sum,n)=>sum+n,0)/values.length,p95_ms:metrics.percentile(values,.95),max_ms:Math.max(...values)}];}))}};
  },{maxPropagationSeconds,scenarioMode,motionWidth,delays,historySeconds,motionBackend,bufferedConfig});
}
async function runSynthetic(page,seconds,targetFPS,anchorDelayMs,maxPropagationSeconds,motionWidth,historySeconds,motionBackend,displayDelaySeconds,syntheticOptions){
  return page.evaluate(async({seconds,targetFPS,anchorDelayMs,maxPropagationSeconds,motionWidth,historySeconds,motionBackend,displayDelaySeconds,syntheticOptions})=>{
    const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms)),api=GoodvisionCameraDisplay;
    const video=document.querySelector("video"),canvas=document.querySelector("#display"),source=document.createElement("canvas");
    source.width=640;source.height=480;const ctx=source.getContext("2d"),started=performance.now();
    const texture=document.createElement("canvas");texture.width=160;texture.height=120;
    const textureContext=texture.getContext("2d"),image=textureContext.createImageData(160,120);
    let seed=91237;for(let i=0;i<160*120;i++){
      seed=(Math.imul(seed,1664525)+1013904223)>>>0;
      const color=60+(seed>>>24)*.7;image.data.set([color,color,color,255],i*4);
    }textureContext.putImageData(image,0,0);
    const base=[[260,65],[300,65],[320,90],[320,150],[345,175],[375,305],[350,315],
      [325,225],[325,430],[295,430],[285,300],[270,430],[240,430],[245,225],
      [225,315],[205,305],[230,175],[250,150],[245,100]];
    let generated=0,currentPolygon;const sourcePolygons=new Map();
    function draw(){
      const t=(performance.now()-started)/1000,dx=35*Math.sin(t*1.2),dy=5*Math.sin(t*2);
      ctx.fillStyle="#384348";ctx.fillRect(0,0,640,480);
      for(let y=0;y<480;y+=32)for(let x=0;x<640;x+=32){ctx.fillStyle=(x+y)%64?"#475157":"#293337";ctx.fillRect(x,y,31,31);}
      currentPolygon=base.map(([x,y])=>[(x+dx)/639,(y+dy)/479]);
      ctx.save();ctx.beginPath();base.forEach(([x,y],i)=>i?ctx.lineTo(x+dx,y+dy):ctx.moveTo(x+dx,y+dy));ctx.closePath();ctx.clip();
      ctx.drawImage(texture,180+dx,40+dy,240,400);ctx.restore();generated++;
      // Bind synthetic geometry to the actual decoded video frame, not the most
      // recently drawn pose (which can lead a video-frame callback by one frame).
      sourcePolygons.set(generated,structuredClone(currentPolygon));
      for(let bit=0;bit<16;bit++){ctx.fillStyle=(generated>>>bit)&1?"#fff":"#000";ctx.fillRect(bit*4,0,4,8);}
    }
    draw();const stream=source.captureStream(30);video.srcObject=stream;await video.play();
    const drawTimer=setInterval(draw,1000/30),anchors=[],anchorSchedule=[],metadataSeen=new Set(),reports=[],failures=[],presented=[],pendingSamples=[],backlogSamples=[],workerMs=[],computeTimings=[],handlerTimings=[],failureCases=[],denseFailureCases=[],backendReady=[],presentationResults=[],sourceTimes=new Map();
    const config={stream_session_id:"synthetic-replay",generation:1,crop_version:0,delay_seconds:displayDelaySeconds,target_fps:targetFPS,
      analysis_sample_fps:3,max_propagation_seconds:maxPropagationSeconds,motion_width:motionWidth,max_history_seconds:historySeconds,motion_backend:motionBackend,anchor_policy:syntheticOptions.anchor_policy};
    let controller;
    const socket={readyState:1,bufferedAmount:0,send(packet){
      const size=new DataView(packet).getUint32(4),metadata=JSON.parse(new TextDecoder().decode(new Uint8Array(packet,8,size)));
      queueMicrotask(()=>controller.acknowledgeUpload({...metadata,type:"camera_frame_ack",accepted:true}));
    }};
    const originalFetch=window.fetch;
    window.fetch=async(url,options)=>{
      if(String(url).endsWith("/metadata")){
        const available=anchors.filter(anchor=>anchor.arrivesAt<=performance.now()&&
          anchor.media_timestamp>=(anchors.at(-1)?.media_timestamp||0)-historySeconds);
        for(const anchor of available)metadataSeen.add(anchor.source_sequence);
        return new Response(JSON.stringify({...config,
          anchors:available.map(({arrivesAt,...anchor})=>anchor),events:[]}),{status:200});
      }
      if(String(url).endsWith("/metrics")){reports.push(JSON.parse(options.body));return new Response("{}",{status:200});}
      return originalFetch(url,options);
    };
    try{
      controller=new api.BrowserController({video,canvas,config,settings:{sample_fps:3,jpeg_quality:85,max_image_side:640},socket,
        onFailure:message=>failures.push(message)});
      if(Math.abs(controller.frames.delay-displayDelaySeconds)>1e-6)throw new Error("Controller did not honor requested display delay");
      const capture=controller.capture.bind(controller);let nextAnchor=0,traceIndex=0;
      const markerCanvas=new OffscreenCanvas(64,8),markerContext=markerCanvas.getContext("2d",{willReadFrequently:true});
      controller.capture=(now,meta)=>{
        const before=controller.sourceSequence;capture(now,meta);
        if(controller.sourceSequence>before){
          const frame=controller.frames.frames.at(-1);
          if(frame)sourceTimes.set(frame.sequence,frame.timestamp);
          const planned=syntheticOptions.trace?.frames[traceIndex],due=syntheticOptions.trace?(planned?.timestamp??Infinity):nextAnchor;
          if(frame&&frame.timestamp>=due){
            markerContext.drawImage(frame.bitmap,0,0,64,8,0,0,64,8);
            const rgba=markerContext.getImageData(0,0,64,8).data;let generatedFrame=0;
            for(let bit=0;bit<16;bit++)if(rgba[(2*64+bit*4+2)*4]>127)generatedFrame|=1<<bit;
            const exactPolygon=sourcePolygons.get(generatedFrame);
            if(!exactPolygon)throw new Error("Synthetic source marker was not decoded; anchor geometry would be unreliable");
            const sourceAgeSeconds=planned?.source_to_result_seconds??anchorDelayMs/1000;
            const arrivesAt=syntheticOptions.trace?frame.capturedAt+sourceAgeSeconds*1000:performance.now()+anchorDelayMs;
            if(syntheticOptions.trace)traceIndex++;else nextAnchor=frame.timestamp+1/syntheticOptions.anchor_hz;
            anchorSchedule.push({sequence:frame.sequence,requested_source_timestamp:due,actual_source_timestamp:frame.timestamp,
              source_selection_delay_seconds:frame.timestamp-due,source_to_result_seconds:sourceAgeSeconds,generated_frame:generatedFrame,
              capture_elapsed_seconds:(frame.capturedAt-started)/1000,arrival_elapsed_seconds:(arrivesAt-started)/1000});
            anchors.push({source_sequence:frame.sequence,media_timestamp:frame.timestamp,epoch:"synthetic",
              arrivesAt,
              subjects:[{track_id:"person",polygons:[exactPolygon],tracking_confirmation:"confirmed"}]});
          }
        }
      };
      const note=controller.frames.notePresented.bind(controller.frames);
      controller.frames.notePresented=(frame,now)=>{
        const all=controller.stats.observed_contour_frames+controller.stats.propagated_contour_frames,
          partial=controller.stats.partial_contour_frames||0,complete=controller.stats.complete_contour_frames??Math.max(0,all-partial);
        presented.push({sequence:frame.sequence,timestamp:frame.timestamp,at:now,age_seconds:(now-frame.capturedAt)/1000,
          coverage:canvas.dataset.contourCoverage,has_contours:all>(presented.at(-1)?.cumulative_contours||0),
          has_complete_contours:complete>(presented.at(-1)?.cumulative_complete_contours||0),
          has_partial_contours:partial>(presented.at(-1)?.cumulative_partial_contours||0),
          cumulative_contours:all,cumulative_complete_contours:complete,cumulative_partial_contours:partial});note(frame,now);
      };
      controller.worker.addEventListener("message",({data})=>{
        if(data.type==="evaluation_failure")failureCases.push(data);
        if(data.type==="evaluation_dense_debug_error")failures.push(data);
        if(data.type==="evaluation_dense_failure"){
          const reference=pixels=>{
            let generatedFrame=0;
            for(let bit=0;bit<16;bit++)if(pixels[Math.floor(2*data.height/480)*data.width+Math.floor((bit*4+2)*data.width/640)]>127)generatedFrame|=1<<bit;
            const polygon=sourcePolygons.get(generatedFrame);
            return {generated_frame:generatedFrame,polygons:polygon?[structuredClone(polygon)]:null};
          };
          data.source_reference=reference(data.source_pixels);data.target_reference=reference(data.target_pixels);
          if(data.predicted_mask&&data.target_reference.polygons){
            const raster=new OffscreenCanvas(data.width,data.height),context=raster.getContext("2d",{willReadFrequently:true});
            context.fillStyle="#fff";context.beginPath();
            data.target_reference.polygons[0].forEach(([x,y],i)=>i?context.lineTo(x*(data.width-1),y*(data.height-1)):context.moveTo(x*(data.width-1),y*(data.height-1)));
            context.closePath();context.fill();const rgba=context.getImageData(0,0,data.width,data.height).data;
            const expected=Uint8Array.from({length:data.width*data.height},(_,i)=>rgba[i*4+3]>=128?1:0);
            data.prediction_vs_generated_reference=ContourQualityMetrics.compareMasks(Uint8Array.from(data.predicted_mask,pixel=>pixel>=128?1:0),expected,data.width,data.height);
          }
          denseFailureCases.push(data);
        }
        if(data.type==="evaluation_compute")handlerTimings.push({operation:data.operation,ms:data.duration_ms});
        if(data.type==="motion_ready")backendReady.push({backend:data.backend,elapsed_ms:performance.now()-started});
        if(data.type==="present")presentationResults.push({sequence:data.sequence,id:data.id,expired:!!data.expired,
          source_timestamp:sourceTimes.get(data.sequence)??null,coverage:data.result?.coverage??null,
          elapsed_ms:performance.now()-started});
        if(Number.isFinite(data.compute_ms))computeTimings.push({operation:data.type==="present"&&data.expired?"present_expired":data.type,ms:data.compute_ms,
          expired:!!data.expired,source_timestamp:sourceTimes.get(data.sequence)??null,
          sequence:data.sequence,id:data.id,anchor_sequence:data.result?.anchor_sequence??null,
          anchor_age_seconds:data.result?.anchor_age_seconds??null,coverage:data.result?.coverage??null,
          elapsed_ms:performance.now()-started});
        if(Number.isFinite(data.queued_at))workerMs.push(performance.now()-data.queued_at);
      });
      const monitor=setInterval(()=>{
        const pending=controller.snapshot().worker_pending_frames;pendingSamples.push(pending);
        backlogSamples.push({source_timestamp:controller.frames.frames.at(-1)?.timestamp??0,
          elapsed_ms:performance.now()-started,pending_frames:pending});
      },25);
      controller.start();await sleep(seconds*1000);clearInterval(monitor);await controller.drain();
      if(controller.grayCanvas.width!==motionWidth)throw new Error("Controller did not honor requested motion_width");
      if(motionBackend==="dense"&&!backendReady.some(item=>item.backend==="dense"))throw new Error("Dense backend was not acknowledged by the real Worker");
      const snapshot=controller.snapshot();controller.stop();await sleep(50);
      const percentile=ContourQualityMetrics.percentile,ages=presented.map(frame=>frame.age_seconds);
      const gaps=presented.slice(1).map((frame,i)=>(frame.at-presented[i].at)/1000);
      const warmupSourceSeconds=syntheticOptions.trace?2:Math.max(2,anchorDelayMs/1000+1);
      const steady=presented.filter(frame=>frame.timestamp>=warmupSourceSeconds&&frame.timestamp<=seconds-2);
      const steadyResults=presentationResults.filter(row=>row.source_timestamp!==null&&row.source_timestamp>=warmupSourceSeconds&&row.source_timestamp<=seconds-2);
      const countCoverage=frames=>{const counts={};for(const frame of frames)counts[frame.coverage]=(counts[frame.coverage]||0)+1;return counts;};
      return {kind:"synthetic_canvas_captureStream_30fps_real_video_callback_real_worker",
        limitation:"Tests browser scheduling/compute only; generated polygons and texture are not real people or camera sensor timing.",
        duration_seconds:seconds,requested_display_fps:targetFPS,analysis_sample_fps:3,requested_display_delay_seconds:displayDelaySeconds,
        anchor_delay_ms:syntheticOptions.trace?null:anchorDelayMs,max_propagation_seconds:maxPropagationSeconds,warmup_source_seconds:warmupSourceSeconds,
        anchor_policy:syntheticOptions.anchor_policy,synthetic_anchor_hz:syntheticOptions.trace?null:syntheticOptions.anchor_hz,
        synthetic_anchor_trace:syntheticOptions.trace?{path:syntheticOptions.trace.path,sha256:syntheticOptions.trace.sha256,scope:syntheticOptions.trace.scope,
          planned_anchor_count:syntheticOptions.trace.frames.length,scheduled_anchor_count:anchorSchedule.length,
          metadata_seen_anchor_count:metadataSeen.size,
          note:"Each trace source time selects the first distinct captured synthetic bitmap at/after that time; its exact marker pose is used. The recorded source-to-result age delays availability from that capture. Actual Controller polling supplies metadata waiting; no extra 150 ms is added."}:null,
        anchor_delivery_schedule:anchorSchedule,
        motion_width:motionWidth,history_seconds:historySeconds,motion_backend:motionBackend,backend_ready:backendReady,
        anchor_geometry_binding:"Frame marker decoded from the captured bitmap identifies the exact generated pose.",
        steady_presented_frames:steady.length,steady_contour_frame_ratio:steady.length?steady.filter(frame=>frame.has_complete_contours).length/steady.length:null,
        steady_complete_contour_frame_ratio:steady.length?steady.filter(frame=>frame.has_complete_contours).length/steady.length:null,
        steady_any_contour_frame_ratio:steady.length?steady.filter(frame=>frame.has_contours||frame.has_partial_contours).length/steady.length:null,
        steady_partial_contour_frame_ratio:steady.length?steady.filter(frame=>frame.has_partial_contours).length/steady.length:null,
        steady_coverage_counts:countCoverage(steady),warmup_coverage_counts:countCoverage(presented.filter(frame=>frame.timestamp<warmupSourceSeconds)),
        worker_presentation_results:presentationResults.length,
        worker_expired_presentations:presentationResults.filter(row=>row.expired).length,
        steady_worker_presentation_results:steadyResults.length,
        steady_worker_expired_presentations:steadyResults.filter(row=>row.expired).length,
        worker_presentation_result_coverage_counts:countCoverage(presentationResults),
        steady_worker_presentation_result_coverage_counts:countCoverage(steadyResults),
        generated_frames:generated,distinct_presented:new Set(presented.map(frame=>frame.sequence)).size,
        duplicate_presentations:presented.length-new Set(presented.map(frame=>frame.sequence)).size,
        steady_distinct_fps:steady.length>1?(steady.length-1)*1000/(steady.at(-1).at-steady[0].at):null,
        source_callback_to_canvas_p50_seconds:percentile(ages,.5),source_callback_to_canvas_p95_seconds:percentile(ages,.95),
        max_present_gap_seconds:Math.max(...gaps),worker_pending_frames_max:Math.max(...pendingSamples),
        steady_worker_pending_frames_max:Math.max(0,...backlogSamples.filter(row=>row.source_timestamp>=warmupSourceSeconds&&row.source_timestamp<=seconds-2).map(row=>row.pending_frames)),
        worker_pending_frames_p95:percentile(pendingSamples,.95),worker_message_latency_p95_ms:percentile(workerMs,.95),
        worker_compute_measurement:computeTimings.length?"Formal ACK compute_ms around appendFrame/select; excludes initialization and queue wait. Expired requests are separate present_expired operations and never lower the valid present mean.":"Synchronous Worker handler wrapper; async initialization queue execution is excluded.",
        worker_compute:ContourQualityMetrics.summarizeWorkerCompute(computeTimings.length?computeTimings:handlerTimings),
        worker_anchor_compute:ContourQualityMetrics.summarizeAnchorCompute(computeTimings),
        worker_anchor_compute_note:"First non-expired present for each observed anchor versus later presents for the same anchor. Later queries may still retry a failed estimate; this grouping does not assume every query is a cache hit.",
        snapshot,failures,reports,presented,worker_operation_records:computeTimings,worker_presentation_records:presentationResults,worker_backlog_samples:backlogSamples,failure_cases:failureCases,dense_failure_cases:denseFailureCases};
    }finally{
      controller?.stop();window.fetch=originalFetch;clearInterval(drawTimer);stream.getTracks().forEach(track=>track.stop());video.srcObject=null;
    }
  },{seconds,targetFPS,anchorDelayMs,maxPropagationSeconds,motionWidth,historySeconds,motionBackend,displayDelaySeconds,syntheticOptions});
}
async function saveVisuals(page,quality){
  const names=[];
  for(const set of [...new Set(quality.rows.map(row=>row.set))].slice(0,2)){
    let candidates=quality.rows.filter(row=>row.set===set&&row.scenario!=="exact_reference_control"&&
      row.people.some(person=>person.predicted||person.partial)).sort((a,b)=>
      Math.max(...b.people.map(person=>person.line_boundary_p95_px||person.boundary_p95_chamfer_px||0))-
      Math.max(...a.people.map(person=>person.line_boundary_p95_px||person.boundary_p95_chamfer_px||0)));
    const focal=quality.rows.find(row=>row.set===set&&row.set==="early"&&row.sequence===94&&row.scenario==="sparse_gap_delayed_2s"&&row.people.some(person=>person.partial));
    if(focal)candidates=[focal,...candidates];
    if(!candidates.length)candidates=quality.rows.filter(row=>row.set===set&&row.coverage==="motion_unreliable");
    if(!candidates.length)continue;
    const row=candidates[0],png=await page.evaluate(async row=>{
      const image=await createImageBitmap(await(await fetch(`/image/${row.set}/${row.sequence}`)).blob());
      const canvas=document.createElement("canvas");canvas.width=image.width;canvas.height=image.height+65;
      const ctx=canvas.getContext("2d");ctx.fillStyle="#10181c";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,65);
      ctx.font="16px sans-serif";ctx.fillStyle="#fff";
      ctx.fillText(`${row.set} source ${row.sequence}, ${row.scenario}`,12,22);
      ctx.fillStyle="#00ff79";ctx.fillText("SAM reference",12,48);ctx.fillStyle="#ff39ef";
      ctx.fillText(row.people.some(person=>person.partial)?"Partial candidate (open lines)":row.people.some(person=>person.predicted)?"Propagated candidate":"No propagated contour",190,48);
      for(const person of row.people)for(const [key,color] of [["reference_polygons","#00ff79"],["predicted_polygons","#ff39ef"]]){
        ctx.strokeStyle=color;ctx.lineWidth=2;
        for(const polygon of person[key]){ctx.beginPath();polygon.forEach(([x,y],i)=>i?ctx.lineTo(x*(image.width-1),65+y*(image.height-1)):ctx.moveTo(x*(image.width-1),65+y*(image.height-1)));ctx.closePath();ctx.stroke();}
      }
      for(const person of row.people){ctx.strokeStyle="#ff39ef";ctx.lineWidth=2;
        for(const segment of person.open_segments||[]){ctx.beginPath();segment.forEach(([x,y],i)=>i?ctx.lineTo(x*(image.width-1),65+y*(image.height-1)):ctx.moveTo(x*(image.width-1),65+y*(image.height-1)));ctx.stroke();}}
      image.close();return canvas.toDataURL("image/png").split(",")[1];
    },row);
    const file=output.replace(/\.json$/i,"")+`-${set}-worst.png`;fs.writeFileSync(file,Buffer.from(png,"base64"));names.push(file);
  }
  return names;
}
async function main(){
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const browser=await chromium.launch({executablePath:args.chrome||"C:/Program Files/Google/Chrome/Application/chrome.exe",headless:true,
    args:["--autoplay-policy=no-user-gesture-required"]});
  try{
    const page=await browser.newPage({viewport:{width:960,height:800}}),errors=[];
    page.on("pageerror",error=>errors.push(error.message));
    await page.route("**/*",route=>new URL(route.request().url()).hostname==="127.0.0.1"?route.continue():route.abort());
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const maxPropagationSeconds=Number(args["max-propagation-seconds"]||2),anchorDelayMs=Number(args["anchor-delay-ms"]||250);
    const motionWidth=Number(args["motion-width"]||160),delays=args.delays?args.delays.split(",").map(Number):null;
    const historySeconds=Number(args["history-seconds"]||6);
    const displayDelaySeconds=Number(args["display-delay-seconds"]||1);
    assert.ok(Number.isFinite(displayDelaySeconds)&&displayDelaySeconds>0&&displayDelaySeconds<=2,"Display delay must be positive and at most two seconds for this experiment");
    const replayMode=args["replay-mode"]||"legacy";assert.ok(["legacy","buffered","trace"].includes(replayMode),"Replay mode must be legacy, buffered or trace");
    const modelDelays=(args["model-delays"]||"0.8,1.2").split(",").map(Number),cadenceHz=Number(args["anchor-cadence-hz"]||1);
    assert.ok(modelDelays.length&&modelDelays.every(delay=>Number.isFinite(delay)&&delay>=0&&delay<=5),"Model delays must be comma-separated seconds between zero and five");
    assert.ok(Number.isFinite(cadenceHz)&&cadenceHz>0&&cadenceHz<=3,"Anchor cadence must be positive and at most 3 Hz");
    const bufferedConfig=replayMode!=="legacy"?{mode:replayMode,model_delays:modelDelays,display_delay_seconds:displayDelaySeconds,cadence_hz:cadenceHz}:null;
    assert.ok([160,240,320].includes(motionWidth),"Motion width must be 160, 240 or 320 for comparable corpus tests");
    assert.ok(!delays||delays.every(lag=>Number.isFinite(lag)&&lag>0&&lag<=6),"Delays must be a comma-separated list of positive source seconds up to six");
    assert.ok(historySeconds>0&&historySeconds<=12,"History must be a positive duration up to twelve seconds");
    assert.ok(maxPropagationSeconds>0&&maxPropagationSeconds<=historySeconds,"Propagation experiment must remain within retained history");
    assert.ok(anchorDelayMs>=0&&anchorDelayMs<=5000,"Anchor delay must be between 0 and 5000 ms");
    console.error(JSON.stringify({stage:args.quality==="false"?"quality_skipped":"quality_started",motion_backend:motionBackend,motion_width:motionWidth}));
    const quality=args.quality==="false"?{summaries:[],rows:[]}:await runReplay(page,maxPropagationSeconds,args.scenarios||"sparse",motionWidth,delays,historySeconds,motionBackend,bufferedConfig);
    for(const summary of quality.summaries.filter(summary=>summary.scenario==="exact_reference_control")){
      assert.equal(summary.person_availability,1);assert.equal(summary.mean_mask_iou_all,1);
    }
    const report={schema:1,source_sha256:crypto.createHash("sha256").update(sourceBytes).digest("hex"),
      evaluator_sha256:crypto.createHash("sha256").update(evaluatorBytes).digest("hex"),
      metrics_sha256:crypto.createHash("sha256").update(metricsBytes).digest("hex"),
      source_path:source,browser_version:browser.version(),created_at:new Date().toISOString(),
      max_propagation_seconds:maxPropagationSeconds,
      motion_width:motionWidth,
      motion_backend:motionBackend,
      buffered_replay:bufferedConfig,
      buffered_replay_note:bufferedConfig?(replayMode==="trace"?"Events use each actual SAM availability_clock_seconds, unchanged source captures and buffered presentations. Supply anchors and evaluation references are separate; explicit cross-run reference correspondence is documented per fixture. Both policies receive identical arrived anchors; every used per-person geometry source is checked against its arrival time.":"Events use original source time: capture, SAM result arrival, then target presentation after the display buffer. Cadence chooses the first existing SAM reference at/after each scheduled tick and preserves its actual timestamp; no new SAM mask or exact 1 Hz frame is fabricated. Both policies receive identical arrived anchors; every used per-person geometry source is checked against its arrival time."):null,
      failure_capture:args["capture-failures"]==="true",
      auxiliary_assets:auxiliarySources.map(asset=>({route:asset.route,path:asset.path,sha256:crypto.createHash("sha256").update(asset.bytes).digest("hex")})),
      history_seconds:historySeconds,
      instrumentation:"Production source and local dense/vendor assets are frozen unchanged. Formal Worker ACK compute_ms is primary; test-only handler timing remains separately identified. Dense replay waits for motion_ready before RPCs; the realtime Controller exercises the actual initialization queue.",
      reference:"Corresponding recorded SAM normalized polygons, not human-labelled ground truth.",
      geometric_matching_note:"Native-ID IoU remains primary. Optimal one-to-one geometric matching is diagnostic only; a different matched ID does not establish which tracker switched a real person's identity.",
      limitations:["Only recorded JPEG source frames are replayed, at their original source timestamps.",
        "Sparse-gap scenarios permit 0.5 s gaps for offline stress evaluation; production remains 0.2 s.",
        "Synthetic anchor delivery delays are source time offsets, not measured model latencies.",
        "No 15 fps real-person display or end-to-end latency acceptance follows from 3 fps JPEGs.",
        "Boundary distances use a symmetric 8-neighbour chamfer approximation at the original JPEG resolution.",
        "Missing predictions contribute zero IoU/recall; undefined boundary errors are reported only for visible predictions."],
      fixture_provenance:fixtureProvenance,
      fixtures:sets.map(set=>({name:set.name,frames:set.sources.length,reference_frames:(set.reference_anchors||set.anchors).length,supply_anchors:set.anchors.length,
        reference_semantics:set.reference_semantics||null,reference_identity_mapping:set.reference_identity_mapping||null,
        first_timestamp:set.sources[0].timestamp,last_timestamp:set.sources.at(-1).timestamp,
        distinct_source_fps:(set.sources.length-1)/(set.sources.at(-1).timestamp-set.sources[0].timestamp)})),
      quality,errors};
    if(args["synthetic-seconds"]){
      console.error(JSON.stringify({stage:"synthetic_started",motion_backend:motionBackend,motion_width:motionWidth,anchor_delay_ms:anchorDelayMs}));
      report.synthetic=await runSynthetic(page,Number(args["synthetic-seconds"]),Number(args["target-fps"]||15),anchorDelayMs,maxPropagationSeconds,motionWidth,historySeconds,motionBackend,displayDelaySeconds,syntheticOptions);
    }
    fs.mkdirSync(path.dirname(output),{recursive:true});report.visuals=await saveVisuals(page,quality);
    fs.writeFileSync(output.replace(/\.json$/i,"")+"-source.js",sourceBytes);
    fs.writeFileSync(output.replace(/\.json$/i,"")+"-metrics.cjs",metricsBytes);
    for(const asset of auxiliarySources)fs.writeFileSync(output.replace(/\.json$/i,"")+asset.suffix,asset.bytes);
    fs.writeFileSync(output,JSON.stringify(report,null,2));
    const syntheticSummary=report.synthetic&&Object.fromEntries(Object.entries(report.synthetic)
      .filter(([key])=>!["failure_cases","dense_failure_cases","presented","reports","anchor_delivery_schedule","worker_operation_records","worker_presentation_records","worker_backlog_samples"].includes(key)));
    console.log(JSON.stringify({output,source_sha256:report.source_sha256,summaries:quality.summaries,
      worker_timings:quality.worker_timings,synthetic:syntheticSummary,errors},null,2));
    assert.equal(errors.length,0,"Browser emitted an error");
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>server.close());
