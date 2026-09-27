#!/usr/bin/env node
/* Private CPU replay, with no browser, network or model calls.
 * node tools/replay_camera_motion.cjs --trace <diagnostic-dir> --output <private.json>
 *   [--from-source 12 --to-source 16] [--from-trace 0 --to-trace 2000]
 *   [--source local-camera-display.js] [--dense-source local-dense.js]
 *   [--opencv-source local-opencv.js] [--include-results true] [--validate-only true]
 *   [--motion-width 160]  Explicit non-exact CPU-resized-gray comparison only.
 * Ranges select reported presentation queries. Every prerequisite input from
 * trace ID zero is replayed, so caches, identity barriers and expiry stay intact.
 * Wait for the live run to end before starting CPU replay. --validate-only true
 * checks finalized files, offsets, paired-prefix integrity and deployment hashes
 * without initializing OpenCV. An incomplete stopped tail does not invalidate
 * its earlier contiguous paired prefix; missing inputs in that prefix are fatal.
 * Example comparison: repeat the same range with
 *   --dense-source src/goodvision_harness/camera_display_dense.js
 * The output records asset hashes and overrides. matching_responses excludes
 * wall-clock compute/queue durations, but compares all geometric response fields
 * exactly by default. --float-tolerance is an explicit numeric comparison option.
 * all_people_complete_ratio uses nonexpired queries with SAM-observed people;
 * it is neither ground-truth recall nor the actual displayed-frame coverage.
 * attempts can include a failed first candidate followed by successful recovery.
 */
"use strict";
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto"),assert=require("node:assert/strict");
const hash=bytes=>crypto.createHash("sha256").update(bytes).digest("hex");
const json=file=>JSON.parse(fs.readFileSync(file,"utf8").replace(/^\uFEFF/,""));
function localFile(directory,name){
  assert.equal(path.basename(name),name,"Trace asset paths must be local basenames");return path.join(directory,name);
}
function loadTrace(directory,overrides={}){
  directory=path.resolve(directory);
  const summaryFile=path.join(directory,"motion-trace-summary.json");
  assert.ok(fs.existsSync(summaryFile),"Trace is not finalized: wait for motion-trace-summary.json before replay");
  const summary=json(summaryFile);
  assert.equal(summary.schema,1);assert.equal(summary.provenance?.source_pixels,"browser_preprocess_exact");
  assert.equal(summary.provenance?.ordered_worker_input,true);
  const eventFile=localFile(directory,summary.files.events),pixelFile=localFile(directory,summary.files.gray),
    eventBytes=fs.readFileSync(eventFile),pixels=fs.readFileSync(pixelFile),inputs=[],outputs=new Map();
  for(const line of eventBytes.toString("utf8").trim().split(/\r?\n/).filter(Boolean)){
    const row=JSON.parse(line),message=row.message,id=message?.trace_id;
    assert.ok(Number.isSafeInteger(id)&&id>=0,"Every recorded Worker message must have an exact trace ID");
    if(row.type==="motion_trace_input"){
      assert.equal(id,inputs.length,"Missing, duplicate or reordered input; no exact replay is possible across this gap");
      assert.ok(["frame","metadata","present"].includes(message.type),"Unknown Worker input");
      if(message.type==="frame"){
        const frame=message.frame;
        assert.ok(Number.isSafeInteger(frame.gray_offset)&&frame.gray_offset>=0);
        assert.equal(frame.gray_length,frame.width*frame.height);
        assert.ok(frame.gray_offset+frame.gray_length<=pixels.length,"Recorded pixels are truncated");
      }
      inputs.push(message);
    }else{
      assert.equal(row.type,"motion_trace_output");assert.ok(!outputs.has(id),"Duplicate Worker response");outputs.set(id,message);
    }
  }
  let prefix=0;while(prefix<inputs.length&&outputs.has(prefix))prefix++;
  const assets={};
  for(const [key,name] of [["source","camera_display.js"],["dense","camera_display_dense.js"],["opencv","opencv-4.13.0.js"]]){
    const saved=summary.provenance.source_assets?.[name],file=overrides[key]?path.resolve(overrides[key]):
      saved?.file&&localFile(directory,saved.file);
    assert.ok(file&&fs.existsSync(file),`Missing saved ${name}; supply an explicit asset override`);
    const sha256=hash(fs.readFileSync(file));
    if(!overrides[key])assert.equal(sha256,saved.sha256,`Saved ${name} differs from the recorded deployment`);
    assets[key]={file,sha256,override:!!overrides[key],recorded_sha256:saved?.sha256||null};
  }
  return {summary,inputs,outputs,pixels,prefix,assets,provenance:{trace_directory:directory,
    summary_sha256:hash(fs.readFileSync(summaryFile)),events_sha256:hash(eventBytes),gray_sha256:hash(pixels),
    recorded_complete:summary.complete,recorded_reason:summary.reason,paired_prefix_messages:prefix,
    unpaired_tail_messages:inputs.length-prefix,assets}};
}
function selected(input,range){return input.type==="present"&&input.trace_id>=range.fromTrace&&input.trace_id<=range.toTrace&&
  input.timestamp>=range.fromSource&&input.timestamp<=range.toSource;}
function firstDifference(a,b,at="$",tolerance=0){
  if(typeof a==="number"&&typeof b==="number"&&Math.abs(a-b)<=tolerance)return null;
  if(Object.is(a,b))return null;
  if(!a||!b||typeof a!=="object"||typeof b!=="object")return {path:at,expected:a,actual:b};
  const keys=[...new Set([...Object.keys(a),...Object.keys(b)])].sort();
  for(const key of keys){
    if(!Object.hasOwn(a,key)||!Object.hasOwn(b,key))return {path:at+"."+key,expected:a[key],actual:b[key]};
    const different=firstDifference(a[key],b[key],at+"."+key,tolerance);if(different)return different;
  }
  return null;
}
function normalizeOutput(message){
  const keys={frame_processed:["type","sequence","accepted","expired"],metadata_processed:["type","accepted","rejected","changed","latest_anchor_timestamp"],
    present:["type","id","sequence","expired","result"],failure:["type","id","sequence","stage"]}[message.type];
  assert.ok(keys,`Unsupported recorded response: ${message.type}`);
  return JSON.parse(JSON.stringify(Object.fromEntries(keys.filter(key=>Object.hasOwn(message,key)).map(key=>[key,message[key]]))));
}
function shapeSize(shape,width,height){
  const points=shape.flat();if(!points.length)return null;
  return {width_pixels:(Math.max(...points.map(point=>point[0]))-Math.min(...points.map(point=>point[0])))*(width-1),
    height_pixels:(Math.max(...points.map(point=>point[1]))-Math.min(...points.map(point=>point[1])))*(height-1)};
}
function resizeGray(frame,width){
  const height=Math.max(16,Math.round(frame.height*width/frame.width)),pixels=new Uint8Array(width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const sx=Math.max(0,Math.min(frame.width-1,(x+.5)*frame.width/width-.5)),
      sy=Math.max(0,Math.min(frame.height-1,(y+.5)*frame.height/height-.5)),
      left=Math.floor(sx),top=Math.floor(sy),right=Math.min(frame.width-1,left+1),bottom=Math.min(frame.height-1,top+1),
      fx=sx-left,fy=sy-top,p=frame.pixels;
    pixels[y*width+x]=Math.round(p[top*frame.width+left]*(1-fx)*(1-fy)+p[top*frame.width+right]*fx*(1-fy)+
      p[bottom*frame.width+left]*(1-fx)*fy+p[bottom*frame.width+right]*fx*fy);
  }
  return {...frame,width,height,pixels};
}
function replayTrace(trace,{History,adapter,range={},includeResults=false,tolerance=0,motionWidth=null}={}){
  range={fromTrace:0,toTrace:Infinity,fromSource:0,toSource:Infinity,...range};
  const targets=trace.inputs.slice(0,trace.prefix).filter(input=>selected(input,range));
  assert.ok(targets.length,"No paired presentation queries match this range");
  const lastId=targets.at(-1).trace_id,identities=new WeakMap(),tags=new WeakMap(),rows=[],attempts=[],mismatches=[],rejections={};
  let active=null,compared=0,matching=0;
  // These wrappers observe the unchanged adapter and never replace geometry,
  // flow, masks, quality gates or failed-attempt recovery.
  for(const method of ["warpPolygons","warpChain","warpPresentation","warpPresentationChain"]){
    const original=adapter[method];if(!original)continue;
    adapter[method]=function(shape,input){
      const motions=method.endsWith("Chain")?input:[input],last=motions.at(-1),identity=tags.get(shape),result=original.call(adapter,shape,input);
      if(result&&identity)tags.set(result,identity);
      if(active&&last&&!result){
        const reason=last.last_warp_rejection||"unspecified";rejections[reason]=(rejections[reason]||0)+1;
        attempts.push({trace_id:active.trace_id,target_sequence:active.sequence,method,reason,identity:identity||null,
          source:identities.get(motions[0].previous)||null,target:identities.get(last.current)||null,
          input_shape_size:shapeSize(shape,last.width,last.height),chain_steps:motions.length,
          diagnostics:structuredClone(last.last_warp_diagnostics)});
      }
      return result;
    };
  }
  const history=new History({...trace.summary.config,...(motionWidth?{motion_width:motionWidth}:{})},adapter);
  for(const input of trace.inputs){
    if(input.trace_id>lastId)break;
    const expected=trace.outputs.get(input.trace_id);let actual;
    active=input.type==="present"&&selected(input,range)?input:null;
    try{
      if(input.type==="frame"&&expected.expired){
        // Admission deadlines were evaluated live. Retired pixels must stay
        // absent from replay history, even though their trace bytes exist.
        actual={type:"frame_processed",sequence:input.frame.sequence,accepted:false,expired:true};
      }else if(input.type==="frame"){
        const {gray_offset,gray_length,...recorded}=input.frame;let frame=recorded;
        frame.pixels=new Uint8Array(trace.pixels.buffer,trace.pixels.byteOffset+gray_offset,gray_length);
        if(motionWidth&&motionWidth!==frame.width)frame=resizeGray(frame,motionWidth);
        identities.set(frame.pixels,{sequence:frame.sequence,timestamp:frame.timestamp});
        actual={type:"frame_processed",sequence:frame.sequence,accepted:history.appendFrame(frame)};
      }else if(input.type==="metadata"){
        const rejected=history.rejected,revision=history.revision,accepted=history.acceptMetadata(input.data);
        actual={type:"metadata_processed",accepted,rejected:history.rejected-rejected,changed:history.revision!==revision,
          latest_anchor_timestamp:accepted&&history.anchors.size?Math.max(...[...history.anchors.values()].map(anchor=>anchor.media_timestamp)):null};
        for(const anchor of history.anchors.values())for(const subject of anchor.subjects)
          tags.set(subject.polygons,{epoch:anchor.epoch,track_id:subject.track_id,anchor_sequence:anchor.source_sequence});
      }else if(expected.type==="present"&&expected.expired){
        // Deadlines were evaluated in the real Worker. Date.now() during an
        // offline run would expire every query and corrupt subsequent caches.
        actual={type:"present",id:input.id,sequence:input.sequence,expired:true,
          result:{subjects:[],coverage:"presentation_expired"}};
      }else actual={type:"present",id:input.id,sequence:input.sequence,result:history.select(input.sequence,input.timestamp)};
    }catch(error){actual={type:"failure",id:input.id,sequence:input.sequence};}
    const original=normalizeOutput(expected),replayed=normalizeOutput(actual),difference=firstDifference(original,replayed,"$",tolerance);
    compared++;if(!difference)matching++;else mismatches.push({trace_id:input.trace_id,type:input.type,
      selected:!!active,difference});
    if(active){
      const past=[...history.anchors.values()].filter(anchor=>anchor.source_sequence<=input.sequence&&anchor.media_timestamp<=input.timestamp)
        .sort((a,b)=>b.source_sequence-a.source_sequence)[0],subjects=actual.result?.subjects||[],expectedPeople=past?.subjects||[],
        expired=actual.expired===true,complete=!expired&&expectedPeople.length>0&&actual.result?.coverage==="anchored"&&
          subjects.length===expectedPeople.length&&subjects.every(subject=>subject.complete!==false);
      rows.push({trace_id:input.trace_id,sequence:input.sequence,timestamp:input.timestamp,
        expired,coverage:actual.result?.coverage||"failure",expected_people:past?expectedPeople.length:null,
        drawn_people:subjects.length,all_people_complete:complete,
        partial_people:!expired&&subjects.length>0&&!complete,
        missing_people:expectedPeople.filter(person=>!subjects.some(subject=>subject.track_id===person.track_id)).map(person=>({
          epoch:past.epoch,track_id:person.track_id,size:shapeSize(person.polygons,history.lastFrame.width,history.lastFrame.height)})),
        original_response_matches:!difference,rejections:actual.result?.motion_rejections||{},
        ...(includeResults?{original_result:expected.result,replayed_result:actual.result}:{})});
    }
  }
  const evaluated=rows.filter(row=>!row.expired),withPeople=evaluated.filter(row=>row.expected_people>0),complete=withPeople.filter(row=>row.all_people_complete).length;
  return {schema:1,provenance:{...trace.provenance,replay_mode:motionWidth?"counterfactual_resized_gray":"recorded_pixels",
      ...(motionWidth?{transformation:{width:motionWidth,method:"CPU bilinear pixel-center downsampling",
        limitation:"Not the original browser GPU resize. Original expired queries stay expired; this does not estimate regained live FPS."}}:{})},
    config:history.config||trace.summary.config,range,
    semantics:"Presentation query coverage, not unique painted-frame FPS or manually labelled accuracy. All prerequisite inputs are replayed from trace zero. Recorded expired queries skip selection. Rejected attempts can recover using another valid anchor.",
    summary:{replayed_inputs:compared,matching_responses:matching,mismatching_responses:mismatches.length,
      selected_queries:rows.length,expired_queries:rows.length-evaluated.length,queries_with_observed_people:withPeople.length,
      all_people_complete_queries:complete,all_people_complete_ratio:withPeople.length?complete/withPeople.length:null,
      partial_people_queries:withPeople.filter(row=>row.partial_people).length,
      zero_people_drawn_queries:withPeople.filter(row=>!row.drawn_people).length,
      empty_observed_scene_queries:evaluated.filter(row=>row.expected_people===0).length,rejected_attempts:rejections},
    rows,attempts,mismatches};
}
async function loadRuntime(file){
  const cv=require(file);
  if(!cv.Mat)await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error("Saved OpenCV did not initialize")),15000);
    cv.onRuntimeInitialized=()=>{clearTimeout(timer);resolve();};
  });
  return {cv}; // Do not assimilate the legacy OpenCV self-returning thenable.
}
async function main(argv=process.argv.slice(2)){
  const args={};for(let index=0;index<argv.length;index+=2){assert.ok(argv[index].startsWith("--")&&argv[index+1]);args[argv[index].slice(2)]=argv[index+1];}
  assert.ok(args.trace&&args.output,"--trace directory and --output private.json are required");
  const trace=loadTrace(args.trace,{source:args.source,dense:args["dense-source"],opencv:args["opencv-source"]}),range={};
  for(const [flag,key] of [["from-source","fromSource"],["to-source","toSource"],["from-trace","fromTrace"],["to-trace","toTrace"]]){
    if(args[flag]!==undefined){range[key]=Number(args[flag]);assert.ok(Number.isFinite(range[key])&&range[key]>=0);}
  }
  let result;
  if(args["validate-only"]==="true")result={validated:true,provenance:trace.provenance,inputs:trace.inputs.length,paired_prefix:trace.prefix};
  else{
    const {cv}=await loadRuntime(trace.assets.opencv.file),{ContourHistory}=require(trace.assets.source.file),
      {createDenseMotionAdapter}=require(trace.assets.dense.file);
    const motionWidth=args["motion-width"]===undefined?null:Number(args["motion-width"]),tolerance=Number(args["float-tolerance"]||0);
    assert.ok(motionWidth===null||Number.isInteger(motionWidth)&&motionWidth>=16&&motionWidth<=320);
    assert.ok(Number.isFinite(tolerance)&&tolerance>=0);
    result=replayTrace(trace,{History:ContourHistory,adapter:createDenseMotionAdapter(cv),range,
      includeResults:args["include-results"]==="true",tolerance,motionWidth});
  }
  fs.mkdirSync(path.dirname(path.resolve(args.output)),{recursive:true});fs.writeFileSync(args.output,JSON.stringify(result));
  console.log(JSON.stringify({output:path.resolve(args.output),summary:result.summary||{validated:true,paired_prefix:trace.prefix}}));
}
module.exports={loadTrace,replayTrace,normalizeOutput,firstDifference,resizeGray,main};
if(require.main===module)main().catch(error=>{console.error(error.message);process.exitCode=1;});
