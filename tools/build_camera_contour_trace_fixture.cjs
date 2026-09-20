#!/usr/bin/env node
// Build a private geometry-only fixture from an actual SAM arrival trace.
"use strict";
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto"),assert=require("node:assert/strict");
const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,"")]=process.argv[i+1];
assert.ok(args.fixtures&&args.trace&&args.output,"Expected --fixtures dir --trace actual-report.json --output private-fixture.json");
const folder=path.resolve(args.fixtures),metadataDelay=Number(args["metadata-delay-seconds"]||.15);
assert.ok(Number.isFinite(metadataDelay)&&metadataDelay>=0);
const read=file=>JSON.parse(fs.readFileSync(file,"utf8").replace(/^\uFEFF/,"")),hash=file=>crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const sourceFile=path.join(folder,"early-source-images.json"),referenceFile=path.join(folder,"early-sam-results.json"),traceFile=path.resolve(args.trace);
const sources=read(sourceFile).sort((a,b)=>a.timestamp-b.timestamp),bySequence=new Map(sources.map(frame=>[frame.sequence,frame])),trace=read(traceFile);
const observations=trace.observations.filter(observation=>bySequence.has(observation.sequence));
assert.ok(observations.length>=2,"Not enough actual SAM observations overlap retained images");
const identities=new Map(),streams=new Set(),anchors=observations.map(observation=>{
  const frame=bySequence.get(observation.sequence),native=observation.native;
  assert.equal(native.frame_sequence,frame.sequence);assert.ok(Math.abs(native.timestamp_seconds-frame.timestamp)<1e-6);
  assert.ok(Math.abs(observation.timestamp-frame.timestamp)<1e-6);
  assert.ok(Math.abs(observation.timestamp+observation.source_to_result_seconds-observation.arrival_elapsed_seconds)<1e-5,"Trace/source clocks differ");
  const people=native.objects.filter(object=>object.label==="person");
  assert.equal(people.length,1,"Explicit single-person correspondence is not valid for this trace");
  const person=people[0],id=String(person.attributes.native_track.object_id),epoch=observation.epoch;
  assert.ok(person.attributes.mask_contours.polygons.length);identities.set(`${epoch}:${id}`,{epoch,track_id:id});streams.add(native.stream_session_id);
  return {source_sequence:frame.sequence,media_timestamp:frame.timestamp,epoch,
    availability_clock_seconds:observation.arrival_elapsed_seconds+metadataDelay,
    sam_arrival_elapsed_seconds:observation.arrival_elapsed_seconds,sam_source_to_result_seconds:observation.source_to_result_seconds,
    metadata_delay_seconds:metadataDelay,subjects:[{track_id:id,polygons:person.attributes.mask_contours.polygons,
      tracking_confirmation:person.attributes.tracking_confirmation||"confirmed"}]};
});
assert.equal(identities.size,1,"More than one actual SAM identity/epoch; manual correspondence is required");assert.equal(streams.size,1);
const identity=[...identities.values()][0],mapping=new Map();
const referenceAnchors=read(referenceFile).flatMap(row=>row.result.frames.filter(frame=>bySequence.has(frame.frame_sequence)).map(frame=>{
  const people=frame.objects.filter(object=>object.label==="person");assert.equal(people.length,1,"Old reference is not single-person");
  assert.ok(Math.abs(frame.timestamp_seconds-bySequence.get(frame.frame_sequence).timestamp)<1e-6);
  const person=people[0],original={epoch:row.result.predictor_epoch,track_id:String(person.attributes.native_track.object_id)};
  mapping.set(`${original.epoch}:${original.track_id}`,{reference_identity:original,evaluation_identity:identity});
  return {source_sequence:frame.frame_sequence,media_timestamp:frame.timestamp_seconds,epoch:identity.epoch,
    original_reference_identity:original,subjects:[{track_id:identity.track_id,polygons:person.attributes.mask_contours.polygons,tracking_confirmation:"confirmed"}]};
}));
const semantics="Geometry-only comparison across independent SAM runs. Both each old reference and each actual new anchor contain exactly one person on the same retained source imagery; the explicitly recorded single-person correspondence is used only for evaluation. Production History receives the new run's real epoch/native ID. This does not establish native identity continuity across runs.";
const manifest={schema:1,run:{stream_session_id:[...streams][0],generation:1,crop_version:0},
  provenance:{trace_path:traceFile,trace_sha256:hash(traceFile),source_sha256:hash(sourceFile),reference_sha256:hash(referenceFile),
    metadata_delay_seconds:metadataDelay,metadata_delay_is_assumed:true,trace_scope:trace.scope},
  sets:[{name:"early",sources,anchors,reference_anchors:referenceAnchors,reference_semantics:semantics,reference_identity_mapping:[...mapping.values()]}]};
fs.mkdirSync(path.dirname(path.resolve(args.output)),{recursive:true});fs.writeFileSync(args.output,JSON.stringify(manifest));
console.log(JSON.stringify({output:path.resolve(args.output),sources:sources.length,actual_anchors:anchors.length,old_reference_frames:referenceAnchors.length,
  real_epoch:identity.epoch,real_track_id:identity.track_id,metadata_delay_seconds:metadataDelay,reference_semantics:semantics}));
