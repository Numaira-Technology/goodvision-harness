#!/usr/bin/env node
/* Offline aggregation only; reads existing private browser reports, no model/browser. */
"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
const {percentile}=require("./contour_quality_metrics.cjs");
function summarizeRows(report){
  const groups=new Map(),motionWidth=report.motion_width||160;
  for(const row of report.quality.rows){
    if(row.coverage==="no_anchor"||row.scenario==="exact_reference_control")continue;
    for(const person of row.people){
      if(!person.reference_polygons)continue;
      const xs=person.reference_polygons.flat().map(point=>point[0]);
      const width=(Math.max(...xs)-Math.min(...xs))*(motionWidth-1);
      const bin=width<4?"under_4":width<8?"4_to_under_8":width<12?"8_to_under_12":"12_or_more";
      const key=JSON.stringify([row.set,row.scenario,bin]);
      if(!groups.has(key))groups.set(key,{set:row.set,scenario:row.scenario,gray_bbox_width_bin:bin,people:[]});
      groups.get(key).people.push({...person,width,coverage:row.coverage});
    }
  }
  return [...groups.values()].map(({people,...group})=>{
    const visible=people.filter(person=>person.predicted),mean=(values)=>values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
    const boundaries=visible.map(person=>person.boundary_p95_chamfer_px).filter(value=>value!==null);
    return {...group,reference_observations:people.length,predicted_observations:visible.length,
      availability:visible.length/people.length,mean_gray_bbox_width:mean(people.map(person=>person.width)),
      mean_native_iou_all:mean(people.map(person=>person.mask_iou||0)),mean_native_iou_visible:mean(visible.map(person=>person.mask_iou)),
      mean_reference_recall_all:mean(people.map(person=>person.reference_recall||0)),
      p95_boundary_chamfer_px_visible:percentile(boundaries,.95),
      mean_geometry_matched_iou_all:people.some(person=>person.geometry_matched_iou!==undefined)?
        mean(people.map(person=>person.geometry_matched_iou||0)):null};
  });
}
if(require.main===module){
  const input=process.argv[2],output=process.argv[3];assert.ok(input,"Pass a saved browser report JSON and optional output JSON");
  const report=JSON.parse(fs.readFileSync(input,"utf8").replace(/^\uFEFF/,""));
  const result={input:path.resolve(input),source_sha256:report.source_sha256,motion_width:report.motion_width||160,
    note:"Reference bbox width at motion resolution; excludes no-anchor warmup and exact-reference controls. Width alone does not guarantee a sufficiently wide textured interior. These remain SAM references, not human truth.",
    groups:summarizeRows(report)};
  if(output)fs.writeFileSync(output,JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
}
module.exports={summarizeRows};
