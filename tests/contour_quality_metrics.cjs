const {test}=require("node:test"),assert=require("node:assert/strict");
const {compareMasks,compareContourLines,bestIoUMatching,summarizeWorkerCompute,summarizeAnchorCompute,sampleAnchorsAtCadence,bufferedReplayEvents}=require("../tools/contour_quality_metrics.cjs");
test("nominal 1 Hz supply selects recorded anchors without inventing or duplicating endpoints",()=>{
  const anchors=[0,.34,.70,1.36,1.70,2.34,4.34,4.67,5.01].map((time,index)=>({source_sequence:index,media_timestamp:time}));
  const selected=sampleAnchorsAtCadence(anchors,1);
  assert.deepEqual(selected.map(anchor=>anchor.media_timestamp),[0,1.36,2.34,4.34,5.01]);
  assert.ok(selected.every(anchor=>anchors.includes(anchor)));assert.equal(new Set(selected).size,selected.length);
});
test("buffered replay exposes future anchors only after capture and model arrival",()=>{
  const sources=[0,.1,1].map((time,index)=>({sequence:index,timestamp:time})),anchors=[{source_sequence:2,media_timestamp:1}];
  const events=bufferedReplayEvents(sources,anchors,.8,1.7);
  const order=events.map(event=>`${event.type}:${event.time.toFixed(1)}`);
  assert.ok(order.indexOf("capture:1.0")<order.indexOf("deliver:1.8"));
  assert.ok(order.indexOf("present:1.7")<order.indexOf("deliver:1.8"));
  assert.ok(order.indexOf("deliver:1.8")<order.indexOf("present:1.8"));
});
test("recorded arrival overrides nominal model latency without moving source capture",()=>{
  const sources=[{sequence:1,timestamp:1}],anchors=[{source_sequence:1,media_timestamp:1,availability_clock_seconds:3.1}];
  const events=bufferedReplayEvents(sources,anchors,.8,1.7);
  assert.deepEqual(events.map(event=>[event.type,event.time]),[["capture",1],["present",2.7],["deliver",3.1]]);
  assert.throws(()=>bufferedReplayEvents(sources,[{...anchors[0],availability_clock_seconds:.9}],.8,1.7),/precedes/);
});
test("anchor startup statistics ignore expired requests and retain all later attempts",()=>{
  const result=summarizeAnchorCompute([
    {operation:"present_expired",expired:true,anchor_sequence:10,ms:0},
    {operation:"present",anchor_sequence:10,ms:100},
    {operation:"present",anchor_sequence:10,ms:10},
    {operation:"present",anchor_sequence:20,ms:200},
    {operation:"present",anchor_sequence:10,ms:20},
    {operation:"present",anchor_sequence:null,ms:1}
  ]);
  assert.equal(result.first_effective_anchor.count,2);assert.equal(result.first_effective_anchor.mean_ms,150);
  assert.equal(result.same_anchor_followup.count,2);assert.equal(result.same_anchor_followup.mean_ms,15);
});
test("expired drawing requests cannot lower the valid present compute mean",()=>{
  const result=summarizeWorkerCompute([
    {operation:"present",ms:20},{operation:"present",ms:40},
    ...Array.from({length:200},()=>({operation:"present",expired:true,ms:0})),
    {operation:"frame_processed",ms:10}
  ]);
  assert.equal(result.present.count,2);assert.equal(result.present.mean_ms,30);
  assert.equal(result.present_expired.count,200);assert.equal(result.present_expired.total_ms,0);
  assert.equal(result.frame_processed.mean_ms,10);
});
test("all-expired requests do not fabricate a zero-cost successful present",()=>{
  const result=summarizeWorkerCompute([{operation:"present_expired",expired:true,ms:0}]);
  assert.equal(result.present,undefined);assert.equal(result.present_expired.count,1);
});
const W=40,H=30;
function rect(x,y,w,h){const mask=new Uint8Array(W*H);for(let py=y;py<y+h;py++)for(let px=x;px<x+w;px++)mask[py*W+px]=1;return mask;}
test("open lines measure only their drawn part and never close the missing edge",()=>{
  const mask=rect(10,5,10,20),points=[[10/39,5/29],[19/39,5/29]],result=compareContourLines([points],mask,W,H);
  assert.equal(result.line_pixels,10);assert.equal(result.line_precision_3px,1);
  assert.ok(result.boundary_recall_3px<.5);assert.equal(result.line_boundary_p95_px,0);
});
test("a missing open contour has zero recall and undefined precision",()=>{
  const result=compareContourLines([],rect(10,5,10,20),W,H);
  assert.equal(result.line_pixels,0);assert.equal(result.boundary_recall_3px,0);assert.equal(result.line_precision_3px,null);
});
test("equal masks are perfect and have zero contour displacement",()=>{
  const mask=rect(10,5,10,20),result=compareMasks(mask,mask,W,H);
  assert.equal(result.mask_iou,1);assert.equal(result.reference_recall,1);
  assert.equal(result.boundary_p95_chamfer_px,0);assert.equal(result.centroid_offset_px,0);
  assert.deepEqual(result.reference_vertical_thirds_recall,[1,1,1]);
});
test("a two-pixel shift is measured independently of contour vertex order",()=>{
  const result=compareMasks(rect(12,5,10,20),rect(10,5,10,20),W,H);
  assert.equal(result.mask_iou,2/3);assert.equal(result.reference_recall,.8);
  assert.equal(result.prediction_precision,.8);assert.equal(result.centroid_offset_px,2);
  assert.equal(result.boundary_p95_chamfer_px,2);
});
test("missing prediction scores zero coverage without silently averaging as accurate",()=>{
  const result=compareMasks(new Uint8Array(W*H),rect(10,5,10,20),W,H);
  assert.equal(result.mask_iou,0);assert.equal(result.reference_recall,0);
  assert.equal(result.prediction_precision,null);assert.equal(result.boundary_mean_chamfer_px,null);
});
test("partial person loses lower-body completeness despite perfect precision",()=>{
  const result=compareMasks(rect(10,5,10,7),rect(10,5,10,20),W,H);
  assert.equal(result.prediction_precision,1);assert.equal(result.reference_recall,.35);
  assert.deepEqual(result.reference_vertical_thirds_recall,[1,0,0]);
});
test("IoU matching finds the global optimum when greedy matching fails",()=>{
  assert.deepEqual(bestIoUMatching([[.9,.8],[.85,0]]),[
    {reference_index:0,predicted_index:1,iou:.8},
    {reference_index:1,predicted_index:0,iou:.85},
  ]);
});
test("extra reference rows cannot reuse one predicted person",()=>{
  assert.deepEqual(bestIoUMatching([[.8],[.9],[.7]]),[
    {reference_index:1,predicted_index:0,iou:.9},
  ]);
});
test("extra predicted columns remain unmatched and inputs are preserved",()=>{
  const matrix=[[.7,.95,.6],[.9,.8,.4]],before=structuredClone(matrix);
  assert.deepEqual(bestIoUMatching(matrix),[
    {reference_index:0,predicted_index:1,iou:.95},
    {reference_index:1,predicted_index:0,iou:.9},
  ]);
  assert.deepEqual(matrix,before);
});
test("empty and zero-IoU inputs produce no fabricated matches",()=>{
  for(const matrix of [[],[[]],[[],[]],[[0,0],[0,0]],[[0,0,0]]])
    assert.deepEqual(bestIoUMatching(matrix),[]);
  assert.deepEqual(bestIoUMatching([[0,0],[0,1e-12]]),[
    {reference_index:1,predicted_index:1,iou:1e-12},
  ]);
});
test("equal weights still assign every reference and prediction at most once",()=>{
  const result=bestIoUMatching([[.5,.5],[.5,.5],[.5,.5]]);
  assert.equal(result.length,2);
  assert.equal(new Set(result.map(m=>m.reference_index)).size,2);
  assert.equal(new Set(result.map(m=>m.predicted_index)).size,2);
});
test("small rectangular assignments agree with an independent exhaustive oracle",()=>{
  function optimum(matrix,row=0,used=0){
    if(row===matrix.length)return 0;
    let best=optimum(matrix,row+1,used);
    for(let col=0;col<matrix[row].length;col++)if(!(used&(1<<col)))
      best=Math.max(best,matrix[row][col]+optimum(matrix,row+1,used|(1<<col)));
    return best;
  }
  let seed=417;
  for(let sample=0;sample<40;sample++){
    const rows=1+sample%4,cols=1+Math.floor(sample/4)%4,
      matrix=Array.from({length:rows},()=>Array.from({length:cols},()=>{
        seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed%11)/10;
      })),result=bestIoUMatching(matrix);
    assert.ok(Math.abs(result.reduce((sum,m)=>sum+m.iou,0)-optimum(matrix))<1e-12);
    assert.equal(new Set(result.map(m=>m.reference_index)).size,result.length);
    assert.equal(new Set(result.map(m=>m.predicted_index)).size,result.length);
    for(const match of result)assert.equal(match.iou,matrix[match.reference_index][match.predicted_index]);
  }
});
test("64-person assignments are supported",()=>{
  const result=bestIoUMatching(Array.from({length:64},(_,row)=>
    Array.from({length:64},(_,col)=>row===col?1:.5)));
  assert.equal(result.length,64);
  for(const match of result){assert.equal(match.reference_index,match.predicted_index);assert.equal(match.iou,1);}
});
test("invalid shapes, scores, and oversized matrices are rejected",()=>{
  for(const matrix of [null,[null],[[0],[0,1]],[[NaN]],[[Infinity]],[[-.1]],[[1.1]],
    [[".5"]],Array(2),[Array(2)],Array.from({length:65},()=>[0]),[Array(65).fill(0)]])
    assert.throws(()=>bestIoUMatching(matrix),/invalid IoU matrix/);
});
