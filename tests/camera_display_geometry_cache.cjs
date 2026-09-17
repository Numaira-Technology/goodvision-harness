const assert=require("node:assert/strict"),{test}=require("node:test");
const {ContourHistory}=require("../src/goodvision_harness/camera_display.js");
const run={stream_session_id:"cache-run",generation:1,crop_version:0};
const shape=[[[.3,.3],[.6,.3],[.6,.6],[.3,.6]]];
const moved=(source,dx)=>source.map(polygon=>polygon.map(([x,y])=>[x+dx,y]));
function fixture({times=[0,.15,.2,.3,.4],options={},reject=false}={}){
  const calls={warps:0,approximate:0,reject},presentations=new WeakMap();
  const warp=(source,motions)=>{
    calls.warps++;
    if(calls.reject){motions.at(-1).last_warp_rejection="boundary_occluded";return null;}
    return moved(source,motions.reduce((sum,motion)=>sum+(motion.current[0]-motion.previous[0])/100,0));
  };
  const adapter={estimateMotion:(previous,current)=>({previous,current}),
    reverseMotion:motion=>({previous:motion.current,current:motion.previous}),
    warpPolygons:(source,motion)=>warp(source,[motion]),warpChain:warp,
    warpApproximateChain:source=>{calls.approximate++;const result=moved(source,0);
      presentations.set(result,{complete:false,approximate:true,polygons:result,open_segments:[],reason:"boundary_occluded"});
      return result;},
    getPresentation:value=>presentations.get(value)||{complete:true,polygons:value,open_segments:[]}};
  const history=new ContourHistory({...run,anchor_policy:"nearest_arrived",max_propagation_seconds:1.5,...options},adapter);
  times.forEach((timestamp,sequence)=>history.appendFrame({sequence,timestamp,width:16,height:16,
    pixels:new Uint8Array(256).fill(sequence)}));
  const anchor=(sequence,{subjects,polygons,epoch="epoch",confirmation="confirmed"}={})=>({
    source_sequence:sequence,media_timestamp:times[sequence],epoch,
    subjects:subjects||[{track_id:"person",polygons:polygons||moved(shape,sequence/100),tracking_confirmation:confirmation}]});
  const metadata=(anchors=[],events=[],extra={})=>history.acceptMetadata({...run,anchors,events,...extra});
  return {history,calls,anchor,metadata};
}
const event=(revision,phase="confirmed",from=0)=>({epoch:"epoch",track_id:"person",event_label:"smoking",
  event_id:"episode",revision,phase,valid_from_timestamp:from,valid_until_timestamp:null});

test("strict backward geometry is reused for repeated targets, distant anchors and event-only updates",()=>{
  const f=fixture();f.metadata([f.anchor(0),f.anchor(2)]);
  const first=f.history.select(1,.15);assert.equal(first.geometry_cache_hit,false);
  assert.equal(first.subjects[0].method,"image_motion_backward");assert.equal(f.calls.warps,1);
  for(let i=0;i<4;i++)assert.equal(f.history.select(1,.15).geometry_cache_hit,true);
  f.metadata([f.anchor(4)]);assert.equal(f.history.select(1,.15).geometry_cache_hit,true);
  f.metadata([],[event(1,"confirmed",.1)]);
  const colored=f.history.select(1,.15);assert.equal(colored.geometry_cache_hit,true);assert.equal(colored.subjects[0].alert,true);
  assert.deepEqual(colored.subjects[0].event_labels,["smoking"]);
  assert.deepEqual(first.subjects[0].event_labels,[],"labels must not mutate a cached or previously delivered result");
  assert.equal(first.subjects[0].alert,false,"recoloring must not mutate previous results");
  f.metadata([],[event(2,"ended",.15)]);
  assert.equal(f.history.select(1,.15).subjects[0].alert,false);assert.equal(f.calls.warps,1);
  assert.deepEqual(f.history.select(1,.15).subjects[0].event_labels,[]);
  assert.deepEqual(f.history.strictSuccesses.get('["epoch","person"]'),[{sequence:1,timestamp:.15}]);
});

test("future geometry correction at the same source invalidates memo despite unchanged sequence",()=>{
  const f=fixture();f.metadata([f.anchor(0),f.anchor(2)]);f.history.select(1,.15);
  f.metadata([f.anchor(2,{polygons:moved(shape,.12)})]);
  const corrected=f.history.select(1,.15);assert.equal(corrected.geometry_cache_hit,false);
  assert.equal(f.calls.warps,2);assert.ok(Math.abs(corrected.subjects[0].polygons[0][0][0]-.41)<1e-9);
  assert.equal(f.history.select(1,.15).geometry_cache_hit,true);
});

test("same-source past confirmation updates invalidate identity without borrowing future confirmation",()=>{
  const f=fixture();f.metadata([f.anchor(0,{confirmation:"provisional"}),f.anchor(2)]);
  f.metadata([],[event(1)]);const first=f.history.select(1,.15);
  assert.equal(first.subjects[0].tracking_confirmation,"provisional");assert.equal(first.subjects[0].alert,false);
  f.metadata([f.anchor(0)]);const next=f.history.select(1,.15);
  assert.equal(next.geometry_cache_hit,false);assert.equal(next.subjects[0].tracking_confirmation,"confirmed");
  assert.equal(next.subjects[0].alert,true);
});

test("an earlier missing-person or epoch future barrier invalidates the chosen future memo",()=>{
  for(const barrier of [{subjects:[]},{epoch:"new-epoch"},{subjects:[{track_id:"other",polygons:shape}]}]){
    const f=fixture({times:[0,.15,.16,.2]});f.metadata([f.anchor(0),f.anchor(3)]);
    assert.equal(f.history.select(1,.15).subjects[0].anchor_sequence,3);
    f.metadata([f.anchor(2,barrier)]);const result=f.history.select(1,.15);
    assert.equal(result.geometry_cache_hit,false);assert.equal(result.subjects[0].anchor_sequence,0);
    assert.equal(result.subjects[0].method,"image_motion");
  }
});

test("new nearest past and future observations replace memo and exact target observation wins",()=>{
  const f=fixture({times:[0,.1,.15,.16,.2]});f.metadata([f.anchor(0),f.anchor(4)]);
  assert.equal(f.history.select(2,.15).subjects[0].anchor_sequence,4);
  f.metadata([f.anchor(3)]);let result=f.history.select(2,.15);
  assert.equal(result.geometry_cache_hit,false);assert.equal(result.subjects[0].anchor_sequence,3);
  f.metadata([f.anchor(1)]);result=f.history.select(2,.15);
  assert.equal(result.geometry_cache_hit,false);assert.equal(result.subjects[0].anchor_sequence,3);
  f.metadata([f.anchor(2,{polygons:moved(shape,.2)})]);result=f.history.select(2,.15);
  assert.equal(result.geometry_cache_hit,false);assert.equal(result.exact,true);
  assert.deepEqual(result.subjects[0].polygons,moved(shape,.2));
});

test("early empty and failed geometry stay retryable rather than becoming negative memo entries",()=>{
  const f=fixture();assert.equal(f.history.select(1,.15).geometry_cache_hit,undefined);
  f.metadata([f.anchor(0),f.anchor(2)]);assert.equal(f.history.select(1,.15).geometry_cache_hit,false);
  const failing=fixture({reject:true});failing.metadata([failing.anchor(0),failing.anchor(2)]);
  assert.equal(failing.history.select(1,.15).subjects.length,0);const count=failing.calls.warps;
  assert.equal(failing.history.select(1,.15).geometry_cache_hit,false);
  assert.ok(failing.calls.warps>count);assert.equal(failing.history.geometryCache.size,0);
});

test("retiring an anchor or target removes memo and a new run never shares it",()=>{
  const f=fixture({options:{max_history_seconds:1}});f.metadata([f.anchor(0),f.anchor(2)]);
  f.history.select(1,.15);assert.equal(f.history.geometryCache.size,1);
  f.history.appendFrame({sequence:5,timestamp:1.12,width:16,height:16,pixels:new Uint8Array(256)});
  assert.equal(f.history.frames.has(1),true);assert.equal(f.history.geometryCache.size,0);
  assert.equal(f.history.select(1,.15).coverage,"no_anchor");
  f.history.appendFrame({sequence:6,timestamp:1.3,width:16,height:16,pixels:new Uint8Array(256)});
  assert.equal(f.history.select(1,.15).coverage,"missing_source");
  const fresh=fixture({options:{stream_session_id:"new",generation:2}});
  assert.equal(fresh.history.geometryCache.size,0);assert.equal(fresh.history.select(1,.15).coverage,"no_anchor");
  assert.equal(fresh.metadata([fresh.anchor(0),fresh.anchor(2)]),false);
});

test("selected anchor lifetime and selection policy cannot be bypassed by memo",()=>{
  const f=fixture();f.metadata([f.anchor(0),f.anchor(2)]);f.history.select(1,.15);
  f.history.config.max_propagation_seconds=.01;
  assert.equal(f.history.select(1,.15).coverage,"anchor_expired");
  f.history.config.max_propagation_seconds=1.5;f.history.config.anchor_policy="past_only";
  const result=f.history.select(1,.15);assert.equal(result.geometry_cache_hit,false);
  assert.equal(result.subjects[0].anchor_sequence,0);
});

test("approximate results are never memoized and old strict cache hits cannot extend their source-time window",()=>{
  const f=fixture({times:[0,.1,.2,.34,.351,.5],reject:true});f.metadata([f.anchor(0),f.anchor(5)]);
  assert.equal(f.history.select(0,0).geometry_cache_hit,false);
  const first=f.history.select(1,.1);assert.equal(first.coverage,"interpolated");
  assert.equal(first.geometry_cache_hit,false);assert.equal(f.history.geometryCache.has(1),false);
  f.history.select(1,.1);assert.equal(f.calls.approximate,2);
  assert.equal(f.history.select(3,.34).coverage,"interpolated");
  f.history.strictSuccesses.clear();
  assert.equal(f.history.select(0,0).geometry_cache_hit,true);
  assert.deepEqual(f.history.strictSuccesses.get('["epoch","person"]'),[{sequence:0,timestamp:0}]);
  assert.equal(f.history.select(4,.351).subjects.length,0);
  assert.equal(f.history.geometryCache.has(4),false);
});
