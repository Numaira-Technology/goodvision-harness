const assert=require("node:assert/strict"),{test}=require("node:test");
const {ContourHistory}=require("../src/goodvision_harness/camera_display.js");
const config={stream_session_id:"run",generation:1,crop_version:0};
const width=80,height=64,shape=[[[.3,.3],[.6,.3],[.6,.6],[.3,.6]]];
function pixels(dx=0){
  let seed=19;const original=Uint8Array.from({length:width*height},()=>{
    seed=(seed*1664525+1013904223)>>>0;return seed>>>24;
  }),result=new Uint8Array(original.length);
  for(let y=0;y<height;y++)for(let x=dx;x<width;x++)result[y*width+x]=original[y*width+x-dx];
  return result;
}
function append(history,sequence,timestamp,data){history.appendFrame({sequence,timestamp,width,height,pixels:data});}
function anchor(history){history.acceptMetadata({...config,anchors:[{source_sequence:0,media_timestamp:0,
  epoch:"epoch",subjects:[{track_id:"person",polygons:shape}]}],events:[]});}

test("motion preparation receives every accepted consecutive motion in source order",()=>{
  const prepared=[],adapter={estimateMotion:()=>({}),warpPolygons:value=>value,
    prepareMotion:motion=>prepared.push(motion.chain_index)},history=new ContourHistory({...config,allow_partial_boundaries:true},adapter);
  append(history,0,0,pixels());append(history,1,.05,pixels());append(history,2,.1,pixels());
  // A missing interval produces no inferred motion; later real neighbors resume.
  append(history,3,.5,pixels());append(history,4,.55,pixels());
  assert.deepEqual(prepared,[0,1,2]);
  assert.equal(history.motions.has(3),false);
});

function approximationHistory({reason="boundary_occluded",seconds=.35,times=[0,.1,.2,.34,.351,.5],
  futureSubjects=[{track_id:"person",polygons:shape}],futureEpoch="epoch",extraConfig={}}={}){
  const handles=new WeakMap(),calls=[],adapter={estimateMotion:(previous,current)=>({previous,current}),
    reverseMotion:motion=>({previous:motion.current,current:motion.previous}),
    warpPolygons:(source,motion)=>{motion.last_warp_rejection=reason;return null;},
    warpChain:(source,motions)=>{motions.at(-1).last_warp_rejection=reason;return null;},
    warpApproximateChain:(source,motions)=>{calls.push({source,motions});const result=structuredClone(source);
      handles.set(result,{complete:false,approximate:true,polygons:result,open_segments:[],reason});return result;},
    getPresentation:value=>handles.get(value)||{complete:true,polygons:value,open_segments:[]}},
    history=new ContourHistory({...config,anchor_policy:"nearest_arrived",max_propagation_seconds:1.5,
      max_approximation_seconds:seconds,...extraConfig},adapter);
  times.forEach((timestamp,sequence)=>append(history,sequence,timestamp,new Uint8Array(width*height).fill(sequence)));
  history.acceptMetadata({...config,anchors:[observation(0,0,[{track_id:"person",polygons:shape}]),
    observation(times.length-1,times.at(-1),futureSubjects,futureEpoch)],events:[]});
  return {history,calls,adapter};
}

test("short local uncertainty uses real flow but cannot renew its last strict success",()=>{
  const {history,calls}=approximationHistory();assert.equal(history.select(0,0).coverage,"anchored");
  for(const [sequence,time] of [[1,.1],[2,.2],[3,.34]]){
    const result=history.select(sequence,time),subject=result.subjects[0];
    assert.equal(result.coverage,"interpolated");assert.equal(result.all_subjects_present,true);
    assert.equal(subject.complete,false);assert.equal(subject.approximate,true);assert.equal(subject.last_strict_sequence,0);
    assert.equal(subject.approximation_age_seconds,time);assert.equal(history.propagated.get(sequence).subjects.length,0);
    history.select(sequence,time); // Repeated geometry queries cannot renew the window.
  }
  history.acceptMetadata({...config,anchors:[],events:[],event_snapshot_revision:1,event_snapshot_timestamp:.34});
  assert.equal(history.select(4,.351).subjects.length,0);
  assert.equal(calls.length,6);assert.equal(history.strictSuccesses.get('["epoch","person"]').length,1);
  assert.equal(history.select(5,.5).subjects[0].complete,true,"A new real 人员追踪模块 observation is strict again");
});

test("interpolation requires a past strict result and cannot borrow a later strict query",()=>{
  const {history,calls}=approximationHistory();assert.equal(history.select(1,.1).subjects.length,0);
  history.select(5,.5);assert.equal(history.select(3,.34).subjects.length,0);assert.equal(calls.length,0);
});

test("nearest future absence or epoch switch blocks interpolation even when a later match exists",()=>{
  for(const options of [{futureSubjects:[]},{futureEpoch:"new"},{futureSubjects:[{track_id:"other",polygons:shape}]}]){
    const {history,calls}=approximationHistory(options);history.select(0,0);
    append(history,6,.6,new Uint8Array(width*height).fill(6));history.acceptMetadata({...config,
      anchors:[observation(6,.6,[{track_id:"person",polygons:shape}])],events:[]});
    assert.equal(history.select(3,.34).subjects.length,0);assert.equal(calls.length,0);
  }
});

test("short interpolation respects both anchor lifetimes, missing source intervals and hard rejection",()=>{
  const cases=[{extraConfig:{max_propagation_seconds:.25}},{times:[0,.05,.1,.4,.5]},
    {reason:"component_appearance"},{reason:"internal_support"},{reason:"invalid_geometry"},{reason:"region_occluded"}];
  for(const options of cases){
    const {history,calls}=approximationHistory(options);history.select(0,0);
    assert.equal(history.select(1,options.times?.[1]??.1).subjects.length,0);assert.equal(calls.length,0);
  }
});

test("revised 人员追踪模块 invalidates continuity and missing persons cannot inherit another person's strict success",()=>{
  const {history,calls}=approximationHistory();history.select(0,0);
  const changed=shape.map(p=>p.map(([x,y])=>[x+.01,y]));
  history.acceptMetadata({...config,anchors:[observation(0,0,[{track_id:"person",polygons:changed}])],events:[]});
  assert.equal(history.select(1,.1).subjects.length,0);assert.equal(calls.length,0);
  history.select(0,0);history.acceptMetadata({...config,anchors:[observation(1,.1,[])],events:[]});
  history.acceptMetadata({...config,anchors:[observation(2,.2,[{track_id:"person",polygons:shape}])],events:[]});
  assert.equal(history.select(3,.34).subjects.length,0,"An intervening disappearance is a continuity barrier");
});

test("unvalidated partial boundaries are disabled in the production default",()=>{
  let partialCalls=0;
  const adapter={estimateMotion:()=>({}),warpPolygons:()=>null,
    warpPresentation:()=>{partialCalls++;return shape;}},history=new ContourHistory(config,adapter);
  append(history,0,0,pixels());anchor(history);append(history,1,.05,pixels());
  assert.equal(history.select(1,.05).subjects.length,0);assert.equal(partialCalls,0);
});

test("partial boundaries expose only open segments while propagation retains the original 人员追踪模块 shape",()=>{
  const seen=[],handles=new WeakMap(),segment=[[.3,.3],[.6,.3]],adapter={
    estimateMotion:()=>({}),warpPolygons:()=>{throw new Error("presentation adapter must be used");},
    warpPresentation:()=>null,
    warpPresentationChain:(source)=>{seen.push(source);const next=structuredClone(source);
      handles.set(next,{complete:false,polygons:[],open_segments:[segment],reason:"boundary_occluded"});return next;},
    getPresentation:value=>handles.get(value)||{complete:true,polygons:value,open_segments:[]},
  },history=new ContourHistory({...config,allow_partial_boundaries:true},adapter);
  append(history,0,0,pixels());anchor(history);append(history,1,.05,pixels());
  const first=history.select(1,.05),subject=first.subjects[0];
  assert.equal(subject.complete,false);assert.deepEqual(subject.polygons,[]);
  assert.deepEqual(subject.open_segments,[segment]);assert.equal(first.coverage,"motion_unreliable");
  assert.equal(first.motion_rejections.boundary_occluded,1);
  append(history,2,.1,pixels());history.select(2,.1);
  assert.deepEqual(seen[1],shape);assert.notEqual(seen[1],subject.polygons);
  // The authoritative 人员追踪模块 observation remains a complete contour.
  assert.deepEqual(history.select(0,0).subjects[0].polygons,shape);
});

test("a partial estimate can recover a complete contour only from retained real anchor pixels",()=>{
  const handles=new WeakMap(),adapter={estimateMotion:()=>({}),warpPolygons:value=>value,
    warpPresentation:source=>structuredClone(source),warpPresentationChain:source=>{
      const next=structuredClone(source);handles.set(next,{complete:false,polygons:[],
        open_segments:[[[.3,.3],[.6,.3]]],reason:"boundary_occluded"});return next;},
    getPresentation:value=>handles.get(value)||{complete:true,polygons:value,open_segments:[]},
  },history=new ContourHistory({...config,allow_partial_boundaries:true},adapter);
  append(history,0,0,pixels());anchor(history);append(history,1,.05,pixels());
  const result=history.select(1,.05);
  assert.equal(result.subjects[0].complete,true);assert.equal(result.subjects[0].relocalized,true);
  assert.equal(result.subjects[0].anchor_sequence,0);assert.equal(result.subjects[0].target_sequence,1);
  assert.equal(result.coverage,"anchored");assert.deepEqual(result.subjects[0].polygons,shape);
});

function bidirectionalHistory({rejectDirection=0,targetAt=.15}={}){
  const adapter={estimateMotion:(previous,current)=>({step:current[0]-previous[0],previous,current}),
    reverseMotion:motion=>({step:-motion.step,previous:motion.current,current:motion.previous}),
    warpPolygons:(value,motion)=>{
      if(rejectDirection&&Math.sign(motion.step)===rejectDirection){motion.last_warp_rejection="boundary_occluded";return null;}
      return value.map(polygon=>polygon.map(([x,y])=>[x+motion.step/100,y]));
    }};
  const history=new ContourHistory({...config,anchor_policy:"nearest_arrived"},adapter);
  for(const [sequence,timestamp] of [[0,0],[1,targetAt],[2,.2]])
    append(history,sequence,timestamp,new Uint8Array(width*height).fill(sequence));
  return history;
}
function observation(source_sequence,media_timestamp,subjects,epoch="epoch"){
  return {source_sequence,media_timestamp,subjects,epoch};
}

test("only an arrived future observation can shorten geometry propagation; event and confirmation stay at target time",()=>{
  const history=bidirectionalHistory(),subject={track_id:"person",polygons:shape,tracking_confirmation:"provisional"};
  history.acceptMetadata({...config,anchors:[observation(0,0,[subject])],events:[]});
  assert.equal(history.select(1,.15).subjects[0].anchor_sequence,0);
  const futureShape=shape.map(p=>p.map(([x,y])=>[x+.02,y]));
  history.acceptMetadata({...config,anchors:[observation(2,.2,[{...subject,polygons:futureShape,
    tracking_confirmation:"confirmed"}])],events:[]});
  const result=history.select(1,.15),predicted=result.subjects[0];
  assert.equal(predicted.anchor_sequence,2);assert.equal(predicted.target_sequence,1);
  assert.equal(predicted.method,"image_motion_backward");assert.ok(Math.abs(predicted.polygons[0][0][0]-.31)<1e-9);
  assert.equal(predicted.tracking_confirmation,"provisional");assert.equal(predicted.alert,false);
  assert.ok(Math.abs(result.anchor_age_seconds-.05)<1e-9);
});

test("failed registration to the nearer anchor tries the other arrived real anchor under the same gates",()=>{
  for(const [rejectDirection,targetAt,expected] of [[-1,.15,0],[1,.05,2]]){
    const history=bidirectionalHistory({rejectDirection,targetAt});
    history.acceptMetadata({...config,anchors:[observation(0,0,[{track_id:"person",polygons:shape}]),
      observation(2,.2,[{track_id:"person",polygons:shape}])],events:[]});
    const result=history.select(1,targetAt);
    assert.equal(result.subjects[0].anchor_sequence,expected);assert.equal(result.coverage,"anchored");
    assert.equal(result.subjects[0].complete,true);assert.ok(result.motion_rejections.boundary_occluded>=1);
  }
});

test("expiry applies to the selected real geometry anchor rather than an older identity observation",()=>{
  const history=bidirectionalHistory();history.config.max_propagation_seconds=.1;
  history.acceptMetadata({...config,anchors:[observation(0,0,[{track_id:"person",polygons:shape}]),
    observation(2,.2,[{track_id:"person",polygons:shape}])],events:[]});
  assert.equal(history.select(1,.15).subjects[0].anchor_sequence,2);
});

test("future-only people and new epochs cannot create earlier people or erase a past person",()=>{
  for(const [epoch,people] of [["epoch",[{track_id:"new",polygons:shape}]],
    ["next",[{track_id:"person",polygons:shape}]], ["epoch",[]]]){
    const history=bidirectionalHistory();history.acceptMetadata({...config,anchors:[
      observation(0,0,[{track_id:"person",polygons:shape}]),observation(2,.2,people,epoch)],events:[]});
    const subjects=history.select(1,.15).subjects;
    assert.deepEqual(subjects.map(s=>s.track_id),["person"]);assert.equal(subjects[0].anchor_sequence,0);
  }
});

test("a missing-person observation blocks later future anchors and exact 人员追踪模块 wins over interpolation",()=>{
  const history=bidirectionalHistory();append(history,3,.22,new Uint8Array(width*height).fill(3));
  history.acceptMetadata({...config,anchors:[observation(0,0,[{track_id:"person",polygons:shape}]),
    observation(2,.2,[]),observation(3,.22,[{track_id:"person",polygons:shape}])],events:[]});
  assert.equal(history.select(1,.15).subjects[0].anchor_sequence,0);
  const exact=[[[.4,.4],[.5,.4],[.5,.5]]];
  history.acceptMetadata({...config,anchors:[observation(1,.15,[{track_id:"person",polygons:exact}])],events:[]});
  assert.deepEqual(history.select(1,.15).subjects[0].polygons,exact);
  assert.equal(history.select(1,.15).exact,true);
});

test("an ambiguous intermediate image can recover only against real retained 人员追踪模块-source pixels",()=>{
  const history=new ContourHistory(config);
  append(history,0,0,pixels());anchor(history);
  append(history,1,.066,new Uint8Array(width*height).fill(110));
  assert.equal(history.select(1,.066).subjects.length,0);
  append(history,2,.132,pixels(2));
  append(history,3,.2,pixels(2));append(history,4,.266,pixels(2));
  const result=history.select(4,.266);
  assert.equal(result.subjects.length,1);
  assert.equal(result.subjects[0].anchor_sequence,0);
  assert.equal(result.subjects[0].target_sequence,4);
  assert.equal(result.subjects[0].relocalized,true);
  assert.ok(Math.abs(result.subjects[0].polygons[0][0][0]-(.3+2/79))<.006);
  assert.equal(result.exact,false);
});

test("one hidden person does not erase a visible person or prevent later image-based recovery",()=>{
  const history=new ContourHistory(config),a=[[[.1,.1],[.4,.1],[.4,.45],[.1,.45]]],
    b=[[[.55,.55],[.9,.55],[.9,.9],[.55,.9]]];
  append(history,0,0,pixels());
  history.acceptMetadata({...config,anchors:[{source_sequence:0,media_timestamp:0,epoch:"epoch",
    subjects:[{track_id:"a",polygons:a},{track_id:"b",polygons:b}]}],events:[]});
  const hidden=pixels();
  for(let y=30;y<height;y++)for(let x=40;x<width;x++)hidden[y*width+x]=110;
  append(history,1,.066,hidden);
  assert.deepEqual(history.select(1,.066).subjects.map(subject=>subject.track_id),["a"]);
  append(history,2,.132,hidden);
  assert.deepEqual(history.select(2,.132).subjects.map(subject=>subject.track_id),["a"]);
  append(history,3,.2,pixels(1));append(history,4,.266,pixels(1));
  const subjects=history.select(4,.266).subjects;
  assert.deepEqual(subjects.map(subject=>subject.track_id),["a","b"]);
  assert.ok(Math.abs(subjects[1].polygons[0][0][0]-(.55+1/79))<.006);
});

test("relocalization does not bridge a missing real-frame interval or extend anchor lifetime",()=>{
  const history=new ContourHistory(config);
  append(history,0,0,pixels());anchor(history);
  append(history,1,.3,pixels(2));
  assert.equal(history.select(1,.3).coverage,"motion_gap");
  append(history,2,2.1,pixels(2));
  assert.equal(history.select(2,2.1).coverage,"anchor_expired");
});

test("new 人员追踪模块 geometry after a failed estimate replaces the old shape on the right source",()=>{
  const history=new ContourHistory(config);
  append(history,0,0,pixels());anchor(history);
  append(history,1,.066,new Uint8Array(width*height).fill(110));
  assert.equal(history.select(1,.066).subjects.length,0);
  const next=[[[.1,.2],[.5,.2],[.5,.8],[.1,.8]]];
  history.acceptMetadata({...config,anchors:[{source_sequence:1,media_timestamp:.066,
    epoch:"epoch",subjects:[{track_id:"person",polygons:next}]}],events:[]});
  const result=history.select(1,.066);
  assert.equal(result.exact,true);assert.deepEqual(result.subjects[0].polygons,next);
});

test("retired server anchors are ignored while missing sources inside retained history are rejected",()=>{
  const history=new ContourHistory(config);
  append(history,0,0,pixels());append(history,100,7,pixels());
  const make=(source_sequence,media_timestamp)=>({source_sequence,media_timestamp,
    epoch:"epoch",subjects:[{track_id:"person",polygons:shape}]});
  for(let i=0;i<3;i++)history.acceptMetadata({...config,anchors:[make(0,0),make(100,7)],events:[]});
  assert.equal(history.rejected,0);assert.equal(history.revision,1);
  assert.equal(history.select(100,7).subjects.length,1);
  history.acceptMetadata({...config,anchors:[make(99,6.95)],events:[]});
  assert.equal(history.rejected,1);
});


test("skipped motion intervals cannot reuse stale contours and fresh exact anchors restore them",()=>{
  const history=new ContourHistory(config);
  append(history,0,0,pixels());anchor(history);
  append(history,1,1/15,pixels(1));
  assert.equal(history.select(1,1/15).subjects.length,1);
  append(history,20,20/15,pixels(2));append(history,21,21/15,pixels(3));
  assert.equal(history.select(20,20/15).subjects.length,0);
  assert.equal(history.select(21,21/15).subjects.length,0);
  history.acceptMetadata({...config,anchors:[{source_sequence:20,media_timestamp:20/15,
    epoch:"epoch",subjects:[{track_id:"person",polygons:shape}]}],events:[]});
  assert.equal(history.select(20,20/15).exact,true);
  assert.equal(history.select(21,21/15).subjects[0].target_sequence,21);
  assert.equal(history.select(1,1/15).subjects.length,1,"old buffered playback retains its own valid segment");
});
