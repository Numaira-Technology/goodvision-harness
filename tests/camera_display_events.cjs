const assert=require("node:assert/strict"),{test}=require("node:test");
const {ContourHistory}=require("../src/goodvision_harness/camera_display.js");
const config={stream_session_id:"event-run",generation:1,crop_version:0,anchor_policy:"nearest_arrived"};
const shape=[[[.3,.3],[.6,.3],[.6,.6],[.3,.6]]];

function observation(source_sequence,media_timestamp){
  return {source_sequence,media_timestamp,epoch:"epoch",subjects:[{
    track_id:"person",tracking_confirmation:"confirmed",
    polygons:shape.map(polygon=>polygon.map(([x,y])=>[x+source_sequence/100,y])),
  }]};
}
function confirmed(revision,from=0,until=null,extra={}){
  return {event_id:"smoking-1",track_id:"person",epoch:"epoch",event_label:"smoking",
    phase:"confirmed",revision,valid_from_timestamp:from,valid_until_timestamp:until,...extra};
}
function metadata(anchors=[],events=[],snapshot={}){return {...config,anchors,events,...snapshot};}
function history({future=true}={}){
  // A cheap deterministic geometry adapter isolates event-time behavior from
  // image matching, while exercising the real reverse-anchor selection path.
  const adapter={estimateMotion:(previous,current)=>({previous,current,step:current[0]-previous[0]}),
    reverseMotion:motion=>({previous:motion.current,current:motion.previous,step:-motion.step}),
    warpPolygons:(polygons,motion)=>polygons.map(polygon=>polygon.map(([x,y])=>[x+motion.step/100,y]))};
  const result=new ContourHistory(config,adapter);
  for(const [sequence,timestamp] of [[0,0],[1,.15],[2,.2],[3,.3],[4,.4]])
    result.appendFrame({sequence,timestamp,width:16,height:16,pixels:new Uint8Array(256).fill(sequence)});
  result.acceptMetadata(metadata(future?[observation(0,0),observation(2,.2)]:[observation(0,0)]));
  return result;
}
function personAt(subjectHistory,sequence,timestamp){
  const selected=subjectHistory.select(sequence,timestamp);
  assert.equal(selected.subjects.length,1);
  assert.equal(selected.subjects[0].tracking_confirmation,"confirmed",
    "A provisional subject would hide an event-time leak by disabling alerts");
  return selected.subjects[0];
}

test("an older full event snapshot cannot resurrect an unknown event but its valid 人员追踪模块 anchor is accepted",()=>{
  const subjectHistory=history({future:false});
  subjectHistory.acceptMetadata(metadata([],[],{event_snapshot_revision:10,event_snapshot_timestamp:.2}));
  assert.equal(subjectHistory.acceptMetadata(metadata([observation(4,.4)],[confirmed(9)],{
    event_snapshot_revision:9,event_snapshot_timestamp:.1,
  })),true);
  assert.equal(subjectHistory.anchors.has(4),true,"Event snapshot ordering does not discard independent 人员追踪模块 observations");
  const subject=personAt(subjectHistory,3,.3);
  assert.equal(subject.anchor_sequence,4);assert.equal(subject.method,"image_motion_backward");
  assert.equal(subject.alert,false,"An event missing from the newer snapshot cannot reopen after that snapshot time");
});

test("an arrived future confirmation cannot color an earlier reverse-propagated subject",()=>{
  const subjectHistory=history();
  subjectHistory.acceptMetadata(metadata([],[confirmed(1,.2)],{
    event_snapshot_revision:1,event_snapshot_timestamp:.2,
  }));
  const subject=personAt(subjectHistory,1,.15);
  assert.equal(subject.method,"image_motion_backward");assert.equal(subject.anchor_sequence,2);
  assert.equal(subject.alert,false);
  assert.equal(personAt(subjectHistory,2,.2).alert,true,"Confirmation applies when its source time is reached");
});

test("a future explicit event end preserves the earlier alert and closes at its exact boundary",()=>{
  const subjectHistory=history();
  subjectHistory.acceptMetadata(metadata([],[confirmed(1)],{
    event_snapshot_revision:1,event_snapshot_timestamp:0,
  }));
  // The server encodes an ended confirmed interval with valid_until and keeps
  // phase=confirmed, so old buffered targets retain the appropriate color.
  subjectHistory.acceptMetadata(metadata([],[confirmed(2,0,.2,{event_state:"ended"})],{
    event_snapshot_revision:2,event_snapshot_timestamp:.2,
  }));
  const earlier=personAt(subjectHistory,1,.15);
  assert.equal(earlier.method,"image_motion_backward");assert.equal(earlier.alert,true);
  assert.equal(personAt(subjectHistory,2,.2).alert,false);
  assert.equal(personAt(subjectHistory,3,.3).alert,false);
});

test("a future omission tombstone closes at snapshot time without clearing an earlier buffered alert",()=>{
  const subjectHistory=history();
  subjectHistory.acceptMetadata(metadata([],[confirmed(1)],{
    event_snapshot_revision:1,event_snapshot_timestamp:0,
  }));
  subjectHistory.acceptMetadata(metadata([],[],{event_snapshot_revision:2,event_snapshot_timestamp:.2}));
  assert.equal(personAt(subjectHistory,1,.15).alert,true);
  assert.equal(personAt(subjectHistory,2,.2).alert,false);
  assert.equal(personAt(subjectHistory,3,.3).alert,false);
});

test("future geometry never borrows a confirmed event from a different predictor epoch",()=>{
  const subjectHistory=history();
  subjectHistory.acceptMetadata(metadata([],[confirmed(1,0,null,{epoch:"another-epoch"})],{
    event_snapshot_revision:1,event_snapshot_timestamp:.2,
  }));
  const subject=personAt(subjectHistory,1,.15);
  assert.equal(subject.method,"image_motion_backward");assert.equal(subject.epoch,"epoch");
  assert.equal(subject.alert,false);
});

test("label names follow the same confirmed identity and source-time interval as event colors",()=>{
  const subjectHistory=history();
  subjectHistory.acceptMetadata(metadata([],[confirmed(1,.2),
    confirmed(1,.2,null,{event_id:"fall-1",event_label:"fall"}),
    confirmed(1,.2,null,{event_id:"duplicate-smoking"}),
    confirmed(1,0,null,{event_id:"other-person",event_label:"weapon",track_id:"other"}),
    confirmed(1,0,null,{event_id:"other-epoch",event_label:"fight",epoch:"other"}),
  ]));
  assert.deepEqual(personAt(subjectHistory,1,.15).event_labels,[]);
  assert.deepEqual(personAt(subjectHistory,2,.2).event_labels,["smoking","fall"]);
  subjectHistory.acceptMetadata(metadata([],[],{event_snapshot_revision:2,event_snapshot_timestamp:.3}));
  assert.deepEqual(personAt(subjectHistory,2,.2).event_labels,["smoking","fall"]);
  assert.deepEqual(personAt(subjectHistory,3,.3).event_labels,[]);
});

test("an ended event still labels an earlier buffered frame and provisional identities get no event labels",()=>{
  const subjectHistory=history();
  subjectHistory.acceptMetadata(metadata([],[confirmed(1,0,.2,{event_state:"ended"})]));
  assert.deepEqual(personAt(subjectHistory,1,.15).event_labels,["smoking"]);
  assert.deepEqual(personAt(subjectHistory,2,.2).event_labels,[]);
  const provisional=observation(0,0);provisional.subjects[0].tracking_confirmation="provisional";
  subjectHistory.acceptMetadata(metadata([provisional]));
  assert.deepEqual(subjectHistory.select(1,.15).subjects[0].event_labels,[]);
});
