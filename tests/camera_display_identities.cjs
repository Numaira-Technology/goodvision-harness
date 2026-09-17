const assert=require("node:assert/strict"),{test}=require("node:test");
const {ContourHistory,DisplayIdentities}=require("../src/goodvision_harness/camera_display.js");
const config={stream_session_id:"name-run",generation:2,crop_version:0,anchor_policy:"nearest_arrived"};
const shape=[[[.2,.2],[.6,.2],[.6,.6]]];
const anchor=(sequence,timestamp,epoch="epoch",track_id="0")=>({source_sequence:sequence,media_timestamp:timestamp,
  epoch,subjects:[{track_id,tracking_confirmation:"confirmed",polygons:shape}]});
const identity=(revision,from=0,until=null,extra={})=>({epoch:"epoch",track_id:"0",person_id:"gallery-person",
  name:"测试姓名",valid_from_timestamp:from,valid_until_timestamp:until,revision,...extra});
const metadata=(revision,identities,extra={})=>({...config,anchors:[],identity_snapshot_timestamp:.4,
  identity_snapshot_revision:revision,identities,...extra});
function fixture({middle=true}={}){
  let warps=0;
  const adapter={estimateMotion:()=>({}),reverseMotion:()=>({}),warpPolygons:polygons=>{warps++;return polygons;}};
  const history=new ContourHistory(config,adapter),names=new DisplayIdentities(config);
  for(let sequence=0;sequence<=4;sequence++)history.appendFrame({sequence,timestamp:sequence/10,
    width:16,height:16,pixels:new Uint8Array(256)});
  history.acceptMetadata({...config,anchors:middle?[anchor(0,0),anchor(2,.2)]:[anchor(0,0)]});
  return {history,names,warps:()=>warps,
    accept:data=>{const accepted=names.accept(data);history.acceptMetadata(data);return accepted;},
    subject:(sequence,timestamp=sequence/10)=>{
      const subject=history.select(sequence,timestamp).subjects[0];
      return {...subject,display_name:names.nameAt(subject,timestamp)};
    }};
}

test("a future native anchor cannot lend its confirmed gallery name to an earlier source frame",()=>{
  const view=fixture({middle:false});view.accept(metadata(1,[identity(1,.4)],{anchors:[anchor(4,.4)]}));
  const before=view.subject(3);assert.equal(before.method,"image_motion_backward");assert.equal(before.display_name,null);
  assert.equal(view.subject(2).display_name,null);assert.equal(view.subject(4).display_name,"测试姓名");
});

test("main-thread name updates and revocation never create worker identity state or invalidate geometry",()=>{
  const view=fixture(),first=view.history.select(1,.1);
  assert.equal(Object.hasOwn(view.history,"identities"),false);
  assert.equal(Object.hasOwn(first.subjects[0],"display_name"),false);
  const calls=view.warps(),revision=view.history.revision;view.accept(metadata(1,[identity(1)]));
  const named=view.history.select(1,.1);assert.equal(named.geometry_cache_hit,true);
  assert.equal(view.names.nameAt(named.subjects[0],.1),"测试姓名");assert.equal(view.warps(),calls);
  assert.equal(Object.hasOwn(named.subjects[0],"display_name"),false);
  assert.equal(view.history.revision,revision,"identity-only updates do not alter worker revisions");
  view.accept(metadata(2,[]));
  const revoked=view.history.select(1,.1);assert.equal(revoked.geometry_cache_hit,true);
  assert.equal(view.names.nameAt(revoked.subjects[0],.1),null);assert.equal(view.warps(),calls);
  view.accept(metadata(1,[identity(1)]));assert.equal(view.subject(1).display_name,null);
});

test("same native ID in another epoch or run never supplies a name and provisional geometry stays anonymous",()=>{
  const view=fixture();
  view.accept(metadata(1,[identity(1,0,null,{epoch:"another-epoch"}),
    identity(1,0,null,{track_id:"another-track"})]));
  assert.equal(view.subject(1).display_name,null);
  assert.equal(view.accept(metadata(2,[identity(2)],{stream_session_id:"other-run"})),false);
  assert.equal(view.subject(1).display_name,null);
  const provisional=anchor(0,0);provisional.subjects[0].tracking_confirmation="provisional";
  view.accept(metadata(2,[identity(2)],{anchors:[provisional]}));
  assert.equal(view.subject(1).display_name,null);
});

test("name changes close at the source boundary and conflicting active identities do not pick an arbitrary name",()=>{
  const view=fixture();view.accept(metadata(2,[identity(2,0,.2),
    identity(2,.2,null,{person_id:"new-person",name:"另一个名字"})]));
  assert.equal(view.subject(1).display_name,"测试姓名");assert.equal(view.subject(2).display_name,"另一个名字");
  view.accept(metadata(3,[identity(3),identity(3,0,null,{person_id:"conflict",name:"Other"})]));
  assert.equal(view.subject(2).display_name,null);
});

test("identity snapshot ordering is independent of old event snapshots and names are validated",()=>{
  const view=fixture();view.history.acceptMetadata({...config,anchors:[],events:[],
    event_snapshot_revision:10,event_snapshot_timestamp:.4});
  view.accept(metadata(1,[identity(1)],{events:[],event_snapshot_revision:1,event_snapshot_timestamp:0}));
  assert.equal(view.subject(1).display_name,"测试姓名");
  for(const name of ["", "bad\nname", {}, "x".repeat(161)]){
    view.accept(metadata(2,[identity(2,0,null,{name})]));
    assert.equal(view.subject(1).display_name,"测试姓名","malformed snapshots cannot replace the last valid table");
  }
  view.accept(metadata(3,[]));assert.equal(view.subject(1).display_name,null);
});
