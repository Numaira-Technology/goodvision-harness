const assert=require("node:assert/strict");
const {test}=require("node:test");
const {FrameTimeline,ContourHistory,estimateMotion,warpPolygons,packCameraFrame}=require(
  "../src/goodvision_harness/camera_display.js");
const config={stream_session_id:"run-a",generation:2,crop_version:0};
const shape=[[[.3,.3],[.6,.3],[.6,.6],[.3,.6]]];
function texture(width=80,height=64){
  let seed=19;return Uint8Array.from({length:width*height},()=>{seed=(seed*1664525+1013904223)>>>0;return seed>>>24;});
}
function shift(input,dx,dy,width=80,height=64){
  const pixels=new Uint8Array(width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const px=x-dx,py=y-dy;
    if(px>=0&&px<width&&py>=0&&py<height)pixels[y*width+x]=input[py*width+px];
  }
  return pixels;
}
function frame(sequence,timestamp,pixels=texture()){return {sequence,timestamp,pixels,width:80,height:64};}
function metadata(anchors,events=[]){return {...config,anchors,events};}
function anchor(source_sequence=0,media_timestamp=0){return {source_sequence,media_timestamp,
  epoch:"epoch-a",subjects:[{track_id:"person-1",polygons:shape}]};}
function event(revision,valid_from_timestamp,phase="confirmed"){
  return {epoch:"epoch-a",track_id:"person-1",event_label:"smoking",revision,
    valid_from_timestamp,valid_until_timestamp:null,phase};
}

test("FIFO presents 15 distinct true frames per second independent of any model result",()=>{
  const queue=new FrameTimeline({delaySeconds:.5});const actual=[];
  for(let i=0;i<45;i++){
    queue.push({sequence:i,timestamp:i/15});
    const due=queue.take(i/15);
    if(due){actual.push(due.sequence);queue.notePresented(due,i*1000/15);}
  }
  assert.ok(actual.length>=37);assert.deepEqual(actual,[...new Set(actual)]);
  assert.deepEqual(actual,[...Array(actual.length).keys()]);
  assert.ok(Math.abs(queue.stats().distinct_presented_fps-15)<1e-6);
});
test("FIFO never substitutes later frames or repeats old analysis frames",()=>{
  const queue=new FrameTimeline({delaySeconds:1});
  queue.push({sequence:10,timestamp:1});queue.push({sequence:20,timestamp:2});
  assert.equal(queue.take(1.99),null);assert.equal(queue.take(2).sequence,10);
  assert.equal(queue.take(2),null);assert.equal(queue.take(3).sequence,20);
  assert.equal(queue.push({sequence:20,timestamp:3}),false);
  assert.equal(queue.push({sequence:21,timestamp:2}),false);
});
test("display capacity overflow is explicit and disposes only display resources",()=>{
  const disposed=[],analysis=[0,1,2];const queue=new FrameTimeline({maxFrames:2,dispose:f=>disposed.push(f.sequence)});
  for(let i=0;i<3;i++)queue.push({sequence:i,timestamp:i});
  assert.deepEqual(disposed,[0]);assert.equal(queue.stats().display_buffer_expired,1);
  assert.deepEqual(analysis,[0,1,2]);queue.reset();assert.deepEqual(disposed,[0,1,2]);
});
test("image-derived movement warps contour to the actual translated image",()=>{
  const pixels=texture(),motion=estimateMotion(pixels,shift(pixels,2,1),80,64);
  assert.ok(motion.validFraction>.75);const result=warpPolygons(shape,motion);
  assert.ok(result);assert.ok(Math.abs(result[0][0][0]-(.3+2/79))<1e-9);
  assert.ok(Math.abs(result[0][0][1]-(.3+1/63))<1e-9);
  assert.deepEqual(shape,[[[.3,.3],[.6,.3],[.6,.6],[.3,.6]]]);
});

test("provisional observed masks can display and propagate without becoming confirmed alert geometry",()=>{
  const history=new ContourHistory(config),pixels=texture(),provisional=anchor();
  provisional.subjects[0].tracking_confirmation="provisional";
  history.appendFrame(frame(0,0,pixels));history.appendFrame(frame(1,.066,shift(pixels,1,0)));
  history.acceptMetadata(metadata([provisional],[event(1,0)]));
  for(const [sequence,time] of [[0,0],[1,.066]]){
    const subjects=history.select(sequence,time).subjects;assert.equal(subjects.length,1);
    assert.equal(subjects[0].tracking_confirmation,"provisional");assert.equal(subjects[0].alert,false);
  }
  const confirmed=anchor(1,.066);history.acceptMetadata(metadata([confirmed],[event(1,0)]));
  assert.equal(history.select(1,.066).subjects[0].tracking_confirmation,"confirmed");
  assert.equal(history.select(1,.066).subjects[0].alert,true);
});
test("flat or unrelated images cannot turn a stale contour into a new observation",()=>{
  const flat=new Uint8Array(80*64).fill(120),motion=estimateMotion(flat,flat,80,64);
  assert.equal(warpPolygons(shape,motion),null);
  const changed=texture();for(let i=0;i<changed.length;i++)changed[i]=255-changed[i];
  assert.equal(warpPolygons(shape,estimateMotion(texture(),changed,80,64)),null);
});

// Independent foreground/background pixels expose errors hidden by translating
// the entire image. These are generated test pixels, never saved camera imagery.
const motionWidth=160,motionHeight=120;
const personOutline=[[64,12],[81,12],[88,20],[87,34],[99,37],[110,69],[102,73],
  [94,52],[96,103],[82,105],[77,70],[73,105],[60,104],[61,51],[50,76],[43,71],[53,39],[65,33],[59,23]];
function motionInside(x,y,p){let hit=false;for(let i=0,j=p.length-1;i<p.length;j=i++){
  const a=p[i],b=p[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])hit=!hit;
}return hit;}
function motionNoise(x,y,seed=19){
  let n=(Math.imul(x+1000,1103515245)^Math.imul(y+1000,12345)^Math.imul(seed,2654435761))>>>0;
  n^=n>>>13;n=Math.imul(n,1274126177);return(n>>>24)&255;
}
function motionTexture(x,y,seed=19){return (motionNoise(x,y,seed)*.5+
  motionNoise(Math.floor(x/4),Math.floor(y/4),seed+4)*.5)|0;}
function motionTextureSample(x,y,seed=19){
  const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;
  return motionTexture(ix,iy,seed)*(1-fx)*(1-fy)+motionTexture(ix+1,iy,seed)*fx*(1-fy)+
    motionTexture(ix,iy+1,seed)*(1-fx)*fy+motionTexture(ix+1,iy+1,seed)*fx*fy;
}
function motionFrame({dx=0,backgroundDx=0,pattern="texture",visible=true,seed=19,occlusion=0,
  polygon=personOutline,moveX=()=>dx,other=null}={}){
  const out=new Uint8Array(motionWidth*motionHeight);
  for(let y=0;y<motionHeight;y++)for(let x=0;x<motionWidth;x++){
    let value=motionTexture(x-backgroundDx,y,4),px=x-moveX(y);
    if(visible&&motionInside(px,y,polygon)){
      value=pattern==="flat"?140:pattern==="checker"?(((Math.floor(px/4)+Math.floor(y/4))&1)?210:50):
        pattern==="torso"&&(y<36||y>69)?140:motionTextureSample(px,y,seed);
      if(occlusion&&y>20&&y<108&&x>43&&x<43+70*occlusion)value=65;
    }
    if(other&&motionInside(x-other.dx,y,other.polygon))value=motionTexture(x-other.dx,y,71);
    out[y*motionWidth+x]=value;
  }return out;
}
const motionShape=(outline=personOutline,dx=0)=>[outline.map(([x,y])=>[(x+dx)/(motionWidth-1),y/(motionHeight-1)])];
function vertexError(result,outline=personOutline,moveX=()=>0){
  assert.ok(result,"Reliable foreground motion must produce a contour");
  return Math.max(...result[0].map((point,i)=>Math.hypot(point[0]*(motionWidth-1)-outline[i][0]-moveX(outline[i][1]),
    point[1]*(motionHeight-1)-outline[i][1])));
}

test("interior person support follows large motion without static or opposing background votes",()=>{
  for(const [dx,backgroundDx,pattern] of [[2,0,"texture"],[8,0,"texture"],[12,0,"texture"],[2,-2,"texture"],[2,0,"torso"]]){
    const before=motionFrame({pattern}),after=motionFrame({dx,backgroundDx,pattern}),
      result=warpPolygons(motionShape(),estimateMotion(before,after,motionWidth,motionHeight));
    assert.ok(vertexError(result,personOutline,()=>dx)<1,`independent motion ${dx}/${backgroundDx}/${pattern}`);
  }
});

test("textured background cannot preserve a vanished, occluded, replaced or unsupported person",()=>{
  const rectangle=[[48,28],[104,28],[104,98],[48,98]];
  for(const [beforeOptions,afterOptions] of [
    [{polygon:rectangle,pattern:"flat"},{polygon:rectangle,visible:false}],
    [{polygon:rectangle,pattern:"flat"},{polygon:rectangle,pattern:"flat",dx:8}],
    [{},{occlusion:1}], [{},{seed:41}],
  ]){
    const motion=estimateMotion(motionFrame(beforeOptions),motionFrame(afterOptions),motionWidth,motionHeight);
    assert.equal(warpPolygons(motionShape(beforeOptions.polygon),motion),null,JSON.stringify([beforeOptions,afterOptions]));
  }
});

test("whole-subject image support resolves a periodic torso only when its silhouette gives a unique displacement",()=>{
  const rectangle=[[48,28],[104,28],[104,98],[48,98]],
    before=motionFrame({polygon:rectangle,pattern:"checker"}),
    after=motionFrame({polygon:rectangle,pattern:"checker",dx:8});
  assert.ok(vertexError(warpPolygons(motionShape(rectangle),estimateMotion(before,after,160,120)),rectangle,()=>8)<1);
  const ambiguous=Uint8Array.from({length:160*120},(_,at)=>((Math.floor(at%160/4)+Math.floor(at/160/4))&1)?210:50);
  assert.equal(warpPolygons(motionShape(rectangle),estimateMotion(ambiguous,ambiguous,160,120)),null);
});

test("small subjects use their own whole-mask pixels without inventing missing corner support",()=>{
  for(const [width,height] of [[5,14],[7,18],[10,20]]){
    const outline=[[60,35],[60+width,35],[60+width,35+height],[60,35+height]];
    for(const dx of [.5,1,3]){
      const before=motionFrame({polygon:outline}),after=motionFrame({polygon:outline,dx}),
        motion=estimateMotion(before,after,160,120),result=warpPolygons(motionShape(outline),motion);
      assert.ok(vertexError(result,outline,()=>dx)<.5,`small subject ${width}x${height}, ${dx}`);
    }
  }
});

test("insufficient narrow silhouettes and disappeared small subjects cannot relocate onto nearby people or background",()=>{
  const narrow=[[60,35],[63,35],[63,43],[60,43]],small=[[60,35],[65,35],[65,49],[60,49]],
    neighbor=[[66,34],[72,34],[72,50],[66,50]];
  for(const dx of [.5,1,3])assert.equal(warpPolygons(motionShape(narrow),
    estimateMotion(motionFrame({polygon:narrow}),motionFrame({polygon:narrow,dx}),160,120)),null);
  const before=motionFrame({polygon:small,other:{polygon:neighbor,dx:0}}),
    disappeared=motionFrame({polygon:small,visible:false,other:{polygon:neighbor,dx:-1}});
  assert.equal(warpPolygons(motionShape(small),estimateMotion(before,disappeared,160,120)),null);
  const similarBackground=new Uint8Array(160*120).fill(140),after=similarBackground.slice();
  for(let y=35;y<49;y++)for(let x=60;x<65;x++)similarBackground[y*160+x]=142;
  assert.equal(warpPolygons(motionShape(small),estimateMotion(similarBackground,after,160,120)),null);
});

test("two seconds of a small moving subject retain subpixel alignment",()=>{
  const outline=[[60,35],[67,35],[67,53],[60,53]];
  let shape=motionShape(outline),before=motionFrame({polygon:outline});
  for(let frame=1;frame<=30;frame++){
    const dx=frame*.25,after=motionFrame({polygon:outline,dx}),motion=estimateMotion(before,after,160,120);
    shape=warpPolygons(shape,motion);
    assert.ok(vertexError(shape,outline,()=>dx)<1.5,`small subject drift at ${frame}/15 seconds`);
    before=after;
  }
});

test("small-subject appearance requires its silhouette transition as well as similar interior grey",()=>{
  const outline=[[60,35],[65,35],[65,49],[60,49]],
    before=motionFrame({polygon:outline,seed:15}),moved=motionFrame({polygon:outline,seed:15,dx:1}),
    disappeared=motionFrame({polygon:outline,visible:false});
  assert.ok(vertexError(warpPolygons(motionShape(outline),estimateMotion(before,moved,160,120)),outline,()=>1)<.5);
  assert.equal(warpPolygons(motionShape(outline),estimateMotion(before,disappeared,160,120)),null);
});

test("four seconds of independent person motion do not drift onto a moving background",()=>{
  for(const pattern of ["texture","torso"]){
    let source=motionShape(),before=motionFrame({pattern});
    for(let i=1;i<=60;i++){
      const dx=Math.round(12*Math.sin(i/8)),after=motionFrame({dx,backgroundDx:-Math.round(4*Math.sin(i/8)),pattern});
      source=warpPolygons(source,estimateMotion(before,after,motionWidth,motionHeight));
      assert.ok(vertexError(source,personOutline,()=>dx)<1,`drift at ${i}/15 seconds, ${pattern}`);
      before=after;
    }
  }
});

test("two separately moving people retain their own internal motion support",()=>{
  const a=[[15,25],[61,25],[61,105],[15,105]],b=[[95,25],[145,25],[145,105],[95,105]],
    before=motionFrame({polygon:a,other:{polygon:b,dx:0}}),
    after=motionFrame({polygon:a,dx:8,other:{polygon:b,dx:-8}}),
    motion=estimateMotion(before,after,motionWidth,motionHeight);
  assert.ok(vertexError(warpPolygons(motionShape(a),motion),a,()=>8)<1);
  assert.ok(vertexError(warpPolygons(motionShape(b),motion),b,()=>-8)<1);
});

test("a supported shear deforms the contour rather than translating every vertex equally",()=>{
  const rectangle=[[40,22],[100,22],[100,102],[40,102]],moveX=y=>Math.round((y-22)*.08),
    before=motionFrame({polygon:rectangle}),after=motionFrame({polygon:rectangle,moveX}),
    result=warpPolygons(motionShape(rectangle),estimateMotion(before,after,motionWidth,motionHeight));
  assert.ok(vertexError(result,rectangle,moveX)<2.5);
});

test("out-of-frame translation clips the visible polygon without attaching it to the old border",()=>{
  const rectangle=[[94,24],[146,24],[146,102],[94,102]],
    before=motionFrame({polygon:rectangle}),after=motionFrame({polygon:rectangle,dx:18}),
    result=warpPolygons(motionShape(rectangle),estimateMotion(before,after,motionWidth,motionHeight));
  assert.ok(result);assert.ok(result.flat().every(point=>point.every(n=>n>=0&&n<=1)));
  assert.ok(Math.abs(Math.min(...result[0].map(p=>p[0]))*159-112)<1);
  assert.equal(Math.max(...result[0].map(p=>p[0])),1);
});

test("two seconds of quarter and half pixel motion preserve subpixel displacement",()=>{
  for(const speed of [.25,.5]){
    let source=motionShape(),before=motionFrame();
    for(let i=1;i<=30;i++){
      const dx=i*speed,after=motionFrame({dx});
      source=warpPolygons(source,estimateMotion(before,after,motionWidth,motionHeight));
      assert.ok(vertexError(source,personOutline,()=>dx)<1.5,`subpixel drift at frame ${i}, step ${speed}`);
      before=after;
    }
  }
});

test("small disconnected mask components use same-person support and still validate their own visibility",()=>{
  const torso=[[48,28],[95,28],[95,98],[48,98]],hand=[[104,52],[109,52],[109,58],[104,58]],
    before=motionFrame({polygon:torso,other:{polygon:hand,dx:0}}),
    after=motionFrame({polygon:torso,dx:3,other:{polygon:hand,dx:3}}),
    source=[...motionShape(torso),...motionShape(hand)],
    result=warpPolygons(source,estimateMotion(before,after,motionWidth,motionHeight));
  assert.ok(result);assert.equal(result.length,2);
  assert.ok(vertexError([result[1]],hand,()=>3)<1);
  const missing=motionFrame({polygon:torso,dx:3});
  assert.equal(warpPolygons(source,estimateMotion(before,missing,motionWidth,motionHeight)),null);
});

test("dense contour vertices cannot bypass a locally disappeared hand",()=>{
  const before=motionFrame(),after=before.slice(),dense=[];
  for(let y=53;y<motionHeight;y++)for(let x=100;x<motionWidth;x++)after[y*motionWidth+x]=motionTexture(x,y,4);
  for(let i=0;i<personOutline.length;i++){
    const a=personOutline[i],b=personOutline[(i+1)%personOutline.length],steps=Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])/4.8);
    for(let j=0;j<steps;j++)dense.push([a[0]+(b[0]-a[0])*j/steps,a[1]+(b[1]-a[1])*j/steps]);
  }
  const motion=estimateMotion(before,after,motionWidth,motionHeight);
  assert.equal(warpPolygons(motionShape(),motion),null);
  assert.equal(warpPolygons(motionShape(dense),motion),null);
});

test("a source crop edge stays at the camera border only when current pixels support the continuation",()=>{
  const rectangle=[[45,25],[105,25],[105,119],[45,119]],source=motionShape(rectangle),
    make=(dy,endsBelow)=>{
      const pixels=new Uint8Array(motionWidth*motionHeight);
      for(let y=0;y<motionHeight;y++)for(let x=0;x<motionWidth;x++){
        const py=y-dy,inside=x>=45&&x<105&&py>=25&&py<(endsBelow?145:120);
        pixels[y*motionWidth+x]=inside?(py>102?190:motionTexture(x,py,19)):35;
      }return pixels;
    },before=make(0,true),continuing=make(-4,true),leaving=make(-4,false),
    result=warpPolygons(source,estimateMotion(before,continuing,motionWidth,motionHeight));
  assert.ok(result);assert.equal(Math.max(...result[0].map(p=>p[1])),1);
  assert.equal(warpPolygons(source,estimateMotion(before,leaving,motionWidth,motionHeight)),null);
  const dense=rectangle.flatMap((a,i)=>{
    const b=rectangle[(i+1)%rectangle.length],steps=Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])/4);
    return Array.from({length:steps},(_,j)=>[a[0]+(b[0]-a[0])*j/steps,a[1]+(b[1]-a[1])*j/steps]);
  });
  const denseResult=warpPolygons(motionShape(dense),estimateMotion(before,continuing,motionWidth,motionHeight));
  assert.ok(denseResult);assert.equal(Math.max(...denseResult[0].map(p=>p[1])),1);
});
test("late 人员追踪模块 anchor follows every retained real-frame motion to its target source",()=>{
  const history=new ContourHistory(config),pixels=texture();
  history.appendFrame(frame(0,0,pixels));history.appendFrame(frame(4,.066,shift(pixels,1,0)));
  history.appendFrame(frame(7,.132,shift(pixels,2,0)));
  history.acceptMetadata(metadata([anchor()]));
  const selected=history.select(7,.132);
  assert.equal(selected.subjects[0].target_sequence,7);
  assert.equal(selected.subjects[0].anchor_sequence,0);
  assert.equal(selected.subjects[0].method,"image_motion");assert.equal(selected.exact,false);
  assert.ok(Math.abs(selected.subjects[0].polygons[0][0][0]-(.3+2/79))<1e-9);
});
test("run, capture generation, crop and source timestamp all gate geometry",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));
  for(const change of [{stream_session_id:"run-b"},{generation:3},{crop_version:1}])
    assert.equal(history.acceptMetadata({...metadata([anchor()]),...change}),false);
  history.acceptMetadata(metadata([anchor(0,.01)]));assert.equal(history.select(0,0).coverage,"no_anchor");
  history.acceptMetadata(metadata([anchor()]));assert.equal(history.select(0,.01).coverage,"missing_source");
  assert.equal(history.select(0,0).subjects.length,1);
});
test("future 人员追踪模块 sources never paint onto earlier display frames",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));history.appendFrame(frame(1,.066));
  history.acceptMetadata(metadata([anchor(1,.066)]));assert.equal(history.select(0,0).coverage,"no_anchor");
  assert.equal(history.select(1,.066).subjects[0].method,"tracking_observed");
});
test("motion gaps, expired anchors and missing pixels yield explicit missing coverage",()=>{
  const history=new ContourHistory({...config,max_propagation_seconds:.1});
  history.appendFrame(frame(0,0));history.acceptMetadata(metadata([anchor()]));
  history.appendFrame(frame(2,.3));assert.equal(history.select(2,.3).coverage,"anchor_expired");
  const gaps=new ContourHistory(config);gaps.appendFrame(frame(0,0));gaps.acceptMetadata(metadata([anchor()]));
  gaps.appendFrame(frame(2,.3));assert.equal(gaps.select(2,.3).coverage,"motion_gap");
});
test("new 人员追踪模块 anchor corrects propagation without mutating the analysis mask",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));history.appendFrame(frame(1,.066));
  const first=anchor(),next=anchor(1,.066);next.subjects[0].polygons=[[[.4,.4],[.7,.4],[.7,.7]]];
  history.acceptMetadata(metadata([first,next]));const selected=history.select(1,.066);
  assert.deepEqual(selected.subjects[0].polygons,next.subjects[0].polygons);
  assert.equal(selected.exact,true);assert.deepEqual(first.subjects[0].polygons,shape);
});
test("event colors require same epoch and a valid source-time interval",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));history.appendFrame(frame(1,.066));
  history.acceptMetadata(metadata([anchor()],[event(1,.05)]));
  assert.equal(history.select(0,0).subjects[0].alert,false);
  assert.equal(history.select(1,.066).subjects[0].alert,true);
  const other=new ContourHistory(config);other.appendFrame(frame(0,0));
  other.acceptMetadata(metadata([anchor()],[{...event(1,0),epoch:"old-epoch"}]));
  assert.equal(other.select(0,0).subjects[0].alert,false);
});
test("stale revisions cannot restore an ended event",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));
  history.acceptMetadata(metadata([anchor()],[event(2,0,"ended")]));
  history.acceptMetadata(metadata([],[event(1,0)]));
  assert.equal(history.select(0,0).subjects[0].alert,false);
});
test("a future event clear does not clear buffered frames before its source time",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));history.appendFrame(frame(1,.066));
  history.acceptMetadata(metadata([anchor()],[event(1,0),event(2,.05,"ended")]));
  assert.equal(history.select(0,0).subjects[0].alert,true);
  assert.equal(history.select(1,.066).subjects[0].alert,false);
});
test("full metadata omission closes an alert at snapshot source time and rejects stale resurrection",()=>{
  const history=new ContourHistory(config);history.appendFrame(frame(0,0));history.appendFrame(frame(1,.066));
  const first={...metadata([anchor()],[event(1,0)]),event_snapshot_timestamp:0,event_snapshot_revision:1};
  history.acceptMetadata(first);
  history.acceptMetadata({...metadata([]),event_snapshot_timestamp:.05,event_snapshot_revision:2});
  assert.equal(history.select(0,0).subjects[0].alert,true);
  assert.equal(history.select(1,.066).subjects[0].alert,false);
  history.acceptMetadata(first);assert.equal(history.select(1,.066).subjects[0].alert,false);
});
test("GVC2 upload carries immutable origin metadata and the original JPEG bytes",()=>{
  const meta={...config,source_sequence:42,media_timestamp:1.2,width:640,height:480};
  const jpeg=new Uint8Array([255,216,10,255,217]),packet=new Uint8Array(packCameraFrame(meta,jpeg));
  assert.equal(new TextDecoder().decode(packet.slice(0,4)),"GVC2");
  const size=new DataView(packet.buffer).getUint32(4);
  assert.deepEqual(JSON.parse(new TextDecoder().decode(packet.slice(8,8+size))),meta);
  assert.deepEqual(packet.slice(8+size),jpeg);
});
