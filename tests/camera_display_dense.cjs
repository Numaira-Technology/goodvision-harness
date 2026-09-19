const {test}=require("node:test");
const assert=require("node:assert/strict");
const {createDenseMotionAdapter}=require("../src/goodvision_harness/camera_display_dense.js");
const runtime=require("../src/goodvision_harness/assets/opencv-4.13.0.js");
// OpenCV's legacy object is self-thenable: resolving it directly never settles.
const ready=new Promise(resolve=>{
  if(runtime.Mat)resolve({cv:runtime});else runtime.onRuntimeInitialized=()=>resolve({cv:runtime});
});
const width=160,height=120;
const rect=(x,y,w,h)=>[[x,y],[x+w,y],[x+w,y+h],[x,y+h]];
const normalized=parts=>parts.map(p=>p.map(([x,y])=>[(x+.5)/width,(y+.5)/height]));
// Real worker contract, including its six-decimal wire precision. These source
// pixels can belong to a different-sized 人员追踪模块 mask than the motion working grid.
const samNormalized=(parts,w,h)=>parts.map(p=>p.map(([x,y])=>[
  Math.round((x+.5)/w*1e6)/1e6,Math.round((y+.5)/h*1e6)/1e6]));
function inside(x,y,p){let result=false;for(let i=0,j=p.length-1;i<p.length;j=i++){
  const a=p[i],b=p[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])result=!result;
}return result;}
function noise(x,y,seed){let value=(Math.imul(x+1000,1103515245)^Math.imul(y+1000,12345)^Math.imul(seed,2654435761))>>>0;
  value^=value>>>13;value=Math.imul(value,1274126177);return(value>>>24)&255;}
const texture=(x,y,seed)=>Math.round((noise(x,y,seed)+noise(Math.floor(x/4),Math.floor(y/4),seed+4))/2);
function pixels(parts,{dx=0,seed=19}={}){const out=new Uint8Array(width*height);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)out[y*width+x]=
    parts.some(p=>inside(x-dx,y,p))?texture(x-dx,y,seed):texture(x,y,4);
  return out;
}
function bounds(polygons){return polygons.map(p=>({left:Math.min(...p.map(q=>q[0]))*width-.5,
  right:Math.max(...p.map(q=>q[0]))*width-.5,bottom:Math.max(...p.map(q=>q[1]))*height-.5}));}
function exactMotion(dx,dy=0){
  const sourceMap=new Float32Array(width*height*2);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    sourceMap[(y*width+x)*2]=x-dx;sourceMap[(y*width+x)*2+1]=y-dy;
  }
  return {kind:"dense_flow",width,height,sourceMap,cycleError:new Uint8Array(width*height),
    appearanceError:new Uint8Array(width*height),matchingPrevious:new Uint8Array(width*height).fill(150),
    matchingCurrent:new Uint8Array(width*height).fill(150)};
}

test("人员追踪模块 pixel-center normalization aligns a tiny far-right head with resized image pixels",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),
    source=samNormalized([rect(360,48,6,8)],384,288),motion=exactMotion(0),
    result=adapter.warpPolygons(source,motion);
  assert.ok(result);assert.deepEqual(bounds(result),[{left:150,right:152,bottom:23}]);
  assert.equal(Math.min(...result[0].map(p=>p[1]))*height-.5,20);
  assert.ok(result[0].some(([x])=>x===150.5/160),"Output retains the 人员追踪模块 pixel-center convention");
  assert.equal(adapter.getPresentation(result).complete,true);
});

test("人员追踪模块 pixel-center mapping preserves tiny masks at both actual camera corners",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),
    source=samNormalized([rect(0,0,6,8),rect(378,279,5,8)],384,288),
    result=adapter.warpPolygons(source,exactMotion(0));
  assert.ok(result);assert.equal(result.length,2);
  const edges=bounds(result).sort((a,b)=>a.left-b.left);
  assert.deepEqual(edges,[{left:0,right:2,bottom:3},{left:157,right:159,bottom:119}]);
  assert.equal(Math.min(...result.flat().map(p=>p[0])),.5/width);
  assert.equal(Math.max(...result.flat().map(p=>p[1])),(height-.5)/height);
});

test("a representable tiny 人员追踪模块 component stays present alongside the large body",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),
    source=[...normalized([rect(30,35,40,65)]),...samNormalized([rect(360,48,6,8)],384,288)],
    result=adapter.warpPolygons(source,exactMotion(0));
  assert.ok(result);assert.equal(result.length,2);assert.equal(adapter.getPresentation(result).complete,true);
  assert.deepEqual(bounds(result).find(part=>part.left>100),{left:150,right:152,bottom:23});
});

test("an unrepresentable 人员追踪模块 component cannot disappear from an allegedly complete silhouette",async()=>{
  const {cv}=await ready;
  // This 8x8 native component is well above 人员追踪模块's four-pixel speckle cutoff,
  // but it becomes a 2x2 raster component (contour area 1) at 160x120.
  const tiny=samNormalized([rect(400,200,7,7)],640,480);
  for(const source of [tiny,[...normalized([rect(30,25,45,70)]),...tiny]]){
    const adapter=createDenseMotionAdapter(cv),motion=exactMotion(0);
    assert.equal(adapter.warpPolygons(source,motion),null);
    assert.equal(motion.last_warp_rejection,"invalid_polygon");
    assert.equal(motion.last_warp_diagnostics.reliable_fraction,1,"No confidence threshold was relaxed");
  }
});

test("real 人员追踪模块 pixel-center anchors retain their positions in both flow directions",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),
    earlier=samNormalized([rect(300,100,10,14)],320,240),
    later=samNormalized([rect(304,100,10,14)],320,240),
    forward={...exactMotion(2),forwardMap:exactMotion(-2).sourceMap},
    projected=adapter.warpPolygons(earlier,forward),
    reversed=adapter.warpPolygons(later,adapter.reverseMotion(forward));
  assert.ok(projected);assert.ok(reversed);
  assert.deepEqual(bounds(projected),[{left:152,right:157,bottom:57}]);
  assert.deepEqual(bounds(reversed),[{left:150,right:155,bottom:57}]);
  // An ordinary serialized/reloaded result has no WeakMap cache. It must still
  // rasterize on exactly the same grid instead of accumulating a center offset.
  assert.deepEqual(adapter.warpPolygons(JSON.parse(JSON.stringify(reversed)),exactMotion(0)),reversed);
});

test("official OpenCV dense correspondence follows an independently moving textured subject",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),parts=[rect(40,25,60,70)],
    original=normalized(parts),frozen=JSON.stringify(original),
    motion=adapter.estimateMotion(pixels(parts),pixels(parts,{dx:2}),width,height),result=adapter.warpPolygons(original,motion);
  assert.ok(result);assert.ok(Math.abs(bounds(result)[0].left-42)<2);
  assert.equal(JSON.stringify(original),frozen,"Predictions do not change the 人员追踪模块 anchor");
  assert.equal(motion.kind,"dense_flow");assert.ok(motion.compute_ms>=0);
});

test("reverse propagation reuses the two existing Farneback solves and owns correctly oriented pixels",async()=>{
  const {cv}=await ready,observing=Object.create(cv);let calls=0;
  observing.calcOpticalFlowFarneback=(...args)=>{calls++;return cv.calcOpticalFlowFarneback(...args);};
  const adapter=createDenseMotionAdapter(observing),parts=[rect(40,25,60,70)],a=pixels(parts),b=pixels(parts,{dx:2}),
    motion=adapter.estimateMotion(a,b,width,height),reverse=adapter.reverseMotion(motion),
    result=adapter.warpPolygons(normalized([rect(42,25,60,70)]),reverse);
  assert.equal(calls,2,"Reverse lookup must not invoke Farneback again");
  assert.equal(reverse.previous,b);assert.equal(reverse.current,a);
  assert.equal(reverse.matchingPrevious,motion.matchingCurrent);assert.equal(reverse.matchingCurrent,motion.matchingPrevious);
  assert.equal(reverse.chain_index,undefined);assert.equal(adapter.reverseMotion(motion),reverse);
  assert.equal(adapter.reverseMotion(reverse),motion);assert.ok(result);
  assert.ok(Math.abs(bounds(result)[0].left-40)<2);
});

test("reverse confidence is evaluated on the earlier target grid, including invalid lookups",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),motion=exactMotion(2),forwardMap=new Float32Array(width*height*2);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const at=y*width+x;forwardMap[at*2]=x+2;forwardMap[at*2+1]=y;
    motion.matchingPrevious[at]=x;motion.matchingCurrent[at]=Math.max(0,x-2);
  }
  motion.forwardMap=forwardMap;
  motion.sourceMap[(60*width+52)*2]+=3;
  motion.matchingCurrent[61*width+52]+=40;
  const reverse=adapter.reverseMotion(motion);
  assert.equal(reverse.sourceMap[(60*width+50)*2],52);
  assert.equal(reverse.cycleError[60*width+50],96,"Round-trip error belongs at the earlier target pixel");
  assert.equal(reverse.appearanceError[61*width+50],40);
  assert.equal(motion.cycleError[60*width+50],0,"The reverse operation does not mutate forward evidence");
  assert.equal(reverse.cycleError[60*width+width-1],255);
  assert.equal(reverse.appearanceError[60*width+width-1],255);
});

test("exact reversed correspondence follows source order from a later anchor to an earlier buffered frame",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),
    a=new Uint8Array(width*height).fill(150),b=new Uint8Array(a),c=new Uint8Array(a),
    ab={...exactMotion(2),forwardMap:exactMotion(-2).sourceMap,previous:a,current:b},
    bc={...exactMotion(2),forwardMap:exactMotion(-2).sourceMap,previous:b,current:c};
  ab.chain_index=0;bc.chain_index=1;adapter.prepareMotion(ab);adapter.prepareMotion(bc);
  const reverseAB=adapter.reverseMotion(ab),reverseBC=adapter.reverseMotion(bc),source=normalized([rect(44,25,59,69)]),
    result=adapter.warpChain(source,[reverseBC,reverseAB]);
  assert.ok(result,JSON.stringify({reason:reverseAB.last_warp_rejection,diagnostics:reverseAB.last_warp_diagnostics}));
  assert.ok(Math.abs(bounds(result)[0].left-40)<2);
  assert.deepEqual(adapter.warpChain(source,[reverseBC,reverseAB]),result,"Warm reuse does not change the geometry");
  assert.equal(adapter.warpChain(source,[reverseAB,reverseBC]),null,"Source order is mandatory in either direction");
  assert.equal(reverseBC.last_warp_rejection,"invalid_input");
});

test("a future person or hand is not drawn into a visibly empty or occluded earlier target",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),torso=rect(50,25,45,70),hand=rect(106,48,10,18),
    cases=[{earlier:[],later:[torso]},{earlier:[torso],later:[torso,hand]}];
  for(const {earlier,later} of cases){
    const motion=adapter.estimateMotion(pixels(earlier),pixels(later),width,height),reverse=adapter.reverseMotion(motion);
    assert.equal(adapter.warpPolygons(normalized(later),reverse),null);
    assert.ok(reverse.last_warp_rejection);
  }
});

test("dense appearance checks reject a disappeared or replaced subject",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),parts=[rect(40,25,60,70)],before=pixels(parts);
  for(const after of [pixels([]),pixels(parts,{seed:41})]){
    const motion=adapter.estimateMotion(before,after,width,height);
    assert.equal(adapter.warpPolygons(normalized(parts),motion),null);
    assert.ok(motion.last_warp_rejection);
  }
});

test("a large visible torso cannot validate a disconnected hand that disappeared",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),torso=rect(50,25,45,70),hand=rect(106,48,10,18),
    motion=adapter.estimateMotion(pixels([torso,hand]),pixels([torso]),width,height);
  assert.equal(adapter.warpPolygons(normalized([torso,hand]),motion),null);
  assert.equal(motion.last_warp_rejection,"component_appearance");
  assert.equal(motion.last_warp_diagnostics.components.length,2);
});

test("100 composed quarter-pixel steps preserve thin components without resampling erosion",async()=>{
  const {cv}=await ready;
  for(const thickness of [2,3,4]){
    const adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,35,65),rect(100,42,thickness,20)]);
    let shape=source;
    for(let frame=1;frame<=100;frame++){
      shape=adapter.warpPolygons(shape,exactMotion(.25));assert.ok(shape);assert.equal(shape.length,2);
    }
    const small=bounds(shape).sort((a,b)=>b.left-a.left)[0];
    assert.ok(Math.abs(small.left-125)<1);assert.ok(Math.abs(small.right-small.left-thickness)<.1);
  }
});

test("a connected two-pixel leg retains its full length after 100 composed steps",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv);
  let shape=normalized([[[45,25],[80,25],[80,90],[62,90],[62,112],[60,112],[60,90],[45,90]]]);
  for(let frame=1;frame<=100;frame++){
    shape=adapter.warpPolygons(shape,exactMotion(.25));assert.ok(shape);
    assert.ok(Math.abs(bounds(shape)[0].bottom-112)<.1);
  }
});

test("a late anchor samples its original mask once after composing a long motion chain",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,35,65),rect(100,42,2,20)]),
    motions=Array.from({length:100},()=>exactMotion(.25)),result=adapter.warpChain(source,motions);
  assert.ok(result);assert.equal(result.length,2);
  const small=bounds(result).sort((a,b)=>b.left-a.left)[0];
  assert.ok(Math.abs(small.left-125)<1);assert.ok(Math.abs(small.right-small.left-2)<.1);
  assert.equal(motions.at(-1).last_warp_diagnostics.chain_steps,100);
});

test("a clean final pair cannot erase unreliable evidence from an earlier chain step",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,35,65)]),
    motions=Array.from({length:20},()=>exactMotion(.25));
  motions[5].appearanceError.fill(200);
  assert.equal(adapter.warpChain(source,motions),null);
  assert.equal(motions.at(-1).last_warp_rejection,"component_appearance");
});

test("shared eight-step blocks match sequential translation at every anchor offset",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,35,65)]),
    motions=Array.from({length:64},(_,chain_index)=>({...exactMotion(.25),chain_index}));
  for(const motion of motions)adapter.prepareMotion(motion);
  for(let start=0;start<8;start++){
    const chain=motions.slice(start,start+48),result=adapter.warpChain(source,chain),diagnostics=chain.at(-1).last_warp_diagnostics,
      sequential=createDenseMotionAdapter(cv).warpChain(source,chain.map(({chain_index,...motion})=>motion));
    assert.deepEqual(result,sequential);assert.ok(diagnostics.chain_blocks>=5);
    assert.ok(diagnostics.chain_combinations<=18);
  }
});

test("block reuse preserves intermediate occlusion and rejects a changed source chain",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,35,65)]),
    motions=Array.from({length:56},(_,chain_index)=>({...exactMotion(.25),chain_index}));
  motions[24].appearanceError.fill(200);
  for(const motion of motions)adapter.prepareMotion(motion);
  for(let start=0;start<8;start++){
    assert.equal(adapter.warpChain(source,motions.slice(start,start+40)),null);
    assert.equal(motions[start+39].last_warp_rejection,"component_appearance");
  }
  const wrong=motions.slice(0,16);wrong[7]={...wrong[7],chain_index:99};
  assert.equal(adapter.warpChain(source,wrong),null);assert.equal(wrong.at(-1).last_warp_rejection,"invalid_input");
  const disconnected=motions.slice(0,16);
  disconnected[6]={...disconnected[6],current:new Uint8Array(1)};
  disconnected[7]={...disconnected[7],previous:new Uint8Array(1)};
  assert.equal(adapter.warpChain(source,disconnected),null);
  assert.equal(disconnected.at(-1).last_warp_rejection,"invalid_input");
});

test("block composition has bounded error for smooth deformation and stable cold/warm output",async()=>{
  const {cv}=await ready,source=normalized([rect(45,25,35,65)]),motions=Array.from({length:56},(_,chain_index)=>{
    const motion={...exactMotion(.2),chain_index};
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      motion.sourceMap[(y*width+x)*2]+=.11*Math.sin(y/13+chain_index/8);
      motion.sourceMap[(y*width+x)*2+1]+=.08*Math.sin(x/17-chain_index/7);
    }
    return motion;
  });
  let captured;const observing=Object.create(cv);
  observing.remap=(source,target,map,...rest)=>{captured=new Float32Array(map.data32F);cv.remap(source,target,map,...rest);};
  for(let start=0;start<8;start++){
    const adapter=createDenseMotionAdapter(observing),chain=motions.slice(start,start+40),cold=adapter.warpChain(source,chain),
      blocked=new Float32Array(captured),warm=adapter.warpChain(source,chain);
    assert.deepEqual(warm,cold);assert.deepEqual(captured,blocked);
    createDenseMotionAdapter(observing).warpChain(source,chain.map(({chain_index,...motion})=>motion));
    let maximum=0;
    for(let y=10;y<height-10;y++)for(let x=20;x<width-20;x++){
      const at=(y*width+x)*2;maximum=Math.max(maximum,Math.hypot(blocked[at]-captured[at],blocked[at+1]-captured[at+1]));
    }
    assert.ok(maximum<.1,`Block regrouping error ${maximum} exceeds a tenth of a motion pixel`);
  }
});

test("connected hand, arm and head disappearance cannot borrow the visible torso's confidence",async()=>{
  const {cv}=await ready,person=[[64,12],[81,12],[88,20],[87,34],[99,37],[110,69],[102,73],[94,52],
    [96,103],[82,105],[77,70],[73,105],[60,104],[61,51],[50,76],[43,71],[53,39],[65,33],[59,23]],before=pixels([person]);
  for(const removed of [(x,y)=>x>=100&&y>=53,(x,y)=>x>=96&&y>=36,(x,y)=>y<33]){
    const after=new Uint8Array(before),adapter=createDenseMotionAdapter(cv);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(removed(x,y))after[y*width+x]=texture(x,y,4);
    const motion=adapter.estimateMotion(before,after,width,height);
    assert.equal(adapter.warpPolygons(normalized([person]),motion),null);assert.ok(motion.last_warp_rejection);
  }
});

test("partial presentation keeps the verified torso and omits a disappeared disconnected hand",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),torso=rect(50,25,45,70),hand=rect(106,48,10,18),
    before=pixels([torso,hand]),after=pixels([torso]),source=normalized([torso,hand]),
    motion=adapter.estimateMotion(before,after,width,height),shape=adapter.warpPresentation(source,motion),
    presentation=adapter.getPresentation(shape);
  assert.ok(shape);assert.equal(presentation.complete,false);assert.deepEqual(presentation.polygons,[]);
  assert.ok(presentation.open_segments.length>0);
  assert.ok(presentation.open_segments.flat().every(([x])=>x*width-.5<102),"No hand or invented bridge is exposed");
  assert.ok(presentation.open_segments.flat().some(([,y])=>y*height-.5>85),"The visible lower torso remains outlined");
  assert.equal(adapter.getPresentation(source).complete,true,"A fresh real 人员追踪模块 anchor retains complete semantics");
  let propagated=shape;
  for(let step=0;step<5;step++){
    const still=adapter.estimateMotion(after,after,width,height);propagated=adapter.warpPresentation(propagated,still);
    assert.ok(propagated);assert.equal(adapter.getPresentation(propagated).complete,false,
      "The incomplete prediction cannot validate its own missing hand against new background");
  }
});

test("partial presentation leaves an open gap at a disappearing connected hand",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),person=[[64,12],[81,12],[88,20],[87,34],[99,37],[110,69],
    [102,73],[94,52],[96,103],[82,105],[77,70],[73,105],[60,104],[61,51],[50,76],[43,71],[53,39],[65,33],[59,23]],
    before=pixels([person]),after=new Uint8Array(before);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(x>=100&&y>=53)after[y*width+x]=texture(x,y,4);
  const motion=adapter.estimateMotion(before,after,width,height),shape=adapter.warpPresentation(normalized([person]),motion),
    presentation=adapter.getPresentation(shape);
  assert.ok(presentation);assert.equal(presentation.complete,false);assert.deepEqual(presentation.polygons,[]);
  assert.ok(presentation.open_segments.length>0);
  assert.ok(presentation.open_segments.every(segment=>JSON.stringify(segment[0])!==JSON.stringify(segment.at(-1))),
    "The connected body's missing boundary remains a gap");
  assert.ok(presentation.open_segments.flat().every(([x,y])=>!(x*width-.5>=100&&y*height-.5>=53)));
});

test("partial presentation does not override whole-subject disappearance or replacement",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),parts=[rect(40,25,60,70)],before=pixels(parts);
  for(const after of [pixels([]),pixels(parts,{seed:41})]){
    const motion=adapter.estimateMotion(before,after,width,height);
    assert.equal(adapter.warpPresentation(normalized(parts),motion),null);
  }
});

test("partial crop continuation preserves verified outer arcs without inventing a closing edge",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,60,94)]),gone=exactMotion(0,-1);
  gone.matchingCurrent.fill(0,(height-1)*width);
  assert.equal(adapter.warpPolygons(source,gone),null,"The strict complete-contour interface still rejects this crop");
  const shape=adapter.warpPresentation(source,gone),view=adapter.getPresentation(shape);
  assert.ok(view);assert.equal(view.complete,false);assert.deepEqual(view.polygons,[]);
  assert.ok(view.open_segments.flat().some(([,y])=>y*height-.5<30),"Verified upper body is retained");
  assert.ok(view.open_segments.flat().every(([,y])=>y*height-.5<height-4),"The unknown camera edge remains open");
  assert.ok(view.open_segments.every(segment=>JSON.stringify(segment[0])!==JSON.stringify(segment.at(-1))));
});

test("a cached partial result retains earlier cycle failure when later appearance and motion look clean",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(50,25,45,70),rect(106,48,10,18)]),
    ambiguous=exactMotion(0);
  for(let y=45;y<=69;y++)for(let x=103;x<=119;x++)ambiguous.cycleError[y*width+x]=160;
  const first=adapter.warpPresentation(source,ambiguous),firstView=adapter.getPresentation(first);
  assert.ok(first);assert.equal(firstView.complete,false);
  assert.ok(firstView.open_segments.flat().every(([x])=>x*width-.5<102));
  let cached=first;
  for(let step=0;step<5;step++){
    cached=adapter.warpPresentationChain(cached,[exactMotion(0),exactMotion(0)]);
    const view=adapter.getPresentation(cached);
    assert.ok(view);assert.equal(view.complete,false,"Cached cycle failure cannot reset at a clean new segment");
    assert.ok(view.open_segments.flat().every(([x])=>x*width-.5<102));
  }
  assert.equal(adapter.getPresentation(source).complete,true,"Only a new real anchor resets the propagation evidence");
});

test("camera crop continuation needs current appearance support and refuses an actual departure",async()=>{
  const {cv}=await ready,source=normalized([rect(45,25,60,94)]),adapter=createDenseMotionAdapter(cv),motion=exactMotion(0,-1),
    continued=adapter.warpPolygons(source,motion);
  assert.ok(continued);assert.equal(Math.max(...continued.flat().map(p=>p[1])),(height-.5)/height);
  const gone=exactMotion(0,-1);gone.matchingCurrent.fill(0,(height-1)*width);
  assert.equal(adapter.warpPolygons(source,gone),null);assert.equal(gone.last_warp_rejection,"crop_continuation");
});

test("uninitialized runtime and inconsistent pixel dimensions fail explicitly",async()=>{
  assert.throws(()=>createDenseMotionAdapter({}),/not initialized/);
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv);
  assert.throws(()=>adapter.estimateMotion(new Uint8Array(1),new Uint8Array(1),width,height),/invalid dense motion pixels/);
});

test("display interpolation tolerates a local crop edge but stays explicitly non-strict",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,60,94)]),motion=exactMotion(0,-1);
  motion.matchingCurrent.fill(0,(height-1)*width);
  assert.equal(adapter.warpChain(source,[motion]),null);assert.equal(motion.last_warp_rejection,"crop_continuation");
  const result=adapter.warpApproximateChain(source,[motion]),view=adapter.getPresentation(result);
  assert.ok(result);assert.equal(view.complete,false);assert.equal(view.approximate,true);
  assert.deepEqual(view.polygons,result);assert.deepEqual(view.open_segments,[]);assert.equal(view.reason,"crop_continuation");
  assert.equal(adapter.warpPolygons(result,exactMotion(0)),null,"An approximation cannot seed a later strict result");
});

test("approximate crop display can omit four isolated raster specks but not an observed source component",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,60,94)]);
  const make=count=>{
    const motion=exactMotion(0,-1);motion.matchingCurrent.fill(0,(height-1)*width);
    for(let i=0;i<count;i++){
      const at=10*width+5+i*3;motion.sourceMap[at*2]=60;motion.sourceMap[at*2+1]=50;
    }
    return motion;
  };
  const motion=make(4);assert.equal(adapter.warpChain(source,[motion]),null);
  const result=adapter.warpApproximateChain(source,[motion]);assert.ok(result);
  assert.equal(adapter.getPresentation(result).approximate,true);assert.equal(result.length,1);
  assert.equal(motion.last_warp_diagnostics.discarded_fragment_pixels,4);
  const tooMany=make(5);assert.equal(adapter.warpApproximateChain(source,[tooMany]),null);
  assert.equal(tooMany.last_warp_rejection,"invalid_polygon");
  const observed=[...source,...normalized([rect(120,15,1,1)])];
  assert.equal(adapter.warpApproximateChain(observed,[make(0)]),null,
    "A real disconnected small component must not be relabelled as a raster speck");
});

test("a local crop warning cannot hide a failed disconnected component during interpolation",async()=>{
  const {cv}=await ready,adapter=createDenseMotionAdapter(cv),source=normalized([rect(45,25,60,94),rect(115,45,12,18)]),
    motion=exactMotion(0,-1);
  motion.matchingCurrent.fill(0,(height-1)*width);
  for(let y=43;y<65;y++)for(let x=113;x<130;x++)motion.appearanceError[y*width+x]=100;
  assert.equal(adapter.warpApproximateChain(source,[motion]),null);
  assert.equal(motion.last_warp_rejection,"component_appearance");
  assert.ok(motion.last_warp_diagnostics.crop_rejected_pixels>0);
  assert.ok(motion.last_warp_diagnostics.mean_appearance_error<25,"Whole-person mean cannot excuse the failed hand");
});

test("interpolation does not bypass global appearance, internal support, region or area gates",async()=>{
  const {cv}=await ready,source=normalized([rect(40,25,60,70)]);
  for(const fail of ["appearance","internal","region","area"]){
    const adapter=createDenseMotionAdapter(cv),motion=exactMotion(0);
    if(fail==="appearance")motion.appearanceError.fill(100);
    if(fail==="internal")motion.cycleError.fill(160);
    if(fail==="region")for(let y=24;y<=60;y++)for(let x=39;x<=70;x++)motion.cycleError[y*width+x]=160;
    if(fail==="area")for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      motion.sourceMap[(y*width+x)*2]=(x-70)*5+70;motion.sourceMap[(y*width+x)*2+1]=(y-60)*5+60;
    }
    assert.equal(adapter.warpApproximateChain(source,[motion]),null,fail);
    assert.equal(motion.last_warp_rejection,{appearance:"component_appearance",internal:"internal_support",region:"region_occluded",area:"invalid_geometry"}[fail]);
  }
});

test("actual disappearance and scene replacement stay rejected by approximate propagation",async()=>{
  const {cv}=await ready,parts=[rect(40,25,60,70)],before=pixels(parts);
  for(const after of [pixels([]),pixels(parts,{seed:41}),new Uint8Array(width*height).fill(240)]){
    const adapter=createDenseMotionAdapter(cv),motion=adapter.estimateMotion(before,after,width,height);
    assert.equal(adapter.warpApproximateChain(normalized(parts),[motion]),null);
    assert.ok(["invalid_geometry","component_appearance","internal_support","region_occluded","component_visibility"].includes(motion.last_warp_rejection));
  }
});
