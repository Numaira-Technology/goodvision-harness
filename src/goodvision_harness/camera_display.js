/* Shared camera/network presentation. Predicted contours never become analysis evidence. */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  else root.GoodvisionCameraDisplay=api;
  if(typeof document==="undefined"&&typeof importScripts==="function"){
    let history=null,loading=false,revision=0,pending=[],grayCanvas=null,grayContext=null,forward=null,
      runtimePromise=null,runtimeLoadMs=null,trace=null;
    function stopTrace(reason){
      if(!trace)return;trace=null;
      try{root.postMessage({type:"motion_trace_stopped",complete:false,reason});}catch(_){}
    }
    function traceInput(message){
      if(!trace)return null;
      try{
        if(performance.now()-trace.started>=trace.maxSeconds*1000){stopTrace("duration_limit");return null;}
        const bytes=message.type==="frame"?message.frame.pixels.byteLength+512:JSON.stringify(message).length*2+256;
        if(trace.bytes+bytes>trace.maxBytes){stopTrace("byte_limit");return null;}
        const copy=message.type==="frame"?{...message,frame:{...message.frame,pixels:message.frame.pixels.slice()}}:message;
        trace.bytes+=bytes;return {...copy,trace_id:trace.next++};
      }catch(_){stopTrace("copy_failure");return null;}
    }
    function reply(input,message){
      if(Number.isSafeInteger(input.trace_id))message.trace_id=input.trace_id;
      root.postMessage(message);
    }
    function release(message){message.frame?.bitmap?.close();}
    function expireFrame(message){
      if(!["frame","frame_bitmap"].includes(message.type)||
        !Number.isFinite(message.motion_deadline_epoch_ms)||Date.now()<=message.motion_deadline_epoch_ms)return false;
      release(message);
      reply(message,{type:"frame_processed",sequence:message.frame.sequence,
        queued_at:message.queued_at,accepted:false,expired:true,compute_ms:0});return true;
    }
    function releasePending(){for(const message of pending)release(message);pending=[];}
    function prepareBitmap(message){
      const frame=message.frame,started=performance.now(),bitmap=frame.bitmap;
      try{
        if(!grayCanvas){grayCanvas=new OffscreenCanvas(frame.width,frame.height);
          grayContext=grayCanvas.getContext("2d",{willReadFrequently:true});}
        if(grayCanvas.width!==frame.width)grayCanvas.width=frame.width;
        if(grayCanvas.height!==frame.height)grayCanvas.height=frame.height;
        grayContext.drawImage(bitmap,0,0,frame.width,frame.height);
        const rgba=grayContext.getImageData(0,0,frame.width,frame.height).data;
        const pixels=new Uint8Array(frame.width*frame.height);
        for(let i=0;i<pixels.length;i++)pixels[i]=(rgba[i*4]*77+rgba[i*4+1]*150+rgba[i*4+2]*29)>>8;
        return {...message,type:"frame",preprocess_ms:performance.now()-started,
          frame:{sequence:frame.sequence,timestamp:frame.timestamp,width:frame.width,height:frame.height,pixels}};
      }finally{bitmap.close();}
    }
    async function encode(message){
      const bitmap=message.bitmap,started=performance.now();
      try{
        const canvas=new OffscreenCanvas(bitmap.width,bitmap.height);
        canvas.getContext("2d").drawImage(bitmap,0,0);
        const blob=await canvas.convertToBlob({type:"image/jpeg",quality:message.quality});
        const bytes=await blob.arrayBuffer();
        root.postMessage({type:"encoded_jpeg",sequence:message.sequence,bytes,
          compute_ms:performance.now()-started},[bytes]);
      }catch(error){root.postMessage({type:"encoding_failure",sequence:message.sequence});}
      finally{bitmap.close();}
    }
    function process(message){
      if(expireFrame(message))return;
      try{
        const started=performance.now();
        if(history&&message.type==="frame"){
          const queueMs=Number.isFinite(message.preprocess_forward_epoch_ms)?
            Math.max(0,Date.now()-message.preprocess_forward_epoch_ms):undefined;
          const accepted=history.appendFrame(message.frame);
          reply(message,{type:"frame_processed",sequence:message.frame.sequence,
            queued_at:message.queued_at,accepted,compute_ms:performance.now()-started,
            preprocess_ms:message.preprocess_ms,motion_queue_ms:queueMs});
        }
        else if(history&&message.type==="metadata"){
          const before=history.rejected,revision=history.revision,anchorsBefore=new Map(history.anchors),
            accepted=history.acceptMetadata(message.data),geometryChanged=anchorsBefore.size!==history.anchors.size||
              [...history.anchors].some(([key,anchor])=>anchorsBefore.get(key)!==anchor);
          reply(message,{type:"metadata_processed",accepted,rejected:history.rejected-before,
            changed:history.revision!==revision,geometry_changed:geometryChanged,queued_at:message.queued_at,
            metadata_revision:message.data.metadata_revision,compute_ms:performance.now()-started,
            latest_anchor_timestamp:accepted&&history.anchors.size?
              Math.max(...[...history.anchors.values()].map(anchor=>anchor.media_timestamp)):null});
        }
        else if(history&&message.type==="present"){
          const queueMs=Number.isFinite(message.preprocess_forward_epoch_ms)?
            Math.max(0,Date.now()-message.preprocess_forward_epoch_ms):undefined;
          if(Number.isFinite(message.deadline_epoch_ms)&&Date.now()>message.deadline_epoch_ms){
            reply(message,{type:"present",id:message.id,sequence:message.sequence,
              queued_at:message.queued_at,expired:true,compute_ms:0,motion_queue_ms:queueMs,
              result:{subjects:[],coverage:"presentation_expired"}});return;
          }
          const result=history.select(message.sequence,message.timestamp);
          reply(message,{type:"present",id:message.id,sequence:message.sequence,
            queued_at:message.queued_at,result,geometry_cache_hit:result.geometry_cache_hit,
            compute_ms:performance.now()-started,motion_queue_ms:queueMs});
        }
      }catch(error){reply(message,{type:"failure",request_type:message.type,id:message.id,sequence:message.frame?.sequence??message.sequence});}
    }
    function loadDenseRuntime(){
      if(runtimePromise)return runtimePromise;
      const started=performance.now();
      runtimePromise=Promise.resolve().then(()=>{
        if(!root.GoodvisionDenseMotion)importScripts("/assets/camera_display_dense.js");
        if(!root.cv)importScripts("/assets/display_motion_runtime.js");
        return new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(new Error("motion runtime unavailable")),10000);
        const ready=cv=>{clearTimeout(timer);resolve({cv});};
        // Some runtime distributions expose a self-returning legacy thenable.
        // Awaiting it directly recurses forever; only unwrap native Promises.
        if(root.cv instanceof Promise)root.cv.then(ready,error=>{clearTimeout(timer);reject(error);});
        else if(root.cv?.Mat)ready(root.cv);
        else if(root.cv)root.cv.onRuntimeInitialized=()=>ready(root.cv);
        else{clearTimeout(timer);reject(new Error("motion runtime missing"));}
        });
      }).then(runtime=>{runtimeLoadMs=performance.now()-started;return runtime;})
        .catch(error=>{runtimePromise=null;throw error;});
      return runtimePromise;
    }
    function loadDense(){
      // Only the runtime is shared. Every run gets a new adapter and its own
      // optical-flow caches, then a fresh history with no identities or masks.
      return loadDenseRuntime().then(({cv})=>root.GoodvisionDenseMotion.createDenseMotionAdapter(cv));
    }
    function receive(message){
      if(message.type==="warmup_motion"){
        loadDenseRuntime().then(()=>root.postMessage({type:"motion_warmup_ready",runtime_load_ms:runtimeLoadMs}))
          .catch(()=>root.postMessage({type:"failure",stage:"motion_warmup"}));
        return;
      }
      if(message.type==="motion_input"){
        receive({type:"reset",config:message.config});
        message.port.onmessage=event=>receive(event.data||{});return;
      }
      if(message.type==="preprocess_output"){
        forward=message.port;
        if(message.trace_enabled===true)trace={started:performance.now(),next:0,bytes:0,
          maxBytes:Math.min(128*1024**2,Math.max(1024,Number(message.trace_max_bytes)||128*1024**2)),
          maxSeconds:Math.min(120,Math.max(1,Number(message.trace_max_seconds)||20))};
        return;
      }
      if(message.type==="stop_motion_trace"){stopTrace(message.reason||"requested");return;}
      if(forward){
        if(expireFrame(message))return;
        const started=performance.now(),queueMs=Number.isFinite(message.queued_epoch_ms)?
          Math.max(0,Date.now()-message.queued_epoch_ms):undefined;
        try{
          if(message.type==="frame_bitmap")message=prepareBitmap(message);
          const traced=traceInput(message);
          const transfers=message.type==="frame"?[message.frame.pixels.buffer]:[];
          // Every frame, metadata update and presentation query uses this one
          // port. Independent workers overlap without allowing an anchor to
          // overtake the pixels of its source frame.
          forward.postMessage({...message,...(traced?{trace_id:traced.trace_id}:{}),preprocess_forward_epoch_ms:Date.now()},transfers);
          if(message.type==="frame")root.postMessage({type:"frame_preprocessed",
            sequence:message.frame.sequence,compute_ms:performance.now()-started,queue_ms:queueMs});
          if(traced)try{root.postMessage({type:"motion_trace_input",message:traced},
            traced.type==="frame"?[traced.frame.pixels.buffer]:[]);}catch(_){stopTrace("delivery_failure");}
        }catch(error){reply(message,{type:"failure",stage:"motion_preprocess",request_type:message.type,
          id:message.id,sequence:message.frame?.sequence??message.sequence});}
        return;
      }
      // Encoding has its own Worker and cannot occupy the motion history queue.
      if(message.type==="encode_jpeg"){encode(message);return;}
      if(message.type==="frame_bitmap"){
        if(expireFrame(message))return;
        if(!loading&&!history){release(message);return;}
        try{message=prepareBitmap(message);}
        catch(error){reply(message,{type:"failure",stage:"motion_preprocess",request_type:message.type,
          sequence:message.frame?.sequence});return;}
      }
      if(message.type==="reset"){
        const current=++revision;history=null;releasePending();
        if(message.config?.motion_backend!=="dense"){
          loading=false;history=new api.ContourHistory(message.config);return;
        }
        loading=true;
        Promise.resolve().then(loadDense).then(adapter=>{
          if(current!==revision)return;
          root.postMessage({type:"motion_runtime_ready",runtime_load_ms:runtimeLoadMs});
          history=new api.ContourHistory(message.config,adapter);loading=false;
          const queued=pending,drainStarted=performance.now();pending=[];
          for(const item of queued)process(item);
          root.postMessage({type:"motion_ready",backend:"dense",startup_drain_ms:performance.now()-drainStarted});
        }).catch(()=>{
          if(current!==revision)return;loading=false;releasePending();
          root.postMessage({type:"failure",stage:"motion_initialization"});
        });
      }else if(loading){
        // Retain source order while WASM initializes; never silently skip the
        // pixels needed by the first delayed tracking anchor.
        if(pending.length<512)pending.push(message);
        else{revision++;loading=false;releasePending();release(message);
          root.postMessage({type:"failure",stage:"motion_initialization"});}
      }else process(message);
    }
    root.onmessage=event=>receive(event.data||{});
  }
})(typeof globalThis!=="undefined"?globalThis:this,function(){
  "use strict";
  const finite=value=>typeof value==="number"&&Number.isFinite(value);
  const sequence=value=>Number.isSafeInteger(value)&&value>=0;
  const safeId=value=>(typeof value==="string"&&/^[A-Za-z0-9_.:-]{1,160}$/.test(value))||sequence(value);
  const clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
  const motionRejectionReasons=["invalid_input","internal_support","support_span","affine_geometry",
    "component_visibility","component_appearance","region_occluded","boundary_occluded",
    "crop_continuation","invalid_geometry","invalid_polygon"];
  function polygons(value){
    if(!Array.isArray(value)||!value.length||value.length>16)return false;
    let count=0;
    return value.every(polygon=>Array.isArray(polygon)&&polygon.length>=3&&polygon.length<=128&&
      (count+=polygon.length)<=1024&&polygon.every(point=>Array.isArray(point)&&point.length===2&&
        point.every(number=>finite(number)&&number>=0&&number<=1)));
  }
  function sameRun(a,b){return a.stream_session_id===b.stream_session_id&&
    a.generation===b.generation&&a.crop_version===b.crop_version;}
  const displayName=value=>typeof value==="string"&&value.trim().length>0&&value.length<=160&&
    !/[\u0000-\u001f\u007f]/.test(value)?value.trim():null;
  class DisplayIdentities{
    constructor(config){this.config=config;this.revision=-1;this.records=[];}
    accept(data){
      if(!data||!sameRun(data,this.config)||!sequence(data.identity_snapshot_revision)||
        data.identity_snapshot_revision<=this.revision||!finite(data.identity_snapshot_timestamp)||
        data.identity_snapshot_timestamp<0||!Array.isArray(data.identities)||data.identities.length>256)return false;
      const valid=data.identities.every(item=>item&&safeId(item.epoch)&&safeId(item.track_id)&&
        typeof item.person_id==="string"&&item.person_id.length>0&&item.person_id.length<=256&&displayName(item.name)&&
        sequence(item.revision)&&item.revision<=data.identity_snapshot_revision&&
        finite(item.valid_from_timestamp)&&item.valid_from_timestamp>=0&&
        (item.valid_until_timestamp===null||finite(item.valid_until_timestamp)&&
          item.valid_until_timestamp>=item.valid_from_timestamp));
      if(!valid)return false;
      // This is a full identity snapshot, independent of incremental geometry.
      // Removing a gallery identity must also remove names on buffered old frames.
      this.records=data.identities.map(item=>({...item,name:displayName(item.name)}));
      this.revision=data.identity_snapshot_revision;return true;
    }
    nameAt(subject,timestamp){
      if(subject.tracking_confirmation!=="confirmed")return null;
      const active=this.records.filter(item=>item.epoch===subject.epoch&&item.track_id===subject.track_id&&
        item.valid_from_timestamp<=timestamp&&
        (item.valid_until_timestamp===null||timestamp<item.valid_until_timestamp));
      if(!active.length||active.some(item=>item.person_id!==active[0].person_id||item.name!==active[0].name))return null;
      return active[0].name;
    }
  }

  // Independent media-clock FIFO. A late renderer is measurable, never a newer analysis frame.
  class FrameTimeline{
    constructor({delaySeconds=1,maxFrames=90,dispose=()=>{}}={}){
      this.delay=delaySeconds;this.maxFrames=maxFrames;this.dispose=dispose;
      this.frames=[];this.lastSequence=-1;this.lastTimestamp=-1;this.presented=0;
      this.expired=0;this.catchupExpired=0;this.rejected=0;this.firstPresentedAt=null;this.lastPresentedAt=null;
    }
    push(frame){
      if(!frame||!sequence(frame.sequence)||!finite(frame.timestamp)||frame.timestamp<0||
        frame.sequence<=this.lastSequence||frame.timestamp<=this.lastTimestamp){this.rejected++;return false;}
      this.lastSequence=frame.sequence;this.lastTimestamp=frame.timestamp;this.frames.push(frame);
      while(this.frames.length>this.maxFrames){this.dispose(this.frames.shift());this.expired++;}
      return true;
    }
    discardBefore(timestamp){
      let last=-1;
      while(this.frames.length&&this.frames[0].timestamp<timestamp){
        const frame=this.frames.shift();last=frame.sequence;
        this.dispose(frame);this.expired++;this.catchupExpired++;
      }
      return last;
    }
    take(nowMedia){
      if(!this.frames.length||!finite(nowMedia)||this.frames[0].timestamp>nowMedia-this.delay)return null;
      return this.frames.shift();
    }
    notePresented(frame,now){
      this.presented++;if(this.firstPresentedAt===null)this.firstPresentedAt=now;
      this.lastPresentedAt=now;this.lastPresentedSequence=frame.sequence;
    }
    reset(){for(const frame of this.frames)this.dispose(frame);this.frames=[];}
    stats(){return {distinct_presented_frames:this.presented,display_buffer_expired:this.expired,
      display_catchup_expired:this.catchupExpired,
      rejected_source_frames:this.rejected,buffered_frames:this.frames.length,
      distinct_presented_fps:this.presented>1&&this.lastPresentedAt>this.firstPresentedAt?
        (this.presented-1)*1000/(this.lastPresentedAt-this.firstPresentedAt):null};}
  }

  function patchScore(a,b,w,h,x,y,dx,dy,radius,cutoff=Infinity){
    if(x-radius<0||y-radius<0||x+radius>=w||y+radius>=h||
      x+dx-radius<0||y+dy-radius<0||x+dx+radius>=w||y+dy+radius>=h)return Infinity;
    let sum=0;const area=(2*radius+1)**2,limit=cutoff*area;
    for(let py=-radius;py<=radius;py++){
      const from=(y+py)*w+x,to=(y+dy+py)*w+x+dx;
      for(let px=-radius;px<=radius;px++)sum+=Math.abs(a[from+px]-b[to+px]);
      if(sum>limit)return Infinity;
    }
    return sum/area;
  }
  const searchOrders=new Map();
  function matchPatch(a,b,w,h,x,y,search,radius,initialX=0,initialY=0){
    let best=Infinity,second=Infinity,bx=0,by=0,at=0;
    const scores=new Float32Array((search*2+1)**2);
    let order=searchOrders.get(search);
    if(!order){
      order=[];for(let dy=-search;dy<=search;dy++)for(let dx=-search;dx<=search;dx++)order.push({dx,dy});
      order.sort((a,b)=>a.dx*a.dx+a.dy*a.dy-b.dx*b.dx-b.dy*b.dy);searchOrders.set(search,order);
    }
    for(const offset of order){
      const dx=initialX+offset.dx,dy=initialY+offset.dy,
        score=patchScore(a,b,w,h,x,y,dx,dy,radius,best+1);
      scores[(offset.dy+search)*(2*search+1)+offset.dx+search]=score;
      if(score<best){best=score;bx=dx;by=dy;}
    }
    // Adjacent pixels are the same subpixel minimum, not a competing match.
    at=0;
    for(let dy=initialY-search;dy<=initialY+search;dy++)for(let dx=initialX-search;dx<=initialX+search;dx++){
      const score=scores[at++];
      if(Math.max(Math.abs(dx-bx),Math.abs(dy-by))>1)second=Math.min(second,score);
    }
    return {dx:bx,dy:by,error:best,separation:second-best};
  }
  function imagePyramid(pixels,width,height){
    const result=[{pixels,width,height}];
    while(result.length<3&&width>=40&&height>=32){
      const w=Math.floor(width/2),h=Math.floor(height/2),small=new Uint8Array(w*h);
      for(let y=0;y<h;y++)for(let x=0;x<w;x++){
        const at=y*2*width+x*2;
        small[y*w+x]=(pixels[at]+pixels[at+1]+pixels[at+width]+pixels[at+width+1]+2)>>2;
      }
      result.push({pixels:small,width:w,height:h});pixels=small;width=w;height=h;
    }
    return result;
  }
  // A corner constrains two-dimensional motion. A single straight edge does not.
  function cornerStrength(pixels,width,x,y){
    let xx=0,xy=0,yy=0;
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
      const at=(y+dy)*width+x+dx,gx=(pixels[at+1]-pixels[at-1])/2,
        gy=(pixels[at+width]-pixels[at-width])/2;
      xx+=gx*gx;xy+=gx*gy;yy+=gy*gy;
    }
    return (xx+yy-Math.hypot(xx-yy,2*xy))/18;
  }
  function pyramidMatch(previous,current,x,y,search,radius){
    const a=previous[0],b=current[0],near=matchPatch(a.pixels,b.pixels,a.width,a.height,x,y,search,radius);
    if(near.error<8&&near.separation>=1)return near;
    let estimate=null;
    for(let level=previous.length-1;level>=0;level--){
      const a=previous[level],b=current[level],scale=2**level,
        px=Math.round(x/scale),py=Math.round(y/scale);
      if(px-radius<0||py-radius<0||px+radius>=a.width||py+radius>=a.height)continue;
      const next=matchPatch(a.pixels,b.pixels,a.width,a.height,px,py,
        estimate?2:search,radius,estimate?estimate.dx*2:0,estimate?estimate.dy*2:0);
      estimate=finite(next.error)?next:null;
    }
    // Fine texture can alias at the top of a pyramid. Keep the better verified
    // full-resolution candidate; coarse motion is never accepted on its own.
    if(!estimate||near.error<=estimate.error)estimate=near;
    if(!finite(estimate.error)||estimate.error>28||estimate.separation<1)return null;
    return estimate;
  }
  function refineTranslation(a,b,width,height,x,y,initial,radius){
    const values=[],gx=[],gy=[];let meanX=0,meanY=0;
    for(let py=-radius;py<=radius;py++)for(let px=-radius;px<=radius;px++){
      const value=samplePixel(a,width,height,x+px,y+py),
        left=samplePixel(a,width,height,x+px-1,y+py),right=samplePixel(a,width,height,x+px+1,y+py),
        top=samplePixel(a,width,height,x+px,y+py-1),bottom=samplePixel(a,width,height,x+px,y+py+1);
      if([value,left,right,top,bottom].some(v=>v===null))return null;
      values.push(value);gx.push((right-left)/2);gy.push((bottom-top)/2);
      meanX+=gx.at(-1);meanY+=gy.at(-1);
    }
    meanX/=values.length;meanY/=values.length;
    let xx=0,xy=0,yy=0;
    for(let i=0;i<values.length;i++){
      gx[i]-=meanX;gy[i]-=meanY;xx+=gx[i]*gx[i];xy+=gx[i]*gy[i];yy+=gy[i]*gy[i];
    }
    const determinant=xx*yy-xy*xy;if(determinant<1e-5)return null;
    let dx=initial.dx,dy=initial.dy;
    for(let iteration=0;iteration<8;iteration++){
      let bx=0,by=0,index=0;
      for(let py=-radius;py<=radius;py++)for(let px=-radius;px<=radius;px++){
        const target=samplePixel(b,width,height,x+px+dx,y+py+dy);
        if(target===null)return null;
        const difference=values[index]-target;bx+=gx[index]*difference;by+=gy[index]*difference;index++;
      }
      const sx=(yy*bx-xy*by)/determinant,sy=(xx*by-xy*bx)/determinant;
      if(!finite(sx)||!finite(sy)||Math.hypot(sx,sy)>1.5)return null;
      dx+=sx;dy+=sy;
      if(Math.hypot(dx-initial.dx,dy-initial.dy)>2)return null;
      if(Math.hypot(sx,sy)<.01)break;
    }
    let error=0,index=0;
    for(let py=-radius;py<=radius;py++)for(let px=-radius;px<=radius;px++){
      const target=samplePixel(b,width,height,x+px+dx,y+py+dy);
      if(target===null)return null;error+=Math.abs(values[index++]-target);
    }
    return error/values.length<=28?{dx,dy,error:error/values.length}:null;
  }
  // Motion is estimated once per real image pair. Subject selection happens later,
  // inside each original contour, so background vectors cannot vote for a person.
  function estimateMotion(previous,current,width,height,{step=8,search=5,radius=3}={}){
    if(previous.length!==width*height||current.length!==width*height)throw new Error("invalid pixels");
    if(width<120)step=Math.min(step,6);
    const vectors=[],features=[],before=imagePyramid(previous,width,height),after=imagePyramid(current,width,height);
    let candidates=0;
    for(let gy=radius+2;gy<height-radius-2;gy+=step)
      for(let gx=radius+2;gx<width-radius-2;gx+=step){
        candidates++;let point=null;
        for(let dy=-2;dy<=2;dy+=2)for(let dx=-2;dx<=2;dx+=2){
          const strength=cornerStrength(previous,width,gx+dx,gy+dy);
          if(strength>=25&&(!point||strength>point.strength))point={x:gx+dx,y:gy+dy,strength};
        }
        if(!point)continue;
        features.push(point);
        const {x,y}=point,forward=pyramidMatch(before,after,x,y,search,radius);
        if(!forward)continue;
        const reverse=pyramidMatch(after,before,x+forward.dx,y+forward.dy,search,radius);
        if(!reverse||Math.hypot(forward.dx+reverse.dx,forward.dy+reverse.dy)>1)continue;
        const refined=forward.error<.5?forward:refineTranslation(previous,current,width,height,x,y,forward,radius);
        if(!refined)continue;
        if(forward.error>=.5){
          const back=refineTranslation(current,previous,width,height,x+refined.dx,y+refined.dy,
            {dx:-refined.dx,dy:-refined.dy},radius);
          if(!back||Math.hypot(refined.dx+back.dx,refined.dy+back.dy)>1)continue;
        }
        const vector={...point,...refined};point.vector=vector;vectors.push(vector);
      }
    return {width,height,step,radius,vectors,features,candidates,previous,current,
      validFraction:candidates?vectors.length/candidates:0};
  }
  function insidePolygon(x,y,polygon){
    let inside=false;
    for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
      const a=polygon[i],b=polygon[j];
      if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])inside=!inside;
    }
    return inside;
  }
  function median(values){const sorted=values.slice().sort((a,b)=>a-b);return sorted[Math.floor(sorted.length/2)];}
  function solveMotion(rows,weights){
    const matrix=Array.from({length:3},()=>[0,0,0,0,0]);
    for(let i=0;i<rows.length;i++){
      const v=rows[i],q=[1,v.x,v.y],weight=weights[i];
      for(let r=0;r<3;r++){
        for(let c=0;c<3;c++)matrix[r][c]+=weight*q[r]*q[c];
        matrix[r][3]+=weight*q[r]*v.dx;matrix[r][4]+=weight*q[r]*v.dy;
      }
    }
    for(let col=0;col<3;col++){
      let pivot=col;
      for(let row=col+1;row<3;row++)if(Math.abs(matrix[row][col])>Math.abs(matrix[pivot][col]))pivot=row;
      if(Math.abs(matrix[pivot][col])<1e-7)return null;
      [matrix[col],matrix[pivot]]=[matrix[pivot],matrix[col]];
      const scale=matrix[col][col];for(let k=col;k<5;k++)matrix[col][k]/=scale;
      for(let row=0;row<3;row++)if(row!==col){
        const amount=matrix[row][col];for(let k=col;k<5;k++)matrix[row][k]-=amount*matrix[col][k];
      }
    }
    return {dx:matrix.map(row=>row[3]),dy:matrix.map(row=>row[4])};
  }
  function affineAt(model,x,y){return {dx:model.dx[0]+model.dx[1]*x+model.dx[2]*y,
    dy:model.dy[0]+model.dy[1]*x+model.dy[2]*y};}
  function fitSubjectMotion(vectors){
    let model={dx:[median(vectors.map(v=>v.dx)),0,0],dy:[median(vectors.map(v=>v.dy)),0,0]};
    const translation=model;
    for(let pass=0;pass<4;pass++){
      const residuals=vectors.map(v=>{const p=affineAt(model,v.x,v.y);return Math.hypot(v.dx-p.dx,v.dy-p.dy);});
      const limit=Math.max(1,median(residuals)*1.5),weights=residuals.map(value=>Math.min(1,limit/Math.max(.001,value)));
      const fitted=solveMotion(vectors,weights);if(!fitted)return null;model=fitted;
    }
    let rigidError=0,affineError=0;
    for(const v of vectors){
      const p=affineAt(model,v.x,v.y);
      rigidError+=(v.dx-translation.dx[0])**2+(v.dy-translation.dy[0])**2;
      affineError+=(v.dx-p.dx)**2+(v.dy-p.dy)**2;
    }
    // Extra affine parameters must explain more than the measured point noise.
    // Otherwise small subpixel errors become an invented scale/rotation that
    // accumulates at unsupported, low-texture extremities.
    const residualVariance=Math.max(.0025,affineError/Math.max(1,vectors.length*2-6));
    if((rigidError-affineError)/4<4*residualVariance)model={...translation,rigid:true};
    const a=1+model.dx[1],b=model.dx[2],c=model.dy[1],d=1+model.dy[2],det=a*d-b*c;
    // A single step cannot legitimately mirror a person or collapse its area.
    if(det<.5||det>1.8||Math.hypot(a,c)<.6||Math.hypot(b,d)<.6)return null;
    return model;
  }
  function subjectDisplacement(model,vectors,x,y,reach){
    const base=affineAt(model,x,y),nearest=[],limit=reach*reach;
    for(const v of vectors){
      const distance=(v.x-x)**2+(v.y-y)**2;
      if(distance>limit||nearest.length===6&&distance>=nearest[5].distance)continue;
      let at=nearest.length;while(at>0&&nearest[at-1].distance>distance)at--;
      nearest.splice(at,0,{v,distance});if(nearest.length>6)nearest.pop();
    }
    let weight=0,dx=0,dy=0;
    for(const {v,distance} of nearest){
      const amount=1/Math.max(4,distance);
      weight+=amount;dx+=v.rx*amount;dy+=v.ry*amount;
    }
    return weight?{dx:base.dx+dx/weight,dy:base.dy+dy/weight}:base;
  }
  function samplePixel(pixels,width,height,x,y){
    if(x<0||y<0||x>width-1||y>height-1)return null;
    const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy,
      nx=Math.min(width-1,ix+1),ny=Math.min(height-1,iy+1);
    return pixels[iy*width+ix]*(1-fx)*(1-fy)+pixels[iy*width+nx]*fx*(1-fy)+
      pixels[ny*width+ix]*(1-fx)*fy+pixels[ny*width+nx]*fx*fy;
  }
  function matchingImage(pixels,width,height){
    // Camera resampling changes the phase of high-frequency texture. A small
    // symmetric low-pass is used only for image correspondence, never analysis.
    const horizontal=new Float32Array(pixels.length),result=new Float32Array(pixels.length);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const row=y*width;
      horizontal[row+x]=(pixels[row+Math.max(0,x-1)]+2*pixels[row+x]+pixels[row+Math.min(width-1,x+1)])/4;
    }
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)result[y*width+x]=
      (horizontal[Math.max(0,y-1)*width+x]+2*horizontal[y*width+x]+horizontal[Math.min(height-1,y+1)*width+x])/4;
    return result;
  }
  function searchSubjectTranslation(points,image,width,height,center,range,stride){
    const score=(dx,dy)=>{
      let sum=0,count=0;
      for(const point of points){
        const value=samplePixel(image,width,height,point.x+dx,point.y+dy);
        if(value!==null){sum+=Math.abs(point.value-value);count++;}
      }
      return count>=points.length*.85?sum/count:Infinity;
    };
    let best={error:Infinity};const scores=[];
    for(let y=-range;y<=range;y+=stride)for(let x=-range;x<=range;x+=stride){
      const dx=center.dx+x,dy=center.dy+y,error=score(dx,dy),candidate={dx,dy,error};
      scores.push(candidate);if(error<best.error)best=candidate;
    }
    if(!finite(best.error))return null;
    for(let step=stride/2;step>=.125;step/=2){
      const current=best;
      for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++){
        const dx=current.dx+x*step,dy=current.dy+y*step,error=score(dx,dy);
        if(error<best.error)best={dx,dy,error};
      }
    }
    const second=Math.min(...scores.filter(p=>Math.hypot(p.dx-best.dx,p.dy-best.dy)>2).map(p=>p.error));
    return {...best,separation:second-best.error};
  }
  function subjectTranslation(source,motion){
    const {width,height}=motion;
    if(!motion.matchingPrevious){
      motion.matchingPrevious=matchingImage(motion.previous,width,height);
      motion.matchingCurrent=matchingImage(motion.current,width,height);
    }
    const previous=motion.matchingPrevious,current=motion.matchingCurrent,points=[];
    let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for(const normalized of source){
      const polygon=normalized.map(([x,y])=>[x*(width-1),y*(height-1)]),
        left=Math.min(...polygon.map(p=>p[0])),right=Math.max(...polygon.map(p=>p[0])),
        top=Math.min(...polygon.map(p=>p[1])),bottom=Math.max(...polygon.map(p=>p[1])),
        spacing=Math.max(1,Math.sqrt((right-left)*(bottom-top)/200));
      minX=Math.min(minX,left);maxX=Math.max(maxX,right);minY=Math.min(minY,top);maxY=Math.max(maxY,bottom);
      for(let y=top+.5;y<bottom;y+=spacing)for(let x=left+.5;x<right;x+=spacing){
        if(insidePolygon(x,y,polygon))points.push({x,y,value:samplePixel(previous,width,height,x,y)});
      }
    }
    const extent=Math.min(maxX-minX,maxY-minY);
    // Below four pixels, smoothing mixes both silhouette edges throughout
    // the object: no independent interior remains to verify a new location.
    if(points.length<12||extent<4)return null;
    const mean=points.reduce((sum,p)=>sum+p.value,0)/points.length,
      variance=points.reduce((sum,p)=>sum+(p.value-mean)**2,0)/points.length;
    if(variance<64)return null;
    // This is a separate two-parameter estimator, not an underconstrained
    // affine fit. Only pixels inside this tracking person's mask vote for it.
    const forward=searchSubjectTranslation(points,current,width,height,{dx:0,dy:0},
      Math.min(16,Math.max(4,Math.ceil(extent))),extent<12?1:2);
    if(!forward||forward.error>18||forward.separation<Math.max(2,forward.error*.15))return null;
    const target=points.map(p=>({x:p.x+forward.dx,y:p.y+forward.dy,
      value:samplePixel(current,width,height,p.x+forward.dx,p.y+forward.dy)})).filter(p=>p.value!==null);
    const reverse=searchSubjectTranslation(target,previous,width,height,
      {dx:-forward.dx,dy:-forward.dy},4,1);
    if(!reverse||reverse.separation<1||Math.hypot(forward.dx+reverse.dx,forward.dy+reverse.dy)>.5)return null;
    return {previous,current,model:{dx:[forward.dx,0,0],dy:[forward.dy,0,0],rigid:true}};
  }
  function clipToFrame(polygon,width,height){
    let output=polygon;
    for(const [axis,bound,sign] of [[0,0,1],[0,width-1,-1],[1,0,1],[1,height-1,-1]]){
      const input=output;output=[];
      for(let i=0;i<input.length;i++){
        const a=input[i],b=input[(i+1)%input.length],insideA=(a[axis]-bound)*sign>=0,
          insideB=(b[axis]-bound)*sign>=0;
        if(insideA)output.push(a);
        if(insideA!==insideB){const t=(bound-a[axis])/(b[axis]-a[axis]);
          output.push([a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t]);}
      }
      if(output.length<3)return null;
    }
    return output;
  }
  function restoreClippedEdges(source,moved,motion){
    const output=moved.map(point=>point.slice()),{width,height}=motion;
    for(const [axis,bound,inward] of [[0,0,1],[0,width-1,-1],[1,0,1],[1,height-1,-1]]){
      const onBorder=source.map(point=>Math.abs(point[axis]-bound)<=1);
      for(let i=0;i<source.length;i++){
        if(!onBorder[i]||onBorder[(i+source.length-1)%source.length])continue;
        let j=i;while(onBorder[(j+1)%source.length]&&(j+1)%source.length!==i)j=(j+1)%source.length;
        if(i===j)continue;
        const a=source[i],b=source[j];
        const gapA=(moved[i][axis]-bound)*inward,gapB=(moved[j][axis]-bound)*inward;
        if(Math.max(gapA,gapB)<.5)continue;
        if(Math.abs(a[1-axis]-b[1-axis])<6)return null;
        const projected=[];
        for(const [at,other] of [[i,(i+source.length-1)%source.length],[j,(j+1)%source.length]]){
          const p=moved[at],q=moved[other],denominator=p[axis]-q[axis];
          if(Math.abs(denominator)<1e-6)return null;
          const ratio=(bound-q[axis])/denominator,point=[q[0]+(p[0]-q[0])*ratio,q[1]+(p[1]-q[1])*ratio];
          if(ratio<0||Math.hypot(point[0]-p[0],point[1]-p[1])>motion.step*3)return null;
          projected.push(point);
        }
        // This edge was the camera crop, not a physical body edge. Newly
        // visible border pixels need their own same-person appearance support;
        // copying an old crop edge onto the chest is not valid optical flow.
        let count=0,consistent=0;const regions=[[0,0],[0,0],[0,0]];
        for(let sample=1;sample<=11;sample++){
          const t=sample/12,prior=[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t],
            target=[projected[0][0]+(projected[1][0]-projected[0][0])*t,
              projected[0][1]+(projected[1][1]-projected[0][1])*t],values=[];
          for(const depth of [1,3,5])for(const lateral of [-1,0,1]){
            const p=prior.slice();p[axis]=bound+depth*inward;p[1-axis]+=lateral;
            if(!insidePolygon(p[0],p[1],source))continue;
            const value=samplePixel(motion.previous,width,height,p[0],p[1]);if(value!==null)values.push(value);
          }
          if(values.length<3)continue;
          values.sort((x,y)=>x-y);
          const low=values[Math.floor(values.length*.2)]-18,high=values[Math.floor(values.length*.8)]+18;
          for(const depth of [.5,Math.min(3,Math.max(.5,(gapA*(1-t)+gapB*t)/2))]){
            const p=target.slice();p[axis]=bound+depth*inward;
            const value=samplePixel(motion.current,width,height,p[0],p[1]);if(value===null)continue;
            const region=regions[Math.min(2,Math.floor(t*3))],matches=value>=low&&value<=high;
            count++;region[0]++;if(matches){consistent++;region[1]++;}
          }
        }
        if(count<12||consistent/count<.8||regions.some(([total,good])=>total>=4&&good/total<.6))return null;
        for(let at=i;;at=(at+1)%source.length){
          const t=(source[at][1-axis]-a[1-axis])/(b[1-axis]-a[1-axis]);
          output[at]=[projected[0][0]+(projected[1][0]-projected[0][0])*t,
            projected[0][1]+(projected[1][1]-projected[0][1])*t];
          if(at===j)break;
        }
      }
    }
    return output;
  }
  function polygonArea(polygon){let area=0;for(let i=0;i<polygon.length;i++){
    const a=polygon[i],b=polygon[(i+1)%polygon.length];area+=a[0]*b[1]-b[0]*a[1];
  }return area/2;}
  function segmentsCross(a,b,c,d){
    const side=(p,q,r)=>(q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]);
    return side(a,b,c)*side(a,b,d)<-1e-7&&side(c,d,a)*side(c,d,b)<-1e-7;
  }
  function validWarpTopology(source,target){
    if(Math.abs(polygonArea(target))<1||polygonArea(source)*polygonArea(target)<=0)return false;
    for(let i=0;i<target.length;i++)for(let j=i+2;j<target.length;j++){
      if(i===0&&j===target.length-1)continue;
      if(segmentsCross(target[i],target[(i+1)%target.length],target[j],target[(j+1)%target.length]))return false;
    }
    return true;
  }
  function warpPolygons(source,motion){
    let result=warpSupportedPolygons(source,motion);
    if(result||!motion||motion.last_warp_rejection==="invalid_input")return result;
    const originalReason=motion.last_warp_rejection,translation=subjectTranslation(source,motion);
    if(translation){
      result=warpSupportedPolygons(source,motion,translation);
      if(result){motion.last_warp_rejection=null;return result;}
    }
    motion.last_warp_rejection=originalReason;return null;
  }
  function warpSupportedPolygons(source,motion,translation=null){
    if(motion)motion.last_warp_rejection=null;
    const reject=reason=>{if(motion)motion.last_warp_rejection=reason;return null;};
    if(!polygons(source)||!motion||!motion.previous||!motion.current)return reject("invalid_input");
    const result=[],{width,height,radius,step}=motion,
      components=source.map(polygon=>polygon.map(point=>[point[0]*(width-1),point[1]*(height-1)]));
    const internal=(x,y,polygon)=>insidePolygon(x,y,polygon)&&[[-radius,-radius],[radius,-radius],
      [-radius,radius],[radius,radius]].every(([dx,dy])=>insidePolygon(x+dx,y+dy,polygon));
    const candidates=motion.features.filter(v=>components.some(polygon=>internal(v.x,v.y,polygon))),
      vectors=candidates.filter(v=>v.vector).map(v=>v.vector);
    // All components belong to one tracking subject. A small hand may use that
    // subject's torso support, but must still pass its own image validation.
    if(!translation&&(vectors.length<4||vectors.length<Math.max(4,candidates.length*.45)))return reject("internal_support");
    const spanX=Math.max(...vectors.map(v=>v.x))-Math.min(...vectors.map(v=>v.x)),
      spanY=Math.max(...vectors.map(v=>v.y))-Math.min(...vectors.map(v=>v.y));
    if(!translation&&(spanX<step||spanY<step))return reject("support_span");
    const model=translation?translation.model:fitSubjectMotion(vectors);if(!model)return reject("affine_geometry");
    const supports=translation?[]:vectors.map(v=>{
      const p=affineAt(model,v.x,v.y),rx=v.dx-p.dx,ry=v.dy-p.dy,
        noise=Math.max(.2,2*v.error/Math.sqrt(Math.max(25,v.strength)));
      return {...v,rx:model.rigid&&Math.hypot(rx,ry)<noise?0:rx,
        ry:model.rigid&&Math.hypot(rx,ry)<noise?0:ry};
    });
    const appearance=translation?{...motion,previous:translation.previous,current:translation.current}:motion;
    for(const polygon of components){
      const
        xs=polygon.map(p=>p[0]),ys=polygon.map(p=>p[1]),left=Math.min(...xs),right=Math.max(...xs),
        top=Math.min(...ys),bottom=Math.max(...ys);
      const reach=Math.max(step*2,Math.min(right-left,bottom-top)*.5),
        move=(x,y)=>subjectDisplacement(model,supports,x,y,reach);
      const probes=[],spacing=Math.max(1,Math.min(4,(right-left)/6,(bottom-top)/6));
      for(let y=top+spacing/2;y<bottom;y+=spacing)for(let x=left+spacing/2;x<right;x+=spacing){
        if(!insidePolygon(x,y,polygon))continue;
        const delta=move(x,y),target=samplePixel(appearance.current,width,height,x+delta.dx,y+delta.dy);
        if(target===null)continue;
        const prior=samplePixel(appearance.previous,width,height,x,y);
        probes.push({x,y,prior,target,error:Math.abs(prior-target)});
      }
      if(probes.length<6)return reject("component_visibility");
      if(translation){
        if(probes.reduce((sum,p)=>sum+p.error,0)/probes.length>18)return reject("component_appearance");
        const meanA=probes.reduce((sum,p)=>sum+p.prior,0)/probes.length,
          meanB=probes.reduce((sum,p)=>sum+p.target,0)/probes.length;
        let aa=0,bb=0,ab=0;
        for(const p of probes){const a=p.prior-meanA,b=p.target-meanB;aa+=a*a;bb+=b*b;ab+=a*b;}
        // Smoothing must not let unrelated regions pass simply because their
        // average grey is similar. Each component retains its own appearance.
        if(aa/probes.length>=25&&bb/probes.length>=25){
          if(ab/Math.sqrt(aa*bb)<.5)return reject("component_appearance");
        }else if(probes.filter(p=>p.error<=12).length/probes.length<.8)return reject("component_appearance");
      }
      // Photometric support is checked throughout the person's interior,
      // including flat areas without corners. A surviving face cannot keep
      // drawing the old torso after the torso has been occluded.
      if(probes.filter(p=>p.error<=32).length/probes.length<.7)return reject("component_appearance");
      for(let row=0;row<2;row++)for(let col=0;col<2;col++){
        const region=probes.filter(p=>(p.x<(left+right)/2?0:1)===col&&(p.y<(top+bottom)/2?0:1)===row);
        if(region.length>=6&&region.filter(p=>p.error<=40).length/region.length<.55)return reject("region_occluded");
      }
      // A thin extremity can occupy too little area to fail a whole-person
      // average. Validate each sufficiently long boundary segment just inside
      // the mask, without borrowing the independently moving background.
      const boundary=[],contrasts=[];let nextSample=1.5;
      for(let i=0;i<polygon.length;i++){
        const a=polygon[i],b=polygon[(i+1)%polygon.length],length=Math.hypot(b[0]-a[0],b[1]-a[1]);
        if(length<1e-8)continue;
        for(let along=nextSample;along<length;along+=3){
          const t=along/length,x=a[0]+(b[0]-a[0])*t,y=a[1]+(b[1]-a[1])*t,
            nx=-(b[1]-a[1])/length*1.5,ny=(b[0]-a[0])/length*1.5,
            sign=insidePolygon(x+nx,y+ny,polygon)?1:-1,px=x+nx*sign,py=y+ny*sign;
          if(!insidePolygon(px,py,polygon)){boundary.push(null);continue;}
          const delta=move(px,py),target=samplePixel(appearance.current,width,height,px+delta.dx,py+delta.dy),
            prior=samplePixel(appearance.previous,width,height,px,py);
          boundary.push(target===null||prior===null?null:Math.abs(prior-target)<=32);
          if(translation&&prior!==null&&target!==null){
            const ox=x-nx*sign,oy=y-ny*sign,
              outsideA=samplePixel(appearance.previous,width,height,ox,oy),
              outsideB=samplePixel(appearance.current,width,height,ox+delta.dx,oy+delta.dy);
            if(outsideA!==null&&outsideB!==null)contrasts.push([prior-outsideA,target-outsideB]);
          }
        }
        nextSample=((nextSample-length)%3+3)%3;
      }
      for(let i=0;i<boundary.length;i++){
        const window=Array.from({length:Math.min(6,boundary.length)},(_,j)=>boundary[(i+j)%boundary.length]).filter(v=>v!==null);
        if(window.length>=3&&window.filter(Boolean).length/window.length<.6)return reject("boundary_occluded");
      }
      if(translation&&contrasts.length>=6){
        let residual=0,energy=0,cross=0,aa=0,bb=0;
        for(const [a,b] of contrasts){residual+=Math.abs(a-b);energy+=Math.abs(a)+Math.abs(b);cross+=a*b;aa+=a*a;bb+=b*b;}
        // Interior colour alone cannot establish that a silhouette still
        // exists. Its inside/outside transition must also move coherently.
        if(energy/contrasts.length<8||residual/energy>.45||cross/Math.sqrt(aa*bb)<.5)
          return reject("boundary_occluded");
      }
      const moved=polygon.map(([x,y])=>{const delta=move(x,y);return [x+delta.dx,y+delta.dy];}),
        restored=restoreClippedEdges(polygon,moved,motion),
        clipped=restored&&clipToFrame(restored,width,height);
      if(!restored)return reject("crop_continuation");
      if(!clipped||!validWarpTopology(polygon,clipped))return reject("invalid_geometry");
      const normalizedResult=clipped.map(([x,y])=>[x/(width-1),y/(height-1)]);
      if(!polygons([normalizedResult]))return reject("invalid_polygon");
      result.push(normalizedResult);
    }
    return result;
  }

  class ContourHistory{
    constructor(config,motionAdapter=null){
      this.config={max_history_seconds:6,max_propagation_seconds:2,max_frame_gap_seconds:.2,...config};
      this.estimateMotion=motionAdapter?motionAdapter.estimateMotion.bind(motionAdapter):estimateMotion;
      const partial=config?.allow_partial_boundaries===true;
      this.warpPolygons=motionAdapter?(partial&&motionAdapter.warpPresentation||motionAdapter.warpPolygons).bind(motionAdapter):warpPolygons;
      this.warpChain=partial&&motionAdapter?.warpPresentationChain?motionAdapter.warpPresentationChain.bind(motionAdapter):
        motionAdapter?.warpChain?motionAdapter.warpChain.bind(motionAdapter):null;
      this.getPresentation=motionAdapter?.getPresentation?motionAdapter.getPresentation.bind(motionAdapter):
        shape=>({complete:true,polygons:shape,open_segments:[]});
      this.prepareMotion=motionAdapter?.prepareMotion?motionAdapter.prepareMotion.bind(motionAdapter):null;
      this.reverseMotion=motionAdapter?.reverseMotion?motionAdapter.reverseMotion.bind(motionAdapter):null;
      this.warpApproximateChain=motionAdapter?.warpApproximateChain?motionAdapter.warpApproximateChain.bind(motionAdapter):null;
      this.approximationSeconds=Math.min(.35,Math.max(0,finite(config?.max_approximation_seconds)?config.max_approximation_seconds:.35));
      this.strictSuccesses=new Map();
      this.geometryCache=new Map();
      this.frames=new Map();this.motions=new Map();this.anchors=new Map();this.events=new Map();this.propagated=new Map();
      this.relocations=new Map();
      this.failedRelocations=new Map();
      this.motionIndex=0;
      this.lastFrame=null;this.rejected=0;this.eventSnapshotRevision=-1;this.revision=0;
    }
    appendFrame(frame){
      if(!sequence(frame.sequence)||!finite(frame.timestamp)||frame.timestamp<0||
        !Number.isInteger(frame.width)||!Number.isInteger(frame.height)||frame.width<16||frame.height<16||
        frame.width>320||frame.height>240||frame.pixels.length!==frame.width*frame.height||
        this.lastFrame&&(frame.sequence<=this.lastFrame.sequence||frame.timestamp<=this.lastFrame.timestamp))
        return false;
      const previous=this.lastFrame;
      if(previous&&previous.width===frame.width&&previous.height===frame.height&&
        frame.timestamp-previous.timestamp<=this.config.max_frame_gap_seconds){
        const motion=this.estimateMotion(previous.pixels,frame.pixels,frame.width,frame.height);
        motion.chain_index=this.motionIndex++;
        if(this.prepareMotion)this.prepareMotion(motion);
        this.motions.set(frame.sequence,{previous:previous.sequence,
          motion});
      }
      this.frames.set(frame.sequence,frame);this.lastFrame=frame;
      const cutoff=frame.timestamp-this.config.max_history_seconds;
      for(const [key,value] of this.frames)if(value.timestamp<cutoff){this.frames.delete(key);this.motions.delete(key);}
      for(const [key,value] of this.anchors)if(value.media_timestamp<cutoff)this.anchors.delete(key);
      for(const [key,value] of this.propagated)if(value.timestamp<cutoff)this.propagated.delete(key);
      for(const [key,value] of this.geometryCache)if(!this.frames.has(key)||
        !this.anchors.has(value.anchor.source_sequence)||
        value.future&&!this.anchors.has(value.future.source_sequence))this.geometryCache.delete(key);
      for(const [key,value] of this.relocations)if(value.sourceTimestamp<cutoff)this.relocations.delete(key);
      for(const [key,value] of this.failedRelocations)if(value.anchorTimestamp<cutoff)this.failedRelocations.delete(key);
      for(const [key,values] of this.strictSuccesses){
        const retained=values.filter(value=>value.timestamp>=cutoff);
        if(retained.length)this.strictSuccesses.set(key,retained);else this.strictSuccesses.delete(key);
      }
      return true;
    }
    acceptMetadata(data){
      if(!data||!sameRun(data,this.config)||!Array.isArray(data.anchors)||data.anchors.length>256)return false;
      for(const anchor of data.anchors){
        // Server snapshots can retain more history than this display worker.
        // A normally retired source is not a corrupt or mismatched anchor.
        if(finite(anchor.media_timestamp)&&this.lastFrame&&
          anchor.media_timestamp<this.lastFrame.timestamp-this.config.max_history_seconds)continue;
        const source=this.frames.get(anchor.source_sequence);
        if(!source||!finite(anchor.media_timestamp)||Math.abs(source.timestamp-anchor.media_timestamp)>1e-6||
          !safeId(anchor.epoch)||!Array.isArray(anchor.subjects)||anchor.subjects.length>64){this.rejected++;continue;}
        const subjects=anchor.subjects.filter(subject=>safeId(subject.track_id)&&polygons(subject.polygons)&&
          [undefined,"confirmed","provisional"].includes(subject.tracking_confirmation));
        if(subjects.length!==anchor.subjects.length){this.rejected++;continue;}
        const existing=this.anchors.get(anchor.source_sequence),signature=JSON.stringify(subjects);
        if(existing&&existing.epoch===anchor.epoch&&existing.signature===signature)continue;
        this.revision++;
        if(existing){
          // A revised real observation invalidates prior display continuity.
          this.strictSuccesses.clear();
          for(const [key,value] of this.propagated)if(value.anchor===anchor.source_sequence)this.propagated.delete(key);
          // Corrected tracking geometry gets a fresh matching attempt immediately.
          for(const subject of subjects)this.failedRelocations.delete(JSON.stringify([anchor.source_sequence,subject.track_id]));
        }
        this.anchors.set(anchor.source_sequence,{...anchor,subjects:structuredClone(subjects),signature});
      }
      // A full newer event snapshot is authoritative even for event IDs that
      // were absent from it. A delayed older response must not resurrect them.
      // tracking anchors above remain independently eligible for display geometry.
      if(sequence(data.event_snapshot_revision)&&data.event_snapshot_revision<this.eventSnapshotRevision)return true;
      const presentEvents=new Set();
      if(Array.isArray(data.events))for(const event of data.events.slice(0,256)){
        if(!safeId(event.track_id)||!safeId(event.epoch)||!safeId(event.event_label)||
          event.event_id!==undefined&&!safeId(event.event_id)||
          !sequence(event.revision)||!finite(event.valid_from_timestamp)||event.valid_from_timestamp<0||
          event.valid_until_timestamp!==null&&(!finite(event.valid_until_timestamp)||
            event.valid_until_timestamp<event.valid_from_timestamp))continue;
        const key=JSON.stringify([event.epoch,event.track_id,event.event_label,event.event_id||""]);
        presentEvents.add(key);
        const versions=this.events.get(key)||[];
        if(!versions.length||event.revision>versions[versions.length-1].revision){
          versions.push({...event});this.events.set(key,versions.slice(-32));this.revision++;
          if(this.events.size>512)this.events.delete(this.events.keys().next().value);
        }
      }
      if(Array.isArray(data.events)&&sequence(data.event_snapshot_revision)&&
        data.event_snapshot_revision>this.eventSnapshotRevision&&
        finite(data.event_snapshot_timestamp)&&data.event_snapshot_timestamp>=0){
        this.eventSnapshotRevision=data.event_snapshot_revision;
        for(const [key,versions] of this.events){
          if(presentEvents.has(key))continue;
          const prior=versions[versions.length-1];
          if(prior.revision>=data.event_snapshot_revision||!["suspected","confirmed"].includes(prior.phase)||
            prior.valid_until_timestamp!==null)continue;
          versions.push({...prior,phase:"ended",revision:data.event_snapshot_revision,
            valid_from_timestamp:Math.max(data.event_snapshot_timestamp,prior.valid_from_timestamp)});
          this.events.set(key,versions.slice(-32));this.revision++;
        }
      }
      return true;
    }
    recordStrictSuccess(epoch,trackId,targetSequence,timestamp){
      const identity=JSON.stringify([epoch,trackId]),values=this.strictSuccesses.get(identity)||[];
      if(!values.some(value=>value.sequence===targetSequence))values.push({sequence:targetSequence,timestamp});
      values.sort((a,b)=>a.sequence-b.sequence);this.strictSuccesses.set(identity,values.slice(-64));
    }
    colorGeometry(result,timestamp,cacheHit){
      const currentEvents=[...this.events.values()].map(versions=>versions.filter(event=>
        event.valid_from_timestamp<=timestamp).at(-1)).filter(Boolean);
      return {...result,geometry_cache_hit:cacheHit,subjects:result.subjects.map(subject=>{
        const active=subject.tracking_confirmation==="confirmed"?currentEvents.filter(event=>
          event.epoch===subject.epoch&&event.track_id===subject.track_id&&["suspected","confirmed"].includes(event.phase)&&
          (event.valid_until_timestamp===null||timestamp<event.valid_until_timestamp)):[];
        return {...subject,alert:active.some(event=>event.phase==="confirmed"),
          warning:active.some(event=>event.phase==="suspected"),
          event_labels:[...new Set(active.map(event=>event.event_label))]};
      })};
    }
    select(targetSequence,timestamp){
      const target=this.frames.get(targetSequence);
      if(!target||Math.abs(target.timestamp-timestamp)>1e-6)return {subjects:[],coverage:"missing_source"};
      const anchors=[...this.anchors.values()].filter(a=>a.source_sequence<=targetSequence&&
        a.media_timestamp<=timestamp).sort((a,b)=>b.source_sequence-a.source_sequence);
      if(!anchors.length)return {subjects:[],coverage:"no_anchor"};
      const anchor=anchors[0];
      if(timestamp-anchor.media_timestamp>this.config.max_propagation_seconds&&this.config.anchor_policy!=="nearest_arrived")
        return {subjects:[],coverage:"anchor_expired"};
      // Display buffering may already have delivered a later real tracking result.
      // Only the nearest future observation is eligible: a missing person or
      // epoch change there is a barrier, never skipped to find a later match.
      const future=this.config.anchor_policy==="nearest_arrived"&&this.reverseMotion?
        [...this.anchors.values()].filter(a=>a.source_sequence>targetSequence&&a.media_timestamp>timestamp)
          .sort((a,b)=>a.source_sequence-b.source_sequence)[0]:null;
      const memo=this.geometryCache.get(targetSequence);
      // Events and distant new anchors do not change these pixels. Both real
      // neighbours participate in the key: a corrected same-source mask or a
      // newly arrived missing-person/epoch barrier must invalidate the result.
      if(memo&&memo.target===target&&memo.anchor===anchor&&memo.future===future&&
        memo.policy===this.config.anchor_policy&&memo.maxAge===this.config.max_propagation_seconds){
        for(const subject of memo.result.subjects)
          this.recordStrictSuccess(subject.epoch,subject.track_id,targetSequence,target.timestamp);
        return this.colorGeometry(memo.result,timestamp,true);
      }
      const cached=[...this.propagated.entries()].filter(([key,value])=>key<=targetSequence&&
        value.anchor===anchor.source_sequence).sort((a,b)=>b[0]-a[0]);
      const subjects=[],propagatedSubjects=[],rejections={},localFailures=new Set();let motionGap=false,anchorExpired=false;
      const reject=motion=>{
        const reason=motion.last_warp_rejection;
        if(motionRejectionReasons.includes(reason))rejections[reason]=(rejections[reason]||0)+1;
      };
      const project=(subject,geometryAnchor)=>{
        if(Math.abs(geometryAnchor.media_timestamp-timestamp)>this.config.max_propagation_seconds){
          anchorExpired=true;return null;
        }
        const backwards=geometryAnchor.source_sequence>targetSequence,
          geometrySubject=geometryAnchor.subjects.find(item=>item.track_id===subject.track_id);
        const prior=backwards?null:cached.find(([,value])=>value.subjects.some(item=>
          item.track_id===subject.track_id&&item.anchor_sequence===anchor.source_sequence));
        const fromSequence=prior?prior[0]:anchor.source_sequence;
        const previous=prior?.[1].subjects.find(item=>item.track_id===subject.track_id);
        const chain=[];let at=backwards?geometryAnchor.source_sequence:targetSequence,gap=false;
        const stopSequence=backwards?targetSequence:fromSequence;
        while(at!==stopSequence){
          const item=this.motions.get(at);
          if(!item||item.previous<stopSequence||chain.length>=180){gap=true;break;}
          chain.push(backwards?this.reverseMotion(item.motion):item.motion);at=item.previous;
        }
        if(gap){motionGap=true;return null;}
        let shape=previous?previous.shape:geometrySubject.polygons,relocalized=false;
        const ordered=backwards?chain:chain.slice().reverse();
        if(chain.length&&this.warpChain){
          shape=this.warpChain(shape,ordered);if(!shape){
            reject(ordered.at(-1));
            if(["crop_continuation","boundary_occluded"].includes(ordered.at(-1).last_warp_rejection))
              localFailures.add(JSON.stringify([geometryAnchor.source_sequence,subject.track_id]));
          }
        }else for(const motion of ordered){
          shape=this.warpPolygons(shape,motion);if(!shape){reject(motion);break;}
        }
        if(!shape||this.getPresentation(shape)?.complete===false){
          // A locally ambiguous intermediate frame need not erase a person.
          // Re-match the actual tracking source pixels to the actual target pixels,
          // using the same foreground/photometric checks as ordinary motion.
          // This does not bridge missing source intervals or expired anchors.
          const original=this.frames.get(geometryAnchor.source_sequence);
          const originalSubject=geometrySubject;
          const retryKey=JSON.stringify([geometryAnchor.source_sequence,subject.track_id]);
          const failed=this.failedRelocations.get(retryKey);
          if(original&&originalSubject&&original.width===target.width&&original.height===target.height&&
            (!failed||timestamp<failed.timestamp-1e-6||timestamp-failed.timestamp>=.2-1e-6)){
            const key=geometryAnchor.source_sequence+":"+targetSequence;
            if(!this.relocations.has(key))this.relocations.set(key,{
              sourceTimestamp:original.timestamp,
              motion:this.estimateMotion(original.pixels,target.pixels,target.width,target.height),
            });
            const recovered=this.warpPolygons(originalSubject.polygons,this.relocations.get(key).motion);
            if(!recovered)reject(this.relocations.get(key).motion);
            if(recovered&&(!shape||this.getPresentation(recovered)?.complete===true)){
              shape=recovered;relocalized=true;this.failedRelocations.delete(retryKey);
            }else this.failedRelocations.set(retryKey,{timestamp,anchorTimestamp:geometryAnchor.media_timestamp});
          }
        }
        if(!shape)return null;
        const presentation=this.getPresentation(shape);
        if(!presentation||!presentation.polygons?.length&&!presentation.open_segments?.length)return null;
        if(!presentation.complete)reject({last_warp_rejection:presentation.reason});
        return {shape,presentation,relocalized,backwards,geometryAnchor};
      };
      const approximate=(subject,candidates)=>{
        if(!this.warpApproximateChain||!this.approximationSeconds||!future||future.epoch!==anchor.epoch||
          !future.subjects.some(item=>item.track_id===subject.track_id)||
          timestamp-anchor.media_timestamp>this.config.max_propagation_seconds||
          future.media_timestamp-timestamp>this.config.max_propagation_seconds)return null;
        const identity=JSON.stringify([anchor.epoch,subject.track_id]),strict=(this.strictSuccesses.get(identity)||[])
          .filter(value=>value.sequence<targetSequence&&value.timestamp<=timestamp)
          .sort((a,b)=>b.sequence-a.sequence)[0];
        if(!strict||timestamp-strict.timestamp>this.approximationSeconds+1e-6)return null;
        if([...this.anchors.values()].some(observation=>observation.source_sequence>strict.sequence&&
          observation.source_sequence<=targetSequence&&(observation.epoch!==anchor.epoch||
            !observation.subjects.some(item=>item.track_id===subject.track_id))))return null;
        // Both real neighbours and the complete source interval are required.
        // A missing future person/epoch is an exit barrier, never skipped.
        const forward=[];let at=future.source_sequence;
        while(at!==anchor.source_sequence){
          const item=this.motions.get(at);
          if(!item||item.previous<anchor.source_sequence||forward.length>=180)return null;
          forward.push({sequence:at,...item});at=item.previous;
        }
        for(const geometryAnchor of candidates){
          if(!localFailures.has(JSON.stringify([geometryAnchor.source_sequence,subject.track_id])))continue;
          const backwards=geometryAnchor===future,chain=backwards?
            forward.filter(item=>item.sequence>targetSequence).map(item=>this.reverseMotion(item.motion)):
            forward.filter(item=>item.sequence<=targetSequence).reverse().map(item=>item.motion);
          if(!chain.length)continue;
          const original=geometryAnchor.subjects.find(item=>item.track_id===subject.track_id),
            shape=this.warpApproximateChain(original.polygons,chain),presentation=shape&&this.getPresentation(shape);
          if(!presentation?.approximate)continue;
          return {shape,presentation,relocalized:false,backwards,geometryAnchor,strict,
            approximateAge:timestamp-strict.timestamp};
        }
        return null;
      };
      for(const subject of anchor.subjects){
        const candidates=[anchor];
        if(future?.epoch===anchor.epoch&&future.subjects.some(item=>item.track_id===subject.track_id))candidates.push(future);
        candidates.sort((a,b)=>Math.abs(a.media_timestamp-timestamp)-Math.abs(b.media_timestamp-timestamp));
        let projected=null;
        for(const candidate of candidates){
          const next=project(subject,candidate);
          if(!next)continue;
          if(!projected||next.presentation.complete)projected=next;
          if(next.presentation.complete)break;
        }
        if(!projected?.presentation.complete){
          const filled=approximate(subject,candidates);if(filled)projected=filled;
        }
        if(!projected)continue;
        const {shape,presentation,relocalized,backwards,geometryAnchor}=projected;
        const confirmation=subject.tracking_confirmation||"confirmed";
        subjects.push({track_id:subject.track_id,epoch:anchor.epoch,polygons:presentation.polygons,
          open_segments:presentation.open_segments,complete:presentation.complete,
          tracking_confirmation:confirmation,
          target_sequence:targetSequence,anchor_sequence:geometryAnchor.source_sequence,
          anchor_age_seconds:Math.abs(timestamp-geometryAnchor.media_timestamp),
          method:targetSequence===anchor.source_sequence?"tracking_observed":backwards?"image_motion_backward":"image_motion",relocalized,
          ...(presentation.approximate?{approximate:true,approximation_reason:presentation.reason,
            approximation_source:"adjacent_tracking_image_motion",last_strict_sequence:projected.strict.sequence,
            approximation_age_seconds:projected.approximateAge}:{}),
          alert:false});
        if(!presentation.approximate)propagatedSubjects.push({track_id:subject.track_id,shape,anchor_sequence:geometryAnchor.source_sequence,
          tracking_confirmation:confirmation});
        if(presentation.complete)this.recordStrictSuccess(anchor.epoch,subject.track_id,targetSequence,timestamp);
      }
      // Cache each successful person independently. Missing estimates never
      // delete a tracking subject; that person retries from its own last success,
      // while an occlusion does not force everyone else to repeat a long chain.
      this.propagated.set(targetSequence,{
        timestamp,anchor:anchor.source_sequence,subjects:propagatedSubjects});
      const geometryAnchors=[...new Set(subjects.map(subject=>subject.anchor_sequence))];
      const allSubjectsPresent=anchor.subjects.length>0&&subjects.length===anchor.subjects.length&&
        subjects.every(subject=>subject.polygons.length>0&&(subject.complete||subject.approximate));
      const result={subjects,all_subjects_present:allSubjectsPresent,
        coverage:subjects.length===anchor.subjects.length&&subjects.every(subject=>subject.complete)?
          "anchored":allSubjectsPresent?"interpolated":motionGap?"motion_gap":anchorExpired?"anchor_expired":"motion_unreliable",
        motion_rejections:rejections,
        anchor_sequence:geometryAnchors.length===1?geometryAnchors[0]:anchor.source_sequence,
        geometry_anchor_sequences:geometryAnchors,
        anchor_age_seconds:subjects.length?Math.max(...subjects.map(subject=>subject.anchor_age_seconds)):
          timestamp-anchor.media_timestamp,
        // Even valid propagation cannot establish coverage for previously unseen people.
        exact:targetSequence===anchor.source_sequence};
      // Incomplete, approximate and negative results remain retryable. In
      // particular a throttled relocalization must not become a cached failure.
      if(subjects.length&&result.coverage==="anchored"&&
        subjects.every(subject=>subject.complete&&!subject.approximate))
        this.geometryCache.set(targetSequence,{target,anchor,future,policy:this.config.anchor_policy,
          maxAge:this.config.max_propagation_seconds,result});
      else this.geometryCache.delete(targetSequence);
      return this.colorGeometry(result,timestamp,false);
    }
  }

  function packCameraFrame(metadata,jpeg){
    const header=new TextEncoder().encode(JSON.stringify(metadata));
    const bytes=jpeg instanceof Uint8Array?jpeg:new Uint8Array(jpeg);
    const packet=new Uint8Array(8+header.length+bytes.length);
    packet.set([71,86,67,50]);new DataView(packet.buffer).setUint32(4,header.length);
    packet.set(header,8);packet.set(bytes,8+header.length);return packet.buffer;
  }

  let warmMotionWorker=null;
  const eventNames={fight:"打架",smoking:"吸烟",fall:"跌倒",weapon:"持械",ppe_violation:"防护装备缺失",ground_litter:"地面垃圾",stranger:"陌生人",identity_pending:"身份待确认"};
  function drawSubjectLabel(context,subject,width,height,displayWidth=width){
    if(subject.track_id===undefined||subject.track_id===null)return;
    const points=[...(subject.polygons||[]),...(subject.open_segments||[])].flat()
      .filter(point=>Number.isFinite(point[0])&&Number.isFinite(point[1]));
    if(!points.length)return;
    // Size text in visible pixels while keeping the original-resolution canvas.
    const fontSize=Math.max(14,Math.round(14*width/(displayWidth||width))),padding=Math.ceil(fontSize*.35),
      lineHeight=Math.ceil(fontSize*1.3),margin=Math.ceil(fontSize*.25),available=width-2*margin-2*padding;
    if(available<=0)return;
    context.save();context.font=`600 ${fontSize}px sans-serif`;context.textBaseline="top";
    const lines=[displayName(subject.display_name)||`ID ${subject.track_id}`];
    if(subject.tracking_confirmation==="confirmed"&&(subject.alert||subject.warning)&&subject.event_labels?.length)
      lines.push([...new Set(subject.event_labels.map(label=>eventNames[label]||label))].join(" · "));
    const fitted=lines.map(line=>{
      if(context.measureText(line).width<=available)return line;
      const chars=Array.from(line);
      while(chars.length&&context.measureText(chars.join("")+"…").width>available)chars.pop();
      return chars.join("")+"…";
    });
    const boxWidth=Math.min(available,Math.max(...fitted.map(line=>context.measureText(line).width)))+2*padding,
      boxHeight=fitted.length*lineHeight+2*padding,
      x=Math.max(margin,Math.min(width-margin-boxWidth,Math.min(...points.map(point=>point[0]))*width)),
      y=Math.max(margin,Math.min(height-margin-boxHeight,Math.min(...points.map(point=>point[1]))*height-boxHeight-margin));
    context.fillStyle="rgba(10, 15, 22, .82)";context.fillRect(x,y,boxWidth,boxHeight);
    context.fillStyle=subject.tracking_confirmation==="provisional"?"#9ca3af":subject.alert?"#ef4444":subject.warning?"#facc15":"#22c55e";
    context.fillRect(x,y,Math.max(2,Math.round(fontSize/7)),boxHeight);
    fitted.forEach((line,index)=>{context.fillStyle=index?"#fca5a5":"#ffffff";
      context.fillText(line,x+padding,y+padding+index*lineHeight,available);});
    context.restore();
  }
  let displayInstanceSequence=0;
  const displayDocumentId=Date.now().toString(36)+"-"+Math.random().toString(36).slice(2);
  class BrowserController{
    static supported(video){return typeof Worker==="function"&&typeof OffscreenCanvas==="function"&&
      typeof MessageChannel==="function"&&
      typeof OffscreenCanvas.prototype.transferToImageBitmap==="function"&&
      (video===null||typeof video?.requestVideoFrameCallback==="function");}
    static warmupMotion({onFailure=()=>{}}={}){
      if(warmMotionWorker)return warmMotionWorker.promise;
      let worker;
      try{worker=new Worker("/assets/camera_display.js");}
      catch(error){onFailure("轮廓跟随预热失败，开始检测时将重试。");return Promise.resolve(false);}
      const warm={worker,resolve:null,promise:null,claimed:false,disposed:false};
      warm.promise=new Promise(resolve=>{warm.resolve=resolve;});warmMotionWorker=warm;
      const fail=()=>{
        if(warm.claimed||warm.disposed)return;warm.disposed=true;
        if(warmMotionWorker===warm)warmMotionWorker=null;
        worker.terminate();warm.resolve(false);onFailure("轮廓跟随预热失败，开始检测时将重试。");
      };
      worker.onmessage=event=>{
        if(warm.disposed)return;
        if(event.data?.type==="motion_warmup_ready")warm.resolve(true);
        else if(event.data?.type==="failure")fail();
      };
      worker.onerror=fail;worker.onmessageerror=fail;
      try{worker.postMessage({type:"warmup_motion"});}catch(error){fail();}
      return warm.promise;
    }
    static discardWarmup(){
      if(!warmMotionWorker)return;
      const warm=warmMotionWorker;warmMotionWorker=null;
      warm.disposed=true;warm.resolve(false);warm.worker.terminate();
    }
    constructor({video,canvas,config,settings,socket,onStatus=()=>{},onFailure=()=>{},onMotionTrace=null,motionTraceOptions={}}){
      this.displayInstanceId=displayDocumentId+"-"+(++displayInstanceSequence);
      this.video=video;this.canvas=canvas;this.config=config;this.settings={...settings};this.socket=socket;
      if(finite(config.analysis_max_image_side)&&config.analysis_max_image_side>0)
        this.settings.max_image_side=Math.min(this.settings.max_image_side,Math.max(1,Math.floor(config.analysis_max_image_side)));
      if(finite(config.analysis_jpeg_quality)&&config.analysis_jpeg_quality>0)
        this.settings.jpeg_quality=Math.min(this.settings.jpeg_quality,clamp(config.analysis_jpeg_quality,1,100));
      this.onStatus=onStatus;this.onFailure=onFailure;this.closed=false;this.capturing=true;
      this.onMotionTrace=typeof onMotionTrace==="function"?onMotionTrace:null;
      this.displayFps=Math.max(15,Number(config.target_fps)||15);
      const requestedDelay=Number(config.delay_seconds),displayDelay=clamp(finite(requestedDelay)?requestedDelay:1,0,config.source_mode==="network"?30:8);
      this.frames=new FrameTimeline({delaySeconds:displayDelay,
        maxFrames:Math.ceil((displayDelay+.5)*this.displayFps)+2,dispose:frame=>frame.bitmap.close()});
      // Native-resolution bitmaps have a separate, bounded presentation budget.
      // Never shorten the requested buffer by silently evicting its first frames.
      this.displayBufferBudgetBytes=1.25*1024**3;this.displayBytesPerFrame=0;this.requiredDisplayBufferBytes=0;
      const warm=config.motion_backend==="dense"?warmMotionWorker:null;
      if(warm){warmMotionWorker=null;warm.claimed=true;}
      this.claimedWarmup=warm;this.worker=warm?.worker||new Worker("/assets/camera_display.js");
      this.preprocessWorker=new Worker("/assets/camera_display.js");
      const motionChannel=new MessageChannel();
      this.worker.postMessage({type:"motion_input",config,port:motionChannel.port1},[motionChannel.port1]);
      this.preprocessWorker.postMessage({type:"preprocess_output",port:motionChannel.port2,...(this.onMotionTrace?{
        trace_enabled:true,trace_max_bytes:motionTraceOptions.maxBytes,trace_max_seconds:motionTraceOptions.maxSeconds}:{}
      )},[motionChannel.port2]);
      this.worker.onmessage=event=>{
        const message=event.data;
        if(this.closed)return;
        this.retireMotionWork(message);
        if(message.type==="motion_trace_input"||message.type==="motion_trace_stopped"){
          this.recordMotionTrace(message);return;
        }
        if(Number.isSafeInteger(message.trace_id))this.recordMotionTrace({type:"motion_trace_output",message});
        if(message.type==="motion_warmup_ready"){this.claimedWarmup?.resolve(true);return;}
        if(message.type==="failure"&&message.stage==="motion_warmup"){
          this.claimedWarmup?.resolve(false);return; // Run initialization reports its own failure below.
        }
        if(message.type==="present"){
          this.pendingPresent.delete(message.sequence);
          const preparedRevision=this.requestedRevision.get(message.sequence);
          this.requestedRevision.delete(message.sequence);
          if(preparedRevision!==undefined&&preparedRevision<this.geometryRevision)this.requested.delete(message.sequence);
          this.stats.worker_results_received++;
          if(finite(message.motion_queue_ms))this.noteTiming("motion_present_queue",message.motion_queue_ms);
          if(finite(message.compute_ms))this.noteTiming("motion_present_compute",message.compute_ms);
          if(message.expired){this.stats.worker_present_expired++;return;}
          if(message.geometry_cache_hit===true)this.stats.geometry_cache_hits++;
          else if(message.geometry_cache_hit===false)this.stats.geometry_cache_misses++;
          if(finite(message.queued_at))this.stats.worker_result_latency_max_seconds=Math.max(
            this.stats.worker_result_latency_max_seconds,(performance.now()-message.queued_at)/1000);
          if(message.sequence>Math.max(this.lastPresentedSequence,this.lastDiscardedSequence)){
            this.overlays.set(message.sequence,message.result);
            if(this.overlays.size>120)this.overlays.delete(this.overlays.keys().next().value);
          }else this.stats.worker_results_late++;
        }else if(message.type==="frame_processed"){
          if(message.expired)this.stats.motion_frames_expired++;
          else this.stats.worker_frames_completed++;
          if(!message.expired&&finite(message.compute_ms))this.noteTiming("motion_compute",message.compute_ms);
          if(finite(message.motion_queue_ms))this.noteTiming("motion_frame_queue",message.motion_queue_ms);
          if(!message.accepted&&!message.expired)this.stats.motion_failures++;
          if(finite(message.queued_at))this.stats.worker_frame_latency_max_seconds=Math.max(
            this.stats.worker_frame_latency_max_seconds,(performance.now()-message.queued_at)/1000);
        }else if(message.type==="frame_preprocessed"){
          this.stats.preprocess_frames_completed++;
          if(finite(message.compute_ms))this.noteTiming("motion_preprocess",message.compute_ms);
          if(finite(message.queue_ms))this.noteTiming("motion_preprocess_queue",message.queue_ms);
        }else if(message.type==="metadata_processed"){
          if(message.accepted){
            this.stats.worker_metadata_accepted++;
            this.lastMetadataAppliedAt=performance.now();
            if(finite(message.queued_at))this.noteTiming("metadata_apply",this.lastMetadataAppliedAt-message.queued_at);
            if(finite(message.latest_anchor_timestamp))this.latestAppliedAnchorTimestamp=message.latest_anchor_timestamp;
            // Coalesce corrections while a query is in flight. Its answer may
            // still serve the frame; at most one newer query follows it.
            if(message.geometry_changed===true){this.geometryRevision++;this.requested.clear();}
          }
          this.stats.worker_anchor_rejections+=message.rejected||0;
        }
        else if(message.type==="motion_runtime_ready"){
          this.claimedWarmup?.resolve(true);
          this.stats.motion_runtime_ready_elapsed_seconds=Math.max(0,(performance.now()-this.startedAt)/1000);
          if(finite(message.runtime_load_ms))this.stats.motion_runtime_load_seconds=message.runtime_load_ms/1000;
        }else if(message.type==="motion_ready"){
          if(message.backend==="dense"){
            this.stats.dense_motion_ready=1;
            this.stats.motion_ready_elapsed_seconds=Math.max(0,(performance.now()-this.startedAt)/1000);
            if(finite(message.startup_drain_ms))this.stats.motion_startup_drain_seconds=message.startup_drain_ms/1000;
          }
        }else if(message.type==="failure"){
          if(message.request_type==="present"){
            this.pendingPresent.delete(message.sequence);this.requestedRevision.delete(message.sequence);
          }
          this.stats.motion_failures++;
          if(message.stage==="motion_initialization"){
            this.disableMotionWork();
            this.stats.motion_initialization_failures++;
            this.onFailure("轮廓跟随组件加载失败，视频继续播放；请重新开始检测。");
          }else if(message.stage==="motion_preprocess"){
            this.stats.preprocess_failures++;
            this.onFailure("轮廓画面预处理失败，视频继续播放；请重新开始检测。");
          }
        }
      };
      this.worker.onerror=()=>{this.claimedWarmup?.resolve(false);this.disableMotionWork();
        this.stats.motion_failures++;this.onFailure("轮廓补帧暂不可用，视频继续播放。");};
      this.worker.onmessageerror=this.worker.onerror;
      this.preprocessWorker.onmessage=this.worker.onmessage;
      this.preprocessWorker.onerror=()=>{
        this.disableMotionWork();
        this.stats.preprocess_failures++;this.stats.motion_failures++;
        this.onFailure("轮廓画面预处理组件失败，视频继续播放；请重新开始检测。");
      };
      this.preprocessWorker.onmessageerror=this.preprocessWorker.onerror;
      this.encodeWorker=new Worker("/assets/camera_display.js");
      this.encodeWorker.onmessage=event=>{
        const message=event.data,pending=this.pendingEncode;
        if(!pending||message.sequence!==pending.sequence||
          !["encoded_jpeg","encoding_failure"].includes(message.type))return;
        clearTimeout(pending.timer);this.pendingEncode=null;
        if(message.type==="encoded_jpeg")pending.resolve(message.bytes);
        else if(message.type==="encoding_failure")pending.reject(new Error("analysis encoding failed"));
      };
      this.encodeWorker.onerror=()=>{
        if(this.pendingEncode){clearTimeout(this.pendingEncode.timer);
          this.pendingEncode.reject(new Error("analysis encoding worker failed"));this.pendingEncode=null;}
      };
      this.encodeWorker.onmessageerror=this.encodeWorker.onerror;
      // Admission spans both workers: releasing a frame after grayscale alone
      // would merely move an unbounded backlog into the motion worker's port.
      this.motionFrames=new Map();this.motionPresents=new Set();this.motionFailed=false;
      this.motionMetadataInFlight=false;this.pendingMotionMetadata=null;
      this.overlays=new Map();this.requested=new Set();this.pendingPresent=new Set();this.requestedRevision=new Map();
      this.identities=new DisplayIdentities(this.config);
      this.geometryRevision=0;
      this.presentPrefetchSeconds=Math.min(1.5,Math.max(.35,this.frames.delay-.25));
      // A zero-delay frame still needs an asynchronous motion-worker round trip.
      // Bound this grace period so a failed worker never stalls live playback.
      this.presentGraceSeconds=Math.max(0,.15-this.frames.delay);
      this.uploads=[];this.uploading=false;
      this.pendingAck=null;this.analysisPause=null;
      this.lastPresentedSequence=-1;this.lastDiscardedSequence=-1;this.sourceSequence=0;this.firstMediaTime=null;
      this.nextCaptureAt=0;this.nextAnalysisAt=0;this.latestClock=null;this.lastVideoFrame=-1;
      this.stats={captured_frames:0,analysis_admitted:0,analysis_sent:0,analysis_upload_failures:0,
        network_requests:0,network_failures:0,network_bytes_received:0,network_frames_received:0,network_source_gap_frames:0,network_queue_dropped_frames:0,
        analysis_prepared:0,
        video_callback_frames:0,video_callback_gap_frames:0,
        analysis_capture_pauses:0,motion_failures:0,metadata_failures:0,frames_without_contours:0,
        display_buffer_capacity_failures:0,
        dense_motion_ready:0,motion_initialization_failures:0,
        motion_warmup_reused:warm?1:0,
        partial_contour_frames:0,complete_contour_frames:0,backward_contour_frames:0,
        approximate_contour_frames:0,approximate_subjects:0,continuous_contour_frames:0,
        observed_contour_frames:0,propagated_contour_frames:0,relocalized_contour_frames:0,last_source_to_present_seconds:null,
        metadata_requests:0,metadata_received:0,metadata_timeouts:0,metadata_http_errors:0,
        metadata_decode_errors:0,metadata_run_mismatches:0,metadata_anchors_received:0,
        metadata_ws_received:0,metadata_duplicate_revisions:0,metadata_invalid_revisions:0,
        metadata_latency_max_seconds:0,worker_frames_completed:0,worker_metadata_accepted:0,
        preprocess_frames_completed:0,preprocess_failures:0,
        motion_frames_admitted:0,motion_frames_skipped:0,motion_frames_expired:0,
        motion_metadata_coalesced:0,
        worker_anchor_rejections:0,worker_results_received:0,worker_results_late:0,
        worker_present_expired:0,
        geometry_cache_hits:0,geometry_cache_misses:0,
        worker_frame_latency_max_seconds:0,worker_result_latency_max_seconds:0,
        coverage_not_ready:0,coverage_no_anchor:0,coverage_anchor_expired:0,coverage_motion_gap:0,
        coverage_motion_unreliable:0,coverage_missing_source:0};
      for(const reason of motionRejectionReasons)this.stats["warp_reject_"+reason]=0;
      this.captureCanvas=new OffscreenCanvas(1,1);this.captureContext=this.captureCanvas.getContext("2d");
      // This canvas only downsizes on the graphics path. All pixel readback and
      // greyscale conversion occur in the motion Worker, away from presentation.
      this.grayCanvas=new OffscreenCanvas(160,90);this.grayContext=this.grayCanvas.getContext("2d");
      this.context=canvas.getContext("2d");this.context.clearRect(0,0,canvas.width,canvas.height);this.lastStatusAt=0;
      for(const key of ["frameSequence","mediaTimestamp","contourCoverage","displayStats"])delete canvas.dataset[key];
      this.startedAt=performance.now();this.lastMetricsAt=this.startedAt;this.reportSequence=0;
      this.metricsPresented=0;this.presentationAges=[];this.maxPresentGap=null;this.metricsPending=false;
      this.timingWindows={};this.metricsCallbackFrames=0;this.metricsCallbackGapFrames=0;
      this.visibilityChanges=0;this.hiddenSeconds=0;this.hiddenSince=null;this.longTaskCount=0;this.longTaskTotalMs=0;
      this.onVisibilityChange=()=>{
        const now=performance.now();this.visibilityChanges++;
        if(document.hidden){if(this.hiddenSince===null)this.hiddenSince=now;}
        else if(this.hiddenSince!==null){this.hiddenSeconds+=(now-this.hiddenSince)/1000;this.hiddenSince=null;}
      };
      if(typeof document!=="undefined"){
        if(document.hidden)this.hiddenSince=this.startedAt;
        document.addEventListener?.("visibilitychange",this.onVisibilityChange);
      }
      if(typeof PerformanceObserver==="function"&&PerformanceObserver.supportedEntryTypes?.includes("longtask")){
        this.longTaskObserver=new PerformanceObserver(list=>{
          for(const entry of list.getEntries())if(entry.startTime>=this.startedAt&&finite(entry.duration)){
            this.longTaskCount++;this.longTaskTotalMs+=entry.duration;this.noteTiming("longtask",entry.duration);
          }
        });
        this.longTaskObserver.observe({type:"longtask"});
      }
    }
    recordMotionTrace(record){
      if(!this.onMotionTrace)return;
      try{if(this.onMotionTrace(record)!==false)return;}
      catch(_){this.motionTraceFailure="consumer_failure";}
      this.onMotionTrace=null;
      try{this.preprocessWorker.postMessage({type:"stop_motion_trace",reason:"consumer_stopped"});}catch(_){}
    }
    noteTiming(name,milliseconds){
      if(!finite(milliseconds)||milliseconds<0)return;
      const window=this.timingWindows[name]||(this.timingWindows[name]={count:0,total:0,max:0,samples:[]});
      window.count++;window.total+=milliseconds;window.max=Math.max(window.max,milliseconds);
      // Fixed-size sample rings bound telemetry memory even if reporting is delayed.
      window.samples[(window.count-1)%256]=milliseconds;
    }
    disableMotionWork(){
      // A terminal worker failure has no future ACKs. Retire actual ownership
      // and stop optional admissions while the original video keeps playing.
      this.motionFailed=true;this.motionFrames.clear();this.motionPresents.clear();
      this.pendingMotionMetadata=null;this.motionMetadataInFlight=false;
      this.pendingPresent.clear();this.requestedRevision.clear();
      this.worker.terminate();this.preprocessWorker.terminate();
    }
    retireMotionWork(message){
      if(message.type==="frame_preprocessed"&&this.motionFrames.has(message.sequence))
        this.motionFrames.set(message.sequence,true);
      if(message.type==="frame_processed"||message.type==="failure"&&
        ["frame","frame_bitmap"].includes(message.request_type))this.motionFrames.delete(message.sequence);
      if(message.type==="present"||message.type==="failure"&&message.request_type==="present")
        this.motionPresents.delete(message.sequence);
      if(message.type==="metadata_processed"||message.type==="failure"&&message.request_type==="metadata"){
        this.motionMetadataInFlight=false;
        const pending=this.pendingMotionMetadata;this.pendingMotionMetadata=null;
        if(pending)try{this.postMotion(pending);}
        catch(error){this.stats.metadata_failures++;}
      }
    }
    postMotion(message,transfers=[]){
      if(this.motionFailed)return false;
      const frame=message.type==="frame_bitmap",present=message.type==="present",metadata=message.type==="metadata";
      if(frame&&this.motionFrames.size>=2||present&&this.motionPresents.size>=2)return false;
      if(metadata&&this.motionMetadataInFlight){
        if(this.pendingMotionMetadata){
          // Delta transports can carry different real anchors. Keep a bounded
          // union, with the newest snapshot authoritative for events/identities.
          const previous=this.pendingMotionMetadata,anchors=new Map();
          for(const anchor of [...previous.data.anchors,...message.data.anchors])
            anchors.set(anchor?.source_sequence,anchor);
          const preserveEvents=sequence(previous.data.event_snapshot_revision)&&
            (!sequence(message.data.event_snapshot_revision)||
              previous.data.event_snapshot_revision>message.data.event_snapshot_revision);
          message={...message,queued_at:previous.queued_at,data:{...message.data,
            ...(preserveEvents?{events:previous.data.events,event_snapshot_revision:previous.data.event_snapshot_revision,
              event_snapshot_timestamp:previous.data.event_snapshot_timestamp}:{}),
            anchors:[...anchors.values()].sort((a,b)=>(a?.source_sequence??-1)-(b?.source_sequence??-1)).slice(-256)}};
          this.stats.motion_metadata_coalesced++;
        }
        this.pendingMotionMetadata=message;return true;
      }
      if(frame){this.motionFrames.set(message.frame.sequence,false);
        message={...message,motion_deadline_epoch_ms:Date.now()+1000};}
      if(present)this.motionPresents.add(message.sequence);
      if(metadata)this.motionMetadataInFlight=true;
      try{this.preprocessWorker.postMessage({...message,queued_epoch_ms:Date.now()},transfers);}
      catch(error){
        if(frame)this.motionFrames.delete(message.frame.sequence);
        if(present)this.motionPresents.delete(message.sequence);
        if(metadata)this.motionMetadataInFlight=false;
        throw error;
      }
      if(frame)this.stats.motion_frames_admitted++;
      return true;
    }
    start(){if(this.config.source_mode!=="network")this.scheduleCapture();this.animate();
      if(!["websocket","frames"].includes(this.config.transport))this.pollMetadata();}
    scheduleCapture(){if(!this.closed&&this.capturing)
      this.captureId=this.video.requestVideoFrameCallback((now,meta)=>{this.capture(now,meta);this.scheduleCapture();});}
    capture(now,meta,externalSource=null){
      if(this.closed||!this.capturing||!finite(meta.mediaTime)||meta.presentedFrames===this.lastVideoFrame)return;
      const captureStarted=performance.now();
      if(finite(this.lastCallbackAt))this.noteTiming("callback_gap",captureStarted-this.lastCallbackAt);
      this.lastCallbackAt=captureStarted;
      if(finite(meta.expectedDisplayTime))this.noteTiming("callback_lateness",Math.max(0,captureStarted-meta.expectedDisplayTime));
      this.stats.video_callback_frames++;
      if(sequence(meta.presentedFrames)&&this.lastVideoFrame>=0)
        this.stats.video_callback_gap_frames+=Math.max(0,meta.presentedFrames-this.lastVideoFrame-1);
      this.lastVideoFrame=meta.presentedFrames;
      if(this.firstMediaTime===null)this.firstMediaTime=meta.mediaTime;
      const timestamp=externalSource?meta.mediaTime:meta.mediaTime-this.firstMediaTime;
      if(timestamp<0)return;
      this.latestClock={timestamp,now};
      if(!externalSource&&timestamp+1e-6<this.nextCaptureAt)return;
      let sample=!externalSource&&!this.analysisPause&&timestamp+1e-6>=this.nextAnalysisAt;
      // Pause new analysis admission, while the independent real-video clock continues.
      // Already admitted JPEG tasks stay in FIFO until the server acknowledges them.
      if(sample&&this.uploads.length>=12){
        this.stats.analysis_capture_pauses++;this.analysisPause="capacity";sample=false;
        this.onFailure("分析上传积压，新的分析采样暂缓；画面继续播放，已接纳帧保留。");
      }
      const width=externalSource?.width||this.video.videoWidth,height=externalSource?.height||this.video.videoHeight;
      if(!width||!height)return;
      if(this.sourceSize&&(this.sourceSize[0]!==width||this.sourceSize[1]!==height)){
        this.capturing=false;this.onFailure("摄像头画面尺寸已改变，请重新开始检测以重新对齐描边。");return;
      }
      if(!this.sourceSize){
        this.displayBytesPerFrame=width*height*4;
        this.requiredDisplayBufferBytes=this.displayBytesPerFrame*this.frames.maxFrames;
        if(this.requiredDisplayBufferBytes>this.displayBufferBudgetBytes){
          this.capturing=false;this.stats.display_buffer_capacity_failures++;
          this.onFailure("当前原图分辨率的画面缓冲需要约 "+Math.ceil(this.requiredDisplayBufferBytes/1024**2)+
            " MiB，超过 1280 MiB 的本地缓冲预算。请缩短显示缓冲后重新开始；画面未缩小。");return;
        }
      }
      this.sourceSize=[width,height];
      const period=1/this.displayFps;
      do{this.nextCaptureAt+=period;}while(this.nextCaptureAt<=timestamp+1e-6);
      // Human presentation retains the camera's original pixels. Model inputs
      // and the motion working image each have their own independent scale.
      const displayWidth=width,displayHeight=height;
      if(this.captureCanvas.width!==displayWidth)this.captureCanvas.width=displayWidth;
      if(this.captureCanvas.height!==displayHeight)this.captureCanvas.height=displayHeight;
      let source=externalSource||this.video;
      let stageStarted=performance.now();
      if(sample){
        // The video decoder can advance between reads. Freeze an admitted
        // source once at native resolution, then derive both display and model
        // images from it, regardless of the independently configured model size.
        if(!this.sourceCanvas){this.sourceCanvas=new OffscreenCanvas(width,height);
          this.sourceContext=this.sourceCanvas.getContext("2d");}
        this.sourceContext.drawImage(this.video,0,0,width,height);source=this.sourceCanvas;
      }
      if(sample)this.noteTiming("source_freeze",performance.now()-stageStarted);
      stageStarted=performance.now();
      this.captureContext.drawImage(source,0,0,this.captureCanvas.width,this.captureCanvas.height);
      const bitmap=this.captureCanvas.transferToImageBitmap(),seq=externalSource?meta.presentedFrames:this.sourceSequence++;
      this.frames.push({sequence:seq,timestamp,capturedAt:now,bitmap});this.stats.captured_frames++;
      this.noteTiming("display_capture",performance.now()-stageStarted);stageStarted=performance.now();
      // Bound motion work by area, not only width: a 4:3 camera otherwise
      // costs a third more pixels than the wide-video acceptance sample.
      // Keep one fixed scale for the run. Optional motion work has a bounded
      // admission window; original playback and model uploads keep their clocks.
      const requestedWidth=clamp(Number(this.config.motion_width)||160,160,320),
        motionScale=Math.min(requestedWidth/width,240/height,Math.sqrt(requestedWidth**2*9/16/(width*height))),
        grayWidth=Math.max(16,Math.floor(width*motionScale)),grayHeight=Math.max(16,Math.floor(height*motionScale));
      if(this.grayCanvas.width!==grayWidth)this.grayCanvas.width=grayWidth;
      if(this.grayCanvas.height!==Math.min(240,grayHeight))this.grayCanvas.height=Math.min(240,grayHeight);
      if(!this.motionFailed&&this.motionFrames.size<2){
        this.grayContext.drawImage(bitmap,0,0,this.grayCanvas.width,this.grayCanvas.height);
        const motionBitmap=this.grayCanvas.transferToImageBitmap();
        try{this.postMotion({type:"frame_bitmap",queued_at:performance.now(),frame:{sequence:seq,timestamp,
          width:this.grayCanvas.width,height:this.grayCanvas.height,bitmap:motionBitmap}},[motionBitmap]);}
        catch(error){motionBitmap.close();this.stats.motion_failures++;
          this.onFailure("轮廓画面传送失败，视频继续播放；请重新开始检测。");}
      }else this.stats.motion_frames_skipped++;
      this.noteTiming("motion_capture",performance.now()-stageStarted);stageStarted=performance.now();
      if(sample){
        const analysisPeriod=1/Math.max(Number(this.config.analysis_sample_fps)||3,Number(this.settings.sample_fps)||0);
        do{this.nextAnalysisAt+=analysisPeriod;}while(this.nextAnalysisAt<=timestamp+1e-6);
        const size=Math.min(1,this.settings.max_image_side/Math.max(width,height));
        const image=new OffscreenCanvas(Math.max(1,Math.round(width*size)),Math.max(1,Math.round(height*size)));
        image.getContext("2d").drawImage(source,0,0,image.width,image.height);
        this.uploads.push({image:image.transferToImageBitmap(),metadata:{stream_session_id:this.config.stream_session_id,
          generation:this.config.generation,crop_version:this.config.crop_version,
          source_sequence:seq,media_timestamp:timestamp,width:image.width,height:image.height}});
        this.stats.analysis_admitted++;this.sendUploads();
        this.noteTiming("analysis_capture",performance.now()-stageStarted);
      }
      this.noteTiming("capture_total",performance.now()-captureStarted);
    }
    async sendUploads(){
      if(this.closed)return;
      if(this.uploading){this.prepareNextUpload();return;}
      this.uploading=true;
      try{
        while(this.uploads.length&&!this.closed){
          if(this.drainDeadline&&performance.now()>this.drainDeadline)throw new Error("upload drain deadline");
          const task=this.uploads[0];
          await this.prepareUpload(task);
          if(this.closed)break;
          if(this.socket.readyState!==1)throw new Error("socket closed");
          if(this.socket.bufferedAmount>1024*1024){await new Promise(resolve=>setTimeout(resolve,20));continue;}
          const acknowledged=new Promise((resolve,reject)=>{this.pendingAck={task,resolve,reject,
            timer:setTimeout(()=>{this.pendingAck=null;reject(new Error("analysis acknowledgement timed out"));},5000)};});
          const sentAt=performance.now();this.socket.send(task.packet);
          this.prepareNextUpload();
          const reply=await acknowledged;
          this.noteTiming("analysis_ack",performance.now()-sentAt);
          if(!reply.accepted){
            if(this.analysisPause!=="capacity")this.stats.analysis_capture_pauses++;
            this.analysisPause=reply.reason==="capacity"?"capacity":"obsolete";
            if(reply.reason==="capacity"&&(task.capacityRetries=(task.capacityRetries||0)+1)<=3){
              this.onFailure("服务器分析队列暂满；画面继续播放，已接纳帧保留并等待重试。");
              await new Promise(resolve=>setTimeout(resolve,250));continue;
            }
            throw new Error(reply.reason==="capacity"?"analysis capacity unavailable":"obsolete analysis source");
          }
          this.uploads.shift();this.stats.analysis_sent++;
          if(this.analysisPause==="capacity")this.analysisPause=null;
        }
      }catch(error){
        if(this.pendingAck){clearTimeout(this.pendingAck.timer);this.pendingAck=null;}
        this.stats.analysis_upload_failures++;this.analysisPause=this.analysisPause||"failure";
        if(!this.closed)this.onFailure(this.analysisPause==="obsolete"?
          "服务器已拒绝旧的采集会话，分析暂停；未确认帧保留，请重新开始检测。":
          "分析帧尚未获服务器确认，分析暂停；画面继续播放，未确认帧保留。");
      }finally{this.uploading=false;}
    }
    prepareUpload(task){
      if(task.packet)return Promise.resolve();
      if(!task.preparation)task.preparation=(async()=>{
        const encodeStarted=performance.now(),bytes=await this.encodeImage(task);
        task.packet=packCameraFrame(task.metadata,bytes);task.image.close();task.image=null;
        this.stats.analysis_prepared++;
        this.noteTiming("analysis_encode",performance.now()-encodeStarted);
      })();
      return task.preparation;
    }
    prepareNextUpload(){
      if(this.closed||!this.pendingAck||this.pendingEncode||this.uploads.length<2)return;
      const next=this.uploads[1];
      if(next.packet||next.preparation)return;
      // One lookahead slot overlaps encoding with the current packet's ACK.
      // Only the foreground FIFO loop sends and removes acknowledged packets.
      this.prepareUpload(next).catch(()=>{/* The same retained task fails when the FIFO reaches it. */});
    }
    encodeImage(task){
      return new Promise((resolve,reject)=>{
        const sequence=task.metadata.source_sequence;
        this.pendingEncode={sequence,resolve,reject,timer:setTimeout(()=>{
          this.pendingEncode=null;reject(new Error("analysis encoding timed out"));
        },10000)};
        // Structured cloning keeps the admitted source in this FIFO until a
        // JPEG is returned. A Worker failure cannot destroy its only pixels.
        try{this.encodeWorker.postMessage({type:"encode_jpeg",sequence,bitmap:task.image,
          quality:this.settings.jpeg_quality/100});}
        catch(error){clearTimeout(this.pendingEncode.timer);this.pendingEncode=null;reject(error);}
      });
    }
    acknowledgeUpload(message){
      const pending=this.pendingAck;
      if(!pending||message?.type!=="camera_frame_ack"||!sameRun(message,this.config)||
        message.source_sequence!==pending.task.metadata.source_sequence||typeof message.accepted!=="boolean")return false;
      clearTimeout(pending.timer);this.pendingAck=null;pending.resolve(message);return true;
    }
    animate(){
      if(this.closed)return;
      const now=performance.now();
      if(this.latestClock){
        const mediaNow=this.latestClock.timestamp+(now-this.latestClock.now)/1000;
        // Preserve the configured playback delay, but never replay a growing
        // backlog after a hidden tab or slow renderer. Only presentation copies
        // older than the delayed playhead plus a bounded grace are discarded.
        const discarded=this.frames.discardBefore(mediaNow-this.frames.delay-.5);
        if(discarded>=0){
          this.lastDiscardedSequence=Math.max(this.lastDiscardedSequence,discarded);
          for(const collection of [this.overlays,this.requested,this.pendingPresent,this.requestedRevision])
            for(const key of collection.keys())if(key<=discarded)collection.delete(key);
        }
        for(const frame of this.frames.frames){
          if(frame.timestamp>mediaNow-this.frames.delay+this.presentPrefetchSeconds)break;
          if(!this.requested.has(frame.sequence)&&!this.pendingPresent.has(frame.sequence)){
            // Display retirement does not cancel a message already in a worker
            // port. Bound actual outstanding queries, independently of overlays.
            if(this.motionFailed||this.motionPresents.size>=2)break;
            this.requested.add(frame.sequence);this.pendingPresent.add(frame.sequence);
            this.requestedRevision.set(frame.sequence,this.geometryRevision);
            this.postMotion({type:"present",id:frame.sequence,queued_at:now,
              deadline_epoch_ms:Date.now()+Math.max(0,(frame.timestamp+this.frames.delay+this.presentGraceSeconds-mediaNow)*1000)+100,
              sequence:frame.sequence,timestamp:frame.timestamp});
          }
        }
        const due=this.frames.frames[0],overlayPending=due&&!this.overlays.has(due.sequence)&&this.pendingPresent.has(due.sequence),
          withinGrace=due&&mediaNow<due.timestamp+this.frames.delay+this.presentGraceSeconds,
          frame=overlayPending&&withinGrace?null:this.frames.take(mediaNow);
        if(frame){
          const presentStarted=performance.now();
          if(this.canvas.width!==frame.bitmap.width)this.canvas.width=frame.bitmap.width;
          if(this.canvas.height!==frame.bitmap.height)this.canvas.height=frame.bitmap.height;
          this.context.drawImage(frame.bitmap,0,0);
          const overlay=this.overlays.get(frame.sequence),subjects=(overlay?.subjects||[])
            .filter(subject=>subject.target_sequence===frame.sequence)
            .map(subject=>({...subject,display_name:this.identities.nameAt(subject,frame.timestamp)}));
          if(subjects.length&&(overlay.coverage!=="anchored"||subjects.some(subject=>subject.complete===false)))
            this.stats.partial_contour_frames++;
          if(subjects.length&&overlay.coverage==="anchored"&&subjects.every(subject=>subject.complete!==false))
            this.stats.complete_contour_frames++;
          const approximations=subjects.filter(subject=>subject.approximate===true).length;
          if(approximations){this.stats.approximate_contour_frames++;this.stats.approximate_subjects+=approximations;}
          if(overlay?.all_subjects_present===true&&subjects.length===overlay.subjects.length&&
            subjects.every(subject=>subject.polygons?.length&&(subject.complete||subject.approximate)))
            this.stats.continuous_contour_frames++;
          for(const reason of motionRejectionReasons){
            const count=overlay?.motion_rejections?.[reason];
            if(sequence(count))this.stats["warp_reject_"+reason]+=count;
          }
          for(const subject of subjects){
            if(subject.target_sequence!==frame.sequence)continue;
            this.context.strokeStyle=subject.tracking_confirmation==="provisional"?"#9ca3af":subject.alert?"#ef4444":subject.warning?"#facc15":"#22c55e";
            this.context.lineWidth=2.5;
            for(const polygon of subject.polygons||[]){this.context.beginPath();
              polygon.forEach((point,index)=>{const x=point[0]*this.canvas.width,y=point[1]*this.canvas.height;
                if(index===0)this.context.moveTo(x,y);else this.context.lineTo(x,y);});
              this.context.closePath();this.context.stroke();}
            for(const segment of subject.open_segments||[]){this.context.beginPath();
              segment.forEach((point,index)=>{const x=point[0]*this.canvas.width,y=point[1]*this.canvas.height;
                if(index===0)this.context.moveTo(x,y);else this.context.lineTo(x,y);});
              this.context.stroke();}
          }
          for(const subject of subjects)
            drawSubjectLabel(this.context,subject,this.canvas.width,this.canvas.height,this.canvas.clientWidth);
          if(!subjects.length){
            this.stats.frames_without_contours++;
            const reason="coverage_"+(overlay?.coverage||"not_ready");
            if(Object.prototype.hasOwnProperty.call(this.stats,reason))this.stats[reason]++;
          }
          else if(overlay.exact)this.stats.observed_contour_frames++;else this.stats.propagated_contour_frames++;
          if(subjects.some(subject=>subject.relocalized))this.stats.relocalized_contour_frames++;
          if(subjects.some(subject=>subject.method==="image_motion_backward"))this.stats.backward_contour_frames++;
          this.canvas.dataset.frameSequence=String(frame.sequence);
          this.canvas.dataset.mediaTimestamp=String(frame.timestamp);
          this.canvas.dataset.contourCoverage=overlay?.coverage||"not_ready";
          if(this.frames.lastPresentedAt!==null)this.maxPresentGap=Math.max(this.maxPresentGap||0,(now-this.frames.lastPresentedAt)/1000);
          this.frames.notePresented(frame,now);this.lastPresentedSequence=frame.sequence;
          this.stats.last_source_to_present_seconds=(now-frame.capturedAt)/1000;
          this.presentationAges.push(this.stats.last_source_to_present_seconds);
          if(this.presentationAges.length>600)this.presentationAges.shift();
          frame.bitmap.close();
          for(const key of this.overlays.keys())if(key<=frame.sequence)this.overlays.delete(key);
          for(const key of this.requested)if(key<=frame.sequence)this.requested.delete(key);
          for(const key of this.pendingPresent)if(key<=frame.sequence){
            this.pendingPresent.delete(key);this.requestedRevision.delete(key);
          }
          this.noteTiming("presentation",performance.now()-presentStarted);
        }
      }
      if(now-this.lastStatusAt>=1000){this.lastStatusAt=now;this.onStatus(this.snapshot());}
      if(now-this.lastMetricsAt>=5000&&!this.metricsPending)this.reportMetrics();
      this.animationId=requestAnimationFrame(()=>this.animate());
    }
    async pollMetadata(){
      if(this.closed||this.metadataPending||["websocket","frames"].includes(this.config.transport))return;
      this.metadataPending=true;
      const controller=new AbortController(),started=performance.now();
      this.metadataController=controller;this.stats.metadata_requests++;
      const timer=setTimeout(()=>controller.abort(),2000);
      let stage="network";
      try{
        const base=this.config.metadata_url||"/v1/video/display/metadata",
          delta=this.config.transport==="stream",
          url=delta?base+"?"+new URLSearchParams({stream_session_id:this.config.stream_session_id,
            revision:this.lastMetadataRevision??-1}):base;
        const response=await fetch(url,{cache:"no-store",signal:controller.signal});
        if(response.status===204)return;
        if(!response.ok){stage="http";throw new Error("metadata unavailable");}
        stage="decode";
        const data=await response.json();
        if(this.closed)return;
        if(!data||!sameRun(data,this.config)){stage="run";throw new Error("metadata run mismatch");}
        if(!Array.isArray(data.anchors))throw new Error("invalid metadata");
        this.stats.metadata_latency_max_seconds=Math.max(this.stats.metadata_latency_max_seconds,
          (performance.now()-started)/1000);
        this.receiveMetadata(data,{transport:delta?"http_delta":"http"});
      }catch(error){if(!this.closed){
        this.stats.metadata_failures++;
        if(controller.signal.aborted)this.stats.metadata_timeouts++;
        else if(stage==="http")this.stats.metadata_http_errors++;
        else if(stage==="decode")this.stats.metadata_decode_errors++;
        else if(stage==="run")this.stats.metadata_run_mismatches++;
      }}
      finally{clearTimeout(timer);this.metadataPending=false;
        if(!this.closed)this.metadataTimer=setTimeout(()=>this.pollMetadata(),150);}
    }
    receiveMetadata(data,{transport="http"}={}){
      if(this.closed)return false;
      if(!data||!sameRun(data,this.config)){
        this.stats.metadata_failures++;this.stats.metadata_run_mismatches++;return false;
      }
      if(!Array.isArray(data.anchors)||data.anchors.length>256){
        this.stats.metadata_failures++;this.stats.metadata_decode_errors++;return false;
      }
      if(["websocket","http_delta"].includes(transport)){
        if(!sequence(data.metadata_revision)){
          this.stats.metadata_failures++;this.stats.metadata_invalid_revisions++;return false;
        }
        if(data.metadata_revision<=(this.lastMetadataRevision??-1)){
          this.stats.metadata_duplicate_revisions++;return false;
        }
        this.lastMetadataRevision=data.metadata_revision;if(transport==="websocket")this.stats.metadata_ws_received++;
      }
      this.stats.metadata_received++;this.lastMetadataReceivedAt=performance.now();
      this.identities.accept(data);
      this.stats.metadata_anchors_received+=data.anchors.length;
      for(const anchor of data.anchors)if(finite(anchor?.media_timestamp))
        this.latestReceivedAnchorTimestamp=Math.max(this.latestReceivedAnchorTimestamp??0,anchor.media_timestamp);
      this.postMotion({type:"metadata",data,queued_at:this.lastMetadataReceivedAt});return true;
    }
    anchorAge(timestamp){
      return finite(timestamp)&&this.latestClock?Math.max(0,this.latestClock.timestamp+
        (performance.now()-this.latestClock.now)/1000-timestamp):null;
    }
    snapshot(){return {display_instance_id:this.displayInstanceId,...this.frames.stats(),...this.stats,analysis_pending:this.uploads.length,
      configured_display_delay_seconds:this.frames.delay,
      presentation_prefetch_seconds:this.presentPrefetchSeconds,pending_present_queries:this.pendingPresent.size,
      source_width:this.sourceSize?.[0]||0,source_height:this.sourceSize?.[1]||0,
      motion_input_width:this.grayCanvas.width,motion_input_height:this.grayCanvas.height,
      estimated_display_buffer_bytes:this.frames.frames.length*this.displayBytesPerFrame,
      required_display_buffer_bytes:this.requiredDisplayBufferBytes,
      analysis_prepared_pending:this.uploads.slice(1).filter(task=>!!task.packet).length,
      source_callback_fps:this.latestClock?.timestamp>0?
        (this.stats.video_callback_frames-1)/(this.latestClock.timestamp-(this.config.source_mode==="network"?this.firstMediaTime:0)||1):null,
      worker_pending_frames:this.motionFrames.size,
      preprocess_pending_frames:[...this.motionFrames.values()].filter(prepared=>!prepared).length,
      motion_pending_frames:[...this.motionFrames.values()].filter(Boolean).length,
      worker_pending_present_queries:this.motionPresents.size,
      worker_pending_metadata:Number(this.motionMetadataInFlight)+Number(!!this.pendingMotionMetadata),
      last_metadata_received_age_seconds:this.lastMetadataReceivedAt===undefined?null:
        Math.max(0,(performance.now()-this.lastMetadataReceivedAt)/1000),
      last_metadata_applied_age_seconds:this.lastMetadataAppliedAt===undefined?null:
        Math.max(0,(performance.now()-this.lastMetadataAppliedAt)/1000),
      latest_received_anchor_age_seconds:this.anchorAge(this.latestReceivedAnchorTimestamp),
      latest_applied_anchor_age_seconds:this.anchorAge(this.latestAppliedAnchorTimestamp),
      page_hidden:typeof document!=="undefined"&&document.hidden?1:0,visibility_changes:this.visibilityChanges,
      hidden_seconds:this.hiddenSeconds+(this.hiddenSince===null?0:(performance.now()-this.hiddenSince)/1000),
      longtask_count:this.longTaskCount,longtask_total_seconds:this.longTaskTotalMs/1000,
      stream_session_id:this.config.stream_session_id,generation:this.config.generation,crop_version:this.config.crop_version};}
    async reportMetrics(final=false){
      if(this.metricsPending&&!final)return;
      const now=performance.now(),stats=this.snapshot(),windowSeconds=Math.max(0,(now-this.lastMetricsAt)/1000),
        windowFrames=stats.distinct_presented_frames-this.metricsPresented,ages=this.presentationAges.slice().sort((a,b)=>a-b),
        percentile=fraction=>ages.length?ages[Math.ceil((ages.length-1)*fraction)]:null;
      const body={...stats,report_sequence:++this.reportSequence,elapsed_seconds:Math.max(0,(now-this.startedAt)/1000),
        window_presented_frames:windowFrames,window_seconds:windowSeconds,
        window_presented_fps:windowSeconds>0?windowFrames/windowSeconds:0,
        source_to_present_p50_seconds:percentile(.5),source_to_present_p95_seconds:percentile(.95),
        source_to_present_max_seconds:ages.length?ages.at(-1):null,
        contour_frame_ratio:stats.distinct_presented_frames>0?
          (stats.observed_contour_frames+stats.propagated_contour_frames)/stats.distinct_presented_frames:null,
        complete_contour_frame_ratio:stats.distinct_presented_frames>0?
          stats.complete_contour_frames/stats.distinct_presented_frames:null,
        continuous_contour_frame_ratio:stats.distinct_presented_frames>0?
          stats.continuous_contour_frames/stats.distinct_presented_frames:null,
        window_callback_frames:this.stats.video_callback_frames-this.metricsCallbackFrames,
        window_callback_gap_frames:this.stats.video_callback_gap_frames-this.metricsCallbackGapFrames,
        max_present_gap_seconds:this.maxPresentGap};
      for(const [name,window] of Object.entries(this.timingWindows)){
        const values=window.samples.sort((a,b)=>a-b),prefix="window_"+name;
        body[prefix+"_count"]=window.count;
        body[prefix+"_sample_count"]=values.length;
        body[prefix+"_mean_seconds"]=window.total/window.count/1000;
        body[prefix+"_p95_seconds"]=values[Math.ceil(values.length*.95)-1]/1000;
        body[prefix+"_max_seconds"]=window.max/1000;
      }
      this.timingWindows={};this.metricsCallbackFrames=this.stats.video_callback_frames;
      this.metricsCallbackGapFrames=this.stats.video_callback_gap_frames;
      this.lastMetricsAt=now;this.metricsPresented=stats.distinct_presented_frames;this.presentationAges=[];
      if(this.config.transport==="websocket"&&this.socket.readyState===1&&this.socket.bufferedAmount<1024*1024){
        try{this.socket.send(JSON.stringify({type:"camera_display_metrics",...body}));return;}
        catch(error){/* A closing socket may fall back to the compatibility endpoint. */}
      }
      this.metricsPending=true;
      const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),2000);
      try{await fetch(this.config.metrics_url||"/v1/video/display/metrics",{method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify(body),keepalive:final,signal:controller.signal});}
      catch(error){/* Telemetry is independent of the video and admitted analysis queue. */}
      finally{clearTimeout(timer);this.metricsPending=false;}
    }
    async drain(){
      this.capturing=false;if(this.captureId!==undefined)this.video.cancelVideoFrameCallback(this.captureId);
      const started=performance.now();
      this.drainDeadline=started+10000;
      while(this.uploading){
        if(performance.now()-started>10000)throw new Error("analysis upload drain timed out; frames remain pending");
        await new Promise(resolve=>setTimeout(resolve,20));
      }
      if(this.uploads.length){await this.sendUploads();if(this.uploads.length)throw new Error("analysis uploads pending");}
      while(this.socket.readyState===1&&this.socket.bufferedAmount>0){
        if(performance.now()-started>10000)throw new Error("camera send queue has not drained");
        await new Promise(resolve=>setTimeout(resolve,20));
      }
      if(this.socket.readyState!==1)throw new Error("camera connection closed before drain");
    }
    stop(){
      if(!this.closed)this.reportMetrics(true);
      this.closed=true;this.capturing=false;
      this.claimedWarmup?.resolve(false);
      if(this.pendingAck){clearTimeout(this.pendingAck.timer);this.pendingAck.reject(new Error("display stopped"));this.pendingAck=null;}
      if(this.pendingEncode){clearTimeout(this.pendingEncode.timer);
        this.pendingEncode.reject(new Error("display stopped"));this.pendingEncode=null;}
      if(this.captureId!==undefined)this.video.cancelVideoFrameCallback(this.captureId);
      cancelAnimationFrame(this.animationId);clearTimeout(this.metadataTimer);this.metadataController?.abort();
      this.longTaskObserver?.disconnect();
      if(typeof document!=="undefined")document.removeEventListener?.("visibilitychange",this.onVisibilityChange);
      this.worker.terminate();this.preprocessWorker.terminate();this.encodeWorker.terminate();this.frames.reset();this.overlays.clear();
      this.requested.clear();this.pendingPresent.clear();this.requestedRevision.clear();
      this.motionFrames.clear();this.motionPresents.clear();this.pendingMotionMetadata=null;this.motionMetadataInFlight=false;
      this.identities.records=[];
      for(const task of this.uploads)task.image?.close();
    }
  }

  function decodeDisplayBatch(buffer){
    if(!(buffer instanceof ArrayBuffer)||buffer.byteLength<8||buffer.byteLength>8*1024**2)throw new Error("invalid display batch");
    const bytes=new Uint8Array(buffer),size=new DataView(buffer).getUint32(4);
    if(String.fromCharCode(...bytes.subarray(0,4))!=="GVD1"||size<1||size>1024**2||size+8>bytes.length)
      throw new Error("invalid display header");
    const data=JSON.parse(new TextDecoder().decode(bytes.subarray(8,8+size)));
    if(data.protocol!==1||typeof data.connection_id!=="string"||!data.config||
      !safeId(data.config.stream_session_id)||!sequence(data.config.generation)||!sequence(data.config.crop_version)||
      !Array.isArray(data.frames)||data.frames.length>64)throw new Error("invalid display identity");
    let offset=8+size,previous=-1,stamp=-1;
    for(const frame of data.frames){
      if(!sequence(frame.source_sequence)||frame.source_sequence<=previous||!finite(frame.media_timestamp)||
        frame.media_timestamp<0||frame.media_timestamp<=stamp||!Number.isInteger(frame.bytes)||frame.bytes<1||
        offset+frame.bytes>bytes.length||!Number.isInteger(frame.width)||!Number.isInteger(frame.height)||
        frame.width<1||frame.height<1||frame.width*frame.height>16777216)throw new Error("invalid display frame");
      frame.jpeg=bytes.subarray(offset,offset+frame.bytes);offset+=frame.bytes;
      previous=frame.source_sequence;stamp=frame.media_timestamp;
    }
    if(offset!==bytes.length)throw new Error("unexpected display bytes");
    return data;
  }

  class DisplayStreamDecoder{
    constructor(){this.prefix=new Uint8Array(4);this.prefixUsed=0;this.body=null;this.used=0;}
    feed(chunk){
      const packets=[];let offset=0;
      while(offset<chunk.length){
        if(!this.body){
          const count=Math.min(4-this.prefixUsed,chunk.length-offset);
          this.prefix.set(chunk.subarray(offset,offset+count),this.prefixUsed);
          this.prefixUsed+=count;offset+=count;
          if(this.prefixUsed<4)break;
          const size=new DataView(this.prefix.buffer).getUint32(0);
          if(size<8||size>8*1024**2)throw new Error("invalid display stream length");
          this.body=new Uint8Array(size);this.used=0;
        }
        const count=Math.min(this.body.length-this.used,chunk.length-offset);
        this.body.set(chunk.subarray(offset,offset+count),this.used);this.used+=count;offset+=count;
        if(this.used===this.body.length){
          const data=decodeDisplayBatch(this.body.buffer);data.byte_length=this.body.length;packets.push(data);
          this.body=null;this.prefixUsed=0;this.used=0;
        }
      }
      return packets;
    }
    finish(){if(this.prefixUsed||this.body)throw new Error("truncated display stream");}
  }

  class NetworkController{
    static supported(){return BrowserController.supported(null)&&typeof createImageBitmap==="function";}
    constructor({canvas,url,metricsUrl,onStatus=()=>{},onFailure=()=>{},onFrame=()=>{}}){
      this.canvas=canvas;this.url=url;this.metricsUrl=metricsUrl;this.onStatus=onStatus;
      this.onFailure=onFailure;this.onFrame=onFrame;this.closed=false;this.after=-1;
      this.connectionId="";this.session="";this.revision=-1;this.controller=null;this.key=null;
      this.socket={readyState:0,bufferedAmount:0};
      this.streamConnections=0;this.queue=[];this.queueBytes=0;this.queueFrames=0;this.draining=false;this.acceptEpoch=0;this.queuedKey=null;
    }
    start(){BrowserController.warmupMotion();this.pull();}
    reset(config,connectionId){
      this.controller?.stop();this.controller=null;this.revision=-1;
      this.session=config.stream_session_id;
      this.key=JSON.stringify([connectionId,config.stream_session_id,config.generation,config.crop_version,
        config.enabled,config.delay_seconds]);
      this.canvas.getContext("2d").clearRect(0,0,this.canvas.width,this.canvas.height);
      if(config.enabled){
        this.controller=new BrowserController({video:null,canvas:this.canvas,socket:this.socket,
          config:{...config,source_mode:"network",transport:config.transport==="stream"?"stream":"frames",
            metadata_url:this.url.replace(/\/frames(?:\?.*)?$/,"/metadata"),metrics_url:this.metricsUrl},
          settings:{sample_fps:0,max_image_side:1280,jpeg_quality:80},
          onStatus:this.onStatus,onFailure:this.onFailure});
        this.controller.stats.network_requests=this.streamConnections;
        this.controller.start();
      }
    }
    enqueue(data){
      const config=data.config,key=JSON.stringify([data.connection_id,config.stream_session_id,
        config.generation,config.crop_version,config.enabled,config.delay_seconds]);
      const changed=key!==this.queuedKey;
      if(changed){this.queue=[];this.queueBytes=0;this.queueFrames=0;this.acceptEpoch++;this.queuedKey=key;}
      if(!changed&&!data.frames.length)return;
      data.received_at=performance.now();
      const size=data.byte_length||data.frames.reduce((sum,frame)=>sum+frame.jpeg.length,0),
        latest=data.frames.at(-1)?.media_timestamp,deadline=latest-(Number(config.delay_seconds)||0)-.5;
      this.queue.push({data,size,epoch:this.acceptEpoch});this.queueBytes+=size;this.queueFrames+=data.frames.length;
      // Keep still-useful pictures through a short decode stall, not an unlimited
      // replay backlog. This queue contains display JPEGs only, never inference work.
      while(this.queue.length>1&&(this.queueBytes>32*1024**2||this.queueFrames>128||
        this.queue[0].data.frames.at(-1)?.media_timestamp<deadline)){
        const old=this.queue.shift();this.queueBytes-=old.size;this.queueFrames-=old.data.frames.length;
        if(this.controller)this.controller.stats.network_queue_dropped_frames+=old.data.frames.length;
      }
      void this.drain();
    }
    async drain(){
      if(this.draining)return;this.draining=true;
      try{while(this.queue.length&&!this.closed){
        const item=this.queue.shift();this.queueBytes-=item.size;this.queueFrames-=item.data.frames.length;
        try{await this.accept(item.data,item.epoch);}
        catch(error){if(!this.closed){if(this.controller)this.controller.stats.network_failures++;
          this.onFailure("画面解码暂时失败，正在继续接收。");}}
      }}finally{this.draining=false;}
    }
    async accept(data,epoch=this.acceptEpoch){
      const config=data.config,key=JSON.stringify([data.connection_id,config.stream_session_id,config.generation,
        config.crop_version,config.enabled,config.delay_seconds]);
      if(this.key!==key)this.reset(config,data.connection_id);
      if(this.connectionId!==data.connection_id)this.after=-1;
      this.connectionId=data.connection_id;
      const latest=data.frames.at(-1)?.media_timestamp,receivedAt=data.received_at??performance.now();
      for(const frame of data.frames){
        if(frame.source_sequence<=this.after)continue;
        const decodeStarted=performance.now();
        const bitmap=await createImageBitmap(new Blob([frame.jpeg],{type:"image/jpeg"}));
        try{
          if(this.closed||epoch!==this.acceptEpoch)return;
          if(bitmap.width!==frame.width||bitmap.height!==frame.height)throw new Error("display dimensions changed");
          if(this.controller){
            this.controller.noteTiming("network_decode",performance.now()-decodeStarted);
            this.controller.stats.network_frames_received++;
            if(this.after>=0)this.controller.stats.network_source_gap_frames+=Math.max(0,frame.source_sequence-this.after-1);
            this.controller.capture(receivedAt-(latest-frame.media_timestamp)*1000,
              {mediaTime:frame.media_timestamp,presentedFrames:frame.source_sequence},bitmap);
          }else{
            this.canvas.width=bitmap.width;this.canvas.height=bitmap.height;
            this.canvas.getContext("2d").drawImage(bitmap,0,0);
            this.canvas.dataset.frameSequence=String(frame.source_sequence);
            this.canvas.dataset.mediaTimestamp=String(frame.media_timestamp);
          }
          this.after=frame.source_sequence;this.onFrame({buffering:!!this.controller&&!this.controller.frames.presented});
        }finally{bitmap.close();}
      }
      if(data.metadata&&this.controller&&this.controller.receiveMetadata(data.metadata,{transport:"websocket"}))
        this.revision=data.metadata.metadata_revision;
    }
    async pull(){
      if(this.closed)return;
      if(typeof document!=="undefined"&&document.hidden){this.timer=setTimeout(()=>this.pull(),250);return;}
      const started=performance.now(),abort=new AbortController();this.abort=abort;
      let timeout,reader,retry=0;
      const touch=()=>{clearTimeout(timeout);timeout=setTimeout(()=>abort.abort(),5000);};touch();
      try{
        const query=new URLSearchParams({after:this.after,connection_id:this.connectionId,
          revision:this.revision,stream_session_id:this.session,continuous:"true"});
        const response=await fetch(this.url+"?"+query,{cache:"no-store",signal:abort.signal});
        if(!response.ok)throw new Error("display unavailable");
        if(response.headers?.get("content-type")?.split(";")[0]==="application/vnd.goodvision.display-stream"){
          this.streamConnections++;if(this.controller)this.controller.stats.network_requests++;
          reader=response.body.getReader();const decoder=new DisplayStreamDecoder();
          while(!this.closed){
            const received=await reader.read();if(received.done){decoder.finish();break;}touch();
            for(const data of decoder.feed(received.value))this.enqueue(data);
            if(this.controller)this.controller.stats.network_bytes_received+=received.value.byteLength;
          }
          if(!this.closed)retry=100;
        }else{
          // Compatibility for older servers. New servers use the independent
          // stream above; the finite transport never has to be deployed first.
          const bytes=await response.arrayBuffer(),downloadSeconds=performance.now()-started;
          if(!this.closed){await this.accept(decodeDisplayBatch(bytes));
            if(this.controller){this.controller.stats.network_requests++;
              this.controller.stats.network_bytes_received+=bytes.byteLength;
              this.controller.noteTiming("network_download",downloadSeconds);}}
        }
      }catch(error){if(!this.closed){if(this.controller)this.controller.stats.network_failures++;
        retry=250;this.onFailure("实时画面暂不可用，正在重新获取。");}}
      finally{
        clearTimeout(timeout);if(reader)await reader.cancel().catch(()=>{});
        if(this.abort===abort)this.abort=null;
        if(!this.closed)this.timer=setTimeout(()=>this.pull(),Math.max(retry,1000/30-(performance.now()-started)));
      }
    }
    stop(){this.closed=true;this.acceptEpoch++;this.queue=[];this.queueBytes=this.queueFrames=0;
      clearTimeout(this.timer);this.abort?.abort();this.controller?.stop();this.controller=null;}
  }

  function eventObservationSummary(report){
    if(!report||report.state!=="completed")return "等待本次事件判断";
    const labels={fight:"打架",smoking:"吸烟",fall:"跌倒",weapon:"持械",ppe_violation:"防护装备缺失",ground_litter:"地面垃圾"};
    const judgments=new Map((report.judgments||[]).map(value=>[value.event_label,value]));
    const parts=Object.entries(labels).filter(([key])=>judgments.has(key)).map(([key,label])=>{
      const value=judgments.get(key),observed=Object.hasOwn(value,"observed_detected")?value.observed_detected:value.detected;
      let result=observed===true?(report.type==="vlm_report"?"发现":value.confirmation_state==="confirmed"?"已确认":"发现，待确认"):observed===false?"未发现":"待判断";
      if(observed===true&&["unavailable","partial"].includes(value.localization_state))result+="，定位待确认";
      return label+"："+result;
    });
    return parts.length?parts.join(" · "):"本次返回尚无可用事件判断";
  }

  // A same-origin diagnostic view reuses the already decoded and annotated
  // fleet canvas. It owns no stream, decoder, motion worker or telemetry slot.
  class SharedController {
    constructor({canvas,source,onStatus=()=>{},requestFrame=callback=>requestAnimationFrame(callback),cancelFrame=id=>cancelAnimationFrame(id)}){
      Object.assign(this,{canvas,source,onStatus,requestFrame,cancelFrame});
      this.closed=true;this.lastKey=null;this.frameId=null;this.lastStatus=0;
    }
    start(){if(!this.closed)return;this.closed=false;this.tick();}
    tick(){
      if(this.closed)return;
      const value=this.source(),media=value?.media,stats=value?.stats;
      const key=String(stats?.display_instance_id||"")+":"+String(media?.dataset?.frameSequence||"");
      const width=media?.naturalWidth||media?.width,height=media?.naturalHeight||media?.height;
      if(width&&height&&key!==this.lastKey){
        if(this.canvas.width!==width)this.canvas.width=width;
        if(this.canvas.height!==height)this.canvas.height=height;
        this.canvas.getContext("2d").drawImage(media,0,0,width,height);
        for(const name of ["frameSequence","observedAt","mediaTimestamp","receivedAt"]){
          if(media.dataset?.[name]!==undefined)this.canvas.dataset[name]=media.dataset[name];
        }
        this.lastKey=key;
      }
      if(stats&&Date.now()-this.lastStatus>=1000){this.lastStatus=Date.now();this.onStatus(stats);}
      this.frameId=this.requestFrame(()=>this.tick());
    }
    stop(){this.closed=true;if(this.frameId!==null)this.cancelFrame(this.frameId);this.frameId=null;}
  }

  return {FrameTimeline,ContourHistory,DisplayIdentities,DisplayStreamDecoder,estimateMotion,warpPolygons,packCameraFrame,sameRun,drawSubjectLabel,
    BrowserController,NetworkController,SharedController,eventObservationSummary,decodeDisplayBatch};
});
