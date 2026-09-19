/* Experimental display-only dense optical flow. No result is analysis evidence. */
(function(root,factory){
  "use strict";
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  else root.GoodvisionDenseMotion=api;
})(typeof globalThis!=="undefined"?globalThis:this,function(){
  "use strict";
  const finite=Number.isFinite;
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
  // tracking normalizes native pixel centers as (pixel + .5) / image size.
  // Canvas-resized motion pixels use the same center convention; align_corners
  // scaling by size-1 shifts small masks by up to a whole rounded pixel.
  const pixelCenter=(coordinate,size)=>Math.round(clamp(coordinate*size-.5,0,size-1));
  const normalizedCenter=(coordinate,size)=>clamp((coordinate+.5)/size,0,1);
  function sample(pixels,width,height,x,y,channels=1,channel=0){
    if(!finite(x)||!finite(y)||x<0||y<0||x>width-1||y>height-1)return null;
    const ix=Math.floor(x),iy=Math.floor(y),nx=Math.min(ix+1,width-1),ny=Math.min(iy+1,height-1),
      fx=x-ix,fy=y-iy;
    return pixels[(iy*width+ix)*channels+channel]*(1-fx)*(1-fy)+pixels[(iy*width+nx)*channels+channel]*fx*(1-fy)+
      pixels[(ny*width+ix)*channels+channel]*(1-fx)*fy+pixels[(ny*width+nx)*channels+channel]*fx*fy;
  }
  function validPolygons(polygons){
    let count=0;
    return Array.isArray(polygons)&&polygons.length>0&&polygons.length<=16&&polygons.every(polygon=>
      Array.isArray(polygon)&&polygon.length>=3&&polygon.length<=128&&(count+=polygon.length)<=1024&&
      polygon.every(point=>Array.isArray(point)&&point.length===2&&point.every(v=>finite(v)&&v>=0&&v<=1)));
  }
  function dispose(...values){for(const value of values)if(value&&typeof value.delete==="function")value.delete();}
  function now(){return typeof performance!=="undefined"?performance.now():Date.now();}
  function createDenseMotionAdapter(cv,options={}){
    if(!cv||typeof cv.Mat!=="function"||typeof cv.calcOpticalFlowFarneback!=="function"||typeof cv.remap!=="function")
      throw new Error("Display motion module is not initialized");
    const config={levels:4,windowSize:21,iterations:4,polynomialSize:7,polynomialSigma:1.5,
      maxCycleError:1.5,maxAppearanceError:32,minReliableFraction:.65,maxCropExtension:8,
      maxMeanAppearanceError:25,minAreaRatio:.3,maxAreaRatio:2.5,...options};
    const masks=new WeakMap(),presentations=new WeakMap(),blocks=new WeakMap(),reversed=new WeakMap(),blockSize=8;
    let prepared=null;
    function anchorAppearance(map,baseImage,motion){
      if(map===motion.sourceMap&&baseImage===motion.matchingPrevious)return motion.appearanceError;
      if(!motion.composedAppearance)motion.composedAppearance=new WeakMap();
      if(motion.composedAppearance.has(map))return motion.composedAppearance.get(map);
      const {width,height}=motion,error=new Uint8Array(width*height);
      for(let at=0;at<error.length;at++){
        const value=sample(baseImage,width,height,map[at*2],map[at*2+1]);
        error[at]=value===null?motion.appearanceError[at]:Math.max(motion.appearanceError[at],
          Math.min(254,Math.round(Math.abs(value-motion.matchingCurrent[at]))));
      }
      motion.composedAppearance.set(map,error);return error;
    }
    function combineEvidence(previous,current){
      if(!current.chainEvidence)current.chainEvidence=new WeakMap();
      if(current.chainEvidence.has(previous.sourceMap))return current.chainEvidence.get(previous.sourceMap);
      const {width,height}=current,sourceMap=new Float32Array(current.sourceMap.length),
        cycleError=new Uint8Array(width*height),appearanceError=new Uint8Array(width*height);
      for(let at=0;at<cycleError.length;at++){
        const x=current.sourceMap[at*2],y=current.sourceMap[at*2+1],
          cx=clamp(x,0,width-1),cy=clamp(y,0,height-1),ix=Math.floor(cx),iy=Math.floor(cy),
          nx=Math.min(ix+1,width-1),ny=Math.min(iy+1,height-1),fx=cx-ix,fy=cy-iy,
          a=iy*width+ix,b=iy*width+nx,c=ny*width+ix,d=ny*width+nx,
          wa=(1-fx)*(1-fy),wb=fx*(1-fy),wc=(1-fx)*fy,wd=fx*fy,prior=previous.sourceMap;
        // Coordinates and both confidence planes share the same four samples.
        // Avoid four independent bilinear lookups for every pixel/chain step.
        sourceMap[at*2]=prior[a*2]*wa+prior[b*2]*wb+prior[c*2]*wc+prior[d*2]*wd+x-cx;
        sourceMap[at*2+1]=prior[a*2+1]*wa+prior[b*2+1]*wb+prior[c*2+1]*wc+prior[d*2+1]*wd+y-cy;
        if(x!==cx||y!==cy){cycleError[at]=255;appearanceError[at]=255;continue;}
        const pc=previous.cycleError,pa=previous.appearanceError;
        cycleError[at]=Math.max(current.cycleError[at],Math.round(pc[a]*wa+pc[b]*wb+pc[c]*wc+pc[d]*wd));
        appearanceError[at]=Math.max(current.appearanceError[at],Math.round(pa[a]*wa+pa[b]*wb+pa[c]*wc+pa[d]*wd));
      }
      const combined={kind:"dense_flow",width,height,sourceMap,cycleError,appearanceError,
        matchingPrevious:previous.matchingPrevious,matchingCurrent:current.matchingCurrent,
        previous:previous.previous,current:current.current};
      current.chainEvidence.set(previous.sourceMap,combined);return combined;
    }
    function continuous(previous,current){
      return current&&previous&&current.kind==="dense_flow"&&previous.kind==="dense_flow"&&
        current.width===previous.width&&current.height===previous.height&&
        Number.isSafeInteger(current.chain_index)&&current.chain_index===previous.chain_index+1&&
        (!previous.current||!current.previous||previous.current===current.previous);
    }
    function prepareMotion(motion){
      if(!motion||motion.kind!=="dense_flow"||!Number.isSafeInteger(motion.chain_index)||motion.chain_index<0){
        prepared=null;return;
      }
      if(motion.chain_index%blockSize===0){prepared={motions:[motion],combined:motion};return;}
      if(!prepared||!continuous(prepared.motions.at(-1),motion)){prepared=null;return;}
      prepared.combined=combineEvidence(prepared.combined,motion);prepared.motions.push(motion);
      if(prepared.motions.length===blockSize){
        blocks.set(prepared.motions[0],prepared);prepared=null;
      }
    }
    function blockAt(motions,offset){
      const first=motions[offset];
      if(!Number.isSafeInteger(first.chain_index)||first.chain_index<0||first.chain_index%blockSize!==0||
        offset+blockSize>motions.length)return null;
      for(let n=1;n<blockSize;n++)if(!continuous(motions[offset+n-1],motions[offset+n]))return null;
      let block=blocks.get(first);
      if(block&&block.motions.every((motion,n)=>motion===motions[offset+n]))return block.combined;
      const list=motions.slice(offset,offset+blockSize);let combined=first;
      for(let n=1;n<blockSize;n++)combined=combineEvidence(combined,list[n]);
      block={motions:list,combined};blocks.set(first,block);return combined;
    }
    function estimateMotion(previous,current,width,height){
      if(!Number.isInteger(width)||!Number.isInteger(height)||width<16||height<16||width>640||height>480||
        previous.length!==width*height||current.length!==width*height)throw new Error("invalid dense motion pixels");
      let a,b,forward,backward,softA,softB;
      const started=now();
      try{
        a=cv.matFromArray(height,width,cv.CV_8UC1,previous);
        b=cv.matFromArray(height,width,cv.CV_8UC1,current);
        forward=new cv.Mat();backward=new cv.Mat();
        cv.calcOpticalFlowFarneback(a,b,forward,.5,config.levels,config.windowSize,
          config.iterations,config.polynomialSize,config.polynomialSigma,0);
        cv.calcOpticalFlowFarneback(b,a,backward,.5,config.levels,config.windowSize,
          config.iterations,config.polynomialSize,config.polynomialSigma,0);
        softA=new cv.Mat();softB=new cv.Mat();
        cv.GaussianBlur(a,softA,new cv.Size(3,3),0,0,cv.BORDER_REPLICATE);
        cv.GaussianBlur(b,softB,new cv.Size(3,3),0,0,cv.BORDER_REPLICATE);
        const forwardData=forward.data32F,forwardMap=new Float32Array(forwardData),
          sourceMap=new Float32Array(backward.data32F),cycleError=new Uint8Array(width*height),
          appearanceError=new Uint8Array(width*height),before=new Uint8Array(softA.data),after=new Uint8Array(softB.data);
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          const at=y*width+x,dx=sourceMap[at*2],dy=sourceMap[at*2+1],sx=x+dx,sy=y+dy;
          sourceMap[at*2]=sx;sourceMap[at*2+1]=sy;
          // The opposite direction was already computed for cycle validation.
          // Retain its coordinates so a later tracking source can describe an older
          // buffered target without another optical-flow model invocation.
          forwardMap[at*2]+=x;forwardMap[at*2+1]+=y;
          const fx=sample(forwardData,width,height,sx,sy,2,0),fy=sample(forwardData,width,height,sx,sy,2,1),
            old=sample(before,width,height,sx,sy);
          cycleError[at]=fx===null||fy===null?255:Math.min(254,Math.round(Math.hypot(dx+fx,dy+fy)*32));
          appearanceError[at]=old===null?255:Math.min(254,Math.round(Math.abs(old-after[at])));
        }
        return {kind:"dense_flow",width,height,sourceMap,forwardMap,cycleError,appearanceError,
          previous,current,matchingPrevious:before,matchingCurrent:after,
          compute_ms:now()-started,last_warp_rejection:null,last_warp_diagnostics:null};
      }finally{dispose(a,b,forward,backward,softA,softB);}
    }
    function reverseMotion(motion){
      if(!motion||motion.kind!=="dense_flow"||
        !motion.forwardMap||motion.forwardMap.length!==motion.width*motion.height*2||
        !motion.sourceMap||motion.sourceMap.length!==motion.forwardMap.length)
        throw new Error("invalid reverse dense motion");
      if(reversed.has(motion))return reversed.get(motion);
      const {width,height}=motion,started=now(),sourceMap=motion.forwardMap,
        cycleError=new Uint8Array(width*height),appearanceError=new Uint8Array(width*height);
      for(let y=0;y<height;y++)for(let x=0;x<width;x++){
        const at=y*width+x,sx=sourceMap[at*2],sy=sourceMap[at*2+1],
          backX=sample(motion.sourceMap,width,height,sx,sy,2,0),
          backY=sample(motion.sourceMap,width,height,sx,sy,2,1),
          source=sample(motion.matchingCurrent,width,height,sx,sy);
        // These errors live on the earlier target's grid. Reusing the old
        // later-target errors or negating its flow would be incorrect.
        cycleError[at]=backX===null||backY===null?255:Math.min(254,Math.round(Math.hypot(backX-x,backY-y)*32));
        appearanceError[at]=source===null?255:Math.min(254,Math.round(Math.abs(source-motion.matchingPrevious[at])));
      }
      const result={kind:"dense_flow",width,height,sourceMap,forwardMap:motion.sourceMap,cycleError,appearanceError,
        previous:motion.current,current:motion.previous,
        matchingPrevious:motion.matchingCurrent,matchingCurrent:motion.matchingPrevious,
        direction:"reverse",compute_ms:now()-started,last_warp_rejection:null,last_warp_diagnostics:null};
      // Reverse chains traverse the real pixel references in reverse source
      // order. They do not inherit forward-only block indices or cache entries.
      reversed.set(motion,result);reversed.set(result,motion);return result;
    }
    function rasterize(polygons,width,height){
      const cached=masks.get(polygons);
      if(cached&&cached.width===width&&cached.height===height)
        return cv.matFromArray(height,width,cv.CV_8UC1,cached.pixels);
      const mask=cv.Mat.zeros(height,width,cv.CV_8UC1),contours=new cv.MatVector();
      try{
        for(const polygon of polygons){
          const coordinates=polygon.flatMap(([x,y])=>[pixelCenter(x,width),pixelCenter(y,height)]);
          const points=cv.matFromArray(polygon.length,1,cv.CV_32SC2,coordinates);
          try{contours.push_back(points);}finally{points.delete();}
        }
        cv.fillPoly(mask,contours,new cv.Scalar(255),cv.LINE_8);
        return mask;
      }catch(error){mask.delete();throw error;}finally{contours.delete();}
    }
    function cropSupported(mask,motion,x,y,sx,sy){
      const {width,height,matchingPrevious:before,matchingCurrent:current}=motion,
        px=clamp(sx,0,width-1),py=clamp(sy,0,height-1),distance=Math.hypot(px-sx,py-sy);
      if(distance>config.maxCropExtension||sample(mask,width,height,px,py)<192)return false;
      // The source mask touched the actual camera crop. Look for its appearance
      // just inside that crop, rather than copying a cropped edge into the torso.
      let count=0,sum=0,squares=0;
      for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
        const xx=Math.round(px)+dx,yy=Math.round(py)+dy;
        if(xx<0||yy<0||xx>=width||yy>=height||mask[yy*width+xx]<192)continue;
        const value=before[yy*width+xx];count++;sum+=value;squares+=value*value;
      }
      if(count<4)return false;
      const mean=sum/count,deviation=Math.sqrt(Math.max(0,squares/count-mean*mean));
      return Math.abs(current[y*width+x]-mean)<=Math.max(18,2*deviation);
    }
    function contoursOf(mask,width,height,approximation=null){
      const contours=new cv.MatVector(),hierarchy=new cv.Mat(),output=[];
      try{
        cv.findContours(mask,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);
        for(let i=0;i<contours.size();i++){
          const contour=contours.get(i);let approx;
          try{
            // A real source component can collapse under motion-grid sampling.
            // Omitting it while returning the other components would claim a
            // complete silhouette with a missing hand/head. Fail explicitly.
            if(cv.contourArea(contour)<2){
              if(!approximation)return null;
              const rect=cv.boundingRect(contour);let pixels=0;
              for(let y=rect.y;y<rect.y+rect.height;y++)for(let x=rect.x;x<rect.x+rect.width;x++)
                if(mask.data[y*width+x])pixels++;
              // Crop clipping can leave a few isolated raster specks. Only
              // explicit approximate display may omit at most four pixels,
              // never a long thin limb or a whole real source component.
              if(pixels>4)return null;
              approximation.discarded_fragment_pixels=(approximation.discarded_fragment_pixels||0)+pixels;
              if(approximation.discarded_fragment_pixels>4)return null;
              continue;
            }
            approx=new cv.Mat();let epsilon=.5;
            do{cv.approxPolyDP(contour,approx,epsilon,true);epsilon*=1.5;}while(approx.rows>128&&epsilon<8);
            if(approx.rows<3||approx.rows>128)return null;
            const polygon=[];
            for(let k=0;k<approx.data32S.length;k+=2)polygon.push([
              normalizedCenter(approx.data32S[k],width),normalizedCenter(approx.data32S[k+1],height)]);
            output.push(polygon);
          }finally{dispose(contour,approx);}
        }
        return validPolygons(output)&&(!approximation||output.length>=approximation.source_components)?output:null;
      }finally{dispose(contours,hierarchy);}
    }
    function openSegments(parts,width,height){
      const output=[];
      for(const {points,trusted} of parts){
        const count=trusted.length;if(count<6)continue;
        // Leave a two-pixel guard on either side of uncertain contour samples.
        // These are original external boundary arcs, never a newly cut mask edge.
        const good=trusted.map((value,i)=>value&&[-2,-1,1,2].every(d=>trusted[(i+d+count)%count]));
        const firstBad=good.indexOf(false),runs=[];
        if(firstBad===-1)runs.push(Array.from({length:count+1},(_,i)=>points[i%count]));
        else{
          let run=[];
          for(let n=1;n<=count;n++){
            const at=(firstBad+n)%count;
            if(good[at])run.push(points[at]);
            else if(run.length){runs.push(run);run=[];}
          }
          if(run.length)runs.push(run);
        }
        for(const run of runs){
          if(run.length<6)continue;
          let source,approx;
          try{
            source=cv.matFromArray(run.length,1,cv.CV_32SC2,run.flat());approx=new cv.Mat();
            let epsilon=.5;
            do{cv.approxPolyDP(source,approx,epsilon,false);epsilon*=1.5;}while(approx.rows>128&&epsilon<8);
            if(approx.rows<2||approx.rows>128)continue;
            const data=approx.data32S,segment=[];
            for(let i=0;i<data.length;i+=2)segment.push([normalizedCenter(data[i],width),normalizedCenter(data[i+1],height)]);
            output.push(segment);
          }finally{dispose(source,approx);}
        }
      }
      return output.length<=32&&output.reduce((sum,segment)=>sum+segment.length,0)<=1024?output:[];
    }
    function localVisibility(binary,motion,support,diagnostics,allowPartial=false,initialReason=null,allowApproximate=false){
      const {width,height}=motion,labels=new cv.Mat(),contours=new cv.MatVector(),hierarchy=new cv.Mat();
      try{
        const count=cv.connectedComponents(binary,labels,8,cv.CV_32S),labelData=labels.data32S,
          components=Array.from({length:count},()=>({pixels:0,reliable:0,error:0}));
        for(let at=0;at<labelData.length;at++){
          const label=labelData[at];if(!label)continue;
          const component=components[label];component.pixels++;
          if(support[at]===1||support[at]===2)component.reliable++;
          if(support[at]!==2&&support[at]!==3)component.error+=motion.appearanceError[at];
        }
        diagnostics.components=components.slice(1).map(c=>({pixels:c.pixels,
          reliable_fraction:c.reliable/c.pixels,mean_appearance_error:c.error/c.pixels}));
        const failedComponents=new Set();
        diagnostics.components.forEach((c,i)=>{if(c.pixels>=4&&(c.reliable_fraction<config.minReliableFraction||
          c.mean_appearance_error>config.maxMeanAppearanceError))failedComponents.add(i+1);});
        // A crop warning must never hide a disconnected head/limb whose
        // appearance failed. Short display interpolation only tolerates local
        // boundary uncertainty; every component still needs real support.
        if(allowApproximate&&failedComponents.size)return {reason:"component_appearance"};
        let reason=initialReason||(failedComponents.size?"component_appearance":null);
        if(reason&&!allowPartial&&!allowApproximate)return {reason};
        cv.findContours(binary,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE);
        const mask=binary.data,parts=[];
        for(let part=0;part<contours.size();part++){
          const contour=contours.get(part);
          try{
            const points=contour.data32S,boundary=[],boundaryEvidence=[],trusted=[],coordinates=[];
            if(failedComponents.has(labelData[points[1]*width+points[0]]))continue;
            for(let i=0;i<points.length;i+=2){
              const x=points[i],y=points[i+1];let pixels=0,reliable=0,error=0,unknown=support[y*width+x]===3?1:0,gx=0,gy=0;
              const value=(px,py)=>px>=0&&py>=0&&px<width&&py<height?mask[py*width+px]:0;
              for(let offset=-1;offset<=1;offset++){
                gx+=value(x+1,y+offset)-value(x-1,y+offset);
                gy+=value(x+offset,y+1)-value(x+offset,y-1);
              }
              const length=Math.hypot(gx,gy),nx=length?gx/length:0,ny=length?gy/length:0;
              let cx=x,cy=y;
              for(const distance of [2,1]){
                const px=Math.round(x+nx*distance),py=Math.round(y+ny*distance);
                if(value(px,py)){cx=px;cy=py;break;}
              }
              // Evidence comes just inside this contour, not from the large
              // torso or another disconnected part. Rasterized boundary pixels
              // mix foreground/background; sample inward before judging motion.
              for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
                const xx=cx+dx,yy=cy+dy;if(xx<0||yy<0||xx>=width||yy>=height)continue;
                const at=yy*width+xx;if(!mask[at])continue;
                pixels++;if(support[at]===1||support[at]===2)reliable++;
                if(support[at]===3)unknown++;else if(support[at]!==2)error+=motion.appearanceError[at];
              }
              boundary.push(unknown>0||pixels>=3&&(reliable/pixels<.6||error/pixels>config.maxMeanAppearanceError));
              boundaryEvidence.push({x,y,reliable_fraction:reliable/pixels,mean_appearance_error:error/pixels});
              if(allowPartial){coordinates.push([x,y]);trusted.push(!unknown&&pixels>=3&&reliable/pixels>=.6&&error/pixels<=config.maxMeanAppearanceError);}
            }
            if(allowPartial)parts.push({points:coordinates,trusted});
            if(boundary.length<6)continue;
            const window=Math.min(12,boundary.length);let bad=0;
            for(let i=0;i<window;i++)bad+=boundary[i]?1:0;
            for(let i=0;i<boundary.length;i++){
              if(bad/window>=.6){
                if(!diagnostics.boundary_failure)diagnostics.boundary_failure={...boundaryEvidence[i],bad_fraction:bad/window};
                reason=reason||"boundary_occluded";
                if(!allowPartial&&!allowApproximate)return {reason};
                break;
              }
              bad-=(boundary[i]?1:0);bad+=(boundary[(i+window)%boundary.length]?1:0);
            }
          }finally{contour.delete();}
        }
        return {reason,segments:reason&&allowPartial?openSegments(parts,width,height):[]};
      }finally{dispose(labels,contours,hierarchy);}
    }
    function warpPolygonsInternal(polygons,motion,allowPartial=false,allowApproximate=false){
      if(motion){motion.last_warp_rejection=null;motion.last_warp_diagnostics=null;}
      const reject=reason=>{if(motion)motion.last_warp_rejection=reason;return null;};
      if(!validPolygons(polygons)||!motion||motion.kind!=="dense_flow")return reject("invalid_input");
      if(presentations.get(polygons)?.approximate)return reject("invalid_input");
      let source,base,map,empty,warped,binary;
      const {width,height}=motion,started=now(),diagnostics={source_pixels:0,warped_pixels:0,reliable_pixels:0,
        crop_extended_pixels:0,crop_rejected_pixels:0,mean_appearance_error:null,reliable_fraction:null};
      motion.last_warp_diagnostics=diagnostics;
      try{
        source=rasterize(polygons,width,height);
        const cached=masks.get(polygons),basePixels=cached?.basePixels||new Uint8Array(source.data),
          baseImage=cached?.baseImage||motion.matchingPrevious,evidence=cached?.evidence?combineEvidence(cached.evidence,motion):motion,
          composed=evidence.sourceMap,appearanceError=anchorAppearance(composed,baseImage,evidence);
        base=cv.matFromArray(height,width,cv.CV_8UC1,basePixels);
        map=cv.matFromArray(height,width,cv.CV_32FC2,composed);empty=new cv.Mat();warped=new cv.Mat();
        // Compose coordinates, not masks. Every frame samples the same real
        // tracking anchor once, so repeated bilinear resampling cannot eat thin limbs.
        cv.remap(base,warped,map,empty,cv.INTER_LINEAR,cv.BORDER_REPLICATE,new cv.Scalar(0));
        let minX=width,maxX=0,minY=height,maxY=0,appearance=0;
        const sourceData=source.data,warpedData=warped.data,support=new Uint8Array(width*height);
        for(let at=0;at<sourceData.length;at++)if(sourceData[at]>=128)diagnostics.source_pixels++;
        const regions=Array.from({length:4},()=>({count:0,reliable:0}));
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          const at=y*width+x;if(warpedData[at]<128)continue;
          const sx=motion.sourceMap[at*2],sy=motion.sourceMap[at*2+1],outside=sx<0||sy<0||sx>width-1||sy>height-1;
          if(outside){
            if(!cropSupported(sourceData,motion,x,y,sx,sy)){
              diagnostics.crop_rejected_pixels++;
              // Approximate closed display uses the same clipped, observed
              // candidate as strict display. BORDER_REPLICATE outside the
              // old camera crop can otherwise grow a long phantom extension.
              if(!allowPartial){warpedData[at]=0;continue;}
              // Keep the original outer boundary geometry. Mark unobserved
              // extension unreliable instead of cutting a new closing edge.
              support[at]=3;
            }else{diagnostics.crop_extended_pixels++;support[at]=2;}
          }
          diagnostics.warped_pixels++;
          if(support[at]===2||!outside&&evidence.cycleError[at]/32<=config.maxCycleError&&appearanceError[at]<=config.maxAppearanceError){
            diagnostics.reliable_pixels++;if(!outside)support[at]=1;
          }
          if(!outside)appearance+=appearanceError[at];
          minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
        }
        if(!diagnostics.warped_pixels)return reject("component_visibility");
        diagnostics.mean_appearance_error=appearance/Math.max(1,diagnostics.warped_pixels-diagnostics.crop_extended_pixels-
          (allowPartial?diagnostics.crop_rejected_pixels:0));
        diagnostics.reliable_fraction=diagnostics.reliable_pixels/diagnostics.warped_pixels;
        const ratio=diagnostics.warped_pixels/Math.max(1,diagnostics.source_pixels);
        if(ratio<config.minAreaRatio||ratio>config.maxAreaRatio)return reject("invalid_geometry");
        if(!allowPartial&&!allowApproximate&&diagnostics.crop_rejected_pixels>Math.max(3,diagnostics.warped_pixels*.01))return reject("crop_continuation");
        if(diagnostics.mean_appearance_error>config.maxMeanAppearanceError)return reject("component_appearance");
        if(diagnostics.reliable_fraction<config.minReliableFraction)return reject("internal_support");
        for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++){
          const at=y*width+x;if(warpedData[at]<128)continue;
          const region=regions[(x>(minX+maxX)/2?1:0)+(y>(minY+maxY)/2?2:0)];region.count++;
          if(support[at]!==3&&evidence.cycleError[at]/32<=config.maxCycleError&&appearanceError[at]<=config.maxAppearanceError)region.reliable++;
        }
        if(regions.some(region=>region.count>=12&&region.reliable/region.count<.4))return reject("region_occluded");
        binary=new cv.Mat();cv.threshold(warped,binary,127,255,cv.THRESH_BINARY);
        const visibility=localVisibility(binary,{...evidence,appearanceError},support,diagnostics,allowPartial,
          (allowPartial||allowApproximate)&&diagnostics.crop_rejected_pixels?"crop_continuation":null,allowApproximate);
        if(visibility.reason&&(allowApproximate?!["crop_continuation","boundary_occluded"].includes(visibility.reason):
          !allowPartial||!visibility.segments?.length))return reject(visibility.reason);
        const output=contoursOf(binary,width,height,allowApproximate?
          Object.assign(diagnostics,{source_components:polygons.length}):null);
        if(!output)return reject("invalid_polygon");
        masks.set(output,{width,height,pixels:new Uint8Array(warped.data),basePixels,baseImage,sourceMap:composed,evidence});
        if(allowApproximate){
          // This closed candidate is presentation-only. It cannot seed a
          // subsequent strict propagation or count as strict quality success.
          presentations.set(output,{complete:false,approximate:true,polygons:output,open_segments:[],
            reason:visibility.reason||"boundary_occluded"});
          diagnostics.approximate_reason=visibility.reason||"boundary_occluded";
        }else if(visibility.reason){
          presentations.set(output,{complete:false,polygons:[],open_segments:visibility.segments,reason:visibility.reason});
          diagnostics.partial_reason=visibility.reason;diagnostics.open_segments=visibility.segments.length;
        }
        return output;
      }finally{
        diagnostics.compute_ms=now()-started;
        dispose(source,base,map,empty,warped,binary);
      }
    }
    function warpChainInternal(polygons,motions,allowPartial=false,allowApproximate=false){
      if(!Array.isArray(motions)||!motions.length)return validPolygons(polygons)?polygons:null;
      const last=motions[motions.length-1],started=now();
      if(!last)return null;
      last.last_warp_rejection=null;last.last_warp_diagnostics=null;
      let combined=motions[0];
      for(let i=0;i<motions.length;i++){
        const motion=motions[i];
        if(!motion||motion.kind!=="dense_flow"||motion.width!==combined.width||motion.height!==combined.height){
          last.last_warp_rejection="invalid_input";return null;
        }
        if(i&&(motions[i-1].current&&motion.previous&&motions[i-1].current!==motion.previous||
          Number.isSafeInteger(motions[i-1].chain_index)&&Number.isSafeInteger(motion.chain_index)&&
          motion.chain_index!==motions[i-1].chain_index+1)){
          last.last_warp_rejection="invalid_input";return null;
        }
      }
      let blocksUsed=0,combinations=0;combined=null;
      for(let i=0;i<motions.length;){
        const block=blockAt(motions,i),motion=block||motions[i];
        if(combined){combined=combineEvidence(combined,motion);combinations++;}else combined=motion;
        if(block){blocksUsed++;i+=blockSize;}else i++;
      }
      // Mask extraction is performed only at the final target. Confidence is
      // carried along the actual mapped path, never reset at an ambiguous step.
      const result=warpPolygonsInternal(polygons,combined,allowPartial,allowApproximate);
      last.last_warp_rejection=combined.last_warp_rejection;
      last.last_warp_diagnostics={...combined.last_warp_diagnostics,chain_steps:motions.length,
        chain_blocks:blocksUsed,chain_combinations:combinations,compute_ms:now()-started};
      return result;
    }
    function warpPolygons(polygons,motion){return warpPolygonsInternal(polygons,motion);}
    function warpChain(polygons,motions){return warpChainInternal(polygons,motions);}
    function warpPresentation(polygons,motion){return warpPolygonsInternal(polygons,motion,true);}
    function warpPresentationChain(polygons,motions){return warpChainInternal(polygons,motions,true);}
    function warpApproximateChain(polygons,motions){return warpChainInternal(polygons,motions,false,true);}
    function getPresentation(shape){
      if(!validPolygons(shape))return null;
      return presentations.get(shape)||{complete:true,polygons:shape,open_segments:[]};
    }
    return {estimateMotion,reverseMotion,warpPolygons,warpChain,warpPresentation,warpPresentationChain,warpApproximateChain,getPresentation,prepareMotion,config};
  }
  return {createDenseMotionAdapter};
});
