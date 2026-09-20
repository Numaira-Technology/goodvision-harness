/* Raster-mask metrics shared by the real-browser replay and deterministic tests. */
(function(root,factory){
  const api=factory();
  if(typeof module==="object"&&module.exports)module.exports=api;
  else root.ContourQualityMetrics=api;
})(typeof globalThis!=="undefined"?globalThis:this,function(){
  "use strict";
  function percentile(values,fraction){
    if(!values.length)return null;
    const sorted=values.slice().sort((a,b)=>a-b);
    return sorted[Math.ceil((sorted.length-1)*fraction)];
  }
  function summarizeWorkerCompute(records){
    const groups=new Map();
    for(const record of records){
      if(!Number.isFinite(record.ms)||record.ms<0)throw new Error("Invalid Worker compute duration");
      const operation=record.operation==="present"&&record.expired?"present_expired":record.operation;
      if(!groups.has(operation))groups.set(operation,[]);groups.get(operation).push(record.ms);
    }
    return Object.fromEntries([...groups].map(([operation,values])=>[operation,{
      count:values.length,total_ms:values.reduce((sum,n)=>sum+n,0),
      mean_ms:values.reduce((sum,n)=>sum+n,0)/values.length,p95_ms:percentile(values,.95),max_ms:Math.max(...values)
    }]));
  }
  function summarizeAnchorCompute(records){
    const seen=new Set(),groups=[];
    for(const record of records){
      if(record.operation!=="present"||record.expired||!Number.isFinite(record.anchor_sequence))continue;
      groups.push({...record,operation:seen.has(record.anchor_sequence)?"same_anchor_followup":"first_effective_anchor"});
      seen.add(record.anchor_sequence);
    }
    return summarizeWorkerCompute(groups);
  }
  function sampleAnchorsAtCadence(anchors,hz=1){
    if(!Number.isFinite(hz)||hz<=0)throw new Error("Invalid anchor cadence");
    const sorted=anchors.slice().sort((a,b)=>a.media_timestamp-b.media_timestamp),selected=[];
    if(!sorted.length)return selected;
    let next=sorted[0].media_timestamp;const step=1/hz;
    for(const anchor of sorted)if(anchor.media_timestamp+1e-6>=next){
      selected.push(anchor);do{next+=step;}while(next<=anchor.media_timestamp+1e-6);
    }
    return selected;
  }
  function bufferedReplayEvents(sources,anchors,modelDelay,displayDelay){
    if(!Number.isFinite(modelDelay)||modelDelay<0||!Number.isFinite(displayDelay)||displayDelay<0)throw new Error("Invalid replay delay");
    return [
      ...sources.map(frame=>({type:"capture",time:frame.timestamp,order:0,frame})),
      ...anchors.map(anchor=>{
        const arrival=anchor.availability_clock_seconds??anchor.media_timestamp+modelDelay;
        if(!Number.isFinite(arrival)||arrival<anchor.media_timestamp)throw new Error("SAM availability precedes its source capture");
        return {type:"deliver",time:arrival,order:1,anchors:[anchor]};
      }),
      ...sources.map(frame=>({type:"present",time:frame.timestamp+displayDelay,order:2,frame}))
    ].sort((a,b)=>a.time-b.time||a.order-b.order);
  }
  function boundary(mask,width,height){
    const out=[];
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=y*width+x;
      if(mask[i]&&(x===0||y===0||x===width-1||y===height-1||
        !mask[i-1]||!mask[i+1]||!mask[i-width]||!mask[i+width]))out.push(i);
    }
    return out;
  }
  function compareContourLines(segments,reference,width,height){
    const pixels=new Set();
    for(const segment of segments||[])for(let i=1;i<segment.length;i++){
      const [ax,ay]=segment[i-1],[bx,by]=segment[i],dx=(bx-ax)*(width-1),dy=(by-ay)*(height-1),steps=Math.max(1,Math.ceil(Math.hypot(dx,dy)*2));
      for(let at=0;at<=steps;at++){
        const x=Math.max(0,Math.min(width-1,Math.round(ax*(width-1)+dx*at/steps))),
          y=Math.max(0,Math.min(height-1,Math.round(ay*(height-1)+dy*at/steps)));
        pixels.add(y*width+x);
      }
    }
    const predicted=[...pixels],expected=boundary(reference,width,height),toReference=distances(expected,width,height),toPredicted=distances(predicted,width,height),
      error=predicted.map(at=>toReference[at]),missing=expected.map(at=>toPredicted[at]);
    const fraction=(values,limit)=>values.length?values.filter(value=>value<=limit).length/values.length:null;
    return {line_pixels:predicted.length,reference_boundary_pixels:expected.length,
      line_precision_3px:fraction(error,3),line_precision_6px:fraction(error,6),
      boundary_recall_3px:fraction(missing,3),boundary_recall_6px:fraction(missing,6),
      line_boundary_mean_px:error.length?error.reduce((sum,value)=>sum+value,0)/error.length:null,
      line_boundary_p95_px:percentile(error,.95)};
  }
  // Eight-neighbour chamfer approximation, in evaluation-raster pixels.
  function distances(points,width,height){
    const out=new Float32Array(width*height).fill(Infinity),diagonal=Math.SQRT2;
    for(const i of points)out[i]=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=y*width+x;
      if(x)out[i]=Math.min(out[i],out[i-1]+1);
      if(y){out[i]=Math.min(out[i],out[i-width]+1);
        if(x)out[i]=Math.min(out[i],out[i-width-1]+diagonal);
        if(x+1<width)out[i]=Math.min(out[i],out[i-width+1]+diagonal);}
    }
    for(let y=height-1;y>=0;y--)for(let x=width-1;x>=0;x--){
      const i=y*width+x;
      if(x+1<width)out[i]=Math.min(out[i],out[i+1]+1);
      if(y+1<height){out[i]=Math.min(out[i],out[i+width]+1);
        if(x)out[i]=Math.min(out[i],out[i+width-1]+diagonal);
        if(x+1<width)out[i]=Math.min(out[i],out[i+width+1]+diagonal);}
    }
    return out;
  }
  function compareMasks(predicted,reference,width,height){
    if(predicted.length!==width*height||reference.length!==width*height)throw new Error("invalid raster size");
    let a=0,b=0,intersection=0,ax=0,ay=0,bx=0,by=0,minY=height,maxY=-1;
    for(let i=0;i<reference.length;i++){
      const x=i%width,y=Math.floor(i/width);
      if(predicted[i]){a++;ax+=x;ay+=y;}
      if(reference[i]){b++;bx+=x;by+=y;minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
      if(predicted[i]&&reference[i])intersection++;
    }
    const partReference=[0,0,0],partIntersection=[0,0,0];
    if(b)for(let i=0;i<reference.length;i++)if(reference[i]){
      const part=Math.min(2,Math.floor(3*(Math.floor(i/width)-minY)/(maxY-minY+1)));
      partReference[part]++;if(predicted[i])partIntersection[part]++;
    }
    const symmetric=[];
    if(a&&b){
      const pa=boundary(predicted,width,height),pb=boundary(reference,width,height);
      const da=distances(pa,width,height),db=distances(pb,width,height);
      for(const i of pa)symmetric.push(db[i]);for(const i of pb)symmetric.push(da[i]);
    }
    return {predicted_pixels:a,reference_pixels:b,intersection_pixels:intersection,
      mask_iou:a+b?intersection/(a+b-intersection):null,
      reference_recall:b?intersection/b:null,prediction_precision:a?intersection/a:null,
      area_ratio:b?a/b:null,centroid_offset_px:a&&b?Math.hypot(ax/a-bx/b,ay/a-by/b):null,
      boundary_mean_chamfer_px:symmetric.length?symmetric.reduce((sum,n)=>sum+n,0)/symmetric.length:null,
      boundary_p95_chamfer_px:percentile(symmetric,.95),
      reference_vertical_thirds_recall:partReference.map((n,i)=>n?partIntersection[i]/n:null)};
  }
  function bestIoUMatching(matrix){
    if(!Array.isArray(matrix)||matrix.length>64)throw new Error("invalid IoU matrix");
    const rows=matrix.length,cols=rows&&Array.isArray(matrix[0])?matrix[0].length:0;
    if(cols>64)throw new Error("invalid IoU matrix");
    for(const row of matrix){
      if(!Array.isArray(row)||row.length!==cols)throw new Error("invalid IoU matrix");
      for(const value of row)if(!Number.isFinite(value)||value<0||value>1)
        throw new Error("invalid IoU matrix");
    }
    if(!rows||!cols)return [];
    // Hungarian assignment minimizes negative IoU in O(n^3). Zero-valued dummy
    // rows/columns allow rectangular inputs; zero-IoU assignments are omitted.
    const n=Math.max(rows,cols),u=new Float64Array(n+1),v=new Float64Array(n+1),
      assignedRow=new Int32Array(n+1),previousColumn=new Int32Array(n+1);
    for(let row=1;row<=n;row++){
      assignedRow[0]=row;
      let column=0;
      const minimum=new Float64Array(n+1).fill(Infinity),used=new Uint8Array(n+1);
      do{
        used[column]=1;
        const currentRow=assignedRow[column];
        let delta=Infinity,nextColumn=0;
        for(let candidate=1;candidate<=n;candidate++)if(!used[candidate]){
          const weight=currentRow<=rows&&candidate<=cols?matrix[currentRow-1][candidate-1]:0,
            cost=-weight-u[currentRow]-v[candidate];
          if(cost<minimum[candidate]){minimum[candidate]=cost;previousColumn[candidate]=column;}
          if(minimum[candidate]<delta){delta=minimum[candidate];nextColumn=candidate;}
        }
        for(let candidate=0;candidate<=n;candidate++){
          if(used[candidate]){u[assignedRow[candidate]]+=delta;v[candidate]-=delta;}
          else minimum[candidate]-=delta;
        }
        column=nextColumn;
      }while(assignedRow[column]!==0);
      do{
        const prior=previousColumn[column];
        assignedRow[column]=assignedRow[prior];column=prior;
      }while(column!==0);
    }
    const matches=[];
    for(let column=1;column<=cols;column++){
      const row=assignedRow[column];
      if(row<=rows&&matrix[row-1][column-1]>0)matches.push({reference_index:row-1,
        predicted_index:column-1,iou:matrix[row-1][column-1]});
    }
    return matches.sort((a,b)=>a.reference_index-b.reference_index);
  }
  return {compareMasks,compareContourLines,percentile,bestIoUMatching,summarizeWorkerCompute,summarizeAnchorCompute,sampleAnchorsAtCadence,bufferedReplayEvents};
});
