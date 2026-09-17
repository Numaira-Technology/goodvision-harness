/* Optional private diagnostic capture. Never installed by the dashboard itself. */
"use strict";
const fs=require("node:fs"),path=require("node:path");
function installMotionTraceCollector({maxBytes,maxSeconds,maxPendingBytes=8*1024**2}){
  const Original=globalThis.GoodvisionCameraDisplay.BrowserController,queue=[];
  const status={enabled:true,complete:true,reason:null,bytes:0,records:0};let pendingBytes=0,controller=null;
  function collect(record){
    if(!status.complete)return false;
    if(record.type==="motion_trace_stopped"){
      status.complete=false;status.reason=record.reason;return false;
    }
    let stored=record;
    if(record.type==="motion_trace_input"&&record.message.type==="frame"){
      const {pixels,...frame}=record.message.frame;
      stored={...record,message:{...record.message,frame:{...frame,
        gray_base64:btoa(String.fromCharCode(...pixels))}}};
    }
    const size=new TextEncoder().encode(JSON.stringify(stored)).byteLength;
    if(status.bytes+size>maxBytes||pendingBytes+size>maxPendingBytes){
      status.complete=false;status.reason=status.bytes+size>maxBytes?"collector_byte_limit":"collector_backpressure";
      return false;
    }
    queue.push(stored);status.bytes+=size;status.records++;pendingBytes+=size;return true;
  }
  globalThis.GoodvisionCameraDisplay.BrowserController=class extends Original{
    constructor(options){super({...options,onMotionTrace:collect,motionTraceOptions:{maxBytes,maxSeconds}});controller=this;}
  };
  globalThis.__drainMotionTrace=()=>{
    if(controller?.motionTraceFailure){status.complete=false;status.reason=controller.motionTraceFailure;}
    const records=queue.splice(0);pendingBytes=0;return {records,status:{...status}};
  };
  globalThis.__stopMotionTrace=reason=>{status.complete=false;status.reason=reason;};
}
function createMotionTraceWriter(directory,{maxBytes=128*1024**2}={}){
  fs.mkdirSync(directory,{recursive:true});
  const eventsFile=path.join(directory,"motion-trace.jsonl"),pixelsFile=path.join(directory,"motion-trace-gray.bin"),
    summaryFile=path.join(directory,"motion-trace-summary.json"),inputs=new Set(),outputs=new Set();
  fs.writeFileSync(eventsFile,"");fs.writeFileSync(pixelsFile,Buffer.alloc(0));
  let offset=0,bytes=0,count=0,stopped=null;
  return {append(batch){
    if(stopped)return stopped;
    for(const record of batch.records){
      let stored=record,pixels=null;
      if(record.type==="motion_trace_input"&&record.message.type==="frame"){
        const {gray_base64,...frame}=record.message.frame;pixels=Buffer.from(gray_base64,"base64");
        if(pixels.length!==frame.width*frame.height){stopped="invalid_pixel_length";return stopped;}
        stored={...record,message:{...record.message,frame:{...frame,gray_offset:offset,gray_length:pixels.length}}};
      }
      const line=JSON.stringify(stored)+"\n",size=Buffer.byteLength(line)+(pixels?.length||0);
      if(bytes+size>maxBytes){stopped="writer_byte_limit";return stopped;}
      if(pixels){fs.appendFileSync(pixelsFile,pixels);offset+=pixels.length;}
      fs.appendFileSync(eventsFile,line);bytes+=size;count++;
      if(record.type==="motion_trace_input")inputs.add(record.message.trace_id);
      if(record.type==="motion_trace_output")outputs.add(record.message.trace_id);
    }
  },finish({status,config,provenance}={}){
    const missing=[...inputs].filter(id=>!outputs.has(id)),orphan=[...outputs].filter(id=>!inputs.has(id));
    const result={schema:1,config,provenance,complete:!stopped&&status?.complete===true&&inputs.size>0&&missing.length===0&&orphan.length===0,
      reason:stopped||status?.reason||(!inputs.size?"no_trace_messages":missing.length||orphan.length?"unpaired_worker_messages":null),
      bytes,records:count,gray_bytes:offset,input_messages:inputs.size,output_messages:outputs.size,
      missing_output_ids:missing,orphan_output_ids:orphan,files:{events:path.basename(eventsFile),gray:path.basename(pixelsFile)}};
    fs.writeFileSync(summaryFile,JSON.stringify(result,null,2));return result;
  }};
}
module.exports={installMotionTraceCollector,createMotionTraceWriter};
