#!/usr/bin/env node
// Render captured synthetic Worker failures. Reads private diagnostics only.
"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict"),{chromium}=require("playwright");
const [input,stem]=process.argv.slice(2);assert.ok(input&&stem,"Expected <private-report.json> <private-output-stem>");
const report=JSON.parse(fs.readFileSync(input,"utf8").replace(/^\uFEFF/,""));
async function main(){
  const browser=await chromium.launch({executablePath:"C:/Program Files/Google/Chrome/Application/chrome.exe",headless:true});
  try{
    const page=await browser.newPage(),cases=report.synthetic.dense_failure_cases.filter(item=>item.reason==="boundary_occluded");
    for(let index=0;index<Math.min(2,cases.length);index++){
      const item=cases[index],png=await page.evaluate(item=>{
        const {width:w,height:h}=item,scale=3,header=75,canvas=document.createElement("canvas");
        canvas.width=w*scale*2;canvas.height=h*scale+header;const ctx=canvas.getContext("2d");
        ctx.fillStyle="#10181c";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font="15px sans-serif";ctx.fillStyle="#fff";
        ctx.fillText(`Actual Worker: source ${item.source.sequence} -> target ${item.target.sequence}, ${item.reason}`,12,22);
        const q=item.prediction_vs_generated_reference;
        ctx.fillText(`Rejected mask IoU ${q.mask_iou.toFixed(3)}, boundary p95 ${q.boundary_p95_chamfer_px.toFixed(2)} gray pixels`,12,44);
        ctx.fillStyle="#ffcf40";ctx.fillText("Input propagated contour",12,66);
        ctx.fillStyle="#00ff79";ctx.fillText("Exact generated target",w*scale+12,66);
        ctx.fillStyle="#ff39ef";ctx.fillText("Rejected mask",w*scale+210,66);
        function panel(pixels,x){
          const small=document.createElement("canvas");small.width=w;small.height=h;const c=small.getContext("2d"),image=c.createImageData(w,h);
          pixels.forEach((v,i)=>image.data.set([v,v,v,255],i*4));c.putImageData(image,0,0);
          ctx.imageSmoothingEnabled=false;ctx.drawImage(small,x,header,w*scale,h*scale);
        }
        function polygons(shapes,x,color){
          ctx.strokeStyle=color;ctx.lineWidth=1.5;
          for(const polygon of shapes||[]){ctx.beginPath();polygon.forEach(([px,py],i)=>i?ctx.lineTo(x+px*(w-1)*scale,header+py*(h-1)*scale):ctx.moveTo(x+px*(w-1)*scale,header+py*(h-1)*scale));ctx.closePath();ctx.stroke();}
        }
        panel(item.source_pixels,0);panel(item.target_pixels,w*scale);polygons(item.input_polygons,0,"#ffcf40");
        const mask=item.predicted_mask;ctx.fillStyle="#ff39ef";
        if(mask)for(let y=0;y<h;y++)for(let x=0;x<w;x++){
          const at=y*w+x;if(mask[at]>=128&&(x===0||y===0||x===w-1||y===h-1||mask[at-1]<128||mask[at+1]<128||mask[at-w]<128||mask[at+w]<128))
            ctx.fillRect(w*scale+x*scale,header+y*scale,scale,scale);
        }
        polygons(item.target_reference.polygons,w*scale,"#00ff79");
        const failure=item.diagnostics.boundary_failure;ctx.strokeStyle="#ff4c32";ctx.lineWidth=2;ctx.beginPath();
        ctx.arc(w*scale+failure.x*scale,header+failure.y*scale,12,0,Math.PI*2);ctx.stroke();
        return canvas.toDataURL("image/png").split(",")[1];
      },item);
      const file=path.resolve(`${stem}-${index+1}.png`);fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,Buffer.from(png,"base64"));console.log(file);
    }
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
