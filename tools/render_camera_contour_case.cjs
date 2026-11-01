#!/usr/bin/env node
// Render one saved quality result; does not rerun tracking or contact a model.
"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict"),{chromium}=require("playwright");
const args={};for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,"")]=process.argv[i+1];
assert.ok(args.report&&args.manifest&&args.set&&args.sequence&&args.scenario&&args.output);
const read=file=>JSON.parse(fs.readFileSync(file,"utf8").replace(/^\uFEFF/,""));
const report=read(args.report),manifest=read(args.manifest),row=report.quality.rows.find(row=>row.set===args.set&&row.sequence===Number(args.sequence)&&row.scenario===args.scenario),
  frame=manifest.sets.find(set=>set.name===args.set)?.sources.find(frame=>frame.sequence===Number(args.sequence));assert.ok(row&&frame);
async function main(){
  const browser=await chromium.launch({executablePath:"C:/Program Files/Google/Chrome/Application/chrome.exe",headless:true});
  try{
    const page=await browser.newPage(),data=await page.evaluate(async({row,jpeg})=>{
      const image=await createImageBitmap(await(await fetch(`data:image/jpeg;base64,${jpeg}`)).blob()),canvas=document.createElement("canvas");
      canvas.width=image.width;canvas.height=image.height+65;const ctx=canvas.getContext("2d");ctx.fillStyle="#10181c";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,65);
      ctx.font="15px sans-serif";ctx.fillStyle="#fff";ctx.fillText(`Source ${row.sequence}, ${row.scenario}`,12,22);
      ctx.fillStyle="#00ff79";ctx.fillText("Old SAM reference",12,48);ctx.fillStyle="#ff39ef";ctx.fillText("Actual trace propagation",220,48);
      for(const person of row.people)for(const [key,color,closed] of [["reference_polygons","#00ff79",true],["predicted_polygons","#ff39ef",true],["open_segments","#ff39ef",false]]){
        ctx.strokeStyle=color;ctx.lineWidth=2;
        for(const polygon of person[key]||[]){ctx.beginPath();polygon.forEach(([x,y],i)=>i?ctx.lineTo(x*(image.width-1),65+y*(image.height-1)):ctx.moveTo(x*(image.width-1),65+y*(image.height-1)));if(closed)ctx.closePath();ctx.stroke();}
      }
      image.close();return canvas.toDataURL("image/png").split(",")[1];
    },{row,jpeg:frame.jpeg});
    fs.mkdirSync(path.dirname(path.resolve(args.output)),{recursive:true});fs.writeFileSync(args.output,Buffer.from(data,"base64"));console.log(path.resolve(args.output));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
