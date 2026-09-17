const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const {test}=require("node:test");

function helper(){
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const start=script.indexOf("function diagGapWarnings(");
  const end=script.indexOf("function renderDiagnostics(");
  assert.ok(start>=0&&end>start,"diagnostics helper must exist");
  const context=vm.createContext({});
  vm.runInContext(script.slice(start,end)+"globalThis.diagGapWarnings=diagGapWarnings;",context);
  return context.diagGapWarnings
}

const branch=(longest,current,p95)=>({
  longest_unanalyzed_gap_seconds:longest,
  current_unanalyzed_gap_seconds:current,
  latency_seconds:{model:{p95}},
});

test("frames still in flight under normal model latency are not a warning",()=>{
  const warn=helper();
  // 2.5 FPS, 1.5 s model latency: a healthy current gap of 1.6 s.
  assert.deepEqual({...warn(branch(0.4,1.6,1.5),0.4)},{gap:"",current:""});
});

test("a current gap beyond sampling spacing plus model latency warns",()=>{
  const warn=helper();
  assert.equal(warn(branch(0.4,3.0,1.5),0.4).current,"diag-warn");
  // Without latency data the sampling spacing alone is the threshold.
  assert.equal(warn(branch(0.4,1.2,null),0.4).current,"diag-warn");
});

test("the longest analyzed gap ignores in-flight latency",()=>{
  const warn=helper();
  assert.equal(warn(branch(1.2,0,5),0.4).gap,"diag-warn");
  assert.equal(warn(branch(0.8,0,5),0.4).gap,"");
});

test("missing period or branch data never warns",()=>{
  const warn=helper();
  assert.deepEqual({...warn(branch(9,9,0),null)},{gap:"",current:""});
  assert.deepEqual({...warn(null,0.4)},{gap:"",current:""});
});
