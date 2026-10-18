const assert=require('node:assert/strict');
const {test}=require('node:test');
const vm=require('node:vm');
const {script}=require('./helpers/dashboard.cjs');
const errorCode=script.slice(script.indexOf('function runtimeErrorText('),script.indexOf('function updateRuntimeSummary('));

test('runtime errors show HTTP reason and independent summary errors',()=>{
  const context=vm.createContext({});vm.runInContext(errorCode,context);
  const failed={last_error:'analysis request failed',pipeline:{branch_errors:{safety:'http_error'},
    branch_error_details:{safety:{code:'http_error',http_status:401,message:'Inference authentication failed. (HTTP 401)'}}},
    workflow:{summaries:{components:[{last_error:{code:'timeout',message:'Summary request timed out.'}}]}}};
  assert.match(context.runtimeErrorText(failed),/Event analysis.*authentication.*401/);
  assert.match(context.runtimeErrorText(failed),/Summary.*timed out/);
  assert.equal(context.runtimeErrorText({pipeline:{branch_errors:{},branch_error_details:{}}}), '');
  assert.match(context.runtimeErrorText({pipeline:{branch_errors:{safety:'timeout'}}}),/Event analysis.*timed out/);
});

function polling(){
  const calls=[],channels=[],records=[],timers=[];
  let response=()=>({ok:true,status:204,json:()=>{throw Error('204 must not be parsed')}});
  const context=vm.createContext({
    AbortController,Map,document:{hidden:false},window:{},
    GoodvisionDiagnostic:true,GoodvisionDiagnosticView:'results',GoodvisionFleet:{detailHidden:()=>false},
    channelSurfaceActive:()=>true,sceneView:{closed:false},state:{sessionId:'one',runtimeErrorMessage:'Event analysis · Authentication failed (HTTP 401)',diagnosticStatus:{vlm:{independent:false}}},
    detectionEvents:['smoking','fall'],ui:{sseChannel:{}},
    setTimeout:(fn,ms)=>{timers.push({fn,ms});return timers.length;},clearTimeout:()=>{},
    fetch:async url=>{calls.push(url);return response(url);},
    setChannel:(_element,text,tone)=>channels.push({text,tone}),analysis:data=>records.push(data),
  });
  context.window.parent=context.window;
  const start=script.indexOf('const diagnosticResults=new Map();');
  vm.runInContext(script.slice(start,script.indexOf('async function connectResultEvents()',start)),context);
  return {context,calls,channels,records,timers,respond:fn=>{response=fn;}};
}

test('empty polling stays quiet and displays inference error, then recovers on same URLs',async()=>{
  const view=polling();await view.context.pollDiagnosticResults();
  assert.equal(view.calls.length,3);
  assert.ok(view.calls.every(url=>url.endsWith('/latest?allow_empty=true')));
  assert.ok(!view.calls.some(url=>url.includes('/vlm/')));
  assert.match(view.channels.at(-1).text,/401/);assert.equal(view.channels.at(-1).tone,'bad');
  assert.equal(view.records.length,0);
  view.context.state.runtimeErrorMessage='';
  await view.context.pollDiagnosticResults();
  assert.match(view.channels.at(-1).text,/Waiting/);assert.equal(view.channels.at(-1).tone,'warn');
  view.respond(url=>url.includes('/smoking/')?{ok:true,status:200,json:async()=>({message_id:'recovered',stream_session_id:'one'})}:{ok:true,status:204});
  await view.context.pollDiagnosticResults();
  assert.equal(view.channels.at(-1).tone,'good');assert.equal(view.records.length,1);
  await view.context.pollDiagnosticResults();assert.equal(view.records.length,1);
});

test('transport and HTTP errors are never shown as a healthy result connection',async()=>{
  const view=polling();view.context.state.runtimeErrorMessage='';
  view.respond(()=>({ok:false,status:503}));await view.context.pollDiagnosticResults();
  assert.equal(view.channels.at(-1).tone,'warn');assert.match(view.channels.at(-1).text,/Could not read/);
  view.respond(()=>{throw Error('offline')});await view.context.pollDiagnosticResults();
  assert.equal(view.channels.at(-1).tone,'warn');
});

test('closed and non-result views stop result requests and timers',async()=>{
  const view=polling();view.context.GoodvisionDiagnosticView='events';await view.context.pollDiagnosticResults();
  assert.equal(view.calls.length,0);assert.equal(view.timers.length,0);
  view.context.GoodvisionDiagnosticView='results';view.context.sceneView.closed=true;await view.context.pollDiagnosticResults();
  assert.equal(view.calls.length,0);assert.equal(view.timers.length,0);
});
