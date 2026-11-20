/* Exercise the shipped object panel, not a stub of its polling function. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const {createDocument} = require('./helpers/dashboard.cjs');
const filename = path.join(__dirname, '../src/goodvision_harness/dashboard.html');
const html = fs.readFileSync(filename, 'utf8');
const start = html.indexOf('function objectPresenceContext(){') >= 0
  ? html.indexOf('function objectPresenceContext(){') : html.indexOf('function renderObjectPresence(');
const end = html.indexOf('\nasync function pollOverallDescription()', start);
const helpers = new vm.Script(html.slice(start, end) + '\nglobalThis.poll=pollObjectPresence;', {filename});
function record(overrides = {}) {
  return {type:'object_presence', node_id:'objects', source_id:'source-a',
    stream_session_id:'objects-session', generation:8, timestamp_seconds:4.8,
    source_frame_sequence:12, error:null, objects:[{id:'knife',name:'knife',zh_name:'刀具',
      present:true,answer:true,present_probability:0.97,gate:0.95}], ...overrides};
}
function fixture({items = [record()], response} = {}) {
  const {document, element} = createDocument();
  const manager = {state:'running',node_id:'objects',stream_session_id:'objects-session',generation:8};
  const state = {running:true,sessionId:'old-native-session',diagnosticStatus:{source_id:'source-a',
    workflow:{enabled:true,stream_session_id:'objects-session',generation:8,objects:manager}}};
  const requests = [], timers = new Map(); let timer = 0;
  const context = vm.createContext({state, document, $:element, AbortController, sceneView:{closed:false},
    GoodvisionI18n:{language:()=> 'en'},
    setTimeout(callback,delay){const id=++timer;timers.set(id,{callback,delay});return id;},
    clearTimeout(id){timers.delete(id);},
    async fetch(url){requests.push(url);const payload=response ? await response(state) : {status:manager,items};
      return {ok:true,status:200,json:async()=>payload};}});
  helpers.runInContext(context);
  return {state,manager,requests,timers,element,context,poll:()=>context.poll()};
}
test('object panel uses workflow identity, independently of the native event session', async()=>{
  const view=fixture(); await view.poll();
  assert.equal(view.element('object-presence').hidden,false);
  assert.match(view.element('object-presence-list').children[0].textContent,/knife.*PRESENT/);
  assert.equal(view.timers.size,0);
});
test('stopping hides retained detections and stops polling object history',async()=>{
  const view=fixture();await view.poll();view.state.running=false;view.manager.state='stopped';await view.poll();
  assert.equal(view.element('object-presence').hidden,true);
  assert.equal(view.requests.length,1);
});
test('a healthy same-session refresh preserves the visible result while awaiting its response',async()=>{
  let hold=false,release;
  const view=fixture({response:()=>hold ? new Promise(resolve=>{release=resolve;})
    : {status:view.manager,items:[record()]}});
  await view.poll();hold=true;
  const pending=view.poll();
  assert.equal(view.element('object-presence').hidden,false);
  assert.match(view.element('object-presence-list').children[0].textContent,/knife.*PRESENT/);
  release({status:view.manager,items:[record({source_frame_sequence:13})]});await pending;
  assert.equal(view.element('object-presence').hidden,false);
});
test('retained server history is hidden if the backend stops before diagnostic status refreshes',async()=>{
  for(const changes of [{state:'stopped'},{state:'stopping'},{stream_session_id:'old'},{generation:7}]){
    let changed=false;
    const view=fixture({response:()=>({status:changed ? {...view.manager,...changes} : view.manager,items:[record()]})});
    await view.poll();assert.equal(view.element('object-presence').hidden,false);
    changed=true;await view.poll();
    assert.equal(view.element('object-presence').hidden,true);
    assert.equal(view.element('object-presence-list').children.length,0);
  }
});
test('workflows without objects and one-shot tests never poll objects',async()=>{
  for(const mode of ['disabled','test','stopping']){
    const view=fixture();
    if(mode==='disabled')view.state.diagnosticStatus.workflow.objects=null;
    else if(mode==='test')view.state.diagnosticStatus.workflow_test_running=true;
    else view.manager.state='stopping';
    await view.poll();assert.equal(view.requests.length,0);assert.equal(view.element('object-presence').hidden,true);
  }
});
test('old sessions, generations, sources and removed nodes cannot appear as current detections',async()=>{
  for(const changes of [{stream_session_id:'old'},{generation:7},{source_id:'other'},{node_id:'removed'},{type:'other'}]){
    const view=fixture({items:[record(changes)]});await view.poll();
    assert.doesNotMatch(view.element('object-presence-list').textContent,/PRESENT/);
    assert.equal(view.element('object-presence-list').children.length,0);
  }
});
test('an in-flight old result cannot paint after workflow replacement or close',async()=>{
  for(const mode of ['session','generation','close','stop']){
    const view=fixture({response:state=>{
      if(mode==='session')state.diagnosticStatus.workflow.stream_session_id='new';
      else if(mode==='generation')state.diagnosticStatus.workflow.generation++;
      else if(mode==='close')view.context.sceneView.closed=true;
      else state.running=false;
      return {status:view.manager,items:[record()]};
    }});
    await view.poll();assert.equal(view.element('object-presence-list').children.length,0);
  }
});
