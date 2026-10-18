const {test}=require('node:test');
const assert=require('node:assert/strict');
const {WorkflowCanvas,definitions,connectionError,validateGraph,effectiveEnabled,nodeReadiness,removeSelection,fitViewport,isEditingTarget,newNode,cameraBusy}=require('../src/goodvision_harness/assets/workflow_canvas.js');
const clone=value=>JSON.parse(JSON.stringify(value));
const node=(id,type,x=0,y=0)=>({id,type,x,y,enabled:true,config:{name:id,...(type==='summary'?{interval_seconds:60,prompt:'Describe this scene.'}:{})}});
const edge=(id,from,fromPort,to,toPort)=>({id,from_node:from,from_port:fromPort,to_node:to,to_port:toPort});
const graph=()=>({nodes:[node('input','source',40,120),node('face','face',330,70),node('summary','summary',330,280),node('out','output',620,140)],edges:[edge('a','input','frames','face','frames'),edge('b','input','frames','summary','frames'),edge('c','face','faces','out','faces'),edge('d','summary','summary','out','summary')]});
function appFixture(){
 const app=Object.create(WorkflowCanvas.prototype),saved=graph();
 Object.assign(app,{data:{document:{revision:4,workflows:[{id:'workflow-1',name:'Entrance',graph:clone(saved)},{id:'workflow-2',name:'Yard',graph:clone(saved)}],groups:[{id:'group-1',name:'Entrance',camera_ids:['camera01'],workflow_id:'workflow-1',applied_graph:clone(saved),enabled:true,block_enabled:{},camera_overrides:{}}],trash:[]},cameras:[{id:'camera01',enabled:true,status:{analysis_state:'idle'}}],event_catalog:{version:1,events:[]}},drafts:new Map(),undo:new Map(),workflowId:'workflow-1',groupId:'group-1',scope:'all',mode:'edit',selected:'face',view:{x:0,y:0,scale:1},busy:false,testBusy:false,library:true,inspector:true,pending:null,testResult:null,notice:'',error:false,testCamera:'camera01',render(){},feedback(){},t:value=>value,el:{ownerDocument:{activeElement:null}}});
 return app;
}
test('typed ports reject mismatches, occupied inputs and self connections',()=>{
 const g=graph();assert.equal(connectionError(g,'face','faces','out','faces'),'This input already has a connection. Disconnect it first.');
 assert.match(connectionError(g,'face','faces','summary','frames'),/types do not match/);
 assert.match(connectionError(g,'face','faces','face','frames'),/itself/);
 assert.equal(connectionError({...g,edges:g.edges.filter(e=>e.id!=='c')},'face','faces','out','faces'),null);
 assert.deepEqual(validateGraph(g,{complete:true}),[]);
});
test('incomplete local edits remain available while Apply requires usable input wiring',()=>{
 assert.deepEqual(validateGraph({nodes:[],edges:[]}),[]);
 assert.match(validateGraph({nodes:[],edges:[]},{complete:true}).join(' '),/workflow input/);
 const g=graph();g.edges=g.edges.filter(e=>e.to_node!=='summary');assert.deepEqual(validateGraph(g),[]);assert.match(validateGraph(g,{complete:true}).join(' '),/summary needs an input/);
 g.nodes.find(n=>n.id==='summary').config.interval_seconds=0;assert.match(validateGraph(g).join(' '),/interval/);
});
test('removing a block removes its incident connections and Undo restores exact graph',()=>{
 const app=appFixture(),before=clone(app.graph);app.deleteSelected();assert(!app.graph.nodes.some(n=>n.id==='face'));assert(!app.graph.edges.some(e=>e.from_node==='face'||e.to_node==='face'));assert.equal(app.graph.nodes.length,3);assert.equal(app.history().length,1);
 app.handleAction('undo',{});assert.deepEqual(app.graph,before);assert.equal(app.selected,'face');assert.equal(app.history().length,0);
 const noWire=removeSelection(before,'d');assert.equal(noWire.nodes.length,4);assert.equal(noWire.edges.length,3);
});
test('adding independent summaries preserves originals; singleton native lanes stay bounded',()=>{
 const app=appFixture(),before=clone(app.graph);app.add('summary',{x:400,y:600});assert.equal(app.graph.nodes.length,before.nodes.length+1);assert.deepEqual(app.graph.nodes.slice(0,before.nodes.length),before.nodes);assert.deepEqual(app.graph.edges,before.edges);
 const summaries=app.graph.nodes.filter(n=>n.type==='summary');assert.equal(new Set(summaries.map(n=>n.id)).size,2);summaries[1].config.prompt='Different instruction';assert.equal(summaries[0].config.prompt,'Describe this scene.');
 app.add('face');assert.equal(app.graph.nodes.filter(n=>n.type==='face').length,1);
 assert.equal(definitions.source.single,true);assert.equal(newNode('summary',app.graph).config.interval_seconds,60);
});
test('group gates always win over camera On, null inherits and camera Off remains local',()=>{
 const n=node('summary','summary'),g={enabled:true,block_enabled:{},camera_overrides:{camera01:{enabled:null,block_enabled:{summary:null}},camera02:{enabled:true,block_enabled:{summary:false}}}};
 assert.equal(effectiveEnabled(g,n,'camera01'),true);assert.equal(effectiveEnabled(g,n,'camera02'),false);
 g.block_enabled.summary=false;g.camera_overrides.camera01.block_enabled.summary=true;assert.equal(effectiveEnabled(g,n,'camera01'),false);
 delete g.block_enabled.summary;g.enabled=false;assert.equal(effectiveEnabled(g,n,'camera01'),false);
 g.enabled=true;g.camera_overrides.camera01.enabled=false;assert.equal(effectiveEnabled(g,n,'camera01'),false);
});
test('pending edits and applied snapshots stay isolated across workflow switches',()=>{
 const app=appFixture(),applied=clone(app.group.applied_graph);app.draft.graph.nodes[1].config.name='Draft face';assert.equal(app.dirty(),true);assert.deepEqual(app.group.applied_graph,applied);
 app.workflowId='workflow-2';assert.equal(app.dirty(),false);app.draft.graph.nodes[1].config.name='Other draft';app.workflowId='workflow-1';assert.equal(app.graph.nodes[1].config.name,'Draft face');assert.equal(app.anyDirty(),true);
 app.mode='groups';assert.equal(app.graph.nodes[1].config.name,'face');
});
test('one Apply atomically submits the selected pending graph and revision without a preceding Save',async()=>{
 const app=appFixture();app.draft.graph.nodes[1].config.name='Saved face';app.workflowId='workflow-2';app.draft.graph.nodes[1].config.name='Keep unsaved';app.workflowId='workflow-1';
 const requests=[];app.request=async(url,options)=>{const body=JSON.parse(options.body);requests.push({url,body});const response=clone(app.data);response.document.revision++;response.document.workflows[0].graph=clone(body.graph);response.document.groups[0].applied_graph=clone(body.graph);return response;};
 await app.handleAction('apply',{});assert.equal(requests.length,1);const request=requests[0];assert.equal(request.url,'/v1/workflows/command');assert.equal(request.body.action,'apply');assert.equal(request.body.revision,4);assert.equal(request.body.workflow_id,'workflow-1');assert.deepEqual(request.body.group_ids,[]);assert.equal(request.body.graph.nodes[1].config.name,'Saved face');assert.equal(app.document.revision,5);assert.equal(app.dirty('workflow-1'),false);assert.equal(app.dirty('workflow-2'),true);assert.equal(app.group.applied_graph.nodes[1].config.name,'Saved face');
});
test('failed or conflicting Apply keeps local edits available for recovery',async()=>{
 const app=appFixture();app.draft.graph.nodes[1].config.name='Keep me';app.request=async()=>{const e=new Error('revision conflict');e.status=409;throw e;};
 await app.handleAction('apply',{});assert.equal(app.dirty(),true);assert.equal(app.graph.nodes[1].config.name,'Keep me');assert.equal(app.busy,false);assert.match(app.notice,/Reload/);
});
test('read-only group view cannot structurally alter its applied graph',()=>{
 const app=appFixture();app.mode='groups';const before=clone(app.graph);app.add('summary');app.deleteSelected();assert.deepEqual(app.graph,before);assert.equal(app.connect('input','frames','face','frames'),undefined);
});
test('Delete/Backspace editing guard covers fields and editable text; running lifecycle is explicit',()=>{
 assert.equal(isEditingTarget({closest:selector=>selector.includes('textarea')?{}:null}),true);assert.equal(isEditingTarget({closest:()=>null}),false);
 for(const state of ['starting','running','stopping','draining'])assert.equal(cameraBusy({status:{analysis_state:state}}),true);
 assert.equal(cameraBusy({status:{analysis_state:'idle'}}),false);
});
test('Fit keeps every block within desktop and narrow viewport bounds',()=>{
 for(const [width,height] of [[900,600],[390,420]]){const nodes=graph().nodes,v=fitViewport(nodes,width,height);for(const n of nodes){assert(n.x*v.scale+v.x>=0);assert(n.y*v.scale+v.y>=0);assert((n.x+176)*v.scale+v.x<=width);assert((n.y+148)*v.scale+v.y<=height);}}
});
test('health refresh updates runtime locks without advancing draft revision or replacing edits',async()=>{
 const app=appFixture();app.draft.graph.nodes[1].config.name='Unsaved';const revision=app.document.revision;let locks=0;app.updateRuntimeLocks=()=>locks++;app.request=async()=>({document:{...clone(app.document),revision:99},cameras:[{id:'camera01',status:{analysis_state:'running'}}],event_catalog:{version:1,events:[]}});await app.refreshMetadata();assert.equal(app.document.revision,revision);assert.equal(app.graph.nodes[1].config.name,'Unsaved');assert.equal(app.cameras[0].status.analysis_state,'running');assert.equal(locks,1);
});
test('per-camera controls lock only their affected camera while group controls lock the whole group',()=>{
 const app=appFixture();app.mode='groups';app.group.camera_ids.push('camera02');app.data.cameras.push({id:'camera02',status:{analysis_state:'idle'}});app.data.cameras[0].status.analysis_state='running';const controls=[{disabled:false}];app.el.querySelectorAll=selector=>selector.includes('data-wc-group-enabled')?controls:[];app.el.querySelector=()=>null;
 app.scope='camera02';app.updateRuntimeLocks();assert.equal(controls[0].disabled,false);app.scope='camera01';app.updateRuntimeLocks();assert.equal(controls[0].disabled,true);app.scope='all';app.updateRuntimeLocks();assert.equal(controls[0].disabled,true);
});
test('live summary history is source-qualified, escapes returned text and labels synthetic data',()=>{
 const app=appFixture();app.liveKey='group-1:all';app.liveResults=[{camera_id:'camera01',summaries:{status:{state:'running'},items:[{node_id:'summary',summary:'<script>unsafe()</script>',synthetic:true,timestamp_seconds:2}]},results:{outputs:[]}}];const html=app.renderLiveRecords();assert(html.includes('camera01'));assert(html.includes('running'));assert(html.includes('Synthetic · test mode'));assert(html.includes('&lt;script&gt;unsafe()&lt;/script&gt;'));assert(!html.includes('<script>'));
});

test('readiness follows upstream gates and distinguishes waiting versus partial output',()=>{
 const g=graph(),group={enabled:true,block_enabled:{face:false},camera_overrides:{}};const face=g.nodes.find(n=>n.id==='face'),out=g.nodes.find(n=>n.id==='out');
 assert.equal(nodeReadiness(g,group,face,'camera01'),'off');assert.equal(nodeReadiness(g,group,out,'camera01'),'partial');group.block_enabled.summary=false;assert.equal(nodeReadiness(g,group,out,'camera01'),'waiting');
 group.block_enabled={};g.edges=g.edges.filter(e=>e.to_node!=='face');assert.equal(nodeReadiness(g,group,face,'camera01'),'waiting');
});

test('Chinese block-library names use declared labels while English remains unchanged',()=>{
 const previous=globalThis.GoodvisionI18n;try{const app=Object.create(WorkflowCanvas.prototype);globalThis.GoodvisionI18n={language:()=> 'zh'};assert.equal(app.t('Every-minute summary'),'每分钟总结');assert.equal(app.t('Face recognition'),'人脸识别');globalThis.GoodvisionI18n={language:()=> 'en'};assert.equal(app.t('Every-minute summary'),'Every-minute summary');}finally{if(previous===undefined)delete globalThis.GoodvisionI18n;else globalThis.GoodvisionI18n=previous;}
});


test('common workflow and camera-group actions are available beside the active picker',()=>{
 const app=appFixture();let html=app.renderContextActions();assert(html.includes('data-wc-action="new-workflow"'));assert(html.includes('data-wc-action="copy"'));assert(html.includes('Workflow list'));assert(!html.includes('>Manage<'));
 app.mode='groups';html=app.renderContextActions();assert(html.includes('data-wc-action="new-group"'));assert(html.includes('data-wc-action="edit-group"'));assert(html.includes('data-wc-action="edit-applied"'));assert(html.includes('Group list'));assert(!html.includes('data-wc-action="new-workflow"'));
 app.groupId=null;html=app.renderContextActions();assert.match(html,/data-wc-action="edit-group" disabled/);assert.match(html,/data-wc-action="edit-applied" disabled/);
});

test('asset lists and trash stay within their explicit workflow or group context',()=>{
 const app=appFixture();app.document.trash=[{id:'deleted-workflow',kind:'workflow',asset:{name:'Deleted workflow'}},{id:'deleted-group',kind:'group',asset:{name:'Deleted group'}}];
 app.dialog={kind:'manage',scope:'workflows',tab:'workflows'};let html=app.renderDialog();assert(html.includes('Workflow list'));assert(html.includes('data-wc-action="manager-workflows"'));assert(!html.includes('data-wc-action="manager-groups"'));assert(!html.includes('data-wc-action="new-workflow"'));
 app.dialog.tab='trash';html=app.renderDialog();assert(html.includes('Deleted workflow'));assert(!html.includes('Deleted group'));
 app.mode='groups';app.dialog={kind:'manage',scope:'groups',tab:'trash'};html=app.renderDialog();assert(html.includes('Group list'));assert(html.includes('Deleted group'));assert(!html.includes('Deleted workflow'));
});

test('block settings keep relevant controls without duplicate page-navigation menus',()=>{
 const app=appFixture();let html=app.renderInspector();assert(html.includes('Manage people library'));assert(!html.includes('data-wc-action="deployment"'));assert(!html.includes('View internal steps'));assert(html.includes('Face detection'));
 app.mode='groups';app.selected='input';html=app.renderInspector();assert(html.includes('data-wc-group-enabled'));assert(!html.includes('data-wc-action="edit-group"'));assert(!html.includes('data-wc-action="live"'));
});

// Exercise the delegated listeners shipped to the browser rather than duplicating
// selection logic in the fixture. No DOM parser is needed for these field events.
function eventEditorFixture(){
 const app=appFixture(),listeners={};
 app.data.event_catalog={version:1,events:[
  {id:'smoking',kind:'safety',labels:{en:'Smoking'},default_enabled:true,sensitivity:{default:'standard',options:[{value:'standard'},{value:'high'}]}},
  {id:'fight',kind:'safety',labels:{en:'Fight'},default_enabled:true,sensitivity:{default:'high',options:[{value:'standard'},{value:'high'}]}},
  {id:'weapon',kind:'safety',labels:{en:'Weapon'},default_enabled:true},
  {id:'fall',kind:'safety',labels:{en:'Fall'},default_enabled:false,sensitivity:{default:'standard',options:[{value:'standard'},{value:'high'}]}},
  {id:'stranger',kind:'identity',labels:{en:'Stranger'},default_enabled:true}
 ]};
 app.workflow.graph={nodes:[node('input','source'),{...node('events','events'),config:{name:'Events',features:['smoking','fight','weapon'],event_sensitivities:{smoking:'high',fight:'standard'}}},node('out','output')],edges:[edge('input-events','input','frames','events','frames'),edge('events-out','events','events','out','events')]};
 app.selected='events';
 app.el.addEventListener=(kind,listener)=>{listeners[kind]=listener;};
 app.el.querySelector=()=>null;app.el.querySelectorAll=()=>[];
 app.bind();
 const fieldTarget=(dataset,{checked,value,disabled=false,type='text'}={})=>({dataset,checked,value,disabled,type,matches:selector=>selector.split(',').some(part=>{
   const key=part.trim().match(/^\[data-([a-z-]+)\]$/)?.[1]?.replace(/-([a-z])/g,(_,c)=>c.toUpperCase());
   return key!==undefined&&Object.hasOwn(dataset,key);
  }),hasAttribute:name=>Object.hasOwn(dataset,name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())),getAttribute:name=>dataset[name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]});
 app.fieldChange=async(dataset,options={})=>{
  const target=fieldTarget(dataset,options);
  await listeners.change({target});return target;
 };
 app.fieldInput=async(dataset,options={})=>{const target=fieldTarget(dataset,options);await listeners.input({target});return target;};
 app.dispatch=async(kind,target,extra={})=>listeners[kind]?.({target,preventDefault(){},...extra});
 return app;
}
function currentEvents(app){return app.graph.nodes.find(n=>n.type==='events');}
function assertSelectedSensitivities(app){
 const config=currentEvents(app).config;
 for(const [id,value] of Object.entries(config.event_sensitivities||{})){
  assert(config.features.includes(id),'unselected event must not retain sensitivity: '+id);
  const definition=app.catalog().find(item=>item.id===id);
  assert(definition?.sensitivity?.options.some(option=>option.value===value),'sensitivity must be a supported option for '+id);
 }
}
test('event checkbox deselection removes its sensitivity from the actual Apply request',async()=>{
 const app=eventEditorFixture();let body;
 await app.fieldChange({wcFeature:'smoking'},{checked:false});
 assert.deepEqual(currentEvents(app).config.features,['fight','weapon']);assertSelectedSensitivities(app);
 app.request=async(url,options)=>{assert.equal(url,'/v1/workflows/command');body=JSON.parse(options.body);const response=clone(app.data);response.document.revision++;response.document.workflows[0].graph=clone(body.graph);return response;};
 await app.handleAction('apply',{});
 assert.equal(body.action,'apply');const config=body.graph.nodes.find(n=>n.type==='events').config;
 assert.deepEqual(config.event_sensitivities,{fight:'standard'});assert.equal(app.dirty(),false);
});
test('event deselect, reselect and undo keep exact feature and sensitivity snapshots',async()=>{
 const app=eventEditorFixture(),before=clone(app.graph);
 await app.fieldChange({wcFeature:'smoking'},{checked:false});const deselected=clone(app.graph);assertSelectedSensitivities(app);
 await app.fieldChange({wcFeature:'smoking'},{checked:true});assert(currentEvents(app).config.features.includes('smoking'));assertSelectedSensitivities(app);
 await app.handleAction('undo',{});assert.deepEqual(app.graph,deselected);
 await app.handleAction('undo',{});assert.deepEqual(app.graph,before);assert.equal(currentEvents(app).config.event_sensitivities.smoking,'high');
});
test('new event block defaults include sensitivity only for its enabled safety events',()=>{
 const app=eventEditorFixture();app.workflow.graph=graph();app.drafts.clear();app.add('events');
 assert.deepEqual(currentEvents(app).config.features,['smoking','fight','weapon']);
 assert.deepEqual(currentEvents(app).config.event_sensitivities,{smoking:'standard',fight:'high'});assertSelectedSensitivities(app);
});
test('event sensitivity selector updates the selected event and undo restores its value',async()=>{
 const app=eventEditorFixture(),before=clone(app.graph);
 await app.fieldChange({wcSensitivity:'fight'},{value:'high'});assert.deepEqual(currentEvents(app).config.event_sensitivities,{smoking:'high',fight:'high'});assertSelectedSensitivities(app);
 await app.handleAction('undo',{});assert.deepEqual(app.graph,before);
});
test('disabled or stale sensitivity controls cannot add settings for unselected events',async()=>{
 const app=eventEditorFixture();await app.fieldChange({wcFeature:'smoking'},{checked:false});const before=clone(app.graph),history=app.history().length;
 await app.fieldChange({wcSensitivity:'smoking'},{value:'high',disabled:true});
 assert.deepEqual(app.graph,before);assert.equal(app.history().length,history);assertSelectedSensitivities(app);
});
test('last event stays selected with a local explanation and no undo entry',async()=>{
 const app=eventEditorFixture();currentEvents(app).config={name:'Events',features:['smoking'],event_sensitivities:{smoking:'high'}};const before=clone(app.graph),history=app.history().length;
 const target=await app.fieldChange({wcFeature:'smoking'},{checked:false});
 assert.equal(target.checked,true);assert.deepEqual(app.graph,before);assert.equal(app.history().length,history);assert.match(app.notice,/at least one event|至少.*事件/i);assert.equal(app.error,true);
});
test('valid event and sensitivity edits invalidate stale test output',async()=>{
 const app=eventEditorFixture();app.testResult={outputs:[{event:'old result'}]};
 await app.fieldChange({wcFeature:'smoking'},{checked:false});assert.equal(app.testResult,null);
 app.testResult={outputs:[{event:'old result'}]};await app.fieldChange({wcSensitivity:'fight'},{value:'high'});assert.equal(app.testResult,null);
});
test('unrecognized event controls and invalid sensitivity options leave the draft intact',async()=>{
 const app=eventEditorFixture(),before=clone(app.graph),history=app.history().length;
 await app.fieldChange({wcFeature:'not-in-catalog'},{checked:true});
 await app.fieldChange({wcSensitivity:'smoking'},{value:'unsupported'});
 await app.fieldChange({wcSensitivity:'fall'},{value:'high'});
 await app.fieldChange({wcSensitivity:'weapon'},{value:'high'});
 assert.deepEqual(app.graph,before);assert.equal(app.history().length,history);
});
test('empty event configuration is explained locally before applying or testing',async()=>{
 const app=eventEditorFixture();currentEvents(app).config={name:'Events',features:[],event_sensitivities:{}};let requests=0;app.request=async()=>{requests++;return clone(app.data);};
 await app.handleAction('apply',{});assert.equal(requests,0);assert.match(app.notice,/event|事件/i);
 await app.runTest();assert.equal(requests,0);assert.match(app.notice,/event|事件/i);assert.equal(app.testBusy,false);
});
test('summary validation matches service interval and instruction bounds',()=>{
 for(const interval of [1,3600]){const g=graph();g.nodes.find(n=>n.type==='summary').config.interval_seconds=interval;assert.deepEqual(validateGraph(g),[]);}
 for(const interval of [0,3600.1,86400]){const g=graph();g.nodes.find(n=>n.type==='summary').config.interval_seconds=interval;assert.match(validateGraph(g).join(' '),/interval/i);}
 const g=graph(),summary=g.nodes.find(n=>n.type==='summary');summary.config.prompt='x'.repeat(4000);assert.deepEqual(validateGraph(g),[]);summary.config.prompt+='x';assert.match(validateGraph(g).join(' '),/summary|instruction|4000/i);
});
test('complete validation requires enabled input, enabled output and a usable route for each active block',()=>{
 const valid=graph();assert.deepEqual(validateGraph(valid,{complete:true}),[]);
 for(const id of ['input','out']){const g=clone(valid);g.nodes.find(n=>n.id===id).enabled=false;assert(validateGraph(g,{complete:true}).length,'disabled '+id+' cannot form a runnable workflow');}
 const noOutput=clone(valid);noOutput.nodes=noOutput.nodes.filter(n=>n.id!=='out');noOutput.edges=noOutput.edges.filter(e=>e.to_node!=='out');assert(validateGraph(noOutput,{complete:true}).length);
 const noOutputRoute=clone(valid);noOutputRoute.edges=noOutputRoute.edges.filter(e=>e.id!=='d');assert(validateGraph(noOutputRoute,{complete:true}).length,'active summary must reach an output');
 const disabledRoute=clone(valid);disabledRoute.nodes.find(n=>n.id==='summary').enabled=false;disabledRoute.edges=disabledRoute.edges.filter(e=>e.id!=='b');assert.deepEqual(validateGraph(disabledRoute,{complete:true}),[],'disabled blocks need no active route');
});
test('explicit reload discards old undo snapshots along with unsaved drafts',async()=>{
 const app=eventEditorFixture(),saved=clone(app.data);await app.fieldChange({wcFeature:'smoking'},{checked:false});assert(app.history().length);assert(app.dirty());app.confirm=()=>true;app.request=async()=>clone(saved);
 const previous=globalThis.requestAnimationFrame;globalThis.requestAnimationFrame=callback=>{callback();return 0;};
 try{await app.handleAction('reload',{});}finally{if(previous===undefined)delete globalThis.requestAnimationFrame;else globalThis.requestAnimationFrame=previous;}
 assert.equal(app.dirty(),false);assert.equal(app.history().length,0);const fresh=clone(app.graph);await app.handleAction('undo',{});assert.deepEqual(app.graph,fresh);
});
function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return{promise,resolve,reject};}
test('run test validates the actual graph before calling inference',async()=>{
 const app=eventEditorFixture();app.graph.edges=app.graph.edges.filter(e=>e.to_node!=='events');let requests=0;app.request=async()=>{requests++;return{outputs:[]};};await app.runTest();
 assert.equal(requests,0);assert.equal(app.testBusy,false);assert.equal(app.error,true);assert.match(app.notice,/input|connect|输入|连接/i);
});
test('in-flight test response cannot populate results after changing the test camera',async()=>{
 const app=eventEditorFixture(),pending=deferred();app.data.cameras.push({id:'camera02',enabled:true,status:{analysis_state:'idle'}});let request;app.request=async(url,options)=>{request={url,body:JSON.parse(options.body)};return pending.promise;};
 const run=app.runTest();assert.equal(request.body.camera_id,'camera01');assert.equal(app.testBusy,true);
 await app.fieldChange({wcTestCamera:''},{value:'camera02'});pending.resolve({outputs:[{camera_id:'camera01',value:'old camera'}]});await run;
 assert.equal(app.testCamera,'camera02');assert.equal(app.testResult,null);assert.equal(app.testBusy,false);
});
test('event edits are blocked during a local test and its unchanged result remains current',async()=>{
 const app=eventEditorFixture(),pending=deferred();app.request=async()=>pending.promise;const run=app.runTest();
 await app.fieldChange({wcFeature:'smoking'},{checked:false});pending.resolve({outputs:[{event:'smoking',value:'current graph'}]});await run;
 assert.equal(app.testResult.outputs[0].value,'current graph');assert.equal(app.testBusy,false);assert(currentEvents(app).config.features.includes('smoking'));
});
test('in-flight test failure does not appear in a newly selected workflow',async()=>{
 const app=eventEditorFixture(),pending=deferred();app.request=async()=>pending.promise;const run=app.runTest();
 await app.fieldChange({wcContext:'workflow'},{value:'workflow-2'});pending.reject(new Error('old workflow inference failed'));await run;
 assert.equal(app.workflowId,'workflow-2');assert.equal(app.testResult,null);assert.equal(app.testBusy,false);assert(!app.notice.includes('old workflow inference failed'));
});
test('typed summary edits are blocked during a local test before blur',async()=>{
 const app=eventEditorFixture(),pending=deferred();app.workflow.graph=graph();app.drafts.clear();app.selected='summary';app.request=async()=>pending.promise;const run=app.runTest();
 await app.fieldInput({wcField:'prompt'},{value:'Describe the new instruction.'});pending.resolve({outputs:[{summary:'output from current instruction'}]});await run;
 assert.equal(app.graph.nodes.find(n=>n.id==='summary').config.prompt,'Describe this scene.');assert.equal(app.testResult.outputs[0].summary,'output from current instruction');assert.equal(app.testBusy,false);
});
test('in-flight inference is discarded if its graph changes outside local authoring controls',async()=>{
 const app=eventEditorFixture(),pending=deferred();app.request=async()=>pending.promise;const run=app.runTest();currentEvents(app).config.name='Changed by a newer external graph';pending.resolve({outputs:[{event:'smoking',value:'old graph'}]});await run;
 assert.equal(app.testResult,null);assert.equal(app.testBusy,false);
});
test('camera editing and test locks include pending starts, cleanup and unknown status',()=>{
 for(const state of ['idle','stopped','ended','error','configuration_error'])assert.equal(cameraBusy({status:{analysis_state:state}}),false,state+' is stopped');
 for(const state of ['starting','running','stopping','draining','unknown','new-runtime-state'])assert.equal(cameraBusy({status:{analysis_state:state}}),true,state+' is not confirmed stopped');
 for(const extra of [{session_start_pending:true},{workflow_test_running:true},{processing_cleanup:'closing'},{processing_cleanup:'stopping'}])assert.equal(cameraBusy({status:{analysis_state:'idle',...extra}}),true);
 assert.equal(cameraBusy({status:{pipeline_state:'stopped'}}),false);assert.equal(cameraBusy({status:{}}),true);assert.equal(cameraBusy({}),true);assert.equal(cameraBusy(undefined),true);
});
function renderedFeedbackFixture(){
 const app=eventEditorFixture(),feedback={innerHTML:'',classList:{toggle(){}},setAttribute(){}};
 app.t=WorkflowCanvas.prototype.t;app.feedback=WorkflowCanvas.prototype.feedback;app.el.querySelector=selector=>selector==='.wc-feedback'?feedback:null;
 return{app,feedback};
}
test('non-revision 409 preserves edits and does not direct users to reload the draft',async()=>{
 const {app,feedback}=renderedFeedbackFixture();await app.fieldChange({wcFeature:'smoking'},{checked:false});const before=clone(app.graph);
 app.request=async()=>{const error=new Error('Stop analysis before applying camera group changes.');error.status=409;throw error;};
 assert.equal(await app.command('apply',{workflow_id:app.workflowId,group_ids:[app.groupId]}),false);
 assert.equal(app.notice,'Stop analysis before applying camera group changes.');assert(!/reload|重新加载/i.test(app.notice));assert(!feedback.innerHTML.includes('data-wc-action="reload"'));assert.deepEqual(app.graph,before);assert.equal(app.dirty(),true);
});
test('revision 409 gives a localized reload explanation while preserving pending edits',async()=>{
 const previous=globalThis.GoodvisionI18n;globalThis.GoodvisionI18n={language:()=> 'zh'};
 try{const {app,feedback}=renderedFeedbackFixture();await app.fieldChange({wcFeature:'smoking'},{checked:false});const before=clone(app.graph);
  app.request=async()=>{const error=new Error('workflow revision conflict');error.status=409;throw error;};
  await app.handleAction('apply',{});
  assert.match(app.notice,/重新加载/);assert(!app.notice.includes('Reload'));assert(feedback.innerHTML.includes('data-wc-action="reload"'));assert.deepEqual(app.graph,before);assert.equal(app.dirty(),true);
 }finally{if(previous===undefined)delete globalThis.GoodvisionI18n;else globalThis.GoodvisionI18n=previous;}
});
test('typing name, prompt and interval creates one undo checkpoint per editing session',async()=>{
 for(const [field,values,type] of [['name',['Gate','Gate camera','Gate camera final'],'text'],['prompt',['Describe people.','Describe all activity.','Describe unusual activity.'],'text'],['interval_seconds',['10','20','30'],'number']]){
  const app=eventEditorFixture();app.workflow.graph=graph();app.drafts.clear();app.selected='summary';const original=clone(app.graph),getValue=()=>app.graph.nodes.find(n=>n.id==='summary').config[field];
  await app.fieldInput({wcField:field},{value:values[0],type});await app.fieldInput({wcField:field},{value:values[1],type});
  assert.equal(app.history().length,1,field+' keystrokes form one undo step');assert.equal(getValue(),type==='number'?Number(values[1]):values[1]);
  await app.fieldChange({wcField:field},{value:values[1],type});await app.fieldInput({wcField:field},{value:values[2],type});assert.equal(app.history().length,2,field+' refocus begins a new undo step');
  await app.handleAction('undo',{});assert.equal(getValue(),type==='number'?Number(values[1]):values[1]);await app.handleAction('undo',{});assert.deepEqual(app.graph,original);
 }
});
function summaryOutputFixture(){
 const app=eventEditorFixture();app.workflow.graph=graph();app.drafts.clear();app.selected='out';app.graph.nodes.push(node('summary-2','summary'),node('summary-3','summary'));
 app.graph.edges.push(edge('input-summary-2','input','frames','summary-2','frames'),edge('input-summary-3','input','frames','summary-3','frames'),edge('summary-2-out','summary-2','summary','out','summary'));
 return app;
}
test('one output aggregates independent summary lanes while rejecting duplicate connections',()=>{
 const app=summaryOutputFixture();assert.equal(connectionError(app.graph,'summary-3','summary','out','summary'),null);assert.match(connectionError(app.graph,'summary','summary','out','summary'),/already has a connection/);
 assert.equal(app.connect('summary-3','summary','out','summary'),true);assert.equal(app.graph.edges.filter(e=>e.to_node==='out'&&e.to_port==='summary').length,3);assert.deepEqual(validateGraph(app.graph,{complete:true}),[]);
 assert.equal(app.connect('summary-3','summary','out','summary'),false);assert.equal(app.graph.edges.filter(e=>e.to_node==='out'&&e.to_port==='summary').length,3);
 const html=app.renderInspector();assert(html.includes('data-wc-edge-id="d"'));assert(html.includes('data-wc-edge-id="summary-2-out"'));
});
test('editing or removing one output binding preserves sibling summary connections and undo',async()=>{
 const app=summaryOutputFixture(),before=clone(app.graph),sibling=clone(app.graph.edges.find(e=>e.id==='summary-2-out'));
 await app.fieldChange({wcBinding:'summary',wcEdgeId:'d'},{value:'summary-3|summary'});
 assert.deepEqual(app.graph.edges.find(e=>e.id==='summary-2-out'),sibling);assert(!app.graph.edges.some(e=>e.from_node==='summary'&&e.to_node==='out'));assert(app.graph.edges.some(e=>e.from_node==='summary-3'&&e.to_node==='out'));
 await app.handleAction('undo',{});assert.deepEqual(app.graph,before);
 await app.fieldChange({wcBinding:'summary',wcEdgeId:'d'},{value:''});assert.deepEqual(app.graph.edges.find(e=>e.id==='summary-2-out'),sibling);assert.equal(app.graph.edges.filter(e=>e.to_node==='out'&&e.to_port==='summary').length,1);
 await app.handleAction('undo',{});assert.deepEqual(app.graph,before);
});
test('attempting to replace an output binding with its sibling is rejected without losing either edge',async()=>{
 const app=summaryOutputFixture(),before=clone(app.graph),history=app.history().length;
 await app.fieldChange({wcBinding:'summary',wcEdgeId:'d'},{value:'summary-2|summary'});
 assert.deepEqual(app.graph,before);assert.equal(app.history().length,history);assert.match(app.notice,/already has a connection/);
});
test('rerender focus tokens identify event fields and the specific repeated output binding',async()=>{
 const app=eventEditorFixture();const feature=await app.fieldChange({wcFeature:'smoking'},{checked:true}),sensitivity=await app.fieldChange({wcSensitivity:'fight'},{value:'standard'});
 assert.deepEqual(app.focusToken(feature),{key:'data-wc-feature',value:'smoking',node:undefined,dir:undefined,edge:undefined});assert.equal(app.focusToken(sensitivity).key,'data-wc-sensitivity');assert.equal(app.focusToken(sensitivity).value,'fight');
 const other=await app.fieldChange({wcBinding:'summary',wcEdgeId:'other-edge'},{value:''}),chosen=await app.fieldChange({wcBinding:'summary',wcEdgeId:'chosen-edge'},{value:''});app.el.querySelectorAll=()=>[other,chosen];
 assert.equal(app.findFocus(app.focusToken(chosen)),chosen);
});

test('typing new configuration removes the visible stale test result while preserving the editor',async()=>{
 const app=eventEditorFixture(),panel={innerHTML:'old visible test output'};
 app.testOpen=true;app.testResult={outputs:[{description:'old visible test output'}]};
 app.el.querySelector=selector=>selector==='.wc-test-results'?panel:null;
 await app.fieldInput({wcField:'name'},{value:'Updated event name'});
 assert.equal(app.testResult,null);assert.doesNotMatch(panel.innerHTML,/old visible test output/);
 assert.match(panel.innerHTML,/No test has run/);assert.equal(currentEvents(app).config.name,'Updated event name');
});

function trackingEditorFixture(configured=true){
 const app=eventEditorFixture();app.data.tracking={configured,reason:configured?'':'SAM person tracking is not configured.',processing_method:'sam_live'};
 app.graph.nodes.push(node('summary','summary'));app.graph.edges.push(edge('input-summary','input','frames','summary','frames'),edge('summary-out','summary','summary','out','summary'));
 return app;
}
async function connectedTrackingFixture(configured=true){
 const app=trackingEditorFixture(configured);app.add('tracking');const tracker=app.graph.nodes.find(n=>n.type==='tracking');
 assert(tracker,'Person tracking is an authorable Canvas block');app.selected=tracker.id;
 await app.fieldChange({wcBinding:'frames',wcEdgeId:''},{value:'input|frames'});app.selected='out';
 await app.fieldChange({wcBinding:'tracks',wcEdgeId:''},{value:tracker.id+'|tracks'});return{app,tracker};
}
test('person tracking is visible in the library and authoring enforces its singleton rule',()=>{
 const app=trackingEditorFixture(false);assert.equal(definitions.tracking.single,true);assert.deepEqual(definitions.tracking.inputs,[['frames','frames']]);assert.deepEqual(definitions.tracking.outputs,[['tracks','tracks']]);assert(definitions.output.inputs.some(port=>port[0]==='tracks'&&port[1]==='tracks'));
 assert(app.renderLibrary().includes('data-wc-add="tracking"'));app.add('tracking');const tracker=app.graph.nodes.find(n=>n.type==='tracking');assert(tracker);assert.deepEqual(Object.keys(tracker.config),['name']);const history=app.history().length;
 app.add('tracking');assert.equal(app.graph.nodes.filter(n=>n.type==='tracking').length,1);assert.equal(app.history().length,history);assert.match(app.renderLibrary(),/data-wc-add="tracking"[^>]*disabled/);
});
test('tracking bindings apply a frames to tracks route beside independent events and summaries',async()=>{
 const {app,tracker}=await connectedTrackingFixture();assert.deepEqual(app.validation(true),[]);assert.equal(connectionError(app.graph,tracker.id,'tracks','out','events'),'These port types do not match.');
 let body;app.request=async(url,options)=>{assert.equal(url,'/v1/workflows/command');body=JSON.parse(options.body);const response=clone(app.data);response.document.revision++;response.document.workflows[0].graph=clone(body.graph);return response;};
 await app.handleAction('apply',{});assert.equal(body.action,'apply');const saved=body.graph.nodes.find(n=>n.type==='tracking');assert.deepEqual(saved.config,{name:tracker.config.name});
 assert(body.graph.edges.some(e=>e.from_node==='input'&&e.to_node===tracker.id&&e.to_port==='frames'));assert(body.graph.edges.some(e=>e.from_node===tracker.id&&e.from_port==='tracks'&&e.to_node==='out'&&e.to_port==='tracks'));
 assert(body.graph.edges.some(e=>e.from_node==='events'&&e.to_port==='events'));assert(body.graph.edges.some(e=>e.from_node==='summary'&&e.to_port==='summary'));assert.equal(app.dirty(),false);
});
test('tracking toggle preserves events and summaries, and undo restores the enabled tracking block',async()=>{
 const {app,tracker}=await connectedTrackingFixture(false),before=clone(app.graph);app.selected=tracker.id;
 await app.fieldChange({wcNodeEnabled:''},{checked:false});assert.equal(tracker.enabled,false);assert.deepEqual(app.graph.nodes.filter(n=>n.type!=='tracking'),before.nodes.filter(n=>n.type!=='tracking'));assert.deepEqual(app.graph.edges,before.edges);assert.deepEqual(app.validation(true),[]);assert.equal(app.trackingUnavailable(),false);
 await app.handleAction('undo',{});assert.deepEqual(app.graph,before);assert.equal(app.trackingUnavailable(),true);
});
test('tracking applies the same group gates and per-camera overrides through the shipped controls',async()=>{
 const {app,tracker}=await connectedTrackingFixture();app.group.applied_graph=clone(app.graph);app.group.camera_ids.push('camera02');app.data.cameras.push({id:'camera02',enabled:true,status:{analysis_state:'idle'}});app.mode='groups';app.selected=tracker.id;const commands=[];
 app.request=async(url,options)=>{assert.equal(url,'/v1/workflows/command');const body=JSON.parse(options.body);commands.push(body);const response=clone(app.data),{revision,action,id,...patch}=body;assert.equal(action,'update_group');Object.assign(response.document.groups.find(group=>group.id===id),patch);response.document.revision++;return response;};
 await app.fieldChange({wcGroupBlock:tracker.id},{checked:false});assert.equal(effectiveEnabled(app.group,tracker,'camera01'),false);assert.equal(effectiveEnabled(app.group,tracker,'camera02'),false);
 app.scope='camera01';await app.fieldChange({wcPolicy:'block',node:tracker.id},{value:'on'});assert.equal(effectiveEnabled(app.group,tracker,'camera01'),false,'camera On cannot bypass a disabled group block');
 app.scope='all';await app.fieldChange({wcGroupBlock:tracker.id},{checked:true});assert.equal(effectiveEnabled(app.group,tracker,'camera01'),true);assert.equal(effectiveEnabled(app.group,tracker,'camera02'),true);
 app.scope='camera01';await app.fieldChange({wcPolicy:'block',node:tracker.id},{value:'off'});assert.equal(effectiveEnabled(app.group,tracker,'camera01'),false);assert.equal(effectiveEnabled(app.group,tracker,'camera02'),true);
 await app.fieldChange({wcPolicy:'block',node:tracker.id},{value:'inherit'});assert.equal(effectiveEnabled(app.group,tracker,'camera01'),true);assert.equal(commands.length,5);assert.equal(app.group.camera_overrides.camera01.block_enabled[tracker.id],null);
 for(const id of ['events','summary'])assert.equal(effectiveEnabled(app.group,app.graph.nodes.find(n=>n.id===id),'camera01'),true,id+' remains independently enabled');
});
test('unconfigured tracking remains applicable and has an explicit capability warning before inference',async()=>{
 const {app,tracker}=await connectedTrackingFixture(false);app.data.tracking.reason='<tracker unavailable>';app.selected=tracker.id;assert.deepEqual(app.validation(),[]);assert.deepEqual(app.validation(true),[]);assert.equal(app.trackingUnavailable(),true);
 const html=app.renderInspector();assert(html.includes('&lt;tracker unavailable&gt;'));assert(!html.includes('<tracker unavailable>'));assert(html.includes('data-wc-node-enabled'));assert(!html.includes('data-wc-field="processing_method"'));
 let requests=0;app.request=async()=>{requests++;return{outputs:[]};};await app.runTest();assert.equal(requests,0);assert.equal(app.testBusy,false);assert.equal(app.error,true);assert.match(app.notice,/track|追踪|跟踪/i);
});
test('turning off unavailable tracking lets event and summary inference run independently',async()=>{
 const {app,tracker}=await connectedTrackingFixture(false);app.selected=tracker.id;await app.fieldChange({wcNodeEnabled:''},{checked:false});const events=clone(currentEvents(app).config),summary=clone(app.graph.nodes.find(n=>n.type==='summary').config);let body;
 app.request=async(url,options)=>{assert.equal(url,'/v1/workflows/test');body=JSON.parse(options.body);return{outputs:[{node_id:'events',result:{detected:false}},{node_id:'summary',summary:'The full scene.'}]};};await app.runTest();
 assert.equal(body.graph.nodes.find(n=>n.type==='tracking').enabled,false);assert.deepEqual(body.graph.nodes.find(n=>n.type==='events').config,events);assert.deepEqual(body.graph.nodes.find(n=>n.type==='summary').config,summary);assert.equal(app.testResult.outputs.length,2);
});
test('configured tracking can be tested using its actual graph without coupling fast events to tracking',async()=>{
 const {app,tracker}=await connectedTrackingFixture(true);assert.equal(app.trackingUnavailable(),false);let body;app.request=async(url,options)=>{body=JSON.parse(options.body);return{outputs:[{node_id:tracker.id,tracks:[]},{node_id:'events',result:{detected:false}}]};};await app.runTest();
 assert(body.graph.nodes.some(n=>n.id===tracker.id&&n.enabled!==false));assert(body.graph.nodes.some(n=>n.type==='events'&&n.enabled!==false));assert.equal(app.testResult.outputs.length,2);assert.equal(app.testBusy,false);
});
test('metadata refresh updates tracking availability without replacing the draft',async()=>{
 const {app}=await connectedTrackingFixture(false);const draft=clone(app.graph),revision=app.document.revision;let locks=0;app.updateRuntimeLocks=()=>locks++;app.request=async()=>({cameras:clone(app.cameras),event_catalog:clone(app.data.event_catalog),tracking:{configured:true,reason:'',processing_method:'sam_live'}});
 await app.refreshMetadata();assert.equal(app.data.tracking.configured,true);assert.equal(app.trackingUnavailable(),false);assert.deepEqual(app.graph,draft);assert.equal(app.document.revision,revision);assert.equal(locks,1);
});

function canvasHtml(app){
 app.el.contains=()=>false;app.el.querySelector=()=>null;app.el.querySelectorAll=()=>[];app.layout=()=>{};app.size=()=>{};app.scheduleLive=()=>{};app.updateRuntimeLocks=()=>{};
 WorkflowCanvas.prototype.render.call(app);return app.el.innerHTML;
}
test('Canvas exposes direct Apply with no Save button or visible draft stage',()=>{
 const app=eventEditorFixture();const html=canvasHtml(app);assert(!html.includes('data-wc-action="save"'));assert(html.includes('data-wc-action="apply"'));assert.doesNotMatch(html,/Save draft|Draft saved|Draft only|Enabled in draft|Save the draft before applying|Edit the shared draft/);
 assert(!app.renderInspector().includes('Enabled in draft'));assert(!app.renderNode(currentEvents(app)).includes('Draft only'));
});
async function withImmediateAnimation(work){
 const previous=globalThis.requestAnimationFrame;globalThis.requestAnimationFrame=callback=>{callback();return 0;};
 try{return await work();}finally{if(previous===undefined)delete globalThis.requestAnimationFrame;else globalThis.requestAnimationFrame=previous;}
}
test('unassigned workflows can Apply pending configuration with an empty group list in one request',async()=>{
 const app=eventEditorFixture();app.document.groups=[];app.groupId=null;await app.fieldInput({wcField:'name'},{value:'Updated unassigned events'});const requests=[];
 app.request=async(url,options)=>{const body=JSON.parse(options.body);requests.push(body);const response=clone(app.data);response.document.revision++;response.document.workflows[0].graph=clone(body.graph);return response;};
 await withImmediateAnimation(()=>app.handleAction('apply',{}));assert.equal(app.dialog.kind,'apply');assert.deepEqual(app.dialog.groups,[]);const html=app.renderDialog();assert(html.includes('Entrance'));assert(!html.includes('Save the draft'));assert.doesNotMatch(html,/data-wc-action="confirm-apply"[^>]*disabled/);
 await app.handleAction('confirm-apply',{});assert.equal(requests.length,1);assert.equal(requests[0].action,'apply');assert.deepEqual(requests[0].group_ids,[]);assert.equal(requests[0].graph.nodes.find(n=>n.type==='events').config.name,'Updated unassigned events');assert.equal(app.dirty(),false);
});
test('assigned camera activity locks its workflow regardless of group gates and unknown state is locked',()=>{
 const statuses=[...['starting','running','stopping','draining','unknown'].map(analysis_state=>({analysis_state})),{analysis_state:'idle',session_start_pending:true},{analysis_state:'idle',workflow_test_running:true},{analysis_state:'idle',processing_cleanup:'closing'},null,undefined];
 for(const status of statuses){const app=eventEditorFixture();app.group.enabled=false;app.group.block_enabled.events=false;app.group.camera_overrides.camera01={enabled:false,block_enabled:{events:false}};app.data.cameras[0].status=status;assert.equal(app.editLocked(),true,JSON.stringify(status)+' must lock assigned workflow');}
 const app=eventEditorFixture();app.group.camera_ids.push('camera-missing');assert.equal(app.editLocked(),true,'missing assigned camera status cannot be treated as idle');
});
test('running workflows reject every authoring path while preserving pending graph and undo history',async()=>{
 const app=eventEditorFixture();await app.fieldChange({wcSensitivity:'fight'},{value:'high'});const before=clone(app.graph),history=app.history().length;app.data.cameras[0].status={analysis_state:'running'};app.pending=null;
 app.add('summary');app.deleteSelected();app.connect('input','frames','events','frames');app.choosePort({dataset:{node:'input',wcPort:'frames',dir:'out',type:'frames'}});assert.equal(app.checkpoint(),false);
 await app.fieldInput({wcField:'name'},{value:'Should not change'});await app.fieldChange({wcFeature:'smoking'},{checked:false});await app.fieldChange({wcSensitivity:'fight'},{value:'standard'});await app.fieldChange({wcNodeEnabled:''},{checked:false});await app.fieldChange({wcBinding:'frames',wcEdgeId:'input-events'},{value:''});
 await app.handleAction('undo',{});await app.handleAction('arrange',{});await app.handleAction('delete',{});
 const keyboardTarget={closest:()=>null};await app.dispatch('keydown',keyboardTarget,{key:'Delete'});await app.dispatch('keydown',keyboardTarget,{key:'z',ctrlKey:true});
 assert.deepEqual(app.graph,before);assert.equal(app.history().length,history);assert.equal(app.pending,null);
});
test('a runtime lock arriving during node or library drag prevents the final drop mutation',async()=>{
 const app=eventEditorFixture(),before=clone(app.graph);app.el.ownerDocument.elementFromPoint=()=>({closest:selector=>selector==='.wc-board'?{}:null});app.data.cameras[0].status={analysis_state:'draining'};
 for(const drag of [{kind:'node',id:'events',startX:0,startY:0,x:0,y:0,moved:true},{kind:'add',type:'summary',startX:0,startY:0,moved:true}]){app.drag=drag;app.pointerUp({clientX:200,clientY:200});assert.deepEqual(app.graph,before);assert.equal(app.history().length,0);}
 const heading={dataset:{wcSelect:'events'},setPointerCapture(){}};const target={closest:selector=>selector==='[data-wc-select]'?heading:selector==='.wc-board'?{}:null};app.drag=null;await app.dispatch('pointerdown',target,{button:0,clientX:0,clientY:0,pointerId:1});assert.equal(app.drag,null);
});
test('activity in another assigned workflow leaves an unrelated workflow editable',async()=>{
 const app=eventEditorFixture();app.data.cameras[0].status={analysis_state:'running'};app.workflowId='workflow-2';app.selected='summary';assert.equal(app.editLocked(),false);const before=app.graph.nodes.length;
 app.add('summary');assert.equal(app.graph.nodes.length,before+1);await app.fieldInput({wcField:'prompt'},{value:'Allowed in the unrelated workflow.'});assert.equal(app.graph.nodes.find(n=>n.id===app.selected).config.prompt,'Allowed in the unrelated workflow.');
});
test('health polling locks editing without replacing pending graph, focused input or revision',async()=>{
 const app=eventEditorFixture();await app.fieldInput({wcField:'name'},{value:'Pending typed name'});const before=clone(app.graph),revision=app.document.revision,input={value:'Pending typed name'},history=app.history().length;app.el.ownerDocument.activeElement=input;let renders=0;app.render=()=>renders++;
 app.request=async()=>({document:{...clone(app.document),revision:999},event_catalog:clone(app.data.event_catalog),cameras:[{id:'camera01',status:{analysis_state:'starting'}}]});await app.refreshMetadata();
 assert.equal(app.editLocked(),true);assert.deepEqual(app.graph,before);assert.equal(app.document.revision,revision);assert.equal(app.history().length,history);assert.equal(app.el.ownerDocument.activeElement,input);assert.equal(input.value,'Pending typed name');assert.equal(renders,0);
 await app.fieldInput({wcField:'name'},{value:'Blocked after polling'});assert.equal(currentEvents(app).config.name,'Pending typed name');
});
test('failed health polling treats assigned camera state as unknown and retains pending edits',async()=>{
 const app=eventEditorFixture();await app.fieldInput({wcField:'name'},{value:'Retain during status outage'});const before=clone(app.graph);app.request=async()=>{throw new Error('status unavailable');};await app.refreshMetadata();
 assert.equal(app.editLocked(),true);assert.deepEqual(app.graph,before);await app.fieldChange({wcFeature:'smoking'},{checked:false});assert.deepEqual(app.graph,before);
});
test('Apply rejected by a camera-start race keeps edits and updates the edit lock',async()=>{
 const app=eventEditorFixture();await app.fieldInput({wcField:'name'},{value:'Keep after race'});const before=clone(app.graph);let requests=0;
 app.request=async()=>{requests++;app.data.cameras[0].status={analysis_state:'running'};const error=new Error('Stop analysis before applying changes.');error.status=409;throw error;};
 await app.handleAction('apply',{});assert.equal(requests,1);assert.deepEqual(app.graph,before);assert.equal(app.dirty(),true);assert.equal(app.editLocked(),true);assert(!/reload/i.test(app.notice));assert.equal(app.busy,false);
});
test('Apply target dialog names the workflow and keeps existing assigned groups mandatory',async()=>{
 const app=eventEditorFixture();app.document.groups.push({id:'group-2',name:'Yard group',workflow_id:'workflow-2',camera_ids:['camera02'],enabled:true,block_enabled:{},camera_overrides:{},applied_graph:graph()});app.data.cameras.push({id:'camera02',status:{analysis_state:'idle'}});await app.fieldInput({wcField:'name'},{value:'Pending event configuration'});const requests=[];
 app.request=async(url,options)=>{const body=JSON.parse(options.body);requests.push(body);const response=clone(app.data);response.document.revision++;response.document.workflows[0].graph=clone(body.graph);return response;};
 await withImmediateAnimation(()=>app.handleAction('assign-groups',{}));assert.equal(app.dialog.kind,'apply');assert.deepEqual(app.dialog.groups,['group-1']);const html=app.renderDialog();assert(html.includes('Workflow to apply'));assert(html.includes('Entrance'));assert.match(html,/data-wc-apply-group="group-1"[^>]*checked/);assert.match(html,/data-wc-apply-group="group-1"[^>]*disabled/);assert(!html.includes('Save the draft'));assert.doesNotMatch(html,/data-wc-action="confirm-apply"[^>]*disabled/);
 await app.fieldChange({wcApplyGroup:'group-1'},{checked:false});assert(app.dialog.groups.includes('group-1'));await app.fieldChange({wcApplyGroup:'group-2'},{checked:true});
 await app.handleAction('confirm-apply',{});assert.equal(requests.length,1);assert.equal(requests[0].action,'apply');assert.deepEqual(requests[0].group_ids.slice().sort(),['group-1','group-2']);assert.equal(requests[0].graph.nodes.find(n=>n.type==='events').config.name,'Pending event configuration');
});
test('Ctrl Save applies the pending graph even while a configuration text field has focus',async()=>{
 const app=eventEditorFixture();const target=await app.fieldInput({wcField:'name'},{value:'Apply from focused field'});target.closest=()=>({});const requests=[];app.request=async(url,options)=>{const body=JSON.parse(options.body);requests.push(body);const response=clone(app.data);response.document.workflows[0].graph=clone(body.graph);response.document.revision++;return response;};
 await app.dispatch('keydown',target,{key:'s',ctrlKey:true});await Promise.resolve();assert.equal(requests.length,1);assert.equal(requests[0].action,'apply');assert.equal(requests[0].graph.nodes.find(n=>n.type==='events').config.name,'Apply from focused field');
});
test('running workflow renders disabled authoring fields and Apply controls',()=>{
 const app=eventEditorFixture();app.data.cameras[0].status={analysis_state:'running'};const html=canvasHtml(app);assert.match(html,/data-wc-edit-fields[^>]*disabled/);assert.match(html,/data-wc-action="apply"[^>]*disabled/);assert.match(html,/data-wc-action="undo"[^>]*disabled/);assert.match(html,/data-wc-add="summary"[^>]*disabled/);
});
test('local test locks editing immediately before camera status polling and unlocks after completion',async()=>{
 const app=eventEditorFixture(),pending=deferred();await app.fieldChange({wcSensitivity:'fight'},{value:'high'});const before=clone(app.graph),history=app.history().length;let requests=0;app.request=async()=>{requests++;return pending.promise;};const run=app.runTest();assert.equal(app.testBusy,true);assert.equal(app.data.cameras[0].status.analysis_state,'idle');assert.equal(app.editLocked(),true);
 app.add('summary');app.deleteSelected();await app.fieldInput({wcField:'name'},{value:'Blocked during local test'});await app.fieldChange({wcFeature:'smoking'},{checked:false});await app.handleAction('undo',{});await app.handleAction('apply',{});assert.deepEqual(app.graph,before);assert.equal(app.history().length,history);assert.equal(requests,1);
 pending.resolve({outputs:[{node_id:'events',result:{detected:false}}]});await run;assert.equal(app.testBusy,false);assert.equal(app.editLocked(),false);assert.equal(app.testResult.outputs.length,1);await app.fieldInput({wcField:'name'},{value:'Editable after test'});assert.equal(currentEvents(app).config.name,'Editable after test');
});
test('status unlock reenables authoring controls while unselected sensitivities and singleton adds stay disabled',()=>{
 const app=eventEditorFixture(),fields=Array.from({length:5},()=>({disabled:false,dataset:{}})),selectedSensitivity={disabled:false,dataset:{wcSensitivity:'smoking'}},unselectedSensitivity={disabled:true,dataset:{wcSensitivity:'fall'}},summaryAdd={disabled:false,dataset:{wcAdd:'summary'}},eventsAdd={disabled:true,dataset:{wcAdd:'events'}},buttons=Object.fromEntries(['apply','assign-groups','undo','delete','arrange'].map(action=>[action,{disabled:false}])),notice={hidden:true,textContent:''};app.history().push({graph:clone(app.graph),selected:app.selected});
 app.el.querySelectorAll=selector=>selector.includes('[data-wc-edit-fields]')?fields:selector==='[data-wc-sensitivity]'?[selectedSensitivity,unselectedSensitivity]:selector==='[data-wc-add]'?[summaryAdd,eventsAdd]:[];app.el.querySelector=selector=>selector==='[data-wc-edit-lock]'?notice:buttons[selector.match(/^\[data-wc-action=([a-z-]+)\]$/)?.[1]]||null;
 app.data.cameras[0].status={analysis_state:'running'};app.updateRuntimeLocks();assert(fields.every(field=>field.disabled));assert.equal(selectedSensitivity.disabled,true);assert.equal(summaryAdd.disabled,true);assert(Object.values(buttons).every(button=>button.disabled));assert.equal(notice.hidden,false);
 app.data.cameras[0].status={analysis_state:'idle'};app.updateRuntimeLocks();assert(fields.every(field=>field.disabled===false));assert.equal(selectedSensitivity.disabled,false);assert.equal(unselectedSensitivity.disabled,true);assert.equal(Boolean(summaryAdd.disabled),false);assert.equal(eventsAdd.disabled,true);assert(Object.values(buttons).every(button=>button.disabled===false));assert.equal(notice.hidden,true);
});

test('group test excludes group-disabled tracking while events and summaries remain testable',async()=>{
 const {app,tracker}=await connectedTrackingFixture(false);app.group.applied_graph=clone(app.graph);app.mode='groups';app.group.block_enabled[tracker.id]=false;const applied=clone(app.graph);let body;
 assert.equal(app.trackingUnavailable(),false);assert.equal(app.testEnabled(),true);assert.deepEqual(validateGraph(app.testGraph(),{complete:true,catalog:app.catalog()}),[]);
 app.request=async(url,options)=>{assert.equal(url,'/v1/workflows/test');body=JSON.parse(options.body);return{outputs:[{node_id:'events'},{node_id:'summary'}]};};await app.runTest();
 assert.equal(body.graph.nodes.find(node=>node.id===tracker.id).enabled,false);for(const id of ['events','summary'])assert.equal(body.graph.nodes.find(node=>node.id===id).enabled,true);assert.deepEqual(app.graph,applied);assert.equal(app.testResult.outputs.length,2);
 assert(!app.renderTest().includes('data-wc-tracking-capability'),'a disabled tracking branch does not show an unavailable-service warning in Test');
});

test('group test uses the selected camera tracking override without changing sibling cameras',async()=>{
 const {app,tracker}=await connectedTrackingFixture(false);app.group.applied_graph=clone(app.graph);app.mode='groups';app.group.camera_ids.push('camera02');app.data.cameras.push({id:'camera02',enabled:true,status:{analysis_state:'idle'}});app.group.camera_overrides.camera01={enabled:null,block_enabled:{[tracker.id]:false}};let requests=0,body;
 app.request=async(url,options)=>{requests++;body=JSON.parse(options.body);return{outputs:[{node_id:'events'}]};};await app.runTest();assert.equal(requests,1);assert.equal(body.camera_id,'camera01');assert.equal(body.graph.nodes.find(node=>node.id===tracker.id).enabled,false);
 await app.fieldChange({wcTestCamera:''},{value:'camera02'});assert.equal(app.trackingUnavailable(),true);await app.runTest();assert.equal(requests,1,'camera02 still requires configured tracking');assert.equal(app.graph.nodes.find(node=>node.id===tracker.id).enabled,true,'the applied snapshot remains enabled');
});

test('disabled group, camera or output prevents group Test without making an inference request',async()=>{
 for(const gate of ['group','camera','output']){
  const {app}=await connectedTrackingFixture(false);app.group.applied_graph=clone(app.graph);app.mode='groups';if(gate==='group')app.group.enabled=false;else if(gate==='camera')app.group.camera_overrides.camera01={enabled:false,block_enabled:{}};else app.group.block_enabled.out=false;
  const applied=clone(app.graph);let requests=0;app.request=async()=>{requests++;return{outputs:[]};};assert.equal(app.testEnabled(),false,gate);assert.equal(app.trackingUnavailable(),false,gate);await app.runTest();assert.equal(requests,0,gate);assert.equal(app.testBusy,false);assert.match(app.notice,/enabled workflow output|启用的输出路径/);assert.deepEqual(app.graph,applied);
 }
});


test('editing a group can remove an existing unavailable camera without offering new unavailable assignments',()=>{
 const app=appFixture();app.data.cameras[0].enabled=false;app.data.cameras.push({id:'camera02',label:'Unavailable extra',enabled:false,status:{analysis_state:'idle'}});app.dialog={kind:'edit-group',id:'group-1',name:'Entrance',cameras:['camera01']};
 let html=app.renderDialog();assert.match(html,/data-wc-group-camera="camera01"[^>]*checked/);assert.doesNotMatch(html,/data-wc-group-camera="camera01"[^>]*disabled/);assert.match(html,/data-wc-group-camera="camera02"[^>]*disabled/);
 app.dialog={kind:'new-group',name:'New group',cameras:[]};html=app.renderDialog();for(const id of ['camera01','camera02'])assert.match(html,new RegExp('data-wc-group-camera="'+id+'"[^>]*disabled'));
});
test('live output refresh disables its button and reenables a replacement button after a pending rerender',async()=>{
 for(const fail of [false,true]){
  const app=appFixture();app.mode='groups';app.liveOpen=true;app.scheduleLive=()=>{};let button={disabled:false};const content={innerHTML:''};app.el.querySelector=selector=>selector==='[data-wc-action=refresh-live]'?button:selector==='.wc-live-content'?content:null;const pending=deferred();app.request=()=>pending.promise;
  const refresh=app.refreshLive();assert.equal(app.liveBusy,true);assert.equal(button.disabled,true);
  button={disabled:true};if(fail)pending.reject(new Error('Inference unavailable'));else pending.resolve({items:[],outputs:[]});await refresh;
  assert.equal(app.liveBusy,false);assert.equal(button.disabled,false);if(fail)assert.match(content.innerHTML,/Inference unavailable/);
 }
});
test('Fit includes the supported 64-block graph and zooming in continues from its fitted scale',()=>{
 const nodes=Array.from({length:64},(_,index)=>({x:330,y:60+index*195,height:190}));
 for(const [width,height] of [[900,600],[390,420]]){
  const app=appFixture();app.view=fitViewport(nodes,width,height);assert(app.view.scale<.12);
  for(const n of nodes){assert(n.x*app.view.scale+app.view.x>=0);assert(n.y*app.view.scale+app.view.y>=0);assert((n.x+176)*app.view.scale+app.view.x<=width);assert((n.y+190)*app.view.scale+app.view.y<=height);}
  const board={getBoundingClientRect:()=>({left:0,top:0,width,height})};app.el.querySelector=()=>board;app.layout=()=>{};const scale=app.view.scale;app.zoom(1.2);assert.equal(app.view.scale,scale*1.2);
 }
});
