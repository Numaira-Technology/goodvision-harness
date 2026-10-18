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
test('incomplete drafts remain saveable while Apply requires usable input wiring',()=>{
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
test('drafts and applied snapshots stay isolated across workflow switches',()=>{
 const app=appFixture(),applied=clone(app.group.applied_graph);app.draft.graph.nodes[1].config.name='Draft face';assert.equal(app.dirty(),true);assert.deepEqual(app.group.applied_graph,applied);
 app.workflowId='workflow-2';assert.equal(app.dirty(),false);app.draft.graph.nodes[1].config.name='Other draft';app.workflowId='workflow-1';assert.equal(app.graph.nodes[1].config.name,'Draft face');assert.equal(app.anyDirty(),true);
 app.mode='groups';assert.equal(app.graph.nodes[1].config.name,'face');
});
test('commands use document revision, save only the selected draft and preserve server snapshot',async()=>{
 const app=appFixture();app.draft.graph.nodes[1].config.name='Saved face';app.workflowId='workflow-2';app.draft.graph.nodes[1].config.name='Keep unsaved';app.workflowId='workflow-1';
 let request;app.request=async(url,options)=>{request={url,body:JSON.parse(options.body)};const response=clone(app.data);response.document.revision++;response.document.workflows[0].graph=clone(request.body.graph);return response;};
 await app.handleAction('save',{});assert.equal(request.url,'/v1/workflows/command');assert.equal(request.body.action,'save_workflow');assert.equal(request.body.revision,4);assert.equal(request.body.id,'workflow-1');assert.equal(app.document.revision,5);assert.equal(app.dirty('workflow-1'),false);assert.equal(app.dirty('workflow-2'),true);assert.equal(app.group.applied_graph.nodes[1].config.name,'face');
});
test('failed or conflicting saves keep local edits available for recovery',async()=>{
 const app=appFixture();app.draft.graph.nodes[1].config.name='Keep me';app.request=async()=>{const e=new Error('revision conflict');e.status=409;throw e;};
 assert.equal(await app.command('save_workflow',{id:app.workflowId,graph:clone(app.graph)},app.workflowId),false);assert.equal(app.dirty(),true);assert.equal(app.graph.nodes[1].config.name,'Keep me');assert.equal(app.busy,false);assert.match(app.notice,/Reload/);
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
