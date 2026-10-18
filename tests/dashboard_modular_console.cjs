const assert=require('node:assert/strict');
const {test}=require('node:test');
const vm=require('node:vm');
const {html,script}=require('./helpers/dashboard.cjs');
const modular=html.match(/<script id="modular-console-script">([\s\S]*?)<\/script>/)[1];

function fixture({hash='',search='',diagnostic=false,diagnosticView='diagnostics'}={}){
  class Element{
    constructor(dataset={}){this.dataset=dataset;this.hidden=false;this.value='';this.attributes={};this.children=[];this.handlers={};this.classList={contains:name=>this.nav&&name==='nav-item'};}
    setAttribute(name,value){this.attributes[name]=value;}
    removeAttribute(name){delete this.attributes[name];delete this[name];}
    addEventListener(name,fn){this.handlers[name]=fn;}
    append(...children){for(const child of children){if(child.parent)child.parent.children=child.parent.children.filter(item=>item!==child);child.parent=this;this.children.push(child);}}
    replaceChildren(...children){this.children=[];this.append(...children);}
    focus(){this.focused=true;}
    scrollIntoView(){}
  }
  const names=['overview','workflows','capabilities','events','outputs','settings'];
  const pages=names.map(name=>new Element({consolePage:name}));
  const links=names.map(name=>Object.assign(new Element({consoleGo:name}),{nav:true}));
  const outputKeys=attribute=>[...html.matchAll(new RegExp('data-output-'+attribute+'="([^"]+)"','g'))].map(match=>match[1]);
  const keys=outputKeys('path');
  const paths=keys.map(key=>new Element({outputPath:key}));
  const outputLinks=outputKeys('link').map(key=>new Element({outputLink:key}));
  const copyButtons=outputKeys('copy').map(key=>new Element({outputCopy:key}));
  const elements=new Map(),handlers={},copied=[];
  const element=id=>{if(!elements.has(id)){const node=new Element();node.id=id;elements.set(id,node);}return elements.get(id);};
  const document={body:{dataset:{},classList:{toggle(){}}},getElementById:element,createElement:()=>new Element(),querySelector:()=>element('scope-note'),querySelectorAll:selector=>({
    '[data-console-page]':pages,'[data-console-go]':links,'[data-output-path]':paths,
    '[data-output-link]':outputLinks,'[data-output-copy]':copyButtons,'.output-scope-note':[element('output-scope')],
  })[selector]||[],createRange:()=>({selectNodeContents(item){element('selection').value=item.textContent;}})};
  const location={hash,search,origin:'http://localhost:7900'};
  const context=vm.createContext({document,location,URL,URLSearchParams,GoodvisionDiagnostic:diagnostic,GoodvisionDiagnosticView:diagnosticView,
    navigator:{clipboard:{writeText:async text=>copied.push(text)}},
    getSelection:()=>({removeAllRanges(){},addRange(){}}),
    history:{pushState:(_,__,value)=>{location.hash=value;}},addEventListener:(name,fn)=>{handlers[name]=fn;},
  });
  vm.runInContext(script.slice(script.indexOf('function consoleJson'),script.indexOf('const ui=')),context);
  vm.runInContext(modular,context);
  const cameras=['camera01','camera02'].map((id,index)=>({id,label:'视频 '+(index+1),enabled:true,diagnostics_url:'/v1/fleet/cameras/'+id+'/diagnostics',
    outputs:Object.fromEntries(keys.map(key=>[key,'/v1/fleet/cameras/'+id+'/'+key]))}));
  const app={metadataOk:true,fleet:{cameras,analysis:{state:'idle'}},rows:new Map(cameras.map(camera=>[camera.id,{status:{connection_state:'disconnected'}}]))};
  return {api:context.GoodvisionConsole,context,element,pages,links,paths,outputLinks,copyButtons,app,copied,handlers,location};
}

test('navigation preserves existing controls, drafts and camera state and restores history',()=>{
  const view=fixture();view.api.sync(view.app);
  const input=view.element('inference-server-url');input.value='http://draft:8000';
  for(const link of view.links){link.handlers.click();assert.equal(view.pages.filter(page=>!page.hidden).length,1);}
  assert.equal(view.element('inference-server-url'),input);assert.equal(input.value,'http://draft:8000');
  assert.equal(view.location.hash,'#settings');assert.equal(view.links[5].attributes['aria-current'],'page');
  view.location.hash='#outputs';view.handlers.hashchange();assert.equal(view.api.currentPage(),'outputs');
  view.location.hash='#unknown';view.handlers.hashchange();assert.equal(view.api.currentPage(),'overview');
  assert.equal(view.app.fleet.analysis.state,'idle');
});

test('outputs follow selected camera and invalidate stale links on metadata failure',()=>{
  const view=fixture({hash:'#outputs'});view.api.sync(view.app);
  const picker=view.element('console-output-camera');picker.value='camera02';picker.handlers.change();
  assert(view.outputLinks.every(link=>link.href.includes('/camera02/')));
  view.api.sync(view.app);assert.equal(picker.value,'camera02');
  view.app.metadataOk=false;view.api.sync(view.app);
  assert(view.outputLinks.every(link=>!link.href));assert(view.copyButtons.every(button=>button.disabled));
  view.app.metadataOk=true;view.app.fleet.cameras.pop();view.api.sync(view.app);
  assert.equal(picker.value,'camera01');assert(view.outputLinks.every(link=>link.href.includes('/camera01/')));
});

test('output links reject external, script, template and malformed addresses',()=>{
  const {api}=fixture();
  for(const value of ['//outside.test/a','https://outside.test','javascript:alert(1)','/\\outside.test','/a/{id}','/a\n/b',null])assert.equal(api.safeAddress(value),null);
  assert.equal(api.safeAddress('/v1/results/latest'),'http://localhost:7900/v1/results/latest');
});

test('copy uses the selected camera address and clipboard denial selects readable text',async()=>{
  const view=fixture({hash:'#outputs'});view.api.sync(view.app);
  await view.copyButtons[0].handlers.click();assert.equal(view.copied[0],view.outputLinks[0].href);
  view.context.navigator.clipboard.writeText=async()=>{throw Error('denied');};
  await view.copyButtons[0].handlers.click();assert.equal(view.element('selection').value,view.paths[0].textContent);
  assert.match(view.element('console-copy-status').textContent,/Ctrl\+C/);
});

test('focused channel views reuse only their own sections without global navigation',()=>{
  const sections={
    results:['diagnostic-result-error','live-result-log'],
    insights:['overall-summary','task-live-judgment'],
    events:['scene-panel','event-archive-panel'],
    diagnostics:['connection-diagnostics','output-metrics','runtime-data-panels','scene-diagnostics','face-diagnostics-panel'],
    camera:['local-camera-panel','analysis-workspace']
  };
  for(const [diagnosticView,ids] of Object.entries(sections)){
    const view=fixture({diagnostic:true,diagnosticView});
    assert.equal(view.pages.filter(page=>!page.hidden).length,0);
    assert.equal(view.element('fleet-detail-console').hidden,true);
    assert.equal(view.context.document.body.dataset.diagnosticView,diagnosticView);
    const content=view.element('console-main').children.find(item=>item.id==='diagnostic-content');
    assert.deepEqual(content.children.map(item=>item.id),ids);
    assert.equal(view.element('event-history-title').textContent,'本路事件记录');
    assert.equal(view.links[0].handlers.click,undefined);
  }
});

test('unknown diagnostic view falls back to service diagnostics',()=>{
  const view=fixture({diagnostic:true,diagnosticView:'<script>'});
  assert.equal(view.context.document.body.dataset.diagnosticView,'diagnostics');
  const content=view.element('console-main').children[0];
  assert.ok(content.children.some(item=>item.id==='runtime-data-panels'));
  assert.ok(!content.children.some(item=>item.id==='event-archive-panel'));
});

test('old diagnostic wrappers select an allowlisted URL view without restarting the service',()=>{
  for(const [search,expected] of [['?view=results','results'],['?view=results&surface=insights','insights'],['?view=events','events'],['?view=camera','camera'],['?view=outputs','diagnostics']]){
    const view=fixture({diagnostic:true,diagnosticView:null,search});
    assert.equal(view.context.GoodvisionDiagnosticView,expected);
    assert.equal(view.context.document.body.dataset.diagnosticView,expected);
  }
  assert.ok(html.includes('body[data-diagnostic-view] #console-main>:not(#diagnostic-content)'));
});

test('pages expose one role each without placeholder navigation or primary-channel duplicates',()=>{
  const overview=html.slice(html.indexOf('id="console-overview"'),html.indexOf('id="console-workflows"'));
  assert.ok(overview.includes('id="fleet-panel"'));
  assert.ok(overview.includes('workspace-guide'));
  assert.ok(!overview.includes('class="pipeline-board"'));
  assert.ok(!overview.includes('class="configure-strip"'));
  const capabilities=html.slice(html.indexOf('id="console-capabilities"'),html.indexOf('id="console-events"'));
  assert.ok(!capabilities.includes('class="pipeline-board"'));
  assert.equal((capabilities.match(/data-event-presets/g)||[]).length,1);
  assert.ok(capabilities.indexOf('id="face-controls"')<capabilities.indexOf('id="capability-catalog-title"'));
  assert.ok(!capabilities.includes('规划中'));
  const events=html.slice(html.indexOf('id="console-events"'),html.indexOf('id="console-outputs"'));
  assert.ok(events.includes('id="console-event-camera"'));assert.ok(!events.includes('id="live-result-log"'));
  const outputs=html.slice(html.indexOf('id="console-outputs"'),html.indexOf('id="console-settings"'));
  for(const legacy of ['output-playback-panel','endpoint-grid','runtime-data-panels','scene-diagnostics','planned-card'])assert.ok(!outputs.includes(legacy));
  assert.ok(outputs.includes('data-output-path="video/stream.mjpeg"'));
  const settings=html.slice(html.indexOf('id="console-settings"'),html.indexOf('<div id="fleet-detail-console" hidden>'));
  assert.ok(settings.includes('id="fleet-detail-toggle"'));
  assert.ok(settings.includes('打开本机摄像头'));
});

test('debug rendering hides model identifiers recursively without changing business data or API objects',()=>{
  const {api}=fixture();
  const data={model:'private-engine',nested:{model_name:'private-name',inference_server_model:'custom-model',person_id:'custom-model'},confidence:.8};
  const rendered=JSON.parse(api.formatJson(data));
  assert.equal(rendered.model,'已隐藏');assert.equal(rendered.nested.model_name,'已隐藏');
  assert.equal(rendered.nested.inference_server_model,'已隐藏');assert.equal(rendered.nested.person_id,'custom-model');
  assert.equal(rendered.confidence,.8);assert.equal(data.model,'private-engine');
  assert.deepEqual(JSON.parse(api.formatJson({model:{name:'private',latency_seconds:1.2}})),{model:{name:'已隐藏',latency_seconds:1.2}});
});


test('console exposes shared catalog containers instead of a fixed event inventory',()=>{
  assert.equal((html.match(/id="archive-feature"/g)||[]).length,1);
  for(const label of ['ppe_violation','ground_litter']){
    assert.ok(!html.includes('data-feature="'+label+'"'));
    assert.ok(!html.includes('<option value="'+label+'">'));
  }
  assert.ok(html.includes('data-event-presets'));
  assert.ok(html.includes('/assets/event_catalog.js'));
});


test('replay navigation selects the requested channel and keeps one scoped archive surface',()=>{
  const view=fixture();view.api.sync(view.app);view.api.openEvents('camera02');
  assert.equal(view.api.currentPage(),'events');assert.equal(view.element('console-event-camera').value,'camera02');
  const host=view.element('console-event-view'),frame=host.children[0];
  assert.equal(frame.src,'http://localhost:7900/v1/fleet/cameras/camera02/diagnostics?view=events');
  assert.equal(host.children.length,1);view.api.sync(view.app);assert.equal(host.children[0],frame);
  view.element('console-event-camera').value='camera01';view.element('console-event-camera').handlers.change();
  assert.equal(host.children.length,1);assert.match(host.children[0].src,/camera01/);
  view.app.metadataOk=false;view.api.sync(view.app);assert.equal(host.children.length,0);
  view.app.metadataOk=true;view.app.fleet.cameras[0].enabled=false;view.api.sync(view.app);
  assert.equal(view.element('console-event-camera').value,'camera02');assert.match(host.children[0].src,/camera02/);
});

test('per-channel polling pauses outside the parent page and for hidden channel cards',()=>{
  const helper=script.slice(script.indexOf('function channelSurfaceActive'),script.indexOf('function consoleJson'));
  const row={card:{hidden:false}},parent={GoodvisionConsole:{currentPage:()=>parent.page},GoodvisionFleet:{app:{rows:new Map([['camera02',row]])}},page:'overview'};
  const window={parent},context=vm.createContext({window,GoodvisionDiagnostic:true,GoodvisionDiagnosticView:'results',GoodvisionDiagnosticCameraId:'camera02'});
  vm.runInContext(helper,context);assert.equal(context.channelSurfaceActive(),true);
  parent.page='settings';assert.equal(context.channelSurfaceActive(),false);
  parent.page='overview';row.card.hidden=true;assert.equal(context.channelSurfaceActive(),false);
  context.GoodvisionDiagnosticView='events';parent.page='events';assert.equal(context.channelSurfaceActive(),true);
  parent.page='outputs';assert.equal(context.channelSurfaceActive(),false);
});


test('legacy diagnostic view resolves before asynchronous runtime startup',()=>{
  const startup=script.slice(script.indexOf('if(globalThis.GoodvisionDiagnostic'),script.indexOf('function channelSurfaceActive'));
  for(const [search,expected] of [['?view=results','results'],['?view=results&surface=insights','insights'],['?view=events','events'],['?view=unknown','diagnostics']]){
    const context=vm.createContext({GoodvisionDiagnostic:true,location:{search},URLSearchParams});
    vm.runInContext(startup,context);assert.equal(context.GoodvisionDiagnosticView,expected);
  }
  const configured=vm.createContext({GoodvisionDiagnostic:true,GoodvisionDiagnosticView:'results',location:{search:'?view=results&surface=insights'},URLSearchParams});
  vm.runInContext(startup,configured);assert.equal(configured.GoodvisionDiagnosticView,'insights');
  const scoped=fixture({diagnostic:true,diagnosticView:'results',search:'?view=results&surface=insights'});
  assert.equal(scoped.context.document.body.dataset.diagnosticView,'insights');
  assert.ok(html.indexOf('data-console-managed')<html.indexOf('<body>'));
});


test('frame result view mounts the actionable inference error outside its hidden summary section',()=>{
  const view=fixture({diagnostic:true,diagnosticView:'results'});
  const banner=view.element('diagnostic-result-error');
  const content=view.element('console-main').children.find(item=>item.id==='diagnostic-content');
  assert.equal(banner.parent,content);
  assert.ok(!content.children.includes(view.element('overall-summary')));
  const ui=new Proxy({}, {get:(_target,name)=>view.element('runtime-'+String(name))});
  Object.assign(view.context,{ui,activeStates:new Set(['starting','running']),
    updateRichAnnotationStatus(){},backgroundMessage:()=>'',backgroundRuns:()=>[],
    count:value=>String(value||0),timeLabel:value=>String(value),setChannel(){},setConnection(){},$:view.element});
  vm.runInContext(script.slice(script.indexOf('function runtimeErrorText('),script.indexOf('async function status()')),view.context);
  view.context.updateRuntimeSummary({pipeline_state:'running',pipeline:{branch_errors:{safety:'http_error'},
    branch_error_details:{safety:{message:'Inference authentication failed. (HTTP 401)'}}}});
  assert.equal(banner.hidden,false);assert.match(banner.textContent,/authentication.*401/);
  view.context.updateRuntimeSummary({pipeline_state:'running',pipeline:{branch_errors:{},branch_error_details:{}}});
  assert.equal(banner.hidden,true);assert.equal(banner.textContent,'');
});


test('workflow outputs and summary history open and copy the selected camera endpoints and clear stale links',async()=>{
 const view=fixture({hash:'#outputs'});view.api.sync(view.app);const picker=view.element('console-output-camera');picker.value='camera02';picker.handlers.change();
 for(const key of ['workflow/results','workflow/summaries']){
  const path=view.paths.find(item=>item.dataset.outputPath===key),link=view.outputLinks.find(item=>item.dataset.outputLink===key),copy=view.copyButtons.find(item=>item.dataset.outputCopy===key);
  assert(path&&link&&copy,key+' exposes its address, Open link and Copy button');const expected='http://localhost:7900/v1/fleet/cameras/camera02/'+key;
  assert.equal(path.textContent,expected);assert.equal(link.href,expected);assert.equal(copy.disabled,false);await copy.handlers.click();assert.equal(view.copied.at(-1),expected);
  delete view.app.fleet.cameras[1].outputs[key];view.api.sync(view.app);assert.equal(link.href,undefined);assert.equal(link.attributes['aria-disabled'],'true');assert.equal(copy.disabled,true);
 }
});
