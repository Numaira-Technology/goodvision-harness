const assert=require('node:assert/strict');
const {test}=require('node:test');
const vm=require('node:vm');
const {html,script}=require('./helpers/dashboard.cjs');
const modular=html.match(/<script id="modular-console-script">([\s\S]*?)<\/script>/)[1];

function fixture({hash='',diagnostic=false}={}){
  class Element{
    constructor(dataset={}){this.dataset=dataset;this.hidden=false;this.value='';this.attributes={};this.children=[];this.handlers={};this.classList={contains:name=>this.nav&&name==='nav-item'};}
    setAttribute(name,value){this.attributes[name]=value;}
    removeAttribute(name){delete this.attributes[name];delete this[name];}
    addEventListener(name,fn){this.handlers[name]=fn;}
    replaceChildren(...children){this.children=children;}
    focus(){this.focused=true;}
    scrollIntoView(){}
  }
  const names=['overview','capabilities','events','outputs','settings'];
  const pages=names.map(name=>new Element({consolePage:name}));
  const links=names.map(name=>Object.assign(new Element({consoleGo:name}),{nav:true}));
  const keys=['video/annotated/stream.mjpeg','results/latest','video/display/frames'];
  const paths=keys.map(key=>new Element({outputPath:key}));
  const outputLinks=keys.map(key=>new Element({outputLink:key}));
  const copyButtons=keys.map(key=>new Element({outputCopy:key}));
  const elements=new Map(),handlers={},copied=[];
  const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const document={getElementById:element,createElement:()=>new Element(),querySelector:()=>element('scope-note'),querySelectorAll:selector=>({
    '[data-console-page]':pages,'[data-console-go]':links,'[data-output-path]':paths,
    '[data-output-link]':outputLinks,'[data-output-copy]':copyButtons,'.output-scope-note':[element('output-scope')],
  })[selector]||[],createRange:()=>({selectNodeContents(item){element('selection').value=item.textContent;}})};
  const location={hash,origin:'http://localhost:7900'};
  const context=vm.createContext({document,location,URL,GoodvisionDiagnostic:diagnostic,
    navigator:{clipboard:{writeText:async text=>copied.push(text)}},
    getSelection:()=>({removeAllRanges(){},addRange(){}}),
    history:{pushState:(_,__,value)=>{location.hash=value;}},addEventListener:(name,fn)=>{handlers[name]=fn;},
  });
  vm.runInContext(script.slice(script.indexOf('function consoleJson'),script.indexOf('const ui=')),context);
  vm.runInContext(modular,context);
  const cameras=['camera01','camera02'].map((id,index)=>({id,label:'视频 '+(index+1),enabled:true,
    outputs:Object.fromEntries(keys.map(key=>[key,'/v1/fleet/cameras/'+id+'/'+key]))}));
  const app={metadataOk:true,fleet:{cameras,analysis:{state:'idle'}},rows:new Map(cameras.map(camera=>[camera.id,{status:{connection_state:'disconnected'}}]))};
  return {api:context.GoodvisionConsole,context,element,pages,links,paths,outputLinks,copyButtons,app,copied,handlers,location};
}

test('navigation preserves existing controls, drafts and camera state and restores history',()=>{
  const view=fixture();view.api.sync(view.app);
  const input=view.element('inference-server-url');input.value='http://draft:8000';
  for(const link of view.links){link.handlers.click();assert.equal(view.pages.filter(page=>!page.hidden).length,1);}
  assert.equal(view.element('inference-server-url'),input);assert.equal(input.value,'http://draft:8000');
  assert.equal(view.location.hash,'#settings');assert.equal(view.links[4].attributes['aria-current'],'page');
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

test('diagnostic pages keep outputs and event review available without global navigation',()=>{
  const view=fixture({diagnostic:true});
  assert.deepEqual(view.pages.filter(page=>!page.hidden).map(page=>page.dataset.consolePage),['events','outputs']);
  assert.equal(view.element('event-history-title').textContent,'本路事件记录');
  assert.equal(view.links[0].handlers.click,undefined);
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
