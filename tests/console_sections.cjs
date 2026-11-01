const assert=require("node:assert/strict");
const {test}=require("node:test");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const source=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8").match(/<script id="console-sections-script">([\s\S]*?)<\/script>/)[1];
const context=vm.createContext({});vm.runInContext(source,context);
const {organizeSettings}=context.GoodvisionConsoleSections;

// The fixture parses the actual settings markup, so adding or relocating a
// control in dashboard.html is covered by the same identity/preservation check.
class Element{
  constructor(tag="div"){
    this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.dataset={};
    this.className="";this.hidden=false;this.handlers={};this.value="";
    this.classList={contains:name=>this.className.split(/\s+/).includes(name),add:name=>{if(!this.classList.contains(name))this.className+=(this.className?" ":"")+name;}};
  }
  append(...nodes){for(const node of nodes){node.remove();node.parentElement=this;this.children.push(node);}}
  insertBefore(node,before){node.remove();node.parentElement=this;this.children.splice(this.children.indexOf(before),0,node);}
  remove(){if(this.parentElement){const parent=this.parentElement;parent.children=parent.children.filter(node=>node!==this);this.parentElement=null;}}
  setAttribute(key,value){this.attributes[key]=value;if(key==="id")this.id=value;if(key==="class")this.className=value;}
  matches(selector){return selector==="[id]"?Boolean(this.id):selector.startsWith(".")?this.classList.contains(selector.slice(1)):selector.startsWith("#")?this.id===selector.slice(1):this.tagName===selector.toUpperCase();}
  closest(selector){for(let node=this;node;node=node.parentElement)if(node.matches(selector))return node;return null;}
  querySelectorAll(selector){return this.children.flatMap(node=>[...(node.matches(selector)?[node]:[]),...node.querySelectorAll(selector)]);}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  get nextElementSibling(){return this.parentElement?.children[this.parentElement.children.indexOf(this)+1]||null;}
  addEventListener(event,fn){this.handlers[event]=fn;}
  focus(options){this.focused=options;}
  scrollIntoView(options){this.scrolled=options;}
}
function fixture(){
  const html=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/dashboard.html"),"utf8");
  const start=html.indexOf('<section class="panel deployment-panel">');
  const markup=html.slice(start,html.indexOf('<section id="cloud-route-panel"',start));
  const page=new Element("section");page.id="console-settings";
  const stack=[page],voidTags=new Set(["input","br","hr"]);
  for(const match of markup.matchAll(/<\/?([a-z][a-z0-9-]*)([^>]*)>/gi)){
    const tag=match[1].toLowerCase();
    if(match[0].startsWith("</")){
      const index=stack.map(node=>node.tagName).lastIndexOf(tag.toUpperCase());
      if(index>0)stack.length=index;continue;
    }
    if(tag==="option"&&stack.at(-1)?.tagName==="OPTION")stack.pop();
    const element=new Element(tag);
    for(const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)){
      element.setAttribute(attr[1],attr[2]??"");
      if(attr[1]==="hidden")element.hidden=true;
    }
    stack.at(-1).append(element);if(!voidTags.has(tag))stack.push(element);
  }
  const document={getElementById:id=>page.id===id?page:page.querySelector("#"+id),createElement:tag=>new Element(tag)};
  return {document,page,panel:page.querySelector(".deployment-panel")};
}

test("settings move each original control once without replacing drafts or handlers",()=>{
  const view=fixture(),original=view.page.querySelectorAll("[id]");
  const input=view.document.getElementById("inference-server-url"),save=view.document.getElementById("save-deployment-button");
  input.value="http://unsaved:8000";input.onchange=()=>"draft changed";save.onclick=()=>"existing save";
  const saveFooter=save.parentElement,status=view.document.getElementById("deployment-note");
  assert.equal(organizeSettings(view.document),true);
  for(const node of original){assert.equal(view.document.getElementById(node.id),node,node.id);assert.equal(view.page.querySelectorAll("#"+node.id).length,1,node.id);}
  assert.equal(input.value,"http://unsaved:8000");assert.equal(input.onchange(),"draft changed");
  assert.equal(save.onclick(),"existing save");assert.equal(save.parentElement,saveFooter);
  assert.equal(view.panel.querySelectorAll(".settings-save-actions").length,1);
  assert.equal(status.parentElement,saveFooter);
  assert.equal(view.document.getElementById("reset-deployment-button").parentElement,saveFooter);
  assert.equal(view.panel.querySelector(".deployment-fields"),null);
  assert.equal(view.panel.querySelector(".deployment-options"),null);
  assert.equal(view.panel.querySelectorAll(".settings-section").length,5);
});

test("settings preserve conditional service wrappers and put controls in their own groups",()=>{
  const view=fixture(),get=id=>view.document.getElementById(id);
  const cloud=get("cloud-model-settings"),server=get("tracking-server-fields"),key=get("tracking-node-key-fields"),legacy=get("local-inference-enabled").parentElement;
  organizeSettings(view.document);
  const expected={analysis:["inference-mode","inference-server-fields"],tracking:["tracking-mode","tracking-server-fields","tracking-node-key-fields","cloud-gpu-panel","tracking-sample-fps","tracking-sample-fps-help","local-inference-enabled","local-tracking-enabled","local-gpu-status","local-gpu-help","processing-method-field","richlabel-model-field","richlabel-model-note"],video:["default-stream-url","default-stream-status","default-camera-layout"],credentials:["default-api-key","default-api-key-status","save-api-key-button","clear-api-key-button"],startup:["default-dry-run","default-autostart","deployment-source-status"]};
  for(const [group,ids] of Object.entries(expected))for(const id of ids)assert.equal(get(id).closest(".settings-section").id,"settings-"+group,id);
  assert.equal(key.parentElement,server);assert.equal(server.hidden,true);assert.equal(key.hidden,true);
  assert.equal(get("default-api-key").closest("#cloud-model-settings"),cloud);
  assert.equal(get("clear-api-key-button").closest("#cloud-model-settings"),cloud);
  assert.equal(get("local-inference-enabled").parentElement,legacy);assert.equal(legacy.hidden,true);
  assert.equal(get("local-tracking-enabled").parentElement,legacy);
  assert.equal(get("default-dry-run").closest("label").nextElementSibling.classList.contains("secret-status"),true);
});

test("section jumps keep the console route and idempotent initialization preserves nodes",()=>{
  const view=fixture();organizeSettings(view.document);
  const nodes=view.page.querySelectorAll("[id]"),links=view.panel.querySelectorAll(".settings-index-link");
  assert.equal(links.length,5);let prevented=false;
  links[0].handlers.click({preventDefault(){prevented=true;}});
  assert.equal(prevented,true);assert.ok(view.document.getElementById("settings-analysis").focused);
  assert.equal(organizeSettings(view.document),false);
  assert.deepEqual(view.page.querySelectorAll("[id]"),nodes);
});

test("diagnostic iframe settings remain untouched and partial pages are harmless",()=>{
  const view=fixture(),nodes=view.page.querySelectorAll("[id]");
  assert.equal(organizeSettings(view.document,{diagnostic:true}),false);
  assert.deepEqual(view.page.querySelectorAll("[id]"),nodes);
  assert.equal(organizeSettings({getElementById:()=>null}),false);
  assert.equal(organizeSettings(null),false);
});
