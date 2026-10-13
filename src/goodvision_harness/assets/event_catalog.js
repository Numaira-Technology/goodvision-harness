/* Shared presentation contract. No built-in event inventory or detector rules. */
(function(root){
  "use strict";
  let events=[];
  const known=new Map();
  const key=value=>String(value||"").trim().toLowerCase().replace(/[\s-]+/g,"_");
  function parse(payload){
    const catalog=payload?.event_catalog;
    if(!catalog||catalog.version!==1||!Array.isArray(catalog.events))throw new Error("事件目录暂不可用，请刷新后重试。");
    const seen=new Set();
    return catalog.events.map(item=>{
      if(!item||typeof item.id!=="string"||!/^[a-z][a-z0-9_]*$/.test(item.id)||seen.has(item.id))throw new Error("事件目录格式无效。");
      seen.add(item.id);
      const options=item.sensitivity?.options;
      if(item.labels&&Object.values(item.labels).some(value=>typeof value!=="string"))throw new Error("事件名称格式无效。");
      if(item.sensitivity&&(!Array.isArray(options)||!options.length||options.some(option=>!option||typeof option.value!=="string"||!option.value)||
        new Set(options.map(option=>option.value)).size!==options.length||
        !options.some(option=>option.value===item.sensitivity.default)))throw new Error("事件参数格式无效。");
      return {...item,labels:{zh:item.id,en:item.id,...item.labels},aliases:Array.isArray(item.aliases)?item.aliases:[],
        sensitivity:item.sensitivity?{...item.sensitivity,options:options.map(option=>({...option}))}:null};
    });
  }
  function registerTranslations(items=events){
    // Keep names of retired presets for historical evidence; membership always
    // comes from the current response, independently for each camera.
    for(const item of items)known.set(item.id,item);
    const entries=items.map(item=>[item.labels.zh,item.labels.en]);
    for(const event of items)for(const option of event.sensitivity?.options||[]){
      if(option.labels?.zh&&option.labels?.en)entries.push([option.labels.zh,option.labels.en]);
      if(option.descriptions?.zh&&option.descriptions?.en)entries.push([option.descriptions.zh,option.descriptions.en]);
    }
    root.GoodvisionI18n?.addTranslations?.(Object.fromEntries(entries));
  }
  function accept(payload){const next=parse(payload);events=next;registerTranslations();return events;}
  function canonical(value){const normalized=key(value);return known.has(normalized)?normalized:[...known.values()].find(item=>item.aliases.includes(normalized))?.id||normalized;}
  function get(value){return known.get(canonical(value));}
  function label(value){const item=get(value);return item?.labels.zh||(/^[a-z][a-z0-9_]*$/.test(String(value))?String(value):"事件");}
  const api={parse,accept,registerTranslations,canonical,get,label,list:()=>events};
  root.GoodvisionEvents=api;
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
})(typeof globalThis!=="undefined"?globalThis:this);
