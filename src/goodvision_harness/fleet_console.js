/* One console for independently managed cameras. Finite JPEG requests keep
 * browser connection slots available for control requests even with 12+ feeds. */
(function(root){
  "use strict";
  const ACTIVE=new Set(["starting","running","stopping","draining"]);
  const FEATURES=["fight","smoking","fall","weapon","stranger"];
  const STATUS={idle:"未连接",starting:"正在连接",running:"检测中",stopping:"正在停止",draining:"正在完成剩余分析",error:"连接异常",stopped:"已停止"};
  async function pooled(items,limit,action){
    let cursor=0;
    await Promise.all(Array.from({length:Math.min(limit,items.length)},async()=>{
      while(cursor<items.length){const item=items[cursor++];await action(item);}
    }));
  }
  function safeEndpoint(value){
    return typeof value==="string"&&/^\/(?!\/)/.test(value)&&!value.includes("\\")?value:null;
  }
  function normalizeFleet(data){
    const allowed=data.multi_camera_allowed===true;
    const cameras=(Array.isArray(data.cameras)?data.cameras:[]).filter(camera=>camera&&typeof camera.id==="string");
    const seen=new Set();
    return {allowed,capacity:Math.max(1,Number(data.capacity)||cameras.length||1),reason:String(data.reason||""),
      cameras:cameras.filter(camera=>{if(seen.has(camera.id))return false;seen.add(camera.id);return true;}).map((camera,index)=>({
        ...camera,label:String(camera.label||"摄像头 "+(index+1)),
        enabled:camera.enabled!==false&&(allowed||camera.own===true||index===0),
        status_url:safeEndpoint(camera.status_url),start_url:safeEndpoint(camera.start_url),
        stop_url:safeEndpoint(camera.stop_url),settings_url:safeEndpoint(camera.settings_url),
        preview_url:safeEndpoint(camera.preview_url),console_url:safeEndpoint(camera.console_url)
      }))};
  }
  class FleetConsole{
    constructor({document,fetch,clock=()=>Date.now(),setTimeout,clearTimeout,AbortController,URL,onDetail=()=>{},onState=()=>{}}){
      Object.assign(this,{document,fetch,clock,setTimeout,clearTimeout,AbortController,URL,onDetail,onState});
      this.rows=new Map();this.visible=new Set();this.fleet=null;this.metadataOk=false;this.loading=false;
      this.closed=false;this.bulk=false;this.lastMetadata=0;this.previewGeneration=0;this.refreshPending=false;
      this.compact=false;this.initializedLayout=false;
      this.ui={};for(const name of ["mode","capacity","add","start-all","stop-all","view-toggle","count","grid","note","detail-toggle","detail-console"]){
        this.ui[name]=document.getElementById("fleet-"+name);
      }
      this.ui.add.onclick=()=>this.add();this.ui["start-all"].onclick=()=>this.all("start");
      this.ui["stop-all"].onclick=()=>this.all("stop");
      this.ui["view-toggle"].onclick=()=>{this.compact=!this.compact;this.update();};
      this.ui["detail-toggle"].onclick=()=>this.showDetail(this.ui["detail-console"].hidden);
    }
    element(tag,className,text){
      const element=this.document.createElement(tag);if(className)element.className=className;
      if(text!==undefined)element.textContent=text;return element;
    }
    say(message,error=false){this.ui.note.textContent=message;this.ui.note.className="status fleet-summary"+(error?" error":"");}
    async request(url,options={},timeoutMs=10000){
      if(!url)throw new Error("该通道尚未提供操作地址");
      const controller=new this.AbortController(),timer=this.setTimeout(()=>controller.abort(),timeoutMs);
      try{
        const response=await this.fetch(url,{cache:"no-store",...options,signal:controller.signal});
        if(!response.ok){
          let detail="服务返回 "+response.status;
          try{const data=await response.json();if(typeof data.detail==="string")detail=data.detail;}catch(_){}
          throw new Error(detail);
        }
        return await response.json();
      }finally{this.clearTimeout(timer);}
    }
    async refresh(){
      if(this.closed)return;
      if(this.loading){this.refreshPending=true;return;}
      this.loading=true;
      const hadMetadataError=!this.metadataOk;
      try{
        this.applyFleet(normalizeFleet(await this.request("/v1/fleet")));
        this.metadataOk=true;this.lastMetadata=this.clock();if(hadMetadataError)this.say("");
      }catch(error){
        this.metadataOk=false;this.say("无法读取摄像头配置："+error.message+"。恢复连接后会自动重试。",true);
        if(!this.fleet){this.ui.mode.textContent="服务连接中断";this.ui.capacity.textContent="可使用下方详细控制检查本路服务。";}
      }finally{this.loading=false;this.update();}
      if(this.refreshPending){this.refreshPending=false;await this.refresh();return;}
      await this.pollStatus(true);
    }
    applyFleet(fleet){
      this.fleet=fleet;
      const ids=new Set(fleet.cameras.map(camera=>camera.id));
      for(const [id,row] of this.rows){if(!ids.has(id)){this.releasePreview(row);row.card.remove();this.rows.delete(id);this.visible.delete(id);}}
      for(const camera of fleet.cameras){
        let row=this.rows.get(camera.id);
        if(!row){row=this.makeRow(camera);this.rows.set(camera.id,row);}
        row.camera=camera;
        if(camera.enabled&&(camera.own===true||!this.visible.size))this.visible.add(camera.id);
        if(!camera.enabled&&!ACTIVE.has(row.status?.pipeline_state))this.visible.delete(camera.id);
        row.card.hidden=!this.visible.has(camera.id);this.updateRow(row);
      }
    }
    makeRow(camera){
      const card=this.element("article","fleet-card"),head=this.element("div","fleet-card-head");
      const title=this.element("h3","",camera.label),state=this.element("span","fleet-camera-state","正在检查");
      head.append(title,state);
      const viewport=this.element("div","fleet-preview"),image=this.element("img");image.alt=camera.label+" 实时人物描边画面";image.hidden=true;
      const placeholder=this.element("div","fleet-preview-note","连接后在此显示实时画面");viewport.append(image,placeholder);
      const fields=this.element("div","fleet-card-fields"),source=this.element("label","field");
      source.append(this.element("span","","网络视频地址"));
      const input=this.element("input");input.type="text";input.autocomplete="off";input.spellcheck=false;
      input.placeholder="RTSP / RTMP / WHEP 地址";input.setAttribute("aria-label",camera.label+" 视频地址");source.append(input);
      const layoutField=this.element("label","field");layoutField.append(this.element("span","","摄像头类型"));
      const layout=this.element("select");layout.setAttribute("aria-label",camera.label+" 摄像头类型");
      for(const [value,label] of [["mono","单目 · 完整画面"],["stereo","双目 · 使用左半幅"]]){const option=this.element("option","",label);option.value=value;layout.append(option);}
      layout.value="mono";layoutField.append(layout);
      const sourceNote=this.element("div","fleet-source","正在读取已保存的视频源…");
      const buttons=this.element("div","fleet-card-buttons"),start=this.element("button","btn primary","连接"),stop=this.element("button","btn danger","停止");
      const save=this.element("button","btn","保存地址"),detail=this.element("button","btn","详细控制"),remove=this.element("button","btn fleet-remove","移除");
      for(const button of [start,stop,save,detail,remove]){button.type="button";buttons.append(button);}
      remove.setAttribute("aria-label","移除 "+camera.label);
      const message=this.element("div","fleet-card-detail");message.setAttribute("role","status");
      fields.append(source,layoutField,sourceNote,buttons,message);card.append(head,viewport,fields);this.ui.grid.append(card);
      const row={camera,card,title,state,image,placeholder,input,layout,sourceNote,start,stop,save,detail,remove,message,
        status:null,busy:false,error:"",notice:"",previewUrl:null,previewError:false,generation:0,layoutDirty:false};
      start.onclick=()=>this.start(row);stop.onclick=()=>this.stop(row);save.onclick=()=>this.save(row);
      remove.onclick=()=>this.remove(row);detail.onclick=()=>{
        if(row.camera.own)this.showDetail(true);
        else if(row.camera.console_url)root.open?.(row.camera.console_url,"_blank","noopener,noreferrer");
      };
      input.oninput=()=>{row.notice="";row.error="";this.updateRow(row);};layout.onchange=()=>{row.layoutDirty=true;};
      return row;
    }
    showDetail(show){
      this.ui["detail-console"].hidden=!show;
      this.ui["detail-toggle"].setAttribute("aria-expanded",String(show));
      this.ui["detail-toggle"].textContent=show?"收起本路详细控制":"展开本路详细控制";
      this.onDetail(show);
      if(show)this.ui["detail-console"].scrollIntoView?.({behavior:"smooth",block:"start"});
    }
    add(){
      if(!this.metadataOk||!this.fleet?.allowed)return;
      const camera=this.fleet.cameras.find(camera=>camera.enabled&&!this.visible.has(camera.id));
      if(!camera)return;
      this.compact=false;this.visible.add(camera.id);const row=this.rows.get(camera.id);row.card.hidden=false;this.updateRow(row);this.update();
      row.input.focus();this.say(camera.label+" 已添加。填写视频地址后连接。");
    }
    remove(row){
      if(row.camera.own||row.busy||ACTIVE.has(row.status?.pipeline_state))return;
      this.visible.delete(row.camera.id);row.card.hidden=true;this.releasePreview(row);this.update();
    }
    update(){
      if(!this.fleet)return;
      const active=[...this.rows.values()].filter(row=>ACTIVE.has(row.status?.pipeline_state)).length;
      const extraActive=[...this.rows.values()].some(row=>!row.camera.own&&ACTIVE.has(row.status?.pipeline_state));
      const displayed=[...this.rows.values()].filter(row=>this.visible.has(row.camera.id));
      const available=this.fleet.cameras.filter(camera=>camera.enabled).length;
      const spare=this.fleet.cameras.some(camera=>camera.enabled&&!this.visible.has(camera.id));
      this.ui.mode.textContent=this.fleet.allowed?"本地 GPU 人物追踪 · 多路模式":"远程人物追踪 · 单路模式";
      this.ui.capacity.textContent=this.fleet.allowed
        ?"已部署 "+this.fleet.capacity+" 路通道，可按需连接任意路数。12 路为容量参考；扩容通道后可继续增加，以实际显卡负载为准。"
        :"当前仅可连接 1 路。启用本地 GPU 人物追踪并保存后，可使用已部署的其他通道。"+(this.fleet.reason?" "+this.fleet.reason:"")+
          (extraActive?" 其他通道仍有历史会话运行，可在下方停止；当前不能新增多路连接。":"");
      this.ui.count.textContent="已连接 "+active+" 路 · 显示 "+displayed.length+" 路 · 可用 "+available+" 路";
      this.ui.grid.dataset.single=String(displayed.length===1);
      this.ui.grid.className="fleet-grid"+(this.compact&&displayed.length>1?" compact":"");
      this.ui["view-toggle"].hidden=displayed.length<2;
      this.ui["view-toggle"].setAttribute("aria-pressed",String(this.compact));
      this.ui["view-toggle"].textContent=this.compact?"展开连接设置":"只看画面";
      this.ui.add.hidden=!this.fleet.allowed;this.ui.add.disabled=!this.metadataOk||!spare||this.bulk;
      this.ui.add.title=spare?"添加一个已部署通道":"所有已部署通道均已显示；增加部署通道数后即可继续添加";
      this.ui["start-all"].disabled=!this.metadataOk||this.bulk||!displayed.some(row=>!row.start.disabled);
      this.ui["stop-all"].disabled=this.bulk||![...this.rows.values()].some(row=>!row.stop.disabled);
      this.ui["start-all"].hidden=available<=1;this.ui["stop-all"].hidden=available<=1&&!extraActive;
      this.onState();
    }
    extraBusy(){return [...this.rows.values()].some(row=>!row.camera.own&&
      (row.busy||ACTIVE.has(row.status?.pipeline_state)||row.camera.enabled&&(!row.status||row.unreachable)));}
    updateRow(row){
      const status=row.status||{},active=ACTIVE.has(status.pipeline_state),stopping=["stopping","draining"].includes(status.pipeline_state);
      const enabled=row.camera.enabled&&this.metadataOk;
      row.title.textContent=row.camera.label+(row.camera.own?" · 本路":"");
      row.state.textContent=row.busy?"正在处理":row.unreachable?"服务离线":STATUS[status.pipeline_state]||(row.status?"未连接":"正在检查");
      row.state.className="fleet-camera-state"+(row.unreachable||row.error||status.last_error?" error":active?" good":"");
      row.input.disabled=active||row.busy;row.layout.disabled=active||row.busy;
      row.start.disabled=!enabled||row.busy||active||!row.status||row.unreachable;
      row.stop.disabled=row.busy||!active||stopping||!row.camera.stop_url;
      row.save.disabled=!enabled||row.busy||active||!row.input.value.trim();
      row.remove.hidden=row.camera.own===true;row.remove.disabled=row.busy||active;
      row.detail.hidden=!row.camera.own&&!row.camera.console_url;
      row.detail.disabled=!row.camera.own&&!row.camera.console_url;
      const configured=status.configured_stream===true;
      row.input.placeholder=configured?"留空使用已保存的视频地址（不回显）":"RTSP / RTMP / WHEP 地址";
      row.sourceNote.textContent=configured?"已保存视频地址 · 连接时可留空；填写新地址仅用于本次连接，点击保存后重启继续使用。":"未保存视频地址 · 填写后可连接，也可保存以便下次使用。";
      if(!row.layoutDirty&&!active)row.layout.value=(status.camera_layout||status.stream?.camera_layout)==="stereo"?"stereo":"mono";
      const serverError=typeof status.last_error==="string"?status.last_error:"";
      row.message.textContent=row.error||serverError||row.notice||(active?(status.dry_run?"测试模式 · 不调用模型":status.ready?"实时检测中":status.readiness_reason||"正在准备视频和模型…"):"默认分析：打架、吸烟、跌倒、武器、陌生人");
      row.message.className="fleet-card-detail"+(row.error||serverError?" error":"");
      const show=active&&status.pipeline_state!=="draining"&&this.visible.has(row.camera.id);
      if(!show)this.releasePreview(row);
      row.placeholder.hidden=!!row.previewUrl&&!row.previewError;
      row.placeholder.className="fleet-preview-note"+(row.previewUrl&&row.previewError?" over-image":"");
      row.placeholder.textContent=row.previewError?"画面暂不可用，正在重试…":row.unreachable?"通道服务暂时离线":active?"正在等待人物描边画面…":"连接后在此显示实时画面";
    }
    async pollStatus(discover=false){
      if(this.closed||this.polling)return;this.polling=true;
      const cameras=[...this.rows.values()].filter(row=>discover||row.camera.enabled||row.busy||ACTIVE.has(row.status?.pipeline_state));
      try{await pooled(cameras,4,async row=>{
        const generation=row.generation;
        try{
          const status=await this.request(row.camera.status_url);
          if(generation!==row.generation||this.closed)return;
          const sessionChanged=row.status?.stream_session_id!==status.stream_session_id;
          if(sessionChanged)this.releasePreview(row);
          row.status=status;if(row.unreachable)row.error="";row.unreachable=false;
          if(ACTIVE.has(status.pipeline_state))this.visible.add(row.camera.id);
          else if(!row.camera.enabled)this.visible.delete(row.camera.id);
          row.card.hidden=!this.visible.has(row.camera.id);
        }catch(error){if(generation!==row.generation||this.closed)return;row.unreachable=true;row.error="无法连接通道服务："+error.message;}
        this.updateRow(row);
      });}finally{
        this.polling=false;
        if(!this.initializedLayout){this.compact=[...this.rows.values()].filter(row=>ACTIVE.has(row.status?.pipeline_state)).length>=4;this.initializedLayout=true;}
        this.update();
      }
    }
    async command(row,url,method,payload){
      if(row.busy)return false;
      row.busy=true;row.error="";row.notice="";row.generation++;this.updateRow(row);this.update();
      try{
        const result=await this.request(url,{method,headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)},75000);
        if(result&&typeof result.pipeline_state==="string")row.status=result;
        return true;
      }catch(error){row.error=error.message;return false;}
      finally{row.busy=false;this.updateRow(row);this.update();}
    }
    async start(row){
      if(row.start.disabled)return false;
      const streamUrl=row.input.value.trim();
      if(!streamUrl&&row.status?.configured_stream!==true){row.error="请先填写该路视频地址。";this.updateRow(row);row.input.focus();return false;}
      const payload={features:[...FEATURES],camera_layout:row.layout.value};
      if(streamUrl)payload.stream_url=streamUrl;
      const ok=await this.command(row,row.camera.start_url,"POST",payload);
      if(ok){row.status={...row.status,pipeline_state:"starting"};row.notice="连接请求已发送。";}
      this.updateRow(row);this.update();await this.pollStatus();return ok;
    }
    async stop(row){
      if(row.stop.disabled)return false;
      const ok=await this.command(row,row.camera.stop_url,"POST",{});
      if(ok){row.status={...row.status,pipeline_state:"stopping"};row.notice="正在停止该路检测…";}
      this.updateRow(row);this.update();await this.pollStatus();return ok;
    }
    async save(row){
      if(row.save.disabled)return false;
      const ok=await this.command(row,row.camera.settings_url,"PUT",{stream_url:row.input.value.trim(),camera_layout:row.layout.value});
      if(ok){row.input.value="";row.status={...row.status,configured_stream:true,camera_layout:row.layout.value};row.notice="视频地址已保存，重启后继续使用。";}
      this.updateRow(row);this.update();return ok;
    }
    async all(action){
      if(this.bulk||action==="start"&&!this.metadataOk)return;
      this.bulk=true;this.update();let succeeded=0,failed=0;
      const candidates=[...this.rows.values()].filter(row=>action==="stop"?!row.stop.disabled:this.visible.has(row.camera.id)&&!row.start.disabled);
      await pooled(candidates,2,async row=>{if(await this[action](row))succeeded++;else failed++;});
      this.bulk=false;this.update();
      this.say((action==="start"?"连接":"停止")+"请求完成："+succeeded+" 路成功"+(failed?"，"+failed+" 路未完成，请查看对应卡片。":"。"),failed>0);
    }
    releasePreview(row){
      row.previewGeneration=(row.previewGeneration||0)+1;
      row.previewController?.abort();row.previewController=null;
      if(row.previewUrl)this.URL.revokeObjectURL(row.previewUrl);
      row.previewUrl=null;row.previewError=false;row.image.removeAttribute("src");row.image.hidden=true;
    }
    async frame(row){
      if(!row.camera.preview_url||this.closed)return;
      const generation=row.previewGeneration||0,controller=new this.AbortController();row.previewController=controller;
      const timer=this.setTimeout(()=>controller.abort(),3000);let nextUrl=null;
      try{
        const response=await this.fetch(row.camera.preview_url,{cache:"no-store",signal:controller.signal});
        if(!response.ok)throw new Error("Frame unavailable");
        nextUrl=this.URL.createObjectURL(await response.blob());
        if(this.closed||generation!==(row.previewGeneration||0)||this.document.hidden)return;
        const previous=row.previewUrl;row.image.src=nextUrl;row.image.hidden=false;row.previewUrl=nextUrl;nextUrl=null;row.previewError=false;
        if(previous)this.URL.revokeObjectURL(previous);
        row.placeholder.hidden=true;
      }catch(_){row.previewError=true;this.updateRow(row);}
      finally{this.clearTimeout(timer);if(nextUrl)this.URL.revokeObjectURL(nextUrl);if(row.previewController===controller)row.previewController=null;}
    }
    async previews(){
      if(this.closed)return;
      const started=this.clock();
      if(!this.document.hidden){
        const rows=[...this.rows.values()].filter(row=>this.visible.has(row.camera.id)&&["starting","running"].includes(row.status?.pipeline_state)&&!row.unreachable);
        await pooled(rows,3,row=>this.frame(row));
      }
      if(!this.closed)this.previewTimer=this.setTimeout(()=>this.previews(),Math.max(50,250-(this.clock()-started)));
    }
    async tick(){
      if(this.closed)return;
      if(!this.document.hidden){if(!this.metadataOk||this.clock()-this.lastMetadata>=10000)await this.refresh();else await this.pollStatus();}
      if(!this.closed)this.timer=this.setTimeout(()=>this.tick(),2000);
    }
    async startPolling(){await this.refresh();if(!this.closed){this.timer=this.setTimeout(()=>this.tick(),2000);this.previews();}}
    close(){this.closed=true;this.clearTimeout(this.timer);this.clearTimeout(this.previewTimer);for(const row of this.rows.values())this.releasePreview(row);}
  }
  const api={FleetConsole,normalizeFleet,pooled};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root.document?.getElementById("fleet-panel")){
    const app=new FleetConsole({document:root.document,fetch:root.fetch.bind(root),setTimeout:root.setTimeout.bind(root),
      clearTimeout:root.clearTimeout.bind(root),AbortController:root.AbortController,URL:root.URL,
      onDetail:show=>{if(show){if(typeof openFeed==="function")openFeed();}else if(typeof closeFeed==="function")closeFeed();},
      onState:()=>{if(typeof controls==="function")controls();}});
    root.GoodvisionFleet={refresh:()=>app.refresh(),detailHidden:()=>app.ui["detail-console"].hidden,extraBusy:()=>app.extraBusy(),app};
    root.addEventListener("beforeunload",()=>app.close());app.startPolling();
  }
})(typeof globalThis!=="undefined"?globalThis:this);
