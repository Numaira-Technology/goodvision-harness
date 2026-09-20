/* Independent video connections with one global analysis cohort. Finite JPEG requests keep
 * browser connection slots available for control requests even with 12+ feeds. */
(function(root){
  "use strict";
  const ACTIVE=new Set(["starting","running","stopping","draining"]);
  const FEATURES=["fight","smoking","fall","weapon","stranger"];
  const LABELS={fight:"打架",smoking:"吸烟",fall:"跌倒",weapon:"持械",stranger:"陌生人"};
  const CONNECTION={connecting:"正在连接视频",connected:"视频已连接",disconnecting:"正在断开视频",disconnected:"视频未连接",error:"视频连接异常"};
  const ANALYSIS={idle:"未分析",starting:"正在开始分析",running:"分析中",stopping:"正在停止分析",draining:"正在完成剩余分析",error:"分析异常",unknown:"正在确认分析状态"};
  function connectionState(status){return status?.connection_state||(status?.connected===true?"connected":"disconnected");}
  function connected(status){return status?.connected===true||connectionState(status)==="connected";}
  function connectionBusy(status){return ["connecting","connected","disconnecting"].includes(connectionState(status));}
  function analyzing(status){return ACTIVE.has(status?.analysis_state||status?.pipeline_state);}
  const MODES={legacy:"原部署追踪服务 · 单路模式",cloud:"云 GPU 人物追踪 · 多路模式",local:"本机 GPU 人物追踪 · 多路模式",server:"独立 GPU 追踪服务器 · 多路模式"};
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
    const count=(value,fallback)=>Number.isFinite(Number(value))&&Number(value)>=0?Math.floor(Number(value)):fallback;
    const seen=new Set();
    return {allowed,trackingEnabled:data.tracking_enabled!==false,trackingMode:Object.hasOwn(MODES,data.tracking_mode)?data.tracking_mode:"legacy",
      analysis:data.analysis&&typeof data.analysis==="object"?data.analysis:{state:"unknown",camera_ids:[],errors:{}},
      analysisStartUrl:safeEndpoint(data.analysis_start_url),analysisStopUrl:safeEndpoint(data.analysis_stop_url),
      capacity:count(data.capacity,cameras.length),usableCapacity:count(data.usable_capacity,cameras.filter(camera=>camera.enabled===true).length),reason:String(data.reason||""),
      cameras:cameras.filter(camera=>{if(seen.has(camera.id))return false;seen.add(camera.id);return true;}).map((camera,index)=>({
        ...camera,label:String(camera.label||"摄像头 "+(index+1)),
        enabled:camera.enabled===true,
        status_url:safeEndpoint(camera.status_url),connect_url:safeEndpoint(camera.connect_url),
        disconnect_url:safeEndpoint(camera.disconnect_url),settings_url:safeEndpoint(camera.settings_url),
        detail_url:safeEndpoint(camera.detail_url),results_url:safeEndpoint(camera.results_url),scene_url:safeEndpoint(camera.scene_url),
        diagnostics_url:safeEndpoint(camera.diagnostics_url),
        preview_url:safeEndpoint(camera.preview_url),display_url:safeEndpoint(camera.display_url),
        display_metrics_url:safeEndpoint(camera.display_metrics_url)
      }))};
  }
  class FleetConsole{
    constructor({document,fetch,clock=()=>Date.now(),setTimeout,clearTimeout,AbortController,URL,onDetail=()=>{},onState=()=>{}}){
      Object.assign(this,{document,fetch,clock,setTimeout,clearTimeout,AbortController,URL,onDetail,onState});
      this.rows=new Map();this.visible=new Set();this.fleet=null;this.metadataOk=false;this.loading=false;
      this.closed=false;this.bulk=false;this.globalBusy=false;this.lastMetadata=0;this.previewGeneration=0;this.refreshPending=false;
      this.compact=false;this.initializedLayout=false;
      this.ui={};for(const name of ["mode","capacity","add","start-all","stop-all","analysis-start","analysis-stop","analysis-state","view-toggle","count","grid","note","detail-toggle","detail-console"]){
        this.ui[name]=document.getElementById("fleet-"+name);
      }
      this.ui.add.onclick=()=>this.add();this.ui["start-all"].onclick=()=>this.all("start");
      this.ui["stop-all"].onclick=()=>this.all("stop");
      this.ui["analysis-start"].onclick=()=>this.analysis("start");this.ui["analysis-stop"].onclick=()=>this.analysis("stop");
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
          const error=new Error(detail);error.httpStatus=response.status;throw error;
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
      for(const [id,row] of this.rows){if(!ids.has(id)){this.closeDiagnostics(row);this.releasePreview(row);row.card.remove();this.rows.delete(id);this.visible.delete(id);}}
      for(const camera of fleet.cameras){
        let row=this.rows.get(camera.id);
        if(!row){row=this.makeRow(camera);this.rows.set(camera.id,row);}
        row.camera=camera;
        if(camera.enabled&&(camera.own===true||!this.visible.size))this.visible.add(camera.id);
        if(!camera.enabled&&!connectionBusy(row.status)&&!analyzing(row.status))this.visible.delete(camera.id);
        row.card.hidden=!this.visible.has(camera.id);this.updateRow(row);
      }
    }
    makeRow(camera){
      const card=this.element("article","fleet-card"),head=this.element("div","fleet-card-head");
      const title=this.element("h3","",camera.label),state=this.element("span","fleet-camera-state","正在检查");
      head.append(title,state);
      const viewport=this.element("div","fleet-preview"),image=this.element("img");image.alt=camera.label+" 实时人物描边画面";image.hidden=true;
      const canvas=this.element("canvas");canvas.hidden=true;
      const placeholder=this.element("div","fleet-preview-note","连接后在此显示实时画面");viewport.append(image,canvas,placeholder);
      const displayNote=this.element("div","fleet-card-detail","播放目标 15 帧/秒 · 画面与分析独立更新");
      const vlmNote=this.element("div","fleet-card-detail","VLM · 等待开始分析");vlmNote.setAttribute("role","status");
      const vlmEvents=this.element("div","fleet-card-detail","本次事件观察 · 等待开始分析");vlmEvents.setAttribute("role","status");
      const fields=this.element("div","fleet-card-fields"),source=this.element("label","field");
      source.append(this.element("span","","网络视频地址"));
      const input=this.element("input");input.type="text";input.autocomplete="off";input.spellcheck=false;
      input.placeholder="RTSP / RTMP / WHEP 地址";input.setAttribute("aria-label",camera.label+" 视频地址");source.append(input);
      const layoutField=this.element("label","field");layoutField.append(this.element("span","","摄像头类型"));
      const layout=this.element("select");layout.setAttribute("aria-label",camera.label+" 摄像头类型");
      for(const [value,label] of [["mono","单目 · 完整画面"],["stereo","双目 · 使用左半幅"]]){const option=this.element("option","",label);option.value=value;layout.append(option);}
      layout.value="mono";layoutField.append(layout);
      const sourceNote=this.element("div","fleet-source","正在读取已保存的视频源…");
      const buttons=this.element("div","fleet-card-buttons"),start=this.element("button","btn primary","连接视频"),stop=this.element("button","btn danger","断开视频");
      const save=this.element("button","btn","保存地址"),detail=this.element("button","btn","详细控制"),remove=this.element("button","btn fleet-remove","移除");
      for(const button of [start,stop,save,detail,remove]){button.type="button";buttons.append(button);}
      remove.setAttribute("aria-label","移除 "+camera.label);
      const message=this.element("div","fleet-card-detail");message.setAttribute("role","status");
      fields.append(source,layoutField,sourceNote,buttons,message);card.append(head,viewport,displayNote,vlmNote,vlmEvents,fields);this.ui.grid.append(card);
      const row={camera,card,title,state,image,canvas,displayNote,vlmNote,vlmEvents,placeholder,input,layout,sourceNote,start,stop,save,detail,remove,message,
        status:null,busy:false,error:"",notice:"",previewUrl:null,previewError:false,generation:0,layoutDirty:false,
        detailOpen:false,prefsLoaded:false,prefsLoading:false,prefsDirty:false,prefsSaving:false,detailPolling:false};
      this.makeDetail(row);
      const diagnostics=this.element("button","btn","本机摄像头 / 诊断");diagnostics.type="button";
      const diagnosticsPanel=this.element("div","fleet-diagnostics");diagnosticsPanel.hidden=true;
      diagnosticsPanel.id="diagnostics-"+camera.id;
      diagnostics.setAttribute("aria-controls",diagnosticsPanel.id);diagnostics.setAttribute("aria-expanded","false");
      diagnostics.setAttribute("aria-label",camera.label+" 本机摄像头 / 诊断");
      buttons.append(diagnostics);card.append(diagnosticsPanel);
      Object.assign(row,{diagnostics,diagnosticsPanel,diagnosticsFrame:null});
      diagnostics.onclick=()=>this.toggleDiagnostics(row);
      start.onclick=()=>this.start(row);stop.onclick=()=>this.stop(row);save.onclick=()=>this.save(row);
      remove.onclick=()=>this.remove(row);detail.onclick=()=>this.toggleDetail(row);
      input.oninput=()=>{row.notice="";row.error="";this.updateRow(row);};layout.onchange=()=>{row.layoutDirty=true;};
      return row;
    }
    toggleDiagnostics(row){
      if(row.diagnosticsFrame){this.closeDiagnostics(row);return;}
      if(!row.camera.diagnostics_url){row.error="该通道暂未提供诊断入口，请更新该通道服务。";this.updateRow(row);return;}
      const frame=this.element("iframe");frame.title=row.camera.label+" 本机摄像头与诊断";
      frame.src=row.camera.diagnostics_url;frame.setAttribute("allow","camera 'self'");
      row.diagnosticsFrame=frame;row.diagnosticsPanel.append(frame);row.diagnosticsPanel.hidden=false;
      row.diagnostics.textContent="收起本机摄像头 / 诊断";row.diagnostics.setAttribute("aria-expanded","true");
    }
    closeDiagnostics(row){
      // Removing the document closes its event streams, timers and local camera.
      row.diagnosticsFrame?.remove();row.diagnosticsFrame=null;row.diagnosticsPanel.hidden=true;
      row.diagnostics.textContent="本机摄像头 / 诊断";row.diagnostics.setAttribute("aria-expanded","false");
    }
    showDetail(show){
      this.ui["detail-console"].hidden=!show;
      this.ui["detail-toggle"].setAttribute("aria-expanded",String(show));
      this.ui["detail-toggle"].textContent=show?"收起本机摄像头 / 诊断":"本机摄像头 / 诊断";
      this.onDetail(show);
      if(show)this.ui["detail-console"].scrollIntoView?.({behavior:"smooth",block:"start"});
    }
    add(){
      if(!this.metadataOk||!this.fleet?.allowed||this.analysisLocked()||this.bulk)return;
      const camera=this.fleet.cameras.find(camera=>camera.enabled&&!this.visible.has(camera.id));
      if(!camera)return;
      this.compact=false;this.visible.add(camera.id);const row=this.rows.get(camera.id);row.card.hidden=false;this.updateRow(row);this.update();
      row.input.focus();this.say(camera.label+" 已添加。填写视频地址后连接。");
    }
    remove(row){
      if(row.camera.own||row.busy||connectionBusy(row.status)||this.analysisLocked())return;
      this.visible.delete(row.camera.id);row.card.hidden=true;this.closeDiagnostics(row);this.releasePreview(row);this.update();
    }
    analysisLocked(){
      const analysis=this.fleet?.analysis,rows=[...this.rows.values()];
      return this.globalBusy||ACTIVE.has(analysis?.state)||analysis?.state==="unknown"||rows.some(row=>analyzing(row.status))||
        analysis?.state==="error"&&((analysis.camera_ids||[]).length>0||rows.some(row=>row.camera.enabled&&(!row.status||row.unreachable)));
    }
    makeDetail(row){
      const panel=this.element("section","fleet-camera-details");panel.hidden=true;panel.id="fleet-detail-"+row.camera.id;
      row.detail.setAttribute("aria-controls",panel.id);row.detail.setAttribute("aria-expanded","false");
      const heading=this.element("h4","","该路检测设置"),features=this.element("div","feature-select");
      const sensitivities=this.element("div","fleet-sensitivities");row.features=new Map();row.sensitivities=new Map();
      for(const key of FEATURES){
        const label=this.element("label","switch"),input=this.element("input");input.type="checkbox";input.checked=true;
        input.setAttribute("aria-label",row.camera.label+" 检测"+LABELS[key]);
        label.append(input,this.element("span","",LABELS[key]));features.append(label);row.features.set(key,input);
        input.onchange=()=>{row.prefsDirty=true;this.updateRow(row);this.update();};
        if(key==="stranger")continue;
        const field=this.element("label","field"),select=this.element("select"),note=this.element("span","note");
        select.setAttribute("aria-label",row.camera.label+" "+LABELS[key]+"敏感度");
        for(const [value,title] of [["standard","标准"],["high","高敏感度"]]){
          const option=this.element("option","",title);option.value=value;select.append(option);
        }
        select.value="standard";select.onchange=()=>{row.prefsDirty=true;this.sensitivityNote(row,key);this.updateRow(row);this.update();};
        field.append(this.element("span","",LABELS[key]+"敏感度"),select,note);sensitivities.append(field);
        row.sensitivities.set(key,{select,note});
      }
      const save=this.element("button","btn","保存该路检测设置"),reload=this.element("button","btn","放弃未保存修改");
      save.type="button";reload.type="button";save.onclick=()=>this.savePreferences(row);reload.onclick=()=>this.loadPreferences(row,true);
      const actions=this.element("div","fleet-card-buttons");actions.append(save,reload);
      const note=this.element("div","fleet-card-detail","正在读取该路已保存设置…");note.setAttribute("role","status");
      const summary=this.element("div","fleet-detail-summary"),result=this.element("pre","fleet-detail-json","暂无分析结果"),scene=this.element("pre","fleet-detail-json","暂无连续观察结果");
      const resultDetails=this.element("details","fleet-result-details"),sceneDetails=this.element("details","fleet-result-details");
      resultDetails.append(this.element("summary","","本路最新分析结果"),result);sceneDetails.append(this.element("summary","","本路连续观察状态"),scene);
      panel.append(heading,features,sensitivities,this.element("p","note","高敏感度更容易触发警报，也可能增加误报。陌生人识别使用独立人脸设置。分析期间检测设置和视频连接保持不变。"),actions,note,summary,resultDetails,sceneDetails);
      row.card.append(panel);Object.assign(row,{detailPanel:panel,prefsSave:save,prefsReload:reload,prefsNote:note,detailSummary:summary,resultJson:result,sceneJson:scene});
    }
    sensitivityNote(row,key){
      const control=row.sensitivities.get(key),option=row.prefOptions?.[key]?.find(item=>item.value===control.select.value);
      control.note.textContent=option?.description||(control.select.value==="high"?"更容易触发警报，可能增加误报。":"使用标准事件判定规则。");
    }
    async toggleDetail(row){
      row.detailOpen=!row.detailOpen;row.detailPanel.hidden=!row.detailOpen;
      row.detail.setAttribute("aria-expanded",String(row.detailOpen));row.detail.textContent=row.detailOpen?"收起详细控制":"详细控制";
      if(row.detailOpen){await this.loadPreferences(row);await this.pollDetail(row);}
    }
    async loadPreferences(row,discard=false){
      if(row.prefsLoading||row.prefsSaving||row.prefsDirty&&!discard||row.prefsLoaded&&!discard)return;
      row.prefsLoading=true;row.prefsNote.textContent="正在读取该路已保存设置…";this.updateRow(row);
      try{
        const values=await this.request(row.camera.detail_url);
        row.prefOptions=values.event_sensitivity_options||{};
        const selected=Array.isArray(values.features)?values.features:FEATURES;
        for(const [key,input] of row.features)input.checked=selected.includes(key);
        for(const [key,{select}] of row.sensitivities){
          if(Array.isArray(row.prefOptions[key])&&row.prefOptions[key].length){
            select.replaceChildren();for(const option of row.prefOptions[key]){
              const item=this.element("option","",option.label||option.value);item.value=option.value;select.append(item);
            }
          }
          select.value=values.event_sensitivities?.[key]||values.event_sensitivity_defaults?.[key]||"standard";
          this.sensitivityNote(row,key);
        }
        row.prefsLoaded=true;row.prefsDirty=false;row.prefsError="";
      }catch(error){row.prefsError="无法读取检测设置："+error.message;}
      finally{row.prefsLoading=false;this.updateRow(row);this.update();}
    }
    async savePreferences(row,forStart=false){
      if(!row.prefsLoaded||row.prefsLoading||row.prefsSaving||!forStart&&this.analysisLocked())return false;
      const features=[...row.features].filter(([,input])=>input.checked).map(([key])=>key);
      if(!features.length){row.prefsError="请为该路至少选择一个检测项目。";this.updateRow(row);return false;}
      row.prefsSaving=true;row.prefsError="";this.updateRow(row);this.update();
      const event_sensitivities=Object.fromEntries([...row.sensitivities].map(([key,{select}])=>[key,select.value]));
      try{
        await this.request(row.camera.detail_url,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({features,event_sensitivities})});
        row.prefsDirty=false;return true;
      }catch(error){row.prefsError="检测设置保存失败："+error.message;return false;}
      finally{row.prefsSaving=false;this.updateRow(row);this.update();}
    }
    async pollDetail(row){
      if(!row.detailOpen||row.detailPolling||this.closed)return;
      row.detailPolling=true;
      try{
        for(const [url,output,label] of [[row.camera.results_url,row.resultJson,"暂无分析结果"],[row.camera.scene_url,row.sceneJson,"暂无连续观察结果"]]){
          if(!url)continue;
          try{const data=await this.request(url);output.textContent=data&&Object.keys(data).length?JSON.stringify(data,null,2):label;}
          catch(error){output.textContent="暂时无法读取："+error.message;}
        }
      }finally{row.detailPolling=false;}
    }
    update(){
      if(!this.fleet)return;
      for(const row of this.rows.values())this.updateRow(row);
      const rows=[...this.rows.values()],active=rows.filter(row=>connected(row.status)).length,analyzed=rows.filter(row=>analyzing(row.status)).length;
      const locked=this.analysisLocked(),disabledActive=rows.some(row=>!row.camera.enabled&&(connectionBusy(row.status)||analyzing(row.status)));
      const displayed=[...this.rows.values()].filter(row=>this.visible.has(row.camera.id));
      const available=this.fleet.cameras.filter(camera=>camera.enabled).length;
      const spare=this.fleet.cameras.some(camera=>camera.enabled&&!this.visible.has(camera.id));
      this.ui.mode.textContent=!this.fleet.trackingEnabled?"视频播放 · 人脸与 VLM 识别":this.fleet.allowed?MODES[this.fleet.trackingMode]:
        MODES[this.fleet.trackingMode].replace("多路模式","当前可用 "+this.fleet.usableCapacity+" 路");
      const capacityNote=!this.fleet.trackingEnabled?"可使用已部署的视频通道；人物描边未启用。":this.fleet.trackingMode==="server"
        ?"可用路数由追踪服务器容量与已部署通道共同决定。"
        :this.fleet.trackingMode==="local"?"可按需添加通道，实际可同时运行的路数以本机 GPU 负载为准。"
        :this.fleet.trackingMode==="cloud"?"可按需部署 N 路通道；各路分配到云 GPU，实际容量取决于租用算力和负载。"
        :"原部署追踪服务仅支持单路；选择本机 GPU 或独立 GPU 追踪服务器后可使用多路通道。";
      this.ui.capacity.textContent="已部署 "+this.fleet.capacity+" 路通道，当前可用 "+this.fleet.usableCapacity+" 路。"+capacityNote+
        (this.fleet.reason?" "+this.fleet.reason:"")+(disabledActive?" 停用通道仍有历史会话运行，可在下方停止。":"");
      this.ui.count.textContent="视频已连接 "+active+" 路 · 正在分析 "+analyzed+" 路 · 显示 "+displayed.length+" 路 · 可用 "+available+" 路";
      this.ui.grid.dataset.single=String(displayed.length===1);
      this.ui.grid.className="fleet-grid"+(this.compact&&displayed.length>1?" compact":"");
      this.ui["view-toggle"].hidden=displayed.length<2;
      this.ui["view-toggle"].setAttribute("aria-pressed",String(this.compact));
      this.ui["view-toggle"].textContent=this.compact?"展开连接设置":"只看画面";
      this.ui.add.hidden=false;this.ui.add.disabled=!this.metadataOk||!this.fleet.allowed||!spare||this.bulk||locked;
      this.ui.add.setAttribute("aria-describedby","fleet-capacity");
      this.ui.add.title=!this.metadataOk?"正在确认可用通道，请等待配置恢复。":this.bulk?"正在处理视频连接，请稍候。":
        locked?"请先停止全部分析，再添加视频。":!this.fleet.allowed?
        this.fleet.reason||"当前可用 "+this.fleet.usableCapacity+" 路；扩容追踪服务和部署通道后可继续添加。":
        spare?"添加一个可用通道":"所有当前可用通道均已显示；扩容追踪服务或部署通道后可继续添加";
      this.ui["start-all"].disabled=!this.metadataOk||this.bulk||locked||!displayed.some(row=>!row.start.disabled);
      this.ui["stop-all"].disabled=this.bulk||locked||!rows.some(row=>!row.stop.disabled);
      const unknown=rows.some(row=>row.camera.enabled&&(!row.status||row.unreachable));
      const pending=rows.some(row=>row.busy||row.prefsLoading||row.prefsSaving||["connecting","disconnecting"].includes(connectionState(row.status)));
      this.ui["analysis-start"].disabled=!this.metadataOk||locked||this.bulk||pending||unknown||!active||!this.fleet.analysisStartUrl;
      const canStop=locked||["error","unknown"].includes(this.fleet.analysis.state);
      this.ui["analysis-stop"].disabled=this.globalBusy||!canStop||!this.fleet.analysisStopUrl;
      const errors=this.fleet.analysis.errors;
      const errorItems=Array.isArray(errors)?errors:errors&&typeof errors==="object"?Object.entries(errors).map(([camera_id,message])=>({camera_id,message})):[];
      const errorText=errorItems.length?" · "+errorItems.map(error=>{
        if(typeof error==="string")return error;
        const name=this.rows.get(error?.camera_id)?.camera.label||error?.camera_id||"全局分析";
        return name+"："+String(error?.message||"状态异常，请重试停止全部分析。");
      }).join("；"):"";
      this.ui["analysis-state"].textContent=(this.globalBusy?"正在处理统一分析请求":ANALYSIS[this.fleet.analysis.state]||"未分析")+
        (locked?" · 分析期间锁定视频连接；先停止全部分析，再增减视频。":this.fleet.analysis.state==="error"?
          " · 各路尚未开始分析，可修正设置后重试；也可停止全部分析以确认状态。":" · 连接视频后，统一开始所有已连接视频的分析。")+errorText;
      this.onState();
    }
    extraBusy(){return this.analysisLocked()||[...this.rows.values()].some(row=>
      row.busy||row.prefsSaving||["connecting","disconnecting"].includes(connectionState(row.status))||row.camera.enabled&&(!row.status||row.unreachable));}
    updateRow(row){
      const status=row.status||{},active=connectionBusy(status),locked=this.analysisLocked();
      const enabled=row.camera.enabled&&this.metadataOk;
      row.title.textContent=row.camera.label;
      const videoFresh=Number.isFinite(row.lastPresentedAt)&&this.clock()-row.lastPresentedAt<2500;
      row.state.textContent=row.busy?"正在处理":row.unreachable?(videoFresh?"视频播放中 · 状态更新延迟":"暂时无法确认服务状态"):(CONNECTION[connectionState(status)]||"正在检查")+" · "+(ANALYSIS[status.analysis_state||status.pipeline_state]||"未分析");
      row.state.className="fleet-camera-state"+(row.unreachable||row.error||status.last_error?" error":active?" good":"");
      const vlm=status.vlm,age=vlm?.last_success_age_seconds;
      if(row.unreachable)row.vlmNote.textContent="VLM · 状态更新延迟，正在重新确认";
      else if(!analyzing(status))row.vlmNote.textContent="VLM · 等待开始分析";
      else if(!vlm?.independent)row.vlmNote.textContent="VLM · 正在读取识别状态";
      else if(vlm.ready){
        row.vlmNote.textContent="VLM · 识别持续返回"+(Number.isFinite(age)?" · 最近成功 "+age.toFixed(1)+" 秒前":"")+
          (vlm.last_attempt_state==="failed"?" · 最近一次失败，继续识别":vlm.analysis_complete===false?" · 部分检测项待确认":"");
      }else if(vlm.running){
        row.vlmNote.textContent="VLM · "+(Number.isFinite(age)?"结果延迟 · 最近成功 "+age.toFixed(1)+" 秒前":"等待首次有效结果")+
          (vlm.last_attempt_state==="failed"?" · 最近一次失败，继续识别":"");
      }else row.vlmNote.textContent="VLM · 识别暂不可用，视频继续播放";
      row.input.disabled=active||row.busy||locked;row.layout.disabled=active||row.busy||locked;
      row.start.disabled=!enabled||row.busy||active||locked||!row.status||row.unreachable||!row.camera.connect_url;
      row.stop.disabled=row.busy||!active||locked||connectionState(status)==="disconnecting"||!row.camera.disconnect_url;
      row.save.disabled=!enabled||row.busy||active||locked||!row.input.value.trim();
      row.remove.hidden=row.camera.own===true;row.remove.disabled=row.busy||active||locked;
      row.detail.hidden=false;row.detail.disabled=false;
      const configured=status.configured_stream===true;
      row.input.placeholder=configured?"留空使用已保存的视频地址（不回显）":"RTSP / RTMP / WHEP 地址";
      row.sourceNote.textContent=configured?"已保存视频地址 · 连接时可留空；填写新地址仅用于本次连接，点击保存后重启继续使用。":"未保存视频地址 · 填写后可连接，也可保存以便下次使用。";
      if(!row.layoutDirty&&!active)row.layout.value=(status.camera_layout||status.stream?.camera_layout)==="stereo"?"stereo":"mono";
      const serverError=typeof status.last_error==="string"?status.last_error:"";
      row.message.textContent=row.error||serverError||row.notice||(analyzing(status)?"正在按本路已保存设置进行分析":active?(connected(status)?"视频已连接；分析请使用顶部“开始全部分析”。":"正在建立或断开视频连接…"):"连接只接收视频；检测项目可在详细控制中分别设置。");
      row.message.className="fleet-card-detail"+(row.error||serverError?" error":"");
      const show=["connecting","connected"].includes(connectionState(status))&&this.visible.has(row.camera.id);
      if(!show)this.releasePreview(row);
      row.placeholder.hidden=!!(row.previewUrl||row.networkHasFrame)&&!row.previewError&&!row.networkBuffering;
      row.placeholder.className="fleet-preview-note"+(row.previewUrl&&row.previewError?" over-image":"");
      row.placeholder.textContent=row.previewError?"画面暂不可用，正在重试…":row.networkBuffering?"正在缓冲连续画面…":row.unreachable?"正在重新确认通道状态…":active?"正在等待视频画面…":"连接后在此显示实时画面";
      const prefsBusy=locked||row.prefsLoading||row.prefsSaving||!row.prefsLoaded;
      for(const [key,input] of row.features){input.disabled=prefsBusy;const sensitivity=row.sensitivities.get(key);if(sensitivity)sensitivity.select.disabled=prefsBusy||!input.checked;}
      row.prefsSave.disabled=prefsBusy||!row.prefsDirty;row.prefsReload.disabled=locked||row.prefsLoading||row.prefsSaving;
      row.prefsNote.textContent=row.prefsError|| (row.prefsLoading?"正在读取该路已保存设置…":row.prefsSaving?"正在保存…":!row.prefsLoaded?"展开时读取该路已保存设置。":row.prefsDirty?"有未保存修改；开始全部分析前会先保存。":"本路检测设置已保存。统一分析会使用这些设置。");
      row.prefsNote.className="fleet-card-detail"+(row.prefsError?" error":"");
      row.detailSummary.textContent="当前状态："+row.state.textContent+" · 已接收 "+(Number(status.source_frames_received)||0)+" 帧 · 已发布 "+(Number(status.results_published)||0)+" 条结果";
    }
    async pollVlm(row){
      if(!analyzing(row.status)||!row.status.vlm?.independent){row.vlmReport=null;row.vlmEvents.textContent="本次事件观察 · 等待开始分析";return;}
      const session=row.status.stream_session_id,generation=row.generation;
      try{
        const url=row.camera.status_url?.replace(/\/status$/, "/vlm/results/latest");
        const report=await this.request(url,{},3000);
        if(this.closed||row.generation!==generation||row.status.stream_session_id!==session||report.stream_session_id!==session)return;
        if(report.state==="completed")row.vlmReport=report;
        const current=row.vlmReport?.stream_session_id===session?row.vlmReport:null;
        const summary=root.GoodvisionCameraDisplay?.eventObservationSummary?.(current)||"等待事件判断";
        const age=current?Math.max(0,(this.clock()-Date.parse(current.emitted_at))/1000):null;
        row.vlmEvents.textContent="本次事件观察 · "+summary+(Number.isFinite(age)?"（"+age.toFixed(1)+" 秒前）":"")+
          (report.state==="failed"?" · 新一轮正在重试":"");
      }catch(_){row.vlmEvents.textContent="事件结果更新延迟，正在重试；请结合上方最近成功时间。";}
    }
    async pollStatus(discover=false){
      if(this.closed||this.polling)return;this.polling=true;
      const cameras=[...this.rows.values()].filter(row=>discover||row.camera.enabled||row.busy||connectionBusy(row.status)||analyzing(row.status));
      try{await pooled(cameras,4,async row=>{
        const generation=row.generation;
        try{
          const statusUrl=row.camera.status_url;
          const status=await this.request(statusUrl?statusUrl+(statusUrl.includes("?")?"&":"?")+"lightweight=true":statusUrl);
          if(generation!==row.generation||this.closed)return;
          const sessionChanged=row.status?.stream_session_id!==status.stream_session_id;
          if(sessionChanged)this.releasePreview(row);
          if(connectionState(row.status)!==connectionState(status))row.notice="";
          row.status=status;if(row.unreachable)row.error="";row.unreachable=false;
          if(connectionBusy(status)||analyzing(status))this.visible.add(row.camera.id);
          else if(!row.camera.enabled)this.visible.delete(row.camera.id);
          row.card.hidden=!this.visible.has(row.camera.id);
        }catch(error){if(generation!==row.generation||this.closed)return;row.unreachable=true;row.error="无法连接通道服务："+error.message;}
        this.updateRow(row);
        if(!row.unreachable)await Promise.all([this.pollDetail(row),this.pollVlm(row)]);
      });}finally{
        this.polling=false;
        if(!this.initializedLayout){this.compact=[...this.rows.values()].filter(row=>connectionBusy(row.status)).length>=4;this.initializedLayout=true;}
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
      if(row.start.disabled||this.analysisLocked())return false;
      const streamUrl=row.input.value.trim();
      if(!streamUrl&&row.status?.configured_stream!==true){row.error="请先填写该路视频地址。";this.updateRow(row);row.input.focus();return false;}
      const payload={camera_layout:row.layout.value};
      if(streamUrl)payload.stream_url=streamUrl;
      const ok=await this.command(row,row.camera.connect_url,"POST",payload);
      if(ok){row.status={...row.status,connection_state:"connecting"};row.notice="视频连接请求已发送。";}
      this.updateRow(row);this.update();await this.pollStatus();return ok;
    }
    async stop(row){
      if(row.stop.disabled||this.analysisLocked())return false;
      const ok=await this.command(row,row.camera.disconnect_url,"POST",{});
      if(ok){row.status={...row.status,connection_state:"disconnecting"};row.notice="正在断开该路视频…";}
      this.updateRow(row);this.update();await this.pollStatus();return ok;
    }
    async save(row){
      if(row.save.disabled||this.analysisLocked())return false;
      const ok=await this.command(row,row.camera.settings_url,"PUT",{stream_url:row.input.value.trim(),camera_layout:row.layout.value});
      if(ok){row.input.value="";row.status={...row.status,configured_stream:true,camera_layout:row.layout.value};row.notice="视频地址已保存，重启后继续使用。";}
      this.updateRow(row);this.update();return ok;
    }
    async all(action){
      if(this.bulk||this.analysisLocked()||action==="start"&&!this.metadataOk)return;
      this.bulk=true;this.update();let succeeded=0,failed=0;
      const candidates=[...this.rows.values()].filter(row=>action==="stop"?!row.stop.disabled:this.visible.has(row.camera.id)&&!row.start.disabled);
      await pooled(candidates,2,async row=>{if(await this[action](row))succeeded++;else failed++;});
      this.bulk=false;this.update();
      this.say((action==="start"?"连接视频":"断开视频")+"请求完成："+succeeded+" 路成功"+(failed?"，"+failed+" 路未完成，请查看对应卡片。":"。"),failed>0);
    }
    async analysis(action){
      if(this.globalBusy||this.ui["analysis-"+action]?.disabled)return false;
      let submitted=false;
      this.globalBusy=true;this.say(action==="start"?"正在保存各路设置并开始全部分析…":"正在停止全部分析，视频连接将保留…");this.update();
      try{
        if(action==="start"){
          const dirty=[...this.rows.values()].filter(row=>row.prefsDirty);let failed=false;
          await pooled(dirty,2,async row=>{if(!await this.savePreferences(row,true))failed=true;});
          if(failed)throw new Error("有摄像头的检测设置未保存，请展开对应详细控制修正后再试。");
        }
        submitted=true;
        const result=await this.request(action==="start"?this.fleet.analysisStartUrl:this.fleet.analysisStopUrl,
          {method:"POST",headers:{"Content-Type":"application/json"},body:"{}"},240000);
        const analysis=result?.analysis||result;
        this.fleet.analysis=analysis&&typeof analysis.state==="string"?analysis:{...this.fleet.analysis,state:action==="start"?"starting":"stopping"};
        this.say(action==="start"?"已请求统一分析所有已连接视频。":"已请求停止全部分析；视频保持连接，可继续预览。");
        return true;
      }catch(error){
        if(submitted&&!error.httpStatus){
          this.fleet.analysis={...this.fleet.analysis,state:"unknown"};
          this.say("统一分析请求的响应中断，正在重新确认服务状态；请求可能仍在执行。请等待状态更新，必要时使用“停止全部分析”。",true);
        }else this.say("统一分析操作失败："+error.message,true);
        return false;
      }
      finally{this.globalBusy=false;await this.refresh();this.update();}
    }
    releasePreview(row){
      row.networkDisplay?.stop();row.networkDisplay=null;row.networkHasFrame=false;row.networkBuffering=false;
      row.lastPresentedAt=null;row.displaySample=null;
      row.canvas.hidden=true;
      row.previewGeneration=(row.previewGeneration||0)+1;
      row.previewController?.abort();row.previewController=null;
      if(row.previewUrl)this.URL.revokeObjectURL(row.previewUrl);
      row.previewUrl=null;row.previewError=false;row.image.removeAttribute("src");row.image.hidden=true;
    }
    async frame(row){
      if(!row.camera.preview_url||this.closed)return;
      const Display=root.GoodvisionCameraDisplay?.NetworkController;
      if(row.camera.display_url&&Display?.supported()){
        if(!row.networkDisplay){
          row.image.hidden=true;row.canvas.hidden=false;
          row.networkDisplay=new Display({canvas:row.canvas,url:row.camera.display_url,metricsUrl:row.camera.display_metrics_url,
            onFrame:({buffering})=>{row.networkHasFrame=true;row.networkBuffering=buffering;row.previewError=false;
              row.placeholder.hidden=!buffering;},
            onStatus:stats=>{row.networkBuffering=stats.distinct_presented_frames===0;
              row.latestDisplayStats=stats;
              row.placeholder.hidden=!row.networkBuffering&&!row.previewError;
              const now=this.clock(),previous=row.displaySample;
              const same=previous&&previous.id===stats.display_instance_id&&stats.distinct_presented_frames>=previous.frames;
              const fps=same&&now>previous.at?(stats.distinct_presented_frames-previous.frames)*1000/(now-previous.at):(stats.distinct_presented_fps??0);
              if((same&&stats.distinct_presented_frames>previous.frames)||(!same&&stats.distinct_presented_frames>0))row.lastPresentedAt=now;
              row.displaySample={id:stats.display_instance_id,at:now,frames:stats.distinct_presented_frames};
              row.displayNote.textContent="播放 "+fps.toFixed(1)+" / 15 帧/秒 · 缓冲 "+stats.configured_display_delay_seconds+" 秒 · 完整描边 "+
                (100*stats.complete_contour_frames/Math.max(1,stats.distinct_presented_frames)).toFixed(0)+"%";},
            onFailure:message=>{row.displayNote.textContent=message;}});
          row.networkDisplay.start();
        }
        return;
      }
      const generation=row.previewGeneration||0,controller=new this.AbortController();row.previewController=controller;
      const timer=this.setTimeout(()=>controller.abort(),3000);let nextUrl=null;
      try{
        const response=await this.fetch(row.camera.preview_url,{cache:"no-store",signal:controller.signal});
        if(!response.ok)throw new Error("Frame unavailable");
        nextUrl=this.URL.createObjectURL(await response.blob());
        if(this.closed||generation!==(row.previewGeneration||0)||this.document.hidden)return;
        const previous=row.previewUrl;row.image.src=nextUrl;row.image.hidden=false;row.previewUrl=nextUrl;nextUrl=null;row.previewError=false;
        row.lastPresentedAt=this.clock();
        if(previous)this.URL.revokeObjectURL(previous);
        row.placeholder.hidden=true;
      }catch(_){row.previewError=true;this.updateRow(row);}
      finally{this.clearTimeout(timer);if(nextUrl)this.URL.revokeObjectURL(nextUrl);if(row.previewController===controller)row.previewController=null;}
    }
    async previews(){
      if(this.closed)return;
      const started=this.clock();
      if(!this.document.hidden){
        const rows=[...this.rows.values()].filter(row=>this.visible.has(row.camera.id)&&["connecting","connected"].includes(connectionState(row.status))&&!row.unreachable);
        await pooled(rows,3,row=>this.frame(row));
      }
      if(!this.closed)this.previewTimer=this.setTimeout(()=>this.previews(),Math.max(50,250-(this.clock()-started)));
    }
    async tick(){
      if(this.closed)return;
      if(!this.document.hidden){if(!this.metadataOk||this.analysisLocked()||this.clock()-this.lastMetadata>=10000)await this.refresh();else await this.pollStatus();}
      if(!this.closed)this.timer=this.setTimeout(()=>this.tick(),2000);
    }
    async startPolling(){await this.refresh();if(!this.closed){this.timer=this.setTimeout(()=>this.tick(),2000);this.previews();}}
    close(){this.closed=true;this.clearTimeout(this.timer);this.clearTimeout(this.previewTimer);for(const row of this.rows.values()){this.closeDiagnostics(row);this.releasePreview(row);}}
  }
  const api={FleetConsole,normalizeFleet,pooled};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(root.document?.getElementById("fleet-panel")){
    const app=new FleetConsole({document:root.document,fetch:root.fetch.bind(root),setTimeout:root.setTimeout.bind(root),
      clearTimeout:root.clearTimeout.bind(root),AbortController:root.AbortController,URL:root.URL,
      onDetail:show=>{if(show){if(typeof openFeed==="function")openFeed();}else if(typeof closeFeed==="function")closeFeed();},
      onState:()=>{if(typeof controls==="function")controls();}});
    root.GoodvisionFleet={refresh:()=>app.refresh(),detailHidden:()=>app.ui["detail-console"].hidden,extraBusy:()=>app.extraBusy(),
      startAnalysis:()=>app.analysis("start"),stopAnalysis:()=>app.analysis("stop"),app};
    root.addEventListener("beforeunload",()=>app.close());app.startPolling();
  }
})(typeof globalThis!=="undefined"?globalThis:this);
