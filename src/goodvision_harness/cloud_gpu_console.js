/* Editable rental settings and explicit start/release controls. */
(function(root){
  "use strict";
  class CloudGPUConsole{
    constructor({document,fetch,setTimeout,clearTimeout,AbortController}){
      Object.assign(this,{document,fetch,setTimeout,clearTimeout,AbortController});
      this.ui={};this.dirty=false;this.busy=false;this.data=null;
      for(const id of ["panel","name","count","streams","disk","script","command","save","start","stop","status"])
        this.ui[id]=document.getElementById("cloud-gpu-"+id);
      for(const id of ["name","count","disk"])this.ui[id].oninput=()=>{this.dirty=true;this.updateScript();};
      this.ui.script.oninput=()=>{this.dirty=true;};
      this.ui.script.onchange=()=>{try{this.fill(this.readScript());}catch(error){this.say(error.message,true);}};
      this.ui.save.onclick=()=>this.action("save");
      this.ui.start.onclick=()=>this.action("start");
      this.ui.stop.onclick=()=>this.action("stop");
    }
    say(text,error=false){this.ui.status.textContent=text;this.ui.status.className="secret-status"+(error?" error":"");}
    fill(settings){
      this.ui.name.value=settings.gpu_name;this.ui.count.value=String(settings.gpu_count);
      this.ui.streams.value=String(settings.streams);this.ui.disk.value=String(settings.disk_gb);
      this.updateScript();
    }
    updateScript(){
      const s={gpu_name:this.ui.name.value.trim(),gpu_count:Number(this.ui.count.value),
        streams:Number(this.ui.streams.value),disk_gb:Number(this.ui.disk.value)};
      this.ui.script.value=JSON.stringify(s,null,2);
      this.ui.command.textContent="python tools/start_sam.py --gpu-name "+s.gpu_name+
        " --gpu-count "+s.gpu_count+" --streams "+s.streams+" --disk-gb "+s.disk_gb;
    }
    readScript(){
      let s;try{s=JSON.parse(this.ui.script.value);}catch(_){throw new Error("租用配置必须是有效的 JSON");}
      if(!s||Object.keys(s).sort().join(",")!=="disk_gb,gpu_count,gpu_name,streams"||
         typeof s.gpu_name!=="string"||!s.gpu_name.trim()||
         ![s.gpu_count,s.streams,s.disk_gb].every(v=>Number.isSafeInteger(v)&&v>0))
        throw new Error("请填写 GPU 型号，数量、视频路数和磁盘容量须为正整数");
      return s;
    }
    render(){
      const data=this.data,active=["starting","ready","stopping","degraded"].includes(data?.state);
      const unavailable=!data?.available;
      for(const id of ["name","count","disk","script","save"])this.ui[id].disabled=this.busy||unavailable||active;
      this.ui.start.disabled=this.busy||unavailable||active||data?.state==="cleanup_required";
      this.ui.stop.disabled=this.busy||unavailable||!active;
    }
    async request(method,path="",payload){
      const controller=new this.AbortController(),timer=this.setTimeout(()=>controller.abort(),75000);
      try{
        const response=await this.fetch("/v1/cloud-gpu"+path,{method,cache:"no-store",signal:controller.signal,
          headers:{"Content-Type":"application/json"},...(payload===undefined?{}:{body:JSON.stringify(payload)})});
        const data=await response.json();if(!response.ok)throw new Error(data.detail||"云 GPU 请求失败");return data;
      }finally{this.clearTimeout(timer);}
    }
    async refresh(){
      if(this.busy)return;
      try{
        this.data=await this.request("GET");
        if(!this.data.available){this.say("当前部署未启用云 GPU 控制器。请使用 tools/start_cloud_gpu.py --streams N 启动主程序。");}
        else{
          if(!this.dirty)this.fill({...this.data.settings,streams:this.data.capacity});
          const messages={stopped:"云 GPU 未运行。保存配置后，点击启动租用。",starting:"正在租用并准备模型，首次准备可能需要数十分钟。",
            ready:"云 GPU 已就绪，可开始全部视频分析。",stopping:"正在释放云 GPU。",degraded:"云 GPU 状态异常，当前租用仍在运行。",
            cleanup_required:"上次租用未确认释放，请检查云平台实例。处理前禁止再次租用。",error:"云 GPU 启动失败，未保留运行中的实例。"};
          this.say((messages[this.data.state]||"状态待确认")+" 已部署 "+this.data.capacity+" 路视频。"+
            (this.data.error?" "+this.data.error:"")+ (this.data.state==="cleanup_required"?" 实例："+(this.data.instance_id||this.data.label||"请查看控制器日志"):""),["cleanup_required","error","degraded"].includes(this.data.state));
        }
      }catch(error){this.data=null;this.say(error.message,true);}finally{this.render();}
    }
    async action(action){
      if(this.busy)return;
      this.busy=true;this.render();
      try{
        if(action!=="stop"){
          const settings=this.readScript();
          this.data=await this.request("PUT","",settings);this.dirty=false;this.fill(this.data.settings);
        }
        if(action!=="save")this.data=await this.request("POST","/"+action,{});
        this.say(action==="save"?"租用配置已保存，点击启动租用使其生效。":action==="start"?"已提交租用请求，正在准备云 GPU。":"已请求释放云 GPU，正在确认状态。");
      }catch(error){this.say(error.message,true);}finally{this.busy=false;this.render();}
    }
  }
  if(typeof module!=="undefined")module.exports={CloudGPUConsole};
  root.GoodvisionCloudGPU={CloudGPUConsole};
  if(root.document?.getElementById("cloud-gpu-panel")){
    const app=new CloudGPUConsole({document:root.document,fetch:root.fetch.bind(root),
      setTimeout:root.setTimeout.bind(root),clearTimeout:root.clearTimeout.bind(root),AbortController:root.AbortController});
    const poll=async()=>{if(!root.document.hidden&&!app.ui.panel.hidden)await app.refresh();root.setTimeout(poll,3000);};
    app.refresh();root.setTimeout(poll,3000);
  }
})(typeof globalThis!=="undefined"?globalThis:this);
