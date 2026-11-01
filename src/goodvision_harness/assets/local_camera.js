(function(root){
  "use strict";
  const messages={
    NotAllowedError:"摄像头权限被拒绝，请在浏览器中允许访问摄像头后重试。",
    NotFoundError:"未找到本机摄像头，请连接摄像头后重试。",
    NotReadableError:"摄像头无法打开，可能正被其他应用占用。"
  };
  class LocalCameraPublisher{
    constructor(doc){
      this.ui={};this.generation=0;this.busy=false;
      for(const name of ["start","stop","copy","url","result","preview","status"])
        this.ui[name]=doc.getElementById("local-camera-"+name);
      this.ui.start.addEventListener("click",()=>this.start());
      this.ui.stop.addEventListener("click",()=>this.stop());
      this.ui.copy.addEventListener("click",()=>this.copy());
      root.addEventListener("pagehide",()=>this.stop());
      this.render();
    }
    say(text,error=false){
      this.ui.status.textContent=text;this.ui.status.className="note"+(error?" error":"");
    }
    render(){
      this.ui.start.disabled=this.busy||Boolean(this.peer);
      this.ui.stop.disabled=!this.busy&&!this.peer;
      this.ui.copy.disabled=!this.address;
      this.ui.result.hidden=!this.address;
      this.ui.url.value=this.address||"";
      this.ui.preview.hidden=!this.stream;
    }
    async request(path,options={}){
      const response=await root.fetch(path,{...options,signal:this.abort?.signal});
      if(!response.ok){
        let detail="";try{detail=(await response.json()).detail||"";}catch(_){}
        const errors={
          "Camera publishing support is not installed":"当前服务未安装摄像头发布组件，请安装 webrtc 依赖后重试。",
          "Camera publication has stopped or expired":"摄像头发布已停止或连接已失效，请重新生成链接。",
          "Stop an existing camera publication first":"摄像头发布数量已满，请先停止已有发布。"
        };
        throw new Error(errors[detail]||"摄像头发布失败，请重试。");
      }
      return response;
    }
    async gather(peer){
      if(peer.iceGatheringState==="complete")return;
      await new Promise((resolve,reject)=>{
        const done=()=>{if(peer.iceGatheringState==="complete"){cleanup();resolve();}};
        const closed=()=>{if(peer.connectionState==="closed"){cleanup();reject(new Error("摄像头发布已停止。"));}};
        const timer=root.setTimeout(()=>{cleanup();reject(new Error("摄像头连接超时，请重试。"));},10000);
        const cleanup=()=>{root.clearTimeout(timer);peer.removeEventListener("icegatheringstatechange",done);peer.removeEventListener("connectionstatechange",closed);};
        peer.addEventListener("icegatheringstatechange",done);peer.addEventListener("connectionstatechange",closed);done();
      });
    }
    async start(){
      if(this.busy||this.peer)return;
      if(!root.isSecureContext||!root.navigator.mediaDevices?.getUserMedia){
        this.say("摄像头访问需要 HTTPS 或 localhost，请使用安全地址打开控制台。",true);return;
      }
      if(!root.RTCPeerConnection){this.say("当前浏览器不支持 WebRTC，请更换浏览器。",true);return;}
      const generation=++this.generation;
      this.busy=true;this.abort=new root.AbortController();this.render();
      this.say("请允许访问摄像头，正在生成 WebRTC 链接…");
      try{
        const stream=await root.navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:30}},audio:false});
        if(generation!==this.generation){stream.getTracks().forEach(track=>track.stop());return;}
        this.stream=stream;this.ui.preview.srcObject=stream;this.render();
        this.ui.preview.play().catch(()=>{});
        const peer=this.peer=new root.RTCPeerConnection({iceServers:[]});
        for(const track of stream.getVideoTracks()){
          peer.addTransceiver(track,{direction:"sendonly"});
          track.addEventListener("ended",()=>{if(generation===this.generation)this.stop("摄像头已断开，请重新生成链接。",true);});
        }
        peer.addEventListener("connectionstatechange",()=>{
          if(generation===this.generation&&["failed","closed"].includes(peer.connectionState))
            this.stop("摄像头连接已断开，请重新生成链接。",true);
        });
        await peer.setLocalDescription(await peer.createOffer());await this.gather(peer);
        if(generation!==this.generation)return;
        const response=await this.request("/v1/local-camera",{method:"POST",headers:{"Content-Type":"application/sdp"},body:peer.localDescription.sdp});
        const resource=new root.URL(response.headers.get("Location"),root.location.origin);
        if(resource.origin!==root.location.origin||!/^\/v1\/local-camera\/[a-f0-9]{32}$/.test(resource.pathname))throw new Error("摄像头发布失败，请重试。");
        if(generation!==this.generation){root.fetch(resource.pathname,{method:"DELETE",keepalive:true}).catch(()=>{});return;}
        this.resource=resource.pathname;
        await peer.setRemoteDescription({type:"answer",sdp:await response.text()});
        let ready=false;
        for(let attempt=0;attempt<40;attempt++){
          if(generation!==this.generation)return;
          const status=await (await this.request(this.resource,{cache:"no-store"})).json();
          if(status.state==="connected"&&status.frames_received>0){ready=true;break;}
          await new Promise(resolve=>root.setTimeout(resolve,250));
        }
        if(!ready)throw new Error("摄像头连接超时，请重试。");
        if(generation!==this.generation)return;
        this.address="webrtc+"+resource.href+"/whep";this.busy=false;this.render();
        this.say("正在发布 · WebRTC 链接已就绪，可复制到工作台该路的视频源配置。");
        this.poll=root.setInterval(async()=>{
          try{
            await this.request(this.resource,{cache:"no-store"});
          }catch(error){if(generation===this.generation)this.stop(error.message,true);}
        },3000);
      }catch(error){
        if(generation===this.generation)this.stop(messages[error.name]||error.message,true);
      }
    }
    stop(message="发布已停止，原链接已失效。",error=false){
      ++this.generation;this.abort?.abort();this.abort=null;
      root.clearInterval(this.poll);this.poll=null;
      const peer=this.peer;this.peer=null;peer?.close();
      this.stream?.getTracks().forEach(track=>track.stop());this.stream=null;
      this.ui.preview.srcObject=null;
      if(this.resource)root.fetch(this.resource,{method:"DELETE",keepalive:true}).catch(()=>{});
      this.resource=null;this.address=null;this.busy=false;this.render();this.say(message,error);
    }
    async copy(){
      if(!this.address)return;
      try{await root.navigator.clipboard.writeText(this.address);this.say("WebRTC 链接已复制。");}
      catch(_){this.ui.url.focus();this.ui.url.select();this.say("已选中链接，请按 Ctrl+C（Mac 使用 ⌘C）复制。");}
    }
  }
  root.GoodvisionLocalCamera={LocalCameraPublisher};
  if(root.document?.getElementById("local-camera-start")&&!root.GoodvisionDiagnostic)
    root.GoodvisionLocalCamera.app=new LocalCameraPublisher(root.document);
})(globalThis);
