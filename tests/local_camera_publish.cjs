const assert=require("node:assert/strict");
const {test}=require("node:test");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const script=fs.readFileSync(path.join(__dirname,"../src/goodvision_harness/assets/local_camera.js"),"utf8");

function fixture({permission,failPublish=false,clipboardDenied=false,secure=true}={}){
  const elements=new Map(),requests=[],copied=[],events={};
  const element=id=>{
    if(!elements.has(id))elements.set(id,{
      handlers:{},addEventListener(name,fn){this.handlers[name]=fn;},
      play:async()=>{},focus(){this.focused=true;},select(){this.selected=true;}
    });
    return elements.get(id);
  };
  const track={stopCount:0,stop(){this.stopCount++;},addEventListener(name,fn){this[name]=fn;}};
  const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
  const peers=[];
  class Peer{
    constructor(){this.iceGatheringState="complete";this.connectionState="connected";this.handlers={};peers.push(this);}
    addEventListener(name,fn){this.handlers[name]=fn;}
    removeEventListener(name){delete this.handlers[name];}
    addTransceiver(track,options){this.track=track;this.direction=options.direction;}
    async createOffer(){return {type:"offer",sdp:"v=0\r\n"};}
    async setLocalDescription(value){this.localDescription=value;}
    async setRemoteDescription(value){this.remoteDescription=value;}
    close(){this.connectionState="closed";this.handlers.connectionstatechange?.();}
  }
  const resource="/v1/local-camera/0123456789abcdef0123456789abcdef";
  const context=vm.createContext({
    document:{getElementById:element},isSecureContext:secure,URL,AbortController,
    navigator:{mediaDevices:{getUserMedia:async options=>{
      assert.equal(options.audio,false);return permission?await permission:stream;
    }},clipboard:{writeText:async address=>{if(clipboardDenied)throw new Error();copied.push(address);}}},
    RTCPeerConnection:Peer,location:{origin:"http://localhost:7900"},
    addEventListener:(name,fn)=>events[name]=fn,setTimeout,clearTimeout,
    setInterval:()=>1,clearInterval:()=>{},
    fetch:async(url,options={})=>{
      requests.push({url,options});
      if(options.method==="POST")return {
        ok:!failPublish,headers:{get:()=>resource},text:async()=>"v=0\r\n",
        json:async()=>({detail:"Camera publishing support is not installed"})
      };
      return {ok:true,json:async()=>({state:"connected",frames_received:2})};
    }
  });
  vm.runInContext(script,context);
  return {app:context.GoodvisionLocalCamera.app,element,requests,copied,track,events,peers,stream,resource};
}

test("one click publishes video and copies a workflow-compatible WHEP link",async()=>{
  const f=fixture();await f.app.start();
  assert.equal(f.peers[0].direction,"sendonly");
  assert.equal(f.app.address,"webrtc+http://localhost:7900"+f.resource+"/whep");
  assert.equal(f.element("local-camera-result").hidden,false);
  assert.equal(f.element("local-camera-start").disabled,true);
  await f.app.start();assert.equal(f.requests.filter(r=>r.options.method==="POST").length,1);
  await f.app.copy();assert.deepEqual(f.copied,[f.app.address]);
  f.app.stop();
  assert.equal(f.peers[0].connectionState,"closed");assert.equal(f.track.stopCount,1);
  assert.equal(f.app.address,null);assert.equal(f.element("local-camera-result").hidden,true);
  assert.ok(f.requests.some(r=>r.url===f.resource&&r.options.method==="DELETE"));
  assert.ok(f.requests.every(r=>!r.url.includes("sessions/start")));
});

test("stopping while camera permission is pending releases a late granted camera",async()=>{
  let resolve;const permission=new Promise(done=>resolve=done);
  const f=fixture({permission});const starting=f.app.start();f.app.stop();
  resolve(f.stream);await starting;
  assert.equal(f.track.stopCount,1);assert.equal(f.peers.length,0);
  assert.equal(f.requests.length,0);assert.equal(f.element("local-camera-start").disabled,false);
});

test("permission denial and missing server support leave no camera running",async()=>{
  const denied=fixture({permission:Promise.reject(Object.assign(new Error("denied"),{name:"NotAllowedError"}))});
  await denied.app.start();
  assert.match(denied.element("local-camera-status").textContent,/权限被拒绝/);
  assert.equal(denied.peers.length,0);
  const unavailable=fixture({failPublish:true});await unavailable.app.start();
  assert.match(unavailable.element("local-camera-status").textContent,/未安装摄像头发布组件/);
  assert.equal(unavailable.track.stopCount,1);assert.equal(unavailable.app.address,null);
});

test("page close and camera disconnection invalidate the link",async()=>{
  const f=fixture();await f.app.start();f.events.pagehide();
  assert.equal(f.app.address,null);assert.equal(f.track.stopCount,1);
  const disconnected=fixture();await disconnected.app.start();disconnected.track.ended();
  assert.equal(disconnected.app.address,null);
  assert.match(disconnected.element("local-camera-status").textContent,/摄像头已断开/);
});

test("clipboard fallback selects the URL and insecure origins never request the camera",async()=>{
  const f=fixture({clipboardDenied:true});await f.app.start();await f.app.copy();
  assert.equal(f.element("local-camera-url").selected,true);f.app.stop();
  const insecure=fixture({secure:false});await insecure.app.start();
  assert.equal(insecure.requests.length,0);assert.match(insecure.element("local-camera-status").textContent,/HTTPS/);
});
