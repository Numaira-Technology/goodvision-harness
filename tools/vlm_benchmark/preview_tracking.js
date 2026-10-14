"use strict";
// Only the preview uses tracking. No tracked box becomes a VLM observation.
window.PreviewTracking = class {
  constructor(video, canvas, note, camera, media) {
    Object.assign(this, {video, canvas, note, camera, media});
    this.stage = window.currentStage;
    this.client = window.trackingClient;
    this.captureCanvas = document.createElement("canvas");
    this.captureCanvas.width = 320;
    this.captureCanvas.height = 180;
    this.busy = false; this.closed = false; this.epoch = 0; this.generation = 0;
    this.lastPaint = 0; this.lastTime = -1; this.playing = false;
    this.lastObjects = []; this.lastObjectsAt = 0;
    // One GPU serves all lanes in round-robin, so a detector box lands every
    // ~3s per lane under 12-stream load. Hold boxes and tolerate round trips
    // past that; the local optical-flow tracker keeps its tight timings.
    this.detectorMode = !!window.previewDetector;
    this.holdMs = this.detectorMode ? 4500 : 3500;
    this.abortMs = this.detectorMode ? 4000 : 1500;
    this.staleMs = this.detectorMode ? 4000 : 1200;
    this.stats = {frames:0, painted:0, requests:0, lost:0, errors:0, framesWithBoxes:0, samples:[], busySkipped:0, staleResponses:0,
      maxPaintGapSeconds:0, roundTripSeconds:[], computeSeconds:[], lastStatus:"waiting_for_grounding"};
    this.ctx = canvas.getContext("2d");
    this.captureContext = this.captureCanvas.getContext("2d");
    video.addEventListener("seeking", () => {this.generation++; this.clear("画面跳转 · 等待重新定位");});
    if(video.requestVideoFrameCallback) this.callback = video.requestVideoFrameCallback((n, m) => this.frame(n, m));
    else { this.note.textContent="当前浏览器不支持逐帧跟踪；保留证据帧"; this.stats.lastStatus="unsupported"; }
  }
  close() {
    this.closed = true; this.generation++;
    if(this.callback) this.video.cancelVideoFrameCallback(this.callback);
    this.controller?.abort();
  }
  clear(message) {
    this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    this.note.textContent=message;
  }
  paint(image, objects=[]) {
    const ctx=this.ctx, w=this.canvas.width, h=this.canvas.height;
    const iw=image.width, ih=image.height, k=Math.min(w/iw,h/ih), rw=iw*k,rh=ih*k,x=(w-rw)/2,y=(h-rh)/2;
    ctx.clearRect(0,0,w,h); ctx.drawImage(image,x,y,rw,rh);
    for(const object of objects) {
      const b=object.bbox;
      // YOLO detector boxes carry event_state (green=normal, red=alert) and are
      // drawn solid; the optical-flow estimate stays a blue dashed box.
      const yolo = object.event_state!==undefined;
      const color = object.event_state==="alert" ? "#ff4d4d" : yolo ? "#3ddc84" : "#57c7ff";
      ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash(yolo?[]:[7,3]);
      ctx.strokeRect(x+b.x1*rw,y+b.y1*rh,(b.x2-b.x1)*rw,(b.y2-b.y1)*rh);
      ctx.setLineDash([]);ctx.fillStyle=color;ctx.font="13px sans-serif";
      const label = yolo ? (object.event_state==="alert" ? `${object.track_id} · 事件` : object.track_id)
        : `T${Number(object.track_id.split(":").at(-1))+1} · 跟踪估计`;
      ctx.fillText(label,x+b.x1*rw,Math.max(14,y+b.y1*rh-4));
    }
    const now=performance.now();
    if(this.lastPaint && this.playing) this.stats.maxPaintGapSeconds=Math.max(this.stats.maxPaintGapSeconds,(now-this.lastPaint)/1000);
    this.lastPaint=now;this.stats.painted++;
  }
  async frame(now, metadata) {
    if(this.closed) return;
    this.callback=this.video.requestVideoFrameCallback((n,m)=>this.frame(n,m));
    if(!this.playing || this.video.seeking || !this.video.videoWidth) return;
    const local=metadata.mediaTime, at=this.epoch*this.media.duration+local;
    if(at===this.lastTime) return;
    this.lastTime=at;this.stats.frames++;
    const height=Math.max(8,Math.round(320*this.video.videoHeight/this.video.videoWidth));
    if(this.captureCanvas.height!==height) this.captureCanvas.height=height;
    this.captureContext.drawImage(this.video,0,0,320,height);
    // A slow tracker cannot freeze the video or leave an old person's box up.
    // Detector boxes update in round-robin across streams; keep the last ones
    // drawn (briefly) so they persist on live frames instead of flickering.
    const held = (this.lastObjects && now-this.lastObjectsAt<this.holdMs) ? this.lastObjects : [];
    if(this.busy) {
      this.stats.busySkipped++;
      if(now-this.lastPaint>100) {this.paint(this.captureCanvas,held);if(!held.length)this.note.textContent="跟踪处理中 · 暂不显示旧框";}
      return;
    }
    // Detector fetches are long (a GPU round trip under 12-stream load can take
    // seconds). The browser allows only ~6 concurrent HTTP/1.1 connections per
    // origin, so 12 lanes fetching at once would starve the evidence-frame image
    // and video requests on the same server ("正在加载对应证据帧" never resolving).
    // Cap how many detector fetches are in flight across all lanes, leaving
    // connections for evidence/video; the server serializes on one GPU anyway,
    // so this costs no box coverage. Non-detector tracking keeps its old pacing.
    if(this.detectorMode) {
      const cls = window.PreviewTracking;
      if(cls.inflight === undefined) { cls.inflight = 0; cls.maxInflight = 4; }
      if(cls.inflight >= cls.maxInflight) { this.stats.budgetSkipped=(this.stats.budgetSkipped||0)+1; return; }
    }
    if(now-this.lastPaint>150) this.paint(this.captureCanvas,held);
    this.busy=true;this.stats.requests++;
    if(this.detectorMode) window.PreviewTracking.inflight++;
    const generation=this.generation, epoch=this.epoch, started=performance.now();
    const snapshot=document.createElement("canvas");snapshot.width=320;snapshot.height=height;
    snapshot.getContext("2d").drawImage(this.captureCanvas,0,0);
    const jpeg=snapshot.toDataURL("image/jpeg",0.85).split(",")[1];
    this.controller=new AbortController();const timeout=setTimeout(()=>this.controller.abort(),this.abortMs);
    try {
      const response=await fetch(`tracking/${encodeURIComponent(this.stage)}/${this.camera}`,{
        method:"POST",headers:{"Content-Type":"application/json"},signal:this.controller.signal,
        body:JSON.stringify({client:this.client,time:at,epoch,jpeg})});
      if(!response.ok) throw new Error("tracking HTTP");
      const result=await response.json(), end=performance.now();
      // YOLO runs on a remote GPU, so allow a larger round trip than the local
      // optical-flow estimate before treating a response as stale.
      const yolo=result.event_state!==undefined;
      if(this.closed || !this.playing || generation!==this.generation || epoch!==this.epoch || end-started>(yolo?this.staleMs:250)) {this.stats.staleResponses++;return;}
      this.stats.lastStatus=result.status;this.stats.lost=result.lost_tracks || 0;
      this.stats.framesWithBoxes+=Number(result.objects.length>0);
      this.stats.samples.push({time:at,anchor:result.anchor_id,status:result.status,objects:result.objects});
      if(this.stats.samples.length>128) this.stats.samples.shift();
      this.stats.roundTripSeconds.push((end-started)/1000);
      this.stats.computeSeconds.push(result.tracking_seconds || 0);
      // Bounded diagnostics even in a long monitoring session.
      if(this.stats.roundTripSeconds.length>5000) {this.stats.roundTripSeconds.shift();this.stats.computeSeconds.shift();}
      this.paint(snapshot,result.objects);
      if(yolo){this.lastObjects=result.objects;this.lastObjectsAt=performance.now();}
      this.note.textContent = yolo ?
          (result.status==="detector_unavailable" ? "检测器暂不可用 · 不显示框" :
           result.objects.length ? `YOLO 检测 ${result.objects.length} 人${result.event_state==="alert"?" · 事件告警":""}` :
           "YOLO · 未检测到人") :
        result.status==="tracking" ? `跟踪估计 ${result.objects.length} 人 · 定位证据已过去 ${result.anchor_age.toFixed(1)} s` :
        result.status==="no_person_boxes" ? "此定位结果未提供人框" : result.status==="waiting_for_grounding" ? "连续播放 · 等待有效定位结果" : "跟踪丢失 / 不确定 · 等待新定位，已隐藏框";
    } catch(error) {
      if(!this.closed) {
        this.stats.errors++;this.stats.lastStatus="error";
        if(this.playing && generation===this.generation && epoch===this.epoch && performance.now()-started<=250) this.paint(snapshot);
        this.note.textContent="跟踪暂不可用 · 不显示旧框";
      }
    } finally {clearTimeout(timeout);this.busy=false;if(this.detectorMode)window.PreviewTracking.inflight--;}
  }
};
