"use strict";
// Only the preview uses tracking. No tracked box becomes a VLM observation.
window.PreviewTracking = class {
  static inflight = 0;
  static maxInflight = 4;
  static waiters = [];
  constructor(video, canvas, note, camera, media, evidence=null) {
    Object.assign(this, {video, canvas, note, camera, media, evidence});
    this.evidenceTime=null;
    this.stage = window.currentStage;
    this.client = window.trackingClient;
    this.captureCanvas = document.createElement("canvas");
    this.captureCanvas.width = 320;
    this.captureCanvas.height = 180;
    this.busy = false; this.closed = false; this.epoch = 0; this.generation = 0;
    this.lastPaint = 0; this.lastTime = -1; this.lastLocal = null; this._playing = false;
    this.lastObjects = []; this.lastObjectsTime = -Infinity;
    this.detectorMode = !!window.previewDetector;
    this.abortMs = this.detectorMode ? 4000 : 1500;
    // A multi-second GPU reply is useful as a timing sample, but its boxes are
    // no longer evidence about the current frame. Never hold them across a loop.
    this.maxSourceAge = 0.25;
    this.stats = {frames:0, painted:0, requests:0, lost:0, errors:0, framesWithBoxes:0, evidenceFramesWithBoxes:0, evidenceSamples:[], samples:[], busySkipped:0, budgetSkipped:0, staleResponses:0,
      maxPaintGapSeconds:0, roundTripSeconds:[], computeSeconds:[], lastStatus:"waiting_for_grounding"};
    this.ctx = canvas.getContext("2d");
    this.captureContext = this.captureCanvas.getContext("2d");
    video.addEventListener("seeking", () => this.invalidate("画面跳转 · 等待重新定位"));
    if(video.requestVideoFrameCallback) this.callback = video.requestVideoFrameCallback((n, m) => this.frame(n, m));
    else { this.note.textContent="当前浏览器不支持逐帧跟踪；保留证据帧"; this.stats.lastStatus="unsupported"; }
  }
  get playing() {return this._playing;}
  set playing(value) {
    value=!!value;
    if(this._playing===value) return;
    this._playing=value;
    if(!value) {this.invalidate("播放已结束 · 已隐藏检测框");this.stats.lastStatus="stopped";}
  }
  close() {
    this.closed = true; this._playing = false;
    this.invalidate("预览已关闭");
    if(this.callback) this.video.cancelVideoFrameCallback(this.callback);
  }
  invalidate(message) {
    const budget=window.PreviewTracking;
    budget.waiters=budget.waiters.filter(waiter=>waiter!==this);
    this.generation++; this.controller?.abort(); this.lastTime=-1;
    this.clear(message);
  }
  clearOverlay(message) {
    this.lastObjects=[]; this.lastObjectsTime=-Infinity;
    this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    this.note.textContent=message;
  }
  clear(message) {
    this.clearOverlay(message);this.evidenceTime=null;
    if(this.evidence) {
      this.evidence.canvas.getContext("2d").clearRect(0,0,this.evidence.canvas.width,this.evidence.canvas.height);
      this.evidence.container.hidden=true;
      this.evidence.note.textContent="";
    }
  }
  drawBoxes(ctx, image, objects, w, h) {
    const iw=image.width, ih=image.height, k=Math.min(w/iw,h/ih), rw=iw*k,rh=ih*k,x=(w-rw)/2,y=(h-rh)/2;
    const scale=Math.min(ctx.canvas.clientWidth/w,ctx.canvas.clientHeight/h) || 1;
    for(const object of objects) {
      const b=object.bbox;
      if(!b || ![b.x1,b.y1,b.x2,b.y2].every(v=>Number.isFinite(v)&&v>=0&&v<=1) || b.x2<=b.x1 || b.y2<=b.y1) continue;
      // Independent detector playback is not aligned with benchmark verdicts;
      // its neutral boxes carry no event/normal assertion.
      const color = this.detectorMode ? "#ffd68b" : "#57c7ff";
      ctx.strokeStyle=color; ctx.lineWidth=2/scale; ctx.setLineDash(this.detectorMode?[]:[7/scale,3/scale]);
      ctx.strokeRect(x+b.x1*rw,y+b.y1*rh,(b.x2-b.x1)*rw,(b.y2-b.y1)*rh);
      ctx.setLineDash([]);ctx.fillStyle=color;ctx.font=`${13/scale}px sans-serif`;
      const label = this.detectorMode ? `${object.track_id} · 检测`
        : `T${Number(String(object.track_id).split(":").at(-1))+1} · 跟踪估计`;
      ctx.fillText(label,x+b.x1*rw,Math.max(14/scale,y+b.y1*rh-4/scale));
    }
  }
  paint(image, objects=[]) {
    // Match the source aspect ratio before CSS contains the transparent overlay.
    const height=Math.round(this.canvas.width*image.height/image.width);
    if(this.canvas.height!==height) this.canvas.height=height;
    // Keep the playing video visible underneath. A delayed network reply must
    // not replace it with the old request JPEG or freeze it after the run stops.
    this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    this.drawBoxes(this.ctx,image,objects,this.canvas.width,this.canvas.height);
    const now=performance.now();
    if(this.lastPaint && this.playing) this.stats.maxPaintGapSeconds=Math.max(this.stats.maxPaintGapSeconds,(now-this.lastPaint)/1000);
    this.lastPaint=now;this.stats.painted++;
  }
  paintEvidence(image, objects, sourceTime) {
    if(!this.evidence) return;
    const canvas=this.evidence.canvas,ctx=canvas.getContext("2d");
    canvas.width=image.width;canvas.height=image.height;
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);
    this.drawBoxes(ctx,image,objects,canvas.width,canvas.height);
    this.evidenceTime=sourceTime;this.evidence.container.hidden=false;
    this.stats.evidenceFramesWithBoxes+=Number(objects.length>0);
    this.stats.evidenceSamples.push({time:sourceTime,epoch:this.epoch,age:Math.max(0,this.lastTime-sourceTime),objects});
    if(this.stats.evidenceSamples.length>128) this.stats.evidenceSamples.shift();
    this.updateEvidenceAge();
  }
  updateEvidenceAge() {
    if(this.evidenceTime===null || !this.evidence) return;
    const local=this.evidenceTime%this.media.duration,age=Math.max(0,this.lastTime-this.evidenceTime);
    this.evidence.note.textContent=`YOLO 证据 ${local.toFixed(2)} s · 第 ${this.epoch+1} 轮 · ${age.toFixed(2)} s 前`;
  }
  acquireBudget(now) {
    // Leave browser connections available for state polls, video and model
    // evidence. A FIFO turn queue prevents the earliest four camera callbacks
    // from monopolizing newly freed slots; each admitted lane captures afresh.
    const budget=window.PreviewTracking;
    budget.waiters=budget.waiters.filter(waiter=>
      !waiter.closed && waiter.playing && !waiter.busy && !waiter.video.seeking
      && now-waiter.budgetWaitingAt<1000);
    this.budgetWaitingAt=now;
    if(!budget.waiters.includes(this)) budget.waiters.push(this);
    if(budget.inflight>=budget.maxInflight || budget.waiters[0]!==this) return null;
    budget.waiters.shift();budget.inflight++;
    return budget;
  }
  async frame(now, metadata) {
    if(this.closed) return;
    this.callback=this.video.requestVideoFrameCallback((n,m)=>this.frame(n,m));
    if(!this.playing || this.video.seeking || !this.video.videoWidth) return;
    const local=metadata.mediaTime;
    if(this.detectorMode && this.lastLocal!==null && local<this.lastLocal-0.05) {
      this.epoch++; this.invalidate("视频循环 · 等待本轮检测框");
    }
    this.lastLocal=local;
    const at=this.epoch*this.media.duration+local;
    if(at===this.lastTime) return;
    this.lastTime=at;this.stats.frames++;this.updateEvidenceAge();
    const height=Math.max(8,Math.round(320*this.video.videoHeight/this.video.videoWidth));
    if(this.captureCanvas.height!==height) this.captureCanvas.height=height;
    this.captureContext.drawImage(this.video,0,0,320,height);
    const held = at>=this.lastObjectsTime && at-this.lastObjectsTime<=this.maxSourceAge ? this.lastObjects : [];
    if(this.busy) {
      this.stats.busySkipped++;
      this.paint(this.captureCanvas,held);
      if(!held.length)this.note.textContent="检测处理中 · 暂不显示旧框";
      return;
    }
    this.paint(this.captureCanvas,held);
    const budget=this.detectorMode ? this.acquireBudget(now) : null;
    if(this.detectorMode && !budget) {
      this.stats.budgetSkipped++;
      if(!held.length)this.note.textContent="检测排队 · 暂不显示旧框";
      return;
    }
    this.busy=true;this.stats.requests++;
    const generation=this.generation, epoch=this.epoch, started=performance.now();
    let timeout;
    try {
      // Capture after acquiring a slot: queued lanes send their current frame,
      // never an old JPEG held while waiting for another lane's network reply.
      const snapshot=document.createElement("canvas");snapshot.width=320;snapshot.height=height;
      snapshot.getContext("2d").drawImage(this.captureCanvas,0,0);
      const jpeg=snapshot.toDataURL("image/jpeg",0.85).split(",")[1];
      const controller=new AbortController();this.controller=controller;
      timeout=setTimeout(()=>controller.abort(),this.abortMs);
      const response=await fetch(`tracking/${encodeURIComponent(this.stage)}/${this.camera}`,{
        method:"POST",headers:{"Content-Type":"application/json"},signal:controller.signal,
        body:JSON.stringify({client:this.client,time:at,epoch,generation,jpeg})});
      if(!response.ok) throw new Error("tracking HTTP");
      const result=await response.json(), end=performance.now();
      this.stats.roundTripSeconds.push((end-started)/1000);
      this.stats.computeSeconds.push(result.tracking_seconds || 0);
      if(this.stats.roundTripSeconds.length>5000) {this.stats.roundTripSeconds.shift();this.stats.computeSeconds.shift();}
      const sourceTime=result.source_time, sourceAge=this.lastTime-sourceTime;
      if(this.closed || !this.playing || generation!==this.generation || epoch!==this.epoch) {this.stats.staleResponses++;return;}
      if(!Number.isFinite(sourceTime) || Math.abs(sourceTime-at)>1e-6 || (result.detector && result.epoch!==epoch)
          || sourceAge<0 || end-started>this.abortMs) {
        this.stats.staleResponses++;this.stats.lastStatus="stale";
        this.clear("检测结果已过期 · 已隐藏旧框");return;
      }
      if(result.status==="stopped") {this.clear("播放已结束 · 已隐藏检测框");this.stats.lastStatus="stopped";return;}
      const objects=Array.isArray(result.objects)?result.objects:[];
      if(this.detectorMode) this.paintEvidence(snapshot,objects,sourceTime);
      // Slow valid detector replies stay inspectable on their captured frame;
      // they must not place that older frame's boxes over the current video.
      if(sourceAge>this.maxSourceAge || end-started>this.maxSourceAge*1000) {
        this.stats.staleResponses++;this.stats.lastStatus="evidence_only";
        this.clearOverlay(this.detectorMode ? "YOLO 延迟结果见证据小窗 · 当前帧不显示旧框" : "跟踪结果已过期 · 已隐藏旧框");return;
      }
      this.stats.lastStatus=result.status;this.stats.lost=result.lost_tracks || 0;
      this.stats.framesWithBoxes+=Number(objects.length>0);
      this.stats.samples.push({time:at,epoch,anchor:result.anchor_id,status:result.status,objects});
      if(this.stats.samples.length>128) this.stats.samples.shift();
      this.lastObjects=objects;this.lastObjectsTime=sourceTime;
      this.paint(this.captureCanvas,objects);
      this.note.textContent = this.detectorMode ?
          (result.status==="detector_unavailable" ? "检测器暂不可用 · 不显示框" :
           result.status==="stopped" ? "播放已结束 · 已隐藏检测框" :
           objects.length ? `YOLO 检测 ${objects.length} 人 · 框不表示事件判断` : "YOLO · 未检测到人") :
        result.status==="tracking" ? `跟踪估计 ${objects.length} 人 · 定位证据已过去 ${result.anchor_age.toFixed(1)} s` :
        result.status==="no_person_boxes" ? "此定位结果未提供人框" : result.status==="waiting_for_grounding" ? "连续播放 · 等待有效定位结果" : "跟踪丢失 / 不确定 · 等待新定位，已隐藏框";
    } catch(error) {
      if(!this.closed && generation===this.generation && epoch===this.epoch && this.playing) {
        this.stats.errors++;this.stats.lastStatus="error";
        this.clear("跟踪暂不可用 · 不显示旧框");
      }
    } finally {
      clearTimeout(timeout);this.busy=false;
      if(budget) budget.inflight--;
    }
  }
};
