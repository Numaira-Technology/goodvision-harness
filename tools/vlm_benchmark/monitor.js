"use strict";
(() => {
  const labels = {fight:"打架",smoking:"吸烟",fall:"跌倒",weapon:"武器",ppe_violation:"防护违规",ground_litter:"地面垃圾"};
  const phaseNames = {preflight:"接口预检",warmup:"预热",measurement:"正式测试",soak:"持续测试"};
  const stateNames = {preparing:"准备播放",playing:"播放与检测中",draining:"播放结束 · 等待剩余结果",finished:"本轮完成",failed:"本轮中断"};
  const byId = id => document.getElementById(id);
  const seconds = v => Number.isFinite(v) ? `${v.toFixed(2)} s` : "—";
  let state, anchoredAt = 0, cards = [], lastPoll = 0, rafFrames = 0;
  window.currentStage = null; window.monitorReady = false; window.finalPainted = false;
  const resetDiagnostics = () => window.monitorDiagnostics = {sawPlaying:false,finalPainted:false,pendingEvidence:0,maxPollGapSeconds:0,maxPlaybackGapSeconds:0,mediaErrors:[],paintFrames:0,phases:[],videoFrames:[]};
  resetDiagnostics();
  function node(tag, cls, text) {
    const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e;
  }
  function setup(data) {
    window.currentStage = data.stage; window.monitorReady = false; window.finalPainted = false; resetDiagnostics();
    rafFrames = 0; cards.forEach(c => { if(c.video) { c.video.pause(); c.video.removeAttribute("src"); c.video.load(); } });
    cards = []; byId("grid").replaceChildren(); byId("summary").hidden = true;
    const count = data.cameras.length;
    document.body.classList.toggle("compact", count > 4);
    document.body.classList.toggle("multicam", count > 1);
    byId("grid").style.setProperty("--cols", count > 8 ? 4 : count > 4 ? 3 : count > 1 ? 2 : 1);
    byId("grid").style.maxWidth = count === 1 ? "1340px" : "none";
    data.cameras.forEach((camera, i) => {
      const card = node("section","camera"), head = node("div","camera-head");
      const clock = node("span","source-clock","等待输入"); head.append(node("strong","",`视频 ${String(i+1).padStart(2,"0")}`),clock); card.append(head);
      const pair = node("div","framepair"), source = node("div"), evidence = node("div");
      source.append(node("div","frame-label","当前播放 · 1×")); evidence.append(node("div","frame-label","结果对应的证据帧"));
      const sourceScreen = node("div","screen"), evidenceScreen = node("div","screen");
      const media = data.media[i % data.media.length]; let video = null, sample = null;
      if(media.url) {
        video = node("video"); video.muted = true; video.playsInline = true; video.preload = "auto"; video.src = media.url;
        video.addEventListener("error", () => window.monitorDiagnostics.mediaErrors.push(`video-${i+1}:decode`));
        sourceScreen.append(video);
      } else { sample = node("img"); sourceScreen.append(sample); }
      const canvas = node("canvas"); canvas.width = 640; canvas.height = 360;
      const empty = node("div","empty","等待对应结果"); evidenceScreen.append(canvas,empty);
      source.append(sourceScreen); evidence.append(evidenceScreen); pair.append(source,evidence); card.append(pair);
      const note = node("div","evidence-note","检测结果按实际返回时刻更新"); card.append(note);
      const laneNodes = {};
      data.lanes.forEach(lane => {
        const row = node("div","lane"), chips = node("div","chips"), timer = node("div","lane-time","尚未收到结果");
        row.append(node("div","lane-name",lane === "events" ? "事件" : "定位"),chips,timer); card.append(row);
        for(const [key,label] of Object.entries(labels)) { const chip = node("span","chip",label); chip.dataset.label = key; chips.append(chip); }
        laneNodes[lane] = {chips,timer};
      });
      const foot = node("div","camera-foot","模型判断不等于已人工核实的事件"); card.append(foot); byId("grid").append(card);
      cards.push({video,sample,clock,canvas,empty,note,foot,lanes:laneNodes,media,evidenceKey:null,evidenceLoading:false,sourceTick:null,playRequested:false,lastPresented:0,progressAt:0});
    });
  }
  function drawEvidence(card, camera, report) {
    const stage = window.currentStage, key = `${stage}:${report.id}:${report.tick}`;
    if(card.evidenceKey === key) return;
    card.evidenceKey = key; card.evidenceLoading = true; card.empty.textContent = "正在加载对应证据帧"; card.empty.hidden = false;
    const ctx = card.canvas.getContext("2d"); ctx.clearRect(0,0,640,360);
    if(report.tick === null) { card.evidenceLoading=false; card.empty.textContent = "此结果缺少可对应的帧"; window.monitorDiagnostics.mediaErrors.push(`evidence-${camera.camera}:missing-frame`); return; }
    const img = new Image();
    img.onload = () => {
      if(card.evidenceKey !== key || window.currentStage !== stage) return;
      const k = Math.min(640/img.naturalWidth,360/img.naturalHeight), w=img.naturalWidth*k, h=img.naturalHeight*k, x=(640-w)/2, y=(360-h)/2;
      ctx.clearRect(0,0,640,360); ctx.drawImage(img,x,y,w,h);
      for(const object of report.objects) {
        const b = object.bbox; if(!b || ![b.x1,b.y1,b.x2,b.y2].every(v => Number.isFinite(v) && v>=0 && v<=1) || b.x2<=b.x1 || b.y2<=b.y1) continue;
        ctx.strokeStyle="#4ee7ba"; ctx.lineWidth=3; ctx.strokeRect(x+b.x1*w,y+b.y1*h,(b.x2-b.x1)*w,(b.y2-b.y1)*h);
      }
      card.empty.hidden = true; card.evidenceLoading = false;
    };
    img.onerror = () => { if(card.evidenceKey === key && window.currentStage === stage) {card.evidenceLoading=false;card.empty.textContent="证据帧加载失败"; window.monitorDiagnostics.mediaErrors.push(`evidence-${camera.camera}:${report.tick}`);} };
    img.src = `frame/${encodeURIComponent(window.currentStage)}/${camera.camera-1}/${report.tick}.jpg`;
  }
  function render(data) {
    if(!data.stage) return;
    if(data.stage !== window.currentStage) setup(data);
    state=data; anchoredAt=performance.now();
    const d=window.monitorDiagnostics;
    if(!d.phases.includes(data.state)) d.phases.push(data.state);
    if(data.state === "playing") d.sawPlaying=true;
    byId("model").textContent=`${data.model} · ${data.backend === "openrouter" ? "OpenRouter 服务商推理" : "自部署服务"}`;
    byId("badge").textContent=data.real ? "真实模型测试" : "模拟调度 · 未调用真实模型";
    byId("stage").textContent=`${phaseNames[data.phase]} / ${data.mode} / ${data.cameras.length} 路 / 第 ${data.repeat} 轮`;
    byId("notice").textContent=`${data.source_count === 1 && data.cameras.length>1 ? "同一视频复制为多路负载；" : ""}模型判断：绿=未检测到，红=检测到，黄=不确定，灰=等待，粉=失败。预览 15 FPS 与请求频率 ${data.observation_fps} Hz 分开。没有定位框不代表无人。`;
    const expected=data.planned;
    const metrics=[];
    data.lanes.forEach(lane => {
      const final=data.summary?.lanes?.[lane], c=data.counters[lane];
      const box=node("div","metric"); box.append(node("span","label",lane === "events" ? "事件完整结果" : "定位分支完整结果"));
      box.append(node("strong","",`${final?.complete_valid_results ?? c.complete} / ${final?.expected_observations ?? expected}`));
      box.append(node("span","muted",`P50 ${seconds(final?.source_to_publish_seconds?.p50 ?? c.p50)} · P95 ${seconds(final?.source_to_publish_seconds?.p95 ?? c.p95)}`)); metrics.push(box);
    }); byId("metrics").replaceChildren(...metrics);
    data.cameras.forEach((camera,i) => {
      const c=cards[i];
      if(c.sample && camera.tick>=0 && c.sourceTick!==camera.tick) {c.sourceTick=camera.tick;c.sample.src=`frame/${encodeURIComponent(data.stage)}/${i}/${camera.tick}.jpg`;}
      for(const lane of data.lanes) {
        const r=camera.lanes[lane], row=c.lanes[lane];
        if(!r) continue;
        for(const chip of row.chips.children) {
          const judgment=r.judgments.find(j=>j.label===chip.dataset.label);
          chip.className=`chip ${r.error || !r.complete ? "failed" : !judgment ? "" : judgment.observed===true ? "positive" : judgment.observed===false ? "negative" : "uncertain"}`;
          chip.title=r.error || (judgment?.observed===true ? "模型检测到" : judgment?.observed===false ? "模型未检测到" : "不确定");
        }
        row.report=r;
      }
      const evidence=camera.lanes.grounding || camera.lanes.events;
      if(evidence) { drawEvidence(c,camera,evidence); c.note.textContent=`证据 ${seconds(evidence.media_seconds)} · 帧 ${evidence.tick ?? "?"} · ${camera.lanes.grounding ? `目标框 ${evidence.objects.length}` : "事件模式不提供定位框"}`; }
      c.foot.textContent=`送帧拒绝 ${camera.rejected} · 保留旧结果 ${camera.out_of_order} · 模型判断未经人工复核`;
    });
    if(["finished","failed"].includes(data.state)) {
      const summary=byId("summary"); summary.hidden=false;
      summary.textContent=data.state === "failed" ? `本轮中断：${data.error}。保留已有结果与部分录屏。` : `最终统计已生成 · 输入时序 ${data.summary.input_timing_valid ? "有效" : "无效"} · 队列 ${data.summary.all_drained ? "已排空" : "未排空"} · 记录丢失 ${data.summary.result_records_missing}。完整输出不代表识别准确率。`;
      requestAnimationFrame(()=>requestAnimationFrame(()=> {
        if(window.currentStage !== data.stage) return;
        d.pendingEvidence=cards.filter(c=>c.evidenceLoading).length;
        window.finalPainted=summary.getBoundingClientRect().bottom<=innerHeight && d.pendingEvidence===0;
        d.finalPainted=window.finalPainted;
      }));
    }
  }
  function animate() {
    if(state?.stage) {
      const elapsed=state.finished_elapsed ?? state.elapsed+(performance.now()-anchoredAt)/1000;
      byId("clock").textContent=`${stateNames[state.state]} · 运行 ${seconds(state.finished_elapsed ?? elapsed)}`;
      cards.forEach((c,i)=> {
        for(const row of Object.values(c.lanes)) {
          const r=row.report; if(!r) continue;
          row.timer.textContent=`证据 ${seconds(r.media_seconds)} · 耗时 ${seconds(r.latency)} · 更新于 ${seconds(Number.isFinite(r.received_elapsed) ? Math.max(0,elapsed-r.received_elapsed) : null)}前${r.error ? ` · ${r.error}` : !r.complete ? " · 输出不完整" : ""}`;
        }
        const time=state.state === "preparing" ? 0 : Math.max(0,Math.min(state.duration-.001,elapsed-state.offsets[i]));
        const pos=time*state.sample_fps, index=Math.floor(pos)%c.media.sample_count;
        const a=c.media.timestamps[index], b=index+1<c.media.sample_count ? c.media.timestamps[index+1] : c.media.duration;
        const media=a+(b-a)*(pos-Math.floor(pos));
        c.clock.textContent=`视频 ${seconds(media)} · 本轮 ${seconds(Math.max(0,Math.min(time,state.duration)))}`;
        if(c.video) {
          const playing=state.state === "playing" && elapsed>=state.offsets[i] && time<state.duration-.01;
          const quality=c.video.getVideoPlaybackQuality();
          const presented=quality.totalVideoFrames-quality.droppedVideoFrames;
          if(!playing || presented>c.lastPresented || !c.progressAt) c.progressAt=performance.now();
          if(playing) window.monitorDiagnostics.maxPlaybackGapSeconds=Math.max(window.monitorDiagnostics.maxPlaybackGapSeconds,(performance.now()-c.progressAt)/1000);
          c.lastPresented=presented;
          if(c.video.readyState>=2) {
            if(Math.abs(c.video.currentTime-media)>.3 && !c.video.seeking) c.video.currentTime=media;
            if(playing && c.video.paused && !c.playRequested) {c.playRequested=true;c.video.play().catch(()=>window.monitorDiagnostics.mediaErrors.push(`video-${i+1}:autoplay`)).finally(()=>c.playRequested=false);}
          }
          if(!playing && !c.video.paused) c.video.pause();
        }
      });
      window.monitorReady=cards.length>0 && cards.every(c=>!c.video || c.video.readyState>=2);
      const d=window.monitorDiagnostics; d.paintFrames=++rafFrames; d.pendingEvidence=cards.filter(c=>c.evidenceLoading).length;
      d.videoFrames=cards.map(c=>c.video ? c.lastPresented : null);
    }
    requestAnimationFrame(animate);
  }
  async function poll() {
    try {
      const response=await fetch("state",{cache:"no-store"}); if(!response.ok) throw new Error("HTTP");
      const data=await response.json(), now=performance.now();
      if(lastPoll && data.stage===window.currentStage) window.monitorDiagnostics.maxPollGapSeconds=Math.max(window.monitorDiagnostics.maxPollGapSeconds,(now-lastPoll)/1000);
      lastPoll=now; render(data); byId("connection").textContent="";
    } catch(_) {byId("connection").textContent="监看连接中断，当前结果可能已过期";}
    setTimeout(poll,150);
  }
  poll(); requestAnimationFrame(animate);
})();
