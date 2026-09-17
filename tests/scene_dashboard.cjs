/* Run the shipped dashboard handlers; no production behavior is duplicated. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

class Element {
  constructor() {
    this.textContent = ""; this.className = ""; this.children = [];
    this.dataset = {}; this.style = {}; this.value = ""; this.checked = false;
    this.classList = {
      add: name => { this.className += " " + name; },
      remove: name => { this.className = this.className.split(" ").filter(x => x !== name).join(" "); },
      toggle: (name, on) => on ? this.classList.add(name) : this.classList.remove(name),
    };
  }
  set innerHTML(value) { throw new Error("Untrusted content must not reach innerHTML: " + value); }
  append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
  appendChild(item) { this.append(item); }
  prepend(item) { item.parent = this; this.children.unshift(item); }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  remove() { this.parent.children = this.parent.children.filter(x => x !== this); }
  get lastChild() { return this.children.at(-1); }
  querySelectorAll() { return []; }
  getAttribute(name) { return this[name] || null; }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
}

function dashboard() {
  let now = 0, timerId = 0;
  const elements = new Map(), timers = new Map(), requests = [], streams = [];
  const features = ["fight", "smoking", "fall", "weapon", "stranger"];
  const inputs = features.map(feature => Object.assign(new Element(), {checked: true, dataset: {feature}}));
  const document = {
    hidden: false,
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    querySelectorAll: () => inputs,
    createElement: () => new Element(),
  };
  class Clock extends Date { static now() { return now; } }
  const context = vm.createContext({document, window: {isSecureContext: true},
    location: {hostname: "localhost"}, Date: Clock, performance: {now: () => now}, AbortController, URLSearchParams,
    EventSource: class {
      constructor(url) { this.url = url; this.listeners = new Map(); streams.push(this); }
      addEventListener(name, listener) { this.listeners.set(name, listener); }
      close() { this.closed = true; }
      emit(name, data) { this.listeners.get(name)({data: JSON.stringify(data)}); }
    },
    setTimeout(callback, delay) { const key = ++timerId; timers.set(key, {callback, delay}); return key; },
    clearTimeout(key) { timers.delete(key); }, clearInterval() {}, requestAnimationFrame() {},
    fetch(url, options) { return new Promise((resolve, reject) => {
      const request = {url, options, resolve, reject}; requests.push(request);
      options.signal?.addEventListener("abort", () => reject(new Error("aborted")));
    }); },
  });
  const html = fs.readFileSync(path.join(__dirname, "../src/goodvision_harness/dashboard.html"), "utf8");
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  new vm.Script(script);
  vm.runInContext(script.slice(0, script.indexOf("ui.defaultStreamUrl.oninput=")) + `
    state.running=true;state.sessionId="session-a";state.sessionFeatures=detectionEvents;
    state.gallery={ready:true};state.galleryLoading=false;
    globalThis.app={state,sceneView,sceneUi,acceptSceneSnapshot,renderScene,renderSceneDecision,
      configureScene,resetSceneView,stopScenePolling,startScenePolling,analysis,analysisFace,
      renderClassification,controls,updateRuntimeSummary,status,message,localCameraDisplay,
      archiveView,loadEventArchive,connectArchiveEvents};
  `, context);
  context.app.sceneView.enabled = true;
  context.app.sceneView.mode = "active";
  return {app: context.app, document, requests, timers, streams, context,
    element: id => document.getElementById(id),
    elapse(ms) { now += ms; },
    tick(delay = 750) {
      const entry = [...timers].find(([, timer]) => timer.delay === delay);
      assert.ok(entry, "Expected timer with delay " + delay);
      timers.delete(entry[0]); entry[1].callback();
    },
  };
}

function snapshot({events = [], entities = [], timestamp = 1, session = "session-a", state = "running", ...rest} = {}) {
  return {enabled: true, mode: "active", state, automatic_temporal_inference: true,
    event_labels: ["smoking", "fight", "fall", "weapon"],
    scene: {stream_session_id: session, source_id: "camera-1", timestamp_seconds: timestamp, entities, events, transitions: []},
    worker: {state: "running", succeeded_requests: 2, failed_requests: 0}, ...rest};
}
function event(phase = "confirmed", extra = {}) {
  return {event_id: "e-1", event_label: "smoking", subject_ids: ["p-1"], phase,
    observation_state: "observed", first_evidence_timestamp_seconds: 0.5,
    last_evidence_timestamp_seconds: 1, ...extra};
}
function person(extra = {}) {
  return {subject_id: "p-1", identity: {}, observation_state: "observed",
    last_observed_timestamp_seconds: 1, ...extra};
}
function text(element) { return element.textContent + element.children.map(text).join(" "); }
const settle = () => new Promise(resolve => setImmediate(resolve));

test("status polling keeps one bounded request so slow status cannot crowd out contours", async () => {
  const view = dashboard();
  const first = view.app.status();
  for(let i=0;i<12;i++)view.app.status();
  assert.equal(view.requests.filter(request=>request.url==="/v1/status").length,1);
  view.tick(3000);await first;
  const retry=view.app.status();
  assert.equal(view.requests.filter(request=>request.url==="/v1/status").length,2);
  view.requests.at(-1).reject(new Error("offline"));await retry;
});

test("active mode points to temporal source evidence and hides the legacy frame links", () => {
  const view = dashboard();
  view.app.configureScene({enabled: true, mode: "active"});
  assert.equal(view.element("legacy-evidence-stream").hidden, true);
  assert.equal(view.element("legacy-evidence-latest").hidden, true);
  assert.equal(view.element("scene-evidence-latest").hidden, false);
  const html = fs.readFileSync(path.join(__dirname, "../src/goodvision_harness/dashboard.html"), "utf8");
  assert.match(html, /id="scene-evidence-latest"[^>]*href="\/v1\/scene\/evidence"/);
  assert.match(html, /evidence\.href="\/v1\/scene\/evidence"/);
  view.app.configureScene({enabled: false, mode: "disabled"});
  assert.equal(view.element("legacy-evidence-stream").hidden, false);
  assert.equal(view.element("legacy-evidence-latest").hidden, false);
  assert.equal(view.element("scene-evidence-latest").hidden, true);
});

test("empty scene waits for evidence and never becomes a green all-clear", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot());
  assert.equal(view.element("verdict").textContent, "等待证据");
  assert.equal(view.element("decision").className, "decision warming");
  assert.match(text(view.element("scene-events")), /等待连续观察证据/);
  assert.ok(view.element("event-statuses").children.every(row => row.className.includes("pending")));
});

test("event lifecycle and occlusion have distinct displays", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event("suspected")], entities: [person()]}));
  assert.equal(view.element("verdict").textContent, "正在复核");
  view.app.acceptSceneSnapshot(snapshot({events: [event()], entities: [person()], timestamp: 2}));
  assert.equal(view.element("verdict").textContent, "事件持续中");
  assert.match(view.element("decision").className, /detected/);
  view.app.acceptSceneSnapshot(snapshot({events: [event("confirmed", {observation_state: "uncertain"})],
    entities: [person({observation_state: "occluded"})], timestamp: 3}));
  assert.match(text(view.element("scene-entities")), /暂时遮挡/);
  assert.match(view.element("description").textContent, /暂不可见或证据不足/);
  view.app.acceptSceneSnapshot(snapshot({events: [event("ended")], timestamp: 4}));
  assert.match(text(view.element("event-statuses")), /已结束，继续观察/);
  assert.equal(view.element("decision").className, "decision warming");
  view.app.acceptSceneSnapshot(snapshot({events: [event("retracted")], timestamp: 5}));
  assert.match(text(view.element("scene-events")), /判断已修订/);
});

test("later per-frame negatives cannot clear a continuous event; face branch is retained", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event()]}));
  view.app.analysis({type: "analysis_result", stream_session_id: "session-a", message_id: "frame-20",
    sequence: 20, frame: {timestamp_seconds: 2, source_frame_index: 20}, model: {},
    judgments: [{event_label: "smoking", detected: false, confidence: 0.99},
      {event_label: "stranger", detected: true, confidence: 0.9, description: "face detail"}]});
  assert.equal(view.element("verdict").textContent, "事件持续中");
  assert.match(view.element("face-diagnostic-summary").textContent, /STRANGER ALERT.*face detail/);
});

test("face-only messages update face diagnostics without replacing event verdict", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event()]}));
  view.app.analysisFace({type: "analysis_result", stream_session_id: "session-a", sequence: 2,
    analysis_kind: "face_recognition", frame: {timestamp_seconds: 2}, model: {}, objects: []});
  assert.equal(view.element("verdict").textContent, "事件持续中");
  assert.match(view.element("face-diagnostic-summary").textContent, /人脸识别/);
  assert.equal(view.element("event-statuses").hidden, false);
});

test("removed shadow mode cannot enable a scene panel or accept snapshots", () => {
  const view = dashboard();
  view.app.configureScene({enabled: true, mode: "shadow"});
  view.element("verdict").textContent = "legacy verdict";
  assert.equal(view.app.acceptSceneSnapshot(snapshot({events: [event()], mode: "shadow"})), false);
  assert.equal(view.app.sceneView.mode, "disabled");
  assert.equal(view.app.sceneView.enabled, false);
  assert.equal(view.app.sceneView.snapshot, null);
  assert.equal(view.element("scene-panel").hidden, true);
  assert.equal(view.requests.length, 0);
  assert.equal(view.element("verdict").textContent, "legacy verdict");
});

test("stopped and stalled sources never imply current clear or current observation", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event()], entities: [person()]}));
  view.elapse(6000); view.app.renderScene();
  assert.match(view.element("scene-status").textContent, /画面未更新/);
  assert.match(text(view.element("scene-entities")), /当前状态待核实/);
  assert.match(text(view.element("event-statuses")), /持续中 · 待核实/);
  // Receiving a duplicate snapshot is not evidence that the camera advanced.
  view.app.acceptSceneSnapshot(snapshot({events: [event()]}));
  assert.match(view.element("scene-status").textContent, /画面未更新/);
  view.app.state.running = false; view.app.renderScene();
  assert.equal(view.element("verdict").textContent, "已停止");
  assert.match(text(view.element("scene-events")), /历史记录/);
  assert.equal(view.element("decision").className, "decision warming");
});

test("post-stop identity and evidence work stays visible without reporting live events", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event()], entities: [person()]}));
  view.elapse(6000);
  view.app.state.running = false;
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", events: [event()], entities: [person()],
    scene_drain: {state: "running", stage: "identity_review"}}));
  assert.equal(view.element("verdict").textContent, "后台收尾中");
  assert.match(view.element("scene-status").textContent, /复核人脸并合并人物/);
  assert.match(text(view.element("scene-events")), /后台整理中/);
  assert.doesNotMatch(text(view.element("scene-events")), /持续中|当前状态待核实/);
  assert.equal(view.element("decision").className, "decision warming");
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", events: [event("ended")],
    scene_drain: {state: "running", stage: "evidence"}}));
  assert.match(view.element("scene-status").textContent, /视频证据/);
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", events: [event("ended")],
    scene_drain: {state: "completed", stage: "evidence"}}));
  assert.equal(view.element("verdict").textContent, "已停止");
  assert.match(view.element("scene-status").textContent, /整理已完成/);
});

test("completed stranger review identifies the person and leaves no misleading active review", () => {
  const view = dashboard(); view.app.state.running = false;
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", entities: [person({identity: {status: "unknown"}})],
    events: [event("ended", {event_label: "stranger"}), event("suspected", {event_id: "e-2",
      review_state: "complete", review_outcome: "unconfirmed"})], stats: {identity_subjects_merged: 3},
    scene_drain: {state: "completed", stage: "evidence"}}));
  assert.match(text(view.element("scene-entities")), /已确认陌生人/);
  assert.doesNotMatch(text(view.element("scene-entities")), /身份待确认/);
  assert.match(text(view.element("scene-events")), /证据不足，未确认/);
  assert.doesNotMatch(text(view.element("scene-events")), /正在复核|持续中/);
  assert.match(view.element("scene-summary").textContent, /已按人脸合并 3 次/);
});

test("unconfirmed face status does not label a person as a confirmed stranger", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({entities: [person({identity: {status: "unknown"}})],
    events: [event("suspected", {event_label: "stranger"})]}));
  assert.match(text(view.element("scene-entities")), /身份待确认/);
  assert.doesNotMatch(text(view.element("scene-entities")), /已确认陌生人/);
});

test("events use confirmed database names and retain aliases for unconfirmed identities", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({entities: [
    person({identity: {status: "known", person_id: "gallery-frank", name: "Frank"}}),
    person({subject_id: "p-2", identity: {status: "unknown", name: "Unconfirmed"}}),
  ], events: [event("confirmed", {subject_ids: ["p-1", "p-2", "p-3"]})]}));
  assert.equal(view.element("scene-events").children[0].children[0].textContent, "吸烟 · Frank、人物 2、人物 3");
  assert.doesNotMatch(text(view.element("scene-events")), /Unconfirmed/);
});

test("late identity review updates historical event names without new source frames", () => {
  const view = dashboard(); view.app.state.running = false;
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", entities: [person()], events: [event("ended")]}));
  assert.equal(view.element("scene-events").children[0].children[0].textContent, "吸烟 · 人物 1");
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", entities: [
    person({identity: {status: "known", person_id: "gallery-frank", name: "Frank"}}),
  ], events: [event("ended")]}));
  assert.equal(view.element("scene-events").children[0].children[0].textContent, "吸烟 · Frank");
});

test("same database names retain subject context until identities are merged", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({entities: [
    person({identity: {status: "known", person_id: "gallery-1", name: "Frank"}}),
    person({subject_id: "p-2", identity: {status: "known", person_id: "gallery-2", name: "Frank"}}),
  ], events: [event("confirmed", {subject_ids: ["p-1", "p-2"]})]}));
  assert.equal(view.element("scene-events").children[0].children[0].textContent,
    "吸烟 · Frank（人物 1）、Frank（人物 2）");
});

test("event names resolve all scene entities, including those beyond the person card limit", () => {
  const view = dashboard();
  const entities = Array.from({length: 65}, (_, index) => person({subject_id: "p-" + index}));
  entities[64].identity = {status: "known", person_id: "gallery-frank", name: "Frank"};
  view.app.acceptSceneSnapshot(snapshot({entities, events: [event("confirmed", {subject_ids: ["p-64"]})]}));
  assert.equal(view.element("scene-entities").children.length, 64);
  assert.equal(view.element("scene-events").children[0].children[0].textContent, "吸烟 · Frank");
});

test("draining disables starting and repeated stopping and shows the current work stage", () => {
  const view = dashboard();
  view.app.state.info = {}; view.element("stream-url").value = "rtsp://camera/live";
  view.app.state.running = false; view.app.state.stopping = true;
  view.app.state.processingStatus = {pipeline_state: "draining", scene_drain: {state: "running", stage: "evidence"}};
  view.app.controls(); view.app.updateRuntimeSummary(view.app.state.processingStatus);
  assert.equal(view.element("start-button").disabled, true);
  assert.equal(view.element("stop-button").disabled, true);
  assert.equal(view.element("stop-button").textContent, "后台收尾中…");
  assert.match(view.element("connection-state").textContent, /后台收尾中/);
  assert.match(view.element("connection-detail").textContent, /视频证据/);
});

test("released live resources permit a new detection while the old run is being organized", async () => {
  for (const pipeline of ["idle", "ended"]) {
    const view = dashboard(); view.app.state.info = {};
    view.element("stream-url").value = "rtsp://camera/live";
    view.app.state.stopping = true;
    view.app.acceptSceneSnapshot(snapshot({state: "stopped", events: [event()],
      scene_drain: {state: "running", stage: "postprocessing"}}));
    const pending = view.app.status();
    view.requests.find(request => request.url === "/v1/status").resolve({ok:true,json: async () => ({
      pipeline_state: pipeline, stream_session_id: "session-a", scene_drain: {state: "running", stage: "postprocessing"},
      postprocessing_runs: [{stream_session_id: "session-a", state: "running", stage: "identity_review"}],
    })});
    await pending;
    assert.equal(view.app.state.stopping, false); assert.equal(view.app.state.running, false);
    assert.equal(view.element("start-button").disabled, false); assert.equal(view.element("stop-button").disabled, true);
    assert.match(view.element("connection-detail").textContent, /后台整理中，可以开始新的检测/);
    assert.match(view.element("scene-status").textContent, /可以开始新的检测/);
    assert.match(view.element("postprocessing-status").textContent, /1 次检测正在后台整理/);
    assert.doesNotMatch(text(view.element("scene-events")), /持续中|当前状态待核实/);
  }
});

test("previous run work and failure stay separate from a new live run", async () => {
  const view = dashboard(); view.app.state.info = {};
  view.app.acceptSceneSnapshot(snapshot({events: [event()]}));
  for (const backgroundState of ["running", "finalized", "failed"]) {
    const pending = view.app.status();
    view.requests.filter(request => request.url === "/v1/status").at(-1).resolve({ok:true,json: async () => ({
      pipeline_state: "running", stream_session_id: "session-a", input_mode: "stream", ready: true,
      scene_drain: null, postprocessing_runs: [{stream_session_id: "old-run", state: backgroundState, stage: "evidence"}],
    })});
    await pending;
    assert.equal(view.app.state.running, true); assert.equal(view.app.state.stopping, false);
    assert.equal(view.app.state.sessionId, "session-a");
    assert.equal(view.element("verdict").textContent, "事件持续中");
    assert.equal(view.element("connection-state").textContent, "Ready and delivering");
    assert.equal(view.element("stop-button").disabled, false);
    if (backgroundState === "failed") assert.match(view.element("postprocessing-status").textContent, /后台整理失败/);
  }
});

test("a completed run cannot hide queued or failed background runs", () => {
  const view = dashboard(); view.app.state.running = false;
  view.app.state.processingStatus = {pipeline_state: "idle", stream_session_id: "session-a",
    scene_drain: {state: "completed"}, postprocessing_runs: [
      {stream_session_id: "session-a", state: "finalized"}, {stream_session_id: "older", state: "queued"},
      {stream_session_id: "failed", state: "failed", error: "<script>untrusted</script>"},
    ]};
  view.app.acceptSceneSnapshot(snapshot({state: "stopped"}));
  view.app.updateRuntimeSummary(view.app.state.processingStatus);
  assert.match(view.element("scene-status").textContent, /等待后台整理.*后台整理失败/);
  assert.doesNotMatch(view.element("scene-status").textContent, /整理已完成/);
  assert.doesNotMatch(view.element("postprocessing-status").textContent, /untrusted|整理已完成/);
});

test("an authoritative empty drain does not revive a stale previous scene snapshot", () => {
  const view = dashboard(); view.app.state.running = false;
  view.app.acceptSceneSnapshot(snapshot({state: "stopped", scene_drain: {state: "running", stage: "postprocessing"}}));
  view.app.state.processingStatus = {pipeline_state: "idle", stream_session_id: "session-a", scene_drain: null};
  view.app.renderScene();
  assert.equal(view.element("verdict").textContent, "已停止");
});

test("camera acknowledgements are dispatched to the current display upload owner", () => {
  const view = dashboard(),received=[];
  view.app.localCameraDisplay.controller = {acknowledgeUpload: item => received.push(item)};
  const ack = {type: "camera_frame_ack", stream_session_id: "session-a", generation: 2,
    crop_version: 0, source_sequence: 42, accepted: true};
  view.app.message(ack);
  assert.deepEqual(received, [ack]); assert.equal(view.app.state.running, true);
  view.app.localCameraDisplay.controller = null; view.app.message(ack);
  assert.equal(received.length, 1);
});

test("a failed durable 人员追踪模块 queue is visible and cannot look like a healthy current observation", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event()]}));
  view.app.state.processingStatus = {pipeline_state: "running", stream_session_id: "session-a",
    rich_labeling: {state: "failed", recovery_required: true, rejected_capacity: 1}};
  view.app.updateRuntimeSummary(view.app.state.processingStatus);view.app.renderScene();
  assert.equal(view.element("rich-annotation-status").hidden, false);
  assert.match(view.element("rich-annotation-status").textContent, /人员追踪模块 分析失败/);
  assert.match(view.element("scene-status").textContent, /暂不可用/);
  view.app.updateRuntimeSummary({pipeline_state: "running", rich_labeling: {state: "running"}});
  assert.equal(view.element("rich-annotation-status").hidden, true);
});

test("stranger journal updates reload one representative row and aliases remove stale rows", async () => {
  const view = dashboard();
  const first = {event_id: "episode-1", event_label: "stranger", subject_type: "person",
    subject_id: "p-1", subject_ids: ["p-1"], created_at: "2026-09-16T00:00:00Z",
    status: "ended", finalized: true, start_seconds: 1, last_detected_seconds: 4};
  const second = {...first, event_id: "episode-2", start_seconds: 8, last_detected_seconds: 11};
  await (async () => {
    const pending = view.app.loadEventArchive();
    view.requests[0].resolve({ok: true, json: async () => ({events: [first], epoch: "run", cursor: 1})});
    await pending;
  })();
  assert.equal(view.element("archive-body").children.length, 1);
  view.streams[0].emit("event_change", {events: [first, second], aliases: []});
  assert.equal(view.requests.length, 2);
  assert.match(view.requests[1].url, /^\/v1\/events\?/);
  view.requests[1].resolve({ok: true, json: async () => ({events: [second], epoch: "run", cursor: 2})});
  await settle();
  assert.equal(view.element("archive-body").children.length, 1);
  assert.deepEqual([...view.app.archiveView.items.keys()], ["episode-2"]);
  assert.match(text(view.element("archive-body")), /8\.00s/);
  const merged = {...second, event_id: "merged-person", subject_id: "p-canonical", subject_ids: ["p-canonical"]};
  view.streams[1].emit("event_change", {events: [], aliases: [
    {event_id: "episode-2", canonical_event_id: "merged-person"},
  ]});
  assert.equal(view.requests.length, 3);
  view.requests[2].resolve({ok: true, json: async () => ({events: [merged], epoch: "run", cursor: 3})});
  await settle();
  assert.equal(view.element("archive-body").children.length, 1);
  assert.deepEqual([...view.app.archiveView.items.keys()], ["merged-person"]);
  assert.match(text(view.element("archive-body")), /p-canonical/);
  assert.equal(view.streams[0].closed, true);
  assert.equal(view.streams[1].closed, true);
});

test("source sessions clear old state and reject late snapshots", async () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({events: [event()], timestamp: 10}));
  assert.equal(view.app.acceptSceneSnapshot(snapshot({timestamp: 9})), false);
  view.app.state.sessionId = "session-b"; view.app.resetSceneView();
  assert.equal(view.app.sceneView.snapshot, null);
  assert.equal(view.element("verdict").textContent, "等待证据");
  assert.equal(view.app.acceptSceneSnapshot(snapshot({events: [event()], timestamp: 11})), false);
  assert.equal(view.app.acceptSceneSnapshot(snapshot({session: "session-b", timestamp: 0})), true);
  assert.equal(view.app.sceneView.aliases.size, 0);
  view.app.stopScenePolling(); await settle();
});

test("all model-supplied identity strings are text, not markup", () => {
  const view = dashboard(), name = '<img src=x onerror="alert(1)">';
  view.app.acceptSceneSnapshot(snapshot({entities: [person({identity: {status: "known", name}})],
    events: [event("confirmed", {event_label: "<script>bad()</script>"})]}));
  assert.match(text(view.element("scene-entities")), /<img src=x/);
  assert.equal(view.element("scene-entities").children[0].children[0].children.length, 0);
  assert.doesNotMatch(text(view.element("scene-events")), /<script>/);
  assert.match(text(view.element("scene-events")), /<img src=x/);
  assert.equal(view.element("scene-events").children[0].children[0].children.length, 0);
});

test("slow scene delivery has one pending request and no polling queue", async () => {
  const view = dashboard(); view.app.startScenePolling(); view.app.startScenePolling();
  assert.equal(view.requests.length, 1);
  assert.equal(view.requests[0].url, "/v1/scene");
  assert.equal(view.requests[0].options.cache, "no-store");
  assert.equal([...view.timers.values()].filter(timer => timer.delay === 750).length, 0);
  view.requests[0].resolve({ok: true, json: async () => snapshot()}); await settle();
  assert.equal(view.timers.size, 1);
  view.tick(); assert.equal(view.requests.length, 2);
  view.app.stopScenePolling(); await settle();
  assert.equal(view.requests[1].options.signal.aborted, true);
  assert.equal(view.timers.size, 0);
});

test("poll failure shows unavailable and page pause cancels requests", async () => {
  const view = dashboard(); view.app.startScenePolling();
  view.requests[0].reject(new Error("offline")); await settle();
  assert.equal(view.element("verdict").textContent, "状态暂不可用");
  assert.equal(view.element("decision").className, "decision warming");
  view.document.hidden = true; view.app.stopScenePolling();
  view.app.startScenePolling(); assert.equal(view.requests.length, 1);
  assert.equal(view.timers.size, 0);
  view.document.hidden = false; view.app.startScenePolling();
  assert.equal(view.requests.length, 2);
  view.app.sceneView.closed = true; view.app.stopScenePolling();
  view.app.startScenePolling(); await settle();
  assert.equal(view.requests.length, 2);
  assert.equal(view.timers.size, 0);
});

test("a timed-out request is aborted before the next poll", async () => {
  const view = dashboard(); view.app.startScenePolling();
  view.tick(4000); await settle();
  assert.equal(view.requests[0].options.signal.aborted, true);
  assert.equal(view.requests.length, 1);
  view.tick(); assert.equal(view.requests.length, 2);
  view.app.stopScenePolling(); await settle();
});

test("native video validation shows its memory boundary and actual processing lag", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({tracking_mode: "batch", memory_scope: "batch",
    tracking_lag_seconds: 5.25}));
  assert.equal(view.element("scene-mode").textContent, "视频片段验证");
  assert.match(view.element("scene-summary").textContent, /分析落后画面 5\.3 秒/);
  assert.match(view.element("scene-summary").textContent, /片段之间人物重新编号/);
  view.app.acceptSceneSnapshot(snapshot({timestamp: 2, tracking_mode: "batch",
    tracking_lag_seconds: null}));
  assert.match(view.element("scene-summary").textContent, /等待首段分析/);
  assert.doesNotMatch(view.element("scene-summary").textContent, /NaN|分析落后画面 0/);
});

test("persistent native memory is distinguished from batch resets and still shows lag", () => {
  const view = dashboard();
  view.app.acceptSceneSnapshot(snapshot({tracking_mode: "batch", memory_scope: "session",
    tracking_lag_seconds: 6.25}));
  assert.equal(view.element("scene-mode").textContent, "连续人物追踪（实验）");
  assert.match(view.element("scene-summary").textContent, /分析落后画面 6\.3 秒/);
  assert.match(view.element("scene-summary").textContent, /当前会话保持人物编号/);
  assert.doesNotMatch(view.element("scene-summary").textContent, /片段之间人物重新编号/);
});
