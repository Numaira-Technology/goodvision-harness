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
  const elements = new Map(), timers = new Map(), requests = [];
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
    location: {hostname: "localhost"}, Date: Clock, AbortController,
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
      renderClassification};
  `, context);
  context.app.sceneView.enabled = true;
  context.app.sceneView.mode = "active";
  return {app: context.app, document, requests, timers, context,
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
