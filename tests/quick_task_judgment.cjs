/* Exercise the shipped polling code with fast-event and frame-result responses.
 * The task judgment must use the quick VLM return, while the result tab keeps
 * the independent per-frame history.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");
const display = require("../src/goodvision_harness/camera_display.js");
const catalog = require("../src/goodvision_harness/assets/event_catalog.js");
const {createDocument} = require("./helpers/dashboard.cjs");
catalog.accept({event_catalog: {version: 1, events: [
  {id: "smoking", kind: "detection", labels: {zh: "吸烟", en: "Smoking"}},
  {id: "fight", kind: "detection", labels: {zh: "打架", en: "Fight"}},
  {id: "stranger", kind: "identity", labels: {zh: "陌生人", en: "Stranger"}},
]}});

const filename = path.join(__dirname, "../src/goodvision_harness/dashboard.html");
const html = fs.readFileSync(filename, "utf8");
const start = html.indexOf("const diagnosticResults=new Map();");
const end = html.indexOf("\nasync function connectResultEvents()", start);
assert.ok(start >= 0 && end > start, "The shipped diagnostic polling handlers must be present");
const handlers = new vm.Script(html.slice(start, end) + "\nglobalThis.poll=pollDiagnosticResults;", {filename});

function frame(feature, message = feature + "-frame") {
  return {type: "analysis_result", message_id: message, stream_session_id: "session-a", feature,
    judgments: [{event_label: feature, detected: false, description: "per-frame result"}]};
}
function quick(state = "completed", session = "session-a", description = "quick task return") {
  return {state, stream_session_id: session,
    judgments: [{event_label: "smoking", detected: true, description}]};
}
function fixture({view = "insights", parentRow = null, response, status = {}} = {}) {
  const requests = [], rendered = [], timers = new Map();
  const {document, element} = createDocument();
  let timerId = 0;
  const localWindow = {};
  localWindow.parent = parentRow ? {GoodvisionFleet: {app: {rows: new Map([["camera-a", parentRow]])}}} : localWindow;
  const state = {running: true, sessionId: "session-a",
    diagnosticStatus: {stream_session_id: "session-a", pipeline_state: "running", analysis_state: "running",
      features: ["smoking", "fight"], vlm: {independent: true}, ...status}};
  const context = vm.createContext({
    window: localWindow, document, AbortController,
    GoodvisionDiagnostic: true, GoodvisionDiagnosticView: view, GoodvisionDiagnosticCameraId: "camera-a",
    GoodvisionConsole: {currentPage: () => "overview"}, GoodvisionFleet: {detailHidden: () => false},
    GoodvisionCameraDisplay: {...display,
      eventObservationSummary: report => report?.judgments?.[0]?.description || "等待事件判断"},
    channelSurfaceActive: () => true, sceneView: {closed: false}, state,
    detectionEvents: ["smoking", "fight"], $: element, ui: {sseChannel: {}},
    analysis: result => rendered.push(result), setChannel() {},
    async pollOverallDescription() {requests.push({url: "/v1/workflow/summaries?limit=100"});return {received:true,unavailable:false};},
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, {callback, delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
    async fetch(url, options) {
      requests.push({url, options});
      const payload = response ? await response(url, state) : url.includes("/vlm/") ? quick() : frame("overall");
      if (payload?.httpStatus) return {ok: payload.httpStatus >= 200 && payload.httpStatus < 300,
        status: payload.httpStatus, json: async () => payload.body};
      return {ok: true, status: payload == null ? 204 : 200, json: async () => payload};
    },
  });
  handlers.runInContext(context);
  return {context, state, element, requests, rendered, timers, poll: () => context.poll()};
}

test("video-side task judgment reuses the lane's quick return and polls only the full-frame description", async () => {
  const text = "任务实时判断 · 快速 events 返回：检测到吸烟（0.2 秒前）";
  const view = fixture({parentRow: {vlmEvents: {textContent: text}, vlmReport: quick()}});
  await view.poll();
  assert.equal(view.element("vlm-current-observation").textContent, text);
  assert.deepEqual(view.requests.map(item => item.url), ["/v1/workflow/summaries?limit=100"]);
  assert.equal(view.rendered.length, 0, "Summary has its own renderer, separate from frame results");
  const cards = view.element("task-judgment-cards").children;
  assert.deepEqual(cards.map(card => [card.dataset.eventLabel, card.dataset.state]), [["smoking", "detected"], ["fight", "pending"]]);
  assert.equal(view.element("vlm-current-observation").hidden, true, "Rendered boxes replace the old joined text");
  assert.ok([...view.timers.values()].some(timer => timer.delay === 2000), "Side view continues finite polling");
});

test("the realtime-results tab still reads every selected frame-result stream and deduplicates repeated results", async () => {
  const view = fixture({view: "results", parentRow: {vlmEvents: {textContent: "quick event"}},
    response: url => frame(url.match(/\/results\/([^/]+)\//)[1])});
  await view.poll();
  assert.deepEqual(view.requests.map(item => item.url), [
    "/v1/results/overall/latest?allow_empty=true",
    "/v1/results/smoking/latest?allow_empty=true",
    "/v1/results/fight/latest?allow_empty=true",
  ]);
  assert.deepEqual(view.rendered.map(result => result.feature), ["overall", "smoking", "fight"]);
  await view.poll();
  assert.equal(view.rendered.length, 3, "Polling the same message must not duplicate frame history");
});

test("standalone task judgment accepts a completed quick return and retains it during a retry", async () => {
  let report = quick();
  const view = fixture({response: url => url.includes("/vlm/") ? report : frame("overall")});
  await view.poll();
  assert.equal(view.element("vlm-current-observation").textContent, "quick task return");
  for (const pendingState of ["running", "failed"]) {
    report = quick(pendingState, "session-a", "unfinished result must not replace success");
    await view.poll();
    assert.match(view.element("vlm-current-observation").textContent, /quick task return/);
    assert.doesNotMatch(view.element("vlm-current-observation").textContent, /unfinished result/);
  }
  assert.ok(view.requests.some(item => item.url === "/v1/vlm/results/latest?allow_empty=true"));
});

test("a new task cannot inherit the previous session's successful quick judgment", async () => {
  let report = quick();
  const view = fixture({response: url => url.includes("/vlm/") ? report : frame("overall")});
  await view.poll();
  view.state.sessionId = "session-b";
  view.state.diagnosticStatus.stream_session_id = "session-b";
  report = quick("running", "session-b", "no new answer yet");
  await view.poll();
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /quick task return|no new answer yet/);
  assert.match(view.element("vlm-current-observation").textContent, /等待/);
});

test("stopping analysis clears the prior quick judgment even if the VLM endpoint retains its last return", async () => {
  const view = fixture();
  await view.poll();
  assert.equal(view.element("vlm-current-observation").textContent, "quick task return");
  view.requests.length = 0;
  view.state.running = false;
  view.state.diagnosticStatus.pipeline_state = "idle";
  view.state.diagnosticStatus.analysis_state = "idle";
  await view.poll();
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /quick task return/);
  assert.match(view.element("vlm-current-observation").textContent, /任务已停止/);
  assert.ok(view.requests.every(item => !item.url.includes("/vlm/")), "Stopped analysis must not revive a retained quick return");
});

test("standalone task judgment ignores a quick return from an older session", async () => {
  const view = fixture({response: url => url.includes("/vlm/") ? quick("completed", "session-old", "stale result") : frame("overall")});
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /等待/);
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /stale result/);
});

test("a task/session change while a quick request is in flight cannot paint the previous task's answer", async () => {
  const view = fixture({response: (url, state) => {
    if (!url.includes("/vlm/")) return frame("overall");
    state.sessionId = "session-b";
    state.diagnosticStatus.stream_session_id = "session-b";
    return quick("completed", "session-a", "old task answer");
  }});
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /等待/);
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /old task answer/);
});

test("a task with no completed quick return waits through initial 404, empty and pending responses", async () => {
  let report = {httpStatus: 404};
  const view = fixture({response: url => url.includes("/vlm/") ? report : frame("overall")});
  for (const pending of [{httpStatus: 404}, null, quick("running", "session-a", "unfinished")]) {
    report = pending;
    await view.poll();
    assert.match(view.element("vlm-current-observation").textContent, /等待/);
    assert.doesNotMatch(view.element("vlm-current-observation").textContent, /unfinished|延迟|重试/);
  }
  report = quick();
  await view.poll();
  assert.equal(view.element("vlm-current-observation").textContent, "quick task return");
});

test("a hidden camera surface does not issue quick or frame-result requests", async () => {
  const view = fixture();
  view.context.channelSurfaceActive = () => false;
  await view.poll();
  assert.equal(view.requests.length, 0);
  assert.ok([...view.timers.values()].some(timer => timer.delay === 2000));
});

test("ordinary event tasks paint their feature result when independent VLM is disabled", async () => {
  const view = fixture({status: {features: ["smoking", "stranger"], vlm: {independent: false, state: "disabled"}},
    response: url => url.includes("/results/smoking/") ? {
      ...frame("smoking"), analysis_complete: false,
      judgments: [{event_label: "smoking", detected: true, description: "ordinary events task return"}],
    } : frame("overall")});
  await view.poll();
  assert.equal(view.element("vlm-current-observation").textContent, "ordinary events task return");
  assert.deepEqual(view.requests.map(item => item.url), [
    "/v1/results/smoking/latest?allow_empty=true", "/v1/workflow/summaries?limit=100",
  ]);
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /未启用/);
});

test("ordinary event tasks retain their last good judgment through errors, then clear it on stop", async () => {
  let unavailable = false;
  const view = fixture({status: {features: ["smoking"], vlm: {independent: false}}, response: url => {
    if (!url.includes("/results/smoking/")) return frame("overall");
    if (unavailable) return {httpStatus: 503};
    return {...frame("smoking"), judgments: [{event_label: "smoking", detected: false, description: "ordinary good return"}]};
  }});
  await view.poll();
  unavailable = true;
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /ordinary good return/);
  assert.match(view.element("vlm-current-observation").textContent, /重试/);
  view.state.running = false;
  view.state.diagnosticStatus.analysis_state = "idle";
  view.requests.length = 0;
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /任务已停止/);
  assert.ok(view.requests.every(item => !item.url.includes("/results/smoking/")));
});

test("an ordinary result arriving after the task changes cannot paint its old answer", async () => {
  const view = fixture({status: {features: ["smoking"], vlm: {independent: false}}, response: (url, state) => {
    if (!url.includes("/results/smoking/")) return frame("overall");
    state.sessionId = "session-b";
    state.diagnosticStatus.stream_session_id = "session-b";
    return {...frame("smoking"), judgments: [{event_label: "smoking", detected: true, description: "old ordinary answer"}]};
  }});
  await view.poll();
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /old ordinary answer/);
  assert.match(view.element("vlm-current-observation").textContent, /等待/);
});

test("only a task with no selected detection events is described as not enabled", async () => {
  const view = fixture({status: {features: ["stranger"], vlm: {independent: false}}});
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /未启用/);
  assert.deepEqual(view.requests.map(item => item.url), ["/v1/workflow/summaries?limit=100"]);
});

test("a pending task configuration does not falsely claim its event detection is disabled", async () => {
  const view = fixture({status: {features: undefined, vlm: {independent: false}}});
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /读取任务配置/);
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /未启用/);
});

test("stopping during an ordinary event request prevents its late result from reviving the box", async () => {
  const view = fixture({status: {features: ["smoking"], vlm: {independent: false}}, response: (url, state) => {
    if (!url.includes("/results/smoking/")) return frame("overall");
    state.running = false;
    state.diagnosticStatus.analysis_state = "idle";
    return {...frame("smoking"), judgments: [{event_label: "smoking", detected: true, description: "late stopped answer"}]};
  }});
  await view.poll();
  assert.doesNotMatch(view.element("vlm-current-observation").textContent, /late stopped answer/);
  await view.poll();
  assert.match(view.element("vlm-current-observation").textContent, /任务已停止/);
});

test("switching to a summary-only workflow clears a retained native judgment, including stale independent VLM state", async () => {
  for (const independent of [false, true]) {
    let pending = false;
    const view = fixture({status: {features: ["smoking"], vlm: {independent}}, response: url => {
      if (url.includes("/results/overall/")) return frame("overall");
      if (pending) return null;
      return independent ? quick("completed", "session-a", "old native observation") :
        {...frame("smoking"), judgments: [{event_label: "smoking", detected: true, description: "old native observation"}]};
    }});
    await view.poll();
    assert.equal(view.element("vlm-current-observation").textContent, "old native observation");
    view.state.diagnosticStatus.workflow = {stream_session_id: "summary-session-b"};
    view.requests.length = 0;
    await view.poll();
    assert.doesNotMatch(view.element("vlm-current-observation").textContent, /old native observation/);
    assert.ok(view.requests.every(item => item.url.includes("/workflow/summaries")));
    delete view.state.diagnosticStatus.workflow;
    pending = true;
    await view.poll();
    assert.doesNotMatch(view.element("vlm-current-observation").textContent, /old native observation/,
      "Returning to a pending task must not restore the cache cleared by the workflow switch");
  }
});

test("workflow tests clear the native judgment and direct the user to workflow output", async () => {
  for (const independent of [false, true]) {
    let pending = false;
    const view = fixture({status: {features: ["smoking"], vlm: {independent}}, response: url => {
      if (url.includes("/results/overall/")) return frame("overall");
      if (pending) return null;
      return independent ? quick("completed", "session-a", "pre-test native observation") :
        {...frame("smoking"), judgments: [{event_label: "smoking", detected: true, description: "pre-test native observation"}]};
    }});
    await view.poll();
    view.state.diagnosticStatus.workflow_test_running = true;
    view.requests.length = 0;
    await view.poll();
    assert.equal(view.element("vlm-current-observation").textContent, "工作流测试中，请查看工作流输出");
    assert.ok(view.requests.every(item => item.url.includes("/workflow/summaries")));
    view.state.diagnosticStatus.workflow_test_running = false;
    pending = true;
    await view.poll();
    assert.doesNotMatch(view.element("vlm-current-observation").textContent, /pre-test native observation/);
  }
});

test("a native request arriving after a workflow session switch or workflow test starts cannot paint its result", async () => {
  for (const independent of [false, true]) {
    for (const transition of ["workflow", "test"]) {
      const view = fixture({status: {features: ["smoking"], vlm: {independent}}, response: (url, state) => {
        if (url.includes("/results/overall/")) return frame("overall");
        if (transition === "workflow") state.diagnosticStatus.workflow = {stream_session_id: "workflow-session-b"};
        else state.diagnosticStatus.workflow_test_running = true;
        return independent ? quick("completed", "session-a", "late native answer") :
          {...frame("smoking"), judgments: [{event_label: "smoking", detected: true, description: "late native answer"}]};
      }});
      await view.poll();
      assert.doesNotMatch(view.element("vlm-current-observation").textContent, /late native answer/);
    }
  }
});

test("event boxes change independently and stop clears alert colors", async () => {
  let report = {...quick(), judgments: [{event_label:"smoking", detected:true}, {event_label:"fight", detected:false}]};
  const view = fixture({response: () => report});
  await view.poll();
  const cards = view.element("task-judgment-cards").children;
  assert.deepEqual(cards.map(card => card.dataset.state), ["detected", "clear"]);
  assert.match(cards[0].className, /detected/);
  assert.match(cards[1].className, /clear/);
  report = {...quick(), judgments: [{event_label:"smoking", detected:false}, {event_label:"fight", detected:null}]};
  await view.poll();
  assert.deepEqual(view.element("task-judgment-cards").children.map(card => card.dataset.state), ["clear", "uncertain"]);
  assert.strictEqual(view.element("task-judgment-cards").children[0], cards[0], "Keep card positions stable between returns");
  assert.equal(cards[1].children[1].textContent, "不确定");
  view.state.running = false;
  await view.poll();
  assert.deepEqual(cards.map(card => card.dataset.state), ["stopped", "stopped"]);
  assert.equal(view.element("vlm-current-observation").hidden, false);
});
