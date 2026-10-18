const assert = require("node:assert/strict");
const {test} = require("node:test");
const catalog = require("../src/goodvision_harness/assets/event_catalog.js");
const {taskObservationItems} = require("../src/goodvision_harness/camera_display.js");

catalog.accept({event_catalog: {version: 1, events: [
  {id: "smoking", kind: "detection", labels: {zh: "吸烟", en: "Smoking"}},
  {id: "fight", kind: "detection", labels: {zh: "打架", en: "Fight"}},
  {id: "spill", kind: "detection", scope: "scene", labels: {zh: "积水", en: "Spill"}},
  {id: "stranger", kind: "identity", labels: {zh: "陌生人", en: "Stranger"}},
  {id: "visitor_identity", kind: "identity", labels: {zh: "访客身份", en: "Visitor identity"}},
]}});

function status(features = ["smoking"], overrides = {}) {
  return {stream_session_id: "current-task", features, ...overrides};
}
function report(judgments, overrides = {}) {
  return {type: "vlm_report", state: "completed", stream_session_id: "current-task", judgments, ...overrides};
}
function item(detected, overrides = {}) {
  return {event_label: "smoking", detected, ...overrides};
}

test("a positive event gets its own detected card immediately", () => {
  assert.deepEqual(taskObservationItems(status(), report([item(true)])), [
    {label: "smoking", name: "吸烟", state: "detected", text: "发现"},
  ]);
});

test("an explicit negative event is clear rather than pending", () => {
  assert.deepEqual(taskObservationItems(status(), report([item(false)])), [
    {label: "smoking", name: "吸烟", state: "clear", text: "未发现"},
  ]);
});

test("an explicit null event is uncertain rather than clear or missing", () => {
  assert.deepEqual(taskObservationItems(status(), report([item(null)])), [
    {label: "smoking", name: "吸烟", state: "uncertain", text: "不确定"},
  ]);
});

test("a selected event without a returned judgment remains pending", () => {
  for (const value of [null, report([])]) {
    assert.deepEqual(taskObservationItems(status(), value), [
      {label: "smoking", name: "吸烟", state: "pending", text: "等待判断"},
    ]);
  }
});

test("the fast observation controls color independently of temporal confirmation", () => {
  for (const [observed, confirmed, state] of [[true, false, "detected"], [false, true, "clear"], [null, true, "uncertain"]]) {
    const cards = taskObservationItems(status(), report([item(confirmed, {
      observed_detected: observed, confirmation_state: confirmed ? "confirmed" : "pending",
    })]));
    assert.equal(cards[0].state, state);
  }
});

test("native feature placeholders remain pending while a returned null is uncertain", () => {
  const native = report([item(null), {event_label: "fight", detected: null}], {
    type: "task_observation",
    observations: {smoking: {type: "analysis_result", judgments: [item(null)]}},
  });
  assert.deepEqual(taskObservationItems(status(["smoking", "fight"]), native).map(card => [card.label, card.state]), [
    ["smoking", "uncertain"], ["fight", "pending"],
  ]);
});

test("each returned native feature keeps an independent state", () => {
  const judgments = [item(true), {event_label: "fight", detected: false}];
  const native = report(judgments, {type: "task_observation", observations: {
    smoking: {type: "analysis_result", judgments: [judgments[0]]},
    fight: {type: "analysis_result", judgments: [judgments[1]]},
  }});
  assert.deepEqual(taskObservationItems(status(["smoking", "fight", "spill"]), native).map(card => card.state),
    ["detected", "clear", "pending"]);
});

test("results from another session cannot color current task cards", () => {
  assert.deepEqual(taskObservationItems(status(), report([item(true)], {stream_session_id: "old-task"})), [
    {label: "smoking", name: "吸烟", state: "pending", text: "等待判断"},
  ]);
});

test("one-shot workflow tests cannot reuse live positive judgments", () => {
  const cards = taskObservationItems(status(["smoking"], {workflow_test_running: true}), report([item(true)]));
  assert.equal(cards[0].state, "pending");
});

test("a new workflow cannot inherit cards selected by an old native session", () => {
  const current = status(["smoking"], {workflow: {stream_session_id: "new-workflow"}});
  assert.deepEqual(taskObservationItems(current, report([item(true)])), []);
});

test("unfinished or failed reports cannot color cards from their embedded judgments", () => {
  for (const state of ["running", "failed", undefined]) {
    assert.equal(taskObservationItems(status(), report([item(true)], {state}))[0].state, "pending");
  }
});

test("cards follow selected event order and exclude identity, overall, and unselected results", () => {
  const current = status(["spill", "stranger", "visitor_identity", "overall", "unknown_person", "smoking", "spill"]);
  const data = report([item(false), {event_label: "fight", detected: true}, {event_label: "stranger", detected: true}]);
  assert.deepEqual(taskObservationItems(current, data).map(card => [card.label, card.state]), [
    ["spill", "pending"], ["smoking", "clear"],
  ]);
  assert.deepEqual(taskObservationItems(status([]), data), []);
});

test("stopping clears every event color while retaining the selected cards", () => {
  const data = report([item(true), {event_label: "fight", detected: false}]);
  assert.deepEqual(taskObservationItems(status(["smoking", "fight", "spill"]), data, {running: false}), [
    {label: "smoking", name: "吸烟", state: "stopped", text: "已停止"},
    {label: "fight", name: "打架", state: "stopped", text: "已停止"},
    {label: "spill", name: "积水", state: "stopped", text: "已停止"},
  ]);
});

test("custom task event labels do not need a built-in event inventory", () => {
  const data = report([{event_label: "door_left_open", detected: true}, item(true)]);
  const cards = taskObservationItems(status(["door_left_open", "delivery_arrived"]), data);
  assert.deepEqual(cards, [
    {label: "door_left_open", name: "door_left_open", state: "detected", text: "发现"},
    {label: "delivery_arrived", name: "delivery_arrived", state: "pending", text: "等待判断"},
  ]);
});

test("fallback cards exclude identity, summary, and unsafe labels when selection metadata is unavailable", () => {
  const data = report([item(true), {event_label: "stranger", detected: true},
    {event_label: "visitor_identity", detected: true}, {event_label: "overall", detected: true},
    {event_label: "../smoking", detected: true}, {event_label: "<script>", detected: true},
    {event_label: "", detected: true}]);
  assert.deepEqual(taskObservationItems({stream_session_id: "current-task"}, data).map(card => card.label), ["smoking"]);
});
