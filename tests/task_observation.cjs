/* Current task observations are available in both independent-VLM and ordinary
 * safety pipelines. The ordinary pipeline exposes a feature-specific latest
 * result even when a faster identity result is the global latest result.
 */
const assert = require("node:assert/strict");
const {test} = require("node:test");
const catalog = require("../src/goodvision_harness/assets/event_catalog.js");
const {taskEventLabels, readTaskObservation, eventObservationSummary} = require("../src/goodvision_harness/camera_display.js");

catalog.accept({event_catalog: {version: 1, events: [
  {id: "smoking", kind: "detection", labels: {zh: "吸烟", en: "Smoking"}},
  {id: "fight", kind: "detection", labels: {zh: "打架", en: "Fight"}},
  {id: "spill", kind: "detection", scope: "scene", labels: {zh: "积水", en: "Spill"}},
  {id: "stranger", kind: "identity", labels: {zh: "陌生人", en: "Stranger"}},
]}});

function status(features = ["smoking", "fight"], overrides = {}) {
  return {pipeline_state: "running", analysis_state: "running", stream_session_id: "session-a",
    features, vlm: {independent: false, state: "disabled"}, ...overrides};
}
function result(label, detected, overrides = {}) {
  return {type: "analysis_result", message_id: label + "-result", stream_session_id: "session-a",
    analysis_kind: "safety", analysis_complete: false, missing_event_labels: ["stranger"],
    emitted_at: "2026-10-18T04:00:00.000Z",
    judgments: [{event_label: label, detected, confidence: 0.8, description: "synthetic " + label}], ...overrides};
}
const judgment = (report, label) => report?.judgments.find(item => item.event_label === label);

test("task selection comes from active features and distinguishes unknown metadata from no detections", () => {
  assert.equal(taskEventLabels(null), null);
  assert.equal(taskEventLabels({vlm: {independent: false}}), null);
  assert.deepEqual(taskEventLabels(status([])), []);
  assert.deepEqual(taskEventLabels(status(["stranger"])), []);
  assert.deepEqual(taskEventLabels(status(["smoking", "stranger", "spill", "smoking"])), ["smoking", "spill"]);
});

test("ordinary events read each selected feature rather than the faster global latest result", async () => {
  const requests = [], smoking = result("smoking", true), fight = result("fight", false, {
    emitted_at: "2026-10-18T04:00:01.000Z",
  });
  const current = await readTaskObservation(status(["smoking", "fight", "stranger"]), async suffix => {
    requests.push(suffix);
    if (suffix === "/results/smoking/latest?allow_empty=true") return smoking;
    if (suffix === "/results/fight/latest?allow_empty=true") return fight;
    return result("stranger", true, {analysis_kind: "fastfaces"});
  });
  assert.deepEqual(requests.sort(), ["/results/fight/latest?allow_empty=true", "/results/smoking/latest?allow_empty=true"]);
  assert.equal(current.retry, "");
  assert.equal(current.report.type, "task_observation");
  assert.equal(current.report.state, "completed");
  assert.equal(current.report.stream_session_id, "session-a");
  assert.equal(judgment(current.report, "smoking").detected, true);
  assert.equal(judgment(current.report, "fight").detected, false);
  assert.equal(judgment(current.report, "stranger"), undefined);
  assert.strictEqual(current.report.observations.smoking, smoking);
  assert.strictEqual(current.report.observations.fight, fight);
  assert.equal(current.report.emitted_at, smoking.emitted_at, "Mixed observations show the oldest retained timestamp");
  assert.match(eventObservationSummary(current.report), /吸烟：发现/);
  assert.doesNotMatch(eventObservationSummary(current.report), /发现，待确认/);
});

test("partial aggregates contribute their requested judgment without borrowing sibling or identity results", async () => {
  const smoking = result("smoking", true, {judgments: [
    {event_label: "smoking", detected: true}, {event_label: "fight", detected: true},
    {event_label: "stranger", detected: true},
  ]});
  const current = await readTaskObservation(status(), async suffix =>
    suffix === "/results/smoking/latest?allow_empty=true" ? smoking : null);
  assert.equal(judgment(current.report, "smoking").detected, true,
    "analysis_complete=false refers to the whole frame and must not discard a valid event");
  assert.equal(judgment(current.report, "fight").detected, null, "An unreturned feature remains pending");
  assert.equal(judgment(current.report, "stranger"), undefined);
  assert.equal(current.retry, "", "An empty feature result is normal while its first analysis is pending");
});

test("a feature endpoint returning only identity data cannot create a false negative event judgment", async () => {
  const current = await readTaskObservation(status(["smoking"]), async () =>
    result("stranger", true, {analysis_kind: "fastfaces"}));
  assert.equal(current.report, null);
  assert.doesNotMatch(eventObservationSummary(current.report), /未发现/);
});

test("same-session successful feature observations survive an empty response and transport failure", async () => {
  const smoking = result("smoking", true), fight = result("fight", false);
  const previous = (await readTaskObservation(status(), async suffix => suffix.includes("smoking") ? smoking : fight)).report;
  const current = await readTaskObservation(status(), async suffix => {
    if (suffix.includes("smoking")) return null;
    throw new Error("temporary unavailable");
  }, previous);
  assert.equal(judgment(current.report, "smoking").detected, true);
  assert.equal(judgment(current.report, "fight").detected, false);
  assert.strictEqual(current.report.observations.smoking, smoking);
  assert.strictEqual(current.report.observations.fight, fight);
  assert.match(current.retry, /重试/);
});

test("initial empty, 404 and 204 feature responses wait without reporting a transport failure", async () => {
  for (const httpStatus of [null, 404, 204]) {
    const current = await readTaskObservation(status(["smoking"]), async () => {
      if (httpStatus) throw Object.assign(new Error("no result yet"), {httpStatus});
      return null;
    });
    assert.equal(current.report, null);
    assert.equal(current.retry, "");
    assert.match(eventObservationSummary(current.report), /等待/);
  }
});

test("late feature responses cannot move a retained same-session observation back to an older frame", async () => {
  const recent = result("smoking", true, {frame: {timestamp_seconds: 12, source_frame_index: 360}});
  const previous = (await readTaskObservation(status(["smoking"]), async () => recent)).report;
  for (const frame of [{timestamp_seconds: 11, source_frame_index: 330}, {timestamp_seconds: 12, source_frame_index: 359}]) {
    const current = await readTaskObservation(status(["smoking"]), async () => result("smoking", false, {frame}), previous);
    assert.equal(judgment(current.report, "smoking").detected, true);
    assert.strictEqual(current.report.observations.smoking, recent);
  }
});

test("new sessions discard cached and late results from the previous task", async () => {
  const oldResult = result("smoking", true);
  const previous = (await readTaskObservation(status(["smoking"]), async () => oldResult)).report;
  const current = await readTaskObservation(status(["smoking"], {stream_session_id: "session-b"}),
    async () => oldResult, previous);
  assert.equal(current.report, null, "No observations from the new session have arrived");
});

test("an independent VLM task preserves the original quick return object and endpoint", async () => {
  const report = {type: "vlm_report", state: "completed", stream_session_id: "session-a",
    judgments: [{event_label: "smoking", detected: true, observed_detected: true, confirmation_state: "pending"}]};
  const requests = [];
  const current = await readTaskObservation(status(["smoking"], {vlm: {independent: true}}), async suffix => {
    requests.push(suffix); return report;
  });
  assert.deepEqual(requests, ["/vlm/results/latest?allow_empty=true"]);
  assert.strictEqual(current.report, report);
  assert.equal(current.retry, "");
  assert.match(eventObservationSummary(current.report), /吸烟：发现/);
  assert.doesNotMatch(eventObservationSummary(current.report), /发现，待确认/);
});

test("pending and failed quick returns preserve only same-session completed reports", async () => {
  const currentStatus = status(["smoking"], {vlm: {independent: true}});
  const previous = {type: "vlm_report", state: "completed", stream_session_id: "session-a", judgments: []};
  const pending = await readTaskObservation(currentStatus,
    async () => ({state: "running", stream_session_id: "session-a"}), previous);
  assert.strictEqual(pending.report, previous);
  assert.equal(pending.retry, "");
  const failed = await readTaskObservation(currentStatus,
    async () => ({state: "failed", stream_session_id: "session-a"}), previous);
  assert.strictEqual(failed.report, previous);
  assert.match(failed.retry, /重试/);
  const next = await readTaskObservation({...currentStatus, stream_session_id: "session-b"}, async () => null, previous);
  assert.equal(next.report, null);
});

test("an identity-only task does not query an event endpoint", async () => {
  let called = false;
  const current = await readTaskObservation(status(["stranger"]), async () => { called = true; return null; });
  assert.equal(called, false);
  assert.equal(current.report, null);
  assert.equal(current.retry, "");
});

test("a new summary-only workflow cannot inherit features or independent VLM from the previous native session", async () => {
  for (const independent of [false, true]) {
    const currentStatus = status(["smoking"], {vlm: {independent}, workflow: {stream_session_id: "workflow-session-b"}});
    const previous = independent ? {type: "vlm_report", state: "completed", stream_session_id: "session-a",
      judgments: [{event_label: "smoking", detected: true}]} :
      (await readTaskObservation(status(["smoking"]), async () => result("smoking", true))).report;
    let called = false;
    assert.deepEqual(taskEventLabels(currentStatus), [], "Retained top-level features belong to the old native session");
    const current = await readTaskObservation(currentStatus, async () => { called = true; return previous; }, previous);
    assert.equal(called, false);
    assert.equal(current.report, null);
    assert.equal(current.retry, "");
  }
});

test("workflow test runs neither read live observations nor retain a previous judgment", async () => {
  for (const independent of [false, true]) {
    const previous = {type: "vlm_report", state: "completed", stream_session_id: "session-a",
      judgments: [{event_label: "smoking", detected: true}]};
    let called = false;
    const current = await readTaskObservation(status(["smoking"], {vlm: {independent}, workflow_test_running: true}),
      async () => { called = true; return previous; }, previous);
    assert.equal(called, false);
    assert.equal(current.report, null);
    assert.equal(current.retry, "");
  }
});

test("a workflow using the current native session can still display its event judgment", async () => {
  const currentStatus = status(["smoking"], {workflow: {stream_session_id: "session-a"}});
  assert.deepEqual(taskEventLabels(currentStatus), ["smoking"]);
  const current = await readTaskObservation(currentStatus, async () => result("smoking", true));
  assert.equal(judgment(current.report, "smoking").detected, true);
});
