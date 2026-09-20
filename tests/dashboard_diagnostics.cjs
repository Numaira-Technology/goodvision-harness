const assert = require("node:assert/strict");
const test = require("node:test");
const {resultText} = require("../src/goodvision_harness/assets/scene_diagnostics.js");

test("missing weapon verdict is not displayed as negative", () => {
  const text = resultText({response_outcome:"validated",result:"partial",
    detected_labels:[],missing_event_labels:["weapon"],unknown_event_labels:[]});
  assert.match(text, /部分完成/); assert.match(text, /weapon/); assert.doesNotMatch(text, /阴性/);
});
test("valid positive survives missing other label", () => {
  assert.match(resultText({response_outcome:"validated",result:"partial",
    detected_labels:["fight"],missing_event_labels:["weapon"]}), /阳性：fight/);
});
test("null and request failures cannot become negatives", () => {
  assert.match(resultText({response_outcome:"validated",unknown_event_labels:["fall"]}), /无法判断/);
  assert.match(resultText({response_outcome:"failed",failure_category:"timeout"}), /请求失败/);
});
