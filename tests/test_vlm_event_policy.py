"""Prompt and wire contracts must keep the same event meaning across modes."""

import json
from dataclasses import replace

import pytest
from test_vlm_independent import config, job, report

from goodvision_harness.event_sensitivity import EVENT_SENSITIVITY_DEFAULTS
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_event_policy import (
    OUTPUT_FORMAT_MARKER,
    annotation_policy,
    event_policy_snapshot,
)
from goodvision_harness.vlm_independent import IndependentEventState
from goodvision_harness.vlm_independent_model import IndependentVLMInspector

LABELS = tuple(EVENT_SENSITIVITY_DEFAULTS)


def response(payload):
    return {"choices": [{
        "finish_reason": "stop", "message": {"content": json.dumps(payload)},
    }]}


@pytest.mark.parametrize("choices", [
    {}, {label: "standard" for label in LABELS},
    {label: "high" for label in LABELS},
])
@pytest.mark.parametrize("followup", [False, True])
def test_modes_and_reference_receive_identical_criteria_and_images(choices, followup):
    cfg = replace(config(), labels=LABELS)
    request = replace(
        job(cfg, 1.2, 18), event_sensitivities=tuple(choices.items()),
        grounding_required=followup,
    )
    reference = annotation_policy(LABELS, choices)
    bodies = []
    for cls in (IndependentVLMInspector, CompactGroundingInspector, EventJSONInspector):
        inspector = cls(
            cfg.model_settings, model="qwen3-vl-test", api_key="test-only",
            event_policy="research-v1",
        )
        try:
            messages = inspector.build_messages(request)
        finally:
            inspector.close()
        common, output = messages[0]["content"].split(OUTPUT_FORMAT_MARKER)
        assert common == reference["decision_rules"]
        # HIGH retains suspicion criteria without demanding fields the event
        # schema forbids. Format changes must not append new sensitivity rules.
        assert "Lower confidence" not in common
        assert "sensitivity HIGH" not in output
        assert "sensitivity STANDARD" not in output
        assert "only all-negative assessments yield a negative scene" in common
        assert "even with one visible participant" in common
        assert "independently verified" in common
        bodies.append(messages[1])
    assert bodies[0] == bodies[1] == bodies[2]
    assert len([p for p in bodies[0]["content"] if p["type"] == "image_url"]) == 2


def test_policy_fingerprint_changes_with_criteria_not_label_order_or_format():
    default = event_policy_snapshot(LABELS)
    explicit = event_policy_snapshot(
        tuple(reversed(LABELS)), EVENT_SENSITIVITY_DEFAULTS,
    )
    assert default == explicit
    standard = event_policy_snapshot(LABELS, {"smoking": "standard"})
    assert standard["sha256"] != default["sha256"]
    assert standard["sensitivities"]["smoking"] == "standard"
    assert event_policy_snapshot(("fall",))["sha256"] != default["sha256"]
    assert default["comparison_field"] == "observed_detected"
    for labels, choices in [
        (("fall", "fall"), {}), (("unknown",), {}), ((), {}),
        (LABELS, {"smoking": "low"}),
    ]:
        with pytest.raises(ValueError):
            event_policy_snapshot(labels, choices)


@pytest.mark.parametrize("votes,expected", [
    ([True, False], True), ([True, None], True),
    ([False, None], None), ([None, None], None), ([False, False], False),
])
def test_grounded_and_events_raw_judgments_agree_despite_missing_boxes(votes, expected):
    cfg = replace(config(), labels=("ppe_violation",))
    request = job(cfg, 1.2, 18)
    grounded = IndependentVLMInspector._parse(request, response({
        "subjects": [{"subject_id": "p1", "bbox": None},
                     {"subject_id": "p2", "bbox": None}],
        "verdicts": [
            {"event_label": "ppe_violation", "subject_ids": [f"p{i+1}"],
             "detected": value, "confidence": 0.01}
            for i, value in enumerate(votes)
        ],
    }))
    events = EventJSONInspector._parse(request, response({"ppe_violation": expected}))
    ground_report = report(IndependentEventState(cfg), 1.2, 18, grounded)
    event_report = report(
        IndependentEventState(replace(cfg, output_mode="events")), 1.2, 18, events,
    )
    assert ground_report["analysis_complete"] and event_report["analysis_complete"]
    assert ground_report["judgments"][0]["observed_detected"] is expected
    assert event_report["judgments"][0]["observed_detected"] is expected
    assert not ground_report["localization_complete"]
    assert ground_report["event_policy"] == event_report["event_policy"]


def test_missing_event_is_incomplete_not_negative():
    cfg = config()
    request = job(cfg, 1.2, 18)
    verdicts = IndependentVLMInspector._parse(request, response({
        "subjects": [], "verdicts": [{"event_label": "fall", "detected": False}],
    }))
    result = report(IndependentEventState(cfg), 1.2, 18, verdicts)
    assert not result["analysis_complete"]
    assert [v["event_label"] for v in result["judgments"]] == ["fall"]


def test_report_fingerprint_uses_the_actual_request_sensitivities():
    from goodvision_harness.vlm_independent import SourceSample, utc_now

    cfg = replace(config(), event_policy="research-v1")
    request = replace(job(cfg, 1.2, 18), event_sensitivities=(("smoking", "standard"),))
    sample = SourceSample(request.frames[-1], utc_now(), 0.0)
    result = IndependentEventState(cfg).report(
        request, sample, (), {"model_seconds": 0}, error="timeout",
    )
    assert result["event_policy"] == event_policy_snapshot(
        cfg.labels, {"smoking": "standard"},
    )
    assert result["judgments"] == [] and not result["analysis_complete"]
