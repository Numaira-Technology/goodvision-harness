"""Short wire output must preserve decisions, associations and failure semantics."""

import copy
import json
from dataclasses import replace

import jsonschema
import pytest
from test_vlm_independent import BOX, config, job, report

from goodvision_harness.scene_model import _ResponseValidationError
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_event_policy import OUTPUT_FORMAT_MARKER, annotation_policy
from goodvision_harness.vlm_independent import IndependentEventState, _inspector
from goodvision_harness.vlm_independent_model import IndependentVLMInspector
from goodvision_harness.vlm_lean_grounding import (
    LeanGroundingInspector,
    response_format,
)
from tools.vlm_benchmark.modes import mode_inspector

LABELS = ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter")


def payload():
    return {
        "v": 1,
        "s": [{"i": "p1", "b": [100, 100, 500, 900]}],
        "e": {
            label: [
                {"p": [] if label == "ground_litter" else ["p1"], "d": value, "c": 0.8}
            ]
            for label, value in zip(
                LABELS, (False, True, None, False, True, False), strict=True
            )
        },
    }


def envelope(raw, finish="stop"):
    return {
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(raw)}}]
    }


def cfg():
    return replace(
        config(), labels=LABELS, output_mode="compact", grounding_format="lean-v1",
        event_policy="research-v1"
    )


def parse(raw, finish="stop"):
    return LeanGroundingInspector._parse(job(cfg(), 1.2, 18), envelope(raw, finish))


def test_all_six_decisions_confidence_boxes_and_public_report_preserved():
    raw = payload()
    legacy = {
        "bbox_coordinate_system": "normalized_1000",
        "subjects": [{"subject_id": "p1", "bbox": [100, 100, 500, 900]}],
        "verdicts": [
            {
                "event_label": label,
                "subject_ids": vote["p"],
                "detected": vote["d"],
                "confidence": vote["c"],
            }
            for label, votes in raw["e"].items()
            for vote in votes
        ],
    }
    votes = parse(raw)
    assert votes == IndependentVLMInspector._parse(
        job(cfg(), 1.2, 18), envelope(legacy)
    )
    assert votes[1].detected is True and votes[1].confidence == 0.8
    assert votes[1].subject_boxes == (("p1", BOX),)
    assert votes[2].detected is None and votes[-1].subject_ids == ()
    result = report(IndependentEventState(cfg()), 1.2, 18, votes)
    assert result["analysis_complete"] and result["localization_complete"]
    assert {x["event_label"]: x["observed_detected"] for x in result["judgments"]} == {
        k: v[0]["d"] for k, v in raw["e"].items()
    }
    assert len(result["objects"]) == 1
    # Size comparison on a fixed example, not a claim about model tokens/latency.
    assert len(json.dumps(raw)) < len(json.dumps(legacy)) * 0.65
    jsonschema.Draft202012Validator(
        response_format(LABELS)["json_schema"]["schema"]
    ).validate(raw)


@pytest.mark.parametrize(
    "values,expected",
    [
        ((True, False), True),
        ((False, None), None),
        ((False, False), False),
    ],
)
def test_multi_person_associations_and_scene_aggregation(values, expected):
    raw = payload()
    raw["s"].append({"i": "p2", "b": [600, 100, 950, 900]})
    raw["e"]["ppe_violation"] = [
        {"p": [f"p{i + 1}"], "d": value, "c": 0.01} for i, value in enumerate(values)
    ]
    votes = parse(raw)
    assert [v.subject_ids for v in votes if v.event_label == "ppe_violation"] == [
        ("p1",),
        ("p2",),
    ]
    result = report(IndependentEventState(cfg()), 1.2, 18, votes)
    judgment = next(
        j for j in result["judgments"] if j["event_label"] == "ppe_violation"
    )
    assert judgment["observed_detected"] is expected


@pytest.mark.parametrize(
    "box",
    [
        None,
        [100, 100, 100, 900],
        [900, 100, 500, 900],
        [-1, 0, 500, 900],
        [100, 100, 1001, 900],
        [0.1, 0.1, 0.5, 0.9],
        [True, 1, 500, 900],
        [100, 100, 500],
        "hidden",
        [float("inf"), 0, 500, 900],
    ],
)
def test_bad_or_unknown_box_never_creates_valid_localization_or_negative_event(box):
    raw = payload()
    raw["s"][0]["b"] = box
    if box is not None and isinstance(box, list) and box[0] == float("inf"):
        with pytest.raises(_ResponseValidationError):
            parse(raw)
        return
    votes = parse(raw)
    assert votes[1].detected is True and votes[2].detected is None
    assert not votes[1].subject_boxes
    result = report(IndependentEventState(cfg()), 1.2, 18, votes)
    assert result["analysis_complete"] and not result["localization_complete"]
    assert not result["objects"]
    assert "invalid_subject_bbox" in votes[1].grounding_issues


def test_missing_person_reference_retains_event_without_inventing_box():
    raw = payload()
    raw["s"] = []
    votes = parse(raw)
    assert votes[1].detected is True
    assert votes[1].grounding_issues == ("missing_subject",)
    assert not votes[1].subject_boxes


def test_transport_validation_failure_is_failed_request_not_negative(monkeypatch):
    import httpx

    from goodvision_harness import scene_model
    from tools.vlm_benchmark.behavior import evaluate

    raw = payload()
    del raw["e"]["weapon"]
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(raw))

    monkeypatch.setattr(
        scene_model, "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    settings = replace(cfg(), api_key="local-test-only")
    inspector = mode_inspector(settings)
    try:
        result = evaluate({}, job(settings, 1.2, 18), settings, inspector)
    finally:
        inspector.close()
    assert len(requests) == 1
    assert requests[0]["response_format"] == response_format(LABELS)
    assert result["state"] == "failed" and result["error_code"] == "invalid_response"
    assert result["judgments"] == {} and not result["report"]["analysis_complete"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(v=True),
        lambda r: r.update(v=2),
        lambda r: r.update(extra=1),
        lambda r: r.pop("v"),
        lambda r: r["e"].pop("weapon"),
        lambda r: r["e"].update(unknown=[]),
        lambda r: r["e"].update(fall=[]),
        lambda r: r.update(e=[]),
        lambda r: r.update(s={}),
        lambda r: r["s"][0].update(extra=1),
        lambda r: r["s"].append(copy.deepcopy(r["s"][0])),
        lambda r: r["s"][0].update(i="person1"),
        lambda r: r["s"][0].pop("b"),
        lambda r: r["e"]["smoking"][0].pop("c"),
        lambda r: r["e"]["smoking"][0].update(d="true"),
        lambda r: r["e"]["smoking"][0].update(d=1),
        lambda r: r["e"]["smoking"][0].update(c=True),
        lambda r: r["e"]["smoking"][0].update(c=2),
        lambda r: r["e"]["smoking"][0].update(c=float("nan")),
        lambda r: r["e"]["smoking"][0].update(p="p1"),
        lambda r: r["e"]["smoking"][0].update(p=["p1", "p1"]),
        lambda r: r["e"]["smoking"][0].update(extra=0),
        lambda r: r["e"]["smoking"].append(copy.deepcopy(r["e"]["smoking"][0])),
        lambda r: r["e"]["ground_litter"][0].update(p=["p1"]),
        lambda r: r["s"].extend({"i": f"p{i}", "b": None} for i in range(2, 66)),
    ],
)
def test_invalid_responses_are_rejected_without_silent_repair(change):
    raw = payload()
    change(raw)
    with pytest.raises(_ResponseValidationError):
        parse(raw)


def test_total_verdict_limit_and_no_all_negative_shortcut():
    raw = payload()
    raw["e"]["smoking"] = [{"p": [f"p{i}"], "d": True, "c": None} for i in range(1, 65)]
    with pytest.raises(_ResponseValidationError):
        parse(raw)
    with pytest.raises(_ResponseValidationError):
        parse({"no_events": 0.9})


@pytest.mark.parametrize("finish", [None, "length", "content_filter"])
def test_truncated_completion_is_not_complete(finish):
    with pytest.raises(_ResponseValidationError, match="completion_not_finished"):
        parse(payload(), finish)


@pytest.mark.parametrize(
    "content",
    [
        '{"v":1,"v":1,"s":[],"e":{}}',
        '{"v":1,"s":[],"e":{"fight":[],"fight":[]}}',
        'prefix {"v":1,"s":[],"e":{}}',
    ],
)
def test_duplicate_fields_and_extra_prose_rejected(content):
    response = envelope(None)
    response["choices"][0]["message"]["content"] = content
    with pytest.raises(_ResponseValidationError):
        LeanGroundingInspector._parse(job(cfg(), 1.2, 18), response)


@pytest.mark.parametrize("followup", [False, True])
def test_policy_and_images_identical_and_factory_schema_scoped_to_grounding(followup):
    request = replace(job(cfg(), 1.2, 18), grounding_required=followup)
    messages = []
    for cls in (LeanGroundingInspector, CompactGroundingInspector, EventJSONInspector):
        inspector = cls(
            cfg().model_settings, model="qwen3-vl-test", api_key="fake",
            event_policy="research-v1",
        )
        try:
            messages.append(inspector.build_messages(request))
        finally:
            inspector.close()
    assert all(
        m[0]["content"].split(OUTPUT_FORMAT_MARKER)[0]
        == annotation_policy(LABELS)["decision_rules"]
        for m in messages
    )
    assert messages[0][1] == messages[1][1] == messages[2][1]
    assert "prior event" in messages[0][0]["content"] if followup else True
    original = replace(cfg(), parameters={"provider": {"only": ["deepinfra"]}})
    ground, event = (
        mode_inspector(original),
        mode_inspector(replace(original, output_mode="events")),
    )
    try:
        assert isinstance(ground, LeanGroundingInspector)
        assert isinstance(event, EventJSONInspector)
        assert ground._parameters["response_format"] == response_format(LABELS)
        assert (
            event._parameters["response_format"]["json_schema"]["name"]
            == "scene_events"
        )
        assert original.parameters == {"provider": {"only": ["deepinfra"]}}
        assert (
            ground.settings.timeout_seconds == original.model_settings.timeout_seconds
        )
    finally:
        ground.close()
        event.close()
    legacy = _inspector(replace(cfg(), grounding_format="legacy"))
    try:
        assert isinstance(legacy, CompactGroundingInspector)
    finally:
        legacy.close()
    with pytest.raises(ValueError, match="wire format"):
        replace(cfg(), grounding_format="unknown")
