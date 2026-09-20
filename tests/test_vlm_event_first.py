"""Event decisions survive unavailable localization without invented subjects."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_local_gpu_runtime import rig as rig
from test_vlm_independent import BOX, config, job, report, verdict

from goodvision_harness.scene_model import _ResponseValidationError
from goodvision_harness.vlm_independent import IndependentEventState
from goodvision_harness.vlm_independent_model import IndependentVLMInspector


def payload(box=None, *, detected=True):
    return {
        "subjects": [{"subject_id": "p1", "bbox": box or [0.1, 0.1, 0.5, 0.9]}],
        "verdicts": [
            {
                "event_label": "smoking",
                "subject_ids": ["p1"],
                "detected": detected,
                "confidence": 0.9,
            }
        ],
    }


def parse(raw):
    return IndependentVLMInspector._parse(
        job(config(), 1.0, 15),
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
            ],
        },
    )


def test_declared_coordinate_scale_converts_exactly_and_legacy_unit_scale_survives():
    raw = payload([100, 100, 500, 900])
    raw["bbox_coordinate_system"] = "normalized_1000"
    (converted,) = parse(raw)
    assert converted.subject_boxes == (("p1", BOX),)
    assert not converted.grounding_issues
    (legacy,) = parse(payload())
    assert legacy.subject_boxes == converted.subject_boxes


@pytest.mark.parametrize(
    "box",
    [
        [100, 100, 500, 900],
        [-0.1, 0.1, 0.5, 0.9],
        [0.5, 0.1, 0.1, 0.9],
        [0.1, 0.1, 0.1, 0.9],
        [True, 0.1, 0.5, 0.9],
        ["0.1", 0.1, 0.5, 0.9],
        [0.1, 0.1, 0.5],
        "unavailable",
    ],
)
def test_bad_box_preserves_reliable_positive_without_creating_confirmed_person(box):
    votes = parse(payload(box))
    assert votes[0].detected is True and votes[0].confidence == 0.9
    assert votes[0].subject_boxes == ()
    state = IndependentEventState(replace(config(), labels=("smoking",)))
    for at, seq in ((1.0, 15), (1.8, 27), (2.6, 39)):
        result = report(state, at, seq, votes)
        assert result["state"] == "completed" and result["analysis_complete"]
        assert not result["localization_complete"]
        assert result["objects"] == result["events"] == []
        (observation,) = result["observations"]
        assert observation["detected"] is True and not observation["accepted"]
        (judgment,) = result["judgments"]
        assert judgment["observed_detected"] is True
        assert judgment["detected"] is None
        assert judgment["confidence"] == 0.9
        assert judgment["localization_state"] == "unavailable"
        assert judgment["confirmation_state"] == "pending"
        assert "invalid_subject_bbox" in judgment["grounding_issues"]


def test_one_bad_person_cannot_be_replaced_by_the_other_fight_participant():
    raw = payload([100, 100, 500, 900])
    raw["subjects"].append({"subject_id": "p2", "bbox": [0.6, 0.1, 0.9, 0.9]})
    raw["verdicts"][0]["subject_ids"] = ["p1", "p2"]
    state = IndependentEventState(config())
    result = report(state, 1.0, 15, parse(raw))
    assert len(result["objects"]) == 1
    assert not result["observations"][0]["accepted"]
    assert result["judgments"][0]["localization_state"] == "partial"
    assert result["judgments"][0]["observed_detected"] is True
    assert result["events"] == []


@pytest.mark.parametrize(
    "subjects", [None, {}, [], [{"subject_id": "p1", "bbox": None}]]
)
def test_missing_localization_never_discards_event_assessment(subjects):
    raw = payload()
    raw["subjects"] = subjects
    result = report(IndependentEventState(config()), 1.0, 15, parse(raw))
    assert result["judgments"][0]["observed_detected"] is True
    assert result["judgments"][0]["localization_state"] == "unavailable"


def test_event_without_localization_fields_retains_its_decision():
    raw = {"verdicts": [{"event_label": "smoking", "detected": True}]}
    result = report(IndependentEventState(config()), 1.0, 15, parse(raw))
    (judgment,) = result["judgments"]
    assert judgment["observed_detected"] is True
    assert judgment["confidence"] is None
    assert judgment["localization_state"] == "unavailable"


def test_duplicate_subject_or_unsupported_units_do_not_guess_a_box():
    raw = payload()
    raw["subjects"] *= 2
    (item,) = parse(raw)
    assert item.subject_boxes == ()
    assert "duplicate_subject" in item.grounding_issues
    raw = payload()
    raw["bbox_coordinate_system"] = "pixels"
    (item,) = parse(raw)
    assert item.subject_boxes == ()
    assert "unsupported_coordinate_system" in item.grounding_issues


@pytest.mark.parametrize(
    "field,value",
    [
        ("detected", "true"),
        ("detected", 1),
        ("confidence", 2),
        ("event_label", "unrequested"),
    ],
)
def test_unreliable_event_fields_still_fail_explicitly(field, value):
    raw = payload()
    raw["verdicts"][0][field] = value
    with pytest.raises(_ResponseValidationError):
        parse(raw)


def test_unlocalized_negative_does_not_clear_confirmed_person_event():
    state = IndependentEventState(config())
    for at, seq in ((1.0, 15), (1.8, 27)):
        report(state, at, seq, (verdict(),))
    for at, seq in ((2.6, 39), (3.4, 51)):
        result = report(
            state, at, seq, parse(payload([100, 100, 500, 900], detected=False))
        )
        assert result["events"][0]["phase"] == "confirmed"
        (judgment,) = result["judgments"]
        assert judgment["observed_detected"] is False
        assert judgment["detected"] is True
        assert not result["observations"][0]["accepted"]


def test_qwen_prompt_declares_native_units_and_prioritizes_all_events():
    cfg = config()
    inspector = IndependentVLMInspector(
        cfg.model_settings, model="qwen/qwen3-vl-30b-a3b-instruct:nitro"
    )
    try:
        text = inspector.build_messages(job(cfg, 1.0, 15))[0]["content"]
        assert 'bbox_coordinate_system to "normalized_1000"' in text
        assert "Event assessment is the primary task" in text
        assert "bbox:null" in text
    finally:
        inspector.close()


def test_recorded_reply_keeps_all_four_decisions_without_guessing_undeclared_units():
    path = Path(__file__).parent / "fixtures/vlm/qwen3vl_recorded_unmarked_boxes.json"
    raw = json.loads(path.read_text())["response"]
    cfg = replace(config(), labels=("smoking", "fight", "fall", "weapon"))
    verdicts = IndependentVLMInspector._parse(
        job(cfg, 53.6, 804),
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
            ],
        },
    )
    result = report(IndependentEventState(cfg), 53.6, 804, verdicts)
    assert result["state"] == "completed" and result["analysis_complete"]
    assert len(result["judgments"]) == 4
    assert all(j["observed_detected"] is False for j in result["judgments"])
    assert all(j["confidence"] is None for j in result["judgments"])
    assert all(j["localization_state"] == "unavailable" for j in result["judgments"])
    assert result["objects"] == []


@pytest.mark.parametrize(
    "endpoint,positive_labels,unlocalized_labels",
    [
        (1.6, {"smoking"}, set()),
        (9.6, {"fight"}, set()),
        (13.6, {"fall"}, set()),
        (24.0, set(), set()),
        (42.4, set(), set()),
        (51.2, set(), {"smoking", "fall", "weapon"}),
    ],
)
def test_recorded_30b_replies_preserve_decisions_and_localization_status(
    endpoint, positive_labels, unlocalized_labels
):
    path = Path(__file__).parent / "fixtures/vlm/qwen3vl_30b_openrouter_20260921.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    window = next(w for w in recorded["windows"] if w["endpoint"] == endpoint)
    cfg = replace(config(), labels=("fight", "smoking", "fall", "weapon"))
    sequence = round(endpoint * 15)
    verdicts = IndependentVLMInspector._parse(job(cfg, endpoint, sequence), window)
    result = report(IndependentEventState(cfg), endpoint, sequence, verdicts)
    assert result["state"] == "completed" and result["analysis_complete"]
    assert len(result["judgments"]) == 4
    # These expectations preserve recorded replies, not claims of visual accuracy.
    assert {v.event_label for v in verdicts if v.detected} == positive_labels
    assert {
        v.event_label for v in verdicts if not v.subject_boxes
    } == unlocalized_labels
    assert result["localization_complete"] is (not unlocalized_labels)
    assert all("invalid_subject_bbox" not in v.grounding_issues for v in verdicts)
    if endpoint == 1.6:
        # Real wire values [483,0,659,662] must use the declared 1000 scale.
        box = verdicts[0].subject_boxes[0][1]
        assert (box.x1, box.y1, box.x2, box.y2) == (0.483, 0.0, 0.659, 0.662)
    for observation in result["observations"]:
        if observation["event_label"] in unlocalized_labels:
            assert observation["detected"] is False
            assert not observation["accepted"]
            assert "unlocalized_event" in observation["grounding_issues"]


def test_public_route_c_reaches_independent_process_as_qwen30b(rig, monkeypatch):
    from test_vlm_independent_service import LocalBranch

    from goodvision_harness import service
    from goodvision_harness.contracts import PUBLIC_MODEL_NAME
    from goodvision_harness.vlm_independent import _inspector

    monkeypatch.setattr(service, "IndependentVLMBranch", LocalBranch)
    runtime = rig.make()
    runtime.config = replace(
        runtime.config,
        runtime=replace(
            runtime.config.runtime,
            scene_independent_vlm=True,
        ),
    )
    assert runtime.select_model_route("C")["selected_id"] == "C"
    runtime.start_camera(dry_run=False, features=["smoking"])
    branch = runtime._vlm_branch
    assert branch.config.model == "qwen/qwen3-vl-30b-a3b-instruct:nitro"
    assert branch.config.route_id == "C"
    assert branch.health()["route_id"] == "C"
    inspector = _inspector(branch.config)
    try:
        assert inspector.model == branch.config.model
        assert runtime.status(lightweight=True)["model"] == PUBLIC_MODEL_NAME
    finally:
        inspector.close()
