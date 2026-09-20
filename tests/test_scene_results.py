"""Legacy result consumers must only see scene-confirmed alerts."""

import copy
import json
from pathlib import Path

import jsonschema

from goodvision_harness.contracts import FrameReference, NormalizedBBox
from goodvision_harness.scene_results import SceneResultPublisher
from goodvision_harness.scene_runtime import SceneRuntime

NOW = "2026-09-19T01:00:00Z"
BOX = NormalizedBBox(0.1, 0.2, 0.5, 0.9)
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas/analysis-result-v2.schema.json")
    .read_text(encoding="utf-8")
)


def setup(*, stranger=False):
    runtime = SceneRuntime("run", "camera", stranger_enabled=stranger)
    publisher = SceneResultPublisher()
    publisher.reset("run")
    return runtime, publisher


def observe(runtime, publisher, sequence, *, subject="person-a", detected=True,
            label="smoking"):
    runtime.scene.observe_person(
        subject, timestamp_seconds=float(sequence), frame_sequence=sequence,
        bbox=BOX,
    )
    runtime.scene.observe_event(
        label, (subject,), timestamp_seconds=float(sequence),
        frame_sequence=sequence, detected=detected,
        evidence_id=f"e-{subject}-{sequence}", confidence=0.9,
    )
    publisher.observe_reference("run", FrameReference(
        sequence, float(sequence), NOW, 1920, 1080,
    ))


def project(runtime, publisher, labels=("smoking",)):
    result = publisher.project(
        runtime.snapshot(include_history=False), labels=labels, emitted_at=NOW,
    )
    if result is not None:
        jsonschema.Draft202012Validator(SCHEMA).validate(result.as_dict())
    return result


def test_suspected_confirmed_clear_and_retracted_follow_public_phase():
    runtime, publisher = setup()
    observe(runtime, publisher, 1)
    suspected = project(runtime, publisher)
    assert suspected.judgments[0].detected is False
    assert suspected.description == "暂无已确认的事件。 / No confirmed events."
    observe(runtime, publisher, 2)
    confirmed = project(runtime, publisher)
    assert confirmed.judgments[0].detected is True
    assert confirmed.judgments[0].object_ids == ("person-a",)
    assert confirmed.frame.width == 1920
    assert confirmed.objects[0].bbox == BOX
    assert confirmed.analysis_kind == "scene_state"
    observe(runtime, publisher, 3, detected=False)
    assert project(runtime, publisher).judgments[0].detected is True
    observe(runtime, publisher, 4, detected=False)
    assert project(runtime, publisher).judgments[0].detected is False
    assert publisher._sequence == 4
    snapshot = runtime.snapshot(include_history=False)
    snapshot["scene"]["events"][0].update(phase="retracted", alert=False)
    result = publisher.project(snapshot, labels=("smoking",), emitted_at=NOW)
    assert result.judgments[0].detected is False


def test_multiple_people_do_not_clear_another_confirmed_event():
    runtime, publisher = setup()
    for i in (1, 2):
        observe(runtime, publisher, i)
    for i in (3, 4):
        observe(runtime, publisher, i, subject="person-b")
    project(runtime, publisher)
    for i in (5, 6):
        observe(runtime, publisher, i, detected=False)
    result = project(runtime, publisher)
    assert result.judgments[0].detected is True
    assert result.judgments[0].object_ids == ("person-b",)


def test_unknown_classes_are_missing_and_empty_scene_does_not_claim_clear():
    runtime, publisher = setup()
    publisher.observe_reference("run", FrameReference(0, 0, NOW, 320, 240))
    assert project(runtime, publisher) is None
    observe(runtime, publisher, 1)
    result = project(runtime, publisher, ("smoking", "fight", "stranger"))
    assert result.missing_event_labels == ("fight", "stranger")
    assert result.as_dict()["analysis_complete"] is False
    assert [j.event_label for j in result.judgments] == ["smoking"]


def test_heartbeats_deduplicate_but_same_frame_revisions_publish():
    runtime, publisher = setup()
    for i in (1, 2):
        observe(runtime, publisher, i)
    first = project(runtime, publisher)
    assert project(runtime, publisher) is None
    runtime.scene.advance(2.1)
    publisher.observe_reference("run", FrameReference(3, 2.1, NOW, 1920, 1080))
    assert project(runtime, publisher) is None
    snapshot = runtime.snapshot(include_history=False)
    snapshot["scene"]["events"][0].update(phase="retracted", alert=False)
    correction = publisher.project(snapshot, labels=("smoking",), emitted_at=NOW)
    assert correction.sequence > first.sequence
    assert correction.judgments[0].detected is False


def test_unknown_face_is_not_positive_before_identity_review():
    runtime, publisher = setup(stranger=True)
    for i in (1, 2):
        observe(runtime, publisher, i, label="stranger")
    result = project(runtime, publisher, ("stranger",))
    assert result.judgments[0].detected is False
    runtime.apply_identity_review({"person-a": "unknown"})
    result = project(runtime, publisher, ("stranger",))
    assert result.judgments[0].detected is True


def test_stale_sources_missing_boxes_and_new_sessions_cannot_invent_results():
    runtime, publisher = setup()
    for i in (1, 2):
        observe(runtime, publisher, i)
    snapshot = runtime.snapshot(include_history=False)
    missing = copy.deepcopy(snapshot)
    missing["scene"]["entities"] = []
    assert publisher.project(missing, labels=("smoking",), emitted_at=NOW) is None
    publisher.reset("new-run")
    publisher.observe_reference("run", FrameReference(2, 2, NOW, 320, 240))
    assert publisher.frame is None
    assert publisher.project(snapshot, labels=("smoking",), emitted_at=NOW) is None
    publisher.reset("run")
    publisher.observe_reference("run", FrameReference(3, 3, NOW, 320, 240))
    assert publisher.project(snapshot, labels=("smoking",), emitted_at=NOW) is None


def test_uncertain_observation_preserves_confirmation_and_marks_box_age():
    runtime, publisher = setup()
    for i in (1, 2):
        observe(runtime, publisher, i)
    runtime.scene.advance(20)
    publisher.observe_reference("run", FrameReference(20, 20, NOW, 320, 240))
    result = project(runtime, publisher)
    assert result.judgments[0].detected is True
    attrs = result.objects[0].attributes
    assert attrs["observation_state"] == "occluded"
    assert attrs["last_observed_timestamp_seconds"] == 2
    assert attrs["scene_events"][0]["observation_state"] == "uncertain"
