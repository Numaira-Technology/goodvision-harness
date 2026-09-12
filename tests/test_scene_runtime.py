"""Exercise the bridge using source observations and explicitly claimed jobs."""

from dataclasses import replace

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict

BOX = NormalizedBBox(0.1, 0.1, 0.4, 0.9)


def tracks(runtime, at, ids=("a",), *, state=None):
    metadata = (
        {}
        if state is None
        else {
            "observation_state": state,
            "last_observed_timestamp_seconds": max(0, at - 1),
            "last_observed_frame_sequence": max(0, int(at * 10) - 1),
        }
    )
    runtime.observe_tracks(
        RichLabelSnapshot(
            runtime.stream_session_id,
            int(at * 10),
            at,
            tuple(RichLabelObject(sid, "person", BOX, **metadata) for sid in ids),
        )
    )


def result(
    at,
    groups=(),
    *,
    detected=True,
    label="smoking",
    message=None,
    objects=(),
    source="camera",
    session="s",
):
    event_objects = tuple(
        GroundedObject(
            f"event-{index}",
            "cigarette",
            BOX,
            attributes={"subject_ids": group},
        )
        for index, group in enumerate(groups)
    )
    return AnalysisResult(
        message or f"r-{at}",
        source,
        session,
        int(at * 10) + 1,
        "2026-09-12T00:00:00Z",
        FrameReference(int(at * 10), at, "2026-09-12T00:00:00Z", 640, 480),
        (
            EventJudgment(
                label,
                detected,
                0.9,
                "observation",
                tuple(obj.object_id for obj in event_objects),
            ),
        ),
        (*event_objects, *objects),
        "test",
        0.5,
    )


def test_duplicate_source_and_late_result_do_not_resurrect_ended_event():
    runtime = SceneRuntime("s", "camera")
    for at, detected in ((1, True), (2, True), (3, False), (4, False)):
        tracks(runtime, at)
        current = result(at, (("a",),) if detected else (), detected=detected)
        runtime.observe_result(current)
        runtime.observe_result(current)
    runtime.observe_result(result(1, (("a",),), message="late"))
    event = runtime.scene.snapshot()["events"][0]
    assert event["phase"] == "ended"
    assert len(event["history"]) == 4


def test_negative_waits_for_exact_source_people_and_never_uses_future_people():
    runtime = SceneRuntime("s", "camera")
    for at in (1, 2):
        tracks(runtime, at)
        runtime.observe_result(result(at, (("a",),)))
    runtime.observe_result(result(3, detected=False))
    assert runtime.scene.snapshot()["events"][0]["negative_observations"] == 0
    tracks(runtime, 3)
    assert runtime.scene.snapshot()["events"][0]["negative_observations"] == 1
    runtime.observe_result(result(4, detected=False))
    tracks(runtime, 5)
    assert runtime.scene.snapshot()["events"][0]["phase"] == "confirmed"


def test_explicit_targeted_negative_does_not_clear_another_person():
    runtime = SceneRuntime("s", "camera")
    for at, groups, detected in (
        (1, (("a",), ("b",)), True),
        (2, (("a",), ("b",)), True),
        (3, (("a",),), False),
        (4, (("a",),), False),
    ):
        tracks(runtime, at, ("a", "b"))
        runtime.observe_result(result(at, groups, detected=detected))
    events = {
        tuple(e["subject_ids"]): e["phase"] for e in runtime.scene.snapshot()["events"]
    }
    assert events == {("a",): "ended", ("b",): "confirmed"}


def test_same_message_regrouping_retracts_old_groups_without_counting_new_frames():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, ("a", "b", "c", "d"))
    runtime.observe_result(result(1, (("a", "b"), ("c", "d")), label="fight"))
    runtime.observe_result(result(1, (("a", "c"), ("b", "d")), label="fight"))
    events = {
        tuple(event["subject_ids"]): event
        for event in runtime.scene.snapshot()["events"]
    }
    assert events[("a", "b")]["phase"] == "retracted"
    assert events[("c", "d")]["phase"] == "retracted"
    assert events[("a", "c")]["phase"] == "suspected"
    assert events[("b", "d")]["phase"] == "suspected"
    assert events[("a", "c")]["positive_observations"] == 1
    assert runtime.snapshot()["stats"]["association_revisions_applied"] == 2


def test_removed_object_slot_is_withdrawn_and_reordered_slots_keep_stable_identity():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, ("a", "b"))
    initial = result(1, (("a",), ("b",)))
    runtime.observe_result(initial)
    original = runtime.scene.snapshot()["events"]
    reordered = replace(
        initial,
        objects=tuple(reversed(initial.objects)),
        judgments=(
            replace(
                initial.judgments[0],
                object_ids=tuple(reversed(initial.judgments[0].object_ids)),
            ),
        ),
    )
    runtime.observe_result(reordered)
    assert runtime.scene.snapshot()["events"] == original
    runtime.observe_result(result(1, (("a",),)))
    events = {
        tuple(event["subject_ids"]): event
        for event in runtime.scene.snapshot()["events"]
    }
    assert events[("a",)]["phase"] == "suspected"
    assert events[("b",)]["phase"] == "retracted"
    runtime.observe_result(result(1, (("a",),)))
    assert runtime.snapshot()["stats"]["association_revisions_applied"] == 1


def test_object_slots_merged_into_one_group_count_one_source_observation():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, ("a", "b"))
    runtime.observe_result(result(1, (("a",), ("b",)), label="fight"))
    runtime.observe_result(result(1, (("a", "b"), ("a", "b")), label="fight"))
    active = [
        event
        for event in runtime.scene.snapshot()["events"]
        if event["phase"] != "retracted"
    ]
    assert len(active) == 1
    assert active[0]["positive_observations"] == 1
    assert active[0]["subject_ids"] == ["a", "b"]


def test_late_owner_correction_replays_subsequent_whole_frame_negatives():
    runtime = SceneRuntime("s", "camera")
    for at, detected in ((1, True), (2, True), (3, False), (4, False)):
        tracks(runtime, at, ("a", "b"))
        runtime.observe_result(
            result(at, (("a",),) if detected else (), detected=detected)
        )
    runtime.observe_result(result(1, (("b",),)))
    runtime.observe_result(result(2, (("b",),)))
    events = {
        tuple(event["subject_ids"]): event
        for event in runtime.scene.snapshot()["events"]
    }
    assert events[("a",)]["phase"] == "retracted"
    assert events[("b",)]["phase"] == "ended"
    assert events[("b",)]["frame_sequence"] == 40


def test_same_message_source_cannot_be_retimed_during_regrouping():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, ("a", "b"))
    runtime.observe_result(result(1, (("a",),), message="fixed-source"))
    original = runtime.scene.snapshot()["events"]
    tracks(runtime, 2, ("a", "b"))
    runtime.observe_result(result(2, (("b",),), message="fixed-source"))
    assert runtime.scene.snapshot()["events"] == original
    assert runtime.snapshot()["stats"]["association_revisions_ignored"] > 0


def test_active_scene_legacy_bridge_only_accepts_stranger_verdicts():
    runtime = SceneRuntime("s", "camera", mode="active")
    tracks(runtime, 1, ("a", "b"))
    runtime.observe_result(result(1, (("a",),)))
    assert not runtime.scene.snapshot()["events"]
    runtime.observe_result(
        result(1, (("a",),), label="stranger", message="face-result")
    )
    runtime.observe_result(
        result(1, (("b",),), label="stranger", message="face-result")
    )
    events = {
        tuple(event["subject_ids"]): event
        for event in runtime.scene.snapshot()["events"]
    }
    assert events[("a",)]["phase"] == "retracted"
    assert events[("b",)]["event_label"] == "stranger"


def test_unassigned_positive_can_be_completed_by_same_message_correction():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1)
    runtime.observe_result(result(1, ((),)))
    assert not runtime.scene.snapshot()["events"]
    runtime.observe_result(result(1, (("a",),)))
    assert runtime.scene.snapshot()["events"][0]["phase"] == "suspected"


def test_source_and_session_isolation_and_predictions():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, state="predicted")
    assert not runtime.scene.snapshot()["entities"]
    tracks(runtime, 1)
    runtime.observe_result(result(1, (("a",),), session="old"))
    runtime.observe_result(result(1, (("a",),), source="other"))
    assert not runtime.scene.snapshot()["events"]


def test_deleted_identity_cannot_return_from_delayed_result_until_reenrolled():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1)
    face = GroundedObject(
        "face",
        "known_person",
        BOX,
        attributes={
            "subject_id": "a",
            "status": "known",
            "name": "Alice",
            "person_id": "alice",
            "face_observation_state": "observed",
        },
    )
    observed = result(1, detected=False, label="stranger", objects=(face,))
    runtime.observe_result(observed)
    assert runtime.scene.snapshot()["entities"][0]["identity"]["name"] == "Alice"
    runtime.forget_identity("alice")
    runtime.observe_result(replace(observed, message_id="delayed"))
    assert not runtime.scene.snapshot()["entities"][0]["identity"]
    runtime.allow_identity("alice")
    runtime.observe_result(replace(observed, message_id="enrolled"))
    assert runtime.scene.snapshot()["entities"][0]["identity"]["name"] == "Alice"


def test_unusable_face_is_not_stranger_clear():
    runtime = SceneRuntime("s", "camera")
    for at in (1, 2):
        tracks(runtime, at)
        runtime.observe_result(result(at, (("a",),), label="stranger"))
    for at in (3, 4):
        tracks(runtime, at)
        runtime.observe_result(
            result(
                at,
                detected=False,
                label="stranger",
                objects=(
                    GroundedObject(
                        "face",
                        "unusable_face",
                        BOX,
                        attributes={
                            "subject_id": "a",
                            "status": "unusable",
                            "face_observation_state": "observed",
                        },
                    ),
                ),
            )
        )
    assert runtime.scene.snapshot()["events"][0]["phase"] == "confirmed"


def prepared():
    runtime = SceneRuntime("s", "camera")
    for at in (0, 0.5):
        tracks(runtime, at)
        runtime.observe_frame(b"jpeg", at, int(at * 10))
    jobs = runtime.take_inspections()
    job = next(job for job in jobs if job.request.subject_ids == ("a",))
    return runtime, job


def test_temporal_inspection_has_multiple_exact_source_frames_and_claims_once():
    runtime, job = prepared()
    assert len(job.frames) == 2
    assert len(job.subject_boxes) == 2
    assert not runtime.take_inspections()
    assert "jpeg_bytes" not in str(runtime.snapshot())
    counts = {}
    assert runtime.complete_inspection(
        job.request.request_id,
        [
            TemporalVerdict("smoking", ("a",), True, 0.9),
        ],
        completion_counts=counts,
    )
    assert counts == {"accepted_verdicts": 1, "rejected_verdicts": 0}
    assert runtime.scene.snapshot()["events"][0]["frame_sequence"] == 5
    assert not runtime.complete_inspection(job.request.request_id, [])


def test_slow_temporal_completion_uses_pinned_evidence_after_15fps_history_rollover():
    runtime, job = prepared()
    for sequence in range(6, 100):
        runtime.observe_tracks(
            RichLabelSnapshot(
                runtime.stream_session_id,
                sequence,
                0.5 + (sequence - 5) / 15,
                (RichLabelObject("a", "person", BOX),),
            )
        )
    before = runtime.scene.snapshot()["entities"][0]
    assert before["last_observed_timestamp_seconds"] > 6
    assert not any(item["frame_sequence"] == 5 for item in before["history"])
    counts = {}
    assert runtime.complete_inspection(
        job.request.request_id,
        [TemporalVerdict("smoking", ("a",), True, 0.9)],
        completion_counts=counts,
    )
    assert counts == {"accepted_verdicts": 1, "rejected_verdicts": 0}
    assert runtime.scene.snapshot()["entities"][0] == before
    assert runtime.scene.snapshot()["events"][0]["frame_sequence"] == 5
    assert job.request.request_id not in runtime.scene._source_pins
    runtime.close()
    assert runtime.scene.snapshot()["source_pins"] == {"jobs": 0, "observations": 0}


def test_latest_successful_inspection_is_immutable_bounded_and_released_on_close():
    runtime, job = prepared()
    assert runtime.latest_inspection() is None
    assert runtime.complete_inspection(job.request.request_id, [])
    assert runtime.latest_inspection() is job
    assert job.request.request_id not in runtime.scene._source_pins
    other = next(iter(runtime._jobs.values()))
    assert runtime.complete_inspection(other.request.request_id, [])
    assert runtime.latest_inspection() is other
    assert len(other.frames) <= 4
    assert sum(len(frame.jpeg_bytes) for frame in other.frames) <= 16 * 1024 * 1024
    runtime.close()
    assert runtime.latest_inspection() is None


def test_failed_and_invalid_inspections_do_not_replace_latest_success():
    runtime, job = prepared()
    assert runtime.complete_inspection(job.request.request_id, [])
    other = next(iter(runtime._jobs.values()))
    with pytest.raises(ValueError):
        runtime.complete_inspection(
            other.request.request_id, [TemporalVerdict("smoking", ("missing",), True)]
        )
    assert runtime.latest_inspection() is job
    assert runtime.complete_inspection(other.request.request_id, [], success=False)
    assert runtime.latest_inspection() is job
    assert not runtime.complete_inspection(other.request.request_id, [])
    assert runtime.latest_inspection() is job
    runtime.close()


def full_scene_with_partial_appearances(first=("a", "b"), second=("a",)):
    runtime = SceneRuntime("s", "camera")
    for at, subjects in ((0, first), (0.5, second)):
        tracks(runtime, at, subjects)
        runtime.observe_frame(b"jpeg", at, int(at * 10))
    job = next(job for job in runtime.take_inspections() if not job.request.subject_ids)
    return runtime, job


def test_full_scene_verdict_uses_last_real_shared_source_without_reviving_visibility():
    runtime, job = full_scene_with_partial_appearances()
    runtime.scene.advance(4)
    counts = {}
    assert runtime.complete_inspection(
        job.request.request_id,
        [TemporalVerdict("smoking", ("b",), True, 0.9)],
        completion_counts=counts,
    )
    assert counts == {"accepted_verdicts": 1, "rejected_verdicts": 0}
    scene = runtime.scene.snapshot()
    event = scene["events"][0]
    assert event["frame_sequence"] == 0
    assert event["last_evidence_timestamp_seconds"] == 0
    assert event["observation_state"] == "uncertain"
    person_b = next(
        entity for entity in scene["entities"] if entity["subject_id"] == "b"
    )
    assert person_b["last_observed_timestamp_seconds"] == 0
    assert person_b["observation_state"] == "occluded"


def test_full_scene_group_without_any_shared_frame_cannot_create_fight():
    runtime, job = full_scene_with_partial_appearances(first=("a",), second=("b",))
    counts = {}
    assert runtime.complete_inspection(
        job.request.request_id,
        [TemporalVerdict("fight", ("a", "b"), True, 0.9)],
        completion_counts=counts,
    )
    assert counts == {"accepted_verdicts": 0, "rejected_verdicts": 1}
    assert not runtime.scene.snapshot()["events"]
    assert job.request.request_id not in runtime.scene._source_pins


def test_old_full_scene_negative_cannot_override_newer_positive_evidence():
    runtime, job = full_scene_with_partial_appearances()
    tracks(runtime, 1, ("b",))
    assert runtime.scene.observe_event(
        "smoking",
        ("b",),
        timestamp_seconds=1,
        frame_sequence=10,
        detected=True,
        evidence_id="newer",
    )
    before = runtime.scene.snapshot()["events"]
    counts = {}
    assert runtime.complete_inspection(
        job.request.request_id,
        [TemporalVerdict("smoking", ("b",), False, 0.9)],
        completion_counts=counts,
    )
    assert counts == {"accepted_verdicts": 0, "rejected_verdicts": 1}
    assert runtime.scene.snapshot()["events"] == before


@pytest.mark.parametrize(
    "bad",
    [
        TemporalVerdict("smoking", ("a",), True, True),
        TemporalVerdict("smoking", ("a",), True, float("nan")),
        TemporalVerdict("smoking", ("other",), True, 0.9),
        TemporalVerdict("stranger", ("a",), True, 0.9),
    ],
)
def test_invalid_temporal_batch_is_atomic_and_still_can_be_failed(bad):
    runtime, job = prepared()
    with pytest.raises(ValueError):
        runtime.complete_inspection(
            job.request.request_id,
            [
                TemporalVerdict("fall", ("a",), True, 0.9),
                bad,
            ],
        )
    assert not runtime.scene.snapshot()["events"]
    assert runtime.complete_inspection(job.request.request_id, [], success=False)


def test_inspection_expires_and_close_releases_all_images():
    runtime, job = prepared()
    runtime.observe_frame(b"new", 20.5, 205)
    assert not runtime.complete_inspection(job.request.request_id, [])
    assert job.request.request_id not in runtime.scene._source_pins
    runtime.close()
    assert len(runtime.evidence) == 0
    assert not runtime.take_inspections()
    assert runtime.snapshot()["state"] == "stopped"
    assert runtime.scene.snapshot()["source_pins"] == {"jobs": 0, "observations": 0}
    runtime.observe_result(result(21, (("a",),)))
    assert not runtime.scene.snapshot()["events"]


def test_inspection_ids_do_not_collide_between_session_instances():
    first, first_job = prepared()
    second, second_job = prepared()
    assert first_job.request.request_id != second_job.request.request_id
    assert not second.complete_inspection(first_job.request.request_id, [])
    first.close()
    second.close()


def test_stranger_refresh_is_not_sent_to_temporal_safety_worker():
    runtime = SceneRuntime("s", "camera")
    for at in (0, 0.5):
        tracks(runtime, at)
        runtime.observe_result(result(at, (("a",),), label="stranger"))
        runtime.observe_frame(b"jpeg", at, int(at * 10))
    assert runtime.take_inspections()
    assert all(
        "stranger" not in job.request.event_labels for job in runtime._jobs.values()
    )
