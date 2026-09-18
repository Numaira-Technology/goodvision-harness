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
            attributes={
                "subject_ids": group,
                **({"status": "unknown", "face_observation_state": "observed",
                    "identity_source": "face_match"} if label == "stranger" else {}),
            },
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


@pytest.mark.parametrize("label", ["fight", "smoking", "fall", "weapon"])
def test_per_frame_safety_results_cannot_create_or_clear_scene_events(label):
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1)
    runtime.observe_result(result(1, (("a",),), label=label))
    assert not runtime.scene.snapshot()["events"]
    assert runtime.scene.observe_event(
        label, ("a",), timestamp_seconds=1, frame_sequence=10,
        detected=True, evidence_id="temporal-positive",
    )
    for at in range(2, 13):
        tracks(runtime, at)
        runtime.observe_result(result(at, detected=False, label=label))
    event = runtime.scene.snapshot()["events"][0]
    assert event["phase"] == "suspected"
    assert event["negative_observations"] == 0


def test_removed_shadow_mode_is_rejected():
    with pytest.raises(ValueError, match="active"):
        SceneRuntime("s", "camera", mode="shadow")


def prepare_batch(runtime, start, subjects=("a",)):
    snapshots = []
    for at in (start, start + 1):
        sequence = int(at * 10)
        runtime.observe_frame(b"jpeg", at, sequence)
        snapshots.append(RichLabelSnapshot(
            runtime.stream_session_id, sequence, at,
            tuple(RichLabelObject(subject, "person", BOX) for subject in subjects),
        ))
    assert runtime.observe_tracking_batch(
        tuple(snapshots), completed_timestamp_seconds=start + 1,
        completed_frame_sequence=int((start + 1) * 10),
    )
    jobs = runtime.take_inspections()
    assert len(jobs) == 1
    return jobs[0]


def test_temporal_clear_requires_two_distinct_endpoints_for_same_entity():
    runtime = SceneRuntime("s", "camera", tracking_mode="batch")
    job = prepare_batch(runtime, 0, ("a", "b"))
    assert runtime.complete_inspection(job.request.request_id, [
        TemporalVerdict("smoking", ("a",), True),
        TemporalVerdict("weapon", ("a",), True),
        TemporalVerdict("smoking", ("b",), True),
    ])
    for index in range(1, 3):
        job = prepare_batch(runtime, index * 2, ("a", "b"))
        assert len(job.frames) == 2
        verdicts = [TemporalVerdict("smoking", ("a",), False)]
        assert runtime.complete_inspection(job.request.request_id, verdicts)
        assert not runtime.complete_inspection(job.request.request_id, verdicts)
        events = {
            (event["event_label"], tuple(event["subject_ids"])): event
            for event in runtime.scene.snapshot()["events"]
        }
        smoking = events[("smoking", ("a",))]
        assert smoking["negative_observations"] == index
        assert smoking["phase"] == ("ended" if index == 2 else "suspected")
        assert events[("smoking", ("b",))]["phase"] == "suspected"
        assert events[("weapon", ("a",))]["phase"] == "suspected"


def test_late_temporal_positive_is_replayed_before_newer_clear_endpoint():
    runtime = SceneRuntime("s", "camera", tracking_mode="batch")
    old = prepare_batch(runtime, 0)
    newer = prepare_batch(runtime, 2)
    assert runtime.complete_inspection(newer.request.request_id, [
        TemporalVerdict("smoking", ("a",), False),
    ])
    assert not runtime.scene.snapshot()["events"]
    assert runtime.complete_inspection(old.request.request_id, [
        TemporalVerdict("smoking", ("a",), True),
    ])
    event = runtime.scene.snapshot()["events"][0]
    assert event["phase"] == "suspected"
    assert event["negative_observations"] == 1
    assert event["first_evidence_timestamp_seconds"] == 1
    assert event["last_evidence_timestamp_seconds"] == 3


def test_face_bridge_only_accepts_stranger_verdicts():
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
    runtime.observe_result(result(1, ((),), label="stranger"))
    assert not runtime.scene.snapshot()["events"]
    runtime.observe_result(result(1, (("a",),), label="stranger"))
    assert runtime.scene.snapshot()["events"][0]["phase"] == "suspected"


def test_source_and_session_isolation_and_predictions():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, state="predicted")
    assert not runtime.scene.snapshot()["entities"]
    tracks(runtime, 1)
    runtime.observe_result(result(1, (("a",),), session="old", label="stranger"))
    runtime.observe_result(result(1, (("a",),), source="other", label="stranger"))
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
    assert runtime.snapshot()["scene"]["events"][0]["phase"] == "suspected"


def test_stranger_needs_both_source_confirmations_and_identity_review():
    runtime = SceneRuntime("s", "camera")
    for at in (1, 2):
        tracks(runtime, at)
        runtime.observe_result(result(at, (("a",),), label="stranger"))
        view = runtime.snapshot()["scene"]
        assert view["events"][0]["phase"] == "suspected"
        assert view["events"][0]["alert"] is False
        assert view["entities"][0]["identity_pending"]
    runtime.apply_identity_review({"a": "unresolved"})
    assert runtime.snapshot()["scene"]["events"][0]["phase"] == "suspected"
    runtime.apply_identity_review({"a": "unknown"})
    view = runtime.snapshot()["scene"]
    assert view["events"][0]["phase"] == "confirmed"
    assert view["events"][0]["positive_observations"] == 2
    assert view["events"][0]["confirmed_since_timestamp_seconds"] == 2
    runtime.observe_result(result(2, (("a",),), label="stranger"))
    assert runtime.snapshot()["scene"]["events"][0]["alert"] is True


def test_identity_review_cannot_replace_n_distinct_source_observations():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1)
    runtime.observe_result(result(1, (("a",),), label="stranger"))
    runtime.observe_result(result(1, (("a",),), label="stranger", message="duplicate"))
    runtime.apply_identity_review({"a": "unknown"})
    event = runtime.snapshot()["scene"]["events"][0]
    assert event["positive_observations"] == 1
    assert event["phase"] == "suspected"


@pytest.mark.parametrize("attributes", [
    {"face_observation_state": "held", "identity_source": "track_history"},
    {"face_observation_state": "observed", "identity_source": "track_history"},
    {"status": "unusable"},
])
def test_held_and_unusable_strangers_never_add_confirmation_votes(attributes):
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1)
    runtime.observe_result(result(1, (("a",),), label="stranger"))
    for at in (2, 3, 4):
        tracks(runtime, at)
        observation = result(at, (("a",),), label="stranger")
        item = observation.objects[0]
        runtime.observe_result(replace(observation, objects=(replace(
            item, attributes={**item.attributes, **attributes},
        ),)))
    runtime.apply_identity_review({"a": "unknown"})
    event = runtime.snapshot()["scene"]["events"][0]
    assert event["positive_observations"] == 1
    assert event["negative_observations"] == 0
    assert event["phase"] == "suspected"


def test_known_merged_identity_suppresses_stranger_but_preserves_safety_alert():
    runtime = SceneRuntime("s", "camera", stranger_enabled=True)
    for at in (1, 2):
        tracks(runtime, at)
        runtime.observe_result(result(at, (("a",),), label="stranger"))
        runtime.scene.observe_event(
            "smoking", ("a",), timestamp_seconds=at, frame_sequence=int(at * 10),
            detected=True, evidence_id=f"smoking-{at}",
        )
    runtime.apply_identity_review({"a": "known"})
    events = {
        item["event_label"]: item for item in runtime.snapshot()["scene"]["events"]
    }
    assert events["stranger"]["phase"] == "retracted"
    assert events["stranger"]["alert"] is False
    assert events["smoking"]["phase"] == "confirmed"
    assert events["smoking"]["alert"] is True


def test_no_face_is_a_yellow_identity_warning_only_when_stranger_is_enabled():
    runtime = SceneRuntime("s", "camera", stranger_enabled=True)
    tracks(runtime, 1)
    view = runtime.snapshot()["scene"]
    assert view["entities"][0]["identity_pending"]
    assert view["events"] == []
    assert runtime.display_snapshot(1)["entities"][0]["identity_pending"]
    runtime.stranger_enabled = False
    assert not runtime.snapshot()["scene"]["entities"][0]["identity_pending"]


def prepared():
    runtime = SceneRuntime("s", "camera")
    for at in (0, 1):
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
    assert runtime.scene.snapshot()["events"][0]["frame_sequence"] == 10
    assert not runtime.complete_inspection(job.request.request_id, [])


def test_slow_temporal_completion_uses_pinned_evidence_after_15fps_history_rollover():
    runtime, job = prepared()
    for sequence in range(11, 110):
        runtime.observe_tracks(
            RichLabelSnapshot(
                runtime.stream_session_id,
                sequence,
                1 + (sequence - 10) / 15,
                (RichLabelObject("a", "person", BOX),),
            )
        )
    before = runtime.scene.snapshot()["entities"][0]
    assert before["last_observed_timestamp_seconds"] > 6
    assert not any(item["frame_sequence"] == 10 for item in before["history"])
    counts = {}
    assert runtime.complete_inspection(
        job.request.request_id,
        [TemporalVerdict("smoking", ("a",), True, 0.9)],
        completion_counts=counts,
    )
    assert counts == {"accepted_verdicts": 1, "rejected_verdicts": 0}
    assert runtime.scene.snapshot()["entities"][0] == before
    assert runtime.scene.snapshot()["events"][0]["frame_sequence"] == 10
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
    for at, subjects in ((0, first), (1, second)):
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
    tracks(runtime, 2, ("b",))
    assert runtime.scene.observe_event(
        "smoking",
        ("b",),
        timestamp_seconds=2,
        frame_sequence=20,
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
    assert counts == {"accepted_verdicts": 1, "rejected_verdicts": 0}
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
    runtime.observe_frame(b"new", 21, 210)
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
    for at in (0, 1):
        tracks(runtime, at)
        runtime.observe_result(result(at, (("a",),), label="stranger"))
        runtime.observe_frame(b"jpeg", at, int(at * 10))
    assert runtime.take_inspections()
    assert all(
        "stranger" not in job.request.event_labels for job in runtime._jobs.values()
    )
