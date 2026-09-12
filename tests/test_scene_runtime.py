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


def test_same_message_regrouping_is_not_new_evidence():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1, ("a", "b", "c", "d"))
    runtime.observe_result(result(1, (("a", "b"), ("c", "d")), label="fight"))
    runtime.observe_result(result(1, (("a", "c"), ("b", "d")), label="fight"))
    assert len(runtime.scene.snapshot()["events"]) == 2
    assert runtime.snapshot()["stats"]["association_revisions_ignored"] > 0


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
    runtime.close()
    assert len(runtime.evidence) == 0
    assert not runtime.take_inspections()
    assert runtime.snapshot()["state"] == "stopped"
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
