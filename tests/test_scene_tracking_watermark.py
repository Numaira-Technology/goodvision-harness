"""Native-video completion must use its own source clock and original JPEGs."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_evidence import SceneEvidenceBuffer
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict
from goodvision_harness.scene_state import SceneState
from goodvision_harness.video_tracking import BACKEND

BOX = NormalizedBBox(0.1, 0.1, 0.4, 0.9)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.3)
NATIVE = {"backend": "sam31", "epoch": "batch-1", "object_id": "7"}


def runtime(**kwargs):
    return SceneRuntime("s", "camera", mode="active", tracking_mode="batch", **kwargs)


def capture(scene, end_sequence, *, start_sequence=0, fps=30):
    for sequence in range(start_sequence, end_sequence + 1):
        scene.observe_frame(f"jpeg-{sequence}".encode(), sequence / fps, sequence)


def track(sequence, *, at=None, ids=("person-a",), native=None, session="s"):
    timestamp = sequence / 30 if at is None else at
    return RichLabelSnapshot(
        session,
        sequence,
        timestamp,
        tuple(
            RichLabelObject(
                sid,
                "person",
                BOX,
                attributes={"native_track": NATIVE if native is None else native},
                observation_state="observed",
                last_observed_timestamp_seconds=timestamp,
                last_observed_frame_sequence=sequence,
            )
            for sid in ids
        ),
    )


def commit(scene, snapshots):
    return scene.observe_tracking_batch(
        snapshots,
        completed_timestamp_seconds=snapshots[-1].timestamp_seconds,
        completed_frame_sequence=snapshots[-1].frame_sequence,
    )


def face_result(sequence, *, session="s"):
    return AnalysisResult(
        f"face-{sequence}",
        "camera",
        session,
        sequence + 1,
        "2026-09-12T00:00:00Z",
        FrameReference(sequence, sequence / 30, "2026-09-12T00:00:00Z", 640, 480),
        (EventJudgment("stranger", False, 0.9, "known face", ()),),
        (
            GroundedObject(
                "face",
                "person",
                FACE,
                attributes={
                    "subject_ids": ["person-a"],
                    "face_observation_state": "observed",
                    "subject_identity": {"status": "known", "name": "Alice"},
                },
            ),
        ),
        "test",
        0.1,
    )


def test_capture_five_seconds_ahead_plans_source_interval_with_true_display_lag():
    scene = runtime()
    capture(scene, 240)
    assert scene.take_inspections() == ()
    # 30 FPS capture and 4 FPS inference do not have matching regular indexes.
    snapshots = tuple(track(round(index * 7.5)) for index in range(13))
    assert commit(scene, snapshots)
    jobs = scene.take_inspections()
    assert len(jobs) == 2
    assert {job.request.reason for job in jobs} == {"full_scene_scan"}
    assert jobs[0].request.end_seconds < jobs[1].request.start_seconds
    assert jobs[0].frames[0].frame_sequence == 0
    assert jobs[-1].frames[-1].frame_sequence == 90
    for job in jobs:
        assert len(job.frames) == 4
        assert all(
            job.request.start_seconds
            <= frame.timestamp_seconds
            <= job.request.end_seconds
            for frame in job.frames
        )
        assert all(
            frame.jpeg_bytes == f"jpeg-{frame.frame_sequence}".encode()
            for frame in job.frames
        )
        assert {sequence for sequence, _, _ in job.subject_boxes} == {
            frame.frame_sequence for frame in job.frames
        }
    snapshot = scene.snapshot()
    assert snapshot["capture_timestamp_seconds"] == 8
    assert snapshot["tracking_completed_timestamp_seconds"] == 3
    assert snapshot["tracking_completed_frame_sequence"] == 90
    assert snapshot["tracking_lag_seconds"] == 5
    assert snapshot["scene"]["timestamp_seconds"] == 8
    person = snapshot["scene"]["entities"][0]
    assert person["observation_state"] == "occluded"
    assert person["last_observed_timestamp_seconds"] == 3
    assert person["native_track"] == NATIVE
    assert scene.scene.snapshot_at(3)["entities"][0]["observation_state"] == "observed"
    assert scene.snapshot()["scene"]["timestamp_seconds"] == 8


def test_capture_and_face_callbacks_do_not_advance_completed_clock_or_plan():
    scene = runtime()
    capture(scene, 240)
    scene.observe_tracks(track(30))
    scene.observe_result(face_result(240))
    assert scene.take_inspections() == ()
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] is None
    assert commit(scene, tuple(track(sequence) for sequence in (0, 30, 60, 90)))
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 3
    # A face ahead of SAM does not prevent delayed body geometry from being used.
    historical = scene.scene.snapshot_at(3, include_history=True)["entities"][0]
    assert historical["bbox"] == BOX.as_dict()
    assert historical["identity"] == {}
    present = scene.snapshot()["scene"]["entities"][0]
    assert present["bbox"] == FACE.as_dict()
    assert present["identity"]["name"] == "Alice"
    assert present["last_observed_timestamp_seconds"] == 8
    jobs = scene.take_inspections()
    assert len(jobs) == 1
    scene.observe_result(face_result(210))
    capture(scene, 270, start_sequence=241)
    assert scene.take_inspections() == ()
    assert scene.snapshot()["tracking_lag_seconds"] == 6


def test_complete_only_once_after_all_thirty_fps_source_observations(monkeypatch):
    scene = runtime()
    capture(scene, 270)
    calls = []
    observe_person = scene.scene.observe_person

    def observe(*args, **kwargs):
        result = observe_person(*args, **kwargs)
        calls.append(kwargs["timestamp_seconds"])
        assert scene.take_inspections() == ()
        return result

    monkeypatch.setattr(scene.scene, "observe_person", observe)
    assert commit(scene, tuple(track(sequence) for sequence in range(121)))
    assert calls == [sequence / 30 for sequence in range(121)]
    person = scene.snapshot()["scene"]["entities"][0]
    assert len(person["history"]) == 121
    jobs = scene.take_inspections()
    assert jobs
    assert len(jobs) == 2
    assert jobs[0].frames[0].timestamp_seconds == 0
    assert jobs[-1].frames[-1].timestamp_seconds == 4
    assert scene.scene.history_size == 64  # Event replay capacity is unchanged.


def test_empty_person_batch_advances_watermark_without_inventing_negative_evidence():
    scene = runtime()
    capture(scene, 300)
    assert commit(scene, (track(0), track(30), track(60), track(90)))
    for job in scene.take_inspections():
        assert scene.complete_inspection(
            job.request.request_id,
            (TemporalVerdict("smoking", ("person-a",), True, 0.9),),
        )
    before = scene.snapshot()["scene"]["events"][0]
    assert commit(
        scene, tuple(track(sequence, ids=()) for sequence in (120, 150, 180, 210))
    )
    snapshot = scene.snapshot()
    assert snapshot["tracking_completed_timestamp_seconds"] == 7
    assert snapshot["tracking_lag_seconds"] == 3
    after = snapshot["scene"]["events"][0]
    assert after["phase"] == before["phase"]
    assert after["negative_observations"] == 0
    assert after["last_evidence_timestamp_seconds"] == 3
    assert after["observation_state"] == "uncertain"
    assert scene.take_inspections() == ()


def test_single_empty_tail_frame_is_a_valid_completion():
    scene = runtime()
    capture(scene, 30)
    assert commit(scene, (track(30, ids=()),))
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 1
    assert scene.snapshot()["scene"]["entities"] == []
    assert scene.snapshot()["scene"]["events"] == []


@pytest.mark.parametrize(
    ("snapshots", "reason"),
    [
        ((track(30), track(0)), "out_of_order"),
        ((track(0), track(0)), "out_of_order"),
        ((track(0), track(30, session="other")), "different_session"),
        ((track(0), track(300)), "source_frame_missing"),
        ((track(0), track(30, at=1.1)), "source_frame_missing"),
        ((track(0), track(30, native={"backend": 7})), "invalid_observation"),
    ],
)
def test_rejected_batches_are_atomic(snapshots, reason):
    scene = runtime()
    capture(scene, 240)
    assert not commit(scene, snapshots)
    snapshot = scene.snapshot()
    assert snapshot["tracking_last_rejection"] == reason
    assert snapshot["tracking_rejections"] == {reason: 1}
    assert snapshot["tracking_completed_timestamp_seconds"] is None
    assert snapshot["scene"]["entities"] == []
    assert snapshot["inspections"] == []
    assert snapshot["stats"]["completed_tracking_batches"] == 0


def test_completion_must_cover_exactly_last_snapshot():
    scene = runtime()
    capture(scene, 60)
    assert not scene.observe_tracking_batch(
        (track(0), track(30)),
        completed_timestamp_seconds=2,
        completed_frame_sequence=60,
    )
    assert scene.snapshot()["tracking_last_rejection"] == "incomplete_watermark"
    assert scene.snapshot()["scene"]["entities"] == []


def test_late_or_previous_session_batch_never_changes_completed_state():
    scene = runtime()
    capture(scene, 240)
    batch = (track(0), track(30), track(60), track(90))
    assert commit(scene, batch)
    before = scene.snapshot()
    assert not commit(scene, batch)
    assert not commit(scene, (track(120, session="previous"),))
    after = scene.snapshot()
    assert after["scene"] == before["scene"]
    assert after["tracking_completed_timestamp_seconds"] == 3
    scene.close()
    assert not commit(scene, (track(120),))
    assert scene.take_inspections() == ()
    closed = scene.snapshot()["scene"]
    assert closed["entities"] == before["scene"]["entities"]
    assert closed["events"] == before["scene"]["events"]
    assert closed["source_pins"] == {"jobs": 0, "observations": 0}


def test_batch_older_than_time_retention_fails_closed_and_can_recover():
    scene = runtime()
    capture(scene, 480)
    assert not commit(scene, (track(0), track(30), track(60), track(90)))
    snapshot = scene.snapshot()
    assert snapshot["tracking_last_rejection"] == "evidence_expired"
    assert snapshot["tracking_completed_timestamp_seconds"] is None
    assert snapshot["scene"]["entities"] == []
    # A newer complete batch is allowed after rejecting the expired one.
    assert commit(scene, (track(150), track(180), track(210), track(240)))
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 8
    assert scene.snapshot()["tracking_lag_seconds"] == 8


@pytest.mark.parametrize("limit", ["frames", "bytes"])
def test_hard_buffer_caps_fail_closed_even_inside_time_retention(limit):
    scene = runtime()
    scene.evidence = SceneEvidenceBuffer(
        max_frames=2 if limit == "frames" else 512,
        max_bytes=12 if limit == "bytes" else 1024,
        history_seconds=12,
    )
    capture(scene, 2)
    assert not commit(scene, (track(0), track(1), track(2)))
    assert scene.snapshot()["tracking_last_rejection"] == "source_frame_missing"
    assert scene.snapshot()["scene"]["entities"] == []
    assert scene.snapshot()["evidence_buffer"][limit] <= (
        2 if limit == "frames" else 12
    )


def test_person_history_has_time_and_count_caps_but_keeps_pinned_source_evidence():
    state = SceneState(
        "s", "camera", person_history_size=512, person_history_seconds=12
    )
    for sequence in range(601):
        assert state.observe_person(
            "person-a",
            timestamp_seconds=sequence / 60,
            frame_sequence=sequence,
            bbox=BOX,
        )
    history = state.snapshot()["entities"][0]["history"]
    assert len(history) == 512
    assert history[0]["frame_sequence"] == 89
    assert state.pin_source_observations("job", (("person-a", 10.0, 600, BOX),))
    state.advance(23)
    assert state.snapshot()["entities"][0]["history"] == []
    assert state.observe_event(
        "smoking",
        ("person-a",),
        timestamp_seconds=10,
        frame_sequence=600,
        detected=True,
        evidence_id="real-pinned-source",
    )
    assert state.snapshot()["entities"][0]["observation_state"] == "occluded"


def test_execution_expiration_releases_jobs_without_waiting_for_next_native_batch(
    monkeypatch,
):
    clock = [100.0]
    monkeypatch.setattr("goodvision_harness.scene_runtime.monotonic", lambda: clock[0])
    scene = runtime()
    capture(scene, 240)
    assert commit(scene, (track(0), track(30), track(60), track(90)))
    jobs = scene.take_inspections()
    assert jobs
    capture(scene, 720, start_sequence=241)
    pending = scene.snapshot()
    assert len(pending["inspections"]) == len(jobs) <= 4
    assert pending["scene"]["source_pins"]["jobs"] == len(jobs)
    assert pending["scene"]["entities"][0]["history"] == []
    assert all(box == BOX for job in jobs for _, _, box in job.subject_boxes)
    assert all(
        frame.jpeg_bytes == f"jpeg-{frame.frame_sequence}".encode()
        for job in jobs
        for frame in job.frames
    )
    assert (
        sum(len(frame.jpeg_bytes) for job in jobs for frame in job.frames)
        <= 16 * 1024 * 1024
    )
    assert pending["tracking_completed_timestamp_seconds"] == 3
    assert pending["tracking_lag_seconds"] == 21
    # Source histories may expire while immutable claimed evidence remains
    # pinned. Only the independent execution deadline retires this native job.
    clock[0] += 20
    capture(scene, 721, start_sequence=721)
    assert scene.snapshot()["inspections"] == []
    assert scene.snapshot()["scene"]["source_pins"]["jobs"] == 0
    assert not scene.complete_inspection(jobs[0].request.request_id, ())
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 3
    assert scene.snapshot()["tracking_lag_seconds"] == pytest.approx(721 / 30 - 3)


def test_native_metadata_whitelist_and_source_projection_are_detached():
    scene = runtime()
    capture(scene, 240)
    metadata = NATIVE | {"private_response": "must not escape"}
    assert commit(scene, (track(0, native=metadata), track(90, native=metadata)))
    snapshot = scene.scene.snapshot_at(3, include_history=True)
    assert snapshot["entities"][0]["native_track"] == NATIVE
    snapshot["entities"][0]["native_track"]["epoch"] = "changed"
    snapshot["entities"][0]["history"][0]["bbox"]["x1"] = 0
    current = scene.snapshot()["scene"]["entities"][0]
    assert current["native_track"] == NATIVE
    assert current["history"][0]["bbox"] == BOX.as_dict()


def test_one_batch_verdict_remains_one_observation_and_unknown_does_not_clear():
    scene = runtime()
    capture(scene, 600)
    # Use current source intervals that are still within the bounded buffer.
    for endpoint, detected in ((300, True), (390, True), (480, None)):
        assert commit(
            scene,
            tuple(
                track(sequence) for sequence in range(endpoint - 60, endpoint + 1, 30)
            ),
        )
        jobs = scene.take_inspections()
        assert jobs
        for job in jobs:
            assert scene.complete_inspection(
                job.request.request_id,
                (TemporalVerdict("smoking", ("person-a",), detected, 0.9),),
            )
        event = scene.snapshot()["scene"]["events"][0]
        if endpoint == 300:
            assert event["phase"] == "suspected"
            assert event["positive_observations"] == 1
        else:
            assert event["phase"] == "confirmed"
            assert event["positive_observations"] == 2
    assert event["negative_observations"] == 0
    assert event["observation_state"] == "uncertain"


@pytest.mark.parametrize("value", ["unknown", None, 5])
def test_unknown_tracking_mode_rejected(value):
    with pytest.raises(ValueError, match="tracking mode"):
        SceneRuntime("s", "camera", tracking_mode=value)


@pytest.mark.parametrize("value", [True, 0, 2, 91, float("nan"), float("inf"), "12"])
def test_tracking_history_limits_are_explicit(value):
    with pytest.raises(ValueError, match="tracking history"):
        runtime(tracking_history_seconds=value)


def test_predicted_observations_are_not_source_person_evidence():
    scene = runtime()
    capture(scene, 240)
    snapshots = []
    for sequence in (30, 60, 90):
        value = track(sequence)
        predicted = replace(
            value.objects[0],
            observation_state="predicted",
            last_observed_timestamp_seconds=0,
            last_observed_frame_sequence=0,
        )
        snapshots.append(replace(value, objects=(predicted,)))
    assert commit(scene, tuple(snapshots))
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 3
    assert scene.snapshot()["scene"]["entities"] == []
    assert scene.take_inspections() == ()


def test_many_native_epochs_retire_expired_people_and_continue_running():
    scene = runtime(event_labels=())
    for batch_index in range(20):
        sequence = batch_index * 90
        scene.observe_frame(f"batch-{batch_index}".encode(), sequence / 30, sequence)
        ids = tuple(f"batch-{batch_index}-person-{index}" for index in range(12))
        assert commit(scene, (track(sequence, ids=ids),))
        assert len(scene.snapshot()["scene"]["entities"]) <= 128
    snapshot = scene.snapshot()
    assert snapshot["tracking_completed_timestamp_seconds"] == 57
    assert snapshot["stats"]["completed_tracking_batches"] == 20
    assert snapshot["stats"]["rejected_tracking_batches"] == 0
    assert snapshot["scene"]["counters"]["expired_batch_entities_retired"] == 112


def test_capacity_rejects_recent_people_then_recovers_after_they_expire():
    scene = runtime(event_labels=())
    scene.observe_frame(b"initial", 0, 0)
    assert commit(scene, (track(0, ids=tuple(f"old-{index}" for index in range(128))),))
    scene.observe_frame(b"too-soon", 1, 30)
    assert not commit(scene, (track(30, ids=("new",)),))
    assert scene.snapshot()["tracking_last_rejection"] == "subject_capacity"
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 0
    scene.observe_frame(b"later", 13, 390)
    assert commit(scene, (track(390, ids=("new",)),))
    snapshot = scene.snapshot()
    assert snapshot["tracking_completed_timestamp_seconds"] == 13
    assert len(snapshot["scene"]["entities"]) == 128


def test_capacity_retirement_preserves_pins_and_active_event_semantics():
    state = SceneState(
        "s",
        "camera",
        max_entities=2,
        person_history_seconds=12,
        person_history_size=512,
    )
    for subject in ("pinned", "active"):
        state.observe_person(subject, timestamp_seconds=0, frame_sequence=0, bbox=BOX)
    assert state.pin_source_observations("job", (("pinned", 0.0, 0, BOX),))
    assert state.observe_event(
        "smoking",
        ("active",),
        timestamp_seconds=0,
        frame_sequence=0,
        detected=True,
        evidence_id="event",
    )
    state.advance(13)
    before = state.snapshot()
    assert not state.reserve_person_capacity(("new",))
    assert state.snapshot() == before
    state.unpin_source_observations("job")
    assert state.reserve_person_capacity(("new",))
    snapshot = state.snapshot()
    assert [person["subject_id"] for person in snapshot["entities"]] == ["active"]
    assert snapshot["events"][0]["phase"] == "suspected"
    assert snapshot["events"][0]["negative_observations"] == 0


def test_capacity_retirement_does_not_evict_people_observed_again_in_same_batch():
    scene = runtime(event_labels=())
    scene.observe_frame(b"initial", 0, 0)
    assert commit(scene, (track(0, ids=tuple(f"old-{index}" for index in range(128))),))
    scene.observe_frame(b"later", 13, 390)
    assert commit(scene, (track(390, ids=("old-0", "new")),))
    people = {
        person["subject_id"]: person for person in scene.snapshot()["scene"]["entities"]
    }
    assert len(people) == 128
    assert people["old-0"]["first_observed_timestamp_seconds"] == 0
    assert people["old-0"]["last_observed_timestamp_seconds"] == 13
    assert people["new"]["last_observed_timestamp_seconds"] == 13


def sampled_batch(
    scene,
    first_sequence=0,
    *,
    count=16,
    subject="person-a",
    epoch="batch-1",
    jpeg=b"source",
):
    snapshots = tuple(
        track(
            sequence,
            at=sequence / 4,
            ids=(subject,),
            native={"backend": BACKEND, "epoch": epoch, "object_id": "7"},
        )
        for sequence in range(first_sequence, first_sequence + count)
    )
    for snapshot in snapshots:
        scene.observe_frame(jpeg, snapshot.timestamp_seconds, snapshot.frame_sequence)
    return snapshots


def test_default_native_batch_two_independent_windows_can_confirm_an_event():
    scene = runtime()
    assert commit(scene, sampled_batch(scene))
    jobs = scene.take_inspections()
    assert len(jobs) == 2
    assert jobs[0].request.end_seconds < jobs[1].request.start_seconds
    assert not {frame.frame_sequence for frame in jobs[0].frames}.intersection(
        frame.frame_sequence for frame in jobs[1].frames
    )
    for index, job in enumerate(jobs):
        assert len(job.frames) == 4
        assert scene.complete_inspection(
            job.request.request_id,
            (TemporalVerdict("smoking", ("person-a",), True, 0.9),),
        )
        event = scene.snapshot()["scene"]["events"][0]
        assert event["phase"] == ("suspected" if index == 0 else "confirmed")
        assert event["positive_observations"] == index + 1
    assert scene.scene.confirm_observations == 2


@pytest.mark.parametrize("count", [4, 7])
def test_short_batch_keeps_one_window_and_does_not_force_confirmation(count):
    scene = runtime()
    assert commit(scene, sampled_batch(scene, count=count))
    jobs = scene.take_inspections()
    assert len(jobs) == 1
    assert scene.complete_inspection(
        jobs[0].request.request_id,
        (TemporalVerdict("smoking", ("person-a",), True, 0.9),),
    )
    assert scene.snapshot()["scene"]["events"][0]["phase"] == "suspected"


def test_pending_windows_are_bounded_and_do_not_overwrite_earlier_intervals():
    scene = runtime()
    for index in range(4):
        assert commit(scene, sampled_batch(scene, index * 16))
    before = scene.snapshot()
    assert len(before["inspections"]) == 4
    assert len(before["pending_tracking_intervals"]) == 4
    ids = {item["request_id"] for item in before["pending_tracking_intervals"]}
    assert commit(scene, sampled_batch(scene, 64))
    snapshot = scene.snapshot()
    assert {
        item["request_id"] for item in snapshot["pending_tracking_intervals"]
    } == ids
    assert snapshot["tracking_interval_rejections"]["queue_capacity"] == 2
    first = scene.take_inspections(limit=1)[0]
    assert scene.complete_inspection(first.request.request_id, ())
    snapshot = scene.snapshot()
    assert len(snapshot["inspections"]) == 4
    assert len(snapshot["pending_tracking_intervals"]) == 3
    assert (
        snapshot["inspections"][-1]["request_id"]
        == before["pending_tracking_intervals"][0]["request_id"]
    )


def test_image_budget_defers_second_window_until_first_completion():
    scene = runtime()
    # The same immutable payload keeps this test small in actual memory while
    # exercising encoded image accounting for four-image inspections.
    jpeg = b"x" * (3 * 1024 * 1024)
    assert commit(scene, sampled_batch(scene, jpeg=jpeg))
    first = scene.take_inspections()
    assert len(first) == 1
    assert len(scene.snapshot()["pending_tracking_intervals"]) == 1
    queued_id = scene.snapshot()["pending_tracking_intervals"][0]["request_id"]
    assert scene.complete_inspection(first[0].request.request_id, ())
    second = scene.take_inspections()
    assert len(second) == 1
    assert second[0].request.request_id == queued_id
    assert scene.snapshot()["pending_tracking_intervals"] == []
    assert sum(len(frame.jpeg_bytes) for frame in second[0].frames) <= 16 * 1024 * 1024


def test_pending_window_expires_when_original_frames_leave_retention():
    scene = runtime()
    for index in range(3):
        assert commit(scene, sampled_batch(scene, index * 16))
    assert len(scene.snapshot()["pending_tracking_intervals"]) == 2
    scene.observe_frame(b"now", 23, 92)
    snapshot = scene.snapshot()
    assert snapshot["pending_tracking_intervals"] == []
    assert (
        snapshot["tracking_interval_rejections"]["source_frame_missing_or_expired"] == 2
    )
    assert snapshot["scene"]["events"] == []


def native_commit(scene, batch, canonicalize):
    return scene.commit_native_tracking_batch(
        batch,
        completed_timestamp_seconds=batch[-1].timestamp_seconds,
        completed_frame_sequence=batch[-1].frame_sequence,
        canonicalize=canonicalize,
    )


def test_native_epoch_change_cannot_continue_previous_event_by_matching_geometry():
    scene = runtime()
    registry = PersonRegistry("s", "camera")

    def canonicalize(snapshots):
        native = snapshots[0].objects[0].attributes["native_track"]
        assert registry.begin_native_epoch(native["backend"], native["epoch"])
        output = []
        for snapshot in snapshots:
            registry.observe_snapshot(snapshot)
            output.append(
                registry.snapshot_at(
                    snapshot.timestamp_seconds, snapshot.frame_sequence
                )
            )
        return tuple(output)

    assert native_commit(scene, sampled_batch(scene), canonicalize)
    for job in scene.take_inspections():
        subject = job.subject_boxes[0][1]
        assert scene.complete_inspection(
            job.request.request_id,
            (TemporalVerdict("smoking", (subject,), True, 0.9),),
        )
    previous = scene.snapshot()["scene"]["events"][0]
    assert previous["phase"] == "confirmed"
    assert native_commit(
        scene, sampled_batch(scene, 16, count=4, epoch="batch-2"), canonicalize
    )
    next_job = scene.take_inspections()[0]
    next_subject = next_job.subject_boxes[0][1]
    assert next_subject not in previous["subject_ids"]
    assert scene.complete_inspection(
        next_job.request.request_id,
        (TemporalVerdict("smoking", (next_subject,), True, 0.9),),
    )
    events = scene.snapshot()["scene"]["events"]
    assert len(events) == 2
    assert events[0]["event_id"] == previous["event_id"]
    assert events[1]["event_id"] != previous["event_id"]
    assert events[1]["phase"] == "suspected"


@pytest.mark.parametrize(
    "reason",
    [
        "expired",
        "missing",
        "session",
        "metadata",
        "capacity",
        "backend",
        "extra_native_field",
    ],
)
def test_native_preflight_rejection_never_mutates_registry(reason):
    scene = runtime(event_labels=())
    batch = sampled_batch(scene)
    if reason == "expired":
        scene.observe_frame(b"later", 30, 120)
    elif reason == "missing":
        scene.evidence.reset()
    elif reason == "session":
        batch = tuple(replace(item, stream_session_id="previous") for item in batch)
    elif reason == "metadata":
        batch = tuple(
            replace(item, objects=(replace(item.objects[0], attributes={}),))
            for item in batch
        )
    elif reason in {"backend", "extra_native_field"}:
        batch = tuple(
            replace(
                item,
                objects=(
                    replace(
                        item.objects[0],
                        attributes={
                            "native_track": dict(
                                item.objects[0].attributes["native_track"]
                            )
                            | (
                                {"backend": "image-detector"}
                                if reason == "backend"
                                else {"unknown": "field"}
                            )
                        },
                    ),
                ),
            )
            for item in batch
        )
    else:
        for index in range(128):
            scene.scene.observe_person(
                f"existing-{index}", timestamp_seconds=3.75, frame_sequence=15, bbox=BOX
            )
    calls = []

    def canonicalize(snapshots):
        calls.append(snapshots)
        return snapshots

    assert not native_commit(scene, batch, canonicalize)
    assert calls == []
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] is None


def test_native_preflight_holds_source_retention_until_mapping_commits():
    scene = runtime()
    batch = sampled_batch(scene)
    entered, release, capture_attempted, captured = Event(), Event(), Event(), Event()

    def canonicalize(snapshots):
        entered.set()
        assert release.wait(2)
        return snapshots

    def capture_ahead():
        capture_attempted.set()
        scene.observe_frame(b"much-later", 30, 120)
        captured.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        mapping = pool.submit(native_commit, scene, batch, canonicalize)
        assert entered.wait(2)
        capturing = pool.submit(capture_ahead)
        assert capture_attempted.wait(2)
        assert not captured.wait(0.05)
        release.set()
        assert mapping.result(timeout=2)
        capturing.result(timeout=2)
    assert scene.snapshot()["tracking_completed_timestamp_seconds"] == 3.75
    assert scene.snapshot()["capture_timestamp_seconds"] == 30
