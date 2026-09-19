"""Completed empty SAM frames can close safety episodes without VLM requests."""

from dataclasses import replace

from goodvision_harness.camera_display import CameraDisplayMetadata
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict


def submit(runtime, sequences, people=("a",), *, session="s"):
    snapshots = []
    for sequence in sequences:
        runtime.observe_frame(b"jpeg", float(sequence), sequence)
        snapshots.append(RichLabelSnapshot(
            session, sequence, float(sequence), tuple(
                RichLabelObject(
                    sid, "person", NormalizedBBox(.1, .1, .4, .9),
                    attributes={"native_track": {
                        "backend": "sam31_stream", "epoch": "epoch", "object_id": sid,
                    }},
                )
                for sid in people
            ),
        ))
    return runtime.observe_tracking_batch(
        tuple(snapshots),
        completed_timestamp_seconds=snapshots[-1].timestamp_seconds,
        completed_frame_sequence=snapshots[-1].frame_sequence,
    )


def active_runtime():
    runtime = SceneRuntime("s", "camera", tracking_mode="batch")
    for start in (0, 2):
        assert submit(runtime, (start, start + 1))
        job, = runtime.take_inspections()
        assert runtime.complete_inspection(job.request.request_id, [
            TemporalVerdict("smoking", ("a",), True),
        ])
    assert runtime.scene.snapshot()["events"][0]["phase"] == "confirmed"
    return runtime


def test_empty_sam_results_close_at_tenth_frame_without_scheduling_vlm():
    runtime = active_runtime()
    for sequence in range(4, 13):
        assert submit(runtime, (sequence,), ())
        assert not runtime.take_inspections()
        assert runtime.scene.snapshot()["events"][0]["phase"] == "confirmed"
    assert submit(runtime, (13,), ())
    assert not runtime.take_inspections()
    event, = runtime.scene.snapshot()["events"]
    assert event["phase"] == "ended"
    assert event["ended_since_timestamp_seconds"] == 13
    assert event["ended_reason"] == "all_subjects_absent"
    assert event["negative_observations"] == 0
    assert event["last_positive_timestamp_seconds"] == 3
    assert len([t for t in runtime.scene.snapshot()["transitions"]
                if t["type"] == "end"]) == 1


def test_successful_sources_count_frames_not_sequence_gaps_or_batches():
    runtime = active_runtime()
    assert submit(runtime, tuple(range(4, 13)), ())
    assert runtime.scene.snapshot()["events"][0]["phase"] == "confirmed"
    # Sampling can skip many camera frame IDs. This is one result, not 18 votes.
    assert submit(runtime, (30,), ())
    event, = runtime.scene.snapshot()["events"]
    assert event["phase"] == "ended"
    assert event["ended_since_timestamp_seconds"] == 30


def test_rejected_duplicate_foreign_or_invalid_results_do_not_count_absence():
    runtime = active_runtime()
    assert submit(runtime, tuple(range(4, 13)), ())
    assert not submit(runtime, (12,), ())
    assert not submit(runtime, (13,), (), session="old-run")
    invalid = RichLabelSnapshot("s", 13, 13., (
        RichLabelObject("a", "person", NormalizedBBox(.1, .1, .4, .9)),
    ))
    invalid = replace(invalid, objects=(replace(
        invalid.objects[0], attributes={"tracking_confirmation": "invalid"},
    ),))
    assert not runtime.observe_tracking_batch(
        (invalid,), completed_timestamp_seconds=13, completed_frame_sequence=13,
    )
    # Capture can continue while a request is failing/in flight.
    runtime.observe_frame(b"jpeg", 14., 14)
    assert runtime.scene.snapshot()["events"][0]["phase"] == "confirmed"
    assert submit(runtime, (14,), ())
    assert runtime.scene.snapshot()["events"][0]["phase"] == "ended"


def test_empty_sam_frames_can_arrive_before_the_first_positive_vlm_result():
    runtime = SceneRuntime("s", "camera", tracking_mode="batch")
    assert submit(runtime, (0, 1))
    job, = runtime.take_inspections()
    assert submit(runtime, tuple(range(2, 12)), ())
    assert not runtime.scene.snapshot()["events"]
    assert runtime.complete_inspection(job.request.request_id, [
        TemporalVerdict("smoking", ("a",), True),
    ])
    event, = runtime.scene.snapshot()["events"]
    assert event["phase"] == "ended"
    assert event["ended_since_timestamp_seconds"] == 11
    assert event["negative_observations"] == 0


def test_absence_end_reaches_display_color_interval_and_persistent_event(tmp_path):
    runtime = active_runtime()
    metadata = CameraDisplayMetadata()
    metadata.reset("s")
    assert metadata.accept_source({
        "stream_session_id": "s", "generation": 1, "crop_version": 0,
        "source_sequence": 0, "media_timestamp": 0., "width": 640, "height": 480,
    })
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    try:
        assert archive.start_session("s", "camera")
        initial = {"scene": runtime.scene.snapshot(include_history=False)}
        metadata.observe_scene(initial)
        assert archive.submit_scene(initial)
        assert archive.flush()
        assert metadata.snapshot()["events"][0]["valid_until_timestamp"] is None
        assert archive.list_events()["events"][0]["state"] == "active"
        assert submit(runtime, tuple(range(4, 14)), ())
        ended = {"scene": runtime.scene.snapshot(include_history=False)}
        metadata.observe_scene(ended)
        assert archive.submit_scene(ended)
        assert archive.flush()
        interval, = metadata.snapshot()["events"]
        assert interval["valid_until_timestamp"] == 13
        assert interval["event_state"] == "ended"
        archived, = archive.list_events()["events"]
        assert archived["state"] == "ended"
        assert archived["end_seconds"] == 13
    finally:
        archive.close()


def test_renaming_subject_preserves_compacted_eight_frame_absence_streak():
    from goodvision_harness.scene_state import SceneState

    scene = SceneState("s", "camera", person_history_size=4)
    for sequence in (1, 2):
        assert scene.observe_person(
            "a", timestamp_seconds=sequence, frame_sequence=sequence,
            bbox=NormalizedBBox(.1, .1, .4, .9),
        )
        assert scene.observe_tracking_presence(
            ("a",), timestamp_seconds=sequence, frame_sequence=sequence,
        )
        assert scene.observe_event(
            "smoking", ("a",), timestamp_seconds=sequence, frame_sequence=sequence,
            detected=True, evidence_id=str(sequence),
        )
    for sequence in range(3, 11):
        assert scene.observe_tracking_presence(
            (), timestamp_seconds=sequence, frame_sequence=sequence,
        )
    assert scene.merge_subject("a", "canonical", reviewed_identity=True)
    for sequence in (11, 12):
        assert scene.observe_tracking_presence(
            (), timestamp_seconds=sequence, frame_sequence=sequence,
        )
        event, = scene.snapshot()["events"]
        assert event["phase"] == ("confirmed" if sequence == 11 else "ended")
    assert event["ended_since_timestamp_seconds"] == 12
