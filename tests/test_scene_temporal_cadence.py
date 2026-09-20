"""Real source-time jobs remain intact across batches, overload and expiry."""

import json
from pathlib import Path

import pytest

from goodvision_harness import scene_runtime
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict
from goodvision_harness.scene_state import SceneState
from goodvision_harness.video_tracking import BACKEND, VideoFrame

BOX = NormalizedBBox(0.1, 0.1, 0.6, 0.9)


def runtime(**kwargs):
    return SceneRuntime(
        "session", "camera", tracking_mode="batch", native_memory_scope="session",
        event_labels=("smoking",), temporal_observation_fps=1.25,
        max_pending_inspections=kwargs.pop("max_pending_inspections", 8), **kwargs,
    )


def source(subject="a", *, sequence, at, epoch="first", confirmation="confirmed"):
    return RichLabelSnapshot("session", sequence, at, (
        RichLabelObject(subject, "person", BOX, attributes={
            "native_track": {"backend": BACKEND, "epoch": epoch, "object_id": subject},
            "tracking_confirmation": confirmation,
        }),
    ))


def feed(scene, sequences, *, epoch="first"):
    snapshots = []
    for sequence in sequences:
        at = sequence / 3
        scene.observe_frame(b"source" + bytes([sequence]), at, sequence)
        snapshots.append(source(sequence=sequence, at=at, epoch=epoch))
    assert scene.observe_tracking_batch(
        snapshots, completed_frame_sequence=snapshots[-1].frame_sequence,
        completed_timestamp_seconds=snapshots[-1].timestamp_seconds,
    )


def test_small_sam_batches_supply_distinct_time_windows_across_batch_edges():
    scene = runtime()
    for sequence in range(13):
        feed(scene, (sequence,))
    jobs = scene.take_inspections()
    assert len(jobs) == 4
    assert [job.frames[-1].frame_sequence for job in jobs] == [3, 5, 8, 10]
    assert all(2 <= len(job.frames) <= 4 for job in jobs)
    assert all(
        job.frames[-1].timestamp_seconds - job.frames[0].timestamp_seconds >= 0.75
        for job in jobs
    )
    assert jobs[1].frames[0].frame_sequence <= jobs[0].frames[-1].frame_sequence
    assert jobs[1].evidence_after_frame_sequence == jobs[0].frames[-1].frame_sequence
    assert scene.snapshot()["stats"]["rejected_tracking_intervals"] == 0


@pytest.mark.parametrize("sample_interval", (0.336, 1.001))
def test_quantized_source_interval_keeps_real_predecessor_for_minimum_span(
    sample_interval,
):
    scene = runtime()
    endpoints = []
    for sequence in range(16):
        at = sequence * sample_interval
        scene.observe_frame(b"original", at, sequence)
        assert scene.observe_tracking_batch(
            (source(sequence=sequence, at=at),),
            completed_frame_sequence=sequence, completed_timestamp_seconds=at,
        )
        for job in scene.take_inspections():
            endpoints.append(job.frames[-1].frame_sequence)
            assert 2 <= len(job.frames) <= 4
            span = job.frames[-1].timestamp_seconds - job.frames[0].timestamp_seconds
            assert span >= .75
            assert scene.complete_inspection(job.request.request_id, ())
    # A one-second hard crop previously admitted no window at either rate:
    # Four .336 s-spaced frames span 1.008 s; two 1 FPS frames span 1.001 s.
    assert len(endpoints) >= 6
    assert endpoints == sorted(set(endpoints))


def test_temporal_predecessor_does_not_bridge_unbounded_source_outage():
    scene = runtime()
    for sequence, at in enumerate((0, 10, 10.4, 10.8)):
        scene.observe_frame(b"original", at, sequence)
        assert scene.observe_tracking_batch(
            (source(sequence=sequence, at=at),),
            completed_frame_sequence=sequence, completed_timestamp_seconds=at,
        )
        if sequence < 3:
            assert scene.take_inspections() == ()
    job, = scene.take_inspections()
    assert [frame.timestamp_seconds for frame in job.frames] == [10, 10.4, 10.8]


def test_recorded_source_timing_supplies_cadence_without_boundary_starvation():
    fixture = json.loads((
        Path(__file__).parent / "fixtures" / "temporal_cadence_quantized_20260917.json"
    ).read_text(encoding="utf-8"))
    timestamps = dict(fixture["sources"])
    batches = fixture["sam_output_batches"]
    provisional = set(fixture["provisional_sequences"])
    assert len(timestamps) == 679
    assert sum(map(len, batches)) == 515
    assert len(fixture["previous_inspection_endpoints"]) == 148
    scene = runtime()
    jobs = []

    def complete_available():
        while available := scene.take_inspections():
            for job in available:
                jobs.append(job)
                assert scene.complete_inspection(job.request.request_id, ())

    for batch in batches:
        snapshots = []
        for sequence in batch:
            at = timestamps[sequence]
            scene.observe_frame(b"original", at, sequence)
            snapshots.append(source(
                sequence=sequence, at=at,
                confirmation="provisional" if sequence in provisional else "confirmed",
            ))
        if not snapshots:
            continue
        assert scene.observe_tracking_batch(
            snapshots, completed_frame_sequence=batch[-1],
            completed_timestamp_seconds=timestamps[batch[-1]],
        )
        complete_available()
    scene.finish_inspection_sources()
    complete_available()

    # Independent source-clock oracle: choose the first confirmed real endpoint
    # on/after each target. Never duplicate a frame to fill a missed clock tick.
    sources = [sequence for batch in batches for sequence in batch
               if sequence not in provisional]
    next_target = timestamps[sources[0]] + .75
    expected = []
    for sequence in sources:
        if timestamps[sequence] + 1e-9 >= next_target:
            expected.append(sequence)
            while next_target <= timestamps[sequence] + 1e-9:
                next_target += .8
    if expected[-1] != sources[-1]:
        expected.append(sources[-1])
    endpoints = [job.frames[-1].frame_sequence for job in jobs]
    assert endpoints == expected
    assert len(endpoints) == 283
    assert endpoints == sorted(set(endpoints))
    previous = -1
    for job in jobs:
        assert job.evidence_after_frame_sequence == previous
        previous = job.frames[-1].frame_sequence
        assert 2 <= len(job.frames) <= 4
        span = job.frames[-1].timestamp_seconds - job.frames[0].timestamp_seconds
        assert .75 <= span <= 1.75
        assert all(timestamps[frame.frame_sequence] == frame.timestamp_seconds
                   for frame in job.frames)
    assert scene.snapshot()["stats"]["rejected_tracking_intervals"] == 0


def test_overlap_does_not_reuse_previous_source_as_second_positive():
    scene = runtime()
    feed(scene, range(6))
    first, second = scene.take_inspections()
    verdict = TemporalVerdict("smoking", ("a",), True, 0.9)
    assert scene.complete_inspection(first.request.request_id, (verdict,))
    assert scene.snapshot()["scene"]["events"][0]["phase"] == "suspected"
    assert not scene.complete_inspection(first.request.request_id, (verdict,))
    assert scene.complete_inspection(second.request.request_id, (verdict,))
    event, = scene.snapshot()["scene"]["events"]
    assert event["phase"] == "confirmed"
    assert event["positive_observations"] == 2
    assert len({item["frame_sequence"] for item in event["history"]}) == 2


def test_full_job_pool_keeps_cursor_and_pending_sources_after_ring_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(scene_runtime, "monotonic", lambda: clock[0])
    scene = runtime(max_pending_inspections=1, tracking_history_seconds=3)
    feed(scene, range(10))
    before = scene.snapshot()
    assert before["temporal_last_endpoint_sequence"] == 3
    assert len(before["inspections"]) == 1
    scene.observe_frame(b"latest", 120, 200, retain_source=False)
    clock[0] += 100
    assert not len(scene.evidence)
    first, = scene.take_inspections()
    assert first.frames[0].jpeg_bytes == b"source\0"
    assert scene.complete_inspection(first.request.request_id, (), success=False)
    second, = scene.take_inspections()
    assert second.frames[-1].frame_sequence == 5
    assert second.frames[0].jpeg_bytes == b"source\x02"
    assert scene.snapshot()["stats"]["expired_inspections"] == 0
    assert scene.snapshot()["scene"]["entities"]
    scene.close()
    assert scene.snapshot()["scene"]["source_pins"]["jobs"] == 0


def test_capacity_rejection_is_atomic_and_does_not_consume_pending_endpoint():
    scene = runtime(max_pending_inspections=1, inspection_image_budget_bytes=34)
    feed(scene, range(4))
    before = scene.snapshot()
    scene.observe_frame(b"source\x04", 4 / 3, 4)
    batch = (source(sequence=4, at=4 / 3),)
    assert not scene.observe_tracking_batch(
        batch, completed_frame_sequence=4, completed_timestamp_seconds=4 / 3,
    )
    after = scene.snapshot()
    assert scene.tracking_last_rejection == "inspection_capacity"
    assert after["tracking_completed_frame_sequence"] == 3
    assert after["temporal_last_endpoint_sequence"] == (
        before["temporal_last_endpoint_sequence"]
    )
    assert after["inspections"] == before["inspections"]
    assert all(entity["frame_sequence"] == 3 for entity in after["scene"]["entities"])


def test_epoch_change_never_creates_mixed_source_job_and_stop_flushes_tail():
    scene = runtime()
    feed(scene, range(5), epoch="old")
    feed(scene, range(5, 10), epoch="new")
    scene.finish_inspection_sources()
    jobs = scene.take_inspections()
    assert jobs
    for job in jobs:
        sequences = [frame.frame_sequence for frame in job.frames]
        assert max(sequences) < 5 or min(sequences) >= 5
    assert jobs[-1].frames[-1].frame_sequence == 9


def test_source_pin_can_be_promoted_after_history_expiry_but_not_fabricated():
    scene = SceneState(
        "session", "camera", person_history_seconds=3, retain_pinned_sources=True,
    )
    assert scene.observe_person("a", timestamp_seconds=0, frame_sequence=0, bbox=BOX)
    observation = (("a", 0, 0, BOX),)
    assert scene.pin_source_observations("source", observation)
    scene.advance(100)
    assert scene.snapshot()["entities"][0]["history"] == []
    assert scene.pin_source_observations("job", observation)
    assert not scene.pin_source_observations("fake", (("a", 1, 0, BOX),))
    scene.unpin_source_observations("source")
    scene.advance(101)
    assert scene.snapshot()["entities"]
    scene.unpin_source_observations("job")
    scene.advance(102)
    assert scene.snapshot()["entities"] == []


def test_compact_runtime_snapshot_does_not_construct_person_event_history():
    scene = runtime()
    feed(scene, range(4))
    full = scene.snapshot()
    compact = scene.snapshot(include_history=False)
    assert "history" in full["scene"]["entities"][0]
    assert "history" not in compact["scene"]["entities"][0]


def test_native_retry_uses_exact_retained_originals_without_moving_capture_clock():
    scene = runtime()
    scene.evidence.max_frames = 1
    snapshots, retained = [], []
    for sequence in range(4):
        at, jpeg = sequence / 3, b"source" + bytes([sequence])
        scene.observe_frame(jpeg, at, sequence)
        snapshots.append(source(sequence=sequence, at=at))
        retained.append(VideoFrame(sequence, at, jpeg))
    assert scene.commit_native_tracking_batch(
        snapshots, completed_frame_sequence=3, completed_timestamp_seconds=1,
        canonicalize=tuple, retained_sources=retained,
    )
    job, = scene.take_inspections()
    assert job.frames[0].jpeg_bytes == b"source\0"
    assert scene.snapshot()["capture_timestamp_seconds"] == 1
    assert not scene._retained_tracking_sources


def test_native_capacity_preflight_does_not_call_identity_canonicalizer():
    scene = runtime(max_pending_inspections=1, inspection_image_budget_bytes=34)
    feed(scene, range(4))
    scene.observe_frame(b"source\x04", 4 / 3, 4)

    def forbidden(_snapshots):
        raise AssertionError("identity allocation ran before capacity preflight")

    assert not scene.commit_native_tracking_batch(
        (source(sequence=4, at=4 / 3),),
        completed_frame_sequence=4, completed_timestamp_seconds=4 / 3,
        canonicalize=forbidden,
        retained_sources=(VideoFrame(4, 4 / 3, b"source\x04"),),
    )
    assert scene.tracking_last_rejection == "inspection_capacity"
    assert not scene._retained_tracking_sources


def test_retained_source_fallback_rejects_changed_pixels_and_overage():
    scene = runtime(tracking_history_seconds=3)
    scene.observe_frame(b"original", 0, 0)
    batch = (source(sequence=0, at=0),)
    with pytest.raises(ValueError, match="changed original"):
        scene.commit_native_tracking_batch(
            batch, completed_frame_sequence=0, completed_timestamp_seconds=0,
            canonicalize=tuple, retained_sources=(VideoFrame(0, 0, b"altered"),),
        )
    scene.observe_frame(b"current", 10, 100, retain_source=False)
    assert not scene.commit_native_tracking_batch(
        batch, completed_frame_sequence=0, completed_timestamp_seconds=0,
        canonicalize=tuple, retained_sources=(VideoFrame(0, 0, b"original"),),
    )
    assert scene.tracking_last_rejection == "evidence_expired"
    assert scene.snapshot()["scene"]["entities"] == []


def test_compact_event_color_timestamps_survive_history_pruning():
    scene = SceneState("session", "camera", history_size=2, clear_observations=2)
    for sequence in range(10):
        scene.observe_person("a", timestamp_seconds=sequence, frame_sequence=sequence,
                             bbox=BOX)
        scene.observe_event(
            "smoking", ("a",), timestamp_seconds=sequence, frame_sequence=sequence,
            detected=sequence < 6, evidence_id=f"event-{sequence}",
        )
    event, = scene.snapshot(include_history=False)["events"]
    assert "history" not in event
    assert event["phase"] == "ended"
    assert event["confirmed_since_timestamp_seconds"] == 1
    assert event["last_positive_timestamp_seconds"] == 5
    assert event["ended_since_timestamp_seconds"] == 7


@pytest.mark.parametrize("kwargs", [
    {"temporal_observation_fps": True}, {"temporal_observation_fps": 0},
    {"temporal_observation_fps": 4}, {"max_pending_inspections": 33},
    {"inspection_image_budget_bytes": 0},
])
def test_temporal_capacity_validation(kwargs):
    with pytest.raises(ValueError):
        SceneRuntime("session", "camera", **kwargs)
