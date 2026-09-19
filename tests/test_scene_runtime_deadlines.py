"""Native delay must not consume the bounded temporal model execution budget."""

from goodvision_harness import scene_runtime
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict


def delayed_runtime(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(scene_runtime, "monotonic", lambda: clock[0])
    runtime = SceneRuntime(
        "session",
        "camera",
        mode="active",
        tracking_mode="batch",
        native_memory_scope="session",
        tracking_history_seconds=32,
        event_labels=("smoking",),
    )
    snapshots = []
    for sequence in range(16):
        at = sequence / 4
        runtime.observe_frame(b"source", at, sequence)
        snapshots.append(
            RichLabelSnapshot(
                "session",
                sequence,
                at,
                (
                    RichLabelObject(
                        "same-person", "person", NormalizedBBox(0.1, 0.1, 0.4, 0.9)
                    ),
                ),
            )
        )
    runtime.observe_frame(b"current", 18.45, 100, retain_source=False)
    assert runtime.observe_tracking_batch(
        tuple(snapshots),
        completed_frame_sequence=15,
        completed_timestamp_seconds=3.75,
    )
    assert runtime.snapshot()["tracking_lag_seconds"] == 14.7
    return runtime, clock


def test_native_delay_and_two_fifteen_second_verdicts_preserve_source_freshness(
    monkeypatch,
):
    runtime, clock = delayed_runtime(monkeypatch)
    verdict = TemporalVerdict("smoking", ("same-person",), True, 0.9)
    (first,) = runtime.take_inspections(limit=1)
    clock[0] += 15
    runtime.observe_frame(b"current", 33.45, 101, retain_source=False)
    assert runtime.complete_inspection(first.request.request_id, (verdict,))
    assert runtime.snapshot()["scene"]["events"][0]["phase"] == "suspected"

    (second,) = runtime.take_inspections(limit=1)
    assert first.frames[-1].frame_sequence < second.frames[0].frame_sequence
    clock[0] += 15
    runtime.observe_frame(b"current", 48.45, 102, retain_source=False)
    assert runtime.complete_inspection(second.request.request_id, (verdict,))
    snapshot = runtime.snapshot()
    (event,) = snapshot["scene"]["events"]
    assert event["phase"] == "confirmed"
    assert event["positive_observations"] == 2
    assert event["last_evidence_timestamp_seconds"] == 3.75
    assert event["observation_state"] == "uncertain"
    assert snapshot["scene"]["entities"][0]["observation_state"] == "occluded"
    assert snapshot["capture_timestamp_seconds"] == 48.45
    assert snapshot["tracking_completed_timestamp_seconds"] == 3.75
    assert snapshot["tracking_lag_seconds"] == 44.7
    assert snapshot["stats"]["completed_inspections"] == 2
    assert snapshot["stats"]["accepted_temporal_verdicts"] == 2
    assert snapshot["inspection_expiry_clock"] == "monotonic_execution"
    assert snapshot["scene"]["source_pins"] == {"jobs": 0, "observations": 0}
    assert not runtime._inspection_deadlines


def test_unclaimed_native_jobs_expire_without_capture_advancing(monkeypatch):
    runtime, clock = delayed_runtime(monkeypatch)
    clock[0] += 20
    assert runtime.take_inspections() == ()
    snapshot = runtime.snapshot()
    assert snapshot["stats"]["expired_inspections"] == 2
    assert snapshot["capture_timestamp_seconds"] == 18.45
    assert snapshot["scene"]["source_pins"]["jobs"] == 0
    assert not runtime._inspection_deadlines


def test_repeated_claim_does_not_extend_deadline_or_resurrect_expired_reply(
    monkeypatch,
):
    runtime, clock = delayed_runtime(monkeypatch)
    jobs = runtime.take_inspections()
    assert len(jobs) == 2
    clock[0] += 19
    assert not runtime.take_inspections()
    assert all(runtime.inspection_active(job.request.request_id) for job in jobs)
    clock[0] += 1
    for job in jobs:
        assert not runtime.complete_inspection(
            job.request.request_id,
            (TemporalVerdict("smoking", ("same-person",), True, 0.9),),
        )
    assert runtime.snapshot()["stats"]["expired_inspections"] == 2
    assert not runtime.scene.snapshot()["events"]
    assert runtime.latest_inspection() is None
    assert not runtime._inspection_deadlines


def test_native_close_clears_deadlines_and_pending_claims(monkeypatch):
    runtime, _ = delayed_runtime(monkeypatch)
    jobs = runtime.take_inspections()
    assert runtime._inspection_deadlines
    runtime.close()
    assert not runtime._inspection_deadlines
    assert not runtime.take_inspections()
    assert all(not runtime.inspection_active(job.request.request_id) for job in jobs)


def test_immediate_mode_still_expires_by_source_time(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(scene_runtime, "monotonic", lambda: clock[0])
    runtime = SceneRuntime("session", "camera")
    for sequence, at in enumerate((0.0, 1.0)):
        runtime.observe_tracks(
            RichLabelSnapshot(
                "session",
                sequence,
                at,
                (
                    RichLabelObject(
                        "person", "person", NormalizedBBox(0.1, 0.1, 0.4, 0.9)
                    ),
                ),
            )
        )
        runtime.observe_frame(b"source", at, sequence)
    jobs = runtime.take_inspections()
    assert jobs
    clock[0] += 1000
    assert all(runtime.inspection_active(job.request.request_id) for job in jobs)
    runtime.observe_frame(b"source", 21, 100)
    assert all(not runtime.inspection_active(job.request.request_id) for job in jobs)
    assert runtime.snapshot()["inspection_expiry_clock"] == "source_time"
