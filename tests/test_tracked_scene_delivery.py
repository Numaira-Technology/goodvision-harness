"""Delayed native display uses exact retained sources without changing state."""

import copy
import io
import json
import logging
import time
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image

from goodvision_harness import scene_delivery
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.result_bus import FramePacket
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_delivery import TrackedSceneDisplayWorker
from goodvision_harness.scene_runtime import SceneRuntime


def jpeg(color):
    buffer = io.BytesIO()
    with Image.new("RGB", (360, 240), color) as image:
        image.save(buffer, "JPEG")
    return buffer.getvalue()


def packet(sequence, timestamp, color="black"):
    return FramePacket(jpeg(color), timestamp, sequence, f"source-observed-{sequence}")


def runtime():
    return SceneRuntime(
        "s",
        "camera",
        mode="active",
        tracking_mode="batch",
        tracking_history_seconds=32,
        native_memory_scope="session",
        event_labels=(),
    )


def capture(scene, worker, frame, *, admitted=True):
    scene.observe_frame(
        frame.jpeg,
        frame.media_timestamp,
        frame.frame_sequence,
        force=True,
        retain_source=admitted,
    )
    worker.submit(frame, retain_source=admitted)


def tracked(scene, frames, *, people=True):
    snapshots = tuple(
        RichLabelSnapshot(
            "s",
            frame.frame_sequence,
            frame.media_timestamp,
            (
                RichLabelObject(
                    "person",
                    "person",
                    NormalizedBBox(0.1, 0.2, 0.7, 0.9),
                    attributes={
                        "native_track": {
                            "backend": "sam3.1_video",
                            "epoch": "epoch",
                            "object_id": "1",
                        }
                    },
                ),
            )
            if people
            else (),
        )
        for frame in frames
    )
    assert scene.observe_tracking_batch(
        snapshots,
        completed_timestamp_seconds=frames[-1].media_timestamp,
        completed_frame_sequence=frames[-1].frame_sequence,
    )


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("tracked display did not reach expected state")
        time.sleep(0.005)


def test_zero_minimum_still_waits_tracking_and_draws_exact_source_not_live_frame(
    monkeypatch,
):
    scene, published, renders = runtime(), [], []
    original = scene_delivery.render_scene_jpeg

    def render(source, snapshot, at, **kwargs):
        renders.append((source, copy.deepcopy(snapshot), at))
        return original(source, snapshot, at, **kwargs)

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    worker.start()
    try:
        source = packet(3, 1.0, "red")
        capture(scene, worker, source)
        capture(scene, worker, packet(100, 15.7, "blue"), admitted=False)
        wait_for(lambda: worker.stats()["unavailable_sources"] > 0)
        assert not published and worker.stats()["state"] == "buffering"
        tracked(scene, (source,))
        before = scene.snapshot()
        worker.notify_tracking()
        wait_for(lambda: len(published) == 1)
        result = published[0]
        assert (result.frame_sequence, result.media_timestamp) == (3, 1.0)
        assert result.observed_at == source.observed_at
        assert result.annotation_mode == "exact"
        assert result.stream_delay_seconds >= 14.7
        assert renders[0][0] is source.jpeg and renders[0][2] == 1.0
        assert renders[0][1]["entities"][0]["last_observed_timestamp_seconds"] == 1.0
        assert scene.snapshot() == before
        assert worker.stats()["pending_bytes"] == 0
        assert worker.wait_until_idle()
    finally:
        worker.close()
        scene.close()


def test_user_minimum_delay_waits_even_if_tracking_finished_and_capture_ended(
    monkeypatch,
):
    now = [100.0]
    monkeypatch.setattr(
        scene_delivery, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(
        scene, published.append, delay_seconds=5, max_fps=60
    )
    source = packet(1, 0.0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    worker.start()
    try:
        now[0] = 104.9
        worker.notify_tracking()
        time.sleep(0.03)
        assert not published
        now[0] = 105.0
        worker.notify_tracking()
        wait_for(lambda: len(published) == 1)
        assert published[0].stream_delay_seconds == 5
        assert published[0].frame_sequence == 1
        assert scene.snapshot()["capture_timestamp_seconds"] == 0
    finally:
        worker.close()
        scene.close()


def test_changed_minimum_delay_applies_to_waiting_sources(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(
        scene_delivery, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(
        scene, published.append, delay_seconds=5, max_fps=60
    )
    source = packet(1, 0.0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    worker.start()
    try:
        now[0] = 11
        worker.set_delay_seconds(2)
        time.sleep(0.03)
        assert not published
        now[0] = 12
        worker.notify_tracking()
        wait_for(lambda: len(published) == 1)
        assert worker.stats()["buffer_delay_seconds"] == 2
    finally:
        worker.close()
        scene.close()


def test_empty_completed_source_is_displayed_without_borrowing_old_person(monkeypatch):
    scene, published, seen = runtime(), [], []
    original = scene_delivery.render_scene_jpeg

    def render(source, snapshot, at, **kwargs):
        seen.append(scene_delivery._people(snapshot, at))
        return original(source, snapshot, at, **kwargs)

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    worker.start()
    try:
        first, empty = packet(1, 0.0, "red"), packet(2, 0.25, "green")
        capture(scene, worker, first)
        tracked(scene, (first,))
        worker.notify_tracking()
        wait_for(lambda: len(published) == 1)
        capture(scene, worker, empty)
        tracked(scene, (empty,), people=False)
        worker.notify_tracking()
        wait_for(lambda: len(published) == 2)
        assert published[0].annotation_mode == "exact"
        assert published[1].annotation_mode == "none"
        assert seen[0] and seen[1] == []
        assert published[1].frame_sequence == 2
        worker.notify_tracking()
        time.sleep(0.04)
        assert len(published) == 2  # No replay or fictitious future source.
    finally:
        worker.close()
        scene.close()


def test_expired_exact_source_does_not_fall_back_to_live_jpeg():
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    source = packet(1, 0.0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    capture(scene, worker, packet(200, 40, "blue"), admitted=False)
    worker.start()
    try:
        wait_for(lambda: worker.stats()["unavailable_sources"] > 0)
        assert not published
        assert worker.wait_until_idle(0.02)
        assert worker.stats()["unavailable_source_frames"] == 1
        assert worker.stats()["state"] == "buffering"
    finally:
        worker.close()
        scene.close()


def test_only_admitted_source_metadata_is_retained_and_has_a_hard_cap():
    scene = runtime()
    worker = TrackedSceneDisplayWorker(scene, lambda frame: None)
    source = packet(0, 0)
    for sequence in range(2000):
        worker.submit(
            FramePacket(source.jpeg, sequence / 30, sequence, "time"),
            retain_source=sequence % 4 == 0,
        )
    assert worker.stats()["pending_frames"] == 500
    for sequence in range(2000, 2020):
        worker.submit(FramePacket(source.jpeg, sequence / 30, sequence, "time"))
    assert (
        worker.stats()["pending_frames"] == worker.stats()["max_pending_frames"] == 512
    )
    assert worker.stats()["source_stamps_dropped"] == 8
    assert worker.stats()["pending_bytes"] == 0
    assert all(not hasattr(stamp, "jpeg") for stamp in worker._sources)
    worker.close()
    assert worker.stats()["pending_frames"] == 0
    scene.close()


def test_stop_during_render_is_bounded_and_never_publishes_old_source(monkeypatch):
    entered, release = Event(), Event()
    scene, published = runtime(), []

    def render(source, *_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return source

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    worker = TrackedSceneDisplayWorker(scene, published.append)
    source = packet(1, 0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    worker.start()
    assert entered.wait(1)
    before = time.monotonic()
    worker.close(timeout_seconds=0.01)
    assert time.monotonic() - before < 0.2
    assert worker.stats()["pending_frames"] == 0
    release.set()
    wait_for(lambda: worker.stats()["state"] == "stopped")
    assert not published
    scene.close()


@pytest.mark.parametrize("delay", [-1, 30.001, 91, True, float("nan"), float("inf")])
def test_delay_configuration_is_bounded(delay):
    scene = runtime()
    try:
        with pytest.raises(ValueError):
            TrackedSceneDisplayWorker(scene, lambda frame: None, delay_seconds=delay)
    finally:
        scene.close()


def test_display_timing_distinguishes_new_sources_from_late_event_repaints(
    monkeypatch, caplog,
):
    now = [100.0]
    monkeypatch.setattr(
        scene_delivery, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    scene, published = runtime(), []
    original_select = scene.tracked_display_frame
    original_snapshot = scene.display_snapshot

    def select(**kwargs):
        now[0] += 0.01
        return original_select(**kwargs)

    def snapshot(*args, **kwargs):
        # Included in the selection phase for both new frames and late events.
        now[0] += 0.02
        return original_snapshot(*args, **kwargs)

    def render(source, *_args, **_kwargs):
        now[0] += 0.03
        return source

    def publish(frame):
        now[0] += 0.04
        published.append(frame)

    monkeypatch.setattr(scene, "tracked_display_frame", select)
    monkeypatch.setattr(scene, "display_snapshot", snapshot)
    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    worker = TrackedSceneDisplayWorker(scene, publish, max_fps=60)

    def timings():
        return [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == "uvicorn.error"
        ]

    # FPS starts with the first valid capture, even before a SAM admission.
    capture(scene, worker, packet(0, 0), admitted=False)
    now[0] = 101.0
    source = packet(1, 1)
    capture(scene, worker, source)
    tracked(scene, (source,))
    worker.start()
    try:
        wait_for(lambda: len(timings()) == 1)
        first = timings()[0]
        assert first.pop("recorded_at").endswith("+00:00")
        assert first == {
            "event": "sam_display_timing",
            "source_frame_sequence": 1,
            "source_timestamp_seconds": 1,
            "source_to_publish_seconds": 0.1,
            "snapshot_seconds": 0.03,
            "render_seconds": 0.03,
            "publish_seconds": 0.04,
            "late_event_repaint": False,
            "distinct_source_frames": 1,
            "same_source_repaints": 0,
            "distinct_source_fps": round(1 / 1.1, 6),
            "stream_session_id": "s",
        }
        now[0] = 102.0
        worker.notify_events()
        wait_for(lambda: len(timings()) == 2)
        repaint = timings()[1]
        assert repaint["late_event_repaint"] is True
        assert repaint["source_frame_sequence"] == 1
        assert repaint["source_to_publish_seconds"] == 1.09
        assert repaint["snapshot_seconds"] == 0.02
        assert repaint["distinct_source_frames"] == 1
        assert repaint["same_source_repaints"] == 1
        assert repaint["distinct_source_fps"] == round(1 / 2.09, 6)
        assert [frame.frame_sequence for frame in published] == [1, 1]

        now[0] = 103.0
        second = packet(2, 3)
        capture(scene, worker, second)
        tracked(scene, (second,))
        worker.notify_tracking()
        wait_for(lambda: len(timings()) == 3)
        stats = worker.stats()
        assert stats["published_frames"] == 3
        assert stats["distinct_source_frames"] == 2
        assert stats["same_source_repaints"] == 1
        assert stats["distinct_source_fps"] == round(2 / 3.1, 6)
        assert stats["last_snapshot_seconds"] == 0.03
        assert stats["last_jpeg_render_seconds"] == 0.03
        assert stats["last_publish_seconds"] == 0.04
        assert stats["last_source_to_publish_seconds"] == 0.1
        assert stats["render_failures"] == stats["publish_failures"] == 0
        now[0] = 104.0
        worker.close()
        assert worker.stats()["distinct_source_fps"] == 0.5
        now[0] = 110.0
        assert worker.stats()["display_elapsed_seconds"] == 4.0
        assert worker.stats()["distinct_source_fps"] == 0.5
    finally:
        worker.close()
        scene.close()


def test_display_timing_omits_unsafe_session_ids(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    scene = runtime()
    worker = TrackedSceneDisplayWorker(scene, lambda frame: None)
    try:
        monkeypatch.setattr(scene, "stream_session_id", "rtsp://user:secret@camera")
        worker._log_display_timing({"event": "sam_display_timing"})
        record = caplog.records[-1]
        assert record.name == "uvicorn.error"
        fields = json.loads(record.getMessage())
        assert fields.pop("recorded_at").endswith("+00:00")
        assert fields == {"event": "sam_display_timing"}
        assert "secret" not in record.getMessage()
        assert worker.stats()["timing_log_failures"] == 0
    finally:
        worker.close()
        scene.close()


def test_timing_logger_failure_does_not_fail_or_retry_a_successful_publish(monkeypatch):
    def fail_log(*_args, **_kwargs):
        raise OSError("diagnostic sink unavailable")

    monkeypatch.setattr(scene_delivery._DISPLAY_LOGGER, "info", fail_log)
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    source = packet(1, 0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    worker.start()
    try:
        wait_for(lambda: worker.stats()["timing_log_failures"] == 1)
        stats = worker.stats()
        assert len(published) == stats["published_frames"] == 1
        assert stats["distinct_source_frames"] == 1
        assert stats["same_source_repaints"] == 0
        assert stats["render_failures"] == stats["publish_failures"] == 0
        assert worker.wait_until_idle()
        time.sleep(0.03)
        assert len(published) == 1
    finally:
        worker.close()
        scene.close()
