"""Display freshness and finite drain on evidence loss; no inference is performed."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_tracked_scene_delivery import capture, packet, runtime, tracked, wait_for

from goodvision_harness import scene_delivery, service
from goodvision_harness.config import load_config
from goodvision_harness.scene_delivery import TrackedSceneDisplayWorker


def test_reported_lag_advances_between_outputs_and_when_capture_stalls(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        scene_delivery, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    source = packet(1, 10.0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    worker.start()
    try:
        wait_for(lambda: len(published) == 1)
        assert worker.stats()["stream_delay_seconds"] == 0
        capture(scene, worker, packet(2, 18.0), admitted=False)
        assert worker.stats()["stream_delay_seconds"] == 8
        clock[0] = 120.0
        assert worker.stats()["stream_delay_seconds"] == 20
        assert len(published) == 1  # Freshness never fabricates a new source frame.
        worker.close()
        clock[0] = 150.0
        assert worker.stats()["stream_delay_seconds"] == 20
    finally:
        worker.close()
        scene.close()


def test_evicted_image_is_discarded_before_requested_delay_without_eof_wait():
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(
        scene, published.append, max_fps=60, delay_seconds=30
    )
    source = packet(1, 0)
    capture(scene, worker, source)
    tracked(scene, (source,))
    capture(scene, worker, packet(2, 40), admitted=False)
    worker.start()
    try:
        assert worker.wait_until_idle(0.5)
        assert not published and worker.stats()["pending_frames"] == 0
        assert worker.stats()["unavailable_source_frames"] == 1
        assert worker.stats()["discarded_frames"] == 1
        assert worker.stats()["stream_delay_seconds"] is None
    finally:
        worker.close()
        scene.close()


def test_byte_cap_eviction_is_discarded_but_retained_unprocessed_source_waits():
    scene = runtime()
    source, retained = packet(1, 0), packet(2, 0.25)
    scene.evidence.max_bytes = len(retained.jpeg)
    worker = TrackedSceneDisplayWorker(scene, lambda frame: None, max_fps=60)
    capture(scene, worker, source)
    capture(scene, worker, retained)
    worker.start()
    try:
        wait_for(lambda: worker.stats()["unavailable_source_frames"] == 1)
        assert not worker.wait_until_idle(0.02)
        assert worker.stats()["pending_frames"] == 1
        tracked(scene, (retained,), people=False)
        worker.notify_tracking()
        assert worker.wait_until_idle(0.5)
        assert worker.stats()["published_frames"] == 1
        assert worker.stats()["last_displayed_frame_sequence"] == 2
    finally:
        worker.close()
        scene.close()


def test_skipped_source_behind_completed_watermark_is_no_longer_pending():
    scene = runtime()
    skipped, completed = packet(1, 0), packet(2, 0.25)
    scene.observe_frame(skipped.jpeg, 0, 1, force=True)
    scene.observe_frame(completed.jpeg, 0.25, 2, force=True)
    assert scene.unavailable_display_sources(((1, 0), (2, 0.25))) == ()
    tracked(scene, (completed,), people=False)
    assert scene.unavailable_display_sources(((1, 0), (2, 0.25))) == (1,)
    # Both bytes still exist: only the completed watermark proves the skipped ID.
    assert (
        scene.evidence.source_frame(
            stream_session_id="s", frame_sequence=1, timestamp_seconds=0
        )
        is not None
    )
    scene.close()
    assert scene.unavailable_display_sources(((1, 0), (2, 0.25))) == (1, 2)


def test_30_second_delay_keeps_450_source_stamps_without_growing_image_budget():
    scene = runtime()
    worker = TrackedSceneDisplayWorker(scene, lambda frame: None, delay_seconds=30)
    source = packet(0, 0)
    for sequence in range(450):
        worker.submit(
            replace(source, frame_sequence=sequence, media_timestamp=sequence / 15)
        )
    assert worker.stats()["pending_frames"] == 450
    assert worker.stats()["source_stamps_dropped"] == 0
    assert worker.stats()["pending_bytes"] == 0
    assert scene.evidence.max_frames == 512
    assert scene.evidence.max_bytes == 64 * 1024 * 1024
    worker.close()
    scene.close()


@pytest.mark.parametrize(
    "sources",
    [None, [(1,)], [(True, 0)], [(1, -1)], [(1, float("inf"))], [(1, 0)] * 513],
)
def test_unavailable_source_query_is_bounded(sources):
    scene = runtime()
    try:
        with pytest.raises(ValueError):
            scene.unavailable_display_sources(sources)
    finally:
        scene.close()


def test_stream_yaml_delay_above_console_limit_fails_before_clients_start(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(
        config,
        display=replace(config.display, delay_seconds=31),
        runtime=replace(config.runtime, scene_mode="active"),
        rich_labeling=replace(
            config.rich_labeling,
            backend="sam31_stream",
            enabled=True,
        ),
    )
    reached = []
    monkeypatch.setattr(
        service,
        "create_tracking_client",
        lambda *_args, **_kwargs: reached.append(True),
    )
    with pytest.raises(ValueError, match="between 0 and 30"):
        service.HarnessRuntime(config)
    assert reached == []


def test_closing_idle_worker_does_not_leave_waiting_thread():
    scene = runtime()
    for _ in range(10):
        worker = TrackedSceneDisplayWorker(scene, lambda frame: None)
        worker.start()
        worker.close(timeout_seconds=0.5)
        assert worker.stats()["state"] == "stopped"
    scene.close()
