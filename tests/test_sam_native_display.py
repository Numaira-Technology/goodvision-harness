"""Native presentation is independent of scene commit and its success flag."""

import threading

from test_sam_tracking_pipeline import setup_pipeline
from test_video_tracking_session import frames, submit, wait_for


def test_native_geometry_is_durable_and_published_before_blocked_scene_commit(tmp_path):
    entered, release = threading.Event(), threading.Event()
    native, committed = [], []

    def scene(batch):
        entered.set()
        assert release.wait(4)
        committed.append(batch)

    _, _, coordinator = setup_pipeline(tmp_path, scene)

    def display(batch, generation):
        assert coordinator._spool.stats()["recoverable_results"] >= 1
        native.append((batch, generation))

    coordinator.on_native_batch = display
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(3)
        assert len(native) == 1 and committed == []
        submit(coordinator, frames(2, 1))
        wait_for(lambda: len(native) == 2)
        assert committed == []
        assert coordinator._spool.stats()["retained_sources"] == 2
        release.set()
        assert coordinator.wait_until_idle(5)
        assert len(committed) == 2
    finally:
        release.set()
        coordinator.close()


def test_presentation_callback_failure_preserves_ordered_analysis_commit(tmp_path):
    committed = []

    def fail_display(batch, generation):
        raise RuntimeError("test display failed")

    _, _, coordinator = setup_pipeline(
        tmp_path, committed.append, on_native_batch=fail_display
    )
    try:
        submit(coordinator, frames(1, 2))
        coordinator.flush()
        assert coordinator.wait_until_idle(5)
        assert sum(len(batch.frames) for batch in committed) == 2
        assert not coordinator.stats()["recovery_required"]
        assert all(row["state"] == "committed" for row in coordinator._spool.records())
    finally:
        coordinator.close()


def test_stale_http_result_after_reset_never_reaches_native_display(tmp_path):
    native = []
    server, _, coordinator = setup_pipeline(
        tmp_path,
        on_native_batch=lambda batch, generation: native.append(batch),
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        coordinator.reset("new")
        server.release.set()
        wait_for(lambda: coordinator.stats()["stale_results"] == 1)
        assert native == []
    finally:
        server.release.set()
        coordinator.close()
