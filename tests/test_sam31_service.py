"""Connect native video batches to actual service, registry and temporal runtime."""

from __future__ import annotations

import io
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.scene_worker import SceneInspectionWorker
from goodvision_harness.video_tracking import (
    VideoFrame,
    VideoTrackingCoordinator,
    VideoTrackingHTTPClient,
)
from goodvision_harness.video_tracking_session import (
    VideoSessionTrackingCoordinator,
    VideoSessionTrackingHTTPClient,
)
from workers.sam_scene.app import create_app
from workers.sam_scene.continuous import SessionEngine, SessionSettings
from workers.sam_scene.sam31 import VideoEngine, VideoSettings

ROOT = Path(__file__).resolve().parents[1]


class NativeDouble:
    def warmup(self):
        pass

    def infer(self, images):
        import numpy as np

        for index, image in enumerate(images):
            mask = np.zeros((1, image.height, image.width), dtype=bool)
            # Same native ID changes position; identity must follow it.
            x = image.width // 4 if index % 2 else image.width // 2
            mask[:, :, x : x + image.width // 4] = True
            yield (
                index,
                {
                    "out_obj_ids": np.array([7]),
                    "out_probs": np.array([0.9]),
                    "out_binary_masks": mask,
                },
            )

    def close(self):
        pass


def jpeg():
    with Image.new("RGB", (80, 48), "gray") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        return buffer.getvalue()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("SAM_SCENE_IMAGE_ENABLED", "false")
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        runtime=replace(config.runtime, scene_mode="active"),
        rich_labeling=replace(
            config.rich_labeling, backend="sam31", enabled=False, tracking_enabled=False
        ),
        events=replace(config.events, directory=tmp_path / "events"),
        service=replace(
            config.service, webhook=replace(config.service.webhook, url=None)
        ),
    )
    instance = service.HarnessRuntime(config)
    instance.start_camera(dry_run=True, features=["smoking"])
    yield instance
    instance.stop()


def test_worker_to_http_parser_to_registry_and_scene(runtime):
    engine = VideoEngine(replace(VideoSettings(), enabled=True), backend=NativeDouble())
    with TestClient(create_app(video_engine=engine)) as worker:

        def transport(request):
            response = worker.request(
                request.method,
                request.url.path,
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(response.status_code, content=response.content)

        client = VideoTrackingHTTPClient(
            "http://worker/v1/video/batches", transport=httpx.MockTransport(transport)
        )
        assert not client.gpu_ready(), "doubles must never attest GPU validation"
        coordinator = VideoTrackingCoordinator(
            client, on_batch=runtime._on_tracking_batch, batch_frames=4
        )
        coordinator.reset(runtime._session.stream_session_id)
        # Injection tests the wire; deployment still requires real GPU readiness.
        runtime._rich_labels = coordinator
        runtime._active_rich_session_id = runtime._session.stream_session_id
        try:
            for sequence in range(4):
                runtime._on_display_frame(jpeg(), sequence * 0.25, sequence)
            assert coordinator.wait_until_idle(5)
            assert coordinator.stats()["failed"] == 0
            scene = runtime.scene_snapshot()
            assert scene["tracking_mode"] == "batch"
            assert scene["tracking_completed_frame_sequence"] == 3
            native = [
                e
                for e in runtime._scene.snapshot()["scene"]["entities"]
                if e.get("native_track")
            ]
            assert len(native) == 1
            assert len(native[0]["history"]) == 4
            first_id = native[0]["subject_id"]
            for sequence in range(4, 8):
                runtime._on_display_frame(jpeg(), sequence * 0.25, sequence)
            assert coordinator.wait_until_idle(5)
            assert coordinator.stats()["failed"] == 0
            latest = runtime._session.person_registry.snapshot_at(1.75, 7)
            assert latest.objects[0].track_id != first_id
            assert runtime.scene_snapshot()["tracking_completed_frame_sequence"] == 7
        finally:
            coordinator.close()


def test_batch_commit_cannot_change_replacement_session(runtime):
    from goodvision_harness.video_tracking import VideoTrackingBatch

    session = runtime._session
    old_session = session.stream_session_id
    assert runtime.stop_session()
    session.join(timeout=5)
    assert not session.running
    runtime._end_drain_thread.join(timeout=5)
    assert not runtime._end_drain_thread.is_alive()
    assert runtime.status()["pipeline_state"] == "idle"
    runtime.start_camera(dry_run=True, features=["smoking"])
    before = runtime.scene_snapshot()
    runtime._on_tracking_batch(VideoTrackingBatch(old_session, "late", "old", (), 0, 0))
    assert runtime.scene_snapshot() == before


def test_late_batches_do_not_appear_current_or_create_unmatched_evidence(runtime):
    from goodvision_harness.rich_labeling import RichLabelSnapshot
    from goodvision_harness.video_tracking import VideoTrackingBatch

    session = runtime._session.stream_session_id
    runtime._on_display_frame(jpeg(), 0, 0)
    runtime._on_display_frame(jpeg(), 20, 20)
    registry_before = runtime._session.person_registry.native_tracking_stats()
    result = VideoTrackingBatch(
        session, "late", "epoch", (RichLabelSnapshot(session, 0, 0, ()),), 0, 0
    )
    with pytest.raises(ValueError, match="retained source evidence"):
        runtime._on_tracking_batch(result)
    assert runtime._session.person_registry.native_tracking_stats() == registry_before
    scene = runtime.scene_snapshot()
    assert scene["capture_timestamp_seconds"] == 20
    assert scene["tracking_completed_timestamp_seconds"] is None
    assert not scene["inspections"]


def test_video_frame_type_does_not_accept_nonfinite_or_boolean_source():
    for sequence, timestamp in ((True, 0), (1, float("nan")), (1, True)):
        with pytest.raises(ValueError):
            VideoFrame(sequence, timestamp, jpeg())


class DelayedSessionDouble:
    """Match the native fourteen-source delay without using a GPU."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []

    def warmup(self):
        pass

    def run(self, session, images, *, final):
        import numpy as np

        self.calls.append((len(images), final))
        self.entered.set()
        assert self.release.wait(5)
        session.native_state = {"test": True}
        end = session.total_frames + len(images)
        stop = end if final else max(0, end - 14)
        width, height = session.output_size
        for index in range(session.emitted_index + 1, stop):
            mask = np.zeros((1, height, width), dtype=bool)
            mask[:, 4:32, 8:24] = True
            yield (
                index,
                {
                    "out_obj_ids": np.array([7]),
                    "out_probs": np.array([0.9]),
                    "out_binary_masks": mask,
                },
            )

    def prune(self, session):
        assert len(session.sources) <= 14

    def discard(self, session):
        session.native_state = None

    def close(self):
        pass


@contextmanager
def continuous(runtime):
    video = VideoEngine(replace(VideoSettings(), enabled=True), backend=NativeDouble())
    backend = DelayedSessionDouble()
    sessions = SessionEngine(video, SessionSettings(enabled=True), backend)
    with TestClient(create_app(video_engine=video, session_engine=sessions)) as worker:

        def transport(request):
            response = worker.request(
                request.method,
                request.url.path,
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(response.status_code, content=response.content)

        client = VideoSessionTrackingHTTPClient(
            "http://worker/v1/video/sessions",
            transport=httpx.MockTransport(transport),
        )
        coordinator = VideoSessionTrackingCoordinator(
            client,
            on_batch=runtime._on_tracking_batch,
        )
        coordinator.reset(runtime._session.stream_session_id)
        runtime._rich_labels = coordinator
        runtime._active_rich_session_id = runtime._session.stream_session_id
        try:
            yield coordinator, backend
        finally:
            backend.release.set()
            coordinator.close(timeout_seconds=5)


def wait_for_postprocessing(runtime, timeout=5):
    deadline = time.monotonic() + timeout
    while runtime.status().get("scene_drain", {}).get("state") == "running":
        assert time.monotonic() < deadline, "background finalization did not finish"
        time.sleep(0.005)


def end_and_join(runtime, timeout=5):
    # Exercise the real outer generation lock, not just the direct helper.
    runtime._guarded_callback(runtime._generation, runtime._on_end, "eof", None)
    thread = runtime._end_drain_thread
    thread.join(timeout)
    assert not thread.is_alive(), "EOF must not wait while holding the callback lock"
    wait_for_postprocessing(runtime, timeout)


@pytest.mark.parametrize("count", [3, 16, 32])
def test_natural_eof_flushes_short_sources_and_native_delayed_tail(runtime, count):
    with continuous(runtime) as (coordinator, backend):
        for sequence in range(count):
            runtime._on_display_frame(jpeg(), sequence / 4, sequence)
            if sequence % 16 == 15:
                assert coordinator.wait_until_idle(5)
        if count >= 16:
            assert coordinator.stats()["pending_frame_count"] == 14
            assert (
                runtime.scene_snapshot()["tracking_completed_frame_sequence"]
                == count - 15
            )
        else:
            assert not backend.calls
        end_and_join(runtime)
        snapshot = runtime.scene_snapshot()
        assert snapshot["tracking_completed_frame_sequence"] == count - 1
        assert snapshot["capture_timestamp_seconds"] == (count - 1) / 4
        assert snapshot["state"] == "stopped"
        assert runtime.status()["scene_drain"] == {"state": "completed"}
        assert coordinator.stats()["pending_frame_count"] == 0
        # The live coordinator is released; the retired run retains every source.
        retired = runtime.scene_snapshot(include_history=True)["scene"]
        assert len(retired["entities"]) == 1
        assert len(retired["entities"][0]["history"]) == count
        assert runtime._session is None
        assert backend.calls[-1] == ((count if count < 16 else 0), True)


def test_natural_eof_waits_for_inflight_tracking_without_blocking_callback(runtime):
    with continuous(runtime) as (coordinator, backend):
        backend.release.clear()
        for sequence in range(16):
            runtime._on_display_frame(jpeg(), sequence / 4, sequence)
        assert backend.entered.wait(3)
        began = time.monotonic()
        runtime._guarded_callback(runtime._generation, runtime._on_end, "eof", None)
        assert time.monotonic() - began < 0.5
        assert runtime.status()["pipeline_state"] == "draining"
        assert runtime.scene_snapshot()["state"] == "running"
        backend.release.set()
        runtime._end_drain_thread.join(5)
        assert not runtime._end_drain_thread.is_alive()
        assert runtime.scene_snapshot()["tracking_completed_frame_sequence"] == 15
        assert backend.calls == [(16, False), (0, True)]
        assert coordinator.stats()["failed"] == 0


def test_eof_drain_must_finish_before_replacement_and_cannot_change_it(runtime):
    with continuous(runtime) as (_, backend):
        backend.release.clear()
        for sequence in range(16):
            runtime._on_display_frame(jpeg(), sequence / 4, sequence)
        assert backend.entered.wait(3)
        runtime._guarded_callback(runtime._generation, runtime._on_end, "eof", None)
        old_thread = runtime._end_drain_thread
        old_scene = runtime._scene
        old_generation = runtime._generation
        assert runtime.stop_session()
        assert runtime.status()["pipeline_state"] == "draining"
        assert runtime._generation == old_generation
        with pytest.raises(RuntimeError, match="already running"):
            runtime.start_camera(dry_run=True, features=["smoking"])
        backend.release.set()
        old_thread.join(5)
        assert not old_thread.is_alive()
        assert old_scene.snapshot()["tracking_completed_frame_sequence"] == 15
        assert runtime.status()["pipeline_state"] == "idle"
        runtime.start_camera(dry_run=True, features=["smoking"])
        replacement = runtime._scene
        runtime._guarded_callback(old_generation, runtime._on_end, "eof", None)
        assert runtime._scene is replacement
        assert replacement.snapshot()["state"] == "running"
        assert replacement.snapshot()["tracking_completed_frame_sequence"] is None
        assert runtime.status()["pipeline_state"] != "ended"
        # Finish the replacement while its injected tracker is still open.
        assert runtime.stop_session()
        runtime._end_drain_thread.join(5)
        assert not runtime._end_drain_thread.is_alive()


def test_explicit_stop_preserves_delayed_native_results_until_tracking_completes(
    runtime,
):
    with continuous(runtime) as (_, backend):
        backend.release.clear()
        for sequence in range(16):
            runtime._on_display_frame(jpeg(), sequence / 4, sequence)
        assert backend.entered.wait(3)
        session = runtime._session
        assert runtime.stop_session()
        # Stop returns before the source's asynchronous end callback necessarily
        # runs. Joining the source publishes the drain thread without waiting
        # for native tracking, which remains blocked until backend.release.
        session.join(timeout=3)
        assert not session.running, "source did not initiate the tracking drain"
        thread = runtime._end_drain_thread
        assert thread.is_alive()
        assert runtime.status()["scene_drain"] == {
            "state": "running", "stage": "tracking",
        }
        assert runtime.scene_snapshot()["state"] == "running"
        assert not runtime.push_camera_frame(jpeg(), width=80, height=48)
        with pytest.raises(RuntimeError, match="already running"):
            runtime.start_camera(dry_run=True, features=["smoking"])
        backend.release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert runtime.status()["pipeline_state"] == "idle"
        wait_for_postprocessing(runtime)
        assert runtime.status()["scene_drain"] == {"state": "completed"}
        assert runtime.scene_snapshot()["state"] == "stopped"
        assert runtime.scene_snapshot()["tracking_completed_frame_sequence"] == 15
        assert backend.calls == [(16, False), (0, True)]


def test_closed_tracker_reports_failed_finalization_instead_of_waiting_forever(
    runtime, monkeypatch,
):
    with continuous(runtime) as (coordinator, _):
        coordinator.close(timeout_seconds=5)
        # A closed tracker may retain unfinished cleanup bookkeeping; no worker
        # remains to turn that pending state into a successful idle result.
        monkeypatch.setattr(coordinator, "wait_until_idle", lambda **kwargs: False)
        session = runtime._session
        assert runtime.stop_session()
        session.join(3)
        runtime._end_drain_thread.join(3)
        assert not runtime._end_drain_thread.is_alive()
        assert runtime.status()["scene_drain"] == {"state": "failed"}
        assert runtime.status()["pipeline_state"] == "error"
        assert runtime.status()["last_error"] == service.PUBLIC_TRACKING_ERROR
        assert runtime._session is None


def test_natural_eof_keeps_source_evidence_until_final_temporal_verdicts(runtime):
    entered, release = threading.Event(), threading.Event()

    class Inspector:
        def inspect(self, job):
            entered.set()
            assert release.wait(5)
            subjects = tuple(sorted({sid for _, sid, _ in job.subject_boxes}))
            return (TemporalVerdict("smoking", subjects, True, 0.9),)

    with continuous(runtime) as (coordinator, _):
        worker = SceneInspectionWorker(runtime._scene, Inspector())
        runtime._scene_worker = worker
        worker.start()
        try:
            for sequence in range(16):
                runtime._on_display_frame(jpeg(), sequence / 4, sequence)
            assert coordinator.wait_until_idle(5)
            runtime._guarded_callback(runtime._generation, runtime._on_end, "eof", None)
            assert entered.wait(3)
            assert runtime._end_drain_thread.is_alive()
            assert runtime._scene.has_pending_inspections()
            assert runtime.scene_snapshot()["state"] == "running"
            release.set()
            runtime._end_drain_thread.join(5)
            assert not runtime._end_drain_thread.is_alive()
            wait_for_postprocessing(runtime)
            assert runtime.status()["scene_drain"] == {"state": "completed"}
            assert worker.stats()["succeeded_requests"] == 2
            assert (
                runtime.scene_snapshot()["scene"]["events"][0]["phase"] == "confirmed"
            )
        finally:
            release.set()
            worker.close()


def test_slow_continuous_samples_survive_high_rate_preview(runtime):
    from goodvision_harness.rich_labeling import RichLabelSnapshot
    from goodvision_harness.scene_runtime import SceneRuntime
    from goodvision_harness.video_tracking import VideoTrackingBatch
    from goodvision_harness.video_tracking_session import (
        VideoSessionTrackingCoordinator,
    )

    class AdmissionOnly(VideoSessionTrackingCoordinator):
        def __init__(self):
            self.calls = 0

        def submit(self, _jpeg, **source):
            self.calls += 1
            return source["frame_sequence"] % 60 == 0

        def close(self, *args, **kwargs):
            pass

        def select(self, *args, **kwargs):
            return None

    session = runtime._session.stream_session_id
    runtime._scene.close()
    runtime._scene = SceneRuntime(
        session,
        "camera",
        mode="active",
        tracking_mode="batch",
        native_memory_scope="session",
        tracking_history_seconds=72,
        event_labels=(),
    )
    coordinator = AdmissionOnly()
    runtime._rich_labels = coordinator
    runtime._active_rich_session_id = session
    source = jpeg()
    try:
        # A 30 FPS preview and 0.5 FPS tracker: the second full input window
        # arrives after 62 seconds, while native output still needs old samples.
        for sequence in range(1861):
            runtime._on_display_frame(source, sequence / 30, sequence)
        assert coordinator.calls == 1861
        before = runtime._scene.snapshot()
        assert before["capture_timestamp_seconds"] == 62
        assert before["evidence_buffer"]["frames"] == 32
        frames = tuple(RichLabelSnapshot(session, i, i / 30, ()) for i in (0, 60))
        runtime._on_tracking_batch(
            VideoTrackingBatch(
                session,
                "delayed",
                "epoch",
                frames,
                60,
                2,
            )
        )
        after = runtime._scene.snapshot()
        assert after["tracking_completed_frame_sequence"] == 60
        assert after["tracking_rejections"] == {}
    finally:
        runtime._rich_labels = None
