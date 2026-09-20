"""The rich worker must prove GPU readiness before any video is submitted."""

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import cv2
import httpx
import numpy as np
import pytest

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.delayed_overlay import OverlaySelection
from goodvision_harness.rich_labeling import (
    RichLabelHTTPClient,
    RichLabelObject,
    RichLabelSnapshot,
)


def _stop_and_wait(runtime):
    session = runtime._session
    assert runtime.stop_session()
    session.join(timeout=5)
    assert not session.running
    drain = getattr(runtime, "_end_drain_thread", None)
    if drain is not None:
        drain.join(timeout=5)
        assert not drain.is_alive()
    assert runtime._session is None
    assert runtime.status()["pipeline_state"] == "idle"


@pytest.mark.parametrize("payload,allowed", [
    ({"ready": True, "gpu_ready": True}, True),
    ({"ready": True, "gpu_ready": False}, False),
    ({"ready": False, "gpu_ready": True}, False),
    ({"ready": True}, False),
    ({"ready": True, "gpu_ready": "true"}, False),
    ({"ready": True, "gpu_ready": 1}, False),
    ([], False),
])
def test_gpu_gate_uses_strict_remote_readiness(payload, allowed):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with RichLabelHTTPClient(
        "http://remote-worker:8091/v1/labels?unneeded=1",
        transport=httpx.MockTransport(handle),
    ) as client:
        assert client.gpu_ready() is allowed
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "http://remote-worker:8091/health/ready"
    assert requests[0].content == b""
    assert requests[0].extensions["timeout"]["read"] == 2.0


@pytest.mark.parametrize("failure", ["timeout", "connection", "http", "json"])
def test_unavailable_or_invalid_worker_never_enables_rich(failure):
    def handle(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private", request=request)
        if failure == "connection":
            raise httpx.ConnectError("private", request=request)
        if failure == "http":
            return httpx.Response(503, json={"ready": True, "gpu_ready": True})
        return httpx.Response(200, content=b"not json")

    with RichLabelHTTPClient(
        "http://remote-worker/v1/labels", transport=httpx.MockTransport(handle),
    ) as client:
        assert client.gpu_ready() is False


def test_cpu_gate_never_constructs_coordinator_or_submits_frames(monkeypatch):
    calls = []
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(config, rich_labeling=replace(config.rich_labeling, enabled=True))

    class Client:
        def __init__(self, *args, **kwargs):
            calls.append("client")

        def gpu_ready(self):
            calls.append("probe")
            return False

        def close(self):
            calls.append("close")

    def forbidden(*args, **kwargs):
        pytest.fail("CPU rich coordinator must never be constructed")

    monkeypatch.setattr(service, "RichLabelHTTPClient", Client)
    monkeypatch.setattr(service, "RichLabelCoordinator", forbidden)
    # This test covers the synchronous startup gate. Separate tests below cover
    # readiness retries, which status/session helpers may start in the background.
    monkeypatch.setattr(
        service.HarnessRuntime, "_ensure_rich_readiness_probe", lambda self: None,
    )
    runtime = service.HarnessRuntime(config)
    try:
        assert runtime._rich_labels is None
        assert runtime.ui_info()["rich_labeling"]["enabled"] is False
        assert runtime.status()["rich_labeling_enabled"] is False
        assert runtime.status()["rich_labeling"] == {
            "state": "disabled", "disabled_reason": "gpu_required",
        }
        runtime._active_rich_session_id = "sample-session"
        runtime._on_display_frame(b"frame-not-submitted-to-rich", 1.0, 1)
        assert calls == ["client", "probe", "close"]
        assert runtime.preview_frames.latest() is not None
    finally:
        runtime.stop()


def test_explicit_disabled_setting_does_not_even_probe_worker(monkeypatch):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(config, rich_labeling=replace(
        config.rich_labeling, enabled=False, tracking_enabled=False,
    ))

    def forbidden(*args, **kwargs):
        pytest.fail("disabled rich settings must not contact the worker")

    monkeypatch.setattr(service, "RichLabelHTTPClient", forbidden)
    runtime = service.HarnessRuntime(config)
    try:
        runtime.start()
        assert runtime.status()["rich_labeling"] == {"state": "disabled"}
        assert runtime._rich_probe_thread is None
    finally:
        runtime.stop()


def test_tracking_stays_active_when_rich_ui_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(
        config,
        rich_labeling=replace(
            config.rich_labeling, enabled=False, tracking_enabled=True,
        ),
        events=replace(config.events, directory=tmp_path / "events"),
    )
    options, submissions = [], []

    class Client:
        def __init__(self, *args, **kwargs):
            options.append(kwargs)

        def gpu_ready(self):
            return True

        def infer(self, jpeg, **metadata):
            submissions.append(metadata)
            return RichLabelSnapshot(**metadata, objects=(RichLabelObject(
                "P1", "person", NormalizedBBox(.2, .1, .7, .9),
                observation_state="observed",
                last_observed_timestamp_seconds=metadata["timestamp_seconds"],
                last_observed_frame_sequence=metadata["frame_sequence"],
            ),))

        def close(self):
            pass

    monkeypatch.setattr(service, "RichLabelHTTPClient", Client)
    runtime = service.HarnessRuntime(config)
    observed = Event()
    try:
        runtime.start()
        runtime._rich_probe_thread.join(timeout=2)
        assert runtime._rich_labels is not None
        assert len(options) == 1
        assert options[0]["mode"] == "tracks"
        assert options[0]["include_attributes"] is False
        runtime.start_camera(dry_run=True, features=["smoking"])
        registry = runtime._session.person_registry
        receive = registry.observe_snapshot

        def record(snapshot):
            receive(snapshot)
            observed.set()

        monkeypatch.setattr(registry, "observe_snapshot", record)
        ok, image = cv2.imencode(".jpg", np.zeros((160, 240, 3), dtype=np.uint8))
        assert ok
        runtime._on_display_frame(image.tobytes(), 2., 2)
        assert observed.wait(2)
        assert len(submissions) == 1
        assert registry.history()
        assert runtime.status()["rich_labeling_enabled"] is False
        assert runtime.ui_info()["rich_labeling"]["enabled"] is False
        assert runtime.latest_rich_labels() is None
    finally:
        runtime.stop()


def test_confirmed_remote_gpu_still_enables_rich(monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(config, rich_labeling=replace(config.rich_labeling, enabled=True))
    monkeypatch.setattr(RichLabelHTTPClient, "gpu_ready", lambda _: True)
    runtime = service.HarnessRuntime(config)
    try:
        assert runtime._rich_labels is not None
        assert runtime.ui_info()["rich_labeling"]["enabled"] is True
        assert runtime.status()["rich_labeling_enabled"] is True
        assert "disabled_reason" not in runtime.status()["rich_labeling"]
    finally:
        runtime._rich_labels.close()


def _requested_config(monkeypatch):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setattr(service, "RICH_READINESS_RETRY_SECONDS", 0.01)
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    return replace(config, rich_labeling=replace(config.rich_labeling, enabled=True))


def test_warming_gpu_recovers_current_face_session_and_delayed_overlay(monkeypatch):
    config = _requested_config(monkeypatch)
    probing, ready = Event(), Event()
    clients, submitted, face_sessions = [], [], []

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False
            clients.append(self)

        def gpu_ready(self):
            if len(clients) == 1:
                return False
            probing.set()
            return ready.wait(2.0)

        def infer(self, jpeg, **metadata):
            submitted.append(metadata)
            return RichLabelSnapshot(
                **metadata,
                objects=(RichLabelObject(
                    "P1", "person", NormalizedBBox(0.2, 0.1, 0.7, 0.9),
                ),),
            )

        def close(self):
            self.closed = True

    class FaceRuntime:
        settings = config.inference

        def warmup(self):
            return 0.0

    class FaceSession(service.FaceRecognitionSession):
        def __init__(self, runtime, **kwargs):
            super().__init__(runtime, **kwargs)
            face_sessions.append(self)

    monkeypatch.setattr(service, "RichLabelHTTPClient", Client)
    monkeypatch.setattr(service, "FaceRecognitionRuntime", FaceRuntime)
    monkeypatch.setattr(service, "FaceRecognitionSession", FaceSession)
    runtime = service.HarnessRuntime(config, face_runtime=FaceRuntime())
    ok, image = cv2.imencode(".jpg", np.zeros((160, 240, 3), dtype=np.uint8))
    assert ok
    source = image.tobytes()
    try:
        session = runtime.start_camera(analysis_mode="face")
        assert probing.wait(1.0)
        face = face_sessions[0]
        tracker = face._tracker
        history = face._rich_labels
        assert history.history() == ()
        runtime._on_display_frame(source, 1.0, 1)
        assert submitted == []
        assert runtime.preview_frames.latest() is not None
        assert runtime.status()["rich_labeling_enabled"] is False

        # Capture the actual callback registered before the worker was ready.
        compose = runtime._delayed_overlay._compose
        assert compose is not None
        assert compose(source, OverlaySelection(None, "none"), 1.0, 1) == source
        ready.set()
        runtime._rich_probe_thread.join(timeout=2.0)
        assert not runtime._rich_probe_thread.is_alive()
        assert runtime.status()["rich_labeling_enabled"] is True
        assert runtime.ui_info()["rich_labeling"]["enabled"] is True
        assert "disabled_reason" not in runtime.status()["rich_labeling"]
        assert runtime._active_rich_session_id == session["stream_session_id"]
        assert face._tracker is tracker
        assert face._rich_labels is history

        runtime._on_display_frame(source, 2.0, 2)
        assert runtime._rich_labels.wait_until_idle(2.0)
        assert [item["stream_session_id"] for item in submitted] == [
            session["stream_session_id"],
        ]
        assert history.history()[0].frame_sequence == 2
        assert runtime.latest_rich_labels()["objects"][0]["track_id"] == "P1"
        rendered = compose(source, OverlaySelection(None, "none"), 2.0, 2)
        pixels = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.count_nonzero(pixels) > 0
        runtime._ensure_rich_readiness_probe()
        assert len(clients) == 2
    finally:
        ready.set()
        runtime.stop()
    assert all(client.closed for client in clients)
    assert not runtime._rich_probe_thread.is_alive()


def test_requested_cpu_worker_retries_without_inference_and_stops_cleanly(monkeypatch):
    config = _requested_config(monkeypatch)
    retried = Event()
    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False
            clients.append(self)

        def gpu_ready(self):
            if len(clients) >= 3:
                retried.set()
            return False

        def close(self):
            self.closed = True

    def forbidden(*args, **kwargs):
        pytest.fail("A CPU worker must never get a rich coordinator")

    monkeypatch.setattr(service, "RichLabelHTTPClient", Client)
    monkeypatch.setattr(service, "RichLabelCoordinator", forbidden)
    runtime = service.HarnessRuntime(config)
    try:
        runtime.start()
        thread = runtime._rich_probe_thread
        runtime._ensure_rich_readiness_probe()
        assert runtime._rich_probe_thread is thread
        assert retried.wait(1.0)
        assert runtime.status()["rich_labeling"] == {
            "state": "disabled", "disabled_reason": "gpu_required",
        }
    finally:
        runtime.stop()
    assert not thread.is_alive()
    assert all(client.closed for client in clients)
    count = len(clients)
    assert runtime._try_enable_rich_labels() is True
    assert len(clients) == count


def test_gpu_success_after_shutdown_never_enables_or_leaks_client(monkeypatch):
    config = _requested_config(monkeypatch)
    probing, release = Event(), Event()
    clients = []

    class Client:
        def __init__(self, *args, **kwargs):
            self.closed = False
            clients.append(self)

        def gpu_ready(self):
            if len(clients) == 1:
                return False
            probing.set()
            assert release.wait(2.0)
            return True

        def close(self):
            self.closed = True

    def forbidden(*args, **kwargs):
        pytest.fail("A late health response must not enable rich during shutdown")

    monkeypatch.setattr(service, "RichLabelHTTPClient", Client)
    monkeypatch.setattr(service, "RichLabelCoordinator", forbidden)
    runtime = service.HarnessRuntime(config)
    runtime.start()
    shutdown = Thread(target=runtime.stop)
    try:
        assert probing.wait(1.0)
        shutdown.start()
        assert runtime._rich_probe_stop.wait(1.0)
        release.set()
        shutdown.join(timeout=2.0)
        assert not shutdown.is_alive()
        assert not runtime._rich_probe_thread.is_alive()
        assert runtime._rich_labels is None
        assert runtime.status()["rich_labeling_enabled"] is False
        assert all(client.closed for client in clients)
    finally:
        release.set()
        if shutdown.ident is not None:
            shutdown.join(timeout=2.0)
        runtime.stop()


def test_gpu_warmup_result_attaches_only_to_current_session(monkeypatch):
    config = _requested_config(monkeypatch)
    probing, ready = Event(), Event()
    count = 0

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def gpu_ready(self):
            nonlocal count
            count += 1
            if count == 1:
                return False
            probing.set()
            return ready.wait(2.0)

        def close(self):
            pass

    monkeypatch.setattr(service, "RichLabelHTTPClient", Client)
    runtime = service.HarnessRuntime(config)
    try:
        first = runtime.start_camera(dry_run=True)
        assert probing.wait(1.0)
        _stop_and_wait(runtime)
        second = runtime.start_camera(dry_run=True)
        assert first["stream_session_id"] != second["stream_session_id"]
        ready.set()
        runtime._rich_probe_thread.join(timeout=2.0)
        assert not runtime._rich_probe_thread.is_alive()
        assert runtime._rich_labels.stats()["stream_session_id"] == second[
            "stream_session_id"
        ]
        assert runtime._rich_labels.history() == ()
    finally:
        ready.set()
        runtime.stop()
