"""Continuous scene operation through real service/session/worker boundaries."""

from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from goodvision_harness import pipeline, service
from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    NormalizedBBox,
)
from goodvision_harness.rich_labeling import (
    RichLabelCoordinator,
    RichLabelObject,
    RichLabelSnapshot,
)
from goodvision_harness.scene_capture import SceneCaptureClient
from goodvision_harness.scene_inspection import build_temporal_messages
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.service import HarnessRuntime

ROOT = Path(__file__).resolve().parents[1]
BOX = NormalizedBBox(0.1, 0.1, 0.6, 0.95)


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    for name in (
        "SCENE_ENGINE_ENABLED",
        "SCENE_ENGINE_MODE",
        "SCENE_TIMEOUT_SECONDS",
        "SCENE_MAX_IMAGE_SIDE",
        "SCENE_MAX_TOKENS",
        "RESULT_WEBHOOK_URL",
        "RICH_LABEL_ENABLED",
        "RICH_LABEL_TRACKING_ENABLED",
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def _jpeg():
    with Image.new("RGB", (240, 160), "gray") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        return buffer.getvalue()


def _await(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("continuous service did not reach expected state")
        time.sleep(0.005)


class OfflineTracker:
    def infer(self, _jpeg, *, stream_session_id, frame_sequence, timestamp_seconds):
        return RichLabelSnapshot(
            stream_session_id, frame_sequence, timestamp_seconds, ()
        )

    def close(self):
        pass


class Inspector:
    def __init__(self, *, blocked=False):
        self.jobs = []
        self.detected = True
        self.started = Event()
        self.release = Event()
        self.closed = Event()
        if not blocked:
            self.release.set()

    def inspect(self, job):
        self.jobs.append(job)
        self.started.set()
        assert self.release.wait(5)
        ids = job.request.subject_ids or tuple(
            dict.fromkeys(subject for _, subject, _ in job.subject_boxes)
        )
        return (
            tuple(
                TemporalVerdict(label, (ids[0],), self.detected, 0.9)
                for label in job.request.event_labels
            )
            if ids
            else ()
        )

    def close(self):
        self.closed.set()

    def stats(self):
        return {"requests": len(self.jobs)}


@pytest.fixture
def factory(tmp_path, monkeypatch):
    runtimes, inspectors = [], []

    def no_remote(*_args, **_kwargs):
        raise AssertionError("unit tests must not construct remote inference")

    monkeypatch.setattr(service, "TemporalVLMInspector", no_remote)
    monkeypatch.setattr(service, "RichLabelHTTPClient", no_remote)

    def make(*, tracker=False, inspector=None, scene_side=720):
        config = load_config(ROOT / "config.yaml")
        config = replace(
            config,
            runtime=replace(
                config.runtime, scene_mode="active", scene_max_image_side=scene_side
            ),
            rich_labeling=replace(
                config.rich_labeling,
                enabled=False,
                tracking_enabled=False,
                backend="paddle",
            ),
            events=replace(
                config.events, directory=tmp_path / f"events-{len(runtimes)}"
            ),
            service=replace(
                config.service, webhook=replace(config.service.webhook, url=None)
            ),
        )
        coordinator = RichLabelCoordinator(OfflineTracker()) if tracker else None
        runtime = HarnessRuntime(config, rich_label_coordinator=coordinator)
        runtime._aiclub_api_key = "TEST_KEY"
        runtimes.append(runtime)
        if inspector is not None:
            inspectors.append(inspector)
            monkeypatch.setattr(
                service, "TemporalVLMInspector", lambda *_a, **_k: inspector
            )
        return runtime

    yield make
    for inspector in inspectors:
        inspector.release.set()
    for runtime in runtimes:
        runtime.stop()


def _observe(runtime, timestamp, sequence):
    runtime._on_tracking_snapshot(
        RichLabelSnapshot(
            runtime._session.stream_session_id,
            sequence,
            timestamp,
            (RichLabelObject("tracker-id", "person", BOX, 0.99),),
        )
    )
    runtime._on_display_frame(_jpeg(), timestamp, sequence)


def test_temporal_evidence_api_returns_exact_source_and_expires_on_reset(factory):
    runtime = factory(tracker=True, inspector=Inspector())
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        assert client.get("/v1/scene/evidence").status_code == 404
        runtime.start_camera(dry_run=False, features=["smoking"])
        _observe(runtime, 0, 0)
        _observe(runtime, 1, 10)
        _await(lambda: runtime._scene.latest_inspection() is not None)
        _await(lambda: not runtime._scene.snapshot()["inspections"])
        response = client.get("/v1/scene/evidence")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        evidence = response.json()
        assert evidence["event_labels"] == ["smoking"]
        assert 2 <= len(evidence["frames"]) <= 4
        assert "base64" not in response.text and "jpeg_bytes" not in response.text
        for frame in evidence["frames"]:
            image = client.get(frame["url"])
            assert image.status_code == 200
            assert image.content == _jpeg()
            assert (
                float(image.headers["x-media-timestamp"]) == frame["timestamp_seconds"]
            )
        _await(lambda: runtime._scene_archive.stats()["stored_events"] > 0)
        history = client.get("/v1/scene/history").json()["events"]
        assert history and history[0]["event_label"] == "smoking"
        assert "identity" not in history[0]
        old_url = evidence["frames"][0]["url"]
        assert runtime.stop_session()
        assert client.get("/v1/scene/evidence").status_code == 404
        assert client.get(old_url).status_code == 404
        runtime.start_camera(dry_run=True, features=["smoking"])
        assert client.get(old_url).status_code == 404


def test_safety_only_capture_avoids_legacy_build_and_classification_with_video(
    factory, monkeypatch
):
    consumed = Event()
    original_windows = pipeline.iter_sliding_windows

    def windows(*args, **kwargs):
        for window in original_windows(*args, **kwargs):
            yield window
            consumed.set()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("active capture must not run the per-frame safety model")

    monkeypatch.setattr(service, "build_selected_features_grounding_client", forbidden)
    monkeypatch.setattr(service, "build_grounding_client", forbidden)
    monkeypatch.setattr(SceneCaptureClient, "classify_window", forbidden)
    monkeypatch.setattr(pipeline, "iter_sliding_windows", windows)
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking", "fall"])
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160)
    assert consumed.wait(2), "the source window must actually traverse the session"
    _await(lambda: runtime.annotated_frames.latest() is not None)
    assert runtime.preview_frames.latest() is not None
    assert runtime._session.running
    assert runtime.status()["last_error"] is None
    assert runtime.results.latest() is None
    assert runtime.scene_snapshot()["mode"] == "active"
    assert runtime.scene_snapshot()["automatic_temporal_inference"] is False
    assert runtime.scene_results.latest() is not None


def test_mixed_active_session_builds_only_legacy_stranger_and_publishes_no_safety(
    factory, monkeypatch
):
    original = service.build_selected_features_grounding_client
    selected = []

    def build(*args, **kwargs):
        selected.append(kwargs["features"])
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "build_selected_features_grounding_client", build)
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking", "stranger"])
    assert selected == [("stranger",)]
    assert runtime._session.client._safety_client is None
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160)
    _await(lambda: runtime.results.latest() is not None)
    judgments = runtime.results.latest().payload["judgments"]
    assert {item["event_label"] for item in judgments} == {"stranger"}
    assert runtime._scene.event_labels == ("smoking",)
    _await(lambda: runtime.frames.latest() is not None)
    exact = runtime.frames.latest()[1]
    assert exact.analysis_message_id == runtime.results.latest().payload["message_id"]
    assert exact.annotation_mode == "none"


@pytest.mark.parametrize(
    "features,scene_side",
    [
        (["smoking"], 720),
        (["smoking", "stranger"], 720),
        (["smoking"], 1600),
        (["smoking", "stranger"], 1600),
    ],
)
def test_active_camera_admits_the_source_resolution_advertised_to_browser(
    factory, features, scene_side
):
    runtime = factory(scene_side=scene_side)
    info = runtime.ui_info()
    settings = info["multi"] if "stranger" in features else info
    side = settings["max_image_side"]
    assert side >= scene_side
    assert info["annotated_output"]["mode"] == "scene"
    assert info["annotated_output"]["buffer_delay_seconds"] == 0
    assert info["annotated_output"]["bbox_max_gap_seconds"] is None
    runtime.start_camera(dry_run=True, features=features)
    assert runtime._session.config.video.max_image_side == side
    height = side // 2
    with Image.new("RGB", (side, height), "gray") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        assert runtime.push_camera_frame(buffer.getvalue(), width=side, height=height)
    _await(lambda: runtime.preview_frames.latest() is not None)
    with Image.open(io.BytesIO(runtime.preview_frames.latest()[1].jpeg)) as image:
        assert image.size == (side, height)
    # Session-specific source resolution must not change legacy model defaults.
    assert runtime.config.video.max_image_side == 320


def test_active_exact_face_frame_keeps_source_bytes_and_rejects_stale_or_foreign(
    factory, monkeypatch
):
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking", "stranger"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "current scene overlays must not alter historical evidence"
        )

    monkeypatch.setattr(runtime._delayed_overlay, "submit_analysis_frame", forbidden)
    monkeypatch.setattr(service, "_render_analysis_jpeg", forbidden)
    jpeg = _jpeg()
    result = AnalysisResult(
        message_id="exact-face-20",
        source_id=runtime.source_id,
        stream_session_id=runtime._session.stream_session_id,
        sequence=20,
        emitted_at="2026-09-12T00:00:00Z",
        frame=FrameReference(20, 2, "2026-09-12T00:00:00Z", 240, 160),
        judgments=(EventJudgment("stranger", False, 0.9, "Known person.", ()),),
        objects=(),
        model_name="offline-face",
        model_latency_seconds=0.1,
    )
    runtime._on_analysis_frame(result, jpeg)
    exact = runtime.frames.latest()
    assert exact[1].jpeg == jpeg
    assert exact[1].analysis_message_id == result.message_id
    assert exact[1].frame_sequence == 20 and exact[1].media_timestamp == 2
    assert exact[1].annotation_mode == "none"
    runtime._on_analysis_frame(
        replace(result, stream_session_id="retired-session"), jpeg
    )
    runtime._on_analysis_frame(
        replace(
            result,
            message_id="old-face",
            frame=replace(result.frame, timestamp_seconds=1, source_frame_index=10),
        ),
        jpeg,
    )
    assert runtime.frames.latest() == exact
    assert runtime.status()["analysis_frames_published"] == 1
    assert runtime.scene_snapshot()["scene"]["events"] == []
    assert runtime.results.latest() is None


def test_live_active_session_requires_a_tracker_before_start(factory):
    runtime = factory()
    with pytest.raises(RuntimeError, match=r"SAM|tracker"):
        runtime.start_camera(dry_run=False, features=["smoking"])
    assert runtime._session is None
    assert runtime._scene_worker is None


def test_actual_worker_confirms_canonical_person_event_and_unknown_does_not_clear(
    factory,
):
    inspector = Inspector()
    runtime = factory(tracker=True, inspector=inspector)
    runtime.start_camera(
        dry_run=False,
        features=["smoking"],
        event_sensitivities={"smoking": "standard", "fight": "high"},
    )
    _observe(runtime, 0, 0)
    _observe(runtime, 1, 10)
    _await(
        lambda: (
            runtime._scene_worker.stats()["succeeded_requests"] >= 1
            and not runtime._scene.snapshot()["inspections"]
        )
    )
    _observe(runtime, 4, 40)
    _await(
        lambda: any(
            event["phase"] == "confirmed"
            for event in runtime.scene_snapshot()["scene"]["events"]
        )
    )
    state = runtime.scene_snapshot()
    assert state["automatic_temporal_inference"] is True
    event = state["scene"]["events"][0]
    canonical = state["scene"]["entities"][0]["subject_id"]
    assert event["subject_ids"] == [canonical]
    assert canonical != "tracker-id"
    assert all(job.request.event_labels == ("smoking",) for job in inspector.jobs)
    assert all(
        dict(job.event_sensitivities) == {"smoking": "standard"}
        for job in inspector.jobs
    )
    assert "STANDARD" in str(build_temporal_messages(inspector.jobs[0]))
    inspector.detected = None
    before = runtime._scene_worker.stats()["succeeded_requests"]
    _observe(runtime, 7, 70)
    _await(lambda: runtime._scene_worker.stats()["succeeded_requests"] > before)
    event = runtime.scene_snapshot()["scene"]["events"][0]
    assert event["phase"] == "confirmed"
    assert event["negative_observations"] == 0
    _await(lambda: runtime.scene_results.latest() is not None)
    assert (
        runtime.scene_results.latest().payload["scene"]["events"][0]["phase"]
        == "confirmed"
    )


def test_restarting_session_cannot_publish_old_model_response(factory):
    inspector = Inspector(blocked=True)
    runtime = factory(tracker=True, inspector=inspector)
    runtime.start_camera(dry_run=False, features=["smoking"])
    _observe(runtime, 0, 0)
    _observe(runtime, 1, 10)
    assert inspector.started.wait(2)
    old_scene = runtime._scene
    assert runtime.stop_session()
    runtime.start_camera(dry_run=True, features=["smoking"])
    current_scene = runtime._scene
    inspector.release.set()
    assert inspector.closed.wait(2)
    assert current_scene.stream_session_id != old_scene.stream_session_id
    assert current_scene.snapshot()["scene"]["events"] == []
    assert current_scene.snapshot()["stats"]["completed_inspections"] == 0
    assert runtime.scene_results.latest() is None
    assert runtime.annotated_frames.latest() is None


def test_active_configuration_environment_is_effective(monkeypatch):
    monkeypatch.setenv("SCENE_ENGINE_MODE", "active")
    monkeypatch.setenv("SCENE_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("SCENE_MAX_IMAGE_SIDE", "1024")
    monkeypatch.setenv("SCENE_MAX_TOKENS", "2048")
    config = load_config(ROOT / "config.yaml")
    assert config.runtime.scene_mode == "active"
    assert config.runtime.scene_timeout_seconds == 12
    assert config.runtime.scene_max_image_side == 1024
    assert config.runtime.scene_max_tokens == 2048


@pytest.mark.parametrize(
    "name,value",
    [
        ("SCENE_ENGINE_MODE", "automatic"),
        ("SCENE_TIMEOUT_SECONDS", "nan"),
        ("SCENE_TIMEOUT_SECONDS", "inf"),
        ("SCENE_TIMEOUT_SECONDS", "21"),
        ("SCENE_MAX_IMAGE_SIDE", "2049"),
        ("SCENE_MAX_TOKENS", "16385"),
    ],
)
def test_invalid_active_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises((ValueError, TypeError)):
        load_config(ROOT / "config.yaml")
