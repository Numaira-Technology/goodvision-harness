from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from goodvision_harness import service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delayed_overlay import OverlaySelection
from goodvision_harness.deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
)
from goodvision_harness.grounding import DETECTION_EVENT_LABELS, DryRunGroundingClient
from goodvision_harness.rich_label_models import DETECTOR_MODEL_HEADER
from goodvision_harness.rich_labeling import (
    FRAME_SEQUENCE_HEADER,
    STREAM_SESSION_HEADER,
    TIMESTAMP_HEADER,
    RichLabelHTTPClient,
    RichLabelObject,
    RichLabelSelection,
    RichLabelSnapshot,
)
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


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


def _jpeg() -> bytes:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _snapshot() -> RichLabelSnapshot:
    return RichLabelSnapshot(
        stream_session_id="stream-rich",
        frame_sequence=4,
        timestamp_seconds=1.0,
        objects=(
            RichLabelObject(
                track_id="P0001",
                label="person",
                bbox=NormalizedBBox(0.2, 0.15, 0.75, 0.9),
                confidence=0.94,
                attributes={
                    "estimated_age_range": "18-60",
                    "estimated_gender": "female-presenting",
                },
            ),
        ),
    )


class _RichCoordinator:
    def __init__(self) -> None:
        self.snapshot = _snapshot()
        self.submissions: list[dict[str, Any]] = []
        self.resets: list[str | None] = []
        self.closed = False

    def submit(self, jpeg: bytes, **metadata: Any) -> bool:
        self.submissions.append({"jpeg": jpeg, **metadata})
        return True

    def reset(self, stream_session_id: str | None = None) -> None:
        self.resets.append(stream_session_id)

    def select(
        self,
        stream_session_id: str,
        timestamp_seconds: float,
        *,
        frame_sequence: int | None = None,
        interpolate: bool = True,
    ) -> RichLabelSelection:
        assert stream_session_id == "stream-rich"
        assert timestamp_seconds == 1.0
        assert frame_sequence == 4
        return RichLabelSelection(self.snapshot, "exact", 4, 4)

    def latest(self, stream_session_id: str | None = None) -> RichLabelSnapshot:
        assert stream_session_id == "stream-rich"
        return self.snapshot

    def stats(self) -> dict[str, Any]:
        return {"state": "ready", "submitted": len(self.submissions)}

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        assert timeout_seconds == 5.0
        self.closed = True


def _runtime(coordinator: _RichCoordinator) -> HarnessRuntime:
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        rich_label_coordinator=coordinator,  # type: ignore[arg-type]
    )
    runtime.autostart = False
    runtime._active_rich_session_id = "stream-rich"
    return runtime


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        message_id="event-rich",
        source_id="camera",
        stream_session_id="stream-rich",
        sequence=1,
        emitted_at="2026-08-30T00:00:01.000Z",
        frame=FrameReference(
            source_frame_index=4,
            timestamp_seconds=1.0,
            observed_at="2026-08-30T00:00:00.000Z",
            width=240,
            height=160,
        ),
        judgments=(
            EventJudgment(
                event_label="fight",
                detected=True,
                confidence=0.91,
                description="Fight evidence.",
                object_ids=("event-person",),
            ),
        ),
        objects=(
            GroundedObject(
                object_id="event-person",
                label="person",
                bbox=NormalizedBBox(0.21, 0.16, 0.74, 0.89),
                confidence=0.9,
            ),
        ),
        model_name="test-model",
        model_latency_seconds=0.2,
    )


def _smoking_stranger_analysis() -> AnalysisResult:
    return AnalysisResult(
        message_id="event-smoking-stranger",
        source_id="camera",
        stream_session_id="stream-rich",
        sequence=2,
        emitted_at="2026-08-30T00:00:01.000Z",
        frame=FrameReference(
            source_frame_index=4,
            timestamp_seconds=1.0,
            observed_at="2026-08-30T00:00:00.000Z",
            width=640,
            height=480,
        ),
        judgments=(
            EventJudgment(
                event_label="smoking",
                detected=True,
                confidence=0.95,
                description="A person is holding a cigarette.",
                object_ids=("smoking-person", "cigarette"),
            ),
            EventJudgment(
                event_label="stranger",
                detected=True,
                confidence=0.875,
                description="Unknown face detected.",
                object_ids=("unknown-face",),
            ),
        ),
        objects=(
            GroundedObject(
                object_id="smoking-person",
                label="person",
                bbox=NormalizedBBox(0.381, 0.344, 0.906, 1.0),
                confidence=0.98,
            ),
            GroundedObject(
                object_id="cigarette",
                label="cigarette",
                bbox=NormalizedBBox(0.781, 0.644, 0.862, 0.738),
                confidence=0.92,
            ),
            GroundedObject(
                object_id="unknown-face",
                label="unknown_person",
                bbox=NormalizedBBox(0.615527, 0.434649, 0.775671, 0.709401),
                confidence=0.875,
            ),
        ),
        model_name="test-model",
        model_latency_seconds=0.2,
        analysis_kind="selected_features",
    )


def test_runtime_queues_display_frames_and_renders_rich_labels() -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    source = _jpeg()
    try:
        runtime._on_display_frame(source, 1.0, 4)

        assert coordinator.submissions == [
            {
                "jpeg": source,
                "stream_session_id": "stream-rich",
                "frame_sequence": 4,
                "timestamp_seconds": 1.0,
            }
        ]
        _, preview = runtime.preview_frames.latest() or (0, None)
        assert preview is not None
        assert preview.jpeg == source

        rendered = runtime._render_rich_label_jpeg(source, 1.0, 4)
        image = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        assert np.count_nonzero(image) > 0
        assert runtime.status()["rich_labeling"]["state"] == "ready"
    finally:
        runtime.stop()
    assert coordinator.closed


def test_only_display_callbacks_disable_rich_interpolation() -> None:
    modes = []

    class Coordinator(_RichCoordinator):
        def select(self, *args: Any, **kwargs: Any) -> RichLabelSelection:
            modes.append(kwargs.get("interpolate", True))
            return super().select(*args, **kwargs)

    runtime = _runtime(Coordinator())
    source, selection = _jpeg(), OverlaySelection(None, "none")
    try:
        assert runtime._delayed_overlay._compose is not None
        runtime._delayed_overlay._compose(source, selection, 1.0, 4)
        runtime._render_rich_label_jpeg(source, 1.0, 4)
        runtime._render_fused_label_jpeg(source, selection, 1.0, 4)
        assert modes == [False, False, True]
    finally:
        runtime.stop()


def test_disabled_rich_display_has_no_composition_callback() -> None:
    config = load_config(ROOT / "config.yaml")
    runtime = HarnessRuntime(replace(
        config,
        rich_labeling=replace(
            config.rich_labeling, enabled=False, tracking_enabled=False,
        ),
    ))
    try:
        assert runtime._delayed_overlay._compose is None
        assert runtime._delayed_overlay._enrich is None
    finally:
        runtime.stop()


def test_exact_analysis_frame_fuses_alert_with_one_jpeg_codec_pass(
    monkeypatch: Any,
) -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    source = _jpeg()
    captured: dict[str, Any] = {}
    decode_calls = 0
    encode_calls = 0
    original_decode = service_module.cv2.imdecode
    original_encode = service_module.cv2.imencode

    def count_decode(*args: Any, **kwargs: Any) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    def count_encode(*args: Any, **kwargs: Any) -> Any:
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(*args, **kwargs)

    def capture_rich(image: Any, objects: Any, **kwargs: Any) -> Any:
        captured["rich_objects"] = tuple(objects)
        captured["alerts"] = kwargs.get("alerts_by_track_id")
        return image.copy()

    def capture_event(image: Any, **kwargs: Any) -> Any:
        captured["event_objects"] = tuple(kwargs["objects"])
        captured["face_objects"] = tuple(kwargs["face_objects"])
        return image.copy()

    monkeypatch.setattr(service_module.cv2, "imdecode", count_decode)
    monkeypatch.setattr(service_module.cv2, "imencode", count_encode)
    monkeypatch.setattr(service_module, "draw_rich_overlay", capture_rich)
    monkeypatch.setattr(service_module, "draw_overlay", capture_event)
    try:
        runtime._on_analysis_frame(_analysis(), source)

        assert [item.track_id for item in captured["rich_objects"]] == ["P0001"]
        assert [
            alert.event_label for alert in captured["alerts"]["P0001"]
        ] == ["fight"]
        assert captured["event_objects"] == ()
        assert decode_calls == 1
        assert encode_calls == 1
        _, published = runtime.frames.latest() or (0, None)
        assert published is not None
        assert published.analysis_message_id == "event-rich"
    finally:
        runtime.stop()


def test_face_reference_box_stays_independent_from_rich_body_track(
    monkeypatch: Any,
) -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    original = _analysis()
    face_reference = replace(
        original,
        analysis_kind="face_reference_match",
        judgments=(
            EventJudgment(
                event_label="unknown_person",
                detected=True,
                confidence=0.91,
                description="Unknown face evidence.",
                object_ids=("event-person",),
            ),
        ),
        objects=(replace(original.objects[0], label="unknown_person"),),
    )
    captured: dict[str, Any] = {}

    def fail_fusion(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("face-reference result reached fusion matcher")

    def capture_rich(image: Any, objects: Any, **kwargs: Any) -> Any:
        captured["rich_objects"] = tuple(objects)
        captured["rich_kwargs"] = kwargs
        return image.copy()

    def capture_event(image: Any, **kwargs: Any) -> Any:
        captured["event_objects"] = tuple(kwargs["objects"])
        captured["face_objects"] = tuple(kwargs["face_objects"])
        return image.copy()

    monkeypatch.setattr(runtime._overlay_fusion, "plan", fail_fusion)
    monkeypatch.setattr(service_module, "draw_rich_overlay", capture_rich)
    monkeypatch.setattr(service_module, "draw_overlay", capture_event)
    try:
        runtime._on_timeline_result(face_reference)
        runtime._on_analysis_frame(face_reference, _jpeg())

        assert [item.track_id for item in captured["rich_objects"]] == ["P0001"]
        assert captured["rich_kwargs"] == {}
        assert captured["event_objects"] == ()
        assert [item.object_id for item in captured["face_objects"]] == [
            "event-person"
        ]
    finally:
        runtime.stop()


def test_selected_stranger_has_independent_face_box_and_smoking_uses_rich_fallback(
    monkeypatch: Any,
) -> None:
    coordinator = _RichCoordinator()
    coordinator.snapshot = RichLabelSnapshot(
        stream_session_id="stream-rich",
        frame_sequence=4,
        timestamp_seconds=1.0,
        objects=(
            RichLabelObject(
                track_id="P0001",
                label="person",
                bbox=NormalizedBBox(0.37, 0.33, 0.92, 1.0),
                confidence=0.94,
            ),
        ),
    )
    runtime = _runtime(coordinator)
    captured: dict[str, Any] = {}

    def capture_rich(image: Any, objects: Any, **kwargs: Any) -> Any:
        captured["rich_objects"] = tuple(objects)
        captured["track_alerts"] = kwargs.get("alerts_by_track_id")
        return image.copy()

    def capture_fallback(image: Any, objects: Any, **kwargs: Any) -> Any:
        captured["fallback_objects"] = tuple(objects)
        captured["fallback_alerts"] = kwargs["alerts_by_object_id"]
        return image.copy()

    def capture_summary(image: Any, **kwargs: Any) -> Any:
        captured["summary_objects"] = tuple(kwargs["objects"])
        captured["face_objects"] = tuple(kwargs["face_objects"])
        return image.copy()

    monkeypatch.setattr(service_module, "draw_rich_overlay", capture_rich)
    monkeypatch.setattr(
        service_module,
        "draw_event_rich_overlay",
        capture_fallback,
    )
    monkeypatch.setattr(service_module, "draw_overlay", capture_summary)
    try:
        runtime._on_analysis_frame(_smoking_stranger_analysis(), _jpeg())

        assert [item.track_id for item in captured["rich_objects"]] == ["P0001"]
        assert captured["track_alerts"] == {}
        assert [item.label for item in captured["face_objects"]] == ["unknown_person"]
        assert [
            item.object_id for item in captured["fallback_objects"]
        ] == ["cigarette"]
        assert [
            alert.event_label
            for alert in captured["fallback_alerts"]["cigarette"]
        ] == ["smoking"]
        assert captured["summary_objects"] == ()
    finally:
        runtime.stop()


def test_exact_analysis_frame_falls_back_to_event_when_rich_select_fails(
    monkeypatch: Any,
) -> None:
    class _FailingSelectCoordinator(_RichCoordinator):
        def select(self, *args: Any, **kwargs: Any) -> RichLabelSelection:
            del args, kwargs
            raise RuntimeError("rich history unavailable")

    coordinator = _FailingSelectCoordinator()
    runtime = _runtime(coordinator)
    captured_objects: list[tuple[GroundedObject, ...]] = []

    def capture_event(image: Any, **kwargs: Any) -> Any:
        captured_objects.append(tuple(kwargs["objects"]))
        return image.copy()

    monkeypatch.setattr(service_module, "draw_overlay", capture_event)
    try:
        runtime._on_analysis_frame(_analysis(), _jpeg())

        assert len(captured_objects) == 1
        assert [item.object_id for item in captured_objects[0]] == ["event-person"]
        _, published = runtime.frames.latest() or (0, None)
        assert published is not None
    finally:
        runtime.stop()


@pytest.mark.parametrize("event_render_fails", [False, True])
def test_exact_analysis_fusion_failure_preserves_original_frame_fallback(
    monkeypatch: Any,
    event_render_fails: bool,
) -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    source = _jpeg()
    event_inputs: list[bytes] = []
    event_objects: list[tuple[GroundedObject, ...]] = []
    rich_calls: list[Any] = []
    original_render = service_module._render_analysis_jpeg

    def fail_fusion(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fusion unavailable")

    def capture_rich(image: Any, *args: Any, **kwargs: Any) -> Any:
        rich_calls.append(image)
        return image.copy()

    def capture_event(image: Any, **kwargs: Any) -> Any:
        event_objects.append(tuple(kwargs["objects"]))
        return image.copy()

    def capture_render(jpeg: bytes, *args: Any, **kwargs: Any) -> bytes:
        event_inputs.append(jpeg)
        if event_render_fails:
            raise ValueError("event renderer unavailable")
        return original_render(jpeg, *args, **kwargs)

    monkeypatch.setattr(runtime._overlay_fusion, "plan", fail_fusion)
    monkeypatch.setattr(service_module, "draw_rich_overlay", capture_rich)
    monkeypatch.setattr(service_module, "draw_overlay", capture_event)
    monkeypatch.setattr(service_module, "_render_analysis_jpeg", capture_render)
    try:
        result = _analysis()
        runtime._on_analysis_frame(result, source)

        assert event_inputs == [source]
        assert rich_calls == []
        if event_render_fails:
            assert event_objects == []
        else:
            assert len(event_objects) == 1
            assert [item.object_id for item in event_objects[0]] == ["event-person"]
            assert event_objects[0][0].bbox == result.objects[0].bbox
        version, published = runtime.frames.latest() or (0, None)
        assert version == 1
        assert published is not None
        assert published.frame_sequence == result.frame.source_frame_index
        assert published.media_timestamp == result.frame.timestamp_seconds
        assert published.analysis_message_id == result.message_id
        if event_render_fails:
            assert published.jpeg == source
    finally:
        runtime.stop()


def test_latest_rich_labels_endpoint_returns_worker_snapshot() -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/v1/rich-labels/latest")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["objects"][0]["track_id"] == "P0001"
    assert response.json()["objects"][0]["attributes"] == {
        "estimated_age_range": "18-60",
        "estimated_gender": "female-presenting",
    }


def test_rich_label_queue_failure_does_not_interrupt_live_preview() -> None:
    class _FailingCoordinator(_RichCoordinator):
        def submit(self, jpeg: bytes, **metadata: Any) -> bool:
            del jpeg, metadata
            raise RuntimeError("worker unavailable")

    coordinator = _FailingCoordinator()
    runtime = _runtime(coordinator)
    source = _jpeg()
    try:
        runtime._on_display_frame(source, 1.0, 4)
        _, preview = runtime.preview_frames.latest() or (0, None)
        assert preview is not None
        assert preview.jpeg == source
    finally:
        runtime.stop()


@pytest.mark.parametrize("backend", ["paddle", "vlm"])
@pytest.mark.parametrize("model", [None, "annotation-model"])
@pytest.mark.parametrize("prompt_cache", [False, True])
def test_runtime_selects_independent_rich_annotation_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    model: str | None,
    prompt_cache: bool,
) -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        rich_labeling=replace(
            config.rich_labeling,
            enabled=True,
            backend=backend,
            model=model,
            timeout_seconds=18,
            vlm_max_tokens=8192,
            vlm_max_image_side=960,
            vlm_prompt_cache=prompt_cache,
        ),
    )
    created: list[tuple[str, Any, dict[str, Any]]] = []

    def paddle_client(endpoint: str, **kwargs: Any) -> object:
        created.append(("paddle", endpoint, kwargs))
        return SimpleNamespace(gpu_ready=lambda: True)

    def vlm_client(settings: Any, **kwargs: Any) -> object:
        created.append(("vlm", settings, kwargs))
        return object()

    monkeypatch.setattr(service_module, "RichLabelHTTPClient", paddle_client)
    monkeypatch.setattr(service_module, "VLMRichLabelClient", vlm_client)
    runtime = HarnessRuntime(config)
    try:
        assert len(created) == 1
        selected_backend, settings, options = created[0]
        assert selected_backend == backend
        if backend == "vlm":
            assert settings == replace(
                config.aiclub, timeout_seconds=18, max_retries=0
            )
            assert options["model"] == (model or config.inference.model)
            assert options["max_tokens"] == 8192
            assert options["max_image_side"] == 960
            assert options["prompt_cache"] is prompt_cache
            assert callable(options["api_key"])
            assert callable(options["dry_run"])
        else:
            assert settings == config.rich_labeling.endpoint
            assert options == {
                "timeout_seconds": 18, "mode": "tracks", "include_attributes": True,
            }

        info = runtime.ui_info()
        assert info["rich_labeling"]["capability"] == "rich_annotation"
        assert info["rich_labeling"]["enabled"] is True
        assert "backend" not in info["rich_labeling"]
        assert "model" not in info["rich_labeling"]
        assert info["supported_features"] == list(DETECTION_EVENT_LABELS)
    finally:
        runtime.stop()


def test_vlm_runtime_starts_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        rich_labeling=replace(config.rich_labeling, enabled=True, backend="vlm"),
    )
    for name in (
        "INFERENCE_API_KEY",
        config.aiclub.api_key_env,
        "AICLUB_API_KEY",
        "MODEL_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime = HarnessRuntime(config)
    try:
        assert runtime.ui_info()["api_key_present"] is False
        assert runtime.status()["rich_labeling"]["failed"] == 0
    finally:
        runtime.stop()


@pytest.mark.parametrize("session_dry_run", [True, False])
def test_vlm_callbacks_follow_deployment_changes_and_session_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_dry_run: bool,
) -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        rich_labeling=replace(config.rich_labeling, enabled=True, backend="vlm"),
    )
    for name in (
        "INFERENCE_API_KEY",
        config.aiclub.api_key_env,
        "AICLUB_API_KEY",
        "MODEL_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    captured: dict[str, Any] = {}

    def vlm_client(settings: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    def event_client(settings: Any, *args: Any, **kwargs: Any) -> Any:
        return DryRunGroundingClient(settings)

    monkeypatch.setattr(service_module, "VLMRichLabelClient", vlm_client)
    monkeypatch.setattr(service_module, "build_grounding_client", event_client)
    runtime = HarnessRuntime(
        config,
        deployment_store=DeploymentSettingsStore(tmp_path / "deployment.json"),
    )
    try:
        assert captured["api_key"]() is None
        runtime.update_deployment_settings(
            {"api_key": "annotation-test-key", "dry_run": not session_dry_run}
        )
        assert captured["api_key"]() == "annotation-test-key"
        assert captured["dry_run"]() is not session_dry_run
        runtime.update_deployment_settings({"clear_api_key": True})
        assert captured["api_key"]() is None

        if not session_dry_run:
            with pytest.raises(RuntimeError, match="missing model API key"):
                runtime.start_camera(
                    dry_run=False, analysis_mode="event", owner="rich-vlm-test",
                )
            runtime.update_deployment_settings({"api_key": "annotation-test-key"})

        runtime.start_camera(
            dry_run=session_dry_run,
            analysis_mode="event",
            owner="rich-vlm-test",
        )
        assert captured["dry_run"]() is session_dry_run

        _stop_and_wait(runtime)
        assert captured["dry_run"]() is not session_dry_run
    finally:
        runtime.stop()


@pytest.fixture
def model_selection_rig(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.delenv("CLOUD_GPU_CONTROLLER_URL", raising=False)
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        rich_labeling=replace(
            config.rich_labeling, enabled=True, backend="sam31_stream",
            processing_method="sam_live",
            endpoint="http://tracking.test/v1/perception/sam_live/sessions",
        ),
        runtime=replace(config.runtime, scene_mode="active"),
    )
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    requests, detector_clients, tracking_clients, runtimes = [], [], [], []
    control = SimpleNamespace(ready=True, probe=None, provider_health=None)

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            ready = control.probe(request) if control.probe else control.ready
            if control.provider_health is not None:
                return httpx.Response(200, json=control.provider_health)
            return httpx.Response(200, json={"ready": ready, "gpu_ready": ready})
        return httpx.Response(200, json={
            "stream_session_id": request.headers[STREAM_SESSION_HEADER],
            "frame_sequence": int(request.headers[FRAME_SEQUENCE_HEADER]),
            "timestamp_seconds": float(request.headers[TIMESTAMP_HEADER]),
            "objects": [],
        })

    def detector_client(endpoint, **options):
        client = RichLabelHTTPClient(
            endpoint, transport=httpx.MockTransport(handle), **options,
        )
        detector_clients.append(client)
        return client

    class TrackingClient:
        startup_frames = 1
        max_append_frames = 4

        def __init__(self, rich, **kwargs):
            self.rich = rich
            self.closed = False
            tracking_clients.append(self)

        def gpu_ready(self):
            return True

        def close(self):
            self.closed = True

        def close_complete(self):
            return self.closed

        def discard_session(self):
            pass

        def transport_stats(self):
            return {}

    monkeypatch.setattr(service_module, "RichLabelHTTPClient", detector_client)
    monkeypatch.setattr(service_module, "create_tracking_client", TrackingClient)

    def make():
        runtime = HarnessRuntime(config, deployment_store=store)
        runtimes.append(runtime)
        return runtime

    yield SimpleNamespace(
        make=make, store=store, config=config, requests=requests,
        detector_clients=detector_clients, tracking_clients=tracking_clients,
        control=control,
    )
    for runtime in runtimes:
        runtime.stop()


def test_saved_detector_switch_rebuilds_existing_client_and_survives_restart(
    model_selection_rig,
):
    rig = model_selection_rig
    rig.store.save(DeploymentSettings("", True, False, rich_label_model="det-fast"))
    runtime = rig.make()
    old = runtime._rich_labels
    assert runtime.config.rich_labeling.endpoint == "http://tracking.test/v1/labels"
    runtime.update_deployment_settings({"rich_label_model": "det-balanced"})
    assert old.close_complete()
    assert runtime._rich_labels is None
    assert runtime._try_enable_rich_labels()
    assert runtime._rich_labels is not old
    current = rig.detector_clients[-1]
    runtime._rich_labels.reset("after-switch")
    current.infer(b"jpeg", stream_session_id="after-switch", frame_sequence=1,
                  timestamp_seconds=0)
    assert rig.requests[-1].url.path == "/v1/tracks"
    assert rig.requests[-1].headers[DETECTOR_MODEL_HEADER] == "rfdetr-l"
    restored = rig.make()
    assert restored.config.rich_labeling == runtime.config.rich_labeling
    assert rig.requests[-1].headers[DETECTOR_MODEL_HEADER] == "rfdetr-l"


def test_detector_selection_and_location_changes_retire_once_and_restore_contour(
    model_selection_rig,
):
    rig = model_selection_rig
    runtime = rig.make()
    contour = runtime._rich_labels
    runtime.update_deployment_settings({"rich_label_model": "det-fast"})
    assert contour.close_complete()
    assert rig.tracking_clients[0].closed
    assert runtime._remote_config.rich_labeling.processing_method == "sam_live"
    assert runtime.config.rich_labeling.backend == "paddle"
    assert runtime.config.rich_labeling.processing_method is None
    assert runtime._try_enable_rich_labels()
    detector = runtime._rich_labels
    revision = runtime._rich_config_revision
    runtime.update_deployment_settings({
        "rich_label_model": "det-accurate", "tracking_mode": "server",
        "tracking_server_url": "http://new-worker.test:8093",
        "tracking_server_protocol": "contour",
    })
    assert detector.close_complete()
    assert runtime._rich_config_revision == revision + 1
    assert runtime.config.rich_labeling.backend == "paddle"
    assert runtime.config.rich_labeling.endpoint == "http://new-worker.test:8093/v1/labels"
    assert runtime._try_enable_rich_labels()
    runtime.update_deployment_settings({"tracking_sample_fps": 3})
    assert runtime.config.rich_labeling.backend == "paddle"
    assert runtime.config.rich_labeling.processing_method is None
    assert runtime._try_enable_rich_labels()
    runtime.update_deployment_settings({"rich_label_model": "contour"})
    assert runtime.config.rich_labeling.backend == "sam31_stream"
    assert runtime.config.rich_labeling.processing_method == "sam_live"
    assert runtime.config.rich_labeling.endpoint == "http://new-worker.test:8093/v1/contour/sessions"
    assert runtime._detector_model_headers() == {}
    assert runtime._try_enable_rich_labels()
    assert rig.tracking_clients[-1].rich == runtime.config.rich_labeling


def test_detector_readiness_cannot_install_client_from_previous_model_selection(
    model_selection_rig,
):
    rig = model_selection_rig
    rig.store.save(DeploymentSettings("", True, False, rich_label_model="det-fast"))
    rig.control.ready = False
    runtime = rig.make()
    entered, release = Event(), Event()

    def blocked(request):
        entered.set()
        assert release.wait(3)
        return True

    rig.control.probe = blocked
    probe = Thread(target=runtime._try_enable_rich_labels)
    probe.start()
    try:
        assert entered.wait(1)
        runtime.update_deployment_settings({"rich_label_model": "det-balanced"})
        release.set()
        probe.join(2)
        assert not probe.is_alive()
        assert runtime._rich_labels is None
        assert all(client._closed for client in rig.detector_clients)
        rig.control.probe = None
        rig.control.ready = True
        assert runtime._try_enable_rich_labels()
        assert rig.requests[-1].headers[DETECTOR_MODEL_HEADER] == "rfdetr-l"
    finally:
        release.set()
        probe.join(3)


def test_optional_segmentation_selection_uses_coreml_readiness(model_selection_rig):
    rig = model_selection_rig
    rig.control.provider_health = {
        "ready": True, "gpu_ready": False, "model": "sam3-onnx",
        "provider_ready": True, "device": "coreml",
        "execution_providers": {
            key: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            for key in ("image_encoder", "language_encoder", "decoder")
        },
    }
    rig.store.save(DeploymentSettings("", True, False, rich_label_model="det-segment"))
    runtime = rig.make()
    assert runtime._rich_labels is not None
    assert runtime.config.rich_labeling.backend == "paddle"
    assert runtime.config.rich_labeling.processing_method is None
    assert rig.requests[-1].headers[DETECTOR_MODEL_HEADER] == "sam3-onnx"
    assert rig.requests[-1].url.path == "/health/tracking"
    restored = rig.make()
    assert restored.config.rich_labeling == runtime.config.rich_labeling
    assert restored._rich_labels is not None


def test_optional_segmentation_unready_does_not_require_cuda(model_selection_rig):
    rig = model_selection_rig
    rig.control.provider_health = {
        "ready": False, "gpu_ready": False, "model": "sam3-onnx",
        "provider_ready": False, "error": "missing local model",
    }
    rig.store.save(DeploymentSettings("", True, False, rich_label_model="det-segment"))
    runtime = rig.make()
    assert runtime._rich_labels is None
    rich_status = runtime.status()["rich_labeling"]
    assert rich_status["disabled_reason"] == "provider_unavailable"
