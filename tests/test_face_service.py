from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.config import InferenceSettings, load_config
from goodvision_harness.contracts import (
    PUBLIC_MODEL_NAME,
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
    GalleryPerson,
)
from goodvision_harness.reference_memory import MemoryContext, MemoryReference
from goodvision_harness.service import (
    FACE_ENROLLMENT_MAX_BYTES,
    HarnessRuntime,
    create_app,
)
from goodvision_harness.visual_memory import VisualMemoryRecord

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


def _jpeg(width: int = 320, height: int = 180) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


class _FakeFaceRuntime:
    def __init__(self) -> None:
        self.settings = InferenceSettings(
            model="buffalo_l",
            event_label="unknown_person",
            system_prompt="unused",
            prompt="unused",
        )
        self.people = {"alice": GalleryPerson("alice", "Alice", 2)}
        self.last_enrollment: tuple[str, str, bytes] | None = None
        self.warmup_calls = 0

    def warmup(self) -> float:
        self.warmup_calls += 1
        return 0.01

    def list_people(self) -> tuple[GalleryPerson, ...]:
        return tuple(self.people.values())

    def enroll(
        self,
        person_id: str,
        name: str,
        image_bytes: bytes,
    ) -> GalleryPerson:
        self.last_enrollment = (person_id, name, image_bytes)
        person = GalleryPerson(person_id, name, 1)
        self.people[person_id] = person
        return person

    def delete_person(self, person_id: str) -> bool:
        return self.people.pop(person_id, None) is not None

    def classify_window(self, _window: Any) -> Any:
        raise AssertionError("this test does not submit a frame")


@pytest.mark.parametrize("name", ["Claude", "Gemini", "Google", "Amazon"])
def test_face_identity_round_trips_without_brand_redaction(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )
    with TestClient(create_app(lambda: runtime)) as client:
        enrolled = client.post(
            f"/v1/faces/gallery/{name}", params={"name": name}, content=_jpeg(),
            headers={"Content-Type": "image/jpeg"},
        )
        assert enrolled.status_code == 200
        identity = enrolled.json()["person"]
        assert (identity["person_id"], identity["name"]) == (name, name)
        gallery = client.get("/v1/faces/gallery").json()["people"]
        assert any(person == identity for person in gallery)
        deleted = client.delete(f"/v1/faces/gallery/{identity['person_id']}")
        assert deleted.json() == {"deleted": True, "person_id": name}
        assert name not in fake.people


def test_face_gallery_http_surface_and_ui_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )
    app = create_app(lambda: runtime)
    image = _jpeg()

    with TestClient(app) as client:
        info = client.get("/v1/ui/info").json()
        assert info["model"] == PUBLIC_MODEL_NAME
        assert info["analysis_modes"] == ["event", "face", "multi"]
        assert info["face"] == {
            "engine": "local",
            "sample_fps": 2.5,
            "max_image_side": 1280,
            "jpeg_quality": 90,
            "threshold": 0.4,
            "min_face_size": 20,
            "enrollment_min_face_size": 80,
        }

        dashboard = client.get("/")
        assert dashboard.status_code == 200
        assert 'id="face-controls" class="face-gallery"' in dashboard.text
        assert 'id="face-controls" class="face-gallery" hidden' not in dashboard.text
        assert 'id="face-analysis-mode"' not in dashboard.text
        assert 'id="event-analysis-mode"' not in dashboard.text
        assert 'id="face-diagnostic-summary"' in dashboard.text
        assert 'analysis_mode:"multi"' in dashboard.text
        assert 'visibleFeatures().includes("stranger")?' in dashboard.text

        gallery = client.get("/v1/faces/gallery")
        assert gallery.status_code == 200
        assert gallery.json()["people"][0]["samples"] == 2

        enrolled = client.post(
            "/v1/faces/gallery/bob?name=Bob",
            content=image,
            headers={"Content-Type": "image/jpeg"},
        )
        assert enrolled.status_code == 200
        assert enrolled.json()["person"]["person_id"] == "bob"
        assert fake.last_enrollment == ("bob", "Bob", image)

        strangers = client.get("/v1/faces/strangers")
        assert strangers.status_code == 410
        assert strangers.json()["detail"]["events_url"] == "/v1/events"
        photo = client.get("/v1/faces/strangers/stranger-test/photo")
        assert photo.status_code == 410
        assert client.get("/v1/faces/strangers/missing/photo").status_code == 410

        deleted = client.delete("/v1/faces/gallery/bob")
        assert deleted.json() == {"deleted": True, "person_id": "bob"}
        assert client.delete("/v1/faces/gallery/missing").status_code == 404


def test_face_enrollment_rejects_oversized_declared_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/faces/gallery/bob?name=Bob",
            content=b"not read",
            headers={"Content-Length": str(FACE_ENROLLMENT_MAX_BYTES + 1)},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "enrollment image is too large"}
    assert fake.last_enrollment is None


def test_face_enrollment_rejects_oversized_stream_without_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )
    app = create_app(lambda: runtime)
    half_limit = FACE_ENROLLMENT_MAX_BYTES // 2
    chunks = iter((b"a" * half_limit, b"b" * half_limit, b"c"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/faces/gallery/bob?name=Bob",
            content=chunks,
            headers={"Content-Type": "image/jpeg"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "enrollment image is too large"}
    assert fake.last_enrollment is None


def test_face_mode_reuses_streaming_session_with_face_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )

    try:
        response = runtime.start_camera(
            analysis_mode="face",
            dry_run=True,
            owner="face-test",
        )
        assert response["analysis_mode"] == "face"
        assert response["event_label"] == "unknown_person"
        assert response["model"] == PUBLIC_MODEL_NAME
        assert runtime.status()["model"] == PUBLIC_MODEL_NAME
        assert response["dry_run"] is False
        assert runtime.status()["analysis_mode"] == "face"
        assert runtime._session is not None
        assert runtime._session.config.video.max_image_side == 1280
        assert runtime._session.config.sliding_window.size_frames == 1
        assert runtime._session.config.runtime.max_concurrent_requests == 1
        assert runtime._session.client is fake
        assert fake.warmup_calls == 1
        runtime._on_pipeline_state(
            {"status": "warming", "last_error": "provider unavailable"}
        )
        assert runtime.status()["last_error"] == "analysis request failed"
    finally:
        runtime.stop()


@pytest.mark.parametrize("analysis_mode", ["face", "multi"])
def test_live_sessions_isolate_face_history_and_bind_rich_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, analysis_mode: str,
) -> None:
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    face_runtime = FaceRecognitionRuntime(FaceRecognizerConfig(
        gallery_path=tmp_path / "gallery.json",
    ))
    monkeypatch.setattr(face_runtime, "warmup", lambda: 0.0)
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config, events=replace(config.events, directory=tmp_path / "events"),
    )
    runtime = HarnessRuntime(
        config, face_runtime=face_runtime
    )
    clients = []
    try:
        for _ in range(2):
            runtime.start_camera(
                analysis_mode=analysis_mode,
                features=["stranger"] if analysis_mode == "multi" else None,
                dry_run=False,
            )
            assert runtime._session is not None
            client = runtime._session.client
            if analysis_mode == "multi":
                client = client._stranger_client
            assert isinstance(client, FaceRecognitionSession)
            assert client.runtime is face_runtime
            assert client._session_id == runtime._session.stream_session_id
            clients.append(client)
            _stop_and_wait(runtime)
        assert clients[0] is not clients[1]
        assert clients[0]._tracker is not clients[1]._tracker
    finally:
        runtime.stop()


def test_local_face_mode_keeps_insightface_runtime_with_reference_memory() -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        memory=MemoryContext(
            purpose="face_recognition",
            references=(
                MemoryReference(
                    memory_id="alice",
                    image_data_url="data:image/jpeg;base64,AA==",
                    width=1,
                    height=1,
                ),
            ),
            instruction="Match the supplied known face.",
        ),
    )
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(config, face_runtime=fake)  # type: ignore[arg-type]

    try:
        response = runtime.start_camera(analysis_mode="face", owner="face-test")

        assert response["analysis_mode"] == "face"
        assert response["event_label"] == "unknown_person"
        assert "reference_memory" not in response
        assert runtime._session is not None
        assert runtime._session.client is fake
        assert fake.warmup_calls == 1
    finally:
        runtime.stop()


def test_multi_dry_run_skips_face_model_warmup() -> None:
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )

    try:
        response = runtime.start_camera(
            analysis_mode="multi",
            dry_run=True,
            owner="multi-dry-run-test",
        )

        assert response["analysis_mode"] == "multi"
        assert response["dry_run"] is True
        assert fake.warmup_calls == 0
    finally:
        runtime.stop()


def test_selected_features_are_canonical_and_skip_unselected_face_runtime() -> None:
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        face_runtime=fake,  # type: ignore[arg-type]
    )

    try:
        response = runtime.start_camera(
            analysis_mode="multi",
            features=["smoking", "fight"],
            dry_run=True,
            owner="selected-test",
        )

        assert response["features"] == ["fight", "smoking"]
        assert response["event_labels"] == ["fight", "smoking"]
        assert runtime.status()["features"] == ["fight", "smoking"]
        assert runtime._session is not None
        assert runtime._session.client.event_labels == ("fight", "smoking")
        assert fake.warmup_calls == 0
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    (
        "features",
        "provider",
        "configured_scheduler",
        "expected_scheduler",
        "expected_concurrency",
    ),
    [
        (["stranger"], "CPUExecutionProvider", "auto", "single_flight", 1),
        (
            ["smoking", "stranger"],
            "CPUExecutionProvider",
            "auto",
            "single_flight",
            6,
        ),
        (["stranger"], "DmlExecutionProvider", "auto", "full", 6),
        (
            ["smoking", "stranger"],
            "CUDAExecutionProvider",
            "auto",
            "full",
            6,
        ),
        (["stranger"], "CPUExecutionProvider", "full", "full", 6),
        (
            ["smoking", "stranger"],
            "CUDAExecutionProvider",
            "single_flight",
            "single_flight",
            6,
        ),
    ],
)
def test_multi_face_live_scheduler_resolves_provider_and_override(
    features: list[str],
    provider: str,
    configured_scheduler: str,
    expected_scheduler: str,
    expected_concurrency: int,
) -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        face=replace(
            config.face,
            provider=provider,
            live_scheduler=configured_scheduler,
        ),
    )
    runtime = HarnessRuntime(config)

    try:
        runtime.start_camera(
            analysis_mode="multi",
            features=features,
            dry_run=True,
            owner="face-concurrency-test",
        )

        assert runtime._session is not None
        assert (
            runtime._session.config.runtime.max_concurrent_requests
            == expected_concurrency
        )
        assert runtime._session.client.face_live_scheduler == expected_scheduler
    finally:
        runtime.stop()


def test_face_only_full_scheduler_keeps_configured_concurrency() -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        face=replace(
            config.face,
            provider="DmlExecutionProvider",
            live_scheduler="auto",
        ),
    )
    fake = _FakeFaceRuntime()
    runtime = HarnessRuntime(config, face_runtime=fake)  # type: ignore[arg-type]

    try:
        runtime.start_camera(analysis_mode="face", owner="face-full-test")

        assert runtime._session is not None
        assert (
            runtime._session.config.runtime.max_concurrent_requests
            == config.runtime.max_concurrent_requests
        )
    finally:
        runtime.stop()


def test_session_http_api_accepts_feature_subsets_and_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        camera = client.post(
            "/v1/sessions/start",
            json={"mode": "camera", "dry_run": True, "features": ["smoking"]},
        )
        assert camera.status_code == 400
        assert camera.json()["detail"] == "camera sessions must be started over /ws"
        stream_request = {
            "mode": "stream",
            "stream_url": "rtsp://127.0.0.1:1/live",
            "dry_run": True,
        }
        assert client.post(
            "/v1/sessions/start",
            json={**stream_request, "features": []},
        ).status_code == 400
        assert client.post(
            "/v1/sessions/start",
            json={**stream_request, "features": "smoking"},
        ).status_code == 400
        assert client.post(
            "/v1/sessions/start",
            json={
                **stream_request,
                "features": ["smoking", "SMOKING"],
            },
        ).status_code == 400
        assert client.post(
            "/v1/sessions/start",
            json={
                **stream_request,
                "features": ["loitering"],
            },
        ).status_code == 400

        recorded: dict[str, Any] = {}

        def fake_start_stream(stream_url: str, **kwargs: Any) -> dict[str, Any]:
            recorded.update({"stream_url": stream_url, **kwargs})
            return {
                "type": "session",
                "mode": "stream",
                "features": ["smoking", "stranger"],
            }

        monkeypatch.setattr(runtime, "start_stream", fake_start_stream)
        started = client.post(
            "/v1/sessions/start",
            json={
                "mode": "stream",
                "stream_url": "rtsp://camera/live",
                "dry_run": True,
                "features": ["stranger", "smoking"],
            },
        )
        assert started.status_code == 200
        assert started.json()["features"] == ["smoking", "stranger"]
        assert recorded["stream_url"] == "rtsp://camera/live"
        assert recorded["features"] == ["stranger", "smoking"]


def test_face_reference_memory_uses_unknown_person_semantics() -> None:
    config = load_config(ROOT / "config.yaml")
    memory = MemoryContext(
        purpose="face_recognition",
        references=(
            MemoryReference(
                memory_id="alice",
                image_data_url="data:image/jpeg;base64,AA==",
                width=1,
                height=1,
            ),
        ),
        instruction="Use the supplied face as known identity context.",
    )
    runtime = HarnessRuntime(replace(config, memory=memory))

    try:
        response = runtime.start_camera(
            analysis_mode="event",
            event_label="known_face",
            dry_run=True,
            owner="memory-test",
        )

        assert response["reference_memory"]["memory_ids"] == ["alice"]
        assert response["event_label"] == "unknown_person"
        assert runtime._session is not None
        assert runtime._session.client.memory_context is memory
        assert runtime._session.config.inference.event_label == "unknown_person"
    finally:
        runtime.stop()


def test_analysis_frame_submits_exact_evidence_to_memory_sink() -> None:
    class _Sink:
        def __init__(self) -> None:
            self.records: list[VisualMemoryRecord] = []

        def submit(self, record: VisualMemoryRecord) -> bool:
            self.records.append(record)
            return True

    sink = _Sink()
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"), memory_sink=sink)
    jpeg = _jpeg()
    result = AnalysisResult(
        message_id="observation-1",
        source_id="camera",
        stream_session_id="session-1",
        sequence=1,
        emitted_at="2026-08-26T00:00:01.000Z",
        frame=FrameReference(
            source_frame_index=1,
            timestamp_seconds=1.0,
            observed_at="2026-08-26T00:00:00.000Z",
            width=320,
            height=180,
        ),
        judgments=(
            EventJudgment(
                event_label="unknown_person",
                detected=True,
                confidence=0.9,
                description="One unknown face.",
                object_ids=("object-1",),
            ),
        ),
        objects=(
            GroundedObject(
                object_id="object-1",
                label="unknown_person",
                bbox=NormalizedBBox(0.1, 0.1, 0.5, 0.8),
                confidence=0.9,
                attributes={"status": "unknown"},
            ),
        ),
        model_name="buffalo_l",
        model_latency_seconds=0.05,
        analysis_kind="face_recognition",
    )

    runtime._on_analysis_frame(result, jpeg)

    assert len(sink.records) == 1
    assert sink.records[0].evidence_jpeg == jpeg
    assert sink.records[0].observation["observation_id"] == "observation-1"
    assert sink.records[0].observation["analysis_kind"] == "face_recognition"
    assert runtime.status()["memory_records_accepted"] == 1
