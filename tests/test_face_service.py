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
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.face_recognition import GalleryPerson
from goodvision_harness.reference_memory import MemoryContext, MemoryReference
from goodvision_harness.service import (
    FACE_ENROLLMENT_MAX_BYTES,
    HarnessRuntime,
    create_app,
)
from goodvision_harness.visual_memory import VisualMemoryRecord

ROOT = Path(__file__).resolve().parents[1]


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
        assert info["analysis_modes"] == ["event", "face"]
        assert info["face"] == {
            "model_name": "buffalo_l",
            "sample_fps": 2.5,
            "max_image_side": 1280,
            "jpeg_quality": 90,
            "threshold": 0.4,
        }

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
        assert response["model"] == "buffalo_l"
        assert response["dry_run"] is False
        assert runtime.status()["analysis_mode"] == "face"
        assert runtime._session is not None
        assert runtime._session.config.video.max_image_side == 1280
        assert runtime._session.config.sliding_window.size_frames == 1
        assert runtime._session.config.runtime.max_concurrent_requests == 1
        assert runtime._session.client is fake
        runtime._on_pipeline_state(
            {"status": "warming", "last_error": "provider unavailable"}
        )
        assert runtime.status()["last_error"] == "provider unavailable"
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
    finally:
        runtime.stop()


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
