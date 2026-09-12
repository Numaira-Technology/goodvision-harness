"""Actionable missing-face startup errors without exposing dependency exceptions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from goodvision_harness import face_recognition, service
from goodvision_harness.config import load_config
from goodvision_harness.grounding import DETECTION_EVENT_LABELS
from goodvision_harness.privacy import (
    PUBLIC_FACE_UNAVAILABLE_ERROR,
    LocalFaceUnavailableError,
    public_error_message,
)
from goodvision_harness.rich_labeling import RichLabelCoordinator


class NoInference:
    def infer(self, *_args, **_kwargs):
        raise AssertionError("failed startup must not run inference")

    def close(self):
        pass


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_kwargs: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(
        config,
        runtime=replace(config.runtime, scene_mode="active"),
        rich_labeling=replace(
            config.rich_labeling, enabled=False, tracking_enabled=False
        ),
        face=replace(config.face, gallery_path=tmp_path / "gallery.json"),
        events=replace(config.events, directory=tmp_path / "events"),
        service=replace(
            config.service, webhook=replace(config.service.webhook, url=None)
        ),
    )
    selected = service.HarnessRuntime(
        config, rich_label_coordinator=RichLabelCoordinator(NoInference())
    )
    selected._aiclub_api_key = "test-only-key"
    real_import = face_recognition.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "insightface.app":
            raise ModuleNotFoundError("private-key=SECRET private-model-directory")
        return real_import(name, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must not silently remove stranger and continue startup")

    monkeypatch.setattr(face_recognition.importlib, "import_module", missing)
    monkeypatch.setattr(service, "build_selected_features_grounding_client", forbidden)
    monkeypatch.setattr(service, "TemporalVLMInspector", forbidden)
    try:
        yield selected
    finally:
        selected.stop()


@pytest.mark.parametrize("boundary", ["websocket", "http"])
def test_formal_all_feature_start_reports_missing_local_face_components(
    runtime, boundary
):
    features = list(DETECTION_EVENT_LABELS)
    assert "stranger" in features
    payload = {
        "type": "start",
        "analysis_mode": "multi",
        "features": features,
        "dry_run": False,
    }
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        if boundary == "websocket":
            with client.websocket_connect("/ws") as websocket:
                assert websocket.receive_json()["type"] == "hello"
                websocket.send_json(payload | {"mode": "camera"})
                response = websocket.receive_json()
                assert response == {
                    "type": "error",
                    "message": PUBLIC_FACE_UNAVAILABLE_ERROR,
                }
        else:
            response = client.post(
                "/v1/sessions/start",
                json=payload
                | {
                    "mode": "stream",
                    "stream_url": "rtmp://camera.example.test/live",
                },
            )
            assert response.status_code == 503
            assert response.json()["detail"] == PUBLIC_FACE_UNAVAILABLE_ERROR
            response = response.json()
        assert runtime._session is None
        assert runtime._face_runtime is not None
        assert runtime.status()["features"] == features
        assert "SECRET" not in str(response)
        assert "private-model" not in str(response)


def test_known_exception_is_typed_and_unknown_exception_text_stays_private(runtime):
    with pytest.raises(LocalFaceUnavailableError) as caught:
        runtime.start_camera(dry_run=False, features=list(DETECTION_EVENT_LABELS))
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert (
        public_error_message(caught.value, fallback="could not start session")
        == PUBLIC_FACE_UNAVAILABLE_ERROR
    )
    assert (
        public_error_message(
            RuntimeError("local face analysis is unavailable; private-key=SECRET"),
            fallback="could not start session",
        )
        == "could not start session"
    )


@pytest.mark.parametrize("boundary", ["websocket", "http"])
def test_unrelated_startup_errors_keep_existing_redacted_fallback(
    runtime, monkeypatch, boundary
):
    def fail(*_args, **_kwargs):
        raise RuntimeError("private-key=SECRET unrelated failure")

    monkeypatch.setattr(runtime, "_ensure_face_runtime", fail)
    payload = {
        "type": "start",
        "analysis_mode": "multi",
        "features": list(DETECTION_EVENT_LABELS),
        "dry_run": False,
    }
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        if boundary == "websocket":
            with client.websocket_connect("/ws") as websocket:
                websocket.receive_json()
                websocket.send_json(payload | {"mode": "camera"})
                response = websocket.receive_json()
                assert response == {
                    "type": "error",
                    "message": "could not start session",
                }
        else:
            response = client.post(
                "/v1/sessions/start",
                json=payload
                | {
                    "mode": "stream",
                    "stream_url": "rtmp://camera.example.test/live",
                },
            )
            assert response.status_code == 503
            assert response.json()["detail"] == "could not start session"
