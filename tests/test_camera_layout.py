from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import goodvision_harness.config as config_module
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettings, DeploymentSettingsStore
from goodvision_harness.pipeline import StreamingSession
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setattr(StreamingSession, "start", lambda self: None)
    config = load_config(ROOT / "config.yaml")
    config = replace(config, events=replace(config.events, directory=tmp_path / "events"))
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    store.save(DeploymentSettings("rtsp://unused.local/live", True, False, 3.75))
    return config, store


@pytest.mark.parametrize("layout,crop", [("mono", "none"), ("stereo", "left_half")])
def test_camera_layout_persists_and_one_run_override_is_separate(configured, layout, crop):
    config, store = configured
    runtime = HarnessRuntime(config, deployment_store=store)
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.put("/v1/deployment/settings", json={"camera_layout": layout})
        assert response.status_code == 200
        assert response.json()["camera_layout"] == layout
        assert response.json()["stream_frame_crop"] == crop
        assert store.load().buffer_delay_seconds == 3.75
        assert client.get("/v1/ui/info").json()["camera_layout"] == layout
        opposite = "mono" if layout == "stereo" else "stereo"
        response = client.post("/v1/sessions/start", json={
            "camera_layout": opposite, "features": ["smoking"], "dry_run": True,
        })
        assert response.status_code == 200
        assert response.json()["camera_layout"] == opposite
        selected = runtime._session.frame_crop_selector.status()
        assert selected["effective"] == ("none" if opposite == "mono" else "left_half")
        assert selected["probed_frames"] == 0
        assert store.load().camera_layout == layout
    restored = HarnessRuntime(config, deployment_store=store)
    assert restored.configured_frame_crop() == crop
    restored.update_deployment_settings({"buffer_delay_seconds": 2})
    assert store.load().camera_layout == layout
    with TestClient(create_app(lambda: restored)) as client:
        response = client.post("/v1/sessions/start", json={"features": ["smoking"], "dry_run": True})
        assert response.json()["camera_layout"] == layout


def test_websocket_start_applies_manual_layout(configured):
    config, store = configured
    runtime = HarnessRuntime(config, deployment_store=store)
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "hello"
            socket.send_json({"type": "start", "mode": "stream", "camera_layout": "stereo",
                              "features": ["smoking"], "dry_run": True})
            response = socket.receive_json()
            assert response["type"] == "session"
            assert response["camera_layout"] == "stereo"
            status = client.get("/v1/status").json()
            assert status["stream"]["frame_crop"] == "left_half"
            assert status["stream"]["crop_selection"]["reason"] == "manual"


@pytest.mark.parametrize("invalid", ["auto", "", "left_half", True, 1, []])
def test_invalid_layout_cannot_change_settings_or_start(configured, invalid):
    config, store = configured
    runtime = HarnessRuntime(config, deployment_store=store)
    original = store.path.read_bytes()
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.put("/v1/deployment/settings", json={"camera_layout": invalid}).status_code == 400
        assert client.post("/v1/sessions/start", json={"camera_layout": invalid}).status_code == 400
        assert store.path.read_bytes() == original
        assert runtime._session is None


def test_running_session_rejects_layout_changes_but_preserves_it_on_buffer_update(configured):
    config, store = configured
    runtime = HarnessRuntime(config, deployment_store=store)
    runtime.update_deployment_settings({"camera_layout": "stereo"})
    class Running:
        running = True
    runtime._session = Running()
    try:
        with pytest.raises(RuntimeError, match="stop the active session"):
            runtime.update_deployment_settings({"camera_layout": "mono"})
        runtime.update_deployment_settings({"buffer_delay_seconds": 1.5})
        assert store.load().camera_layout == "stereo"
    finally:
        runtime._session = None


def test_old_auto_config_and_settings_default_to_mono(configured, tmp_path):
    config, store = configured
    legacy = replace(config, stream=replace(config.stream, frame_crop="auto"))
    runtime = HarnessRuntime(legacy, deployment_store=store)
    assert runtime.configured_camera_layout() == "mono"
    assert runtime.configured_frame_crop() == "none"
    path = tmp_path / "old-auto.yaml"
    path.write_text((ROOT / "config.yaml").read_text(encoding="utf-8").replace(
        "frame_crop: none", "frame_crop: auto"), encoding="utf-8")
    assert load_config(path).stream.frame_crop == "none"
