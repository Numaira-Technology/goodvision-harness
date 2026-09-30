"""A new installation can provision credentials entirely through the console."""

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.pipeline import StreamingSession
from goodvision_harness.privacy import PUBLIC_MISSING_API_KEY_ERROR
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


def test_no_env_console_key_save_and_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("INFERENCE_API_KEY", "OPENROUTER_API_KEY", "AICLUB_API_KEY", "MODEL_API_KEY",
                 "DASHSCOPE_API_KEY", "HARNESS_STREAM_URL", "RTSP_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "false")
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    # Use a standalone configuration and working directory without a .env.
    path = tmp_path / "config.yaml"
    path.write_bytes((ROOT / "config.delivery.yaml").read_bytes())
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    config = load_config(path)
    config = replace(config, events=replace(config.events, directory=tmp_path / "events"))
    store = DeploymentSettingsStore(tmp_path / "data" / "settings.json")
    runtime = HarnessRuntime(config, deployment_store=store)
    monkeypatch.setattr(StreamingSession, "start", lambda self: None)
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.get("/").status_code == 200
        assert "save-api-key-button" in client.get("/").text
        assert 'src="/assets/console_i18n.js"' in client.get("/").text
        language_script = client.get("/assets/console_i18n.js")
        assert language_script.status_code == 200
        assert "application/javascript" in language_script.headers["content-type"]
        assert "goodvision.console.language" in language_script.text
        info = client.get("/v1/ui/info").json()
        assert info["api_key_present"] is False
        assert info["deployment"]["persistence_available"] is True
        # Validate before face warmup or video capture, even with every feature selected.
        failed = client.post("/v1/sessions/start", json={"dry_run": False})
        assert failed.status_code == 400
        assert failed.json()["detail"] == PUBLIC_MISSING_API_KEY_ERROR
        assert runtime._session is None and runtime._face_runtime is None
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "start", "mode": "camera", "dry_run": False})
            assert socket.receive_json() == {"type": "error", "message": PUBLIC_MISSING_API_KEY_ERROR}
        saved = client.put("/v1/deployment/settings", json={"api_key": "offline-fixture-key"})
        assert saved.status_code == 200
        assert saved.json()["api_key_present"] and saved.json()["api_key_persisted"]
        assert "offline-fixture-key" not in saved.text
        assert "offline-fixture-key" not in client.get("/v1/ui/info").text
        # No frames are supplied: real client construction, with no external API calls.
        started = client.post("/v1/sessions/start", json={"features": ["smoking"], "dry_run": False})
        assert started.status_code == 200
    restored = HarnessRuntime(config, deployment_store=store)
    with TestClient(create_app(lambda: restored)) as client:
        assert client.get("/v1/ui/info").json()["api_key_present"]
        assert client.post("/v1/sessions/start", json={"features": ["smoking"], "dry_run": False}).status_code == 200
    assert not (tmp_path / ".env").exists()
