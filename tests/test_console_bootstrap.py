"""A new installation can provision credentials entirely through the console."""

from dataclasses import replace
from pathlib import Path

from fake_session_owner import FakeSessionOwner
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.privacy import PUBLIC_MISSING_API_KEY_ERROR
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


def test_production_console_persists_by_default_and_rejects_synthetic_requests(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_CONFIG", str(ROOT / "config.delivery.yaml"))
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    monkeypatch.delenv("HARNESS_DEPLOYMENT_SETTINGS_PATH", raising=False)
    runtime = HarnessRuntime.from_environment()
    with TestClient(create_app(lambda: runtime)) as client:
        info = client.get("/v1/ui/info").json()
        assert info["dry_run"] is False
        assert info["deployment"]["persistence_available"] is True
        html = client.get("/").text
        assert 'id="default-dry-run"' not in html
        assert 'id="dry-run"' not in html
        assert "Test mode" not in html
        assert client.put(
            "/v1/deployment/settings", json={"buffer_delay_seconds": 3.25},
        ).status_code == 200
        for endpoint, method in (
            ("/v1/deployment/settings", client.put),
            ("/v1/sessions/start", client.post),
        ):
            response = method(endpoint, json={"dry_run": True})
            assert response.status_code == 400
            assert response.json() == {
                "detail": "Synthetic analysis is no longer supported.",
            }
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({"type": "start", "mode": "camera", "dry_run": True})
            assert socket.receive_json() == {
                "type": "error", "message": "Synthetic analysis is no longer supported.",
            }
        assert runtime._session is None
    restored = HarnessRuntime.from_environment()
    assert restored.deployment_settings()["buffer_delay_seconds"] == 3.25
    assert restored.dry_run is False
    assert (tmp_path / ".goodvision" / "deployment-settings.json").is_file()


def test_production_reads_dotenv_storage_and_ignores_saved_synthetic_mode(
    tmp_path, monkeypatch,
):
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_CONFIG", str(ROOT / "config.delivery.yaml"))
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    # Track the key before dotenv inserts it so monkeypatch restores the
    # original environment after this test.
    monkeypatch.setenv("HARNESS_DEPLOYMENT_SETTINGS_PATH", "")
    monkeypatch.delenv("HARNESS_DEPLOYMENT_SETTINGS_PATH", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "HARNESS_DEPLOYMENT_SETTINGS_PATH=data/settings.json\n", encoding="utf-8",
    )
    monkeypatch.setattr(
        "goodvision_harness.config.find_dotenv", lambda **_: str(dotenv),
    )
    data = tmp_path / "data"
    data.mkdir()
    path = data / "settings.json"
    path.write_text(json.dumps({
        "schema_version": 1, "stream_url": "", "dry_run": True, "autostart": False,
    }), encoding="utf-8")
    runtime = HarnessRuntime.from_environment()
    assert runtime.dry_run is False
    assert runtime.deployment_settings()["settings_saved"] is True
    runtime.update_deployment_settings({"buffer_delay_seconds": 1.75})
    assert json.loads(path.read_text(encoding="utf-8"))["dry_run"] is False
    assert not (tmp_path / ".goodvision" / "deployment-settings.json").exists()


def test_no_env_console_key_save_and_restart(tmp_path, monkeypatch, request):
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
    first_owner = FakeSessionOwner(runtime, monkeypatch, request)
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
        assert first_owner.workers == [runtime._identity_worker]
        assert first_owner.sessions == [runtime._session]
    restored = HarnessRuntime(config, deployment_store=store)
    restored_owner = FakeSessionOwner(restored, monkeypatch, request)
    with TestClient(create_app(lambda: restored)) as client:
        assert client.get("/v1/ui/info").json()["api_key_present"]
        assert client.post("/v1/sessions/start", json={"features": ["smoking"], "dry_run": False}).status_code == 200
        assert restored_owner.workers == [restored._identity_worker]
        assert restored_owner.sessions == [restored._session]
        assert restored_owner.workers[0] is not first_owner.workers[0]
    assert not (tmp_path / ".env").exists()
