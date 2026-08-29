from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import goodvision_harness.service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
)
from goodvision_harness.grounding import DryRunGroundingClient
from goodvision_harness.service import HarnessRuntime, create_app


ROOT = Path(__file__).resolve().parents[1]


def test_deployment_store_round_trip_and_write_only_key(tmp_path: Path) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    settings = DeploymentSettings(
        stream_url="rtsp://camera-user:camera-secret@camera.local:8554/live",
        dry_run=True,
        autostart=False,
    )

    assert store.load() is None
    assert store.load_api_key() is None
    store.save(settings)
    store.save_api_key("provider-secret")

    assert store.load() == settings
    assert store.load_api_key() == "provider-secret"
    settings_payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert settings_payload["schema_version"] == 1
    assert "provider-secret" not in store.path.read_text(encoding="utf-8")
    assert "camera-secret" not in store.api_key_path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.api_key_path.stat().st_mode) == 0o600

    assert store.delete_api_key() is True
    assert store.delete_api_key() is False
    assert store.load_api_key() is None


def test_legacy_api_key_record_is_migrated_to_generic_name(tmp_path: Path) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    legacy = tmp_path / "openrouter-api-key"
    legacy.write_text("legacy-secret\n", encoding="utf-8")

    assert store.load_api_key() == "legacy-secret"
    assert store.api_key_path.name == "model-api-key"
    assert store.api_key_path.read_text(encoding="utf-8").strip() == "legacy-secret"
    assert not legacy.exists()


def test_deployment_settings_api_persists_without_returning_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "false")
    monkeypatch.setenv("AICLUB_API_KEY", "environment-key")
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )
    app = create_app(lambda: runtime)
    stream_url = "rtsp://camera-user:camera-secret@camera.local:8554/live"

    with TestClient(app) as client:
        initial = client.get("/v1/deployment/settings")
        assert initial.status_code == 200
        assert initial.json()["settings_saved"] is False
        response = client.put(
            "/v1/deployment/settings",
            json={
                "stream_url": stream_url,
                "dry_run": True,
                "autostart": False,
                "api_key": "persisted-provider-secret",
            },
        )
        assert response.status_code == 200
        public = response.json()
        info = client.get("/v1/ui/info").json()

    rendered = json.dumps({"public": public, "info": info}, sort_keys=True)
    assert public["settings_saved"] is True
    assert public["settings_source"] == "persistent"
    assert public["configured_stream"] is True
    assert public["configured_stream_protocol"] == "rtsp"
    assert public["dry_run"] is True
    assert public["autostart"] is False
    assert public["api_key_present"] is True
    assert public["api_key_persisted"] is True
    assert "stream_url" not in public
    assert stream_url not in rendered
    assert "camera-secret" not in rendered
    assert "persisted-provider-secret" not in rendered
    assert runtime.configured_stream_url() == stream_url
    assert runtime.dry_run is True
    assert runtime.autostart is False

    restored = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )
    restored_public = restored.deployment_settings()
    assert restored.configured_stream_url() == stream_url
    assert restored.dry_run is True
    assert restored.autostart is False
    assert restored_public["api_key_present"] is True
    assert restored_public["api_key_persisted"] is True
    restored.update_deployment_settings({"clear_api_key": True})
    assert restored.deployment_settings()["api_key_persisted"] is False
    assert restored.deployment_settings()["api_key_present"] is True


def test_persistent_settings_override_bootstrap_environment_and_autostart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RTSP_URL", "rtsp://environment.invalid/live")
    monkeypatch.setenv("HARNESS_DRY_RUN", "false")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    saved_url = "rtsp://host.docker.internal:8554/live/livestream"
    store.save(
        DeploymentSettings(stream_url=saved_url, dry_run=True, autostart=True)
    )
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )
    started: list[str] = []
    runtime.start_stream = lambda url: started.append(url) or {}  # type: ignore[method-assign]

    runtime.start()

    assert runtime.configured_stream_url() == saved_url
    assert runtime.dry_run is True
    assert runtime.autostart is True
    assert started == [saved_url]


def test_generic_stream_environment_overrides_legacy_rtsp_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whep_url = (
        "webrtc+http://192.168.20.199:1985/rtc/v1/whep/"
        "?app=live&stream=livestream"
    )
    monkeypatch.setenv("HARNESS_STREAM_URL", whep_url)
    monkeypatch.setenv("RTSP_URL", "rtsp://legacy.invalid/live")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))

    assert runtime.configured_stream_url() == whep_url
    assert runtime.deployment_settings()["settings_source"] == "environment"
    assert runtime.deployment_settings()["configured_stream_protocol"] == "webrtc"


def test_deployment_update_rejects_active_session_and_invalid_values(
    tmp_path: Path,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )

    with pytest.raises(ValueError, match="unsupported stream URL"):
        runtime.update_deployment_settings({"stream_url": "ftp://example.test/video"})
    with pytest.raises(ValueError, match="dry_run must be boolean"):
        runtime.update_deployment_settings({"dry_run": "yes"})

    class _RunningSession:
        running = True

    runtime._session = _RunningSession()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="stop the active session"):
        runtime.update_deployment_settings({"dry_run": True})


@pytest.mark.parametrize(
    "invalid_key",
    [
        "line\nbreak",
        "carriage\rreturn",
        "tab\tcharacter",
        "nul\x00character",
        "delete\x7fcharacter",
    ],
)
def test_deployment_update_rejects_api_key_control_characters(
    tmp_path: Path,
    invalid_key: str,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )

    with pytest.raises(ValueError, match="control characters"):
        runtime.update_deployment_settings({"api_key": invalid_key})

    assert store.load_api_key() is None


@pytest.mark.parametrize(
    ("invalid_key", "expected_detail"),
    [
        ("", "api_key must not be empty"),
        (
            "invalid\nprovider-key",
            "api_key must not contain control characters",
        ),
    ],
)
def test_mixed_invalid_deployment_update_has_no_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_key: str,
    expected_detail: str,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    original = DeploymentSettings(
        stream_url="rtsp://original-user:original-secret@camera.local/live",
        dry_run=True,
        autostart=False,
    )
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    store.save(original)
    store.save_api_key("original-provider-key")
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.put(
            "/v1/deployment/settings",
            json={
                "stream_url": "rtsp://replacement.local/live",
                "dry_run": False,
                "autostart": True,
                "api_key": invalid_key,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail
    assert store.load() == original
    assert store.load_api_key() == "original-provider-key"
    assert runtime.configured_stream_url() == original.stream_url
    assert runtime.dry_run is original.dry_run
    assert runtime.autostart is original.autostart
    assert runtime.deployment_settings()["api_key_present"] is True


def test_deployment_settings_request_body_is_limited_to_64_kib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=store,
    )
    app = create_app(lambda: runtime)
    oversized_body = json.dumps(
        {"api_key": "x" * (64 * 1024)}
    ).encode("utf-8")

    with TestClient(app) as client:
        response = client.put(
            "/v1/deployment/settings",
            content=oversized_body,
            headers={"Content-Type": "application/json"},
        )

    assert len(oversized_body) > 64 * 1024
    assert response.status_code == 413
    assert response.json()["detail"] == (
        "deployment settings body exceeds 64 KiB"
    )
    assert store.load() is None
    assert store.load_api_key() is None


def test_persisted_api_key_is_injected_without_changing_model_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    store.save_api_key("persisted-provider-secret")
    config = load_config(ROOT / "config.yaml")
    assert config.stream.url == (
        "webrtc+http://192.168.20.199:1985/rtc/v1/whep/"
        "?app=live&stream=livestream"
    )
    captured: dict[str, object] = {}

    def fake_builder(settings, aiclub_settings, **kwargs):
        captured.update(
            model=settings.model,
            api_key=kwargs.get("api_key"),
            endpoint=aiclub_settings.endpoint,
        )
        return DryRunGroundingClient(settings, kwargs.get("memory_context"))

    monkeypatch.setattr(service_module, "build_grounding_client", fake_builder)
    runtime = HarnessRuntime(config, deployment_store=store)
    try:
        runtime.start_camera(
            dry_run=False,
            analysis_mode="event",
            owner="deployment-key-test",
        )
    finally:
        runtime.stop()

    assert captured["api_key"] == "persisted-provider-secret"
    assert captured["model"] == config.inference.model
    assert captured["model"] != "numaira-beta-fast"
