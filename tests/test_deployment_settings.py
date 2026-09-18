from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import goodvision_harness.config as config_module
import goodvision_harness.service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
)
from goodvision_harness.grounding import DryRunGroundingClient
from goodvision_harness.privacy import contains_private_implementation_detail
from goodvision_harness.service import HarnessRuntime, create_app


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def ignore_local_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")


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


@pytest.mark.parametrize("seconds", [0, 3.75, 30])
def test_buffer_delay_api_applies_while_running_and_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: float,
) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    config = load_config(ROOT / "config.yaml")
    runtime = HarnessRuntime(config, deployment_store=store)

    class RunningSession:
        running = True

    session = RunningSession()
    with TestClient(create_app(lambda: runtime)) as client:
        initial = client.get("/v1/deployment/settings").json()
        assert initial["buffer_delay_seconds"] == 2.5
        session_dry_run = not runtime.dry_run
        runtime._update_state(dry_run=session_dry_run)
        runtime._session = session  # type: ignore[assignment]
        try:
            response = client.put(
                "/v1/deployment/settings", json={"buffer_delay_seconds": seconds},
            )
            assert response.status_code == 200
            assert runtime._session is session
            assert runtime._state["dry_run"] is session_dry_run
            assert response.json()["buffer_delay_seconds"] == seconds
            assert runtime._delayed_overlay.stats()["buffer_delay_seconds"] == seconds
            info = client.get("/v1/ui/info").json()
            assert info["annotated_output"]["buffer_delay_seconds"] == seconds
            mixed = client.put(
                "/v1/deployment/settings",
                json={"buffer_delay_seconds": 1, "dry_run": True},
            )
            assert mixed.status_code == 409
            assert store.load().buffer_delay_seconds == seconds
        finally:
            runtime._session = None

    restored = HarnessRuntime(config, deployment_store=store)
    assert restored.deployment_settings()["buffer_delay_seconds"] == seconds
    assert restored._delayed_overlay.stats()["buffer_delay_seconds"] == seconds
    restored.update_deployment_settings({"autostart": False})
    assert store.load().buffer_delay_seconds == seconds


def test_old_deployment_record_keeps_configured_buffer_default(tmp_path: Path) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    store.save(DeploymentSettings(stream_url="", dry_run=True, autostart=False))
    assert "buffer_delay_seconds" not in json.loads(store.path.read_text())
    config = load_config(ROOT / "config.yaml")
    config = replace(config, display=replace(config.display, delay_seconds=4.25))
    runtime = HarnessRuntime(config, deployment_store=store)
    assert runtime.deployment_settings()["buffer_delay_seconds"] == 4.25
    assert runtime._delayed_overlay.stats()["buffer_delay_seconds"] == 4.25
    runtime.update_deployment_settings({"autostart": False})
    assert store.load().buffer_delay_seconds is None


@pytest.mark.parametrize(
    "invalid", [-1, 30.1, True, "3", None, float("nan"), float("inf"), 10 ** 400],
)
def test_invalid_buffer_delay_does_not_change_disk_or_runtime(
    tmp_path: Path, invalid: object,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    original = DeploymentSettings("", True, False, buffer_delay_seconds=3.5)
    store.save(original)
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"), deployment_store=store)
    with pytest.raises(ValueError, match="buffer_delay_seconds"):
        runtime.update_deployment_settings({"buffer_delay_seconds": invalid})
    assert store.load() == original
    assert runtime._delayed_overlay.stats()["buffer_delay_seconds"] == 3.5


def test_stored_api_keys_are_independent_for_each_provider(tmp_path: Path) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    providers = ("aiclub", "openrouter", "dashscope")
    store.save_api_key("default-secret")
    for provider in providers[1:]:
        assert store.load_api_key(provider=provider) is None
        assert store.delete_api_key(provider=provider) is False
    assert store.load_api_key() == "default-secret"

    for provider in providers:
        store.save_api_key(f"secret-{provider}", provider=provider)
    for provider in providers:
        assert store.load_api_key(provider=provider) == f"secret-{provider}"

    assert store.delete_api_key(provider="openrouter") is True
    assert store.load_api_key(provider="openrouter") is None
    assert store.load_api_key() == "secret-aiclub"
    assert store.load_api_key(provider="dashscope") == "secret-dashscope"
    assert all(
        provider not in path.name
        for path in tmp_path.iterdir()
        for provider in providers
    )


@pytest.mark.parametrize("provider", ["openrouter", "dashscope"])
def test_other_providers_never_migrate_or_delete_legacy_credentials(
    tmp_path: Path, provider: str,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    legacy = tmp_path / "openrouter-api-key"
    legacy.write_text("legacy-default-secret\n", encoding="utf-8")

    assert store.load_api_key(provider=provider) is None
    store.save_api_key("new-route-secret", provider=provider)
    assert store.delete_api_key(provider=provider) is True
    assert legacy.read_text(encoding="utf-8").strip() == "legacy-default-secret"
    assert store.load_api_key() == "legacy-default-secret"


@pytest.mark.parametrize("operation", ["load", "save", "delete"])
def test_api_key_store_rejects_unknown_provider(
    tmp_path: Path, operation: str,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    with pytest.raises(ValueError, match="unsupported inference provider"):
        if operation == "save":
            store.save_api_key("secret", provider="../outside")
        elif operation == "load":
            store.load_api_key(provider="../outside")
        else:
            store.delete_api_key(provider="../outside")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("provider", ["openrouter", "dashscope"])
def test_runtime_console_credentials_remain_scoped_across_route_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str,
) -> None:
    for name in (
        "INFERENCE_API_KEY", "MODEL_API_KEY", "AICLUB_API_KEY",
        "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    store.save_api_key("default-route-secret")
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    config = load_config(ROOT / "config.yaml")
    runtime = HarnessRuntime(config, deployment_store=store)
    assert runtime._aiclub_api_key is None
    assert runtime.deployment_settings()["api_key_persisted"] is False

    public = runtime.update_deployment_settings({"api_key": "selected-route-secret"})
    assert public["api_key_persisted"] is True
    assert store.load_api_key(provider=provider) == "selected-route-secret"
    restored = HarnessRuntime(config, deployment_store=store)
    assert restored._aiclub_api_key == "selected-route-secret"

    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    default = HarnessRuntime(load_config(ROOT / "config.yaml"), deployment_store=store)
    assert default._aiclub_api_key == "default-route-secret"
    restored.update_deployment_settings({"clear_api_key": True})
    assert store.load_api_key(provider=provider) is None
    assert store.load_api_key() == "default-route-secret"
    rendered = json.dumps(public).lower()
    assert "selected-route-secret" not in rendered
    assert provider not in rendered


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

    try:
        runtime.start()

        assert runtime.configured_stream_url() == saved_url
        assert runtime.dry_run is True
        assert runtime.autostart is True
        assert started == [saved_url]
    finally:
        runtime.stop()


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


@pytest.mark.parametrize("provider", ["aiclub", "openrouter", "dashscope"])
def test_persisted_api_key_is_injected_without_changing_model_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    store.save_api_key("persisted-provider-secret", provider=provider)
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    config = load_config(ROOT / "config.yaml")
    assert config.stream.url == (
        "webrtc+http://192.168.20.199:1985/rtc/v1/whep/"
        "?app=live&stream=livestream"
    )
    captured: dict[str, object] = {}

    def fake_builder(settings, model_api, **kwargs):
        captured.update(
            model=settings.model,
            api_key=kwargs.get("api_key"),
            endpoint=model_api.endpoint,
            provider=model_api.provider,
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
    assert captured["endpoint"] == config.model_api.endpoint
    assert captured["provider"] == provider


@pytest.fixture
def public_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "INFERENCE_API_KEY", "AICLUB_API_KEY", "MODEL_API_KEY",
        "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY", "INFERENCE_PROVIDER",
        "INFERENCE_MODEL", "INFERENCE_ENDPOINT", "LEGACY_CUSTOM_KEY",
        "HARNESS_STREAM_URL", "RTSP_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")


def test_control_errors_do_not_reflect_arbitrary_commands(
    public_settings_environment: None,
    tmp_path: Path,
) -> None:
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=DeploymentSettingsStore(tmp_path / "settings.json"),
    )
    with TestClient(create_app(lambda: runtime)) as client:
        invalid_setting = client.put(
            "/v1/deployment/settings", json={"PaddleX OpenRouter invalid setting": True}
        )
        assert invalid_setting.status_code == 400
        assert invalid_setting.json() == {"detail": "unknown deployment setting"}
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json()["type"] == "hello"
            websocket.send_json({"type": "PaddleX OpenRouter invalid setting"})
            assert websocket.receive_json() == {
                "type": "error", "message": "unknown command",
            }
            websocket.send_json({
                "type": "start",
                "mode": "camera",
                "features": ["PaddleX OpenRouter invalid setting"],
            })
            assert websocket.receive_json() == {
                "type": "error", "message": "unsupported feature",
            }


@pytest.mark.parametrize("legacy", (
    "AICLUB_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY", "LEGACY_CUSTOM_KEY",
))
def test_neutral_credentials_take_precedence_and_legacy_still_works(
    public_settings_environment: None, monkeypatch: pytest.MonkeyPatch, legacy: str
) -> None:
    if legacy == "OPENROUTER_API_KEY":
        monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    config = load_config(ROOT / "config.yaml")
    if legacy == "LEGACY_CUSTOM_KEY":
        config = replace(config, aiclub=replace(config.aiclub, api_key_env=legacy))
    monkeypatch.setenv(legacy, "legacy-fixture")
    assert HarnessRuntime(config)._aiclub_api_key == "legacy-fixture"
    monkeypatch.setenv("INFERENCE_API_KEY", "neutral-fixture")
    assert HarnessRuntime(config)._aiclub_api_key == "neutral-fixture"


def test_saved_credentials_keep_precedence_over_environment(
    public_settings_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    store.save_api_key("saved-fixture")
    monkeypatch.setenv("INFERENCE_API_KEY", "neutral-fixture")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"), deployment_store=store)
    assert runtime._aiclub_api_key == "saved-fixture"


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [
        ("aiclub", "AICLUB_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("dashscope", "DASHSCOPE_API_KEY"),
    ],
)
def test_runtime_uses_only_the_selected_provider_credentials(
    public_settings_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_name: str,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    for name in ("AICLUB_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(name, f"secret-for-{name}")
    config = load_config(ROOT / "config.yaml")

    runtime = HarnessRuntime(config)
    assert runtime._aiclub_api_key == f"secret-for-{key_name}"
    assert not contains_private_implementation_detail(runtime.ui_info())

    monkeypatch.delenv(key_name)
    missing = HarnessRuntime(config)
    assert missing._aiclub_api_key is None
    assert missing.deployment_settings()["api_key_present"] is False

    monkeypatch.setenv("INFERENCE_API_KEY", "explicit-current-route-key")
    assert HarnessRuntime(config)._aiclub_api_key == "explicit-current-route-key"


def test_compose_passes_route_overrides_and_independent_credentials() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    environment = compose["services"]["harness"]["environment"]
    for key in (
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "INFERENCE_API_KEY",
        "AICLUB_API_KEY",
        "OPENROUTER_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        # Empty values must retain YAML/preset defaults, rather than fixing a route.
        assert environment[key] == "${" + key + ":-}"


def test_packaged_config_matches_repository_default() -> None:
    assert (ROOT / "config.yaml").read_bytes() == (
        ROOT / "src" / "goodvision_harness" / "config.yaml"
    ).read_bytes()
