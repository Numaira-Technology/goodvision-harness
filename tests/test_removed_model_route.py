"""The retired provider cannot be selected or donate credentials to OpenRouter."""

import json
from pathlib import Path

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.model_api import ModelAPISettings, environment_api_key


@pytest.mark.parametrize(
    "configuration",
    [
        {"inference_service": {"provider": "aiclub"}},
        {"aiclub": {"api_key_env": "OLD_KEY"}},
    ],
)
def test_removed_provider_config_is_rejected_before_building_a_client(
    tmp_path, monkeypatch, configuration
):
    import goodvision_harness.config as module

    monkeypatch.setattr(module, "find_dotenv", lambda **kwargs: "")
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("HARNESS_INFERENCE_PROFILE", raising=False)
    path = tmp_path / "removed.json"
    path.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="AIClub has been removed"):
        load_config(path)


def test_removed_provider_environment_cannot_override_default(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    with pytest.raises(ValueError, match="AIClub has been removed"):
        load_config(Path(__file__).parents[1] / "config.yaml")


@pytest.mark.parametrize(
    "settings",
    [
        {"provider": "aiclub"},
        {"endpoint": "https://aiclub.zimaocloud.com/model/openApi/v1/chat/completions"},
    ],
)
def test_removed_provider_cannot_be_built_directly(settings):
    with pytest.raises(ValueError):
        ModelAPISettings(**settings)


@pytest.mark.parametrize("operation", ["load", "save", "delete"])
def test_retired_storage_is_not_read_or_modified(tmp_path, operation):
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    retired = tmp_path / "model-api-key"
    retired.write_text("retired-fixture")
    with pytest.raises(ValueError, match="unsupported inference provider"):
        method = getattr(store, operation + "_api_key")
        method("new-fixture", provider="aiclub") if operation == "save" else method(
            provider="aiclub"
        )
    assert retired.read_text() == "retired-fixture"
    assert store.load_api_key() is None
    store.save_api_key("openrouter-fixture")
    assert store.api_key_path.name == "model-api-key-2"
    assert store.load_api_key(provider="openrouter") == "openrouter-fixture"


def test_retired_env_key_is_not_a_fallback(monkeypatch):
    for name in ("INFERENCE_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AICLUB_API_KEY", "retired-fixture")
    assert environment_api_key(ModelAPISettings()) == ""


@pytest.mark.parametrize(
    "relative",
    [
        "config.yaml",
        "config.vlm-rich.yaml",
        "tools/vlm_benchmark/config.json",
    ],
)
def test_shipped_default_route_and_model_match_openrouter(monkeypatch, relative):
    import goodvision_harness.config as module

    monkeypatch.setattr(module, "find_dotenv", lambda **kwargs: "")
    for name in (
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "HARNESS_INFERENCE_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = load_config(Path(__file__).parents[1] / relative)
    assert config.model_api.provider == "openrouter"
    assert config.model_api.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert config.inference.model.startswith("qwen/")
