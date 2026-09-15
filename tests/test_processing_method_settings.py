"""Perception choices persist independently of credentials and model API routing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
    validate_processing_method,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.delenv("PERCEPTION_METHOD", raising=False)


@pytest.mark.parametrize("method", ["sam_live", "hybrid"])
def test_processing_choice_round_trip_keeps_other_deployment_fields_and_api_route(
    tmp_path: Path,
    method: str,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    legacy = DeploymentSettings("rtsp://camera.test/live", True, False, 1.25, "stereo")
    store.save(legacy)
    assert "processing_method" not in json.loads(store.path.read_text())
    assert store.load().processing_method is None
    store.save_api_key("test-secret", provider="openrouter")
    key_before = store.path.with_name("model-api-key-2").read_bytes()
    settings = replace(legacy, processing_method=method)
    store.save(settings)
    restored = DeploymentSettingsStore(store.path)
    assert restored.load() == settings
    assert json.loads(store.path.read_text())["processing_method"] == method
    assert store.path.with_name("model-api-key-2").read_bytes() == key_before
    assert not store.path.with_name("selected-route.json").exists()


@pytest.mark.parametrize(
    "invalid", [None, "", "SAM_LIVE", "sam", "openrouter", True, 1, []]
)
def test_unknown_processing_choice_is_rejected(invalid: object) -> None:
    with pytest.raises(ValueError, match="processing_method"):
        validate_processing_method(invalid)


@pytest.mark.parametrize("invalid", ["sam31_stream", "A", "openrouter", True, 1, []])
def test_invalid_processing_choice_does_not_replace_saved_settings(
    tmp_path: Path,
    invalid: object,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    settings = DeploymentSettings("", False, False, processing_method="sam_live")
    store.save(settings)
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="processing_method"):
        store.save(replace(settings, processing_method=invalid))
    assert store.path.read_bytes() == before


def test_invalid_persisted_processing_choice_does_not_silently_pick_a_route(
    tmp_path: Path,
) -> None:
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    store.path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stream_url": "",
                "dry_run": True,
                "autostart": False,
                "processing_method": "removed-plugin",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="processing_method"):
        store.load()


def test_missing_processing_choice_preserves_legacy_config() -> None:
    config = load_config(ROOT / "config.yaml")
    assert config.rich_labeling.processing_method is None


@pytest.mark.parametrize("method", ["sam_live", "hybrid"])
def test_processing_environment_override_does_not_change_model_route(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    original = load_config(ROOT / "config.yaml")
    monkeypatch.setenv("PERCEPTION_METHOD", " " + method + " ")
    configured = load_config(ROOT / "config.yaml")
    assert configured.rich_labeling.processing_method == method
    assert configured.model_api == original.model_api
    assert configured.inference == original.inference
    assert configured.rich_labeling.backend == original.rich_labeling.backend


def test_yaml_processing_choice_and_empty_environment_use_same_persistable_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload.setdefault("rich_labeling", {})["processing_method"] = "hybrid"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("PERCEPTION_METHOD", " ")
    assert load_config(config_path).rich_labeling.processing_method == "hybrid"
    monkeypatch.setenv("PERCEPTION_METHOD", "sam_live")
    assert load_config(config_path).rich_labeling.processing_method == "sam_live"
    monkeypatch.setenv("PERCEPTION_METHOD", "openrouter")
    with pytest.raises(ValueError, match="processing_method"):
        load_config(config_path)
