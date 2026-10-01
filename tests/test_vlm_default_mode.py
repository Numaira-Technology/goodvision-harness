"""Default/legacy configuration must start parallel VLM unless explicitly overridden."""

from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import RuntimeSettings, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.delenv("SCENE_VLM_OUTPUT_MODE", raising=False)


@pytest.mark.parametrize("profile", [
    "config.yaml", "config.delivery.yaml", "config.vlm-rich.yaml",
    "src/goodvision_harness/config.yaml",
])
def test_shipped_profiles_default_to_parallel(profile):
    assert load_config(ROOT / profile).runtime.scene_vlm_output_mode == "parallel"


def test_older_config_without_output_mode_defaults_to_parallel(tmp_path, monkeypatch):
    document = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    document["runtime"].pop("scene_vlm_output_mode")
    path = tmp_path / "old-config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    # An empty environment value must also use the default, never legacy full.
    monkeypatch.setenv("SCENE_VLM_OUTPUT_MODE", "")
    assert load_config(path).runtime.scene_vlm_output_mode == "parallel"
    assert RuntimeSettings().scene_vlm_output_mode == "parallel"


@pytest.mark.parametrize("mode", ["full", "compact", "parallel"])
def test_explicit_mode_override_is_preserved(tmp_path, monkeypatch, mode):
    document = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    document["runtime"]["scene_vlm_output_mode"] = mode
    path = tmp_path / "explicit-config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert load_config(path).runtime.scene_vlm_output_mode == mode
    monkeypatch.setenv("SCENE_VLM_OUTPUT_MODE", mode)
    assert load_config(ROOT / "config.yaml").runtime.scene_vlm_output_mode == mode


def test_invalid_mode_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("SCENE_VLM_OUTPUT_MODE", "paralell")
    with pytest.raises(ValueError, match="scene_vlm_output_mode"):
        load_config(ROOT / "config.yaml")
