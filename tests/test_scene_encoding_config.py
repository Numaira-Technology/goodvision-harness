"""Independent VLM encoding budget; camera and SAM compression stay separate."""

from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import load_config


@pytest.fixture
def scene_config(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.delenv("SCENE_JPEG_QUALITY", raising=False)
    path = tmp_path / "scene.yaml"
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
    )

    def write(value=None):
        if value is not None:
            data.setdefault("runtime", {})["scene_jpeg_quality"] = value
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return write


def test_scene_quality_default_config_and_environment_override(
    scene_config, monkeypatch
):
    assert load_config(scene_config()).runtime.scene_jpeg_quality == 90
    path = scene_config(82)
    original = load_config(path)
    assert original.runtime.scene_jpeg_quality == 82
    monkeypatch.setenv("SCENE_JPEG_QUALITY", "70")
    monkeypatch.setenv("SCENE_MAX_IMAGE_SIDE", "640")
    settings = load_config(path)
    assert settings.runtime.scene_jpeg_quality == 70
    assert settings.runtime.scene_max_image_side == 640
    assert settings.video == original.video
    assert settings.rich_labeling == original.rich_labeling


@pytest.mark.parametrize("value", [0, 101, True, 70.5, "70"])
def test_scene_quality_config_rejects_invalid_values(scene_config, value):
    with pytest.raises(ValueError, match="scene_jpeg_quality"):
        load_config(scene_config(value))


@pytest.mark.parametrize("value", ["0", "101", "70.5", "true"])
def test_scene_quality_environment_rejects_invalid_values(
    scene_config, monkeypatch, value
):
    monkeypatch.setenv("SCENE_JPEG_QUALITY", value)
    with pytest.raises(ValueError):
        load_config(scene_config())
