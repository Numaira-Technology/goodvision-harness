from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import goodvision_harness.config as config_module
from goodvision_harness.aiclub import AIClubClient, AIClubSettings, api_key_present
from goodvision_harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def neutral_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    for key in (
        "FACE_PROVIDER", "FACE_ACCELERATOR", "INFERENCE_API_KEY", "AICLUB_API_KEY",
        "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY", "MODEL_API_KEY",
        "INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    connection = data.pop("inference_service")
    for key in ("endpoint", "api_key_env", "provider", "providers"):
        connection.pop(key, None)
    data["inference_service"] = connection
    data["inference"].pop("model", None)
    data["inference"]["profile"] = "standard"
    data["face"].pop("model_name")
    data["face"].pop("provider", None)
    data["face"]["model_profile"] = "standard"
    data["face"]["accelerator"] = "gpu"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_neutral_profiles_preserve_actual_engines(neutral_config: Path) -> None:
    current = load_config(neutral_config)
    assert current.inference.model == "qwen3.5-plus"
    assert current.face.model_name == "buffalo_l"
    assert current.face.provider == "CUDAExecutionProvider"
    assert current.aiclub.api_key_env == "INFERENCE_API_KEY"
    assert current.aiclub.endpoint == AIClubSettings().endpoint
    assert current.aiclub.max_retries == 0
    assert current.model_api is current.aiclub


def test_new_accelerator_environment_wins_over_legacy(
    neutral_config: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_PROVIDER", "CPUExecutionProvider")
    monkeypatch.setenv("FACE_ACCELERATOR", "gpu")
    assert load_config(neutral_config).face.provider == "CUDAExecutionProvider"


def test_legacy_accelerator_override_remains_supported(
    neutral_config: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_PROVIDER", "DmlExecutionProvider")
    assert load_config(neutral_config).face.provider == "DmlExecutionProvider"


@pytest.mark.parametrize("section,key", [
    ("inference", "profile"), ("face", "model_profile"), ("face", "accelerator"),
])
def test_unknown_profile_fails_instead_of_changing_engine(
    neutral_config: Path, section: str, key: str,
) -> None:
    data = yaml.safe_load(neutral_config.read_text(encoding="utf-8"))
    data[section][key] = "unknown"
    neutral_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=section):
        load_config(neutral_config)


def test_neutral_credentials_take_priority_and_legacy_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICLUB_API_KEY", "old-test-key")
    monkeypatch.setenv("INFERENCE_API_KEY", "new-test-key")
    settings = AIClubSettings(api_key_env="INFERENCE_API_KEY")
    assert AIClubClient(settings)._api_key == "new-test-key"
    monkeypatch.delenv("INFERENCE_API_KEY")
    assert api_key_present(settings)
    assert AIClubClient(settings)._api_key == "old-test-key"
