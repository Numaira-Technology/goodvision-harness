from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_route_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    for name in ("INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


def test_rich_labeling_defaults_are_safe_and_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RICH_LABEL_ENABLED", raising=False)
    monkeypatch.delenv("RICH_LABEL_ENDPOINT", raising=False)
    monkeypatch.delenv("RICH_LABEL_BACKEND", raising=False)

    settings = load_config(ROOT / "config.yaml").rich_labeling

    assert settings.enabled is False
    assert settings.backend == "paddle"
    assert settings.model is None
    assert settings.vlm_max_tokens == 4096
    assert settings.vlm_max_image_side == 1280
    assert settings.endpoint == "http://rich-label-worker:8091/v1/labels"
    assert settings.max_fps == 7.5
    assert settings.max_interpolation_gap_seconds == 2.0
    assert settings.max_hold_seconds == 2.0


def test_rich_labeling_environment_overrides_connection_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_ENABLED", "true")
    monkeypatch.setenv("RICH_LABEL_ENDPOINT", "http://127.0.0.1:18091/v1/labels")
    monkeypatch.setenv("RICH_LABEL_BACKEND", " VLM ")

    settings = load_config(ROOT / "config.yaml").rich_labeling

    assert settings.enabled is True
    assert settings.backend == "vlm"
    assert settings.endpoint == "http://127.0.0.1:18091/v1/labels"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "unsupported", "backend must be paddle or vlm"),
        ("model", 42, "model must be a string or null"),
        ("vlm_max_tokens", 0, "positive integer"),
        ("vlm_max_tokens", True, "positive integer"),
        ("vlm_max_image_side", 0, "positive integer"),
        ("vlm_max_image_side", 1280.5, "positive integer"),
        ("endpoint", "file:///tmp/worker", "must use http or https"),
        ("timeout_seconds", 0, "positive number"),
        ("max_fps", 0, "positive number"),
        ("history_size", 0, "positive integer"),
        ("history_size", 1, "at least 2"),
        ("max_interpolation_gap_seconds", -1, "positive number"),
        ("max_hold_seconds", -1, "positive number"),
    ],
)
def test_rich_labeling_config_validation_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.delenv("RICH_LABEL_ENABLED", raising=False)
    monkeypatch.delenv("RICH_LABEL_ENDPOINT", raising=False)
    monkeypatch.delenv("RICH_LABEL_BACKEND", raising=False)
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["rich_labeling"][field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_config(config_path)


def test_rich_labeling_vlm_model_and_limits_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RICH_LABEL_BACKEND", raising=False)
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["rich_labeling"].update(
        backend="vlm",
        model=" annotation-model ",
        vlm_max_tokens=8192,
        vlm_max_image_side=960,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    settings = load_config(config_path).rich_labeling

    assert settings.backend == "vlm"
    assert settings.model == "annotation-model"
    assert settings.vlm_max_tokens == 8192
    assert settings.vlm_max_image_side == 960


def test_rich_labeling_invalid_environment_backend_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_BACKEND", "unsupported")

    with pytest.raises(ValueError, match="backend must be paddle or vlm"):
        load_config(ROOT / "config.yaml")


def _write_route_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "route-config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_rich_vlm_inherits_selected_profile_with_its_own_output_budget(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    provider = {"only": ["deepinfra"], "allow_fallbacks": False}
    payload["inference_service"] = {
        "provider": "openrouter",
        "providers": {
            "openrouter": {
                "model": "qwen/qwen3.5-35b-a3b",
                "request_parameters": {
                    "provider": provider,
                    "reasoning": {"effort": "none"},
                    "temperature": 0.1,
                    "max_tokens": 777,
                },
            },
        },
    }
    payload["inference"]["request_parameters"] = {
        "temperature": 0.4, "max_tokens": 1200
    }
    payload["rich_labeling"].update(
        backend="vlm",
        vlm_max_tokens=8192,
        vlm_request_parameters={
            "temperature": 0.0,
            "top_p": 0.8,
            "reasoning": {"enabled": True},
            "provider": {"only": ["coreweave"]},
            "response_format": {"type": "json_object"},
        },
    )

    config = load_config(_write_route_config(tmp_path, payload))

    assert config.model_api.provider == "openrouter"
    assert config.inference.model == "qwen/qwen3.5-35b-a3b"
    assert config.rich_labeling.model is None
    assert config.rich_labeling.vlm_max_tokens == 8192
    assert config.rich_labeling.vlm_request_parameters == {
        "provider": provider,
        "reasoning": {"effort": "none"},
        "temperature": 0.1,
        "top_p": 0.8,
        "response_format": {"type": "json_object"},
    }
    assert config.inference.request_parameters["provider"] == provider
    assert config.inference.request_parameters["reasoning"] == {"effort": "none"}
    assert config.inference.request_parameters["max_tokens"] == 777
    assert "top_p" not in config.inference.request_parameters


def test_switching_to_aiclub_removes_old_rich_openrouter_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["inference_service"] = {
        "provider": "openrouter",
        "providers": {
            "openrouter": {
                "model": "qwen/qwen3.5-35b-a3b",
                "request_parameters": {
                    "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
                    "reasoning": {"effort": "none"},
                },
            },
            "aiclub": {
                "model": "qwen3.5-plus",
                "request_parameters": {"enable_thinking": False},
            },
        },
    }
    payload["inference"]["request_parameters"] = {
        "temperature": 0.0,
        "max_tokens": 1200,
        "provider": {"only": ["deepinfra"]},
        "reasoning": {"enabled": False},
    }
    payload["rich_labeling"]["vlm_request_parameters"] = {
        "provider": {"only": ["coreweave"]},
        "reasoning": {"enabled": True},
        "enable_thinking": True,
        "response_format": {"type": "json_object"},
    }
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")

    config = load_config(_write_route_config(tmp_path, payload))

    assert config.model_api.provider == "aiclub"
    assert config.inference.model == "qwen3.5-plus"
    assert config.rich_labeling.vlm_request_parameters == {
        "temperature": 0.0,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    assert "provider" not in config.inference.request_parameters
    assert "reasoning" not in config.inference.request_parameters


def test_switching_to_default_openrouter_inherits_default_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["inference_service"] = {"provider": "aiclub"}
    payload["inference"]["request_parameters"] = {
        "temperature": 0.0, "max_tokens": 1200, "enable_thinking": False
    }
    payload["rich_labeling"]["vlm_request_parameters"] = {"enable_thinking": False}
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")

    config = load_config(_write_route_config(tmp_path, payload))

    assert config.model_api.provider == "openrouter"
    assert config.inference.model == "qwen/qwen3.5-35b-a3b"
    assert config.rich_labeling.vlm_request_parameters == {
        "temperature": 0.0, "reasoning": {"enabled": False}
    }
    assert config.inference.request_parameters["max_tokens"] == 1200


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "other-model"), ("messages", []), ("max_tokens", 12), ("stream", False)],
)
def test_rich_vlm_explicit_fixed_fields_are_not_silently_filtered(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["rich_labeling"]["vlm_request_parameters"] = {field: value}

    with pytest.raises(ValueError, match="vlm_request_parameters must not override"):
        load_config(_write_route_config(tmp_path, payload))
