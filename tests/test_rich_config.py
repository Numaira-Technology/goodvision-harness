from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_route_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    for name in (
        "INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT",
        "RICH_LABEL_MAX_FPS",
    ):
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
    assert settings.vlm_wire_format == "objects"
    assert settings.vlm_execution_mode == "single"
    assert settings.vlm_structured_output is False
    assert settings.vlm_prompt_cache is False
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
        ("backend", "unsupported",
         "backend must be paddle, vlm, sam, sam31 or sam31_stream"),
        ("vlm_wire_format", "unsupported",
         "vlm_wire_format must be objects or compact"),
        ("vlm_execution_mode", "unsupported",
         "vlm_execution_mode must be single or parallel"),
        ("vlm_execution_mode", "parallel", "requires vlm_wire_format: compact"),
        ("vlm_structured_output", True, "requires vlm_wire_format: compact"),
        ("vlm_structured_output", "true", "must be true or false"),
        ("vlm_prompt_cache", "true", "must be true or false"),
        ("vlm_prompt_cache", 1, "must be true or false"),
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
        vlm_wire_format="compact",
        vlm_execution_mode="parallel",
        vlm_structured_output=True,
        vlm_prompt_cache=True,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    settings = load_config(config_path).rich_labeling

    assert settings.backend == "vlm"
    assert settings.model == "annotation-model"
    assert settings.vlm_max_tokens == 8192
    assert settings.vlm_max_image_side == 960
    assert settings.vlm_wire_format == "compact"
    assert settings.vlm_execution_mode == "parallel"
    assert settings.vlm_structured_output is True
    assert settings.vlm_prompt_cache is True


def test_rich_labeling_invalid_environment_backend_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_BACKEND", "unsupported")

    with pytest.raises(
        ValueError, match="backend must be paddle, vlm, sam, sam31 or sam31_stream"
    ):
        load_config(ROOT / "config.yaml")


def test_persistent_native_backend_is_opt_in_with_explicit_endpoint(monkeypatch):
    monkeypatch.setenv("RICH_LABEL_BACKEND", "sam31_stream")
    monkeypatch.setenv("RICH_LABEL_ENDPOINT", "http://localhost:18092/v1/video/sessions")
    monkeypatch.setenv("SAM31_BATCH_FRAMES", "16")
    monkeypatch.setenv("RICH_LABEL_MAX_FPS", "4")
    settings = load_config(ROOT / "config.yaml").rich_labeling
    assert settings.backend == "sam31_stream"
    assert settings.endpoint.endswith("/v1/video/sessions")
    assert settings.video_batch_frames == 16


@pytest.mark.parametrize(("frames", "fps"), [(8, 0.1), (16, 0.1)])
def test_persistent_native_startup_cannot_outgrow_retained_evidence(
    monkeypatch, frames, fps
):
    monkeypatch.setenv("RICH_LABEL_BACKEND", "sam31_stream")
    monkeypatch.setenv("SAM31_BATCH_FRAMES", str(frames))
    monkeypatch.setenv("RICH_LABEL_MAX_FPS", str(fps))
    with pytest.raises(ValueError, match="sam31_stream requires"):
        load_config(ROOT / "config.yaml")


def test_native_stream_append_target_and_transport_are_independent_of_startup(
    monkeypatch,
):
    monkeypatch.setenv("RICH_LABEL_BACKEND", "sam31_stream")
    monkeypatch.setenv("SAM31_BATCH_FRAMES", "4")
    monkeypatch.setenv("SAM31_BATCH_SECONDS", "0.5")
    monkeypatch.setenv("SAM31_TRANSPORT_MAX_SIDE", "512")
    monkeypatch.setenv("SAM31_TRANSPORT_JPEG_QUALITY", "55")
    rich = load_config(ROOT / "config.yaml").rich_labeling
    assert (rich.video_batch_frames, rich.video_batch_seconds) == (4, 0.5)
    assert rich.video_transport_max_side == 512
    assert rich.video_transport_jpeg_quality == 55


def test_h264_transport_is_explicit_session_only_with_bounded_crf(monkeypatch):
    monkeypatch.setenv("RICH_LABEL_BACKEND", "sam31_stream")
    monkeypatch.setenv("SAM31_TRANSPORT_CODEC", "h264")
    monkeypatch.setenv("SAM31_TRANSPORT_CRF", "23")
    rich = load_config(ROOT / "config.yaml").rich_labeling
    assert rich.video_transport_codec == "h264"
    assert rich.video_transport_crf == 23


@pytest.mark.parametrize("backend", ["paddle", "vlm", "sam", "sam31"])
def test_h264_cannot_silently_change_other_backends(monkeypatch, backend):
    monkeypatch.setenv("RICH_LABEL_BACKEND", backend)
    monkeypatch.setenv("SAM31_TRANSPORT_CODEC", "h264")
    with pytest.raises(ValueError, match="requires sam31_stream"):
        load_config(ROOT / "config.yaml")


@pytest.mark.parametrize("value", ["17", "36", "23.5", "nan"])
def test_h264_crf_rejects_out_of_range_or_noninteger_values(monkeypatch, value):
    monkeypatch.setenv("SAM31_TRANSPORT_CRF", value)
    with pytest.raises(ValueError):
        load_config(ROOT / "config.yaml")


@pytest.mark.parametrize(("name", "value"), [
    ("SAM31_TRANSPORT_MAX_SIDE", "127"), ("SAM31_TRANSPORT_MAX_SIDE", "1281"),
    ("SAM31_TRANSPORT_MAX_SIDE", "512.5"), ("SAM31_TRANSPORT_JPEG_QUALITY", "0"),
    ("SAM31_TRANSPORT_JPEG_QUALITY", "101"), ("SAM31_TRANSPORT_JPEG_QUALITY", "nan"),
])
def test_native_transport_rejects_invalid_explicit_options(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_config(ROOT / "config.yaml")


def test_sam31_uses_separate_bounded_video_settings(monkeypatch):
    monkeypatch.setenv("RICH_LABEL_BACKEND", "sam31")
    monkeypatch.setenv("SAM31_BATCH_FRAMES", "8")
    monkeypatch.setenv("SAM31_BATCH_SECONDS", "2")
    monkeypatch.setenv("SAM31_CLIENT_TIMEOUT_SECONDS", "70")
    rich = load_config(ROOT / "config.yaml").rich_labeling
    assert rich.backend == "sam31"
    assert (rich.video_batch_frames, rich.video_batch_seconds) == (8, 2)
    assert rich.video_timeout_seconds == 70


@pytest.mark.parametrize(("name", "value"), [
    ("SAM31_BATCH_FRAMES", "0"), ("SAM31_BATCH_FRAMES", "17"),
    ("SAM31_BATCH_FRAMES", "3.5"), ("SAM31_BATCH_SECONDS", "11"),
    ("SAM31_CLIENT_TIMEOUT_SECONDS", "nan"),
])
def test_sam31_rejects_unbounded_configuration(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_config(ROOT / "config.yaml")


def _write_route_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "route-config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_neutral_customer_profile_preserves_distinct_rich_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["inference_service"] = {"timeout_seconds": 8, "max_retries": 0}
    rich_route = {"only": ["seed/fp8"], "allow_fallbacks": False}
    payload["rich_labeling"].update(
        backend="vlm", model="bytedance-seed/seed-2.0-mini",
        vlm_request_parameters={
            "provider": rich_route, "reasoning": {"enabled": False},
        },
    )
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    config = load_config(_write_route_config(tmp_path, payload))
    assert config.model_api.provider == "openrouter"
    assert config.inference.request_parameters["provider"] == {
        "sort": "throughput", "allow_fallbacks": True,
        "ignore": ["coreweave", "novita"],
    }
    assert config.rich_labeling.vlm_request_parameters["provider"] == rich_route
    assert config.rich_labeling.vlm_request_parameters["reasoning"] == {
        "enabled": False,
    }


@pytest.mark.parametrize("rich_model", [None, "qwen/qwen3.5-35b-a3b"])
def test_rich_vlm_inherits_selected_profile_with_its_own_output_budget(
    tmp_path: Path,
    rich_model: str | None,
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
        model=rich_model,
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
    assert config.rich_labeling.model == rich_model
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


def test_distinct_rich_model_uses_its_provider_without_changing_event_route(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    event_parameters = {
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        "reasoning": {"effort": "none"},
        "temperature": 0.1,
        "max_tokens": 777,
    }
    payload["inference_service"] = {
        "provider": "openrouter",
        "providers": {"openrouter": {
            "model": "qwen/qwen3.5-35b-a3b",
            "request_parameters": event_parameters,
        }},
    }
    payload["inference"]["request_parameters"] = {"temperature": 0.4}
    rich_parameters = {
        "provider": {"only": ["alibaba"], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    payload["rich_labeling"].update(
        backend="vlm", model=" qwen/qwen3-vl-8b-instruct ",
        vlm_max_tokens=2048, vlm_request_parameters=rich_parameters,
    )

    config = load_config(_write_route_config(tmp_path, payload))

    assert config.model_api.provider == "openrouter"
    assert config.rich_labeling.model == "qwen/qwen3-vl-8b-instruct"
    assert config.rich_labeling.vlm_request_parameters == rich_parameters
    assert config.rich_labeling.vlm_max_tokens == 2048
    assert config.inference.model == "qwen/qwen3.5-35b-a3b"
    assert config.inference.request_parameters == event_parameters


@pytest.mark.parametrize("rich_model", [None, "qwen/qwen3-vl-8b-instruct"])
def test_switching_to_aiclub_removes_old_rich_openrouter_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rich_model: str | None,
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
    payload["rich_labeling"]["model"] = rich_model
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


@pytest.mark.parametrize("rich_model", [None, "qwen3-vl-flash"])
def test_switching_to_default_openrouter_inherits_default_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rich_model: str | None,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["inference_service"] = {"provider": "aiclub"}
    payload["inference"]["request_parameters"] = {
        "temperature": 0.0, "max_tokens": 1200, "enable_thinking": False
    }
    payload["rich_labeling"]["model"] = rich_model
    payload["rich_labeling"]["vlm_request_parameters"] = {"enable_thinking": False}
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")

    config = load_config(_write_route_config(tmp_path, payload))

    assert config.model_api.provider == "openrouter"
    assert config.inference.model == "qwen/qwen3.5-35b-a3b:nitro"
    assert config.rich_labeling.vlm_request_parameters == {
        "temperature": 0.0, "reasoning": {"enabled": False},
        "provider": {
            "sort": "throughput", "allow_fallbacks": True,
            "ignore": ["coreweave", "novita"],
        },
    }
    assert config.inference.request_parameters["max_tokens"] == 1200


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "other-model"), ("messages", []), ("max_tokens", 12), ("stream", False)],
)
@pytest.mark.parametrize("rich_model", [None, "qwen/qwen3-vl-8b-instruct"])
def test_rich_vlm_explicit_fixed_fields_are_not_silently_filtered(
    tmp_path: Path,
    field: str,
    value: object,
    rich_model: str | None,
) -> None:
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["rich_labeling"]["model"] = rich_model
    payload["rich_labeling"]["vlm_request_parameters"] = {field: value}

    with pytest.raises(ValueError, match="vlm_request_parameters must not override"):
        load_config(_write_route_config(tmp_path, payload))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_realtime_queue_configuration_preserves_display_settings(
    monkeypatch, batch_size,
):
    before = load_config(ROOT / "config.yaml").display
    monkeypatch.setenv("PERCEPTION_METHOD", "sam_live")
    monkeypatch.setenv("SAM31_QUEUE_POLICY", "realtime_latest")
    monkeypatch.setenv("SAM31_REALTIME_APPEND_FRAMES", str(batch_size))
    monkeypatch.setenv("SAM31_REALTIME_PENDING_FRAMES", "2")
    monkeypatch.setenv("SAM31_REALTIME_MAX_AGE_SECONDS", "2")
    result = load_config(ROOT / "config.yaml")
    assert result.rich_labeling.video_queue_policy == "realtime_latest"
    assert result.rich_labeling.video_realtime_append_frames == batch_size
    assert result.rich_labeling.video_realtime_pending_frames == 2
    assert result.display == before


@pytest.mark.parametrize("name,value", [
    ("SAM31_QUEUE_POLICY", "unknown"),
    ("SAM31_REALTIME_APPEND_FRAMES", "3"),
    ("SAM31_REALTIME_PENDING_FRAMES", "0"),
    ("SAM31_REALTIME_MAX_AGE_SECONDS", "nan"),
    ("SAM31_RESULT_DEADLINE_SECONDS", "0"),
])
def test_realtime_queue_rejects_invalid_configuration(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_config(ROOT / "config.yaml")


def test_realtime_queue_rejects_noncausal_route(monkeypatch):
    monkeypatch.setenv("SAM31_QUEUE_POLICY", "realtime_latest")
    monkeypatch.delenv("PERCEPTION_METHOD", raising=False)
    with pytest.raises(ValueError, match="requires sam_live"):
        load_config(ROOT / "config.yaml")


@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostic_latest_only_environment_toggle(monkeypatch, enabled):
    monkeypatch.setenv("RICH_LABEL_LATEST_ONLY", str(enabled).lower())
    monkeypatch.setenv("SAM31_QUEUE_POLICY", "durable_ordered")
    rich = load_config(ROOT / "config.yaml").rich_labeling
    assert rich.video_latest_only is enabled
    assert rich.video_queue_policy == "durable_ordered"
