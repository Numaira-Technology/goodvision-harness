from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import cv2
import numpy as np
import pytest
import yaml

import goodvision_harness.config as config_module
import goodvision_harness.model_api as model_api_module
from goodvision_harness.aiclub import AIClubClient, AIClubError, AIClubSettings
from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    SAFETY_EVENT_LABELS,
    DryRunGroundingClient,
    build_fixed_five_grounding_client,
    build_grounding_client,
    build_selected_features_grounding_client,
)
from goodvision_harness.inference import DryRunWindowClient, build_client
from goodvision_harness.model_api import (
    ModelAPIClient,
    ModelAPIError,
    ModelAPISettings,
    api_key_present,
    environment_api_key,
)
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def _model_jpeg(shade: int = 0) -> bytes:
    """A decodable frame already at the configured 360x202 model canvas."""

    ok, encoded = cv2.imencode(".jpg", np.full((202, 360, 3), shade, np.uint8))
    assert ok
    return encoded.tobytes()

ROOT = Path(__file__).resolve().parents[1]
ROUTES = (
    (
        "aiclub",
        "https://aiclub.zimaocloud.com/model/openApi/v1/chat/completions",
        "AICLUB_API_KEY",
        "qwen3.5-plus",
        {},
    ),
    (
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "OPENROUTER_API_KEY",
        "qwen/qwen3.5-35b-a3b:nitro",
        {"reasoning": {"enabled": False}},
    ),
    (
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "DASHSCOPE_API_KEY",
        "qwen3.5-plus",
        {"enable_thinking": False},
    ),
)


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    for key in (
        "INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT",
        "INFERENCE_API_KEY", "MODEL_API_KEY", "CUSTOM_MODEL_KEY",
        "AICLUB_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        (ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8",
    )
    return path


class _Response:
    def __init__(self, content: dict[str, Any]) -> None:
        self._body = json.dumps({
            "choices": [{"message": {"content": json.dumps(content)}}],
        }).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _window() -> FrameWindow:
    return FrameWindow(index=0, frames=(SampledFrame(
        sample_index=0,
        source_frame_index=0,
        timestamp_seconds=0.0,
        jpeg_bytes=_model_jpeg(),
        width=360,
        height=202,
    ),))


def test_aiclub_imports_remain_compatibility_aliases() -> None:
    assert AIClubSettings is ModelAPISettings
    assert AIClubClient is ModelAPIClient
    assert AIClubError is ModelAPIError


@pytest.mark.parametrize("kind", ["binary", "grounding", "selected", "fixed_five"])
@pytest.mark.parametrize("provider,endpoint,key_env,model,route_parameters", ROUTES)
def test_live_builders_send_the_selected_route(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    provider: str,
    endpoint: str,
    key_env: str,
    model: str,
    route_parameters: dict[str, Any],
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    # Having every credential installed must still choose only the selected key.
    for route, _, env_name, _, _ in ROUTES:
        monkeypatch.setenv(env_name, f"test-{route}-key")
    config = load_config(ROOT / "config.yaml")
    calls: list[Request] = []
    clear = {
        "detected": False, "confidence": 0.9,
        "description": "未发现异常事件。 / No visible event.", "objects": [],
    }
    labels = ("smoking",) if kind == "selected" else SAFETY_EVENT_LABELS
    content = (
        {"events": [{"event_label": label, **clear} for label in labels]}
        if kind in {"selected", "fixed_five"} else clear
    )

    def capture(request: Request, *, timeout: float) -> _Response:
        assert timeout == config.model_api.timeout_seconds
        calls.append(request)
        return _Response(content)

    monkeypatch.setattr(model_api_module, "urlopen", capture)
    if kind == "binary":
        client = build_client(config.inference, config.model_api)
    elif kind == "grounding":
        client = build_grounding_client(config.inference, config.model_api)
    elif kind == "selected":
        composite = build_selected_features_grounding_client(
            config.inference, config.model_api, features=("smoking",),
        )
        client = composite._safety_client
    else:
        composite = build_fixed_five_grounding_client(
            config.inference, config.model_api,
            stranger_client=DryRunGroundingClient(config.inference),
        )
        # Face recognition is local; exercise the model branch of each composite.
        client = composite._safety_client
    assert client is not None
    assert client.classify_window(_window()).ok
    assert len(calls) == 1
    request = calls[0]
    assert config.model_api.provider == provider
    assert config.model_api.api_key_env == key_env
    assert request.full_url == endpoint
    assert request.get_header("Authorization") == f"Bearer test-{provider}-key"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data)
    assert payload["model"] == model
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] >= 1200
    assert {
        key: payload[key]
        for key in ("reasoning", "enable_thinking") if key in payload
    } == route_parameters
    assert payload["messages"][0]["role"] == "system"
    blocks = payload["messages"][1]["content"]
    assert any(block.get("type") == "image_url" for block in blocks)


@pytest.mark.parametrize("provider,endpoint,key_env,model,route_parameters", ROUTES)
def test_missing_selected_key_never_falls_back_to_another_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    endpoint: str,
    key_env: str,
    model: str,
    route_parameters: dict[str, Any],
) -> None:
    for route, _, env_name, _, _ in ROUTES:
        if route != provider:
            monkeypatch.setenv(env_name, f"test-{route}-key")
    settings = ModelAPISettings(provider=provider)
    assert not environment_api_key(settings)
    assert not api_key_present(settings)
    with pytest.raises(ModelAPIError, match=key_env):
        ModelAPIClient(settings)


def test_credentials_follow_explicit_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ModelAPISettings(
        provider="openrouter", api_key_env="CUSTOM_MODEL_KEY",
        endpoint="https://custom.example/v1/chat/completions",
    )
    monkeypatch.setenv("MODEL_API_KEY", "legacy-generic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    monkeypatch.setenv("CUSTOM_MODEL_KEY", "custom-key")
    monkeypatch.setenv("INFERENCE_API_KEY", " neutral-key ")
    assert environment_api_key(settings) == "neutral-key"
    monkeypatch.setenv("INFERENCE_API_KEY", "  ")
    assert environment_api_key(settings) == "custom-key"
    monkeypatch.delenv("CUSTOM_MODEL_KEY")
    assert environment_api_key(settings) == "legacy-generic"
    monkeypatch.delenv("MODEL_API_KEY")
    assert not environment_api_key(settings)
    assert not api_key_present(settings)
    with pytest.raises(ModelAPIError, match="CUSTOM_MODEL_KEY"):
        ModelAPIClient(settings)


def test_neutral_key_names_can_use_selected_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    for key_env in ("INFERENCE_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY"):
        settings = ModelAPISettings(provider="openrouter", api_key_env=key_env)
        assert environment_api_key(settings) == "provider-key"


def test_environment_model_and_endpoint_overrides_reach_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("INFERENCE_MODEL", "example/vision-model")
    monkeypatch.setenv(
        "INFERENCE_ENDPOINT", "https://gateway.example/v1/chat/completions",
    )
    monkeypatch.setenv("INFERENCE_API_KEY", "explicit-route-key")
    calls: list[Request] = []

    def capture(request: Request, *, timeout: float) -> _Response:
        calls.append(request)
        return _Response({
            "detected": False, "confidence": 0.9, "description": "Clear.",
        })

    monkeypatch.setattr(model_api_module, "urlopen", capture)
    config = load_config(ROOT / "config.yaml")
    client = build_client(config.inference, config.model_api)
    assert client.classify_window(_window()).ok
    assert calls[0].full_url == "https://gateway.example/v1/chat/completions"
    assert calls[0].get_header("Authorization") == "Bearer explicit-route-key"
    assert json.loads(calls[0].data)["model"] == "example/vision-model"


def test_invalid_provider_is_rejected_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "unrecognized-route")
    with pytest.raises(ValueError, match="inference_service.provider"):
        load_config(ROOT / "config.yaml")


def test_nested_route_options_override_common_settings_without_leaking(
    config_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    service = data["inference_service"]
    service["extra_headers"] = {"X-Shared": "common"}
    service["providers"]["aiclub"]["extra_headers"] = {"X-AIClub": "local"}
    service["providers"]["openrouter"] = {
        "endpoint": "https://route.example/v1/chat/completions",
        "api_key_env": "CUSTOM_MODEL_KEY",
        "timeout_seconds": 17,
        "model": "custom/vision-model",
        "extra_headers": {"X-Shared": "route", "X-OpenRouter": "remote"},
        "request_parameters": {"temperature": 0.3, "provider": {"sort": "latency"}},
    }
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    selected = load_config(config_path)
    assert selected.model_api.endpoint == "https://route.example/v1/chat/completions"
    assert selected.model_api.api_key_env == "CUSTOM_MODEL_KEY"
    assert selected.model_api.timeout_seconds == 17
    assert selected.model_api.extra_headers == {
        "X-Shared": "route", "X-OpenRouter": "remote",
    }
    assert selected.inference.model == "custom/vision-model"
    assert selected.inference.request_parameters == {
        "temperature": 0.3, "max_tokens": 1200, "provider": {"sort": "latency"},
    }

    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    switched = load_config(config_path)
    assert switched.model_api.endpoint == ROUTES[0][1]
    assert switched.model_api.api_key_env == "AICLUB_API_KEY"
    assert switched.model_api.extra_headers == {
        "X-Shared": "common", "X-AIClub": "local",
    }
    assert switched.inference.request_parameters == {
        "temperature": 0.0, "max_tokens": 1200,
    }


@pytest.mark.parametrize("nested_override", [False, True])
def test_switch_discards_original_service_endpoint_and_key_name(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_override: bool,
) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    service = data["inference_service"]
    service["endpoint"] = "https://original.example/v1/chat/completions"
    service["api_key_env"] = "AICLUB_API_KEY"
    service["extra_headers"] = {"HTTP-Referer": "https://original.example"}
    data["inference"]["request_parameters"].update({
        "provider": {"only": ["legacy-backend"]},
        "reasoning": {"enabled": True},
        "enable_thinking": True,
    })
    if nested_override:
        service["providers"]["openrouter"].update({
            "endpoint": "https://selected.example/v1/chat/completions",
            "api_key_env": "CUSTOM_MODEL_KEY",
        })
    else:
        service.pop("provider")
        service.pop("providers")
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")

    selected = load_config(config_path)
    assert selected.model_api.endpoint == (
        "https://selected.example/v1/chat/completions"
        if nested_override else ROUTES[1][1]
    )
    assert selected.model_api.api_key_env == (
        "CUSTOM_MODEL_KEY" if nested_override else "OPENROUTER_API_KEY"
    )
    assert selected.model_api.timeout_seconds == service["timeout_seconds"]
    assert selected.model_api.extra_headers == (
        service["extra_headers"] if nested_override else {}
    )
    assert selected.inference.request_parameters == {
        "temperature": 0.0, "max_tokens": 1200, "reasoning": {"enabled": False},
        "provider": {"sort": "throughput", "allow_fallbacks": True,
                     "ignore": ["coreweave", "novita"]},
    }


def test_optional_null_route_blocks_keep_default_behavior(config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["aiclub"] = None
    service = data["inference_service"]
    service["extra_headers"] = None
    service["providers"] = {"aiclub": None}
    data["inference"]["request_parameters"] = None
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    selected = load_config(config_path)
    assert selected.model_api.endpoint == ROUTES[0][1]
    assert selected.model_api.extra_headers == {}
    assert selected.inference.request_parameters == {}

    service["providers"] = None
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(config_path).model_api.endpoint == ROUTES[0][1]


def test_mixed_legacy_and_service_settings_preserve_precedence(
    config_path: Path,
) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["aiclub"] = {
        "endpoint": "https://legacy.example/v1/chat/completions",
        "api_key_env": "AICLUB_API_KEY",
        "timeout_seconds": 9,
        "max_retries": 2,
    }
    service = data["inference_service"] = {
        "endpoint": "https://service.example/v1/chat/completions",
        "timeout_seconds": 20,
    }
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    neutral = load_config(config_path).model_api
    assert neutral.endpoint == "https://service.example/v1/chat/completions"
    assert neutral.timeout_seconds == 20
    assert neutral.max_retries == 2
    assert neutral.api_key_env == "AICLUB_API_KEY"

    service["api_key_env"] = "CUSTOM_MODEL_KEY"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(config_path).model_api.api_key_env == "CUSTOM_MODEL_KEY"

    service["providers"] = {"aiclub": {
        "endpoint": "https://selected.example/v1/chat/completions",
        "api_key_env": "SELECTED_MODEL_KEY",
        "timeout_seconds": 30,
    }}
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    selected = load_config(config_path).model_api
    assert selected.endpoint == "https://selected.example/v1/chat/completions"
    assert selected.api_key_env == "SELECTED_MODEL_KEY"
    assert selected.timeout_seconds == 30
    assert selected.max_retries == 2


@pytest.mark.parametrize("legacy_provider", ["aiclub", "openrouter"])
def test_legacy_connection_and_model_survive_until_route_is_changed(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_provider: str,
) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.pop("inference_service")
    data[legacy_provider] = {
        "endpoint": "https://legacy.example/v1/chat/completions",
        "api_key_env": f"{legacy_provider.upper()}_API_KEY",
        "timeout_seconds": 12,
        "max_retries": 1,
    }
    data["inference"].pop("profile", None)
    data["inference"]["model"] = "legacy-vision-model"
    data["inference"]["request_parameters"].update({
        "provider": {"only": ["legacy-backend"]},
        "reasoning": {"enabled": True},
        "enable_thinking": True,
    })
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    legacy = load_config(config_path)
    assert legacy.model_api.provider == legacy_provider
    assert legacy.model_api.endpoint == "https://legacy.example/v1/chat/completions"
    assert legacy.inference.model == "legacy-vision-model"
    assert legacy.model_api.timeout_seconds == 12
    assert legacy.model_api.max_retries == 1
    assert legacy.inference.request_parameters["reasoning"] == {"enabled": True}

    monkeypatch.setenv("INFERENCE_PROVIDER", "dashscope")
    switched = load_config(config_path)
    assert switched.model_api.provider == "dashscope"
    assert switched.model_api.endpoint == ROUTES[2][1]
    assert switched.model_api.api_key_env == "DASHSCOPE_API_KEY"
    assert switched.inference.model == "qwen3.5-plus"
    assert switched.inference.request_parameters == {
        "temperature": 0.0, "max_tokens": 1200, "enable_thinking": False,
    }


@pytest.mark.parametrize("provider", ["aiclub", "openrouter", "dashscope"])
def test_dry_run_builders_need_no_provider_credentials(
    monkeypatch: pytest.MonkeyPatch, provider: str,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    config = load_config(ROOT / "config.yaml")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("dry run must not call a provider")

    monkeypatch.setattr(model_api_module, "urlopen", forbidden)
    monkeypatch.setattr(model_api_module.time, "sleep", lambda _: None)
    binary = build_client(config.inference, config.model_api, dry_run=True)
    grounded = build_grounding_client(config.inference, config.model_api, dry_run=True)
    selected = build_selected_features_grounding_client(
        config.inference, config.model_api, features=("smoking",), dry_run=True,
    )
    fixed = build_fixed_five_grounding_client(
        config.inference, config.model_api, stranger_client=grounded, dry_run=True,
    )
    assert isinstance(binary, DryRunWindowClient)
    assert isinstance(grounded, DryRunGroundingClient)
    assert binary.classify_window(_window()).ok
    assert grounded.classify_window(_window()).ok
    assert selected._safety_client.classify_window(_window()).ok
    assert fixed._safety_client.classify_window(_window()).ok
