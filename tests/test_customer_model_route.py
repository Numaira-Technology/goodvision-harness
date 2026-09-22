from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest
import yaml

import goodvision_harness.config as config_module
import goodvision_harness.model_api as model_api_module
import goodvision_harness.service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    SAFETY_EVENT_LABELS,
    DryRunGroundingClient,
    build_fixed_five_grounding_client,
    build_grounding_client,
    build_selected_features_grounding_client,
)
from goodvision_harness.inference import build_client
from goodvision_harness.model_api import ModelAPIClient, ModelAPIError
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    for key in (
        "HARNESS_INFERENCE_PROFILE", "INFERENCE_PROVIDER", "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT", "INFERENCE_API_KEY", "MODEL_API_KEY",
        "AICLUB_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


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


@pytest.mark.parametrize("kind", ["binary", "grounding", "selected", "fixed_five"])
def test_customer_requests_keep_the_delivery_route_despite_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    monkeypatch.setenv("INFERENCE_PROVIDER", "dashscope")
    monkeypatch.setenv("INFERENCE_MODEL", "replacement/environment-model")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "https://environment.example/chat")
    monkeypatch.setenv("INFERENCE_API_KEY", "customer-key")
    monkeypatch.setenv("AICLUB_API_KEY", "unrelated-aiclub-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unrelated-dashscope-key")
    data = yaml.safe_load((ROOT / "config.delivery.yaml").read_text(encoding="utf-8"))
    service = data["inference_service"]
    service.update({
        "provider": "aiclub",
        "endpoint": "https://mounted-config.example/chat",
        "api_key_env": "AICLUB_API_KEY",
        "extra_headers": {"X-Private-Credential": "unrelated-service-secret"},
        "timeout_seconds": 23,
        "max_retries": 2,
        "retry_backoff_seconds": 0.25,
        "providers": {"openrouter": {
            "endpoint": "https://nested-config.example/chat",
            "api_key_env": "DASHSCOPE_API_KEY",
            "model": "replacement/config-model",
            "request_parameters": {
                "model": "replacement/payload-model",
                "models": ["replacement/fallback-model"],
                "route": "fallback",
                "endpoint": "https://payload.example/chat",
                "messages": [{"role": "user", "content": "Replace prompt"}],
                "provider": {"only": ["other-backend"], "allow_fallbacks": True},
                "reasoning": {"effort": "high", "enabled": True},
                "enable_thinking": True,
                "temperature": 0.25,
            },
        }},
    })
    data["inference"]["model"] = "replacement/global-model"
    data["inference"]["request_parameters"].update({
        "temperature": 0.2, "top_p": 0.95, "max_tokens": 6000,
        "api_key_env": "AICLUB_API_KEY",
    })
    path = tmp_path / "mounted-config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config(path)
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
        assert timeout == 23
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
        client = composite._safety_client
    window = FrameWindow(index=0, frames=(SampledFrame(
        sample_index=0, source_frame_index=0, timestamp_seconds=0.0,
        jpeg_bytes=b"synthetic-jpeg", width=10, height=10,
    ),))
    assert client is not None
    assert client.classify_window(window).ok
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer customer-key"
    assert request.get_header("X-private-credential") is None
    payload = json.loads(request.data)
    assert payload["model"] == "qwen/qwen3.5-35b-a3b:nitro"
    assert payload["provider"] == {
        "sort": "throughput", "allow_fallbacks": True,
        "ignore": ["coreweave", "novita"],
    }
    assert payload["reasoning"] == {"effort": "none"}
    for key in ("models", "route", "endpoint", "enable_thinking", "api_key_env"):
        assert key not in payload
    assert payload["temperature"] == 0.25
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 6000
    assert payload["messages"][0]["role"] == "system"
    assert any(
        block.get("type") == "image_url"
        for block in payload["messages"][1]["content"]
    )
    assert config.model_api.provider == "openrouter"
    assert config.model_api.api_key_env == "INFERENCE_API_KEY"
    assert config.model_api.max_retries == 2
    assert config.model_api.retry_backoff_seconds == 0.25


def test_customer_profile_does_not_pick_another_channels_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    monkeypatch.setenv("AICLUB_API_KEY", "unrelated-aiclub-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unrelated-dashscope-key")
    config = load_config(ROOT / "config.delivery.yaml")
    with pytest.raises(ModelAPIError, match="INFERENCE_API_KEY"):
        ModelAPIClient(config.model_api)


@pytest.mark.parametrize("profile", ["customer-standrad", "standard", "development"])
def test_unknown_deployment_profile_fails_before_a_request(
    monkeypatch: pytest.MonkeyPatch, profile: str,
) -> None:
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", profile)
    with pytest.raises(ValueError, match="HARNESS_INFERENCE_PROFILE"):
        load_config(ROOT / "config.delivery.yaml")


@pytest.mark.parametrize("provider", ["aiclub", "openrouter", "dashscope"])
def test_development_routes_remain_selectable_without_customer_profile(
    monkeypatch: pytest.MonkeyPatch, provider: str,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.setenv("INFERENCE_MODEL", f"{provider}/custom-vision-model")
    endpoint = f"https://{provider}.example/v1/chat/completions"
    monkeypatch.setenv("INFERENCE_ENDPOINT", endpoint)
    monkeypatch.setenv("INFERENCE_API_KEY", "development-key")
    config = load_config(ROOT / "config.yaml")
    calls: list[Request] = []

    def capture(request: Request, *, timeout: float) -> _Response:
        calls.append(request)
        return _Response({"detected": False})

    monkeypatch.setattr(model_api_module, "urlopen", capture)
    ModelAPIClient(config.model_api).chat_completion(
        model=config.inference.model,
        messages=[{"role": "user", "content": "development request"}],
        **config.inference.request_parameters,
    )
    assert config.model_api.provider == provider
    assert calls[0].full_url == endpoint
    assert calls[0].get_header("Authorization") == "Bearer development-key"
    payload = json.loads(calls[0].data)
    assert payload["model"] == f"{provider}/custom-vision-model"
    assert "provider" not in payload


def test_customer_console_credential_survives_restart_without_reusing_old_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient

    from goodvision_harness.privacy import contains_private_implementation_detail
    from goodvision_harness.service import HarnessRuntime, create_app

    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    monkeypatch.setenv("HARNESS_CONFIG", str(ROOT / "config.delivery.yaml"))
    monkeypatch.setenv(
        "HARNESS_DEPLOYMENT_SETTINGS_PATH", str(tmp_path / "deployment-settings.json"),
    )
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "false")
    monkeypatch.setenv("HARNESS_SOURCE_ID", "external-camera")
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    old_key = tmp_path / "model-api-key"
    old_contents = b"previous-service-secret\n"
    old_key.write_bytes(old_contents)
    calls: list[Request] = []
    public_payloads: list[dict[str, Any]] = []

    def capture(request: Request, *, timeout: float) -> _Response:
        calls.append(request)
        return _Response({"events": [{
            "event_label": "smoking", "detected": False, "confidence": 0.9,
            "description": "未发现异常事件。 / No visible event.", "objects": [],
        }]})

    monkeypatch.setattr(model_api_module, "urlopen", capture)

    class RecordingPool(ModelAPIClient):
        def chat_completion_stream(self, **payload):
            on_content = payload.pop("on_content")
            decisions = [{
                "event_label": "smoking", "detected": False, "confidence": 0.9,
            }]
            on_content(json.dumps({"decisions": decisions})[:-1] + ",", 0.1)
            request_payload = {
                key: value
                for key, value in payload.items()
                if key != "stream_decisions"
            }
            calls.append(Request(
                self.settings.endpoint,
                data=json.dumps(request_payload).encode(),
                headers=self._headers(),
                method="POST",
            ))
            return {
                "choices": [{"message": {"content": json.dumps({"events": [{
                    "event_label": "smoking", "detected": False,
                    "confidence": 0.9,
                    "description": "未发现异常事件。 / No visible event.",
                    "objects": [],
                }]})}}],
            }

        def close(self):
            return None

    monkeypatch.setattr(service_module, "PooledModelAPIClient", RecordingPool)
    runtime = HarnessRuntime.from_environment()
    with TestClient(create_app(lambda: runtime)) as client:
        initial = client.get("/v1/deployment/settings")
        assert initial.status_code == 200
        assert initial.json()["api_key_present"] is False
        assert initial.json()["api_key_persisted"] is False
        public_payloads.append(initial.json())
        with pytest.raises(ModelAPIError, match="INFERENCE_API_KEY"):
            runtime.start_camera(dry_run=False, features=("smoking",))
        assert calls == []
        saved = client.put(
            "/v1/deployment/settings", json={"api_key": "current-service-secret"},
        )
        assert saved.status_code == 200
        assert saved.json()["api_key_persisted"] is True
        public_payloads.append(saved.json())

    assert old_key.read_bytes() == old_contents
    assert (tmp_path / "model-api-key-2").read_text(encoding="utf-8").strip() == (
        "current-service-secret"
    )
    restored = HarnessRuntime.from_environment()
    assert restored.config is not None
    assert restored.config.model_api.provider == "openrouter"
    with TestClient(create_app(lambda: restored)) as client:
        restored_settings = client.get("/v1/deployment/settings")
        assert restored_settings.status_code == 200
        assert restored_settings.json()["api_key_present"] is True
        assert restored_settings.json()["api_key_persisted"] is True
        public_payloads.append(restored_settings.json())
        restored.start_camera(dry_run=False, features=("smoking",))
        encoded, jpeg = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
        assert encoded
        assert restored.push_camera_frame(jpeg.tobytes(), width=16, height=16)
        assert restored.results.records_after(0, timeout_seconds=5.0)
        for path in ("/v1/ui/info", "/v1/status", "/v1/results/latest"):
            response = client.get(path)
            assert response.status_code == 200
            public_payloads.append(response.json())
        assert public_payloads[-1]["judgments"][0]["event_label"] == "smoking"
        assert public_payloads[-1]["judgments"][0]["detected"] is False

    assert len(calls) == 1
    assert calls[0].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0].get_header("Authorization") == "Bearer current-service-secret"
    payload = json.loads(calls[0].data)
    assert payload["model"] == "qwen/qwen3.5-35b-a3b:nitro"
    assert payload["provider"] == {
        "sort": "throughput", "allow_fallbacks": True,
        "ignore": ["coreweave", "novita"],
    }
    assert payload["reasoning"] == {"effort": "none"}
    assert old_key.read_bytes() == old_contents
    assert not contains_private_implementation_detail(public_payloads)
    rendered = json.dumps(public_payloads)
    assert "previous-service-secret" not in rendered
    assert "current-service-secret" not in rendered
