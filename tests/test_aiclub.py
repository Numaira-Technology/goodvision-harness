from __future__ import annotations

import io
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

import goodvision_harness.aiclub as aiclub_module
from goodvision_harness.aiclub import (
    DEFAULT_AICLUB_ENDPOINT,
    AIClubClient,
    AIClubError,
    AIClubSettings,
    api_key_present,
)
from goodvision_harness.config import InferenceSettings
from goodvision_harness.inference import QwenWindowClient, build_client


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _http_error(code: int, payload: object) -> HTTPError:
    return HTTPError(
        url=DEFAULT_AICLUB_ENDPOINT,
        code=code,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def test_settings_defaults_target_aiclub() -> None:
    settings = AIClubSettings.from_mapping({})

    assert settings.endpoint == DEFAULT_AICLUB_ENDPOINT
    assert settings.api_key_env == "AICLUB_API_KEY"
    assert settings.timeout_seconds == 90.0
    assert settings.max_retries == 3


def test_settings_reject_invalid_endpoint_and_key_name() -> None:
    with pytest.raises(ValueError, match="aiclub.endpoint"):
        AIClubSettings(endpoint="ftp://example.com/chat")
    with pytest.raises(ValueError, match="aiclub.api_key_env"):
        AIClubSettings(api_key_env=" ")


def test_api_key_helpers_use_configured_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICLUB_API_KEY", "test-key")
    assert api_key_present(AIClubSettings())
    monkeypatch.setenv("AICLUB_API_KEY", "")
    assert not api_key_present(AIClubSettings())
    with pytest.raises(AIClubError, match="AICLUB_API_KEY"):
        AIClubClient(AIClubSettings())


def test_chat_completion_posts_openai_compatible_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Request] = {}
    settings = AIClubSettings(max_retries=0)

    def fake_urlopen(request: Request, timeout: float = 0.0) -> _FakeResponse:
        captured["request"] = request
        assert timeout == settings.timeout_seconds
        return _FakeResponse(
            {"choices": [{"message": {"content": "  hello  "}}]}
        )

    monkeypatch.setattr(aiclub_module, "urlopen", fake_urlopen)
    client = AIClubClient(settings, api_key="test-key")
    response = client.chat_completion(
        model="qwen3.5-plus",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=96,
        unused=None,
    )

    request = captured["request"]
    assert request.full_url == DEFAULT_AICLUB_ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "application/json"
    header_names = {key.lower() for key, _value in request.header_items()}
    assert "http-referer" not in header_names
    assert "x-title" not in header_names
    assert json.loads(request.data.decode("utf-8")) == {
        "temperature": 0.0,
        "max_tokens": 96,
        "model": "qwen3.5-plus",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert client.assistant_text(response) == "hello"


def test_assistant_text_joins_structured_text_blocks() -> None:
    text = AIClubClient.assistant_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "first"},
                            {"type": "image_url", "text": "ignore"},
                            {"type": "text", "text": "second"},
                        ]
                    }
                }
            ]
        }
    )

    assert text == "first\nsecond"


def test_retryable_http_error_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(_request: Request, timeout: float = 0.0) -> _FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(429, {"error": {"message": "slow down"}})
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(aiclub_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(aiclub_module.time, "sleep", lambda _delay: None)
    client = AIClubClient(
        AIClubSettings(max_retries=1, retry_backoff_seconds=0.0),
        api_key="test-key",
    )

    response = client.chat_completion(
        model="qwen3.5-plus",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert calls == 2
    assert client.assistant_text(response) == "ok"


def test_build_client_uses_aiclub_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = InferenceSettings(
        model="qwen3.5-plus",
        event_label="smoking",
        system_prompt="system",
        prompt="prompt",
    )

    monkeypatch.setenv("AICLUB_API_KEY", "test-key")
    client = build_client(settings, AIClubSettings())

    assert isinstance(client, QwenWindowClient)
    assert isinstance(client._transport, AIClubClient)


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
