from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import pytest

from goodvision_harness.aiclub import (
    DEFAULT_AICLUB_ENDPOINT,
    AIClubClient,
    AIClubError,
    AIClubSettings,
)
from goodvision_harness.config import RichLabelSettings, StreamSettings, VideoSettings
from tools import vlm_rich_smoke as smoke


@pytest.fixture(autouse=True)
def isolated_config_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("smoke tests must use mocked config and transport")

    # Prevent accidental .env loading or external calls if a test omits its mocks.
    monkeypatch.setattr(smoke, "load_config", unexpected)
    monkeypatch.setattr(AIClubClient, "_post_json", unexpected)
    for name in (
        "INFERENCE_API_KEY", "MODEL_API_KEY", "AICLUB_API_KEY",
        "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _config(endpoint: str = DEFAULT_AICLUB_ENDPOINT) -> SimpleNamespace:
    return SimpleNamespace(
        aiclub=AIClubSettings(endpoint=endpoint),
        rich_labeling=RichLabelSettings(
            enabled=True, backend="vlm",
            vlm_request_parameters={"enable_thinking": False},
        ),
        inference=SimpleNamespace(model="test-vlm"),
        video=VideoSettings(),
        stream=StreamSettings(),
    )


def _arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> Any:
    monkeypatch.setattr(smoke, "load_config", lambda _: _config())
    monkeypatch.setenv("INFERENCE_API_KEY", "test-secret-never-persist")
    image = tmp_path / "frame.jpg"
    ok, encoded = cv2.imencode(".jpg", np.zeros((64, 96, 3), dtype=np.uint8))
    assert ok
    image.write_bytes(encoded.tobytes())
    return smoke.build_parser().parse_args([
        "--image", str(image), "--output-dir", str(tmp_path / "result"), *extra,
    ])


def _reply(text: str = '{"objects": []}', finish: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_expected_endpoint_defaults_to_exact_aiclub_destination() -> None:
    arguments = smoke.build_parser().parse_args(["--image", "unused.jpg"])
    assert arguments.expected_endpoint == DEFAULT_AICLUB_ENDPOINT
    assert arguments.save_model_response is False


@pytest.mark.parametrize("endpoint", [
    "https://different.example/v1/chat/completions?key=private-value",
    DEFAULT_AICLUB_ENDPOINT + "/",
    DEFAULT_AICLUB_ENDPOINT + "?route=other",
])
def test_endpoint_mismatch_precedes_key_media_and_client_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoint: str,
) -> None:
    monkeypatch.setattr(smoke, "load_config", lambda _: _config(endpoint))

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("mismatched destination must stop before key/media/client access")

    monkeypatch.setattr(smoke, "api_key_present", unexpected)
    monkeypatch.setattr(smoke, "_sample_frames", unexpected)
    monkeypatch.setattr(smoke, "VLMRichLabelClient", unexpected)
    arguments = smoke.build_parser().parse_args([
        "--image", "unused.jpg", "--output-dir", str(tmp_path / "result"),
    ])
    arguments.image = SimpleNamespace(is_file=unexpected)
    with pytest.raises(smoke.EndpointMismatchError, match="exactly match"):
        smoke.run_smoke(arguments)
    assert not arguments.output_dir.exists()


def test_endpoint_mismatch_has_clear_safe_cli_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        smoke, "load_config",
        lambda _: _config("https://different.example/?key=private-value"),
    )
    assert smoke.main(["--image", "unused.jpg", "--dry-run"]) == 1
    error = capsys.readouterr().err
    assert "does not exactly match --expected-endpoint" in error
    assert "before reading media or making API requests" in error
    assert "private-value" not in error
    assert "different.example" not in error


def test_diagnostic_saves_only_assistant_content_and_token_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    arguments = _arguments(monkeypatch, tmp_path, "--save-model-response")
    response = _reply()
    secret = "test-secret-never-persist"
    response.update(api_key=secret, headers={"Authorization": secret})
    response["request"] = {"messages": [secret], "api_key": secret}
    response["choices"][0]["message"]["request"] = secret
    response["usage"].update(
        api_key=secret, request={"body": secret},
        completion_tokens_details={"reasoning_tokens": 2, "api_key": secret},
    )
    sent: list[dict[str, Any]] = []

    def post(_client: AIClubClient, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        return response

    monkeypatch.setattr(AIClubClient, "_post_json", post)
    metrics = smoke.run_smoke(arguments)
    assert metrics["status"] == "success"
    assert metrics["endpoint"] == DEFAULT_AICLUB_ENDPOINT
    assert metrics["model"] == "test-vlm"
    assert sent[0]["enable_thinking"] is False
    diagnostic = json.loads(
        (arguments.output_dir / "response-001.json").read_text(encoding="utf-8")
    )
    assert diagnostic == {
        "assistant_text": '{"objects": []}',
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    for path in arguments.output_dir.glob("*.json"):
        assert secret not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(("text", "finish", "reason"), [
    ('{"objects": [', "length", "rich annotation response was truncated"),
    ("invalid json", "stop", "rich annotation must be JSON"),
])
def test_protocol_failure_keeps_local_reason_and_preparse_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    text: str, finish: str, reason: str,
) -> None:
    arguments = _arguments(monkeypatch, tmp_path, "--save-model-response")
    monkeypatch.setattr(AIClubClient, "_post_json", lambda *_: _reply(text, finish))
    metrics = smoke.run_smoke(arguments)
    assert metrics["status"] == "failure"
    assert metrics["error_type"] == "RichLabelProtocolError"
    assert metrics["error_reason"] == reason
    assert metrics["requests"][0]["error_reason"] == reason
    assert metrics["successful_samples"] == 0
    diagnostic = json.loads(
        (arguments.output_dir / "response-001.json").read_text(encoding="utf-8")
    )
    assert diagnostic["assistant_text"] == text
    assert diagnostic["finish_reason"] == finish


def test_remote_failure_never_saves_raw_error_or_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    arguments = _arguments(monkeypatch, tmp_path, "--save-model-response")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AIClubError("remote body with test-secret-never-persist")

    monkeypatch.setattr(AIClubClient, "_post_json", fail)
    metrics = smoke.run_smoke(arguments)
    assert metrics["status"] == "failure"
    assert metrics["error_type"] == AIClubError.__name__
    assert "error_reason" not in metrics
    assert "error_reason" not in metrics["requests"][0]
    assert list(arguments.output_dir.glob("response-*.json")) == []
    assert "test-secret-never-persist" not in (
        arguments.output_dir / "metrics.json"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(("cause", "expected"), [
    (
        HTTPError(
            "https://private.example/?key=test-secret-never-persist", 504,
            "remote body with test-secret-never-persist", None, None,
        ),
        {"http_status": 504},
    ),
    (
        TimeoutError("timeout with test-secret-never-persist"),
        {"error_category": "timeout"},
    ),
    (
        URLError("network reason with test-secret-never-persist"),
        {"error_category": "network_error", "network_reason_type": "str"},
    ),
    (
        URLError(TimeoutError("nested test-secret-never-persist")),
        {"error_category": "network_error", "network_reason_type": "TimeoutError"},
    ),
])
def test_transport_failure_records_safe_immediate_cause_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    cause: Exception, expected: dict[str, Any],
) -> None:
    arguments = _arguments(monkeypatch, tmp_path, "--save-model-response")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AIClubError("raw remote test-secret-never-persist") from cause

    monkeypatch.setattr(AIClubClient, "_post_json", fail)
    metrics = smoke.run_smoke(arguments)
    assert metrics["status"] == "failure"
    assert metrics["error_type"] == AIClubError.__name__
    assert "error_reason" not in metrics
    for key, value in expected.items():
        assert metrics[key] == value
        assert metrics["requests"][0][key] == value
    if "http_status" in expected:
        assert "error_category" not in metrics
        assert "network_reason_type" not in metrics
    else:
        assert "http_status" not in metrics
    assert list(arguments.output_dir.glob("response-*.json")) == []
    for path in arguments.output_dir.glob("*.json"):
        saved = path.read_text(encoding="utf-8")
        assert "test-secret-never-persist" not in saved
        assert "private.example" not in saved


@pytest.mark.parametrize("save_response", [False, True])
def test_records_selected_route_and_returned_openrouter_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, save_response: bool,
) -> None:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    extra = ["--expected-endpoint", endpoint]
    if save_response:
        extra.append("--save-model-response")
    arguments = _arguments(monkeypatch, tmp_path, *extra)
    config = _config(endpoint)
    config.aiclub = AIClubSettings(endpoint=endpoint, provider="openrouter")
    route = {"only": ["DeepInfra"], "order": ["DeepInfra"], "allow_fallbacks": False}
    secret = "test-secret-never-persist"
    parameters = {
        "provider": {**route, "api_key": secret, "headers": {"Authorization": secret}},
        "enable_thinking": False,
    }
    config.rich_labeling = replace(
        config.rich_labeling, vlm_request_parameters=parameters,
    )
    monkeypatch.setattr(smoke, "load_config", lambda _: config)
    response = {**_reply(), "provider": "DeepInfra", "headers": {"secret": secret}}
    sent: list[dict[str, Any]] = []

    def post(_client: AIClubClient, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        return response

    monkeypatch.setattr(AIClubClient, "_post_json", post)
    metrics = smoke.run_smoke(arguments)
    assert metrics["status"] == "success"
    assert metrics["selected_provider"] == "openrouter"
    assert metrics["request_route"] == route
    assert sent[0]["provider"]["only"] == ["DeepInfra"]
    assert metrics["response_providers"] == ["DeepInfra"]
    assert metrics["requests"][0]["response_provider"] == "DeepInfra"
    response_path = arguments.output_dir / "response-001.json"
    assert response_path.exists() is save_response
    if save_response:
        saved_response = json.loads(response_path.read_text(encoding="utf-8"))
        assert saved_response["provider"] == "DeepInfra"
    for path in arguments.output_dir.glob("*.json"):
        saved = path.read_text(encoding="utf-8")
        assert secret not in saved
        assert "Authorization" not in saved
