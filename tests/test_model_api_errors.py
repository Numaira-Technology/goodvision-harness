"""Transport failures expose bounded diagnostics without provider response data."""

from __future__ import annotations

import io
from dataclasses import replace
from urllib.error import HTTPError, URLError

import httpx
import pytest

import goodvision_harness.model_api as model_api_module
from goodvision_harness.model_api import (
    ModelAPIClient,
    ModelAPIError,
    ModelAPISettings,
    PooledModelAPIClient,
)

PRIVATE_DETAIL = "private-provider-token-and-response"


@pytest.fixture(params=["urllib", "pooled"])
def transport_client(request):
    settings = ModelAPISettings(
        endpoint=f"https://example.invalid/completions?token={PRIVATE_DETAIL}",
        max_retries=0, retry_backoff_seconds=0,
    )
    client_type = ModelAPIClient if request.param == "urllib" else PooledModelAPIClient
    client = client_type(settings, api_key=PRIVATE_DETAIL)
    yield client, request.param
    if isinstance(client, PooledModelAPIClient):
        client.close()


def _intercept(monkeypatch, client, backend, handler):
    if backend == "urllib":
        monkeypatch.setattr(model_api_module, "urlopen", handler)
    else:
        monkeypatch.setattr(client._http, "post", handler)


def _request(client):
    return client.chat_completion(
        model="test", messages=[{"role": "user", "content": "test"}]
    )


def _assert_error(error, code, status=None):
    assert error.code == code
    assert error.status_code == status
    assert PRIVATE_DETAIL not in str(error)
    assert PRIVATE_DETAIL not in repr(vars(error))
    assert "example.invalid" not in str(error)


def test_model_error_preserves_legacy_constructor_and_bounds_metadata():
    error = ModelAPIError("existing message")
    assert str(error) == "existing message"
    assert error.code == "request_failed"
    assert error.status_code is None
    untrusted = ModelAPIError("safe message", code=PRIVATE_DETAIL, status_code=True)
    assert untrusted.code == "request_failed"
    assert untrusted.status_code is None


@pytest.mark.parametrize("retries", [0, 1])
def test_http_429_retains_only_status_after_retry_exhaustion(
    transport_client, monkeypatch, retries,
):
    client, backend = transport_client
    client.settings = replace(client.settings, max_retries=retries)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if backend == "urllib":
            raise HTTPError(
                client.settings.endpoint, 429, PRIVATE_DETAIL, {},
                io.BytesIO(PRIVATE_DETAIL.encode()),
            )
        return httpx.Response(429, text=PRIVATE_DETAIL)

    _intercept(monkeypatch, client, backend, fail)
    with pytest.raises(ModelAPIError) as raised:
        _request(client)
    _assert_error(raised.value, "http_error", 429)
    assert len(calls) == retries + 1


@pytest.mark.parametrize("failure", ["timeout", "connection"])
def test_transport_failure_categories_do_not_copy_private_exception_details(
    transport_client, monkeypatch, failure,
):
    client, backend = transport_client

    def fail(*args, **kwargs):
        if backend == "urllib":
            if failure == "timeout":
                raise TimeoutError(PRIVATE_DETAIL)
            raise URLError(PRIVATE_DETAIL)
        error_type = httpx.ReadTimeout if failure == "timeout" else httpx.ConnectError
        raise error_type(PRIVATE_DETAIL)

    _intercept(monkeypatch, client, backend, fail)
    with pytest.raises(ModelAPIError) as raised:
        _request(client)
    _assert_error(raised.value, failure)


def test_urllib_wrapped_socket_timeout_is_timeout(monkeypatch):
    client = ModelAPIClient(ModelAPISettings(max_retries=0), api_key="test")

    def fail(*args, **kwargs):
        raise URLError(TimeoutError(PRIVATE_DETAIL))

    monkeypatch.setattr(model_api_module, "urlopen", fail)
    with pytest.raises(ModelAPIError) as raised:
        _request(client)
    _assert_error(raised.value, "timeout")


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"not-json-private-provider-token-and-response", "invalid_response"),
        (b"\xff", "invalid_response"),
        (b"[]", "invalid_response"),
        (b" \r\n", "empty_response"),
    ],
)
def test_invalid_or_empty_http_body_has_safe_diagnostics(
    transport_client, monkeypatch, body, code,
):
    client, backend = transport_client

    def respond(*args, **kwargs):
        return (
            io.BytesIO(body) if backend == "urllib"
            else httpx.Response(200, content=body)
        )

    _intercept(monkeypatch, client, backend, respond)
    with pytest.raises(ModelAPIError) as raised:
        _request(client)
    _assert_error(raised.value, code)


@pytest.mark.parametrize(
    "content", [None, "", " \n", [], [{"type": "text", "text": " "}]],
)
def test_empty_assistant_content_is_distinct_from_invalid_payload(content):
    with pytest.raises(ModelAPIError) as raised:
        ModelAPIClient.assistant_text({"choices": [{"message": {"content": content}}]})
    _assert_error(raised.value, "empty_response")


@pytest.mark.parametrize(
    "response",
    [
        {"error": PRIVATE_DETAIL},
        {"choices": []},
        {"choices": [{"message": {"content": 42}}]},
        {"choices": [{"message": {"content": [
            {"type": "text", "text": {"private": PRIVATE_DETAIL}},
        ]}}]},
    ],
)
def test_malformed_assistant_content_is_invalid_response(response):
    with pytest.raises(ModelAPIError) as raised:
        ModelAPIClient.assistant_text(response)
    _assert_error(raised.value, "invalid_response")


def test_valid_structured_assistant_text_is_preserved():
    assert ModelAPIClient.assistant_text({"choices": [{"message": {"content": [
        {"type": "text", "text": " first "},
        {"type": "image", "image": "ignored"},
        {"type": "text", "text": "second"},
    ]}}]}) == "first \nsecond"
