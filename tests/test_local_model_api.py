from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from goodvision_harness import model_api, scene_model, vlm_rich_labeling
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.model_api import (
    DEFAULT_LOCAL_API_KEY_ENV,
    DEFAULT_LOCAL_ENDPOINT,
    ModelAPIClient,
    ModelAPISettings,
    PooledModelAPIClient,
    environment_api_key,
)
from goodvision_harness.model_routing import resolve_model_route
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.streaming_model_api import complete_stream


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "INFERENCE_PROVIDER", "INFERENCE_ENDPOINT", "INFERENCE_MODEL",
        "HARNESS_INFERENCE_PROFILE", "LOCAL_INFERENCE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "INFERENCE_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY",
        "AICLUB_API_KEY", "DASHSCOPE_API_KEY",
    ):
        monkeypatch.setenv(name, "cloud-secret-" + name)


def _completion(text: str = "local reply") -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def _jpeg() -> bytes:
    with Image.new("RGB", (48, 32), "red") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        return buffer.getvalue()


def test_local_provider_defaults_and_isolated_environment_key(monkeypatch) -> None:
    settings, model, parameters = resolve_model_route(
        {"inference_service": {"provider": "local"}}, {},
    )
    assert settings.endpoint == DEFAULT_LOCAL_ENDPOINT
    assert settings.api_key_env == DEFAULT_LOCAL_API_KEY_ENV
    assert model == "local-vision"
    assert parameters == {}
    assert environment_api_key(settings) == ""
    for cloud_name in ("INFERENCE_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY"):
        copied = ModelAPISettings(provider="local", api_key_env=cloud_name)
        assert copied.api_key_env == DEFAULT_LOCAL_API_KEY_ENV
        assert environment_api_key(copied) == ""
    monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", " local-key ")
    assert environment_api_key(settings) == "local-key"


@pytest.mark.parametrize("customer_profile", [False, True])
def test_switching_to_local_discards_cloud_route_and_private_headers(
    monkeypatch, customer_profile,
) -> None:
    monkeypatch.setenv("INFERENCE_PROVIDER", "local")
    if customer_profile:
        monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    settings, model, parameters = resolve_model_route({
        "inference_service": {
            "provider": "openrouter",
            "endpoint": "https://cloud.example/v1/chat/completions",
            "api_key_env": "OPENROUTER_API_KEY",
            "extra_headers": {"X-Private": "cloud-secret", "Authorization": "old"},
            "timeout_seconds": 17,
            "providers": {
                "openrouter": {"extra_headers": {"X-Cloud": "private"}},
                "local": {
                    "endpoint": "http://127.0.0.1:9000/v1/chat/completions",
                    "model": "customer-vision",
                    "extra_headers": {"X-Local": "local-value"},
                },
            },
        },
    }, {
        "model": "cloud/model",
        "request_parameters": {
            "temperature": 0.1, "max_tokens": 1200,
            "provider": {"only": ["cloud"]}, "reasoning": {"enabled": False},
            "enable_thinking": False, "models": ["cloud/fallback"],
            "endpoint": "https://cloud.example", "route": "cloud",
            "api_key_env": "MODEL_API_KEY",
        },
    })
    assert settings.provider == "local"
    assert settings.endpoint == "http://127.0.0.1:9000/v1/chat/completions"
    assert settings.timeout_seconds == 17
    assert settings.api_key_env == DEFAULT_LOCAL_API_KEY_ENV
    assert settings.extra_headers == {"X-Local": "local-value"}
    assert model == "customer-vision"
    assert parameters == {"temperature": 0.1, "max_tokens": 1200}


def test_explicit_local_configuration_overrides_customer_cloud_profile(monkeypatch):
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    settings, model, _ = resolve_model_route({
        "inference_service": {"provider": "local"},
    }, {"model": "installed-vision"})
    assert settings.provider == "local"
    assert model == "installed-vision"


@pytest.mark.parametrize("local_key", [None, "local-secret"])
def test_standard_local_request_uses_only_optional_local_auth(monkeypatch, local_key):
    if local_key:
        monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", local_key)
    seen = []

    def capture(request, **_kwargs):
        seen.append(request)
        return io.BytesIO(json.dumps(_completion()).encode())

    monkeypatch.setattr(model_api, "urlopen", capture)
    client = ModelAPIClient(ModelAPISettings(
        provider="local", extra_headers={"aUthOriZation": "cloud-secret"},
    ))
    result = client.chat_completion(
        model="local-vision", messages=[{"role": "user", "content": "hello"}],
    )
    assert result == _completion()
    request = seen[0]
    assert request.full_url == DEFAULT_LOCAL_ENDPOINT
    assert request.get_header("Authorization") == (
        f"Bearer {local_key}" if local_key else None
    )
    assert "cloud-secret" not in str(request.header_items())
    assert json.loads(request.data)["model"] == "local-vision"


def test_pooled_local_request_works_without_credentials(monkeypatch):
    seen = []

    def capture(request):
        seen.append(request)
        return httpx.Response(200, json=_completion())

    transport_client = httpx.Client(transport=httpx.MockTransport(capture))
    monkeypatch.setattr(model_api.httpx, "Client", lambda **_kwargs: transport_client)
    client = PooledModelAPIClient(ModelAPISettings(provider="local"))
    try:
        result = client.chat_completion(
            model="local-vision", messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        client.close()
    assert result == _completion()
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize("local_key", [None, "local-secret"])
def test_streaming_local_isolates_reused_pool_credentials(monkeypatch, local_key):
    if local_key:
        monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", local_key)
    seen = []

    def capture(request):
        seen.append(request)
        chunk = {"choices": [{
            "index": 0, "delta": {"content": "local reply"}, "finish_reason": "stop",
        }]}
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"},
            content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode(),
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(capture),
            headers={"Authorization": "cloud-secret", "X-Private": "cloud-secret"},
            cookies={"cloud_session": "cloud-secret"},
            params={"cloud_key": "cloud-secret"},
            auth=("cloud-user", "cloud-secret"),
        ) as client:
            result = await complete_stream(
                client, settings=ModelAPISettings(
                    provider="local", extra_headers={"authorization": "stale"},
                ), model="local-vision",
                messages=[{"role": "user", "content": "hello"}],
                deadline_seconds=1,
            )
            assert client.headers["Authorization"] == "cloud-secret"
            return result

    result = asyncio.run(run())
    assert result["choices"][0]["message"]["content"] == "local reply"
    request = seen[0]
    assert str(request.url) == DEFAULT_LOCAL_ENDPOINT
    assert request.headers.get("authorization") == (
        f"Bearer {local_key}" if local_key else None
    )
    assert "x-private" not in request.headers and "cookie" not in request.headers
    assert "cloud-secret" not in str(request.headers)


def test_temporal_local_inspection_needs_no_cloud_key_and_isolates_pool(monkeypatch):
    seen = []

    def capture(request):
        seen.append(request)
        return httpx.Response(200, json=_completion("[]"))

    monkeypatch.setattr(
        scene_model, "create_vlm_http_client", lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(capture),
            headers={"Authorization": "cloud-secret", "X-Private": "cloud-secret"},
            cookies={"cloud_session": "cloud-secret"},
            auth=("cloud-user", "cloud-secret"),
        ),
    )
    jpeg = _jpeg()
    bbox = NormalizedBBox(0.1, 0.1, 0.6, 0.9)
    job = TemporalInspection(
        "session", "camera",
        InspectionRequest("request", ("person-1",), "new_person", ("smoking",), 0, 1),
        (EvidenceFrame("session", 0, 0, jpeg), EvidenceFrame("session", 1, 1, jpeg)),
        ((0, "person-1", bbox), (1, "person-1", bbox)),
    )
    inspector = TemporalVLMInspector(
        ModelAPISettings(provider="local"), model="local-vision",
    )
    try:
        assert inspector.inspect(job) == ()
    finally:
        inspector.close()
    assert str(seen[0].url) == DEFAULT_LOCAL_ENDPOINT
    for header in ("authorization", "x-private", "cookie"):
        assert header not in seen[0].headers


def test_local_rich_annotation_does_not_require_cloud_credentials(monkeypatch):
    keys = []
    client = vlm_rich_labeling.VLMRichLabelClient(
        ModelAPISettings(provider="local"), model="local-vision",
    )

    def infer(_image, key, _started):
        keys.append(key)
        return (), _completion('{"objects": []}')

    monkeypatch.setattr(client, "_infer_frame", infer)
    try:
        result = client.infer(
            _jpeg(), stream_session_id="session", frame_sequence=1,
            timestamp_seconds=0,
        )
    finally:
        client.close()
    assert result.objects == ()
    assert keys == [""]
