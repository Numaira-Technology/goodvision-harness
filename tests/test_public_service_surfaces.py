from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from goodvision_harness import service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.delivery import WebhookDelivery
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.result_bus import ResultBus
from goodvision_harness.rich_labeling import RichLabelCoordinator
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEXT = "OpenRouter Qwen InsightFace PaddleX PP-LCNet_x1_0_pedestrian_attribute"
PRIVATE_MARKERS = ("openrouter", "qwen", "insightface", "paddle", "pp-lcnet")


def _assert_public(value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False).lower()
    assert not any(marker in rendered for marker in PRIVATE_MARKERS)


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Credential precedence tests must never reload a developer's real .env.
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    for name in (
        "INFERENCE_API_KEY", "AICLUB_API_KEY", "MODEL_API_KEY",
        "OPENROUTER_API_KEY", "LEGACY_CUSTOM_KEY", "HARNESS_STREAM_URL", "RTSP_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")


@pytest.mark.parametrize("path", ("/docs", "/redoc"))
def test_api_reference_is_self_contained_and_preserves_specification(path: str) -> None:
    client = TestClient(create_app())
    response = client.get(path)
    specification = client.get("/openapi.json")
    assert response.status_code == specification.status_code == 200
    text = response.text.lower()
    assert "goodvision api reference" in text
    assert "/v1/results/events" in text
    assert "/v1/results/stranger/latest" in text
    assert "/v1/faces/strangers/{stranger_id}/photo" in text
    assert 'href="/openapi.json"' in text
    assert not any(name in text for name in ("swagger", "redoc", "fastapi", "jsdelivr"))
    assert not re.search(r'(?:src|href)\s*=\s*[\"\'](?:https?:)?//', text)
    assert "<script" not in text
    assert "default-src 'none'" in response.headers["content-security-policy"]
    _assert_public(specification.json())


def test_validation_keeps_error_contract_without_echoing_input() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/results/events", params={"after": PRIVATE_TEXT})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["query", "after"]
    assert error["type"] == "int_parsing"
    assert error["input"] == "[redacted]"
    assert isinstance(error["msg"], str)
    _assert_public(response.json())


def test_http_exception_detail_is_projected_and_status_and_headers_remain() -> None:
    app = create_app()

    @app.get("/audit-error")
    async def audit_error() -> None:
        raise HTTPException(409, detail={"reason": PRIVATE_TEXT}, headers={"Retry-After": "5"})

    response = TestClient(app).get("/audit-error")
    assert response.status_code == 409
    assert response.headers["retry-after"] == "5"
    assert "reason" in response.json()["detail"]
    _assert_public(response.json())


def test_control_errors_do_not_reflect_arbitrary_commands(
    clean_environment: None,
    tmp_path: Path,
) -> None:
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=DeploymentSettingsStore(tmp_path / "settings.json"),
    )
    with TestClient(create_app(lambda: runtime)) as client:
        invalid_setting = client.put("/v1/deployment/settings", json={PRIVATE_TEXT: True})
        assert invalid_setting.status_code == 400
        assert invalid_setting.json() == {"detail": "unknown deployment setting"}
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json()["type"] == "hello"
            websocket.send_json({"type": PRIVATE_TEXT})
            assert websocket.receive_json() == {"type": "error", "message": "unknown command"}
            websocket.send_json({"type": "start", "mode": "camera", "features": [PRIVATE_TEXT]})
            assert websocket.receive_json() == {"type": "error", "message": "unsupported feature"}


def test_result_bus_projects_public_records_without_mutating_caller() -> None:
    payload = {"message_id": "observation-1", "description": PRIVATE_TEXT, "nested": [PRIVATE_TEXT]}
    bus = ResultBus()
    published = bus.publish(payload)
    _assert_public(published.payload)
    _assert_public(bus.latest().payload)
    _assert_public(bus.records_after(0)[0].payload)
    assert payload["description"] == PRIVATE_TEXT


def test_webhook_projects_payload_and_preserves_matching_idempotency() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    delivery = WebhookDelivery("http://receiver.test/events", transport=httpx.MockTransport(handler))
    delivery.start()
    try:
        assert delivery.submit({"message_id": "Qwen", "description": PRIVATE_TEXT})
        assert delivery.flush(timeout_seconds=2)
    finally:
        delivery.close()
    payload = json.loads(requests[0].content)
    assert requests[0].headers["idempotency-key"] == payload["message_id"]
    _assert_public(payload)
    assert delivery.stats()["delivered"] == 1


def test_webhook_network_failure_reports_failure_without_raw_exception() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(PRIVATE_TEXT)

    delivery = WebhookDelivery(
        "http://receiver.test/events", transport=httpx.MockTransport(handler), max_retries=0
    )
    delivery.start()
    try:
        assert delivery.submit({"message_id": "observation-1"})
        assert delivery.flush(timeout_seconds=2)
    finally:
        delivery.close()
    assert delivery.stats()["failed"] == 1
    assert delivery.stats()["last_error"] == "webhook request failed"


def test_rich_failure_does_not_publish_exception_class_or_text(caplog: pytest.LogCaptureFixture) -> None:
    class PaddleRuntimeError(RuntimeError):
        pass

    class Client:
        def infer(self, *_args: Any, **_kwargs: Any) -> None:
            raise PaddleRuntimeError(PRIVATE_TEXT)

    coordinator = RichLabelCoordinator(Client(), max_fps=10_000)
    try:
        assert coordinator.submit(b"fixture", stream_session_id="session-1", frame_sequence=1, timestamp_seconds=0)
        assert coordinator.wait_until_idle()
        assert coordinator.stats()["failed"] == 1
        assert coordinator.stats()["state"] == "degraded"
        assert coordinator.stats()["last_error"] == "rich-label inference failed"
        _assert_public(coordinator.stats())
        _assert_public(caplog.text)
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "legacy", ("AICLUB_API_KEY", "MODEL_API_KEY", "OPENROUTER_API_KEY")
)
def test_neutral_credentials_take_precedence_and_legacy_still_works(
    clean_environment: None, monkeypatch: pytest.MonkeyPatch, legacy: str
) -> None:
    config = load_config(ROOT / "config.yaml")
    monkeypatch.setenv(legacy, "legacy-fixture")
    assert HarnessRuntime(config)._aiclub_api_key == "legacy-fixture"
    monkeypatch.setenv("INFERENCE_API_KEY", "neutral-fixture")
    assert HarnessRuntime(config)._aiclub_api_key == "neutral-fixture"


def test_explicit_provider_key_does_not_borrow_delivery_credentials(
    clean_environment: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config, aiclub=replace(config.aiclub, api_key_env="LEGACY_CUSTOM_KEY"),
    )
    monkeypatch.setenv("INFERENCE_API_KEY", "other-provider-fixture")
    assert HarnessRuntime(config)._aiclub_api_key is None
    monkeypatch.setenv("LEGACY_CUSTOM_KEY", "explicit-provider-fixture")
    assert HarnessRuntime(config)._aiclub_api_key == "explicit-provider-fixture"


def test_saved_credentials_keep_precedence_over_environment(
    clean_environment: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    store.save_api_key("saved-fixture")
    monkeypatch.setenv("INFERENCE_API_KEY", "neutral-fixture")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"), deployment_store=store)
    assert runtime._aiclub_api_key == "saved-fixture"


def test_service_launcher_hides_server_header_and_maps_capture_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(service_module.uvicorn, "run", lambda *_args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("VIDEO_CAPTURE_OPTIONS", "neutral-options")
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "legacy-options")
    service_module.main()
    assert calls[0]["server_header"] is False
    assert service_module.os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "neutral-options"


def test_service_launcher_keeps_legacy_capture_options_when_neutral_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module.uvicorn, "run", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("VIDEO_CAPTURE_OPTIONS", raising=False)
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "legacy-options")
    service_module.main()
    assert service_module.os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "legacy-options"
