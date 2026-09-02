from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import goodvision_harness.config as config_module
import goodvision_harness.overlay as overlay_module
from goodvision_harness.contracts import (
    PUBLIC_MODEL_NAME,
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delivery import WebhookDelivery
from goodvision_harness.privacy import (
    contains_private_implementation_detail,
    public_safe_value,
    redact_public_text,
)
from goodvision_harness.result_bus import ResultBus

NAMES = ("Claude", "Gemini", "Google", "Amazon", "Timm", "Azure")


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)


@pytest.mark.parametrize("value", [
    *NAMES, "torch", "torches", "torchbearer", "pandas", "buffalo", "antelope",
    "Claude-user", "Gemini-person", "GoogleSmith", "Amazonia",
    "Claude carries a torch near Amazon's entrance.",
])
def test_ordinary_names_objects_and_ids_are_preserved(value: str) -> None:
    assert redact_public_text(value) == value
    assert public_safe_value({value: [value]}) == {value: [value]}
    # Logs and release audits remain strict; data projection has more context.
    assert contains_private_implementation_detail(value)


@pytest.mark.parametrize("value", [
    "torch.nn.Module", "torchvision", "TorchRuntimeError", "torch-2.5.1",
    "pandas.DataFrame", "timm.create_model", "import torch", "torch backend",
    "claude-3.5-sonnet", "claude-sonnet-4", "gemini-2.5-flash", "Gemini model",
    "GOOGLE_API_KEY", "AWS_SDK", "AmazonClientError", "Google Cloud",
    "https://generativelanguage.googleapis.com/v1/models",
    "https://api.anthropic.com/v1/messages", "https://bedrock.amazonaws.com",
    "buffalo_l", "antelopev2",
])
def test_actual_implementation_forms_stay_redacted(value: str) -> None:
    projected = redact_public_text(value)
    assert projected != value
    assert not contains_private_implementation_detail(projected)


def _result() -> AnalysisResult:
    return AnalysisResult(
        message_id="Google",
        source_id="Amazon",
        stream_session_id="Gemini-person",
        sequence=1,
        emitted_at="2026-09-03T00:00:00Z",
        frame=FrameReference(1, 1.0, "2026-09-03T00:00:00Z", 320, 180),
        judgments=(EventJudgment(
            "weapon", True, 0.9, "Claude carries a torch.",
            object_ids=("torch-object",),
        ),),
        objects=(GroundedObject(
            "torch-object", "torch", NormalizedBBox(0.1, 0.1, 0.8, 0.8),
            confidence=0.8, attributes={"person_id": "Claude-user", "name": "Claude"},
        ),),
        model_name="qwen3.5-plus",
        model_latency_seconds=0.1,
    )


def test_serialization_sse_history_and_webhook_preserve_domain_values() -> None:
    payload = _result().as_dict()
    assert payload["message_id"] == "Google"
    assert payload["source_id"] == "Amazon"
    assert payload["objects"][0]["label"] == "torch"
    assert payload["objects"][0]["attributes"]["name"] == "Claude"
    assert payload["judgments"][0]["object_ids"] == ["torch-object"]
    assert payload["judgments"][0]["description"] == "Claude carries a torch."
    assert payload["model"]["name"] == PUBLIC_MODEL_NAME
    bus = ResultBus()
    assert bus.publish(payload).payload == payload
    assert bus.latest().payload == payload
    assert bus.records_after(0)[0].payload == payload

    calls: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(204)

    delivery = WebhookDelivery(
        "http://receiver.test/events", transport=httpx.MockTransport(capture),
    )
    delivery.start()
    try:
        assert delivery.submit(payload)
        assert delivery.flush(timeout_seconds=2)
    finally:
        delivery.close()
    assert json.loads(calls[0].content) == payload
    assert calls[0].headers["idempotency-key"] == "Google"


@pytest.mark.parametrize("name", NAMES)
def test_object_and_face_overlay_keep_domain_labels(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    labels: list[str] = []

    def capture(_image: object, text: str, *_args: object, **_kwargs: object) -> None:
        labels.append(text)

    monkeypatch.setattr(overlay_module, "_text_with_background", capture)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    bbox = NormalizedBBox(0.1, 0.1, 0.8, 0.8)
    overlay_module.draw_overlay(
        image, event_label="weapon", detected=True, confidence=0.9,
        description=f"{name} carries a torch.",
        objects=(SimpleNamespace(label="torch", bbox=bbox, confidence=0.8),),
    )
    overlay_module.draw_overlay(
        image, event_label="unknown_person", detected=False, confidence=0.9,
        description="Known person.", analysis_kind="face_recognition",
        objects=(SimpleNamespace(
            label="known_person", bbox=bbox, confidence=0.9,
            attributes={"status": "known", "name": name},
        ),),
    )
    assert "torch 0.80" in labels
    assert f"{name} carries a torch." in labels
    assert name in labels
