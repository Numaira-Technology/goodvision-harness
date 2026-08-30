from __future__ import annotations

import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import numpy as np
import pytest

import goodvision_harness.openrouter as transport_module
import goodvision_harness.overlay as overlay_module
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
    PUBLIC_MODEL_NAME,
)
from goodvision_harness.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    OpenRouterSettings,
)
from goodvision_harness.privacy import public_safe_value, redact_public_text
from goodvision_harness.vision_result import VisionResult


PRIVATE_MARKERS = (
    "openrouter",
    "qwen",
    "coreweave",
    "deepinfra",
    "insightface",
    "buffalo_l",
    "onnxruntime",
    "cpuexecutionprovider",
    "dmlexecutionprovider",
)
PRIVATE_TEXT = (
    "OpenRouter https://openrouter.ai qwen/qwen3.5-35b-a3b CoreWeave DeepInfra "
    "InsightFace buffalo_l onnxruntime CPUExecutionProvider DmlExecutionProvider"
)


def _assert_public(value: object) -> None:
    rendered = json.dumps(value, ensure_ascii=False, default=str).lower()
    assert all(marker not in rendered for marker in PRIVATE_MARKERS)


def test_recursive_public_projection_removes_implementation_markers() -> None:
    projected = public_safe_value(
        {"description": PRIVATE_TEXT, "nested": [PRIVATE_TEXT]}
    )

    _assert_public(projected)
    assert "analysis engine" in str(projected)


def test_serialized_result_records_never_emit_private_model_details() -> None:
    result = AnalysisResult(
        message_id="message-1",
        source_id="camera-1",
        stream_session_id="session-1",
        sequence=1,
        emitted_at="2026-08-28T00:00:00.000Z",
        frame=FrameReference(
            source_frame_index=1,
            timestamp_seconds=1.0,
            observed_at="2026-08-28T00:00:00.000Z",
            width=320,
            height=180,
        ),
        judgments=(
            EventJudgment(
                event_label="fight",
                detected=True,
                confidence=0.9,
                description=PRIVATE_TEXT,
                object_ids=("object-1",),
            ),
        ),
        objects=(
            GroundedObject(
                object_id="object-1",
                label=f"person {PRIVATE_TEXT}",
                bbox=NormalizedBBox(0.1, 0.1, 0.8, 0.8),
                attributes={"engine_note": PRIVATE_TEXT},
            ),
        ),
        model_name="qwen/qwen3.5-35b-a3b",
        model_latency_seconds=0.2,
    )
    vision = VisionResult(
        event_label="fight",
        detected=True,
        confidence=0.9,
        description=PRIVATE_TEXT,
        window_index=1,
        window_start_seconds=0.0,
        window_end_seconds=1.0,
    )

    payload = result.as_dict()
    assert payload["model"]["name"] == PUBLIC_MODEL_NAME
    _assert_public(payload)
    _assert_public(vision.as_dict())


def test_overlay_redacts_description_label_and_enrolled_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def capture(_image: object, text: str, *_args: object, **_kwargs: object) -> None:
        captured.append(text)

    monkeypatch.setattr(overlay_module, "_text_with_background", capture)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    bbox = NormalizedBBox(0.1, 0.1, 0.8, 0.8)

    overlay_module.draw_overlay(
        image,
        event_label="fight",
        detected=True,
        confidence=0.9,
        description=PRIVATE_TEXT,
        objects=(SimpleNamespace(label=PRIVATE_TEXT, bbox=bbox, confidence=0.8),),
    )
    overlay_module.draw_overlay(
        image,
        event_label="unknown_person",
        detected=False,
        confidence=0.9,
        description=PRIVATE_TEXT,
        objects=(
            SimpleNamespace(
                label="known_person",
                bbox=bbox,
                confidence=0.8,
                attributes={"status": "known", "name": PRIVATE_TEXT},
            ),
        ),
        analysis_kind="face_recognition",
    )

    _assert_public(captured)


def test_provider_error_body_and_network_exception_are_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = HTTPError(
        "https://openrouter.ai/api/v1/chat/completions",
        500,
        "failed",
        {},
        io.BytesIO(PRIVATE_TEXT.encode("utf-8")),
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(transport_module, "urlopen", fail)
    client = OpenRouterClient(
        OpenRouterSettings(max_retries=0),
        api_key="not-a-real-key",
    )

    with pytest.raises(OpenRouterError) as raised:
        client.chat_completion(model="private-model", messages=[{"role": "user"}])

    assert str(raised.value) == "model API returned HTTP 500"
    _assert_public(str(raised.value))
    _assert_public(redact_public_text(PRIVATE_TEXT))
