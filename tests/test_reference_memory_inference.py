from __future__ import annotations

import base64
import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import PUBLIC_MODEL_NAME
from goodvision_harness.grounding import (
    AIClubGroundingClient,
    DryRunGroundingClient,
    FACE_REFERENCE_ANALYSIS_KIND,
    REFERENCE_MEMORY_ANALYSIS_KIND,
    GroundingResponse,
    build_grounding_messages,
    parse_grounded_classification,
)
from goodvision_harness.inference import (
    DryRunWindowClient,
    QwenWindowClient,
    WindowResponse,
)
from goodvision_harness.organizer import ResultOrganizer
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.prompt import (
    BinaryClassification,
    build_window_messages,
    parse_binary_classification,
)
from goodvision_harness.reference_memory import MemoryContext, MemoryReference
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


ROOT = Path(__file__).resolve().parents[1]


def _memory_context(
    *,
    purpose: str = "face_recognition",
    newest_frame_only: bool = True,
) -> MemoryContext:
    return MemoryContext(
        purpose=purpose,
        references=(
            MemoryReference(
                memory_id="alice",
                image_data_url="data:image/jpeg;base64,cmVmZXJlbmNl",
                width=10,
                height=10,
            ),
        ),
        instruction="Match only a live face to a supplied reference.",
        newest_frame_only=newest_frame_only,
    )


def _window(index: int = 3) -> FrameWindow:
    frames = tuple(
        SampledFrame(
            sample_index=index,
            source_frame_index=index,
            timestamp_seconds=float(index),
            jpeg_bytes=f"frame-{index}".encode(),
            width=20,
            height=10,
        )
        for index in range(2)
    )
    return FrameWindow(index=index, frames=frames)


class _Transport:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, **request: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "id": "response-1",
            "choices": [{"message": {"content": self.text}}],
        }

    @staticmethod
    def assistant_text(response: dict[str, Any]) -> str:
        return str(response["choices"][0]["message"]["content"])


def test_binary_memory_prompt_prefix_and_newest_frame_only() -> None:
    memory = _memory_context()
    messages = build_window_messages(
        _window(),
        event_label="known face",
        system_prompt="system",
        prompt="Find {event} in {window_label}.",
        memory_context=memory,
    )

    content = messages[1]["content"]
    assert content[:4] == memory.prompt_prefix
    assert content[4]["type"] == "text"
    assert "BEGIN LIVE CAMERA INPUT" in content[4]["text"]
    assert "Target event: unknown_person" in content[4]["text"]
    assert "Find known face" not in content[4]["text"]
    assert '"matched_memory_ids"' in content[4]["text"]
    assert "same stranger-detection semantics as InsightFace" in content[4]["text"]
    assert len(content) == 6
    assert content[-1]["image_url"]["url"] == (
        "data:image/jpeg;base64," + base64.b64encode(b"frame-1").decode()
    )


def test_memory_context_can_keep_the_full_live_window() -> None:
    memory = _memory_context(newest_frame_only=False)
    messages = build_grounding_messages(
        _window(),
        event_label="known face",
        system_prompt="system",
        prompt="prompt",
        memory_context=memory,
    )

    content = messages[1]["content"]
    live_images = content[len(memory.prompt_prefix) + 1 :]
    assert len(live_images) == 2
    assert "matched_memory_ids" in content[len(memory.prompt_prefix)]["text"]
    assert "detected and matched_memory_ids are independent" in content[
        len(memory.prompt_prefix)
    ]["text"]
    assert "bbox_2d" in content[len(memory.prompt_prefix)]["text"]


@pytest.mark.parametrize(
    ("payload", "allowed", "error"),
    [
        (
            {
                "detected": True,
                "confidence": 0.9,
                "description": "match",
            },
            {"alice"},
            "omitted matched_memory_ids",
        ),
        (
            {
                "detected": True,
                "confidence": 0.9,
                "description": "match",
                "matched_memory_ids": ["mallory"],
            },
            {"alice"},
            "unknown memory IDs",
        ),
        (
            {
                "detected": True,
                "confidence": 0.9,
                "description": "match",
                "matched_memory_ids": ["alice", "alice"],
            },
            {"alice"},
            "must not contain duplicates",
        ),
        (
            {
                "detected": True,
                "confidence": 0.9,
                "description": "match",
                "matched_memory_ids": ["alice"],
            },
            None,
            "without a memory context",
        ),
    ],
)
def test_binary_memory_result_validation_is_strict(
    payload: dict[str, Any],
    allowed: set[str] | None,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        parse_binary_classification(
            json.dumps(payload),
            allowed_memory_ids=allowed,
        )


@pytest.mark.parametrize(
    ("detected", "matched_memory_ids"),
    [
        (False, []),
        (False, ["alice"]),
        (True, []),
        (True, ["alice"]),
    ],
)
def test_binary_memory_result_keeps_detection_independent_from_known_matches(
    detected: bool,
    matched_memory_ids: list[str],
) -> None:
    classification = parse_binary_classification(
        json.dumps(
            {
                "detected": detected,
                "confidence": 0.9,
                "description": "Face state classified.",
                "matched_memory_ids": matched_memory_ids,
            }
        ),
        allowed_memory_ids={"alice"},
    )

    assert classification.detected is detected
    assert classification.matched_memory_ids == tuple(matched_memory_ids)


def test_qwen_client_applies_the_memory_allowlist() -> None:
    config = load_config(ROOT / "config.yaml")
    transport = _Transport(
        '{"detected":true,"confidence":0.9,"description":"match",'
        '"matched_memory_ids":["unknown"]}'
    )
    client = QwenWindowClient(
        config.inference,
        transport,  # type: ignore[arg-type]
        _memory_context(),
    )

    response = client.classify_window(_window())

    assert response.classification is None
    assert response.error is not None
    assert response.error == "analysis request failed"
    assert config.inference.model == "qwen3.5-plus"
    assert transport.calls[0]["model"] == config.inference.model
    assert transport.calls[0]["model"] != PUBLIC_MODEL_NAME
    assert transport.calls[0]["messages"][1]["content"][0]["text"].startswith(
        "BEGIN MEMORY CONTEXT"
    )


@pytest.mark.parametrize("client_type", [DryRunWindowClient, DryRunGroundingClient])
@pytest.mark.parametrize(
    ("window_index", "detected", "matched_memory_ids"),
    [
        (0, False, ("alice",)),
        (3, True, ()),
    ],
)
def test_dry_run_memory_keeps_detection_independent_from_known_matches(
    client_type: type[DryRunWindowClient] | type[DryRunGroundingClient],
    window_index: int,
    detected: bool,
    matched_memory_ids: tuple[str, ...],
) -> None:
    config = load_config(ROOT / "config.yaml")
    client = client_type(config.inference, _memory_context())

    response = client.classify_window(_window(window_index))

    assert response.classification is not None
    assert response.classification.detected is detected
    assert response.classification.matched_memory_ids == matched_memory_ids


@pytest.mark.parametrize(
    ("purpose", "expected_kind"),
    [
        ("face_recognition", FACE_REFERENCE_ANALYSIS_KIND),
        ("product_reference", REFERENCE_MEMORY_ANALYSIS_KIND),
    ],
)
def test_grounding_memory_preserves_boxes_and_sets_distinct_analysis_kind(
    purpose: str,
    expected_kind: str,
) -> None:
    config = load_config(ROOT / "config.yaml")
    transport = _Transport(
        json.dumps(
            {
                "detected": True,
                "confidence": 0.93,
                "description": (
                    "画面中有一张未知人脸和 Alice。 / "
                    "An unknown face and Alice are visible."
                ),
                "matched_memory_ids": ["alice"],
                "objects": [
                    {
                        "label": "unknown_face",
                        "confidence": 0.91,
                        "bbox_2d": [100, 200, 900, 800],
                    }
                ],
            }
        )
    )
    client = AIClubGroundingClient(
        config.inference,
        transport,  # type: ignore[arg-type]
        _memory_context(purpose=purpose),
    )

    response = client.classify_window(_window())

    assert response.classification is not None
    assert response.classification.detected is True
    assert response.classification.analysis_kind == expected_kind
    assert response.classification.matched_memory_ids == ("alice",)
    assert response.classification.objects[0].bbox.as_dict() == {
        "coordinate_system": "source_normalized_xyxy",
        "x1": 0.1,
        "y1": 0.2,
        "x2": 0.9,
        "y2": 0.8,
    }


def test_grounding_memory_accepts_known_only_as_clear() -> None:
    classification = parse_grounded_classification(
        json.dumps(
            {
                "detected": False,
                "confidence": 0.94,
                "description": "Alice is the only visible face.",
                "matched_memory_ids": ["alice"],
                "objects": [],
            }
        ),
        allowed_memory_ids={"alice"},
    )

    assert classification.detected is False
    assert classification.matched_memory_ids == ("alice",)
    assert classification.objects == ()


def test_grounding_memory_still_requires_a_box_for_an_unknown_face() -> None:
    with pytest.raises(ValueError, match="at least one grounded object"):
        parse_grounded_classification(
            json.dumps(
                {
                    "detected": True,
                    "confidence": 0.9,
                    "description": "Unknown face.",
                    "matched_memory_ids": [],
                }
            ),
            allowed_memory_ids={"alice"},
        )


def test_organizer_surfaces_memory_matches_in_record_and_decision() -> None:
    window = _window()
    response = WindowResponse(
        window=window,
        classification=BinaryClassification(
            detected=False,
            confidence=0.9,
            description="Alice is visible.",
            matched_memory_ids=("alice",),
        ),
        raw_text=None,
        latency_seconds=0.1,
    )

    record, snapshot = ResultOrganizer("known face").apply(
        response,
        elapsed_seconds=0.2,
    )

    assert record["matched_memory_ids"] == ["alice"]
    assert snapshot["decision"]["matched_memory_ids"] == ["alice"]
    assert snapshot["decision"]["detected"] is False


def test_streaming_session_preserves_grounded_memory_metadata() -> None:
    config = load_config(ROOT / "config.yaml")
    transport = _Transport(
        json.dumps(
            {
                "detected": True,
                "confidence": 0.93,
                "description": (
                    "画面中有一张未知人脸和 Alice。 / "
                    "An unknown face and Alice are visible."
                ),
                "matched_memory_ids": ["alice"],
                "objects": [
                    {
                        "label": "unknown_face",
                        "bbox_2d": [100, 200, 900, 800],
                    }
                ],
            }
        )
    )
    client = AIClubGroundingClient(
        config.inference,
        transport,  # type: ignore[arg-type]
        _memory_context(),
    )
    window = _window()
    model_response = client.classify_window(window)
    assert isinstance(model_response, GroundingResponse)
    emitted = []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(on_analysis_result=emitted.append),
    )
    future: Future[Any] = Future()
    future.set_result(model_response)

    session._on_window_done(future, window)

    assert len(emitted) == 1
    assert emitted[0].analysis_kind == FACE_REFERENCE_ANALYSIS_KIND
    assert emitted[0].judgments[0].event_label == "unknown_person"
    assert emitted[0].judgments[0].matched_memory_ids == ("alice",)
    assert emitted[0].as_dict()["judgments"][0]["matched_memory_ids"] == [
        "alice"
    ]
