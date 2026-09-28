"""Prompt delivery regressions, not model-accuracy or live-camera tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    BILINGUAL_DESCRIPTION_RULE,
    SAFETY_EVENT_LABELS,
    STREAMING_DECISION_PREFIX_RULE,
    STREAMING_MULTI_DECISION_PREFIX_RULE,
    AIClubGroundingClient,
    AIClubMultiGroundingClient,
    build_grounding_messages,
    parse_multi_grounded_classification,
    parse_streaming_decision_prefix,
    parse_streaming_multi_decision_prefix,
)
from goodvision_harness.model_api import ModelAPIClient
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_path", [
        # Package/default byte equality is checked in test_deployment_settings.py.
        ROOT / "config.yaml",
        ROOT / "config.delivery.yaml",
    ]
)
@pytest.mark.parametrize(
    ("legacy_single", "event_labels"),
    [
        (True, ("smoking",)),
        (True, ("fall",)),
        (False, ("smoking",)),
        (False, ("fall",)),
        (False, ("smoking", "fall")),
        (False, SAFETY_EVENT_LABELS),
    ],
)
def test_evidence_rules_reach_model_in_each_detection_path(
    config_path: Path, legacy_single: bool, event_labels: tuple[str, ...]
) -> None:
    config = load_config(config_path)
    settings = config.inference
    assert config.sliding_window.size_frames == 1
    assert config.sliding_window.stride_frames == 1
    assert config.video.fps == 2.5
    assert config.video.max_image_side == 320
    assert config.video.jpeg_quality == 75
    negative = {
        "detected": False,
        "confidence": 0.9,
        "description": "未发现目标证据。 / No target evidence visible.",
        "objects": [],
    }
    payload = negative if legacy_single else {
        "events": [{"event_label": label, **negative} for label in event_labels]
    }
    transport = Mock(spec=ModelAPIClient)
    transport.assistant_text.return_value = json.dumps(payload)
    if legacy_single:
        client = AIClubGroundingClient(
            replace(settings, event_label=event_labels[0]), transport
        )
    else:
        client = AIClubMultiGroundingClient(settings, transport, event_labels)
    window = FrameWindow(
        index=0,
        frames=(SampledFrame(
            sample_index=0,
            source_frame_index=0,
            timestamp_seconds=0.0,
            jpeg_bytes=b"test-frame",
            width=320,
            height=180,
        ),),
    )

    response = client.classify_window(window)

    assert response.error is None
    assert response.classification is not None
    assert response.classification.detected is False
    assert response.classification.objects == ()
    transport.chat_completion.assert_called_once()
    request = transport.chat_completion.call_args.kwargs
    assert request["model"] == settings.model
    for key, value in settings.request_parameters.items():
        assert request[key] == value
    system_text = request["messages"][0]["content"]
    assert "For smoking and fall" in system_text
    assert "take precedence over this high-recall preference" in system_text
    if "smoking" in event_labels:
        assert "Mandatory smoking evidence rule" in system_text
        assert "One or two fingers held to the mouth" in system_text
        assert "Never invent a concealed cigarette" in system_text
    content = request["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 1
    prompt = content[0]["text"]
    expected = (
        settings.multi_event_prompt
        if not legacy_single and event_labels == SAFETY_EVENT_LABELS
        else settings.prompt
    )
    if not legacy_single and event_labels != SAFETY_EVENT_LABELS:
        expected = expected.split("Respond with only this JSON object:")[0].rstrip()
        assert "extend that same JSON object" not in prompt
    assert prompt.startswith(expected)
    normalized = " ".join(prompt.split())
    for required_rule in (
        "use detected=false, not a low-confidence positive",
        "requires an identifiable cigarette",
        "The item may be lit or unlit",
        "A hand near or covering the mouth",
        "coughing, yawning, wiping the face",
        "eating, or drinking is insufficient without an identifiable smoking item",
        "A generic small white object alone is not enough",
        "Do not invent a hidden item, smoke, or a bounding box",
        "requires visible evidence of loss of balance or support",
        "stable, supported squat or crouch",
        "kneeling, bending to pick something up",
        "normal sitting or lying is not a fall",
        "A person can fall from any posture, including a squat",
        "With only one frame, do not invent downward motion",
        "The full impact is not required",
        "at least partially visible in the NEWEST frame",
    ):
        assert required_rule in normalized


def test_single_and_multi_prompts_use_identical_event_evidence_rules() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    for single_label, multi_label in (("Smoking", "smoking"), ("Falling", "fall")):
        def rule(prompt: str, label: str) -> str:
            section = prompt.split(f"- {label}: ", 1)[1]
            return " ".join(section.split("\n- ", 1)[0].split("\n\n", 1)[0].split())

        assert rule(settings.prompt, single_label) == rule(
            settings.multi_event_prompt, multi_label
        )


@pytest.mark.parametrize(
    ("legacy_single", "event_labels"),
    [
        (True, ("smoking",)),
        (False, ("smoking",)),
        (False, ("smoking", "fall")),
        (False, SAFETY_EVENT_LABELS),
    ],
)
def test_bare_finger_evidence_rule_overrides_stale_permissive_prompts(
    legacy_single: bool, event_labels: tuple[str, ...]
) -> None:
    settings = replace(
        load_config(ROOT / "config.yaml").inference,
        system_prompt="Use a low detection threshold and prefer false positives.",
        prompt="A hand near the mouth plausibly indicates smoking; choose true.",
        multi_event_prompt="Prefer a low-confidence positive for ambiguous cues.",
    )
    negative = {
        "detected": False,
        "confidence": 0.95,
        "description": (
            "两根空手指接触嘴唇，未见吸烟物品。 / "  # noqa: RUF001
            "Two bare fingers touch the lips; no smoking item is visible."
        ),
        "objects": [],
    }
    transport = Mock(spec=ModelAPIClient)
    transport.assistant_text.return_value = json.dumps(
        negative if legacy_single else {
            "events": [{"event_label": label, **negative} for label in event_labels]
        }
    )
    client = (
        AIClubGroundingClient(settings, transport)
        if legacy_single
        else AIClubMultiGroundingClient(settings, transport, event_labels)
    )
    window = FrameWindow(index=0, frames=(SampledFrame(
        sample_index=0, source_frame_index=0, timestamp_seconds=0.0,
        jpeg_bytes=b"test-frame", width=320, height=180,
    ),))

    response = client.classify_window(window)

    assert response.error is None
    assert response.classification is not None
    assert response.classification.detected is False
    assert response.classification.objects == ()
    messages = transport.chat_completion.call_args.kwargs["messages"]
    system_text = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert system_text.count("Mandatory smoking evidence rule") == 1
    assert system_text.count(BILINGUAL_DESCRIPTION_RULE) == 1
    for required in (
        "overrides any high-recall preference",
        "detected=true requires BOTH an identifiable cigarette",
        "visibly separate from the person's fingers",
        "AND a visible action using, holding, or preparing that item",
        "One or two fingers held to the mouth",
        "skin, fingernails, or miming a puff are not smoking items",
        "A pen, straw, food, or a generic small white/cylindrical object",
        "cannot be distinguished from a finger or pen, use detected=false",
        "not a low-confidence positive",
        "Never invent a concealed cigarette or put a cigarette box on fingers",
        "its distinguishing visual feature",
    ):
        assert required in system_text


def test_smoking_evidence_rule_does_not_change_other_event_requests() -> None:
    window = FrameWindow(index=0, frames=(SampledFrame(
        sample_index=0, source_frame_index=0, timestamp_seconds=0.0,
        jpeg_bytes=b"test-frame", width=320, height=180,
    ),))
    messages = build_grounding_messages(
        window, event_label="fall", system_prompt="Existing fall policy",
        prompt="Evaluate fall",
    )

    assert messages[0]["content"] == (
        f"Existing fall policy\n\n{BILINGUAL_DESCRIPTION_RULE}"
    )
    assert "Mandatory smoking evidence rule" not in messages[0]["content"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"detected":true,"confidence":0.85,', (True, 0.85)),
        (' { "detected" : false, "confidence" : 1 }', (False, 1.0)),
        ('{"confidence":0.9,"detected":true,', None),
        ('{"detected":true,"confidence":', None),
        ('{"detected":true,"confidence":1.1,', None),
        ('prefix {"detected":true,"confidence":0.9,', None),
    ],
)
def test_streaming_decision_prefix_requires_complete_ordered_values(
    text: str,
    expected: tuple[bool, float] | None,
) -> None:
    assert parse_streaming_decision_prefix(text) == expected


def test_streaming_grounder_emits_early_verdict_and_keeps_final_validation() -> None:
    settings = replace(load_config(ROOT / "config.yaml").inference, event_label="fall")
    full = json.dumps({
        "detected": True,
        "confidence": 0.91,
        "description": "一人正在跌倒。 / A person is falling.",
        "objects": [{
            "label": "person",
            "confidence": 0.92,
            "bbox_2d": [100, 100, 700, 950],
        }],
    }, ensure_ascii=False)

    class StreamTransport:
        assistant_text = staticmethod(ModelAPIClient.assistant_text)

        def chat_completion_stream(self, **kwargs):
            assert kwargs["on_content"](
                '{"detected":true,"confidence":0.91,', 0.2,
            )
            assert STREAMING_DECISION_PREFIX_RULE in (
                kwargs["messages"][0]["content"]
            )
            return {"choices": [{"message": {"content": full}}]}

    preliminary = []
    client = AIClubGroundingClient(
        settings,
        StreamTransport(),
        stream_decisions=True,
        preliminary_callback=preliminary.append,
    )
    window = FrameWindow(index=7, frames=(SampledFrame(
        sample_index=7,
        source_frame_index=21,
        timestamp_seconds=2.8,
        jpeg_bytes=b"test-frame",
        width=320,
        height=180,
    ),))

    response = client.classify_window(window)

    assert response.error is None
    assert response.classification is not None
    assert response.classification.detected is True
    assert len(preliminary) == 1
    assert preliminary[0].detected is True
    assert preliminary[0].confidence == 0.91
    assert preliminary[0].window_index == 7
    assert preliminary[0].as_dict()["type"] == "preliminary_vision_result"


def test_streaming_grounder_does_not_confirm_failed_preliminary_delivery() -> None:
    settings = replace(load_config(ROOT / "config.yaml").inference, event_label="fall")
    callback_results = []

    class StreamTransport:
        assistant_text = staticmethod(ModelAPIClient.assistant_text)

        def chat_completion_stream(self, **kwargs):
            callback_results.append(kwargs["on_content"](
                '{"detected":true,"confidence":0.91,', 0.2,
            ))
            return {"choices": [{"message": {"content": json.dumps({
                "detected": False,
                "confidence": 0.8,
                "description": "未检测到跌倒。 / No fall was detected.",
                "objects": [],
            }, ensure_ascii=False)}}]}

    def fail_delivery(_result) -> None:
        raise OSError("output is unavailable")

    client = AIClubGroundingClient(
        settings,
        StreamTransport(),
        stream_decisions=True,
        preliminary_callback=fail_delivery,
    )
    window = FrameWindow(index=8, frames=(SampledFrame(
        sample_index=8,
        source_frame_index=24,
        timestamp_seconds=3.2,
        jpeg_bytes=b"test-frame",
        width=320,
        height=180,
    ),))

    response = client.classify_window(window)

    assert callback_results == [False]
    assert response.error is None
    assert response.classification is not None
    assert response.classification.detected is False


def test_streaming_multi_grounder_emits_each_decision_before_final_details() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    labels = ("smoking", "fall")
    decisions = [
        {"event_label": "smoking", "detected": False, "confidence": 0.93},
        {"event_label": "fall", "detected": True, "confidence": 0.88},
    ]
    full = json.dumps({
        "decisions": decisions,
        "events": [
            {
                "event_label": "smoking", "detected": False, "confidence": 0.93,
                "description": "未检测到吸烟。 / No smoking detected.", "objects": [],
            },
            {
                "event_label": "fall", "detected": True, "confidence": 0.88,
                "description": "一人正在跌倒。 / A person is falling.",
                "objects": [{
                    "label": "person", "confidence": 0.9,
                    "bbox_2d": [100, 100, 700, 950],
                }],
            },
        ],
    }, ensure_ascii=False)
    prefix = json.dumps({"decisions": decisions}, ensure_ascii=False)[:-1] + ","

    class StreamTransport:
        assistant_text = staticmethod(ModelAPIClient.assistant_text)

        def chat_completion_stream(self, **kwargs):
            assert kwargs["on_content"](prefix, 0.2)
            assert STREAMING_MULTI_DECISION_PREFIX_RULE in (
                kwargs["messages"][0]["content"]
            )
            return {"choices": [{"message": {"content": full}}]}

    preliminary = []
    client = AIClubMultiGroundingClient(
        settings, StreamTransport(), labels,
        stream_decisions=True, preliminary_callback=preliminary.append,
    )
    window = FrameWindow(index=7, frames=(SampledFrame(
        sample_index=7, source_frame_index=21, timestamp_seconds=2.8,
        jpeg_bytes=b"test-frame", width=320, height=180,
    ),))

    response = client.classify_window(window)

    assert response.ok
    assert [(item.event_label, item.detected) for item in preliminary] == [
        ("smoking", False), ("fall", True),
    ]
    assert {item.source_frame_index for item in preliminary} == {21}
    assert parse_streaming_multi_decision_prefix(prefix, event_labels=labels) == (
        ("smoking", False, 0.93), ("fall", True, 0.88),
    )


def test_streaming_multi_parser_expands_only_omitted_negative_details() -> None:
    labels = ("smoking", "fall", "weapon")
    decisions = [
        {"event_label": "smoking", "detected": False, "confidence": 0.93},
        {"event_label": "fall", "detected": True, "confidence": 0.88},
        {"event_label": "weapon", "detected": False, "confidence": 0.91},
    ]
    payload = {
        "decisions": decisions,
        "events": [{
            "event_label": "fall",
            "detected": True,
            "confidence": 0.88,
            "description": "一人正在跌倒。 / A person is falling.",
            "objects": [{
                "label": "person",
                "confidence": 0.9,
                "bbox_2d": [100, 100, 700, 950],
            }],
        }],
    }

    result = parse_multi_grounded_classification(
        json.dumps(payload, ensure_ascii=False),
        event_labels=labels,
        require_bilingual=True,
    )

    assert [(item.event_label, item.detected) for item in result.judgments] == [
        ("smoking", False), ("fall", True), ("weapon", False),
    ]
    assert result.judgments[0].confidence == 0.93
    assert result.judgments[2].objects == ()


def test_streaming_multi_parser_never_invents_missing_positive_evidence() -> None:
    payload = {
        "decisions": [
            {"event_label": "smoking", "detected": False, "confidence": 0.93},
            {"event_label": "fall", "detected": True, "confidence": 0.88},
        ],
        "events": [],
    }

    with pytest.raises(ValueError, match="missing event labels: fall"):
        parse_multi_grounded_classification(
            json.dumps(payload), event_labels=("smoking", "fall"),
        )
