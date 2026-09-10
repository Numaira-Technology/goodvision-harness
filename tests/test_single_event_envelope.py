"""A valid single-target smoking verdict must survive legacy JSON wrapping."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    SAFETY_EVENT_LABELS,
    AIClubMultiGroundingClient,
    DryRunStrangerGroundingClient,
    SelectedFeaturesGroundingClient,
    parse_multi_grounded_classification,
)
from goodvision_harness.model_api import ModelAPIClient
from goodvision_harness.route_probe import probe_window

ROOT = Path(__file__).resolve().parents[1]


def _positive():
    # The real failed response on 2026-09-10, without any image or identifiers.
    return {
        "detected": True,
        "confidence": 0.95,
        "description": (
            "男子嘴边有一根白色细长物体，疑似香烟 / "  # noqa: RUF001
            "A man has a white slender object near his mouth, resembling a cigarette"
        ),
        "objects": [{
            "label": "cigarette", "confidence": 0.98,
            "bbox_2d": [476, 666, 541, 738],
        }],
    }


def _client(payload, labels=("smoking",), *, config_path="config.delivery.yaml"):
    transport = Mock(spec=ModelAPIClient)
    transport.chat_completion.return_value = {
        "choices": [{
            "message": {"content": json.dumps(payload, ensure_ascii=False)},
            "finish_reason": "stop",
        }],
        "usage": {"completion_tokens": 105},
    }
    transport.assistant_text.side_effect = ModelAPIClient.assistant_text
    settings = load_config(ROOT / config_path).inference
    return AIClubMultiGroundingClient(settings, transport, labels), transport


@pytest.mark.parametrize("box", [[476, 666, 541, 738], [437, 658, 511, 762]])
def test_real_smoking_positive_is_identical_to_canonical_envelope(box):
    original = _positive()
    original["objects"][0]["bbox_2d"] = box
    unchanged = deepcopy(original)
    canonical = {"events": [{"event_label": "smoking", **original}]}
    actual = parse_multi_grounded_classification(
        json.dumps(original), event_labels=("smoking",), require_bilingual=True,
    )
    expected = parse_multi_grounded_classification(
        json.dumps(canonical), event_labels=("smoking",), require_bilingual=True,
    )
    assert actual == expected
    assert actual.detected and actual.confidence == 0.95
    assert actual.objects[0].bbox.x1 == box[0] / 1000
    assert original == unchanged


def test_legacy_envelope_preserves_optional_actor_ownership():
    payload = _positive()
    payload["objects"][0]["actor_ids"] = ["a1"]
    payload["actors"] = [{
        "actor_id": "a1", "confidence": 0.9, "bbox_2d": [100, 50, 900, 1000],
    }]
    parsed = parse_multi_grounded_classification(
        json.dumps(payload), event_labels=("smoking",), require_bilingual=True,
    )
    assert parsed.objects[0].actor_ids == ("a1",)
    assert parsed.actors[0].actor_id == "a1"


@pytest.mark.parametrize("change", [
    {"event_label": "fight"}, {"event_label": None}, {"detected": "true"},
    {"confidence": -1}, {"confidence": 1.1}, {"confidence": True},
    {"description": ""}, {"description": "English only"}, {"objects": []},
    {"objects": [{"label": "cigarette", "bbox_2d": [500, 600, 500, 700]}]},
    {"objects": [{"label": "cigarette", "bbox_2d": [0, 0, 1001, 700]}]},
    {"detected": False}, {"events": None}, {"events": []}, {"no_events": 0.95},
])
def test_envelope_compatibility_never_repairs_invalid_verdicts_or_boxes(change):
    with pytest.raises((TypeError, ValueError)):
        parse_multi_grounded_classification(
            json.dumps({**_positive(), **change}),
            event_labels=("smoking",), require_bilingual=True,
        )


@pytest.mark.parametrize("labels", [("smoking", "fall"), SAFETY_EVENT_LABELS])
def test_single_envelope_never_invents_other_requested_verdicts(labels):
    client, transport = _client(_positive(), labels)
    result = client.classify_window(probe_window())
    assert not result.ok and result.classification is None
    assert result.branch_errors == (("safety", "invalid_response"),)
    assert result.classification_diagnostics == {
        "category": "invalid_envelope", "recovered": False,
        "response_chars": len(json.dumps(_positive(), ensure_ascii=False)),
        "finish_reason": "stop", "completion_tokens": 105,
    }
    transport.chat_completion.assert_called_once()


@pytest.mark.parametrize("include_face", [False, True])
def test_recovered_positive_reaches_selected_feature_client_once(include_face):
    safety, transport = _client(_positive())
    client = SelectedFeaturesGroundingClient(
        safety,
        DryRunStrangerGroundingClient(safety.settings) if include_face else None,
        safety_event_labels=("smoking",), include_stranger=include_face,
    )
    window = probe_window()
    response = client.classify_window(window)
    assert response.ok and response.window is window
    smoking = response.classification.judgments[0]
    assert smoking.event_label == "smoking" and smoking.detected
    assert smoking.confidence == 0.95 and smoking.objects[0].label == "cigarette"
    assert response.classification_diagnostics["category"] == "single_event_envelope"
    assert response.classification_diagnostics["recovered"] is True
    assert response.raw_text is None
    assert response.branch_errors == ()
    transport.chat_completion.assert_called_once()


def test_failed_envelope_diagnostic_survives_selected_feature_wrapper():
    safety, _ = _client({"events": "private-model-output"})
    client = SelectedFeaturesGroundingClient(
        safety, None, safety_event_labels=("smoking",), include_stranger=False,
    )
    response = client.classify_window(probe_window())
    assert response.classification is None
    diagnostic = response.classification_diagnostics
    assert diagnostic["category"] == "invalid_envelope"
    assert "private-model-output" not in json.dumps(diagnostic)


@pytest.mark.parametrize("config_path", [
    "config.delivery.yaml", "config.yaml", "config.vlm-rich.yaml",
    "src/goodvision_harness/config.yaml",
])
@pytest.mark.parametrize("labels", [("smoking",), ("smoking", "fall")])
def test_selected_prompt_has_one_output_contract_and_keeps_detection_policy(
    config_path, labels,
):
    client, transport = _client({"no_events": 0.95}, labels, config_path=config_path)
    assert client.classify_window(probe_window()).ok
    messages = transport.chat_completion.call_args.kwargs["messages"]
    user_text = messages[1]["content"][0]["text"]
    assert "Respond with only this JSON object:" not in user_text
    assert "extend that same JSON object" not in user_text
    assert "events array" in user_text
    assert "even for ONE requested event" in messages[0]["content"]
    assert "Mandatory smoking evidence rule" in messages[0]["content"]
    assert "Smoking:" in user_text
    assert "Falling:" in user_text
