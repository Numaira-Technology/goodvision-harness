"""The small model response remains a complete, source-correlated verdict."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    COMPACT_NO_EVENTS_RULE,
    SAFETY_EVENT_LABELS,
    AIClubMultiGroundingClient,
    DryRunStrangerGroundingClient,
    SelectedFeaturesGroundingClient,
    parse_multi_grounded_classification,
)
from goodvision_harness.model_api import ModelAPIClient
from goodvision_harness.route_probe import ROUTES, probe_sample, probe_window

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("labels", [
    ("smoking",), ("fight", "fall"), SAFETY_EVENT_LABELS,
])
@pytest.mark.parametrize("config_path", [
    "config.yaml", "config.delivery.yaml", "src/goodvision_harness/config.yaml",
])
def test_compact_negative_expands_without_another_model_call(labels, config_path):
    settings = load_config(ROOT / config_path).inference
    transport = Mock(spec=ModelAPIClient)
    transport.assistant_text.return_value = '{"no_events":0.97}'
    client = AIClubMultiGroundingClient(settings, transport, labels)
    window = probe_window()

    response = client.classify_window(window)

    assert response.ok
    assert response.window is window
    assert response.completed_branches == ("safety",)
    result = response.classification
    assert not result.detected
    assert result.missing_event_labels == ()
    assert result.objects == result.actors == ()
    assert tuple(j.event_label for j in result.judgments) == labels
    for judgment in result.judgments:
        assert not judgment.detected
        assert judgment.confidence == 0.97
        assert " / " in judgment.description
        assert judgment.objects == ()
    transport.chat_completion.assert_called_once()
    sent = transport.chat_completion.call_args.kwargs
    assert COMPACT_NO_EVENTS_RULE in sent["messages"][0]["content"]
    assert sent["max_tokens"] == 1200  # Positive events retain their output budget.


@pytest.mark.parametrize("payload", [
    "{}", '{"events":[]}', '{"no_events":false}', '{"no_events":null}',
    '{"no_events":"0.9"}', '{"no_events":-0.1}', '{"no_events":1.1}',
    '{"no_events":NaN}', '{"no_events":Infinity}', '{"no_events":0.9',
    '{"no_events":0.9,"events":[]}', '{"no_events":0.9,"actors":[]}',
    '{"no_events":0.9,"detected":true}',
])
def test_incomplete_or_contradictory_response_is_not_a_clear_verdict(payload):
    with pytest.raises((TypeError, ValueError)):
        parse_multi_grounded_classification(payload, require_bilingual=True)


def test_compact_safety_result_keeps_face_branch_and_original_frame():
    settings = load_config(ROOT / "config.yaml").inference
    transport = Mock(spec=ModelAPIClient)
    transport.assistant_text.return_value = '{"no_events":0.93}'
    client = SelectedFeaturesGroundingClient(
        AIClubMultiGroundingClient(settings, transport),
        DryRunStrangerGroundingClient(settings),
    )
    window = probe_window()
    response = client.classify_window(window)
    assert response.ok
    assert response.window is window
    assert response.completed_branches == ("safety", "stranger")
    assert response.classification.missing_event_labels == ()
    assert [j.event_label for j in response.classification.judgments] == [
        *SAFETY_EVENT_LABELS, "stranger",
    ]
    transport.chat_completion.assert_called_once()


def test_route_probe_accepts_the_same_compact_response_as_live_inference():
    settings = load_config(ROOT / "config.yaml").inference
    transport = Mock(spec=ModelAPIClient)
    transport.assistant_text.return_value = json.dumps({"no_events": 0.96})
    assert probe_sample(transport, settings, ROUTES[0], 0)["ok"]
    sent = transport.chat_completion.call_args.kwargs
    assert COMPACT_NO_EVENTS_RULE in sent["messages"][0]["content"]
