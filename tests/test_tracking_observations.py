from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness.config import InferenceSettings
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.grounding import (
    ACTOR_ASSOCIATION_RULE,
    FACE_REFERENCE_ANALYSIS_KIND,
    POSITIVE_EVENT_ACTOR_RULE,
    AIClubMultiGroundingClient,
    GroundedClassification,
    GroundingResponse,
    ModelObject,
    SelectedFeaturesGroundingClient,
    build_grounding_messages,
    build_multi_grounding_messages,
    observation_bundle,
    parse_grounded_classification,
    parse_multi_grounded_classification,
)
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def _payload(*, detected: bool = True) -> dict:
    return {
        "detected": detected,
        "confidence": 0.9,
        "description": "发现烟具。 / A cigarette is visible.",
        "objects": [
            {
                "label": "cigarette",
                "bbox_2d": [200, 200, 220, 230],
                "actor_ids": ["a1"],
            }
        ]
        if detected
        else [],
        "actors": [
            {
                "actor_id": "a1",
                "bbox_2d": [100, 100, 500, 900],
                "confidence": 0.9,
            }
        ],
    }


def _window() -> FrameWindow:
    jpeg = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))[1].tobytes()
    return FrameWindow(0, (SampledFrame(0, 0, 0.0, jpeg, 320, 240),))


@pytest.mark.parametrize("detected", [True, False])
def test_actors_are_private_and_available_on_negative_frames(detected: bool) -> None:
    classification = parse_grounded_classification(
        json.dumps(_payload(detected=detected))
    )
    bundle = observation_bundle(classification)
    assert classification.detected is detected
    assert len(bundle.actors) == 1
    assert bundle.actors[0].bbox == NormalizedBBox(0.1, 0.1, 0.5, 0.9)
    assert len(classification.objects) == int(detected)
    if detected:
        assert classification.objects[0].label == "cigarette"
        assert classification.objects[0].attributes == {}
        assert bundle.object_actor_ids == (("a1",),)


@pytest.mark.parametrize(
    "actors",
    [
        None,
        "wrong",
        [None],
        [{"actor_id": "a1", "bbox_2d": [0, 0, 0, 5]}],
        [{"actor_id": "a1", "bbox_2d": [0, 0, 10, 10], "confidence": "bad"}],
    ],
)
def test_bad_actor_metadata_does_not_discard_valid_event(actors: object) -> None:
    payload = _payload()
    payload["actors"] = actors
    classification = parse_grounded_classification(json.dumps(payload))
    assert classification.detected
    assert classification.actors == ()
    assert classification.objects[0].label == "cigarette"


def test_duplicate_actor_ids_are_unusable_and_actor_count_is_bounded() -> None:
    payload = _payload()
    payload["actors"] *= 2
    assert parse_grounded_classification(json.dumps(payload)).actors == ()
    payload["actors"] = [
        {"actor_id": str(index), "bbox_2d": [0, 0, 10, 10]} for index in range(20)
    ]
    assert len(parse_grounded_classification(json.dumps(payload)).actors) == 8


def test_old_model_response_without_actors_is_accepted() -> None:
    payload = _payload()
    del payload["actors"]
    del payload["objects"][0]["actor_ids"]
    classification = parse_grounded_classification(json.dumps(payload))
    assert classification.detected
    assert observation_bundle(classification).actors == ()
    assert observation_bundle(classification).object_actor_ids == ((),)


def test_memory_identity_is_actor_scoped_and_only_allowed_for_face_references() -> None:
    payload = _payload(detected=False)
    payload["matched_memory_ids"] = ["alice"]
    payload["actors"][0]["matched_memory_id"] = "alice"
    generic = parse_grounded_classification(
        json.dumps(payload), allowed_memory_ids={"alice"}
    )
    face = parse_grounded_classification(
        json.dumps(payload),
        allowed_memory_ids={"alice"},
        analysis_kind=FACE_REFERENCE_ANALYSIS_KIND,
    )
    assert generic.actors[0].matched_memory_id is None
    assert face.actors[0].matched_memory_id == "alice"
    payload["actors"][0]["matched_memory_id"] = "not-supplied"
    invalid = parse_grounded_classification(
        json.dumps(payload),
        allowed_memory_ids={"alice"},
        analysis_kind=FACE_REFERENCE_ANALYSIS_KIND,
    )
    assert invalid.actors[0].matched_memory_id is None
    assert invalid.matched_memory_ids == ("alice",)


def test_actor_contract_is_appended_for_custom_and_fixed_multi_prompts() -> None:
    custom = build_grounding_messages(
        _window(), event_label="custom", system_prompt="system", prompt="custom body"
    )
    fixed = build_multi_grounding_messages(
        _window(),
        system_prompt="system",
        prompt="old fixed prompt",
        append_contract=False,
    )
    for messages, rule in (
        (custom, ACTOR_ASSOCIATION_RULE), (fixed, POSITIVE_EVENT_ACTOR_RULE),
    ):
        content = messages[1]["content"]
        assert rule in content[0]["text"]
        assert len([item for item in content if item["type"] == "image_url"]) == 1
    assert "at most 8" in ACTOR_ASSOCIATION_RULE
    assert "every event is negative" in ACTOR_ASSOCIATION_RULE


def test_actors_do_not_change_existing_request_budget_or_add_model_calls() -> None:
    labels = ("fight", "smoking", "fall", "weapon")
    payload = {
        "events": [
            {**_payload(detected=False), "event_label": label} for label in labels
        ],
        "actors": _payload()["actors"],
    }
    calls = []
    transport = SimpleNamespace(
        chat_completion=lambda **kwargs: calls.append(kwargs) or {},
        assistant_text=lambda response: json.dumps(payload),
    )
    settings = InferenceSettings(
        "model", "smoking", "system", "prompt", {"max_tokens": 1024}, "fixed prompt"
    )
    response = AIClubMultiGroundingClient(settings, transport).classify_window(
        _window()
    )
    assert response.ok
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 1024
    assert len(response.classification.actors) == 1


def test_composite_keeps_known_faces_separate_from_event_evidence() -> (
    None
):
    face = ModelObject(
        "known_person",
        NormalizedBBox(0.1, 0.1, 0.2, 0.2),
        0.9,
        {"status": "known", "person_id": "alice"},
    )
    face_classification = GroundedClassification(
        False, 0.9, "Known", (face,), event_object_indexes=()
    )
    safety = parse_multi_grounded_classification(
        json.dumps(
            {
                "events": [{**_payload(), "event_label": "smoking"}],
                "actors": _payload()["actors"],
            }
        ),
        event_labels=("smoking",),
    )

    def client(classification: object) -> SimpleNamespace:
        return SimpleNamespace(
            settings=InferenceSettings("model", "smoking", "system", "prompt"),
            classify_window=lambda window: GroundingResponse(
                window, classification, 0.1
            ),
        )

    result = SelectedFeaturesGroundingClient(
        client(safety),
        client(face_classification),
        safety_event_labels=("smoking",),
        include_stranger=True,
    ).classify_window(_window())
    assert result.ok
    bundle = observation_bundle(result.classification)
    assert bundle.face_observations == (face,)
    assert len(bundle.actors) == 1
    assert bundle.object_actor_ids == (("a1",), ())
    assert [item.label for item in result.classification.objects] == [
        "cigarette", "known_person",
    ]
    stranger = next(
        item for item in result.classification.judgments
        if item.event_label == "stranger"
    )
    assert not stranger.detected
    assert stranger.objects == ()
    assert result.classification.supporting_objects == (face,)
