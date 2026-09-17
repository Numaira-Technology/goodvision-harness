import base64
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from goodvision_harness.config import InferenceSettings, _model_image_size, load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.diagnostics import capture_call, failure_category
from goodvision_harness.grounding import (
    AIClubGroundingClient,
    AIClubMultiGroundingClient,
    GroundedClassification,
    GroundedEventClassification,
    GroundingResponse,
    ModelActor,
    ModelObject,
    MultiGroundedClassification,
)
from goodvision_harness.letterbox import classify_letterboxed, prepare_window
from goodvision_harness.model_api import ModelAPIClient
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def window(width=640, height=480):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return FrameWindow(
        0,
        (
            SampledFrame(
                0,
                7,
                1.2,
                encoded.tobytes(),
                width,
                height,
                source_width=width,
                source_height=height,
            ),
        ),
    )


@pytest.mark.parametrize(
    "dimensions", [(640, 480), (360, 202), (1920, 1080), (100, 400), (80, 40)]
)
def test_model_canvas_preserves_full_content_and_source_metadata(dimensions):
    original = window(*dimensions)
    prepared, geometry = prepare_window(original, (360, 202), 75)
    frame = prepared.frames[0]
    image = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (202, 360, 3)
    assert frame.original_dimensions == dimensions
    assert frame.timestamp_seconds == 1.2 and frame.source_frame_index == 7
    g = geometry
    assert (
        image[
            g.top : g.top + g.content_height, g.left : g.left + g.content_width
        ].mean()
        > 250
    )
    if g.left > 3:
        assert image[:, : g.left - 2].mean() < 3
    if g.top > 3:
        assert image[: g.top - 2].mean() < 3
    assert (
        abs(g.content_width / g.content_height - dimensions[0] / dimensions[1]) < 0.02
    )
    if dimensions == (360, 202):
        assert frame.jpeg_bytes == original.frames[0].jpeg_bytes


@pytest.mark.parametrize("multi", [False, True])
def test_model_boxes_and_actors_return_to_original_coordinates(multi):
    original = window()
    observed = []

    def classify(prepared):
        # The complete 640x480 image fills x=45..314 of the 360x202 canvas.
        item = ModelObject("person", NormalizedBBox(45 / 360, 0, 314 / 360, 1))
        actor = ModelActor("a1", item.bbox)
        if multi:
            result = MultiGroundedClassification(
                (
                    GroundedEventClassification(
                        "fight",
                        True,
                        0.9,
                        "test",
                        (item,),
                    ),
                ),
                actors=(actor,),
                supporting_objects=(item,),
            )
        else:
            result = GroundedClassification(
                True,
                0.9,
                "test",
                (item,),
                actors=(actor,),
                event_object_indexes=(0,),
            )
        return GroundingResponse(prepared, result, 0.1)

    response = classify_letterboxed(classify, original, (360, 202), 75, observed.append)
    assert response.ok and response.window is original
    assert observed[0].frames[0].width == 360
    for item in (*response.classification.objects, *response.classification.actors):
        assert item.bbox == NormalizedBBox(0, 0, 1, 1)


def test_padding_only_detection_is_an_error_not_a_negative_or_fake_edge_box():
    def classify(prepared):
        item = ModelObject("weapon", NormalizedBBox(0, 0, 0.05, 0.5))
        return GroundingResponse(
            prepared, GroundedClassification(True, 0.9, "test", (item,)), 0.1
        )

    with capture_call() as capture:
        response = classify_letterboxed(classify, window(), (360, 202), 75)
    assert not response.ok and response.error
    assert response.branch_errors == (("safety", "invalid_response"),)
    # The model answered, so this is an unusable reply, not an input problem.
    assert [failure_category(f) for f in capture.failures] == ["parse"]



_INSIDE = NormalizedBBox(45 / 360, 0, 314 / 360, 1)
_PADDING = NormalizedBBox(0, 0, 0.05, 0.5)


def test_a_padding_only_event_is_missing_while_other_events_survive():
    def classify(prepared):
        return GroundingResponse(prepared, MultiGroundedClassification((
            GroundedEventClassification(
                "fight", True, 0.9, "test", (ModelObject("person", _INSIDE),),
            ),
            GroundedEventClassification(
                "weapon", True, 0.8, "test", (ModelObject("gun", _PADDING),),
            ),
            GroundedEventClassification(
                "smoking", True, 0.7, "test",
                (ModelObject("cigarette", _PADDING), ModelObject("cigarette", _INSIDE)),
            ),
            GroundedEventClassification("fall", False, 0.1, "test"),
        ), actors=(ModelActor("a1", _PADDING), ModelActor("a2", _INSIDE))), 0.1)

    with capture_call() as capture:
        response = classify_letterboxed(classify, window(), (360, 202), 75)
    assert response.ok
    result = response.classification
    assert [j.event_label for j in result.judgments] == ["fight", "smoking", "fall"]
    assert result.missing_event_labels == ("weapon",)
    smoking = result.judgments[1]
    assert [o.bbox for o in smoking.objects] == [NormalizedBBox(0, 0, 1, 1)]
    assert [a.actor_id for a in result.actors] == ["a2"]
    assert capture.failures == []


def test_event_object_indexes_follow_the_objects_that_remain():
    def classify(prepared):
        return GroundingResponse(prepared, GroundedClassification(
            True, 0.9, "test",
            (ModelObject("gun", _PADDING), ModelObject("person", _INSIDE)),
            event_object_indexes=(1,),
        ), 0.1)

    response = classify_letterboxed(classify, window(), (360, 202), 75)
    assert response.ok
    assert response.classification.event_object_indexes == (0,)
    assert response.classification.objects[0].label == "person"


def test_a_frame_already_at_the_canvas_is_sent_without_decoding(monkeypatch):
    original = window(360, 202)
    decode = Mock(side_effect=AssertionError("decoded again"))
    monkeypatch.setattr("goodvision_harness.letterbox.cv2.imdecode", decode)
    prepared, geometry = prepare_window(original, (360, 202), 75)
    assert prepared.frames == original.frames
    assert (geometry.left, geometry.top) == (0, 0)
    decode.assert_not_called()

@pytest.mark.parametrize("multi", [False, True])
def test_actual_api_payload_contains_fixed_jpeg_without_network(multi):
    settings = InferenceSettings(
        "test", "fight", "system", "prompt", image_size=(360, 202)
    )
    transport = Mock(spec=ModelAPIClient)
    transport.assistant_text.return_value = (
        '{"no_events":0.95}'
        if multi
        else '{"detected":false,"confidence":0.95,'
        '"description":"无事件。 / No event.","objects":[]}'
    )
    client = (AIClubMultiGroundingClient if multi else AIClubGroundingClient)(
        settings, transport
    )
    original = window()
    response = client.classify_window(original)
    assert response.ok and response.window is original
    messages = transport.chat_completion.call_args.kwargs["messages"]
    images = [
        item["image_url"]["url"]
        for msg in messages
        if isinstance(msg["content"], list)
        for item in msg["content"]
        if item.get("type") == "image_url"
    ]
    assert images
    for url in images:
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert image.shape == (202, 360, 3)


@pytest.mark.parametrize(
    "value", [[360], [360, 0], [True, 202], [360.0, 202], "360x202"]
)
def test_invalid_size_rejected(value):
    with pytest.raises(ValueError):
        _model_image_size(value)


def test_size_is_opt_in_and_loaded_from_video_config():
    assert _model_image_size(None) is None
    assert InferenceSettings("m", "e", "s", "p").image_size is None
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert config.inference.image_size == (360, 202)
    assert config.inference.image_jpeg_quality == config.video.jpeg_quality == 75


def test_corrupt_frame_does_not_call_model():
    original = window()
    broken = replace(original, frames=(replace(original.frames[0], jpeg_bytes=b"bad"),))
    classify = Mock()
    with capture_call() as capture:
        response = classify_letterboxed(classify, broken, (360, 202), 75)
    assert not response.ok
    classify.assert_not_called()
    # Nothing was sent, so diagnostics must not blame the model's reply.
    assert [failure_category(f) for f in capture.failures] == ["input"]
