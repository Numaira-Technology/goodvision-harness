from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import httpx
import jsonschema
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.requests import Request

import goodvision_harness.config as config_module
from goodvision_harness import service as service_module
from goodvision_harness.aiclub import AIClubSettings, api_key_present
from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    PUBLIC_MODEL_NAME,
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delivery import WebhookDelivery
from goodvision_harness.grounding import (
    ACTOR_ASSOCIATION_RULE,
    BILINGUAL_DESCRIPTION_RULE,
    DETECTION_EVENT_LABELS,
    SAFETY_EVENT_LABELS,
    SMOKING_EVIDENCE_RULE,
    DryRunGroundingClient,
    build_grounding_messages,
    build_multi_grounding_messages,
    parse_grounded_classification,
)
from goodvision_harness.inference import DryRunWindowClient, WindowResponse
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.privacy import contains_private_implementation_detail
from goodvision_harness.prompt import BinaryClassification
from goodvision_harness.result_bus import FrameBus, FramePacket, ResultBus
from goodvision_harness.service import (
    HarnessRuntime,
    _mjpeg_part,
    _render_analysis_jpeg,
    create_app,
)
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from goodvision_harness.video_to_stream import LiveFrame
from goodvision_harness.vision_result import VisionResult

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SYSTEM_PROMPT = (
    "You are a risk-sensitive multi-label safety-event perception module for live "
    "AI glasses. You receive "
    "short, chronologically ordered windows of frames sampled from a live "
    "camera. Use visible objects, human interaction, posture, motion, and "
    "temporal context to detect plausible events, including precursors and "
    "partial events. Use a deliberately low detection threshold and prefer "
    "false positives over false negatives. Do not require certainty or a "
    "textbook-complete event. Evaluate every requested event independently. "
    "For smoking and fall, the event-specific minimum evidence requirements "
    "and negative examples take precedence over this high-recall preference. "
    "A low-confidence guess cannot replace the required evidence. "
    "Answer with a single JSON object - no markdown, no extra text."
)
EXPECTED_PROMPT = (
    """Decide whether the target event is happening, starting, imminent, or
plausibly indicated in the NEWEST (last) frame. Use earlier frames as
temporal context.

This is a high-recall safety detector with a deliberately low threshold.
Set detected=true whenever visible evidence makes the target event a
reasonable possibility, even when the evidence is partial or ambiguous.
A related object combined with a person's interaction, posture, movement,
or context can be sufficient. When choosing between a plausible positive
and a negative, choose detected=true; lower confidence may express the
uncertainty. Use detected=false only when there is no meaningful visible
cue for the event or the evidence clearly indicates an unrelated activity.

For smoking and fall, apply the minimum evidence requirements below
before the high-recall preference. If those requirements are not met,
use detected=false, not a low-confidence positive.

Event-specific sensitivity examples:
- Smoking: detected=true requires an identifiable cigarette, cigar,
  pipe, or e-cigarette at least partially visible in the NEWEST frame and
  held, handled, carried, or used by a person. The item may be lit or unlit;
  a flame, glowing tip, inhalation, exhalation, or visible smoke is not
  required. An unattended item alone is insufficient.
  A hand near or covering the mouth, coughing, yawning, wiping the face,
  eating, or drinking is insufficient without an identifiable smoking
  item. A generic small white object alone is not enough to identify it
  as a cigarette. Do not invent a hidden item, smoke, or a bounding box.
  For a positive, name the visible smoking item in the description;
  do not use a hand-to-mouth gesture alone as evidence.
- Fighting: detected=true for striking or kicking, pushing, grabbing,
  wrestling, aggressive lunging or chasing, raised fists, an attack
  posture, or an apparently imminent attack. Do not require visible impact.
- Falling: detected=true requires visible evidence of loss of balance
  or support, uncontrolled descent, or collapse. A stable, supported
  squat or crouch, kneeling, bending to pick something up, or normal
  sitting or lying is not a fall. Low body height, a forward lean, or
  being on the ground alone is insufficient. A person can fall from any
  posture, including a squat, when loss of support or collapse is visible.
  With only one frame, do not invent downward motion or a before/after
  transition. The full impact is not required. For a positive, describe
  the visible loss of balance, loss of support, or collapse evidence.

Respond with only this JSON object:
"""
    '{"detected": true or false, "confidence": 0.0-1.0, '
    '"description": "one short sentence of visible evidence"}'
)
BBOX_PROMPT_APPEND = (
    'For the delivery contract, extend that same JSON object with an "objects"\n'
    """array. Each relevant visible entity must be
{"label": "...", "confidence": 0.0-1.0, "bbox_2d": [x1, y1, x2, y2]}.
bbox_2d must tightly enclose the object in the NEWEST frame and use the
top-left-origin relative 0..1000 xyxy coordinate grid.

If detected=true, include at least one valid target object. For fight and
fall, return only the involved person's tight body box or boxes. For
smoking, return only the cigarette, cigar, pipe, or e-cigarette box; do not
return the smoker's person box. For weapon, return only the weapon box; do
not return the holder's person box. If detected=false, use "objects": []."""
)


def _analysis_result(*, detected: bool = True) -> AnalysisResult:
    objects = (
        (
            GroundedObject(
                object_id="object-1",
                label="person",
                confidence=0.91,
                bbox=NormalizedBBox(0.1, 0.2, 0.9, 0.8),
            ),
        )
        if detected
        else ()
    )
    return AnalysisResult(
        message_id="message-1",
        source_id="external-camera",
        stream_session_id="session-1",
        sequence=1,
        emitted_at="2026-08-25T10:00:01.000Z",
        frame=FrameReference(
            source_frame_index=42,
            timestamp_seconds=1.25,
            observed_at="2026-08-25T10:00:00.000Z",
            width=1920,
            height=1080,
        ),
        judgments=(
            EventJudgment(
                event_label="smoking",
                detected=detected,
                confidence=0.94,
                description="Visible event." if detected else "Event is not visible.",
                object_ids=("object-1",) if detected else (),
            ),
        ),
        objects=objects,
        model_name="test-model",
        model_latency_seconds=0.25,
    )


def _jpeg(width: int = 200, height: int = 100) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_harness_defaults_and_high_recall_prompt() -> None:
    config = load_config(ROOT / "config.yaml")

    assert config.default_video is None
    assert config.stream.frame_crop == "auto"
    assert (
        config.video.fps,
        config.video.max_image_side,
        config.video.jpeg_quality,
    ) == (2.5, 320, 75)
    assert (
        config.display.fps,
        config.display.max_image_side,
        config.display.jpeg_quality,
        config.display.delay_seconds,
        config.display.bbox_max_gap_seconds,
    ) == (15.0, 720, 70, 2.5, 0.8)
    assert (
        config.sliding_window.size_frames,
        config.sliding_window.stride_frames,
        config.sliding_window.include_partial_final_window,
    ) == (1, 1, False)
    assert config.runtime.max_concurrent_requests == 6
    assert config.model_api.max_retries == 0
    assert config.model_api.api_key_env == "AICLUB_API_KEY"
    assert config.model_api.provider == "aiclub"
    assert config.aiclub is config.model_api
    # Branding is applied only at public serialization boundaries; the
    # provider requests use the selected route's actual model identifier.
    assert config.inference.model == "qwen3.5-plus"
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 7900
    assert config.face.allow_model_download is True
    assert config.face.min_face_size == 20
    assert config.face.enrollment_min_face_size == 80
    assert config.inference.system_prompt == EXPECTED_SYSTEM_PROMPT
    assert config.inference.prompt == (
        f"{EXPECTED_PROMPT}\n\n{BBOX_PROMPT_APPEND}"
    )
    assert config.inference.multi_event_prompt.startswith(
        "Evaluate exactly these four safety events in the NEWEST (last) frame:"
    )
    assert '"event_label":"fight"' in config.inference.multi_event_prompt
    assert '"event_label":"smoking"' in config.inference.multi_event_prompt
    assert '"event_label":"fall"' in config.inference.multi_event_prompt
    assert '"event_label":"weapon"' in config.inference.multi_event_prompt
    assert "Every detected=false event must use \"objects\": []." in (
        config.inference.multi_event_prompt
    )
    assert "return only the smoking item box, not the smoker" in (
        config.inference.multi_event_prompt
    )
    assert "return only the weapon box, not the holder" in (
        config.inference.multi_event_prompt
    )
    assert config.inference.request_parameters == {
        "temperature": 0.0,
        "max_tokens": 1200,
    }
    packaged = ROOT / "src" / "goodvision_harness" / "config.yaml"
    assert packaged.read_text(encoding="utf-8") == (ROOT / "config.yaml").read_text(
        encoding="utf-8"
    )


def test_face_provider_can_be_selected_by_deployment_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_PROVIDER", "CUDAExecutionProvider")

    config = load_config(ROOT / "config.yaml")

    assert config.face.provider == "CUDAExecutionProvider"


def test_directml_face_provider_can_be_selected_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_PROVIDER", "DmlExecutionProvider")

    config = load_config(ROOT / "config.yaml")

    assert config.face.provider == "DmlExecutionProvider"


@pytest.mark.parametrize("scheduler", ["auto", "full", "single_flight"])
def test_face_live_scheduler_can_be_selected_by_environment(
    scheduler: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_LIVE_SCHEDULER", scheduler)

    config = load_config(ROOT / "config.yaml")

    assert config.face.live_scheduler == scheduler


def test_face_live_scheduler_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_LIVE_SCHEDULER", "sometimes")

    with pytest.raises(ValueError, match=r"face\.live_scheduler must be"):
        load_config(ROOT / "config.yaml")


def test_stream_crop_is_applied_before_display_sampling_and_source_mapping() -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(config, stream=replace(config.stream, frame_crop="left_half"))
    image = np.zeros((100, 400, 3), dtype=np.uint8)
    image[:, :200] = (12, 34, 220)
    image[:, 200:] = (210, 32, 14)

    class OneFrameStream:
        realtime = True

        def frames(self):
            yield LiveFrame(index=7, timestamp_seconds=0.0, bgr=image)

        def stop(self) -> None:
            pass

    displayed: list[bytes] = []
    session = StreamingSession(
        config,
        DryRunGroundingClient(config.inference),
        SessionCallbacks(
            on_display_frame=lambda jpeg, _timestamp, _index: displayed.append(jpeg)
        ),
        mode="stream",
        video_stream=OneFrameStream(),  # type: ignore[arg-type]
    )

    sampled = next(session._sampled_frames())
    sampled_image = cv2.imdecode(
        np.frombuffer(sampled.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    display_image = cv2.imdecode(
        np.frombuffer(displayed[0], dtype=np.uint8), cv2.IMREAD_COLOR
    )

    assert sampled.original_dimensions == (200, 100)
    assert sampled_image.shape[:2] == (100, 200)
    assert display_image.shape[:2] == (100, 200)
    assert float(sampled_image[:, :, 2].mean()) > 180
    assert float(sampled_image[:, :, 0].mean()) < 60

    result = replace(
        _analysis_result(),
        frame=FrameReference(
            source_frame_index=7,
            timestamp_seconds=0.0,
            observed_at="2026-08-28T00:00:00.000Z",
            width=sampled.original_dimensions[0],
            height=sampled.original_dimensions[1],
        ),
    )
    rendered = _render_analysis_jpeg(displayed[0], result, quality=90)
    rendered_image = cv2.imdecode(
        np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert rendered_image.shape[:2] == (100, 200)
    # The normalized x1=0.1 edge must land at x=20 on the cropped 200px view,
    # not at x=40 as it would on the original side-by-side frame.
    box_edge = rendered_image[60:75, 18:23].mean(axis=(0, 1))
    assert box_edge[2] > box_edge[1] + 40


def test_annotated_output_rate_covers_the_fastest_analysis_source() -> None:
    config = load_config(ROOT / "config.yaml")
    fast_config = replace(
        config,
        video=replace(config.video, fps=24.0),
        face=replace(config.face, sample_fps=30.0),
    )
    runtime = HarnessRuntime(fast_config)
    try:
        assert runtime.ui_info()["annotated_output"]["output_fps"] == 30.0
        assert runtime.status()["annotated_output"]["output_fps"] == 30.0
    finally:
        runtime.stop()


def test_model_prompt_has_one_bbox_append_and_dynamic_suffix() -> None:
    config = load_config(ROOT / "config.yaml")
    frame = SampledFrame(
        sample_index=9,
        source_frame_index=42,
        timestamp_seconds=1.25,
        jpeg_bytes=b"jpeg",
        width=320,
        height=180,
    )
    window = FrameWindow(index=7, frames=(frame,))

    messages = build_grounding_messages(
        window,
        event_label=config.inference.event_label,
        system_prompt=config.inference.system_prompt,
        prompt=config.inference.prompt,
    )
    assert messages[0] == {
        "role": "system",
        "content": (
            f"{EXPECTED_SYSTEM_PROMPT}\n\n{SMOKING_EVIDENCE_RULE}"
            f"\n\n{BILINGUAL_DESCRIPTION_RULE}"
        ),
    }
    prompt_text = messages[1]["content"][0]["text"]
    assert prompt_text == (
        f"{EXPECTED_PROMPT}\n\n{BBOX_PROMPT_APPEND}\n\n"
        "Target event: smoking\n"
        "Window: W7 · f9-f9\n"
        "Sample timestamps: f9=1.250s\n"
        "The JPEG images follow in chronological order."
        f"\n\n{ACTOR_ASSOCIATION_RULE}"
    )
    assert prompt_text.count("For the delivery contract") == 1
    assert prompt_text.count('"bbox_2d"') == 1

    multi_messages = build_multi_grounding_messages(
        window,
        event_labels=SAFETY_EVENT_LABELS,
        system_prompt=config.inference.system_prompt,
        prompt=config.inference.multi_event_prompt,
        append_contract=False,
    )
    assert multi_messages[0]["content"].endswith(
        "independently judge every fixed safety event and allow multiple "
        "simultaneous positives."
    )
    multi_prompt_text = multi_messages[1]["content"][0]["text"]
    assert multi_prompt_text.startswith(config.inference.multi_event_prompt)
    assert (
        "Target safety events (exact order): fight, smoking, fall, weapon"
        in multi_prompt_text
    )
    assert multi_prompt_text.count('"events":[') == 1
    assert "Authoritative fixed safety-event output contract" not in (
        multi_prompt_text
    )


def test_grounded_dry_run_preserves_original_pattern_latency_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "config.yaml")
    sleeps: list[float] = []
    monkeypatch.setattr("goodvision_harness.grounding.time.sleep", sleeps.append)
    client = DryRunGroundingClient(config.inference)
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=0,
        timestamp_seconds=0.0,
        jpeg_bytes=b"jpeg",
        width=320,
        height=180,
    )

    responses = [
        client.classify_window(FrameWindow(index=index, frames=(frame,)))
        for index in range(6)
    ]
    classifications = [response.classification for response in responses]
    assert all(item is not None for item in classifications)
    assert [item.detected for item in classifications if item is not None] == [
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert sleeps == pytest.approx(
        [0.35 + (index * 13 % 40) / 100.0 for index in range(6)]
    )
    positive = classifications[2]
    assert positive is not None
    assert positive.confidence == 0.896
    assert positive.description == (
        "演示帧W2 · f0-f0中检测到吸烟事件。 / "
        "Dry run: smoking visibly active in W2 · f0-f0."
    )
    assert len(positive.objects) == 1
    assert positive.objects[0].bbox == NormalizedBBox(0.18, 0.12, 0.72, 0.9)
    negative = classifications[1]
    assert negative is not None
    assert negative.confidence == 0.748
    assert negative.description == (
        "演示帧W1 · f0-f0中未检测到吸烟事件。 / "
        "Dry run: smoking not visible in W1 · f0-f0."
    )
    assert negative.objects == ()
    assert all(response.raw_text is None for response in responses)


def test_vision_result_contract_is_unchanged() -> None:
    result = VisionResult(
        event_label="smoking",
        detected=True,
        confidence=0.9126,
        description="Visible evidence.",
        window_index=4,
        window_start_seconds=1.23456,
        window_end_seconds=1.67891,
    )
    assert result.as_dict() == {
        "type": "vision_result",
        "schema_version": "1",
        "event_label": "smoking",
        "detected": True,
        "confidence": 0.913,
        "description": "Visible evidence.",
        "window_index": 4,
        "window_start_seconds": 1.235,
        "window_end_seconds": 1.679,
    }


def test_legacy_window_client_keeps_vision_result_without_fabricating_bbox() -> None:
    config = load_config(ROOT / "config.yaml")
    client = DryRunWindowClient(config.inference)
    legacy: list[VisionResult] = []
    grounded: list[AnalysisResult] = []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_vision_result=legacy.append,
            on_analysis_result=grounded.append,
        ),
    )
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=8,
        timestamp_seconds=1.5,
        jpeg_bytes=_jpeg(),
        width=200,
        height=100,
    )
    window = FrameWindow(index=4, frames=(frame,))
    future: Future[Any] = Future()
    future.set_result(
        WindowResponse(
            window=window,
            classification=BinaryClassification(
                detected=True,
                confidence=0.88,
                description="Visible evidence.",
            ),
            raw_text=None,
            latency_seconds=0.25,
        )
    )

    session._on_window_done(future, window)

    assert [result.as_dict() for result in legacy] == [
        {
            "type": "vision_result",
            "schema_version": "1",
            "event_label": "smoking",
            "detected": True,
            "confidence": 0.88,
            "description": "Visible evidence.",
            "window_index": 4,
            "window_start_seconds": 1.5,
            "window_end_seconds": 1.5,
        }
    ]
    assert grounded == []

def test_qwen_bbox_is_normalized_and_validated() -> None:
    classification = parse_grounded_classification(
        json.dumps(
            {
                "detected": True,
                "confidence": 0.93,
                "description": "A visible event.",
                "objects": [
                    {
                        "label": "person",
                        "confidence": 0.9,
                        "bbox_2d": [100, 200, 900, 800],
                    }
                ],
            }
        )
    )
    assert classification.objects[0].bbox == NormalizedBBox(0.1, 0.2, 0.9, 0.8)

    negative = parse_grounded_classification(
        '{"detected":false,"confidence":0.8,"description":"clear"}'
    )
    assert negative.objects == ()

    with pytest.raises(ValueError, match="at least one grounded object"):
        parse_grounded_classification(
            '{"detected":true,"confidence":0.8,"description":"positive"}'
        )
    with pytest.raises(ValueError, match=r"0\.\.1000"):
        parse_grounded_classification(
            '{"detected":true,"confidence":0.8,"description":"x",'
            '"objects":[{"label":"person","bbox_2d":[0,0,1001,500]}]}'
        )
    with pytest.raises(ValueError, match="x2>x1"):
        NormalizedBBox(0.8, 0.1, 0.2, 0.9)


@pytest.mark.parametrize(
    ("detected", "description"),
    [(True, "smoking: Visible event."), (False, "无异常 / No anomalies detected")],
)
def test_analysis_result_matches_versioned_json_schema(
    detected: bool, description: str,
) -> None:
    payload = _analysis_result(detected=detected).as_dict()
    assert payload["model"]["name"] == PUBLIC_MODEL_NAME
    assert payload["description"] == description
    schema = json.loads(
        (ROOT / "schemas" / "analysis-result-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    invalid = json.loads(json.dumps(payload))
    invalid["judgments"][0]["detected"] = True
    invalid["judgments"][0]["object_ids"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    if detected:
        assert payload["objects"][0]["bbox"]["coordinate_system"] == (
            "source_normalized_xyxy"
        )
    assert payload["frame"]["width"] == 1920


@pytest.mark.parametrize("description", ["", None, 42])
def test_analysis_schema_requires_nonempty_frame_description(description: Any) -> None:
    schema = json.loads(
        (ROOT / "schemas" / "analysis-result-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    payload = _analysis_result().as_dict()
    del payload["description"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)
    payload["description"] = description
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


@pytest.mark.parametrize("complete,missing,valid", (
    (True, [], True), (False, ["stranger"], True),
    (True, ["stranger"], False), (False, [], False),
))
def test_v2_schema_enforces_completeness_consistency(
    complete: bool, missing: list[str], valid: bool,
) -> None:
    payload = _analysis_result(detected=False).as_dict()
    payload.update(analysis_complete=complete, missing_event_labels=missing)
    schema = json.loads(
        (ROOT / "schemas" / "analysis-result-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert jsonschema.Draft202012Validator(schema).is_valid(payload) is valid


@pytest.mark.parametrize("detected", (False, True))
def test_video_overlay_does_not_report_partial_results_as_clear(
    monkeypatch: pytest.MonkeyPatch, detected: bool,
) -> None:
    import goodvision_harness.overlay as overlay_module

    captions = []
    monkeypatch.setattr(
        overlay_module, "_text_with_background",
        lambda image, text, *args, **kwargs: captions.append(text),
    )
    partial = replace(
        _analysis_result(detected=detected), missing_event_labels=("stranger",)
    )
    service_module._draw_analysis_overlay(
        np.zeros((100, 200, 3), dtype=np.uint8), partial
    )
    assert not any(text.startswith("CLEAR") for text in captions)
    if detected:
        assert any(text.startswith("SMOKING") for text in captions)
        assert not any("incomplete" in text.lower() for text in captions)
    else:
        assert captions == []


def test_historical_v1_records_keep_the_original_contract() -> None:
    historical = _analysis_result().as_dict()
    historical["schema_version"] = "1"
    for key in ("description", "analysis_complete", "missing_event_labels"):
        historical.pop(key)
    schema = json.loads(
        (ROOT / "schemas" / "analysis-result-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(historical, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_analysis_result().as_dict(), schema)


def test_frame_description_summarizes_only_detected_events() -> None:
    result = _analysis_result()
    result = replace(
        result,
        judgments=(
            replace(
                result.judgments[0],
                description="画面中有人吸烟。 / A person is smoking in the frame.",
            ),
            EventJudgment("fall", False, 0.98, "No fall is visible."),
            EventJudgment(
                "stranger", True, 0.9,
                "画面中有陌生人。 / An unknown person is visible.", ("object-1",),
            ),
        ),
    )

    payload = result.as_dict()

    assert payload["description"] == (
        "smoking: 画面中有人吸烟。 / A person is smoking in the frame.; "
        "stranger: 画面中有陌生人。 / An unknown person is visible."
    )
    assert payload["judgments"][1]["description"] == "No fall is visible."


def test_frame_description_is_clear_with_only_known_people() -> None:
    result = _analysis_result()
    result = replace(
        result,
        analysis_kind="face_recognition",
        judgments=(
            EventJudgment(
                "unknown_person", False, 0.99, "Only an authorized person is visible.",
                matched_memory_ids=("alice",),
            ),
        ),
        objects=(replace(result.objects[0], label="known_person"),),
    )

    payload = result.as_dict()

    assert payload["objects"]
    assert payload["description"] == "无异常 / No anomalies detected"
    assert payload["judgments"][0]["description"] == (
        "Only an authorized person is visible."
    )


def test_pipeline_emits_grounded_result_with_source_frame_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "config.yaml")
    monkeypatch.setattr("goodvision_harness.grounding.time.sleep", lambda _: None)
    client = DryRunGroundingClient(config.inference)
    emitted: list[AnalysisResult] = []
    legacy: list[VisionResult] = []
    emitted_frames: list[tuple[AnalysisResult, bytes]] = []
    callback_order: list[str] = []

    def on_analysis_frame(result: AnalysisResult, jpeg: bytes) -> None:
        callback_order.append("analysis_frame")
        emitted_frames.append((result, jpeg))

    def on_analysis_result(result: AnalysisResult) -> None:
        callback_order.append("analysis_result")
        emitted.append(result)

    def on_vision_result(result: VisionResult) -> None:
        callback_order.append("vision_result")
        legacy.append(result)

    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_event=lambda _: callback_order.append("event"),
            on_state=lambda _: callback_order.append("state"),
            on_analysis_frame=on_analysis_frame,
            on_analysis_result=on_analysis_result,
            on_vision_result=on_vision_result,
        ),
        source_id="camera-a",
    )
    presentation_jpeg = _jpeg(width=320, height=180)
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=77,
        timestamp_seconds=2.5,
        jpeg_bytes=_jpeg(),
        width=200,
        height=100,
        source_width=1920,
        source_height=1080,
        observed_at="2026-08-25T10:00:00.000Z",
        presentation_jpeg_bytes=presentation_jpeg,
    )
    window = FrameWindow(index=2, frames=(frame,))

    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))
    session._on_window_done(future, window)

    assert callback_order == [
        "event",
        "state",
        "vision_result",
        "analysis_frame",
        "analysis_result",
    ]
    assert len(emitted) == 1
    assert len(legacy) == 1
    assert legacy[0].detected is True
    assert legacy[0].confidence == emitted[0].judgments[0].confidence
    payload = emitted[0].as_dict()
    assert payload["source_id"] == "camera-a"
    assert payload["frame"]["source_frame_index"] == 77
    assert payload["frame"]["width"] == 1920
    assert payload["judgments"][0]["object_ids"] == ["object-1"]
    assert emitted_frames == [(emitted[0], presentation_jpeg)]


def test_pipeline_keeps_late_success_for_private_overlay_timeline() -> None:
    config = load_config(ROOT / "config.yaml")
    client = DryRunGroundingClient(config.inference)
    public_results: list[AnalysisResult] = []
    timeline_results: list[AnalysisResult] = []
    records: list[dict[str, Any]] = []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_event=records.append,
            on_analysis_result=public_results.append,
            on_timeline_result=timeline_results.append,
        ),
    )
    classification = parse_grounded_classification(
        '{"detected":true,"confidence":0.9,"description":"visible",'
        '"objects":[{"label":"person","bbox_2d":[100,100,500,900]}]}'
    )

    def completed_window(
        index: int,
        timestamp: float,
    ) -> tuple[Future[Any], FrameWindow]:
        frame = SampledFrame(
            sample_index=index,
            source_frame_index=index * 10,
            timestamp_seconds=timestamp,
            jpeg_bytes=_jpeg(),
            width=200,
            height=100,
        )
        window = FrameWindow(index=index, frames=(frame,))
        future: Future[Any] = Future()
        future.set_result(
            WindowResponse(
                window=window,
                classification=classification,
                raw_text=None,
                latency_seconds=2.0,
            )
        )
        return future, window

    newer_future, newer_window = completed_window(2, 0.8)
    older_future, older_window = completed_window(1, 0.4)
    session._on_window_done(newer_future, newer_window)
    session._on_window_done(older_future, older_window)

    assert [item.frame.source_frame_index for item in public_results] == [20]
    assert [item.frame.source_frame_index for item in timeline_results] == [20, 10]
    assert [item["applied"] for item in records] == [True, False]


def test_pending_future_remains_until_delivery_callbacks_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "config.yaml")
    monkeypatch.setattr("goodvision_harness.grounding.time.sleep", lambda _: None)
    client = DryRunGroundingClient(config.inference)
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def block_result(_result: AnalysisResult) -> None:
        callback_entered.set()
        assert release_callback.wait(2.0)

    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(on_analysis_result=block_result),
    )
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=1,
        timestamp_seconds=0.5,
        jpeg_bytes=_jpeg(),
        width=200,
        height=100,
    )
    window = FrameWindow(index=2, frames=(frame,))
    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))
    with session._pending_lock:
        session._pending.add(future)

    worker = threading.Thread(target=session._on_window_done, args=(future, window))
    worker.start()
    try:
        assert callback_entered.wait(1.0)
        assert session._has_pending()
    finally:
        release_callback.set()
        worker.join(2.0)
    assert not worker.is_alive()
    assert not session._has_pending()


def test_overlay_burns_normalized_box_without_mutating_contract() -> None:
    result = _analysis_result()
    before = result.as_dict()
    rendered = _render_analysis_jpeg(_jpeg(), result, quality=85)
    decoded = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # The 0.1 x-coordinate maps to x=20 on a 200-pixel frame. Sample the
    # vertical box edge well below the text overlay and compare its red channel
    # with an untouched interior region.
    edge = decoded[60:75, 18:23].mean(axis=(0, 1))
    interior = decoded[60:75, 90:110].mean(axis=(0, 1))
    assert edge[2] > edge[1] + 40
    assert edge.mean() > interior.mean() + 20
    assert result.as_dict() == before


def test_result_and_frame_buses_are_bounded_and_resumable() -> None:
    results = ResultBus(history_size=2)
    results.publish({"message_id": "one"})
    results.publish({"message_id": "two"})
    results.publish({"message_id": "three"})
    replay = results.records_after(0)
    assert [item.payload["message_id"] for item in replay] == ["two", "three"]
    assert results.records_after(replay[-1].cursor, timeout_seconds=0.01) == ()

    frames = FrameBus()
    packet = FramePacket(
        jpeg=_jpeg(),
        media_timestamp=1.5,
        frame_sequence=9,
        observed_at="2026-08-25T10:00:00.000Z",
        analysis_message_id="message-9",
        next_analysis_message_id="message-10",
        annotation_mode="interpolated",
        stream_delay_seconds=2.4,
    )
    version = frames.publish(packet)
    assert frames.wait_after(0) == (version, packet)
    part = _mjpeg_part(packet)
    assert b"Content-Type: image/jpeg" in part
    assert b"X-Analysis-Message-Id: message-9" in part
    assert b"X-Next-Analysis-Message-Id: message-10" in part
    assert b"X-Annotation-Mode: interpolated" in part
    assert b"X-Stream-Delay: 2.400" in part
    assert packet.jpeg in part


def test_result_and_frame_bus_reset_clear_old_session_and_remain_usable() -> None:
    results = ResultBus(history_size=2)
    first = results.publish({"message_id": "old-1"})
    second = results.publish({"message_id": "old-2"})
    old_epoch = results.epoch

    results.reset()

    assert results.epoch != old_epoch
    assert results.latest() is None
    assert results.records_after(0) == ()
    replacement = results.publish({"message_id": "new-1"})
    assert replacement.cursor == second.cursor + 1
    assert [item.payload for item in results.records_after(second.cursor)] == [
        {"message_id": "new-1"}
    ]
    assert first.cursor < second.cursor < replacement.cursor

    frames = FrameBus()
    old_packet = FramePacket(
        jpeg=_jpeg(),
        media_timestamp=1.0,
        frame_sequence=1,
        observed_at="2026-08-25T10:00:00.000Z",
        analysis_message_id="old-message",
    )
    old_version = frames.publish(old_packet)
    reset_version = frames.reset()

    assert reset_version > old_version
    assert frames.latest() is None
    assert frames.wait_after(old_version) is None

    new_packet = FramePacket(
        jpeg=_jpeg(),
        media_timestamp=2.0,
        frame_sequence=2,
        observed_at="2026-08-25T10:00:01.000Z",
        analysis_message_id="new-message",
    )
    new_version = frames.publish(new_packet)
    assert new_version > reset_version
    assert frames.wait_after(reset_version) == (new_version, new_packet)


class _StubRuntime:
    def __init__(self) -> None:
        self.results = ResultBus()
        self.frames = FrameBus()
        self.preview_frames = FrameBus()
        self.annotated_frames = FrameBus()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self.results.publish(_analysis_result().as_dict())
        self.frames.publish(
            FramePacket(
                jpeg=_jpeg(),
                media_timestamp=1.0,
                frame_sequence=3,
                observed_at="2026-08-25T10:00:00.000Z",
                analysis_message_id="message-1",
            )
        )
        self.preview_frames.publish(
            FramePacket(
                jpeg=_jpeg(),
                media_timestamp=1.1,
                frame_sequence=4,
                observed_at="2026-08-25T10:00:00.100Z",
            )
        )
        self.annotated_frames.publish(
            FramePacket(
                jpeg=_jpeg(),
                media_timestamp=0.6,
                frame_sequence=2,
                observed_at="2026-08-25T10:00:00.200Z",
                analysis_message_id="message-1",
                next_analysis_message_id="message-2",
                annotation_mode="interpolated",
                stream_delay_seconds=2.4,
            )
        )

    def stop(self) -> None:
        self.stopped = True

    def readiness(self) -> tuple[bool, str]:
        return True, "ready"

    def status(self) -> dict[str, Any]:
        return {"pipeline_state": "running", "ready": True}


def test_ui_info_preserves_core_defaults_without_exposing_stream_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_url = (
        "rtsp://camera-user:camera-secret@private-camera.example:8554/live"
    )
    monkeypatch.setenv("HARNESS_STREAM_URL", configured_url)
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    config = load_config(ROOT / "config.yaml")
    runtime = HarnessRuntime(config)
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/v1/ui/info")
        assert response.status_code == 200
        info = response.json()
        status = client.get("/v1/status").json()

    assert info["model"] == PUBLIC_MODEL_NAME
    assert status["model"] == PUBLIC_MODEL_NAME
    assert info["sample_fps"] == 2.5
    assert info["max_image_side"] == 320
    assert info["jpeg_quality"] == 75
    assert info["max_concurrent_requests"] == 6
    assert info["annotated_output"] == {
        "output_fps": 15.0,
        "max_image_side": 720,
        "buffer_delay_seconds": 2.5,
        "bbox_max_gap_seconds": 0.8,
    }
    assert info["configured_stream"] is True
    assert info["configured_stream_protocol"] == "rtsp"
    assert info["stream_frame_crop"] == "auto"
    assert "stream_url" not in info
    rendered = json.dumps({"info": info, "status": status}, sort_keys=True)
    assert config.inference.model not in rendered
    assert configured_url not in rendered
    assert "camera-secret" not in rendered
    assert "private-camera.example" not in rendered


@pytest.mark.parametrize("detected", (True, False))
@pytest.mark.parametrize("partial", (True, False))
def test_overall_description_latest_is_correlated_and_compact(
    detected: bool, partial: bool,
) -> None:
    stub = _StubRuntime()
    app = create_app(lambda: stub)  # type: ignore[arg-type]
    with TestClient(app) as client:
        result = replace(
            _analysis_result(detected=detected),
            missing_event_labels=("stranger",) if partial else (),
        )
        judgment = replace(
            result.judgments[0],
            description="有人吸烟。 / A person is smoking.",
        )
        payload = replace(result, judgments=(judgment,)).as_dict()
        record = stub.results.publish(payload)
        response = client.get("/v1/results/overall/latest")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.headers["X-Result-Cursor"] == str(record.cursor)
        assert response.headers["X-Result-Stream"] == stub.results.epoch
        assert response.headers["Cache-Control"] == "no-store"
        expected = {
            key: value for key, value in payload.items()
            if key not in {"judgments", "objects", "model", "analysis_kind"}
        }
        expected["type"] = "overall_description"
        assert response.json() == expected
        if partial:
            assert "Analysis incomplete" in response.json()["description"]
            assert "No anomalies detected" not in response.json()["description"]
        else:
            assert response.json()["description"] == (
                "smoking: 有人吸烟。 / A person is smoking."
                if detected else "无异常 / No anomalies detected"
            )
        assert client.get("/v1/results/latest").json() == payload
        stub.results.reset()
        assert client.get("/v1/results/overall/latest").status_code == 404
        assert client.get("/v1/results/overall/events?after=-1").status_code == 422


@pytest.mark.parametrize("resume", ("query", "header", "stale", "precedence"))
def test_overall_description_sse_delivers_and_resumes(resume: str) -> None:
    stub = _StubRuntime()
    app = create_app(lambda: stub)  # type: ignore[arg-type]
    app.state.runtime = stub
    positive = _analysis_result().as_dict()
    negative = replace(
        _analysis_result(detected=False), message_id="message-2", sequence=2,
        missing_event_labels=("stranger",),
    ).as_dict()
    stub.results.publish(positive)
    stub.results.publish(negative)
    headers = []
    if resume in {"header", "stale", "precedence"}:
        epoch = "old-stream" if resume == "stale" else stub.results.epoch
        cursor = 2 if resume == "precedence" else 1
        headers = [(b"last-event-id", f"{epoch}:{cursor}".encode())]
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {"type": "http", "app": app, "headers": headers}, receive=receive
    )
    route = next(
        route for route in app.routes
        if route.path == "/v1/results/overall/events"
    )

    async def read_messages() -> list[str]:
        response = await route.endpoint(
            request, after=1 if resume in {"query", "precedence"} else 0
        )
        assert response.media_type == "text/event-stream"
        assert response.headers["X-Accel-Buffering"] == "no"
        chunks = []
        try:
            for _ in range(2 if resume == "stale" else 1):
                chunk = await anext(response.body_iterator)
                chunks.append(chunk.decode())
        finally:
            await response.body_iterator.aclose()
        return chunks

    chunks = asyncio.run(read_messages())
    assert len(chunks) == (2 if resume == "stale" else 1)
    for chunk, payload in zip(
        chunks, [positive, negative] if resume == "stale" else [negative], strict=True
    ):
        assert "event: overall_description\n" in chunk
        assert chunk.endswith("\n\n")
        data = json.loads(chunk.split("data: ", 1)[1])
        assert data["message_id"] == payload["message_id"]
        assert data["description"] == payload["description"]
        assert data["analysis_complete"] == payload["analysis_complete"]
        assert data["missing_event_labels"] == payload["missing_event_labels"]
        assert data["frame"] == payload["frame"]
        assert data["type"] == "overall_description"
        assert "judgments" not in data and "objects" not in data
    assert chunks[-1].startswith(f"id: {stub.results.epoch}:2\n")


def test_http_delivery_surface_and_degraded_liveness() -> None:
    stub = _StubRuntime()
    app = create_app(lambda: stub)  # type: ignore[arg-type]
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").status_code == 200
        latest = client.get("/v1/results/latest")
        assert latest.status_code == 200
        assert latest.json()["message_id"] == "message-1"
        assert latest.json()["model"]["name"] == PUBLIC_MODEL_NAME
        assert latest.json()["description"] == "smoking: Visible event."
        assert latest.headers["X-Result-Stream"] == stub.results.epoch
        smoking = client.get("/v1/results/smoking/latest")
        assert smoking.status_code == 200
        assert smoking.headers["X-Result-Feature"] == "smoking"
        assert smoking.json()["description"] == latest.json()["description"]
        assert [item["event_label"] for item in smoking.json()["judgments"]] == [
            "smoking"
        ]
        assert [item["object_id"] for item in smoking.json()["objects"]] == [
            "object-1"
        ]
        assert client.get("/v1/results/fight/latest").status_code == 404
        frame = client.get("/v1/video/latest.jpg")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.headers["X-Analysis-Message-Id"] == "message-1"
        live_frame = client.get("/v1/video/live/latest.jpg")
        assert live_frame.status_code == 200
        assert "X-Analysis-Message-Id" not in live_frame.headers
        annotated = client.get("/v1/video/annotated/latest.jpg")
        assert annotated.status_code == 200
        assert annotated.headers["X-Analysis-Message-Id"] == "message-1"
        assert annotated.headers["X-Next-Analysis-Message-Id"] == "message-2"
        assert annotated.headers["X-Annotation-Mode"] == "interpolated"
        assert annotated.headers["X-Stream-Delay"] == "2.400"
        dashboard = client.get("/")
        assert dashboard.status_code == 200
        assert "qwen/qwen3-vl-8b-instruct" not in dashboard.text
        assert "qwen3.5-plus" not in dashboard.text
        assert "Persistent deployment defaults" in dashboard.text
        assert "/v1/video/annotated/stream.mjpeg" in dashboard.text
        assert "Continuous delayed annotation" in dashboard.text
        assert "Math.max(displayFps,analysisFps)" in dashboard.text
        paths = {route.path for route in app.routes}
        assert "/v1/results/events" in paths
        for feature in ("fight", "smoking", "fall", "weapon", "stranger"):
            assert f"/v1/results/{feature}/latest" in paths
            assert f"/v1/results/{feature}/events" in paths
        assert "/v1/deployment/settings" in paths
        assert "/v1/video/annotated/stream.mjpeg" in paths
        assert "/v1/video/stream.mjpeg" in paths
        assert "/v1/video/live/stream.mjpeg" in paths
    assert stub.started and stub.stopped

    degraded = HarnessRuntime(None, config_error="bad config")
    degraded_app = create_app(lambda: degraded)
    with TestClient(degraded_app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get("/v1/status").json()["pipeline_state"] == (
            "configuration_error"
        )


def test_dashboard_annotation_workspace_has_fullscreen_contract() -> None:
    stub = _StubRuntime()
    app = create_app(lambda: stub)  # type: ignore[arg-type]

    with TestClient(app) as client:
        dashboard = client.get("/")

    assert dashboard.status_code == 200
    html = dashboard.text
    assert html.count('id="analysis-workspace"') == 1
    assert html.count('id="fullscreen-button"') == 1
    assert html.count('id="detection-overlay-toggle"') == 1
    assert 'class="grid analysis-workspace"' in html
    assert 'aria-controls="analysis-workspace" aria-pressed="false"' in html
    assert 'aria-controls="decision" aria-expanded="true"' in html
    assert ".analysis-workspace:fullscreen" in html
    assert ".analysis-workspace.fullscreen-active .classification-panel" in html
    assert "ui.analysisWorkspace.requestFullscreen" in html
    assert "document.exitFullscreen" in html
    assert 'document.addEventListener("fullscreenchange",syncFullscreenUi)' in html
    assert 'new ResizeObserver(()=>alignAnnotationLayer()).observe(ui.viewport)' in html


def test_browser_camera_dry_run_delivers_raw_and_correlated_bbox_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    owner = "browser-test"
    camera_jpeg = _jpeg(width=200, height=100)

    try:
        session = runtime.start_camera(dry_run=True, owner=owner)
        assert session["mode"] == "camera"
        assert session["dry_run"] is True
        assert runtime.push_camera_frame(
            camera_jpeg,
            width=200,
            height=100,
            owner=owner,
        )
        # This frame is still published to the display path but falls between
        # two 2.5 FPS model samples.
        assert runtime.push_camera_frame(
            camera_jpeg,
            width=200,
            height=100,
            owner=owner,
        )
        time.sleep(0.41)
        assert runtime.push_camera_frame(
            camera_jpeg,
            width=200,
            height=100,
            owner=owner,
        )

        feature_records = {}
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            feature_records = {
                label: runtime.results.latest(feature=label)
                for label in DETECTION_EVENT_LABELS
            }
            if all(
                record is not None
                and record.payload["frame"]["source_frame_index"] == 2
                for record in feature_records.values()
            ):
                break
            time.sleep(0.01)

        assert all(record is not None for record in feature_records.values())
        source_results = {
            record.payload["message_id"]: record.payload
            for record in runtime.results.records_after(0)
            if record.payload["frame"]["source_frame_index"] == 2
        }
        assert len(source_results) == 2
        assert len({
            payload["frame"]["timestamp_seconds"]
            for payload in source_results.values()
        }) == 1
        schema = json.loads((
            ROOT / "schemas/analysis-result-v2.schema.json"
        ).read_text(encoding="utf-8"))
        observed = {}
        for payload in source_results.values():
            jsonschema.validate(payload, schema)
            assert payload["analysis_kind"] == "selected_features"
            assert payload["frame"]["width"] == 200
            assert payload["frame"]["height"] == 100
            labels = [item["event_label"] for item in payload["judgments"]]
            assert labels in [list(SAFETY_EVENT_LABELS), ["stranger"]]
            assert payload["analysis_complete"] is False
            assert set(payload["missing_event_labels"]) == (
                set(DETECTION_EVENT_LABELS) - set(labels)
            )
            objects = {item["object_id"]: item for item in payload["objects"]}
            for item in payload["judgments"]:
                observed[item["event_label"]] = item
                if not item["detected"]:
                    assert item["object_ids"] == []
                else:
                    assert item["object_ids"]
                    for object_id in item["object_ids"]:
                        assert objects[object_id]["bbox"]["coordinate_system"] == (
                            "source_normalized_xyxy"
                        )
        assert set(observed) == set(DETECTION_EVENT_LABELS)
        assert {label for label, item in observed.items() if item["detected"]} == {
            "smoking", "stranger",
        }
        for label, record in feature_records.items():
            assert record.payload == source_results[record.payload["message_id"]]
            projection = service_module._feature_result_payload(record.payload, label)
            assert projection["frame"]["source_frame_index"] == 2
            assert projection["judgments"] == [observed[label]]

        raw_update = runtime.preview_frames.latest()
        assert raw_update is not None
        _, raw_packet = raw_update
        raw_image = cv2.imdecode(
            np.frombuffer(raw_packet.jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert raw_image is not None
        assert raw_image.shape[:2] == (100, 200)
        assert raw_packet.frame_sequence == 2
        assert raw_packet.analysis_message_id is None

        annotated_update = runtime.frames.latest()
        assert annotated_update is not None
        _, annotated_packet = annotated_update
        linked_result = source_results[annotated_packet.analysis_message_id]
        assert annotated_packet.analysis_message_id == (
            runtime.results.latest().payload["message_id"]
        )
        assert round(annotated_packet.media_timestamp, 3) == linked_result["frame"][
            "timestamp_seconds"
        ]
        assert annotated_packet.frame_sequence == linked_result["frame"][
            "source_frame_index"
        ]
        assert cv2.imdecode(
            np.frombuffer(annotated_packet.jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        ) is not None
    finally:
        runtime.stop()


def test_camera_runtime_rejects_invalid_frames_and_second_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))

    try:
        runtime.start_camera(dry_run=True, owner="owner-a")
        with pytest.raises(RuntimeError, match="already running"):
            runtime.start_camera(dry_run=True, owner="owner-b")

        with pytest.raises(ValueError, match="exceeds 1280px"):
            runtime.push_camera_frame(
                _jpeg(width=1281, height=100),
                width=1281,
                height=100,
                owner="owner-a",
            )
        with pytest.raises(ValueError, match="dimensions do not match"):
            runtime.push_camera_frame(
                _jpeg(width=200, height=100),
                width=199,
                height=100,
                owner="owner-a",
            )
        with pytest.raises(ValueError, match="not a valid JPEG"):
            runtime.push_camera_frame(
                b"not-a-jpeg",
                width=1,
                height=1,
                owner="owner-a",
            )
        assert not runtime.push_camera_frame(
            _jpeg(width=200, height=100),
            width=200,
            height=100,
            owner="owner-b",
        )
    finally:
        runtime.stop()


def test_readiness_requires_video_and_successful_analysis() -> None:
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))

    class _RunningSession:
        running = True

        def stop(self) -> None:
            self.running = False

        def join(self, timeout: float | None = None) -> None:
            del timeout

    try:
        runtime._session = _RunningSession()  # type: ignore[assignment]
        runtime._update_state(pipeline_state="running")
        runtime._on_display_frame(_jpeg(), 1.0, 1)
        assert runtime.readiness() == (False, "waiting_for_first_analysis")

        result = _analysis_result()
        runtime._on_analysis_frame(result, _jpeg())
        runtime._on_analysis_result(result)
        assert runtime.readiness() == (True, "ready")
        _, packet = runtime.frames.latest() or (0, None)
        assert packet is not None
        assert packet.analysis_message_id == result.message_id
        assert packet.frame_sequence == result.frame.source_frame_index
    finally:
        runtime.stop()


@pytest.mark.parametrize("detected", [True, False])
def test_webhook_uses_stable_idempotency_and_hmac_signature(detected: bool) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    delivery = WebhookDelivery(
        "http://receiver.test/events",
        secret="shared-secret",
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )
    delivery.start()
    payload = _analysis_result(detected=detected).as_dict()
    assert delivery.submit(payload)
    assert delivery.flush(timeout_seconds=2.0)
    delivery.close()

    assert len(requests) == 1
    request = requests[0]
    assert json.loads(request.content) == payload
    assert json.loads(request.content)["description"] == (
        "smoking: Visible event." if detected else "无异常 / No anomalies detected"
    )
    assert request.headers["Idempotency-Key"] == payload["message_id"]
    expected = hmac.new(
        b"shared-secret",
        request.content,
        hashlib.sha256,
    ).hexdigest()
    assert request.headers["X-Harness-Signature-256"] == f"sha256={expected}"
    assert delivery.stats()["delivered"] == 1


def test_webhook_close_discards_backlog_and_worker_can_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = WebhookDelivery(
        "http://receiver.test/events",
        queue_size=1,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_delivery(_client: httpx.Client, _item: object) -> None:
        entered.set()
        assert release.wait(2.0)

    monkeypatch.setattr(delivery, "_deliver", blocked_delivery)
    delivery.start()
    payload = _analysis_result().as_dict()
    assert delivery.submit(payload)
    assert entered.wait(3.0)
    payload["message_id"] = "message-2"
    assert delivery.submit(payload)
    try:
        delivery.close(timeout_seconds=0.01)
        assert delivery.stats()["dropped"] == 1
    finally:
        release.set()
        if delivery._thread is not None:
            delivery._thread.join(2.0)
    assert delivery._thread is not None
    assert not delivery._thread.is_alive()


def test_invalid_dry_run_env_degrades_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_DRY_RUN", "not-a-boolean")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    runtime.start()
    status = runtime.status()
    assert status["pipeline_state"] == "configuration_error"
    assert status["last_error"] == "configuration error"
    assert "HARNESS_DRY_RUN" not in str(status)


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
    assert not contains_private_implementation_detail(specification.text)


def test_webhook_network_failure_reports_failure_without_raw_exception() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("PaddleX OpenRouter network failure")

    delivery = WebhookDelivery(
        "http://receiver.test/events",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    )
    delivery.start()
    try:
        assert delivery.submit({"message_id": "observation-1"})
        assert delivery.flush(timeout_seconds=2)
    finally:
        delivery.close()
    assert delivery.stats()["failed"] == 1
    assert delivery.stats()["last_error"] == "webhook request failed"


def test_service_launcher_hides_server_header_and_maps_capture_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service_module.uvicorn, "run", lambda *_args, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setenv("VIDEO_CAPTURE_OPTIONS", "neutral-options")
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "legacy-options")
    service_module.main()
    assert calls[0]["server_header"] is False
    assert (
        service_module.os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "neutral-options"
    )


def test_service_launcher_keeps_legacy_capture_options_when_neutral_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module.uvicorn, "run", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("VIDEO_CAPTURE_OPTIONS", raising=False)
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "legacy-options")
    service_module.main()
    assert (
        service_module.os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "legacy-options"
    )


@pytest.fixture
def neutral_profile_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    for key in (
        "FACE_PROVIDER", "FACE_ACCELERATOR", "INFERENCE_API_KEY", "AICLUB_API_KEY",
        "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY", "MODEL_API_KEY",
        "INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    connection = data.pop("inference_service")
    for key in ("endpoint", "api_key_env", "provider", "providers"):
        connection.pop(key, None)
    data["inference_service"] = connection
    data["inference"].pop("model", None)
    data["inference"]["profile"] = "standard"
    data["face"].pop("model_name")
    data["face"].pop("provider", None)
    data["face"]["model_profile"] = "standard"
    data["face"]["accelerator"] = "gpu"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_neutral_profiles_preserve_actual_engines(neutral_profile_config: Path) -> None:
    current = load_config(neutral_profile_config)
    assert current.inference.model == "qwen3.5-plus"
    assert current.face.model_name == "buffalo_l"
    assert current.face.provider == "CUDAExecutionProvider"
    assert current.aiclub.api_key_env == "INFERENCE_API_KEY"
    assert current.aiclub.endpoint == AIClubSettings().endpoint
    assert current.aiclub.max_retries == 0
    assert current.model_api is current.aiclub


def test_new_accelerator_environment_wins_over_legacy(
    neutral_profile_config: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_PROVIDER", "CPUExecutionProvider")
    monkeypatch.setenv("FACE_ACCELERATOR", "gpu")
    assert load_config(neutral_profile_config).face.provider == "CUDAExecutionProvider"


def test_legacy_accelerator_override_remains_supported(
    neutral_profile_config: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_PROVIDER", "DmlExecutionProvider")
    assert load_config(neutral_profile_config).face.provider == "DmlExecutionProvider"


@pytest.mark.parametrize("section,key", [
    ("inference", "profile"), ("face", "model_profile"), ("face", "accelerator"),
])
def test_unknown_profile_fails_instead_of_changing_engine(
    neutral_profile_config: Path, section: str, key: str,
) -> None:
    data = yaml.safe_load(neutral_profile_config.read_text(encoding="utf-8"))
    data[section][key] = "unknown"
    neutral_profile_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=section):
        load_config(neutral_profile_config)


@pytest.mark.parametrize("new_key_env", (None, "OVERRIDE_FIXTURE_KEY"))
def test_mixed_connection_settings_preserve_legacy_key_unless_explicitly_overridden(
    neutral_profile_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    new_key_env: str | None,
) -> None:
    data = yaml.safe_load(neutral_profile_config.read_text(encoding="utf-8"))
    data["aiclub"] = {
        "endpoint": "https://receiver.invalid/v1/chat",
        "api_key_env": "CUSTOM_FIXTURE_KEY",
    }
    data["inference_service"]["timeout_seconds"] = 15
    expected_key = new_key_env or "CUSTOM_FIXTURE_KEY"
    if new_key_env is not None:
        data["inference_service"]["api_key_env"] = new_key_env
    monkeypatch.setenv(expected_key, "synthetic-fixture")
    neutral_profile_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    current = load_config(neutral_profile_config)
    assert current.aiclub.endpoint == "https://receiver.invalid/v1/chat"
    assert current.aiclub.timeout_seconds == 15
    assert current.aiclub.api_key_env == expected_key
    assert api_key_present(current.aiclub)
