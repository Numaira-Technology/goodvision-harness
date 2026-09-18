from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest

import goodvision_harness.overlay as overlay_module
import goodvision_harness.service as service_module
from goodvision_harness.config import InferenceSettings, load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.grounding import (
    DETECTION_EVENT_LABELS,
    FIXED_FIVE_ANALYSIS_KIND,
    SAFETY_EVENT_LABELS,
    SELECTED_FEATURES_ANALYSIS_KIND,
    AIClubGroundingClient,
    AIClubMultiGroundingClient,
    FixedFiveGroundingClient,
    GroundedClassification,
    GroundedEventClassification,
    GroundingResponse,
    ModelObject,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
    parse_multi_grounded_classification,
)
from goodvision_harness.model_api import ModelAPIClient, ModelAPIError
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]


def _jpeg(width: int = 640, height: int = 360) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _window() -> FrameWindow:
    return FrameWindow(
        index=7,
        frames=(
            SampledFrame(
                sample_index=7,
                source_frame_index=42,
                timestamp_seconds=1.25,
                jpeg_bytes=_jpeg(),
                width=640,
                height=360,
                source_width=1920,
                source_height=1080,
                observed_at="2026-08-27T10:00:00.000Z",
                presentation_jpeg_bytes=_jpeg(width=960, height=540),
            ),
        ),
    )


def _model_object(
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float,
    attributes: dict[str, Any],
) -> ModelObject:
    return ModelObject(
        label=label,
        bbox=NormalizedBBox(*bbox),
        confidence=confidence,
        attributes=attributes,
    )


def _safety_classification() -> MultiGroundedClassification:
    return MultiGroundedClassification(
        judgments=(
            GroundedEventClassification(
                event_label="fight",
                detected=True,
                confidence=0.91,
                description="A raised fist and forceful contact are visible.",
                objects=(
                    _model_object(
                        "person",
                        (0.05, 0.1, 0.35, 0.9),
                        confidence=0.93,
                        attributes={"evidence": "raised_fist"},
                    ),
                ),
            ),
            GroundedEventClassification(
                event_label="smoking",
                detected=True,
                confidence=0.88,
                description="A cigarette is held at the mouth.",
                objects=(
                    _model_object(
                        "cigarette",
                        (0.38, 0.25, 0.43, 0.31),
                        confidence=0.82,
                        attributes={"evidence": "at_mouth"},
                    ),
                ),
            ),
            GroundedEventClassification(
                event_label="fall",
                detected=False,
                confidence=0.96,
                description="No fall is visible.",
            ),
            GroundedEventClassification(
                event_label="weapon",
                detected=False,
                confidence=0.94,
                description="No weapon is visible.",
            ),
        ),
    )


def _stranger_classification(*, unknown: bool) -> GroundedClassification:
    known = _model_object(
        "known_person",
        (0.45, 0.1, 0.65, 0.8),
        confidence=0.97,
        attributes={
            "status": "known",
            "subject_id": "subject-known-alice",
            "subject_ids": ["subject-known-alice"],
            "person_id": "alice",
            "name": "Alice",
            "similarity": 0.94,
        },
    )
    objects = (known,)
    event_object_indexes: tuple[int, ...] = ()
    if unknown:
        objects += (
            _model_object(
                "unknown_person",
                (0.7, 0.12, 0.9, 0.85),
                confidence=0.95,
                attributes={
                    "status": "unknown",
                    "similarity": 0.18,
                    "subject_id": "subject-unknown-1",
                    "subject_ids": ["subject-unknown-1"],
                },
            ),
        )
        event_object_indexes = (1,)
    return GroundedClassification(
        detected=unknown,
        confidence=0.95,
        description=(
            "One known and one unknown face are visible."
            if unknown
            else "Only an authorized person is visible."
        ),
        objects=objects,
        analysis_kind="face_recognition",
        event_object_indexes=event_object_indexes,
        matched_memory_ids=("alice",),
    )


class _StaticClient:
    def __init__(
        self,
        settings: InferenceSettings,
        classification: GroundedClassification | MultiGroundedClassification,
    ) -> None:
        self.settings = settings
        self.classification = classification
        self.windows: list[FrameWindow] = []

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        self.windows.append(window)
        return GroundingResponse(
            window=window,
            classification=self.classification,
            latency_seconds=0.05,
        )


def _combined_client(*, unknown: bool) -> FixedFiveGroundingClient:
    settings = load_config(ROOT / "config.yaml").inference
    safety = _StaticClient(settings, _safety_classification())
    stranger = _StaticClient(
        replace(settings, event_label="unknown_person"),
        _stranger_classification(unknown=unknown),
    )
    return FixedFiveGroundingClient(
        safety,
        stranger,
        safety_max_image_side=320,
        safety_jpeg_quality=75,
    )


def _event_payload(event_label: str, *, detected: bool = False) -> dict[str, Any]:
    return {
        "event_label": event_label,
        "detected": detected,
        "confidence": 0.9,
        "description": f"{event_label} verdict.",
        "objects": (
            [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "bbox_2d": [100, 100, 500, 900],
                }
            ]
            if detected
            else []
        ),
    }


def test_safety_parser_canonicalizes_fixed_order_and_rejects_bad_label_sets() -> None:
    events = [
        _event_payload(label, detected=label == "fight")
        for label in reversed(SAFETY_EVENT_LABELS)
    ]
    parsed = parse_multi_grounded_classification(json.dumps({"events": events}))
    assert tuple(item.event_label for item in parsed.judgments) == (
        SAFETY_EVENT_LABELS
    )

    missing = {"events": events[1:]}
    with pytest.raises(ValueError, match="missing event labels"):
        parse_multi_grounded_classification(json.dumps(missing))

    duplicate = json.loads(json.dumps({"events": events}))
    duplicate["events"][-1]["event_label"] = duplicate["events"][0][
        "event_label"
    ]
    with pytest.raises(ValueError, match="duplicate event_label"):
        parse_multi_grounded_classification(json.dumps(duplicate))

    unknown = json.loads(json.dumps({"events": events}))
    unknown["events"][-1]["event_label"] = "loitering"
    with pytest.raises(ValueError, match="unknown event_label"):
        parse_multi_grounded_classification(json.dumps(unknown))


def test_fixed_five_combination_uses_only_unknown_faces_for_stranger() -> None:
    positive_response = _combined_client(unknown=True).classify_window(_window())
    positive = positive_response.classification
    assert isinstance(positive, MultiGroundedClassification)
    assert positive.analysis_kind == FIXED_FIVE_ANALYSIS_KIND
    assert tuple(item.event_label for item in positive.judgments) == (
        DETECTION_EVENT_LABELS
    )
    stranger = positive.judgments[-1]
    assert stranger.detected is True
    assert [item.label for item in stranger.objects] == ["unknown_person"]
    assert stranger.objects[0].attributes["status"] == "unknown"
    assert stranger.matched_memory_ids == ("alice",)
    assert len(positive.supporting_objects) == 1
    assert positive.supporting_objects[0].attributes["person_id"] == "alice"
    assert positive.supporting_objects[0] in positive.objects

    known_only_response = _combined_client(unknown=False).classify_window(_window())
    known_only = known_only_response.classification
    assert isinstance(known_only, MultiGroundedClassification)
    stranger = known_only.judgments[-1]
    assert stranger.detected is False
    assert stranger.objects == ()
    assert stranger.matched_memory_ids == ("alice",)
    assert known_only.supporting_objects == positive.supporting_objects


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("safety_event", [None, *SAFETY_EVENT_LABELS, "unavailable"])
def test_joint_known_face_renders_independently_of_safety_verdict(
    monkeypatch: pytest.MonkeyPatch, unknown: bool, safety_event: str | None,
) -> None:
    config = load_config(ROOT / "config.yaml")
    safety = MultiGroundedClassification(
        judgments=tuple(
            GroundedEventClassification(
                event_label=label,
                detected=label == safety_event,
                confidence=0.95,
                description="Safety result.",
                objects=(
                    (_model_object(
                        "person", (0.05, 0.2, 0.25, 0.9),
                        confidence=0.95, attributes={},
                    ),)
                    if label == safety_event else ()
                ),
            )
            for label in SAFETY_EVENT_LABELS
        ),
    )
    safety_client = _StaticClient(config.inference, safety)
    if safety_event == "unavailable":
        monkeypatch.setattr(
            safety_client, "classify_window",
            lambda window: GroundingResponse(window, None, 0.1, error="failed"),
        )
    client = FixedFiveGroundingClient(
        safety_client,
        _StaticClient(config.inference, _stranger_classification(unknown=unknown)),
        safety_max_image_side=320,
        safety_jpeg_quality=75,
    )
    results: list[AnalysisResult] = []
    session = StreamingSession.for_camera(
        config, client, SessionCallbacks(on_analysis_result=results.append),
    )
    window = _window()
    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))
    session._on_window_done(future, window)
    analysis = results[0]
    known = next(item for item in analysis.objects if item.label == "known_person")
    assert known.attributes["person_id"] == "alice"
    assert known.bbox == NormalizedBBox(0.45, 0.1, 0.65, 0.8)
    assert all(known.object_id not in item.object_ids for item in analysis.judgments)
    stranger = next(
        item for item in analysis.judgments if item.event_label == "stranger"
    )
    assert stranger.detected is unknown
    assert bool(stranger.object_ids) is unknown
    texts: list[str] = []
    original_text = overlay_module._text_with_background

    def capture_text(image: Any, text: str, *args: Any, **kwargs: Any) -> None:
        texts.append(text)
        original_text(image, text, *args, **kwargs)

    monkeypatch.setattr(overlay_module, "_text_with_background", capture_text)
    rendered = service_module._draw_analysis_overlay(
        np.zeros((360, 640, 3), dtype=np.uint8), analysis,
    )
    has_green = tuple(rendered[287, 352]) == (55, 180, 70)
    assert has_green
    assert any(text.startswith("ID: alice") for text in texts)
    if safety_event in SAFETY_EVENT_LABELS:
        assert tuple(rendered[323, 96]) == (40, 40, 220)
    if unknown:
        assert tuple(rendered[305, 512]) == (40, 40, 220)


def test_selected_feature_subset_runs_only_requested_detectors() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    smoking = _safety_classification().judgments[1]
    safety = _StaticClient(
        settings,
        MultiGroundedClassification((smoking,)),
    )
    stranger = _StaticClient(
        replace(settings, event_label="unknown_person"),
        _stranger_classification(unknown=True),
    )
    client = SelectedFeaturesGroundingClient(
        safety,
        stranger,
        safety_event_labels=("smoking",),
        include_stranger=True,
    )

    response = client.classify_window(_window())

    assert response.ok
    classification = response.classification
    assert isinstance(classification, MultiGroundedClassification)
    assert classification.analysis_kind == SELECTED_FEATURES_ANALYSIS_KIND
    assert [item.event_label for item in classification.judgments] == [
        "smoking",
        "stranger",
    ]
    assert len(safety.windows) == 1
    assert len(stranger.windows) == 1


def test_selected_features_keep_successful_branch_when_other_branch_fails() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    smoking = _safety_classification().judgments[1]
    safety = _StaticClient(
        settings,
        MultiGroundedClassification((smoking,)),
    )

    class _FailingStrangerClient:
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            raise RuntimeError("face detector unavailable")

    client = SelectedFeaturesGroundingClient(
        safety,
        _FailingStrangerClient(),
        safety_event_labels=("smoking",),
        include_stranger=True,
    )

    response = client.classify_window(_window())

    assert response.ok
    assert response.completed_branches == ("safety",)
    assert response.branch_errors == (("stranger", "request_failed"),)
    classification = response.classification
    assert isinstance(classification, MultiGroundedClassification)
    assert [item.event_label for item in classification.judgments] == ["smoking"]
    assert classification.analysis_kind == SELECTED_FEATURES_ANALYSIS_KIND


@pytest.mark.parametrize("failure", ["timeout", "http_error", "invalid_response"])
def test_partial_face_success_preserves_safety_failure_diagnostics(
    failure: str, caplog: pytest.LogCaptureFixture,
) -> None:
    settings = load_config(ROOT / "config.yaml").inference

    class _Transport:
        assistant_text = staticmethod(ModelAPIClient.assistant_text)

        def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            if failure == "invalid_response":
                return {"choices": [{"message": {
                    "content": "private response body, not classification JSON",
                }}]}
            raise ModelAPIError(
                "private URL / secret / provider response",
                code=failure,
                status_code=429 if failure == "http_error" else None,
            )

    client = SelectedFeaturesGroundingClient(
        AIClubMultiGroundingClient(settings, _Transport()),
        _StaticClient(settings, _stranger_classification(unknown=True)),
    )
    response = client.classify_window(_window())
    assert response.ok
    assert response.completed_branches == ("stranger",)
    assert response.branch_errors == (("safety", failure),)
    assert [j.event_label for j in response.classification.judgments] == ["stranger"]
    assert "branch=safety" in caplog.text
    assert f"code={failure}" in caplog.text
    assert "window=7" in caplog.text
    assert "private" not in caplog.text
    assert "secret" not in caplog.text
    if failure == "http_error":
        assert "http_status=429" in caplog.text


def test_both_failed_branches_remain_distinct() -> None:
    settings = load_config(ROOT / "config.yaml").inference

    class _FailingClient:
        def __init__(self, code: str) -> None:
            self.settings = settings
            self.code = code

        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            raise ModelAPIError("not public", code=self.code)

    response = SelectedFeaturesGroundingClient(
        _FailingClient("timeout"), _FailingClient("connection"),
    ).classify_window(_window())
    assert not response.ok
    assert response.branch_errors == (
        ("safety", "timeout"), ("stranger", "connection"),
    )
    assert response.completed_branches == ()


def test_selected_features_drop_busy_face_samples_without_building_a_queue() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    smoking = _safety_classification().judgments[1]
    safety = _StaticClient(
        settings,
        MultiGroundedClassification((smoking,)),
    )
    face_started = threading.Event()
    release_face = threading.Event()

    class _BlockingStrangerClient:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            with self.lock:
                self.calls.append(window.index)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            face_started.set()
            try:
                assert release_face.wait(3.0)
                return GroundingResponse(
                    window=window,
                    classification=_stranger_classification(unknown=True),
                    latency_seconds=0.05,
                )
            finally:
                with self.lock:
                    self.active -= 1

    stranger = _BlockingStrangerClient()
    client = SelectedFeaturesGroundingClient(
        safety,
        stranger,
        safety_event_labels=("smoking",),
        include_stranger=True,
    )

    with ThreadPoolExecutor(max_workers=6) as executor:
        primary = executor.submit(
            client.classify_window,
            replace(_window(), index=0),
        )
        assert face_started.wait(2.0)
        busy = [
            executor.submit(
                client.classify_window,
                replace(_window(), index=index),
            )
            for index in range(1, 6)
        ]
        busy_responses = [future.result(timeout=2.0) for future in busy]

        # All remote safety branches complete while the one face
        # sample is still blocked; no caller queues behind the face runtime.
        assert stranger.calls == [0]
        assert len(safety.windows) == 6
        assert all(window.newest_frame.width == 320 for window in safety.windows)
        for response in busy_responses:
            assert response.completed_branches == ("safety",)
            assert response.branch_errors == ()
            classification = response.classification
            assert isinstance(classification, MultiGroundedClassification)
            assert classification.missing_event_labels == ("stranger",)
            assert [item.event_label for item in classification.judgments] == [
                "smoking"
            ]

        release_face.set()
        primary_response = primary.result(timeout=2.0)

    primary_classification = primary_response.classification
    assert primary_response.completed_branches == ("safety", "stranger")
    assert primary_response.branch_errors == ()
    assert isinstance(primary_classification, MultiGroundedClassification)
    assert primary_classification.missing_event_labels == ()
    assert [
        item.event_label for item in primary_classification.judgments
    ] == ["smoking", "stranger"]
    assert stranger.max_active == 1

    follow_up = client.classify_window(replace(_window(), index=6))
    assert follow_up.ok
    assert stranger.calls == [0, 6]


def test_selected_features_full_face_scheduler_runs_every_concurrent_sample() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    smoking = _safety_classification().judgments[1]
    safety = _StaticClient(
        settings,
        MultiGroundedClassification((smoking,)),
    )
    all_faces_started = threading.Event()
    release_faces = threading.Event()

    class _BlockingStrangerClient:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            with self.lock:
                self.calls.append(window.index)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if len(self.calls) == 6:
                    all_faces_started.set()
            try:
                assert release_faces.wait(3.0)
                return GroundingResponse(
                    window=window,
                    classification=_stranger_classification(unknown=True),
                    latency_seconds=0.05,
                )
            finally:
                with self.lock:
                    self.active -= 1

    stranger = _BlockingStrangerClient()
    client = SelectedFeaturesGroundingClient(
        safety,
        stranger,
        safety_event_labels=("smoking",),
        include_stranger=True,
        face_live_scheduler="full",
    )

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                client.classify_window,
                replace(_window(), index=index),
            )
            for index in range(6)
        ]
        assert all_faces_started.wait(2.0)
        assert sorted(stranger.calls) == list(range(6))
        release_faces.set()
        responses = [future.result(timeout=2.0) for future in futures]

    assert stranger.max_active == 6
    assert len(safety.windows) == 6
    for response in responses:
        classification = response.classification
        assert isinstance(classification, MultiGroundedClassification)
        assert [item.event_label for item in classification.judgments] == [
            "smoking",
            "stranger",
        ]


def test_selected_features_release_face_slot_after_failure() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    smoking = _safety_classification().judgments[1]
    safety = _StaticClient(
        settings,
        MultiGroundedClassification((smoking,)),
    )

    class _FailOnceStrangerClient:
        def __init__(self) -> None:
            self.calls = 0

        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary face failure")
            return GroundingResponse(
                window=window,
                classification=_stranger_classification(unknown=True),
                latency_seconds=0.05,
            )

    stranger = _FailOnceStrangerClient()
    client = SelectedFeaturesGroundingClient(
        safety,
        stranger,
        safety_event_labels=("smoking",),
        include_stranger=True,
    )

    first = client.classify_window(replace(_window(), index=0))
    second = client.classify_window(replace(_window(), index=1))

    assert first.ok
    assert second.ok
    assert stranger.calls == 2
    classification = second.classification
    assert isinstance(classification, MultiGroundedClassification)
    assert [item.event_label for item in classification.judgments] == [
        "smoking",
        "stranger",
    ]


def test_stranger_only_selection_keeps_public_stranger_label() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    stranger = _StaticClient(
        replace(settings, event_label="unknown_person"),
        _stranger_classification(unknown=True),
    )
    client = SelectedFeaturesGroundingClient(
        None,
        stranger,
        safety_event_labels=(),
        include_stranger=True,
    )

    response = client.classify_window(_window())

    classification = response.classification
    assert isinstance(classification, MultiGroundedClassification)
    assert [item.event_label for item in classification.judgments] == ["stranger"]
    assert classification.analysis_kind == SELECTED_FEATURES_ANALYSIS_KIND
    assert len(stranger.windows) == 1


def test_fixed_five_runs_safety_and_stranger_in_parallel_on_same_frame() -> None:
    settings = load_config(ROOT / "config.yaml").inference
    barrier = threading.Barrier(2)

    class _BarrierClient(_StaticClient):
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            self.windows.append(window)
            barrier.wait(timeout=2.0)
            return GroundingResponse(
                window=window,
                classification=self.classification,
                latency_seconds=0.05,
            )

    safety = _BarrierClient(settings, _safety_classification())
    stranger = _BarrierClient(
        replace(settings, event_label="unknown_person"),
        _stranger_classification(unknown=True),
    )
    client = FixedFiveGroundingClient(
        safety,
        stranger,
        safety_max_image_side=320,
        safety_jpeg_quality=75,
    )

    response = client.classify_window(_window())

    assert response.ok
    assert safety.windows[0].newest_frame.width == 320
    assert stranger.windows[0].newest_frame.width == 640
    assert (
        safety.windows[0].newest_frame.timestamp_seconds
        == stranger.windows[0].newest_frame.timestamp_seconds
    )

@pytest.mark.parametrize(
    "scenario", ("face_failure", "safety_failure", "busy", "disabled")
)
def test_partial_negative_results_never_report_all_clear(scenario: str) -> None:
    config = load_config(ROOT / "config.yaml")
    smoking = replace(
        _safety_classification().judgments[1], detected=False, objects=(),
        description="未发现吸烟。 / No smoking detected.",
    )
    safety = _StaticClient(config.inference, MultiGroundedClassification((smoking,)))
    stranger = _StaticClient(config.inference, _stranger_classification(unknown=False))
    if scenario == "face_failure":
        stranger.classification = None
    if scenario == "safety_failure":
        safety.classification = None
    client = SelectedFeaturesGroundingClient(
        safety, None if scenario == "disabled" else stranger,
        safety_event_labels=("smoking",), include_stranger=scenario != "disabled",
    )
    if scenario == "busy":
        client._stranger_live_slot.acquire()
    try:
        window = _window()
        response = client.classify_window(window)
    finally:
        if scenario == "busy":
            client._stranger_live_slot.release()
    emitted = []
    session = StreamingSession.for_camera(
        config, client, SessionCallbacks(on_analysis_result=emitted.append)
    )
    future: Future[Any] = Future()
    future.set_result(response)
    session._on_window_done(future, window)
    assert len(emitted) == 1
    payload = emitted[0].as_dict()
    overall = service_module._overall_description_payload(payload)
    expected_missing = (
        [] if scenario == "disabled"
        else ["smoking"] if scenario == "safety_failure" else ["stranger"]
    )
    for result in (payload, overall):
        assert result["missing_event_labels"] == expected_missing
        assert result["analysis_complete"] is (scenario == "disabled")
        if expected_missing:
            assert "Analysis incomplete" in result["description"]
            assert "无异常" not in result["description"]
            assert "No anomalies detected" not in result["description"]
        else:
            assert result["description"] == "无异常 / No anomalies detected"


@pytest.mark.parametrize("multi", (True, False))
@pytest.mark.parametrize("description,valid", (
    ("No smoking detected.", False),
    ("未发现吸烟。", False),
    ("No smoking detected. / 未发现吸烟。", True),
    ("未发现吸烟。 / ", False),
    ("未发现吸烟。 / No smoking detected.", True),
))
def test_live_grounding_rejects_invalid_bilingual_descriptions(
    multi: bool, description: str, valid: bool,
) -> None:
    class Transport:
        def chat_completion(self, **kwargs: Any) -> dict:
            return {}

        def assistant_text(self, response: dict) -> str:
            judgment = {
                "detected": False, "confidence": 0.9,
                "description": description, "objects": [],
            }
            return json.dumps(
                {"events": [{"event_label": "smoking", **judgment}]}
                if multi else judgment
            )

    settings = load_config(ROOT / "config.yaml").inference
    client = (
        AIClubMultiGroundingClient(settings, Transport(), event_labels=("smoking",))
        if multi else AIClubGroundingClient(settings, Transport())
    )
    response = client.classify_window(_window())
    assert response.ok is valid
    if valid:
        classification = response.classification
        actual = classification.judgments[0] if multi else classification
        assert actual.description == "未发现吸烟。 / No smoking detected."
    else:
        assert response.classification is None
        assert response.raw_text is None


def test_pipeline_emits_one_rich_fixed_five_analysis_result() -> None:
    config = load_config(ROOT / "config.yaml")
    client = _combined_client(unknown=True)
    emitted: list[AnalysisResult] = []
    emitted_frames: list[tuple[AnalysisResult, bytes]] = []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_analysis_result=emitted.append,
            on_analysis_frame=lambda result, jpeg: emitted_frames.append(
                (result, jpeg)
            ),
        ),
        source_id="camera-multi",
    )
    window = _window()
    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))

    session._on_window_done(future, window)

    assert len(emitted) == 1
    result = emitted[0]
    payload = result.as_dict()
    assert payload["analysis_kind"] == FIXED_FIVE_ANALYSIS_KIND
    assert payload["observation_id"] == payload["message_id"]
    assert payload["source_id"] == "camera-multi"
    assert payload["frame"]["source_frame_index"] == 42
    assert payload["frame"]["width"] == 1920
    assert payload["description"] == (
        "fight: A raised fist and forceful contact are visible.; "
        "smoking: A cigarette is held at the mouth.; "
        "stranger: One known and one unknown face are visible."
    )
    assert [item["event_label"] for item in payload["judgments"]] == list(
        DETECTION_EVENT_LABELS
    )

    judgments = {item["event_label"]: item for item in payload["judgments"]}
    assert [
        label for label, item in judgments.items() if item["detected"]
    ] == ["fight", "smoking", "stranger"]
    assert judgments["fall"]["object_ids"] == []
    assert judgments["weapon"]["object_ids"] == []
    assert judgments["stranger"]["matched_memory_ids"] == ["alice"]

    objects = {item["object_id"]: item for item in payload["objects"]}
    fight_object = objects[judgments["fight"]["object_ids"][0]]
    smoking_object = objects[judgments["smoking"]["object_ids"][0]]
    stranger_object = objects[judgments["stranger"]["object_ids"][0]]
    assert fight_object["attributes"] == {"evidence": "raised_fist"}
    assert smoking_object["attributes"] == {"evidence": "at_mouth"}
    assert stranger_object["label"] == "unknown_person"
    assert stranger_object["attributes"]["status"] == "unknown"
    assert stranger_object["attributes"]["subject_id"] == "subject-unknown-1"
    assert stranger_object["attributes"]["subject_ids"] == ["subject-unknown-1"]
    assert "stranger_id" not in stranger_object["attributes"]
    assert "photo_url" not in stranger_object["attributes"]
    # Person association uses the existing open object-attributes extension.
    # The version and top-level contract are unchanged.
    assert payload["schema_version"] == "2"
    assert "subject_id" not in payload
    schema = json.loads(
        (ROOT / "schemas/analysis-result-v2.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(payload)
    known_objects = [
        item for item in payload["objects"]
        if item["attributes"].get("status") == "known"
    ]
    assert len(known_objects) == 1
    assert known_objects[0]["attributes"]["person_id"] == "alice"
    assert known_objects[0]["object_id"] not in judgments["stranger"]["object_ids"]
    assert emitted_frames == [(result, window.newest_frame.presentation_jpeg_bytes)]

    # Capability views keep the original frame summary, including when their
    # own judgment is negative but other events were detected in the frame.
    for feature in ("smoking", "fall"):
        filtered = service_module._feature_result_payload(payload, feature)
        assert filtered is not None
        assert len(filtered["judgments"]) == 1
        assert filtered["description"] == payload["description"]


def test_renderer_passes_each_shared_positive_object_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = GroundedObject(
        object_id="object-1",
        label="person",
        bbox=NormalizedBBox(0.1, 0.1, 0.7, 0.9),
        confidence=0.93,
        attributes={"evidence": "shared"},
    )
    unreferenced = GroundedObject(
        object_id="object-2",
        label="background-item",
        bbox=NormalizedBBox(0.75, 0.1, 0.9, 0.3),
        confidence=0.72,
    )
    analysis = AnalysisResult(
        message_id="message-multi",
        source_id="camera-multi",
        stream_session_id="session-multi",
        sequence=1,
        emitted_at="2026-08-27T10:00:01.000Z",
        frame=FrameReference(
            source_frame_index=42,
            timestamp_seconds=1.25,
            observed_at="2026-08-27T10:00:00.000Z",
            width=200,
            height=100,
        ),
        judgments=(
            EventJudgment(
                "fight",
                True,
                0.91,
                "Fight evidence.",
                ("object-1",),
            ),
            EventJudgment("smoking", False, 0.96, "No smoking."),
            EventJudgment("fall", False, 0.97, "No fall."),
            EventJudgment(
                "weapon",
                True,
                0.89,
                "Weapon evidence.",
                ("object-1",),
            ),
            EventJudgment("stranger", False, 0.95, "No stranger."),
        ),
        objects=(shared, unreferenced),
        model_name="test-model",
        model_latency_seconds=0.1,
        analysis_kind=FIXED_FIVE_ANALYSIS_KIND,
    )
    captured: dict[str, Any] = {}

    def capture_overlay(image: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return image

    monkeypatch.setattr(service_module, "draw_overlay", capture_overlay)
    rendered = service_module._render_analysis_jpeg(
        _jpeg(200, 100), analysis, quality=85
    )

    assert cv2.imdecode(
        np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR
    ) is not None
    assert captured["event_label"] == "fight | weapon"
    assert captured["detected"] is True
    assert [item.object_id for item in captured["objects"]] == ["object-1"]
    assert captured["objects"][0].label == "fight / weapon | person"


def test_renderer_names_only_selected_features_when_all_are_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = AnalysisResult(
        message_id="message-selected-clear",
        source_id="camera-multi",
        stream_session_id="session-multi",
        sequence=1,
        emitted_at="2026-08-27T10:00:01.000Z",
        frame=FrameReference(
            source_frame_index=42,
            timestamp_seconds=1.25,
            observed_at="2026-08-27T10:00:00.000Z",
            width=200,
            height=100,
        ),
        judgments=(
            EventJudgment("smoking", False, 0.96, "No smoking."),
        ),
        objects=(),
        model_name="test-model",
        model_latency_seconds=0.1,
        analysis_kind=SELECTED_FEATURES_ANALYSIS_KIND,
    )
    captured: dict[str, Any] = {}

    def capture_overlay(image: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return image

    monkeypatch.setattr(service_module, "draw_overlay", capture_overlay)
    service_module._render_analysis_jpeg(_jpeg(200, 100), analysis, quality=85)

    assert captured["event_label"] == "smoking"
    assert captured["detected"] is False
