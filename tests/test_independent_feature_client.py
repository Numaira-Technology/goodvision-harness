from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from goodvision_harness.config import InferenceSettings
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
)
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundedEventClassification,
    GroundingResponse,
    ModelActor,
    ModelObject,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
    build_face_diagnostics,
)
from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def _window(index: int = 0) -> FrameWindow:
    ok, encoded = cv2.imencode(".jpg", np.zeros((360, 640, 3), dtype=np.uint8))
    assert ok
    return FrameWindow(
        index=index,
        frames=(
            SampledFrame(
                sample_index=index,
                source_frame_index=index,
                timestamp_seconds=index * 0.4,
                jpeg_bytes=encoded.tobytes(),
                width=640,
                height=360,
                source_width=1920,
                source_height=1080,
            ),
        ),
    )


def _face(status: str = "unknown", **attributes: Any) -> ModelObject:
    return ModelObject(
        "face",
        NormalizedBBox(0.2, 0.1, 0.4, 0.4),
        0.95,
        {"status": status, **attributes},
    )


def _safety() -> MultiGroundedClassification:
    actor = ModelActor("a1", NormalizedBBox(0.1, 0.1, 0.8, 0.9))
    item = ModelObject(
        "cigarette", NormalizedBBox(0.3, 0.3, 0.4, 0.4), actor_ids=("a1",)
    )
    return MultiGroundedClassification(
        (
            GroundedEventClassification("smoking", True, 0.9, "Smoking.", (item,)),
            GroundedEventClassification("fall", False, 0.9, "No fall."),
        ),
        actors=(actor,),
    )


def _stranger(*, positive: bool = True) -> GroundedClassification:
    faces = (_face("known"), _face("unusable", reason="identity_unconfirmed"))
    if positive:
        faces += (_face(),)
    return GroundedClassification(
        positive,
        0.9,
        "Local face result.",
        faces,
        event_object_indexes=(2,) if positive else (),
        face_observations=faces,
    )


class _Client:
    settings = InferenceSettings("model", "smoking", "system", "prompt")

    def __init__(
        self, classification: GroundedClassification | MultiGroundedClassification
    ) -> None:
        self.classification = classification
        self.windows: list[FrameWindow] = []

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        self.windows.append(window)
        return GroundingResponse(window, self.classification, 0.01)


def _combined(
    safety: _Client | None = None, stranger: _Client | None = None
) -> SelectedFeaturesGroundingClient:
    return SelectedFeaturesGroundingClient(
        safety or _Client(_safety()),
        stranger or _Client(_stranger()),
        safety_event_labels=("smoking", "fall"),
    )


@pytest.mark.parametrize("branch", ["safety", "stranger"])
def test_independent_branch_returns_while_other_branch_is_blocked(branch: str) -> None:
    entered = threading.Event()
    release = threading.Event()

    class _BlockingClient(_Client):
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            entered.set()
            assert release.wait(3)
            return super().classify_window(window)

    blocked_branch = "stranger" if branch == "safety" else "safety"
    blocked = _BlockingClient(_stranger() if branch == "safety" else _safety())
    client = _combined(**{blocked_branch: blocked})
    window = _window()
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(
            getattr(client, f"classify_{blocked_branch}_window"), window
        )
        try:
            assert entered.wait(2)
            response = executor.submit(
                getattr(client, f"classify_{branch}_window"), window
            ).result(timeout=2)
            assert response.completed_branches == (branch,)
            assert response.branch_errors == ()
            assert response.window is window
            assert not pending.done()
        finally:
            release.set()
        assert pending.result(timeout=2).completed_branches == (blocked_branch,)


def test_independent_branches_keep_original_frame_and_private_observations() -> None:
    safety, stranger = _Client(_safety()), _Client(_stranger())
    client = _combined(safety, stranger)
    window = _window()
    safety_response = client.classify_safety_window(window)
    assert stranger.windows == []
    face_response = client.classify_stranger_window(window)
    assert len(safety.windows) == 1
    assert client.independent_live_branches
    assert safety_response.window is face_response.window is window
    assert safety.windows[0].newest_frame.width == 320
    assert safety.windows[0].newest_frame.height == 180
    assert stranger.windows == [window]
    assert stranger.windows[0].newest_frame.jpeg_bytes is window.newest_frame.jpeg_bytes
    safety_result = safety_response.classification
    face_result = face_response.classification
    assert isinstance(safety_result, MultiGroundedClassification)
    assert isinstance(face_result, MultiGroundedClassification)
    assert safety_result.missing_event_labels == ("stranger",)
    assert face_result.missing_event_labels == ("smoking", "fall")
    assert safety_result.actors == safety.classification.actors
    assert safety_result.objects[0].actor_ids == ("a1",)
    assert face_result.actors == ()
    assert face_result.face_observations == stranger.classification.face_observations
    assert len(face_result.objects) == 1
    assert safety_result.face_observations == ()
    assert safety_response.face_diagnostics is None


def test_face_branch_ignores_legacy_internal_single_flight_slot() -> None:
    client = _combined()
    client._stranger_live_slot.acquire()
    try:
        response = client.classify_stranger_window(_window())
    finally:
        client._stranger_live_slot.release()
    assert response.completed_branches == ("stranger",)
    assert response.classification.detected


def test_missing_face_response_is_failure_not_scheduler_skip() -> None:
    class _MissingResponseClient(_Client):
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            return None

    response = _combined(
        stranger=_MissingResponseClient(_stranger())
    ).classify_stranger_window(_window())
    assert response.classification is None
    assert response.completed_branches == ()
    assert response.branch_errors == (("stranger", "request_failed"),)


@pytest.mark.parametrize("invalid_jpeg", [b"invalid image", b"\xff\xd8\xff"])
def test_safety_input_failure_does_not_prevent_independent_face(
    invalid_jpeg: bytes,
) -> None:
    client = _combined()
    window = _window()
    window = replace(
        window, frames=(replace(window.newest_frame, jpeg_bytes=invalid_jpeg),)
    )
    safety_response = client.classify_safety_window(window)
    face_response = client.classify_stranger_window(window)
    assert safety_response.classification is None
    assert safety_response.completed_branches == ()
    assert safety_response.branch_errors == (("safety", "invalid_response"),)
    assert face_response.completed_branches == ("stranger",)
    assert face_response.branch_errors == ()


@pytest.mark.parametrize("branch", ["safety", "stranger"])
def test_branch_failure_only_reports_its_own_error(
    branch: str, caplog: pytest.LogCaptureFixture
) -> None:
    class _FailingClient(_Client):
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            raise ModelAPIError("private response url and credential", code="timeout")

    client = _combined(**{branch: _FailingClient(_stranger())})
    response = getattr(client, f"classify_{branch}_window")(_window())
    assert response.classification is None
    assert response.completed_branches == ()
    assert response.branch_errors == ((branch, "timeout"),)
    assert "private" not in caplog.text
    assert "credential" not in caplog.text


@pytest.mark.parametrize("branch", ["safety", "stranger"])
def test_wrong_classification_type_fails_only_own_branch(branch: str) -> None:
    client = _combined(
        **{branch: _Client(_stranger() if branch == "safety" else _safety())}
    )
    response = getattr(client, f"classify_{branch}_window")(_window())
    assert response.classification is None
    assert response.branch_errors == ((branch, "invalid_response"),)
    assert response.completed_branches == ()


@pytest.mark.parametrize("branch", ["safety", "stranger"])
def test_branch_retains_own_reported_error_and_ignores_foreign_health(
    branch: str,
) -> None:
    class _FailedResponseClient(_Client):
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            return GroundingResponse(
                window,
                None,
                0.01,
                branch_errors=(("safety", "timeout"), ("stranger", "connection")),
                completed_branches=("foreign",),
            )

    client = _combined(**{branch: _FailedResponseClient(_stranger())})
    response = getattr(client, f"classify_{branch}_window")(_window())
    expected_code = "timeout" if branch == "safety" else "connection"
    assert response.branch_errors == ((branch, expected_code),)
    assert response.completed_branches == ()


@pytest.mark.parametrize("branch", ["safety", "stranger"])
def test_single_branch_selection_keeps_synchronous_contract(branch: str) -> None:
    client = SelectedFeaturesGroundingClient(
        _Client(_safety()) if branch == "safety" else None,
        _Client(_stranger()) if branch == "stranger" else None,
        safety_event_labels=("smoking", "fall") if branch == "safety" else (),
        include_stranger=branch == "stranger",
    )
    assert not client.independent_live_branches
    response = client.classify_window(_window())
    assert response.completed_branches == (branch,)
    assert response.classification.missing_event_labels == ()
    absent = "stranger" if branch == "safety" else "safety"
    with pytest.raises(ValueError, match="not selected"):
        getattr(client, f"classify_{absent}_window")(_window())


def test_negative_face_result_preserves_unconfirmed_and_quality_diagnostics() -> None:
    raw = (_face(), _face("unusable", reason="low_detection_confidence"))
    stable = (
        _face("unusable", reason="identity_unconfirmed", unknown_confirmations=2),
        raw[1],
    )
    diagnostics = build_face_diagnostics(raw, stable)

    class _DiagnosticClient(_Client):
        def classify_window(self, window: FrameWindow) -> GroundingResponse:
            return GroundingResponse(
                window,
                replace(
                    _stranger(positive=False), objects=stable, face_observations=stable
                ),
                0.01,
                face_diagnostics=diagnostics,
            )

    client = _combined(stranger=_DiagnosticClient(_stranger()))
    response = client.classify_stranger_window(_window())
    assert response.face_diagnostics is diagnostics
    assert not response.classification.detected
    assert response.classification.objects == ()
    assert response.classification.face_observations == stable
    assert diagnostics["detected_face_count"] == 2
    assert diagnostics["raw_status_counts"]["unknown"] == 1
    assert diagnostics["stable_status_counts"]["unusable"] == 2
    assert diagnostics["reason_counts"] == {
        "identity_unconfirmed": 1, "low_detection_confidence": 1
    }
    assert diagnostics["faces"][0]["unknown_confirmations"] == 2


def test_face_diagnostics_are_bounded_and_do_not_copy_identity_data() -> None:
    item = _face(
        "known",
        name="PRIVATE PERSON",
        person_id="PRIVATE ID",
        face_width=65,
        face_height=80,
        detection_score=0.95,
        reason="PRIVATE REASON" * 1000,
        unknown_confirmations="PRIVATE COUNTER",
        source_url="PRIVATE URL",
    )
    diagnostics = build_face_diagnostics((item,) * 100, (item,) * 100)
    assert diagnostics["face_count"] == 100
    assert diagnostics["truncated"] is True
    assert len(diagnostics["faces"]) == len(diagnostics["raw_faces"]) == 8
    assert diagnostics["reason_counts"] == {"other": 100}
    assert diagnostics["faces"][0]["face_width"] == 65
    assert "unknown_confirmations" not in diagnostics["faces"][0]
    assert "PRIVATE" not in json.dumps(diagnostics)


def test_legacy_face_client_does_not_claim_raw_detector_observations() -> None:
    response = _combined().classify_stranger_window(_window())
    diagnostics = response.face_diagnostics
    assert diagnostics["state"] == "unavailable"
    assert diagnostics["raw_observations_available"] is False
    assert diagnostics["detected_face_count"] is None
    assert diagnostics["raw_status_counts"] is None


@pytest.mark.parametrize("mixed", [False, True])
def test_runtime_diagnostics_distinguish_single_frame_and_held_stranger(
    tmp_path: Path, mixed: bool,
) -> None:
    face = SimpleNamespace(
        bbox=np.array([100, 50, 180, 150]),
        normed_embedding=np.array([1.0, 0.0]),
        det_score=0.95,
    )
    observations = iter([[face], [face], [face], []])
    runtime = FaceRecognitionRuntime(
        FaceRecognizerConfig(gallery_path=tmp_path / "gallery.json")
    )
    runtime._model = SimpleNamespace(get=lambda image: next(observations))
    session = FaceRecognitionSession(runtime)
    classify = (
        _combined(stranger=session).classify_stranger_window
        if mixed else session.classify_window
    )
    responses = [classify(_window(index)) for index in range(4)]
    for index in range(3):
        diagnostics = responses[index].face_diagnostics
        assert diagnostics["raw_status_counts"]["unknown"] == 1
        assert responses[index].classification.detected
        assert diagnostics["stable_status_counts"]["unknown"] == 1
        assert diagnostics["reason_counts"] == {"none": 1}
        assert diagnostics["faces"][0]["unknown_confirmations"] == index + 1
    assert responses[2].classification.detected
    assert responses[2].face_diagnostics["stable_status_counts"]["unknown"] == 1
    held = responses[3]
    assert held.classification.detected
    assert held.face_diagnostics["state"] == "no_faces"
    assert held.face_diagnostics["detected_face_count"] == 0
    assert held.face_diagnostics["face_count"] == 1
    assert held.face_diagnostics["raw_status_counts"]["unknown"] == 0
    assert held.face_diagnostics["faces"][0]["identity_source"] == "track_history"
