from __future__ import annotations

import json
from dataclasses import replace

import pytest

from goodvision_harness.grounding import (
    GroundedEventClassification,
    GroundingResponse,
    MultiGroundedClassification,
)
from goodvision_harness.inference import WindowResponse
from goodvision_harness.organizer import ResultOrganizer
from goodvision_harness.privacy import PUBLIC_ANALYSIS_ERROR
from goodvision_harness.prompt import BinaryClassification
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def _window(index: int) -> FrameWindow:
    return FrameWindow(index=index, frames=(SampledFrame(
        sample_index=index, source_frame_index=index,
        timestamp_seconds=index * 0.4, jpeg_bytes=b"fixture", width=10, height=10,
    ),))


def _legacy(index: int, *, failed: bool = False) -> WindowResponse:
    return WindowResponse(
        window=_window(index),
        classification=None if failed else BinaryClassification(
            detected=False, confidence=0.9, description="No event.",
        ),
        raw_text=None,
        latency_seconds=0.5,
        error="private provider details" if failed else None,
    )


def _branches(
    index: int,
    *,
    completed: tuple[str, ...] = (),
    errors: tuple[tuple[str, str], ...] = (),
) -> GroundingResponse:
    judgments = tuple(
        GroundedEventClassification(
            event_label="smoking" if branch == "safety" else "stranger",
            detected=False, confidence=0.9, description="No event.",
        )
        for branch in completed
    )
    return GroundingResponse(
        window=_window(index),
        classification=MultiGroundedClassification(judgments=judgments)
        if judgments else None,
        latency_seconds=0.5,
        error=PUBLIC_ANALYSIS_ERROR if not judgments else None,
        branch_errors=errors,
        completed_branches=completed,
    )


def _apply(
    organizer: ResultOrganizer, response: WindowResponse | GroundingResponse,
) -> tuple[dict, dict]:
    organizer.note_submitted(response.window, elapsed_seconds=0.0)
    return organizer.apply(response, elapsed_seconds=1.0)


def test_newer_legacy_success_clears_error_and_keeps_failure_count() -> None:
    organizer = ResultOrganizer("smoking")
    failed, state = _apply(organizer, _legacy(1, failed=True))
    assert failed["error"] == PUBLIC_ANALYSIS_ERROR
    assert state["last_error"] == PUBLIC_ANALYSIS_ERROR

    _, recovered = _apply(organizer, _legacy(2))

    assert recovered["last_error"] is None
    assert recovered["branch_errors"] == {}
    assert recovered["counts"]["failed"] == 1
    assert recovered["counts"]["completed"] == 1
    assert recovered["counts"]["in_flight"] == 0


@pytest.mark.parametrize("success_index", [1, 2])
def test_old_or_duplicate_success_cannot_hide_newer_failure(success_index: int) -> None:
    organizer = ResultOrganizer("smoking")
    _apply(organizer, _legacy(2, failed=True))

    _, state = _apply(organizer, _legacy(success_index))

    assert state["last_error"] == PUBLIC_ANALYSIS_ERROR
    assert state["branch_errors"] == {"generic": "request_failed"}
    assert state["decision"]["window_index"] == success_index


def test_late_failure_is_recorded_without_reintroducing_recovered_error() -> None:
    organizer = ResultOrganizer("smoking")
    _apply(organizer, _legacy(3))

    record, state = _apply(organizer, _legacy(2, failed=True))

    assert record["state"] == "failed"
    assert record["error"] == PUBLIC_ANALYSIS_ERROR
    assert state["counts"]["failed"] == 1
    assert state["last_error"] is None
    assert state["branch_errors"] == {}
    assert state["decision"]["window_index"] == 3


def test_partial_face_success_does_not_clear_safety_error() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _, partial = _apply(organizer, _branches(
        1, completed=("stranger",), errors=(("safety", "timeout"),),
    ))
    assert partial["counts"]["completed"] == 1
    assert partial["counts"]["failed"] == 0
    assert partial["branch_errors"] == {"safety": "timeout"}

    _, face_only = _apply(organizer, _branches(2, completed=("stranger",)))
    assert face_only["last_error"] == PUBLIC_ANALYSIS_ERROR
    assert face_only["branch_errors"] == {"safety": "timeout"}

    _, recovered = _apply(organizer, _branches(3, completed=("safety",)))
    assert recovered["last_error"] is None
    assert recovered["branch_errors"] == {}


def test_skipped_face_branch_neither_fails_nor_recovers_face_health() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _, initial = _apply(organizer, _branches(1, completed=("safety",)))
    assert initial["last_error"] is None
    assert initial["branch_errors"] == {}
    _apply(organizer, _branches(
        2, completed=("safety",), errors=(("stranger", "request_failed"),),
    ))

    _, skipped = _apply(organizer, _branches(3, completed=("safety",)))

    assert skipped["branch_errors"] == {"stranger": "request_failed"}
    _, recovered = _apply(organizer, _branches(4, completed=("stranger",)))
    assert recovered["last_error"] is None


def test_branch_recovery_is_independent_of_newer_partial_decision() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _apply(organizer, _branches(1, errors=(("safety", "connection"),)))
    _apply(organizer, _branches(3, completed=("stranger",)))

    record, state = _apply(organizer, _branches(2, completed=("safety",)))

    assert record["applied"] is False
    assert state["decision"]["window_index"] == 3
    assert state["last_error"] is None
    assert state["branch_errors"] == {}


def test_old_branch_success_cannot_clear_newer_branch_failure() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _apply(organizer, _branches(3, errors=(("safety", "http_error"),)))

    _, state = _apply(organizer, _branches(2, completed=("safety",)))

    assert state["branch_errors"] == {"safety": "http_error"}
    assert state["last_error"] == PUBLIC_ANALYSIS_ERROR


def test_old_branch_failure_cannot_override_newer_branch_success() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _apply(organizer, _branches(3, completed=("safety",)))

    _, state = _apply(organizer, _branches(2, errors=(("safety", "timeout"),)))

    assert state["branch_errors"] == {}
    assert state["last_error"] is None
    assert state["counts"]["failed"] == 1


def test_each_failed_branch_requires_its_own_recovery() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _apply(organizer, _branches(1, errors=(
        ("safety", "timeout"), ("stranger", "connection"),
    )))

    _, safety_recovered = _apply(organizer, _branches(2, completed=("safety",)))
    assert safety_recovered["branch_errors"] == {"stranger": "connection"}
    assert safety_recovered["last_error"] == PUBLIC_ANALYSIS_ERROR
    _, face_recovered = _apply(organizer, _branches(3, completed=("stranger",)))
    assert face_recovered["branch_errors"] == {}
    assert face_recovered["last_error"] is None


def test_generic_failure_requires_complete_recovery_from_branch_aware_client() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _apply(organizer, _legacy(1, failed=True))

    _, partial = _apply(organizer, _branches(2, completed=("stranger",)))
    assert partial["branch_errors"] == {"generic": "request_failed"}
    _, complete = _apply(organizer, _branches(3, completed=("safety", "stranger")))
    assert complete["branch_errors"] == {}
    assert complete["last_error"] is None


def test_unannotated_partial_response_cannot_hide_generic_failure() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _apply(organizer, _legacy(1, failed=True))
    partial = replace(_branches(2, completed=("stranger",)), completed_branches=())

    _, state = _apply(organizer, partial)

    assert state["last_error"] == PUBLIC_ANALYSIS_ERROR
    assert state["branch_errors"] == {"generic": "request_failed"}


def test_unknown_error_code_cannot_leak_provider_details() -> None:
    organizer = ResultOrganizer("smoking,stranger")
    _, state = _apply(organizer, _branches(1, errors=(
        ("safety", "https://private-provider.example/secret"),
    )))

    assert state["branch_errors"] == {"safety": "request_failed"}
    assert state["last_error"] == PUBLIC_ANALYSIS_ERROR
    assert "private-provider" not in json.dumps(state)
