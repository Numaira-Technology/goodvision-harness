"""Order asynchronous Qwen window responses into one live binary decision.

Window requests run in parallel, so responses can arrive out of order. The
organizer is the single source of truth for "what should the light show right
now": only a successful response for a window newer than the newest applied
one may change the decision, late older responses are recorded but cannot
roll the display backward, and failures update counters without flipping the
light.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from .privacy import PUBLIC_ANALYSIS_ERROR, public_safe_value
from .sliding_window import FrameWindow

if TYPE_CHECKING:
    from .grounding import GroundingResponse
    from .inference import WindowResponse

# Decision lifecycle: warming (no result yet) -> live (newest window applied).
STATUS_WARMING = "warming"
STATUS_LIVE = "live"
_BRANCHES = frozenset({"safety", "stranger"})
_ERROR_CODES = frozenset({
    "request_failed", "timeout", "connection", "http_error",
    "invalid_response", "empty_response",
})


class ResultOrganizer:
    """Thread-safe newest-window-wins state over window responses."""

    def __init__(self, event_label: str) -> None:
        self.event_label = event_label
        self._lock = threading.Lock()
        self._revision = 0
        self._applied_index = -1
        self._status = STATUS_WARMING
        self._decision: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._branch_health: dict[str, tuple[int, str | None]] = {}
        self._counts = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "dropped": 0,
            "detected_windows": 0,
        }

    def note_submitted(
        self, window: FrameWindow, *, elapsed_seconds: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            self._counts["submitted"] += 1
            record = self._window_record(
                window, state="submitted", elapsed_seconds=elapsed_seconds
            )
            return record, self._snapshot_locked()

    def note_dropped(
        self, window: FrameWindow, *, elapsed_seconds: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Record a window skipped because all request slots were busy."""

        with self._lock:
            self._counts["dropped"] += 1
            record = self._window_record(
                window, state="dropped", elapsed_seconds=elapsed_seconds
            )
            return record, self._snapshot_locked()

    def apply(
        self, response: WindowResponse | GroundingResponse, *, elapsed_seconds: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fold one finished response into counters and, if newest, the decision."""

        window = response.window
        with self._lock:
            classification = response.classification
            self._update_branch_health_locked(response)
            if classification is None:
                self._counts["failed"] += 1
                record = self._window_record(
                    window,
                    state="failed",
                    elapsed_seconds=elapsed_seconds,
                    latency_seconds=response.latency_seconds,
                    error=PUBLIC_ANALYSIS_ERROR,
                    applied=False,
                )
                return record, self._snapshot_locked()

            self._counts["completed"] += 1
            if classification.detected:
                self._counts["detected_windows"] += 1
            matched_memory_ids = tuple(
                getattr(classification, "matched_memory_ids", ())
            )

            applied = window.index > self._applied_index
            if applied:
                self._applied_index = window.index
                self._status = STATUS_LIVE
                self._decision = {
                    "detected": classification.detected,
                    "confidence": classification.confidence,
                    "description": public_safe_value(classification.description),
                    "matched_memory_ids": list(matched_memory_ids),
                    "window_index": window.index,
                    "window_label": window.label,
                    "window_start": round(window.start_timestamp, 3),
                    "window_end": round(window.end_timestamp, 3),
                    "model_latency": round(response.latency_seconds, 3),
                }

            record = self._window_record(
                window,
                state="completed",
                elapsed_seconds=elapsed_seconds,
                latency_seconds=response.latency_seconds,
                detected=classification.detected,
                confidence=classification.confidence,
                description=public_safe_value(classification.description),
                matched_memory_ids=matched_memory_ids,
                applied=applied,
            )
            return record, self._snapshot_locked()

    def _update_branch_health_locked(
        self, response: WindowResponse | GroundingResponse,
    ) -> None:
        """Only newer evidence for the same branch can change its health.

        A skipped branch supplies no evidence of either failure or recovery.
        Health has its own ordering because a partial decision may be newer
        than the response that finally resolves another branch's error.
        """

        index = response.window.index
        completed = {
            branch for branch in getattr(response, "completed_branches", ())
            if branch in _BRANCHES
        }
        errors = {
            branch if branch in _BRANCHES else "generic": (
                code if code in _ERROR_CODES else "request_failed"
            )
            for branch, code in getattr(response, "branch_errors", ())
        }
        if completed or errors:
            updates: dict[str, str | None] = {
                branch: None for branch in completed
            }
            updates.update(errors)
            if not errors and self._is_complete_response(response):
                # Unexpected future exceptions use the legacy generic error.
                # A partial face-only result cannot resolve such a failure.
                updates["generic"] = None
        elif response.classification is None or response.error:
            updates = {"generic": "request_failed"}
        else:
            updates = {"generic": None} if self._is_complete_response(response) else {}
        for branch, code in updates.items():
            previous_index = self._branch_health.get(branch, (-1, None))[0]
            if index > previous_index:
                self._branch_health[branch] = (index, code)
        self._last_error = (
            PUBLIC_ANALYSIS_ERROR
            if any(code is not None for _, code in self._branch_health.values())
            else None
        )

    def _is_complete_response(
        self, response: WindowResponse | GroundingResponse,
    ) -> bool:
        classification = response.classification
        if classification is None or response.error:
            return False
        judgments = getattr(classification, "judgments", None)
        if judgments is None:
            return True
        expected = {
            label.strip() for label in self.event_label.split(",") if label.strip()
        }
        return expected <= {judgment.event_label for judgment in judgments}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        self._revision += 1
        counts = dict(self._counts)
        counts["in_flight"] = (
            counts["submitted"] - counts["completed"] - counts["failed"]
        )
        snapshot: dict[str, Any] = {
            "type": "state",
            "revision": self._revision,
            "event_label": self.event_label,
            "status": self._status,
            "decision": dict(self._decision) if self._decision else None,
            "last_error": self._last_error,
            "branch_errors": {
                branch: code
                for branch, (_, code) in sorted(self._branch_health.items())
                if code is not None
            },
            "counts": counts,
        }
        return snapshot

    @staticmethod
    def _window_record(
        window: FrameWindow,
        *,
        state: str,
        elapsed_seconds: float,
        latency_seconds: float | None = None,
        detected: bool | None = None,
        confidence: float | None = None,
        description: str | None = None,
        matched_memory_ids: tuple[str, ...] | None = None,
        error: str | None = None,
        applied: bool | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "type": "window",
            "state": state,
            "window_index": window.index,
            "window_label": window.label,
            "start_seconds": round(window.start_timestamp, 3),
            "end_seconds": round(window.end_timestamp, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        if latency_seconds is not None:
            record["latency_seconds"] = round(latency_seconds, 3)
        if detected is not None:
            record["detected"] = detected
        if confidence is not None:
            record["confidence"] = round(confidence, 3)
        if description is not None:
            record["description"] = description
        if matched_memory_ids is not None:
            record["matched_memory_ids"] = list(matched_memory_ids)
        if error is not None:
            record["error"] = error
        if applied is not None:
            record["applied"] = applied
        return record
