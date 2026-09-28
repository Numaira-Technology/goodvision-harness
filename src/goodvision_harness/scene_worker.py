"""Bounded temporal inference on a session-owned background thread."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, RLock, Thread, current_thread
from typing import Any

from .diagnostics import note
from .event_registry import SAFETY_BRANCH, branch_events, labels
from .latency_trace import log_timing, trace_id
from .model_api import ModelAPIError
from .scene_inspection import (
    TemporalInspector,
    _read_image,
    _validate_job,
    _validate_verdicts,
)
from .scene_runtime import BenchmarkInspection, SceneRuntime, TemporalInspection, TemporalVerdict

# Deliberately the built-in safety labels rather than whatever a session
# requested: the verdict log must stay restricted to fixed public labels.
_PUBLIC_VERDICT_LABELS = frozenset(labels(branch_events(SAFETY_BRANCH)))


@contextmanager
def _timed_phase(timing: dict[str, Any], phase: str):
    started = time.monotonic()
    try:
        yield
    finally:
        timing[phase] = time.monotonic() - started


@dataclass(slots=True)
class _InspectionOutcome:
    job: TemporalInspection
    started: float
    timing: dict[str, Any]
    verdicts: tuple[TemporalVerdict, ...] = ()
    success: bool = False
    discarded: bool = False
    finished: float = 0.0
    benchmark: BenchmarkInspection | None = None


def _verdict_log_fields(outcome: _InspectionOutcome) -> dict[str, Any]:
    """Count validated response groups, not accepted evidence or person coverage.

    Omitted means that a requested event label has no returned group. Reporting
    one group does not prove coverage of every person or possible interaction.
    A failed/unavailable response has no measurable verdict coverage, including
    no manufactured unknown/negative verdict. Only fixed public labels enter logs.
    """
    requested = sorted(
        set(outcome.job.request.event_labels).intersection(_PUBLIC_VERDICT_LABELS)
    )
    counts = {
        label: dict.fromkeys(("true", "false", "unknown", "omitted"), None)
        for label in requested
    }
    reported = omitted = None
    if outcome.success:
        state = "validated" if outcome.verdicts else "validated_empty"
        for row in counts.values():
            row.update(dict.fromkeys(row, 0))
        for verdict in outcome.verdicts:
            field = (
                "true" if verdict.detected is True else
                "false" if verdict.detected is False else "unknown"
            )
            counts[verdict.event_label][field] += 1
        for row in counts.values():
            row["omitted"] = int(not any(row[field] for field in (
                "true", "false", "unknown",
            )))
        omitted = sum(row["omitted"] for row in counts.values())
        reported = len(counts) - omitted
    else:
        error = outcome.timing.get("error_type")
        state = "failed" if error and error != "InspectionExpired" else "unavailable"
    return {
        "verdict_response_state": state,
        "event_label_coverage": {
            "requested": len(counts), "reported": reported, "omitted": omitted,
        },
        "event_verdict_counts": counts,
    }


@dataclass(slots=True)
class _InspectionSlot:
    inspector: TemporalInspector
    queue: Queue = field(default_factory=Queue)
    thread: Thread | None = None
    busy: bool = False


class SceneInspectionWorker:
    """Bounded independent model clients, with one ordered completion owner.

    close() stops new work immediately and joins only for its supplied budget.
    In-flight responses are discarded; their client is closed by the worker
    after the call returns. The owner also closes the runtime on session reset
    and must generation-guard on_update against a replaced service session.
    The callback runs outside all worker/runtime locks and should return promptly.
    """

    def __init__(
        self,
        runtime: SceneRuntime,
        inspector: TemporalInspector,
        *,
        on_update: Callable[[], None] | None = None,
        poll_seconds: float = 0.1,
        max_in_flight: int = 1,
        inspector_factory: Callable[[], TemporalInspector] | None = None,
        on_benchmark: Callable | None = None,
    ) -> None:
        if (
            type(poll_seconds) not in (int, float)
            or not math.isfinite(poll_seconds)
            or not 0 < poll_seconds <= 60
        ):
            raise ValueError("poll_seconds must be finite and within (0, 60]")
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= 4:
            raise ValueError("max_in_flight must be within 1..4")
        if max_in_flight > 1 and not callable(inspector_factory):
            raise ValueError("parallel inspection requires independent client factory")
        self.runtime = runtime
        self.inspector = inspector
        self._on_update = on_update
        self._on_benchmark = on_benchmark
        self._poll_seconds = poll_seconds
        self._max_in_flight = max_in_flight
        self._inspector_factory = inspector_factory
        self._slots = [_InspectionSlot(inspector)]
        self._completed: Queue = Queue()
        self._pending: dict[
            str, tuple[TemporalInspection, _InspectionOutcome | None]
        ] = {}
        self._dispatch_times: dict[str, float] = {}
        self._wake = Event()
        self._stop = Event()
        self._lock = RLock()
        self._thread: Thread | None = None
        self._stats = {
            "claimed_requests": 0,
            "succeeded_requests": 0,
            "failed_requests": 0,
            "discarded_requests": 0,
            "submitted_verdicts": 0,
            "accepted_verdicts": 0,
            "rejected_verdicts": 0,
            "callback_failures": 0,
            "worker_failures": 0,
            "active_requests": 0,
            "pending_completions": 0,
            "expired_while_running": 0,
        }

    def start(self) -> None:
        with self._lock:
            if self._stop.is_set() or self._thread is not None:
                return
            self._thread = Thread(
                target=self._run, name="scene-temporal-inspection", daemon=True
            )
            self._thread.start()

    def wake(self) -> None:
        if not self._stop.is_set():
            self._wake.set()

    def _run(self) -> None:
        try:
            if not self._stop.is_set():
                while len(self._slots) < self._max_in_flight:
                    assert self._inspector_factory is not None
                    inspector = self._inspector_factory()
                    if any(slot.inspector is inspector for slot in self._slots):
                        raise ValueError("inspection clients must be independent")
                    self._slots.append(_InspectionSlot(inspector))
                for index, slot in enumerate(self._slots):
                    slot.thread = Thread(
                        target=self._run_slot, args=(slot,),
                        name=f"scene-temporal-model-{index}", daemon=True,
                    )
                    slot.thread.start()
            while not self._stop.is_set():
                self._wake.clear()
                self._collect_outcomes()
                self._commit_ready()
                available = [slot for slot in self._slots if not slot.busy]
                capacity = min(
                    len(available), self._max_in_flight - len(self._pending),
                )
                if capacity:
                    jobs = self.runtime.take_inspections(limit=capacity)
                    with self._lock:
                        self._stats["claimed_requests"] += len(jobs)
                    for slot, job in zip(available, jobs, strict=False):
                        key = job.request.request_id
                        if key in self._pending:
                            raise ValueError("duplicate inspection claim")
                        self._pending[key] = (job, None)
                        self._dispatch_times[key] = time.monotonic()
                        slot.busy = True
                        slot.queue.put(job)
                with self._lock:
                    self._stats["pending_completions"] = len(self._pending)
                self._wake.wait(self._poll_seconds)
        except Exception:
            # Neither model exceptions nor callback exceptions may leak inputs.
            with self._lock:
                self._stats["worker_failures"] += 1
        finally:
            # Each client is closed only after its own in-flight call finishes.
            # The coordinator remains alive while draining, but close() itself
            # still joins for only the caller's supplied budget.
            for slot in self._slots:
                if slot.thread is not None:
                    slot.queue.put(None)
            for slot in self._slots:
                if slot.thread is not None:
                    slot.thread.join()
            self._collect_outcomes()
            for job, outcome in self._pending.values():
                if outcome is None:
                    outcome = self._new_outcome(job)
                outcome.discarded = True
                self._commit_outcome(outcome)
            self._pending.clear()
            self._dispatch_times.clear()
            with self._lock:
                self._stats["pending_completions"] = 0
            for slot in self._slots:
                self._close_inspector(slot.inspector)

    def _close_inspector(self, inspector: TemporalInspector) -> None:
        closer = getattr(inspector, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                with self._lock:
                    self._stats["worker_failures"] += 1

    def _run_slot(self, slot: _InspectionSlot) -> None:
        while True:
            job = slot.queue.get()
            if job is None:
                return
            try:
                outcome = self._evaluate_one(job, slot.inspector)
            except Exception as exc:
                outcome = self._new_outcome(job)
                outcome.timing["error_type"] = type(exc).__name__
                with self._lock:
                    self._stats["worker_failures"] += 1
            self._completed.put((slot, outcome))
            self._wake.set()

    def _collect_outcomes(self) -> None:
        while True:
            try:
                slot, outcome = self._completed.get_nowait()
            except Empty:
                return
            slot.busy = False
            key = outcome.job.request.request_id
            if key in self._pending:
                self._pending[key] = (outcome.job, outcome)
            else:
                if self._on_benchmark is not None:
                    self._on_benchmark(
                        outcome.job, outcome.benchmark,
                        accepted=False, error="late_response",
                    )
                log_timing(
                    "vlm_late_result_timing",
                    stream_session_id=trace_id(outcome.job.stream_session_id),
                    request_id=trace_id(key), completed_observations=0,
                    context_frame_sequences=[
                        frame.frame_sequence for frame in outcome.job.frames
                    ],
                    observation_end_frame_sequence=(
                        outcome.job.frames[-1].frame_sequence
                        if outcome.job.frames else None
                    ),
                    observation_end_timestamp_seconds=(
                        outcome.job.frames[-1].timestamp_seconds
                        if outcome.job.frames else None
                    ),
                    total_seconds=round(time.monotonic() - outcome.started, 6),
                    **_verdict_log_fields(outcome),
                    **outcome.timing,
                )
            # An expired call still owns its client until it returns; a late
            # response cannot create a second completion or negative evidence.

    def _commit_ready(self) -> None:
        for key, (job, outcome) in tuple(self._pending.items()):
            if outcome is None:
                if self.runtime.inspection_active(key):
                    break
                outcome = self._new_outcome(job)
                outcome.started = self._dispatch_times.get(key, outcome.started)
                outcome.discarded = True
                outcome.timing["error_type"] = "InspectionExpired"
                # The client is still running. These durations/attempts are
                # unknown here, not zero; its eventual timing is logged separately.
                for field in ("model_attempts", "model_seconds", "prepare_seconds",
                              "verdict_validation_seconds"):
                    outcome.timing[field] = None
                with self._lock:
                    self._stats["expired_while_running"] += 1
            self._pending.pop(key)
            self._dispatch_times.pop(key, None)
            self._commit_outcome(outcome)

    @staticmethod
    def _new_outcome(job: TemporalInspection) -> _InspectionOutcome:
        timing: dict[str, Any] = dict.fromkeys(
            ("prepare_seconds", "model_seconds", "verdict_validation_seconds",
             "complete_seconds", "callback_seconds", "reorder_wait_seconds"), 0.0,
        )
        timing["model_attempts"] = 0
        return _InspectionOutcome(job, time.monotonic(), timing)

    def _evaluate_one(
        self, job: TemporalInspection, inspector: TemporalInspector, *,
        capacity_exceeded: bool = False,
    ) -> _InspectionOutcome:
        outcome = self._new_outcome(job)
        timing = outcome.timing
        if self._stop.is_set() or not self.runtime.inspection_active(
            job.request.request_id
        ):
            outcome.discarded = True
            outcome.finished = time.monotonic()
            return outcome
        with self._lock:
            self._stats["active_requests"] += 1
        try:
            with _timed_phase(timing, "prepare_seconds"):
                if capacity_exceeded:
                    raise ValueError("temporal execution capacity exceeded")
                allowed = _validate_job(job)
                for frame in job.frames:
                    with _read_image(frame.jpeg_bytes):
                        pass
            timing["model_attempts"] = 1
            with _timed_phase(timing, "model_seconds"):
                response = inspector.inspect(job)
            with _timed_phase(timing, "verdict_validation_seconds"):
                if isinstance(response, BenchmarkInspection):
                    outcome.benchmark = response
                    response = response.verdicts
                outcome.verdicts = _validate_verdicts(job, response, allowed)
            note(job, "responded", verdicts=outcome.verdicts)
            outcome.success = True
        except Exception as exc:
            # A missing response never becomes negative event evidence.
            note(job, "responded", error=exc)
            outcome.verdicts = ()
            if job.benchmark_questions and getattr(exc, "benchmark_response", None) is not None:
                outcome.benchmark = BenchmarkInspection((), (), exc.benchmark_response)
            timing["error_type"] = type(exc).__name__
            if isinstance(exc, ModelAPIError):
                timing["error_code"] = exc.code
                timing["http_status"] = exc.status_code
        finally:
            outcome.finished = time.monotonic()
            with self._lock:
                self._stats["active_requests"] -= 1
        return outcome

    def _inspect_one(
        self, job: TemporalInspection, *, capacity_exceeded: bool = False
    ) -> None:
        """Synchronous seam retained for deterministic adapter/timing tests."""
        self._commit_outcome(self._evaluate_one(
            job, self.inspector, capacity_exceeded=capacity_exceeded,
        ))

    def _commit_outcome(self, outcome: _InspectionOutcome) -> None:
        job, timing = outcome.job, outcome.timing
        counts = dict.fromkeys((
            "succeeded_requests", "failed_requests", "discarded_requests",
            "submitted_verdicts", "accepted_verdicts", "rejected_verdicts",
            "callback_failures",
        ), 0)
        recorded = False
        acknowledged = False
        try:
            if outcome.finished:
                timing["reorder_wait_seconds"] = max(
                    0.0, time.monotonic() - outcome.finished,
                )
            if outcome.discarded or self._stop.is_set() or not (
                self.runtime.inspection_active(job.request.request_id)
            ):
                counts["discarded_requests"] = 1
                return
            completed: dict[str, int] = {}
            with _timed_phase(timing, "complete_seconds"):
                acknowledged = self.runtime.complete_inspection(
                    job.request.request_id,
                    outcome.verdicts,
                    success=outcome.success,
                    completion_counts=completed,
                )
            if not acknowledged:
                counts["discarded_requests"] = 1
            elif outcome.success:
                counts["succeeded_requests"] = 1
                counts["submitted_verdicts"] = len(outcome.verdicts)
                counts["accepted_verdicts"] = completed.get("accepted_verdicts", 0)
                counts["rejected_verdicts"] = completed.get("rejected_verdicts", 0)
            else:
                counts["failed_requests"] = 1
            with self._lock:
                for key, value in counts.items():
                    self._stats[key] += value
            recorded = True
            if acknowledged and not self._stop.is_set() and self._on_update is not None:
                try:
                    with _timed_phase(timing, "callback_seconds"):
                        self._on_update()
                except Exception as exc:
                    timing["callback_error_type"] = type(exc).__name__
                    counts["callback_failures"] = 1
        finally:
            if self._on_benchmark is not None:
                try:
                    self._on_benchmark(
                        job, outcome.benchmark,
                        accepted=acknowledged and outcome.success,
                        error=timing.get("error_type"),
                    )
                except Exception:
                    counts["callback_failures"] += 1
            # Per-job counts are independent of other parallel requests.
            with self._lock:
                for key, value in counts.items():
                    if not recorded or key == "callback_failures":
                        self._stats[key] += value
            try:
                log_timing(
                    "vlm_inspection_timing",
                    branch="vlm",
                    stream_session_id=trace_id(job.stream_session_id),
                    request_id=trace_id(job.request.request_id),
                    context_frame_sequences=[
                        frame.frame_sequence for frame in job.frames
                    ],
                    context_frames=len(job.frames),
                    observation_end_frame_sequence=(
                        job.frames[-1].frame_sequence if job.frames else None
                    ),
                    observation_end_timestamp_seconds=(
                        job.frames[-1].timestamp_seconds if job.frames else None
                    ),
                    completed_observations=counts["succeeded_requests"],
                    total_seconds=round(time.monotonic() - outcome.started, 6),
                    **_verdict_log_fields(outcome),
                    **{key: round(value, 6) if isinstance(value, float) else value
                       for key, value in timing.items()},
                    **counts,
                )
            except Exception:
                pass

    def stats(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            return {
                **self._stats,
                "state": (
                    "stopping"
                    if self._stop.is_set() and thread and thread.is_alive()
                    else "stopped"
                    if self._stop.is_set()
                    else "running"
                    if thread and thread.is_alive()
                    else "failed"
                    if thread
                    else "ready"
                ),
                "max_in_flight": self._max_in_flight,
            }

    def close(self, timeout_seconds: float = 1.0) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("close timeout must be finite and non-negative")
        self._stop.set()
        self._wake.set()
        with self._lock:
            # An unstarted worker still owns its inspector. A cleanup thread
            # keeps caller shutdown bounded even for an arbitrary adapter.
            if self._thread is None:
                self._thread = Thread(
                    target=self._run, name="scene-temporal-cleanup", daemon=True
                )
                self._thread.start()
            thread = self._thread
        if thread is not current_thread():
            thread.join(timeout=timeout_seconds)
