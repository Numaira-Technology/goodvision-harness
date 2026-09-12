"""Bounded temporal inference on a session-owned background thread."""

from __future__ import annotations

import math
from collections.abc import Callable
from threading import Event, RLock, Thread, current_thread
from typing import Any

from .scene_inspection import (
    TemporalInspector,
    _read_image,
    _validate_job,
    _validate_verdicts,
)
from .scene_runtime import SceneRuntime, TemporalInspection


class SceneInspectionWorker:
    """One in-flight model call per session, independent of video callbacks.

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
    ) -> None:
        if (
            type(poll_seconds) not in (int, float)
            or not math.isfinite(poll_seconds)
            or not 0 < poll_seconds <= 60
        ):
            raise ValueError("poll_seconds must be finite and within (0, 60]")
        self.runtime = runtime
        self.inspector = inspector
        self._on_update = on_update
        self._poll_seconds = poll_seconds
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
            while not self._stop.is_set():
                self._wake.clear()
                jobs = self.runtime.take_inspections(limit=1)
                with self._lock:
                    self._stats["claimed_requests"] += len(jobs)
                for index, job in enumerate(jobs):
                    if self._stop.is_set():
                        with self._lock:
                            self._stats["discarded_requests"] += len(jobs) - index
                        break
                    self._inspect_one(job, capacity_exceeded=index >= 4)
                self._wake.wait(self._poll_seconds)
        except Exception:
            # Neither model exceptions nor callback exceptions may leak inputs.
            with self._lock:
                self._stats["worker_failures"] += 1
        finally:
            closer = getattr(self.inspector, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    with self._lock:
                        self._stats["worker_failures"] += 1

    def _inspect_one(
        self, job: TemporalInspection, *, capacity_exceeded: bool = False
    ) -> None:
        if not self.runtime.inspection_active(job.request.request_id):
            with self._lock:
                self._stats["discarded_requests"] += 1
            return
        verdicts = ()
        success = False
        with self._lock:
            self._stats["active_requests"] = 1
        try:
            if capacity_exceeded:
                raise ValueError("temporal execution capacity exceeded")
            allowed = _validate_job(job)
            for frame in job.frames:
                with _read_image(frame.jpeg_bytes):
                    pass
            verdicts = _validate_verdicts(job, self.inspector.inspect(job), allowed)
            success = True
        except Exception:
            # Failure/absence of a response is never negative event evidence.
            verdicts = ()
        finally:
            with self._lock:
                self._stats["active_requests"] = 0
        if self._stop.is_set() or not self.runtime.inspection_active(
            job.request.request_id
        ):
            with self._lock:
                self._stats["discarded_requests"] += 1
            return
        counts: dict[str, int] = {}
        acknowledged = self.runtime.complete_inspection(
            job.request.request_id,
            verdicts,
            success=success,
            completion_counts=counts,
        )
        with self._lock:
            if not acknowledged:
                self._stats["discarded_requests"] += 1
            elif success:
                self._stats["succeeded_requests"] += 1
                self._stats["submitted_verdicts"] += len(verdicts)
                self._stats["accepted_verdicts"] += counts.get("accepted_verdicts", 0)
                self._stats["rejected_verdicts"] += counts.get("rejected_verdicts", 0)
            else:
                self._stats["failed_requests"] += 1
        if acknowledged and not self._stop.is_set() and self._on_update is not None:
            try:
                self._on_update()
            except Exception:
                with self._lock:
                    self._stats["callback_failures"] += 1

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
                "max_in_flight": 1,
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
