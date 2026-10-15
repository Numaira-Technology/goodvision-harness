"""Schedule fixed offline windows with native models, prompts, parsing, and publication.

Process frozen windows under backpressure without dropping them or sending warmups.
The two lanes publish independently of person detection. The cumulative ledger
guards paid requests. This scheduler is for offline capture, not realtime service.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
import time

from ..model_api import ModelAPIError
from ..result_bus import ResultBus
from ..runtime.scene import TemporalInspection
from ..runtime.scheduling import InspectionRequest
from ..vlm_independent import IndependentEventState, _inspector, utc_now


class ObservationFailure(RuntimeError):
    """Preserve failures for the controller; do not recast 402/connection faults as question failures."""

    def __init__(self, failures):
        super().__init__("independent_window_failed; see persisted lane reports")
        self.failures = tuple(failures)


def fixed_window_plan(frame_plan, profile, *, observation_fps=1.25):
    """Freeze up to four source-PTS frames per window with the native 0.75-second context rule."""
    if not 0 < observation_fps <= 3:
        raise ValueError("observation cadence must be within (0, 3]")
    rows, history, next_due, previous = [], [], 0.0, None
    for position, frame in enumerate(frame_plan):
        history.append(frame)
        history = [row for row in history if frame["t"] - row["t"] <= 3.0][-12:]
        # Include the last decoded frame even off cadence; do not substitute an unseen media end time.
        if frame["t"] + 1e-9 < next_due and position + 1 != len(frame_plan):
            continue
        window = history[-4:]
        if len(window) == 4 and window[-1]["t"] - window[0]["t"] < .75:
            older = next((row for row in reversed(history[:-4])
                          if window[-1]["t"] - row["t"] >= .75), None)
            if older is not None:
                window = [older, *window[-3:]]
        if len(window) < 2 or window[-1]["t"] - window[0]["t"] < .75:
            continue
        row = profile.window(window_id=f"window-{len(rows):06d}",
                             start=window[0]["t"], end=frame["t"])
        row.update(frame_sequences=[item["frame"] for item in window],
                   evidence_after_frame_sequence=previous)
        rows.append(row)
        previous = frame["frame"]
        next_due = frame["t"] + 1.0 / observation_fps
    return rows


class FixedWindowObserver:
    def __init__(self, config, *, recorder, on_grounding, ledger, inspector_factory=_inspector):
        if config.output_mode != "parallel":
            raise ValueError("formal observation requires both independent lanes")
        options = dict(config.options)
        self.transport = options.pop("evaluation_transport", {})
        self.config = replace(config, options=options)
        self.recorder, self.on_grounding = recorder, on_grounding
        self.factory, self.ledger = inspector_factory, ledger
        # Preserve main's events+compact parallel lanes rather than switching to full prompts.
        self.results = {lane: ResultBus() for lane in ("events", "compact")}
        self.executors = {lane: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"eval-{lane}")
                          for lane in self.results}
        self.clients = {}
        self.state = IndependentEventState(config)
        self.lock = Lock()
        self.pending = []
        self.closed = False
        self.completed_windows = 0

    def submit(self, window, samples):
        if self.closed or self.pending:
            raise RuntimeError("previous fixed window must drain before another submission")
        frames = tuple(sample.frame for sample in samples)
        if ([frame.frame_sequence for frame in frames] != window["frame_sequences"]
                or frames[0].timestamp_seconds != window["start"]
                or frames[-1].timestamp_seconds != window["end"]):
            raise ValueError("frozen independent window input mismatch")
        request_id = f"{self.config.stream_session_id}:{window['window_id']}"
        request = InspectionRequest(request_id, (), "independent_source", self.config.labels,
                                    window["start"], window["end"])
        job = TemporalInspection(
            self.config.stream_session_id, self.config.source_id, request, frames,
            event_sensitivities=self.config.sensitivities,
            evidence_after_frame_sequence=window["evidence_after_frame_sequence"],
            allowed_event_labels=self.config.allowed_event_labels,
            scene_only_event_labels=self.config.scene_only_event_labels,
            event_definitions=self.config.event_definitions,
        )
        # Pass source images and the fixed profile, without person boxes, questions, or ground truth.
        self.pending = [self.executors[lane].submit(self._observe, lane, job, samples[-1])
                        for lane in self.results]

    def _observe(self, lane, job, sample):
        sequences = [frame.frame_sequence for frame in job.frames]
        budget_key = lane + ":" + ",".join(map(str, sequences))
        options = dict(self.config.options)
        options["max_tokens"] = self.transport.get("legacy_window_budgets", {}).get(
            budget_key, self.transport.get("lane_max_tokens", {}).get(
                lane, options.get("max_tokens", 1600)))
        client_key = (lane, options["max_tokens"])
        inspector = self.clients.get(client_key)
        if inspector is None:
            inspector = self.factory(replace(self.config, output_mode=lane, options=options))
            self.ledger.attach(inspector, {"stream_session_id": self.config.stream_session_id,
                                          "source_id": self.config.source_id, "lane": lane})
            self.clients[client_key] = inspector
        inspector.evaluation_context = {"request_id": job.request.request_id,
                                        "source_frame_sequences": [frame.frame_sequence for frame in job.frames]}
        inspector.last_ledger_receipt = None
        started, started_at = time.monotonic(), utc_now()
        verdicts, failure = (), None
        backoff_seconds = 0.0
        # Retry explicit 429s only; ledger accounting enforces three sends per complete input.
        # Keep backoff outside the 75-second HTTP deadline to avoid cancellation/unknown outcomes.
        for attempt in range(3):
            try:
                verdicts = inspector.inspect(job)
                failure = None
                break
            except ModelAPIError as exc:
                failure = exc
                if exc.status_code != 429 or attempt == 2:
                    break
                delay = max(60.0, getattr(exc, "evaluation_diagnostic", {}).get("retry_after_seconds") or 0)
                if delay > 300:
                    break
                self.recorder.timing({"kind": "vlm_rate_limit_backoff",
                    "clock_domain": self.recorder.clock_domain,
                    "lane": lane, "request_id": job.request.request_id, "seconds": delay,
                    "ledger_receipt": inspector.last_ledger_receipt, "same_payload": True})
                waiting = time.monotonic()
                time.sleep(delay)
                backoff_seconds += time.monotonic() - waiting
        # Return programming/I/O exceptions through the future for controller recovery.
        ended = time.monotonic()
        timing = dict(request_started_at=started_at, request_ended_at=utc_now(),
                      request_started_monotonic=started, model_seconds=ended - started - backoff_seconds,
                      rate_limit_backoff_seconds=backoff_seconds,
                      observation_wall_seconds=ended - started,
                      http_status=getattr(failure, "status_code", None),
                      validation_reason=getattr(failure, "validation_reason", None),
                      effective_timeout_seconds=inspector.settings.timeout_seconds,
                      effective_max_tokens=options["max_tokens"],
                      source_to_request_seconds=started - sample.received_monotonic)
        with self.lock:
            report = self.state.report(job, sample, verdicts, timing,
                                       error=getattr(failure, "code", None) if failure else None)
            report.update(vlm_lane="information" if lane == "events" else "grounding",
                          output_mode=lane, window_id=job.request.request_id,
                          historical_only=False, scheduling_mode="offline_fixed_backpressure")
            published = time.monotonic()
            report["timing"].update(published_at=utc_now(), published_monotonic=published,
                                    source_to_publish_seconds=published - sample.received_monotonic,
                                    clock_domain=self.recorder.clock_domain)
            self.results[lane].publish(report)
            self.recorder.independent_report(report)
            self.recorder.timing({"clock_domain": self.recorder.clock_domain,
                                  "kind": "independent_publication", "lane": report["vlm_lane"],
                                  "request_id": job.request.request_id,
                                  "frame_sequence": sample.frame.frame_sequence,
                                  "source_ingress_monotonic": sample.received_monotonic,
                                  "request_started_monotonic": started,
                                  "response_received_monotonic": ended,
                                  "published_monotonic": published,
                                  "cached": (inspector.last_ledger_receipt or {}).get("cached"),
                                  "ledger_receipt": inspector.last_ledger_receipt,
                                  "success": failure is None})
            if lane != "events" and failure is None:
                self.on_grounding(report, job.frames)
        return failure

    def drain_window(self):
        failures, exceptions = [], []
        futures, self.pending = self.pending, []
        for future in futures:
            try:
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
            except Exception as exc:
                exceptions.append(exc)
        if exceptions:
            raise exceptions[0]
        if failures:
            raise ObservationFailure(failures) from failures[0]
        if futures:
            self.completed_windows += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Close on the lane's inspect thread to release its asyncio connection pool.
        close_errors = []
        try:
            for future in self.pending:
                try:
                    future.result()
                except Exception:
                    pass
            self.pending = []
            for lane, executor in self.executors.items():
                try:
                    executor.submit(self._close_lane, lane).result()
                except Exception as exc:
                    close_errors.append(exc)
        finally:
            for executor in self.executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
        if close_errors:
            raise close_errors[0]

    def _close_lane(self, lane):
        for (client_lane, _), inspector in tuple(self.clients.items()):
            if client_lane == lane:
                inspector.close()
