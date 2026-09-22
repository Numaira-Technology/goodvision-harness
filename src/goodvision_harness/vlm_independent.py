"""Independent source admission, spawned VLM process and immediate result delivery.

No service, SAM, scene, archive or identity lock is reachable from the child or
the output reader. Fusion has its own bounded consumer after publication.
"""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import os
import time
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Empty, Full
from threading import Event, Lock, Thread, local
from uuid import uuid4

from PIL import Image

from .contracts import PUBLIC_MODEL_NAME
from .latency_trace import log_timing
from .result_bus import ResultBus
from .scene_evidence import EvidenceFrame, InspectionRequest
from .scene_runtime import TemporalInspection
from .vlm_independent_model import IndependentVLMInspector

MAX_FRAME_BYTES = 4 * 1024**2
INPUT_CAPACITY = 8
OUTPUT_CAPACITY = 16
REORDER_CAPACITY = 32


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VLMProcessConfig:
    stream_session_id: str
    source_id: str
    generation: int
    labels: tuple[str, ...]
    model_settings: object
    model: str
    api_key: str = field(default="", repr=False)
    parameters: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    sensitivities: tuple[tuple[str, str], ...] = ()
    observation_fps: float = 1.25
    max_in_flight: int = 4
    route_id: str | None = None

    def __post_init__(self):
        if not self.stream_session_id or not self.source_id or not self.labels:
            raise ValueError("VLM process requires a scoped source and event labels")
        if not 0 < self.observation_fps <= 3 or not math.isfinite(self.observation_fps):
            raise ValueError("VLM cadence must be in (0, 3]")
        if type(self.max_in_flight) is not int or not 1 <= self.max_in_flight <= 4:
            raise ValueError("VLM concurrency must be within 1..4")


@dataclass(frozen=True)
class SourceSample:
    frame: EvidenceFrame
    received_at: str
    received_monotonic: float


class IndependentEventState:
    """Publish current observations; the tracked scene owns person-event lifecycles.

    Subject references are scoped to one report, even when boxes or model-local
    IDs repeat. There is no VLM track history or independent confirmation vote.
    """

    def __init__(self, config):
        self.config = config
        self.sequence = 0

    def report(self, job, sample, verdicts, timing, *, error=None):
        at, sequence = sample.frame.timestamp_seconds, sample.frame.frame_sequence
        message_id = str(uuid4())
        boxes = {sid: box for verdict in verdicts for sid, box in verdict.subject_boxes}
        mapping = {sid: f"vlm-{message_id}-{sid}" for sid in boxes}
        observations = []
        for index, verdict in enumerate(verdicts):
            ids = tuple(
                sorted(mapping[sid] for sid in verdict.subject_ids if sid in mapping)
            )
            grounded = bool(verdict.subject_ids) and len(ids) == len(
                verdict.subject_ids
            )
            evidence_id = f"vlm:{job.request.request_id}:{index}"
            # Acceptance only admits this observation to background association.
            # Confirmation/clear votes belong to the matched tracking IDs there.
            observations.append(
                {
                    "event_label": verdict.event_label,
                    "subject_ids": list(ids),
                    "detected": verdict.detected,
                    "confidence": verdict.confidence,
                    "evidence_id": evidence_id,
                    "accepted": grounded,
                    "grounded": grounded,
                    "grounding_issues": list(verdict.grounding_issues),
                }
            )
        reported = {v.event_label for v in verdicts}
        judgments = []
        for label in self.config.labels:
            if label not in reported:
                continue
            votes = [v for v in verdicts if v.event_label == label]
            assessments = [o for o in observations if o["event_label"] == label]
            observed_detected = (
                True
                if any(v.detected is True for v in votes)
                else False
                if all(v.detected is False for v in votes)
                else None
            )
            observation_name = (
                "positive"
                if observed_detected is True
                else "negative"
                if observed_detected is False
                else "uncertain"
            )
            localization_state = (
                "available"
                if all(o["grounded"] for o in assessments)
                else "partial"
                if any(o["subject_ids"] for o in assessments)
                else "unavailable"
            )
            confirmation_state = (
                "pending"
                if observed_detected is True
                else "not_detected"
                if observed_detected is False
                else "uncertain"
            )
            detected = False if observed_detected is False else None
            judgments.append(
                {
                    "event_label": label,
                    "detected": detected,
                    "observed_detected": observed_detected,
                    "localization_state": localization_state,
                    "confirmation_state": confirmation_state,
                    "grounding_issues": sorted(
                        {issue for vote in votes for issue in vote.grounding_issues}
                    ),
                    "confidence": max(
                        (v.confidence for v in votes if v.confidence is not None),
                        default=None,
                    ),
                    "description": (
                        f"{label}: model observation {observation_name}; "
                        f"confirmation {confirmation_state}; "
                        f"localization {localization_state}"
                    ),
                    "object_ids": [],
                    "event_ids": [],
                }
            )
        with Image.open(io.BytesIO(sample.frame.jpeg_bytes)) as image:
            width, height = image.size
        self.sequence += 1
        return {
            "schema_version": "1",
            "type": "vlm_report",
            "analysis_kind": "vlm_event",
            "stream_session_id": self.config.stream_session_id,
            "source_id": self.config.source_id,
            "generation": self.config.generation,
            "message_id": message_id,
            "observation_id": message_id,
            "subject_id_scope": "observation",
            "description": "Independent VLM temporal report",
            "analysis_complete": not error and reported == set(self.config.labels),
            "localization_complete": bool(observations)
            and all(item["grounded"] for item in observations),
            "request_id": job.request.request_id,
            "sequence": self.sequence,
            "state": "failed" if error else "completed",
            "error_code": error,
            "emitted_at": utc_now(),
            "frame": {
                "source_frame_index": sequence,
                "timestamp_seconds": at,
                "observed_at": sample.received_at,
                "width": width,
                "height": height,
            },
            "source_clock": "source_pts_and_local_branch_ingress",
            "evidence": [
                {
                    "frame_sequence": frame.frame_sequence,
                    "timestamp_seconds": frame.timestamp_seconds,
                }
                for frame in job.frames
            ],
            "timing": {
                **timing,
                "evidence_received_at": sample.received_at,
                "evidence_received_monotonic": sample.received_monotonic,
            },
            "model": {
                "name": PUBLIC_MODEL_NAME,
                "route_id": self.config.route_id,
                "latency_seconds": timing["model_seconds"],
            },
            "judgments": judgments,
            "observations": observations,
            "objects": [
                {
                    "object_id": mapping[sid],
                    "label": "person",
                    "bbox": box.as_dict(),
                    "attributes": {
                        "subject_id": mapping[sid],
                        "identity_status": "unassociated",
                        "observation_state": "observed",
                        "source_timestamp_seconds": at,
                        "source_frame_sequence": sequence,
                    },
                }
                for sid, box in boxes.items()
            ],
        }


def _inspector(config):
    return IndependentVLMInspector(
        config.model_settings,
        model=config.model,
        api_key=config.api_key,
        request_parameters=config.parameters,
        **config.options,
    )


def run_vlm_process(config, incoming, outgoing, stop, inspector_factory=_inspector):
    """Spawn target: image windows, inference, observation reports, then IPC."""
    state = IndependentEventState(config)
    stats = Counter()
    history = deque(maxlen=12)
    inflight = OrderedDict()
    ready = {}
    next_dispatch_order = next_publish_order = 0
    thread_state, clients = local(), []
    last_sequence, last_endpoint, next_due = -1, -1, -math.inf
    last_status = 0.0
    source_finished = False
    executor = ThreadPoolExecutor(
        max_workers=config.max_in_flight, thread_name_prefix="vlm-model"
    )

    def emit(item):
        try:
            outgoing.put_nowait(item)
        except Full:
            stats["output_dropped"] += 1

    def evaluate(job):
        started, started_at = time.monotonic(), utc_now()
        verdicts, error = (), None
        validation_reason = None
        try:
            if not hasattr(thread_state, "inspector"):
                thread_state.inspector = inspector_factory(config)
                clients.append(thread_state.inspector)
            verdicts = thread_state.inspector.inspect(job)
        except Exception as exc:
            # Closed vocabulary only: no endpoints, credentials or model text.
            error = getattr(exc, "code", None)
            if error not in {
                "timeout",
                "connection",
                "http_error",
                "invalid_response",
                "empty_response",
            }:
                error = "inspection_failed"
            reason = getattr(exc, "validation_reason", None)
            if reason in {
                "invalid_subject_bbox",
                "invalid_subject",
                "invalid_verdict",
                "duplicate_verdict",
                "subject_fields",
                "verdict_fields",
                "independent_object_required",
                "independent_capacity",
                "completion_not_finished",
                "duplicate_json_field",
                "nonfinite_json_number",
                "invalid_json_syntax",
                "invalid_json_encoding",
                "json_nesting_limit",
                "response_size_limit",
            }:
                validation_reason = reason
            elif error in {"invalid_response", "empty_response"}:
                validation_reason = "invalid_response_shape"
        return (
            verdicts,
            error,
            {
                "request_started_at": started_at,
                "request_ended_at": utc_now(),
                "request_started_monotonic": started,
                "model_seconds": time.monotonic() - started,
                "validation_reason": validation_reason,
            },
        )

    def metrics():
        active = len(inflight) + len(ready)
        return {
            **stats,
            "active_requests": active,
            "inflight_requests": len(inflight),
            "pending_requests": len(ready),
            "reorder_capacity": REORDER_CAPACITY,
            "reorder_buffer_full": active >= REORDER_CAPACITY,
        }

    try:
        emit(
            {
                "kind": "status",
                "state": "running",
                "pid": os.getpid(),
                "stats": dict(stats),
            }
        )
        while not stop.is_set():
            for _ in range(INPUT_CAPACITY):
                try:
                    sample = incoming.get(timeout=0.01 if _ == 0 else 0)
                except Empty:
                    break
                if sample is None:
                    source_finished = True
                    break
                frame = sample.frame
                if (
                    frame.stream_session_id != config.stream_session_id
                    or frame.frame_sequence <= last_sequence
                ):
                    stats["stale_source_dropped"] += 1
                    continue
                last_sequence = frame.frame_sequence
                if time.monotonic() - sample.received_monotonic > 3.0:
                    stats["expired_source_dropped"] += 1
                    continue
                if (
                    history
                    and frame.timestamp_seconds <= history[-1].frame.timestamp_seconds
                ):
                    stats["source_clock_rejected"] += 1
                    continue
                history.append(sample)
                stats["source_samples"] += 1
                while history and (
                    frame.timestamp_seconds - history[0].frame.timestamp_seconds > 3.0
                    or sum(len(x.frame.jpeg_bytes) for x in history) > 16 * 1024**2
                ):
                    history.popleft()

            # A completed future no longer consumes model-call capacity. Results
            # wait in a separately bounded buffer until every earlier request is
            # available for source-ordered publication.
            for order, (job, sample, future, dispatched) in tuple(inflight.items()):
                if not future.done():
                    continue
                inflight.pop(order)
                ready[order] = (job, sample, future.result(), dispatched)

            while next_publish_order in ready:
                job, sample, result, dispatched = ready.pop(next_publish_order)
                next_publish_order += 1
                verdicts, error, timing = result
                now = time.monotonic()
                timing.update(
                    dispatch_wait_seconds=timing["request_started_monotonic"]
                    - dispatched,
                    source_to_request_seconds=timing["request_started_monotonic"]
                    - sample.received_monotonic,
                    reorder_wait_seconds=max(
                        0.0,
                        now
                        - timing["request_started_monotonic"]
                        - timing["model_seconds"],
                    ),
                )
                report = state.report(job, sample, verdicts, timing, error=error)
                stats["failed_requests" if error else "succeeded_requests"] += 1
                if not error and not report["analysis_complete"]:
                    stats["incomplete_reports"] += 1
                if not error and not verdicts:
                    stats["empty_reports"] += 1
                stats["ungrounded_verdicts"] += sum(
                    not item["grounded"] for item in report["observations"]
                )
                if verdicts and not report["localization_complete"]:
                    stats["unlocalized_reports"] += 1
                if any("invalid_subject_bbox" in v.grounding_issues for v in verdicts):
                    stats["invalid_bbox_reports"] += 1
                report["process_metrics"] = metrics()
                emit({"kind": "report", "report": report})

            active_requests = len(inflight) + len(ready)
            if (
                history
                and len(inflight) < config.max_in_flight
                and active_requests < REORDER_CAPACITY
            ):
                sample = history[-1]
                frame = sample.frame
                window = tuple(x.frame for x in list(history)[-4:])
                if (
                    len(window) >= 2
                    and window[-1].timestamp_seconds - window[0].timestamp_seconds
                    >= 0.75
                    and frame.frame_sequence > last_endpoint
                    and frame.timestamp_seconds + 1e-9 >= next_due
                    and time.monotonic() - sample.received_monotonic <= 3.0
                ):
                    request = InspectionRequest(
                        str(uuid4()),
                        (),
                        "independent_source",
                        config.labels,
                        window[0].timestamp_seconds,
                        frame.timestamp_seconds,
                    )
                    job = TemporalInspection(
                        config.stream_session_id,
                        config.source_id,
                        request,
                        window,
                        event_sensitivities=config.sensitivities,
                        evidence_after_frame_sequence=last_endpoint,
                    )
                    dispatched = time.monotonic()
                    order = next_dispatch_order
                    next_dispatch_order += 1
                    inflight[order] = (
                        job,
                        sample,
                        executor.submit(evaluate, job),
                        dispatched,
                    )
                    last_endpoint = frame.frame_sequence
                    next_due = frame.timestamp_seconds + 1.0 / config.observation_fps
                    stats["claimed_requests"] += 1
            if time.monotonic() - last_status >= 1.0:
                emit(
                    {
                        "kind": "status",
                        "state": "drained"
                        if source_finished and not inflight and not ready
                        else "running",
                        "pid": os.getpid(),
                        "stats": {
                            **metrics(),
                            "retained_frames": len(history),
                            "retained_bytes": sum(
                                len(x.frame.jpeg_bytes) for x in history
                            ),
                        },
                    }
                )
                last_status = time.monotonic()
    except Exception:
        emit(
            {
                "kind": "status",
                "state": "failed",
                "pid": os.getpid(),
                "error_code": "process_failed",
                "stats": dict(stats),
            }
        )
        raise RuntimeError("independent VLM process failed") from None
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.close()
        outgoing.close()
        outgoing.cancel_join_thread()


class IndependentVLMBranch:
    """One immutable run owner; only small branch locks, never a service callback."""

    def __init__(
        self,
        config,
        *,
        results=None,
        connection=None,
        fusion=None,
        camera_callback=None,
        inspector_factory=_inspector,
    ):
        self.config, self.connection, self.fusion = config, connection, fusion
        self.results = ResultBus(128)
        self._shared_results = results
        self._shared_epoch = results.epoch if results is not None else None
        context = mp.get_context("spawn")
        self._incoming = context.Queue(INPUT_CAPACITY)
        self._outgoing = context.Queue(OUTPUT_CAPACITY)
        self._process_stop = context.Event()
        self._stop, self._lock = Event(), Lock()
        self._source_finished = Event()
        self._process = context.Process(
            target=run_vlm_process,
            args=(
                config,
                self._incoming,
                self._outgoing,
                self._process_stop,
                inspector_factory,
            ),
            name="goodvision-vlm",
            daemon=True,
        )
        self._status = {
            "state": "starting",
            "succeeded_requests": 0,
            "failed_requests": 0,
            "last_attempt_state": None,
            "last_attempt_error": None,
            "analysis_complete": False,
            "localization_complete": False,
        }
        self._stats = Counter()
        self._last_sequence = -1
        self._last_source_monotonic = self._last_publish_monotonic = None
        self._last_success_monotonic = None
        self._next_sample = -math.inf
        self._started = False
        self._reader = self._source = self._reaper = None
        self._evidence = OrderedDict()
        self._evidence_bytes = 0
        self._camera_dispatch = (
            CameraSourceDispatch(camera_callback) if camera_callback else None
        )

    def start(self):
        if self._started or self._stop.is_set():
            return
        if self._shared_results is not None:
            self._shared_epoch = self._shared_results.epoch
        self._process.start()
        self._started = True
        self._reader = Thread(
            target=self._read, name="vlm-independent-results", daemon=True
        )
        self._reader.start()
        if self._camera_dispatch is not None:
            self._camera_dispatch.start()
        if self.fusion is not None:
            self.fusion.start()
        if self.connection is not None:
            self._source = Thread(
                target=self._sample_connection, name="vlm-original-source", daemon=True
            )
            self._source.start()

    def offer(self, jpeg, timestamp, sequence):
        if (
            self._stop.is_set()
            or self._source_finished.is_set()
            or not isinstance(jpeg, bytes)
            or not jpeg
            or len(jpeg) > MAX_FRAME_BYTES
            or type(sequence) is not int
            or sequence < 0
            or type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            return False
        now = time.monotonic()
        with self._lock:
            if self._source_finished.is_set() or self._stop.is_set():
                return False
            if sequence <= self._last_sequence or timestamp + 1e-9 < self._next_sample:
                return False
            self._last_sequence = sequence
            self._next_sample = timestamp + 1.0 / max(
                2.0, self.config.observation_fps * 2.0
            )
            sample = SourceSample(
                EvidenceFrame(self.config.stream_session_id, sequence, timestamp, jpeg),
                utc_now(),
                now,
            )
            try:
                self._incoming.put_nowait(sample)
            except Full:
                self._stats["input_backpressure_dropped"] += 1
                return False
            self._stats["admitted_samples"] += 1
            self._last_source_monotonic = now
            self._evidence[sequence] = sample
            self._evidence_bytes += len(jpeg)
            while self._evidence and (
                len(self._evidence) > 64 or self._evidence_bytes > 32 * 1024**2
            ):
                _, old = self._evidence.popitem(last=False)
                self._evidence_bytes -= len(old.frame.jpeg_bytes)
            return True

    def _sample_connection(self):
        cursor = -1
        try:
            while not self._stop.is_set():
                frames = self.connection.display_frames(
                    cursor, max_age_seconds=1.0, limit=32
                )
                for sequence, timestamp, _width, _height, jpeg in frames:
                    self.offer(jpeg, timestamp, sequence)
                    cursor = sequence
                self._stop.wait(0.025)
        except Exception:
            with self._lock:
                self._status.update(
                    source_state="failed", source_error="source_reader_failed"
                )

    def _read(self):
        while not self._stop.is_set():
            try:
                message = self._outgoing.get(timeout=0.1)
            except Empty:
                if not self._process.is_alive():
                    with self._lock:
                        self._status.update(state="failed", error_code="process_exited")
                    return
                continue
            if self._stop.is_set():
                return
            if message["kind"] == "status":
                with self._lock:
                    self._status.update(
                        {
                            key: value
                            for key, value in message.items()
                            if key not in {"kind", "stats"}
                        }
                    )
                    self._status.update(message.get("stats", {}))
                continue
            report = message["report"]
            if (
                report.get("stream_session_id") != self.config.stream_session_id
                or report.get("generation") != self.config.generation
            ):
                continue
            now = time.monotonic()
            report["emitted_at"] = utc_now()
            report["timing"]["published_at"] = report["emitted_at"]
            report["timing"]["source_to_publish_seconds"] = max(
                0.0, now - report["timing"].pop("evidence_received_monotonic")
            )
            report["timing"].pop("request_started_monotonic", None)
            # Independent publication precedes every optional fusion/archive action.
            self.results.publish(report)
            if self._shared_results is not None:
                self._shared_results.publish(report, expected_epoch=self._shared_epoch)
            with self._lock:
                self._status.update(report.get("process_metrics", {}))
                self._status.update(
                    last_result_at=report["emitted_at"],
                    last_attempt_state=report["state"],
                    last_attempt_error=report.get("error_code"),
                    last_attempt_validation_reason=report["timing"].get(
                        "validation_reason"
                    ),
                    analysis_complete=report.get("analysis_complete", False),
                    localization_complete=report.get("localization_complete", False),
                    last_source_to_publish_seconds=report["timing"][
                        "source_to_publish_seconds"
                    ],
                )
                self._stats["published_reports"] += 1
                self._last_publish_monotonic = now
                if report["state"] == "completed":
                    self._last_success_monotonic = now
                    self._status["last_success_at"] = report["emitted_at"]
                evidence = tuple(
                    self._evidence[item["frame_sequence"]].frame
                    for item in report["evidence"]
                    if item["frame_sequence"] in self._evidence
                )
            if self.fusion is not None and report["state"] == "completed":
                self.fusion.offer(report, evidence)
            log_timing(
                "vlm_independent_report",
                stream_session_id=report["stream_session_id"],
                request_id=report["request_id"],
                source_sequence=report["frame"]["source_frame_index"],
                source_timestamp_seconds=report["frame"]["timestamp_seconds"],
                source_to_publish_seconds=report["timing"]["source_to_publish_seconds"],
                state=report["state"],
                error_code=report.get("error_code"),
                validation_reason=report["timing"].get("validation_reason"),
                process_pid=self._process.pid,
            )

    def finish_source(self):
        with self._lock:
            if self._source_finished.is_set():
                return
            self._source_finished.set()

        def send_end():
            while not self._stop.is_set():
                try:
                    self._incoming.put(None, timeout=0.1)
                    return
                except Full:
                    continue

        Thread(target=send_end, name="vlm-source-end", daemon=True).start()

    def wait_until_idle(self, timeout_seconds=0.1):
        self.finish_source()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.stats()["state"] == "drained":
                return True
            self._stop.wait(0.02)
        return self.stats()["state"] == "drained"

    def wake(self):
        """Compatibility with scene observers; no dependency on their wakeups."""

    def finish_camera_sources(self):
        return self._camera_dispatch is None or self._camera_dispatch.finish()

    def dispatch_camera_source(self, jpeg, timestamp, sequence):
        if self._camera_dispatch is None:
            return False
        return self._camera_dispatch.offer(jpeg, timestamp, sequence)

    def health(self):
        """Bounded local counters only; never query fusion, archives or workers."""
        with self._lock:
            result = {
                **self._status,
                **self._stats,
                "pid": self._process.pid,
                "stream_session_id": self.config.stream_session_id,
                "generation": self.config.generation,
                "route_id": self.config.route_id,
                "event_labels": list(self.config.labels),
                "independent": True,
                "last_source_age_seconds": (
                    None
                    if self._last_source_monotonic is None
                    else max(0.0, time.monotonic() - self._last_source_monotonic)
                ),
                "last_result_age_seconds": (
                    None
                    if self._last_publish_monotonic is None
                    else max(0.0, time.monotonic() - self._last_publish_monotonic)
                ),
                "last_success_age_seconds": (
                    None
                    if self._last_success_monotonic is None
                    else max(0.0, time.monotonic() - self._last_success_monotonic)
                ),
                "pending_requests": self._status.get("pending_requests", 0),
                "inflight_requests": self._status.get("inflight_requests", 0),
                "input_capacity": INPUT_CAPACITY,
                "output_capacity": OUTPUT_CAPACITY,
                "reorder_capacity": REORDER_CAPACITY,
                "source_sample_fps": max(2.0, self.config.observation_fps * 2.0),
                "observation_fps": self.config.observation_fps,
            }
        return result

    def stats(self):
        result = self.health()
        if self.fusion is not None:
            result["fusion"] = self.fusion.stats()
        if self._camera_dispatch is not None:
            result["camera_source_dispatch"] = self._camera_dispatch.stats()
        return result

    def close(self, timeout_seconds=0.0):
        if not self._stop.is_set():
            self._stop.set()
            self._process_stop.set()
            if self.fusion is not None:
                self.fusion.close()
            if self._camera_dispatch is not None:
                self._camera_dispatch.close()
            with self._lock:
                self._status["state"] = "stopping"
            self._reaper = Thread(
                target=self._cleanup, name="vlm-process-cleanup", daemon=True
            )
            self._reaper.start()
        if self._reaper is not None and timeout_seconds > 0:
            self._reaper.join(timeout_seconds)

    def _cleanup(self):
        if self._started:
            self._process.join(2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1.0)
        for thread in (self._reader, self._source):
            if thread is not None:
                thread.join(1.0)
        for queue in (self._incoming, self._outgoing):
            queue.close()
            queue.cancel_join_thread()
        with self._lock:
            self._evidence.clear()
            self._evidence_bytes = 0
            self._status["state"] = "stopped"


class CameraSourceDispatch:
    """Do not hold the upload/owner lock while waiting for SAM or scene callbacks.

    A queued picture has not yet been admitted to SAM. Accepted in-flight SAM
    work is untouched; unsent pictures are replaceable and counted explicitly.
    """

    def __init__(self, callback):
        self.callback = callback
        self._lock, self._stop, self._wake = Lock(), Event(), Event()
        self._queue = deque()
        self._finishing = False
        self._active = False
        self._stats = Counter()
        self._thread = None

    def start(self):
        self._thread = Thread(
            target=self._run, name="camera-source-dispatch", daemon=True
        )
        self._thread.start()

    def offer(self, jpeg, timestamp, sequence):
        with self._lock:
            if self._stop.is_set() or self._finishing or len(jpeg) > MAX_FRAME_BYTES:
                return False
            self._queue.append((jpeg, timestamp, sequence))
            while len(self._queue) > 4:
                self._queue.popleft()
                self._stats["unsent_sources_superseded"] += 1
        self._wake.set()
        return True

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                item = self._queue.popleft() if self._queue else None
                self._active = item is not None
            if item is not None:
                try:
                    self.callback(*item)
                    with self._lock:
                        self._stats["callbacks_completed"] += 1
                except Exception:
                    with self._lock:
                        self._stats["callback_failures"] += 1
                finally:
                    with self._lock:
                        self._active = False
            else:
                self._wake.wait(0.1)
                self._wake.clear()

    def finish(self):
        with self._lock:
            self._finishing = True
            return not self._queue and not self._active

    def stats(self):
        with self._lock:
            return {
                **self._stats,
                "queued_sources": len(self._queue),
                "callback_active": self._active,
                "source_finished": self._finishing,
            }

    def close(self):
        self._stop.set()
        self._wake.set()
        with self._lock:
            self._stats["cancelled_sources"] += len(self._queue)
            self._queue.clear()
