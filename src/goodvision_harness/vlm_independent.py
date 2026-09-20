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
from .scene_state import SceneState
from .vlm_independent_model import IndependentVLMInspector

MAX_FRAME_BYTES = 4 * 1024**2
INPUT_CAPACITY = 8
OUTPUT_CAPACITY = 16


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def overlap(a, b):
    intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0.0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return intersection / union if union else 0.0


def unambiguous_matches(boxes, candidates, *, pose_tolerance=False):
    """Mutual unique overlap; ambiguous people never share confirmation votes.

    Nearby VLM windows may describe the same person standing or on the floor.
    Permit a moderate overlap only when both box centers stay inside each other.
    SAM association retains its stricter overlap threshold.
    """

    def score(box, other):
        value = overlap(box, other)
        if pose_tolerance and value >= 0.35:
            bx, by = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
            ox, oy = (other.x1 + other.x2) / 2, (other.y1 + other.y2) / 2
            if not (
                box.x1 <= ox <= box.x2
                and box.y1 <= oy <= box.y2
                and other.x1 <= bx <= other.x2
                and other.y1 <= by <= other.y2
            ):
                return 0.0
        return value

    scores = {
        key: {sid: score(box, other) for sid, other in candidates.items()}
        for key, box in boxes.items()
    }
    minimum = 0.35 if pose_tolerance else 0.5

    def winner(rows):
        ranked = sorted(rows, reverse=True)
        if (
            ranked
            and ranked[0][0] >= minimum
            and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.15)
        ):
            return ranked[0][1]
        return None

    matches = {}
    for key, values in scores.items():
        target = winner((value, sid) for sid, value in values.items())
        if (
            target is not None
            and winner((values[target], other) for other, values in scores.items())
            == key
        ):
            matches[key] = target
    return matches


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
    """Use the existing two-positive/two-negative source evidence rules unchanged."""

    def __init__(self, config):
        self.config = config
        self.scene = SceneState(config.stream_session_id, config.source_id)
        self.subjects = {}
        self.namespace = uuid4().hex[:12]
        self.sequence = 0
        self.subject_sequence = 0

    def report(self, job, sample, verdicts, timing, *, error=None):
        at, sequence = sample.frame.timestamp_seconds, sample.frame.frame_sequence
        self.subjects = {
            sid: item for sid, item in self.subjects.items() if at - item[1] <= 3.0
        }
        boxes = {sid: box for verdict in verdicts for sid, box in verdict.subject_boxes}
        mapping = unambiguous_matches(
            boxes, {sid: item[0] for sid, item in self.subjects.items()}
        )
        nearby = unambiguous_matches(
            boxes,
            {
                sid: item[0]
                for sid, item in self.subjects.items()
                if at - item[1] <= 1.2
            },
            pose_tolerance=True,
        )
        for key, sid in nearby.items():
            if key not in mapping and sid not in mapping.values():
                mapping[key] = sid
        for sid, box in boxes.items():
            if sid not in mapping:
                self.subject_sequence += 1
                mapping[sid] = f"vlm-{self.namespace}-{self.subject_sequence}"
            public_id = mapping[sid]
            self.subjects[public_id] = (box, at)
            self.scene.observe_person(
                public_id, timestamp_seconds=at, frame_sequence=sequence, bbox=box
            )
        observations = []
        for index, verdict in enumerate(verdicts):
            ids = tuple(sorted(mapping[sid] for sid in verdict.subject_ids))
            evidence_id = f"vlm:{job.request.request_id}:{index}"
            # An unlocalized negative may coexist with a valid positive in a
            # model response. Retain it for audit, but it cannot clear a person
            # event, count as evaluated coverage or cancel the grounded positive.
            accepted = bool(ids) and self.scene.observe_event(
                verdict.event_label,
                ids,
                timestamp_seconds=at,
                frame_sequence=sequence,
                detected=verdict.detected,
                confidence=verdict.confidence,
                evidence_id=evidence_id,
            )
            observations.append(
                {
                    "event_label": verdict.event_label,
                    "subject_ids": list(ids),
                    "detected": verdict.detected,
                    "confidence": verdict.confidence,
                    "evidence_id": evidence_id,
                    "accepted": accepted,
                    "grounded": bool(ids),
                }
            )
        self.scene.advance(at)
        snapshot = self.scene.snapshot(include_history=False)
        events = [
            {**event, "alert": event["phase"] == "confirmed"}
            for event in snapshot["events"]
        ]
        reported = {v.event_label for v in verdicts if v.subject_ids}
        judgments = []
        for label in self.config.labels:
            if label not in reported:
                continue
            relevant = [event for event in events if event["event_label"] == label]
            confirmed = [event for event in relevant if event["phase"] == "confirmed"]
            votes = [v for v in verdicts if v.event_label == label and v.subject_ids]
            detected = (
                True
                if confirmed
                else (False if all(v.detected is False for v in votes) else None)
            )
            judgments.append(
                {
                    "event_label": label,
                    "detected": detected,
                    "confidence": max(
                        (v.confidence or 0.0 for v in votes), default=0.0
                    ),
                    "description": f"{label}: independent temporal observation",
                    "object_ids": sorted(
                        {sid for e in confirmed for sid in e["subject_ids"]}
                    ),
                    "event_ids": [e["event_id"] for e in relevant],
                }
            )
        with Image.open(io.BytesIO(sample.frame.jpeg_bytes)) as image:
            width, height = image.size
        self.sequence += 1
        message_id = str(uuid4())
        return {
            "schema_version": "1",
            "type": "vlm_report",
            "analysis_kind": "vlm_event",
            "stream_session_id": self.config.stream_session_id,
            "source_id": self.config.source_id,
            "generation": self.config.generation,
            "message_id": message_id,
            "observation_id": message_id,
            "description": "Independent VLM temporal report",
            "analysis_complete": not error and reported == set(self.config.labels),
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
                "latency_seconds": timing["model_seconds"],
            },
            "judgments": judgments,
            "observations": observations,
            "events": events,
            "missing_event_labels": [
                label for label in self.config.labels if label not in reported
            ],
            "objects": [
                {
                    "object_id": person["subject_id"],
                    "label": "person",
                    "bbox": person["bbox"],
                    "attributes": {
                        "subject_id": person["subject_id"],
                        "identity_status": "unassociated",
                        "observation_state": person["observation_state"],
                        "source_timestamp_seconds": person[
                            "last_observed_timestamp_seconds"
                        ],
                        "source_frame_sequence": person["frame_sequence"],
                    },
                }
                for person in snapshot["entities"]
            ],
            "association": {"state": "independent", "results_url": "/v1/scene"},
            "confirmation": {
                "positive_observations": self.scene.confirm_observations,
                "negative_observations": self.scene.clear_observations,
            },
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
    """Spawn target: image windows, inference, evidence confirmation, then IPC."""
    state = IndependentEventState(config)
    stats = Counter()
    history = deque(maxlen=12)
    pending = OrderedDict()
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
        return (
            verdicts,
            error,
            {
                "request_started_at": started_at,
                "request_ended_at": utc_now(),
                "request_started_monotonic": started,
                "model_seconds": time.monotonic() - started,
            },
        )

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
            while pending:
                key, (job, sample, future, dispatched) = next(iter(pending.items()))
                if not future.done():
                    break
                pending.pop(key)
                verdicts, error, timing = future.result()
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
                stats["ungrounded_verdicts"] += sum(not v.subject_ids for v in verdicts)
                report["process_metrics"] = dict(stats)
                emit({"kind": "report", "report": report})
            if history and len(pending) < config.max_in_flight:
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
                    pending[request.request_id] = (
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
                        if source_finished and not pending
                        else "running",
                        "pid": os.getpid(),
                        "stats": {
                            **stats,
                            "active_requests": len(pending),
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
        }
        self._stats = Counter()
        self._last_sequence = -1
        self._last_source_monotonic = self._last_publish_monotonic = None
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
                    last_source_to_publish_seconds=report["timing"][
                        "source_to_publish_seconds"
                    ],
                )
                self._stats["published_reports"] += 1
                self._last_publish_monotonic = now
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

    def stats(self):
        with self._lock:
            result = {
                **self._status,
                **self._stats,
                "pid": self._process.pid,
                "stream_session_id": self.config.stream_session_id,
                "generation": self.config.generation,
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
                "input_capacity": INPUT_CAPACITY,
                "output_capacity": OUTPUT_CAPACITY,
                "source_sample_fps": max(2.0, self.config.observation_fps * 2.0),
                "observation_fps": self.config.observation_fps,
            }
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
