"""Independent source admission, spawned VLM process and immediate result delivery.

No service, SAM, scene, archive or identity lock is reachable from the child or
the output reader. Fusion has its own bounded consumer after publication.
"""

from __future__ import annotations

import copy
import io
import math
import multiprocessing as mp
import os
import time
from bisect import bisect_left
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from queue import Empty, Full
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread, local
from uuid import uuid4

from PIL import Image

from .contracts import PUBLIC_MODEL_NAME
from .independent_diagnostics import (
    DIAGNOSTIC_QUEUE_CAPACITY,
    IndependentDiagnostics,
    IndependentRequestDiagnostics,
)
from .latency_trace import configure_worker_timing, log_timing
from .result_bus import ResultBus
from .runtime.scene import TemporalInspection
from .runtime.scheduling import EvidenceFrame, InspectionRequest
from .vlm_admission import AdmissionError, AdmissionSettings, SharedAdmission
from .vlm_event_policy import EVENT_POLICY_PROFILES, event_policy_snapshot
from .vlm_independent_model import IndependentVLMInspector

MAX_FRAME_BYTES = 4 * 1024**2
INPUT_CAPACITY = 8
OUTPUT_CAPACITY = 16
REORDER_CAPACITY = 32  # Per lane, including in-flight and completed requests.
INFORMATION_HTTP_DEADLINE_SECONDS = 5.0


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
    output_mode: str = "parallel"
    diagnostics_enabled: bool = False
    admission: AdmissionSettings | None = None
    event_policy: str = "main-baseline"
    retain_historical_results: bool = False
    event_definitions: tuple[tuple[str, str], ...] = ()
    candidate_fps: float | None = None
    observation_window_frames: int = 4
    observation_frame_delta_seconds: float | None = None
    videojev_question_layout: str = "repeated"
    videojev_prompt_style: str = "full"

    def __post_init__(self):
        if self.output_mode not in {"full", "compact", "parallel", "events"}:
            raise ValueError("invalid VLM output mode")
        if self.videojev_question_layout not in {
            "repeated", "shared", "prefix", "prefix-cache", "prefix-cache-aligned",
        }:
            raise ValueError("invalid VideoJev question layout")
        if self.videojev_prompt_style not in {"full", "scene", "brief"}:
            raise ValueError("invalid VideoJev prompt style")
        if not self.stream_session_id or not self.source_id or not self.labels:
            raise ValueError("VLM process requires a scoped source and event labels")
        if self.event_policy not in EVENT_POLICY_PROFILES:
            raise ValueError("invalid event policy profile")
        if type(self.retain_historical_results) is not bool:
            raise ValueError("historical retention must be boolean")
        if (
            not isinstance(self.event_definitions, tuple)
            or len(self.event_definitions) > 64
            or any(
                not isinstance(item, tuple) or len(item) != 2
                or not isinstance(item[0], str) or item[0] not in self.labels
                or not isinstance(item[1], str) or not item[1].strip()
                or len(item[1]) > 4096
                for item in self.event_definitions
            )
        ):
            raise ValueError("invalid fixed event definitions")
        defined = {label for label, _ in self.event_definitions}
        if (len(defined) != len(self.event_definitions)
                or self.event_definitions and defined != set(self.labels)):
            raise ValueError("fixed definitions must uniquely cover the requested events")
        if not 0 < self.observation_fps <= 3 or not math.isfinite(self.observation_fps):
            raise ValueError("VLM cadence must be in (0, 3]")
        if self.candidate_fps is not None and (
            type(self.candidate_fps) not in (int, float)
            or not 1.0 / 3600 <= self.candidate_fps <= 120
            or not math.isfinite(self.candidate_fps)
        ):
            raise ValueError("candidate sampling FPS must be within 1/3600..120")
        if type(self.observation_window_frames) is not int or not (
            2 <= self.observation_window_frames <= 64
        ):
            raise ValueError("observation window must contain 2..64 source frames")
        if self.observation_frame_delta_seconds is not None and (
            type(self.observation_frame_delta_seconds) not in (int, float)
            or not 0 < self.observation_frame_delta_seconds <= 3600
            or not math.isfinite(self.observation_frame_delta_seconds)
        ):
            raise ValueError("observation frame interval must be in (0, 3600]")
        if type(self.max_in_flight) is not int or not 1 <= self.max_in_flight <= 4:
            raise ValueError("VLM concurrency must be within 1..4")


@dataclass(frozen=True)
class SourceSample:
    frame: EvidenceFrame
    received_at: str
    received_monotonic: float


def _source_sample_fps(config):
    return (
        config.candidate_fps if config.candidate_fps is not None
        else max(2.0, config.observation_fps * 2.0)
    )


def _observation_history_limits(config):
    if (
        config.candidate_fps is None
        and config.observation_window_frames == 4
        and config.observation_frame_delta_seconds is None
    ):
        return 12, 3.0
    fps = _source_sample_fps(config)
    interval = 1.0 / fps
    delta = config.observation_frame_delta_seconds
    span = (config.observation_window_frames - 1) * (
        interval if delta is None else delta
    )
    seconds = max(3.0, span + max(interval, delta or interval))
    capacity = max(12, config.observation_window_frames, math.ceil(seconds * fps) + 2)
    return capacity, seconds


def _observation_window(
    history, frame_count=4, frame_delta_seconds=None, candidate_fps=None,
):
    """Select distinct arrived originals, retaining the latest source endpoint."""
    samples = tuple(history)
    if frame_delta_seconds is not None:
        if len(samples) < frame_count:
            return ()
        timestamps = tuple(sample.frame.timestamp_seconds for sample in samples)
        endpoint = timestamps[-1]
        if endpoint - (frame_count - 1) * frame_delta_seconds + 1e-9 < timestamps[0]:
            return ()
        intervals = sorted(
            newer.frame.timestamp_seconds - older.frame.timestamp_seconds
            for older, newer in zip(samples, samples[1:])
        )
        middle = len(intervals) // 2
        observed_interval = (
            intervals[middle] if len(intervals) % 2
            else (intervals[middle - 1] + intervals[middle]) / 2.0
        )
        sampling_interval = (
            observed_interval if candidate_fps is None
            else max(1.0 / candidate_fps, observed_interval)
        )
        tolerance = min(frame_delta_seconds / 2.0, sampling_interval / 2.0) + 1e-9
        selected = [samples[-1].frame]
        available_end = len(samples) - 1
        for offset in range(1, frame_count):
            if not available_end:
                return ()
            target = endpoint - offset * frame_delta_seconds
            insertion = bisect_left(timestamps, target, 0, available_end)
            candidates = {
                max(0, insertion - 1), min(available_end - 1, insertion),
            }
            index = min(candidates, key=lambda i: (abs(timestamps[i] - target), -i))
            frame = samples[index].frame
            if abs(frame.timestamp_seconds - target) > tolerance:
                return ()
            selected.append(frame)
            available_end = index
        return tuple(reversed(selected))
    window = tuple(sample.frame for sample in samples[-frame_count:])
    if (
        len(window) == frame_count
        and window[-1].timestamp_seconds - window[0].timestamp_seconds < 0.75
    ):
        # Dense sampling can put all four recent frames inside 0.75 seconds.
        # Replace only the oldest image, preserving the latest endpoint and
        # image budget without changing windows at the default sampling rate.
        for sample in reversed(samples[:-frame_count]):
            if window[-1].timestamp_seconds - sample.frame.timestamp_seconds >= 0.75:
                return (sample.frame, *window[1:])
    return window


def _native_event_policy(config, job):
    if not job.event_definitions:
        return event_policy_snapshot(
            job.request.event_labels, dict(job.event_sensitivities),
            profile=config.event_policy,
        )
    return {
        "version": "fixed-evaluation-events-v1",
        "profile": "fixed-evaluation",
        "event_labels": list(job.request.event_labels),
        "definitions": dict(job.event_definitions),
        "comparison_field": "observed_detected",
    }


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
            grounded = verdict.event_label == "ground_litter" or (
                bool(verdict.subject_ids) and len(ids) == len(verdict.subject_ids)
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
            if self.config.output_mode == "events":
                # Scene-level observations do not wait for a SAM identity.
                # They are not confirmed person-specific event lifecycles.
                detected = observed_detected
                confirmation_state = "scene_observation"
                localization_state = "not_requested"
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
            **({"event_policy": _native_event_policy(self.config, job)} if (
                self.config.model_settings.provider == "local"
                and self.config.model == "ZhuYuhan/VideoJev-4B"
            ) else {}),
            "event_scope": "scene"
            if self.config.output_mode == "events"
            else "observation",
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
                "name": (
                    self.config.model
                    if self.config.model_settings.provider == "local"
                    and self.config.model == "ZhuYuhan/VideoJev-4B"
                    else PUBLIC_MODEL_NAME
                ),
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
    if (
        config.model_settings.provider == "local"
        and config.model == "ZhuYuhan/VideoJev-4B"
    ):
        from .videojev_model import VideoJevInspector

        return VideoJevInspector(config)
    from .vlm_compact_model import CompactGroundingInspector, EventJSONInspector

    cls = {"events": EventJSONInspector, "compact": CompactGroundingInspector}.get(
        config.output_mode, IndependentVLMInspector
    )
    options = dict(config.options)
    settings = config.model_settings
    if config.output_mode == "events":
        options["max_tokens"] = 160
        # The transport uses asyncio.wait_for: expiry cancels the HTTP request
        # and releases its worker, rather than abandoning a running Future.
        settings = replace(
            settings,
            timeout_seconds=min(
                settings.timeout_seconds,
                INFORMATION_HTTP_DEADLINE_SECONDS,
            ),
        )
    return cls(
        settings,
        model=config.model,
        api_key=config.api_key,
        request_parameters=config.parameters,
        **options,
    )


def run_vlm_process(
    config,
    incoming,
    outgoing,
    stop,
    inspector_factory=_inspector,
    grounding_required=None,
    startup_ready=None,
    diagnostic_outgoing=None,
):
    """Spawn target: image windows, inference, observation reports, then IPC."""
    if mp.current_process().name != "MainProcess":
        configure_worker_timing()
    state = IndependentEventState(config)
    admission = SharedAdmission(config.admission) if config.admission else None
    stats = Counter()
    history_capacity, history_seconds = _observation_history_limits(config)
    history = deque(maxlen=history_capacity)
    lanes = (
        ("events", "compact")
        if config.output_mode == "parallel"
        else (config.output_mode,)
    )
    queues = {lane: OrderedDict() for lane in lanes}
    accounting = {lane: Counter(submitted=0, finished=0, published=0) for lane in lanes}
    ready = {lane: {} for lane in lanes}
    next_dispatch_order = {lane: 0 for lane in lanes}
    next_publish_order = {lane: 0 for lane in lanes}
    next_report_sequence = 1
    last_published = {lane: -1 for lane in lanes}
    last_lane_endpoint = {lane: -1 for lane in lanes}
    thread_state, clients = local(), []
    last_sequence, last_endpoint, next_due = -1, -1, -math.inf
    pending_event_window = False
    last_status = 0.0
    source_finished = False
    executors = {
        lane: ThreadPoolExecutor(
            max_workers=config.max_in_flight, thread_name_prefix="vlm-" + lane
        )
        for lane in lanes
    }

    def emit_status(item):
        # Status is replaceable; reports retain their slot until delivery.
        try:
            outgoing.put_nowait(item)
        except Full:
            stats["output_dropped"] += 1

    def worker_inspector(lane):
        if not hasattr(thread_state, "inspector"):
            thread_state.inspector = inspector_factory(
                replace(config, output_mode=lane)
            )
            clients.append(thread_state.inspector)
        return thread_state.inspector

    warm_barrier = Barrier(len(lanes) * config.max_in_flight)

    def prepare_worker(lane):
        # Occupy every worker before any prime task can finish. Otherwise a
        # fast HEAD could prime one idle thread repeatedly and leave others cold.
        try:
            warm_barrier.wait(timeout=5.0)
        except BrokenBarrierError:
            return False
        if stop.is_set():
            return False
        try:
            warmup = getattr(worker_inspector(lane), "warmup", None)
            return bool(warmup()) if callable(warmup) else None
        except Exception:
            # A failed best-effort transport probe must not disable inference.
            return False

    def evaluate(job, lane, sample):
        submitted = time.monotonic()
        started, started_at = submitted, utc_now()
        model_seconds = 0.0
        admission_wait = 0.0
        verdicts, error = (), None
        validation_reason = None
        http_status = None
        response_diagnostics = {}
        shape = {}
        inspector = None
        inspection_attempted = False
        diagnostic = None
        failure = None
        if config.diagnostics_enabled and diagnostic_outgoing is not None:
            try:
                diagnostic = IndependentRequestDiagnostics(
                    job, lane, sample, config.generation)
                job = replace(job, diagnostics=diagnostic)
            except Exception:
                pass  # Optional diagnostics cannot prevent inference.
        try:
            # Capacity belongs to the source, not each independent process/lane.
            gate = (admission.slot(
                config.stream_session_id, stop=stop,
                source_deadline=sample.received_monotonic + 3.0,
                lane=lane,
            ) if admission else nullcontext(0.0))
            with gate as admission_wait:
                started, started_at = time.monotonic(), utc_now()
                try:
                    inspector = worker_inspector(lane)
                    inspection_attempted = True
                    verdicts = inspector.inspect(job)
                    shape = getattr(inspector, "last_response_diagnostics", {})
                finally:
                    model_seconds = time.monotonic() - started
        except AdmissionError as exc:
            failure = exc
            error = exc.code
            admission_wait = time.monotonic() - submitted
            started, started_at = time.monotonic(), utc_now()
        except Exception as exc:
            failure = exc
            # Closed vocabulary only: no endpoints, credentials or model text.
            error = getattr(exc, "code", None)
            status = getattr(exc, "status_code", None)
            if type(status) is int and 100 <= status <= 599:
                http_status = status
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
            shape = getattr(exc, "response_diagnostics", None)
        if isinstance(shape, dict):
            for key in ("choices_count", "content_characters", "content_parts"):
                value = shape.get(key)
                if type(value) is int and 0 <= value <= 10**9:
                    response_diagnostics[key] = value
            for key, allowed in {
                "model_match": {"matched", "mismatch", "unreported"},
                "provider_category": {
                    "Darkbloom", "DeepInfra", "Parasail", "Alibaba",
                    "Alibaba Cloud Int.", "AtlasCloud", "SiliconFlow",
                    "Venice", "other",
                },
                "finish_reason": {
                    "stop", "length", "content_filter", "tool_calls",
                    "function_call", "missing", "other",
                },
                "content_kind": {"text", "parts", "missing", "other"},
                "content_wrapper": {"code_fence", "none"},
                "native_service_error": {
                    "cuda_shared_failure", "cuda_recovery_failed", "result_log_failure",
                    "gpu_worker_stopped", "cpu_worker_stopped", "invalid_response",
                    "preprocess_failed", "window_failed", "cuda_out_of_memory",
                    "queue_full", "superseded", "expired_before_preprocess",
                    "expired_before_gpu", "client_timeout_before_start",
                    "request_deadline", "service_unavailable", "service_shutdown",
                    "cancelled_before_start",
                },
            }.items():
                value = shape.get(key)
                if isinstance(value, str) and value in allowed:
                    response_diagnostics[key] = value
        # Only named numeric metrics cross the process boundary. Exclude all
        # request/response values, URLs, headers and provider exception text.
        http_timing = {}
        measurements = getattr(inspector, "last_http_timing", {})
        for name in (
            "request_build_seconds", "tcp_connect_seconds",
            "tls_handshake_seconds", "send_request_headers_seconds",
            "send_request_body_seconds", "receive_response_headers_seconds",
            "http_until_headers_seconds", "upload_to_first_non_whitespace_seconds",
            "response_read_seconds", "response_json_decode_seconds", "total_seconds",
            "request_bytes", "response_bytes", "encoded_image_bytes",
            "prompt_tokens", "completion_tokens", "cached_prompt_tokens",
            "reasoning_tokens",
        ):
            value = measurements.get(name) if isinstance(measurements, dict) else None
            if (
                type(value) in (int, float)
                and math.isfinite(value) and 0 <= value <= 10**9
            ):
                http_timing[name] = value
        for name in (
            "videojev_preprocess_seconds", "videojev_visual_encode_seconds",
            "videojev_decision_seconds", "videojev_model_pipeline_seconds",
            "videojev_server_queue_seconds", "videojev_peak_allocated_bytes",
            "videojev_peak_reserved_bytes", "videojev_input_frame_count",
            "videojev_internal_temporal_padding_frames", "videojev_visual_tokens",
            "videojev_native_temporal_fps", "videojev_gpu_pipeline_seconds",
            "videojev_context", "videojev_max_pixels", "videojev_source_width",
            "videojev_source_height", "videojev_processed_width",
            "videojev_processed_height", "videojev_cpu_queue_seconds",
            "videojev_gpu_queue_seconds", "videojev_server_preprocess_seconds",
            "videojev_server_gpu_seconds", "videojev_server_elapsed_seconds",
            "videojev_processor_max_pixels",
            "videojev_native_group_count", "videojev_native_input_tokens",
            "videojev_question_group_prepare_seconds", "videojev_gpu_scoring_seconds",
        ):
            value = measurements.get(name) if isinstance(measurements, dict) else None
            if (
                type(value) in (int, float)
                and value >= 0
                and (type(value) is int or math.isfinite(value))
            ):
                http_timing[name] = value
        http_attempted = (
            measurements.get("http_request_attempted")
            if isinstance(measurements, dict) else None
        )
        if not inspection_attempted:
            http_attempted = False
        elif type(http_attempted) is not bool:
            http_attempted = None
        inspection_timing = {}
        measurements = getattr(inspector, "last_inspection_timing", {})
        for name in (
            "request_prepare_seconds", "verdict_parse_seconds",
            "client_initialize_seconds",
            "server_request_prepare_seconds", "server_verdict_parse_seconds",
        ):
            value = measurements.get(name) if isinstance(measurements, dict) else None
            if (
                type(value) in (int, float)
                and math.isfinite(value) and 0 <= value <= 3600
            ):
                inspection_timing[name] = value
        if diagnostic is not None:
            try:
                diagnostic.publish(
                    diagnostic_outgoing, verdicts=verdicts, error=failure
                )
            except Exception:
                pass  # Separate bounded diagnostics never gate normal reports.
        completed = time.monotonic()
        return (
            verdicts,
            error,
            {
                "request_started_at": started_at,
                "request_ended_at": utc_now(),
                "request_started_monotonic": started,
                "model_seconds": model_seconds,
                "admission_wait_seconds": admission_wait,
                "validation_reason": validation_reason,
                "http_status": http_status,
                "response_diagnostics": response_diagnostics,
                "http_timing": http_timing,
                "inspection_timing": inspection_timing,
                "inspection_attempted": inspection_attempted,
                "http_request_attempted": http_attempted,
                "completion_monotonic": completed,
                "source_to_completion_seconds": max(
                    0.0, completed - sample.received_monotonic
                ),
            },
        )

    def metrics():
        lane_counts = {
            lane: {
                "inflight_requests": len(queues[lane]),
                "pending_requests": len(ready[lane]),
                "active_requests": len(queues[lane]) + len(ready[lane]),
            }
            for lane in lanes
        }
        return {
            **stats,
            **{
                key: sum(counts[key] for counts in lane_counts.values())
                for key in ("active_requests", "inflight_requests", "pending_requests")
            },
            "reorder_capacity": REORDER_CAPACITY * len(lanes),
            "pending_observation_windows": int(pending_event_window),
            "reorder_capacity_per_lane": REORDER_CAPACITY,
            "reorder_buffer_full": all(
                counts["active_requests"] >= REORDER_CAPACITY
                for counts in lane_counts.values()
            ),
            "lanes": lane_counts,
            "request_accounting": {
                "events" if lane == "events" else "grounding": dict(counts)
                for lane, counts in accounting.items()
            },
        }

    try:
        emit_status({"kind": "status", "state": "warming", "pid": os.getpid()})
        warm_started = time.monotonic()
        primes = [
            executors[lane].submit(prepare_worker, lane)
            for lane in lanes
            for _ in range(config.max_in_flight)
        ]
        outcomes = [future.result() for future in primes]
        stats["transport_warmup_slots"] = len(outcomes)
        stats["transport_warmup_ready"] = sum(x is True for x in outcomes)
        stats["transport_warmup_failed"] = sum(x is False for x in outcomes)
        log_timing(
            "vlm_independent_warmup_timing",
            stream_session_id=config.stream_session_id,
            slots=len(outcomes),
            ready=stats["transport_warmup_ready"],
            failed=stats["transport_warmup_failed"],
            total_seconds=time.monotonic() - warm_started,
        )
        if startup_ready is not None:
            startup_ready.set()
        emit_status(
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
                    frame.timestamp_seconds - history[0].frame.timestamp_seconds
                    > history_seconds
                    or sum(len(x.frame.jpeg_bytes) for x in history) > 16 * 1024**2
                ):
                    history.popleft()
            # Collect every completed call before publishing. Waiting for an
            # earlier result or IPC capacity must not occupy an inference slot.
            for lane, pending in queues.items():
                for order, (job, sample, future, dispatched) in tuple(pending.items()):
                    if not future.done():
                        continue
                    pending.pop(order)
                    result = future.result()
                    accounting[lane]["finished"] += 1
                    accounting[lane]["inspection_attempted"] += int(
                        result[2]["inspection_attempted"]
                    )
                    http_attempted = result[2]["http_request_attempted"]
                    accounting[lane][
                        "http_attempt_unknown" if http_attempted is None
                        else "http_attempted" if http_attempted
                        else "http_not_attempted"
                    ] += 1
                    if lane == "events" and result[1] == "timeout":
                        stats["information_http_timeouts"] += 1
                    ready[lane][order] = (job, sample, result, dispatched, None)

            for lane in lanes:
                while ready[lane] and not stop.is_set():
                    # Information bypasses unfinished older calls. Grounding/full
                    # retain dispatch order independently within their own lane.
                    order = (min(ready[lane]) if lane == "events"
                             else next_publish_order[lane])
                    if order not in ready[lane]:
                        break
                    job, sample, result, dispatched, report = ready[lane][order]
                    verdicts, error, timing = result
                    historical = (
                        lane == "events"
                        and sample.frame.frame_sequence <= last_published[lane]
                    )
                    if (historical and config.output_mode != "events"
                            and not config.retain_historical_results):
                        ready[lane].pop(order)
                        stats["late_information_reports_discarded"] += 1
                        log_timing(
                            "vlm_information_report_discarded",
                            stream_session_id=config.stream_session_id,
                            request_id=job.request.request_id,
                            source_sequence=sample.frame.frame_sequence,
                            reason="superseded",
                            error_code=error,
                            model_seconds=timing["model_seconds"],
                        )
                        continue
                    if report is None:
                        now = time.monotonic()
                        timing.update(
                            dispatch_wait_seconds=timing["request_started_monotonic"]
                            - dispatched,
                            source_to_request_seconds=timing["request_started_monotonic"]
                            - sample.received_monotonic,
                            reorder_wait_seconds=max(
                                0.0,
                                now - timing["completion_monotonic"],
                            ),
                        )
                        report = state.report(
                            job, sample, verdicts, timing, error=error,
                        )
                        report["historical_only"] = historical
                        if historical:
                            stats["late_information_reports_preserved"] += 1
                        stats["failed_requests" if error else "succeeded_requests"] += 1
                        if error and error.startswith("admission_"):
                            stats[error] += 1
                        if timing.get("http_status") is not None:
                            stats[f"http_status_{timing['http_status']}"] += 1
                        if not error and not report["analysis_complete"]:
                            stats["incomplete_reports"] += 1
                        if not error and not verdicts:
                            stats["empty_reports"] += 1
                        stats["ungrounded_verdicts"] += sum(
                            not item["grounded"] for item in report["observations"]
                        )
                        if verdicts and not report["localization_complete"]:
                            stats["unlocalized_reports"] += 1
                        if any(
                            "invalid_subject_bbox" in v.grounding_issues
                            for v in verdicts
                        ):
                            stats["invalid_bbox_reports"] += 1
                        if config.output_mode != "full":
                            report["vlm_lane"] = (
                                "information" if lane == "events" else
                                "grounding" if config.output_mode == "parallel"
                                else "combined"
                            )
                            report["window_id"] = job.request.request_id
                            report["output_mode"] = lane
                        # Retry this exact report without rebuilding its ID or
                        # counting the result a second time under backpressure.
                        ready[lane][order] = (job, sample, result, dispatched, report)
                    report["process_metrics"] = metrics()
                    # A report prepared in a blocked lane may be overtaken by
                    # the other lane; sequence describes actual IPC publication.
                    report["sequence"] = next_report_sequence
                    try:
                        outgoing.put_nowait({"kind": "report", "report": report})
                    except Full:
                        stats["output_backpressure"] += 1
                        break
                    next_report_sequence += 1
                    accounting[lane]["published"] += 1
                    ready[lane].pop(order)
                    last_published[lane] = max(
                        last_published[lane], sample.frame.frame_sequence
                    )
                    if lane != "events":
                        next_publish_order[lane] += 1

            available_lanes = [
                lane for lane in lanes
                if len(queues[lane]) < config.max_in_flight
                and len(queues[lane]) + len(ready[lane]) < REORDER_CAPACITY
            ]
            if history:
                sample = history[-1]
                frame = sample.frame
                window = _observation_window(
                    history, config.observation_window_frames,
                    config.observation_frame_delta_seconds,
                    _source_sample_fps(config),
                )
                window_ready = (
                    len(window) == config.observation_window_frames
                    if config.observation_frame_delta_seconds is not None
                    else len(window) >= 2
                    and window[-1].timestamp_seconds - window[0].timestamp_seconds
                    >= 0.75
                )
                if (
                    window_ready
                    and frame.frame_sequence > last_endpoint
                    and (
                        frame.timestamp_seconds + 1e-9 >= next_due
                        or pending_event_window
                    )
                    and time.monotonic() - sample.received_monotonic <= 3.0
                ):
                    newly_due = frame.timestamp_seconds + 1e-9 >= next_due
                    if config.output_mode == "events" and newly_due:
                        period = 1.0 / config.observation_fps
                        if not math.isfinite(next_due):
                            due_slots = 1
                            next_due = frame.timestamp_seconds
                        else:
                            due_slots = 1 + max(
                                0,
                                math.floor(
                                    (frame.timestamp_seconds - next_due + 1e-9) / period
                                ),
                            )
                        stats["scheduled_observation_windows"] += due_slots
                        stats["cadence_windows_missed"] += (
                            due_slots - 1 + int(pending_event_window)
                        )
                        next_due += due_slots * period
                        pending_event_window = True
                    if not available_lanes:
                        if config.output_mode == "events" and newly_due:
                            stats["events_busy_windows_deferred"] += 1
                    else:
                        pending_event_window = False
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
                            grounding_required=bool(
                                grounding_required and grounding_required.is_set()
                            ),
                            event_definitions=config.event_definitions,
                        )
                        dispatched = time.monotonic()
                        for lane in lanes:
                            if lane not in available_lanes:
                                stats[lane + "_busy_windows_skipped"] += 1
                                continue
                            lane_job = replace(
                                job,
                                evidence_after_frame_sequence=last_lane_endpoint[lane],
                            )
                            order = next_dispatch_order[lane]
                            next_dispatch_order[lane] += 1
                            queues[lane][order] = (
                                lane_job,
                                sample,
                                executors[lane].submit(
                                    evaluate, lane_job, lane, sample
                                ),
                                dispatched,
                            )
                            accounting[lane]["submitted"] += 1
                            last_lane_endpoint[lane] = frame.frame_sequence
                        last_endpoint = frame.frame_sequence
                        if config.output_mode != "events":
                            next_due = (
                                frame.timestamp_seconds + 1.0 / config.observation_fps
                            )
                        stats["claimed_requests"] += len(available_lanes)
            if (
                pending_event_window
                and history
                and time.monotonic() - history[-1].received_monotonic > 3.0
            ):
                pending_event_window = False
                stats["expired_waiting_windows"] += 1
            if time.monotonic() - last_status >= 1.0:
                emit_status(
                    {
                        "kind": "status",
                        "state": "drained"
                        if source_finished
                        and not pending_event_window
                        and not any(queues.values())
                        and not any(ready.values())
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
        emit_status(
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
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.close()
        outgoing.close()
        outgoing.cancel_join_thread()
        if diagnostic_outgoing is not None:
            diagnostic_outgoing.close()
            diagnostic_outgoing.cancel_join_thread()


class IndependentVLMBranch:
    """One immutable run owner; only small branch locks, never a service callback."""

    def __init__(
        self,
        config,
        *,
        results=None,
        connection=None,
        fusion=None,
        display_fusion=None,
        delivery=None,
        camera_callback=None,
        inspector_factory=_inspector,
        report_observer=None,
        rejected_report_observer=None,
        publication_observer=None,
    ):
        if report_observer is not None and rejected_report_observer is None:
            raise ValueError("evaluation recording requires a rejected-report observer")
        if publication_observer is not None and report_observer is None:
            raise ValueError("publication timing requires a report observer")
        self.config, self.connection, self.fusion = config, connection, fusion
        self._display_fusion = display_fusion
        self._delivery = delivery
        self._report_observer = report_observer
        self._rejected_report_observer = rejected_report_observer
        self._publication_observer = publication_observer
        self.results = ResultBus(128)
        self.grounding_results = ResultBus(128)
        self._shared_results = results
        self._shared_epoch = results.epoch if results is not None else None
        context = mp.get_context("spawn")
        self._incoming = context.Queue(INPUT_CAPACITY)
        self._outgoing = context.Queue(OUTPUT_CAPACITY)
        self._diagnostic_outgoing = None
        self.diagnostics = None
        if config.diagnostics_enabled:
            try:
                self.diagnostics = IndependentDiagnostics(
                    config.stream_session_id, config.generation)
                self._diagnostic_outgoing = context.Queue(DIAGNOSTIC_QUEUE_CAPACITY)
            except Exception:
                self.diagnostics = None  # Optional telemetry cannot prevent startup.
        self._diagnostic_reader = None
        self._process_stop = context.Event()
        self._grounding_required = context.Event()
        self._startup_ready = context.Event()
        self._stop, self._lock = Event(), Lock()
        self._cleanup_started = False
        self._source_finished = Event()
        self._process = context.Process(
            target=run_vlm_process,
            args=(
                config,
                self._incoming,
                self._outgoing,
                self._process_stop,
                inspector_factory,
                self._grounding_required,
                self._startup_ready,
                self._diagnostic_outgoing,
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
        if self._diagnostic_outgoing is not None:
            try:
                self._diagnostic_reader = Thread(
                    target=self._read_diagnostics, name="vlm-diagnostics", daemon=True)
                self._diagnostic_reader.start()
            except Exception:
                self._diagnostic_reader = None  # Full telemetry queues drop records.
        if self._camera_dispatch is not None:
            self._camera_dispatch.start()
        if self.fusion is not None:
            self.fusion.start()
        # start_camera runs off the HTTP event loop. Finish normal transport
        # preparation before its session ACK allows the first uploaded frame.
        # Keep failures bounded; readiness still requires a real model result.
        deadline = time.monotonic() + 10.0
        while not self._startup_ready.wait(0.05):
            if (
                self._stop.is_set()
                or not self._process.is_alive()
                or time.monotonic() >= deadline
            ):
                break
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
            self._next_sample = timestamp + 1.0 / _source_sample_fps(self.config)
            sample = SourceSample(
                EvidenceFrame(self.config.stream_session_id, sequence, timestamp, jpeg),
                utc_now(),
                now,
            )
            try:
                self._incoming.put_nowait(sample)
            except Full:
                if self.config.output_mode != "events":
                    self._stats["input_backpressure_dropped"] += 1
                    return False
                try:
                    self._incoming.get_nowait()
                    self._stats["input_superseded_samples"] += 1
                except Empty:
                    # multiprocessing's feeder can temporarily hide a full
                    # queue from get_nowait. Never block camera ingestion.
                    self._stats["input_backpressure_dropped"] += 1
                    return False
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

    def _read_diagnostics(self):
        while not self._stop.is_set():
            try:
                packet = self._diagnostic_outgoing.get(timeout=0.1)
            except Empty:
                if not self._process.is_alive():
                    return
                continue
            except (EOFError, OSError, ValueError):
                return
            if self._stop.is_set():
                return
            try:
                self.diagnostics.receive(packet)
            except Exception:
                # Malformed optional telemetry cannot kill result delivery.
                continue

    def _read(self):
        while not self._stop.is_set():
            if self._display_fusion is not None:
                if getattr(self._display_fusion, "grounding_required", False):
                    self._grounding_required.set()
                else:
                    self._grounding_required.clear()
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
                if self._rejected_report_observer is not None:
                    try:
                        self._rejected_report_observer(
                            copy.deepcopy(report), reason="stream_session_or_generation_mismatch"
                        )
                    except Exception as exc:
                        self._capture_failed(exc, "evaluation_rejected_archive_failed")
                    else:
                        self._capture_failed(None, "evaluation_report_identity_mismatch")
                    return
                continue
            now = time.monotonic()
            report["emitted_at"] = utc_now()
            report["timing"]["published_at"] = report["emitted_at"]
            report["timing"]["source_to_publish_seconds"] = max(
                0.0, now - report["timing"].pop("evidence_received_monotonic")
            )
            report["timing"].pop("request_started_monotonic", None)
            completed = report["timing"].pop("completion_monotonic", None)
            if completed is not None:
                report["timing"]["completion_to_publish_seconds"] = max(
                    0.0, now - completed
                )
            # Independent publication precedes every optional fusion/archive action.
            information = report.get("vlm_lane") != "grounding"
            grounding = report.get("vlm_lane") != "information"
            publication = (self.results if information else self.grounding_results).publish(
                report, measure_publication=self._publication_observer is not None)
            if self._report_observer is not None:
                try:
                    reference = self._report_observer(copy.deepcopy(report))
                    if self._publication_observer is not None:
                        self._publication_observer(publication, report_reference=reference)
                except Exception as exc:
                    self._capture_failed(exc, "evaluation_capture_failed")
                    return
            if information and self._shared_results is not None:
                self._shared_results.publish(report, expected_epoch=self._shared_epoch)
            if (
                information
                and self._delivery is not None
                and report["state"] == "completed"
            ):
                try:
                    # submit only admits to the webhook's own bounded queue.
                    # Network I/O and SAM association cannot delay later reports.
                    self._delivery.submit(report)
                except Exception as exc:
                    log_timing(
                        "vlm_delivery_admission_failure", error_type=type(exc).__name__
                    )
            if grounding and self._display_fusion is not None:
                try:
                    self._display_fusion.offer_vlm(report)
                except Exception as exc:
                    log_timing(
                        "vlm_display_handoff_failure", error_type=type(exc).__name__
                    )
            with self._lock:
                if not report.get("historical_only"):
                    self._status[
                        report.get("vlm_lane", "combined") + "_last_result"
                    ] = {
                        "frame_sequence": report["frame"]["source_frame_index"],
                        "state": report["state"],
                        "seconds": report["timing"]["source_to_publish_seconds"],
                    }
                self._status.update(report.get("process_metrics", {}))
                current_status = dict(
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
                if not report.get("historical_only"):
                    self._status.update(current_status)
                self._stats["published_reports"] += 1
                if not report.get("historical_only"):
                    self._last_publish_monotonic = now
                if report["state"] == "completed" and not report.get("historical_only"):
                    self._last_success_monotonic = now
                    self._status["last_success_at"] = report["emitted_at"]
                evidence = tuple(
                    self._evidence[item["frame_sequence"]].frame
                    for item in report["evidence"]
                    if item["frame_sequence"] in self._evidence
                )
            if grounding and self.fusion is not None and report["state"] == "completed":
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
                "reorder_capacity": REORDER_CAPACITY * (
                    2 if self.config.output_mode == "parallel" else 1
                ),
                "reorder_capacity_per_lane": REORDER_CAPACITY,
                "source_sample_fps": _source_sample_fps(self.config),
                "observation_fps": self.config.observation_fps,
                "observation_window_frames": self.config.observation_window_frames,
                "observation_frame_delta_seconds": (
                    self.config.observation_frame_delta_seconds
                ),
                "output_mode": self.config.output_mode,
                "shared_admission_enabled": self.config.admission is not None,
                "paired_admission": False,
                "independent_lane_admission": self.config.output_mode == "parallel",
            }
        return result

    def stats(self):
        result = self.health()
        if self.fusion is not None:
            result["fusion"] = self.fusion.stats()
        if self._camera_dispatch is not None:
            result["camera_source_dispatch"] = self._camera_dispatch.stats()
        return result

    def _capture_failed(self, exc, code):
        with self._lock:
            self._status.update(state="failed", error_code=code)
            if exc is not None:
                self._status["capture_error_type"] = type(exc).__name__
                self._status["capture_error_errno"] = getattr(exc, "errno", None)
        self._process_stop.set()
        self._stop.set()
        self.close()

    def close(self, timeout_seconds=0.0):
        with self._lock:
            start_cleanup = not self._cleanup_started
            self._cleanup_started = True
            self._stop.set()
            self._process_stop.set()
            if start_cleanup and self._status.get("state") != "failed":
                self._status["state"] = "stopping"
        if start_cleanup:
            try:
                if self.fusion is not None:
                    self.fusion.close()
            finally:
                try:
                    if self._camera_dispatch is not None:
                        self._camera_dispatch.close()
                finally:
                    with self._lock:
                        self._reaper = Thread(
                            target=self._cleanup, name="vlm-process-cleanup", daemon=True
                        )
                        self._reaper.start()
        if self._reaper is not None and timeout_seconds > 0:
            self._reaper.join(timeout_seconds)

    def close_complete(self):
        """Confirm process and callback retirement, not just cleanup initiation."""
        with self._lock:
            complete = self._status.get("cleanup_complete") is True
        return (complete and (not self._started or not self._process.is_alive())
                and all(thread is None or not thread.is_alive() for thread in
                        (self._reader, self._source, self._diagnostic_reader, self._reaper)))

    def _cleanup(self):
        if self._started:
            self._process.join(2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1.0)
        for thread in (self._reader, self._source, self._diagnostic_reader):
            if thread is not None:
                thread.join(1.0)
        for queue in (self._incoming, self._outgoing):
            queue.close()
            queue.cancel_join_thread()
        if self._diagnostic_outgoing is not None:
            try:
                self._diagnostic_outgoing.close()
                self._diagnostic_outgoing.cancel_join_thread()
            except Exception:
                pass  # Telemetry cleanup must not prevent the stopped state.
        with self._lock:
            self._evidence.clear()
            self._evidence_bytes = 0
            if self._status.get("state") != "failed":
                self._status["state"] = "stopped"
            self._status["cleanup_complete"] = True


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
