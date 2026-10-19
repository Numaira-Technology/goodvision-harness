"""Timed source playback through the production background processing components.

The outer owner supplies an already allocated, isolated detector endpoint. This
module opens neither the production HTTP service nor a second GPU. Source input
continues while inference runs; native admission and replacement policies apply.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, RLock
import time
from urllib.parse import urlparse

from ..config import load_config
from ..perception import create_tracking_client, create_tracking_coordinator
from ..person_registry import PersonRegistry
from ..pipeline import SessionCallbacks, StreamingSession
from ..rich_labeling import RichLabelCoordinator, RichLabelHTTPClient
from ..scene_capture import SceneCaptureClient
from ..video_tracking_session import continuous_source_history_seconds
from ..vlm_fusion import VLMFusion
from ..vlm_independent import IndependentVLMBranch
from .capture_source import _read_frozen
from .privacy import reject_private
from .profile import load_profile
from .recording import publish_once
from .session import UnifiedCaptureSession
from .stream_ledger import LedgerInspectorFactory
from .stream_recording import StreamingRecorder
from .evidence import NATIVE_SKIP_POLICY
from .failures import native_report_failure_policy
from .configuration import source_settings


class NativeSourceFailure(RuntimeError):
    """A source-local interruption, eligible to skip only after confirmed cleanup."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def effective_config_snapshot(harness):
    """Record evaluation settings, excluding service and source credentials."""
    names = ("video", "display", "rich_labeling", "sliding_window", "inference", "runtime")
    values = {name: asdict(getattr(harness, name)) for name in names}
    def redact(value):
        if isinstance(value, dict):
            return {key: ("[REDACTED]" if key.lower() in {
                "api_key", "authorization", "password", "secret", "token", "headers"}
                else redact(item)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [redact(item) for item in value]
        return value
    # Endpoints belong to the separately checked transport, not this snapshot.
    values["rich_labeling"].pop("endpoint", None)
    return json.loads(json.dumps(redact(values), default=str, allow_nan=False))


class NativeCapture:
    """Use the same registry/scene commits as the service and archive every callback."""

    _check = UnifiedCaptureSession._check
    _canonicalize = UnifiedCaptureSession._canonicalize
    _persist_tracks = UnifiedCaptureSession._persist_tracks
    _capture_scene = UnifiedCaptureSession._capture_scene

    def __init__(self, recorder, *, profile, source_fps, native_batches, vlm_enabled):
        self.recorder = recorder
        self.sid = recorder.manifest["stream_session_id"]
        self.source_id = recorder.manifest["source_id"]
        self.registry = PersonRegistry(self.sid, self.source_id)
        self.scene = profile.scene_runtime(
            stream_session_id=self.sid, source_id=self.source_id,
            tracking_mode="batch" if native_batches else "immediate",
            native_memory_scope="session" if native_batches else None,
            tracking_history_seconds=continuous_source_history_seconds(source_fps)
            if native_batches else 12.0)
        self._record_lock = RLock()
        self._wake = Event()
        self._callback_error = None
        self._shared_failure = False
        self._sealed = self._closed = False
        self._submitted, self._tracked = set(), set()
        self._frame_plan = recorder.frames
        self._sources = {}
        self.fusion = VLMFusion(self.scene, on_update=self._capture_scene,
                                on_skip=self.fusion_skip)
        self.vlm_enabled = vlm_enabled
        self.worker = self.tracker = None

    def callback(self, operation, *args):
        try:
            with self._record_lock:
                self._check()
                return operation(*args)
        except BaseException as exc:
            self._callback_error = exc
            self._wake.set()
            raise

    def source(self, jpeg, timestamp, sequence):
        try:
            return self._source(jpeg, timestamp, sequence)
        except BaseException as exc:
            # StreamingSession exposes a generic end error; retain the cause so
            # callback/recording failures cannot become source-local timeouts.
            self._callback_error = exc
            self._wake.set()
            raise

    def _source(self, jpeg, timestamp, sequence):
        self._check()
        with self._record_lock:
            self.recorder.source_frame(jpeg, timestamp, sequence)
            self._submitted.add(sequence)
            self.scene.observe_frame(jpeg, timestamp, sequence, force=True)
        # Independent observation admission never waits for a detector response.
        vlm_admitted = self.worker.offer(jpeg, timestamp, sequence) if self.worker else None
        admitted = self.tracker.submit(jpeg, stream_session_id=self.sid,
                                      frame_sequence=sequence, timestamp_seconds=timestamp)
        self.recorder.record("admission", {
            "frame_sequence": sequence, "timestamp_seconds": timestamp,
            "tracking_admitted": admitted, "vlm_admitted": vlm_admitted,
            "tracking_statistics": self.tracker.stats()})

    def snapshot(self, snapshot):
        return self.callback(UnifiedCaptureSession.tracking_snapshot, self, snapshot)

    def source_input(self, payload):
        return self.callback(self.recorder.record, "frame-input", payload)

    def publication(self, record, *, report_reference):
        lower, upper = record.readable_monotonic_bounds
        return self.callback(self.recorder.record, "publication", {
            "report_reference": report_reference,
            "lane": record.payload.get("vlm_lane", "combined"),
            "cursor": record.cursor, "state": record.payload["state"],
            "readable_monotonic_lower": lower,
            "readable_monotonic_upper": upper,
            "boundary_width_seconds": upper - lower,
        })

    def independent_report(self, report):
        # Archive the original window before acting on its failure scope. The
        # recorder and evidence adapter retain failed windows as failed, not as
        # successful observations or negatives.
        try:
            reference = self.recorder.independent_report(report)
            failure = native_report_failure_policy(report)
            if failure and failure["scope"] == "shared":
                self.recorder.request_failure(stage="events", reason="shared_window_failure",
                    detail={**failure, "request_id": report["request_id"], "reference": reference})
                raise RuntimeError("native window requires shared-service recovery")
            return reference
        except BaseException:
            self._shared_failure = True
            raise

    def fusion_skip(self, report, reason):
        """Archive unavailable association times without discarding other lanes."""
        try:
            stamps = report["evidence"]
            self.recorder.request_failure(stage="fusion", reason=reason, detail={
                "request_id": report["request_id"], "message_id": report["message_id"],
                "start": min(item["timestamp_seconds"] for item in stamps),
                "end": max(item["timestamp_seconds"] for item in stamps),
                "frame_sequences": [item["frame_sequence"] for item in stamps],
                "affected_lane": "person_association", "status": "skipped"})
        except BaseException as exc:
            self._callback_error = exc
            self._shared_failure = True
            self._wake.set()
            raise

    def batch(self, batch):
        def commit():
            if not batch.frames:
                return
            if batch.stream_session_id != self.sid:
                raise ValueError("native tracking session changed")
            accepted = getattr(self.tracker, "accepts_batch", None)
            if accepted is not None and not accepted(batch):
                raise ValueError("native tracking completion generation rejected")
            retained = self.tracker.retained_sources(batch)
            canonical = []

            def map_frames(snapshots):
                if not self.registry.begin_native_epoch(
                        getattr(batch, "backend", "sam3.1_video"), batch.predictor_epoch):
                    raise ValueError("native predictor epoch rejected")
                canonical.extend(self._canonicalize(snapshots))
                return tuple(canonical)

            if not self.scene.commit_native_tracking_batch(
                    batch.frames, completed_timestamp_seconds=batch.completed_timestamp_seconds,
                    completed_frame_sequence=batch.completed_frame_sequence,
                    canonicalize=map_frames, retained_sources=retained):
                raise RuntimeError("native scene completion rejected")
            self._persist_tracks(batch.frames, canonical)
            self._capture_scene()

        return self.callback(commit)

    def check_runtime(self):
        self._check()
        stats = self.tracker.stats()
        if stats.get("failed", 0) or stats.get("state") == "failed":
            self.recorder.request_failure(stage="tracking", reason="native_worker_failed", detail=stats)
            raise RuntimeError("native tracking worker failed; see saved runtime statistics")
        if self.worker:
            stats = self.worker.stats()
            if stats.get("state") == "failed":
                self.recorder.request_failure(stage="events", reason="native_worker_failed", detail=stats)
                raise RuntimeError("native VLM worker failed; retain ledger outcomes for recovery")


def capture_native_stream(config_path, *, output, resources, config_sha256=None):
    path = Path(config_path).resolve()
    base = path.parent
    config = _read_frozen(path)
    reject_private(config, observation=True)
    evaluation_timing = config.get("evaluation_timing", False)
    if type(evaluation_timing) is not bool:
        raise ValueError("evaluation_timing must be an explicit boolean")
    if (config.get("schema_version") != "native-stream-source-v1"
            or type(config.get("vlm_enabled")) is not bool
            or config.get("face_enabled") is not False
            or config.get("question_conditioned") is not False
            or config.get("sam_internal_patch") is not False):
        raise ValueError("explicit native stream, VLM choice, stock model and face-disabled identity required")
    resources.prepare(config)
    coverage_policy = config.get("coverage_policy",
        "all_delivered_person_frames_and_native_admitted_vlm_windows")
    if coverage_policy not in {"all_delivered_person_frames_and_native_admitted_vlm_windows",
                               NATIVE_SKIP_POLICY}:
        raise ValueError("unknown frozen streaming coverage policy")
    harness_path = base / config["harness_config"]
    harness = load_config(harness_path)
    effective = effective_config_snapshot(harness)
    if ("effective_harness_config" in config
            and effective != config["effective_harness_config"]):
        raise ValueError("resolved Harness settings differ from the frozen effective configuration")
    if (harness.video.realtime is not True or harness.video.playback_speed != 1.0
            or harness.video.max_sampled_frames is not None or harness.memory is not None):
        raise ValueError("native evaluation requires full timed playback at source speed")
    for key in ("max_source_wall_seconds", "drain_timeout_seconds"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("explicit finite source and drain time limits required")
    media = base / config["media_path"]
    profile = load_profile(base / config["event_definitions"])
    rich = harness.rich_labeling
    if not rich.enabled:
        raise ValueError("native answer capture requires its selected tracker enabled")
    native_batches = rich.backend == "sam31_stream"
    tracking = config["tracking"]
    expected_backend = {"sam3.1": "sam31_stream", "yolo26-s": "paddle",
                        "yolo26-l": "paddle", "rfdetr-l": "paddle"}.get(tracking["model"])
    if (expected_backend is None or rich.backend != expected_backend
            or rich.backend != tracking["backend"] or rich.endpoint != tracking["endpoint"]
            or rich.model != tracking["model"] or not rich.tracking_enabled):
        raise ValueError("selected detector must match the frozen native backend, model and endpoint")
    if native_batches and rich.processing_method != "sam_live":
        raise ValueError("native SAM capture requires the explicit causal sam_live route")
    if rich.backend == "sam31":
        raise ValueError("batch-reset SAM is not the continuous native method")
    provenance = {**config["provenance"], "run_config_path": str(path),
                  "source_configuration": source_settings(config),
                  "media_path": str(media.resolve()), "harness_config_path": str(harness_path.resolve()),
                  "effective_harness_config": effective,
                  "question_conditioned": False, "face_enabled": False,
                  "vlm_enabled": config["vlm_enabled"], "event_profile": profile.provenance(),
                  "tracking_model": config["tracking"]["model"],
                  "scheduling_mode": "native_streaming",
                  "window_plan_semantics": "independent_native_lanes_v1",
                  "runtime_scope": "production_components_with_timed_file_source",
                  "coverage_policy": coverage_policy,
                  "evaluation_timing": evaluation_timing,
                  "timing_protocol": "paced_raw_input_to_independent_bus_v1"
                  if evaluation_timing else None,
                  "source_preprocessing": {"display_fps": harness.display.fps,
                      "model_fps": harness.video.fps, "max_image_side": harness.display.max_image_side,
                      "jpeg_quality": harness.display.jpeg_quality}}
    recorder = StreamingRecorder(output, video_id=config["video_id"],
        stream_session_id=config["stream_session_id"], source_id=config["source_id"],
        source=config["source"], provenance=provenance, profile=profile)
    capture = NativeCapture(recorder, profile=profile,
        source_fps=max(harness.display.fps, harness.video.fps),
        native_batches=native_batches, vlm_enabled=config["vlm_enabled"])
    source_done = Event()
    terminal = {}
    session = None
    failure = None
    started = time.monotonic()
    try:
        if native_batches:
            client = create_tracking_client(rich)
            capture.tracker = create_tracking_coordinator(client, rich, capture.batch)
            if not hasattr(capture.tracker, "retained_sources"):
                raise ValueError("native streaming tracker must retain admitted source evidence")
        else:
            client = RichLabelHTTPClient(rich.endpoint, timeout_seconds=rich.timeout_seconds,
                                         headers={"x-goodvision-detector-model": tracking["model"]},
                                         mode="tracks", include_attributes=False)
            capture.tracker = RichLabelCoordinator(client, max_fps=rich.max_fps,
                history_size=rich.history_size,
                max_interpolation_gap_seconds=rich.max_interpolation_gap_seconds,
                max_hold_seconds=rich.max_hold_seconds, on_snapshot=capture.snapshot)
        capture.tracker.reset(config["stream_session_id"])
        if config["vlm_enabled"]:
            settings = harness.model_api
            endpoint = urlparse(settings.endpoint)
            if (settings.max_retries != 0 or settings.inspection_endpoint
                    or endpoint.scheme != "https" or endpoint.username or endpoint.password
                    or endpoint.query or endpoint.fragment
                    or harness.inference.model != "qwen/qwen3-vl-8b-instruct"):
                raise ValueError("explicit Qwen3-VL-8B HTTPS route with ledger-owned retries required")
            vlm = profile.vlm_config(stream_session_id=capture.sid, source_id=capture.source_id,
                generation=1, model_settings=settings, model=harness.inference.model,
                api_key=os.environ.get(settings.api_key_env, ""),
                parameters=harness.inference.request_parameters,
                options=config["vlm"]["inspector_settings"],
                observation_fps=harness.runtime.scene_observation_fps,
                max_in_flight=harness.runtime.scene_max_in_flight,
                route_id=config["vlm"]["route_id"])
            vlm = replace(vlm, retain_historical_results=True)
            ledger = config["model_ledger"]
            factory = LedgerInspectorFactory(str((base / ledger["output_directory"]).resolve()),
                str((base / ledger["seed_manifest"]).resolve()),
                str((base / ledger["owner_lock"]).resolve()))
            capture.worker = IndependentVLMBranch(vlm, fusion=capture.fusion,
                inspector_factory=factory, report_observer=capture.independent_report,
                rejected_report_observer=recorder.rejected_report,
                publication_observer=capture.publication if evaluation_timing else None)
            capture.worker.start()

        def ended(reason, error):
            terminal.update(reason=reason, error=error, ended_monotonic=time.monotonic())
            source_done.set()

        session = StreamingSession.for_file(harness,
            SceneCaptureClient(harness.inference, profile.labels if config["vlm_enabled"] else ()),
            SessionCallbacks(on_display_frame=capture.source, on_end=ended,
                on_source_input=capture.source_input if evaluation_timing else None),
            video_path=media, realtime=True, source_id=capture.source_id,
            stream_session_id=capture.sid)
        session.start()
        while not source_done.wait(.1):
            capture.check_runtime()
            if time.monotonic() - started > config["max_source_wall_seconds"]:
                raise NativeSourceFailure("native_source_wall_time_exceeded")
        capture.check_runtime()
        recorder.record("source-end", terminal)
        if terminal.get("reason") != "eof" or terminal.get("error") is not None:
            raise RuntimeError("native source ended without decoder EOF")
        if capture.worker:
            capture.worker.finish_source()
        if hasattr(capture.tracker, "mark_capture_finished"):
            if not capture.tracker.mark_capture_finished(stream_session_id=capture.sid,
                                                         generation=capture.tracker._generation):
                raise RuntimeError("native tracking rejected the capture-end identity")
        flush = getattr(capture.tracker, "flush", None)
        if flush:
            flush()
        deadline = time.monotonic() + config["drain_timeout_seconds"]
        while True:
            capture.check_runtime()
            tracks_done = capture.tracker.wait_until_idle(.1)
            if native_batches:
                tracks_done = tracks_done and capture.tracker.stats().get("state") == "finished"
            vlm_done = capture.worker.wait_until_idle(.1) if capture.worker else True
            if tracks_done and vlm_done:
                break
            if time.monotonic() >= deadline:
                raise NativeSourceFailure("native_workers_drain_timeout")
        capture.scene.finish_tracking_sources()
        capture.scene.finish_inspection_sources()
        capture.fusion.finish()
        while capture.fusion.stats()["pending_reports"]:
            capture.check_runtime()
            if time.monotonic() >= deadline:
                raise NativeSourceFailure("native_fusion_drain_timeout")
            capture._wake.wait(.1)
            capture._wake.clear()
        capture._capture_scene()
        stats = {"tracking": capture.tracker.stats(), "vlm": capture.worker.stats() if capture.worker else None,
                 "fusion": capture.fusion.stats(), "source": terminal,
                 "processing_wall_seconds": time.monotonic() - started}
        if evaluation_timing:
            stats["input_timing"] = session.video_stream.input_timing_statistics
        # Skipped associations have already been archived with their source times.
        for key in ("association_failures", "callback_failures"):
            if stats["fusion"].get(key, 0):
                recorder.request_failure(stage="fusion", reason=key, detail=stats["fusion"][key])
        with capture._record_lock:
            capture._check()
            evidence = recorder.finish(source_eof=True, tracking_drained=True, events_drained=True,
                                       fusion_drained=True, runtime_statistics=stats)
            capture._sealed = True
        outcome = {"status": "captured" if evidence["observation_manifest"]["complete"] else "incomplete",
                   "video_id": config["video_id"], "stream_session_id": capture.sid,
                   "config_path": str(path), "source_id": capture.source_id,
                   "vlm_enabled": config["vlm_enabled"], "scope": "native_streaming",
                   "final_answer_generated": False}
        publish_once(recorder.root / "outcome.json", outcome)
        return outcome
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if session:
            session.stop()
            session.join(5)
        if capture.worker:
            capture.worker.close(timeout_seconds=10)
        if capture.tracker:
            capture.tracker.close(timeout_seconds=10)
        capture.fusion.close(timeout_seconds=10)
        capture.scene.close()
        capture._closed = True
        tracking_stats = capture.tracker.stats() if capture.tracker else {}
        vlm_stats = capture.worker.stats() if capture.worker else {}
        shared_failure = bool(capture._shared_failure or capture._callback_error is not None
            or tracking_stats.get("failed", 0) or tracking_stats.get("state") == "failed"
            or vlm_stats.get("state") == "failed"
            or capture.fusion.stats().get("callback_failures", 0)
            or capture.fusion.stats().get("association_failures", 0))
        # A close-time HTTP timeout can be counted as stale instead of failed.
        # Require a normal accepted completion for every started detector request.
        tracking_settled = (not native_batches and capture.tracker is not None
            and tracking_stats.get("started") == tracking_stats.get("succeeded")
            and tracking_stats.get("in_flight") is False
            and tracking_stats.get("pending") is False)
        cleanup = {
            "source_stopped": session is None or not session.running,
            "tracking_stopped": capture.tracker is None or capture.tracker.close_complete(),
            "vlm_stopped": capture.worker is None or capture.worker.close_complete(),
            "fusion_stopped": capture.fusion.close_complete(),
            "shared_failure": shared_failure,
            "tracking_requests_settled": tracking_settled,
            "tracking_session_scope": "native_session" if native_batches else "stream_session_id",
            "tracking_statistics": tracking_stats, "vlm_statistics": vlm_stats,
            "gpu_release_owned_by_outer_controller": True}
        if evaluation_timing:
            cleanup["input_timing"] = (session.video_stream.input_timing_statistics
                                        if session is not None else None)
        if failure is not None:
            # A fatal callback can arrive while the main thread waits for idle.
            # Decide scope only after shutdown, with shared failures taking precedence.
            local = (isinstance(failure, NativeSourceFailure) and not shared_failure
                and tracking_settled and all(cleanup[key] for key in
                    ("source_stopped", "tracking_stopped", "vlm_stopped", "fusion_stopped")))
            publish_once(recorder.root / "failure.json", {"error_type": type(failure).__name__,
                "config_path": str(path), "stream_session_id": capture.sid,
                "failure_scope": "source" if local else "shared",
                "reason": failure.reason if local else "shared_capture_interruption",
                "status": "interrupted", "preserve_ledger_and_raw_records": True})
        publish_once(recorder.root / "cleanup.json", cleanup)
