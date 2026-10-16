"""Timed source playback through the production background processing components.

The outer owner supplies an already allocated, isolated detector endpoint. This
module opens neither the production HTTP service nor a second GPU. Source input
continues while inference runs; native admission and replacement policies apply.
"""
from __future__ import annotations

import hashlib
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


def effective_config_sha256(harness):
    """Bind resolved settings, including environment overrides, without exporting secrets."""
    encoded = json.dumps(asdict(harness), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        self._sealed = self._closed = False
        self._submitted, self._tracked = set(), set()
        self._frame_plan = recorder.frames
        self._sources = {}
        self.fusion = VLMFusion(self.scene, on_update=self._capture_scene)
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
            if stats.get("state") == "failed" or stats.get("failed_requests", 0):
                self.recorder.request_failure(stage="events", reason="native_worker_failed", detail=stats)
                raise RuntimeError("native VLM worker failed; retain ledger outcomes for recovery")


def capture_native_stream(config_path, *, config_sha256, output, resources):
    path = Path(config_path).resolve()
    base = path.parent
    config = _read_frozen(path, config_sha256)
    reject_private(config, observation=True)
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
    if hashlib.sha256(harness_path.read_bytes()).hexdigest() != config["harness_config_sha256"]:
        raise ValueError("Harness config hash changed")
    harness = load_config(harness_path)
    if effective_config_sha256(harness) != config["effective_harness_config_sha256"]:
        raise ValueError("resolved Harness settings differ from the frozen effective configuration")
    if (harness.video.realtime is not True or harness.video.playback_speed != 1.0
            or harness.video.max_sampled_frames is not None or harness.memory is not None):
        raise ValueError("native evaluation requires full timed playback at source speed")
    for key in ("max_source_wall_seconds", "drain_timeout_seconds"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("explicit finite source and drain time limits required")
    media = base / config["media_path"]
    media_digest = hashlib.sha256()
    with media.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            media_digest.update(chunk)
    digest = media_digest.hexdigest()
    if digest != config["media_sha256"]:
        raise ValueError("media file hash changed")
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
    provenance = {**config["provenance"], "run_config_sha256": config_sha256,
                  "media_sha256": digest, "harness_config_sha256": config["harness_config_sha256"],
                  "effective_harness_config_sha256": config["effective_harness_config_sha256"],
                  "question_conditioned": False, "face_enabled": False,
                  "vlm_enabled": config["vlm_enabled"], "event_profile": profile.provenance(),
                  "tracking_model": config["tracking"]["model"],
                  "scheduling_mode": "native_streaming",
                  "window_plan_semantics": "independent_native_lanes_v1",
                  "runtime_scope": "production_components_with_timed_file_source",
                  "coverage_policy": coverage_policy,
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
                str((base / ledger["seed_manifest"]).resolve()), ledger["seed_manifest_sha256"],
                str((base / ledger["owner_lock"]).resolve()))
            capture.worker = IndependentVLMBranch(vlm, fusion=capture.fusion,
                inspector_factory=factory, report_observer=recorder.independent_report,
                rejected_report_observer=recorder.rejected_report)
            capture.worker.start()

        def ended(reason, error):
            terminal.update(reason=reason, error=error, ended_monotonic=time.monotonic())
            source_done.set()

        session = StreamingSession.for_file(harness,
            SceneCaptureClient(harness.inference, profile.labels if config["vlm_enabled"] else ()),
            SessionCallbacks(on_display_frame=capture.source, on_end=ended),
            video_path=media, realtime=True, source_id=capture.source_id,
            stream_session_id=capture.sid)
        session.start()
        while not source_done.wait(.1):
            capture.check_runtime()
            if time.monotonic() - started > config["max_source_wall_seconds"]:
                raise TimeoutError("native source exceeded its frozen wall-time limit")
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
                raise TimeoutError("native workers exceeded frozen drain limit")
        capture.scene.finish_tracking_sources()
        capture.scene.finish_inspection_sources()
        capture.fusion.finish()
        while capture.fusion.stats()["pending_reports"]:
            capture.check_runtime()
            if time.monotonic() >= deadline:
                raise TimeoutError("native fusion exceeded frozen drain limit")
            capture._wake.wait(.1)
            capture._wake.clear()
        capture._capture_scene()
        stats = {"tracking": capture.tracker.stats(), "vlm": capture.worker.stats() if capture.worker else None,
                 "fusion": capture.fusion.stats(), "source": terminal,
                 "processing_wall_seconds": time.monotonic() - started}
        for key in ("capacity_dropped_reports", "expired_reports", "association_failures", "callback_failures"):
            if stats["fusion"].get(key, 0):
                recorder.request_failure(stage="fusion", reason=key, detail=stats["fusion"][key])
        with capture._record_lock:
            capture._check()
            evidence = recorder.finish(source_eof=True, tracking_drained=True, events_drained=True,
                                       fusion_drained=True, runtime_statistics=stats)
            capture._sealed = True
        outcome = {"status": "captured" if evidence["observation_manifest"]["complete"] else "incomplete",
                   "video_id": config["video_id"], "stream_session_id": capture.sid,
                   "config_sha256": config_sha256, "evidence_sha256": evidence["evidence_sha256"],
                   "vlm_enabled": config["vlm_enabled"], "scope": "native_streaming",
                   "final_answer_generated": False}
        publish_once(recorder.root / "outcome.json", outcome)
        return outcome
    except BaseException as exc:
        publish_once(recorder.root / "failure.json", {"error_type": type(exc).__name__,
            "config_sha256": config_sha256, "stream_session_id": capture.sid,
            "status": "interrupted", "preserve_ledger_and_raw_records": True})
        raise
    finally:
        if session:
            session.stop()
            session.join(5)
        if capture.worker:
            capture.worker.close(timeout_seconds=10)
        if capture.tracker:
            capture.tracker.close(timeout_seconds=10)
        capture.fusion.close()
        capture.scene.close()
        capture._closed = True
        publish_once(recorder.root / "cleanup.json", {
            "source_stopped": session is None or not session.running,
            "tracking_stopped": capture.tracker is None or capture.tracker.close_complete(),
            "vlm_stopped": capture.worker is None or capture.worker.stats().get("cleanup_complete") is True,
            "gpu_release_owned_by_outer_controller": True})
