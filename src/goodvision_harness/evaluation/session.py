"""Wire source frames to independent publication, person registration, and native fusion.

The entry point supplies models and sources; this module neither allocates GPUs
nor reads questions/ground truth or loads faces. SAM requires native completion
batches; immediate detector results cannot stand in for those batches.
"""
from __future__ import annotations

from threading import Event, RLock
import time

from ..contracts import json_mapping_as_dict
from ..person_registry import PersonRegistry
from ..runtime.scheduling import EvidenceFrame
from ..video_tracking import VideoFrame
from ..video_tracking_session import continuous_source_history_seconds
from ..vlm_fusion import VLMFusion
from ..vlm_independent import SourceSample, utc_now
from .offline_observer import FixedWindowObserver
from .recording import ProductionEvidenceRecorder


class UnifiedCaptureSession:
    def __init__(self, directory, *, video_id, stream_session_id, source_id, source,
                 frame_plan, window_plan, provenance, profile, vlm_config, tracking_mode,
                 source_fps, max_entities, ledger):
        if tracking_mode not in {"immediate", "batch"}:
            raise ValueError("explicit native tracking mode required")
        if (vlm_config.stream_session_id != stream_session_id
                or vlm_config.source_id != source_id or vlm_config.labels != profile.labels):
            raise ValueError("independent configuration scope mismatch")
        self.recorder = ProductionEvidenceRecorder(
            directory, video_id=video_id, stream_session_id=stream_session_id, source_id=source_id,
            source=source, frame_plan=frame_plan, window_plan=window_plan,
            provenance={**provenance, "event_profile": profile.provenance(),
                        "question_conditioned": False, "face_enabled": False,
                        "scheduling_mode": "offline_fixed_backpressure",
                        "tracking_mode": tracking_mode,
                        "source_fps": source_fps, "scene_max_entities": max_entities,
                        "independent_lanes": ["events", "compact"],
                        "sam_display_projection": "disabled_for_all_backends",
                        "detector_specific_grounding_feedback": False,
                        "runtime_scope": "native_components_without_face_not_http_service"})
        self.sid, self.source_id = stream_session_id, source_id
        self.registry = PersonRegistry(stream_session_id, source_id)
        self.scene = profile.scene_runtime(stream_session_id=stream_session_id, source_id=source_id,
                                           tracking_mode=tracking_mode,
                                           native_memory_scope="session" if tracking_mode == "batch" else None,
                                           tracking_history_seconds=continuous_source_history_seconds(source_fps)
                                           if tracking_mode == "batch" else 12.0,
                                           max_entities=max_entities)
        self._record_lock = RLock()
        self._callback_error = None
        self._sealed = False
        self._closed = False
        self._wake = Event()
        self._sources, self._samples = {}, {}
        self._submitted, self._tracked = set(), set()
        self._frame_plan = {row["frame"]: row for row in frame_plan}
        self._ordered_frames = tuple(self._frame_plan)
        self._windows = {row["frame_sequences"][-1]: row for row in window_plan}
        if len(self._windows) != len(window_plan):
            raise ValueError("one fixed window per source endpoint required")
        self.fusion = VLMFusion(self.scene, on_update=self._capture_scene)
        self.observer = FixedWindowObserver(vlm_config, recorder=self.recorder,
                                            on_grounding=self.fusion.offer, ledger=ledger)
        self.fusion.start()

    def _check(self):
        if self._closed or self._sealed:
            raise RuntimeError("capture session is closed")
        if self._callback_error is not None:
            raise RuntimeError("persistent scene callback failed") from self._callback_error

    def source_frame(self, frame):
        """Admit source images to both lanes without conditioning VLM admission on person detections."""
        self._check()
        if (not isinstance(frame, VideoFrame)
                or len(self._submitted) >= len(self._ordered_frames)
                or frame.frame_sequence != self._ordered_frames[len(self._submitted)]):
            raise ValueError("source frame order differs from frozen plan")
        row = self._frame_plan[frame.frame_sequence]
        if row["t"] != frame.timestamp_seconds:
            raise ValueError("source frame timestamp differs from the source plan")
        if self.observer.pending:
            # Apply offline backpressure rather than dropping a busy window.
            self.observer.drain_window()
        available = time.monotonic()
        sample = SourceSample(EvidenceFrame(self.sid, frame.frame_sequence,
                                            frame.timestamp_seconds, frame.jpeg_bytes),
                              utc_now(), available)
        self._submitted.add(frame.frame_sequence)
        self._sources[frame.frame_sequence] = frame
        self._samples[frame.frame_sequence] = sample
        self.recorder.timing({"clock_domain": self.recorder.clock_domain,
                              "kind": "source_ingress", "frame_sequence": frame.frame_sequence,
                              "timestamp_seconds": frame.timestamp_seconds,
                              "frame_available": available})
        self.scene.observe_frame(frame.jpeg_bytes, frame.timestamp_seconds,
                                 frame.frame_sequence, force=True)
        window = self._windows.get(frame.frame_sequence)
        if window is not None:
            self.observer.submit(window, [self._samples[index] for index in window["frame_sequences"]])
        # Keep originals until tracking receipts arrive; retain VLM images needed by future windows.
        future_needed = {index for end, plan in self._windows.items() if end > frame.frame_sequence
                         for index in plan["frame_sequences"]}
        self._samples = {index: value for index, value in self._samples.items() if index in future_needed}

    def _canonicalize(self, snapshots):
        canonical = []
        for snapshot in snapshots:
            if snapshot.stream_session_id != self.sid:
                raise ValueError("tracking response session mismatch")
            self.registry.observe_snapshot(snapshot)
            mapped = self.registry.snapshot_at(snapshot.timestamp_seconds, snapshot.frame_sequence)
            if mapped is None or (mapped.timestamp_seconds, mapped.frame_sequence) != (
                    snapshot.timestamp_seconds, snapshot.frame_sequence):
                raise ValueError("registry did not produce the exact requested source")
            canonical.append(mapped)
        self.scene.apply_subject_merges(self.registry.drain_subject_merges())
        return tuple(canonical)

    def _persist_tracks(self, raw, canonical):
        for original, mapped in zip(raw, canonical, strict=True):
            index = mapped.frame_sequence
            if index not in self._submitted or index in self._tracked:
                raise ValueError("unsubmitted or duplicate tracking completion")
            subjects = [{"id": obj.track_id, "label": obj.label, "bbox": obj.bbox.as_dict(),
                         "observation_state": obj.observation_state or obj.attributes.get("observation_state"),
                         "last_observed_frame_sequence": obj.last_observed_frame_sequence,
                         "last_observed_timestamp_seconds": obj.last_observed_timestamp_seconds,
                         "attributes": json_mapping_as_dict(obj.attributes)} for obj in mapped.objects]
            if any(row["observation_state"] not in {"observed", "predicted"} for row in subjects):
                raise ValueError("tracking completion lacks explicit observation state")
            self.recorder.tracking_frame(frame=index, timestamp_seconds=mapped.timestamp_seconds,
                                          subjects=subjects,
                                          raw_response=original.as_dict())
            self._tracked.add(index)
            self._sources.pop(index, None)

    def tracking_snapshot(self, snapshot):
        self._check()
        if self.scene.tracking_mode != "immediate":
            raise ValueError("SAM requires native completion batches")
        canonical = self._canonicalize((snapshot,))
        self.scene.observe_tracks(canonical[0])
        self._persist_tracks((snapshot,), canonical)
        self._capture_scene()

    def tracking_batch(self, batch):
        self._check()
        if self.scene.tracking_mode != "batch" or batch.stream_session_id != self.sid:
            raise ValueError("unexpected native tracking batch")
        if not batch.frames:
            return  # Accepted is not completed; do not fabricate an empty-frame receipt.
        canonical = []

        def map_frames(snapshots):
            if not self.registry.begin_native_epoch(batch.backend, batch.predictor_epoch):
                raise ValueError("native tracking epoch rejected")
            canonical.extend(self._canonicalize(snapshots))
            return tuple(canonical)

        retained = [self._sources[frame.frame_sequence] for frame in batch.frames]
        accepted = self.scene.commit_native_tracking_batch(
            batch.frames, completed_timestamp_seconds=batch.completed_timestamp_seconds,
            completed_frame_sequence=batch.completed_frame_sequence,
            canonicalize=map_frames, retained_sources=retained)
        if not accepted:
            raise ValueError(f"native completion rejected: {self.scene.tracking_last_rejection}")
        self._persist_tracks(batch.frames, canonical)
        self._capture_scene()

    def _capture_scene(self):
        try:
            with self._record_lock:
                if not self._sealed and not self._closed:
                    self.recorder.scene_snapshot(self.scene.snapshot(include_history=True),
                                                 association_links=self.fusion.links())
        except Exception as exc:
            self._callback_error = exc
            raise
        finally:
            self._wake.set()

    def finish(self, *, source_eof, tracking_drained, drain_timeout_seconds=10):
        self._check()
        if type(source_eof) is not bool or type(tracking_drained) is not bool:
            raise ValueError("actual source EOF and tracking drain states required")
        self.observer.drain_window()
        all_sources = self._submitted == set(self._frame_plan)
        all_tracks = self._tracked == self._submitted and not self._sources
        if source_eof and not all_sources or tracking_drained and not all_tracks:
            raise ValueError("claimed source/tracking completion is missing fixed frames")
        if tracking_drained:
            self.scene.finish_tracking_sources()
        self.fusion.finish()
        until = time.monotonic() + drain_timeout_seconds
        while self.fusion.stats()["pending_reports"] and time.monotonic() < until:
            self._wake.wait(.1)
            self._wake.clear()
            self._check()
        self._capture_scene()
        self._check()
        stats = self.fusion.stats()
        for key in ("capacity_dropped_reports", "expired_reports", "association_failures", "callback_failures"):
            if stats.get(key, 0):
                self.recorder.request_failure(stage="fusion", reason=key, detail={"count": stats[key]})
        # Preserve unresolved associations without forcing IDs or deleting raw events.
        with self._record_lock:
            result = self.recorder.finish(
                source_eof=source_eof, tracking_drained=tracking_drained,
                events_drained=self.observer.completed_windows == len(self._windows),
                fusion_drained=stats["pending_reports"] == 0)
            self._sealed = True
        return result

    def close(self):
        if self._closed:
            return
        try:
            self.observer.close()
        finally:
            try:
                self.fusion.close()
            finally:
                with self._record_lock:
                    self._closed = True
                    self.scene.close()
