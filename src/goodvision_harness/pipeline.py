"""Glue a live frame source to windowed Qwen inference and the organizer.

A ``StreamingSession`` owns one run: it walks a live source (a wall-time
video file, network stream, or pushed camera frames), fans decoded frames out
to the display callback, samples model frames at the configured FPS, builds
sliding windows, and schedules each window on a bounded worker pool. Finished
responses flow through the ``ResultOrganizer`` and out via callbacks, so the
same session drives both the web server and the headless CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from .config import DEFAULT_CONFIG_PATH, HarnessConfig, load_config
from .contracts import (
    PUBLIC_MODEL_NAME,
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
)
from .event_archive import EventArchive
from .face_admission import FaceAdmissionQueue
from .face_recognition import FaceRecognitionRuntime, FaceRecognitionSession
from .frame_crop import FrameCropSelector
from .grounding import (
    GroundingClient,
    MultiGroundedClassification,
    build_grounding_client,
    observation_bundle,
)
from .inference import WindowClient, WindowResponse
from .latency_trace import log_timing, trace_id
from .organizer import ResultOrganizer
from .person_registry import PersonRegistry
from .privacy import (
    PUBLIC_ANALYSIS_ERROR,
    public_error_message,
    public_safe_value,
)
from .reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    UNKNOWN_FACE_EVENT_LABEL,
)
from .sliding_window import FrameWindow, SampledFrame, iter_sliding_windows
from .stream_adapter import NetworkStreamAdapter, StreamAdapterError
from .video_to_stream import (
    PushSampledSource,
    VideoFileStream,
    encode_jpeg,
)
from .vision_result import VisionResult

FILE_MODE = "file"
CAMERA_MODE = "camera"
STREAM_MODE = "stream"
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionCallbacks:
    """Thread-safe sinks for everything a running session produces.

    Callbacks may be invoked from the session thread and from inference
    worker threads; implementations must marshal to their own event loop or
    lock as needed.
    """

    on_display_frame: Callable[[bytes, float, int], None] | None = None
    on_event: Callable[[dict[str, Any]], None] | None = None
    on_state: Callable[[dict[str, Any]], None] | None = None
    on_end: Callable[[str, str | None], None] | None = None
    on_vision_result: Callable[[VisionResult], None] | None = None
    # Only the newest successful result in a branch may supply its exact frame.
    on_analysis_frame: Callable[[AnalysisResult, bytes], None] | None = None
    # Every valid grounded observation, in arrival order, including superseded
    # positives and negatives. Consumers keep source-ordered live state and can
    # confirm a clear using the same source observations as the video overlay.
    on_analysis_result: Callable[[AnalysisResult], None] | None = None
    # Every successful grounded keyframe, including a response superseded in
    # its own branch. Mixed local-face/safety sessions publish the newest result
    # of each branch independently, always with that branch's exact source frame.
    on_timeline_result: Callable[[AnalysisResult], None] | None = None
    # Evaluation-only file ingress, before cadence selection and JPEG encoding.
    on_source_input: Callable[[dict[str, Any]], None] | None = None


class StreamingSession:
    """One live perception run over a file, network stream, or pushed camera."""

    def __init__(
        self,
        config: HarnessConfig,
        client: WindowClient | GroundingClient,
        callbacks: SessionCallbacks,
        *,
        mode: str,
        video_stream: VideoFileStream | NetworkStreamAdapter | None = None,
        camera_source: PushSampledSource | None = None,
        source_id: str = "default",
        stream_session_id: str | None = None,
        archive: EventArchive | None = None,
    ) -> None:
        if mode not in {FILE_MODE, CAMERA_MODE, STREAM_MODE}:
            raise ValueError(f"unknown session mode: {mode}")
        if mode != CAMERA_MODE and video_stream is None:
            raise ValueError(f"{mode} mode requires a video_stream")
        if mode == CAMERA_MODE and camera_source is None:
            raise ValueError("camera mode requires a camera_source")
        if not source_id.strip():
            raise ValueError("source_id must not be empty")
        self.config = config
        self.client = client
        self.callbacks = callbacks
        self.mode = mode
        self.frame_crop_selector = FrameCropSelector(
            config.stream.frame_crop if mode == STREAM_MODE else "none",
        )
        self.video_stream = video_stream
        self.camera_source = camera_source
        self.source_id = source_id.strip()
        self.stream_session_id = stream_session_id or str(uuid4())
        self.archive = archive
        # Service-owned runs finish the archive after tracking and identity review.
        self.defer_archive_end = False
        self.person_registry = PersonRegistry(
            stream_session_id=self.stream_session_id, source_id=self.source_id,
        )
        face_client = getattr(client, "_stranger_client", client)
        self._ordered_face_client = (
            face_client if isinstance(face_client, FaceRecognitionSession) else None
        )
        bind_registry = getattr(face_client, "bind_person_registry", None)
        if bind_registry is not None:
            bind_registry(self.person_registry)
        memory_context = getattr(client, "memory_context", None)
        event_labels = tuple(getattr(client, "event_labels", ()))
        if event_labels:
            self.event_label = ",".join(event_labels)
        else:
            self.event_label = (
                UNKNOWN_FACE_EVENT_LABEL
                if memory_context is not None
                and memory_context.purpose == FACE_RECOGNITION_PURPOSE
                else client.settings.event_label
            )
        self._independent_branches = (
            getattr(client, "independent_live_branches", False) is True
        )
        self.organizer = ResultOrganizer(
            ",".join(label for label in event_labels if label != "stranger")
            if self._independent_branches else self.event_label
        )
        self._face_organizer = ResultOrganizer("stranger")
        self._face_pending: set[Future[Any]] = set()
        self._face_executor: ThreadPoolExecutor | None = None
        self._branch_diagnostics: dict[str, dict[str, Any]] = {}
        self._archive_result_windows: OrderedDict[str, str | int] = OrderedDict()
        # Live sources drop windows when all request slots are busy (the feed
        # cannot wait); offline runs block instead so every window is scored.
        self._drop_when_saturated = (
            video_stream.realtime if video_stream is not None else True
        )
        self._stop = threading.Event()
        self._drain_on_stop = False
        self._emit_open = True
        self._pending: set[Future[Any]] = set()
        self._pending_lock = threading.Lock()
        self._apply_lock = threading.Lock()
        self._result_sequence = 0
        self._started_wall: float | None = None
        self._thread: threading.Thread | None = None
        self._face_admission = (
            FaceAdmissionQueue(self._admit_ordered_face)
            if mode == STREAM_MODE and self._drop_when_saturated
            and self._ordered_face_client is not None
            else None
        )

    @classmethod
    def for_file(
        cls,
        config: HarnessConfig,
        client: WindowClient | GroundingClient,
        callbacks: SessionCallbacks,
        *,
        video_path: str | Path,
        start_seconds: float = 0.0,
        realtime: bool | None = None,
        source_id: str = "file",
        archive: EventArchive | None = None,
        stream_session_id: str | None = None,
    ) -> StreamingSession:
        use_realtime = config.video.realtime if realtime is None else realtime
        stream = VideoFileStream(
            video_path,
            # Keep source cadence sufficient for both inference and evidence.
            display_fps=max(
                config.display.fps, config.video.fps,
                config.events.fps if archive is not None else 0,
            ),
            playback_speed=config.video.playback_speed,
            start_seconds=start_seconds,
            realtime=use_realtime,
            input_observer=callbacks.on_source_input,
        )
        return cls(
            config,
            client,
            callbacks,
            mode=FILE_MODE,
            video_stream=stream,
            source_id=source_id,
            archive=archive,
            stream_session_id=stream_session_id,
        )

    @classmethod
    def for_stream(
        cls,
        config: HarnessConfig,
        client: WindowClient | GroundingClient,
        callbacks: SessionCallbacks,
        *,
        stream_url: str,
        whep_headers: Mapping[str, str] | None = None,
        source_id: str = "camera",
        archive: EventArchive | None = None,
        video_stream: Any | None = None,
    ) -> StreamingSession:
        stream = video_stream if video_stream is not None else NetworkStreamAdapter(
            stream_url,
            output_fps=max(
                config.display.fps, config.video.fps,
                config.events.fps if archive is not None else 0,
            ),
            whep_headers=whep_headers,
        )
        return cls(
            config,
            client,
            callbacks,
            mode=STREAM_MODE,
            video_stream=stream,
            source_id=source_id,
            archive=archive,
        )

    @classmethod
    def for_camera(
        cls,
        config: HarnessConfig,
        client: WindowClient | GroundingClient,
        callbacks: SessionCallbacks,
        *,
        source_id: str = "camera",
        archive: EventArchive | None = None,
    ) -> StreamingSession:
        return cls(
            config,
            client,
            callbacks,
            mode=CAMERA_MODE,
            camera_source=PushSampledSource(),
            source_id=source_id,
            archive=archive,
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("session already started")
        if self.archive is not None:
            self.archive.start_session(self.stream_session_id, self.source_id)
        self._thread = threading.Thread(
            target=self._run,
            name="streaming-session",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, drain: bool = False) -> None:
        self._drain_on_stop = drain
        if not drain:
            self._emit_open = False
        self._stop.set()
        if self.video_stream is not None:
            self.video_stream.stop()
        if self.camera_source is not None:
            self.camera_source.close()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def push_camera_frame(
        self,
        jpeg_bytes: bytes,
        *,
        width: int,
        height: int,
        timestamp_seconds: float | None = None,
        source_frame_index: int | None = None,
    ) -> bool:
        if self.camera_source is None:
            return False
        return self.camera_source.push_jpeg(
            jpeg_bytes,
            width=width,
            height=height,
            timestamp_seconds=timestamp_seconds,
            source_frame_index=source_frame_index,
        )

    # ------------------------------------------------------------------ run

    def _run(self) -> None:
        self._started_wall = time.monotonic()
        reason = {"file": "eof", "stream": "disconnected"}.get(self.mode, "stopped")
        error: str | None = None
        executor = ThreadPoolExecutor(
            max_workers=self.config.runtime.max_concurrent_requests,
            thread_name_prefix="analysis-window",
        )
        if self._independent_branches:
            self._face_executor = ThreadPoolExecutor(
                max_workers=self._face_concurrency(),
                thread_name_prefix="analysis-face",
            )
        try:
            windows = iter_sliding_windows(
                self._sampled_frames(),
                size_frames=self.config.sliding_window.size_frames,
                stride_frames=self.config.sliding_window.stride_frames,
                include_partial_final_window=(
                    self.config.sliding_window.include_partial_final_window
                ),
            )
            for window in windows:
                if self._stop.is_set() and not self._drain_on_stop:
                    break
                if not getattr(self.client, "scene_source_only", False):
                    self._schedule_window(window, executor)
            if self._stop.is_set():
                reason = "stopped"
        except StreamAdapterError as exc:
            LOGGER.warning("stream_failure code=%s", exc.code)
            reason, error = "error", exc.public_message
        except Exception:
            reason, error = "error", PUBLIC_ANALYSIS_ERROR
        finally:
            if self._face_admission is not None:
                try:
                    self._face_admission.close()
                except Exception:
                    reason, error = "error", PUBLIC_ANALYSIS_ERROR
            # Stopping capture must not discard accepted face/model requests.
            while self._has_pending() and (
                not self._stop.is_set() or self._drain_on_stop
            ):
                time.sleep(0.02)
            if self._stop.is_set() and error is None:
                reason = "stopped"
            self._emit_open = False
            executor.shutdown(wait=False, cancel_futures=True)
            if self._face_executor is not None:
                self._face_executor.shutdown(wait=False, cancel_futures=True)
            if self._ordered_face_client is not None:
                self._ordered_face_client.close(
                    wait=not self._stop.is_set() or self._drain_on_stop,
                )
            if self.archive is not None and not self.defer_archive_end:
                self.archive.end_session(self.stream_session_id, reason)
            if self.callbacks.on_end is not None:
                self.callbacks.on_end(reason, error)

    def _sampled_frames(self) -> Iterator[SampledFrame]:
        if self.mode == CAMERA_MODE:
            assert self.camera_source is not None
            yield from self.camera_source.frames()
            return

        assert self.video_stream is not None
        video = self.config.video
        display = self.config.display
        sample_period = 1.0 / video.fps
        next_sample_time = 0.0
        sample_index = 0
        for frame in self.frame_crop_selector.frames(
            self.video_stream.frames(), stopped=self._stop.is_set,
        ):
            if self._face_admission is not None:
                self._face_admission.raise_if_failed()
            # Decide before display, tracking, evidence and inference see any
            # frames, so every consumer uses one stable coordinate system.
            frame_bgr = frame.bgr
            presentation_jpeg_bytes: bytes | None = None
            if self.callbacks.on_display_frame is not None or self.archive is not None:
                presentation_jpeg_bytes, _, _ = encode_jpeg(
                    frame_bgr,
                    max_image_side=display.max_image_side,
                    quality=display.jpeg_quality,
                )
                if self.archive is not None:
                    self.archive.submit_frame(
                        presentation_jpeg_bytes, frame.timestamp_seconds, frame.index,
                        self.stream_session_id,
                    )
                if self.callbacks.on_display_frame is not None:
                    self.callbacks.on_display_frame(
                        presentation_jpeg_bytes, frame.timestamp_seconds, frame.index
                    )

            if frame.timestamp_seconds + 1e-9 < next_sample_time:
                continue
            while next_sample_time <= frame.timestamp_seconds + 1e-9:
                next_sample_time += sample_period

            jpeg_bytes, width, height = encode_jpeg(
                frame_bgr,
                max_image_side=video.max_image_side,
                quality=video.jpeg_quality,
            )
            source_height, source_width = frame_bgr.shape[:2]
            yield SampledFrame(
                sample_index=sample_index,
                source_frame_index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                jpeg_bytes=jpeg_bytes,
                width=width,
                height=height,
                source_width=int(source_width),
                source_height=int(source_height),
                observed_at=_utc_now(),
                presentation_jpeg_bytes=presentation_jpeg_bytes,
            )
            sample_index += 1
            if (
                video.max_sampled_frames is not None
                and sample_index >= video.max_sampled_frames
            ):
                return

    # -------------------------------------------------------------- windows

    def _schedule_window(
        self, window: FrameWindow, executor: ThreadPoolExecutor
    ) -> None:
        if self._independent_branches:
            # Face uses its durable ordered pool when supported. Legacy custom
            # branch clients retain the generic scheduler contract.
            assert self._face_executor is not None
            self._schedule_branch(window, self._face_executor, "stranger")
            self._schedule_branch(window, executor, "safety")
            return
        self._schedule_branch(window, executor, None)

    def _face_concurrency(self) -> int:
        return (
            1
            if getattr(self.client, "face_live_scheduler", "single_flight")
            == "single_flight"
            else self.config.runtime.max_concurrent_requests
        )

    def _schedule_branch(
        self,
        window: FrameWindow,
        executor: ThreadPoolExecutor,
        branch: str | None,
    ) -> None:
        if self._stop.is_set() and not self._drain_on_stop:
            return
        if self._ordered_face_client is not None and branch != "safety":
            self._schedule_ordered_face(window, branch=branch)
            return
        pending = self._face_pending if branch == "stranger" else self._pending
        organizer = self._face_organizer if branch == "stranger" else self.organizer
        capacity = (
            self._face_concurrency()
            if branch == "stranger"
            else self.config.runtime.max_concurrent_requests
        )
        while True:
            with self._pending_lock:
                saturated = len(pending) >= capacity
            if not saturated:
                break
            if self._drop_when_saturated:
                # A live feed cannot wait for the model; skip, don't queue.
                record, snapshot = organizer.note_dropped(
                    window, elapsed_seconds=self._elapsed()
                )
                self._emit_branch_state(record, snapshot, branch)
                self._archive_window(window, "dropped", branch=branch)
                return
            if self._stop.is_set() and not self._drain_on_stop:
                return
            time.sleep(0.02)

        if self._stop.is_set() and not self._drain_on_stop:
            return

        record, snapshot = organizer.note_submitted(
            window, elapsed_seconds=self._elapsed()
        )
        self._emit_branch_state(record, snapshot, branch)
        self._archive_window(window, "admitted", branch=branch)
        classify = self.client.classify_window
        if branch == "safety":
            classify = self.client.classify_safety_window
        elif branch == "stranger":
            classify = self.client.classify_stranger_window
        future = executor.submit(classify, window)
        with self._pending_lock:
            pending.add(future)
        future.add_done_callback(
            lambda done, window=window: self._on_window_done(
                done, window, branch=branch,
            )
        )

    def _schedule_ordered_face(self, window, *, branch):
        if self._face_admission is None:
            return self._admit_ordered_face(window, branch=branch)
        if not self._face_admission.offer(window, branch=branch):
            # Reject only this not-yet-admitted face sample. Capture/SAM continue;
            # durable face observations and their ordered identity commits survive.
            organizer = self._face_organizer if branch == "stranger" else self.organizer
            record, snapshot = organizer.note_dropped(
                window, elapsed_seconds=self._elapsed(),
            )
            self._emit_branch_state(record, snapshot, branch)
            self._archive_window(window, "dropped", branch=branch)
            log_timing(
                "face_ingress_backpressure",
                stream_session_id=trace_id(self.stream_session_id),
                source_frame_sequence=window.newest_frame.source_frame_index,
                outcome="not_admitted",
            )

    def _admit_ordered_face(self, window, *, branch):
        face = self._ordered_face_client
        organizer = self._face_organizer if branch == "stranger" else self.organizer
        pending = self._face_pending if branch == "stranger" else self._pending
        delivered = Future()

        def completed(future):
            try:
                self._on_ordered_face_done(future, branch=branch)
            except Exception as exc:
                delivered.set_exception(exc)
                raise
            else:
                delivered.set_result(None)

        def admitted(future):
            record, snapshot = organizer.note_submitted(
                window, elapsed_seconds=self._elapsed(),
            )
            self._emit_branch_state(record, snapshot, branch)
            self._archive_window(window, "admitted", branch=branch)
            with self._pending_lock:
                pending.add(future)
            # This closure retains no JPEG/window. The durable source is used
            # for an error; a successful response carries its exact source.
            future.add_done_callback(completed)

        try:
            face.submit_window(window, on_admitted=admitted)
            return delivered
        except Exception as exc:
            self._archive_window(window, "capacity_fault", branch=branch)
            log_timing(
                "face_pipeline_admission_failure",
                stream_session_id=trace_id(self.stream_session_id),
                source_frame_sequence=window.newest_frame.source_frame_index,
                error_type=type(exc).__name__, outcome="capacity_fault",
            )
            # Halt ingress explicitly. Previously admitted work still drains.
            if self.camera_source is not None:
                self.camera_source.close()
            raise

    def _on_ordered_face_done(self, future, *, branch):
        mapped = Future()
        window = None
        try:
            try:
                response = future.result()
                window = response.window
                merge = getattr(self.client, "_merge_responses", None)
                if merge is not None:
                    response = merge(
                        window, started=time.monotonic() - response.latency_seconds,
                        include_safety=False, include_stranger=True,
                        stranger_response=response,
                    )
                mapped.set_result(response)
            except Exception as exc:
                if window is None:
                    window = self._ordered_face_client.window_for(future)
                mapped.set_exception(exc)
            self._apply_window_future(mapped, window, branch=branch)
        finally:
            with self._pending_lock:
                pending = self._face_pending if branch == "stranger" else self._pending
                pending.discard(future)

    def _on_window_done(
        self, future: Future[Any], window: FrameWindow, *, branch: str | None = None,
    ) -> None:
        try:
            self._apply_window_future(future, window, branch=branch)
        finally:
            # Keep the future pending until apply and every delivery callback has
            # finished. The session thread must not close emission too early.
            with self._pending_lock:
                pending = self._face_pending if branch == "stranger" else self._pending
                pending.discard(future)

    def _log_window_completion(
        self, window: FrameWindow, response: Any, *,
        branch: str | None, exception: BaseException | None,
    ) -> None:
        """One real completed request, independent of object/event multiplicity."""
        try:
            face_only = branch is None and (
                isinstance(
                    self.client, (FaceRecognitionRuntime, FaceRecognitionSession)
                )
                or tuple(getattr(self.client, "event_labels", ())) == ("stranger",)
                or getattr(response.classification, "analysis_kind", None)
                == "face_recognition"
            )
            success = (
                exception is None and response.classification is not None
                and response.error is None
            )
            log_timing(
                "analysis_branch_timing",
                branch="insightface" if branch == "stranger" or face_only
                else "vlm_frame",
                stream_session_id=trace_id(self.stream_session_id),
                window_index=window.index,
                source_frame_sequence=window.newest_frame.source_frame_index,
                source_timestamp_seconds=window.newest_frame.timestamp_seconds,
                context_frame_sequences=[
                    frame.source_frame_index for frame in window.frames
                ],
                context_frames=len(window.frames),
                completed_requests=1,
                completed_observations=int(success),
                success=success,
                # An escaped exception has no measured response duration; zero
                # is the public fallback, not an actual instantaneous inference.
                model_seconds=response.latency_seconds if exception is None else None,
                error_type=type(exception).__name__ if exception is not None
                else "ResponseError" if not success else None,
            )
        except Exception:
            pass

    def _apply_window_future(
        self, future: Future[Any], window: FrameWindow, *, branch: str | None = None,
    ) -> None:
        if future.cancelled():
            self._archive_window(window, "cancelled", branch=branch)
            return
        exc = future.exception()
        if exc is not None:
            response = WindowResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=0.0,
                error=PUBLIC_ANALYSIS_ERROR,
            )
        else:
            response = future.result()
        self._log_window_completion(window, response, branch=branch, exception=exc)
        with self._apply_lock:
            organizer = self._face_organizer if branch == "stranger" else self.organizer
            record, snapshot = organizer.apply(
                response, elapsed_seconds=self._elapsed()
            )
            if branch is not None:
                self._branch_diagnostics[branch] = {
                    "window_index": window.index,
                    "source_frame_index": window.newest_frame.source_frame_index,
                    "source_timestamp_seconds": window.end_timestamp,
                    "latency_seconds": response.latency_seconds,
                    "applied": record.get("applied") is True,
                    "error": response.error,
                    **(
                        {"face": dict(response.face_diagnostics)}
                        if getattr(response, "face_diagnostics", None) is not None
                        else {}
                    ),
                    **(
                        {"classification": dict(response.classification_diagnostics)}
                        if getattr(response, "classification_diagnostics", None)
                        is not None
                        else {}
                    ),
                }
                record = {**record, "branch": branch}
                if branch == "stranger":
                    record["diagnostics"] = self._branch_diagnostics[branch]
                snapshot = self._branch_snapshot(branch, snapshot)
            vision_result = None
            analysis_result = None
            timeline_result = None
            analysis_frame_jpeg = None
            classification = response.classification
            applied = record.get("applied") is True
            if applied and classification is not None:
                # Preserve the original callback and result shape for existing
                # CLI/library consumers.
                vision_result = VisionResult(
                    event_label=str(public_safe_value(self.event_label)),
                    detected=classification.detected,
                    confidence=classification.confidence,
                    description=str(public_safe_value(classification.description)),
                    window_index=window.index,
                    window_start_seconds=window.start_timestamp,
                    window_end_seconds=window.end_timestamp,
                )
            if classification is not None:
                if (
                    self.callbacks.on_analysis_frame is not None
                    or self.callbacks.on_analysis_result is not None
                    or self.callbacks.on_timeline_result is not None
                    or self.archive is not None
                ):
                    grounded_objects: tuple[GroundedObject, ...]
                    judgments: tuple[EventJudgment, ...] | None
                    if isinstance(classification, MultiGroundedClassification):
                        object_records: list[GroundedObject] = []
                        judgment_records: list[EventJudgment] = []
                        for judgment in classification.judgments:
                            object_ids: list[str] = []
                            for item in judgment.objects:
                                object_id = f"object-{len(object_records) + 1}"
                                object_records.append(
                                    GroundedObject(
                                        object_id=object_id,
                                        label=str(public_safe_value(item.label)),
                                        bbox=item.bbox,
                                        confidence=item.confidence,
                                        attributes=public_safe_value(
                                            getattr(item, "attributes", {})
                                        ),
                                    )
                                )
                                object_ids.append(object_id)
                            judgment_records.append(
                                EventJudgment(
                                    event_label=judgment.event_label,
                                    detected=judgment.detected,
                                    confidence=judgment.confidence,
                                    description=str(
                                        public_safe_value(judgment.description)
                                    ),
                                    object_ids=tuple(object_ids),
                                    matched_memory_ids=judgment.matched_memory_ids,
                                )
                            )
                        for item in classification.supporting_objects:
                            object_records.append(
                                GroundedObject(
                                    object_id=f"object-{len(object_records) + 1}",
                                    label=str(public_safe_value(item.label)),
                                    bbox=item.bbox,
                                    confidence=item.confidence,
                                    attributes=public_safe_value(item.attributes),
                                )
                            )
                        grounded_objects = tuple(object_records)
                        judgments = tuple(judgment_records)
                    else:
                        grounded_objects = tuple(
                            GroundedObject(
                                object_id=f"object-{index}",
                                label=str(public_safe_value(item.label)),
                                bbox=item.bbox,
                                confidence=item.confidence,
                                attributes=public_safe_value(
                                    getattr(item, "attributes", {})
                                ),
                            )
                            for index, item in enumerate(
                                getattr(classification, "objects", ()), start=1
                            )
                        )
                        event_object_indexes = getattr(
                            classification, "event_object_indexes", None
                        )
                        if event_object_indexes is None:
                            selected_indexes = (
                                tuple(range(len(grounded_objects)))
                                if classification.detected
                                else ()
                            )
                        else:
                            selected_indexes = tuple(event_object_indexes)
                        event_object_ids = tuple(
                            grounded_objects[index].object_id
                            for index in selected_indexes
                        )
                        # The customer contract requires a bbox for a positive.
                        # Legacy callbacks still accept an ungrounded positive.
                        judgments = (
                            (
                                EventJudgment(
                                    event_label=self.event_label,
                                    detected=classification.detected,
                                    confidence=classification.confidence,
                                    description=str(
                                        public_safe_value(classification.description)
                                    ),
                                    object_ids=event_object_ids,
                                    matched_memory_ids=getattr(
                                        classification,
                                        "matched_memory_ids",
                                        (),
                                    ),
                                ),
                            )
                            if not classification.detected or event_object_ids
                            else None
                        )

                    if judgments is not None:
                        newest = window.newest_frame
                        frame_width, frame_height = newest.original_dimensions
                        self._result_sequence += 1
                        timeline_result = AnalysisResult(
                            message_id=str(uuid4()),
                            source_id=self.source_id,
                            stream_session_id=self.stream_session_id,
                            sequence=self._result_sequence,
                            emitted_at=_utc_now(),
                            frame=FrameReference(
                                source_frame_index=newest.source_frame_index,
                                timestamp_seconds=newest.timestamp_seconds,
                                observed_at=newest.observed_at or _utc_now(),
                                width=frame_width,
                                height=frame_height,
                            ),
                            judgments=judgments,
                            missing_event_labels=getattr(
                                classification, "missing_event_labels", ()
                            ),
                            objects=grounded_objects,
                            model_name=PUBLIC_MODEL_NAME,
                            model_latency_seconds=response.latency_seconds,
                            analysis_kind=getattr(
                                classification, "analysis_kind", "event"
                            ),
                        )
                        try:
                            timeline_result = self.person_registry.prepare_analysis(
                                timeline_result, observation_bundle(classification),
                            )
                        except Exception:
                            LOGGER.warning(
                                "could not associate observation with a person"
                            )
                        analysis_result = timeline_result
                        if applied:
                            # A late positive still reaches event notifications,
                            # but must not replace the current source image.
                            analysis_frame_jpeg = (
                                newest.presentation_jpeg_bytes or newest.jpeg_bytes
                            )
            self._emit(
                record,
                snapshot,
                vision_result=vision_result,
                analysis_result=analysis_result,
                timeline_result=timeline_result,
                analysis_frame_jpeg=analysis_frame_jpeg,
            )
            corrections = (
                self.person_registry.drain_reassociated_results()
                if timeline_result is not None and self._emit_open else ()
            )
            # A later face observation can resolve an earlier actor. Update the
            # private display state without republishing a new model verdict.
            for corrected in corrections:
                if self._emit_open and self.callbacks.on_timeline_result is not None:
                    _invoke_callback(
                        "private overlay association correction",
                        self.callbacks.on_timeline_result,
                        corrected,
                    )
            # Archive work is strictly downstream of the immediate result. These
            # methods only enqueue; disk, replay and encoding never run here.
            if self.archive is not None and self._emit_open:
                if timeline_result is not None:
                    window_id = self._archive_window_id(window, branch)
                    self._archive_result_windows[timeline_result.message_id] = window_id
                    while len(self._archive_result_windows) > 2048:
                        self._archive_result_windows.popitem(last=False)
                    accepted = self.archive.submit_result(
                        timeline_result, window_id=window_id,
                    )
                    self._archive_window(
                        window, "completed" if accepted else "dropped", branch=branch,
                    )
                    for corrected in corrections:
                        self.archive.submit_result(
                            corrected,
                            window_id=self._archive_result_windows.get(corrected.message_id),
                        )
                else:
                    self._archive_window(window, "failed", branch=branch)

    def _branch_snapshot(self, branch: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        safety = snapshot if branch == "safety" else self.organizer.snapshot()
        face = snapshot if branch == "stranger" else self._face_organizer.snapshot()
        errors = {**safety.get("branch_errors", {}), **face.get("branch_errors", {})}
        return {
            **safety,
            "event_label": self.event_label,
            "branch_errors": errors,
            "branch_error_details": {
                **safety.get("branch_error_details", {}),
                **face.get("branch_error_details", {}),
            },
            "last_error": safety.get("last_error") or face.get("last_error"),
            "branches": {
                "safety": {
                    **safety,
                    "last_completion": self._branch_diagnostics.get("safety"),
                },
                "stranger": {
                    **face,
                    "scheduler": getattr(
                        self.client, "face_live_scheduler", "single_flight",
                    ),
                    "last_completion": self._branch_diagnostics.get("stranger"),
                },
            },
        }

    def _emit_branch_state(
        self, record: dict[str, Any], snapshot: dict[str, Any], branch: str | None,
    ) -> None:
        if branch is not None:
            record = {**record, "branch": branch}
            snapshot = self._branch_snapshot(branch, snapshot)
        self._emit(record, snapshot)

    @staticmethod
    def _archive_window_id(window: FrameWindow, branch: str | None) -> str | int:
        return f"{branch}:{window.index}" if branch is not None else window.index

    def _archive_window(
        self, window: FrameWindow, outcome: str, *, branch: str | None = None,
    ) -> None:
        if self.archive is not None:
            labels = tuple(getattr(self.client, "event_labels", ())) or (
                self.event_label,
            )
            if branch == "stranger":
                labels = ("stranger",)
            elif branch == "safety":
                labels = tuple(label for label in labels if label != "stranger")
            self.archive.note_window(
                self.stream_session_id, self._archive_window_id(window, branch),
                window.end_timestamp, expected_labels=labels, outcome=outcome,
                source_frame_index=window.newest_frame.source_frame_index,
            )

    def _has_pending(self) -> bool:
        with self._pending_lock:
            return bool(self._pending or self._face_pending)

    def _elapsed(self) -> float:
        if self._started_wall is None:
            return 0.0
        return time.monotonic() - self._started_wall

    def _emit(
        self,
        record: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        vision_result: VisionResult | None = None,
        analysis_result: AnalysisResult | None = None,
        timeline_result: AnalysisResult | None = None,
        analysis_frame_jpeg: bytes | None = None,
    ) -> None:
        if not self._emit_open:
            return
        if self.callbacks.on_event is not None:
            _invoke_callback("event", self.callbacks.on_event, record)
        if self._emit_open and self.callbacks.on_state is not None:
            _invoke_callback("state", self.callbacks.on_state, snapshot)
        if (
            self._emit_open and vision_result is not None
            and self.callbacks.on_vision_result is not None
        ):
            _invoke_callback(
                "vision result",
                self.callbacks.on_vision_result,
                vision_result,
            )

        if (
            self._emit_open and timeline_result is not None
            and self.callbacks.on_timeline_result is not None
        ):
            _invoke_callback(
                "private overlay timeline",
                self.callbacks.on_timeline_result,
                timeline_result,
            )

        if (
            self._emit_open and analysis_result is not None
            and analysis_frame_jpeg is not None
            and self.callbacks.on_analysis_frame is not None
        ):
            _invoke_callback(
                "exact analysis frame",
                self.callbacks.on_analysis_frame,
                analysis_result,
                analysis_frame_jpeg,
            )
        if (
            self._emit_open and analysis_result is not None
            and self.callbacks.on_analysis_result is not None
        ):
            _invoke_callback(
                "analysis result",
                self.callbacks.on_analysis_result,
                analysis_result,
            )


def _invoke_callback(
    name: str,
    callback: Callable[..., None],
    *args: Any,
) -> None:
    """Keep an optional delivery sink from breaking the other outputs."""

    try:
        callback(*args)
    except Exception:
        LOGGER.error("%s callback failed", name)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# ----------------------------------------------------------------- headless


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--video",
        type=Path,
        help="video path (defaults to input.default_video in the config)",
    )
    source_group.add_argument(
        "--stream-url",
        help="RTSP, RTMP, or WHEP URL; overrides the configured stream source",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="seek to this video timestamp before streaming",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise streaming and windowing without calling the model",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="process as fast as possible instead of matching the wall clock",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write event JSONL here instead of standard output",
    )
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    stream_url: str | None = None
    whep_headers: Mapping[str, str] = {}
    if args.video is not None:
        video = args.video
    elif args.stream_url is not None:
        video = None
        stream_url = args.stream_url
    elif config.stream.url is not None:
        video = None
        stream_url = config.stream.url
        whep_headers = config.stream.whep_headers
    else:
        video = config.default_video
    if video is None and stream_url is None:
        raise SystemExit(
            "provide --video, --stream-url, or set stream.url/input.default_video"
        )

    output: TextIO = sys.stdout
    close_output = False
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("w", encoding="utf-8")
        close_output = True

    write_lock = threading.Lock()
    done = threading.Event()

    def emit(record: dict[str, Any]) -> None:
        safe = public_safe_value(record)
        assert isinstance(safe, dict)
        with write_lock:
            print(json.dumps(safe, ensure_ascii=False), file=output, flush=True)

    session: StreamingSession | None = None

    def on_end(reason: str, error: str | None) -> None:
        record: dict[str, Any] = {"type": "ended", "reason": reason}
        if error:
            record["error"] = public_error_message(error)
        if session is not None:
            record["final_state"] = session.organizer.snapshot()
        emit(record)
        done.set()

    if (
        config.memory is not None
        and config.memory.purpose == FACE_RECOGNITION_PURPOSE
        and config.inference.event_label != UNKNOWN_FACE_EVENT_LABEL
    ):
        config = replace(
            config,
            inference=replace(
                config.inference,
                event_label=UNKNOWN_FACE_EVENT_LABEL,
            ),
        )

    client = build_grounding_client(
        config.inference,
        config.model_api,
        memory_context=config.memory,
        dry_run=args.dry_run,
    )
    callbacks = SessionCallbacks(
        on_event=emit,
        on_end=on_end,
        on_vision_result=lambda result: emit(result.as_dict()),
    )
    archive = EventArchive(config.events)
    if stream_url is not None:
        session = StreamingSession.for_stream(
            config,
            client,
            callbacks,
            stream_url=stream_url,
            whep_headers=whep_headers,
            archive=archive,
        )
    else:
        assert video is not None
        session = StreamingSession.for_file(
            config,
            client,
            callbacks,
            video_path=video,
            start_seconds=args.start_seconds,
            realtime=False if args.no_realtime else None,
            archive=archive,
        )
    try:
        session.start()
        while not done.wait(0.2):
            pass
    except KeyboardInterrupt:
        session.stop()
        done.wait(5.0)
    finally:
        archive.close(timeout=15.0)
        if close_output:
            output.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI with a stable public error boundary (no raw traceback)."""

    try:
        return _main(argv)
    except Exception:
        record = {
            "type": "ended",
            "reason": "error",
            "error": PUBLIC_ANALYSIS_ERROR,
        }
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
