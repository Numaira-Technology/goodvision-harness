"""HTTP delivery service for the independent streaming harness."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import struct
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from .config import (
    DEFAULT_CONFIG_PATH,
    HarnessConfig,
    ServiceSettings,
    load_config,
)
from .contracts import AnalysisResult
from .delivery import WebhookDelivery
from .face_recognition import FaceRecognitionRuntime, FaceRecognizerConfig
from .grounding import build_grounding_client
from .openrouter import api_key_present
from .overlay import draw_overlay
from .pipeline import CAMERA_MODE, STREAM_MODE, SessionCallbacks, StreamingSession
from .reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    UNKNOWN_FACE_EVENT_LABEL,
)
from .result_bus import FrameBus, FramePacket, ResultBus
from .stream_adapter import parse_stream_source
from .visual_memory import (
    VisualMemoryRecord,
    VisualMemorySink,
)

LOGGER = logging.getLogger(__name__)
MJPEG_BOUNDARY = "goodvision-frame"
CAMERA_FRAME_HEADER = struct.Struct(">HH")
CAMERA_FRAME_MAX_BYTES = 2_000_000
EVENT_LABEL_MAX_CHARS = 80
PERSON_FIELD_MAX_CHARS = 120
FACE_ENROLLMENT_MAX_BYTES = 10_000_000
EVENT_ANALYSIS_MODE = "event"
FACE_ANALYSIS_MODE = "face"
ANALYSIS_MODES = frozenset({EVENT_ANALYSIS_MODE, FACE_ANALYSIS_MODE})


class HarnessRuntime:
    """Own one active stream or browser-camera session."""

    def __init__(
        self,
        config: HarnessConfig | None,
        *,
        config_error: str | None = None,
        memory_sink: VisualMemorySink | None = None,
        face_runtime: FaceRecognitionRuntime | None = None,
    ) -> None:
        try:
            dry_run = _env_bool("HARNESS_DRY_RUN", False)
        except ValueError as exc:
            dry_run = False
            config_error = config_error or f"configuration error: {exc}"
        self.config = config
        self.config_error = config_error
        self.dry_run = dry_run
        self.settings = config.service if config is not None else ServiceSettings()
        self.source_id = (
            os.environ.get("HARNESS_SOURCE_ID", "").strip() or self.settings.source_id
        )
        self.results = ResultBus(self.settings.result_history)
        self.frames = FrameBus()
        self.preview_frames = FrameBus()
        self._state_lock = RLock()
        self._control_lock = RLock()
        self._session: StreamingSession | None = None
        self._session_mode: str | None = None
        self._camera_owner: str | None = None
        self._camera_frame_sequence = 0
        self._camera_max_image_side = (
            config.video.max_image_side if config is not None else 0
        )
        self._session_started_monotonic: float | None = None
        self._generation = 0
        self._delivery: WebhookDelivery | None = None
        self._delivery_initialized = False
        self._memory_sink = memory_sink
        self._face_runtime = face_runtime
        self._last_frame_monotonic: float | None = None
        self._last_result_monotonic: float | None = None
        self._closed = False
        self._state: dict[str, Any] = {
            "service_started_at": _utc_now(),
            "source_id": self.source_id,
            "pipeline_state": (
                "configuration_error" if config_error else "idle"
            ),
            "input_mode": None,
            "analysis_mode": EVENT_ANALYSIS_MODE,
            "generation": 0,
            "ready": False,
            "last_frame_at": None,
            "last_result_at": None,
            "source_frames_received": 0,
            "analysis_frames_published": 0,
            "results_published": 0,
            "last_error": config_error,
            "end_reason": None,
            "stream": None,
            "stream_session_id": None,
            "event_label": (
                config.inference.event_label if config is not None else None
            ),
            "model": config.inference.model if config is not None else None,
            "dry_run": self.dry_run,
            "webhook_enabled": False,
            "memory_enabled": memory_sink is not None,
            "memory_records_accepted": 0,
            "memory_records_dropped": 0,
            "pipeline": None,
        }

    @classmethod
    def from_environment(cls) -> HarnessRuntime:
        config_path = Path(os.environ.get("HARNESS_CONFIG", str(DEFAULT_CONFIG_PATH)))
        try:
            config = load_config(config_path)
        except Exception as exc:
            LOGGER.exception("could not load harness configuration")
            return cls(None, config_error=f"configuration error: {exc}")
        return cls(config)

    def configured_stream_url(self) -> str:
        if self.config is None:
            return ""
        return (
            os.environ.get("RTSP_URL", "").strip()
            or os.environ.get("HARNESS_STREAM_URL", "").strip()
            or (self.config.stream.url or "")
        )

    def ui_info(self) -> dict[str, Any]:
        if self.config is None:
            return {
                "configured": False,
                "error": self.config_error or "configuration unavailable",
            }
        stream_url = self.configured_stream_url()
        protocol = None
        if stream_url:
            try:
                protocol = parse_stream_source(stream_url).protocol
            except ValueError:
                protocol = None
        return {
            "configured": True,
            "model": self.config.inference.model,
            "event_label": self.config.inference.event_label,
            "sample_fps": self.config.video.fps,
            "max_image_side": self.config.video.max_image_side,
            "jpeg_quality": self.config.video.jpeg_quality,
            "window": {
                "size_frames": self.config.sliding_window.size_frames,
                "stride_frames": self.config.sliding_window.stride_frames,
            },
            "max_concurrent_requests": self.config.runtime.max_concurrent_requests,
            "api_key_present": api_key_present(self.config.openrouter),
            "dry_run": self.dry_run,
            "configured_stream": bool(stream_url),
            "configured_stream_protocol": protocol,
            "analysis_modes": sorted(ANALYSIS_MODES),
            "reference_memory": (
                {
                    "purpose": self.config.memory.purpose,
                    "reference_count": len(self.config.memory.references),
                    "newest_frame_only": self.config.memory.newest_frame_only,
                }
                if self.config.memory is not None
                else None
            ),
            "face": {
                "model_name": self.config.face.model_name,
                "sample_fps": self.config.face.sample_fps,
                "max_image_side": self.config.face.max_image_side,
                "jpeg_quality": self.config.face.jpeg_quality,
                "threshold": self.config.face.threshold,
            },
        }

    def face_gallery(self) -> dict[str, Any]:
        """Return an embedding-free view of the local known-person gallery."""

        try:
            people = self._ensure_face_runtime().list_people()
        except Exception as exc:
            LOGGER.warning("could not open face gallery", exc_info=True)
            return {
                "ready": False,
                "people": [],
                "sample_count": 0,
                "error": str(exc),
            }
        public_people = [
            {
                "person_id": person.person_id,
                "name": person.name,
                "samples": person.sample_count,
                "sample_count": person.sample_count,
            }
            for person in people
        ]
        return {
            "ready": True,
            "people": public_people,
            "sample_count": sum(person.sample_count for person in people),
        }

    def enroll_face(
        self,
        person_id: str,
        name: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        person = self._ensure_face_runtime().enroll(person_id, name, image_bytes)
        return {
            "ready": True,
            "person": {
                "person_id": person.person_id,
                "name": person.name,
                "samples": person.sample_count,
                "sample_count": person.sample_count,
            },
        }

    def delete_face(self, person_id: str) -> bool:
        return self._ensure_face_runtime().delete_person(person_id)

    def _ensure_face_runtime(self) -> FaceRecognitionRuntime:
        if self.config is None:
            raise RuntimeError(self.config_error or "configuration unavailable")
        with self._control_lock:
            if self._face_runtime is None:
                face = self.config.face
                self._face_runtime = FaceRecognitionRuntime(
                    FaceRecognizerConfig(
                        model_name=face.model_name,
                        model_root=face.model_root,
                        allow_model_download=face.allow_model_download,
                        provider=face.provider,
                        det_size=face.det_size,
                        threshold=face.threshold,
                        min_face_size=face.min_face_size,
                        gallery_path=face.gallery_path,
                    )
                )
            return self._face_runtime

    def start(self) -> None:
        """Initialize delivery and optionally start the configured network stream."""

        if self.config is None or self.config_error is not None:
            return
        try:
            with self._control_lock:
                self._ensure_delivery()
            autostart = _env_bool("HARNESS_AUTOSTART", True)
            stream_url = self.configured_stream_url()
            if autostart and stream_url:
                self.start_stream(stream_url)
                return
            self._update_state(
                pipeline_state="idle",
                last_error=None,
                webhook_enabled=self._delivery is not None,
            )
        except Exception as exc:
            LOGGER.exception("could not start harness runtime")
            self._update_state(
                pipeline_state="error",
                last_error=str(exc),
                ready=False,
            )

    def start_stream(
        self,
        stream_url: str | None = None,
        *,
        dry_run: bool | None = None,
        event_label: str | None = None,
        analysis_mode: str = EVENT_ANALYSIS_MODE,
    ) -> dict[str, Any]:
        url = (stream_url or "").strip() or self.configured_stream_url()
        if not url:
            raise ValueError("provide an RTSP, RTMP, or WebRTC/WHEP stream URL")
        source = parse_stream_source(url)
        return self._start_session(
            STREAM_MODE,
            stream_url=url,
            stream_protocol=source.protocol,
            stream_host=source.host,
            dry_run=dry_run,
            event_label=event_label,
            analysis_mode=analysis_mode,
        )

    def start_camera(
        self,
        *,
        dry_run: bool | None = None,
        event_label: str | None = None,
        analysis_mode: str = EVENT_ANALYSIS_MODE,
        owner: str | None = None,
    ) -> dict[str, Any]:
        return self._start_session(
            CAMERA_MODE,
            dry_run=dry_run,
            event_label=event_label,
            analysis_mode=analysis_mode,
            camera_owner=owner,
        )

    def _start_session(
        self,
        mode: str,
        *,
        stream_url: str | None = None,
        stream_protocol: str | None = None,
        stream_host: str | None = None,
        dry_run: bool | None = None,
        event_label: str | None = None,
        analysis_mode: str = EVENT_ANALYSIS_MODE,
        camera_owner: str | None = None,
    ) -> dict[str, Any]:
        if self.config is None or self.config_error is not None:
            raise RuntimeError(self.config_error or "configuration unavailable")
        if mode not in {CAMERA_MODE, STREAM_MODE}:
            raise ValueError(f"unsupported input mode: {mode}")

        selected_analysis = str(analysis_mode).strip().lower()
        if selected_analysis not in ANALYSIS_MODES:
            raise ValueError("analysis_mode must be event or face")

        if selected_analysis == FACE_ANALYSIS_MODE:
            label = UNKNOWN_FACE_EVENT_LABEL
            use_dry_run = False
        else:
            face_reference_memory = (
                self.config.memory is not None
                and self.config.memory.purpose == FACE_RECOGNITION_PURPOSE
            )
            if face_reference_memory:
                label = UNKNOWN_FACE_EVENT_LABEL
            else:
                label = (event_label or self.config.inference.event_label).strip()
                if not label:
                    label = self.config.inference.event_label
            label = label[:EVENT_LABEL_MAX_CHARS]
            use_dry_run = self.dry_run if dry_run is None else bool(dry_run)

        with self._control_lock:
            if self._closed:
                raise RuntimeError("harness service is stopping")
            if self._session is not None and self._session.running:
                raise RuntimeError(
                    "a harness session is already running; stop it first"
                )
            self._ensure_delivery()

            if selected_analysis == FACE_ANALYSIS_MODE:
                face_runtime = self._ensure_face_runtime()
                face = self.config.face
                session_config = replace(
                    self.config,
                    video=replace(
                        self.config.video,
                        fps=face.sample_fps,
                        max_image_side=face.max_image_side,
                        jpeg_quality=face.jpeg_quality,
                    ),
                    sliding_window=replace(
                        self.config.sliding_window,
                        size_frames=1,
                        stride_frames=1,
                        include_partial_final_window=False,
                    ),
                    inference=face_runtime.settings,
                    runtime=replace(
                        self.config.runtime,
                        max_concurrent_requests=1,
                    ),
                )
                client = face_runtime
            else:
                session_config = self.config
                if label != self.config.inference.event_label:
                    session_config = replace(
                        self.config,
                        inference=replace(self.config.inference, event_label=label),
                    )
                client = build_grounding_client(
                    session_config.inference,
                    session_config.openrouter,
                    memory_context=session_config.memory,
                    dry_run=use_dry_run,
                )
            generation = self._generation + 1
            callbacks = SessionCallbacks(
                on_display_frame=(
                    (
                        lambda jpeg, timestamp, sequence: self._guarded_callback(
                            generation,
                            self._on_display_frame,
                            jpeg,
                            timestamp,
                            sequence,
                        )
                    )
                    if mode == STREAM_MODE
                    else None
                ),
                on_state=lambda state: self._guarded_callback(
                    generation, self._on_pipeline_state, state
                ),
                on_end=lambda reason, error: self._guarded_callback(
                    generation, self._on_end, reason, error
                ),
                on_analysis_frame=lambda result, jpeg: self._guarded_callback(
                    generation, self._on_analysis_frame, result, jpeg
                ),
                on_analysis_result=lambda result: self._guarded_callback(
                    generation, self._on_analysis_result, result
                ),
            )
            if mode == STREAM_MODE:
                assert stream_url is not None
                session = StreamingSession.for_stream(
                    session_config,
                    client,
                    callbacks,
                    stream_url=stream_url,
                    whep_headers=session_config.stream.whep_headers,
                    source_id=self.source_id,
                )
            else:
                session = StreamingSession.for_camera(
                    session_config,
                    client,
                    callbacks,
                    source_id=self.source_id,
                )

            self.results.reset()
            self.frames.reset()
            self.preview_frames.reset()
            with self._state_lock:
                self._generation = generation
                self._session = session
                self._session_mode = mode
                self._camera_owner = camera_owner if mode == CAMERA_MODE else None
                self._camera_frame_sequence = 0
                self._camera_max_image_side = session_config.video.max_image_side
                self._session_started_monotonic = time.monotonic()
                self._last_frame_monotonic = None
                self._last_result_monotonic = None
                self._state.update(
                    {
                        "pipeline_state": "starting",
                        "input_mode": mode,
                        "analysis_mode": selected_analysis,
                        "generation": generation,
                        "last_frame_at": None,
                        "last_result_at": None,
                        "source_frames_received": 0,
                        "analysis_frames_published": 0,
                        "results_published": 0,
                        "last_error": None,
                        "end_reason": None,
                        "stream": (
                            {
                                "protocol": stream_protocol,
                                "host": stream_host,
                            }
                            if mode == STREAM_MODE
                            else None
                        ),
                        "stream_session_id": session.stream_session_id,
                        "event_label": label,
                        "model": session_config.inference.model,
                        "dry_run": use_dry_run,
                        "webhook_enabled": self._delivery is not None,
                        "pipeline": None,
                    }
                )
            try:
                session.start()
            except Exception:
                with self._state_lock:
                    self._session = None
                    self._session_mode = None
                    self._camera_owner = None
                    self._state.update(
                        {
                            "pipeline_state": "error",
                            "input_mode": None,
                            "last_error": "could not start streaming session",
                        }
                    )
                raise

            response: dict[str, Any] = {
                "type": "session",
                "mode": mode,
                "analysis_mode": selected_analysis,
                "generation": generation,
                "stream_session_id": session.stream_session_id,
                "event_label": label,
                "model": session_config.inference.model,
                "dry_run": use_dry_run,
            }
            if stream_protocol is not None:
                response["stream_protocol"] = stream_protocol
            if (
                selected_analysis == EVENT_ANALYSIS_MODE
                and session_config.memory is not None
            ):
                response["reference_memory"] = {
                    "purpose": session_config.memory.purpose,
                    "reference_count": len(session_config.memory.references),
                    "newest_frame_only": session_config.memory.newest_frame_only,
                    "memory_ids": list(session_config.memory.memory_ids),
                }
            return response

    def stop_session(self, *, camera_owner: str | None = None) -> bool:
        """Stop the current run but keep the service available for another."""

        with self._control_lock:
            with self._state_lock:
                if (
                    camera_owner is not None
                    and (
                        self._session_mode != CAMERA_MODE
                        or self._camera_owner != camera_owner
                    )
                ):
                    return False
                session = self._session
                if session is None:
                    self._state.update(
                        {"pipeline_state": "idle", "input_mode": None, "ready": False}
                    )
                    return False
                self._generation += 1
                self._state.update(
                    {
                        "pipeline_state": "stopping",
                        "generation": self._generation,
                        "ready": False,
                    }
                )

            session.stop()
            session.join(timeout=5.0)
            if session.running:
                self._update_state(
                    pipeline_state="error",
                    last_error="the previous video session did not stop in time",
                )
                return False

            with self._state_lock:
                self._session = None
                self._session_mode = None
                self._camera_owner = None
                self._session_started_monotonic = None
                self._camera_max_image_side = self.config.video.max_image_side
                self._state.update(
                    {
                        "pipeline_state": "idle",
                        "input_mode": None,
                        "ready": False,
                        "stream": None,
                    }
                )
            return True

    def stop_camera_owned_by(self, owner: str) -> None:
        self.stop_session(camera_owner=owner)

    def stop(self) -> None:
        with self._control_lock:
            self._closed = True
        self.stop_session()
        if self._delivery is not None:
            self._delivery.close(timeout_seconds=5.0)
        self._update_state(pipeline_state="stopped", ready=False)

    def push_camera_frame(
        self,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        owner: str | None = None,
    ) -> bool:
        if self.config is None:
            return False
        if not jpeg or len(jpeg) > CAMERA_FRAME_MAX_BYTES:
            raise ValueError("camera JPEG is empty or too large")
        if width < 1 or height < 1:
            raise ValueError("camera frame dimensions must be positive")
        max_image_side = self._camera_max_image_side
        if max(width, height) > max_image_side:
            raise ValueError(
                f"camera frame exceeds {max_image_side}px"
            )
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("camera payload is not a valid JPEG")
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (width, height):
            raise ValueError("camera JPEG dimensions do not match its header")

        with self._control_lock:
            session = self._session
            if (
                session is None
                or self._session_mode != CAMERA_MODE
                or not session.running
                or (self._camera_owner is not None and owner != self._camera_owner)
            ):
                return False
            accepted = session.push_camera_frame(jpeg, width=width, height=height)
            if not accepted:
                return False
            sequence = self._camera_frame_sequence
            self._camera_frame_sequence += 1
            started = self._session_started_monotonic or time.monotonic()
            timestamp = max(0.0, time.monotonic() - started)
            generation = self._generation
            self._guarded_callback(
                generation,
                self._on_display_frame,
                jpeg,
                timestamp,
                sequence,
            )
            return True

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            payload = copy.deepcopy(self._state)
        ready, reason = self.readiness()
        payload["ready"] = ready
        payload["readiness_reason"] = reason
        if self._delivery is not None:
            payload["webhook"] = self._delivery.stats()
        return payload

    def readiness(self) -> tuple[bool, str]:
        with self._state_lock:
            state = self._state["pipeline_state"]
            last_frame = self._last_frame_monotonic
            last_result = self._last_result_monotonic
            session = self._session
        if state in {
            "configuration_error",
            "error",
            "stopped",
            "idle",
            "ended",
            "stopping",
        }:
            return False, str(state)
        if session is None or not session.running:
            return False, "pipeline_not_running"
        if last_frame is None:
            return False, "waiting_for_first_frame"
        now = time.monotonic()
        if now - last_frame > self.settings.ready_max_frame_age_seconds:
            return False, "frame_stale"
        if last_result is None:
            return False, "waiting_for_first_analysis"
        if now - last_result > self.settings.ready_max_result_age_seconds:
            return False, "analysis_stale"
        return True, "ready"

    def _ensure_delivery(self) -> None:
        if self._delivery_initialized:
            return
        self._delivery = self._build_delivery()
        if self._delivery is not None:
            self._delivery.start()
        self._delivery_initialized = True

    def _build_delivery(self) -> WebhookDelivery | None:
        assert self.config is not None
        webhook = self.settings.webhook
        url = os.environ.get("RESULT_WEBHOOK_URL", "").strip() or webhook.url
        if not url:
            return None
        secret = os.environ.get(webhook.secret_env)
        return WebhookDelivery(
            url,
            secret=secret,
            timeout_seconds=webhook.timeout_seconds,
            max_retries=webhook.max_retries,
            retry_backoff_seconds=webhook.retry_backoff_seconds,
            queue_size=webhook.queue_size,
        )

    def _guarded_callback(
        self,
        generation: int,
        callback: Callable[..., None],
        *args: Any,
    ) -> None:
        with self._state_lock:
            if generation != self._generation:
                return
        callback(*args)

    def _on_analysis_frame(self, result: AnalysisResult, jpeg: bytes) -> None:
        source_jpeg = bytes(jpeg)
        if self._memory_sink is not None:
            try:
                accepted = self._memory_sink.submit(
                    VisualMemoryRecord(
                        observation=result.as_dict(),
                        evidence_jpeg=source_jpeg,
                    )
                )
            except Exception:
                LOGGER.warning(
                    "visual memory sink rejected an observation",
                    exc_info=True,
                )
                accepted = False
            memory_key = (
                "memory_records_accepted" if accepted else "memory_records_dropped"
            )
            self._update_state(**{memory_key: int(self._state[memory_key]) + 1})
        try:
            rendered = _render_analysis_jpeg(
                source_jpeg,
                result,
                quality=self.settings.overlay_jpeg_quality,
            )
        except (cv2.error, TypeError, ValueError):
            LOGGER.warning("could not render analysis overlay", exc_info=True)
            rendered = source_jpeg
        self.frames.publish(
            FramePacket(
                jpeg=rendered,
                media_timestamp=result.frame.timestamp_seconds,
                frame_sequence=result.frame.source_frame_index,
                observed_at=result.frame.observed_at,
                analysis_message_id=result.message_id,
            )
        )
        frame_count = int(self._state["analysis_frames_published"]) + 1
        self._update_state(
            analysis_frames_published=frame_count,
            last_analysis_frame_at=_utc_now(),
        )

    def _on_analysis_result(self, result: AnalysisResult) -> None:
        payload = result.as_dict()
        self.results.publish(payload)
        result_count = int(self._state["results_published"]) + 1
        self._last_result_monotonic = time.monotonic()
        self._update_state(
            last_result_at=_utc_now(),
            results_published=result_count,
            stream_session_id=result.stream_session_id,
        )
        if self._delivery is not None:
            self._delivery.submit(payload)

    def _on_display_frame(
        self,
        jpeg: bytes,
        media_timestamp: float,
        frame_sequence: int,
    ) -> None:
        observed_at = _utc_now()
        self.preview_frames.publish(
            FramePacket(
                jpeg=bytes(jpeg),
                media_timestamp=float(media_timestamp),
                frame_sequence=int(frame_sequence),
                observed_at=observed_at,
            )
        )
        frame_count = int(self._state["source_frames_received"]) + 1
        self._last_frame_monotonic = time.monotonic()
        self._update_state(
            pipeline_state="running",
            last_frame_at=observed_at,
            source_frames_received=frame_count,
            last_media_timestamp=round(float(media_timestamp), 3),
            last_frame_sequence=int(frame_sequence),
        )

    def _on_pipeline_state(self, state: dict[str, Any]) -> None:
        values: dict[str, Any] = {"pipeline": copy.deepcopy(state)}
        pipeline_error = state.get("last_error")
        if pipeline_error:
            values["last_error"] = str(pipeline_error)
        self._update_state(**values)

    def _on_end(self, reason: str, error: str | None) -> None:
        next_state = "error" if error else "ended"
        self._update_state(
            pipeline_state=next_state,
            ready=False,
            end_reason=reason,
            last_error=error,
        )

    def _update_state(self, **values: Any) -> None:
        with self._state_lock:
            self._state.update(values)


def _render_analysis_jpeg(
    jpeg: bytes,
    analysis: AnalysisResult,
    *,
    quality: int,
) -> bytes:
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("display callback did not contain a valid JPEG")
    judgment = analysis.judgments[0]
    if analysis.analysis_kind == "face_recognition":
        objects = analysis.objects
    else:
        referenced = set(judgment.object_ids)
        objects = tuple(
            item for item in analysis.objects if item.object_id in referenced
        )
    rendered = draw_overlay(
        image,
        event_label=judgment.event_label,
        detected=judgment.detected,
        confidence=judgment.confidence,
        description=judgment.description,
        objects=objects,
        analysis_kind=analysis.analysis_kind,
    )
    ok, output = cv2.imencode(
        ".jpg",
        rendered,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise ValueError("could not encode annotated JPEG")
    return output.tobytes()


def create_app(
    runtime_factory: Callable[[], HarnessRuntime] | None = None,
) -> FastAPI:
    factory = runtime_factory or HarnessRuntime.from_environment

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = factory()
        application.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    application = FastAPI(
        title="Goodvision Streaming Harness",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    @application.get("/v1/ui/info")
    async def ui_info(request: Request) -> dict[str, Any]:
        return _runtime(request).ui_info()

    @application.websocket("/ws")
    async def control_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime: HarnessRuntime = websocket.app.state.runtime
        owner = str(uuid4())
        await websocket.send_json({"type": "hello", **runtime.ui_info()})
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text_message = message.get("text")
                if text_message is not None:
                    try:
                        payload = json.loads(text_message)
                    except json.JSONDecodeError:
                        await websocket.send_json(
                            {"type": "error", "message": "control message must be JSON"}
                        )
                        continue
                    if not isinstance(payload, dict):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "control message must be an object",
                            }
                        )
                        continue
                    command = payload.get("type")
                    if command == "start":
                        mode = str(payload.get("mode") or "")
                        analysis_mode = str(
                            payload.get("analysis_mode") or EVENT_ANALYSIS_MODE
                        )
                        requested_dry_run = payload.get("dry_run")
                        if requested_dry_run is not None and not isinstance(
                            requested_dry_run, bool
                        ):
                            await websocket.send_json(
                                {"type": "error", "message": "dry_run must be boolean"}
                            )
                            continue
                        event_label = str(payload.get("event_label") or "").strip()
                        try:
                            if mode == CAMERA_MODE:
                                response = await asyncio.to_thread(
                                    runtime.start_camera,
                                    dry_run=requested_dry_run,
                                    event_label=event_label,
                                    analysis_mode=analysis_mode,
                                    owner=owner,
                                )
                            elif mode == STREAM_MODE:
                                response = await asyncio.to_thread(
                                    runtime.start_stream,
                                    str(payload.get("stream_url") or ""),
                                    dry_run=requested_dry_run,
                                    event_label=event_label,
                                    analysis_mode=analysis_mode,
                                )
                            else:
                                raise ValueError("mode must be camera or stream")
                        except Exception as exc:
                            await websocket.send_json(
                                {"type": "error", "message": str(exc)}
                            )
                        else:
                            await websocket.send_json(response)
                    elif command == "stop":
                        await asyncio.to_thread(runtime.stop_session)
                        await websocket.send_json(
                            {"type": "ended", "reason": "stopped", "error": None}
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"unknown command: {command!r}",
                            }
                        )
                    continue

                data = message.get("bytes")
                if data is None:
                    continue
                if len(data) <= CAMERA_FRAME_HEADER.size:
                    await websocket.send_json(
                        {"type": "error", "message": "camera frame is truncated"}
                    )
                    continue
                width, height = CAMERA_FRAME_HEADER.unpack_from(data)
                try:
                    await asyncio.to_thread(
                        runtime.push_camera_frame,
                        data[CAMERA_FRAME_HEADER.size :],
                        width=width,
                        height=height,
                        owner=owner,
                    )
                except ValueError as exc:
                    await websocket.send_json(
                        {"type": "error", "message": str(exc)}
                    )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await asyncio.to_thread(runtime.stop_camera_owned_by, owner)

    @application.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        ready, reason = runtime.readiness()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "reason": reason},
            status_code=200 if ready else 503,
        )

    @application.get("/v1/status")
    async def status(request: Request) -> dict[str, Any]:
        return _runtime(request).status()

    @application.get("/v1/faces/gallery")
    async def face_gallery(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_runtime(request).face_gallery)

    @application.post("/v1/faces/gallery/{person_id}")
    async def enroll_face(
        person_id: str,
        request: Request,
        name: str = Query(min_length=1, max_length=PERSON_FIELD_MAX_CHARS),
    ) -> dict[str, Any]:
        rendered_id = person_id.strip()
        if not rendered_id or len(rendered_id) > PERSON_FIELD_MAX_CHARS:
            raise HTTPException(status_code=400, detail="invalid person_id")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > FACE_ENROLLMENT_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="enrollment image is too large",
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="invalid Content-Length",
                ) from None
        image_buffer = bytearray()
        async for chunk in request.stream():
            if len(image_buffer) + len(chunk) > FACE_ENROLLMENT_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="enrollment image is too large",
                )
            image_buffer.extend(chunk)
        image_bytes = bytes(image_buffer)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="enrollment image is empty")
        try:
            return await asyncio.to_thread(
                _runtime(request).enroll_face,
                rendered_id,
                name.strip(),
                image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.delete("/v1/faces/gallery/{person_id}")
    async def delete_face(person_id: str, request: Request) -> dict[str, Any]:
        try:
            deleted = await asyncio.to_thread(
                _runtime(request).delete_face,
                person_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="person not found")
        return {"deleted": True, "person_id": person_id}

    @application.get("/v1/results/latest")
    async def latest_result(request: Request) -> JSONResponse:
        record = _runtime(request).results.latest()
        if record is None:
            raise HTTPException(status_code=404, detail="no analysis result yet")
        return JSONResponse(
            record.payload,
            headers={
                "Cache-Control": "no-store",
                "X-Result-Cursor": str(record.cursor),
                "X-Result-Stream": _runtime(request).results.epoch,
            },
        )

    @application.get("/v1/results/events")
    async def result_events(
        request: Request,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        runtime = _runtime(request)
        header_cursor = request.headers.get("last-event-id")
        cursor = after
        if header_cursor and after == 0:
            epoch, separator, raw_cursor = header_cursor.rpartition(":")
            if separator and epoch == runtime.results.epoch:
                try:
                    cursor = max(0, int(raw_cursor))
                except ValueError:
                    cursor = 0
            else:
                cursor = 0

        async def generate() -> AsyncIterator[bytes]:
            current = cursor
            while not await request.is_disconnected():
                records = await asyncio.to_thread(
                    runtime.results.records_after,
                    current,
                    timeout_seconds=15.0,
                )
                if not records:
                    yield b": heartbeat\n\n"
                    continue
                for record in records:
                    current = record.cursor
                    data = json.dumps(
                        record.payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {runtime.results.epoch}:{record.cursor}\n"
                        f"event: analysis_result\ndata: {data}\n\n"
                    ).encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/v1/video/latest.jpg")
    async def latest_frame(request: Request) -> Response:
        return _snapshot_response(
            _runtime(request).frames,
            missing_detail="no analyzed frame yet",
        )

    @application.get("/v1/video/stream.mjpeg")
    async def video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).frames)

    @application.get("/v1/video/live/latest.jpg")
    async def latest_live_frame(request: Request) -> Response:
        return _snapshot_response(
            _runtime(request).preview_frames,
            missing_detail="no live source frame yet",
        )

    @application.get("/v1/video/live/stream.mjpeg")
    async def live_video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).preview_frames)

    return application


def _runtime(request: Request) -> HarnessRuntime:
    return request.app.state.runtime


def _frame_headers(packet: FramePacket) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-store",
        "X-Frame-Sequence": str(packet.frame_sequence),
        "X-Media-Timestamp": f"{packet.media_timestamp:.3f}",
        "X-Observed-At": packet.observed_at,
    }
    if packet.analysis_message_id is not None:
        headers["X-Analysis-Message-Id"] = packet.analysis_message_id
    return headers


def _snapshot_response(bus: FrameBus, *, missing_detail: str) -> Response:
    latest = bus.latest()
    if latest is None:
        raise HTTPException(status_code=404, detail=missing_detail)
    _, packet = latest
    return Response(
        packet.jpeg,
        media_type="image/jpeg",
        headers=_frame_headers(packet),
    )


def _mjpeg_response(request: Request, bus: FrameBus) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        version = 0
        latest = bus.latest()
        if latest is not None:
            version, packet = latest
            yield _mjpeg_part(packet)
        while not await request.is_disconnected():
            update = await asyncio.to_thread(
                bus.wait_after,
                version,
                timeout_seconds=15.0,
            )
            if update is None:
                continue
            version, packet = update
            yield _mjpeg_part(packet)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


def _mjpeg_part(packet: FramePacket) -> bytes:
    analysis_header = ""
    if packet.analysis_message_id is not None:
        analysis_header = f"X-Analysis-Message-Id: {packet.analysis_message_id}\r\n"
    headers = (
        f"--{MJPEG_BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(packet.jpeg)}\r\n"
        f"X-Frame-Sequence: {packet.frame_sequence}\r\n"
        f"X-Media-Timestamp: {packet.media_timestamp:.3f}\r\n"
        f"X-Observed-At: {packet.observed_at}\r\n"
        f"{analysis_header}\r\n"
    ).encode("ascii")
    return headers + packet.jpeg + b"\r\n"


def _utc_now() -> str:
    from datetime import datetime

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    rendered = value.strip().lower()
    if rendered in {"1", "true", "yes", "on"}:
        return True
    if rendered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def main() -> None:
    host = os.environ.get("HARNESS_HOST", "0.0.0.0")
    port = int(os.environ.get("HARNESS_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


app = create_app()


if __name__ == "__main__":
    main()
