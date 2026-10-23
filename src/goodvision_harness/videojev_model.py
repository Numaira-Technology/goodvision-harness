"""Native VideoJev choice scoring on the existing independent event branch."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import time
from threading import RLock
from urllib.parse import urlparse

import httpx
import cv2
import numpy as np
from PIL import Image

from .diagnostics import note
from .model_api import ModelAPIError
from .scene_inspection import _validate_job
from .scene_model import _json
from .videojev_contract import MODEL, parse_choices, scene_question_template
from .video_to_stream import encode_jpeg
from .vlm_independent_model import IndependentVerdict
from .vlm_transport import create_vlm_http_client

_MAX_RESPONSE_BYTES = 256 * 1024
_MAX_REQUEST_BYTES = 23 * 1024 * 1024
NATIVE_SHARED_ERRORS = frozenset((
    "cuda_shared_failure", "cuda_recovery_failed", "result_log_failure",
    "gpu_worker_stopped", "cpu_worker_stopped",
))
_NATIVE_SERVICE_ERRORS = NATIVE_SHARED_ERRORS | frozenset((
    "invalid_response", "preprocess_failed", "window_failed", "cuda_out_of_memory",
    "queue_full", "superseded", "expired_before_preprocess", "expired_before_gpu",
    "client_timeout_before_start", "request_deadline", "service_unavailable",
    "service_shutdown", "cancelled_before_start", "template_missing",
))
_SERVICE_METRICS = frozenset((
    "preprocess_seconds", "visual_encode_seconds", "decision_seconds",
    "model_pipeline_seconds", "gpu_pipeline_seconds", "server_queue_seconds",
    "cpu_queue_seconds", "gpu_queue_seconds", "server_preprocess_seconds",
    "server_gpu_seconds", "server_elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
    "input_frame_count", "internal_temporal_padding_frames", "visual_tokens",
    "native_temporal_fps", "context", "max_pixels", "source_width",
    "source_height", "processed_width", "processed_height", "processor_max_pixels",
    "native_group_count", "native_input_tokens", "question_group_prepare_seconds",
    "gpu_scoring_seconds",
    "forward_input_tokens", "prefix_tokens", "prefix_cache_hits", "prefix_cache_misses",
))


class VideoJevInspector:
    """One pooled local HTTP client per Harness event worker, without retries.

    Scene choices are not person grounding, generated JSON, or temporal decoding.
    An HTTP timeout does not cancel already-started work in the GPU service.
    """

    def __init__(self, config):
        if (
            config.model_settings.provider != "local"
            or config.model != MODEL
            or config.output_mode != "events"
        ):
            raise ValueError("VideoJev requires its explicit local events route")
        settings = config.model_settings
        endpoint = urlparse(settings.endpoint)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("invalid VideoJev service endpoint")
        if not math.isfinite(settings.timeout_seconds):
            raise ValueError("VideoJev requires a finite request deadline")
        self.config = config
        self.settings = settings
        self._lock = RLock()
        self._loop = None
        self._http = None
        self._question_template = scene_question_template(
            labels=config.labels, sensitivities=dict(config.sensitivities),
        )
        self._expected = tuple(
            {"id": row["id"], "options": row["options"]}
            for row in self._question_template
        )
        self._template_registered = False
        self.last_http_timing = {}
        self.last_inspection_timing = {}
        self.last_response_diagnostics = {}

    def close(self):
        with self._lock:
            if self._loop is not None:
                try:
                    if self._http is not None:
                        self._loop.run_until_complete(self._http.aclose())
                finally:
                    self._http = None
                    self._loop.close()
                    self._loop = None

    def _ensure_transport(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = create_vlm_http_client(trust_env=False)

    def _payload(self, job):
        _validate_job(job)
        # The native model uses uniform FPS for positional encoding. Actual
        # source timestamps remain separate; irregular input is not retimed.
        delta = self.config.observation_frame_delta_seconds
        fps = (
            1.0 / delta if delta is not None
            else self.config.candidate_fps
            if self.config.candidate_fps is not None
            else max(2.0, self.config.observation_fps * 2)
        )
        images, encoded = [], []
        for frame in job.frames:
            jpeg = frame.jpeg_bytes
            with Image.open(io.BytesIO(jpeg)) as image:
                width, height = image.size
                if max(width, height) > 720:
                    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    jpeg, width, height = encode_jpeg(bgr, max_image_side=720, quality=90)
            images.append(base64.b64encode(jpeg).decode("ascii"))
            encoded.append((frame, jpeg, width, height))
        payload = {
            "request_id": job.request.request_id,
            "session": job.stream_session_id,
            "fps": fps,
            "frame_times": [frame.timestamp_seconds for frame in job.frames],
            "images": images,
        }
        if not self._template_registered:
            payload["question_template"] = list(self._question_template)
        if job.diagnostics is not None:
            note(job, "prepared", encoded)
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("VideoJev request exceeds the encoded memory limit")
        return body, self._expected

    def inspect(self, job):
        with self._lock:
            start = time.monotonic()
            verdicts, failure = (), None
            self.last_http_timing = {"http_request_attempted": False}
            self.last_inspection_timing = {}
            self.last_response_diagnostics = {}
            try:
                body, expected = self._payload(job)
                self.last_http_timing.update(
                    request_build_seconds=time.monotonic() - start,
                    request_bytes=len(body),
                    encoded_image_bytes=sum(len(f.jpeg_bytes) for f in job.frames),
                )
                self.last_inspection_timing["request_prepare_seconds"] = (
                    time.monotonic() - start
                )
                initialize = time.monotonic()
                self._ensure_transport()
                self.last_inspection_timing["client_initialize_seconds"] = (
                    time.monotonic() - initialize
                )
                response = self._loop.run_until_complete(asyncio.wait_for(
                    self._request(job, body), timeout=self.settings.timeout_seconds,
                ))
                parse_started = time.monotonic()
                try:
                    if (
                        not isinstance(response, dict)
                        or response.get("request_id") != job.request.request_id
                        or not isinstance(response.get("timings"), dict)
                    ):
                        raise ValueError("invalid native scoring envelope")
                    self._template_registered = True
                    for name, value in response["timings"].items():
                        if (name in _SERVICE_METRICS
                                and type(value) in (int, float)
                                and math.isfinite(value) and value >= 0):
                            self.last_http_timing["videojev_" + name] = value
                    values = parse_choices(response.get("raw"), expected)
                    verdicts = tuple(
                        IndependentVerdict(
                            label, (), values["scene/" + label], None, (),
                            ("scene_only",),
                        ) for label in job.request.event_labels
                    )
                finally:
                    self.last_inspection_timing["verdict_parse_seconds"] = (
                        time.monotonic() - parse_started
                    )
                self.last_response_diagnostics = {
                    "choices_count": len(expected), "model_match": "unreported",
                }
                return verdicts
            except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
                failure = ModelAPIError("VideoJev deadline exceeded", code="timeout")
                raise failure from None
            except httpx.RequestError:
                failure = ModelAPIError("VideoJev connection failed", code="connection")
                raise failure from None
            except ModelAPIError as error:
                failure = error
                raise
            except (ValueError, TypeError, KeyError, IndexError, RecursionError,
                    OverflowError, OSError):
                if not self.last_http_timing.get("http_request_attempted", False):
                    failure = ValueError("VideoJev input preparation failed")
                    raise failure from None
                failure = ModelAPIError(
                    "VideoJev returned an invalid response", code="invalid_response",
                )
                raise failure from None
            except Exception as error:
                failure = error
                raise
            finally:
                note(job, "responded", verdicts=verdicts, error=failure,
                     transport=self.last_http_timing)

    async def _request(self, job, body):
        started = time.monotonic()
        timing = self.last_http_timing
        timing["response_bytes"] = 0
        try:
            request = self._http.build_request(
                "POST", self.settings.endpoint, content=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=self.settings.timeout_seconds,
            )
            note(job, "attempted")
            timing["http_request_attempted"] = True
            response = await self._http.send(request, stream=True, follow_redirects=False)
            try:
                timing["http_status"] = response.status_code
                timing["http_until_headers_seconds"] = time.monotonic() - started
                chunks = []
                read_started = time.monotonic()
                async for chunk in response.aiter_bytes():
                    timing["response_bytes"] += len(chunk)
                    if timing["response_bytes"] > _MAX_RESPONSE_BYTES:
                        raise ValueError("native scoring response exceeds its memory limit")
                    chunks.append(chunk)
                timing["response_read_seconds"] = time.monotonic() - read_started
                decode_started = time.monotonic()
                try:
                    raw = b"".join(chunks)
                    if response.status_code != 200:
                        code = "http_error"
                        try:
                            detail = _json(raw, stage="native_scoring_error")
                        except (ValueError, TypeError, RecursionError):
                            detail = None
                        if isinstance(detail, dict):
                            service_error = detail.get("error")
                            if service_error in _NATIVE_SERVICE_ERRORS:
                                self.last_response_diagnostics = {
                                    "native_service_error": service_error,
                                }
                            if detail.get("error") == "invalid_response":
                                code = "invalid_response"
                            if service_error == "template_missing":
                                self._template_registered = False
                            service = detail.get("timings")
                            if isinstance(service, dict):
                                for name, value in service.items():
                                    if (name in _SERVICE_METRICS
                                            and type(value) in (int, float)
                                            and math.isfinite(value) and value >= 0):
                                        timing["videojev_" + name] = value
                        failure = ModelAPIError(
                            "VideoJev service request failed", code=code,
                            status_code=response.status_code,
                        )
                        failure.retry_after = response.headers.get("Retry-After")
                        failure.response_diagnostics = dict(self.last_response_diagnostics)
                        raise failure
                    return _json(raw, stage="native_scoring_response")
                finally:
                    timing["response_json_decode_seconds"] = time.monotonic() - decode_started
            finally:
                await response.aclose()
        finally:
            timing["total_seconds"] = time.monotonic() - started
