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
from PIL import Image

from .diagnostics import note
from .model_api import ModelAPIError
from .scene_inspection import _validate_job
from .scene_model import _json
from .videojev_contract import MODEL, parse_choices, questions
from .vlm_independent_model import IndependentVerdict
from .vlm_transport import create_vlm_http_client

_MAX_RESPONSE_BYTES = 256 * 1024
_MAX_REQUEST_BYTES = 23 * 1024 * 1024
_SERVICE_METRICS = frozenset((
    "preprocess_seconds", "visual_encode_seconds", "decision_seconds",
    "model_pipeline_seconds", "gpu_pipeline_seconds", "server_queue_seconds",
    "cpu_queue_seconds", "gpu_queue_seconds", "server_preprocess_seconds",
    "server_gpu_seconds", "server_elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
    "input_frame_count", "internal_temporal_padding_frames", "visual_tokens",
    "native_temporal_fps", "context", "max_pixels", "source_width",
    "source_height", "processed_width", "processed_height", "processor_max_pixels",
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
        window = {"frames": [
            {"media_seconds": frame.timestamp_seconds} for frame in job.frames
        ]}
        expected = questions(
            window, labels=job.request.event_labels,
            sensitivities=dict(job.event_sensitivities),
            event_policy_profile=self.config.event_policy,
            definitions=job.event_definitions,
        )
        # The native model uses uniform FPS for positional encoding. Actual
        # source timestamps remain separate; irregular input is not retimed.
        delta = self.config.observation_frame_delta_seconds
        fps = (
            1.0 / delta if delta is not None
            else self.config.candidate_fps
            if self.config.candidate_fps is not None
            else max(2.0, self.config.observation_fps * 2)
        )
        payload = {
            "request_id": job.request.request_id,
            "session": job.stream_session_id,
            "fps": fps,
            "frame_times": [frame.timestamp_seconds for frame in job.frames],
            "questions": expected,
            "images": [
                base64.b64encode(frame.jpeg_bytes).decode("ascii")
                for frame in job.frames
            ],
        }
        if job.diagnostics is not None:
            encoded = []
            for frame in job.frames:
                with Image.open(io.BytesIO(frame.jpeg_bytes)) as image:
                    width, height = image.size
                encoded.append((frame, frame.jpeg_bytes, width, height))
            note(job, "prepared", encoded)
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("VideoJev request exceeds the encoded memory limit")
        return body, expected

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
                            if detail.get("error") == "invalid_response":
                                code = "invalid_response"
                            service = detail.get("timings")
                            if isinstance(service, dict):
                                for name, value in service.items():
                                    if (name in _SERVICE_METRICS
                                            and type(value) in (int, float)
                                            and math.isfinite(value) and value >= 0):
                                        timing["videojev_" + name] = value
                        raise ModelAPIError(
                            "VideoJev service request failed", code=code,
                            status_code=response.status_code,
                            retry_after=response.headers.get("Retry-After"),
                        )
                    return _json(raw, stage="native_scoring_response")
                finally:
                    timing["response_json_decode_seconds"] = time.monotonic() - decode_started
            finally:
                await response.aclose()
        finally:
            timing["total_seconds"] = time.monotonic() - started
