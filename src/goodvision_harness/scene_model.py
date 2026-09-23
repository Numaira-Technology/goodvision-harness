"""Strict temporal VLM adapter with a reusable pool and a total request deadline."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import RLock
from typing import Any

import httpx

from .latency_trace import log_timing, trace_id
from .model_api import (
    ModelAPIClient,
    ModelAPIError,
    ModelAPISettings,
    environment_api_key,
    model_request_headers,
)
from .scene_inspection import (
    _validate_job,
    _validate_verdicts,
    build_temporal_messages,
    build_benchmark_answer_messages,
    add_independent_benchmark_context,
)
from .scene_runtime import BenchmarkInspection, TemporalInspection, TemporalVerdict
from .vlm_transport import create_vlm_http_client

_MAX_RESPONSE_BYTES = 256 * 1024
_FIELDS = {"event_label", "subject_ids", "detected", "confidence"}


class _ResponseValidationError(ValueError):
    """Closed diagnostic vocabulary; response values never enter the message."""

    def __init__(self, reason: str, stage: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ResponseValidationError("duplicate_json_field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise _ResponseValidationError("nonfinite_json_number")


def _json(value: str | bytes, *, stage: str) -> Any:
    try:
        return json.loads(
            value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except _ResponseValidationError as error:
        error.stage = stage
        raise
    except json.JSONDecodeError:
        raise _ResponseValidationError("invalid_json_syntax", stage) from None
    except UnicodeDecodeError:
        raise _ResponseValidationError("invalid_json_encoding", stage) from None
    except RecursionError:
        raise _ResponseValidationError("json_nesting_limit", stage) from None


def _verdict_rejection_reason(job, verdicts, allowed) -> str:
    """Explain an already rejected batch without relaxing the shared validator."""
    seen = set()
    for verdict in verdicts:
        if verdict.event_label not in job.request.event_labels:
            return "unrequested_event_label"
        ids = verdict.subject_ids
        if not ids:
            return "empty_subject_ids"
        if any(not isinstance(value, str) or not value for value in ids):
            return "invalid_subject_id_type"
        if not set(ids) <= allowed:
            return "unknown_subject_id"
        if len(set(ids)) != len(ids):
            return "duplicate_subject_id"
        if type(verdict.detected) not in (bool, type(None)):
            return "invalid_detected_type"
        if verdict.confidence is not None:
            if type(verdict.confidence) not in (int, float):
                return "invalid_confidence_type"
            if (
                not math.isfinite(verdict.confidence)
                or not 0 <= verdict.confidence <= 1
            ):
                return "invalid_confidence_range"
        key = (verdict.event_label, tuple(sorted(ids)))
        if key in seen:
            return "duplicate_verdict"
        seen.add(key)
    return "invalid_or_ungrounded_verdict"


def _response_shape(response: object) -> dict[str, object]:
    """Only fixed enums and aggregate sizes; no model text or arbitrary keys."""
    if not isinstance(response, Mapping):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list):
        return {"choices_count": None}
    result: dict[str, object] = {"choices_count": len(choices)}
    if not choices or not isinstance(choices[0], Mapping):
        return result
    choice = choices[0]
    finish = choice.get("finish_reason")
    result["finish_reason"] = (
        finish
        if isinstance(finish, str)
        and finish
        in {
            "stop",
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
        }
        else "missing"
        if finish is None
        else "other"
    )
    message = choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        result["content_kind"] = (
            "text"
            if isinstance(content, str)
            else "parts"
            if isinstance(content, list)
            else "missing"
            if content is None
            else "other"
        )
        if isinstance(content, str):
            result["content_characters"] = len(content)
        elif isinstance(content, list):
            result["content_parts"] = len(content)
    return result


class TemporalVLMInspector:
    """Serialize calls through one verified pool; the owning worker closes it.

    Retry scheduling belongs to SceneRuntime. Route parameters are copied as
    configured, and no alternate model, provider, or credential is selected.
    JSON schema support is provider dependent, so response_format is sent only
    when explicitly configured. The portable prompt also permits plain JSON.
    """

    def __init__(
        self,
        settings: ModelAPISettings,
        *,
        model: str,
        api_key: str | Callable[[], str] = "",
        request_parameters: Mapping[str, Any] | None = None,
        max_tokens: int = 1600,
        max_image_side: int = 720,
        jpeg_quality: int = 90,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("temporal model must be configured")
        if not isinstance(api_key, str) and not callable(api_key):
            raise ValueError("api_key must be text or a credential provider")
        if (
            type(settings.timeout_seconds) not in (int, float)
            or not math.isfinite(settings.timeout_seconds)
            or not 0 < settings.timeout_seconds <= 300
        ):
            raise ValueError("temporal request timeout must be within (0, 300] seconds")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 16384:
            raise ValueError("max_tokens must be an integer from 1 through 16384")
        if type(max_image_side) is not int or not 1 <= max_image_side <= 2048:
            raise ValueError("max_image_side must be an integer from 1 through 2048")
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer from 1 through 100")
        if request_parameters is not None and not isinstance(
            request_parameters, Mapping
        ):
            raise ValueError("temporal request parameters must be a mapping")
        parameters = copy.deepcopy(dict(request_parameters or {}))
        if {"model", "messages"} & parameters.keys():
            raise ValueError("temporal parameters cannot replace model or messages")
        if "stream" in parameters and parameters["stream"] is not False:
            raise ValueError("temporal inspection requires a non-streaming completion")
        parameters.setdefault("max_tokens", max_tokens)
        if (
            type(parameters["max_tokens"]) is not int
            or not 1 <= parameters["max_tokens"] <= 16384
        ):
            raise ValueError("temporal output token budget must be bounded")
        # This inspection has an independent budget from legacy single-frame
        # inference. Preserve routing options while honoring the scene setting.
        parameters["max_tokens"] = max_tokens
        parameters.setdefault("temperature", 0)
        parameters["stream"] = False
        # Reject non-JSON options before any images or credentials are sent.
        json.dumps(parameters, allow_nan=False)
        self.settings = replace(settings, max_retries=0)
        self.model = model
        self._api_key = api_key
        self._parameters = parameters
        self._max_image_side = max_image_side
        self._jpeg_quality = jpeg_quality
        self._call_lock = RLock()
        self._stats_lock = RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._http: httpx.AsyncClient | None = None
        self._closed = False
        self._stats: dict[str, Any] = {
            "provider": settings.provider,
            "requests": 0,
            "succeeded_requests": 0,
            "failed_requests": 0,
            "empty_verdict_requests": 0,
            "last_latency_seconds": None,
            "total_latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "last_error_code": None,
            "last_http_status": None,
        }

    def _ensure_transport(self) -> None:
        """Initialize on the owning worker, which also uses and closes the pool."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = create_vlm_http_client()

    def warmup(self) -> bool:
        """Best-effort connection preparation, with no model request or credentials.

        Call from the same owning worker as inspect/close. A response, including
        401 or 405, only confirms transport reachability, not model readiness.
        Warmups do not contribute to inference statistics or consume API keys.
        """
        started = time.monotonic()
        outcome = "failed"
        try:
            with self._call_lock:
                if self._closed:
                    outcome = "closed"
                    return False
                # Real TLS handshakes can exceed two seconds on this route.
                deadline = min(5.0, self.settings.timeout_seconds)
                self._ensure_transport()
                remaining = deadline - (time.monotonic() - started)
                if remaining <= 0:
                    outcome = "timeout"
                    return False
                assert self._loop is not None
                self._loop.run_until_complete(
                    asyncio.wait_for(self._warmup_request(remaining), remaining)
                )
                outcome = "transport_ready"
                return True
        except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            outcome = "timeout"
            return False
        except Exception:
            # Initialization, TLS, and transport failures are best effort. Never
            # log exception text (URLs/credentials) or disable later inference.
            return False
        finally:
            log_timing(
                "vlm_connection_warmup_timing",
                outcome=outcome,
                total_seconds=time.monotonic() - started,
            )

    async def _warmup_request(self, timeout: float) -> None:
        assert self._http is not None
        # Do not copy client headers/cookies/params/auth or endpoint credentials.
        # HEAD uses the configured route without sending evidence or generating
        # output; even an unsupported-method response can warm the connection.
        endpoint = httpx.URL(self.settings.endpoint).copy_with(
            username="", password="", query=None, fragment=None,
        )
        request = httpx.Request(
            "HEAD", endpoint,
            extensions={"timeout": httpx.Timeout(timeout).as_dict()},
        )
        response = await self._http.send(
            request, stream=True, auth=None, follow_redirects=False,
        )
        try:
            # Consume the HEAD response to EOF so HTTP/1.1 can return the
            # connection to the pool. Correct HEAD responses carry no body.
            if not response.is_stream_consumed:
                async for _ in response.aiter_raw():
                    pass
        finally:
            await response.aclose()

    def inspect(self, job: TemporalInspection) -> tuple[TemporalVerdict, ...] | BenchmarkInspection:
        started = time.monotonic()
        timing = {
            "stream_session_id": trace_id(job.stream_session_id),
            "request_id": trace_id(job.request.request_id),
            "outcome": "error",
        }
        with self._call_lock:
            timing["inspector_lock_wait_seconds"] = time.monotonic() - started
            try:
                verdicts = self._inspect_locked(job, timing=timing)
                rows = verdicts.verdicts if isinstance(verdicts, BenchmarkInspection) else verdicts
                timing.update(
                    outcome="completed", verdict_count=len(rows),
                    returned_event_count=len({v.event_label for v in rows}),
                    requested_event_count=len(job.request.event_labels),
                    empty_verdicts=not rows,
                )
                return verdicts
            except ModelAPIError as error:
                timing["error_code"] = error.code
                raise
            finally:
                timing["total_seconds"] = time.monotonic() - started
                log_timing("vlm_inspection_timing", **timing)

    def build_messages(self, job):
        response_format = self._parameters.get("response_format")
        return build_temporal_messages(
            job, max_image_side=self._max_image_side, jpeg_quality=self._jpeg_quality,
            verdicts_wrapper=(isinstance(response_format, Mapping)
                              and response_format.get("type") == "json_object"),
        )

    def _inspect_locked(
        self, job: TemporalInspection, *, timing: dict
    ) -> tuple[TemporalVerdict, ...] | BenchmarkInspection:
        # Called with the inspector lock held, including preparation and parsing.
        if self._closed:
            raise ModelAPIError("temporal inspector is closed")
        preparation_started = time.monotonic()
        messages = self.build_messages(job)
        if job.benchmark_questions and getattr(self, "benchmark_result_wrapper", False):
            add_independent_benchmark_context(messages, job)
        key = self._api_key() if callable(self._api_key) else self._api_key
        if self.settings.provider == "local" and not key:
            key = environment_api_key(self.settings)
        if not isinstance(key, str) or (
            not key.strip() and self.settings.provider != "local"
        ):
            raise ModelAPIError("temporal model credential is unavailable")
        payload = {**self._parameters, "model": self.model, "messages": messages}
        if job.benchmark_questions and payload.get("response_format") is not None:
            payload["response_format"] = {"type": "json_object"}
        # These data URLs were just built locally from bounded JPEGs. Count
        # decoded bytes from Base64 length without copying/decoding pixels.
        encoded_image_bytes = 0
        for part in messages[1]["content"]:
            if part["type"] == "image_url":
                data = part["image_url"]["url"].split(",", 1)[1]
                encoded_image_bytes += len(data) // 4 * 3 - (
                    len(data) - len(data.rstrip("="))
                )
        http_timing = {
            "stream_session_id": trace_id(job.stream_session_id),
            "request_id": trace_id(job.request.request_id),
            "frame_count": len(job.frames),
            "encoded_image_bytes": encoded_image_bytes,
            "max_image_side": self._max_image_side,
            "jpeg_quality": self._jpeg_quality,
        }
        headers = httpx.Headers(model_request_headers(self.settings, key.strip()))
        timing["request_prepare_seconds"] = time.monotonic() - preparation_started
        started = time.monotonic()
        response = None
        validation_stage = validation_reason = None
        with self._stats_lock:
            self._stats["requests"] += 1
        try:
            client_started = time.monotonic()
            self._ensure_transport()
            timing["client_initialize_seconds"] = time.monotonic() - client_started
            response = self._loop.run_until_complete(
                asyncio.wait_for(
                    self._request(payload, headers, timing=http_timing,
                                  capture_response=bool(job.benchmark_questions)),
                    timeout=self.settings.timeout_seconds,
                )
            )
            self._record_usage(response)
            parse_started = time.monotonic()
            try:
                if job.benchmark_questions and getattr(self, "benchmark_result_wrapper", False):
                    verdicts = self._parse_benchmark(job, response)
                else:
                    verdicts = self._parse(job, response)
            finally:
                timing["verdict_parse_seconds"] = time.monotonic() - parse_started
        except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            error = ModelAPIError(
                "temporal model deadline exceeded", code="timeout"
            )
        except httpx.RequestError:
            error = ModelAPIError(
                "temporal model connection failed", code="connection"
            )
        except ModelAPIError as failure:
            error = failure
            validation_stage = getattr(failure, "validation_stage", None)
            validation_reason = getattr(failure, "validation_reason", None)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            RecursionError,
        ) as failure:
            if isinstance(failure, _ResponseValidationError):
                validation_stage, validation_reason = failure.stage, failure.reason
            else:
                validation_stage, validation_reason = (
                    "response",
                    "unexpected_response_shape",
                )
            error = ModelAPIError(
                "temporal model returned an invalid response",
                code="invalid_response",
            )
            if job.benchmark_questions and hasattr(failure, "benchmark_response"):
                error.benchmark_response = failure.benchmark_response
        else:
            self._record_completion(started, verdicts=(
                verdicts.verdicts if isinstance(verdicts, BenchmarkInspection) else verdicts
            ))
            return verdicts
        self._record_completion(started, error=error)
        if error.code in {"invalid_response", "empty_response"}:
            # Preserve the safe closed-vocabulary diagnosis across the process
            # boundary; never attach the model's response or request content.
            error.validation_stage = validation_stage or "response"
            error.validation_reason = validation_reason or "invalid_response_shape"
            log_timing(
                "vlm_response_validation",
                stream_session_id=trace_id(job.stream_session_id),
                request_id=trace_id(job.request.request_id),
                error_code=error.code,
                validation_stage=validation_stage or "response",
                validation_reason=validation_reason or "invalid_response_shape",
                **_response_shape(response),
            )
        if job.benchmark_questions and isinstance(response, dict):
            error.benchmark_response = response
        raise error from None

    async def _request(
        self, payload: Mapping[str, Any], headers: httpx.Headers, *, timing: dict,
        capture_response: bool = False,
    ) -> dict[str, Any]:
        assert self._http is not None
        started = time.monotonic()
        transport_starts = {}
        dispatch_started = upload_completed = None
        timing.update(
            outcome="error",
            transport_trace_observed=False,
            response_read_seconds=None,
            response_chunks=0,
            response_bytes=0,
            response_first_chunk_seconds=None,
            response_last_chunk_seconds=None,
            response_first_non_whitespace_seconds=None,
            response_leading_whitespace_bytes=0,
            response_max_chunk_gap_seconds=0.0,
            response_chunk_samples=[],
            response_chunk_samples_dropped=0,
        )
        response = None
        transport_metrics = {
            "connect_tcp": "tcp_connect_seconds",
            "start_tls": "tls_handshake_seconds",
            "send_request_headers": "send_request_headers_seconds",
            "send_request_body": "send_request_body_seconds",
            "receive_response_headers": "receive_response_headers_seconds",
        }

        async def transport_trace(name, _info):
            nonlocal upload_completed
            # Trace info can contain credentials, endpoints and body objects.
            # Only fixed event names and monotonic durations are inspected.
            parts = name.split(".")
            if (
                len(parts) != 3
                or parts[0]
                not in {
                    "connection",
                    "proxy",
                    "http11",
                    "http2",
                }
                or parts[2] not in {"started", "complete", "failed"}
            ):
                return
            metric = transport_metrics.get(parts[1])
            if metric is None:
                return
            timing["transport_trace_observed"] = True
            identity, state, now = ".".join(parts[:2]), parts[2], time.monotonic()
            if dispatch_started is not None:
                # This includes pool/scheduling/transport startup; it is not a
                # direct measurement of connection-pool queueing alone.
                timing.setdefault(
                    "dispatch_to_first_transport_event_seconds", now - dispatch_started
                )
            if state == "started":
                transport_starts[identity] = (metric, now)
                if parts[1] == "send_request_body":
                    timing["request_body_completed"] = False
            else:
                phase = transport_starts.pop(identity, None)
                if phase is not None:
                    timing[metric] = timing.get(metric, 0.0) + max(0.0, now - phase[1])
                if parts[1] == "send_request_body" and state == "complete":
                    timing["request_body_completed"] = True
                    upload_completed = now
                    timing["request_body_sent_at_seconds"] = now - started

        try:
            if self.settings.provider == "local":
                # Do not inherit credentials, cookies, query parameters or
                # private headers from a previously configured cloud pool.
                request = httpx.Request(
                    "POST", self.settings.endpoint, json=payload, headers=headers,
                    extensions={
                        "trace": transport_trace,
                        "timeout": httpx.Timeout(
                            self.settings.timeout_seconds
                        ).as_dict(),
                    },
                )
            else:
                request = self._http.build_request(
                    "POST",
                    self.settings.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.settings.timeout_seconds,
                    extensions={"trace": transport_trace},
                )
            # Use the exact HTTPX serialization sent, including Base64 and JSON.
            timing["request_bytes"] = len(request.content)
            timing["request_build_seconds"] = time.monotonic() - started
            dispatch_started = time.monotonic()
            if self.settings.provider == "local":
                response = await self._http.send(
                    request, stream=True, auth=None, follow_redirects=False,
                )
            else:
                response = await self._http.send(request, stream=True)
            timing["http_until_headers_seconds"] = time.monotonic() - started
            timing["http_status"] = response.status_code
            if not 200 <= response.status_code < 300:
                raise ModelAPIError(
                    "temporal model HTTP request failed",
                    code="http_error",
                    status_code=response.status_code,
                )
            body = bytearray()
            read_started = time.monotonic()
            previous_chunk_seconds = 0.0
            try:
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise _ResponseValidationError(
                            "response_size_limit", "envelope_body"
                        )
                    if chunk:
                        # Offsets start when body reading starts. A chunk can be
                        # whitespace/keepalive data, not necessarily model output.
                        elapsed = time.monotonic() - read_started
                        if timing["response_first_chunk_seconds"] is None:
                            timing["response_first_chunk_seconds"] = elapsed
                        timing["response_last_chunk_seconds"] = elapsed
                        timing["response_chunks"] += 1
                        timing["response_bytes"] += len(chunk)
                        timing["response_max_chunk_gap_seconds"] = max(
                            timing["response_max_chunk_gap_seconds"],
                            elapsed - previous_chunk_seconds,
                        )
                        previous_chunk_seconds = elapsed
                        # JSON whitespace is only SP/TAB/CR/LF. Log sizes and
                        # offsets, never response content, headers or tokens.
                        remaining = chunk.lstrip(b" \t\r\n")
                        if timing["response_first_non_whitespace_seconds"] is None:
                            timing["response_leading_whitespace_bytes"] += (
                                len(chunk) - len(remaining)
                            )
                            if remaining:
                                timing["response_first_non_whitespace_seconds"] = (
                                    elapsed
                                )
                                timing["http_first_non_whitespace_seconds"] = (
                                    read_started + elapsed - started
                                )
                                if upload_completed is not None:
                                    timing["upload_to_first_non_whitespace_seconds"] = (
                                        read_started + elapsed - upload_completed
                                    )
                        if len(timing["response_chunk_samples"]) < 8:
                            timing["response_chunk_samples"].append({
                                "offset_seconds": elapsed, "bytes": len(chunk),
                                "whitespace_only": not remaining,
                            })
                        else:
                            timing["response_chunk_samples_dropped"] += 1
                    body.extend(chunk)
            finally:
                # Preserve partial progress when the total deadline cancels a
                # stalled body read, and exclude response cleanup from this phase.
                timing["response_read_seconds"] = time.monotonic() - read_started
            if not body.strip():
                error = ModelAPIError(
                    "temporal model response is empty", code="empty_response"
                )
                error.validation_stage = "envelope_body"
                error.validation_reason = "empty_response_body"
                if capture_response:
                    error.benchmark_response = {"raw_body": bytes(body).decode("utf-8", errors="replace")}
                raise error
            decode_started = time.monotonic()
            try:
                decoded = _json(bytes(body), stage="envelope_json")
                if not isinstance(decoded, dict):
                    raise _ResponseValidationError("envelope_not_object", "envelope_schema")
            except _ResponseValidationError as error:
                if capture_response:
                    error.benchmark_response = {"raw_body": bytes(body).decode("utf-8", errors="replace")}
                raise
            finally:
                timing["response_json_decode_seconds"] = (
                    time.monotonic() - decode_started
                )
            generation_id = decoded.get("id")
            if isinstance(generation_id, str) and re.fullmatch(
                r"gen-[A-Za-z0-9-]{1,156}", generation_id
            ):
                timing["generation_ref"] = hashlib.sha256(
                    generation_id.encode()
                ).hexdigest()
            usage = decoded.get("usage")
            if isinstance(usage, Mapping):
                for field in ("prompt_tokens", "completion_tokens"):
                    value = usage.get(field)
                    if type(value) is int and value >= 0:
                        timing[field] = value
                details = usage.get("completion_tokens_details")
                if isinstance(details, Mapping):
                    value = details.get("reasoning_tokens")
                    if type(value) is int and value >= 0:
                        timing["reasoning_tokens"] = value
                details = usage.get("prompt_tokens_details")
                if isinstance(details, Mapping):
                    value = details.get("cached_tokens")
                    if type(value) is int and value >= 0:
                        timing["cached_prompt_tokens"] = value
            timing["outcome"] = "completed"
            return decoded
        except asyncio.CancelledError:
            timing["outcome"] = "cancelled"
            raise
        finally:
            for metric, phase_started in transport_starts.values():
                timing[metric] = timing.get(metric, 0.0) + max(
                    0.0,
                    time.monotonic() - phase_started,
                )
            try:
                close_started = time.monotonic()
                if response is not None:
                    await response.aclose()
            finally:
                timing["response_close_seconds"] = time.monotonic() - close_started
                timing["total_seconds"] = time.monotonic() - started
                log_timing("vlm_http_timing", **timing)

    @staticmethod
    def _parse(
        job: TemporalInspection, response: Mapping[str, Any]
    ) -> tuple[TemporalVerdict, ...] | BenchmarkInspection:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _ResponseValidationError("missing_choices", "envelope_schema")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _ResponseValidationError("choice_not_object", "envelope_schema")
        if choice.get("finish_reason") not in (None, "stop"):
            raise _ResponseValidationError("completion_not_finished", "envelope_schema")
        try:
            content = ModelAPIClient.assistant_text(response).strip()
        except ModelAPIError as error:
            error.validation_stage = "assistant_content"
            error.validation_reason = (
                "empty_assistant_text"
                if error.code == "empty_response"
                else "invalid_assistant_text"
            )
            raise
        # Some vision routes wrap otherwise valid JSON in one Markdown fence.
        # Remove only a whole-message wrapper; prose, partial JSON, nested fences,
        # extra fields and type coercions remain invalid.
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content, re.DOTALL
        )
        raw = _json(fenced.group(1) if fenced else content, stage="verdict_json")
        observations = []
        if job.benchmark_questions:
            if not isinstance(raw, dict) or set(raw) != {"verdicts", "qa_observations"}:
                raise _ResponseValidationError("benchmark_object_required", "benchmark_schema")
            observations = raw["qa_observations"]
            if not isinstance(observations, list):
                raise _ResponseValidationError("observations_array_required", "benchmark_schema")
            question_ids = {q["qa_id"] for q in job.benchmark_questions}
            allowed_subjects = _validate_job(job)
            for observation in observations:
                if (
                    not isinstance(observation, dict)
                    or set(observation) != {"qa_id", "subject_ids", "observation"}
                    or observation["qa_id"] not in question_ids
                    or not isinstance(observation["observation"], str)
                    or not isinstance(observation["subject_ids"], list)
                    or any(subject not in allowed_subjects for subject in observation["subject_ids"])
                ):
                    raise _ResponseValidationError("invalid_observation", "benchmark_schema")
            raw = raw["verdicts"]
        try:
            verdicts = TemporalVLMInspector._parse_verdicts(job, raw)
        except _ResponseValidationError as error:
            if not job.benchmark_questions:
                raise
            # Safety events still require grounded people. A rejected event
            # batch must not discard independently valid question observations.
            return BenchmarkInspection(
                (), tuple(observations), dict(response),
                f"{error.stage}:{error.reason}",
            )
        if job.benchmark_questions:
            return BenchmarkInspection(verdicts, tuple(observations), dict(response))
        return verdicts

    @staticmethod
    def _parse_verdicts(job, raw):
        if isinstance(raw, dict) and set(raw) == {"verdicts"}:
            raw = raw["verdicts"]
        if not isinstance(raw, list) or len(raw) > 64:
            raise _ResponseValidationError(
                "verdict_array_required_or_limit", "verdict_schema"
            )
        verdicts = []
        for item in raw:
            if not isinstance(item, dict):
                raise _ResponseValidationError("non_object", "verdict_schema")
            if _FIELDS - set(item):
                raise _ResponseValidationError("missing_fields", "verdict_schema")
            if set(item) - _FIELDS:
                raise _ResponseValidationError("extra_fields", "verdict_schema")
            if not isinstance(item["event_label"], str):
                raise _ResponseValidationError(
                    "event_label_not_string", "verdict_schema"
                )
            if not isinstance(item["subject_ids"], list):
                raise _ResponseValidationError("subject_ids_not_list", "verdict_schema")
            verdicts.append(
                TemporalVerdict(
                    event_label=item["event_label"],
                    subject_ids=tuple(item["subject_ids"]),
                    detected=item["detected"],
                    confidence=item["confidence"],
                )
            )
        allowed = _validate_job(job)
        try:
            verdicts = _validate_verdicts(job, verdicts, allowed)
        except ValueError:
            raise _ResponseValidationError(
                _verdict_rejection_reason(job, verdicts, allowed), "verdict_grounding"
            ) from None
        return verdicts

    def answer_benchmark(self, questions, evidence):
        """One text-only completion on the same frozen route as observation."""
        with self._call_lock:
            key = self._api_key() if callable(self._api_key) else self._api_key
            if self.settings.provider == "local" and not key:
                key = environment_api_key(self.settings)
            if not key and self.settings.provider != "local":
                raise ModelAPIError("benchmark model credential is unavailable")
            payload = {**self._parameters, "model": self.model,
                       "messages": build_benchmark_answer_messages(questions, evidence)}
            if payload.get("response_format") is not None:
                payload["response_format"] = {"type": "json_object"}
            self._ensure_transport()
            response = self._loop.run_until_complete(asyncio.wait_for(
                self._request(payload, httpx.Headers(model_request_headers(self.settings, key)),
                              timing={}, capture_response=True),
                timeout=self.settings.timeout_seconds,
            ))
            self._record_usage(response)
            try:
                choice = response["choices"][0]
                if choice.get("finish_reason") not in (None, "stop"):
                    raise ValueError("benchmark completion is incomplete")
                content = ModelAPIClient.assistant_text(response).strip()
                fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content, re.DOTALL)
                parsed = _json(fenced.group(1) if fenced else content, stage="benchmark_answer")
                rows = parsed["predictions"]
                expected = {question["qa_id"] for question in questions}
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict) or set(row) != {"qa_id", "prediction"}
                    or not isinstance(row["prediction"], str) for row in rows
                ) or len(rows) != len(expected) or {row["qa_id"] for row in rows} != expected:
                    raise ValueError("benchmark predictions must include each qa_id once")
            except (KeyError, IndexError, TypeError, ValueError, ModelAPIError) as exc:
                error = ModelAPIError("benchmark model returned an incomplete or invalid answer", code="invalid_response")
                error.benchmark_response = response
                raise error from exc
            return {"predictions": rows, "response": response}

    def _record_usage(self, response: Mapping[str, Any]) -> None:
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            return
        with self._stats_lock:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if type(value) is int and 0 <= value <= 100_000_000:
                    self._stats[key] += value

    def _record_completion(
        self,
        started: float,
        *,
        verdicts: tuple[TemporalVerdict, ...] = (),
        error: ModelAPIError | None = None,
    ) -> None:
        elapsed = round(time.monotonic() - started, 6)
        with self._stats_lock:
            self._stats["last_latency_seconds"] = elapsed
            self._stats["total_latency_seconds"] += elapsed
            self._stats["last_error_code"] = error.code if error else None
            self._stats["last_http_status"] = error.status_code if error else None
            self._stats["failed_requests" if error else "succeeded_requests"] += 1
            if not error and not verdicts:
                self._stats["empty_verdict_requests"] += 1

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)

    def close(self) -> None:
        """Close after the owning worker has finished its bounded request."""
        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._http is not None and self._loop is not None:
                    self._loop.run_until_complete(self._http.aclose())
            finally:
                if self._loop is not None:
                    self._loop.close()
