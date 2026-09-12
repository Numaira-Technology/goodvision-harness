"""Strict temporal VLM adapter with a reusable pool and a total request deadline."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import RLock
from typing import Any

import httpx

from .model_api import ModelAPIClient, ModelAPIError, ModelAPISettings
from .scene_inspection import (
    _validate_job,
    _validate_verdicts,
    build_temporal_messages,
)
from .scene_runtime import TemporalInspection, TemporalVerdict
from .vlm_transport import create_vlm_http_client

_MAX_RESPONSE_BYTES = 256 * 1024
_FIELDS = {"event_label", "subject_ids", "detected", "confidence"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _json(value: str | bytes) -> Any:
    return json.loads(
        value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )


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
        api_key: str | Callable[[], str],
        request_parameters: Mapping[str, Any] | None = None,
        max_tokens: int = 1600,
        max_image_side: int = 720,
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

    def inspect(self, job: TemporalInspection) -> tuple[TemporalVerdict, ...]:
        with self._call_lock:
            if self._closed:
                raise ModelAPIError("temporal inspector is closed")
            messages = build_temporal_messages(job, max_image_side=self._max_image_side)
            messages[0]["content"] += (
                '\nAn object with the single field "verdicts" containing that same '
                "array is also accepted. Every verdict must include all four fields."
            )
            key = self._api_key() if callable(self._api_key) else self._api_key
            if not isinstance(key, str) or not key.strip():
                raise ModelAPIError("temporal model credential is unavailable")
            payload = {**self._parameters, "model": self.model, "messages": messages}
            headers = httpx.Headers(self.settings.extra_headers)
            headers.update(
                {
                    "Authorization": f"Bearer {key.strip()}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "goodvision-harness/1",
                }
            )
            started = time.monotonic()
            with self._stats_lock:
                self._stats["requests"] += 1
            try:
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                if self._http is None:
                    self._http = create_vlm_http_client()
                response = self._loop.run_until_complete(
                    asyncio.wait_for(
                        self._request(payload, headers),
                        timeout=self.settings.timeout_seconds,
                    )
                )
                self._record_usage(response)
                verdicts = self._parse(job, response)
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
            except (ValueError, TypeError, KeyError, IndexError, RecursionError):
                error = ModelAPIError(
                    "temporal model returned an invalid response",
                    code="invalid_response",
                )
            else:
                self._record_completion(started, verdicts=verdicts)
                return verdicts
            self._record_completion(started, error=error)
            raise error from None

    async def _request(
        self, payload: Mapping[str, Any], headers: httpx.Headers
    ) -> dict[str, Any]:
        assert self._http is not None
        async with self._http.stream(
            "POST",
            self.settings.endpoint,
            json=payload,
            headers=headers,
            timeout=self.settings.timeout_seconds,
        ) as response:
            if not 200 <= response.status_code < 300:
                raise ModelAPIError(
                    "temporal model HTTP request failed",
                    code="http_error",
                    status_code=response.status_code,
                )
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                    raise ModelAPIError(
                        "temporal response exceeds size limit", code="invalid_response"
                    )
                body.extend(chunk)
            if not body.strip():
                raise ModelAPIError(
                    "temporal model response is empty", code="empty_response"
                )
            decoded = _json(bytes(body))
            if not isinstance(decoded, dict):
                raise ValueError("completion must be a JSON object")
            return decoded

    @staticmethod
    def _parse(
        job: TemporalInspection, response: Mapping[str, Any]
    ) -> tuple[TemporalVerdict, ...]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("completion has no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("completion choice must be an object")
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError("incomplete or refused temporal completion")
        content = ModelAPIClient.assistant_text(response).strip()
        # Some vision routes wrap otherwise valid JSON in one Markdown fence.
        # Remove only a whole-message wrapper; prose, partial JSON, nested fences,
        # extra fields and type coercions remain invalid.
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content, re.DOTALL
        )
        raw = _json(fenced.group(1) if fenced else content)
        if isinstance(raw, dict) and set(raw) == {"verdicts"}:
            raw = raw["verdicts"]
        if not isinstance(raw, list) or len(raw) > 64:
            raise ValueError("temporal completion must be a bounded verdict array")
        verdicts = []
        for item in raw:
            if (
                not isinstance(item, dict)
                or set(item) != _FIELDS
                or not isinstance(item["event_label"], str)
                or not isinstance(item["subject_ids"], list)
            ):
                raise ValueError("temporal verdict fields are invalid")
            verdicts.append(
                TemporalVerdict(
                    event_label=item["event_label"],
                    subject_ids=tuple(item["subject_ids"]),
                    detected=item["detected"],
                    confidence=item["confidence"],
                )
            )
        return _validate_verdicts(job, verdicts, _validate_job(job))

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
