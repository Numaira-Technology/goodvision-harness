"""Shared client for selectable OpenAI-compatible chat-completions services."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import httpx

DEFAULT_AICLUB_ENDPOINT = (
    "https://aiclub.zimaocloud.com/model/openApi/v1/chat/completions"
)
DEFAULT_AICLUB_API_KEY_ENV = "AICLUB_API_KEY"
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DASHSCOPE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_LOCAL_API_KEY_ENV = "LOCAL_INFERENCE_API_KEY"
OPENROUTER_EXCLUDED_PROVIDERS = frozenset({"novita", "coreweave"})
PROVIDER_DEFAULTS = {
    "aiclub": (DEFAULT_AICLUB_ENDPOINT, DEFAULT_AICLUB_API_KEY_ENV, "qwen3.5-plus"),
    "openrouter": (
        DEFAULT_OPENROUTER_ENDPOINT, "OPENROUTER_API_KEY", "qwen/qwen3.5-35b-a3b:nitro"
    ),
    "dashscope": (DEFAULT_DASHSCOPE_ENDPOINT, "DASHSCOPE_API_KEY", "qwen3.5-plus"),
    "local": (DEFAULT_LOCAL_ENDPOINT, DEFAULT_LOCAL_API_KEY_ENV, "local-vision"),
}


def nitro_provider_preferences() -> dict[str, Any]:
    """Let Nitro choose a fast endpoint while retaining operator exclusions."""
    return {
        "sort": "throughput",
        "allow_fallbacks": True,
        "ignore": sorted(OPENROUTER_EXCLUDED_PROVIDERS),
    }


class ModelAPIError(RuntimeError):
    """Raised when the selected service cannot return a usable chat completion."""

    _CODES = frozenset({
        "request_failed", "timeout", "connection", "http_error",
        "invalid_response", "empty_response", "cancelled", "overloaded",
    })

    def __init__(
        self, message: str, *, code: str = "request_failed",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        # Only these bounded fields are suitable for diagnostic logs. Exception
        # messages and causes can contain provider details or request secrets.
        self.code = (
            code if isinstance(code, str) and code in self._CODES else "request_failed"
        )
        self.status_code = (
            status_code
            if isinstance(status_code, int) and not isinstance(status_code, bool)
            and 100 <= status_code <= 599
            else None
        )


@dataclass(frozen=True, slots=True)
class ModelAPISettings:
    """Connection settings that do not depend on a specific model or prompt."""

    endpoint: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 90.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    provider: str = "aiclub"

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                "inference_service.provider must be aiclub, openrouter, "
                "dashscope, or local"
            )
        endpoint, api_key_env, _model = PROVIDER_DEFAULTS[self.provider]
        if not self.endpoint:
            object.__setattr__(self, "endpoint", endpoint)
        if not self.api_key_env or self.provider == "local":
            # A copied cloud configuration must never send its credentials to
            # the local server. Local authentication has a dedicated variable.
            object.__setattr__(self, "api_key_env", api_key_env)
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "inference_service.endpoint must be an absolute HTTP(S) URL"
            )
        if not self.api_key_env.strip():
            raise ValueError("inference_service.api_key_env must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("inference_service.timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("inference_service.max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "inference_service.retry_backoff_seconds must be non-negative"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelAPISettings:
        extra_headers = value.get("extra_headers") or {}
        if not isinstance(extra_headers, Mapping):
            raise TypeError("inference_service.extra_headers must be a mapping")
        return cls(
            endpoint=str(value.get("endpoint", "")),
            api_key_env=str(value.get("api_key_env", "")),
            provider=str(value.get("provider", "aiclub")).strip().lower(),
            timeout_seconds=float(value.get("timeout_seconds", 90.0)),
            max_retries=int(value.get("max_retries", 3)),
            retry_backoff_seconds=float(value.get("retry_backoff_seconds", 1.0)),
            extra_headers={str(key): str(item) for key, item in extra_headers.items()},
        )


def api_key_present(settings: ModelAPISettings) -> bool:
    """True when the configured API-key environment variable is set."""

    return bool(environment_api_key(settings))


def environment_api_key(settings: ModelAPISettings) -> str:
    if settings.provider == "local":
        return os.environ.get(DEFAULT_LOCAL_API_KEY_ENV, "").strip()
    # Generic overrides are explicit; never guess using another route's key.
    names = ["INFERENCE_API_KEY", settings.api_key_env]
    if settings.api_key_env in {"INFERENCE_API_KEY", "MODEL_API_KEY"}:
        names.append(PROVIDER_DEFAULTS[settings.provider][1])
    names.append("MODEL_API_KEY")
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def model_request_headers(
    settings: ModelAPISettings, api_key: str, *, accept: str = "application/json",
) -> dict[str, str]:
    """Build headers with authentication coming only from the selected key."""

    headers = {
        str(key): str(value) for key, value in settings.extra_headers.items()
        if str(key).lower() != "authorization"
    }
    headers.update({
        "Content-Type": "application/json",
        "Accept": accept,
        "User-Agent": "goodvision-harness/1",
    })
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class MissingAPIKeyError(ModelAPIError):
    """The operator must configure a credential before live model calls."""


class ModelAPIClient:
    """Generic JSON client used by the video inference layer."""

    _RETRYABLE_STATUSES: ClassVar[set[int]] = {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        settings: ModelAPISettings,
        *,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings
        self._api_key = (api_key or environment_api_key(settings)).strip()
        if not self._api_key and settings.provider != "local":
            raise MissingAPIKeyError(
                f"missing model API key; set {settings.api_key_env}"
            )

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        """POST a model-agnostic chat completion and return its JSON object."""

        if not model.strip():
            raise ValueError("model must not be empty")
        if not messages:
            raise ValueError("messages must not be empty")

        payload = {key: value for key, value in parameters.items() if value is not None}
        payload.update({"model": model, "messages": list(messages)})
        return self._post_json(payload)

    @staticmethod
    def assistant_text(response: Mapping[str, Any]) -> str:
        """Extract text from either string or structured assistant content."""

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError(
                "response did not contain an assistant message",
                code="invalid_response",
            ) from exc

        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
            raise ModelAPIError(
                "assistant message did not contain text", code="empty_response"
            )
        if content is None:
            raise ModelAPIError(
                "assistant message did not contain text", code="empty_response"
            )
        if isinstance(content, Sequence):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, Mapping) and block.get("type") == "text"
            ]
            if any(not isinstance(part, str) for part in parts):
                raise ModelAPIError(
                    "assistant message contained invalid text", code="invalid_response"
                )
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
            raise ModelAPIError(
                "assistant message did not contain text", code="empty_response"
            )
        raise ModelAPIError(
            "assistant message did not contain text", code="invalid_response"
        )

    def _headers(self) -> dict[str, str]:
        return model_request_headers(self.settings, self._api_key)

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.settings.endpoint,
            data=encoded,
            headers=self._headers(),
            method="POST",
        )

        for attempt in range(self.settings.max_retries + 1):
            try:
                with urlopen(
                    request, timeout=self.settings.timeout_seconds
                ) as response:
                    body = response.read()
                if not body.strip():
                    raise ModelAPIError(
                        "model API returned an empty response", code="empty_response"
                    )
                try:
                    decoded = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ModelAPIError(
                        "model API returned invalid JSON", code="invalid_response"
                    ) from exc
                if not isinstance(decoded, dict):
                    raise ModelAPIError(
                        "model API returned an invalid response",
                        code="invalid_response",
                    )
                return decoded
            except HTTPError as exc:
                # Consume the provider body so the connection can be released,
                # but never copy vendor details into logs or public records.
                exc.read()
                if (
                    exc.code in self._RETRYABLE_STATUSES
                    and attempt < self.settings.max_retries
                ):
                    retry_after = None
                    if exc.headers is not None:
                        retry_after = exc.headers.get("Retry-After")
                    self._sleep_before_retry(attempt, retry_after)
                    continue
                raise ModelAPIError(
                    f"model API returned HTTP {int(exc.code)}",
                    code="http_error", status_code=int(exc.code),
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.settings.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                timed_out = isinstance(exc, TimeoutError) or (
                    isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError)
                )
                raise ModelAPIError(
                    "model API request failed",
                    code="timeout" if timed_out else "connection",
                ) from exc

        raise AssertionError("retry loop terminated unexpectedly")

    def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        delay: float | None = None
        if retry_after:
            try:
                delay = max(0.0, float(retry_after))
            except ValueError:
                delay = None
        if delay is None:
            delay = self.settings.retry_backoff_seconds * (2**attempt)
        if delay:
            time.sleep(delay)


class PooledModelAPIClient(ModelAPIClient):
    """Shared, thread-safe verified HTTPS pool for route probes and live runs."""

    def __init__(
        self,
        settings: ModelAPISettings,
        *,
        api_key: str | None = None,
        timing_sink: Callable[[Mapping[str, Any]], None] | None = None,
        trace_transport: bool = False,
    ):
        super().__init__(settings, api_key=api_key)
        self._timing_sink = timing_sink
        self._trace_transport = trace_transport
        self._sequence_lock = threading.Lock()
        self._request_sequence = 0
        self._http = httpx.Client(
            http2=False, follow_redirects=False,
            limits=httpx.Limits(
                max_connections=6, max_keepalive_connections=6,
                keepalive_expiry=120,
            ),
        )

    def close(self) -> None:
        self._http.close()

    def _next_request_sequence(self) -> int:
        with self._sequence_lock:
            sequence = self._request_sequence
            self._request_sequence += 1
            return sequence

    def _emit_timing(self, timing: Mapping[str, Any]) -> None:
        if self._timing_sink is None:
            return
        try:
            self._timing_sink(dict(timing))
        except Exception:
            # Diagnostics must never turn a successful request into a failure.
            pass

    def _start_timing(
        self,
        request_sequence: int,
        attempt: int,
        *,
        streaming: bool,
    ) -> tuple[float, dict[str, Any], Callable[[str, dict[str, Any]], None]]:
        started = time.perf_counter()
        timing: dict[str, Any] = {
            "type": "model_transport_timing",
            "request_sequence": request_sequence,
            "attempt": attempt + 1,
            "outcome": "error",
            "streaming": streaming,
            "trace_enabled": self._trace_transport,
            "trace_events_observed": False,
            "tcp_connect_attempt_count": 0,
            "tcp_connect_count": 0,
            "tcp_connect_failure_count": 0,
        }
        phase_starts: dict[str, float] = {}
        phases = {
            "connect_tcp": "connect_tcp_seconds",
            "start_tls": "tls_seconds",
            "send_request_headers": "request_headers_seconds",
            "send_request_body": "request_body_seconds",
            "receive_response_headers": "wait_response_headers_seconds",
            "receive_response_body": "response_body_seconds",
        }

        def trace(name: str, _info: dict[str, Any]) -> None:
            # Trace metadata can contain URLs and headers. Retain only fixed
            # phase names and durations, never the provider-supplied info.
            parts = name.split(".")
            if (
                len(parts) != 3
                or parts[0] not in {"connection", "proxy", "http11", "http2"}
                or parts[1] not in phases
                or parts[2] not in {"started", "complete", "failed"}
            ):
                return
            timing["trace_events_observed"] = True
            identity = ".".join(parts[:2])
            now = time.perf_counter()
            if parts[2] == "started":
                phase_starts[identity] = now
                if parts[1] == "connect_tcp":
                    timing["tcp_connect_attempt_count"] += 1
                return
            phase_started = phase_starts.pop(identity, None)
            if phase_started is not None:
                metric = phases[parts[1]]
                timing[metric] = round(
                    timing.get(metric, 0.0) + now - phase_started, 6,
                )
            if parts[1:] == ["connect_tcp", "complete"]:
                timing["tcp_connect_count"] += 1
            elif parts[1:] == ["connect_tcp", "failed"]:
                timing["tcp_connect_failure_count"] += 1

        return started, timing, trace

    def _finish_timing(
        self,
        timing: dict[str, Any],
        started: float,
        outcome: str,
    ) -> None:
        timing["outcome"] = outcome
        timing["total_seconds"] = round(time.perf_counter() - started, 6)
        timing["connection_reused"] = (
            timing["tcp_connect_attempt_count"] == 0
            if timing["trace_events_observed"] else None
        )
        self._emit_timing(timing)

    @staticmethod
    def _response_metadata(
        response: httpx.Response,
        timing: dict[str, Any],
    ) -> None:
        version = response.extensions.get("http_version")
        if version in (b"HTTP/1.0", b"HTTP/1.1", b"HTTP/2", b"HTTP/3"):
            timing["http_version"] = version.decode("ascii")
        timing["http_status"] = response.status_code

    @staticmethod
    def _sse_data(response: httpx.Response) -> Iterator[str]:
        data: list[str] = []
        for line in response.iter_lines():
            if not line:
                if data:
                    yield "\n".join(data)
                    data.clear()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if data:
            yield "\n".join(data)

    def chat_completion_stream(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        on_content: Callable[[str, float], bool] | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Complete one SSE response while exposing a validated text prefix."""

        if not model.strip():
            raise ValueError("model must not be empty")
        if not messages:
            raise ValueError("messages must not be empty")
        if {"model", "messages", "stream", "stream_options"} & parameters.keys():
            raise ValueError("stream parameters override a fixed field")
        payload = {
            key: value for key, value in parameters.items() if value is not None
        }
        payload.update({
            "model": model,
            "messages": list(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        return self._post_stream(payload, on_content=on_content)

    def _post_stream(
        self,
        payload: Mapping[str, Any],
        *,
        on_content: Callable[[str, float], bool] | None,
    ) -> dict[str, Any]:
        request_sequence = self._next_request_sequence()
        for attempt in range(self.settings.max_retries + 1):
            started, timing, trace = self._start_timing(
                request_sequence, attempt, streaming=True,
            )
            content = ""
            finish_reason: str | None = None
            usage: dict[str, Any] = {}
            done = False
            preliminary_emitted = False
            chunk_count = 0
            try:
                with self._http.stream(
                    "POST",
                    self.settings.endpoint,
                    headers=model_request_headers(
                        self.settings, self._api_key, accept="text/event-stream",
                    ),
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                    extensions={"trace": trace} if self._trace_transport else None,
                ) as response:
                    self._response_metadata(response, timing)
                    timing["response_headers_seconds"] = round(
                        time.perf_counter() - started, 6,
                    )
                    if not 200 <= response.status_code < 300:
                        response.read()
                        timing["response_bytes"] = len(response.content)
                        self._finish_timing(timing, started, "http_error")
                        if (
                            response.status_code in self._RETRYABLE_STATUSES
                            and attempt < self.settings.max_retries
                        ):
                            self._sleep_before_retry(
                                attempt, response.headers.get("Retry-After"),
                            )
                            continue
                        raise ModelAPIError(
                            f"model API returned HTTP {response.status_code}",
                            code="http_error",
                            status_code=response.status_code,
                        )
                    media_type = response.headers.get("content-type", "").split(
                        ";", 1,
                    )[0].strip().lower()
                    if media_type != "text/event-stream":
                        response.read()
                        timing["response_bytes"] = len(response.content)
                        self._finish_timing(timing, started, "invalid_response")
                        raise ModelAPIError(
                            "model stream returned an invalid content type",
                            code="invalid_response",
                        )
                    for data in self._sse_data(response):
                        if not data.strip():
                            continue
                        if done:
                            self._finish_timing(
                                timing, started, "invalid_response",
                            )
                            raise ModelAPIError(
                                "model stream contains data after DONE",
                                code="invalid_response",
                            )
                        if data.strip() == "[DONE]":
                            done = True
                            timing["complete_seconds"] = round(
                                time.perf_counter() - started, 6,
                            )
                            continue
                        chunk_count += 1
                        try:
                            chunk = json.loads(data)
                        except (TypeError, ValueError):
                            self._finish_timing(
                                timing, started, "invalid_response",
                            )
                            raise ModelAPIError(
                                "model stream contains invalid JSON",
                                code="invalid_response",
                            ) from None
                        if not isinstance(chunk, Mapping) or "error" in chunk:
                            self._finish_timing(
                                timing, started, "invalid_response",
                            )
                            raise ModelAPIError(
                                "model stream contains an invalid chunk",
                                code="invalid_response",
                            )
                        raw_usage = chunk.get("usage")
                        if isinstance(raw_usage, Mapping):
                            for key in (
                                "prompt_tokens", "completion_tokens", "total_tokens",
                            ):
                                value = raw_usage.get(key)
                                if (
                                    isinstance(value, int)
                                    and not isinstance(value, bool)
                                    and value >= 0
                                ):
                                    usage[key] = value
                        choices = chunk.get("choices", [])
                        if not isinstance(choices, list):
                            self._finish_timing(
                                timing, started, "invalid_response",
                            )
                            raise ModelAPIError(
                                "model stream contains invalid choices",
                                code="invalid_response",
                            )
                        for choice in choices:
                            if not isinstance(choice, Mapping):
                                continue
                            if choice.get("index", 0) != 0:
                                continue
                            delta = choice.get("delta", {})
                            if not isinstance(delta, Mapping):
                                continue
                            token = delta.get("content")
                            if token is not None and not isinstance(token, str):
                                self._finish_timing(
                                    timing, started, "invalid_response",
                                )
                                raise ModelAPIError(
                                    "model stream contains invalid content",
                                    code="invalid_response",
                                )
                            if token:
                                elapsed = round(time.perf_counter() - started, 6)
                                timing.setdefault(
                                    "first_content_token_seconds", elapsed,
                                )
                                timing["last_content_token_seconds"] = elapsed
                                content += token
                                if on_content is not None and not preliminary_emitted:
                                    try:
                                        preliminary_emitted = bool(
                                            on_content(content, elapsed)
                                        )
                                    except Exception:
                                        # Optional early delivery must not corrupt
                                        # the authoritative final classification.
                                        on_content = None
                                    if preliminary_emitted:
                                        timing["preliminary_decision_seconds"] = elapsed
                            finish = choice.get("finish_reason")
                            if finish is not None:
                                if not isinstance(finish, str):
                                    self._finish_timing(
                                        timing, started, "invalid_response",
                                    )
                                    raise ModelAPIError(
                                        "model stream has invalid finish reason",
                                        code="invalid_response",
                                    )
                                finish_reason = finish
                                timing["finish_reason_seconds"] = round(
                                    time.perf_counter() - started, 6,
                                )
                    timing["response_eof_seconds"] = round(
                        time.perf_counter() - started, 6,
                    )
            except httpx.RequestError as exc:
                outcome = (
                    "timeout" if isinstance(exc, httpx.TimeoutException)
                    else "connection"
                )
                self._finish_timing(timing, started, outcome)
                if (
                    not preliminary_emitted
                    and attempt < self.settings.max_retries
                ):
                    self._sleep_before_retry(attempt, None)
                    continue
                raise ModelAPIError(
                    "model API request failed", code=outcome,
                ) from exc
            if not done or finish_reason != "stop" or not content.strip():
                self._finish_timing(timing, started, "invalid_response")
                raise ModelAPIError(
                    "model stream did not finish normally",
                    code="invalid_response",
                )
            timing["stream_chunk_count"] = chunk_count
            timing["response_chars"] = len(content)
            first = timing.get("first_content_token_seconds")
            last = timing.get("last_content_token_seconds")
            if isinstance(first, float) and isinstance(last, float):
                timing["generation_seconds"] = round(last - first, 6)
            self._finish_timing(timing, started, "completed")
            return {
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
            }
        raise AssertionError("retry loop terminated unexpectedly")

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_sequence = self._next_request_sequence()
        for attempt in range(self.settings.max_retries + 1):
            started, timing, trace = self._start_timing(
                request_sequence, attempt, streaming=False,
            )

            response = None
            try:
                response = self._http.post(
                    self.settings.endpoint, headers=self._headers(), json=payload,
                    timeout=self.settings.timeout_seconds,
                    extensions={"trace": trace} if self._trace_transport else None,
                )
            except httpx.RequestError as exc:
                outcome = (
                    "timeout" if isinstance(exc, httpx.TimeoutException)
                    else "connection"
                )
                self._finish_timing(timing, started, outcome)
                if attempt < self.settings.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise ModelAPIError(
                    "model API request failed",
                    code=(
                        "timeout" if isinstance(exc, httpx.TimeoutException)
                        else "connection"
                    ),
                ) from exc
            self._response_metadata(response, timing)
            timing["response_bytes"] = len(response.content)
            if not 200 <= response.status_code < 300:
                self._finish_timing(timing, started, "http_error")
                if (response.status_code in self._RETRYABLE_STATUSES
                        and attempt < self.settings.max_retries):
                    self._sleep_before_retry(
                        attempt, response.headers.get("Retry-After")
                    )
                    continue
                raise ModelAPIError(
                    f"model API returned HTTP {response.status_code}",
                    code="http_error", status_code=response.status_code,
                )
            if not response.content.strip():
                self._finish_timing(timing, started, "empty_response")
                raise ModelAPIError(
                    "model API returned an empty response", code="empty_response"
                )
            parse_started = time.perf_counter()
            try:
                decoded = response.json()
            except (ValueError, UnicodeDecodeError) as exc:
                timing["response_json_decode_seconds"] = round(
                    time.perf_counter() - parse_started, 6,
                )
                self._finish_timing(timing, started, "invalid_response")
                raise ModelAPIError(
                    "model API returned invalid JSON", code="invalid_response"
                ) from exc
            timing["response_json_decode_seconds"] = round(
                time.perf_counter() - parse_started, 6,
            )
            if not isinstance(decoded, dict):
                self._finish_timing(timing, started, "invalid_response")
                raise ModelAPIError(
                    "model API returned an invalid response", code="invalid_response"
                )
            self._finish_timing(timing, started, "completed")
            return decoded
        raise AssertionError("retry loop terminated unexpectedly")
