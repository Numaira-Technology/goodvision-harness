"""vLLM streaming transport with historical benchmark event/person parsers."""

import asyncio
import json
import math
import time

from goodvision_harness.latency_trace import log_timing
from goodvision_harness.model_api import ModelAPIError
from tools.benchmark_runtime.scene_model import (
    _ResponseValidationError,
    _retry_after_seconds,
)
from tools.benchmark_runtime.vlm_rate_limit import read_rate_limit_diagnostics

MAX_BYTES = 2 * 1024 * 1024


class SSECompletion:
    """Reassemble one answer; heartbeat/role/usage chunks are not answer tokens."""

    def __init__(self):
        self.content = []
        self.metadata = {}
        self.usage = None
        self.finish = None
        self.done = False
        self.first = self.last = None
        self.reasoning_first = None
        self.content_chunks = 0

    def feed(self, data, elapsed):
        if self.done:
            raise _ResponseValidationError("data_after_done", "stream")
        if data == "[DONE]":
            self.done = True
            return
        try:
            row = json.loads(data)
        except (ValueError, RecursionError) as exc:
            raise _ResponseValidationError("invalid_stream_json", "stream") from exc
        if not isinstance(row, dict):
            raise _ResponseValidationError("invalid_stream_object", "stream")
        if row.get("error"):
            # HTTP 200 does not mean success when the SSE body reports failure.
            error = row["error"]
            code = error.get("code") if isinstance(error, dict) else None
            status = (
                int(code)
                if str(code) in {"401", "402", "403", "429", "502", "503"}
                else None
            )
            raise ModelAPIError(
                "model stream failed",
                code="http_error" if status else "request_failed",
                status_code=status,
            )
        for key in ("id", "model", "provider"):
            value = row.get(key)
            if value is not None:
                if not isinstance(value, str) or len(value) > 256:
                    raise _ResponseValidationError("invalid_stream_metadata", "stream")
                if key in self.metadata and self.metadata[key] != value:
                    raise _ResponseValidationError("changed_stream_metadata", "stream")
                self.metadata[key] = value
        if isinstance(row.get("usage"), dict):
            self.usage = row["usage"]
        choices = row.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise _ResponseValidationError("invalid_stream_choices", "stream")
        if not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise _ResponseValidationError("invalid_stream_choice", "stream")
        delta = choice.get("delta", {})
        if not isinstance(delta, dict) or delta.get("tool_calls"):
            raise _ResponseValidationError("invalid_stream_delta", "stream")
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise _ResponseValidationError("invalid_stream_content", "stream")
        if content:
            if self.finish is not None:
                raise _ResponseValidationError("content_after_finish", "stream")
            self.content.append(content)
            # Leading whitespace is not the first useful answer content.
            if self.first is not None or content.strip():
                if self.first is None:
                    self.first = elapsed
                self.last = elapsed
                self.content_chunks += 1
        if delta.get("reasoning") or delta.get("reasoning_details"):
            if self.reasoning_first is None:
                self.reasoning_first = elapsed
        finish = choice.get("finish_reason")
        if finish is not None:
            if self.finish is not None and self.finish != finish:
                raise _ResponseValidationError("changed_stream_finish", "stream")
            self.finish = finish

    def result(self, timing):
        if not self.done or self.finish != "stop" or self.first is None:
            raise _ResponseValidationError("incomplete_stream", "stream")
        timing["stream_first_content_seconds"] = self.first
        timing["stream_content_span_seconds"] = self.last - self.first
        timing["stream_content_chunks"] = self.content_chunks
        if self.reasoning_first is not None:
            timing["stream_first_reasoning_seconds"] = self.reasoning_first
        return {
            **self.metadata,
            "usage": self.usage,
            "choices": [
                {
                    "finish_reason": self.finish,
                    "message": {"role": "assistant", "content": "".join(self.content)},
                }
            ],
        }


def usage_metrics(usage, timing):
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if type(value) is int and value >= 0:
            timing[key] = value
    for group, key, target in (
        ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ("prompt_tokens_details", "cached_tokens", "cached_prompt_tokens"),
    ):
        details = usage.get(group)
        value = details.get(key) if isinstance(details, dict) else None
        if type(value) is int and value >= 0:
            timing[target] = value
    cost = usage.get("cost")
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        timing["cost_usd"] = cost
    # Only an explicitly reported reasoning count permits subtraction. Missing
    # usage is unknown, never zero; SSE chunk count is never a token count.
    completion, reasoning = (
        timing.get("completion_tokens"),
        timing.get("reasoning_tokens"),
    )
    if completion is not None and reasoning is not None and reasoning <= completion:
        timing["visible_completion_tokens"] = completion - reasoning


class StreamingMixin:
    async def _request(self, payload, headers, *, timing):
        started = time.monotonic()
        timing.update(
            http_request_attempted=False,
            outcome="error",
            dispatch_pacing_wait_seconds=0.0,
            dispatch_pacing_granted=0,
            dispatch_pacing_expired=0,
        )
        self.last_rate_limit_diagnostics = {}
        response = None
        dispatched = None
        try:
            payload = {**payload, "stream": True}
            request = self._http.build_request(
                "POST",
                self.settings.endpoint,
                json=payload,
                headers={**headers, "Accept": "text/event-stream"},
                timeout=self.settings.timeout_seconds,
            )
            timing["request_bytes"] = len(request.content)
            timing["request_build_seconds"] = time.monotonic() - started
            pacer = getattr(self, "dispatch_pacer", None)
            if pacer is not None:
                await pacer.wait(timing)
            dispatched = time.monotonic()
            timing["http_dispatch_monotonic_seconds"] = dispatched
            timing["http_request_attempted"] = True
            response = await self._http.send(
                request, stream=True, auth=None, follow_redirects=False
            )
            timing["http_until_headers_seconds"] = time.monotonic() - dispatched
            timing["http_status"] = response.status_code
            if response.status_code in (402, 429):
                self.last_rate_limit_diagnostics = await read_rate_limit_diagnostics(
                    response
                )
                retry = _retry_after_seconds(response.headers.get("Retry-After"))
                if retry is not None:
                    timing["retry_after_seconds"] = retry
            if not 200 <= response.status_code < 300:
                raise ModelAPIError(
                    "model HTTP request failed",
                    code="http_error",
                    status_code=response.status_code,
                )
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise _ResponseValidationError("non_stream_response", "stream")
            parser, pending, data, size = SSECompletion(), b"", [], 0
            read_started = time.monotonic()
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                timing["response_bytes"] = size
                if size > MAX_BYTES:
                    raise _ResponseValidationError("response_size_limit", "stream")
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if not line:
                        if data:
                            try:
                                parser.feed(
                                    b"\n".join(data).decode("utf-8"),
                                    time.monotonic() - dispatched,
                                )
                            except UnicodeError as exc:
                                raise _ResponseValidationError(
                                    "invalid_stream_utf8", "stream"
                                ) from exc
                            data = []
                    elif line.startswith(b"data:"):
                        data.append(line[5:].removeprefix(b" "))
            timing["response_read_seconds"] = time.monotonic() - read_started
            if pending or data:
                raise _ResponseValidationError("truncated_stream", "stream")
            usage_metrics(parser.usage, timing)
            result = parser.result(timing)
            timing["outcome"] = "completed"
            return result
        except asyncio.CancelledError:
            timing["outcome"] = "cancelled"
            raise
        finally:
            try:
                if response is not None:
                    try:
                        await response.aclose()
                    except Exception:
                        # Cleanup must not turn an upstream 402/429 into a
                        # generic connection failure and obscure the bottleneck.
                        if timing.get("outcome") == "completed":
                            raise
            finally:
                if dispatched is not None:
                    timing["http_elapsed_seconds"] = time.monotonic() - dispatched
                timing["total_seconds"] = time.monotonic() - started
                self.last_http_timing = timing
                log_timing("vlm_http_timing", **timing)
