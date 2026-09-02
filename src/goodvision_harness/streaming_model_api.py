"""Complete one SSE chat response using a caller-owned asynchronous HTTP pool."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from typing import Any

import httpx

from .model_api import ModelAPIError, ModelAPISettings


def _safe_name(value: Any) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= 256 and value.isprintable():
        return value
    return None


def _safe_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        number = value.get(key)
        if (
            isinstance(number, (int, float)) and not isinstance(number, bool)
            and math.isfinite(number) and number >= 0
        ):
            result[key] = number
    for key in ("prompt_tokens_details", "completion_tokens_details"):
        details = value.get(key)
        if isinstance(details, Mapping):
            result[key] = {
                name: number for name, number in details.items()
                if isinstance(name, str) and name.endswith("_tokens")
                and isinstance(number, int) and not isinstance(number, bool)
                and number >= 0
            }
    return result


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    data: list[str] = []
    async with aclosing(response.aiter_lines()) as lines:
        async for line in lines:
            if not line:
                if data:
                    yield "\n".join(data)
                    data.clear()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


async def _collect(
    client: httpx.AsyncClient,
    *,
    settings: ModelAPISettings,
    payload: dict[str, Any],
    api_key: str,
    deadline_seconds: float,
    started: float,
) -> dict[str, Any]:
    timing: dict[str, float] = {}
    response: dict[str, Any] = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                     "finish_reason": None}],
        "usage": {},
        "_timing": timing,
    }
    choice_result = response["choices"][0]
    message = choice_result["message"]
    done = False

    def elapsed() -> float:
        return round(time.perf_counter() - started, 6)

    headers = httpx.Headers(settings.extra_headers)
    headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    async with client.stream(
        "POST", settings.endpoint, json=payload, headers=headers,
        timeout=httpx.Timeout(deadline_seconds), follow_redirects=False,
    ) as stream:
        timing["response_headers_seconds"] = elapsed()
        if stream.status_code != 200:
            raise ModelAPIError(f"model stream returned HTTP {stream.status_code}")
        media_type = stream.headers.get("content-type", "").split(";", 1)[0]
        if media_type.strip().lower() != "text/event-stream":
            raise ModelAPIError("model stream has unexpected content type")
        async with aclosing(_sse_data(stream)) as events:
            async for data in events:
                if not data.strip():
                    continue
                if done:
                    raise ModelAPIError("model stream contains data after DONE")
                if data.strip() == "[DONE]":
                    done = True
                    timing["complete_seconds"] = elapsed()
                    # Reading EOF returns the connection to the caller's pool.
                    continue
                try:
                    chunk = json.loads(data)
                except (TypeError, ValueError):
                    raise ModelAPIError("model stream contains invalid JSON") from None
                if not isinstance(chunk, Mapping):
                    raise ModelAPIError("model stream contains an invalid chunk")
                if "error" in chunk:
                    raise ModelAPIError("model stream reported a remote error")
                for field in ("id", "provider"):
                    value = _safe_name(chunk.get(field))
                    if value is not None:
                        response[field] = value
                response["usage"].update(_safe_usage(chunk.get("usage")))
                choices = chunk.get("choices", [])
                if not isinstance(choices, list):
                    raise ModelAPIError("model stream contains invalid choices")
                for choice in choices:
                    if not isinstance(choice, Mapping):
                        raise ModelAPIError("model stream contains an invalid choice")
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta", {})
                    if not isinstance(delta, Mapping):
                        raise ModelAPIError("model stream contains an invalid delta")
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise ModelAPIError("model stream contains invalid content")
                    if content:
                        timing.setdefault("first_content_token_seconds", elapsed())
                        message["content"] += content
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        if not isinstance(finish, str):
                            raise ModelAPIError(
                                "model stream has invalid finish reason"
                            )
                        choice_result["finish_reason"] = finish
        timing["response_eof_seconds"] = elapsed()
        if not done:
            raise ModelAPIError("model stream is missing DONE")
        if choice_result["finish_reason"] == "length":
            raise ModelAPIError("model stream was truncated")
        if choice_result["finish_reason"] != "stop":
            raise ModelAPIError("model stream did not finish normally")
        if not message["content"].strip():
            raise ModelAPIError("model stream contains no assistant content")
    timing["elapsed_seconds"] = elapsed()
    return response


async def complete_stream(
    client: httpx.AsyncClient,
    *,
    settings: ModelAPISettings,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    request_parameters: Mapping[str, Any] | None = None,
    api_key: str,
    deadline_seconds: float,
) -> dict[str, Any]:
    """Return a completed chat response or raise a safe ModelAPIError.

    The supplied client stays open. The wall deadline includes the final EOF,
    no retry is attempted, and external cancellation propagates unchanged after
    response cleanup. Response text still needs the caller's domain validation.
    """

    if (
        isinstance(deadline_seconds, bool)
        or not isinstance(deadline_seconds, (int, float))
        or not math.isfinite(deadline_seconds) or deadline_seconds <= 0
    ):
        raise ModelAPIError("model stream deadline must be finite and positive")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ModelAPIError("model stream API key is missing")
    if not isinstance(model, str) or not model.strip() or not messages:
        raise ModelAPIError("model stream requires model and messages")
    if request_parameters is not None and not isinstance(request_parameters, Mapping):
        raise ModelAPIError("model stream parameters must be a mapping")
    parameters = dict(request_parameters or {})
    if {"model", "messages", "stream", "stream_options"} & parameters.keys():
        raise ModelAPIError("model stream parameters override a fixed field")
    payload = {
        **parameters, "model": model, "messages": list(messages), "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    try:
        return await asyncio.wait_for(
            _collect(
                client, settings=settings, payload=payload, api_key=api_key.strip(),
                deadline_seconds=deadline_seconds, started=started,
            ),
            timeout=deadline_seconds,
        )
    except (asyncio.TimeoutError, TimeoutError):
        raise ModelAPIError("model stream exceeded its wall deadline") from None
    except ModelAPIError:
        raise
    except httpx.TimeoutException:
        raise ModelAPIError("model stream socket timed out") from None
    except Exception:
        # HTTPX exceptions can include credentials, URL queries, or remote text.
        raise ModelAPIError("model stream request failed") from None
