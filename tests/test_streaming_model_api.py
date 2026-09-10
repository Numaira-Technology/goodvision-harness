from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.streaming_model_api import complete_stream


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self.chunks = chunks
        self.started = asyncio.Event()
        self.closed = False
        self.exhausted = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        for delay, chunk in self.chunks:
            await asyncio.sleep(delay)
            yield chunk
        self.exhausted = True

    async def aclose(self) -> None:
        self.closed = True


def _kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "settings": ModelAPISettings(provider="openrouter"),
        "model": "test-model", "messages": [{"role": "user", "content": "test"}],
        "api_key": "fake-key", "deadline_seconds": 1.0, **overrides,
    }


def _event(content: Any = None, finish: Any = None) -> bytes:
    value = {
        "id": "gen-test", "provider": "DeepInfra",
        "choices": [{"index": 0, "delta": {"content": content},
                     "finish_reason": finish}],
    }
    return ("data: " + json.dumps(value) + "\n\n").encode()


def _pending_tasks() -> list[asyncio.Task[Any]]:
    return [
        task for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]


def test_reuses_caller_client_with_dynamic_keys_and_drains_eof() -> None:
    requests: list[httpx.Request] = []
    streams: list[_Chunks] = []
    usage = {
        "choices": [], "usage": {
            "prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16,
            "cost": 0.001, "secret": "do-not-retain",
            "completion_tokens_details": {"reasoning_tokens": 2, "secret": 9},
        },
    }

    async def handle(request: httpx.Request) -> httpx.Response:
        if streams:
            assert streams[-1].exhausted and streams[-1].closed
        requests.append(request)
        stream = _Chunks([
            (0, b": heartbeat\n\n"), (0, _event('{"o":')),
            (0, _event("[]}", "stop")),
            (0, ("data: " + json.dumps(usage) + "\n\n").encode()),
            (0, b"data: [DONE]\n\n"),
            (0.01, b": trailing keepalive\n\ndata:\n\n"),
        ])
        streams.append(stream)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream; charset=utf-8"},
            stream=stream,
        )

    async def run() -> list[dict[str, Any]]:
        results = []
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), headers={"X-Client": "unchanged"},
        ) as client:
            for key in ("first-key", "second-key"):
                results.append(await complete_stream(client, **_kwargs(
                    api_key=key,
                    settings=ModelAPISettings(
                        provider="openrouter", extra_headers={"authorization": "old"},
                    ),
                    request_parameters={"max_tokens": 100, "provider": {"only": ["x"]}},
                )))
                assert not client.is_closed
                assert "authorization" not in client.headers
                assert client.headers["X-Client"] == "unchanged"
                assert not _pending_tasks()
        assert client.is_closed
        return results

    results = asyncio.run(run())
    assert [request.headers["Authorization"] for request in requests] == [
        "Bearer first-key", "Bearer second-key",
    ]
    for request in requests:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["max_tokens"] == 100 and payload["provider"] == {"only": ["x"]}
        assert len(request.headers.get_list("authorization")) == 1
        assert "trace" not in request.extensions
    for result in results:
        assert result["choices"][0]["message"]["content"] == '{"o":[]}'
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["id"] == "gen-test" and result["provider"] == "DeepInfra"
        assert result["usage"] == {
            "prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16,
            "cost": 0.001, "completion_tokens_details": {"reasoning_tokens": 2},
        }
        timing = result["_timing"]
        assert (
            timing["response_headers_seconds"] <= timing["first_content_token_seconds"]
        )
        assert timing["complete_seconds"] < timing["response_eof_seconds"]
        assert timing["response_eof_seconds"] <= timing["elapsed_seconds"]
        assert (
            timing["first_content_token_seconds"]
            <= timing["last_content_token_seconds"]
            <= timing["finish_reason_seconds"]
            <= timing["complete_seconds"]
        )
        assert result["_transport"] == {"trace_enabled": False}
        saved = json.dumps(result)
        assert "first-key" not in saved and "second-key" not in saved
        assert "do-not-retain" not in saved and "authorization" not in saved.lower()


@pytest.mark.parametrize("chunks", [
    [_event("{}", "stop")],
    [_event("{", "length"), b"data: [DONE]\n\n"],
    [_event("{}", "content_filter"), b"data: [DONE]\n\n"],
    [_event("   ", "stop"), b"data: [DONE]\n\n"],
    [_event([], "stop"), b"data: [DONE]\n\n"],
    [b'data: {"error":{"message":"remote-secret"}}\n\n'],
    [b"data: invalid remote-secret\n\n"],
    [_event("{}", "stop"), b"data: [DONE]\n\n", _event("extra")],
])
def test_incomplete_and_error_streams_never_return_partial_response(
    chunks: list[bytes],
) -> None:
    stream = _Chunks([(0, chunk) for chunk in chunks])

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(ModelAPIError) as caught:
                await complete_stream(client, **_kwargs())
            assert "remote-secret" not in str(caught.value)
            assert not client.is_closed and stream.closed
            assert not _pending_tasks()

    asyncio.run(run())


def test_timing_separates_content_finish_done_and_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        "goodvision_harness.streaming_model_api.time.perf_counter", lambda: clock[0],
    )
    usage = {
        "choices": [], "service_tier": "priority",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    class TimedChunks(_Chunks):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for timestamp, chunk in [
                (0.5, _event("{")), (0.9, _event("}")),
                (1.0, _event(finish="stop")),
                (1.1, ("data: " + json.dumps(usage) + "\n\n").encode()),
                (1.2, b"data: [DONE]\n\n"), (1.7, b": tail\n\n"),
            ]:
                clock[0] = timestamp
                yield chunk
            self.exhausted = True

    stream = TimedChunks([])

    async def handle(request: httpx.Request) -> httpx.Response:
        clock[0] = 0.2
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def run() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await complete_stream(client, **_kwargs())

    result = asyncio.run(run())
    assert result["_timing"] == {
        "response_headers_seconds": 0.2,
        "first_content_token_seconds": 0.5,
        "last_content_token_seconds": 0.9,
        "finish_reason_seconds": 1.0,
        "complete_seconds": 1.2,
        "response_eof_seconds": 1.7,
        "elapsed_seconds": 1.7,
        "generation_seconds": 0.4,
        "completion_tail_seconds": 0.3,
        "eof_tail_seconds": 0.5,
    }
    assert result["service_tier"] == "priority"
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}
    assert stream.closed and stream.exhausted


@pytest.mark.parametrize("protocol", ["http11", "http2"])
def test_transport_trace_measures_phases_and_reuse_without_retaining_payloads(
    monkeypatch: pytest.MonkeyPatch, protocol: str,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        "goodvision_harness.streaming_model_api.time.perf_counter", lambda: clock[0],
    )
    count = 0
    http_version = b"HTTP/1.1" if protocol == "http11" else b"HTTP/2"

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal count
        trace = request.extensions["trace"]
        phases = [] if count else ["connection.connect_tcp", "connection.start_tls"]
        phases.extend([
            f"{protocol}.send_request_headers", f"{protocol}.send_request_body",
            f"{protocol}.receive_response_headers",
        ])
        for phase in phases:
            await trace(phase + ".started", {
                "request": request, "host": "private-host", "headers": request.headers,
            })
            clock[0] += 0.1
            await trace(phase + ".complete", {"return_value": "private-response"})
        await trace("unknown.private-phase.started", {"secret": "private-trace"})
        count += 1
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            extensions={"http_version": http_version},
            stream=_Chunks([(0, _event("{}", "stop")), (0, b"data: [DONE]\n\n")]),
        )

    async def run() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return [
                await complete_stream(client, **_kwargs(trace_transport=True))
                for _ in range(2)
            ]

    first, reused = asyncio.run(run())
    for index, result in enumerate((first, reused)):
        assert result["_transport"] == {
            "trace_enabled": True, "trace_events_observed": True,
            "tcp_connect_count": 1 if index == 0 else 0,
            "http_version": http_version.decode("ascii"),
        }
        for name in (
            "request_headers_seconds", "request_body_seconds",
            "wait_response_headers_seconds",
        ):
            assert result["_timing"][name] == 0.1
        assert "private" not in json.dumps(result)
        assert "fake-key" not in json.dumps(result)
    assert first["_timing"]["connect_tcp_seconds"] == 0.1
    assert first["_timing"]["tls_seconds"] == 0.1
    assert "connect_tcp_seconds" not in reused["_timing"]
    assert "tls_seconds" not in reused["_timing"]


@pytest.mark.parametrize("after_done", [False, True])
def test_wall_deadline_includes_eof_and_closes_only_the_response(
    after_done: bool,
) -> None:
    chunks = [(0.5, _event("{}", "stop"))]
    if after_done:
        chunks = [
            (0, _event("{}", "stop")), (0, b"data: [DONE]\n\n"),
            (0.5, b": unfinished tail\n\n"),
        ]
    stream = _Chunks(chunks)

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            started = time.perf_counter()
            with pytest.raises(ModelAPIError, match="wall deadline"):
                await complete_stream(client, **_kwargs(deadline_seconds=0.03))
            assert time.perf_counter() - started < 0.4
            assert stream.closed and not stream.exhausted and not client.is_closed
            assert not _pending_tasks()

    asyncio.run(run())


def test_external_cancellation_propagates_after_response_cleanup() -> None:
    stream = _Chunks([(10, _event("{}", "stop"))])

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            task = asyncio.create_task(complete_stream(client, **_kwargs()))
            await stream.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed and not client.is_closed
            assert not _pending_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("status", [302, 503])
def test_http_error_never_reads_body_retries_or_follows_redirects(status: int) -> None:
    stream = _Chunks([(0, b"private remote error text")])
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, headers={
            "location": "https://other.example/?secret=never-follow",
        }, stream=stream)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), follow_redirects=True,
        ) as client:
            with pytest.raises(ModelAPIError, match=f"HTTP {status}") as caught:
                await complete_stream(client, **_kwargs())
            assert len(requests) == 1
            assert stream.closed and not stream.started.is_set()
            assert not client.is_closed
            assert "secret" not in str(caught.value)
            assert "private" not in str(caught.value)

    asyncio.run(run())


def test_network_failure_hides_exception_url_and_credentials() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("https://example.invalid/?secret=private-key")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(ModelAPIError) as caught:
                await complete_stream(client, **_kwargs())
            assert str(caught.value) == "model stream request failed"
            assert caught.value.__suppress_context__
            assert not client.is_closed

    asyncio.run(run())
