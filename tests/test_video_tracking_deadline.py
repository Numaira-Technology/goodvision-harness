"""Actual socket stalls and cancellation/resource contracts; no model inference."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import httpx
import pytest

from goodvision_harness import video_tracking
from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.video_tracking import (
    MAX_RESPONSE_BYTES,
    VideoFrame,
    VideoTrackingHTTPClient,
)


def infer(client):
    return client.infer_batch((VideoFrame(0, 0, b"jpeg"),), stream_session_id="camera")


def wait_inactive(client):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with client._lock:
            if client._active is None:
                return
        time.sleep(0.005)
    pytest.fail("cancelled operation did not finish cleanup")


@contextmanager
def stalled_server(stage):
    stopped, arrived, disconnected = (threading.Event() for _ in range(3))
    chunks = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            arrived.set()
            if stage == "upload":
                stopped.wait(3)
                return
            self.rfile.read(int(self.headers["content-length"]))
            if stage == "headers":
                stopped.wait(3)
                return
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while not stopped.wait(0.02):
                try:
                    self.wfile.write(b"1\r\n \r\n")
                    self.wfile.flush()
                    chunks.append(1)
                except OSError:
                    disconnected.set()
                    return

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/video/batches", arrived, chunks
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize("stage", ["headers", "body"])
def test_real_socket_headers_and_dripping_body_obey_total_deadline(stage):
    with stalled_server(stage) as (url, arrived, chunks):
        client = VideoTrackingHTTPClient(url, timeout_seconds=1.0)
        started = time.monotonic()
        try:
            with pytest.raises(httpx.TimeoutException):
                infer(client)
            assert time.monotonic() - started < 1.8
            assert arrived.is_set()
            if stage == "body":
                assert (
                    len(chunks) > 2
                )  # Each chunk arrived well inside the I/O timeout.
            wait_inactive(client)
        finally:
            client.close()
        assert not client._thread.is_alive()


def test_real_socket_upload_is_included_in_total_deadline():
    with stalled_server("upload") as (url, arrived, _chunks):
        client = VideoTrackingHTTPClient(url, timeout_seconds=1.0)
        jpeg = b"j" * (6 * 1024 * 1024)
        inputs = tuple(VideoFrame(i, i, jpeg) for i in range(3))
        started = time.monotonic()
        try:
            with pytest.raises(httpx.TimeoutException):
                client.infer_batch(inputs, stream_session_id="camera")
            assert time.monotonic() - started < 1.8
            assert arrived.is_set()
            wait_inactive(client)
        finally:
            client.close()
        assert not client._thread.is_alive()


def test_health_has_five_second_total_budget_even_when_video_budget_is_75():
    cancelled = threading.Event()

    async def handler(_request):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        timeout_seconds=75,
        transport=httpx.MockTransport(handler),
    )
    started = time.monotonic()
    try:
        assert client.gpu_ready() is False
        assert 4.5 <= time.monotonic() - started < 6.5
        assert cancelled.wait(1)
    finally:
        client.close()
    assert not client._thread.is_alive()


def patch_operation_future(monkeypatch, future_class):
    # Do not change the process-wide Future used by asyncio or other clients.
    monkeypatch.setattr(
        video_tracking,
        "concurrent",
        SimpleNamespace(futures=SimpleNamespace(
            Future=future_class,
            TimeoutError=concurrent.futures.TimeoutError,
            CancelledError=concurrent.futures.CancelledError,
        )),
    )


@pytest.mark.parametrize("first_cancel", ["caller", "watchdog"])
def test_timeout_keeps_admission_until_actual_cleanup_and_reuses_one_thread(
    monkeypatch, first_cancel,
):
    entered, cleaning, release, cleaned, watched = (
        threading.Event() for _ in range(5)
    )
    watch_gate = asyncio.Event()
    clock = SimpleNamespace(now=0.0)
    calls, operations = [], []

    class CountedTask(asyncio.Task):
        def __init__(self, *args, **kwargs):
            self.cancel_calls = 0
            super().__init__(*args, **kwargs)

        def cancel(self, *args, **kwargs):
            self.cancel_calls += 1
            return super().cancel(*args, **kwargs)

    class EnteredFuture(concurrent.futures.Future):
        def __init__(self):
            super().__init__()
            operations.append(self)

        def result(self, timeout=None):
            assert entered.wait(2), "request did not enter the handler"
            # Expire only after entry: cold setup belongs to the separate
            # pre-handler tests below, not this cleanup-ownership contract.
            clock.now += 1
            if first_cancel == "watchdog":
                client._loop.call_soon_threadsafe(watch_gate.set)
                assert cleaning.wait(1), "watchdog did not start cleanup"
            return super().result(timeout=0)

    patch_operation_future(monkeypatch, EnteredFuture)
    monkeypatch.setattr(
        video_tracking, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )

    async def handler(_request):
        calls.append(threading.get_ident())
        entered.set()
        try:
            await asyncio.sleep(60)
        finally:
            cleaning.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            cleaned.set()

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        timeout_seconds=0.05,
        transport=httpx.MockTransport(handler),
    )
    original_watch = client._watch_deadline
    original_loop = client._run_loop

    def counted_loop():
        # Task.cancelling() is unavailable on supported Python 3.10. Count the
        # real cancel calls without changing Task's cancellation behavior.
        client._loop.set_task_factory(
            lambda loop, coro, **kwargs: CountedTask(coro, loop=loop, **kwargs)
        )
        original_loop()

    async def watch_after_entry(*args):
        # A real wait_for timer must not expire while the test clock is held
        # during setup. Once released, use the unchanged production watcher.
        await watch_gate.wait()
        try:
            await original_watch(*args)
        finally:
            watched.set()

    monkeypatch.setattr(client, "_watch_deadline", watch_after_entry)
    monkeypatch.setattr(client, "_run_loop", counted_loop)
    worker = None
    try:
        for attempt in range(3):
            for event in (entered, cleaning, release, cleaned, watched):
                event.clear()
            watch_gate.clear()
            with pytest.raises(httpx.TimeoutException):
                infer(client)
            assert entered.is_set() and cleaning.wait(1)
            for _ in range(3):
                with pytest.raises(httpx.PoolTimeout):
                    infer(client)
            assert len(calls) == attempt + 1
            operation = operations[-1]
            assert operation.cancelled()
            client._loop.call_soon_threadsafe(watch_gate.set)
            assert watched.wait(1)
            drained = threading.Event()

            def repeat_cancellation(operation=operation, drained=drained):
                # Caller, watchdog and close-style callbacks may all race.
                for timed_out in (True, False, True):
                    client._cancel_on_loop(operation, timed_out=timed_out)
                drained.set()

            client._loop.call_soon_threadsafe(repeat_cancellation)
            assert drained.wait(1)
            with client._lock:
                assert client._active is operation
                assert not client._task.done()
                assert client._task.cancel_calls == 1
            assert not cleaned.is_set()
            worker = worker or client._thread
            assert client._thread is worker
            release.set()
            wait_inactive(client)
            assert cleaned.is_set()
        assert len(calls) == 3
        assert len(set(calls)) == 1
    finally:
        release.set()
        if client._loop is not None:
            client._loop.call_soon_threadsafe(watch_gate.set)
        client.close()
    assert not client._thread.is_alive()


@pytest.mark.parametrize("stage", ["worker_start", "task_created", "task_expired"])
def test_timeout_before_handler_releases_admission_and_reuses_worker(
    monkeypatch, stage,
):
    blocked, release = threading.Event(), threading.Event()
    calls, created, exchange_entries = [], [], []

    async def handler(_request):
        calls.append(threading.get_ident())
        return httpx.Response(200, json={
            "ready": True, "gpu_ready": True, "backend": "sam3.1_video",
            "native_sam_video_tracking": True, "memory_scope": "batch",
        })

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches", transport=httpx.MockTransport(handler),
    )
    original_start, original_exchange = client._start_operation, client._exchange

    async def record_exchange(*args, **kwargs):
        exchange_entries.append(True)
        return await original_exchange(*args, **kwargs)

    monkeypatch.setattr(client, "_exchange", record_exchange)
    if stage == "worker_start":
        original_loop = client._run_loop

        def gated_loop():
            blocked.set()
            assert release.wait(2)
            original_loop()

        monkeypatch.setattr(client, "_run_loop", gated_loop)
    else:
        class CreatedFuture(concurrent.futures.Future):
            def result(self, timeout=None):
                # Establish task creation before allowing the first caller's
                # ordinary Future wait to expire. Keep its actual 50 ms budget.
                assert blocked.wait(2), "task creation did not occur"
                return super().result(timeout=timeout)

        patch_operation_future(monkeypatch, CreatedFuture)

        def gated_start(operation, *args):
            if stage == "task_created":
                # Queue cancellation ahead of the newly created task's first
                # step. The callback still runs only after the caller expires.
                client._loop.call_soon(client._cancel_on_loop, operation)
            original_start(operation, *args)
            created.append(client._task)
            blocked.set()
            assert release.wait(2)

        monkeypatch.setattr(client, "_start_operation", gated_start)

    try:
        with pytest.raises(httpx.TimeoutException):
            client._request(
                "GET", client._health, deadline=time.monotonic() + 0.05,
            )
        assert blocked.wait(1)
        assert calls == [] and exchange_entries == []
        with client._lock:
            operation = client._active
            assert operation is not None and operation.cancelled()
            if stage == "worker_start":
                assert client._task is None
            else:
                assert created == [client._task] and client._task is not None
                assert not client._task.done()
        for _ in range(3):
            with pytest.raises(httpx.PoolTimeout):
                client._request(
                    "GET", client._health, deadline=time.monotonic() + 0.05,
                )
        worker = client._thread
        monkeypatch.setattr(client, "_start_operation", original_start)
        release.set()
        wait_inactive(client)
        assert calls == []
        assert len(exchange_entries) == (1 if stage == "task_expired" else 0)
        if created:
            assert created[0].done()
            assert created[0].cancelled() == (stage == "task_created")
        # Recovery uses the existing health API's five-second cap, not a
        # fabricated success or a larger budget for the expired request.
        for _ in range(2):
            assert client.gpu_ready()
            wait_inactive(client)
            assert client._thread is worker
        assert len(calls) == 2 and len(set(calls)) == 1
    finally:
        release.set()
        client.close()
    assert not client._thread.is_alive()


def test_close_cancels_pending_headers_and_joins_without_deadlocking():
    entered, cancelled = threading.Event(), threading.Event()
    results = []

    async def handler(_request):
        entered.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches", transport=httpx.MockTransport(handler)
    )

    def request():
        try:
            infer(client)
        except httpx.RequestError:
            results.append("cancelled")

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    assert entered.wait(2)
    started = time.monotonic()
    client.close()
    thread.join(2)
    assert time.monotonic() - started < 1
    assert not thread.is_alive()
    assert not client._thread.is_alive()
    assert cancelled.is_set()
    assert results == ["cancelled"]
    client.close()
    with pytest.raises(httpx.RequestError, match="closed"):
        infer(client)


@pytest.mark.parametrize(
    "body",
    [
        b'{"ready":false,"ready":true}',
        b'{"x":{"y":1,"y":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b"[" * 1500 + b"]" * 1500,
    ],
)
def test_strict_json_rejects_duplicate_nested_fields_constants_and_excess_depth(body):
    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
    )
    try:
        with pytest.raises(RichLabelProtocolError):
            infer(client)
    finally:
        client.close()


class RawStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.read_count = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.read_count += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate", "identity, gzip"])
def test_compressed_response_rejected_before_read_or_decompression(encoding):
    stream = RawStream([b"compressed bytes never read"])
    seen = []

    def handler(request):
        seen.append(request.headers["accept-encoding"])
        return httpx.Response(
            200, headers={"content-encoding": encoding}, stream=stream
        )

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(RichLabelProtocolError, match="compressed"):
            infer(client)
        assert stream.read_count == 0
        assert stream.closed
        assert seen == ["identity"]
    finally:
        client.close()


def test_raw_stream_stops_at_byte_limit_and_closes_response():
    stream = RawStream([b" " * (16 * 1024)] * (MAX_RESPONSE_BYTES // (16 * 1024) + 5))
    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)),
    )
    try:
        with pytest.raises(RichLabelProtocolError, match="byte limit"):
            infer(client)
        assert stream.read_count == MAX_RESPONSE_BYTES // (16 * 1024) + 1
        assert stream.closed
    finally:
        client.close()


def test_slow_json_parse_cannot_deliver_late_success_or_start_parallel_request(
    monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    original = json.loads

    def blocked_parse(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    response = httpx.Response(200, content=b"{}")
    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        timeout_seconds=0.05,
        transport=httpx.MockTransport(lambda _: response),
    )
    monkeypatch.setattr("goodvision_harness.video_tracking.json.loads", blocked_parse)
    started = time.monotonic()
    try:
        with pytest.raises(httpx.TimeoutException):
            infer(client)
        assert time.monotonic() - started < 0.5
        assert entered.is_set()
        with pytest.raises(httpx.PoolTimeout):
            infer(client)
    finally:
        release.set()
        wait_inactive(client)
        client.close()


def test_unused_client_closes_without_creating_a_thread():
    client = VideoTrackingHTTPClient("http://worker/v1/video/batches")
    client.close()
    assert client._thread is None


def test_oversized_sequence_is_rejected_before_materializing_or_starting_io():
    class TooManyFrames:
        def __len__(self):
            return 10**9

        def __iter__(self):
            pytest.fail("oversized input was materialized")

    client = VideoTrackingHTTPClient("http://worker/v1/video/batches")
    try:
        with pytest.raises(ValueError, match="1-16"):
            client.infer_batch(TooManyFrames(), stream_session_id="camera")
        assert client._thread is None
    finally:
        client.close()
