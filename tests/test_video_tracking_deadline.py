"""Actual socket stalls and cancellation/resource contracts; no model inference."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

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


def test_timeout_keeps_admission_until_actual_cleanup_and_reuses_one_thread():
    cleaning, release = threading.Event(), threading.Event()
    calls = []

    async def handler(_request):
        calls.append(threading.get_ident())
        try:
            await asyncio.sleep(60)
        finally:
            cleaning.set()
            while not release.is_set():
                await asyncio.sleep(0.005)

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        timeout_seconds=0.05,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(httpx.TimeoutException):
            infer(client)
        assert cleaning.wait(1)
        for _ in range(3):
            with pytest.raises(httpx.PoolTimeout):
                infer(client)
        assert len(calls) == 1
        thread = client._thread
        release.set()
        wait_inactive(client)
        for _ in range(2):
            with pytest.raises(httpx.TimeoutException):
                infer(client)
            wait_inactive(client)
        assert len(calls) == 3
        assert len(set(calls)) == 1
        assert client._thread is thread
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
