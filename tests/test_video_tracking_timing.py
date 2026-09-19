"""Exercise timing through real HTTP clients; transport/clock are deterministic."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from test_perception_client import RouteServer
from test_video_tracking_session import SessionServer

from goodvision_harness import video_tracking
from goodvision_harness.perception import PerceptionSessionHTTPClient
from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.video_tracking import VideoFrame, VideoTrackingHTTPClient
from goodvision_harness.video_tracking_session import VideoSessionTrackingHTTPClient


class Clock:
    def __init__(self):
        self.value = 100.0

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class TimedBody(httpx.AsyncByteStream):
    def __init__(self, clock, chunks, *, error=None):
        self.clock, self.chunks, self.error = clock, chunks, error

    async def __aiter__(self):
        for content, seconds in self.chunks:
            self.clock.advance(seconds)
            yield content
        if self.error is not None:
            raise self.error


def timing_records(caplog):
    return [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.name == "uvicorn.error"
        and record.getMessage().startswith("sam_client_http_timing ")
    ]


@pytest.mark.parametrize("message,reason", [
    ("Server disconnected without sending a response.", "eof_before_headers"),
    ("peer closed connection (incomplete body)", "incomplete_body"),
    ("illegal header line", "malformed_headers"),
    ("some other protocol error", "other"),
])
def test_protocol_failure_logs_only_a_safe_reason_not_server_detail(
    monkeypatch, caplog, message, reason,
):
    monkeypatch.setattr(video_tracking, "time", Clock())
    private = " https://private-user:private-token@worker-private.test/secret"

    def broken(_request):
        raise httpx.RemoteProtocolError(message + private) from OSError(private)

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with pytest.raises(httpx.RemoteProtocolError):
            exchange(httpx.MockTransport(broken))
    record, = timing_records(caplog)
    assert record["protocol_error_reason"] == reason
    assert record["cause_type"] == "OSError"
    assert "private-token" not in json.dumps(record)
    assert "worker-private.test" not in json.dumps(record)


def payload():
    return {
        "stream_session_id": "source.001",
        "request_id": "append.001",
        "is_last_batch": False,
        "frames": [
            {
                "frame_sequence": 3,
                "timestamp_seconds": 0.25,
                "jpeg_base64": "private-image-body",
            },
            {
                "frame_sequence": 15,
                "timestamp_seconds": 1.25,
                "jpeg_base64": "private-image-body",
            },
        ],
    }


def exchange(transport, factory=payload, parser=None, *, deadline=1000.0):
    client = VideoTrackingHTTPClient(
        "https://private-user:private-token@worker-private.test/v1/video/sessions",
        transport=transport,
    )

    async def run():
        try:
            return await client._exchange(
                "POST", client.endpoint + "/private-session/append", factory, parser,
                deadline=deadline,
            )
        finally:
            if client._client is not None:
                await client._client.aclose()

    return asyncio.run(run())


@pytest.mark.parametrize("route", ["legacy", "sam_live"])
def test_public_append_timing_correlates_request_and_sparse_sources(route, caplog):
    server = RouteServer(route) if route == "sam_live" else SessionServer()
    client_type = (
        PerceptionSessionHTTPClient
        if route == "sam_live" else VideoSessionTrackingHTTPClient
    )
    options = {"method": route} if route == "sam_live" else {}
    client = client_type(
        "http://worker-private.test/v1/video/sessions",
        transport=httpx.MockTransport(server), timeout_seconds=5,
        **options,
    )
    source = (
        VideoFrame(3, 0.25, b"private-original-image"),
        VideoFrame(15, 1.25, b"private-original-image"),
    )
    try:
        with caplog.at_level("INFO", logger="uvicorn.error"):
            client.open_session("source.001")
            result = client.append(
                source, stream_session_id="source.001", request_id="append.001",
                is_last_batch=True,
            )
        fields, = timing_records(caplog)
        assert (
            fields["request_id"] == server.requests[-1]["request_id"]
            == result.request_id
        )
        assert fields["stream_session_id"] == result.stream_session_id == "source.001"
        assert fields["input_frames"] == 2
        assert fields["route"] == route
        assert fields["first_source_sequence"] == 3
        assert fields["last_source_sequence"] == 15
        assert fields["first_source_timestamp_seconds"] == 0.25
        assert fields["last_source_timestamp_seconds"] == 1.25
        assert fields["final"] is True
        assert fields["outcome"] == "ok" and fields["http_status"] == 200
        assert [item.frame_sequence for item in result.frames] == [3, 15]
        record_text = json.dumps(fields)
        for private in (
            "worker-private", "private-original-image", "jpeg_base64",
            base64.b64encode(source[0].jpeg_bytes).decode(), result.predictor_epoch,
        ):
            assert private not in record_text
    finally:
        client.close()


def test_append_stage_times_measure_real_http_boundaries(monkeypatch, caplog):
    clock, received = Clock(), []
    monkeypatch.setattr(video_tracking, "time", clock)

    def prepare():
        clock.advance(2)
        return payload()

    def handler(request):
        received.append(request)
        clock.advance(3)
        return httpx.Response(
            200, stream=TimedBody(clock, [(b'{"ok":', 4), (b"true}", 6)])
        )

    def loads(*args, **kwargs):
        clock.advance(7)
        return json.loads(*args, **kwargs)

    monkeypatch.setattr(video_tracking, "json", SimpleNamespace(loads=loads))

    def validate(value):
        clock.advance(11)
        assert value == {"ok": True}
        return "validated-result"

    with caplog.at_level("INFO", logger="uvicorn.error"):
        result = exchange(httpx.MockTransport(handler), prepare, validate)
    assert result == "validated-result"
    fields, = timing_records(caplog)
    expected = {
        "payload_prepare_seconds": 2,
        "http_until_headers_seconds": 3,
        "response_read_seconds": 10,
        "response_json_seconds": 7,
        "result_validation_seconds": 11,
    }
    for name, seconds in expected.items():
        assert fields[name] == seconds
    assert fields["total_seconds"] == sum(expected.values()) == 33
    assert fields["request_bytes"] == len(received[0].content)
    assert fields["response_bytes"] == len(b'{"ok":true}')
    assert fields["transport_trace_observed"] is False
    assert "send_request_body_seconds" not in fields  # unavailable is not zero
    assert "request_body_completed" not in fields


@pytest.mark.parametrize("protocol", ["http11", "http2"])
def test_transport_trace_separates_upload_and_header_wait_without_private_info(
    protocol, monkeypatch, caplog,
):
    clock, requests = Clock(), []
    monkeypatch.setattr(video_tracking, "time", clock)
    secret = "private-user private-token private-image-body private-session"

    async def handler(request):
        requests.append(request)
        trace = request.extensions["trace"]
        # Info may hold entire request/response objects. It must never be read.
        info = {"request": request, "body": secret, "return_value": object()}
        phases = [
            ("connection.connect_tcp", 0.3),
            ("connection.start_tls", 0.2),
            (f"{protocol}.send_request_headers", 0.1),
            (f"{protocol}.send_request_body", 1.36),
            (f"{protocol}.receive_response_headers", 0.385),
        ]
        await trace("private.unknown.started", info)
        for name, seconds in phases:
            await trace(name + ".started", info)
            clock.advance(seconds)
            await trace(name + ".complete", info)
        return httpx.Response(200, json={"ok": True})

    with caplog.at_level("INFO", logger="uvicorn.error"):
        assert exchange(httpx.MockTransport(handler)) == {"ok": True}
    fields, = timing_records(caplog)
    assert fields["transport_trace_observed"] is True
    assert fields["request_body_completed"] is True
    assert fields["tcp_connections_opened"] == 1
    assert fields["tcp_connect_seconds"] == pytest.approx(0.3)
    assert fields["tls_handshake_seconds"] == pytest.approx(0.2)
    assert fields["send_request_headers_seconds"] == pytest.approx(0.1)
    assert fields["send_request_body_seconds"] == pytest.approx(1.36)
    assert fields["receive_response_headers_seconds"] == pytest.approx(0.385)
    assert fields["http_until_headers_seconds"] == pytest.approx(2.345)
    assert len(requests) == 1
    assert requests[0].headers["content-length"] == str(len(requests[0].content))
    assert "transfer-encoding" not in requests[0].headers
    assert not any(key.startswith("_") for key in fields)
    assert not any(word in json.dumps(fields) for word in secret.split())


@pytest.mark.parametrize("terminal_event", ["failed", "cancelled_without_event"])
def test_transport_trace_records_incomplete_body_on_failure_or_cancellation(
    terminal_event, monkeypatch, caplog,
):
    clock = Clock()
    monkeypatch.setattr(video_tracking, "time", clock)
    error = (
        httpx.WriteTimeout("private-token") if terminal_event == "failed"
        else asyncio.CancelledError("private-token")
    )

    async def handler(request):
        trace = request.extensions["trace"]
        await trace("http11.send_request_body.started", {"request": request})
        clock.advance(0.7)
        if terminal_event == "failed":
            await trace("http11.send_request_body.failed", {"exception": error})
        raise error

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with pytest.raises(type(error)) as caught:
            exchange(httpx.MockTransport(handler))
    assert caught.value is error
    fields, = timing_records(caplog)
    assert fields["send_request_body_seconds"] == pytest.approx(0.7)
    assert fields["request_body_completed"] is False
    assert fields["transport_trace_observed"] is True
    assert not any(key.startswith("_") for key in fields)
    assert "private-token" not in json.dumps(fields)


def test_real_httpcore_trace_preserves_prepared_body_and_reuses_connection(caplog):
    # Loopback HTTP only: exercise the real trace API, not a model or external
    # endpoint. Timing magnitudes are deliberately not asserted as performance.
    seen, connections = [], []
    wire = json.dumps(payload()).encode()
    prepared = video_tracking.SerializedVideoPayload(wire, payload(), 0.1, {})

    async def handle(reader, writer):
        connections.append(writer)
        try:
            while True:
                try:
                    headers = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    break
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                seen.append(await reader.readexactly(length))
                assert b"transfer-encoding:" not in headers.lower()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n"
                    b"Content-Type: application/json\r\n\r\n{\"ok\":true}"
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def run():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = VideoTrackingHTTPClient(f"http://127.0.0.1:{port}/v1/video/sessions")
        async with server:
            try:
                for _ in range(2):
                    assert await client._exchange(
                        "POST", client.endpoint + "/session/append",
                        lambda: prepared, None,
                        deadline=asyncio.get_running_loop().time() + 5,
                    ) == {"ok": True}
            finally:
                if client._client is not None:
                    await client._client.aclose()

    with caplog.at_level("INFO", logger="uvicorn.error"):
        asyncio.run(run())
    first, second = timing_records(caplog)
    assert len(connections) == 1 and seen == [wire, wire]
    assert first["tcp_connections_opened"] == 1
    assert "tcp_connections_opened" not in second
    for fields in (first, second):
        assert fields["transport_trace_observed"] is True
        assert fields["request_body_completed"] is True
        assert fields["send_request_body_seconds"] >= 0
        assert fields["receive_response_headers_seconds"] >= 0
        assert fields["request_bytes"] == len(wire)
        assert "private-image-body" not in json.dumps(fields)


@pytest.mark.parametrize("failure", ["headers", "read", "json", "validation"])
def test_failed_append_records_phase_without_private_error_content(
    failure, monkeypatch, caplog
):
    clock = Clock()
    monkeypatch.setattr(video_tracking, "time", clock)
    secret = "private-image-body https://worker-private.test/private-token"

    def handler(_request):
        clock.advance(3)
        if failure == "headers":
            raise httpx.ReadTimeout(secret)
        return httpx.Response(
            200,
            stream=TimedBody(
                clock, [(b'{"ok":true}' if failure != "json" else b"{private", 5)],
                error=httpx.ReadTimeout(secret) if failure == "read" else None,
            ),
        )

    def validate(value):
        clock.advance(7)
        raise RichLabelProtocolError(secret)

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with pytest.raises((httpx.ReadTimeout, RichLabelProtocolError)):
            exchange(httpx.MockTransport(handler), parser=validate)
    fields, = timing_records(caplog)
    expected = {
        "headers": ("http_until_headers", "ReadTimeout", 3),
        "read": ("response_read", "ReadTimeout", 5),
        "json": ("response_json", "RichLabelProtocolError", 0),
        "validation": ("result_validation", "RichLabelProtocolError", 7),
    }
    phase, error_type, duration = expected[failure]
    assert fields["outcome"] == (
        "timeout" if failure in {"headers", "read"} else "error"
    )
    assert fields["request_id"] == "append.001"
    assert fields["phase"] == phase and fields["error_type"] == error_type
    assert fields[f"{phase}_seconds"] == duration
    record_text = json.dumps(fields)
    for private in (
        "private-image-body", "private-token", "worker-private", "private-user",
        "private-session", "jpeg_base64", "{private",
    ):
        assert private not in record_text


def test_unsafe_correlation_values_are_omitted(monkeypatch, caplog):
    monkeypatch.setattr(video_tracking, "time", Clock())
    invalid = payload() | {
        "request_id": "private\nforged-log-line",
        "stream_session_id": "https://private-source.test/path",
    }
    with caplog.at_level("INFO", logger="uvicorn.error"):
        assert exchange(
            httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True})),
            lambda: invalid,
        ) == {"ok": True}
    fields, = timing_records(caplog)
    assert fields["request_id"] is None and fields["stream_session_id"] is None
    assert "private" not in json.dumps(fields)


def test_preparation_deadline_records_work_without_opening_network(monkeypatch, caplog):
    clock, requests = Clock(), []
    monkeypatch.setattr(video_tracking, "time", clock)

    def prepare():
        clock.advance(11)
        return payload()

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with pytest.raises(httpx.TimeoutException):
            exchange(httpx.MockTransport(requests.append), prepare, deadline=105)
    fields, = timing_records(caplog)
    assert requests == []
    assert fields["phase"] == "payload_prepare"
    assert fields["payload_prepare_seconds"] == fields["total_seconds"] == 11
    assert fields["request_id"] == "append.001"
    assert fields["error_type"] == "TimeoutException"
    assert fields["outcome"] == "timeout"


def test_late_validated_response_logs_timeout_and_keeps_public_deadline(
    monkeypatch, caplog
):
    clock = Clock()
    monkeypatch.setattr(video_tracking, "time", clock)
    client = VideoTrackingHTTPClient(
        "http://worker.test/v1/video/sessions",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"ok": True})
        ),
    )

    def validate(value):
        clock.advance(11)
        return value

    try:
        with caplog.at_level("INFO", logger="uvicorn.error"):
            with pytest.raises(httpx.TimeoutException):
                client._request(
                    "POST", client.endpoint + "/session/append", deadline=105,
                    payload_factory=payload, result_parser=validate,
                )
        fields, = timing_records(caplog)
        assert fields["outcome"] == "timeout"
        assert fields["deadline_exceeded"] is True
        assert fields["phase"] == "result_validation"
        assert fields["result_validation_seconds"] == fields["total_seconds"] == 11
        assert fields["request_id"] == "append.001"
    finally:
        client.close()


@pytest.mark.parametrize("past_deadline", [False, True])
def test_cancellation_timing_distinguishes_deadline_without_replacing_exception(
    past_deadline, monkeypatch, caplog
):
    clock = Clock()
    monkeypatch.setattr(video_tracking, "time", clock)
    original = asyncio.CancelledError()

    def cancel(_request):
        clock.advance(11 if past_deadline else 1)
        raise original

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with pytest.raises(asyncio.CancelledError) as caught:
            exchange(httpx.MockTransport(cancel), deadline=105)
    assert caught.value is original
    fields, = timing_records(caplog)
    assert fields["outcome"] == ("timeout" if past_deadline else "error")
    assert fields["error_type"] == "CancelledError"
    assert fields["phase"] == "http_until_headers"


@pytest.mark.parametrize("failed_request", [False, True])
def test_logging_handler_failure_does_not_change_result(
    failed_request, monkeypatch, caplog
):
    monkeypatch.setattr(video_tracking, "time", Clock())

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise RuntimeError("logging storage unavailable")

    handler = BrokenHandler()
    logger = logging.getLogger("uvicorn.error")
    logger.addHandler(handler)
    expected_error = httpx.ReadTimeout("original transport error")

    def request(_):
        if failed_request:
            raise expected_error
        return httpx.Response(200, json={"ok": True})

    try:
        with caplog.at_level("INFO", logger="uvicorn.error"):
            if failed_request:
                with pytest.raises(httpx.ReadTimeout) as caught:
                    exchange(httpx.MockTransport(request))
                assert caught.value is expected_error
            else:
                assert exchange(httpx.MockTransport(request)) == {"ok": True}
    finally:
        logger.removeHandler(handler)
