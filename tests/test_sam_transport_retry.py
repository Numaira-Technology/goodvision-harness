"""Bounded SAM transport recovery against the real causal worker protocol.

Inference is an explicit deterministic double; these are never GPU tests.
"""

import asyncio
import base64
import json
import threading
import time
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from test_perception_client import RouteServer
from test_perception_worker_routes import causal_engine
from test_sam31_continuous_worker import payload
from test_video_tracking_session import SessionServer, frames

from goodvision_harness.perception import METHODS, PerceptionSessionHTTPClient
from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.video_tracking import VideoFrame
from goodvision_harness.video_tracking_session import VideoSessionTrackingHTTPClient
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import register_session_routes

TRANSIENT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
)


def causal_app():
    selected = causal_engine("sam_live")
    selected.backend.route_metadata = lambda session, snapshots: {
        "confirmation_policy": METHODS["sam_live"].confirmation_policy,
        "observations": [
            {"frame_sequence": row["frame_sequence"], "provisional_object_ids": ["7"]}
            for row in snapshots
        ],
    }
    app = FastAPI()
    register_session_routes(
        app,
        selected,
        threading.Lock(),
        decode_jpeg,
        api_prefix="/v1/perception/sam_live",
        health_path="/health/perception/sam_live",
        legacy=False,
    )
    return selected, app


def real_frame(index):
    row = json.loads(payload(1, start=index))["frames"][0]
    return VideoFrame(
        row["frame_sequence"],
        row["timestamp_seconds"],
        base64.b64decode(row["jpeg_base64"]),
    )


class LostResponseTransport(httpx.AsyncBaseTransport):
    """Complete the real ASGI request, then lose its response on the wire."""

    def __init__(self, app):
        self.inner = httpx.ASGITransport(app=app)
        self.requests = []
        self.lost = set()

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        if request.url.path.endswith("/append"):
            body = bytes(request.content)
            self.requests.append(body)
            data = json.loads(body)
            # Exercise lost first-frame and lost EOF responses, with an ordinary
            # later source between them. Replays must not advance native state.
            lose = data["request_id"] in {"first", "eof"}
            if lose and data["request_id"] not in self.lost:
                assert response.status_code == 200
                await response.aread()
                await response.aclose()
                self.lost.add(data["request_id"])
                raise httpx.RemoteProtocolError("test response lost", request=request)
        return response

    async def aclose(self):
        await self.inner.aclose()


@pytest.mark.parametrize("prepared", [False, True], ids=["direct", "prepared"])
def test_completed_worker_response_loss_replays_identical_body_and_preserves_fifo_eof(
    prepared,
):
    selected, app = causal_app()
    transport = LostResponseTransport(app)
    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=transport,
        timeout_seconds=3,
    )
    try:
        session = client.open_session("camera")
        source = (real_frame(0),)
        packet = (
            client.prepare_append(
                source,
                stream_session_id="camera",
                request_id="first",
            )
            if prepared
            else None
        )
        first = client.append(
            source,
            stream_session_id="camera",
            request_id="first",
            prepared=packet,
        )
        assert selected.backend.calls == 1
        assert transport.requests[0] == transport.requests[1]
        if packet is not None:
            assert transport.requests[0] == packet.payload.body
        assert client.stats()["accepted_frames"] == 1
        second = client.append(
            (real_frame(1),),
            stream_session_id="camera",
            request_id="second",
        )
        assert selected.backend.calls == 2
        eof = client.append(
            (),
            stream_session_id="camera",
            request_id="eof",
            is_last_batch=True,
        )
        assert selected.backend.calls == 3
        assert transport.requests[3] == transport.requests[4]
        assert [json.loads(body)["request_id"] for body in transport.requests] == [
            "first",
            "first",
            "second",
            "eof",
            "eof",
        ]
        assert [
            row.frame_sequence for result in (first, second) for row in result.frames
        ] == [0, 2]
        assert {result.predictor_epoch for result in (first, second, eof)} == {
            session.predictor_epoch,
        }
        assert eof.frames == () and eof.state == "closed"
        assert client.stats()["accepted_frames"] == 2
        assert client.stats()["state"] == "closed"
    finally:
        client.close()


@pytest.mark.parametrize("error_type", TRANSIENT_ERRORS)
def test_retry_exhaustion_is_two_attempts_then_session_fails_closed(error_type):
    server = RouteServer("sam_live")
    bodies = []

    async def exchange(request):
        if request.url.path.endswith("/append"):
            bodies.append(bytes(request.content))
            raise error_type("test transport failure", request=request)
        return await server(request)

    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(exchange),
        timeout_seconds=2,
    )
    try:
        client.open_session("camera")
        with pytest.raises(error_type):
            client.append(frames(1, 1), stream_session_id="camera")
        assert len(bodies) == 2 and bodies[0] == bodies[1]
        assert client.stats()["state"] == "invalid"
        assert client.stats()["accepted_frames"] == 0
        with pytest.raises(RichLabelProtocolError):
            client.append(frames(2, 1), stream_session_id="camera")
        assert len(bodies) == 2  # Later FIFO work cannot bypass failed source.
    finally:
        client.close()


@pytest.mark.parametrize("failure", [409, 429, 503, "schema", "timeout"])
def test_business_schema_and_timeout_errors_are_not_retried(failure):
    server = RouteServer("sam_live")
    calls = []

    async def exchange(request):
        if request.url.path.endswith("/append"):
            calls.append(bytes(request.content))
            if failure == "timeout":
                raise httpx.ReadTimeout("test deadline", request=request)
            if failure == "schema":
                return httpx.Response(200, json={"invalid": True})
            return httpx.Response(failure, json={"detail": "test rejection"})
        return await server(request)

    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(exchange),
        timeout_seconds=2,
    )
    expected = (
        httpx.ReadTimeout
        if failure == "timeout"
        else RichLabelProtocolError
        if failure == "schema"
        else httpx.HTTPStatusError
    )
    try:
        client.open_session("camera")
        with pytest.raises(expected):
            client.append(frames(1, 1), stream_session_id="camera")
        assert len(calls) == 1
        assert client.stats()["state"] == "invalid"
        assert client.stats()["accepted_frames"] == 0
    finally:
        client.close()


def test_legacy_append_preserves_default_no_retry():
    server = SessionServer()
    calls = []

    async def exchange(request):
        if request.url.path.endswith("/append"):
            calls.append(bytes(request.content))
            raise httpx.RemoteProtocolError("test lost response", request=request)
        return await server(request)

    client = VideoSessionTrackingHTTPClient(
        "http://worker.test/v1/video/sessions",
        transport=httpx.MockTransport(exchange),
    )
    try:
        client.open_session("camera")
        with pytest.raises(httpx.RemoteProtocolError):
            client.append(frames(), stream_session_id="camera")
        assert len(calls) == 1
        assert client.stats()["state"] == "invalid"
    finally:
        client.close()


def test_retry_keeps_original_deadline_and_cancels_blocked_second_attempt():
    server = RouteServer("sam_live")
    calls, deadlines = [], []
    cancelled = threading.Event()

    async def exchange(request):
        if request.url.path.endswith("/append"):
            calls.append(bytes(request.content))
            if len(calls) == 1:
                raise httpx.ReadError("test transport failure", request=request)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return await server(request)

    class DeadlineClient(PerceptionSessionHTTPClient):
        def _request(self, method, url, **kwargs):
            if url.endswith("/append"):
                deadlines.append(kwargs["deadline"])
            return super()._request(method, url, **kwargs)

    client = DeadlineClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(exchange),
        timeout_seconds=0.6,
    )
    try:
        client.open_session("camera")
        with pytest.raises(httpx.TimeoutException):
            client.append(frames(1, 1), stream_session_id="camera")
        assert len(calls) == 2 and calls[0] == calls[1]
        assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
        assert cancelled.wait(1)
        assert client.stats()["state"] == "invalid"
    finally:
        client.close()


def test_prepared_packet_expiring_before_backoff_does_not_gain_new_budget():
    server = RouteServer("sam_live")
    calls = []

    async def exchange(request):
        if request.url.path.endswith("/append"):
            calls.append(bytes(request.content))
            raise httpx.RemoteProtocolError("test response lost", request=request)
        return await server(request)

    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(exchange),
        timeout_seconds=2,
    )
    try:
        client.open_session("camera")
        source = frames(1, 1)
        packet = client.prepare_append(source, stream_session_id="camera")
        packet = replace(packet, deadline=time.monotonic() + 0.1)
        with pytest.raises(httpx.RemoteProtocolError):
            client.append(source, stream_session_id="camera", prepared=packet)
        assert calls == [packet.payload.body]
        assert client.stats()["state"] == "invalid"
    finally:
        client.close()


def test_retry_retains_session_lock_so_a_later_frame_cannot_overtake():
    server = RouteServer("sam_live")
    calls, results, failures = [], [], []
    retry_entered, release = threading.Event(), threading.Event()

    async def exchange(request):
        if request.url.path.endswith("/append"):
            calls.append(bytes(request.content))
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("test response lost", request=request)
            if len(calls) == 2:
                retry_entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
        return await server(request)

    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(exchange),
        timeout_seconds=3,
    )

    def first_append():
        try:
            results.append(client.append(frames(1, 1), stream_session_id="camera"))
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=first_append)
    try:
        client.open_session("camera")
        thread.start()
        assert retry_entered.wait(2)
        with pytest.raises(httpx.PoolTimeout):
            client.append(frames(2, 1), stream_session_id="camera")
        assert len(calls) == 2 and calls[0] == calls[1]
        release.set()
        thread.join(2)
        assert not thread.is_alive() and not failures
        assert results[0].frames[0].frame_sequence == 3
        second = client.append(frames(2, 1), stream_session_id="camera")
        assert second.frames[0].frame_sequence == 6
        assert len(calls) == 3
        assert client.stats()["accepted_frames"] == 2
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(3)
        client.close()
