"""Offline timing/correlation checks: mocked models and in-process ASGI only."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_sam31_continuous_worker import engine, payload

from workers.sam_scene import cloud_pool, continuous
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.multistream import FairAdmission, FairGPUAdmission


def _fields(caplog, message):
    return [
        record.sam_timing
        for record in caplog.records
        if record.getMessage().startswith(message)
    ]


def test_worker_correlates_camera_session_request_attempt_and_response_send(
    caplog, monkeypatch
):
    clock = [0.0]

    def charge(seconds):
        clock[0] += seconds

    monkeypatch.setattr(
        continuous,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0], monotonic=time.monotonic),
    )
    selected = engine(hotstart_delay=3)
    run = selected.backend.run

    def measured_run(*args, **kwargs):
        charge(7.0)
        yield from run(*args, **kwargs)

    selected.backend.run = measured_run
    app = FastAPI()
    prefix = "/v1/cameras/camera-07/perception/sam_live"
    continuous.register_session_routes(
        app,
        selected,
        FairGPUAdmission(4),
        decode_jpeg,
        api_prefix=prefix,
        legacy=False,
        queued_admission=True,
        preparation_admission=FairAdmission(4, capacity=2),
    )

    class SendDelay:
        def __init__(self, application):
            self.application = application

        async def __call__(self, scope, receive, send):
            async def measured_send(message):
                if message["type"].startswith("http.response"):
                    charge(3.0)
                await send(message)

            await self.application(scope, receive, measured_send)

    with caplog.at_level("INFO", logger="goodvision.sam31.session"):
        with TestClient(SendDelay(app)) as client:
            token = client.post(
                prefix + "/sessions", json={"stream_session_id": "camera"}
            ).json()["video_session_id"]
            response = client.post(
                f"{prefix}/sessions/{token}/append",
                content=payload(16, request_id="payload-request"),
                headers={
                    "X-Goodvision-Request-ID": "transport-request",
                    "X-Goodvision-HTTP-Attempt-ID": "attempt-1",
                },
            )
    assert response.status_code == 200
    native = _fields(caplog, "Native session append timing")[0]
    http = _fields(caplog, "Native session request completed")[-1]
    sent = _fields(caplog, "Native session response timing")[-1]
    for fields in (native, http, sent):
        assert fields["camera_id"] == "camera-07"
        assert fields["request_id"] == "payload-request"
        assert fields["transport_request_id"] == "transport-request"
        assert fields["http_attempt_id"] == "attempt-1"
        assert fields["session_ref"] == continuous._session_ref(token)
        assert fields["timing_basis"] == "host_wall_clock"
        assert fields["recorded_at"]
    assert native["inference_wall_seconds"] == native["native_advance_seconds"] == 7
    assert native["geometry_wall_seconds"] == native["snapshot_seconds"]
    assert http["gpu_active_at_enqueue"] == 0
    assert http["gpu_waiters_at_enqueue"] == 0
    assert sent["asgi_send_seconds"] == 6
    assert sent["request_through_send_seconds"] >= http["total_seconds"] + 6
    assert token not in caplog.text


def test_diagnostics_reject_free_text_and_use_configured_stderr_logger(
    caplog, monkeypatch
):
    for value in ("https://private.example/token", "private\nidentity", "x" * 161):
        assert continuous._trace_identifier(value) is None
    assert continuous._trace_identifier("request-7:attempt.1") == "request-7:attempt.1"
    monkeypatch.setattr(continuous.LOGGER, "level", logging.WARNING)
    with caplog.at_level("INFO", logger="uvicorn.error"):
        continuous._timing_log("Native session diagnostic", {"request_id": "safe"})
    records = [
        r
        for r in caplog.records
        if r.getMessage().startswith("Native session diagnostic")
    ]
    assert len(records) == 1 and records[0].name == "uvicorn.error"


def test_pool_measures_forward_headers_stream_and_asgi_without_buffering(
    caplog, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        cloud_pool, "time", SimpleNamespace(perf_counter=lambda: clock[0])
    )
    observed = []

    class ResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            clock[0] += 3
            yield b"first"
            clock[0] += 4
            yield b"second"

    async def handler(request):
        observed.append(
            (request.url.path, dict(request.headers), await request.aread())
        )
        clock[0] += 11
        return httpx.Response(200, stream=ResponseStream())

    async def run():
        app = cloud_pool.create_app(
            2, 4, launch_workers=False, transport=httpx.MockTransport(handler)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://pool"
            ) as client:
                response = await client.post(
                    "/v1/cameras/camera-04/perception/sam_live/sessions/private-token/append",
                    content=b"unchanged-input-bytes",
                    headers={
                        "X-Goodvision-Request-ID": "request-1",
                        "X-Goodvision-HTTP-Attempt-ID": "attempt-1",
                        "Authorization": "Bearer do-not-log-me",
                    },
                )
                assert response.content == b"firstsecond"

    with caplog.at_level("INFO", logger="goodvision.sam31.pool"):
        asyncio.run(run())
    fields = _fields(caplog, "SAM cloud proxy timing")[0]
    assert fields["camera_id"] == "camera-04"
    assert fields["worker_camera_id"] == "camera-02"
    assert fields["gpu_index"] == 1
    assert fields["transport_request_id"] == "request-1"
    assert fields["http_attempt_id"] == "attempt-1"
    assert fields["session_ref"] == continuous._session_ref("private-token")
    assert fields["upstream_headers_seconds"] == 11
    assert fields["response_stream_seconds"] == fields["asgi_send_seconds"] == 7
    assert fields["total_seconds"] == 18
    assert fields["request_bytes"] == len(b"unchanged-input-bytes")
    assert fields["response_bytes"] == 11
    assert observed[0][0].endswith(
        "/camera-02/perception/sam_live/sessions/private-token/append"
    )
    assert observed[0][1]["x-goodvision-http-attempt-id"] == "attempt-1"
    assert observed[0][2] == b"unchanged-input-bytes"
    assert "private-token" not in caplog.text
    assert "do-not-log-me" not in caplog.text
    assert "unchanged-input-bytes" not in caplog.text


def test_pool_logs_failed_attempt_without_exception_text_or_body(caplog):
    async def handler(_request):
        raise httpx.ReadError("private endpoint and token must stay out of logs")

    with caplog.at_level("INFO", logger="goodvision.sam31.pool"):
        with TestClient(
            cloud_pool.create_app(
                1, 1, launch_workers=False, transport=httpx.MockTransport(handler)
            )
        ) as client:
            response = client.post(
                "/v1/cameras/camera-01/perception/sam_live/sessions/private-token/append",
                content=b"private-image",
                headers={"X-Goodvision-HTTP-Attempt-ID": "invalid/attempt"},
            )
    assert response.status_code == 503
    fields = _fields(caplog, "SAM cloud proxy timing")[0]
    assert fields["error_type"] == "ReadError"
    assert fields["outcome"] == "upstream_failed"
    assert fields["http_attempt_id"] is None
    assert "private" not in caplog.text
