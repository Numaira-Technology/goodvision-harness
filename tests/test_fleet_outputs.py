"""Output contracts, actual streaming/cancellation, media leases and lane isolation."""

from __future__ import annotations

import asyncio
import json
import struct
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Query, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from test_fleet import make_app

from goodvision_harness.event_routes import install_event_routes
from goodvision_harness.fleet_outputs import _SSE, JSON_LIMIT, OUTPUT_PATHS, PREFIX
from goodvision_harness.service import create_app


def test_catalog_covers_all_native_read_outputs_and_documents_queries(monkeypatch):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "3")
    app = create_app()
    native = {
        r.path[3:]
        for r in app.routes
        if "GET" in getattr(r, "methods", ())
        and r.path.startswith(
            (
                "/v1/results/",
                "/v1/events",
                "/v1/scene",
                "/v1/faces/",
                "/v1/rich-labels/",
                "/v1/video/",
            )
        )
    }
    assert native <= set(OUTPUT_PATHS)
    schema = app.openapi()["paths"]
    for suffix in OUTPUT_PATHS:
        assert PREFIX + suffix in schema
    event = schema[PREFIX + "/events"]["get"]
    params = {p["name"]: p for p in event["parameters"]}
    assert params["limit"]["schema"]["maximum"] == 200
    assert params["camera_id"]["required"]
    assert "Last-Event-ID" not in params
    assert "application/json" in event["responses"]["200"]["content"]
    video = schema[PREFIX + "/events/{event_id}/video"]["get"]
    assert {p["name"] for p in video["parameters"]} >= {
        "camera_id",
        "event_id",
        "revision",
        "Range",
        "If-Range",
    }
    assert "206" in video["responses"]
    for suffix, media in [
        ("/results/events", "text/event-stream"),
        ("/video/annotated/stream.mjpeg", "multipart/x-mixed-replace"),
        ("/video/display/frames", "application/vnd.goodvision.display"),
    ]:
        assert media in schema[PREFIX + suffix]["get"]["responses"]["200"]["content"]
    assert app.openapi() == app.openapi()  # No duplicated parameters on repeated reads.


def test_camera_json_identity_links_filters_and_no_browser_credentials(monkeypatch):
    requests = []

    def peer(request):
        requests.append(request)
        lane = request.url.host
        return httpx.Response(
            200,
            json={
                "source": lane,
                "source_id": "same-source",
                "stream_session_id": "session",
                "event_id": "same-event",
                "subject_id": "same-subject",
                "evidence": {
                    "available": {"url": "/v1/events/same-event/video?revision=2"}
                },
                "frames": [{"url": "/v1/scene/evidence/request/17.jpg"}],
                "external_url": "https://example.org/video",
            },
            headers={"X-Result-Cursor": "epoch:7", "Set-Cookie": "private=hidden"},
        )

    app = make_app(monkeypatch, capacity=13, peer_handler=peer)
    with TestClient(app) as client:
        for camera, host in [("camera02", "harness-02"), ("camera13", "harness-13")]:
            base = PREFIX.format(camera_id=camera)
            response = client.get(
                base + "/events?subject_id=same-subject&limit=2",
                headers={
                    "Authorization": "Bearer secret",
                    "Cookie": "secret=1",
                    "X-Fleet-Coordinator": "1",
                },
            )
            body = response.json()
            assert body["camera_id"] == camera
            assert response.headers["X-Camera-Id"] == camera
            assert response.headers["X-Result-Cursor"] == "epoch:7"
            assert "set-cookie" not in response.headers
            assert body["source"] == host
            assert body["source_id"] == "same-source"
            assert body["event_id"] == "same-event"
            assert (
                body["evidence"]["available"]["url"]
                == base + "/events/same-event/video?revision=2"
            )
            assert body["frames"][0]["url"] == base + "/scene/evidence/request/17.jpg"
            assert body["external_url"] == "https://example.org/video"
        assert all(r.url.params["subject_id"] == "same-subject" for r in requests)
        assert all(
            not ({"authorization", "cookie", "x-fleet-coordinator"} & set(r.headers))
            for r in requests
        )
        assert (
            client.post(PREFIX.format(camera_id="camera02") + "/events").status_code
            == 405
        )
        assert (
            client.get(PREFIX.format(camera_id="camera14") + "/events").status_code
            == 404
        )
        assert (
            client.get(
                PREFIX.format(camera_id="camera02") + "/deployment/routes"
            ).status_code
            == 404
        )


def test_local_validation_errors_attributed_and_legacy_unmodified(monkeypatch):
    app = make_app(monkeypatch)

    @app.get("/v1/scene/history")
    def history(limit: int = Query(100, ge=1, le=1000)):
        return {"events": [], "limit": limit}

    with TestClient(app) as client:
        base = PREFIX.format(camera_id="camera01")
        assert "camera_id" not in client.get("/v1/scene/history").json()
        assert client.get(base + "/scene/history?limit=2").json() == {
            "camera_id": "camera01",
            "limit": 2,
            "events": [],
        }
        error = client.get(base + "/scene/history?limit=0")
        assert error.status_code == 422
        assert error.json()["camera_id"] == "camera01"
        assert client.get(base + "/faces/fast").status_code == 404


def test_native_mp4_ranges_pending_and_lease_release(monkeypatch, tmp_path):
    app = make_app(monkeypatch)
    content = bytes(range(256)) * 100
    video = tmp_path / "clip.mp4"
    video.write_bytes(content)
    releases = []
    archive = SimpleNamespace(
        get_event=lambda event_id: {
            "event_id": event_id,
            "evidence": {"status": "pending"},
        },
        acquire_video=lambda event_id, revision=None: (
            (video, lambda: releases.append(event_id)) if event_id == "ready" else None
        ),
    )
    app.state.runtime.event_archive = lambda: archive
    install_event_routes(app)
    with TestClient(app) as client:
        base = PREFIX.format(camera_id="camera01") + "/events/"
        full = client.get(base + "ready/video")
        assert full.content == content
        partial = client.get(
            base + "ready/video?revision=1", headers={"Range": "bytes=12-31"}
        )
        assert partial.status_code == 206
        assert partial.content == content[12:32]
        assert partial.headers["Content-Range"] == f"bytes 12-31/{len(content)}"
        assert partial.headers["X-Camera-Id"] == "camera01"
        assert (
            client.get(
                base + "ready/video", headers={"Range": "bytes=999999-"}
            ).status_code
            == 416
        )
        assert releases == ["ready"] * 3
        pending = client.get(base + "waiting/video")
        assert pending.status_code == 202
        assert pending.headers["Retry-After"] == "1"
        assert pending.json()["camera_id"] == "camera01"
        assert client.get(base + "waiting/video?revision=0").status_code == 422


def test_peer_range_and_jpeg_metadata_and_gvd_packet(monkeypatch):
    requests = []
    header = json.dumps(
        {"connection_id": "connection", "frames": [{"bytes": 4}]}
    ).encode()
    packet = b"GVD1" + struct.pack(">I", len(header)) + header + b"JPEG"

    def peer(request):
        requests.append(request)
        if request.url.path.endswith("/video"):
            return httpx.Response(
                206,
                content=b"1234",
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Range": "bytes 1-4/10",
                    "Accept-Ranges": "bytes",
                    "ETag": '"revision1"',
                },
            )
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                content=packet,
                headers={"Content-Type": "application/vnd.goodvision.display"},
            )
        return httpx.Response(
            200,
            content=b"JPEG",
            headers={
                "Content-Type": "image/jpeg",
                "X-Frame-Sequence": "42",
                "X-Media-Timestamp": "1.5",
            },
        )

    app = make_app(monkeypatch, peer_handler=peer)
    with TestClient(app) as client:
        base = PREFIX.format(camera_id="camera02")
        result = client.get(
            base + "/events/e/video?revision=1",
            headers={"Range": "bytes=1-4", "If-Range": '"revision1"'},
        )
        assert result.content == b"1234" and result.status_code == 206
        assert result.headers["Content-Range"] == "bytes 1-4/10"
        assert requests[-1].headers["Range"] == "bytes=1-4"
        assert requests[-1].headers["If-Range"] == '"revision1"'
        image = client.get(base + "/scene/evidence/r/42.jpg")
        assert image.content == b"JPEG"
        assert image.headers["X-Frame-Sequence"] == "42"
        result = client.get(base + "/video/display/frames?after=17&revision=3")
        assert result.headers["Content-Length"] == str(len(result.content))
        length = struct.unpack(">I", result.content[4:8])[0]
        assert json.loads(result.content[8 : 8 + length])["camera_id"] == "camera02"
        assert result.content[8 + length :] == b"JPEG"
        assert requests[-1].url.params["after"] == "17"


@pytest.mark.parametrize("camera", ["camera01", "camera02"])
@pytest.mark.parametrize("kind", ["sse", "mjpeg"])
def test_first_chunk_is_live_disconnect_closes_and_control_still_works(
    monkeypatch, camera, kind
):
    async def exercise():
        closed = asyncio.Event()
        disconnected = asyncio.Event()
        delivered = asyncio.Event()
        requests = []
        if kind == "sse":
            payload = (
                "id: epoch:9\r\nevent: event_change\r\ndata: "
                + json.dumps(
                    {
                        "event_id": "e",
                        "description": "第二路",
                        "url": "/v1/events/e/video?revision=1",
                    },
                    ensure_ascii=False,
                )
                + "\r\n\r\n"
            ).encode()
            path, media = "/events/stream", "text/event-stream"
        else:
            payload = (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                b"X-Frame-Sequence: 9\r\n\r\nJPEG\r\n"
            )
            path, media = (
                "/video/annotated/stream.mjpeg",
                "multipart/x-mixed-replace; boundary=frame",
            )

        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                # Split UTF-8 and event delimiters across arbitrary transport chunks.
                for byte in payload:
                    yield bytes([byte])
                await asyncio.Event().wait()

            async def aclose(self):
                closed.set()

        def peer(request):
            requests.append(request)
            return httpx.Response(
                200, stream=SlowStream(), headers={"Content-Type": media}
            )

        app = make_app(monkeypatch, peer_handler=peer)
        # The control client has its own finite response transport.
        await app.state.fleet._peer_client.aclose()
        app.state.fleet._peer_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"ok": True})
            )
        )

        @app.get("/v1" + path)
        async def local(request: Request):
            requests.append(request)

            async def stream():
                try:
                    yield payload
                    await asyncio.Event().wait()
                finally:
                    closed.set()

            return StreamingResponse(stream(), media_type=media)

        messages = []

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if b"JPEG" in message.get("body", b"") or b"data:" in message.get(
                "body", b""
            ):
                delivered.set()
            # MJPEG may forward a byte per message: never wait for a complete frame.
            if kind == "mjpeg" and message.get("body"):
                delivered.set()

        url = PREFIX.format(camera_id=camera) + path
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": url,
            "raw_path": url.encode(),
            "query_string": b"after=epoch%3A8",
            "root_path": "",
            "headers": [(b"last-event-id", b"epoch:8")],
            "server": ("test", 80),
            "client": ("test", 123),
        }
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(delivered.wait(), timeout=2)
            assert not task.done(), (
                "an infinite response must deliver before completion"
            )
            control = await app.state.fleet.request_json(2, "GET", "/v1/status")
            assert control.status_code == 200
            assert requests[0].headers["last-event-id"] == "epoch:8"
            headers = dict(messages[0]["headers"])
            assert headers[b"x-camera-id"] == camera.encode()
            if kind == "sse":
                output = b"".join(m.get("body", b"") for m in messages)
                assert b"id: epoch:9" in output
                assert f'"camera_id":"{camera}"'.encode() in output
                assert (
                    PREFIX.format(camera_id=camera) + "/events/e/video?revision=1"
                ).encode() in output
        finally:
            disconnected.set()
            await asyncio.wait_for(task, timeout=2)
            await app.state.fleet.aclose()
        assert closed.is_set(), "disconnect must release the source stream"

    asyncio.run(exercise())


def test_sse_heartbeat_reset_utf8_and_size_limit():
    parser = _SSE("camera02")
    assert list(parser.feed(b": keep-alive\n\n")) == [b": keep-alive\n\n"]
    data = b'id: epoch:0\nevent: reset\ndata: {"reason":"history_lost"}\n\n'
    out = next(iter(parser.feed(data)))
    assert b"event: reset" in out and b'"camera_id":"camera02"' in out
    with pytest.raises(ValueError):
        list(parser.feed(b"data: " + b"x" * JSON_LIMIT))


def test_real_analysis_result_matches_schema_and_retired_links_stay_in_lane(
    monkeypatch, tmp_path
):
    from pathlib import Path

    import jsonschema
    from test_event_service import make_runtime, observation

    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "1")
    runtime = make_runtime(tmp_path, monkeypatch)
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        runtime._on_analysis_result(observation(1, "message"))
        base = PREFIX.format(camera_id="camera01")
        original = client.get("/v1/results/latest")
        scoped = client.get(base + "/results/latest")
        assert scoped.status_code == original.status_code == 200
        assert scoped.json() == dict(original.json(), camera_id="camera01")
        schema = json.loads(
            Path("schemas/analysis-result-v2.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(scoped.json(), schema)
        for path in ("/faces/strangers", "/faces/strangers/unused/photo"):
            response = client.get(base + path)
            assert response.status_code == 410
            assert response.json()["detail"]["events_url"] == base + "/events"
