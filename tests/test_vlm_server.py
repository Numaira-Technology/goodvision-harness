"""CPU companion integration with a mocked, authenticated model endpoint."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from goodvision_harness import scene_inspection, vlm_server
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_independent_model import IndependentVLMInspector
from goodvision_harness.vlm_offload import decode_inspection, encode_inspection
from goodvision_harness.vlm_server import VLMServerSettings, create_app


def _job():
    with Image.new("RGB", (96, 64), "red") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        jpeg = buffer.getvalue()
    box = NormalizedBBox(0.1, 0.1, 0.6, 0.9)
    return TemporalInspection(
        "session",
        "rtsp://private-camera?key=SECRET",
        InspectionRequest(
            "request-01", ("person-1",), "new_person", ("smoking",), 0, 1
        ),
        (EvidenceFrame("session", 0, 0, jpeg), EvidenceFrame("session", 1, 1, jpeg)),
        ((0, "person-1", box), (1, "person-1", box)),
    )


def _wire(kind="full", deadline=1):
    return encode_inspection(
        _job(),
        kind,
        "local-vision",
        {"temperature": 0},
        {"max_tokens": 1600, "max_image_side": 32, "jpeg_quality": 40},
        deadline,
    )


def _reply(kind="full"):
    if kind == "events":
        content = {"smoking": False}
    elif kind == "compact":
        content = {"no_events": 0.9}
    elif kind == "temporal":
        content = [
            {
                "event_label": "smoking",
                "subject_ids": ["person-1"],
                "detected": False,
                "confidence": 0.9,
            }
        ]
    else:
        content = {
            "subjects": [],
            "verdicts": [
                {
                    "event_label": "smoking",
                    "subject_ids": [],
                    "detected": False,
                    "confidence": 0.9,
                }
            ],
        }
    return {
        "choices": [
            {"message": {"content": json.dumps(content)}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _app(handler, **options):
    seen = []

    def record(request):
        seen.append(request)
        return handler(request)

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(record), trust_env=False)

    settings = VLMServerSettings(
        upstream_endpoint="http://model:8000/v1/chat/completions",
        timeout_seconds=1,
        **options,
    )
    return create_app(settings, client_factory=factory), seen


def test_wire_roundtrip_retains_originals_without_decoding(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wire codec must not decode pixels")

    monkeypatch.setattr(scene_inspection, "_read_image", forbidden)
    job = _job()
    wire = _wire()
    assert "private-camera" not in json.dumps(wire)
    decoded = decode_inspection(wire)
    assert decoded.job.frames == job.frames
    assert decoded.job.subject_boxes == job.subject_boxes
    assert decoded.job.source_id == ""
    assert decoded.kind == "full"


@pytest.mark.parametrize("kind", ["temporal", "full", "compact", "events"])
def test_cpu_prompt_and_parser_reuse_for_every_lane(monkeypatch, kind):
    cls = {
        "temporal": TemporalVLMInspector,
        "full": IndependentVLMInspector,
        "compact": CompactGroundingInspector,
        "events": EventJSONInspector,
    }[kind]
    reference = cls(
        ModelAPISettings(provider="local"),
        model="local-vision",
        request_parameters={"temperature": 0},
        max_image_side=32,
        jpeg_quality=40,
    )
    expected = reference.build_messages(_job())
    reference.close()
    threads = []
    original = scene_inspection._read_image

    def read_image(jpeg):
        threads.append(threading.current_thread().name)
        return original(jpeg)

    monkeypatch.setattr(scene_inspection, "_read_image", read_image)
    app, seen = _app(lambda _: httpx.Response(200, json=_reply(kind)))
    with TestClient(app) as client:
        response = client.post("/v1/inspections", json=_wire(kind))
    assert response.status_code == 200
    envelope = response.json()
    assert envelope["version"] == 1
    assert envelope["request_id"] == "request-01"
    assert envelope["completion"] == _reply(kind)
    assert envelope["usage"]["total_tokens"] == 15
    assert envelope["timing"]["request_prepare_seconds"] >= 0
    assert threads and all(name.startswith("vlm-companion") for name in threads)
    sent = json.loads(seen[0].content)
    assert sent["messages"] == expected
    assert sent["stream"] is False
    images = [p for p in sent["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 2
    encoded = images[0]["image_url"]["url"].split(",")[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (32, 21)


def test_local_auth_and_cloud_credential_isolation(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "CLOUD_SECRET")
    monkeypatch.setenv("OPENROUTER_API_KEY", "CLOUD_SECRET")
    app, seen = _app(
        lambda _: httpx.Response(200, json=_reply()), api_key="LOCAL_SECRET"
    )
    with TestClient(app) as client:
        assert client.post("/v1/inspections", json=_wire()).status_code == 401
        assert (
            client.post(
                "/v1/inspections",
                json=_wire(),
                headers={"Authorization": "Bearer CLOUD_SECRET"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/inspections",
                json=_wire(),
                headers={"Authorization": "Bearer LOCAL_SECRET"},
            ).status_code
            == 200
        )
    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer LOCAL_SECRET"
    assert "CLOUD_SECRET" not in seen[0].content.decode()


@pytest.mark.parametrize(
    "corruption", ["base64", "jpeg", "sequence", "version", "upstream", "model"]
)
def test_invalid_jobs_never_contact_upstream(corruption):
    wire = copy.deepcopy(_wire())
    if corruption == "base64":
        wire["job"]["frames"][0]["jpeg_base64"] = "!!not-base64"
    elif corruption == "jpeg":
        wire["job"]["frames"][0]["jpeg_base64"] = base64.b64encode(b"not-jpeg").decode()
    elif corruption == "sequence":
        wire["job"]["frames"][1]["frame_sequence"] = 0
    elif corruption == "version":
        wire["version"] = True
    elif corruption == "upstream":
        wire["endpoint"] = "http://untrusted-host"
    else:
        wire["model"] = "another-model"
    app, seen = _app(lambda _: httpx.Response(200, json=_reply()))
    with TestClient(app) as client:
        assert client.post("/v1/inspections", json=wire).status_code == 400
    assert not seen


def test_size_limit_rejects_header_and_chunked_body_before_json(monkeypatch):
    monkeypatch.setattr(vlm_server, "MAX_REQUEST_BYTES", 64)
    app, seen = _app(lambda _: httpx.Response(200, json=_reply()))
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/inspections", content=b"x", headers={"Content-Length": "65"}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/inspections", content=iter([b"x" * 40, b"x" * 40])
            ).status_code
            == 413
        )
        assert client.post("/v1/inspections", content=b"not-json").status_code == 400
    assert not seen


def test_invalid_model_reply_is_sanitized_and_next_request_recovers():
    replies = iter(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "PRIVATE_SECRET bad response"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
            httpx.Response(200, json=_reply()),
        ]
    )
    app, _ = _app(lambda _: next(replies))
    with TestClient(app) as client:
        bad = client.post("/v1/inspections", json=_wire())
        assert bad.status_code == 502
        assert "PRIVATE_SECRET" not in bad.text
        assert client.post("/v1/inspections", json=_wire()).status_code == 200


@pytest.mark.parametrize("alias,expected", [("local-vision", 200), ("other", 503)])
def test_health_requires_matching_upstream_model(alias, expected):
    app, seen = _app(lambda _: httpx.Response(200, json={"data": [{"id": alias}]}))
    with TestClient(app) as client:
        result = client.get("/health")
    assert result.status_code == expected
    assert result.json()["ready"] is (expected == 200)
    assert str(seen[0].url) == "http://model:8000/v1/models"


def test_legacy_streaming_proxy_uses_fixed_upstream_and_selected_auth():
    data = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    app, seen = _app(lambda _: httpx.Response(200, content=data), api_key="LOCAL")
    with TestClient(app) as client:
        result = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-vision",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer LOCAL"},
        )
        assert result.content == data
        assert result.headers["content-type"].startswith("text/event-stream")
        assert app.state.admission.active == 0
    assert str(seen[0].url) == "http://model:8000/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer LOCAL"


def test_legacy_proxy_rejects_corrupt_upstream_json_without_leaking_it():
    app, _ = _app(lambda _: httpx.Response(200, text="PRIVATE corrupt JSON"))
    with TestClient(app) as client:
        result = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-vision",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert result.status_code == 502
        assert result.json()["code"] == "invalid_response"
        assert "PRIVATE" not in result.text
        assert app.state.admission.active == 0


@pytest.mark.parametrize("path", ["/v1/inspections", "/v1/chat/completions"])
def test_alternate_token_budget_is_bounded_before_contacting_upstream(path):
    payload = (
        _wire()
        if path.endswith("inspections")
        else {
            "model": "local-vision",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    parameters = payload["parameters"] if path.endswith("inspections") else payload
    parameters["max_completion_tokens"] = 100000
    app, seen = _app(lambda _: httpx.Response(200, json=_reply()))
    with TestClient(app) as client:
        assert client.post(path, json=payload).status_code == 400
    assert not seen


def test_proxy_without_token_budget_sets_bounded_default():
    app, seen = _app(lambda _: httpx.Response(200, json=_reply()))
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-vision",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ).status_code
            == 200
        )
    assert json.loads(seen[0].content)["max_tokens"] == 1600


def test_shorter_declared_deadline_keeps_slot_until_cpu_finishes(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = scene_inspection._read_image

    def blocked(jpeg):
        entered.set()
        release.wait(1)
        return original(jpeg)

    monkeypatch.setattr(scene_inspection, "_read_image", blocked)
    app, seen = _app(lambda _: httpx.Response(200, json=_reply()), max_concurrency=1)
    with TestClient(app) as client:
        try:
            started = time.monotonic()
            result = client.post("/v1/inspections", json=_wire(deadline=0.03))
            assert entered.is_set()
            assert result.status_code == 504
            assert time.monotonic() - started < 0.3
            assert client.post("/v1/inspections", json=_wire()).status_code == 429
        finally:
            release.set()
        for _ in range(100):
            if app.state.admission.active == 0:
                break
            time.sleep(0.005)
        assert app.state.admission.active == 0
        assert not seen
        assert client.post("/v1/inspections", json=_wire()).status_code == 200


@pytest.mark.parametrize("path", ["/v1/inspections", "/v1/chat/completions"])
def test_controller_disconnect_cancels_upstream_and_releases_its_slot(path):
    entered, cancelled = threading.Event(), threading.Event()

    async def slow(_request):
        entered.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(slow))

    app = create_app(
        VLMServerSettings(timeout_seconds=10, max_concurrency=1), client_factory=factory
    )
    payload = (
        _wire(deadline=10)
        if path == "/v1/inspections"
        else {
            "model": "local-vision",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    body = json.dumps(payload).encode()
    sent = []

    async def run():
        uploaded = False

        async def receive():
            nonlocal uploaded
            if not uploaded:
                uploaded = True
                return {"type": "http.request", "body": body, "more_body": False}
            assert await asyncio.to_thread(entered.wait, 2)
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("localhost", 8080),
        }
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), 2)
            for _ in range(100):
                if app.state.admission.active == 0:
                    break
                await asyncio.sleep(0.005)
            assert app.state.admission.active == 0
            assert cancelled.is_set()

    asyncio.run(run())
    assert sent[0]["status"] == 499


def test_stream_disconnect_finishes_yielding_upstream_close():
    closed = {"started": False, "done": False}

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: {}\n\n"
            await asyncio.sleep(10)

        async def aclose(self):
            closed["started"] = True
            await asyncio.sleep(0.01)
            closed["done"] = True

    def factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    stream=SlowStream(),
                    headers={"Content-Type": "text/event-stream"},
                ),
            )
        )

    app = create_app(
        VLMServerSettings(timeout_seconds=10, max_concurrency=1),
        client_factory=factory,
    )
    body = json.dumps(
        {
            "model": "local-vision",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()

    async def run():
        uploaded = False
        first_body = asyncio.Event()

        async def receive():
            nonlocal uploaded
            if not uploaded:
                uploaded = True
                return {"type": "http.request", "body": body, "more_body": False}
            await first_body.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("localhost", 8080),
        }
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), 2)
            assert closed == {"started": True, "done": True}
            assert app.state.admission.active == 0

    asyncio.run(run())
