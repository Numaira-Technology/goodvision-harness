"""Fleet gateway isolation, admission policy and bounded network IO."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient

from goodvision_harness.fleet import (
    CONTROL_MAX_BYTES,
    PREVIEW_MAX_BYTES,
    RESPONSE_MAX_BYTES,
    install_fleet_routes,
)


def make_app(
    monkeypatch,
    *,
    capacity=3,
    index=1,
    tracking=True,
    inference=False,
    available=True,
    peer_handler=None,
):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", str(capacity))
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", str(index))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await app.state.fleet.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.runtime = SimpleNamespace(
        _local_tracking_enabled=tracking,
        _local_inference_enabled=inference,
        _local_gpu=SimpleNamespace(available=available),
        _aiclub_api_key=None,
        route_probe_status=lambda: {"state": "idle", "supported": False},
    )
    fleet = install_fleet_routes(app)
    app.state.peer_requests = []

    def default_peer(request):
        app.state.peer_requests.append(request)
        if request.url.path == "/v1/status":
            return httpx.Response(200, json={"pipeline_state": "idle"})
        if request.url.path.endswith("latest.jpg"):
            return httpx.Response(
                200,
                content=b"jpeg",
                headers={
                    "content-type": "image/jpeg",
                },
            )
        return httpx.Response(200, json={"ok": True})

    fleet._peer_client = httpx.AsyncClient(
        transport=httpx.MockTransport(peer_handler or default_peer),
        trust_env=False,
        follow_redirects=False,
    )

    @app.get("/v1/status")
    async def own_status():
        return {"pipeline_state": "idle", "source": "own"}

    @app.post("/v1/sessions/start")
    async def own_start(request: Request):
        return {"own": True, "payload": await request.json()}

    @app.post("/v1/sessions/stop")
    async def own_stop():
        return {"stopped": True}

    @app.put("/v1/deployment/settings")
    async def own_settings(request: Request):
        async with fleet.control_lock:
            return {"saved": await request.json()}

    @app.get("/v1/video/annotated/latest.jpg")
    async def own_preview():
        return Response(b"own-jpeg", media_type="image/jpeg")

    return app


@pytest.mark.parametrize("capacity", [1, 2, 8, 12, 16, 24, 100])
def test_capacity_is_provisioned_without_a_twelve_camera_limit(monkeypatch, capacity):
    app = make_app(monkeypatch, capacity=capacity)
    with TestClient(app) as client:
        response = client.get("/v1/fleet")
        fleet = response.json()
        assert fleet["capacity"] == capacity
        assert len(fleet["cameras"]) == capacity
        assert fleet["cameras"][-1]["id"] == f"camera{capacity:02d}"
        assert all(camera["enabled"] for camera in fleet["cameras"])
        for camera in fleet["cameras"]:
            assert camera["preview_url"].endswith("/video/annotated/latest.jpg")
            assert all(
                value.startswith("/")
                for key, value in camera.items()
                if key.endswith("_url") and value is not None
            )
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "tracking,inference,available,allowed",
    [
        (True, True, True, True),
        (True, False, True, True),
        (False, True, True, False),
        (False, False, True, False),
        (True, True, False, False),
        (False, False, False, False),
    ],
)
def test_multicamera_requires_only_local_tracking_and_nvidia_capability(
    monkeypatch,
    tracking,
    inference,
    available,
    allowed,
):
    app = make_app(
        monkeypatch,
        tracking=tracking,
        inference=inference,
        available=available,
    )
    with TestClient(app) as client:
        fleet = client.get("/v1/fleet").json()
        assert fleet["multi_camera_allowed"] is allowed
        assert fleet["cameras"][0]["enabled"]
        assert fleet["cameras"][1]["enabled"] is allowed
        response = client.post("/v1/fleet/cameras/camera02/start", json={})
        assert response.status_code == (200 if allowed else 409)
        assert (
            client.post("/v1/fleet/cameras/camera01/start", json={}).status_code == 200
        )


def test_own_camera_reuses_original_handlers_without_network_or_settings_deadlock(
    monkeypatch,
):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        prefix = "/v1/fleet/cameras/camera01"
        assert client.get(prefix + "/status").json()["source"] == "own"
        assert client.post(prefix + "/start", json={"stream_url": "saved"}).json() == {
            "own": True,
            "payload": {"stream_url": "saved"},
        }
        assert client.put(prefix + "/settings", json={"dry_run": True}).json() == {
            "saved": {"dry_run": True},
        }
        assert client.get(prefix + "/video/annotated/latest.jpg").content == b"own-jpeg"
        assert not app.state.peer_requests


def test_peer_start_syncs_global_gpu_modes_credentials_and_cloud_route(monkeypatch):
    app = make_app(monkeypatch)
    app.state.runtime._aiclub_api_key = "private-api-key"
    app.state.runtime.route_probe_status = lambda: {
        "state": "idle",
        "supported": True,
        "selected_id": "route-A",
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/fleet/cameras/camera02/start?target=http://other-host",
            json={"stream_url": "rtsp://user:password@camera/live"},
            headers={"Authorization": "Bearer browser-secret", "Cookie": "private=1"},
        )
        assert response.status_code == 200
        requests = app.state.peer_requests
        assert [request.url.path for request in requests] == [
            "/v1/status",
            "/v1/deployment/settings",
            "/v1/deployment/routes/select",
            "/v1/sessions/start",
        ]
        assert all(request.url.host == "harness-02" for request in requests)
        assert all(not request.url.query for request in requests)
        assert all("authorization" not in request.headers for request in requests)
        assert all("cookie" not in request.headers for request in requests)
        synchronized = json.loads(requests[1].content)
        assert synchronized == {
            "local_tracking_enabled": True,
            "local_inference_enabled": False,
            "dry_run": False,
            "autostart": False,
            "api_key": "private-api-key",
        }
        assert "stream_url" not in synchronized
        assert "private-api-key" not in response.text


def test_peer_start_copies_primary_saved_test_mode(monkeypatch):
    app = make_app(monkeypatch)
    app.state.runtime.dry_run = True
    with TestClient(app) as client:
        response = client.post("/v1/fleet/cameras/camera02/start", json={})
        assert response.status_code == 200
        settings = json.loads(app.state.peer_requests[1].content)
        assert settings["dry_run"] is True
        assert "dry_run" not in json.loads(app.state.peer_requests[-1].content)


def test_absent_primary_key_clears_old_peer_key(monkeypatch):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        assert (
            client.post("/v1/fleet/cameras/camera02/start", json={}).status_code == 200
        )
        settings = json.loads(app.state.peer_requests[1].content)
        assert settings["clear_api_key"] is True
        assert "api_key" not in settings


@pytest.mark.parametrize(
    "payload",
    [
        {"local_tracking_enabled": False},
        {"local_inference_enabled": True},
        {"dry_run": True},
        {"autostart": True},
    ],
)
def test_peers_cannot_override_global_gpu_modes_or_autostart(monkeypatch, payload):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        response = client.put("/v1/fleet/cameras/camera02/settings", json=payload)
        assert response.status_code == 400
        assert not app.state.peer_requests


def test_camera_control_bodies_are_bounded_and_validated_before_forwarding(monkeypatch):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        path = "/v1/fleet/cameras/camera02/start"
        assert (
            client.post(path, content=b"x" * (CONTROL_MAX_BYTES + 1)).status_code == 413
        )
        assert client.post(path, content=b"not-json").status_code == 400
        assert client.post(path, json=[]).status_code == 400
        assert not app.state.peer_requests


@pytest.mark.parametrize(
    "camera_id", ["camera00", "camera04", "camera002", "2", "camera-2"]
)
def test_only_canonical_provisioned_camera_ids_can_be_proxied(monkeypatch, camera_id):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        assert client.get(f"/v1/fleet/cameras/{camera_id}/status").status_code == 404
        assert not app.state.peer_requests


def test_stop_and_status_remain_available_when_multicamera_becomes_disabled(
    monkeypatch,
):
    app = make_app(monkeypatch, tracking=False)
    with TestClient(app) as client:
        path = "/v1/fleet/cameras/camera02"
        assert client.get(path + "/status").status_code == 200
        assert client.post(path + "/stop").status_code == 200
        assert client.get(path + "/video/annotated/latest.jpg").status_code == 409


@pytest.mark.parametrize(
    "state", ["starting", "running", "draining", "stopping", "unknown"]
)
def test_disabling_local_tracking_refuses_any_busy_or_unknown_peer(monkeypatch, state):
    app = make_app(
        monkeypatch,
        peer_handler=lambda _: httpx.Response(
            200,
            json={"pipeline_state": state},
        ),
    )
    with TestClient(app) as client:
        with pytest.raises(HTTPException) as error:
            client.portal.call(app.state.fleet.assert_can_disable_multicamera)
        assert error.value.status_code == 409


def test_disabling_local_tracking_requires_every_peer_reachable_and_idle(monkeypatch):
    def handler(request):
        if request.url.host == "harness-03":
            raise httpx.ConnectError("private network info", request=request)
        return httpx.Response(200, json={"pipeline_state": "idle"})

    app = make_app(monkeypatch, peer_handler=handler)
    with TestClient(app) as client:
        with pytest.raises(HTTPException) as error:
            client.portal.call(app.state.fleet.assert_can_disable_multicamera)
        assert error.value.status_code == 409
        assert "private network info" not in error.value.detail


def test_all_idle_peers_allow_mode_change(monkeypatch):
    app = make_app(monkeypatch, capacity=16)
    with TestClient(app) as client:
        client.portal.call(app.state.fleet.assert_can_disable_multicamera)
        assert len(app.state.peer_requests) == 15


def test_primary_route_probe_blocks_new_peer_start(monkeypatch):
    app = make_app(monkeypatch)
    app.state.runtime.route_probe_status = lambda: {"state": "running"}
    with TestClient(app) as client:
        assert (
            client.post("/v1/fleet/cameras/camera02/start", json={}).status_code == 409
        )
        assert not app.state.peer_requests


@pytest.mark.parametrize(
    "response,code",
    [
        (httpx.Response(302, headers={"location": "http://untrusted-host"}), 502),
        (httpx.Response(200, content=b"x" * (RESPONSE_MAX_BYTES + 1)), 502),
        (httpx.Response(200, content=b"not-json"), 502),
    ],
)
def test_invalid_peer_responses_are_bounded_and_do_not_redirect(
    monkeypatch, response, code
):
    app = make_app(monkeypatch, peer_handler=lambda _: response)
    with TestClient(app) as client:
        assert client.get("/v1/fleet/cameras/camera02/status").status_code == code


def test_preview_is_a_finite_uncached_jpeg_with_no_forwarded_browser_headers(
    monkeypatch,
):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/v1/fleet/cameras/camera02/video/annotated/latest.jpg")
        assert response.content == b"jpeg"
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "response,code",
    [
        (httpx.Response(404), 404),
        (httpx.Response(302, headers={"location": "http://other"}), 502),
        (
            httpx.Response(
                200, content=b"text", headers={"content-type": "text/plain"}
            ),
            502,
        ),
        (
            httpx.Response(
                200,
                content=b"x" * (PREVIEW_MAX_BYTES + 1),
                headers={"content-type": "image/jpeg"},
            ),
            502,
        ),
    ],
)
def test_preview_rejects_missing_invalid_and_oversized_frames(
    monkeypatch, response, code
):
    app = make_app(monkeypatch, peer_handler=lambda _: response)
    with TestClient(app) as client:
        result = client.get("/v1/fleet/cameras/camera02/video/annotated/latest.jpg")
        assert result.status_code == code


def test_secondary_cannot_act_as_an_alternate_fleet_controller(monkeypatch):
    app = make_app(monkeypatch, index=2)
    with TestClient(app) as client:
        assert client.get("/v1/fleet").status_code == 409
        assert (
            client.post("/v1/fleet/cameras/camera03/start", json={}).status_code == 409
        )
        assert not app.state.peer_requests


def test_start_and_global_mode_change_are_serialized(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()
        peer_running = False

        async def handler(request):
            nonlocal peer_running
            if request.url.path == "/v1/status":
                return httpx.Response(
                    200,
                    json={
                        "pipeline_state": "running" if peer_running else "idle",
                    },
                )
            if request.url.path == "/v1/sessions/start":
                started.set()
                await finish.wait()
                peer_running = True
            return httpx.Response(200, json={"ok": True})

        app = make_app(monkeypatch, capacity=2, peer_handler=handler)
        fleet = app.state.fleet

        async def disable():
            async with fleet.control_lock:
                await fleet.assert_can_disable_multicamera()
                app.state.runtime._local_tracking_enabled = False

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            start = asyncio.create_task(
                client.post(
                    "/v1/fleet/cameras/camera02/start",
                    json={},
                )
            )
            await started.wait()
            change = asyncio.create_task(disable())
            await asyncio.sleep(0)
            assert not change.done()
            finish.set()
            assert (await start).status_code == 200
            with pytest.raises(HTTPException):
                await change
            assert app.state.runtime._local_tracking_enabled
        await fleet.aclose()

    asyncio.run(scenario())
