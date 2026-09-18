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
    connected=(),
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
    app.state.own_requests = []
    app.state.statuses = {
        number: {
            "pipeline_state": "idle",
            "analysis_state": "idle",
            "connected": number in connected,
        }
        for number in range(1, capacity + 1)
    }

    def default_peer(request):
        app.state.peer_requests.append(request)
        number = int(request.url.host.removeprefix("harness-"))
        status = app.state.statuses[number]
        if request.url.path == "/v1/status":
            return httpx.Response(200, json=status)
        if request.url.path == "/v1/video/connect":
            status["connected"] = True
        if request.url.path == "/v1/video/disconnect":
            status["connected"] = False
        if request.url.path == "/v1/analysis/start":
            status.update(pipeline_state="running", analysis_state="running")
        if request.url.path == "/v1/analysis/stop":
            status.update(pipeline_state="idle", analysis_state="idle")
        if request.url.path.endswith("latest.jpg"):
            return httpx.Response(
                200,
                content=b"jpeg",
                headers={
                    "content-type": "image/jpeg",
                },
            )
        return httpx.Response(200, json={"ok": True})

    app.state.default_peer = default_peer
    fleet._peer_client = httpx.AsyncClient(
        transport=httpx.MockTransport(peer_handler or default_peer),
        trust_env=False,
        follow_redirects=False,
    )

    fleet._output_client = httpx.AsyncClient(
        transport=httpx.MockTransport(peer_handler or default_peer),
        trust_env=False,
        follow_redirects=False,
    )

    @app.get("/v1/status")
    async def own_status():
        return {**app.state.statuses[index], "source": "own"}

    @app.post("/v1/video/connect")
    async def own_connect(request: Request):
        app.state.statuses[index]["connected"] = True
        return {"own": True, "payload": await request.json()}

    @app.post("/v1/video/disconnect")
    async def own_disconnect():
        app.state.statuses[index]["connected"] = False
        return {"disconnected": True}

    @app.post("/v1/analysis/{action}")
    async def own_analysis(action: str, request: Request):
        app.state.own_requests.append((action, dict(request.headers)))
        if action == "start":
            app.state.statuses[index].update(
                pipeline_state="running", analysis_state="running"
            )
        elif action == "stop":
            app.state.statuses[index].update(
                pipeline_state="idle", analysis_state="idle"
            )
        return {"ok": True}

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


def test_connect_is_independent_of_model_keys_routes_and_analysis(monkeypatch):
    app = make_app(monkeypatch)
    app.state.runtime.route_probe_status = lambda: {"state": "running"}
    with TestClient(app) as client:
        response = client.post(
            "/v1/fleet/cameras/camera02/connect",
            json={"stream_url": "rtsp://camera/live"},
        )
        assert response.status_code == 200
        commands = [
            request for request in app.state.peer_requests if request.method != "GET"
        ]
        assert [request.url.path for request in commands] == ["/v1/video/connect"]
        assert commands[0].headers["X-Fleet-Coordinator"] == "1"
        assert app.state.statuses[2] == {
            "pipeline_state": "idle",
            "analysis_state": "idle",
            "connected": True,
        }


@pytest.mark.parametrize("action", ["start", "stop"])
def test_individual_analysis_buttons_cannot_be_called_through_old_api(
    monkeypatch, action
):
    app = make_app(monkeypatch, connected=(1, 2))
    with TestClient(app) as client:
        for camera in ["camera01", "camera02"]:
            result = client.post(f"/v1/fleet/cameras/{camera}/{action}", json={})
            assert result.status_code == 409
        assert not app.state.peer_requests
        assert not app.state.own_requests


def test_global_start_includes_all_connected_and_no_unconnected_camera(monkeypatch):
    app = make_app(monkeypatch, capacity=5, connected=(1, 3, 5))
    with TestClient(app) as client:
        result = client.post("/v1/fleet/analysis/start", json={})
        assert result.status_code == 200
        assert result.json()["analysis"]["camera_ids"] == [
            "camera01",
            "camera03",
            "camera05",
        ]
        assert [action for action, _ in app.state.own_requests] == ["prepare", "start"]
        requests = [
            request
            for request in app.state.peer_requests
            if request.url.path.startswith("/v1/analysis/")
        ]
        assert {request.url.host for request in requests} == {
            "harness-03",
            "harness-05",
        }
        assert all(
            request.headers["X-Fleet-Coordinator"] == "1" for request in requests
        )
        assert all(json.loads(request.content) == {} for request in requests)
        descriptor = client.get("/v1/fleet").json()
        assert descriptor["analysis"] == {
            "state": "running",
            "camera_ids": ["camera01", "camera03", "camera05"],
            "errors": [],
        }
        for camera in descriptor["cameras"]:
            for key in [
                "connect_url",
                "disconnect_url",
                "detail_url",
                "results_url",
                "scene_url",
            ]:
                assert camera[key].startswith(f"/v1/fleet/cameras/{camera['id']}/")


def test_global_start_requires_a_connected_cohort_and_rejects_subsets(monkeypatch):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 400
        assert (
            client.post(
                "/v1/fleet/analysis/start", json={"camera_ids": ["camera01"]}
            ).status_code
            == 400
        )
        assert not app.state.own_requests
        assert all(request.method == "GET" for request in app.state.peer_requests)


@pytest.mark.parametrize("state", ["connecting", "disconnecting"])
def test_global_start_waits_for_all_connection_transitions(monkeypatch, state):
    app = make_app(monkeypatch, capacity=2, connected=(1,))
    app.state.statuses[2]["connection_state"] = state
    with TestClient(app) as client:
        result = client.post("/v1/fleet/analysis/start", json={})
        assert result.status_code == 409
        assert not app.state.own_requests
        assert all(request.method == "GET" for request in app.state.peer_requests)
        app.state.statuses[2].update(connection_state="connected", connected=True)
        result = client.post("/v1/fleet/analysis/start", json={})
        assert result.status_code == 200
        assert result.json()["analysis"]["camera_ids"] == ["camera01", "camera02"]


def test_failed_preflight_starts_no_analysis_and_preserves_connections(monkeypatch):
    def handler(request):
        if (
            request.url.host == "harness-03"
            and request.url.path == "/v1/analysis/prepare"
        ):
            return httpx.Response(409, json={"detail": "tracking not ready"})
        return app.state.default_peer(request)

    app = make_app(monkeypatch, connected=(1, 2, 3), peer_handler=handler)
    with TestClient(app) as client:
        result = client.post("/v1/fleet/analysis/start", json={})
        assert result.status_code == 409
        assert result.json()["analysis"]["errors"] == [
            {"camera_id": "camera03", "message": "tracking not ready"},
        ]
        assert not any(
            request.url.path == "/v1/analysis/start"
            for request in app.state.peer_requests
        )
        assert [action for action, _ in app.state.own_requests] == ["prepare"]
        assert all(
            status["connected"] and status["analysis_state"] == "idle"
            for status in app.state.statuses.values()
        )
        assert client.get("/v1/fleet").json()["analysis"]["state"] == "error"


@pytest.mark.parametrize("failure", ["response", "timeout", "inactive_status"])
def test_failed_global_start_rolls_back_every_attempted_lane(monkeypatch, failure):
    def handler(request):
        if (
            request.url.host == "harness-03"
            and request.url.path == "/v1/analysis/start"
        ):
            # The worker started even if the network then lost its response.
            app.state.default_peer(request)
            if failure == "timeout":
                raise httpx.ReadTimeout("private internal details", request=request)
            if failure == "inactive_status":
                return httpx.Response(200, json={"analysis_state": "error"})
            return httpx.Response(503, json={"detail": "worker unavailable"})
        return app.state.default_peer(request)

    app = make_app(monkeypatch, connected=(1, 2, 3), peer_handler=handler)
    with TestClient(app) as client:
        result = client.post("/v1/fleet/analysis/start", json={})
        assert result.status_code == 502
        assert "private internal details" not in result.text
        assert [action for action, _ in app.state.own_requests] == [
            "prepare",
            "start",
            "stop",
        ]
        assert {
            request.url.host
            for request in app.state.peer_requests
            if request.url.path == "/v1/analysis/stop"
        } == {"harness-02", "harness-03"}
        assert all(
            status["connected"] and status["analysis_state"] == "idle"
            for status in app.state.statuses.values()
        )


def test_analysis_state_is_recovered_after_controller_restart(monkeypatch):
    app = make_app(monkeypatch, connected=(1, 3))
    for index in [1, 3]:
        app.state.statuses[index].update(
            pipeline_state="running", analysis_state="running"
        )
    with TestClient(app) as client:
        assert client.get("/v1/fleet").json()["analysis"] == {
            "state": "running",
            "camera_ids": ["camera01", "camera03"],
            "errors": [],
        }
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 409
        stopped = client.post("/v1/fleet/analysis/stop", json={})
        assert stopped.status_code == 200
        assert stopped.json()["analysis"]["state"] == "idle"
        assert app.state.statuses[1]["connected"]
        assert app.state.statuses[3]["connected"]


def test_partial_stop_failure_remains_retryable_and_stops_disabled_lanes(monkeypatch):
    reachable = False

    def handler(request):
        if request.url.host == "harness-03" and not reachable:
            raise httpx.ConnectError("peer down", request=request)
        return app.state.default_peer(request)

    app = make_app(monkeypatch, tracking=False, peer_handler=handler)
    app.state.statuses[3].update(pipeline_state="running", analysis_state="running")
    with TestClient(app) as client:
        failed = client.post("/v1/fleet/analysis/stop", json={})
        assert failed.status_code == 503
        assert failed.json()["analysis"]["state"] == "error"
        assert failed.json()["analysis"]["errors"]
        reachable = True
        stopped = client.post("/v1/fleet/analysis/stop", json={})
        assert stopped.status_code == 200
        assert stopped.json()["analysis"] == {
            "state": "idle",
            "camera_ids": [],
            "errors": [],
        }
        assert any(
            request.url.host == "harness-03" and request.url.path == "/v1/analysis/stop"
            for request in app.state.peer_requests
        )


@pytest.mark.parametrize("state", ["starting", "running", "draining", "unknown"])
def test_global_analysis_locks_connections_and_camera_settings(monkeypatch, state):
    app = make_app(monkeypatch)
    app.state.statuses[1].update(pipeline_state=state, analysis_state=state)
    with TestClient(app) as client:
        prefix = "/v1/fleet/cameras/camera02"
        assert client.post(prefix + "/connect", json={}).status_code == 409
        assert client.post(prefix + "/disconnect", json={}).status_code == 409
        assert client.put(prefix + "/details", json={"features": []}).status_code == 409
        assert (
            client.put(
                prefix + "/settings", json={"stream_url": "rtsp://camera/new"}
            ).status_code
            == 409
        )
        assert all(request.method == "GET" for request in app.state.peer_requests)


def test_each_camera_details_and_results_are_proxied_to_its_own_runtime(monkeypatch):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        prefix = "/v1/fleet/cameras/camera03"
        assert client.get(prefix + "/details").status_code == 200
        assert (
            client.put(
                prefix + "/details",
                json={"features": ["events"], "event_sensitivities": {}},
            ).status_code
            == 200
        )
        assert client.get(prefix + "/results/latest").status_code == 200
        assert client.get(prefix + "/scene").status_code == 200
        commands = [
            request
            for request in app.state.peer_requests
            if request.url.path != "/v1/status"
        ]
        assert [(request.method, request.url.path) for request in commands] == [
            ("GET", "/v1/analysis/settings"),
            ("PUT", "/v1/analysis/settings"),
            ("GET", "/v1/results/latest"),
            ("GET", "/v1/scene"),
        ]
        assert all(request.url.host == "harness-03" for request in commands)
        assert commands[1].headers["X-Fleet-Coordinator"] == "1"


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
        response = client.post("/v1/fleet/cameras/camera02/connect", json={})
        assert response.status_code == (200 if allowed else 409)
        assert (
            client.post("/v1/fleet/cameras/camera01/connect", json={}).status_code
            == 200
        )


def test_own_camera_reuses_original_handlers_without_network_or_settings_deadlock(
    monkeypatch,
):
    app = make_app(monkeypatch, capacity=1)
    with TestClient(app) as client:
        prefix = "/v1/fleet/cameras/camera01"
        assert client.get(prefix + "/status").json()["source"] == "own"
        assert client.post(
            prefix + "/connect", json={"stream_url": "saved"}
        ).json() == {
            "own": True,
            "payload": {"stream_url": "saved"},
        }
        assert client.put(prefix + "/settings", json={"dry_run": True}).json() == {
            "saved": {"dry_run": True},
        }
        assert client.get(prefix + "/video/annotated/latest.jpg").content == b"own-jpeg"
        assert not app.state.peer_requests


@pytest.mark.parametrize("tracking_key", [None, "private-node-key"])
def test_peer_start_syncs_global_gpu_modes_credentials_and_cloud_route(
    monkeypatch, tracking_key
):
    app = make_app(monkeypatch, connected=(2,))
    app.state.runtime._aiclub_api_key = "private-api-key"
    app.state.runtime._tracking_api_key = tracking_key
    app.state.runtime.route_probe_status = lambda: {
        "state": "idle",
        "supported": True,
        "selected_id": "route-A",
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/fleet/analysis/start?target=http://other-host",
            json={},
            headers={"Authorization": "Bearer browser-secret", "Cookie": "private=1"},
        )
        assert response.status_code == 200
        requests = [
            request
            for request in app.state.peer_requests
            if request.url.path != "/v1/status"
        ]
        assert [request.url.path for request in requests] == [
            "/v1/deployment/settings",
            "/v1/deployment/routes/select",
            "/v1/analysis/prepare",
            "/v1/analysis/start",
        ]
        assert all(request.url.host == "harness-02" for request in requests)
        assert all(not request.url.query for request in requests)
        assert all("authorization" not in request.headers for request in requests)
        assert all("cookie" not in request.headers for request in requests)
        synchronized = json.loads(requests[0].content)
        assert synchronized == {
            "local_tracking_enabled": True,
            "local_inference_enabled": False,
            "dry_run": False,
            "autostart": False,
            "api_key": "private-api-key",
            **(
                {"tracking_api_key": tracking_key}
                if tracking_key
                else {"clear_tracking_api_key": True}
            ),
        }
        assert "stream_url" not in synchronized
        assert "private-api-key" not in response.text
        assert "private-node-key" not in response.text


def test_peer_start_copies_primary_saved_test_mode(monkeypatch):
    app = make_app(monkeypatch, connected=(2,))
    app.state.runtime.dry_run = True
    with TestClient(app) as client:
        response = client.post("/v1/fleet/analysis/start", json={})
        assert response.status_code == 200
        requests = [
            request
            for request in app.state.peer_requests
            if request.url.path != "/v1/status"
        ]
        settings = json.loads(requests[0].content)
        assert settings["dry_run"] is True
        assert "dry_run" not in json.loads(requests[-1].content)


def test_absent_primary_key_clears_old_peer_key(monkeypatch):
    app = make_app(monkeypatch, connected=(2,))
    with TestClient(app) as client:
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 200
        settings = json.loads(
            next(
                request.content
                for request in app.state.peer_requests
                if request.url.path == "/v1/deployment/settings"
            )
        )
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
        path = "/v1/fleet/cameras/camera02/connect"
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
        assert client.post(path + "/disconnect").status_code == 200
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
    app = make_app(monkeypatch, connected=(2,))
    app.state.runtime.route_probe_status = lambda: {"state": "running"}
    with TestClient(app) as client:
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 409
        assert all(
            request.url.path == "/v1/status" for request in app.state.peer_requests
        )


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
            client.post("/v1/fleet/cameras/camera03/connect", json={}).status_code
            == 409
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
                        "connected": True,
                    },
                )
            if request.url.path == "/v1/analysis/start":
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
                    "/v1/fleet/analysis/start",
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


@pytest.mark.parametrize(
    "provisioned,server_capacity,usable",
    [
        (24, 16, 16),
        (16, 24, 16),
        (3, 1, 1),
        (100, 100, 100),
    ],
)
def test_self_hosted_tracking_on_another_machine_enables_only_available_lanes(
    monkeypatch,
    provisioned,
    server_capacity,
    usable,
):
    app = make_app(monkeypatch, capacity=provisioned, tracking=False, available=False)
    app.state.runtime.tracking_camera_capacity = lambda: server_capacity
    app.state.runtime.service_execution_settings = lambda: {
        "inference_mode": "cloud",
        "tracking_mode": "server",
        "inference_server_url": "",
        "inference_server_model": "local-vision",
        "tracking_server_url": "http://other-gpu-machine:8092",
        "tracking_server_capacity": server_capacity,
    }
    with TestClient(app) as client:
        descriptor = client.get("/v1/fleet").json()
        assert descriptor["tracking_mode"] == "server"
        assert descriptor["capacity"] == provisioned
        assert descriptor["usable_capacity"] == usable
        assert descriptor["multi_camera_allowed"] is (usable > 1)
        assert sum(camera["enabled"] for camera in descriptor["cameras"]) == usable
        response = client.post(f"/v1/fleet/cameras/camera{usable:02d}/connect", json={})
        assert response.status_code == 200
        if usable < provisioned:
            assert (
                client.post(
                    f"/v1/fleet/cameras/camera{usable + 1:02d}/connect",
                    json={},
                ).status_code
                == 409
            )


def test_peer_start_synchronizes_modular_server_settings_without_cloud_credentials(
    monkeypatch,
):
    app = make_app(monkeypatch, tracking=False, available=False, connected=(2,))
    execution = {
        "inference_mode": "server",
        "tracking_mode": "server",
        "inference_server_url": "http://vlm-machine:8000",
        "inference_server_model": "supplied-vision",
        "tracking_server_url": "http://tracking-machine:8092",
        "tracking_server_capacity": 24,
    }
    app.state.runtime.tracking_camera_capacity = lambda: 24
    app.state.runtime.service_execution_settings = lambda: {
        **execution,
        "unrelated_public_field": "must-not-be-forwarded",
    }
    with TestClient(app) as client:
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 200
        payload = json.loads(
            next(
                request.content
                for request in app.state.peer_requests
                if request.url.path == "/v1/deployment/settings"
            )
        )
        assert payload.items() >= execution.items()
        assert "unrelated_public_field" not in payload
        assert payload["local_tracking_enabled"] is False
        assert payload["local_inference_enabled"] is False
        assert payload["clear_api_key"] is True


@pytest.mark.parametrize(
    "field",
    [
        "inference_mode",
        "tracking_mode",
        "inference_server_url",
        "inference_server_model",
        "tracking_server_url",
        "tracking_server_capacity",
    ],
)
def test_peer_settings_cannot_change_global_service_locations(monkeypatch, field):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        response = client.put(
            "/v1/fleet/cameras/camera02/settings", json={field: "new"}
        )
        assert response.status_code == 400
        assert not app.state.peer_requests
