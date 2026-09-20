import json

import httpx
import pytest
from fastapi.testclient import TestClient
from test_fleet import make_app


@pytest.mark.parametrize("capacity,lane", [(1, 1), (2, 2), (37, 37)])
def test_diagnostic_document_is_scoped_and_has_no_global_controls(
    monkeypatch,
    capacity,
    lane,
):
    app = make_app(monkeypatch, capacity=capacity)
    camera = f"camera{lane:02}"
    with TestClient(app) as client:
        descriptor = client.get("/v1/fleet").json()["cameras"][lane - 1]
        page = client.get(descriptor["diagnostics_url"])
        assert page.status_code == 200
        assert page.headers["x-frame-options"] == "SAMEORIGIN"
        assert f'fetch("/v1/fleet/cameras/{camera}/status"' in page.text
        assert f'fetch("/v1/fleet/cameras/{camera}/deployment/settings"' in page.text
        assert (
            f'new EventSource("/v1/fleet/cameras/{camera}/results/events' in page.text
        )
        assert f'href="/v1/fleet/cameras/{camera}/video/stream.mjpeg"' in page.text
        assert 'src="/assets/fleet_console.js"' not in page.text
        assert 'fetch("/v1/status"' not in page.text
        assert "loadRoutes();setInterval(loadRoutes,1500)" not in page.text
        assert "globalThis.GoodvisionFleet={detailHidden:()=>false" in page.text


def test_peer_buffer_updates_while_analysis_runs_without_affecting_primary(monkeypatch):
    app = make_app(monkeypatch)
    app.state.statuses[1]["analysis_state"] = "running"
    app.state.statuses[2]["analysis_state"] = "running"
    with TestClient(app) as client:
        response = client.put(
            "/v1/fleet/cameras/camera02/deployment/settings",
            json={"buffer_delay_seconds": 8},
        )
        assert response.status_code == 200
        assert len(app.state.peer_requests) == 1
        call = app.state.peer_requests[0]
        assert call.url.host == "harness-02"
        assert call.url.path == "/v1/deployment/settings"
        assert json.loads(call.content) == {"buffer_delay_seconds": 8}
        assert not app.state.own_requests
        assert app.state.statuses[1]["analysis_state"] == "running"


@pytest.mark.parametrize("camera", ["camera01", "camera02"])
def test_diagnostics_reject_global_writes(monkeypatch, camera):
    app = make_app(monkeypatch)
    with TestClient(app) as client:
        response = client.put(
            f"/v1/fleet/cameras/{camera}/deployment/settings",
            json={"buffer_delay_seconds": 2, "dry_run": True},
        )
        assert response.status_code == 400
        assert not app.state.peer_requests


def test_info_and_links_come_from_selected_lane(monkeypatch):
    def peer(request):
        assert request.url.host == "harness-02"
        assert request.url.path == "/v1/ui/info"
        return httpx.Response(
            200,
            json={
                "name": "second",
                "annotated_output": {"url": "/v1/video/annotated/latest.jpg"},
            },
        )

    app = make_app(monkeypatch, peer_handler=peer)
    with TestClient(app) as client:
        info = client.get("/v1/fleet/cameras/camera02/ui/info").json()
        assert info["name"] == "second"
        assert info["annotated_output"]["url"].startswith("/v1/fleet/cameras/camera02/")
        assert client.get("/v1/fleet/cameras/camera99/diagnostics").status_code == 404
        assert client.get("/v1/fleet/cameras/camera2/diagnostics").status_code == 404


def test_offline_lane_does_not_fall_back_to_primary(monkeypatch):
    def peer(request):
        raise httpx.ConnectError("unreachable", request=request)

    app = make_app(monkeypatch, peer_handler=peer)
    with TestClient(app) as client:
        assert client.get("/v1/fleet/cameras/camera02/ui/info").status_code == 503
