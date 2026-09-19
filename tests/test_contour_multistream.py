"""Two independent public node lanes; doubles test contracts, not GPU capacity."""

import asyncio
import base64
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_contour_node import AUTH, BASE, KEY, LIMITS, health
from test_sam_multistream import _payload, _setup

from goodvision_harness.perception import METHODS, ContourSessionHTTPClient
from goodvision_harness.video_tracking import VideoFrame
from workers.contour.app import create_app
from workers.contour.settings import Settings


def path(index):
    return f"/v1/cameras/camera-{index:02d}/contour/sessions"


@pytest.mark.parametrize("count", [2, 4])
def test_public_clients_isolate_lanes_and_close_without_stopping_siblings(count):
    worker, fleet, _gate, registry = _setup(count)
    fleet.source.backend.route_metadata = lambda session, frames: {
        "confirmation_policy": METHODS["sam_live"].confirmation_policy,
        "observations": [
            {"frame_sequence": f["frame_sequence"], "provisional_object_ids": []}
            for f in frames
        ],
    }
    # Protocol fixture only: native GPU readiness is covered by the worker's CUDA probe.
    for lane in fleet.engines.values():
        lane._production = True
    gateway = create_app(
        Settings(api_key=KEY, engine_url="http://worker", camera_capacity=count),
        transport=httpx.ASGITransport(app=worker),
    )
    clients = [
        ContourSessionHTTPClient(
            "http://node" + path(i),
            api_key=KEY,
            transport=httpx.ASGITransport(app=gateway),
        )
        for i in range(1, count + 1)
    ]
    try:
        with TestClient(gateway) as console:
            results, epochs, tokens = [], [], []
            for i, client in enumerate(clients, 1):
                assert client.ready_for_new_session()
                opened = client.open_session(f"source-{i}")
                epochs.append(opened.predictor_epoch)
                tokens.append(opened.video_session_id)
                assert not client.ready_for_new_session()
                row = _payload(i)["frames"][0]
                frame = VideoFrame(
                    row["frame_sequence"],
                    row["timestamp_seconds"],
                    base64.b64decode(row["jpeg_base64"]),
                )
                results.append(client.append((frame,), stream_session_id=f"source-{i}"))
            assert len(set(epochs)) == len(set(tokens)) == count
            assert len({r.frames[0].objects[0].track_id for r in results}) == count
            status = console.get("/api/status", headers=AUTH).json()
            assert status["ready"] and status["camera_capacity"] == count
            assert status["active_sessions"] == count
            assert len(status["cameras"]) == count
            assert "sam_live" not in str(status)
            assert (
                console.delete(path(2) + "/" + tokens[0], headers=AUTH).status_code
                == 404
            )
            assert (
                console.post(path(count + 1), json={}, headers=AUTH).status_code == 404
            )
            assert (
                console.post(
                    "/v1/contour/sessions",
                    json={"stream_session_id": "alias"},
                    headers=AUTH,
                ).status_code
                == 409
            )
            clients[0].discard_session()
            assert clients[0].ready_for_new_session()
            assert not clients[1].ready_for_new_session()
            row = _payload(2, start=1)["frames"][0]
            frame = VideoFrame(
                row["frame_sequence"],
                row["timestamp_seconds"],
                base64.b64decode(row["jpeg_base64"]),
            )
            result = clients[1].append((frame,), stream_session_id="source-2")
            assert result.predictor_epoch == epochs[1]
            assert result.frames[0].stream_session_id == "source-2"
            for client in clients[1:]:
                client.discard_session()
            assert (
                console.get("/api/status", headers=AUTH).json()["active_sessions"] == 0
            )
    finally:
        for client in clients:
            client.close()
        registry.close()


def test_gateway_accepts_parallel_lanes_but_rejects_overlap_within_one_lane():
    async def run():
        release, arrived = asyncio.Event(), asyncio.Event()
        calls = []

        async def backend(request):
            calls.append(request.url.path)
            if len(calls) == 2:
                arrived.set()
            await release.wait()
            return httpx.Response(200, json={**BASE, "limits": LIMITS})

        app = create_app(
            Settings(api_key=KEY, engine_url="http://engine", camera_capacity=2),
            transport=httpx.MockTransport(backend),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://node",
                headers=AUTH,
            ) as client,
        ):
            tasks = [asyncio.create_task(client.post(path(i), json={})) for i in (1, 2)]
            try:
                await asyncio.wait_for(arrived.wait(), 2)
                conflict = await client.post("/v1/contour/sessions", json={})
                assert conflict.status_code == 429
            finally:
                release.set()
            assert all(r.status_code == 200 for r in await asyncio.gather(*tasks))
            assert len(calls) == 2

    asyncio.run(run())


def test_capacity_mismatch_does_not_advertise_ready():
    def backend(request):
        if request.url.path == "/health/cameras":
            return httpx.Response(200, json={"max_active_sessions": 1, "cameras": []})
        return httpx.Response(200, json=health())

    app = create_app(
        Settings(api_key=KEY, engine_url="http://engine", camera_capacity=2),
        transport=httpx.MockTransport(backend),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready", headers=AUTH).status_code == 503
        assert not client.get("/api/status", headers=AUTH).json()["ready"]
        assert (
            client.get("/api/cameras/camera-03/status", headers=AUTH).status_code == 404
        )
        assert client.get("/api/cameras/camera-01/status").status_code == 401
    for capacity in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            replace(Settings(), camera_capacity=capacity)
