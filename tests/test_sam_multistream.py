"""Camera isolation and scheduler correctness with doubles; no GPU speed claim."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_perception_worker_routes import causal_engine
from test_sam31_continuous_worker import engine as legacy_engine
from test_sam31_continuous_worker import payload

from goodvision_harness.perception import METHODS, PerceptionSessionHTTPClient
from goodvision_harness.video_tracking import VideoFrame
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import SessionRegistry, register_session_routes
from workers.sam_scene.multistream import (
    CameraFleet,
    FairGPUAdmission,
    camera_count_from_env,
    register_camera_routes,
)


def _setup(count=12):
    registry = SessionRegistry()
    legacy = legacy_engine()
    registry.add(legacy)
    source = causal_engine(registry=registry)
    fleet = CameraFleet(source, registry, count)
    fleet.warmup()
    gate = FairGPUAdmission(count * 2)
    app = FastAPI()
    register_session_routes(app, legacy, gate, decode_jpeg)
    register_camera_routes(app, fleet, gate, decode_jpeg)
    return app, fleet, gate, registry


def _path(index):
    return f"/v1/cameras/camera-{index:02d}/perception/sam_live/sessions"


def _payload(index, *, start=0):
    value = json.loads(payload(1, start=start, request_id=f"request-{start}"))
    value["stream_session_id"] = f"source-{index}"
    return value


@pytest.mark.parametrize("count", [2, 12, 16])
def test_camera_lanes_share_one_backend_but_isolate_state_and_ownership(count):
    app, fleet, _gate, registry = _setup(count)
    try:
        with TestClient(app) as client:
            tokens = []
            epochs = []
            for index in range(1, count + 1):
                response = client.post(
                    _path(index),
                    json={
                        "stream_session_id": f"source-{index}",
                    },
                )
                assert response.status_code == 200
                created = response.json()
                tokens.append(created["video_session_id"])
                epochs.append(created["predictor_epoch"])
                frame = client.post(
                    f"{_path(index)}/{tokens[-1]}/append",
                    json=_payload(index),
                )
                assert frame.status_code == 200
                assert frame.json()["accepted_frame_sequence"] == 0
            assert len(set(tokens)) == len(set(epochs)) == count
            assert (
                len({id(e.session.native_state) for e in fleet.engines.values()})
                == count
            )
            assert all(
                e.backend.source is fleet.source.backend for e in fleet.engines.values()
            )
            health = client.get("/health/cameras").json()
            assert health["active_sessions"] == count
            assert health["max_active_sessions"] == count
            assert health["max_queued_requests"] == count * 2
            # Test doubles never advertise actual CUDA readiness.
            assert client.get("/health/cameras").json()["gpu_ready"] is False
            assert (
                client.post(
                    _path(count + 1), json={"stream_session_id": "x"}
                ).status_code
                == 404
            )
            assert (
                client.post(_path(0), json={"stream_session_id": "x"}).status_code
                == 404
            )
            assert (
                client.post(_path(1), json={"stream_session_id": "new"}).status_code
                == 409
            )
            assert (
                client.post(
                    f"{_path(2)}/{tokens[0]}/append",
                    json=_payload(2, start=1),
                ).status_code
                == 404
            )
            assert client.delete(f"{_path(2)}/{tokens[0]}").status_code == 404
            assert (
                client.post(
                    "/v1/video/sessions",
                    json={"stream_session_id": "legacy"},
                ).status_code
                == 409
            )
            assert client.delete(f"{_path(1)}/{tokens[0]}").status_code == 200
            appended = client.post(
                f"{_path(2)}/{tokens[1]}/append",
                json=_payload(2, start=1),
            )
            assert appended.status_code == 200
            assert appended.json()["predictor_epoch"] == epochs[1]
            assert appended.json()["accepted_frame_sequence"] == 2
            for index in range(2, count + 1):
                assert (
                    client.delete(
                        f"{_path(index)}/{tokens[index - 1]}",
                    ).status_code
                    == 200
                )
            assert not fleet.has_active_session
            legacy = client.post(
                "/v1/video/sessions",
                json={"stream_session_id": "legacy"},
            )
            assert legacy.status_code == 200
            assert (
                client.post(_path(1), json={"stream_session_id": "new"}).status_code
                == 409
            )
    finally:
        registry.close()


def test_fifo_admission_is_bounded_and_does_not_allow_legacy_barging():
    async def run():
        gate = FairGPUAdmission(3)
        assert gate.acquire()
        order = []

        async def wait(index):
            assert await gate.acquire_async(time.monotonic() + 1)
            order.append(index)
            await asyncio.sleep(0)
            gate.release()

        tasks = []
        for index in range(3):
            tasks.append(asyncio.create_task(wait(index)))
            await asyncio.sleep(0)
        assert gate.waiting == 3
        assert not await gate.acquire_async(time.monotonic() + 1)
        gate.release()
        assert not gate.acquire()
        await asyncio.gather(*tasks)
        assert order == [0, 1, 2]
        assert gate.waiting == 0 and not gate.locked()

    asyncio.run(run())


@pytest.mark.parametrize("count", [2, 12, 16])
def test_real_host_clients_validate_worker_lane_routes_and_readiness(count):
    app, fleet, _gate, registry = _setup(count)
    fleet.source.backend.route_metadata = lambda session, snapshots: {
        "confirmation_policy": METHODS["sam_live"].confirmation_policy,
        "observations": [
            {
                "frame_sequence": frame["frame_sequence"],
                "provisional_object_ids": ["7"],
            }
            for frame in snapshots
        ],
    }
    # Advertise ready wire fields only for this protocol fixture; native CUDA
    # capability is separately gated by the production startup probe.
    for lane in fleet.engines.values():
        lane._production = True
    clients = [
        PerceptionSessionHTTPClient(
            "http://worker" + _path(index),
            method="sam_live",
            transport=httpx.ASGITransport(app=app),
        )
        for index in range(1, count + 1)
    ]
    try:
        epochs = []
        for index, client in enumerate(clients, 1):
            assert client.ready_for_new_session()
            session = client.open_session(f"source-{index}")
            epochs.append(session.predictor_epoch)
            assert not client.ready_for_new_session()
            if index == 1:
                assert clients[1].ready_for_new_session()
            row = _payload(index)["frames"][0]
            frame = VideoFrame(
                row["frame_sequence"],
                row["timestamp_seconds"],
                base64.b64decode(row["jpeg_base64"]),
            )
            result = client.append((frame,), stream_session_id=f"source-{index}")
            assert result.backend == "sam3.1_video"
            assert result.predictor_epoch == session.predictor_epoch
            assert result.accepted_frame_sequence == result.emitted_frame_sequence == 0
            assert (
                result.frames[0].objects[0].attributes["tracking_confirmation"]
                == "provisional"
            )
        assert len(set(epochs)) == count
    finally:
        for client in clients:
            client.close()
        registry.close()


@pytest.mark.parametrize("cancel_after_grant", [False, True])
def test_cancelled_queue_waiter_cannot_leak_or_steal_gpu_slot(cancel_after_grant):
    async def run():
        gate = FairGPUAdmission(2)
        assert gate.acquire()
        pending = asyncio.create_task(gate.acquire_async(time.monotonic() + 1))
        await asyncio.sleep(0)
        if cancel_after_grant:
            gate.release()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert gate.waiting == 0
        assert gate.locked() is (not cancel_after_grant)
        if not cancel_after_grant:
            gate.release()
        assert gate.acquire()
        gate.release()

    asyncio.run(run())


def test_queue_deadline_does_not_release_active_native_operation():
    async def run():
        gate = FairGPUAdmission(1)
        assert gate.acquire()
        with pytest.raises(TimeoutError):
            await gate.acquire_async(time.monotonic() + 0.01)
        assert gate.waiting == 0 and gate.locked()
        gate.release()
        assert gate.acquire()
        gate.release()

    asyncio.run(run())


def test_queued_http_timeout_preserves_the_existing_session():
    app, fleet, gate, registry = _setup(2)
    lane = fleet.engines["camera-01"]
    try:
        with TestClient(app) as client:
            token = client.post(
                _path(1), json={"stream_session_id": "source-1"}
            ).json()["video_session_id"]
            lane.video.settings = replace(
                lane.video.settings, request_timeout_seconds=0.1
            )
            assert gate.acquire()
            try:
                response = client.delete(f"{_path(1)}/{token}")
                assert response.status_code == 504
                assert lane.session.status == "active"
                assert gate.locked() and gate.waiting == 0
            finally:
                gate.release()
            assert client.delete(f"{_path(1)}/{token}").status_code == 200
    finally:
        registry.close()


def test_timed_out_cuda_work_keeps_other_cameras_queued_until_actual_completion():
    app, fleet, gate, registry = _setup(2)
    backend = fleet.source.backend
    backend.release = threading.Event()
    lane = fleet.engines["camera-01"]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://worker",
        ) as client:
            created = await client.post(
                _path(1), json={"stream_session_id": "source-1"}
            )
            token = created.json()["video_session_id"]
            lane.video.settings = replace(
                lane.video.settings, request_timeout_seconds=0.2
            )
            failed = await client.post(
                f"{_path(1)}/{token}/append",
                json=_payload(1),
            )
            assert failed.status_code == 504
            assert gate.locked() and backend.entered.is_set()
            other = asyncio.create_task(
                client.post(
                    _path(2),
                    json={"stream_session_id": "source-2"},
                )
            )
            await asyncio.sleep(0.01)
            assert not other.done() and gate.waiting == 1
            backend.release.set()
            assert (await other).status_code == 200
            assert lane.session is None
            assert not gate.locked()

    try:
        asyncio.run(run())
    finally:
        backend.release.set()
        registry.close()


@pytest.mark.parametrize("value", ["0", "abc", "1.5", "-1", ""])
def test_camera_count_rejects_invalid_capacity(monkeypatch, value):
    monkeypatch.setenv("SAM31_LIVE_CAMERAS", value)
    with pytest.raises(ValueError, match="positive integer"):
        camera_count_from_env()


def test_camera_count_defaults_to_legacy_single_session(monkeypatch):
    monkeypatch.delenv("SAM31_LIVE_CAMERAS", raising=False)
    assert camera_count_from_env() == 1


@pytest.mark.parametrize("count", [1, 2, 6, 12, 13, 16, 24, 100])
def test_camera_count_accepts_configured_positive_capacity(monkeypatch, count):
    monkeypatch.setenv("SAM31_LIVE_CAMERAS", str(count))
    assert camera_count_from_env() == count


@pytest.mark.parametrize("count", [0, 1, -1, 1.5, "2", True])
def test_camera_fleet_rejects_invalid_lane_count(count):
    with pytest.raises(ValueError, match="at least 2 lanes"):
        CameraFleet(None, None, count)


@pytest.mark.parametrize("count", [13, 16, 24, 100])
def test_camera_fleet_creates_all_configured_lane_ids(count):
    registry = SessionRegistry()
    source = causal_engine(registry=registry)
    try:
        fleet = CameraFleet(source, registry, count)
        assert list(fleet.engines) == [
            f"camera-{index:02d}" for index in range(1, count + 1)
        ]
        assert len({id(engine) for engine in fleet.engines.values()}) == count
    finally:
        registry.close()


def test_failed_shared_warmup_never_advertises_camera_readiness(monkeypatch):
    registry = SessionRegistry()
    source = causal_engine(registry=registry)
    source._production = True
    fleet = CameraFleet(source, registry, 12)

    def failure():
        raise RuntimeError("interleaved state probe failed")

    monkeypatch.setattr(fleet, "_verify_interleaved_states", failure)
    with pytest.raises(RuntimeError, match="probe failed"):
        fleet.warmup()
    assert all(
        not engine.ready and not engine.gpu_ready for engine in fleet.engines.values()
    )
    assert all(
        engine.error_code == "camera_initialization_failed"
        for engine in fleet.engines.values()
    )
    registry.close()
