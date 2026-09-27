"""Real host transport and worker APIs with explicit, deterministic model doubles."""

import base64
import json
import threading
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_perception_worker_routes import causal_engine
from test_sam31_continuous_worker import payload
from test_video_tracking_session import SessionServer, frames, submit, wait_for

from goodvision_harness.config import RichLabelSettings
from goodvision_harness.perception import (
    METHODS,
    PerceptionSessionHTTPClient,
    create_tracking_client,
    create_tracking_coordinator,
    processing_methods,
    processing_status,
)
from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.video_tracking import VideoFrame
from goodvision_harness.video_tracking_session import VideoSessionTrackingHTTPClient
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.cloud_pool import create_app as pool_app
from workers.sam_scene.continuous import register_session_routes


class RouteServer(SessionServer):
    """Adapt the legacy double to an explicit causal route wire contract."""

    def __init__(self, method):
        super().__init__()
        self.method = METHODS[method]
        self.settings.update(
            startup_frames=1,
            max_append_frames=4,
            hotstart_delay=1,
            max_pending_frames=0,
        )
        self.route_mutate = None
        self.paths = []
        self.active_session = False

    async def __call__(self, request):
        self.paths.append(request.url.path)
        prefix = f"/v1/perception/{self.method.id}/sessions"
        path = request.url.path.replace(prefix, "/v1/video/sessions")
        proxy = httpx.Request(
            request.method, request.url.copy_with(path=path), content=request.content
        )
        response = await super().__call__(proxy)
        body = response.json()
        if "backend" in body:
            body["backend"] = self.method.backend
        if "native_sam_video_tracking" in body:
            body["native_sam_video_tracking"] = self.method.native_sam_video_tracking
            body["active_session"] = self.active_session
        if "frames" in body:
            observations = []
            for frame in body["frames"]:
                for obj in frame["objects"]:
                    obj["attributes"]["native_track"]["backend"] = self.method.backend
                observations.append(
                    {
                        "frame_sequence": frame["frame_sequence"],
                        "provisional_object_ids": ["1"] if frame["objects"] else [],
                    }
                )
            body["route_metadata"] = {
                "route": self.method.id,
                "confirmation_policy": self.method.confirmation_policy,
                "observations": observations,
            }
            if self.route_mutate:
                self.route_mutate(body)
        return httpx.Response(response.status_code, json=body)


@pytest.fixture(params=tuple(METHODS))
def route(request, tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_SAM_SPOOL_DIRECTORY", str(tmp_path / "sam-spool"))
    server = RouteServer(request.param)
    client = PerceptionSessionHTTPClient(
        "http://worker.test/v1/video/sessions",
        method=request.param,
        transport=httpx.MockTransport(server),
        timeout_seconds=2,
    )
    try:
        yield server, client
    finally:
        server.release.set()
        client.close()


def test_causal_client_keeps_backend_source_and_confirmation(route):
    server, client = route
    assert client.gpu_ready()  # protocol double, not physical GPU readiness
    assert client.startup_frames == 1
    client.open_session("camera")
    result = client.append(frames(1, 1), stream_session_id="camera")
    assert result.backend == server.method.backend
    assert result.pending_frame_count == 0
    snapshot = result.frames[0]
    assert snapshot.frame_sequence == 3
    assert snapshot.timestamp_seconds == 0.25
    assert snapshot.objects[0].attributes["tracking_confirmation"] == "provisional"
    result = client.append((), stream_session_id="camera", is_last_batch=True)
    assert result.state == "closed" and result.frames == ()
    client.discard_session()
    assert server.paths[0] == f"/health/perception/{server.method.id}"
    assert all("/v1/video" not in path for path in server.paths)


def test_new_session_readiness_requires_confirmed_global_idle(route):
    server, client = route
    assert client.ready_for_new_session()
    for active in (True, None, "false", 0):
        server.active_session = active
        assert client.gpu_ready()
        assert not client.ready_for_new_session()
    server.active_session = False
    assert client.ready_for_new_session()


@pytest.mark.parametrize("mutation,expected", [
    (lambda body: None, True),
    (lambda body: body.update(active_session=True), False),
    (lambda body: body.update(active_session=None), False),
    (lambda body: body.update(gpu_ready=False), False),
    (lambda body: body.update(ready=False), False),
    (lambda body: body.update(backend="unknown"), False),
    (lambda body: body.update(unknown_field=True), False),
    (lambda body: body.pop("ready"), False),
    (lambda body: body["limits"].update(hotstart_delay=5), False),
])
def test_causal_client_accepts_cloud_pool_health_without_bypassing_session_checks(
    mutation, expected,
):
    worker = RouteServer("sam_live")
    pool = pool_app(
        1, 1, launch_workers=False, transport=httpx.MockTransport(worker),
    )
    with TestClient(pool) as gateway:
        response = gateway.get("/health/perception/sam_live")
        assert response.status_code == 200
        body = response.json()
        assert body["model_instances"] == 1
        assert body["gpus"] == [{"index": 0, "ready": True}]
    mutation(body)
    client = PerceptionSessionHTTPClient(
        "http://pool.test/v1/video/sessions", method="sam_live",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    )
    try:
        assert client.ready_for_new_session() is expected
    finally:
        client.close()


def test_legacy_session_health_still_rejects_cloud_metadata():
    worker = SessionServer()

    async def augmented_health(request):
        response = await worker(request)
        return httpx.Response(200, json={**response.json(), "model_instances": 1})

    client = VideoSessionTrackingHTTPClient(
        "http://worker.test", transport=httpx.MockTransport(augmented_health),
    )
    try:
        assert not client.gpu_ready()
    finally:
        client.close()


def test_close_timeout_does_not_claim_background_work_has_finished(route):
    server, client = route
    rich = replace(
        RichLabelSettings(),
        backend="sam31_stream",
        processing_method=server.method.id,
        max_fps=4,
    )
    coordinator = create_tracking_coordinator(client, rich, lambda _: None)
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        coordinator.close(timeout_seconds=0)
        assert not coordinator.close_complete()
        server.release.set()
        wait_for(coordinator.close_complete, timeout=4)
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["route_metadata"].update(route="unknown"),
        lambda p: p["route_metadata"].update(confirmation_policy="trust_everything"),
        lambda p: p["route_metadata"].update(observations=[]),
        lambda p: p["route_metadata"]["observations"][0].update(frame_sequence=4),
        lambda p: p["route_metadata"]["observations"][0].update(
            provisional_object_ids=["missing"]
        ),
        lambda p: p["route_metadata"].update(diagnostics={"nested": {"untrusted": 1}}),
        lambda p: p["route_metadata"].update(extra=True),
        lambda p: p.update(backend="wrong-backend"),
        lambda p: p["frames"][0]["objects"][0]["attributes"]["native_track"].update(
            backend="wrong-backend"
        ),
        lambda p: p.pop("route_metadata"),
    ],
)
def test_route_contract_rejects_wrong_provenance_and_invalidates_session(
    route, mutation
):
    server, client = route
    client.open_session("camera")
    server.route_mutate = mutation
    with pytest.raises(RichLabelProtocolError):
        client.append(frames(1, 1), stream_session_id="camera")
    assert client.stats()["state"] == "invalid"
    assert client.stats()["accepted_frames"] == 0


def test_legacy_client_does_not_accept_causal_limits_or_route_metadata():
    server = RouteServer("sam_live")
    client = VideoSessionTrackingHTTPClient(
        "http://worker.test",
        transport=httpx.MockTransport(server),
    )
    try:
        assert not client.gpu_ready()
        with pytest.raises(RichLabelProtocolError):
            client.open_session("camera")
    finally:
        client.close()


def test_route_dispatches_first_frame_and_preserves_ordered_backlog_without_gpu(route):
    server, client = route
    rich = replace(
        RichLabelSettings(),
        backend="sam31_stream",
        processing_method=server.method.id,
        max_fps=4,
        video_queue_policy="durable_ordered",
    )
    received = []
    coordinator = create_tracking_coordinator(client, rich, received.append)
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)  # no 16-frame startup or output wait
        submit(coordinator, frames(2, 10))
        status = processing_status(coordinator, rich.processing_method)
        assert status["queue_frames"] == 10
        assert status["dropped_frames"] == 0
        assert status["source_to_result_seconds"] is None
        assert status["processed_frames"] == 0
        server.release.set()
        wait_for(lambda: coordinator.stats()["processed_frames"] == 11)
        status = processing_status(coordinator, rich.processing_method)
        assert status["source_to_result_seconds"] >= 0
        assert status["processed_frames"] == 11
        assert coordinator.stats()["backend"] == server.method.backend
        assert server.requests[0]["frames"][0]["frame_sequence"] == 3
        assert all(1 <= len(request["frames"]) <= 4 for request in server.requests)
        assert [frame["frame_sequence"] for request in server.requests
                for frame in request["frames"]] == list(range(3, 34, 3))
        assert [
            snapshot.frame_sequence for batch in received for snapshot in batch.frames
        ] == list(range(3, 34, 3))
        coordinator.flush()
        assert coordinator.wait_until_idle(2)
        coordinator.reset("new-source")
        assert (
            processing_status(coordinator, rich.processing_method)["processed_frames"]
            == 0
        )
        assert (
            processing_status(coordinator, rich.processing_method)[
                "source_to_result_seconds"
            ]
            is None
        )
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize("method", tuple(METHODS))
@pytest.mark.parametrize("transport_mode", ["direct", "prepared_jpeg", "prepared_h264"])
def test_real_worker_route_and_real_client_agree_on_source_contract(
    method, transport_mode,
):
    selected = causal_engine(method)
    codec = "h264" if transport_mode == "prepared_h264" else "jpeg"
    if codec == "h264":
        pytest.importorskip("av")
        selected.settings = replace(selected.settings, h264_enabled=True)
    selected.backend.route_metadata = lambda session, snapshots: {
        "confirmation_policy": METHODS[method].confirmation_policy,
        "observations": [
            {"frame_sequence": frame["frame_sequence"], "provisional_object_ids": ["7"]}
            for frame in snapshots
        ],
    }
    app = FastAPI()
    register_session_routes(
        app,
        selected,
        threading.Lock(),
        decode_jpeg,
        api_prefix=f"/v1/perception/{method}",
        health_path=f"/health/perception/{method}",
        legacy=False,
    )
    client = PerceptionSessionHTTPClient(
        "http://test",
        method=method,
        transport=httpx.ASGITransport(app=app),
        transport_codec=codec,
    )
    try:
        assert not client.gpu_ready()  # actual worker does not attest a double
        client.open_session("camera")
        row = json.loads(payload(1))["frames"][0]
        frame = VideoFrame(
            row["frame_sequence"],
            row["timestamp_seconds"],
            base64.b64decode(row["jpeg_base64"]),
        )
        prepared = (
            client.prepare_append((frame,), stream_session_id="camera")
            if transport_mode != "direct" else None
        )
        result = client.append((frame,), stream_session_id="camera", prepared=prepared)
        assert result.backend == METHODS[method].backend
        assert (
            result.frames[0].objects[0].attributes["tracking_confirmation"]
            == "provisional"
        )
        assert (
            client.append((), stream_session_id="camera", is_last_batch=True).frames
            == ()
        )
    finally:
        client.close()
        selected.close()


def test_factory_and_offline_catalog_require_no_model_packages():
    rich = replace(
        RichLabelSettings(), backend="sam31_stream", processing_method="sam_live"
    )
    client = create_tracking_client(rich)
    try:
        assert isinstance(client, PerceptionSessionHTTPClient)
        assert client.backend == "sam3.1_video"
        assert [row["id"] for row in processing_methods(rich)] == ["sam_live"]
        assert all(row["available"] for row in processing_methods(rich))
        assert not any(row["available"] for row in processing_methods(None))
    finally:
        client.close()
    with pytest.raises(ValueError, match="continuous"):
        create_tracking_client(replace(rich, backend="sam31"))
