"""Real host/gateway/worker contracts; inference doubles are not GPU validation."""

import base64
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_contour_node import KEY, LIMITS
from test_perception_worker_routes import causal_engine
from test_sam31_continuous_worker import payload

from goodvision_harness.config import RichLabelSettings, load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.perception import (
    ContourSessionHTTPClient,
    create_tracking_client,
)
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.service import HarnessRuntime
from goodvision_harness.service import create_app as main_app
from goodvision_harness.service_locations import ServiceLocations
from goodvision_harness.video_tracking import VideoFrame
from workers.contour.app import create_app
from workers.contour.settings import Settings
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import register_session_routes
from workers.sam_scene.sam_live import SamLiveContinuation


@pytest.mark.parametrize("codec", ["jpeg", "h264"])
def test_main_client_through_real_gateway_and_worker(codec, tmp_path):
    if codec == "h264":
        pytest.importorskip("av")
    selected = causal_engine()
    selected.settings = replace(selected.settings, h264_enabled=True)
    selected.backend.route_metadata = MethodType(
        SamLiveContinuation.route_metadata, selected.backend
    )
    worker = FastAPI()
    upstream_headers = []

    @worker.middleware("http")
    async def record_upstream(request, call_next):
        upstream_headers.append(request.headers.get("authorization"))
        return await call_next(request)

    register_session_routes(
        worker,
        selected,
        threading.Lock(),
        decode_jpeg,
        api_prefix="/v1/perception/sam_live",
        legacy=False,
    )
    gateway = create_app(
        Settings(
            api_key=KEY,
            checkpoint_path=tmp_path / "weights",
            engine_url="http://engine",
        ),
        transport=httpx.ASGITransport(app=worker),
    )
    headers = []

    @gateway.middleware("http")
    async def record_public(request, call_next):
        headers.append(
            (request.method, request.url.path, request.headers.get("authorization"))
        )
        return await call_next(request)

    client = ContourSessionHTTPClient(
        "http://node.test:8093",
        api_key=KEY,
        transport_codec=codec,
        transport=httpx.ASGITransport(app=gateway),
    )
    try:
        with TestClient(gateway):
            assert not client.gpu_ready()  # A double must not attest CUDA readiness.
            client.open_session("camera")
            row = json.loads(payload(1))["frames"][0]
            frame = VideoFrame(
                row["frame_sequence"],
                row["timestamp_seconds"],
                base64.b64decode(row["jpeg_base64"]),
            )
            prepared = client.prepare_append((frame,), stream_session_id="camera")
            result = client.append(
                (frame,), stream_session_id="camera", prepared=prepared
            )
            assert result.backend == "goodvision_contour"
            item = result.frames[0].objects[0]
            assert item.track_id.startswith("ct-")
            assert item.attributes["mask_contours"]["polygons"]
            assert item.attributes["tracking_confirmation"] in (
                "confirmed",
                "provisional",
            )
            registry = PersonRegistry()
            registry.observe_snapshot(result.frames[0])
            assert registry.native_tracking_stats()["backend"] == "goodvision_contour"
            assert registry.native_tracking_stats()["tracks"] == 1
            scene = SceneRuntime(
                "camera",
                "source",
                tracking_mode="batch",
                native_memory_scope="session",
                event_labels=(),
            )
            scene.observe_frame(
                frame.jpeg_bytes,
                frame.timestamp_seconds,
                frame.frame_sequence,
                force=True,
            )
            assert scene.commit_native_tracking_batch(
                result.frames,
                completed_timestamp_seconds=result.completed_timestamp_seconds,
                completed_frame_sequence=result.completed_frame_sequence,
                canonicalize=lambda frames: tuple(
                    registry.snapshot_at(item.timestamp_seconds, item.frame_sequence)
                    for item in frames
                ),
                retained_sources=(frame,),
            ), scene.tracking_last_rejection
            assert len(scene.snapshot()["scene"]["entities"]) == 1
            assert (
                client.append((), stream_session_id="camera", is_last_batch=True).state
                == "closed"
            )
            client.discard_session()
            assert headers[-1][0] == "DELETE"
            assert all(auth == "Bearer " + KEY for _, _, auth in headers)
            assert all(
                path.startswith(("/api/status", "/v1/contour/"))
                for _, path, _ in headers
            )
            assert not any(
                upstream_headers
            )  # Node key never goes to the private worker.
    finally:
        client.close()
        selected.close()


@pytest.mark.parametrize("frame_limit", [1, 4])
def test_contour_readiness_requires_auth_model_flags_and_idle_owner(frame_limit):
    body = {
        "model_id": "goodvision-contour-1",
        "ready": True,
        "gpu_ready": True,
        "accepting_sessions": True,
        "active_session": False,
        "limits": LIMITS | {"max_append_frames": frame_limit},
    }
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer " + KEY
        return httpx.Response(200, json=body)

    client = ContourSessionHTTPClient(
        "http://node", api_key=KEY, transport=httpx.MockTransport(handle)
    )
    try:
        assert client.ready_for_new_session()
        assert client.max_append_frames == frame_limit
        body["active_session"] = True
        assert client.gpu_ready() and not client.ready_for_new_session()
        for name in ("ready", "gpu_ready", "accepting_sessions"):
            body[name] = "true"
            assert not client.gpu_ready()
            body[name] = True
        body["model_id"] = "unknown"
        assert not client.gpu_ready()
        assert all(request.url.path == "/api/status" for request in calls)
    finally:
        client.close()


def test_contour_auth_failure_and_redirect_never_claim_ready():
    for status in (401, 302):
        calls = []

        def handle(request, calls=calls, status=status):
            calls.append(request)
            return httpx.Response(status, headers={"Location": "http://other.test"})

        client = ContourSessionHTTPClient(
            "http://node", api_key=KEY, transport=httpx.MockTransport(handle)
        )
        try:
            assert not client.gpu_ready()
            assert len(calls) == 1
        finally:
            client.close()
    client = ContourSessionHTTPClient("http://node")
    try:
        assert not client.gpu_ready()
        assert client._thread is None
    finally:
        client.close()


def test_contour_configuration_and_legacy_credentials_stay_separate():
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    locations = ServiceLocations(
        tracking_mode="server",
        tracking_server_protocol="contour",
        tracking_server_url="http://192.168.20.93:8093",
    )
    configured = locations.configure(config, LocalGPUDeployment())
    assert (
        configured.rich_labeling.endpoint
        == "http://192.168.20.93:8093/v1/contour/sessions"
    )
    client = create_tracking_client(configured.rich_labeling, api_key=KEY)
    legacy = create_tracking_client(
        replace(
            RichLabelSettings(), backend="sam31_stream", processing_method="sam_live"
        ),
        api_key=KEY,
    )
    try:
        assert isinstance(client, ContourSessionHTTPClient)
        assert not legacy._request_headers
        assert client._trust_env is False
    finally:
        client.close()
        legacy.close()
    multiple = replace(locations, tracking_server_capacity=2)
    for index in (1, 2):
        lane = multiple.configure(config, LocalGPUDeployment(), camera_index=index)
        assert lane.rich_labeling.endpoint.endswith(
            f"/v1/cameras/camera-{index:02d}/contour/sessions"
        )
        client = create_tracking_client(lane.rich_labeling, api_key=KEY)
        try:
            assert isinstance(client, ContourSessionHTTPClient)
            assert client._health.endswith(f"/api/cameras/camera-{index:02d}/status")
            assert client._request_headers["Authorization"] == "Bearer " + KEY
        finally:
            client.close()


def test_node_key_api_persists_without_echo_and_rejects_mixed_invalid_update(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    monkeypatch.setattr(
        "goodvision_harness.service.create_tracking_client",
        lambda *a, **k: SimpleNamespace(gpu_ready=lambda: False, close=lambda: None),
    )
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(
        config, events=replace(config.events, directory=tmp_path / "events")
    )
    store = DeploymentSettingsStore(tmp_path / "settings.json")
    store.save_api_key("cloud-key-kept", provider=config.model_api.provider)
    runtime = HarnessRuntime(config, deployment_store=store)
    with TestClient(main_app(lambda: runtime)) as client:
        response = client.put(
            "/v1/deployment/settings",
            json={
                "tracking_mode": "server",
                "tracking_server_protocol": "contour",
                "tracking_server_url": "http://192.168.20.93:8093",
                "tracking_server_capacity": 1,
                "tracking_api_key": KEY,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["tracking_key_present"] is True
        assert response.json()["tracking_server_protocol"] == "contour"
        assert KEY not in response.text
        assert KEY not in client.get("/v1/deployment/settings").text
        assert KEY not in store.path.read_text()
        assert store.load_api_key(provider="tracking") == KEY
        assert (
            store.load_api_key(provider=config.model_api.provider) == "cloud-key-kept"
        )
        revision = runtime._rich_config_revision
        response = client.put(
            "/v1/deployment/settings", json={"tracking_api_key": KEY + "x"}
        )
        assert response.status_code == 200
        assert runtime._rich_config_revision > revision
        original = store.path.read_bytes()
        invalid = client.put(
            "/v1/deployment/settings",
            json={
                "tracking_server_url": "http://different-node",
                "tracking_api_key": "invalid\nkey",
            },
        )
        assert invalid.status_code == 400
        assert store.path.read_bytes() == original
        assert store.load_api_key(provider="tracking") == KEY + "x"
    restarted = HarnessRuntime(config, deployment_store=store)
    try:
        assert restarted._tracking_api_key == KEY + "x"
        assert restarted.deployment_settings()["tracking_key_present"] is True
        result = restarted.update_deployment_settings({"clear_tracking_api_key": True})
        assert result["tracking_key_present"] is False
        assert store.load_api_key(provider="tracking") is None
        assert (
            store.load_api_key(provider=config.model_api.provider) == "cloud-key-kept"
        )
    finally:
        restarted.stop()
