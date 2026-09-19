"""Standalone node boundaries; fake transports never validate real GPU inference."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from types import MethodType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from workers.contour.app import create_app
from workers.contour.initialize import initialize
from workers.contour.settings import Settings

KEY = "contour-test-key-" + "x" * 32
AUTH = {"Authorization": "Bearer " + KEY}
TOKEN = "a" * 32
EPOCH = "b" * 32


def test_jetson_fixed_prompt_cache_isolated_and_rejects_other_prompts():
    from workers.contour.jetson_runtime import cache_person_text

    calls = []

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.value)

    def original(captions, device):
        calls.append((tuple(captions), device))
        return {
            key: Tensor(7)
            for key in ("language_features", "language_mask", "language_embeds")
        }

    encoder = SimpleNamespace(
        parameters=lambda: [], to=lambda device: calls.append(("unload", device))
    )
    backbone = SimpleNamespace(forward_text=original, language_backbone=encoder)
    runtime = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    cache_person_text(backbone, runtime)
    cache_person_text(backbone, runtime)  # Jetson + opt-in must not wrap twice.
    with pytest.raises(RuntimeError, match="fixed person"):
        backbone.forward_text(["car", "visual", "geometric"])
    assert not calls
    first = backbone.forward_text(["person", "visual", "geometric"])
    first["language_features"].value = 99
    second = backbone.forward_text(["person", "visual", "geometric"])
    assert second["language_features"].value == 7
    assert calls == [(("person", "visual", "geometric"), "cuda"), ("unload", "meta")]
    with pytest.raises(RuntimeError, match="fixed person"):
        backbone.forward_text(["person", "visual", "geometric"], additional_text=["x"])


def test_jetson_guard_refuses_unrelated_cgroups():
    from workers.contour.run_jetson import cgroup_path

    container_id = "a" * 64
    for group in ("/", "/system.slice", "/docker/other", "/../docker/" + container_id):
        with pytest.raises(RuntimeError):
            cgroup_path(group, container_id)
    assert (
        cgroup_path(f"/system.slice/docker-{container_id}.scope", container_id).name
        == f"docker-{container_id}.scope"
    )


def test_jetson_guard_refuses_unsafe_container_configuration():
    import copy

    from workers.contour.run_jetson import validate_container

    config = {
        "Id": "a" * 64,
        "Config": {
            "Labels": {
                "com.docker.compose.project": "goodvision-contour",
                "com.docker.compose.service": "engine",
            }
        },
        "HostConfig": {
            "RestartPolicy": {"Name": "no"},
            "Memory": 4500 * 1024 * 1024,
            "MemorySwap": 4500 * 1024 * 1024,
        },
    }
    validate_container(config)
    for property_name, replacement in (
        ("RestartPolicy", {"Name": "always"}),
        ("Memory", 0),
        ("MemorySwap", -1),
    ):
        changed = copy.deepcopy(config)
        changed["HostConfig"][property_name] = replacement
        with pytest.raises(RuntimeError):
            validate_container(changed)
    config["Config"]["Labels"]["com.docker.compose.service"] = "unrelated-service"
    with pytest.raises(RuntimeError):
        validate_container(config)


BASE = {
    "video_session_id": TOKEN,
    "stream_session_id": "customer-sam-camera",
    "predictor_epoch": EPOCH,
    "backend": "sam3.1_video",
    "memory_scope": "session",
    "state": "active",
}
LIMITS = {
    "max_frames": 16,
    "max_request_bytes": 33554432,
    "max_jpeg_bytes": 8388608,
    "max_total_jpeg_bytes": 25165824,
    "max_image_pixels": 8294400,
    "max_total_image_pixels": 67108864,
    "inference_max_side": 1280,
    "max_objects": 16,
    "request_timeout_seconds": 60.0,
    "max_inflight_requests": 1,
    "startup_frames": 1,
    "max_append_frames": 4,
    "hotstart_delay": 1,
    "max_pending_frames": 0,
    "history_frames": 32,
    "max_session_frames": 65536,
    "max_cumulative_native_ids": 2048,
    "idle_seconds": 60.0,
    "max_active_sessions": 1,
    "h264_transport": 1,
}


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=KEY, checkpoint_path=tmp_path / "contour.pt")


def health(**changes):
    return {
        "enabled": True,
        "ready": True,
        "gpu_ready": True,
        "backend": "sam3.1_video",
        "limits": LIMITS,
        "active_session": False,
        "sam3_commit": "private-source-version",
    } | changes


def test_framework_is_live_but_never_claims_model_readiness(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/api/status").status_code == 401
        value = client.get("/health/ready", headers=AUTH)
        assert value.status_code == 503
        assert value.json()["state"] == "waiting_for_model"
        assert value.json()["gpu_ready"] is False
        assert value.json()["checkpoint"]["present"] is False
        assert client.post(
            "/v1/contour/sessions",
            json={"stream_session_id": "camera"},
            headers=AUTH,
        ).json() == {"error": {"code": "model_not_installed"}}


def test_downloaded_file_does_not_attest_gpu_readiness(settings):
    settings.checkpoint_path.write_bytes(b"placeholder-only-for-status-test")
    with TestClient(create_app(settings)) as client:
        result = client.get("/health/ready", headers=AUTH)
        assert result.status_code == 503
        assert result.json()["state"] == "model_staged"
        assert result.json()["gpu_ready"] is False


@pytest.mark.parametrize("frame_limit", [1, 4])
def test_gateway_preserves_supported_node_frame_limits(settings, frame_limit):
    limits = LIMITS | {"max_append_frames": frame_limit}

    def engine(request):
        value = (
            health(limits=limits)
            if request.method == "GET"
            else BASE | {"limits": limits}
        )
        return httpx.Response(200, json=value)

    app = create_app(
        replace(settings, engine_url="http://engine"),
        transport=httpx.MockTransport(engine),
    )
    with TestClient(app) as client:
        status = client.get("/health/ready", headers=AUTH)
        assert status.status_code == 200
        assert status.json()["limits"]["max_append_frames"] == frame_limit
        created = client.post(
            "/v1/contour/sessions",
            json={"stream_session_id": BASE["stream_session_id"]},
            headers=AUTH,
        )
        assert created.status_code == 200
        assert created.json()["limits"]["max_append_frames"] == frame_limit


def test_no_key_is_fail_closed_and_initializer_never_replaces_key(settings, tmp_path):
    path = initialize(tmp_path)
    original = path.read_bytes()
    assert len(original) > 32
    assert initialize(tmp_path).read_bytes() == original
    with TestClient(create_app(replace(settings, api_key=""))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status", headers=AUTH).status_code == 503
    with pytest.raises(ValueError):
        replace(settings, api_key="short")


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_ready": False},
        {"ready": False},
        {"enabled": False},
        {"gpu_ready": "true"},
        {"backend": "wrong"},
        {"active_session": "no"},
    ],
)
def test_only_real_boolean_readiness_and_expected_backend_are_accepted(
    settings, changes
):
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json=health(**changes))
    )
    app = create_app(replace(settings, engine_url="http://engine"), transport=transport)
    with TestClient(app) as client:
        response = client.get("/health/ready", headers=AUTH)
        assert response.status_code == 503
        assert response.json()["gpu_ready"] is False


def test_pause_drains_existing_sessions_and_does_not_publish_internal_metadata(
    settings,
):
    seen = []

    def engine(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=health())
        return httpx.Response(
            200,
            json=BASE
            | {
                "state": "closed",
                "discarded_pending_frames": 0,
                "diagnostics": "SAM internal driver trace",
            },
        )

    app = create_app(
        replace(settings, engine_url="http://engine"),
        transport=httpx.MockTransport(engine),
    )
    with TestClient(app) as client:
        assert client.get("/health/ready", headers=AUTH).status_code == 200
        response = client.post(
            "/api/admission",
            json={"accepting_sessions": False},
            headers=AUTH,
        )
        assert response.status_code == 200
        assert client.get("/health/ready", headers=AUTH).status_code == 503
        assert (
            client.post(
                "/v1/contour/sessions",
                json={},
                headers=AUTH,
            ).json()["error"]["code"]
            == "new_sessions_paused"
        )
        deleted = client.delete(f"/v1/contour/sessions/{TOKEN}", headers=AUTH)
        assert deleted.status_code == 200
        assert deleted.json()["backend"] == "goodvision_contour"
        assert deleted.json()["stream_session_id"] == "customer-sam-camera"
        assert "diagnostics" not in deleted.json()
        assert seen[-1].headers.get("authorization") is None
        status = client.get("/api/status", headers=AUTH)
        assert status.json()["requests"]["completed"] == 1
        assert "sam3" not in status.text.lower()
        assert "private-source-version" not in status.text
        assert KEY not in status.text


@pytest.mark.parametrize("status", [400, 409, 429, 503, 504, 302, 500])
def test_internal_errors_and_headers_do_not_escape(settings, status):
    app = create_app(
        replace(settings, engine_url="http://engine"),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status,
                text="SAM 3.1 CUDA traceback /private/token",
                headers={
                    "Location": "http://private-node/sam",
                    "X-Engine": "SAM",
                },
            )
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/contour/sessions", json={}, headers=AUTH)
        assert response.status_code == (502 if status == 302 else status)
        assert "SAM" not in response.text
        assert "location" not in response.headers
        assert "x-engine" not in response.headers


def test_body_response_limits_and_unknown_routes(settings):
    seen = []

    def engine(request):
        seen.append(request)
        return httpx.Response(200, content=b"x" * 1024)

    app = create_app(
        replace(settings, engine_url="http://engine", max_response_bytes=100),
        transport=httpx.MockTransport(engine),
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/contour/sessions",
                content=b"x" * 4097,
                headers=AUTH | {"Content-Type": "application/json"},
            ).status_code
            == 413
        )
        assert not seen
        assert (
            client.post(
                "/v1/contour/sessions",
                content="hello",
                headers=AUTH,
            ).status_code
            == 415
        )
        assert client.get("/openapi.json", headers=AUTH).status_code == 404
        assert (
            client.get("/health/perception/sam_live", headers=AUTH).status_code == 404
        )
        assert (
            client.post("/v1/contour/sessions", json={}, headers=AUTH).status_code
            == 502
        )


def test_deadlines_and_parallel_admission(settings):
    entered = threading.Event()

    async def engine(_request):
        entered.set()
        await asyncio.sleep(2)
        return httpx.Response(200, json=BASE | {"limits": LIMITS})

    app = create_app(
        replace(settings, engine_url="http://engine", request_timeout=0.1),
        transport=httpx.MockTransport(engine),
    )
    with TestClient(app) as client:
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                client.post(
                    "/v1/contour/sessions",
                    json={},
                    headers=AUTH,
                )
            )
        )
        thread.start()
        assert entered.wait(2)
        assert (
            client.post("/v1/contour/sessions", json={}, headers=AUTH).status_code
            == 429
        )
        thread.join(3)
        assert result[0].status_code == 504
        assert (
            client.get("/api/status", headers=AUTH).json()["requests"]["in_flight"] == 0
        )


def test_real_worker_protocol_through_gateway_preserves_contours_and_retry_identity(
    settings,
):
    # Exercise the current worker's real serializer/route handlers, with an
    # injected inference double only. No weights, CUDA or network are involved.
    from test_perception_worker_routes import causal_engine
    from test_sam31_continuous_worker import payload

    from workers.sam_scene.app import decode_jpeg
    from workers.sam_scene.continuous import register_session_routes
    from workers.sam_scene.sam_live import SamLiveContinuation

    worker = FastAPI()
    selected = causal_engine()
    selected.backend.route_metadata = MethodType(
        SamLiveContinuation.route_metadata, selected.backend
    )
    register_session_routes(
        worker,
        selected,
        threading.Lock(),
        decode_jpeg,
        api_prefix="/v1/perception/sam_live",
        legacy=False,
    )
    node = create_app(
        replace(settings, engine_url="http://engine"),
        transport=httpx.ASGITransport(app=worker),
    )
    with TestClient(node) as client:
        created = client.post(
            "/v1/contour/sessions",
            json={"stream_session_id": "camera"},
            headers=AUTH,
        )
        assert created.status_code == 200, created.text
        path = "/v1/contour/sessions/" + created.json()["video_session_id"]
        body = json.loads(payload(1))
        first = client.post(path + "/append", json=body, headers=AUTH)
        assert first.status_code == 200, first.text
        obj = first.json()["frames"][0]["objects"][0]
        assert obj["track_id"].startswith("ct-")
        assert obj["attributes"]["mask_contours"]["polygons"]
        assert obj["attributes"]["native_track"]["backend"] == "goodvision_contour"
        assert "sam" not in first.text.lower()
        retry = client.post(path + "/append", json=body, headers=AUTH)
        assert retry.json() == first.json()
        client.post("/api/admission", json={"accepting_sessions": False}, headers=AUTH)
        final = client.post(
            path + "/append",
            json=json.loads(payload(0, request_id="end", final=True)),
            headers=AUTH,
        )
        assert final.status_code == 200, final.text
        assert final.json()["state"] == "closed"
        assert client.delete(path, headers=AUTH).status_code == 200
