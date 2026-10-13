"""Local switches route independently and preserve the original deployment."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.service import HarnessRuntime, create_app


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "true")
    monkeypatch.setenv("LOCAL_GPU_COUNT", "2")
    for name in (
        "LOCAL_INFERENCE_ENABLED",
        "LOCAL_TRACKING_ENABLED",
        "LOCAL_INFERENCE_GPU_ID",
        "LOCAL_TRACKING_GPU_ID",
        "LOCAL_INFERENCE_API_KEY",
        "HARNESS_INFERENCE_PROFILE",
        "INFERENCE_EXECUTION_MODE", "TRACKING_EXECUTION_MODE",
        "INFERENCE_SERVER_URL", "INFERENCE_SERVER_MODEL",
        "TRACKING_SERVER_URL", "TRACKING_SERVER_CAPACITY",
        "HARNESS_FLEET_CAPACITY", "HARNESS_FLEET_CAMERA_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INFERENCE_API_KEY", "cloud-secret")
    monkeypatch.setattr(
        "goodvision_harness.service.create_tracking_client",
        lambda rich: SimpleNamespace(gpu_ready=lambda: False, close=lambda: None),
    )
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config = replace(
        config,
        aiclub=ModelAPISettings(provider="openrouter"),
        rich_labeling=replace(
            config.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            endpoint="http://remote-tracker:8092/v1/video/sessions",
            backend="sam31_stream",
            processing_method=None,
        ),
        runtime=replace(config.runtime, scene_mode="active"),
        events=replace(config.events, directory=tmp_path / "events"),
    )
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    store.save_api_key("saved-cloud-secret", provider="openrouter")
    runtimes = []

    def make():
        runtime = HarnessRuntime(config, deployment_store=store)
        runtimes.append(runtime)
        return runtime

    yield SimpleNamespace(make=make, config=config, store=store)
    for runtime in runtimes:
        runtime.stop()


@pytest.mark.parametrize(
    "inference,tracking", [(False, False), (True, False), (False, True), (True, True)]
)
def test_independent_local_choices_and_restart(rig, inference, tracking):
    runtime = rig.make()
    payload = {"local_inference_enabled": inference, "local_tracking_enabled": tracking}
    result = runtime.update_deployment_settings(payload)
    assert result.items() >= payload.items()
    assert result["local_gpu"]["inference_device"] == 0
    assert result["local_gpu"]["tracking_device"] == 1
    assert (runtime.config.model_api.provider == "local") is inference
    assert ("127.0.0.1" in runtime.config.rich_labeling.endpoint) is tracking
    assert runtime._aiclub_api_key == (None if inference else "saved-cloud-secret")
    if inference:
        assert not runtime.config.model_api.extra_headers
        assert "provider" not in runtime.config.inference.request_parameters
        assert "chat_template_kwargs" not in runtime.config.inference.request_parameters
    restarted = rig.make()
    assert restarted.config.model_api == runtime.config.model_api
    assert restarted.config.rich_labeling == runtime.config.rich_labeling
    assert restarted.deployment_settings()["local_tracking_enabled"] is tracking


def test_remote_routes_and_key_restored_after_local_roundtrip(rig):
    runtime = rig.make()
    original = runtime.config
    runtime.update_deployment_settings(
        {"local_inference_enabled": True, "local_tracking_enabled": True}
    )
    runtime.update_deployment_settings(
        {"local_inference_enabled": False, "local_tracking_enabled": False}
    )
    assert runtime.config == original
    assert runtime._aiclub_api_key == "saved-cloud-secret"


def test_tracking_only_deployment_keeps_cloud_vlm_and_rejects_unprovisioned_vlm(
    rig, monkeypatch,
):
    monkeypatch.setenv("LOCAL_INFERENCE_AVAILABLE", "false")
    monkeypatch.setenv("LOCAL_TRACKING_AVAILABLE", "true")
    monkeypatch.setenv("LOCAL_TRACKING_GPU_ID", "0")
    monkeypatch.setenv("TRACKING_EXECUTION_MODE", "local")
    endpoint = "http://local-tracking:8092/v1/perception/sam_live/sessions"
    monkeypatch.setenv("LOCAL_TRACKING_ENDPOINT", endpoint)
    runtime = rig.make()
    assert runtime.config_error is None
    assert runtime.config.rich_labeling.endpoint == endpoint
    assert runtime.config.rich_labeling.processing_method == "sam_live"
    assert runtime.config.model_api.provider == "openrouter"
    assert runtime._aiclub_api_key == "saved-cloud-secret"
    status = runtime.deployment_settings()["local_gpu"]
    assert status["available"] is True
    assert status["tracking_available"] is True
    assert status["inference_available"] is False
    assert status["tracking_device"] == 0
    assert status["inference_device"] is None
    assert status["assignment"] == "tracking_only"
    before = runtime.config
    with pytest.raises(ValueError, match="本地大模型服务未部署"):
        runtime.update_deployment_settings({"inference_mode": "local"})
    assert runtime.config == before
    runtime.update_deployment_settings({"tracking_mode": "legacy"})
    assert runtime.config.rich_labeling.endpoint == rig.config.rich_labeling.endpoint
    runtime.update_deployment_settings({"tracking_mode": "local"})
    assert runtime.config.rich_labeling.endpoint == endpoint
    assert rig.make().config.rich_labeling.endpoint == endpoint


@pytest.mark.parametrize("bad", [None, "true", 1, [], {}])
def test_invalid_switch_is_atomic(rig, bad):
    runtime = rig.make()
    runtime.update_deployment_settings({"dry_run": True})
    before = rig.store.path.read_bytes()
    with pytest.raises(ValueError, match="boolean"):
        runtime.update_deployment_settings(
            {"local_inference_enabled": bad, "dry_run": False}
        )
    assert rig.store.path.read_bytes() == before
    assert runtime.dry_run
    assert runtime.config.model_api.provider == "openrouter"


def test_unprovisioned_nvidia_fails_without_switching_or_persisting(rig, monkeypatch):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    runtime = rig.make()
    original = runtime.config
    with TestClient(create_app(lambda: runtime)) as client:
        reply = client.put(
            "/v1/deployment/settings", json={"local_tracking_enabled": True}
        )
        assert reply.status_code == 400
        assert "GPU" in reply.json()["detail"]
    assert not rig.store.path.exists()
    assert runtime.config == original


def test_switch_locked_while_running_or_draining(rig):
    runtime = rig.make()
    for state in ("starting", "running", "draining", "stopping"):
        runtime._state["pipeline_state"] = state
        with pytest.raises(RuntimeError, match="stop"):
            runtime.update_deployment_settings({"local_inference_enabled": True})
    runtime._state["pipeline_state"] = "idle"
    assert not rig.store.path.exists()


def test_console_sampling_updates_runtime_persists_and_rejects_active_change(rig):
    runtime = rig.make()
    original_display, original_face = runtime.config.display, runtime.config.face
    with TestClient(create_app(lambda: runtime)) as client:
        reply = client.put("/v1/deployment/settings", json={"tracking_sample_fps": 1.5})
        assert reply.status_code == 200, reply.text
        assert reply.json()["tracking_sample_fps"] == 1.5
        assert runtime.config.rich_labeling.max_fps == 1.5
        assert runtime.config.display == original_display
        assert runtime.config.face == original_face
        before = rig.store.path.read_bytes()
        rejected = client.put("/v1/deployment/settings", json={"tracking_sample_fps": True, "dry_run": True})
        assert rejected.status_code == 400
        assert rig.store.path.read_bytes() == before
        runtime._state["pipeline_state"] = "running"
        try:
            busy = client.put("/v1/deployment/settings", json={"tracking_sample_fps": 3})
            assert busy.status_code == 409
            assert runtime.config.rich_labeling.max_fps == 1.5
        finally:
            runtime._state["pipeline_state"] = "idle"
    restarted = rig.make()
    assert restarted.config.rich_labeling.max_fps == 1.5
    assert restarted.deployment_settings()["tracking_sample_fps"] == 1.5


def test_local_start_needs_no_cloud_key(rig, monkeypatch):
    runtime = rig.make()
    runtime.update_deployment_settings({"local_inference_enabled": True})
    runtime.config = replace(
        runtime.config, runtime=replace(runtime.config.runtime, scene_mode="disabled")
    )
    runtime._scene_mode = "disabled"
    runtime._scene_enabled = False
    monkeypatch.setattr(
        "goodvision_harness.pipeline.StreamingSession.start", lambda _: None
    )
    with TestClient(create_app(lambda: runtime)) as client:
        reply = client.post(
            "/v1/sessions/start", json={"features": ["smoking"], "dry_run": False}
        )
        assert reply.status_code == 200, reply.text
        assert client.get("/v1/ui/info").json()["api_key_required"] is False


def test_gpu_host_assignment(monkeypatch):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "true")
    monkeypatch.setenv("LOCAL_GPU_COUNT", "1")
    assert LocalGPUDeployment.from_environment().tracking_device == 0
    monkeypatch.setenv("LOCAL_GPU_COUNT", "2")
    assert LocalGPUDeployment.from_environment().tracking_device == 1
    monkeypatch.setenv("LOCAL_TRACKING_GPU_ID", "0")
    with pytest.raises(ValueError, match="separated"):
        LocalGPUDeployment.from_environment()


def test_full_console_save_does_not_revert_a_saved_remote_method(rig):
    runtime = rig.make()
    runtime.update_deployment_settings({"processing_method": "sam_live"})
    revision = runtime._rich_config_revision
    runtime.update_deployment_settings(
        {
            "local_inference_enabled": False,
            "local_tracking_enabled": False,
            "processing_method": "sam_live",
            "dry_run": True,
        }
    )
    assert runtime.config.rich_labeling.processing_method == "sam_live"
    assert runtime._remote_config.rich_labeling.processing_method == "sam_live"
    assert runtime._rich_config_revision == revision
    runtime.update_deployment_settings({"local_inference_enabled": True})
    assert runtime.config.rich_labeling.processing_method == "sam_live"
    assert runtime._remote_config.rich_labeling.processing_method == "sam_live"
    assert runtime._rich_config_revision == revision


def test_full_console_save_detaches_only_once_when_method_changes(rig, monkeypatch):
    runtime = rig.make()
    calls = []
    original = runtime._replace_processing_method

    def detach(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_replace_processing_method", detach)
    runtime.update_deployment_settings(
        {
            "local_inference_enabled": True,
            "local_tracking_enabled": False,
            "processing_method": "sam_live",
        }
    )
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["stream", "camera"])
def test_secondary_runtime_requires_local_tracking_for_all_inputs(
    rig, monkeypatch, mode,
):
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "2")
    runtime = rig.make()
    with pytest.raises(RuntimeError, match="人员追踪模块"):
        if mode == "stream":
            runtime.start_stream("rtsp://camera.example/live")
        else:
            runtime.start_camera(owner="test")


def test_secondary_never_autostarts_without_primary_admission(rig, monkeypatch):
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "2")
    runtime = rig.make()
    runtime.update_deployment_settings({
        "autostart": True, "local_tracking_enabled": True,
    })
    starts = []
    connects = []
    monkeypatch.setattr(runtime, "start_stream", lambda url: starts.append(url))
    monkeypatch.setattr(runtime, "connect_stream", lambda url: connects.append(url))
    monkeypatch.setattr(runtime, "configured_stream_url", lambda: "rtsp://camera/live")
    runtime.start()
    assert starts == []
    assert connects == []


@pytest.mark.parametrize("peer_state", ["running", "draining", "stopping", "unknown"])
def test_fleet_busy_peer_blocks_global_changes_through_original_api(
    rig, monkeypatch, peer_state,
):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "3")
    runtime = rig.make()
    runtime.update_deployment_settings({"local_tracking_enabled": True})
    app = create_app(lambda: runtime)
    app.state.fleet._peer_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"pipeline_state": peer_state},
        )),
    )
    before = rig.store.path.read_bytes()
    with TestClient(app) as client:
        for payload in (
            {"local_tracking_enabled": False},
            {"local_inference_enabled": True},
            {"dry_run": not runtime.dry_run},
            {"api_key": "replacement-test-key"},
        ):
            reply = client.put("/v1/deployment/settings", json=payload)
            assert reply.status_code == 409, reply.text
        assert client.post("/v1/deployment/routes/check").status_code == 409
        assert client.post(
            "/v1/deployment/routes/select", json={"route_id": "unused"},
        ).status_code == 409
        assert rig.store.path.read_bytes() == before
        # Independent camera address edits do not alter the fleet mode.
        assert client.put("/v1/deployment/settings", json={
            "stream_url": "rtsp://camera.example/live",
            "local_tracking_enabled": True,
            "local_inference_enabled": False,
        }).status_code == 200


def test_fleet_remote_switch_after_idle_disables_extra_routes(rig, monkeypatch):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "16")
    # This scenario requests tracking in remote mode too. Face/VLM-only
    # deployments instead use the provisioned application-worker capacity.
    runtime = HarnessRuntime(replace(
        rig.config, rich_labeling=replace(rig.config.rich_labeling, tracking_enabled=True),
    ), deployment_store=rig.store)
    runtime.update_deployment_settings({"local_tracking_enabled": True})
    app = create_app(lambda: runtime)
    app.state.fleet._peer_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"pipeline_state": "idle"},
        )),
    )
    with TestClient(app) as client:
        descriptor = client.get("/v1/fleet").json()
        script = client.get("/assets/fleet_console.js")
        assert script.status_code == 200
        assert "FleetConsole" in script.text
        assert descriptor["multi_camera_allowed"] is True
        assert len(descriptor["cameras"]) == 16
        assert runtime._local_inference_enabled is False
        assert client.put("/v1/deployment/settings", json={
            "local_tracking_enabled": False,
        }).status_code == 200
        descriptor = client.get("/v1/fleet").json()
        assert descriptor["multi_camera_allowed"] is False
        assert sum(camera["enabled"] for camera in descriptor["cameras"]) == 1
        assert client.post(
            descriptor["cameras"][1]["start_url"], json={},
        ).status_code == 409
        status = client.get(descriptor["cameras"][0]["status_url"])
        assert isinstance(status.json()["configured_stream"], bool)
        assert status.json()["camera_layout"] == "mono"


@pytest.mark.parametrize("local_inference", [False, True])
def test_fleet_start_synchronizes_real_peer_runtime_before_start(
    rig, monkeypatch, local_inference,
):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
    primary = rig.make()
    primary.update_deployment_settings({
        "local_tracking_enabled": True,
        "local_inference_enabled": local_inference,
    })
    app = create_app(lambda: primary)
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "2")
    peer_store = DeploymentSettingsStore(rig.store.path.with_name("peer.json"))
    peer = HarnessRuntime(rig.config, deployment_store=peer_store)
    peer.update_deployment_settings({"stream_url": "rtsp://peer.example/live"})
    peer_app = create_app(lambda: peer)
    peer_app.state.runtime = peer
    starts = []

    def start(stream_url, **kwargs):
        starts.append((
            peer._local_tracking_enabled, peer._local_inference_enabled,
            peer._aiclub_api_key, peer.configured_stream_url(),
        ))
        peer._state["pipeline_state"] = "running"
        return {"pipeline_state": "starting"}

    monkeypatch.setattr(peer, "start_stream", start)
    monkeypatch.setattr(peer, "start_connected_analysis", lambda: start(
        peer.configured_stream_url(),
    ))
    monkeypatch.setattr(peer, "prepare_analysis", lambda: {"prepared": True})
    original_status = peer.status
    monkeypatch.setattr(peer, "status", lambda **kwargs: {
        **original_status(**kwargs), "connected": True, "connection_state": "connected",
    })
    app.state.fleet._peer_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=peer_app),
    )
    try:
        with TestClient(app) as client:
            response = client.post("/v1/fleet/analysis/start", json={})
            assert response.status_code == 200, response.text
            assert starts == [(
                True, local_inference,
                None if local_inference else "saved-cloud-secret",
                "rtsp://peer.example/live",
            )]
            assert "secret" not in response.text
            assert peer_store.load_api_key(provider="local") is None
    finally:
        peer.stop()


def test_slow_start_is_visible_cancellable_and_blocks_configuration(rig, monkeypatch):
    runtime = rig.make()
    # Readiness gates fused sessions; independent VLM sessions deliberately
    # bypass this optional tracker gate (covered in test_vlm_independent_service).
    runtime.config = replace(runtime.config, runtime=replace(
        runtime.config.runtime, scene_independent_vlm=False,
    ))
    entered, release = Event(), Event()

    def wait_for_readiness():
        entered.set()
        assert release.wait(10)
        return None

    monkeypatch.setattr(runtime, "_require_processing_ready", wait_for_readiness)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runtime.start_camera, owner="camera-tab")
        try:
            assert entered.wait(5)
            status = runtime.status()
            assert status["session_start_pending"] is True
            assert status["pipeline_state"] == "starting"
            with pytest.raises(RuntimeError, match="active session"):
                runtime.update_deployment_settings({"local_tracking_enabled": True})
            with pytest.raises(RuntimeError, match="already starting"):
                runtime.start_camera(owner="another-tab")
            assert runtime.stop_session(camera_owner="unrelated-tab") is False
            assert runtime.stop_session(camera_owner="camera-tab") is True
            assert runtime.status()["pipeline_state"] == "stopping"
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            future.result(timeout=5)
    assert runtime.status()["session_start_pending"] is False
    assert runtime._session is None


@pytest.mark.parametrize("capacity", [1, 3, 12, 16])
def test_server_tracking_without_main_gpu_survives_restart(rig, monkeypatch, capacity):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", str(capacity))
    runtime = rig.make()
    result = runtime.update_deployment_settings({
        "tracking_mode": "server",
        "tracking_server_url": "http://sam-gpu.example:8092/",
        "tracking_server_capacity": capacity,
    })
    assert result["tracking_server_url"] == "http://sam-gpu.example:8092"
    assert result["local_tracking_enabled"] is False
    assert result["local_gpu"]["available"] is False
    assert runtime.tracking_camera_capacity() == capacity
    assert runtime.config.model_api.provider == "openrouter"
    assert runtime._aiclub_api_key == "saved-cloud-secret"
    path = ("/v1/cameras/camera-01/perception/sam_live/sessions"
            if capacity > 1 else "/v1/video/sessions")
    assert runtime.config.rich_labeling.endpoint == "http://sam-gpu.example:8092" + path
    restarted = rig.make()
    assert restarted.config == runtime.config
    assert (
        restarted.service_execution_settings() == runtime.service_execution_settings()
    )


def test_server_inference_isolated_from_cloud_auth_and_route(rig, monkeypatch):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    runtime = rig.make()
    original = runtime.config
    result = runtime.update_deployment_settings({
        "inference_mode": "server",
        "inference_server_url": "http://qwen-gpu.example:8000",
        "inference_server_model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    })
    assert result["inference_server_url"] == "http://qwen-gpu.example:8000"
    assert result["inference_server_model"] == runtime.config.inference.model
    assert runtime.config.model_api.provider == "local"
    assert runtime.config.model_api.endpoint == (
        "http://qwen-gpu.example:8000/v1/chat/completions"
    )
    assert runtime._aiclub_api_key is None
    assert not runtime._route_pool_enabled
    assert not runtime.config.model_api.extra_headers
    assert "provider" not in runtime.config.inference.request_parameters
    assert result["api_key_required"] is False
    assert runtime.config.rich_labeling == original.rich_labeling
    restarted = rig.make()
    assert restarted.config == runtime.config
    runtime.update_deployment_settings({"inference_mode": "cloud"})
    assert runtime.config == original
    assert runtime._aiclub_api_key == "saved-cloud-secret"


@pytest.mark.parametrize("change", [
    {"tracking_sample_fps": 1.5},
    {"tracking_mode": "legacy"},
    {"local_tracking_enabled": False},
    {"tracking_server_url": "http://replacement:8092"},
    {"tracking_server_capacity": 16},
    {"inference_mode": "server", "inference_server_url": "http://inference:8000"},
    {"inference_server_model": "alternate-model"},
])
def test_busy_server_camera_blocks_global_topology_changes(rig, monkeypatch, change):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "3")
    runtime = rig.make()
    runtime.update_deployment_settings({
        "tracking_mode": "server", "tracking_server_url": "http://tracking:8092",
        "tracking_server_capacity": 3,
    })
    app = create_app(lambda: runtime)
    app.state.fleet._peer_client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"pipeline_state": "running"}),
    ))
    before = rig.store.path.read_bytes()
    with TestClient(app) as client:
        reply = client.put("/v1/deployment/settings", json=change)
        assert reply.status_code == 409, reply.text
    assert rig.store.path.read_bytes() == before


def test_server_settings_validation_does_not_partially_save(rig, monkeypatch):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    runtime = rig.make()
    runtime.update_deployment_settings({"dry_run": True})
    before = rig.store.path.read_bytes()
    with pytest.raises(ValueError):
        runtime.update_deployment_settings({
            "dry_run": False, "tracking_mode": "server",
            "tracking_server_url": "http://user:secret@tracking:8092",
        })
    assert rig.store.path.read_bytes() == before
    assert runtime.dry_run
    assert runtime.service_execution_settings()["tracking_mode"] == "legacy"


def test_server_peer_sync_uses_own_camera_lane_without_main_gpu(rig, monkeypatch):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "16")
    primary = rig.make()
    primary.update_deployment_settings({
        "tracking_mode": "server", "tracking_server_url": "http://sam-gpu:8092",
        "tracking_server_capacity": 16,
    })
    app = create_app(lambda: primary)
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "16")
    peer_store = DeploymentSettingsStore(rig.store.path.with_name("peer16.json"))
    peer = HarnessRuntime(rig.config, deployment_store=peer_store)
    peer.update_deployment_settings({"stream_url": "rtsp://camera16.example/live"})
    peer_app = create_app(lambda: peer)
    peer_app.state.runtime = peer
    starts = []

    def start(stream_url, **kwargs):
        starts.append((peer.config.rich_labeling.endpoint, peer._aiclub_api_key))
        peer._state["pipeline_state"] = "running"
        return {"pipeline_state": "starting"}

    monkeypatch.setattr(peer, "start_stream", start)
    monkeypatch.setattr(peer, "start_connected_analysis", lambda: start(
        peer.configured_stream_url(),
    ))
    monkeypatch.setattr(peer, "prepare_analysis", lambda: {"prepared": True})
    original_status = peer.status
    monkeypatch.setattr(peer, "status", lambda **kwargs: {
        **original_status(**kwargs), "connected": True, "connection_state": "connected",
    })
    transport = httpx.ASGITransport(app=peer_app)

    async def dispatch(request):
        if request.url.host == "harness-16":
            return await transport.handle_async_request(request)
        return httpx.Response(200, json={
            "pipeline_state": "idle", "connected": False,
            "connection_state": "disconnected", "analysis_state": "idle",
        })

    app.state.fleet._peer_client = httpx.AsyncClient(
        transport=httpx.MockTransport(dispatch),
    )
    try:
        with TestClient(app) as client:
            result = client.post("/v1/fleet/analysis/start", json={})
            assert result.status_code == 200, result.text
            assert starts == [(
                "http://sam-gpu:8092/v1/cameras/camera-16/perception/sam_live/sessions",
                "saved-cloud-secret",
            )]
            assert (
                peer.service_execution_settings()
                == primary.service_execution_settings()
            )
            assert peer.tracking_camera_capacity() == 16
    finally:
        peer.stop()


def test_server_capacity_blocks_direct_secondary_start(rig, monkeypatch):
    monkeypatch.setenv("LOCAL_GPU_AVAILABLE", "false")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "16")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "4")
    runtime = rig.make()
    runtime.update_deployment_settings({
        "tracking_mode": "server", "tracking_server_url": "http://sam-gpu:8092",
        "tracking_server_capacity": 3,
    })
    with pytest.raises(RuntimeError, match="配置容量"):
        runtime.start_camera(owner="test")
