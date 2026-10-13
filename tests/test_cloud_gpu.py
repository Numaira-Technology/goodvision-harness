"""Cloud rental controls, exact hardware requests and isolated GPU routing."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettings
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.service_locations import ServiceLocations
from tools.cloud_gpu_controller import CloudController, create_app
from tools.cloud_gpu_settings import CloudGPUSettings
from tools.start_cloud_gpu import fleet_overlay
from workers.sam_scene.cloud_pool import assignment, worker_plan
from workers.sam_scene.cloud_pool import create_app as pool_app


def test_cloud_has_no_fixed_camera_limit_and_uses_distinct_routes(monkeypatch):
    monkeypatch.setenv("CLOUD_TRACKING_URL", "http://cloud:8092")
    base = load_config(Path(__file__).parents[1] / "config.yaml")
    locations = ServiceLocations(tracking_mode="cloud", tracking_sample_fps=3)
    gpu = LocalGPUDeployment(available=False)
    for count in (1, 5, 37, 128, 1024):
        assert locations.tracking_capacity(count, gpu) == count
        config = locations.configure(base, gpu, camera_index=count)
        assert (
            config.rich_labeling.endpoint
            == f"http://cloud:8092/v1/cameras/camera-{count:02d}/perception/sam_live/sessions"
        )
        assert config.rich_labeling.max_fps == 3
        assert config.rich_labeling.processing_method == "sam_live"


def test_saved_legacy_cloud_deployment_migrates_only_on_managed_cloud(monkeypatch):
    saved = DeploymentSettings("", False, False, tracking_mode="legacy")
    monkeypatch.delenv("CLOUD_GPU_CONTROLLER_URL", raising=False)
    assert (
        ServiceLocations.resolve(
            saved, local_inference=False, local_tracking=False
        ).tracking_mode
        == "legacy"
    )
    monkeypatch.setenv("CLOUD_GPU_CONTROLLER_URL", "http://sam-remote:8093")
    assert (
        ServiceLocations.resolve(
            saved, local_inference=False, local_tracking=False
        ).tracking_mode
        == "cloud"
    )


def test_exact_five_a100_request_has_no_single_gpu_fallback():
    config = CloudGPUSettings("A100", 5, 37, 80)
    assert "num_gpus=5 " in config.query()
    assert "gpu_name in [A100_" in config.query()
    assert "disk_space>=80" in config.query()
    assert config.arguments() == [
        "--gpu-name",
        "A100",
        "--gpu-count",
        "5",
        "--streams",
        "37",
        "--disk-gb",
        "80",
    ]
    assert "gpu_name=A100_SXM4 " in CloudGPUSettings("A100 SXM4", 5).query()


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_name": "A100 num_gpus=8"},
        {"gpu_name": "A100;rm"},
        {"gpu_count": 0},
        {"gpu_count": True},
        {"gpu_count": 1.5},
        {"streams": 0},
        {"disk_gb": "80"},
    ],
)
def test_invalid_rental_settings_rejected(change):
    with pytest.raises(ValueError):
        CloudGPUSettings(**{**CloudGPUSettings().public(), **change})


def test_video_assignment_is_stable_and_uses_all_five_gpus():
    plan = worker_plan(5, 37)
    assert [p["lanes"] for p in plan] == [8, 8, 7, 7, 7]
    assert [p["port"] for p in plan] == list(range(8100, 8105))
    pairs = [assignment(i, 5, 37) for i in range(1, 38)]
    assert len(set(pairs)) == 37
    assert {p[0] for p in pairs} == set(range(5))
    assert pairs[0] == (0, 1) and pairs[5] == (0, 2) and pairs[36] == (1, 8)
    with pytest.raises(HTTPException):
        assignment(38, 5, 37)


class Stream(httpx.AsyncByteStream):
    def __init__(self, value):
        self.value = value

    async def __aiter__(self):
        yield json.dumps(self.value).encode()


def test_cloud_proxy_preserves_session_body_and_routes_to_fixed_gpu_and_lane():
    requests = []

    async def handler(request):
        body = await request.aread()
        requests.append((request.url.port, request.url.path, body))
        return httpx.Response(
            200,
            stream=Stream({"ready": True, "gpu_ready": True, "enabled": True}),
            headers={"Content-Type": "application/json"},
        )

    app = pool_app(5, 37, launch_workers=False, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        for camera in (1, 6, 37):
            path = (
                f"/v1/cameras/camera-{camera:02d}/perception/sam_live/"
                "sessions/session-a/append"
            )
            assert client.post(path, content=b"exact-frame-payload").status_code == 200
            gpu, lane = assignment(camera, 5, 37)
            assert requests[-1] == (
                8100 + gpu,
                f"/v1/cameras/camera-{lane:02d}/perception/sam_live/sessions/session-a/append",
                b"exact-frame-payload",
            )
        assert (
            client.get("/health/cameras/camera-37/perception/sam_live").status_code
            == 200
        )
        assert requests[-1][0:2] == (
            8101,
            "/health/cameras/camera-08/perception/sam_live",
        )
        assert (
            client.delete(
                "/v1/cameras/camera-37/perception/sam_live/sessions/session-a"
            ).status_code
            == 200
        )
        assert requests[-1][0] == 8101
        for camera in ("camera-00", "camera-01x", "camera-038", "camera-38"):
            assert (
                client.post(
                    f"/v1/cameras/{camera}/perception/sam_live/sessions"
                ).status_code
                == 404
            )
        health = client.get("/health/perception/sam_live").json()
        assert health["ready"] and health["model_instances"] == 5
        assert health["max_gpu_operations"] == 5 and health["max_active_sessions"] == 37


def test_pool_readiness_requires_every_gpu_and_never_migrates_failed_sessions():
    async def handler(request):
        if request.url.port == 8103:
            raise httpx.ConnectError("unreachable")
        return httpx.Response(
            200, stream=Stream({"ready": True, "gpu_ready": True, "enabled": True})
        )

    with TestClient(
        pool_app(5, 37, launch_workers=False, transport=httpx.MockTransport(handler))
    ) as client:
        assert client.get("/health/perception/sam_live").status_code == 503
        assert (
            client.post(
                "/v1/cameras/camera-04/perception/sam_live/sessions"
            ).status_code
            == 503
        )
        assert (
            client.post(
                "/v1/cameras/camera-05/perception/sam_live/sessions"
            ).status_code
            == 200
        )


@pytest.mark.parametrize("count,visible", [(1, ""), (5, ""), (5, "2,3,5,7,9")])
def test_pool_launches_and_closes_one_process_per_visible_gpu(
    monkeypatch, count, visible
):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count)),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    processes = [Mock() for _ in range(count)]
    for process in processes:
        process.poll.return_value = None
    spawn = Mock(side_effect=processes)
    monkeypatch.setattr("workers.sam_scene.cloud_pool.subprocess.Popen", spawn)
    with TestClient(pool_app(count, 37)):
        assert spawn.call_count == count
        environments = [call.kwargs["env"] for call in spawn.call_args_list]
        assert [env["CUDA_VISIBLE_DEVICES"] for env in environments] == (
            visible.split(",") if visible else list(map(str, range(count)))
        )
        assert all(int(env["SAM31_LIVE_CAMERAS"]) >= 2 for env in environments)
    for process in processes:
        process.terminate.assert_called_once()
        process.wait.assert_called_once()


def test_busy_gpu_does_not_serialize_requests_to_another_gpu():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            await request.aread()
            if request.url.port == 8100:
                entered.set()
                await release.wait()
            return httpx.Response(200, stream=Stream({"gpu": request.url.port - 8100}))

        app = pool_app(
            5, 37, launch_workers=False, transport=httpx.MockTransport(handler)
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://pool"
            ) as client:
                slow = asyncio.create_task(
                    client.post("/v1/cameras/camera-01/perception/sam_live/sessions")
                )
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    response = await asyncio.wait_for(
                        client.post(
                            "/v1/cameras/camera-02/perception/sam_live/sessions"
                        ),
                        1,
                    )
                    assert response.json()["gpu"] == 1
                    assert not slow.done()
                finally:
                    release.set()
                    await slow

    asyncio.run(run())


def test_controller_save_start_stop_persistence_and_duplicate_protection(
    tmp_path, monkeypatch
):
    process = Mock()
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr("tools.cloud_gpu_controller.subprocess.Popen", spawn)
    app = create_app(tmp_path, 37)
    with TestClient(app) as client:
        assert (
            client.put(
                "/settings", json=CloudGPUSettings("A100", 5, 37, 80).public()
            ).status_code
            == 200
        )
        spawn.assert_not_called()
        assert CloudController(tmp_path, 37).settings().gpu_count == 5
        assert client.post("/start").json()["state"] == "starting"
        argv = spawn.call_args.args[0]
        assert argv[3:11] == [
            "--gpu-name",
            "A100",
            "--gpu-count",
            "5",
            "--streams",
            "37",
            "--disk-gb",
            "80",
        ]
        assert "shell" not in spawn.call_args.kwargs
        assert client.post("/start").status_code == 409
        assert (
            client.put(
                "/settings", json=CloudGPUSettings("A100", 8, 37).public()
            ).status_code
            == 409
        )
        spawn.assert_called_once()

        def release(**_):
            process.poll.return_value = 130
            app.state.manager.status_path.write_text(
                json.dumps({"state": "stopped", "cleanup_confirmed": True})
            )

        process.wait.side_effect = release
        assert client.post("/stop").json()["state"] == "stopped"
        process.send_signal.assert_called_once()


def test_controller_refuses_mismatched_capacity_and_uncertain_cleanup(
    tmp_path, monkeypatch
):
    manager = CloudController(tmp_path, 37)
    with pytest.raises(ValueError):
        manager.save(CloudGPUSettings("A100", 5, 38).public())
    manager.save(CloudGPUSettings("A100", 5, 37).public())
    manager.status_path.write_text(
        json.dumps({"state": "ready", "cleanup_confirmed": False, "instance_id": 42})
    )
    spawn = Mock()
    monkeypatch.setattr("tools.cloud_gpu_controller.subprocess.Popen", spawn)
    assert manager.status()["state"] == "cleanup_required"
    with pytest.raises(HTTPException):
        manager.start()
    spawn.assert_not_called()


def test_running_rental_health_is_rechecked_without_another_rental(
    tmp_path, monkeypatch
):
    app = create_app(tmp_path, 37)
    manager = app.state.manager
    manager.process = Mock()
    manager.process.poll.return_value = None
    manager.status_path.write_text(
        json.dumps({"state": "ready", "cleanup_confirmed": False})
    )
    monkeypatch.setattr(manager, "pool_ready", lambda: False)
    monkeypatch.setattr(manager, "stop", AsyncMock())
    with TestClient(app) as client:
        assert client.get("/settings").json()["state"] == "degraded"
        assert client.post("/start").status_code == 409
        monkeypatch.setattr(manager, "pool_ready", lambda: True)
        assert client.get("/settings").json()["state"] == "ready"


def test_cloud_launcher_moves_all_lanes_to_gpu_host_and_mounts_old_data_for_migration():
    text = fleet_overlay(37)
    assert "harness-37:" not in text
    assert "camera-37-data:/legacy/camera-37:ro" in text
    assert '"SAM31_LIVE_CAMERAS": "37"' in text
    assert "LOCAL_CAMERA_37_STREAM_URL" in text


def test_cloud_launcher_dry_run_never_calls_docker_or_creates_files(
    monkeypatch, tmp_path, capsys
):
    from tools import start_cloud_gpu

    monkeypatch.setattr(start_cloud_gpu, "ROOT", tmp_path)
    docker = Mock(side_effect=AssertionError("dry run must not invoke Docker"))
    monkeypatch.setattr(start_cloud_gpu.subprocess, "run", docker)
    assert start_cloud_gpu.main(["--streams", "37", "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert json.loads(capsys.readouterr().out)["streams"] == 37
    docker.assert_not_called()


def test_cloud_gpu_console_controls():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for console interaction tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("cloud_gpu_console.cjs"))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_main_proxy_serializes_actions_and_blocks_mutation_while_analyzing(monkeypatch):
    from test_fleet import make_app

    monkeypatch.setenv("CLOUD_GPU_CONTROLLER_URL", "http://sam-remote:8093")
    app = make_app(monkeypatch, capacity=2)
    app.state.runtime.service_execution_settings = lambda: {"tracking_mode": "cloud"}
    app.state.runtime.tracking_camera_capacity = lambda: 2
    # The real proxy should never be reached while any lane is analyzing.
    app.state.statuses[2]["analysis_state"] = "running"
    with TestClient(app) as client:
        for method, path in (
            ("PUT", "/v1/cloud-gpu"),
            ("POST", "/v1/cloud-gpu/start"),
            ("POST", "/v1/cloud-gpu/stop"),
        ):
            assert client.request(method, path, json={}).status_code == 409
        assert client.get("/v1/fleet").json()["usable_capacity"] == 2


@pytest.mark.parametrize("streams", [2, 12, 37, 1024])
def test_gateway_container_budget_covers_whep_slots_and_reconnects(streams):
    import yaml

    from goodvision_harness.whep_process import DEFAULT_FRAME_CAPACITY

    services = yaml.safe_load(fleet_overlay(streams))["services"]
    reservation = int(services["harness"]["shm_size"].removesuffix("m")) * 1024**2
    # All original receivers and their replacements can coexist; metadata
    # arenas/semaphores need additional space beyond the full pixel slots.
    assert reservation >= 2 * streams * DEFAULT_FRAME_CAPACITY + 16 * 1024**2
    recipe = yaml.safe_load(
        (Path(__file__).parents[1] / "compose.analysis-gateway.yaml").read_text("utf-8")
    )["services"]["harness"]
    assert "HARNESS_GATEWAY_SHM_SIZE" in recipe["shm_size"]
    assert recipe["tmpfs"] == ["/tmp:size=64m"]


@pytest.mark.parametrize("recipe", ["cloud", "legacy"])
def test_cloud_compose_keeps_existing_server_inference_configuration(
    tmp_path, recipe,
):
    from compose_support import isolated_docker_config

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is required for configuration rendering")
    root = Path(__file__).parents[1]
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    docker_config = tmp_path / "docker"
    docker_config.mkdir()
    selected = {
        "INFERENCE_EXECUTION_MODE": "server",
        "INFERENCE_SERVER_URL": "https://vision.example",
        "INFERENCE_SERVER_MODEL": "installed-vision",
        "INFERENCE_SERVER_PROTOCOL": "goodvision",
        "LOCAL_INFERENCE_API_KEY": "companion-test-key",
        "LOCAL_INFERENCE_MAX_TOKENS": "2048",
    }
    environment = {
        **os.environ, **selected,
        "DOCKER_CONFIG": isolated_docker_config(docker_config),
        "VAST_API_KEY": "rental-test-key", "HF_TOKEN": "download-test-token",
        "SAM_SSH_KEY": str(tmp_path / "ssh-key"),
    }
    files = (
        ["compose.cloud-gpu.yaml"] if recipe == "cloud"
        else ["compose.yaml", "compose.sam31-remote.yaml"]
    )
    command = [
        docker, "compose", "--project-directory", str(root),
        "--env-file", str(empty_env),
    ]
    for filename in files:
        command.extend(["-f", str(root / filename)])
    rendered = subprocess.run(
        [*command, "config", "--format", "json"], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert rendered.returncode == 0, rendered.stderr
    services = json.loads(rendered.stdout)["services"]
    controller = services["sam-remote"]["environment"]
    # The old cloud main application received these through main-server; the
    # controller now must pass them to the remote analysis process instead.
    for key, value in selected.items():
        assert controller.get(key) == value, key
    assert services["harness"]["environment"]["HARNESS_ANALYSIS_URL"] == (
        "http://sam-remote:8094"
    )
