"""Local worker ownership, model readiness, and deployment entry points."""

import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_workflow_routes import workflow_app

from goodvision_harness import module_service, module_services
from goodvision_harness.module_services import LocalModuleServices, worker_lock
from goodvision_harness.workflow_modules import MODELS


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_MODULE_DATA_DIR", str(tmp_path))
    for engine in ("bbox", "system1", "neural"):
        monkeypatch.delenv(f"GOODVISION_{engine.upper()}_LOCAL_URL", raising=False)
    return LocalModuleServices()


class Process:
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, **kwargs):
        return self.returncode


def test_start_is_nonblocking_reuses_worker_and_only_closes_owned_processes(
    services, monkeypatch
):
    launched = []
    health = {"state": "unavailable"}
    monkeypatch.setattr(services, "_health", lambda *args: health["state"])

    def launch(command, **options):
        process = Process()
        launched.append((command, options, process))
        return process

    monkeypatch.setattr(module_services.subprocess, "Popen", launch)
    monkeypatch.setenv("GOODVISION_SYSTEM1_PYTHON", "isolated-python")
    assert services.start("system1")["state"] == "starting"
    command, options, process = launched[0]
    assert command[:5] == [
        "isolated-python",
        "-u",
        "-m",
        "goodvision_harness.module_service",
        "system1",
    ]
    assert "--singleton" in command
    assert command[command.index("--port") + 1] == "8097"
    assert options.get("shell") is not True
    assert "src" in options["env"]["PYTHONPATH"]
    assert services.start("system1")["state"] == "starting"
    assert len(launched) == 1
    health["state"] = "ready"
    assert services.start("system1")["ready"]
    assert services.start("neural")["ready"]  # Externally owned ready worker.
    assert len(launched) == 1
    services.close()
    assert process.returncode == 0


def test_failed_worker_reports_error_without_claiming_ready(services, monkeypatch):
    process = Process()
    process.returncode = 1
    services._processes["bbox"] = process
    monkeypatch.setattr(services, "_health", lambda *args: "unavailable")
    assert services.status("bbox")["state"] == "error"
    monkeypatch.setattr(services, "_health", lambda *args: "error")
    assert services.status("bbox")["ready"] is False


def test_provisioned_local_service_never_spawns_on_a_remote_host(services, monkeypatch):
    monkeypatch.setenv("GOODVISION_BBOX_LOCAL_URL", "http://bbox-worker:8096")
    monkeypatch.setattr(services, "_health", lambda *args: "unavailable")
    monkeypatch.setattr(
        module_services.subprocess, "Popen", lambda *a, **k: pytest.fail("local spawn")
    )
    with pytest.raises(ValueError, match="provisioned local"):
        services.start("bbox")


def test_worker_os_lock_prevents_duplicate_model_loading(tmp_path):
    with worker_lock(tmp_path / "bbox.lock") as first:
        assert first
        with worker_lock(tmp_path / "bbox.lock") as second:
            assert not second
    with worker_lock(tmp_path / "bbox.lock") as third:
        assert third


@pytest.mark.parametrize("engine", ["bbox", "system1", "neural"])
def test_health_requires_exact_model_and_real_readiness(services, monkeypatch, engine):
    model = "yolo26-s" if engine == "bbox" else MODELS[engine]
    payload = {"ready": True, "gpu_ready": True, "model": model}
    seen = []
    client_class = httpx.Client

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        module_services.httpx,
        "Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    endpoint = (
        "http://worker/v1/labels" if engine == "bbox" else "http://worker/predict"
    )
    assert services._health(engine, endpoint) == "ready"
    assert seen[0].url.path == ("/health/tracking" if engine == "bbox" else "/health")
    if engine == "bbox":
        assert seen[0].headers["x-goodvision-detector-model"] == "yolo26-s"
        payload["gpu_ready"] = False
        assert services._health(engine, endpoint) == "error"
        assert services.status(engine)["state"] == "error"
        payload.update(device="cpu", runtime="ultralytics", provider_ready=True)
        assert services._health(engine, endpoint) == "ready"
        payload["provider_ready"] = False
        assert services._health(engine, endpoint) == "error"
    payload["model"] = "unrelated-model"
    assert services._health(engine, endpoint) == "error"
    payload.update(model=model, ready=False, error="model weights unavailable")
    assert services._health(engine, endpoint) == "error"


def test_busy_existing_worker_does_not_spawn_a_duplicate(services, monkeypatch):
    monkeypatch.setattr(services, "_health", lambda *args: "starting")
    monkeypatch.setattr(
        module_services.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate")
    )
    assert services.start("bbox")["state"] == "starting"


@pytest.mark.parametrize(
    ("engine", "module"),
    [("system1", "videojev_service"), ("neural", "neural_service")],
)
def test_server_entrypoint_reuses_existing_service_and_forwards_bind(
    tmp_path, monkeypatch, engine, module
):
    seen = []
    monkeypatch.setenv("GOODVISION_MODULE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setitem(
        sys.modules,
        "goodvision_harness." + module,
        SimpleNamespace(main=lambda: seen.append(list(sys.argv))),
    )
    monkeypatch.setattr(sys, "argv", [])
    module_service.serve(engine, "0.0.0.0", 9100, tmp_path / "logs")
    assert seen[0][seen[0].index("--host") + 1] == "0.0.0.0"
    assert seen[0][seen[0].index("--port") + 1] == "9100"
    assert "--cache-dir" in seen[0]


def test_local_module_api_respects_fleet_lifecycle_and_fixed_engine_allowlist(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path)
    seen = []
    fake = SimpleNamespace(
        start=lambda engine: (
            seen.append(engine) or {"state": "starting", "ready": False}
        ),
        status=lambda engine: {"state": "ready", "ready": True},
    )
    monkeypatch.setattr(module_services, "local_modules", fake)
    with TestClient(app) as client:
        assert client.get("/v1/workflow-modules/bbox/local").json()["ready"]
        assert (
            client.post("/v1/workflow-modules/bbox/local", json={}).status_code == 200
        )
        assert seen == ["bbox"]
        assert (
            client.post("/v1/workflow-modules/arbitrary/local", json={}).status_code
            == 404
        )
        assert (
            client.post(
                "/v1/workflow-modules/bbox/local", json={"command": "x"}
            ).status_code
            == 400
        )
        app.state.statuses[2].update(analysis_state="running", pipeline_state="running")
        assert (
            client.post("/v1/workflow-modules/neural/local", json={}).status_code == 409
        )
        assert seen == ["bbox"]
