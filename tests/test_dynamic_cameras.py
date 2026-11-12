"""Dynamic camera allocation, persistence, isolation and recovery contracts."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_fleet import make_app

from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.dynamic_cameras import DynamicCameras


def manager(app, directory):
    app.state.runtime.config = SimpleNamespace(
        face=SimpleNamespace(gallery_path=directory / "people.json")
    )
    result = DynamicCameras(app.state.fleet, directory)
    app.state.fleet.dynamic = result
    return result


def test_page_adds_sources_beyond_deployed_and_tracking_capacity_and_restores(
    monkeypatch,
    tmp_path,
):
    app = make_app(monkeypatch, capacity=1, tracking=False)
    cameras = manager(app, tmp_path)

    async def ready(index):
        app.state.statuses[index] = {
            "analysis_state": "idle",
            "connected": False,
        }
        cameras.origins[index] = f"http://harness-{index:02d}:8080"

    monkeypatch.setattr(cameras, "ensure", ready)
    with TestClient(app) as client:
        first = client.post("/v1/fleet/cameras", json={})
        assert first.status_code == 201
        assert first.json()["camera"]["id"] == "camera02"
        assert first.json()["camera"]["status"]["analysis_state"] == "idle"
        assert (
            client.post("/v1/fleet/cameras", json={}).json()["camera"]["id"]
            == "camera03"
        )
        state = client.get("/v1/fleet").json()
        assert state["dynamic_sources"] is True
        assert all(camera["enabled"] for camera in state["cameras"])
        assert state["capacity"] == 3
        assert (
            client.post("/v1/fleet/cameras", json={"capacity": 100}).status_code == 400
        )
    restored = make_app(monkeypatch, capacity=1)
    manager(restored, tmp_path)
    assert restored.state.fleet.capacity == 3
    assert restored.state.fleet.resolve_camera("camera03") == 3
    assert "http" not in cameras.registry.read_text()


def test_failed_creation_keeps_registry_and_existing_cameras(monkeypatch, tmp_path):
    app = make_app(monkeypatch, capacity=1)
    cameras = manager(app, tmp_path)

    async def fail(index):
        raise HTTPException(503, "worker could not start")

    monkeypatch.setattr(cameras, "ensure", fail)
    with TestClient(app) as client:
        assert client.post("/v1/fleet/cameras", json={}).status_code == 503
        assert app.state.fleet.capacity == 1
        assert not cameras.registry.exists()
        assert client.get("/v1/fleet/cameras/camera01/status").status_code == 200


def test_worker_environment_preserves_models_but_isolates_source_and_evidence(
    monkeypatch,
    tmp_path,
):
    app = make_app(monkeypatch, capacity=1)
    cameras = manager(app, tmp_path)
    monkeypatch.setenv("HARNESS_STREAM_URL", "rtsp://private-primary-camera/live")
    monkeypatch.setenv("HARNESS_ANALYSIS_URL", "http://remote-analysis:8080")
    monkeypatch.setenv("FACE_MODEL_ROOT", "/models/insightface")
    env2, env3 = cameras.environment(2), cameras.environment(3)
    assert env2["HARNESS_SOURCE_ID"] != env3["HARNESS_SOURCE_ID"]
    assert env2["HARNESS_STREAM_URL"] == env2["RTSP_URL"] == ""
    assert "HARNESS_ANALYSIS_URL" not in env2
    assert env2["HARNESS_FLEET_DYNAMIC"] == "false"
    assert env2["FACE_MODEL_ROOT"] == "/models/insightface"
    assert env2["FACE_GALLERY_PATH"] == env3["FACE_GALLERY_PATH"]
    assert env2["EVENT_EVIDENCE_DIRECTORY"] != env3["EVENT_EVIDENCE_DIRECTORY"]
    store = DeploymentSettingsStore(env2["HARNESS_DEPLOYMENT_SETTINGS_PATH"])
    assert store.load().stream_url == ""
    assert store.load().autostart is False


def test_broken_registry_is_not_silently_overwritten(monkeypatch, tmp_path):
    app = make_app(monkeypatch, capacity=1)
    path = tmp_path / "video-sources.json"
    path.write_text(json.dumps({"schema_version": 1, "capacity": True}))
    with pytest.raises(ValueError, match="registry"):
        manager(app, tmp_path)


@pytest.mark.parametrize("provisioned,saved,total", [(3, 2, 3), (2, 4, 4)])
def test_saved_registry_allows_provisioned_capacity_changes(
    monkeypatch, tmp_path, provisioned, saved, total,
):
    app = make_app(monkeypatch, capacity=provisioned)
    path = tmp_path / "video-sources.json"
    path.write_text(json.dumps({"schema_version": 1, "capacity": saved}))
    cameras = manager(app, tmp_path)
    assert app.state.fleet.capacity == total
    assert cameras.base_capacity == provisioned
    assert not cameras.owns(provisioned)
    assert cameras.owns(total + 1)
    assert json.loads(path.read_text())["capacity"] == saved


def test_registry_write_failure_rolls_back_worker_and_capacity(monkeypatch, tmp_path):
    app = make_app(monkeypatch, capacity=1)
    cameras = manager(app, tmp_path)
    stopped = []

    async def ready(index):
        pass

    async def stop(index):
        stopped.append(index)

    def disk_full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(cameras, "ensure", ready)
    monkeypatch.setattr(cameras, "stop", stop)
    monkeypatch.setattr("os.fsync", disk_full)
    with TestClient(app) as client:
        response = client.post("/v1/fleet/cameras", json={})
        assert response.status_code == 503
        assert "save" in response.json()["detail"]
    assert stopped == [2]
    assert app.state.fleet.capacity == 1
    assert not cameras.registry.exists()
    assert not list(tmp_path.glob(".video-sources-*"))


def test_dead_worker_is_replaced_before_next_control_request(monkeypatch, tmp_path):
    app = make_app(monkeypatch, capacity=1)
    cameras = manager(app, tmp_path)
    calls = []

    class Process:
        returncode = None

        def terminate(self):
            self.returncode = -15

        async def wait(self):
            return self.returncode

    async def spawn(*args, **kwargs):
        calls.append(kwargs["env"]["HARNESS_SOURCE_ID"])
        return Process()

    async def healthy(client, *args, **kwargs):
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("httpx.AsyncClient.get", healthy)

    async def run():
        await asyncio.gather(cameras.ensure(2), cameras.ensure(2))
        assert calls == ["camera02"]
        cameras.processes[2].returncode = 1
        await cameras.ensure(2)
        assert calls == ["camera02", "camera02"]
        await cameras.close()
        assert not cameras.processes

    asyncio.run(run())
