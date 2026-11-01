"""Workflow command atomicity and integration with the real fleet control path."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
from test_fleet import make_app
from test_video_connection import ControlledAdapter, frame, publish, wait_for
from test_workflows import graph

import goodvision_harness.config as config_module
import goodvision_harness.workflows as workflow_model
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.service import HarnessRuntime, create_app
from goodvision_harness.workflow_routes import (
    MAX_COMMAND_BYTES,
    install_workflow_routes,
)
from goodvision_harness.workflows import WorkflowConflictError, WorkflowStore

ROOT = Path(__file__).resolve().parents[1]


def workflow_app(monkeypatch, tmp_path, *, connected=(), index=1):
    app = None

    def peer(request):
        number = int(request.url.host.removeprefix("harness-"))
        app.state.trace.append((request.url.path, number))
        if request.url.path == "/v1/status" and number in app.state.unreachable:
            return httpx.Response(503, json={"detail": "camera unavailable"})
        if request.url.path == "/v1/analysis/workflow":
            app.state.bindings.append(
                (number, json.loads(request.content), dict(request.headers))
            )
            if number in app.state.failed_sync:
                return httpx.Response(503, json={"detail": "workflow install failed"})
        if request.url.path.startswith("/v1/workflow/"):
            app.state.peer_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "outputs": [
                        {
                            "node_id": "display",
                            "inputs": [
                                {
                                    "value": {
                                        "jpeg_url": "/v1/video/live/latest.jpg",
                                    }
                                }
                            ],
                        }
                    ],
                    "items": [{"node_id": "summary", "text": "Actual camera output"}],
                },
            )
        return app.state.default_peer(request)

    app = make_app(
        monkeypatch, capacity=3, index=index, connected=connected, peer_handler=peer
    )
    app.state.bindings = []
    app.state.trace = []
    app.state.unreachable = set()
    app.state.failed_sync = set()
    coordinator = install_workflow_routes(app)
    coordinator._store = WorkflowStore(tmp_path / "workflow-assets.json")

    @app.put("/v1/analysis/workflow")
    async def configure(request: Request):
        app.state.trace.append((request.url.path, index))
        payload = await request.json()
        app.state.bindings.append((index, payload, dict(request.headers)))
        return {"configured": True, "managed": payload["binding"] is not None}

    @app.post("/v1/analysis/workflow/test")
    async def test_workflow(request: Request):
        return {
            "tested": await request.json(),
            "coordinator": request.headers.get("X-Fleet-Coordinator"),
        }

    return app


def seed(coordinator, action, **payload):
    previous = coordinator.store.load()
    candidate = coordinator.store.candidate(action, payload, previous["revision"])
    return coordinator.store.commit(candidate, previous["revision"])


def assets(app, *, cameras=("camera02",), applied=False):
    control = app.state.workflows
    doc = seed(control, "create_workflow", name="Entrance", graph=graph())
    workflow_id = doc["workflows"][0]["id"]
    doc = seed(
        control, "create_group", name="Entrance cameras", camera_ids=list(cameras)
    )
    group_id = doc["groups"][0]["id"]
    if applied:
        seed(control, "apply", workflow_id=workflow_id, group_ids=[group_id])
    return workflow_id, group_id


def post_command(client, app, action, **payload):
    return client.post(
        "/v1/workflows/command",
        json={
            "revision": app.state.workflows.store.load()["revision"],
            "action": action,
            **payload,
        },
    )


def test_initial_api_reports_real_camera_status_and_catalog(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.get("/v1/workflows")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        data = response.json()
        assert data["document"]["revision"] == 0
        assert data["document"]["groups"] == []
        assert {c["id"] for c in data["cameras"]} == {
            "camera01",
            "camera02",
            "camera03",
        }
        assert all(c["status"]["analysis_state"] == "idle" for c in data["cameras"])
        assert any(item["id"] == "stranger" for item in data["event_catalog"]["events"])
        assert data["tracking"]["configured"] is False
        assert data["tracking"]["processing_method"] == "sam_live"


def test_canvas_reports_tracking_configuration_without_exposing_service_secrets(
    monkeypatch,
    tmp_path,
):
    app = workflow_app(monkeypatch, tmp_path)
    capability = {
        "configured": True,
        "processing_method": "sam_live",
        "reason": (
            "Tracking service configured; readiness is checked when analysis starts."
        ),
    }
    app.state.runtime.workflow_tracking_capability = lambda: capability
    with TestClient(app) as client:
        response = client.get("/v1/workflows")
        assert response.status_code == 200
        assert response.json()["tracking"] == capability


def test_workflow_save_rejects_analysis_without_a_hidden_saved_version(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, _ = assets(app, applied=True)
    app.state.statuses[2].update(analysis_state="running", pipeline_state="running")
    previous = app.state.workflows.store.load()
    with TestClient(app) as client:
        value = graph()
        value["nodes"][3]["config"]["prompt"] = "A draft prompt."
        response = post_command(
            client, app, "save_workflow", id=workflow_id, name="Draft", graph=value
        )
        assert response.status_code == 409
        assert app.state.workflows.store.load() == previous
        assert app.state.statuses[2]["analysis_state"] == "running"
        assert not app.state.bindings
        assert not app.state.own_requests


def test_apply_checks_only_affected_cameras_and_never_writes_remote_state(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, group_id = assets(app)
    app.state.statuses[1].update(analysis_state="running", pipeline_state="running")
    with TestClient(app) as client:
        response = post_command(
            client, app, "apply", workflow_id=workflow_id, group_ids=[group_id]
        )
        assert response.status_code == 200
        assert response.json()["document"]["groups"][0]["workflow_id"] == workflow_id
        assert app.state.statuses[1]["analysis_state"] == "running"
        assert not app.state.bindings
        assert all(request.method == "GET" for request in app.state.peer_requests)


@pytest.mark.parametrize(
    "state", ["running", "starting", "stopping", "draining", "unknown", "unreachable"]
)
def test_apply_rejection_is_atomic_for_busy_or_unknown_targets(
    monkeypatch, tmp_path, state
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, first = assets(app, cameras=("camera01",))
    doc = seed(
        app.state.workflows, "create_group", name="Second", camera_ids=["camera02"]
    )
    second = doc["groups"][1]["id"]
    if state == "unreachable":
        app.state.unreachable.add(2)
    else:
        app.state.statuses[2].update(analysis_state=state, pipeline_state=state)
    path = app.state.workflows.store.path
    before = path.read_bytes()
    with TestClient(app) as client:
        response = post_command(
            client, app, "apply", workflow_id=workflow_id, group_ids=[first, second]
        )
        assert response.status_code == 409
        assert path.read_bytes() == before
        assert all(
            group["applied_graph"] is None
            for group in app.state.workflows.store.load()["groups"]
        )
        assert not app.state.bindings


def test_same_graph_reapply_also_requires_stopped_targets(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, group_id = assets(app, applied=True)
    app.state.statuses[2].update(analysis_state="running", pipeline_state="running")
    before = app.state.workflows.store.load()
    with TestClient(app) as client:
        response = post_command(
            client, app, "apply", workflow_id=workflow_id, group_ids=[group_id]
        )
        assert response.status_code == 409
        assert app.state.workflows.store.load() == before


@pytest.mark.parametrize("revision", [None, True, -1, "0", 1.0, 2**53])
def test_malformed_revision_is_400_without_mutation(monkeypatch, tmp_path, revision):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/v1/workflows/command",
            json={"revision": revision, "action": "create_workflow", "name": "Invalid"},
        )
        assert response.status_code == 400
        assert app.state.workflows.store.load()["revision"] == 0


def test_stale_revision_and_commit_race_are_409(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert (
            post_command(client, app, "create_workflow", name="First").status_code
            == 200
        )
        previous = app.state.workflows.store.load()
        response = client.post(
            "/v1/workflows/command",
            json={"revision": 0, "action": "create_workflow", "name": "Stale"},
        )
        assert response.status_code == 409

        def conflict(*_args, **_kwargs):
            raise WorkflowConflictError("concurrent save")

        monkeypatch.setattr(app.state.workflows.store, "commit", conflict)
        assert (
            post_command(client, app, "create_workflow", name="Race").status_code == 409
        )
        assert app.state.workflows.store.load() == previous


@pytest.mark.parametrize(
    ("camera", "status"), [("camera04", 404), ("camera1", 400), ("../camera01", 400)]
)
def test_group_camera_ids_must_be_valid_provisioned_lanes(
    monkeypatch, tmp_path, camera, status
):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = post_command(
            client, app, "create_group", name="Invalid", camera_ids=[camera]
        )
        assert response.status_code == status
        assert app.state.workflows.store.load()["groups"] == []


def test_invalid_graph_and_overlapping_groups_return_400_without_partial_changes(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, _ = assets(app)
    previous = app.state.workflows.store.load()
    invalid = graph()
    invalid["edges"][0]["to_port"] = "summary"
    with TestClient(app) as client:
        assert (
            post_command(
                client, app, "save_workflow", id=workflow_id, name="Bad", graph=invalid
            ).status_code
            == 400
        )
        assert (
            post_command(
                client, app, "create_group", name="Overlap", camera_ids=["camera02"]
            ).status_code
            == 400
        )
        assert app.state.workflows.store.load() == previous


def test_disk_write_failure_is_503_and_keeps_previous_revision(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path)
    assets(app)
    previous = app.state.workflows.store.load()

    def reject(*_args):
        raise PermissionError("disk unavailable")

    monkeypatch.setattr(workflow_model.os, "replace", reject)
    with TestClient(app) as client:
        assert (
            post_command(client, app, "create_workflow", name="Not saved").status_code
            == 503
        )
        assert app.state.workflows.store.load() == previous


def test_corrupt_storage_fails_closed_for_reads_commands_and_fleet_start(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path, connected=(1, 2))
    path = app.state.workflows.store.path
    path.write_bytes(b"{corrupt")
    with TestClient(app) as client:
        assert client.get("/v1/workflows").status_code == 503
        response = client.post(
            "/v1/workflows/command",
            json={"revision": 0, "action": "create_workflow", "name": "No fallback"},
        )
        assert response.status_code == 503
        assert client.get("/v1/fleet").status_code == 503
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 503
        assert path.read_bytes() == b"{corrupt"
        assert not app.state.bindings
        assert not app.state.own_requests


def test_command_body_is_bounded_and_deep_json_is_a_client_error(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        oversized = client.post(
            "/v1/workflows/command", content=b"x" * (MAX_COMMAND_BYTES + 1)
        )
        assert oversized.status_code == 413
        nested = b'{"graph":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}"
        assert client.post("/v1/workflows/command", content=nested).status_code == 400
        assert client.post("/v1/workflows/command", json=[]).status_code == 400
        assert app.state.workflows.store.load()["revision"] == 0


def test_secondary_runtime_cannot_manage_global_assets(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path, index=2)
    with TestClient(app) as client:
        assert client.get("/v1/workflows").status_code == 409
        assert client.post("/v1/workflows/command", json={}).status_code == 409
        assert client.post("/v1/workflows/test", json={}).status_code == 409
        assert app.state.workflows.store.load()["revision"] == 0


def test_fleet_start_clears_every_unmanaged_binding_at_revision_zero(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path, connected=(1, 2))
    with TestClient(app) as client:
        response = client.post("/v1/fleet/analysis/start", json={})
        assert response.status_code == 200
        assert [(index, payload) for index, payload, _ in app.state.bindings] == [
            (1, {"binding": None}),
            (2, {"binding": None}),
        ]
        assert all(
            headers["x-fleet-coordinator"] == "1"
            for _, _, headers in app.state.bindings
        )
        prepare = app.state.trace.index(("/v1/analysis/prepare", 2))
        assert all(
            app.state.trace.index(("/v1/analysis/workflow", index)) < prepare
            for index in (1, 2)
        )


def test_fleet_installs_the_same_atomically_saved_and_applied_workflow(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path, connected=(1, 2))
    workflow_id, _ = assets(app, applied=True)
    draft = graph()
    draft["nodes"][3]["config"]["prompt"] = "Unsaved to groups."
    seed(
        app.state.workflows,
        "save_workflow",
        id=workflow_id,
        name="New draft",
        graph=draft,
    )
    with TestClient(app) as client:
        assert client.post("/v1/fleet/analysis/start", json={}).status_code == 200
        peer_binding = next(
            payload["binding"] for index, payload, _ in app.state.bindings if index == 2
        )
        assert peer_binding["workflow_id"] == workflow_id
        assert (
            peer_binding["graph"]["nodes"][3]["config"]["prompt"]
            == "Unsaved to groups."
        )


def test_failed_binding_sync_starts_no_camera_and_retains_connections(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path, connected=(1, 2, 3))
    assets(app, applied=True)
    app.state.failed_sync.add(2)
    with TestClient(app) as client:
        response = client.post("/v1/fleet/analysis/start", json={})
        assert response.status_code == 409
        assert not app.state.own_requests
        assert not any(
            request.url.path in {"/v1/analysis/prepare", "/v1/analysis/start"}
            for request in app.state.peer_requests
        )
        assert all(status["connected"] for status in app.state.statuses.values())


@pytest.mark.parametrize(
    "detail",
    [
        "Person tracking is not configured for this deployment.",
        "PRIVATE_CREDENTIAL",
        "Person tracking is not configured for this deployment. PRIVATE_CREDENTIAL",
    ],
)
def test_fleet_start_explains_only_known_public_tracking_binding_failure(
    monkeypatch,
    tmp_path,
    detail,
):
    from fastapi.responses import JSONResponse
    from test_workflows import tracking_graph

    app = workflow_app(monkeypatch, tmp_path, connected=(1,))
    workflow_id, group_id = assets(app, cameras=("camera01",))
    seed(
        app.state.workflows,
        "save_workflow",
        id=workflow_id,
        name="Tracking",
        graph=tracking_graph(),
    )
    seed(
        app.state.workflows,
        "apply",
        workflow_id=workflow_id,
        group_ids=[group_id],
    )
    fleet = app.state.fleet
    request_json = fleet.request_json

    async def reject(index, method, path, *args, **kwargs):
        if path == "/v1/analysis/workflow":
            return JSONResponse({"detail": detail}, status_code=400)
        return await request_json(index, method, path, *args, **kwargs)

    monkeypatch.setattr(fleet, "request_json", reject)
    with TestClient(app) as client:
        response = client.post("/v1/fleet/analysis/start", json={})
        assert response.status_code == 409
        if detail == "Person tracking is not configured for this deployment.":
            assert "camera01 未配置人物追踪服务" in response.json()["detail"]
            assert "Canvas 关闭人物追踪" in response.json()["detail"]
        else:
            assert response.json()["detail"] == (
                "Could not prepare workflow for camera01. No cameras started."
            )
        assert "PRIVATE_CREDENTIAL" not in response.text
        assert not app.state.own_requests
        assert app.state.statuses[1]["connected"]
        assert all(
            status["analysis_state"] == "idle" for status in app.state.statuses.values()
        )


def test_all_off_workflow_skips_its_lane_during_global_start(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path, connected=(1, 2))
    _, group_id = assets(app, applied=True)
    seed(app.state.workflows, "update_group", id=group_id, enabled=False)
    with TestClient(app) as client:
        response = client.post("/v1/fleet/analysis/start", json={})
        assert response.status_code == 200
        assert response.json()["analysis"]["camera_ids"] == ["camera01"]
        assert [index for index, _, _ in app.state.bindings] == [1]
        assert app.state.statuses[2]["analysis_state"] == "idle"


def test_managed_camera_manual_settings_are_locked_but_legacy_camera_is_unchanged(
    monkeypatch, tmp_path
):
    app = workflow_app(monkeypatch, tmp_path)
    _, group_id = assets(app, applied=True)
    with TestClient(app) as client:
        cameras = client.get("/v1/fleet").json()["cameras"]
        assert cameras[1]["workflow"]["group_id"] == group_id
        assert cameras[2]["workflow"] is None
        managed = client.put(
            "/v1/fleet/cameras/camera02/details", json={"features": ["fall"]}
        )
        assert managed.status_code == 409
        legacy = client.put(
            "/v1/fleet/cameras/camera03/details", json={"features": ["fall"]}
        )
        assert legacy.status_code == 200
        writes = [
            request
            for request in app.state.peer_requests
            if request.method == "PUT" and request.url.path == "/v1/analysis/settings"
        ]
        assert [request.url.host for request in writes] == ["harness-03"]


@pytest.mark.parametrize("suffix", ["results", "summaries?limit=2&node_id=summary"])
def test_workflow_outputs_forward_with_camera_identity_and_scoped_media_links(
    monkeypatch, tmp_path, suffix
):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.get(f"/v1/fleet/cameras/camera02/workflow/{suffix}")
        assert response.status_code == 200
        assert response.headers["x-camera-id"] == "camera02"
        payload = response.json()
        assert payload["camera_id"] == "camera02"
        assert (
            payload["outputs"][0]["inputs"][0]["value"]["jpeg_url"]
            == "/v1/fleet/cameras/camera02/video/live/latest.jpg"
        )


@pytest.mark.parametrize("camera_id", ["camera01", "camera02"])
def test_source_image_forwarding_preserves_frame_identity(monkeypatch, camera_id):
    headers = {
        "Content-Type": "image/jpeg",
        "X-Source-Frame-Sequence": "42",
        "X-Source-Timestamp": "2.8",
        "X-Connection-Id": "active-connection",
        "Set-Cookie": "private=hidden",
    }

    def peer(request):
        assert request.url.path == "/v1/video/source/latest.jpg"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"source-image", headers=headers)

    app = make_app(monkeypatch, peer_handler=peer)

    @app.get("/v1/video/source/latest.jpg")
    def own_source():
        return Response(b"source-image", headers=headers)

    with TestClient(app) as client:
        result = client.get(
            f"/v1/fleet/cameras/{camera_id}/video/source/latest.jpg",
            headers={"Authorization": "Bearer viewer-secret"},
        )
        assert result.status_code == 200
        assert result.content == b"source-image"
        assert result.headers["content-type"] == "image/jpeg"
        assert result.headers["x-camera-id"] == camera_id
        for name in (
            "X-Source-Frame-Sequence",
            "X-Source-Timestamp",
            "X-Connection-Id",
        ):
            assert result.headers[name] == headers[name]
        assert result.headers["cache-control"] == "no-store"
        assert "set-cookie" not in result.headers


def test_source_only_workflow_serves_real_preview_without_native_analysis(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "1")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "1")
    monkeypatch.delenv("PERCEPTION_METHOD", raising=False)
    base = load_config(ROOT / "config.yaml")
    config = replace(
        base,
        rich_labeling=replace(
            base.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            processing_method=None,
        ),
        runtime=replace(base.runtime, scene_mode="disabled"),
        events=replace(base.events, directory=tmp_path / "events"),
        face=replace(base.face, gallery_path=tmp_path / "gallery.json"),
    )
    runtime = HarnessRuntime(
        config,
        deployment_store=DeploymentSettingsStore(tmp_path / "deployment.json"),
    )
    monkeypatch.setattr(
        runtime,
        "_require_processing_ready",
        lambda: pytest.fail("source output must not prepare native inference"),
    )
    monkeypatch.setattr(
        runtime,
        "_make_summary_manager",
        lambda *args, **kwargs: pytest.fail("source output must not start a model"),
    )
    native = "/v1/video/source/latest.jpg"
    scoped = "/v1/fleet/cameras/camera01/video/source/latest.jpg"
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.get(native).status_code == 404
        assert client.get(scoped).status_code == 404
        adapter = ControlledAdapter("rtsp://camera.test/source")
        runtime.connect_stream("rtsp://camera.test/source", _adapter=adapter)
        connection = runtime._video_connection
        publish(connection, adapter, frame(15))
        wait_for(lambda: connection.latest_jpeg() is not None)
        assert runtime._session is None
        runtime.configure_workflow(
            {
                "binding": {
                    "workflow_id": "preview",
                    "group_id": "entrance",
                    "graph": {
                        "nodes": [
                            {"id": "source", "type": "source"},
                            {"id": "display", "type": "output"},
                        ],
                        "edges": [
                            {
                                "id": "source-display",
                                "from_node": "source",
                                "from_port": "frames",
                                "to_node": "display",
                                "to_port": "frames",
                            }
                        ],
                    },
                    "enabled": True,
                    "block_enabled": {},
                    "camera_override": {},
                }
            }
        )
        started = client.post("/v1/analysis/start", json={})
        assert started.status_code == 200
        assert started.json()["analysis_state"] == "running"
        assert runtime._session is None
        image = client.get(scoped)
        assert image.status_code == 200
        assert image.content == client.get(native).content
        assert image.content.startswith(b"\xff\xd8")
        assert image.headers["x-connection-id"] == connection.connection_id
        assert image.headers["x-source-frame-sequence"] == "15"
        assert float(image.headers["x-source-timestamp"]) == 1.0
        output = client.get("/v1/fleet/cameras/camera01/workflow/results").json()
        value = output["outputs"][0]["inputs"][0]["value"]
        assert value["jpeg_url"] == scoped
        assert value["connection_id"] == image.headers["x-connection-id"]
        assert value["source_frame_sequence"] == 15
        assert client.post("/v1/analysis/stop", json={}).status_code == 200
        assert runtime._session is None
        assert client.get(scoped).status_code == 200
        assert client.post("/v1/video/disconnect", json={}).status_code == 200
        wait_for(lambda: not connection.running)
        assert client.get(native).status_code == 404
        assert client.get(scoped).status_code == 404


@pytest.mark.parametrize("camera_id", [None, 1, {}, []])
def test_snapshot_test_rejects_nonstring_camera_id_with_400(
    monkeypatch, tmp_path, camera_id
):
    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/workflows/test", json={"camera_id": camera_id, "graph": graph()}
            ).status_code
            == 400
        )


def test_snapshot_test_uses_coordinator_without_modifying_assets(monkeypatch, tmp_path):
    app = workflow_app(monkeypatch, tmp_path)
    previous = app.state.workflows.store.load()
    with TestClient(app) as client:
        result = client.post(
            "/v1/workflows/test", json={"camera_id": "camera01", "graph": graph()}
        )
        assert result.status_code == 200
        assert result.json()["coordinator"] == "1"
        assert app.state.workflows.store.load() == previous
        app.state.statuses[1].update(analysis_state="running", pipeline_state="running")
        assert (
            client.post(
                "/v1/workflows/test", json={"camera_id": "camera01", "graph": graph()}
            ).status_code
            == 409
        )


def test_snapshot_test_releases_fleet_lock_after_admission_for_stop(
    monkeypatch, tmp_path
):
    async def exercise():
        app = workflow_app(monkeypatch, tmp_path, connected=(2,))
        workflow_id, group_id = assets(app, applied=True)
        previous = app.state.workflows.store.load()
        dispatched, admitted, admit, release = (asyncio.Event() for _ in range(4))
        stopped = asyncio.Event()

        async def peer(request):
            index = int(request.url.host.removeprefix("harness-"))
            if request.url.path == "/v1/analysis/workflow/test":
                dispatched.set()
                await admit.wait()
                app.state.statuses[index].update(
                    pipeline_state="starting",
                    analysis_state="starting",
                    workflow_test_running=True,
                )
                admitted.set()
                await release.wait()
                app.state.statuses[index].update(
                    pipeline_state="idle",
                    analysis_state="idle",
                    workflow_test_running=False,
                )
                return httpx.Response(200, json={"test": True})
            if request.url.path == "/v1/analysis/stop" and index == 2:
                stopped.set()
                # A cancelled model may still be draining. Stop must return
                # without waiting for that model, and mutations stay blocked.
                app.state.statuses[index].update(
                    pipeline_state="stopping", analysis_state="stopping"
                )
                return httpx.Response(200, json={"ok": True})
            return app.state.default_peer(request)

        fleet = app.state.fleet
        await fleet._peer_client.aclose()
        fleet._peer_client = httpx.AsyncClient(transport=httpx.MockTransport(peer))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                testing = asyncio.create_task(
                    client.post(
                        "/v1/workflows/test",
                        json={"camera_id": "camera02", "graph": graph()},
                    )
                )
                stopping = None
                try:
                    await asyncio.wait_for(dispatched.wait(), 1)
                    # Before runtime admission, other fleet controls must not
                    # overtake this request and allow it to start after Stop.
                    stopping = asyncio.create_task(
                        client.post("/v1/fleet/analysis/stop", json={})
                    )
                    await asyncio.sleep(0.1)
                    assert not stopped.is_set()
                    admit.set()
                    await asyncio.wait_for(admitted.wait(), 1)
                    response = await asyncio.wait_for(stopping, 1)
                    assert response.status_code == 200
                    assert stopped.is_set()
                    assert not testing.done()
                    # The in-flight model still owns this camera lifecycle.
                    for command in (
                        {"action": "update_group", "id": group_id, "enabled": False},
                        {
                            "action": "apply",
                            "workflow_id": workflow_id,
                            "group_ids": [group_id],
                        },
                    ):
                        result = await asyncio.wait_for(
                            client.post(
                                "/v1/workflows/command",
                                json={"revision": previous["revision"], **command},
                            ),
                            1,
                        )
                        assert result.status_code == 409
                    result = await asyncio.wait_for(
                        client.post("/v1/fleet/analysis/start", json={}), 1
                    )
                    assert result.status_code == 409
                    assert app.state.workflows.store.load() == previous
                    release.set()
                    assert (await asyncio.wait_for(testing, 1)).status_code == 200
                finally:
                    admit.set()
                    release.set()
                    await asyncio.gather(
                        testing,
                        *([stopping] if stopping is not None else []),
                        return_exceptions=True,
                    )
        finally:
            await fleet.aclose()

    asyncio.run(exercise())


def test_native_workflow_controls_require_coordinator_and_bound_request_bodies(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "1")
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        deployment_store=DeploymentSettingsStore(tmp_path / "deployment.json"),
    )
    observed = []
    monkeypatch.setattr(
        runtime, "test_workflow", lambda value: observed.append(value) or {"test": True}
    )
    with TestClient(create_app(lambda: runtime)) as client:
        assert (
            client.put("/v1/analysis/workflow", json={"binding": None}).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/analysis/workflow/test", json={"graph": graph()}
            ).status_code
            == 409
        )
        headers = {"X-Fleet-Coordinator": "1"}
        assert (
            client.put(
                "/v1/analysis/workflow", json={"binding": None}, headers=headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/analysis/workflow/test", json={"graph": graph()}, headers=headers
            ).status_code
            == 200
        )
        assert observed == [graph()]
        oversized = json.dumps({"graph": "x" * (256 * 1024 + 1)})
        assert (
            client.post(
                "/v1/analysis/workflow/test", content=oversized, headers=headers
            ).status_code
            == 413
        )
        assert len(observed) == 1


@pytest.mark.parametrize("action", ["apply", "save_workflow"])
@pytest.mark.parametrize("unchanged", [False, True])
@pytest.mark.parametrize(
    "lifecycle",
    [
        {"analysis_state": "running"},
        {"analysis_state": "starting"},
        {"analysis_state": "stopping"},
        {"analysis_state": "draining"},
        {"analysis_state": "unknown"},
        {"workflow_test_running": True},
        {"session_start_pending": True},
        {"processing_cleanup": "closing"},
        {"processing_cleanup": "stopping"},
        {"unreachable": True},
    ],
)
def test_workflow_edit_locks_every_existing_camera_even_when_gated_off_or_unselected(
    monkeypatch,
    tmp_path,
    action,
    unchanged,
    lifecycle,
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, old_group = assets(app, applied=True)
    seed(
        app.state.workflows,
        "update_group",
        id=old_group,
        enabled=False,
        block_enabled={"source": False},
        camera_overrides={"camera02": {"enabled": False}},
    )
    created = seed(
        app.state.workflows, "create_group", name="New target", camera_ids=["camera01"]
    )
    new_group = created["groups"][-1]["id"]
    if lifecycle.get("unreachable"):
        app.state.unreachable.add(2)
    else:
        app.state.statuses[2].update(lifecycle)
    before = app.state.workflows.store.load()
    before_bytes = app.state.workflows.store.path.read_bytes()
    value = graph()
    if not unchanged:
        value["nodes"][3]["config"]["prompt"] = "Apply this change directly."
    payload = (
        {"workflow_id": workflow_id, "group_ids": [new_group], "graph": value}
        if action == "apply"
        else {"id": workflow_id, "name": "Renamed", "graph": value}
    )
    with TestClient(app) as client:
        response = post_command(client, app, action, **payload)
        assert response.status_code == 409
        assert app.state.workflows.store.load() == before
        assert app.state.workflows.store.path.read_bytes() == before_bytes
        assert not app.state.bindings


def test_one_apply_updates_saved_graph_existing_assignment_and_new_group_atomically(
    monkeypatch,
    tmp_path,
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, first = assets(app, applied=True)
    created = seed(
        app.state.workflows, "create_group", name="New target", camera_ids=["camera01"]
    )
    second = created["groups"][-1]["id"]
    app.state.statuses[3].update(analysis_state="running", pipeline_state="running")
    before = app.state.workflows.store.load()
    value = graph()
    value["nodes"][3]["config"]["prompt"] = "Count people directly."
    with TestClient(app) as client:
        response = post_command(
            client,
            app,
            "apply",
            workflow_id=workflow_id,
            group_ids=[second],
            graph=value,
        )
        assert response.status_code == 200
        after = response.json()["document"]
        assert after["revision"] == before["revision"] + 1
        canonical = after["workflows"][0]["graph"]
        assert canonical["nodes"][3]["config"]["prompt"] == "Count people directly."
        assert {group["id"] for group in after["groups"]} == {first, second}
        assert all(group["applied_graph"] == canonical for group in after["groups"])
        assert all(group["workflow_id"] == workflow_id for group in after["groups"])
        assert app.state.statuses[3]["analysis_state"] == "running"
        assert not app.state.bindings


def test_apply_without_new_group_selection_updates_existing_associations(
    monkeypatch,
    tmp_path,
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, _ = assets(app, applied=True)
    before = app.state.workflows.store.load()
    value = graph()
    value["nodes"][3]["config"]["interval_seconds"] = 15
    with TestClient(app) as client:
        response = post_command(
            client, app, "apply", workflow_id=workflow_id, group_ids=[], graph=value
        )
        assert response.status_code == 200
        after = response.json()["document"]
        assert after["revision"] == before["revision"] + 1
        assert after["groups"][0]["applied_graph"] == after["workflows"][0]["graph"]
        assert (
            after["groups"][0]["applied_graph"]["nodes"][3]["config"][
                "interval_seconds"
            ]
            == 15
        )


def test_apply_unassigned_workflow_allows_empty_groups_despite_unrelated_analysis(
    monkeypatch,
    tmp_path,
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, _ = assets(app)
    app.state.statuses[2].update(analysis_state="running", pipeline_state="running")
    with TestClient(app) as client:
        response = post_command(
            client, app, "apply", workflow_id=workflow_id, group_ids=[], graph=graph()
        )
        assert response.status_code == 200
        assert response.json()["document"]["groups"][0]["workflow_id"] is None


@pytest.mark.parametrize("failure", ["invalid", "stale", "write"])
def test_direct_apply_failure_never_leaves_a_saved_but_unapplied_version(
    monkeypatch,
    tmp_path,
    failure,
):
    app = workflow_app(monkeypatch, tmp_path)
    workflow_id, _ = assets(app, applied=True)
    before = app.state.workflows.store.load()
    before_bytes = app.state.workflows.store.path.read_bytes()
    value = graph()
    value["nodes"][3]["config"]["prompt"] = "This must not be saved separately."
    if failure == "invalid":
        value["edges"] = []
    elif failure == "write":

        def unavailable(*_):
            raise PermissionError("private storage failure")

        monkeypatch.setattr(workflow_model.os, "replace", unavailable)
    with TestClient(app) as client:
        response = client.post(
            "/v1/workflows/command",
            json={
                "revision": before["revision"] - (failure == "stale"),
                "action": "apply",
                "workflow_id": workflow_id,
                "group_ids": [],
                "graph": value,
            },
        )
        assert (
            response.status_code
            == {"invalid": 400, "stale": 409, "write": 503}[failure]
        )
        assert app.state.workflows.store.load() == before
        assert app.state.workflows.store.path.read_bytes() == before_bytes
