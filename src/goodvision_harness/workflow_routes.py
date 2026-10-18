"""Primary-owned workflow drafts, applied snapshots and camera-group controls."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .event_catalog import event_catalog
from .workflows import (
    WorkflowConflictError,
    WorkflowStorageError,
    WorkflowStore,
    changed_cameras,
    compile_document,
    compile_graph,
)

NO_STORE = {"Cache-Control": "no-store"}
MAX_COMMAND_BYTES = 256 * 1024


async def body(request):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_COMMAND_BYTES:
            raise HTTPException(413, "workflow request exceeds 256 KiB")
        raw.extend(chunk)
    try:
        payload = json.loads(raw or b"{}")
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise HTTPException(400, "workflow request must be JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "workflow request must be an object")
    return payload


class WorkflowCoordinator:
    def __init__(self, application):
        self.application = application
        self._store = None

    @property
    def store(self):
        if self._store is None:
            runtime = self.application.state.runtime
            deployment = getattr(runtime, "_deployment_store", None)
            configured = os.environ.get("HARNESS_WORKFLOW_SETTINGS_PATH", "").strip()
            path = (
                Path(configured)
                if configured
                else (
                    deployment.path.with_name("workflow-assets.json")
                    if deployment
                    else Path(".goodvision/workflow-assets.json")
                )
            )
            self._store = WorkflowStore(path)
        return self._store

    def bindings(self):
        document = self.store.load()
        bindings = {}
        for group in document["groups"]:
            if group["applied_graph"] is None:
                continue
            for camera_id in group["camera_ids"]:
                bindings[camera_id] = {
                    "workflow_id": group["workflow_id"],
                    "group_id": group["id"],
                    "graph": group["applied_graph"],
                    "enabled": group["enabled"],
                    "block_enabled": group["block_enabled"],
                    "camera_override": group["camera_overrides"].get(camera_id, {}),
                }
        return document, bindings, compile_document(document)

    def managed_cameras(self):
        document = self.store.load()
        return {
            cid: {
                "group_id": group["id"],
                "group_name": group["name"],
                "workflow_id": group["workflow_id"],
            }
            for group in document["groups"]
            if group["applied_graph"] is not None
            for cid in group["camera_ids"]
        }

    async def state(self):
        fleet = self.application.state.fleet
        statuses = await fleet.refresh_analysis()
        cameras = fleet.descriptor()["cameras"]
        for camera in cameras:
            camera["status"] = statuses.get(camera["index"])
        return {
            "document": self.store.load(),
            "cameras": cameras,
            "event_catalog": event_catalog(),
        }

    async def require_stopped(self, camera_ids):
        fleet = self.application.state.fleet
        if not camera_ids:
            return
        if fleet._analysis_transition:
            raise HTTPException(409, "wait for the analysis operation to finish")
        statuses = await fleet.refresh_analysis()
        for camera_id in camera_ids:
            index = fleet.resolve_camera(camera_id)
            status = statuses.get(index)
            if status is None or fleet._analysis_state(status) != "idle":
                raise HTTPException(
                    409,
                    f"Stop {camera_id} and wait for cleanup "
                    "before changing its workflow.",
                )


def install_workflow_routes(application):
    control = WorkflowCoordinator(application)
    application.state.workflows = control
    fleet = application.state.fleet

    @application.get("/v1/workflows")
    async def state():
        fleet._require_primary()
        try:
            return JSONResponse(await control.state(), headers=NO_STORE)
        except (OSError, ValueError, TypeError, WorkflowStorageError):
            raise HTTPException(
                503, "workflow storage is unavailable or invalid"
            ) from None

    @application.post("/v1/workflows/command")
    async def command(request: Request):
        fleet._require_primary()
        payload = await body(request)
        revision = payload.pop("revision", None)
        action = payload.pop("action", None)
        if type(revision) is not int or not 0 <= revision < 2**53:
            raise HTTPException(400, "workflow revision must be a nonnegative integer")
        if not isinstance(action, str) or not action:
            raise HTTPException(400, "workflow action must be a nonempty string")
        try:
            async with fleet.control_lock:
                previous = control.store.load()
                if revision != previous["revision"]:
                    raise HTTPException(
                        409, "Workflow state changed. Reload before saving."
                    )
                candidate = control.store.candidate(
                    action, payload, expected_revision=revision
                )
                # Only provisioned camera IDs can be assigned. Existing camera
                # configuration/credentials remain in their original stores.
                for group in candidate["groups"]:
                    for camera_id in group["camera_ids"]:
                        fleet.resolve_camera(camera_id)
                affected = set(changed_cameras(previous, candidate))
                if action == "apply":
                    affected.update(
                        cid
                        for group in candidate["groups"]
                        if group["id"] in payload["group_ids"]
                        for cid in group["camera_ids"]
                    )
                await control.require_stopped(affected)
                control.store.commit(candidate, expected_revision=revision)
                return JSONResponse(await control.state(), headers=NO_STORE)
        except WorkflowConflictError:
            raise HTTPException(
                409, "Workflow state changed. Reload before saving."
            ) from None
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        except (OSError, WorkflowStorageError):
            raise HTTPException(503, "workflow changes could not be saved") from None

    @application.post("/v1/workflows/test")
    async def test(request: Request):
        fleet._require_primary()
        payload = await body(request)
        if set(payload) != {"graph", "camera_id"}:
            raise HTTPException(400, "test requires graph and camera_id")
        if not isinstance(payload["camera_id"], str):
            raise HTTPException(400, "camera_id must be a string")
        index = fleet.resolve_camera(payload["camera_id"])
        fleet.require_enabled(index)
        try:
            compile_graph(payload["graph"])
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        async with fleet.control_lock:
            await control.require_stopped([payload["camera_id"]])
            return await fleet.request_json(
                index,
                "POST",
                "/v1/analysis/workflow/test",
                {"graph": payload["graph"]},
                timeout=180,
                coordinated=True,
            )

    @application.get("/assets/workflow_canvas.js", include_in_schema=False)
    async def script():
        return Response(
            (Path(__file__).parent / "assets/workflow_canvas.js").read_text(
                encoding="utf-8"
            ),
            media_type="application/javascript",
        )

    @application.get("/assets/workflow_canvas.css", include_in_schema=False)
    async def stylesheet():
        return Response(
            (Path(__file__).parent / "assets/workflow_canvas.css").read_text(
                encoding="utf-8"
            ),
            media_type="text/css",
        )

    return control
