"""Isolated diagnostic documents for individual, fixed-topology camera lanes."""
# ruff: noqa: RUF001

import json
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .fleet_outputs import scoped_payload

DiagnosticView = Literal["results", "events", "diagnostics", "camera"]


def diagnostic_document(camera_id: str, view: DiagnosticView = "diagnostics") -> str:
    # The caller validates the canonical camera ID before rendering. Keeping a
    # separate document per lane also isolates timers, results and unsaved input.
    page = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    page = page.replace("/v1/", f"/v1/fleet/cameras/{camera_id}/")
    page = page.replace(
        '<div id="fleet-detail-console" hidden>', '<div id="fleet-detail-console">'
    )
    page = page.replace('<script src="/assets/fleet_console.js"></script>', "")
    page = page.replace('<script src="/assets/workflow_canvas.js"></script>', "")
    page = page.replace("loadRoutes();setInterval(loadRoutes,1500);", "")
    # Global settings and the shared gallery stay in the parent console. Their
    # elements remain for the existing UI bindings, but have no write API here.
    setup = """
<script>
globalThis.GoodvisionDiagnostic=true;
globalThis.GoodvisionDiagnosticCameraId=CAMERA_ID;
globalThis.GoodvisionDiagnosticView=DIAGNOSTIC_VIEW;
globalThis.GoodvisionFleet={detailHidden:()=>false,extraBusy:()=>false};
</script>
"""
    # JSON string encoding keeps values as data in the inline setup script.
    setup = setup.replace("CAMERA_ID", json.dumps(camera_id).replace("<", "\\u003c"))
    setup = setup.replace("DIAGNOSTIC_VIEW", json.dumps(view).replace("<", "\\u003c"))
    page = page.replace("</head>", setup + "</head>")
    return page


def install_diagnostic_routes(application, controller, json_body):
    prefix = "/v1/fleet/cameras/{camera_id}"

    @application.get(
        prefix + "/diagnostics", response_class=HTMLResponse, include_in_schema=False
    )
    async def document(
        camera_id: str,
        view: Annotated[DiagnosticView, Query()] = "diagnostics",
    ):
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        return HTMLResponse(
            diagnostic_document(camera_id, view),
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Options": "SAMEORIGIN",
            },
        )

    @application.get(prefix + "/ui/info")
    async def info(camera_id: str):
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        response = await controller.request_json(index, "GET", "/v1/ui/info")
        # Runtime info contains output links as well as scalar capabilities.
        return JSONResponse(
            scoped_payload(json.loads(response.body), camera_id),
            status_code=response.status_code,
        )

    @application.get(prefix + "/deployment/settings")
    async def settings(camera_id: str):
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        return await controller.request_json(index, "GET", "/v1/deployment/settings")

    @application.put(prefix + "/deployment/settings")
    async def buffer_settings(camera_id: str, request: Request):
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        payload = await json_body(request)
        if set(payload) != {"buffer_delay_seconds"}:
            raise HTTPException(400, "此处仅修改本路播放缓冲；全局设置请使用主控制台。")
        # Display delay is intentionally adjustable during analysis. The runtime
        # validates and persists it under its own lock; never change cohort state.
        return await controller.request_json(
            index,
            "PUT",
            "/v1/deployment/settings",
            payload,
        )
