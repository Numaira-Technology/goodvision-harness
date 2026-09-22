"""Isolated diagnostic documents for individual, fixed-topology camera lanes."""
# ruff: noqa: RUF001

import json
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .fleet_outputs import scoped_payload


def diagnostic_document(camera_id: str) -> str:
    # The caller validates the canonical camera ID before rendering. Keeping a
    # separate document per lane also isolates timers, results and unsaved input.
    page = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    page = page.replace("/v1/", f"/v1/fleet/cameras/{camera_id}/")
    page = page.replace(
        '<div id="fleet-detail-console" hidden>', '<div id="fleet-detail-console">'
    )
    page = page.replace('<script src="/assets/fleet_console.js"></script>', "")
    page = page.replace("loadRoutes();setInterval(loadRoutes,1500);", "")
    # Global settings and the shared gallery stay in the parent console. Their
    # elements remain for the existing UI bindings, but have no write API here.
    setup = """
<style>
header,.deployment-panel,#cloud-route-panel,#fleet-panel,#face-controls{
  display:none!important
}
body{background:#fff}main{max-width:none;margin:0;padding:12px}
.fleet-detail-label{position:sticky;top:0;background:#fff;padding:8px 0;z-index:10}
</style>
<script>
globalThis.GoodvisionDiagnostic=true;
globalThis.GoodvisionFleet={detailHidden:()=>false,extraBusy:()=>false};
</script>
"""
    page = page.replace("</head>", setup + "</head>")
    page = page.replace(
        "本机摄像头 / 诊断 · 网络视频请使用上方各路卡片设置，"
        "并通过顶部统一开始、停止分析。此处保留浏览器摄像头和服务诊断。",
        f"摄像头 {int(camera_id[6:])} · 本路诊断与播放缓冲设置。"
        "网络视频和检测项目在本路卡片中调整，分析仍通过顶部统一开始、停止。"
        "浏览器摄像头按钮仅用于本机预览检查。",
    )
    return page


def local_camera_document() -> str:
    """Render the single-runtime console with browser-camera controls enabled."""
    page = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    page = page.replace(
        '<div id="fleet-detail-console" hidden>', '<div id="fleet-detail-console">'
    )
    page = page.replace('<script src="/assets/fleet_console.js"></script>', "")
    page = page.replace(
        'mode("stream");info().finally(loadDeploymentSettings);',
        'mode("camera");info().finally(loadDeploymentSettings);',
    )
    setup = """
<style>
#fleet-panel{display:none!important}
</style>
"""
    page = page.replace("</head>", setup + "</head>")
    page = page.replace(
        "本机摄像头 / 诊断 · 网络视频请使用上方各路卡片设置，"
        "并通过顶部统一开始、停止分析。此处保留浏览器摄像头和服务诊断。",
        "Mac 本机摄像头检测 · 启用摄像头后，选择检测项目并开始分析。",
    )
    return page


def install_diagnostic_routes(application, controller, json_body):
    prefix = "/v1/fleet/cameras/{camera_id}"

    @application.get(
        "/local-camera", response_class=HTMLResponse, include_in_schema=False
    )
    async def local_camera():
        return HTMLResponse(
            local_camera_document(),
            headers={"Cache-Control": "no-store"},
        )

    @application.get(
        prefix + "/diagnostics", response_class=HTMLResponse, include_in_schema=False
    )
    async def document(camera_id: str):
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        return HTMLResponse(
            diagnostic_document(camera_id),
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
