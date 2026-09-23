"""Same-origin controls for the private cloud rental controller."""

# ruff: noqa: RUF001 -- User-facing Chinese punctuation.

from __future__ import annotations

import os

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


def install_cloud_gpu_routes(application, fleet):
    from .fleet import _json_body
    from .service_locations import _service_origin

    origin = _service_origin(
        os.environ.get("CLOUD_GPU_CONTROLLER_URL", ""), "CLOUD_GPU_CONTROLLER_URL"
    )

    async def proxy(method, path, payload=None):
        fleet._require_primary()
        if not origin:
            if method == "GET":
                return JSONResponse({"available": False, "capacity": fleet.capacity})
            raise HTTPException(
                503, "当前部署未启用云 GPU 控制器，请使用云 GPU 启动入口"
            )
        try:
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=False, timeout=70
            ) as client:
                response = await client.request(method, origin + path, json=payload)
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("invalid controller response")
                return JSONResponse(
                    data,
                    status_code=response.status_code,
                    headers={"Cache-Control": "no-store"},
                )
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "云 GPU 控制器无法连接，请检查部署状态") from None

    async def require_idle():
        statuses = await fleet.snapshot()
        if any(
            status is None or fleet._analysis_state(status) != "idle"
            for status in statuses.values()
        ):
            raise HTTPException(409, "请先停止全部分析，再修改、启动或释放云 GPU")

    @application.get("/v1/cloud-gpu")
    async def status():
        return await proxy("GET", "/settings")

    @application.put("/v1/cloud-gpu")
    async def save(request: Request):
        payload = await _json_body(request)
        async with fleet.control_lock:
            fleet._require_primary()
            await require_idle()
            return await proxy("PUT", "/settings", payload)

    @application.post("/v1/cloud-gpu/{action}")
    async def control(action: str, request: Request):
        if action not in {"start", "stop"}:
            raise HTTPException(404, "unknown cloud GPU action")
        if await _json_body(request):
            raise HTTPException(400, "请先保存租用配置")
        async with fleet.control_lock:
            fleet._require_primary()
            await require_idle()
            if action == "start" and fleet.tracking_mode() != "cloud":
                raise HTTPException(409, "请先把人物追踪运行位置保存为云 GPU 服务器")
            return await proxy("POST", "/" + action)
