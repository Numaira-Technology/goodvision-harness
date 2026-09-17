"""Same-origin controls for launcher-provisioned, isolated camera runtimes.

Only the first runtime is published on the host. Camera destinations and routes
are derived from the launch topology; callers cannot choose a proxy target.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .privacy import public_safe_value

CONTROL_MAX_BYTES = 64 * 1024
RESPONSE_MAX_BYTES = 2 * 1024 * 1024
PREVIEW_MAX_BYTES = 8 * 1024 * 1024
_IDLE_STATES = {"idle", "stopped", "ended", "error", "configuration_error"}
_NO_STORE = {"Cache-Control": "no-store"}


def _positive_environment_integer(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


async def _json_body(request: Request) -> dict[str, Any]:
    """Bound the bytes actually received, including chunked request bodies."""
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > CONTROL_MAX_BYTES:
            raise HTTPException(413, "camera control body exceeds 64 KiB")
        body.extend(chunk)
    try:
        payload = json.loads(body or b"{}")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "camera control body must be JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "camera control body must be an object")
    return payload


class FleetController:
    def __init__(self, application: FastAPI) -> None:
        self.application = application
        self.capacity = _positive_environment_integer("HARNESS_FLEET_CAPACITY", 1)
        self.camera_index = _positive_environment_integer(
            "HARNESS_FLEET_CAMERA_INDEX",
            1,
        )
        if self.camera_index > self.capacity:
            raise ValueError("HARNESS_FLEET_CAMERA_INDEX exceeds fleet capacity")
        self.control_lock = asyncio.Lock()
        self._peer_client: httpx.AsyncClient | None = None
        self._own_client: httpx.AsyncClient | None = None

    @property
    def is_primary(self) -> bool:
        return self.camera_index == 1

    def _client(self, index: int) -> httpx.AsyncClient:
        if index == self.camera_index:
            if self._own_client is None:
                self._own_client = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=self.application),
                    base_url="http://fleet-self",
                    trust_env=False,
                    follow_redirects=False,
                )
            return self._own_client
        if self._peer_client is None:
            self._peer_client = httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(60, connect=3, pool=3),
                limits=httpx.Limits(
                    max_connections=max(16, self.capacity * 2 + 8),
                    max_keepalive_connections=max(8, self.capacity),
                ),
            )
        return self._peer_client

    async def aclose(self) -> None:
        for client in (self._peer_client, self._own_client):
            if client is not None:
                await client.aclose()
        self._peer_client = self._own_client = None

    def _url(self, index: int, path: str) -> str:
        if index == self.camera_index:
            return f"http://fleet-self{path}"
        service = "harness" if index == 1 else f"harness-{index:02d}"
        return f"http://{service}:8080{path}"

    def resolve_camera(self, camera_id: str) -> int:
        # Canonical equality rejects signs, alternate zero padding and path data.
        try:
            index = int(camera_id.removeprefix("camera"))
        except ValueError:
            raise HTTPException(404, "camera not provisioned") from None
        if not 1 <= index <= self.capacity or camera_id != f"camera{index:02d}":
            raise HTTPException(404, "camera not provisioned")
        return index

    def multi_camera_allowed(self) -> bool:
        runtime = self.application.state.runtime
        return bool(
            self.is_primary
            and getattr(runtime, "_local_tracking_enabled", False)
            and getattr(getattr(runtime, "_local_gpu", None), "available", False)
            and not getattr(runtime, "_local_configuration_error", None)
        )

    def _require_primary(self) -> None:
        if not self.is_primary:
            raise HTTPException(409, "请使用主控制台管理视频通道。")

    def require_enabled(self, index: int) -> None:
        self._require_primary()
        if index != 1 and not self.multi_camera_allowed():
            raise HTTPException(409, "多路视频需要启用本地 NVIDIA GPU 人物追踪。")

    def descriptor(self) -> dict[str, Any]:
        self._require_primary()
        allowed = self.multi_camera_allowed()
        return {
            "multi_camera_allowed": allowed,
            "capacity": self.capacity,
            "mode": "multi" if allowed else "single",
            "reason": "" if allowed else "人物追踪使用远程服务时仅支持一路视频。",
            "cameras": [
                self._camera_descriptor(index, allowed)
                for index in range(1, self.capacity + 1)
            ],
        }

    def _camera_descriptor(self, index: int, allowed: bool) -> dict[str, Any]:
        camera_id = f"camera{index:02d}"
        base = f"/v1/fleet/cameras/{camera_id}"
        return {
            "id": camera_id,
            "index": index,
            "label": f"视频 {index}",
            "enabled": index == 1 or allowed,
            "own": index == self.camera_index,
            "status_url": f"{base}/status",
            "start_url": f"{base}/start",
            "stop_url": f"{base}/stop",
            "settings_url": f"{base}/settings",
            "preview_url": f"{base}/video/annotated/latest.jpg",
            "console_url": "/" if index == self.camera_index else None,
        }

    async def request_json(
        self,
        index: int,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 60,
    ) -> JSONResponse:
        try:
            async with self._client(index).stream(
                method,
                self._url(index, path),
                json=payload,
                timeout=httpx.Timeout(timeout, connect=3, pool=3),
            ) as upstream:
                if 300 <= upstream.status_code < 400:
                    raise HTTPException(502, "camera service returned a redirect")
                body = bytearray()
                async for chunk in upstream.aiter_bytes():
                    if len(body) + len(chunk) > RESPONSE_MAX_BYTES:
                        raise HTTPException(502, "camera response exceeds size limit")
                    body.extend(chunk)
                try:
                    result = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    raise HTTPException(502, "camera response must be JSON") from None
                return JSONResponse(
                    public_safe_value(result),
                    status_code=upstream.status_code,
                    headers=_NO_STORE,
                )
        except httpx.HTTPError:
            raise HTTPException(503, "视频通道暂时无法连接。请检查本地服务。") from None

    async def assert_can_disable_multicamera(self) -> None:
        """Called with control_lock held, before changing the primary GPU mode."""
        if not self.is_primary or self.capacity == 1:
            return
        # A bounded fan-out also handles deployments larger than twelve cameras.
        semaphore = asyncio.Semaphore(16)

        async def camera_idle(index: int) -> bool:
            async with semaphore:
                try:
                    response = await self.request_json(
                        index,
                        "GET",
                        "/v1/status",
                        timeout=5,
                    )
                    status = json.loads(response.body)
                except (HTTPException, ValueError):
                    return False
                return bool(
                    response.status_code == 200
                    and isinstance(status, dict)
                    and status.get("pipeline_state") in _IDLE_STATES
                    and status.get("processing_cleanup")
                    not in {
                        "closing",
                        "stopping",
                    }
                )

        checks = await asyncio.gather(
            *(camera_idle(index) for index in range(2, self.capacity + 1))
        )
        if not all(checks):
            raise HTTPException(
                409,
                "请先停止其他视频通道。无法连接的通道需恢复连接并确认已停止。"
                "然后再修改全局运行设置。",
            )

    async def synchronize_camera(self, index: int) -> None:
        """Copy global inference choices without replacing camera-local defaults."""
        runtime = self.application.state.runtime
        route = runtime.route_probe_status()
        if route.get("state") == "running":
            raise HTTPException(409, "请等待大模型线路检查完成后再连接视频。")
        current = await self.request_json(index, "GET", "/v1/status", timeout=5)
        status = json.loads(current.body)
        if (
            current.status_code != 200
            or not isinstance(status, dict)
            or status.get("pipeline_state") not in _IDLE_STATES
        ):
            raise HTTPException(409, "请先停止此视频通道。等待清理完成后再连接。")
        payload: dict[str, Any] = {
            "local_tracking_enabled": runtime._local_tracking_enabled,
            "local_inference_enabled": runtime._local_inference_enabled,
            "dry_run": bool(getattr(runtime, "dry_run", False)),
            "autostart": False,
        }
        key = getattr(runtime, "_aiclub_api_key", None)
        if key:
            payload["api_key"] = key
        else:
            payload["clear_api_key"] = True
        response = await self.request_json(
            index,
            "PUT",
            "/v1/deployment/settings",
            payload,
        )
        if response.status_code != 200:
            raise HTTPException(409, "视频通道尚未准备好。请稍后重试连接。")
        if route.get("supported") and route.get("selected_id"):
            response = await self.request_json(
                index,
                "POST",
                "/v1/deployment/routes/select",
                {"route_id": route["selected_id"]},
            )
            if response.status_code != 200:
                raise HTTPException(409, "视频通道的大模型线路同步失败。请重试。")

    async def preview(self, index: int) -> Response:
        self.require_enabled(index)
        try:
            async with self._client(index).stream(
                "GET",
                self._url(index, "/v1/video/annotated/latest.jpg"),
                timeout=httpx.Timeout(5, connect=3, pool=3),
            ) as upstream:
                if upstream.status_code == 404:
                    raise HTTPException(404, "视频通道尚无预览画面。")
                if (
                    upstream.status_code != 200
                    or upstream.headers.get("content-type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                    != "image/jpeg"
                ):
                    raise HTTPException(502, "视频通道未返回可用的预览。")
                body = bytearray()
                async for chunk in upstream.aiter_bytes():
                    if len(body) + len(chunk) > PREVIEW_MAX_BYTES:
                        raise HTTPException(502, "视频预览超过大小限制。")
                    body.extend(chunk)
                return Response(
                    bytes(body),
                    media_type="image/jpeg",
                    headers=_NO_STORE,
                )
        except httpx.HTTPError:
            raise HTTPException(503, "视频预览暂时无法连接。") from None


def install_fleet_routes(application: FastAPI) -> FleetController:
    controller = FleetController(application)
    application.state.fleet = controller

    @application.get("/v1/fleet")
    async def fleet() -> JSONResponse:
        return JSONResponse(controller.descriptor(), headers=_NO_STORE)

    @application.get("/v1/fleet/cameras/{camera_id}/status")
    async def status(camera_id: str) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        controller._require_primary()
        # Status and stop remain reachable if a mode change is detected, so an
        # already-running extra camera can always be inspected and stopped.
        return await controller.request_json(index, "GET", "/v1/status", timeout=5)

    @application.get("/v1/fleet/cameras/{camera_id}/settings")
    async def settings(camera_id: str) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        return await controller.request_json(index, "GET", "/v1/deployment/settings")

    @application.put("/v1/fleet/cameras/{camera_id}/settings")
    async def save_settings(camera_id: str, request: Request) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        payload = await _json_body(request)
        if index != 1 and {
            "local_tracking_enabled",
            "local_inference_enabled",
            "dry_run",
        }.intersection(payload):
            raise HTTPException(400, "请在主控制台设置全局运行模式。")
        if index != 1 and payload.get("autostart") is True:
            raise HTTPException(400, "其他视频通道需通过主控制台连接。")
        # The primary settings handler holds the shared lock for mode changes.
        if index == 1:
            return await controller.request_json(
                index,
                "PUT",
                "/v1/deployment/settings",
                payload,
            )
        async with controller.control_lock:
            controller.require_enabled(index)
            return await controller.request_json(
                index,
                "PUT",
                "/v1/deployment/settings",
                payload,
            )

    @application.post("/v1/fleet/cameras/{camera_id}/start")
    async def start(camera_id: str, request: Request) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        payload = await _json_body(request)
        async with controller.control_lock:
            controller.require_enabled(index)
            if index != 1:
                await controller.synchronize_camera(index)
            return await controller.request_json(
                index,
                "POST",
                "/v1/sessions/start",
                payload,
            )

    @application.post("/v1/fleet/cameras/{camera_id}/stop")
    async def stop(camera_id: str, request: Request) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        controller._require_primary()
        await _json_body(request)
        return await controller.request_json(index, "POST", "/v1/sessions/stop")

    @application.get("/v1/fleet/cameras/{camera_id}/video/annotated/latest.jpg")
    async def preview(camera_id: str) -> Response:
        return await controller.preview(controller.resolve_camera(camera_id))

    return controller
