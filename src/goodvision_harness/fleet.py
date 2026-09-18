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
from fastapi.responses import JSONResponse

from .privacy import public_safe_value
from .service_locations import SERVICE_LOCATION_FIELDS

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
        self._output_client: httpx.AsyncClient | None = None
        self._analysis_indices: set[int] = set()
        self._analysis_transition: str | None = None
        self._analysis_errors: list[dict[str, str]] = []
        self._analysis: dict[str, Any] = {
            "state": "idle",
            "camera_ids": [],
            "errors": [],
        }

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

    def output_client(self) -> httpx.AsyncClient:
        if self._output_client is None:
            self._output_client = httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(None, connect=3, pool=3, write=5),
                limits=httpx.Limits(
                    max_connections=max(64, self.capacity * 32),
                    max_keepalive_connections=max(8, self.capacity * 2),
                ),
            )
        return self._output_client

    async def aclose(self) -> None:
        for client in (self._peer_client, self._own_client, self._output_client):
            if client is not None:
                await client.aclose()
        self._peer_client = self._own_client = self._output_client = None

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
        return self.is_primary and self.usable_capacity() > 1

    def tracking_mode(self) -> str:
        runtime = self.application.state.runtime
        settings = getattr(runtime, "service_execution_settings", None)
        if callable(settings):
            return settings().get("tracking_mode", "legacy")
        return (
            "local" if getattr(runtime, "_local_tracking_enabled", False) else "legacy"
        )

    def usable_capacity(self) -> int:
        runtime = self.application.state.runtime
        if getattr(runtime, "_local_configuration_error", None):
            return 1
        capacity = getattr(runtime, "tracking_camera_capacity", None)
        if callable(capacity):
            value = capacity()
            return min(self.capacity, value) if type(value) is int and value > 0 else 1
        return (
            self.capacity
            if (
                getattr(runtime, "_local_tracking_enabled", False)
                and getattr(getattr(runtime, "_local_gpu", None), "available", False)
            )
            else 1
        )

    def _require_primary(self) -> None:
        if not self.is_primary:
            raise HTTPException(409, "请使用主控制台管理视频通道。")

    def require_enabled(self, index: int) -> None:
        self._require_primary()
        if index > self.usable_capacity():
            raise HTTPException(
                409,
                "此通道超出人物追踪服务可用路数。请配置本地或自建追踪服务并扩容。",
            )

    def descriptor(self) -> dict[str, Any]:
        self._require_primary()
        allowed = self.multi_camera_allowed()
        usable = self.usable_capacity()
        tracking_mode = self.tracking_mode()
        reason = ""
        if tracking_mode == "legacy":
            reason = "原远程人物追踪仅支持一路。配置本地或自建追踪服务可使用多路。"
        elif usable < self.capacity:
            reason = f"当前人物追踪服务允许 {usable} 路。其余通道需增加服务容量后使用。"
        elif usable == 1:
            reason = "当前部署配置一路视频。增加部署通道后可连接更多视频。"
        return {
            "multi_camera_allowed": allowed,
            "capacity": self.capacity,
            "usable_capacity": usable,
            "tracking_mode": tracking_mode,
            "mode": "multi" if allowed else "single",
            "reason": reason,
            "analysis": self._analysis,
            "analysis_start_url": "/v1/fleet/analysis/start",
            "analysis_stop_url": "/v1/fleet/analysis/stop",
            "cameras": [
                self._camera_descriptor(index, usable)
                for index in range(1, self.capacity + 1)
            ],
        }

    def _camera_descriptor(self, index: int, usable: int) -> dict[str, Any]:
        from .fleet_outputs import output_urls

        camera_id = f"camera{index:02d}"
        base = f"/v1/fleet/cameras/{camera_id}"
        return {
            "id": camera_id,
            "index": index,
            "label": f"视频 {index}",
            "enabled": index <= usable,
            "own": index == self.camera_index,
            "status_url": f"{base}/status",
            "start_url": f"{base}/start",
            "stop_url": f"{base}/stop",
            "connect_url": f"{base}/connect",
            "disconnect_url": f"{base}/disconnect",
            "detail_url": f"{base}/details",
            "results_url": f"{base}/results/latest",
            "scene_url": f"{base}/scene",
            "settings_url": f"{base}/settings",
            "preview_url": f"{base}/video/annotated/latest.jpg",
            "display_url": f"{base}/video/display/frames",
            "display_metrics_url": f"{base}/video/display/metrics",
            "mjpeg_url": f"{base}/video/annotated/stream.mjpeg",
            "outputs": output_urls(camera_id),
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
        coordinated: bool = False,
    ) -> JSONResponse:
        try:
            async with self._client(index).stream(
                method,
                self._url(index, path),
                json=payload,
                headers={"X-Fleet-Coordinator": "1"} if coordinated else None,
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

    @staticmethod
    def _analysis_state(status: dict[str, Any]) -> str:
        if status.get("session_start_pending"):
            return "starting"
        if status.get("processing_cleanup") in {"closing", "stopping"}:
            return "stopping"
        state = status.get("analysis_state", status.get("pipeline_state"))
        if state in _IDLE_STATES:
            return "idle"
        if state in {"starting", "running", "stopping", "draining"}:
            return "stopping" if state == "draining" else state
        return "unknown"

    async def snapshot(self) -> dict[int, dict[str, Any] | None]:
        """Inspect every provisioned lane, including disabled surviving sessions."""
        semaphore = asyncio.Semaphore(16)

        async def read(index: int) -> tuple[int, dict[str, Any] | None]:
            async with semaphore:
                try:
                    response = await self.request_json(
                        index,
                        "GET",
                        "/v1/status",
                        timeout=5,
                    )
                    status = json.loads(response.body)
                    if response.status_code == 200 and isinstance(status, dict):
                        return index, status
                except (HTTPException, ValueError):
                    pass
                return index, None

        return dict(
            await asyncio.gather(
                *(read(index) for index in range(1, self.capacity + 1))
            )
        )

    def _update_analysis(self, statuses: dict[int, dict[str, Any] | None]) -> None:
        states = {
            index: self._analysis_state(status) if status is not None else "unknown"
            for index, status in statuses.items()
        }
        active = {
            index
            for index, state in states.items()
            if state
            in {
                "starting",
                "running",
                "stopping",
            }
        }
        errors = list(self._analysis_errors)
        for index, state in states.items():
            if state == "unknown":
                errors.append(
                    {
                        "camera_id": f"camera{index:02d}",
                        "message": (
                            "无法确认此路分析状态。请恢复连接后重试停止全部分析。"
                        ),
                    }
                )
        if self._analysis_transition:
            state = self._analysis_transition
        elif active:
            missing = self._analysis_indices - active
            if missing:
                errors.extend(
                    {
                        "camera_id": f"camera{index:02d}",
                        "message": "此路分析已停止。请停止全部分析后重新启动。",
                    }
                    for index in sorted(missing)
                    if states.get(index) == "idle"
                )
            self._analysis_indices.update(active)
            state = (
                "error"
                if errors
                else "stopping"
                if any(states[index] == "stopping" for index in active)
                else "starting"
                if any(states[index] == "starting" for index in active)
                else "running"
            )
        else:
            if all(value == "idle" for value in states.values()):
                self._analysis_indices.clear()
            state = "error" if errors else "idle"
        self._analysis = {
            "state": state,
            "camera_ids": [
                f"camera{index:02d}" for index in sorted(self._analysis_indices)
            ],
            "errors": errors,
        }

    async def refresh_analysis(self) -> dict[int, dict[str, Any] | None]:
        statuses = await self.snapshot()
        self._update_analysis(statuses)
        return statuses

    async def assert_connections_mutable(self) -> None:
        statuses = await self.refresh_analysis()
        if self._analysis_transition or any(
            status is None or self._analysis_state(status) != "idle"
            for status in statuses.values()
        ):
            raise HTTPException(
                409,
                "请先停止全部分析。再连接、断开视频或修改该路设置。",
            )

    async def _analysis_requests(
        self,
        indices: set[int],
        action: str,
    ) -> list[dict[str, str]]:
        semaphore = asyncio.Semaphore(16)

        async def request(index: int) -> dict[str, str] | None:
            async with semaphore:
                try:
                    response = await self.request_json(
                        index,
                        "POST",
                        f"/v1/analysis/{action}",
                        {},
                        timeout=60,
                        coordinated=True,
                    )
                    data = json.loads(response.body)
                    start_rejected = (
                        action == "start"
                        and isinstance(data, dict)
                        and ("analysis_state" in data or "pipeline_state" in data)
                        and self._analysis_state(data) not in {"starting", "running"}
                    )
                    if 200 <= response.status_code < 300 and not start_rejected:
                        return None
                    detail = data.get("detail") if isinstance(data, dict) else None
                    if start_rejected and not detail:
                        detail = "视频分析启动后未处于运行状态。"
                    message = (
                        detail if isinstance(detail, str) else "视频分析操作失败。"
                    )
                except HTTPException as exc:
                    message = str(exc.detail)
                return {"camera_id": f"camera{index:02d}", "message": message}

        results = await asyncio.gather(*(request(index) for index in sorted(indices)))
        return [result for result in results if result is not None]

    async def start_analysis(self) -> JSONResponse:
        """Start the fixed connected cohort only after every lane passes preflight."""
        self._require_primary()
        statuses = await self.refresh_analysis()
        if any(
            status is None or self._analysis_state(status) != "idle"
            for status in statuses.values()
        ):
            raise HTTPException(409, "请先停止全部分析并确认所有通道可连接。")
        if any(
            status and status.get("connection_state") in {"connecting", "disconnecting"}
            for status in statuses.values()
        ):
            raise HTTPException(409, "请等待所有视频完成连接或断开后再开始全部分析。")
        cohort = {
            index
            for index, status in statuses.items()
            if status
            and (
                status.get("connected") is True
                or status.get("connection_state") == "connected"
            )
        }
        if not cohort:
            raise HTTPException(400, "请先连接至少一路视频。再开始全部分析。")
        for index in cohort:
            self.require_enabled(index)
        self._analysis_indices = cohort
        self._analysis_errors = []
        self._analysis_transition = "starting"
        self._update_analysis(statuses)
        try:
            # Global configuration is copied at analysis start. Connecting a
            # video must not require a model key, model route or GPU readiness.
            semaphore = asyncio.Semaphore(16)

            async def synchronize(index: int) -> dict[str, str] | None:
                async with semaphore:
                    try:
                        await self.synchronize_camera(index)
                        return None
                    except HTTPException as exc:
                        return {
                            "camera_id": f"camera{index:02d}",
                            "message": str(exc.detail),
                        }

            synced = await asyncio.gather(
                *(synchronize(index) for index in sorted(cohort - {self.camera_index}))
            )
            errors = [error for error in synced if error is not None]
            if errors:
                self._analysis_errors = errors
                return JSONResponse(
                    {
                        "detail": "全局分析设置同步失败。未启动任何一路分析。",
                        "analysis": {
                            "state": "error",
                            "camera_ids": [],
                            "errors": errors,
                        },
                    },
                    status_code=409,
                    headers=_NO_STORE,
                )
            errors = await self._analysis_requests(cohort, "prepare")
            if errors:
                self._analysis_errors = errors
                return JSONResponse(
                    {
                        "detail": "部分视频尚未准备好。未启动任何一路分析。",
                        "analysis": {
                            "state": "error",
                            "camera_ids": [],
                            "errors": errors,
                        },
                    },
                    status_code=409,
                    headers=_NO_STORE,
                )
            errors = await self._analysis_requests(cohort, "start")
            if errors:
                # A timeout is ambiguous: stop the entire attempted cohort,
                # including lanes whose start response was lost.
                errors.extend(await self._analysis_requests(cohort, "stop"))
                self._analysis_errors = errors
                return JSONResponse(
                    {
                        "detail": "部分视频启动失败。已向全部参与视频发出停止指令。",
                        "analysis": {
                            "state": "error",
                            "camera_ids": [f"camera{i:02d}" for i in sorted(cohort)],
                            "errors": errors,
                        },
                    },
                    status_code=502,
                    headers=_NO_STORE,
                )
            return JSONResponse(
                {
                    "ok": True,
                    "analysis": {
                        "state": "starting",
                        "camera_ids": [f"camera{i:02d}" for i in sorted(cohort)],
                        "errors": [],
                    },
                },
                headers=_NO_STORE,
            )
        except HTTPException as exc:
            self._analysis_errors = [{"camera_id": "", "message": str(exc.detail)}]
            raise
        finally:
            self._analysis_transition = None
            await self.refresh_analysis()

    async def stop_analysis(self) -> JSONResponse:
        """Retry every provisioned lane, so controller restarts cannot strand work."""
        self._require_primary()
        self._analysis_transition = "stopping"
        self._analysis_errors = []
        try:
            self._analysis_errors = await self._analysis_requests(
                set(range(1, self.capacity + 1)),
                "stop",
            )
        finally:
            self._analysis_transition = None
            await self.refresh_analysis()
        return JSONResponse(
            {
                "ok": not self._analysis_errors,
                "analysis": self._analysis,
                **(
                    {"detail": "部分通道未确认停止。请重试停止全部分析。"}
                    if self._analysis_errors
                    else {}
                ),
            },
            status_code=503 if self._analysis_errors else 200,
            headers=_NO_STORE,
        )

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
                    and self._analysis_state(status) == "idle"
                )

        checks = await asyncio.gather(
            *(camera_idle(index) for index in range(2, self.capacity + 1))
        )
        if not all(checks):
            raise HTTPException(
                409,
                "请先停止全部分析。无法连接的通道需恢复连接并确认已停止。"
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
            or self._analysis_state(status) != "idle"
        ):
            raise HTTPException(409, "请先停止此视频通道。等待清理完成后再连接。")
        payload: dict[str, Any] = {
            "local_tracking_enabled": runtime._local_tracking_enabled,
            "local_inference_enabled": runtime._local_inference_enabled,
            "dry_run": bool(getattr(runtime, "dry_run", False)),
            "autostart": False,
        }
        settings = getattr(runtime, "service_execution_settings", None)
        if callable(settings):
            primary = settings()
            payload.update(
                {
                    name: primary[name]
                    for name in SERVICE_LOCATION_FIELDS
                    if name in primary
                }
            )
        key = getattr(runtime, "_aiclub_api_key", None)
        if key:
            payload["api_key"] = key
        else:
            payload["clear_api_key"] = True
        tracking_key = getattr(runtime, "_tracking_api_key", None)
        if tracking_key:
            payload["tracking_api_key"] = tracking_key
        else:
            payload["clear_tracking_api_key"] = True
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



def install_fleet_routes(application: FastAPI) -> FleetController:
    controller = FleetController(application)
    application.state.fleet = controller

    @application.get("/v1/fleet")
    async def fleet() -> JSONResponse:
        controller._require_primary()
        await controller.refresh_analysis()
        return JSONResponse(controller.descriptor(), headers=_NO_STORE)

    @application.post("/v1/fleet/analysis/start")
    async def start_analysis(request: Request) -> JSONResponse:
        payload = await _json_body(request)
        if payload:
            raise HTTPException(
                400, "分析会包含所有已连接视频。请在各路详细控制中保存设置。"
            )
        async with controller.control_lock:
            return await controller.start_analysis()

    @application.post("/v1/fleet/analysis/stop")
    async def stop_analysis(request: Request) -> JSONResponse:
        await _json_body(request)
        async with controller.control_lock:
            return await controller.stop_analysis()

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
        if index != 1 and (
            {
                "local_tracking_enabled",
                "local_inference_enabled",
                "dry_run",
            }
            | SERVICE_LOCATION_FIELDS
        ).intersection(payload):
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
            await controller.assert_connections_mutable()
            return await controller.request_json(
                index,
                "PUT",
                "/v1/deployment/settings",
                payload,
            )

    @application.post("/v1/fleet/cameras/{camera_id}/connect")
    async def connect(camera_id: str, request: Request) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        payload = await _json_body(request)
        async with controller.control_lock:
            controller.require_enabled(index)
            await controller.assert_connections_mutable()
            return await controller.request_json(
                index,
                "POST",
                "/v1/video/connect",
                payload,
                coordinated=True,
            )

    @application.post("/v1/fleet/cameras/{camera_id}/disconnect")
    async def disconnect(camera_id: str, request: Request) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        controller._require_primary()
        await _json_body(request)
        async with controller.control_lock:
            await controller.assert_connections_mutable()
            return await controller.request_json(
                index,
                "POST",
                "/v1/video/disconnect",
                coordinated=True,
            )

    @application.post("/v1/fleet/cameras/{camera_id}/start")
    @application.post("/v1/fleet/cameras/{camera_id}/stop")
    async def old_camera_analysis_control(
        camera_id: str, request: Request
    ) -> JSONResponse:
        controller.resolve_camera(camera_id)
        controller._require_primary()
        await _json_body(request)
        raise HTTPException(409, "分析须统一控制。请使用开始全部分析或停止全部分析。")

    @application.get("/v1/fleet/cameras/{camera_id}/details")
    async def details(camera_id: str) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        controller.require_enabled(index)
        return await controller.request_json(index, "GET", "/v1/analysis/settings")

    @application.put("/v1/fleet/cameras/{camera_id}/details")
    async def save_details(camera_id: str, request: Request) -> JSONResponse:
        index = controller.resolve_camera(camera_id)
        payload = await _json_body(request)
        async with controller.control_lock:
            controller.require_enabled(index)
            await controller.assert_connections_mutable()
            return await controller.request_json(
                index,
                "PUT",
                "/v1/analysis/settings",
                payload,
                coordinated=True,
            )

    @application.post("/v1/fleet/cameras/{camera_id}/video/display/metrics")
    async def display_metrics(camera_id: str, request: Request) -> JSONResponse:
        return await controller.request_json(controller.resolve_camera(camera_id), "POST",
                                             "/v1/video/display/metrics", await _json_body(request))

    from .fleet_outputs import install_output_routes

    install_output_routes(application, controller)
    return controller
