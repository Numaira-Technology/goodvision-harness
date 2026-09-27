"""Authenticated standalone gateway and console. One process per node."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import time
from collections import deque
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException

from . import engine_adapter
from .contracts import GPUMemoryStatus, Limits
from .settings import MODEL_ID, PRODUCT, ROOT, VERSION, Settings

ERRORS = {
    400: "invalid_request",
    401: "invalid_access_key",
    404: "not_found",
    405: "method_not_allowed",
    409: "session_conflict",
    413: "request_too_large",
    415: "json_required",
    422: "invalid_request",
    429: "node_busy",
    500: "node_error",
    502: "engine_response_invalid",
    503: "engine_unavailable",
    504: "request_outcome_unknown",
}


def failure(status: int, code: str | None = None):
    headers = {"Retry-After": "1"} if status == 429 else {}
    return JSONResponse(
        {"error": {"code": code or ERRORS.get(status, "request_failed")}},
        status_code=status,
        headers=headers,
    )


def create_app(settings: Settings | None = None, *, transport=None):
    config = settings or Settings.from_env()
    started = time.monotonic()
    events: deque[dict] = deque(maxlen=12)
    counters = {"completed": 0, "failed": 0, "in_flight": 0, "last_request_ms": None}
    accepting = True
    admissions = {
        index: asyncio.Lock() for index in range(1, config.camera_capacity + 1)
    }

    def event(code):
        events.appendleft({"time": int(time.time()), "code": code})

    @asynccontextmanager
    async def lifespan(application):
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=config.request_timeout,
            limits=httpx.Limits(
                max_connections=config.camera_capacity + 4,
                max_keepalive_connections=config.camera_capacity + 2,
            ),
        ) as client:
            application.state.engine = client
            event("node_started")
            yield

    app = FastAPI(
        title=PRODUCT,
        version=VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def access(request: Request, call_next):
        public = {"/", "/console.css", "/console.js", "/health/live", "/favicon.svg"}
        if request.url.path not in public:
            if not config.api_key:
                response = failure(503, "access_key_not_configured")
            elif not hmac.compare_digest(
                request.headers.get("authorization", "").encode(),
                ("Bearer " + config.api_key).encode(),
            ):
                response = failure(401)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'self'"
                ),
            }
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc):
        return failure(exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc):
        return failure(400)

    @app.exception_handler(Exception)
    async def unexpected_error(_request, _exc):
        return failure(500)

    @app.get("/")
    async def console():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/console.css")
    async def style():
        return FileResponse(ROOT / "static/console.css")

    @app.get("/console.js")
    async def script():
        return FileResponse(ROOT / "static/console.js")

    @app.get("/favicon.svg")
    async def icon():
        return FileResponse(ROOT / "static/favicon.svg")

    @app.get("/health/live")
    async def live():
        return {"service": PRODUCT, "version": VERSION, "live": True}

    async def upstream(method, path, body=None, *, timeout=None):
        if not config.engine_url:
            raise HTTPException(503)
        async with asyncio.timeout(timeout or config.request_timeout):
            async with app.state.engine.stream(
                method,
                config.engine_url + path,
                content=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                # Never copy internal errors, redirects, headers or stack traces.
                if response.status_code != 200:
                    status = response.status_code
                    raise HTTPException(status if status in ERRORS else 502)
                size = 0
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > config.max_response_bytes:
                        raise HTTPException(502)
                    data.extend(chunk)
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise ValueError("Expected object")
                return value

    def camera_index(camera_id):
        if not re.fullmatch(r"camera-(?:0[1-9]|[1-9][0-9]+)", camera_id):
            raise HTTPException(404)
        index = int(camera_id.removeprefix("camera-"))
        if index > config.camera_capacity:
            raise HTTPException(404)
        return index

    async def status_snapshot(index=None):
        try:
            size = config.checkpoint_path.stat().st_size
            checkpoint_present = config.checkpoint_path.is_file() and size > 0
        except OSError:
            size, checkpoint_present = 0, False
        state = (
            "engine_unavailable"
            if config.engine_url
            else "model_staged"
            if checkpoint_present
            else "waiting_for_model"
        )
        gpu_ready, active_session, limits = False, None, None
        active_sessions, cameras = None, []
        if config.engine_url:
            try:
                _, health_path = engine_adapter.lane_paths(
                    index or 1, config.camera_capacity
                )
                health = await upstream("GET", health_path, timeout=3)
                limits = Limits.model_validate(health["limits"]).model_dump()
                gpu_ready = (
                    health.get("enabled") is True
                    and health.get("ready") is True
                    and health.get("gpu_ready") is True
                    and health.get("backend") == engine_adapter.BACKEND
                )
                if type(health.get("active_session")) is not bool:
                    raise ValueError("Missing session ownership")
                active_session = health["active_session"]
                active_sessions = int(active_session)
                if index is None and config.camera_capacity > 1:
                    fleet = await upstream("GET", "/health/cameras", timeout=3)
                    rows = fleet["cameras"]
                    if (
                        type(fleet.get("max_active_sessions")) is not int
                        or fleet["max_active_sessions"] != config.camera_capacity
                        or not isinstance(rows, list)
                        or len(rows) != config.camera_capacity
                    ):
                        raise ValueError("Unexpected node capacity")
                    for number, row in enumerate(rows, 1):
                        if (
                            row.get("id") != f"camera-{number:02d}"
                            or type(row.get("ready")) is not bool
                            or type(row.get("active_session")) is not bool
                        ):
                            raise ValueError("Invalid camera state")
                        cameras.append(
                            {key: row[key] for key in ("id", "ready", "active_session")}
                        )
                    gpu_ready = (
                        gpu_ready
                        and fleet.get("enabled") is True
                        and fleet.get("ready") is True
                        and fleet.get("gpu_ready") is True
                        and all(row["ready"] for row in cameras)
                    )
                    active_sessions = sum(row["active_session"] for row in cameras)
                    active_session = active_sessions > 0
                state = "ready" if gpu_ready else "engine_unavailable"
            except (
                httpx.HTTPError,
                HTTPException,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                TimeoutError,
            ):
                gpu_ready = False
                active_sessions, cameras = None, []
                state = (
                    "engine_unavailable" if checkpoint_present else "waiting_for_model"
                )
        if gpu_ready and not accepting:
            state = "draining"
        return {
            "service": PRODUCT,
            "version": VERSION,
            "model_id": MODEL_ID,
            "state": state,
            "ready": gpu_ready and accepting,
            "gpu_ready": gpu_ready,
            "accepting_sessions": accepting,
            "active_session": active_session,
            "camera_capacity": config.camera_capacity,
            "active_sessions": active_sessions,
            "cameras": cameras,
            "checkpoint": {"present": checkpoint_present, "bytes": size},
            "mode": "connected" if config.engine_url else "framework",
            "uptime_seconds": int(time.monotonic() - started),
            "limits": limits,
            "requests": dict(counters),
            "events": list(events),
        }

    @app.get("/api/gpu-memory")
    async def gpu_memory():
        try:
            value = await upstream("GET", "/health/gpu-memory", timeout=3)
            result = GPUMemoryStatus.model_validate(value).model_dump()
            if result["available"] and any(
                result[name] is None
                for name in (
                    "allocated_bytes",
                    "reserved_bytes",
                    "unused_reserved_bytes",
                    "peak_allocated_bytes",
                    "device_free_bytes",
                    "device_total_bytes",
                )
            ):
                return failure(502)
            return result
        except HTTPException as exc:
            return failure(exc.status_code)
        except (httpx.HTTPError, TimeoutError):
            return failure(503)
        except (ValueError, TypeError):
            return failure(502)

    @app.get("/api/status")
    async def status():
        return await status_snapshot()

    @app.get("/api/cameras/{camera_id}/status")
    async def camera_status(camera_id: str):
        return await status_snapshot(camera_index(camera_id))

    @app.get("/health/ready")
    async def ready():
        value = await status_snapshot()
        return JSONResponse(value, status_code=200 if value["ready"] else 503)

    @app.get("/v1/models")
    async def models():
        value = await status_snapshot()
        return {
            "models": [
                {
                    "id": MODEL_ID,
                    "name": PRODUCT,
                    "ready": value["ready"],
                    "capabilities": [
                        "video_sessions",
                        "person_tracking",
                        "mask_contours",
                    ],
                }
            ]
        }

    async def read_body(request, bound):
        if request.method != "DELETE" and (
            request.headers.get("content-type", "").split(";")[0].lower()
            != "application/json"
        ):
            raise HTTPException(415)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > bound:
                raise HTTPException(413)
            body.extend(chunk)
        return bytes(body)

    @app.post("/api/admission")
    async def set_admission(request: Request):
        nonlocal accepting
        try:
            async with asyncio.timeout(5):
                value = json.loads(await read_body(request, 1024))
            if (
                type(value) is not dict
                or set(value) != {"accepting_sessions"}
                or type(value["accepting_sessions"]) is not bool
            ):
                return failure(400)
            accepting = value["accepting_sessions"]
            event("admission_opened" if accepting else "admission_paused")
            return {"accepting_sessions": accepting}
        except (ValueError, TimeoutError):
            return failure(400)

    async def forward(request, operation, token=None, *, index=1):
        if operation == "create" and not accepting:
            return failure(503, "new_sessions_paused")
        if not config.engine_url:
            return failure(503, "model_not_installed")
        if token is not None and re.fullmatch(r"[a-f0-9]{32}", token) is None:
            return failure(400)
        admission = admissions[index]
        if admission.locked():
            return failure(429)
        async with admission:
            counters["in_flight"] += 1
            request_started = time.monotonic()
            try:
                async with asyncio.timeout(config.request_timeout):
                    bound = config.max_request_bytes if operation == "append" else 4096
                    body = await read_body(request, bound)
                    if operation == "delete" and body:
                        raise HTTPException(400)
                    path, _ = engine_adapter.lane_paths(index, config.camera_capacity)
                    if token:
                        path += "/" + token
                    if operation == "append":
                        path += "/append"
                    result = await upstream(request.method, path, body)
                    result = engine_adapter.project_response(operation, result)
                counters["completed"] += 1
                if operation != "append":
                    event(
                        "session_opened" if operation == "create" else "session_closed"
                    )
                return JSONResponse(result)
            except HTTPException as exc:
                counters["failed"] += 1
                event("request_rejected")
                return failure(exc.status_code)
            except (httpx.TimeoutException, TimeoutError):
                counters["failed"] += 1
                event("request_timeout")
                return failure(504)
            except httpx.HTTPError:
                counters["failed"] += 1
                event("engine_unreachable")
                return failure(503)
            except (ValueError, KeyError, TypeError, AttributeError):
                counters["failed"] += 1
                event("engine_response_invalid")
                return failure(502)
            finally:
                counters["in_flight"] -= 1
                counters["last_request_ms"] = round(
                    (time.monotonic() - request_started) * 1000, 1
                )

    @app.post("/v1/contour/sessions")
    async def create(request: Request):
        return await forward(request, "create")

    @app.post("/v1/contour/sessions/{session_id}/append")
    async def append(session_id: str, request: Request):
        return await forward(request, "append", session_id)

    @app.delete("/v1/contour/sessions/{session_id}")
    async def delete(session_id: str, request: Request):
        return await forward(request, "delete", session_id)

    @app.post("/v1/cameras/{camera_id}/contour/sessions")
    async def create_camera(camera_id: str, request: Request):
        return await forward(request, "create", index=camera_index(camera_id))

    @app.post("/v1/cameras/{camera_id}/contour/sessions/{session_id}/append")
    async def append_camera(camera_id: str, session_id: str, request: Request):
        return await forward(
            request, "append", session_id, index=camera_index(camera_id)
        )

    @app.delete("/v1/cameras/{camera_id}/contour/sessions/{session_id}")
    async def delete_camera(camera_id: str, session_id: str, request: Request):
        return await forward(
            request, "delete", session_id, index=camera_index(camera_id)
        )

    return app


app = create_app()
