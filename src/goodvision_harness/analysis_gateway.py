"""A console/video gateway with no local analysis runtime or model fallback."""

# ruff: noqa: RUF001 -- User-facing Chinese punctuation.

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.websockets import WebSocketDisconnect

from .gateway_video import CONNECT, DISCONNECT, GatewayVideo
from .service_locations import _service_origin

HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
}
UNAVAILABLE = "GPU 主机上的分析服务不可用，请检查服务器或启动云 GPU。"


def forwarded_headers(headers):
    excluded = HOP_HEADERS | {
        part.strip().lower() for part in headers.get("connection", "").split(",")
    }
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


def create_app(origin=None, *, controller=None, transport=None):
    origin = _service_origin(
        origin or os.environ.get("HARNESS_ANALYSIS_URL", ""),
        "HARNESS_ANALYSIS_URL",
    )
    if not origin:
        raise ValueError("HARNESS_ANALYSIS_URL is required for the analysis gateway")
    controller = _service_origin(
        controller or os.environ.get("CLOUD_GPU_CONTROLLER_URL", ""),
        "CLOUD_GPU_CONTROLLER_URL",
    )
    video = GatewayVideo(origin)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            transport=transport,
            timeout=httpx.Timeout(None, connect=5, write=60, pool=5),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=32),
        ) as client:
            app.state.client = client
            try:
                yield
            finally:
                await video.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.control_lock = asyncio.Lock()
    app.state.video = video

    @app.get("/gateway/video-source")
    async def video_source():
        return video.settings()

    @app.put("/gateway/video-source")
    async def set_video_source(request: Request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 1024:
                raise HTTPException(413, "video settings too large")
        try:
            return video.save(json.loads(data))
        except (ValueError, TypeError):
            raise HTTPException(400, "pull_location must be local or gpu") from None

    async def require_remote():
        if controller:
            try:
                response = await app.state.client.get(
                    controller + "/settings?check_health=false",
                    timeout=5,
                )
                if (
                    response.status_code == 200
                    and response.json().get("state") == "ready"
                ):
                    return
            except (httpx.HTTPError, ValueError):
                pass
            raise HTTPException(503, UNAVAILABLE)

    @app.get("/health/live")
    async def live():
        return {
            "status": "ok",
            "execution_location": "gpu_host",
            "local_analysis": False,
        }

    @app.get("/gateway/status")
    async def remote_status():
        await require_remote()
        try:
            response = await app.state.client.get(origin + "/health/live", timeout=5)
            if response.status_code == 200:
                return {"ready": True, "execution_location": "gpu_host"}
        except httpx.HTTPError:
            pass
        raise HTTPException(503, UNAVAILABLE)

    async def controller_request(method, path, payload=None):
        if not controller:
            raise HTTPException(503, "当前部署未启用云 GPU 控制器")
        try:
            response = await app.state.client.request(
                method,
                controller + path,
                json=payload,
                # Current controllers acknowledge background release; retain
                # this budget for older synchronous controller deployments.
                timeout=450 if path == "/stop" else 180,
            )
            return JSONResponse(response.json(), status_code=response.status_code)
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "云 GPU 控制器暂时不可用") from None

    @app.get("/v1/cloud-gpu")
    async def cloud_status():
        if not controller:
            return {"available": False}
        return await controller_request("GET", "/settings")

    @app.put("/v1/cloud-gpu")
    @app.post("/v1/cloud-gpu/{action}")
    async def cloud_control(request: Request, action: str = ""):
        if action not in {"", "start", "stop"}:
            raise HTTPException(404, "unknown cloud GPU action")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                raise HTTPException(413, "租用配置过大")
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(400, "租用配置必须是 JSON") from None
        if not isinstance(payload, dict) or (action and payload):
            raise HTTPException(400, "请先保存租用配置")
        async with app.state.control_lock:
            # The controller drains and backs up the complete remote runtime
            # before releasing it; the gateway never owns analysis state.
            return await controller_request(
                request.method,
                "/" + action if action else "/settings",
                payload,
            )

    @app.get("/gateway")
    async def gateway_page():
        return HTMLResponse(
            Path(__file__).with_name("analysis_gateway.html").read_text("utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/assets/cloud_gpu_console.js")
    async def cloud_script():
        return Response(
            Path(__file__).with_name("cloud_gpu_console.js").read_bytes(),
            media_type="application/javascript",
        )

    @app.websocket("/ws")
    async def camera_socket(socket: WebSocket):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidHandshake

        url = ("wss" if origin.startswith("https:") else "ws") + origin[
            origin.index(":") :
        ]
        try:
            await require_remote()
            async with connect(
                url + "/ws",
                compression=None,
                max_size=16 * 1024 * 1024,
                open_timeout=5,
                proxy=None,
            ) as remote:
                await socket.accept()

                async def upload():
                    while True:
                        message = await socket.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if message.get("text") and video.location == "local":
                            try:
                                command = json.loads(message["text"])
                            except ValueError:
                                command = None
                            if (
                                isinstance(command, dict)
                                and command.get("type") == "start"
                                and command.get("mode") == "stream"
                            ):
                                try:
                                    result = await video.prepare_legacy_start(
                                        app.state.client, command
                                    )
                                except (HTTPException, httpx.HTTPError):
                                    result = JSONResponse({}, status_code=503)
                                if result is not None:
                                    await socket.send_json(
                                        {"type": "error", "message": "视频连接失败"}
                                    )
                                    continue
                        await remote.send(
                            message["bytes"]
                            if message.get("bytes") is not None
                            else message["text"],
                        )

                async def download():
                    async for message in remote:
                        if isinstance(message, bytes):
                            await socket.send_bytes(message)
                        else:
                            await socket.send_text(message)

                tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except (
            OSError,
            TimeoutError,
            ConnectionClosed,
            InvalidHandshake,
            WebSocketDisconnect,
            HTTPException,
        ):
            pass
        finally:
            try:
                await socket.close(code=1013)
            except RuntimeError:
                pass

    @app.api_route(
        "/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    async def forward(request: Request, path: str):
        # Preserve escaped paths and queries without accepting a caller-selected
        # origin. Stream both directions, including camera uploads, SSE and MJPEG.
        url = httpx.URL(origin).copy_with(
            raw_path=request.scope.get("raw_path", request.url.path.encode())
            + (
                b"?" + request.scope["query_string"]
                if request.scope["query_string"]
                else b""
            ),
        )
        try:
            await require_remote()
            content = request.stream()
            if (
                request.method == "POST"
                and video.location == "local"
                and (
                    CONNECT.fullmatch(request.url.path)
                    or request.url.path == "/v1/sessions/start"
                )
            ):
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 64 * 1024:
                        raise HTTPException(413, "video connection body too large")
                try:
                    payload = json.loads(body or b"{}")
                except ValueError:
                    raise HTTPException(400, "video connection must be JSON") from None
                if not isinstance(payload, dict):
                    raise HTTPException(400, "video connection must be an object")
                if CONNECT.fullmatch(request.url.path):
                    return await video.connect(
                        app.state.client,
                        request.url.path,
                        payload,
                        forwarded_headers(request.headers),
                    )
                if payload.get("mode", "stream") == "stream":
                    response = await video.prepare_legacy_start(
                        app.state.client, payload, forwarded_headers(request.headers)
                    )
                    if response is not None:
                        return response
                content = bytes(body)
            upstream = await app.state.client.send(
                app.state.client.build_request(
                    request.method,
                    url,
                    headers=forwarded_headers(request.headers),
                    content=content,
                ),
                stream=True,
            )
        except HTTPException as error:
            if path == "" and request.method == "GET" and error.status_code == 503:
                return await gateway_page()
            raise
        except httpx.HTTPError:
            if path == "" and request.method == "GET":
                return await gateway_page()
            raise HTTPException(503, UNAVAILABLE) from None
        if (
            request.method == "POST"
            and DISCONNECT.fullmatch(request.url.path)
            and upstream.status_code == 200
        ):
            await video.disconnect(request.url.path)

        async def response_body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        headers = forwarded_headers(upstream.headers)
        if headers.get("location", "").startswith(origin + "/"):
            headers["location"] = headers["location"][len(origin) :]
        return StreamingResponse(
            response_body(),
            status_code=upstream.status_code,
            headers=headers,
            background=BackgroundTask(upstream.aclose),
        )

    return app


def main():
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("HARNESS_HOST", "0.0.0.0"),
        port=int(os.environ.get("HARNESS_PORT", "8080")),
        ws_per_message_deflate=False,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
