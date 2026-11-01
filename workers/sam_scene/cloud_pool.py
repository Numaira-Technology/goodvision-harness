"""One isolated SAM process per GPU, with stable camera-to-GPU routing."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

if __package__:
    from .continuous import _session_ref, _timing_log, _trace_identifier
else:
    from continuous import _session_ref, _timing_log, _trace_identifier

LOGGER = logging.getLogger("goodvision.sam31.pool")

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


def assignment(camera: int, gpu_count: int, streams: int):
    if not 1 <= camera <= streams:
        raise HTTPException(404, "camera not provisioned")
    return (camera - 1) % gpu_count, (camera - 1) // gpu_count + 1


def worker_plan(gpu_count: int, streams: int):
    if gpu_count < 1 or streams < 1 or gpu_count > 57436:
        raise ValueError("invalid cloud GPU topology")
    return [
        {
            "gpu": gpu,
            "port": 8100 + gpu,
            # Two lanes keeps the named-camera API available even on an idle GPU.
            "lanes": max(2, (streams + gpu_count - 1 - gpu) // gpu_count),
        }
        for gpu in range(gpu_count)
    ]


def create_app(gpu_count: int, streams: int, *, launch_workers=True, transport=None):
    plans = worker_plan(gpu_count, streams)

    @asynccontextmanager
    async def lifespan(app):
        processes = []
        try:
            if launch_workers:
                import torch

                if torch.cuda.device_count() != gpu_count:
                    raise RuntimeError(
                        "Visible GPU count differs from the rental request"
                    )
                visible = [
                    value.strip()
                    for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                    if value.strip()
                ]
                for plan in plans:
                    device = (
                        visible[plan["gpu"]]
                        if len(visible) == gpu_count
                        else str(plan["gpu"])
                    )
                    environment = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=device,
                        SAM31_LIVE_CAMERAS=str(plan["lanes"]),
                    )
                    processes.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "uvicorn",
                                "app:app",
                                "--host",
                                "127.0.0.1",
                                "--port",
                                str(plan["port"]),
                                "--workers",
                                "1",
                            ],
                            env=environment,
                        )
                    )
            app.state.processes = processes
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                transport=transport,
                timeout=httpx.Timeout(90, connect=3),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
            ) as client:
                app.state.client = client
                yield
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    await asyncio.to_thread(process.wait, timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/perception/sam_live")
    @app.get("/health/cameras")
    async def health():
        async def probe(plan):
            try:
                response = await app.state.client.get(
                    f"http://127.0.0.1:{plan['port']}/health/cameras",
                    timeout=3,
                )
                data = response.json()
                return response.status_code == 200 and all(
                    data.get(key) is True for key in ("enabled", "ready", "gpu_ready")
                )
            except (httpx.HTTPError, ValueError):
                return False

        states = await asyncio.gather(*(probe(plan) for plan in plans))
        ready = all(states) and all(p.poll() is None for p in app.state.processes)
        # Retain the legacy single-camera protocol/limits for direct CLI clients.
        first = {}
        try:
            response = await app.state.client.get(
                "http://127.0.0.1:8100/health/cameras/camera-01/perception/sam_live",
                timeout=3,
            )
            if response.status_code == 200:
                first = response.json()
                ready = (
                    ready
                    and isinstance(first, dict)
                    and all(
                        first.get(key) is True
                        for key in ("enabled", "ready", "gpu_ready")
                    )
                )
                if not isinstance(first, dict):
                    first = {}
            else:
                ready = False
        except (httpx.HTTPError, ValueError):
            ready = False
        return JSONResponse(
            {
                **first,
                "enabled": True,
                "ready": ready,
                "gpu_ready": ready,
                "model_instances": gpu_count,
                "max_gpu_operations": gpu_count,
                "max_active_sessions": streams,
                "gpus": [
                    {"index": i, "ready": state} for i, state in enumerate(states)
                ],
            },
            status_code=200 if ready else 503,
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy(path: str, request: Request):
        started = time.perf_counter()
        path = "/" + path
        match = re.match(r"^/(v1|health)/cameras/(camera-(\d+))(/.*)$", path)
        gpu = 0
        camera_id = worker_camera_id = "camera-01"
        if match:
            camera = int(match[3])
            if match[2] != f"camera-{camera:02d}":
                raise HTTPException(404, "camera not provisioned")
            gpu, lane = assignment(camera, gpu_count, streams)
            camera_id, worker_camera_id = match[2], f"camera-{lane:02d}"
            path = f"/{match[1]}/cameras/camera-{lane:02d}{match[4]}"
        elif path.startswith("/v1/video/sessions"):
            path = path.replace(
                "/v1/video/sessions",
                "/v1/cameras/camera-01/perception/sam_live/sessions",
                1,
            )
        elif path.startswith("/v1/perception/sam_live/sessions"):
            path = path.replace(
                "/v1/perception/sam_live",
                "/v1/cameras/camera-01/perception/sam_live",
                1,
            )
        elif path != "/v1/perception/routes":
            raise HTTPException(404, "unknown cloud tracking route")
        # Observe session traffic, not periodic health polling. No body parsing
        # or buffering is needed to correlate the existing append request ID.
        session_match = re.search(r"/sessions(?:/([^/]+))?", path)
        observe = session_match is not None
        request_bytes = response_bytes = 0
        forward_started = forward_finished = headers_finished = None
        stream_started = stream_finished = None
        stream_outcome = "completed"
        fields = {
            "operation": (
                "append" if path.endswith("/append") else
                "delete" if request.method == "DELETE" else "create"
            ),
            "transport_request_id": _trace_identifier(
                request.headers.get("X-Goodvision-Request-ID")
            ),
            "http_attempt_id": _trace_identifier(
                request.headers.get("X-Goodvision-HTTP-Attempt-ID")
            ),
            "camera_id": camera_id,
            "worker_camera_id": worker_camera_id,
            "gpu_index": gpu,
            "session_ref": _session_ref(
                session_match[1] if session_match else None
            ),
            "timing_basis": "host_wall_clock",
        }

        def log_proxy(outcome, status_code, *, asgi_send_seconds=0.0, error=None):
            if not observe:
                return
            finished = time.perf_counter()
            _timing_log(
                "SAM cloud proxy timing",
                fields | {
                    "outcome": outcome,
                    "status_code": status_code,
                    "error_type": type(error).__name__ if error else None,
                    "request_bytes": request_bytes,
                    "response_bytes": response_bytes,
                    # This includes forwarding and worker processing; it must
                    # not be labelled pure network or GPU execution time.
                    "upstream_headers_seconds": (
                        (headers_finished or finished) - started
                    ),
                    "request_forward_seconds": (
                        (forward_finished or finished) - forward_started
                        if forward_started is not None else 0.0
                    ),
                    # These spans overlap and include downstream backpressure.
                    "response_stream_seconds": (
                        (stream_finished or finished) - stream_started
                        if stream_started is not None else 0.0
                    ),
                    "asgi_send_seconds": asgi_send_seconds,
                    "total_seconds": finished - started,
                },
                level=logging.INFO if outcome == "completed" else logging.WARNING,
                logger=LOGGER,
            )

        if app.state.processes and app.state.processes[gpu].poll() is not None:
            log_proxy("worker_exited", 503)
            raise HTTPException(
                503, "assigned GPU worker exited; session cannot migrate"
            )
        client = app.state.client

        async def forward_body():
            nonlocal request_bytes, forward_started, forward_finished
            forward_started = time.perf_counter()
            try:
                async for chunk in request.stream():
                    request_bytes += len(chunk)
                    yield chunk
            finally:
                forward_finished = time.perf_counter()

        upstream = client.build_request(
            request.method,
            f"http://127.0.0.1:{8100 + gpu}{path}",
            params=request.query_params,
            headers={
                k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
            },
            content=forward_body(),
        )
        try:
            response = await client.send(upstream, stream=True)
            headers_finished = time.perf_counter()
        except httpx.HTTPError as exc:
            log_proxy("upstream_failed", 503, error=exc)
            raise HTTPException(503, "assigned GPU worker unavailable") from None
        except asyncio.CancelledError as exc:
            log_proxy("cancelled", 499, error=exc)
            raise

        async def stream_body():
            nonlocal response_bytes, stream_started, stream_finished, stream_outcome
            stream_started = time.perf_counter()
            try:
                async for chunk in response.aiter_raw():
                    response_bytes += len(chunk)
                    yield chunk
            except BaseException:
                stream_outcome = "stream_failed"
                raise
            finally:
                stream_finished = time.perf_counter()

        class TimedResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                send_started = time.perf_counter()
                outcome = "completed"
                error = None
                try:
                    return await super().__call__(scope, receive, send)
                except BaseException as exc:
                    outcome, error = "send_failed", exc
                    raise
                finally:
                    log_proxy(
                        stream_outcome if outcome == "completed" else outcome,
                        self.status_code,
                        asgi_send_seconds=time.perf_counter() - send_started,
                        error=error,
                    )

        return TimedResponse(
            stream_body(),
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in HOP_HEADERS
            },
            background=BackgroundTask(response.aclose),
        )

    return app


app = create_app(
    int(os.environ.get("SAM_GPU_COUNT", "1")),
    int(os.environ.get("SAM31_LIVE_CAMERAS", "1")),
)
