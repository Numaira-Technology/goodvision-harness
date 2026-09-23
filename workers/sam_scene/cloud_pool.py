"""One isolated SAM process per GPU, with stable camera-to-GPU routing."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

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
        path = "/" + path
        match = re.match(r"^/(v1|health)/cameras/(camera-(\d+))(/.*)$", path)
        gpu = 0
        if match:
            camera = int(match[3])
            if match[2] != f"camera-{camera:02d}":
                raise HTTPException(404, "camera not provisioned")
            gpu, lane = assignment(camera, gpu_count, streams)
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
        if app.state.processes and app.state.processes[gpu].poll() is not None:
            raise HTTPException(
                503, "assigned GPU worker exited; session cannot migrate"
            )
        client = app.state.client
        upstream = client.build_request(
            request.method,
            f"http://127.0.0.1:{8100 + gpu}{path}",
            params=request.query_params,
            headers={
                k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
            },
            content=request.stream(),
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError:
            raise HTTPException(503, "assigned GPU worker unavailable") from None
        return StreamingResponse(
            response.aiter_raw(),
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
