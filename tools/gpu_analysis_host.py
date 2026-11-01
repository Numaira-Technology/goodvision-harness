"""Run isolated Harness analysis lanes on the cloud GPU host.

The management listener is loopback-only and reachable through the controller's
SSH connection. SAM keeps its separate, pinned Python environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException


def lane_environment(base, root: Path, index: int, streams: int):
    if not 1 <= index <= streams <= 55536:
        raise ValueError("analysis lanes exceed available TCP ports")
    data = root / "data" / f"camera-{index:02d}"
    data.mkdir(parents=True, exist_ok=True)
    origins = [f"http://127.0.0.1:{10000 + lane}" for lane in range(streams)]
    environment = {
        **base,
        "HARNESS_ANALYSIS_URL": "",
        "HARNESS_CONFIG": str(root / "source/config.yaml"),
        "HARNESS_HOST": "127.0.0.1",
        "HARNESS_PORT": str(9999 + index),
        "HARNESS_AUTOSTART": "false",
        "HARNESS_MANAGED_ANALYSIS": "true",
        "HARNESS_DRY_RUN": base.get("HARNESS_DRY_RUN", "false"),
        "HARNESS_DEPLOYMENT_SETTINGS_PATH": str(data / "deployment-settings.json"),
        "HARNESS_FLEET_CAPACITY": str(streams),
        "HARNESS_FLEET_CAMERA_INDEX": str(index),
        "HARNESS_FLEET_PRIMARY_URL": origins[0],
        "HARNESS_FLEET_ORIGINS": json.dumps(origins),
        "HARNESS_SOURCE_ID": f"camera-{index:02d}",
        "HARNESS_STREAM_URL": base.get(f"LOCAL_CAMERA_{index:02d}_STREAM_URL")
        or (base.get("HARNESS_STREAM_URL", "") if index == 1 else ""),
        "RTSP_URL": base.get("RTSP_URL", "") if index == 1 else "",
        "FACE_PROVIDER": "CPUExecutionProvider",
        "FACE_MODEL_ROOT": str(root / "models/face"),
        "FACE_GALLERY_PATH": str(data / "face-gallery.json"),
        "EVENT_EVIDENCE_DIRECTORY": str(data / "event-evidence"),
        "GOODVISION_SAM_SPOOL_DIRECTORY": str(data / "sam-spool"),
        "LOCAL_GPU_AVAILABLE": "false",
        "LOCAL_GPU_COUNT": "0",
        "LOCAL_TRACKING_ENABLED": "false",
        "LOCAL_INFERENCE_ENABLED": "false",
        "TRACKING_EXECUTION_MODE": "cloud",
        "CLOUD_TRACKING_URL": "http://127.0.0.1:8092",
        "CLOUD_GPU_CONTROLLER_URL": "http://127.0.0.1:8096",
        "RICH_LABEL_ENABLED": "true",
        "RICH_LABEL_BACKEND": "sam31_stream",
        "RICH_LABEL_ENDPOINT": "http://127.0.0.1:8092/v1/video/sessions",
        "PERCEPTION_METHOD": "sam_live",
        "SCENE_ENGINE_MODE": "active",
        "PERSON_TRACKING_ENABLED": "true",
        "SAM31_TRANSPORT_CODEC": "h264",
    }
    return environment


def create_app(root: Path, streams: int, base):
    processes = []
    lock = asyncio.Lock()
    stopped = False

    async def stop():
        nonlocal stopped
        async with lock:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            # Signal all lanes before waiting, so shutdown time is per fleet.
            try:
                await asyncio.gather(
                    *(
                        asyncio.to_thread(process.wait, timeout=100)
                        for process in processes
                    )
                )
            except subprocess.TimeoutExpired:
                # No snapshot may be accepted while a writer remains alive.
                raise HTTPException(
                    503, "analysis processes have not finished draining"
                ) from None
            stopped = True

    @asynccontextmanager
    async def lifespan(app):
        try:
            for index in range(1, streams + 1):
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-m", "goodvision_harness.service"],
                        env=lane_environment(base, root, index, streams),
                        cwd=root / "source",
                        stdin=subprocess.DEVNULL,
                    )
                )
            yield
        finally:
            await stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        ready = (
            not stopped
            and len(processes) == streams
            and all(p.poll() is None for p in processes)
        )
        if ready:
            async with httpx.AsyncClient(trust_env=False, timeout=3) as client:

                async def probe(index):
                    try:
                        result = await client.get(
                            f"http://127.0.0.1:{9999 + index}/health/live"
                        )
                        return result.status_code == 200
                    except httpx.HTTPError:
                        return False

                ready = all(
                    await asyncio.gather(*(probe(i) for i in range(1, streams + 1)))
                )
        return {"ready": ready, "stopped": stopped, "streams": streams}

    @app.post("/stop")
    async def shutdown():
        await stop()
        return {"stopped": True}

    return app


if __name__ == "__main__":
    import uvicorn

    root = Path(os.environ["GPU_ANALYSIS_ROOT"])
    base = {**os.environ, **json.loads((root / "environment.json").read_text("utf-8"))}
    uvicorn.run(
        create_app(root, int(os.environ["SAM31_LIVE_CAMERAS"]), base),
        host="127.0.0.1",
        port=8095,
        timeout_graceful_shutdown=110,
    )
