"""On-demand, process-isolated camera workers inside the main container.

Only loopback sockets are opened. The registry and camera-local settings live
beside the primary deployment settings, so replacing the image preserves them.
Existing launcher-provisioned peers keep their original destinations and data.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import HTTPException


class DynamicCameras:
    @staticmethod
    def supported() -> bool:
        # Workers inherit a bound socket through pass_fds. Native Windows
        # cannot do this; Linux containers on Windows use the POSIX path.
        return os.name == "posix"

    def __init__(self, fleet, directory: Path):
        self.fleet = fleet
        self.directory = directory
        self.registry = directory / "video-sources.json"
        self.base_capacity = fleet.capacity
        self.processes = {}
        self.origins = {}
        self.locks = {}
        self.closed = False
        if self.registry.exists():
            value = json.loads(self.registry.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != 1
                or type(value.get("capacity")) is not int
                or value["capacity"] < 1
            ):
                raise ValueError("invalid video source registry")
            fleet.capacity = max(self.base_capacity, value["capacity"])

    def owns(self, index):
        return index > self.base_capacity

    def environment(self, index):
        from .deployment_settings import DeploymentSettings, DeploymentSettingsStore

        root = self.directory / "cameras" / f"camera{index:02d}"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings_path = root / "deployment-settings.json"
        if not settings_path.exists():
            # A new channel must never inherit the primary stream URL, including
            # image defaults or RTSP_URL. Shared service choices sync at Start.
            DeploymentSettingsStore(settings_path).save(
                DeploymentSettings(
                    stream_url="",
                    dry_run=False,
                    autostart=False,
                )
            )
        env = dict(os.environ)
        env.update(
            {
                "HARNESS_FLEET_DYNAMIC": "false",
                "HARNESS_FLEET_CAPACITY": str(index),
                "HARNESS_FLEET_CAMERA_INDEX": str(index),
                "HARNESS_FLEET_ORIGINS": "[]",
                "HARNESS_DEPLOYMENT_SETTINGS_PATH": str(settings_path),
                "HARNESS_WORKFLOW_SETTINGS_PATH": str(root / "workflow-assets.json"),
                "HARNESS_SOURCE_ID": f"camera{index:02d}",
                "HARNESS_AUTOSTART": "false",
                "HARNESS_MANAGED_ANALYSIS": "true",
                "HARNESS_STREAM_URL": "",
                "RTSP_URL": "",
                "EVENT_EVIDENCE_DIRECTORY": str(root / "events"),
                "GOODVISION_SAM_SPOOL_DIRECTORY": str(root / "sam-spool"),
                # All cameras read the same people library; only the primary UI
                # exposes enrollment. Models and global admission remain shared.
                "FACE_GALLERY_PATH": str(
                    self.fleet.application.state.runtime.config.face.gallery_path
                ),
            }
        )
        env.pop("HARNESS_ANALYSIS_URL", None)
        env.pop("CLOUD_GPU_CONTROLLER_URL", None)
        return env

    async def ensure(self, index):
        if not self.owns(index):
            return
        async with self.locks.setdefault(index, asyncio.Lock()):
            if self.closed:
                raise HTTPException(503, "Video workers are shutting down.")
            process = self.processes.get(index)
            if process is not None and process.returncode is None:
                return
            # Passing the bound socket avoids a port selection race. Ports are
            # internal and need no Docker mapping or additional containers.
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "goodvision_harness.dynamic_cameras",
                    str(listener.fileno()),
                    env=self.environment(index),
                    pass_fds=(listener.fileno(),),
                    stdin=asyncio.subprocess.DEVNULL,
                )
                self.processes[index] = process
                self.origins[index] = f"http://127.0.0.1:{port}"
            except OSError:
                raise HTTPException(503, "Could not create the video worker.") from None
            finally:
                listener.close()

            async def wait_ready():
                async with httpx.AsyncClient(trust_env=False) as client:
                    while process.returncode is None:
                        try:
                            response = await client.get(
                                self.origins[index] + "/v1/status?lightweight=true",
                                timeout=2,
                            )
                            if response.status_code == 200:
                                return
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.1)
                raise RuntimeError("worker exited")

            try:
                await asyncio.wait_for(wait_ready(), timeout=90)
            except (asyncio.TimeoutError, RuntimeError, asyncio.CancelledError) as exc:
                await self.stop(index)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise HTTPException(
                    503,
                    "Video worker could not start. Check container resources and logs.",
                ) from None

    async def add(self):
        index = self.fleet.capacity + 1
        await self.ensure(index)
        temporary = self.registry.with_name(".video-sources-" + uuid4().hex)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump({"schema_version": 1, "capacity": index}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.registry)
        except OSError:
            await self.stop(index)
            raise HTTPException(503, "Could not save the video source.") from None
        finally:
            temporary.unlink(missing_ok=True)
        self.fleet.capacity = index
        return self.fleet._camera_descriptor(index, index)

    async def stop(self, index):
        process = self.processes.pop(index, None)
        self.origins.pop(index, None)
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=20)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def close(self):
        self.closed = True
        await asyncio.gather(*(self.stop(index) for index in list(self.processes)))


def main():
    import uvicorn

    capture_options = os.environ.get("VIDEO_CAPTURE_OPTIONS")
    if capture_options is not None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = capture_options
    uvicorn.run(
        "goodvision_harness.service:app",
        fd=int(sys.argv[1]),
        timeout_graceful_shutdown=5,
        server_header=False,
        ws_per_message_deflate=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
