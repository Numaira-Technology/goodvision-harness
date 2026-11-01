"""Private console API; rental starts only after an explicit start request."""

# ruff: noqa: RUF001 -- User-facing Chinese punctuation.

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from fastapi import FastAPI, HTTPException, Request

if __package__:
    from .cloud_gpu_settings import CloudGPUSettings
    from .relay_bootstrap import controller_environment
else:
    from cloud_gpu_settings import CloudGPUSettings
    from relay_bootstrap import controller_environment


class CloudController:
    def __init__(self, directory: Path, capacity: int):
        self.directory = directory
        self.capacity = capacity
        self.config_path = directory / "rental.json"
        self.status_path = directory / "rental-state.json"
        self.process = None
        self.release_process = None
        self.release_task = None
        self.release_error = None
        self.lock = asyncio.Lock()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.child_environment = controller_environment(os.environ, self.directory)

    def settings(self):
        return (
            CloudGPUSettings.from_payload(
                json.loads(self.config_path.read_text("utf-8"))
            )
            if self.config_path.exists()
            else CloudGPUSettings.from_environment()
        )

    def record(self):
        if self.status_path.exists():
            return json.loads(self.status_path.read_text("utf-8"))
        return {"state": "stopped", "cleanup_confirmed": True}

    def pool_ready(self):
        try:
            with build_opener(ProxyHandler({})).open(
                "http://127.0.0.1:8092/health/cameras",
                timeout=8,
            ) as response:
                data = json.load(response)
            ready = all(
                data.get(key) is True for key in ("enabled", "ready", "gpu_ready")
            )
            if ready and os.environ.get("SAM_ANALYSIS_ENABLED") == "true":
                with build_opener(ProxyHandler({})).open(
                    "http://127.0.0.1:8095/health", timeout=8,
                ) as response:
                    ready = json.load(response).get("ready") is True
            return ready
        except (URLError, OSError, ValueError):
            return False

    def status(self):
        record = self.record()
        live = self.process is not None and self.process.poll() is None
        if not live and not record.get("cleanup_confirmed"):
            record["state"] = "cleanup_required"
        elif live and record["state"] == "stopped":
            record["state"] = "stopping"
        # The child may still report ready immediately after SIGTERM. Block
        # analysis synchronously while the owned release continues in background.
        if self.release_task is not None and not self.release_task.done():
            record["state"] = "stopping"
        if self.release_error:
            record.update(
                state="cleanup_required", cleanup_confirmed=False,
                error=self.release_error,
            )
        config = self.settings()
        return {
            "available": True,
            "capacity": self.capacity,
            "settings": config.public(),
            "command": "python tools/start_sam.py " + " ".join(config.arguments()),
            **record,
        }

    def save(self, payload):
        if (
            (self.process is not None and self.process.poll() is None)
            or (self.release_task is not None and not self.release_task.done())
        ):
            raise HTTPException(409, "请先释放当前云 GPU，再修改租用配置")
        settings = CloudGPUSettings.from_payload(payload)
        if settings.streams != self.capacity:
            raise ValueError(
                f"视频路数须与主程序已部署的 {self.capacity} 路一致；"
                "扩容请修改启动参数 --streams"
            )
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings.public()), encoding="utf-8")
        temporary.replace(self.config_path)
        return self.status()

    def start(self):
        status = self.status()
        if (
            (self.process is not None and self.process.poll() is None)
            or (self.release_task is not None and not self.release_task.done())
        ):
            raise HTTPException(409, "云 GPU 已启动、正在启动或正在释放")
        if status["state"] == "cleanup_required":
            raise HTTPException(
                409,
                "上次租用尚未确认释放。请在云平台核对实例并处理后"
                "清除控制器的 rental-state.json，避免重复租用",
            )
        settings = self.settings()
        if settings.streams != self.capacity:
            raise HTTPException(409, "租用配置的路数与主程序不一致，请重新保存租用配置")
        self.status_path.write_text(
            json.dumps({"state": "starting", "cleanup_confirmed": False}),
            encoding="utf-8",
        )
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("start_sam.py")),
                    *settings.arguments(),
                    "--status-file",
                    str(self.status_path),
                ],
                stdin=subprocess.DEVNULL,
                env=self.child_environment,
            )
        except OSError:
            self.status_path.write_text(
                json.dumps({"state": "stopped", "cleanup_confirmed": True}),
                encoding="utf-8",
            )
            raise HTTPException(503, "无法启动云 GPU 租用进程") from None
        self.release_process = self.release_task = self.release_error = None
        return self.status()

    async def wait_release(self, process):
        try:
            # Backup operations have byte-aware budgets. No HTTP request or
            # fixed fleet-duration timeout can abandon this owned cleanup.
            await asyncio.to_thread(process.wait)
        except Exception:
            self.release_error = "无法确认云 GPU 释放，请检查实例和控制器日志"

    async def stop(self):
        if self.release_process is self.process and self.release_task is not None:
            return self.status()
        if self.process is not None and self.process.poll() is None:
            # One signal per rental; repeated stop and lifespan shutdown reuse
            # this task even if a browser disconnects during the request.
            self.release_process = self.process
            self.release_error = None
            try:
                self.process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass  # A naturally exited child can still be reaped safely.
            self.release_task = asyncio.create_task(self.wait_release(self.process))
        return self.status()

    async def close(self):
        await self.stop()
        if self.release_task is not None:
            await asyncio.shield(self.release_task)


def create_app(directory=None, capacity=None):
    manager = CloudController(
        Path(directory or os.environ.get("SAM_CONTROL_DIRECTORY", "/data/cloud-gpu")),
        capacity or int(os.environ.get("SAM31_LIVE_CAMERAS", "1")),
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        await manager.close()

    app = FastAPI(lifespan=lifespan)
    app.state.manager = manager

    @app.get("/settings")
    async def status(check_health: bool = True):
        state = manager.status()
        if check_health and state["state"] == "ready" and not await asyncio.to_thread(
            manager.pool_ready
        ):
            state.update(
                state="degraded",
                error="部分云 GPU 不可用，请停止分析并检查或释放后重启",
            )
        return state

    @app.put("/settings")
    async def save(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                raise HTTPException(413, "租用配置过大")
        async with manager.lock:
            try:
                return manager.save(json.loads(body))
            except (ValueError, TypeError):
                raise HTTPException(
                    400,
                    "租用配置无效；请检查 GPU 型号、正整数数量、磁盘容量和已部署路数",
                ) from None

    @app.post("/start")
    async def start():
        async with manager.lock:
            return manager.start()

    @app.post("/stop")
    async def stop():
        async with manager.lock:
            return await manager.stop()

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8093, timeout_graceful_shutdown=435)
