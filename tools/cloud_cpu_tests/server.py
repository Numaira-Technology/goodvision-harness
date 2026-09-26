"""Run on the rented Linux host: two isolated CPU Harness runtimes, one SAM GPU."""

import argparse
import hashlib
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import httpx
import psutil
from fastapi import HTTPException

from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.service import HarnessRuntime, create_app
from tools.seg_pipeline.__main__ import child_environment, stop
from tools.seg_pipeline.service import (
    ROOT,
    benchmark_config,
    configure_runtime_environment,
    fixed_protocol,
)


class TimingRecords(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = deque(maxlen=20000)

    def emit(self, record):
        if record.args and len(record.args) == 2:
            event, data = record.args
            if isinstance(event, str) and event.startswith("sam_"):
                try:
                    self.rows.append({"event": event, **json.loads(data)})
                except (ValueError, TypeError):
                    pass


def gpu_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,"
            "power.limit,power.draw,clocks.sm,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return result.stdout.strip()


def build_app(args):
    state = args.output.resolve() / f"lane-{args.lane}"
    state.mkdir(parents=True, exist_ok=True)
    endpoint = (
        f"{args.sam_endpoint}/v1/cameras/camera-{args.lane:02d}"
        "/perception/sam_live/sessions"
    )
    config = benchmark_config(
        {
            "output": str(state),
            "endpoint": endpoint,
            "face_model_root": str(args.face_model_root.resolve()),
            "allow_face_download": args.allow_face_download,
            "inference_provider": "openrouter",
        }
    )
    config = replace(
        config, service=replace(config.service, source_id=f"cloud-test-{args.lane}")
    )
    configure_runtime_environment({"output": str(state)}, config)
    if not os.environ.get(config.model_api.api_key_env):
        raise ValueError(f"Set {config.model_api.api_key_env} on the cloud host")
    # Persist buffer changes only within this isolated experiment directory.
    store = DeploymentSettingsStore(state / "deployment.json")
    timings = TimingRecords()
    telemetry = deque(maxlen=10000)
    trial = {"id": None, "delay": None}
    process = psutil.Process()
    stop_monitor = threading.Event()
    identity = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "lane": args.lane,
        "cpu_count": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "sam_endpoint": endpoint,
        "gpu_at_start": gpu_snapshot(),
        "config_sha256": hashlib.sha256(
            (ROOT / "config.yaml").read_bytes()
        ).hexdigest(),
        "harness_source_sha256": hashlib.sha256(
            (ROOT / "src/goodvision_harness/service.py").read_bytes()
        ).hexdigest(),
        "worker_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "workers/sam_scene").glob("*.py"))
        },
    }
    app = create_app(
        runtime_factory=lambda: HarnessRuntime(config, deployment_store=store)
    )

    def monitor():
        while not stop_monitor.wait(2):
            try:
                telemetry.append(
                    {
                        "wall_time": time.time(),
                        "trial_id": trial["id"],
                        "process_cpu_percent": process.cpu_percent(),
                        "process_rss_bytes": process.memory_info().rss,
                        "system_cpu_percent": psutil.cpu_percent(),
                        "available_ram_bytes": psutil.virtual_memory().available,
                        "swap_used_bytes": psutil.swap_memory().used,
                        "children": [
                            {
                                "pid": p.pid,
                                "name": p.name(),
                                "rss_bytes": p.memory_info().rss,
                            }
                            for p in process.children(recursive=True)
                        ],
                        "gpu": gpu_snapshot(),
                    }
                )
            except (psutil.Error, subprocess.SubprocessError, OSError) as error:
                telemetry.append(
                    {"error": type(error).__name__, "wall_time": time.time()}
                )

    original_lifespan = app.router.lifespan_context

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(application):
        logging.getLogger("uvicorn.error").addHandler(timings)
        thread = threading.Thread(target=monitor, daemon=True)
        try:
            async with original_lifespan(application):
                thread.start()
                yield
        finally:
            stop_monitor.set()
            if thread.ident:
                thread.join(timeout=7)
            logging.getLogger("uvicorn.error").removeHandler(timings)

    app.router.lifespan_context = lifespan

    @app.get("/benchmark/config")
    def info():
        runtime = app.state.runtime
        protocol = fixed_protocol(runtime.config)
        protocol["effective_display_delay_seconds"] = (
            runtime._display_settings.delay_seconds
        )
        return {
            "kind": "sam_cloud_cpu_tests",
            "host": identity,
            "trial": dict(trial),
            "fixed_main": protocol,
        }

    @app.post("/benchmark/trial")
    def begin(payload: dict):
        runtime = app.state.runtime
        if runtime._processing_lifecycle_busy():
            raise HTTPException(409, "Previous trial has not drained")
        if set(payload) != {"id", "delay"} or not isinstance(payload["id"], str):
            raise HTTPException(400, "Expected trial id and delay")
        delay = payload["delay"]
        if type(delay) not in (int, float) or not 0 <= delay <= 30:
            raise HTTPException(400, "Delay must be in 0..30 seconds")
        runtime.update_deployment_settings({"buffer_delay_seconds": delay})
        os.environ["LOCAL_CAMERA_DISPLAY_DELAY_SECONDS"] = str(delay)
        timings.rows.clear()
        telemetry.clear()
        trial.update(id=payload["id"], delay=delay)
        return info()

    @app.get("/benchmark/evidence")
    def evidence():
        return {**info(), "timings": list(timings.rows), "telemetry": list(telemetry)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-endpoint", default="http://127.0.0.1:8092")
    parser.add_argument("--base-port", type=int, default=18094)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/cloud-cpu-server"
    )
    parser.add_argument(
        "--face-model-root", type=Path, default=ROOT / ".models/insightface"
    )
    parser.add_argument("--allow-face-download", action="store_true")
    parser.add_argument("--lane", type=int, choices=(1, 2), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this service on the cloud Linux host, not on the Mac")
    if not 1024 <= args.base_port <= 65534:
        parser.error("Choose two adjacent unprivileged ports")
    if args.lane:
        os.environ.update(
            HARNESS_AUTOSTART="false",
            HARNESS_DRY_RUN="false",
            LOCAL_INFERENCE_ENABLED="false",
            LOCAL_TRACKING_ENABLED="false",
            INFERENCE_EXECUTION_MODE="cloud",
            TRACKING_EXECUTION_MODE="legacy",
        )
        import uvicorn

        uvicorn.run(
            build_app(args), host="127.0.0.1", port=args.base_port + args.lane - 1
        )
        return
    # Provision exactly two lanes in the same shared SAM model before testing.
    with httpx.Client(trust_env=False, timeout=10) as client:
        response = client.get(args.sam_endpoint.rstrip("/") + "/health/cameras")
        response.raise_for_status()
        health = response.json()
    if not health.get("gpu_ready") or health.get("max_active_sessions") != 2:
        parser.error("Start the SAM worker with SAM31_LIVE_CAMERAS=2 first")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sam-health.json").write_text(json.dumps(health, indent=2))
    children = []

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for lane in (1, 2):
            directory = args.output / f"lane-{lane}"
            directory.mkdir(exist_ok=True)
            env = child_environment(directory)
            command = [
                sys.executable,
                "-m",
                "tools.cloud_cpu_tests.server",
                *sys.argv[1:],
                "--lane",
                str(lane),
            ]
            children.append(subprocess.Popen(command, cwd=ROOT, env=env))
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError("A Harness lane exited; inspect the cloud terminal")
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            stop(child)


if __name__ == "__main__":
    main()
