"""Persist, verify and supervise the frozen detector on loopback only."""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from . import fp8_profile as p
from . import runtime


def weights(root):
    return Path(root) / "detector" / p.YOLO_SHA256 / "yolo26s.pt"


def verify_weights(root):
    path = weights(root)
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(Path(root).resolve())
        or runtime.sha(path) != p.YOLO_SHA256
    ):
        raise ValueError("YOLO weight integrity mismatch")
    return path


def prepare(root):
    root = runtime.volume_root(root)
    with runtime.exclusive(root / ".detector-weights.lock"):
        path = weights(root)
        if path.exists():
            return verify_weights(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        try:
            with (
                urllib.request.urlopen(p.YOLO_URL, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                total = 0
                while chunk := response.read(1024**2):
                    total += len(chunk)
                    if total > 200 * 1024**2:
                        raise ValueError("unexpected detector weight size")
                    out.write(chunk)
            if runtime.sha(temporary) != p.YOLO_SHA256:
                raise ValueError("downloaded YOLO checksum differs")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return verify_weights(root)


def verify_active(root):
    root = runtime.volume_root(root)
    data = json.loads((root / "runtime/detector-active.json").read_text())
    if (
        data["version"] != p.VERSION
        or data["weights_sha256"] != p.YOLO_SHA256
        or data.get("boot_id") != runtime.boot_id()
        or runtime.process_start_ticks(data["pid"]) != data["start_ticks"]
    ):
        raise ValueError("no matching live YOLO service")
    verify_weights(root)
    return data


def start(root, python):
    root = runtime.volume_root(root)
    python = Path(python).absolute()
    if not python.is_file():
        raise ValueError("run setup-yolo.sh --execute first")
    with (
        runtime.exclusive(root / ".detector-server.lock"),
        runtime.exclusive(root / ".detector-weights.lock"),
    ):
        path = verify_weights(root)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 8092))
        worker = Path(__file__).resolve().parents[2] / "workers/detector_rich_label"
        logs = root / "runtime"
        logs.mkdir(exist_ok=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("DETECTOR_")
            and not any(s in k.upper() for s in ("API_KEY", "HF_TOKEN"))
        }
        env.update(
            DETECTOR_MODEL="yolo26-s",
            DETECTOR_DEVICE="cuda",
            DETECTOR_WEIGHTS=str(path),
            DETECTOR_CONF="0.35",
            DETECTOR_IMGSZ="640",
        )
        # Confirm the separate detector environment and actual GPU, before launch.
        code = (
            "import torch, ultralytics; "
            'assert ultralytics.__version__ == "8.4.181"; '
            "assert torch.cuda.device_count() == 1; "
            "g=torch.cuda.get_device_properties(0); "
            'assert g.total_memory >= 46*1024**3 and "4090" in g.name; '
            "assert (g.major,g.minor)==(8,9)"
        )
        subprocess.run([str(python), "-c", code], env=env, check=True, timeout=60)
        active = logs / "detector-active.json"
        with (logs / f"detector-{time.time_ns()}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "uvicorn",
                    "app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8092",
                ],
                cwd=worker,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

            def terminate(*_):
                raise KeyboardInterrupt

            previous = signal.signal(signal.SIGTERM, terminate)
            try:
                from tools.tracked_grounding.detector import DetectorBroker

                deadline = time.monotonic() + 120
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("YOLO exited before readiness")
                    try:
                        DetectorBroker(p.DETECTOR_URL, logs).check_ready()
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("YOLO readiness exceeded 120s") from None
                        time.sleep(1)
                runtime.write_json(
                    active,
                    {
                        "version": p.VERSION,
                        "pid": process.pid,
                        "start_ticks": runtime.process_start_ticks(process.pid),
                        "boot_id": runtime.boot_id(),
                        "weights_sha256": p.YOLO_SHA256,
                    },
                )
                if process.wait():
                    raise RuntimeError("YOLO exited; inspect detector log")
            finally:
                signal.signal(signal.SIGTERM, previous)
                active.unlink(missing_ok=True)
                runtime.stop_process_group(process, timeout=15)
