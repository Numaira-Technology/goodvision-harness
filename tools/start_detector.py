"""Rent a GPU and run the YOLO / RF-DETR detector rich-label worker, SAM-style.

Mirrors tools/start_sam.py: same Vast rental / SSH-key / tunnel machinery (reused
by import), but ships workers/detector_rich_label and speaks the /v1/labels
contract on port 8092. Pick the model with DETECTOR_MODEL (yolo26-s / yolov8-m /
rfdetr-l). Weights auto-download (ultralytics / rfdetr) unless DETECTOR_WEIGHTS_DIR
is pre-staged. Stop the Compose app (or Ctrl+C) to release the GPU.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

# Reuse the proven, worker-agnostic Vast machinery from the SAM launcher.
from start_sam import (  # type: ignore[import-not-found]
    ROOT,
    create_from_offers,
    creation_diagnostic,
    prepare_ssh_key,
    release_instance,
    required,
    run,
    stop_process,
    wait_for_ssh,
)

REMOTE = "/workspace/goodvision-detector"
WORKER_FILES = ("app.py", "tracker.py", "detectors.py", "requirements.txt")
WORKER_ENV = {
    "DETECTOR_MODEL": os.environ.get("DETECTOR_MODEL", "yolo26-s"),
    "DETECTOR_DEVICE": os.environ.get("DETECTOR_DEVICE", "cuda"),
    "DETECTOR_WEIGHTS_DIR": os.environ.get("DETECTOR_WEIGHTS_DIR", f"{REMOTE}/models"),
    "DETECTOR_CONF": os.environ.get("DETECTOR_CONF", "0.35"),
    "DETECTOR_IMGSZ": os.environ.get("DETECTOR_IMGSZ", "640"),
    "PYTHONUNBUFFERED": "1",
}

RUNTIME_CHECK = (
    "import importlib,sys; "
    "[importlib.import_module(m) for m in ('cv2','numpy','fastapi','ultralytics')]; "
    "print('detector runtime ok')"
)

SETUP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd {REMOTE}/worker
if [ -x {REMOTE}/env/bin/python ] && {REMOTE}/env/bin/python -c {shlex.quote(RUNTIME_CHECK)}; then
  echo 'Reusing verified detector runtime; installation skipped.'
  exit 0
fi
apt-get update
apt-get install -y --no-install-recommends \\
  curl ca-certificates git libgl1 libglib2.0-0
cd {REMOTE}
if [ ! -x {REMOTE}/conda/bin/conda ]; then
  curl --fail --location --retry 3 \\
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \\
  -o miniforge.sh
  bash miniforge.sh -b -p {REMOTE}/conda
fi
if [ ! -x {REMOTE}/env/bin/python ]; then
  {REMOTE}/conda/bin/conda create -y \\
    -p {REMOTE}/env python=3.12 pip
fi
{REMOTE}/env/bin/python -m pip install --no-cache-dir \\
  torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
{REMOTE}/env/bin/python -m pip install --no-cache-dir -r worker/requirements.txt
cd {REMOTE}/worker
{REMOTE}/env/bin/python -c {shlex.quote(RUNTIME_CHECK)}
"""


def start_worker(vast, instance_id: int, options: list[str], *, port: int):
    print("Waiting for SSH...", flush=True)
    host, ssh_port = wait_for_ssh(vast, instance_id, options)
    ssh = ["ssh", *options, "-p", str(ssh_port), f"root@{host}"]
    print("Uploading detector worker source...", flush=True)
    run([*ssh, f"mkdir -p {REMOTE}/worker {REMOTE}/models"])
    worker_dir = ROOT / "workers" / "detector_rich_label"
    run([
        "scp", *options, "-P", str(ssh_port),
        *(str(worker_dir / name) for name in WORKER_FILES),
        f"root@{host}:{REMOTE}/worker/",
    ])
    print("Installing the remote environment (logs follow)...", flush=True)
    run([*ssh, "bash -s"], input=SETUP, timeout=45 * 60)
    worker_environment = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in WORKER_ENV.items()
    )
    worker_command = (
        f"cd {REMOTE}/worker && exec env {worker_environment} "
        f"{REMOTE}/env/bin/python -m uvicorn app:app --host 127.0.0.1 "
        f"--port 8092 --workers 1"
    )
    print("Starting detector worker + SSH tunnel; warming up (logs follow)...",
          flush=True)
    return subprocess.Popen([
        *ssh[:-1], "-o", "ExitOnForwardFailure=yes",
        "-L", f"0.0.0.0:{port}:127.0.0.1:8092",
        ssh[-1], worker_command,
    ], stdin=subprocess.DEVNULL, start_new_session=True)


def wait_for_worker(process: subprocess.Popen, port: int) -> None:
    import time
    import urllib.request

    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("detector worker exited before readiness")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health/ready", timeout=3
            ) as response:
                import json

                body = json.loads(response.read().decode())
                if body.get("ready") and body.get("gpu_ready"):
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("detector worker did not become ready")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-key", type=Path)
    parser.add_argument("--disk-gb", type=int,
                        default=int(os.environ.get("SAM_DISK_GB", "50")))
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    for item in (signal.SIGINT, signal.SIGTERM):
        signal.signal(item, interrupt)

    vast = instance_id = ssh_directory = worker = None
    attempted: set[str] = set()
    label = f"goodvision-detector-{uuid4().hex[:12]}"
    exit_code = 1
    try:
        from dotenv import load_dotenv
        from vastai import VastAI

        load_dotenv(ROOT / ".env", override=False)
        gpu_name = (os.environ.get("SAM_GPU_NAME", "").strip().replace(" ", "_")
                    or "RTX_4090")
        geo = os.environ.get("SAM_GEOLOCATION", "!=CN").strip() or "!=CN"
        order = os.environ.get("SAM_OFFER_ORDER", "dph_total").strip() or "dph_total"
        api_key = required("VAST_API_KEY")
        ssh_key = (args.ssh_key or Path(required("SAM_SSH_KEY"))).expanduser().resolve()
        ssh_directory = tempfile.TemporaryDirectory(prefix="goodvision-det-ssh-")
        ssh_key = prepare_ssh_key(ssh_key, Path(ssh_directory.name))
        vast = VastAI(api_key=api_key, retry=1)
        print(f"Searching on-demand {gpu_name} (geolocation{geo}; order={order})...",
              flush=True)
        offers = vast.search_offers(
            query=(f"gpu_name={gpu_name} num_gpus=1 rentable=true direct_port_count>=1 "
                   f"cuda_max_good>=12.6 disk_space>={args.disk_gb} geolocation{geo}"),
            type="on-demand", order=order, storage=float(args.disk_gb), limit=10,
        )
        if not offers:
            raise RuntimeError("No eligible GPU offer; nothing created.")
        options = [
            "-i", str(ssh_key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
            "-o", f"UserKnownHostsFile={Path(ssh_directory.name) / 'known_hosts'}",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        ]
        while True:
            if instance_id is None:
                instance_id = create_from_offers(
                    vast, offers, disk_gb=args.disk_gb, label=label,
                    attempted_machines=attempted,
                )
                print(f"Created instance: {instance_id}", flush=True)
            try:
                worker = start_worker(vast, instance_id, options, port=args.port)
                wait_for_worker(worker, args.port)
            except Exception as error:
                if len(attempted) >= 3:
                    raise
                print(f"Startup failed on {instance_id}: "
                      f"{creation_diagnostic({'message': str(error)})}; releasing.",
                      flush=True)
                stop_process(worker)
                worker = None
                release_instance(vast, instance_id)
                instance_id = None
                label = f"goodvision-detector-{uuid4().hex[:12]}"
                continue
            break
        print(
            f"Detector ready: {WORKER_ENV['DETECTOR_MODEL']} passed its health check.\n"
            f"Labels endpoint: http://127.0.0.1:{args.port}/v1/labels\n"
            "Stop the Compose application to release this GPU.",
            flush=True,
        )
        worker.wait()
        raise RuntimeError(f"detector/SSH exited ({worker.returncode}); releasing.")
    except KeyboardInterrupt:
        print("\nStopping detector session...", flush=True)
        exit_code = 130
    except Exception as error:
        print(f"Start failed: {error}", file=sys.stderr, flush=True)
    finally:
        for item in (signal.SIGINT, signal.SIGTERM):
            signal.signal(item, signal.SIG_IGN)
        stop_process(worker)
        if vast is not None and instance_id is not None:
            try:
                release_instance(vast, instance_id)
            except Exception as error:
                print(f"Release failed: {error}; destroy {instance_id} in Vast.",
                      file=sys.stderr, flush=True)
        if ssh_directory is not None:
            ssh_directory.cleanup()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
