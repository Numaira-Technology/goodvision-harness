"""Manage one remote SAM GPU and SSH tunnel for the Compose sam-remote service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/goodvision-sam"
IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
SAM_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
WORKER_FILES = (
    "app.py", "sam31.py", "continuous.py", "streaming_state.py",
    "h264_transport.py", "requirements.txt",
)

# Execute inside the rented container, without nesting another Docker daemon.
# Keep credentials out of this script and out of remote command arguments.
SETUP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git build-essential libgomp1 libglib2.0-0
cd {REMOTE}
curl --fail --location --retry 3 \\
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \\
  -o miniforge.sh
bash miniforge.sh -b -p {REMOTE}/conda
{REMOTE}/conda/bin/conda create -y -p {REMOTE}/env python=3.12 pip
{REMOTE}/env/bin/python -m pip install --no-cache-dir \\
  torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
{REMOTE}/env/bin/python -m pip install --no-cache-dir -r worker/requirements.txt
{REMOTE}/env/bin/python -m pip install --no-cache-dir \\
  git+https://github.com/facebookresearch/sam3.git@{SAM_COMMIT}
"""


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the repository .env or your terminal.")
    return value


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(command: list[str], *, input: str | None = None, timeout: int = 120) -> None:
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        text=True, start_new_session=True,
    )
    try:
        process.communicate(input=input, timeout=timeout)
        if process.returncode:
            raise RuntimeError(f"{Path(command[0]).name} exited with status {process.returncode}.")
    finally:
        stop_process(process)


def wait_for_ssh(vast, instance_id: int, options: list[str]) -> tuple[str, int]:
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        instance = vast.show_instance(id=instance_id) or {}
        if instance.get("actual_status") in {"offline", "exited", "stopped", "frozen"}:
            raise RuntimeError(f"Instance stopped before SSH was ready: {instance.get('status_msg', '')}")
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
        ssh_ports = (instance.get("ports") or {}).get("22/tcp", [])
        if instance.get("public_ipaddr") and ssh_ports:
            # Prefer the published port over Vast's shared SSH gateway.
            host, port = instance["public_ipaddr"], ssh_ports[0]["HostPort"]
        if host and port:
            command = ["ssh", *options, "-p", str(port), f"root@{host}", "true"]
            try:
                result = subprocess.run(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=15,
                )
                if result.returncode == 0:
                    print(f"SSH connected: {host}:{port}", flush=True)
                    return str(host), int(port)
            except subprocess.TimeoutExpired:
                pass
        time.sleep(5)
    raise RuntimeError("SSH did not become available within 20 minutes.")


def wait_for_worker(process: subprocess.Popen, port: int) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 10 * 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SAM worker/SSH exited with status {process.returncode}; see logs above.")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health/video/session", timeout=3) as response:
                state = json.load(response)
            if state.get("ready") and state.get("gpu_ready"):
                return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError("SAM did not finish loading within 10 minutes; see logs above.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-key", type=Path, help="SSH private key; defaults to SAM_SSH_KEY.")
    parser.add_argument("--disk-gb", type=int, default=os.environ.get("SAM_DISK_GB", "50"), help="Rented disk size; defaults to SAM_DISK_GB or 50 GB.")
    parser.add_argument("--sam-port", type=int, default=8092, help="SSH forwarding port inside the controller (default: 8092).")
    args = parser.parse_args()

    vast = None
    instance_id = None
    creation_attempted = False
    worker = None
    ssh_directory = None
    exit_code = 1
    label = f"goodvision-sam-{uuid4().hex[:12]}"

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {item: signal.signal(item, interrupt) for item in handled_signals}
    try:
        try:
            from dotenv import load_dotenv
            from vastai import VastAI
        except ImportError as error:
            raise RuntimeError(
                "Missing controller dependencies; rebuild the sam-remote image with Docker Compose."
            ) from error
        load_dotenv(ROOT / ".env", override=False)
        for program in ("ssh", "scp"):
            if shutil.which(program) is None:
                raise RuntimeError(f"Required command is not on PATH: {program}")
        api_key, hf_token = required("VAST_API_KEY"), required("HF_TOKEN")
        ssh_key = (args.ssh_key or Path(required("SAM_SSH_KEY"))).expanduser().resolve()
        if not ssh_key.is_file():
            raise RuntimeError(f"SSH private key does not exist: {ssh_key}")
        if args.disk_gb <= 0 or not 1 <= args.sam_port <= 65535:
            raise RuntimeError("Disk size must be positive and the forwarding port must be within 1..65535.")
        for filename in WORKER_FILES:
            if not (ROOT / "workers" / "sam_scene" / filename).is_file():
                raise RuntimeError(f"Missing SAM worker source: {filename}")

        print("Searching for one on-demand RTX 4090 (CUDA >= 12.6)...", flush=True)
        # vastai 1.5.6 counts attempts, so retry=1 sends exactly one request.
        vast = VastAI(api_key=api_key, retry=1)
        offers = vast.search_offers(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rentable=true direct_port_count>=1 "
                f"cuda_max_good>=12.6 disk_space>={args.disk_gb}"
            ),
            type="on-demand", order="dph_total", storage=float(args.disk_gb), limit=10,
        )
        if not offers:
            raise RuntimeError("No eligible RTX 4090 offer is available. No instance was created.")
        offer = min(offers, key=lambda item: float(item["dph_total"]))
        print(f"Selected offer: {offer['id']}; estimated total hourly price: ${float(offer['dph_total']):.4f}", flush=True)
        print(f"Creating instance with label: {label}", flush=True)
        creation_attempted = True
        created = vast.create_instance(
            id=int(offer["id"]), image=IMAGE, disk=args.disk_gb,
            label=label, ssh=True, direct=True, cancel_unavail=True,
        )
        instance_id = int(created["new_contract"])
        print(f"Created instance: {instance_id}\nWaiting for SSH...", flush=True)
        ssh_directory = tempfile.TemporaryDirectory(prefix="goodvision-sam-ssh-")
        options = [
            "-i", str(ssh_key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
            "-o", f"UserKnownHostsFile={Path(ssh_directory.name) / 'known_hosts'}",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        ]
        host, ssh_port = wait_for_ssh(vast, instance_id, options)
        ssh = ["ssh", *options, "-p", str(ssh_port), f"root@{host}"]
        print("Uploading local SAM worker source...", flush=True)
        run([*ssh, f"mkdir -p {REMOTE}/worker {REMOTE}/models"])
        run([
            "scp", *options, "-P", str(ssh_port),
            *(str(ROOT / "workers" / "sam_scene" / name) for name in WORKER_FILES),
            f"root@{host}:{REMOTE}/worker/",
        ])
        print("Installing the remote Conda environment (logs follow)...", flush=True)
        run([*ssh, "bash -s"], input=SETUP, timeout=45 * 60)
        print("Downloading approved SAM 3.1 weights on the GPU instance...", flush=True)
        download = (
            "import sys; from huggingface_hub import hf_hub_download; "
            "hf_hub_download(repo_id='facebook/sam3.1', filename='sam3.1_multiplex.pt', "
            f"local_dir='{REMOTE}/models', token=sys.stdin.read().strip())"
        )
        run([
            *ssh, f"HF_HUB_OFFLINE=0 {REMOTE}/env/bin/python -c {shlex.quote(download)}",
        ], input=hf_token, timeout=30 * 60)
        worker_command = (
            f"cd {REMOTE}/worker && exec env "
            "SAM31_ENABLED=true SAM31_SESSION_ENABLED=true SAM_SCENE_IMAGE_ENABLED=false "
            "SAM31_SESSION_H264_ENABLED=true SAM31_SESSION_HOTSTART_DELAY=5 "
            "SAM31_SESSION_POSTPROCESS_BATCH_SIZE=1 SAM31_COMPILE_MODE=off "
            "SAM31_INFERENCE_MAX_SIDE=640 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "
            f"SAM31_CHECKPOINT_PATH={REMOTE}/models/sam3.1_multiplex.pt "
            f"{REMOTE}/env/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8092 --workers 1"
        )
        print("Starting SAM and its SSH tunnel; waiting for model loading (logs follow)...", flush=True)
        worker = subprocess.Popen([
            *ssh[:-1], "-o", "ExitOnForwardFailure=yes",
            # Compose keeps this port inside its network; it is not published.
            "-L", f"0.0.0.0:{args.sam_port}:127.0.0.1:8092",
            ssh[-1], worker_command,
        ], stdin=subprocess.DEVNULL, start_new_session=True)
        wait_for_worker(worker, args.sam_port)
        print(
            f"SAM is ready at http://sam-remote:{args.sam_port}/v1/video/sessions.\n"
            "Stop the Compose application to release this GPU. Closing the browser does not release it.",
            flush=True,
        )
        worker.wait()
        raise RuntimeError(f"SAM/SSH exited with status {worker.returncode}; releasing the instance.")
    except KeyboardInterrupt:
        print("\nStopping SAM session...", flush=True)
        exit_code = 130
    except Exception as error:
        print(f"Start failed: {error}", file=sys.stderr, flush=True)
    finally:
        # A second Ctrl+C must not interrupt the short instance-release request.
        for item in handled_signals:
            signal.signal(item, signal.SIG_IGN)
        if vast is not None:
            # Fit lookup + destruction inside Compose's 90-second shutdown grace.
            vast.client.timeout = (5, 20)
        try:
            stop_process(worker)
        except OSError:
            pass
        if creation_attempted and instance_id is None:
            # A lost create response does not mean the server created nothing.
            try:
                matches = [item for item in vast.show_instances() if item.get("label") == label]
                if len(matches) == 1:
                    instance_id = int(matches[0]["id"])
            except Exception as error:
                print(f"Could not look up the unconfirmed creation: {error}", file=sys.stderr, flush=True)
            if instance_id is None:
                print(
                    f"Creation remains unconfirmed. Check Vast Instances for label {label} before retrying.\n"
                    "An instance may appear after this lookup. Destroy that instance in Vast Instances.",
                    file=sys.stderr, flush=True,
                )
        if instance_id is not None:
            print(f"Destroying instance: {instance_id}", flush=True)
            try:
                result = vast.destroy_instance(id=instance_id)
                if not isinstance(result, dict) or result.get("success") is not True:
                    raise RuntimeError("Vast did not confirm instance destruction.")
                print(f"Released instance: {instance_id}", flush=True)
            except Exception as error:
                exit_code = 1
                print(
                    f"Release failed: {error}\nInstance {instance_id} may still be billed. "
                    "Destroy it in Vast Instances.",
                    file=sys.stderr, flush=True,
                )
        if ssh_directory is not None:
            ssh_directory.cleanup()
        for item, handler in previous_handlers.items():
            signal.signal(item, handler)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
