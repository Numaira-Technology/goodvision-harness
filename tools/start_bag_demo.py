"""Rent one Vast GPU, bring up the native VideoJev scorer from R2, tunnel :8093.

For the 翻包 demo only. Reuses the vetted Vast machinery in start_sam.py (rent, SSH
wait, release-on-exit) but swaps in the VideoJev CUDA image + env recipe from
docs/guides/videojev-candidate.md, pulls the pinned weights from Cloudflare R2
(creds over SSH stdin, HF fallback), and runs tools/videojev_benchmark/server.py.

Cleanup is attempted on exit/signal/failure. Unconfirmed creation or destruction
requires checking the reported instance/label in Vast; release cannot be guaranteed.

    PYTHONPATH=src .venv/bin/python -m tools.start_bag_demo
    PYTHONPATH=src .venv/bin/python -m tools.start_bag_demo --resume-instance-id <id>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from tools import start_sam as S
from tools.cloud_prefetch import credential_blob
from tools.videojev_probe.contract import MODEL, REVISION

ROOT = Path(__file__).resolve().parents[1]
# Base image + env recipe verified on RTX 4090 24GB, CUDA 12.8 (guide, 2026-10-19).
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
BASE = "/workspace/videojev"
REPO = f"{BASE}/repo"
CACHE = f"{BASE}/models/huggingface"
LOCAL_PORT = 8093
R2_KEYS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")

SETUP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p {CACHE}
py={BASE}/env/bin/python
pip={BASE}/env/bin/pip
if [ -x "$py" ] && "$py" -c '
import flash_attn, transformers
assert transformers.__version__ == "5.14.1"
' 2>/dev/null; then
  echo 'Reusing verified VideoJev env; install skipped.'
else
  if ! command -v python3.12 >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends software-properties-common \\
      curl ca-certificates git build-essential
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y --no-install-recommends \\
      python3.12 python3.12-venv python3.12-dev
  fi
  [ -x "$py" ] || python3.12 -m venv {BASE}/env
  "$pip" install --no-cache-dir --upgrade pip setuptools wheel packaging ninja
  "$pip" install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \\
    --index-url https://download.pytorch.org/whl/cu128
  "$pip" install --no-cache-dir -r {REPO}/tools/videojev_probe/requirements.txt
  "$pip" install --no-cache-dir flash-attn==2.8.3.post2 causal-conv1d==1.7.0 \\
    --no-build-isolation
fi
"$py" -c '
import torch
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "no CUDA/BF16"
print("cuda ok", torch.cuda.get_device_name(0))
'
"""


def _public_key(key_path: Path) -> str:
    r = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(key_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return r.stdout.strip()


def _attach_key(vast, instance_id: int, key_path: Path) -> None:
    """Inject our public key into the instance (team API key doesn't at boot).

    Done right after creation so it propagates during the SSH wait, not after it.
    """
    try:
        res = vast.attach_ssh(instance_id=instance_id, ssh_key=_public_key(key_path))
        if isinstance(res, dict) and res.get("success") is False:
            raise RuntimeError("key attachment refused")
        print(f"Attached SSH key to instance {instance_id}", flush=True)
    except Exception as error:
        print(
            f"attach_ssh warning (will still wait for SSH): {type(error).__name__}",
            flush=True,
        )


def _vast_query(disk_gb: int) -> str:
    gpu = os.environ.get("BAG_DEMO_GPU", "RTX_4090")
    return (
        f"gpu_name={gpu} num_gpus=1 cuda_vers>=12.8 gpu_ram>=23 "
        f"disk_space>={disk_gb} inet_down>=200 reliability>0.95 rentable=true "
        "verified=true datacenter=true"
    )


def _upload_repo(ssh):
    S.run([*ssh, f"mkdir -p {REPO}"])
    # Only tracked runtime sources: never send local .env, media, logs or caches.
    files = subprocess.run(
        ["git", "ls-files", "-z", "--", "src", "tools"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout
    if not files:
        raise RuntimeError("no tracked runtime sources to upload")
    key_opts = " ".join(
        shlex.quote(o) for o in ssh[1:-1]
    )  # ssh args minus 'ssh' and target
    rsync_ssh = f"ssh {key_opts}"
    S.run(
        [
            "rsync",
            "-az",
            "--no-links",
            "--from0",
            "--files-from=-",
            "-e",
            rsync_ssh,
            f"{ROOT}/",
            f"{ssh[-1]}:{REPO}/",
        ],
        input=files,
        timeout=600,
    )


def _prefetch_from_r2(ssh, hf_token):
    r2_env = {k: os.environ.get(k) for k in R2_KEYS}
    blob = credential_blob(r2_env, hf_token)
    cmd = (
        f"set -a; . /dev/stdin; set +a; cd {REPO} && "
        f"PYTHONPATH=src {BASE}/env/bin/python -m tools.model_source "
        f"--model videojev --cache-dir {shlex.quote(CACHE)}"
    )
    print("Prefetching VideoJev from R2 (HF fallback)...", flush=True)
    S.run([*ssh, cmd], input=blob, timeout=45 * 60)


def _server_process(ssh):
    audit = f"{BASE}/audit.jsonl"
    remote = (
        f"cd {REPO} && exec env PYTHONPATH=src HF_HUB_OFFLINE=1 "
        f"{BASE}/env/bin/python -m tools.videojev_benchmark.server "
        f"--cache-dir {shlex.quote(CACHE)} --audit {shlex.quote(audit)}"
    )
    print(
        f"Starting VideoJev scorer + SSH tunnel on :{LOCAL_PORT} (logs follow)...",
        flush=True,
    )
    return subprocess.Popen(
        [
            *ssh[:-1],
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:8093",
            ssh[-1],
            remote,
        ],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_ready(process, timeout=20 * 60):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"scorer/SSH exited with status {process.returncode}; see logs above."
            )
        try:
            with opener.open(f"http://127.0.0.1:{LOCAL_PORT}/", timeout=4) as r:
                state = json.load(r)
            if isinstance(state, dict) and state.get("ready") is True:
                if (state.get("model"), state.get("revision")) != (MODEL, REVISION):
                    raise RuntimeError("unexpected scorer model/revision on local port")
                print(
                    f"VideoJev ready: {MODEL} @ {REVISION[:12]} "
                    f"on {state.get('runtime', {}).get('gpu')}",
                    flush=True,
                )
                return
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(3)
    raise RuntimeError("VideoJev scorer did not become ready in time; see logs above.")


def main(argv=None) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--disk-gb", type=int, default=int(os.environ.get("BAG_DEMO_DISK_GB", "70"))
    )
    ap.add_argument("--resume-instance-id", type=int)
    ap.add_argument("--status-file", type=Path)
    args = ap.parse_args(argv)
    if args.disk_gb <= 0 or (
        args.resume_instance_id is not None and args.resume_instance_id <= 0
    ):
        ap.error("disk size and resume instance ID must be positive")

    from vastai import VastAI

    api_key = S.required("VAST_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")  # optional for R2 and public HF fallback
    ssh_key_path = Path(S.required("SAM_SSH_KEY")).expanduser().resolve()
    if not ssh_key_path.is_file():
        raise RuntimeError(f"SSH key not found: {ssh_key_path}")

    vast = VastAI(api_key=api_key, retry=1)
    instance_id = None
    worker = None
    label = f"goodvision-bagdemo-{uuid4().hex[:12]}"
    exit_code = 1
    creation_attempted = False
    cleanup_confirmed = False
    ssh_dir = None

    def report(state):
        if args.status_file:
            value = {
                "state": state,
                "instance_id": instance_id,
                "label": label,
                "cleanup_confirmed": cleanup_confirmed,
            }
            tmp = args.status_file.with_name(args.status_file.name + ".tmp")
            tmp.write_text(json.dumps(value) + "\n", encoding="utf-8")
            tmp.replace(args.status_file)

    def interrupt(_s, _f):
        raise KeyboardInterrupt

    handled = [signal.SIGINT, signal.SIGTERM] + (
        [signal.SIGHUP] if hasattr(signal, "SIGHUP") else []
    )
    prev = {s: signal.signal(s, interrupt) for s in handled}
    try:
        for program in ("ssh", "ssh-keygen", "rsync", "git"):
            if shutil.which(program) is None:
                raise RuntimeError(f"required local tool missing: {program}")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", LOCAL_PORT))
        report("starting")  # validate status destination before any rental
        ssh_dir = tempfile.TemporaryDirectory(prefix="bagdemo-ssh-")
        key = S.prepare_ssh_key(ssh_key_path, Path(ssh_dir.name))
        options = [
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            f"UserKnownHostsFile={Path(ssh_dir.name) / 'known_hosts'}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if args.resume_instance_id is not None:
            inst = vast.show_instance(id=args.resume_instance_id) or {}
            if not isinstance(inst, dict) or not str(inst.get("label", "")).startswith(
                "goodvision-bagdemo-"
            ):
                raise RuntimeError("Resume requires an existing bag-demo instance.")
            instance_id = args.resume_instance_id
            label = inst["label"]
            print(f"Recovering instance {instance_id}; no new rental.", flush=True)
        else:
            print(f"Searching Vast for {_vast_query(args.disk_gb)} ...", flush=True)
            offers = vast.search_offers(
                query=_vast_query(args.disk_gb),
                type="on-demand",
                order="dph_total",
                storage=float(args.disk_gb),
                limit=10,
            )
            if not offers:
                raise RuntimeError("No eligible RTX 4090 / CUDA 12.8 offer available.")
            creation_attempted = True
            previous_image = S.IMAGE
            try:
                S.IMAGE = IMAGE  # create_from_offers reads this module global
                instance_id = S.create_from_offers(
                    vast, offers, disk_gb=args.disk_gb, label=label
                )
            finally:
                S.IMAGE = previous_image
            print(f"Created instance: {instance_id}", flush=True)

        # Team API key does not inject our key at boot; attach it now so it
        # propagates while we poll SSH (the earlier run attached too late).
        report("starting")
        _attach_key(vast, instance_id, key)
        print("Waiting for SSH...", flush=True)
        host, port = S.wait_for_ssh(vast, instance_id, options)
        ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
        print("Uploading repo (src + tools)...", flush=True)
        _upload_repo(ssh)
        print(
            "Provisioning VideoJev CUDA env (this takes a while; logs follow)...",
            flush=True,
        )
        S.run([*ssh, "bash -s"], input=SETUP, timeout=50 * 60)
        _prefetch_from_r2(ssh, hf_token)
        worker = _server_process(ssh)
        _wait_ready(worker)
        print(
            f"\nVideoJev scorer is live on http://127.0.0.1:{LOCAL_PORT}\n"
            f"Run the demo:  VIDEOJEV_ENDPOINT=http://127.0.0.1:{LOCAL_PORT} "
            f"PYTHONPATH=src .venv/bin/python -m tools.bag_demo.server\n"
            f"Instance {instance_id} is billing; Ctrl+C attempts cleanup.\n",
            flush=True,
        )
        report("ready")
        worker.wait()
        raise RuntimeError(f"scorer/SSH exited with status {worker.returncode}.")
    except KeyboardInterrupt:
        print("\nStopping bag-demo GPU session...", flush=True)
        exit_code = 130
    except Exception as error:
        print(
            f"bag-demo start failed: {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        for s in handled:
            signal.signal(s, signal.SIG_IGN)
        vast.client.timeout = (5, 20)
        try:
            S.stop_process(worker)
        except Exception as error:
            print(f"Worker cleanup failed: {type(error).__name__}", file=sys.stderr)
        # Reconcile a lost-create response so a silently-created box is not leaked.
        if creation_attempted and instance_id is None:
            try:
                matches = [
                    i
                    for i in vast.show_instances()
                    if isinstance(i, dict) and i.get("label") == label
                ]
                if (
                    len(matches) == 1
                    and type(matches[0].get("id")) is int
                    and matches[0]["id"] > 0
                ):
                    instance_id = int(matches[0]["id"])
            except Exception:
                pass
            if instance_id is None:
                print(
                    "Creation remains unconfirmed. Check Vast Instances "
                    f"for label {label} "
                    "before retrying; an instance may appear after this lookup. "
                    "Destroy that instance to stop billing.",
                    file=sys.stderr,
                    flush=True,
                )
        if instance_id is not None:
            try:
                S.release_instance(vast, instance_id)
                cleanup_confirmed = True
            except Exception as error:
                exit_code = 1
                print(
                    f"RELEASE FAILED for instance {instance_id}: "
                    f"{type(error).__name__}\n"
                    "Destroy it in the Vast console to stop billing.",
                    file=sys.stderr,
                    flush=True,
                )
        elif not creation_attempted:
            cleanup_confirmed = True
        try:
            if ssh_dir is not None:
                ssh_dir.cleanup()
        finally:
            for s, h in prev.items():
                signal.signal(s, h)
            try:
                report("stopped" if cleanup_confirmed else "cleanup_required")
            except OSError:
                print(
                    "Could not update the status file; use the cleanup result above.",
                    file=sys.stderr,
                    flush=True,
                )
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
