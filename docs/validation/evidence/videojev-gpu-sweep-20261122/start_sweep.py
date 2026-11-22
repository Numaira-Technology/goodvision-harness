"""Rent one Vast GPU, run the VideoJev gpu_sweep for BF16 then FP8 on it, release.

The client runs on the GPU box (like the frozen protocol), so network latency
stays out of the numbers. Results are copied back to a private local folder.

    SWEEP_REPO=<checkout> SWEEP_ENV_FILE=<.env> SWEEP_BUNDLE=<bundle> SWEEP_OUT=<dir> \
      PYTHONPATH=<checkout>/src:<checkout> python start_sweep.py --gpu RTX_4090 --tag 4090
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from tools import start_sam as S
from tools.cloud_prefetch import credential_blob

ROOT = Path(os.environ["SWEEP_REPO"])
ENV_FILE = Path(os.environ["SWEEP_ENV_FILE"])
BUNDLE = Path(os.environ["SWEEP_BUNDLE"])
OUT = Path(os.environ["SWEEP_OUT"])
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
BASE = "/workspace/videojev"
REPO = f"{BASE}/repo"
CACHE = f"{BASE}/models/huggingface"
RBUNDLE = f"{BASE}/bundle"
ROUT = f"{BASE}/sweep"
R2_KEYS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
LABEL_PREFIX = "goodvision-gpusweep-"

SETUP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p {CACHE}
py={BASE}/env/bin/python
pip={BASE}/env/bin/pip
if ! command -v python3.12 >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends software-properties-common curl ca-certificates git build-essential rsync
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
fi
[ -x "$py" ] || python3.12 -m venv {BASE}/env
"$pip" install --no-cache-dir --upgrade pip setuptools wheel packaging ninja
"$pip" install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$pip" install --no-cache-dir -r {REPO}/tools/videojev_probe/requirements-fp8.txt
"$pip" install --no-cache-dir flash-attn==2.8.3.post2 causal-conv1d==1.7.0 --no-build-isolation
"$py" -c '
import torch, torchao, transformers
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print("cuda ok", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torchao.__version__, transformers.__version__)
'
"""


def sweep_script(tag):
    service = (
        f"PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {BASE}/env/bin/python "
        f"-m goodvision_harness.videojev_service --cache-dir {CACHE} "
        "--ready-capacity 4 --gpu-batch-size 4 --max-pixels 409600 --prefix-cache "
        "--cpu-threads 4 --request-timeout 20 --queue-age 19.5 --queue-capacity 48 "
        "--port 8093 --host 127.0.0.1"
    )
    return f"""set -euo pipefail
cd {REPO}
mkdir -p {ROUT}
for q in bf16 fp8; do
  echo "=== {tag} $q: starting service"
  {service} --quantization $q --result-log {ROUT}/service-{tag}-$q.jsonl > {ROUT}/service-{tag}-$q.log 2>&1 &
  pid=$!
  ok=0
  for i in $(seq 1 400); do
    if ! kill -0 $pid 2>/dev/null; then echo "service exited"; tail -40 {ROUT}/service-{tag}-$q.log; exit 1; fi
    if curl -fs http://127.0.0.1:8093/ 2>/dev/null | {BASE}/env/bin/python -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("ready") is True else 1)' 2>/dev/null; then ok=1; break; fi
    sleep 3
  done
  [ $ok = 1 ] || {{ echo "service not ready"; tail -40 {ROUT}/service-{tag}-$q.log; kill $pid; exit 1; }}
  echo "=== {tag} $q: ready; running sweep"
  PYTHONPATH=src:. {BASE}/env/bin/python -m tools.unified_benchmark.gpu_sweep run \\
    --bundle {RBUNDLE} --url http://127.0.0.1:8093 --label {tag}-$q --output {ROUT}/{tag}-$q --warmup 3
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > {ROUT}/{tag}-$q/nvidia-smi.csv || true
  # Background jobs of a non-interactive bash ignore SIGINT; stop with TERM, then KILL.
  kill -TERM $pid 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done
  kill -KILL $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  sleep 5
done
echo "=== {tag}: done"
"""


def main():
    from dotenv import load_dotenv
    from vastai import VastAI

    load_dotenv(ENV_FILE, override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help="Vast gpu_name, e.g. RTX_4090 or L40S")
    ap.add_argument("--gpu-ram", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--disk-gb", type=int, default=80)
    ap.add_argument("--max-dph", type=float, default=1.5)
    ap.add_argument("--any-host", action="store_true", help="drop datacenter=true")
    args = ap.parse_args()

    vast = VastAI(api_key=S.required("VAST_API_KEY"), retry=1)
    key_path = Path(S.required("SAM_SSH_KEY")).expanduser().resolve()
    label = f"{LABEL_PREFIX}{args.tag}-{uuid4().hex[:10]}"
    status = OUT / f"status-{args.tag}.json"
    instance_id, creation_attempted, released = None, False, False

    def report(state):
        status.write_text(json.dumps({"state": state, "instance_id": instance_id,
                                      "label": label, "released": released}) + "\n")

    def interrupt(_s, _f):
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, interrupt)
    ssh_dir = tempfile.TemporaryDirectory(prefix="sweep-ssh-")
    code = 1
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        report("starting")
        key = S.prepare_ssh_key(key_path, Path(ssh_dir.name))
        options = ["-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
                   "-o", f"UserKnownHostsFile={Path(ssh_dir.name) / 'known_hosts'}",
                   "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        query = (f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 gpu_ram>={args.gpu_ram} "
                 f"disk_space>={args.disk_gb} inet_down>=200 reliability>0.95 rentable=true "
                 f"verified=true dph_total<={args.max_dph}" + ("" if args.any_host else " datacenter=true"))
        print("Searching:", query, flush=True)
        offers = vast.search_offers(query=query, type="on-demand", order="dph_total",
                                    storage=float(args.disk_gb), limit=10)
        if not offers:
            raise RuntimeError("no eligible offer")
        print("cheapest offer $/h:", [round(o.get("dph_total", 0), 3) for o in offers[:5]], flush=True)
        creation_attempted = True
        S.IMAGE = IMAGE
        instance_id = S.create_from_offers(vast, offers, disk_gb=args.disk_gb, label=label)
        report("created")
        print("Created", instance_id, label, flush=True)
        pub = subprocess.run(["ssh-keygen", "-y", "-f", str(key)], capture_output=True,
                             text=True, check=True).stdout.strip()
        try:
            vast.attach_ssh(instance_id=instance_id, ssh_key=pub)
        except Exception as error:  # still wait; gateway may already have the key
            print("attach_ssh warning:", type(error).__name__, flush=True)
        host, port = S.wait_for_ssh(vast, instance_id, options)
        ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
        rsync_e = "ssh " + " ".join(shlex.quote(o) for o in [*options, "-p", str(port)])
        S.run([*ssh, f"mkdir -p {REPO} {RBUNDLE}; command -v rsync || (apt-get update && apt-get install -y rsync)"],
              timeout=600)
        files = subprocess.run(["git", "ls-files", "-z", "--", "src", "tools"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout
        S.run(["rsync", "-az", "--no-links", "--from0", "--files-from=-", "-e", rsync_e,
               f"{ROOT}/", f"root@{host}:{REPO}/"], input=files, timeout=900)
        S.run(["rsync", "-az", "-e", rsync_e, f"{BUNDLE}/", f"root@{host}:{RBUNDLE}/"], timeout=900)
        print("Provisioning env...", flush=True)
        S.run([*ssh, "bash -s"], input=SETUP, timeout=60 * 60)
        blob = credential_blob({k: os.environ.get(k) for k in R2_KEYS}, os.environ.get("HF_TOKEN"))
        S.run([*ssh, f"set -a; . /dev/stdin; set +a; cd {REPO} && PYTHONPATH=src "
               f"{BASE}/env/bin/python -m tools.model_source --model videojev --cache-dir {CACHE}"],
              input=blob, timeout=45 * 60)
        report("running")
        try:
            S.run([*ssh, "bash -s"], input=sweep_script(args.tag), timeout=50 * 60)
            code = 0
        finally:
            dest = OUT / args.tag
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-az", "-e", rsync_e, f"root@{host}:{ROUT}/", f"{dest}/"],
                           timeout=900, check=False)
            print("Copied results to", dest, flush=True)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
        code = 130
    except Exception as error:
        print("sweep failed:", type(error).__name__, str(error)[:300], file=sys.stderr, flush=True)
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        if creation_attempted and instance_id is None:
            try:
                matches = [i for i in vast.show_instances() if isinstance(i, dict) and i.get("label") == label]
                if len(matches) == 1:
                    instance_id = int(matches[0]["id"])
            except Exception:
                pass
        if instance_id is not None:
            try:
                S.release_instance(vast, instance_id)
                released = True
                print("Released", instance_id, flush=True)
            except Exception as error:
                print(f"RELEASE FAILED {instance_id} {label}: {type(error).__name__}", file=sys.stderr, flush=True)
                code = 1
        elif creation_attempted:
            print(f"Creation unconfirmed; check Vast for label {label}", file=sys.stderr, flush=True)
        else:
            released = True
        report("stopped" if released else "cleanup_required")
        ssh_dir.cleanup()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
