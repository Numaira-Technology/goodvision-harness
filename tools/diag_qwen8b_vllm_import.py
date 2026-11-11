"""One RTX 4090: restore the pinned vLLM R2 pack and capture import traces.

Authorized import diagnostic only. Does not prefetch weights, start vLLM, or
run eval_local. Always attach SSH at create and release this labelled instance.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from tools import registry
from tools import start_qwen8b_r2 as H
from tools import start_qwen8b_vllm as V
from tools import start_sam as S
from tools.cloud_env import restore_env

ENV_NAME = "qwen8b-vllm"
PRIOR_DPH = 1.2296
# Report-and-stop well below the eval launcher cap if the market jumps.
MAX_DPH = 1.50
DISK_GB = 40
REMOTE = "/workspace/qwen8b-vllm-diag"
OUT_DEFAULT = Path("outputs/qwen8b-r2-vllm/diag-import.json")
BOOTSTRAP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update
apt-get install -y --no-install-recommends python3 python3-pip ca-certificates
python3 -m pip install --break-system-packages --no-cache-dir \\
  boto3==1.43.123 PyYAML==6.0.2
mkdir -p {REMOTE}
"""
OLD_PROBE = (
    "import vllm, torch\n"
    "print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
    "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
    "assert torch.version.cuda, 'restored torch is not a CUDA wheel'\n"
)
NEW_PROBE = (
    "import sys\n"
    "print('executable', sys.executable)\n"
    "import vllm, torch\n"
    "print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
    "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
    "assert torch.version.cuda, 'restored torch is not a CUDA wheel'\n"
    "from torch.utils.cpp_extension import load_inline\n"
    "import vllm.entrypoints.openai.api_server as _api\n"
    "print('api_server', _api.__file__)\n"
)
CUDA_PROBE = (
    "import sys, os, torch, vllm\n"
    "print('executable', sys.executable)\n"
    "print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
    "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
    "print('cuda_is_available', torch.cuda.is_available())\n"
    "print('device_count', torch.cuda.device_count())\n"
    "assert torch.cuda.is_available(), 'CUDA unavailable after restore'\n"
    "assert torch.cuda.device_count() > 0, 'No CUDA device after restore'\n"
    "print('device0', torch.cuda.get_device_name(0))\n"
    "print('capability', '.'.join(str(x) for x in "
    "torch.cuda.get_device_capability(0)))\n"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    art = Path("/opt/cursor/artifacts") / path.name
    try:
        art.write_text(path.read_text())
    except OSError:
        pass


def _ssh_run(ssh: list[str], remote: str, *, timeout: int = 180) -> dict:
    started = time.monotonic()
    proc = subprocess.run(
        [*ssh, remote],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "seconds": round(time.monotonic() - started, 3),
    }


def _old_verify_remote(packed_python: str) -> str:
    # cloud_env returns bin/python. Only this historical comparison uses activate.
    activate = str(Path(packed_python).with_name("activate"))
    inner = (
        f"set +u; source {shlex.quote(activate)} || exit $?; set -u; "
        "python - <<'PY'\n" + OLD_PROBE + "PY"
    )
    return "bash -lc " + shlex.quote(inner)


def _isolated_probe_remote(packed_python: str, code: str) -> str:
    python = V._packed_python(packed_python)
    guarded = V._prefix_guard(V._unpack_to(python)) + code
    return (
        "unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1; "
        f"{shlex.quote(python)} -s -c {shlex.quote(guarded)}"
    )


def _new_verify_remote(packed_python: str) -> str:
    return (
        "set +u; "
        "(command -v nvidia-smi >/dev/null && "
        "nvidia-smi --query-gpu=driver_version,name --format=csv,noheader) "
        "|| echo nvidia-smi=n/a; " + _isolated_probe_remote(packed_python, NEW_PROBE)
    )


def _eligible(offers: list[dict]) -> list[dict]:
    usable = [
        o
        for o in offers
        if int(o.get("num_gpus") or 1) == 1
        and float(o.get("gpu_ram") or 0) <= 30000
        and float(o["dph_total"]) <= MAX_DPH
        and V._cuda_compatible(o)
    ]
    if not offers:
        raise RuntimeError("No RTX 4090 offers returned.")
    if not usable:
        cheapest = min(float(o["dph_total"]) for o in offers)
        raise RuntimeError(
            f"No CUDA 13.0 / driver >=580 single 24GB 4090 at <= ${MAX_DPH}/hr "
            f"(cheapest ${cheapest:.4f}/hr); "
            "not renting a much more expensive or multi-GPU card."
        )
    cheapest = min(float(o["dph_total"]) for o in usable)
    if cheapest > PRIOR_DPH * 1.25:
        raise RuntimeError(
            f"Cheapest eligible 4090 is ${cheapest:.4f}/hr "
            f"vs prior ${PRIOR_DPH:.4f}/hr; "
            "not renting without a new price authorization."
        )
    return usable


def main() -> int:
    from dotenv import load_dotenv
    from vastai import VastAI

    load_dotenv(H.ROOT / ".env", override=False)
    S.IMAGE = H.IMAGE
    api_key = S.required("VAST_API_KEY")
    ssh_key_path = Path(S.required("SAM_SSH_KEY")).expanduser().resolve()
    r2_env = {k: os.environ.get(k, "") for k in H.R2_KEYS}
    if not ssh_key_path.is_file():
        raise RuntimeError(f"SSH key not found: {ssh_key_path}")
    if not all(r2_env.values()):
        raise RuntimeError("Set R2_* credentials")
    env_spec = copy.deepcopy(registry.env_of(ENV_NAME))
    V._require_r2_pack(env_spec)

    out_path = OUT_DEFAULT
    label = f"goodvision-qwen8b-vllm-diag-{uuid4().hex[:12]}"
    report: dict = {
        "phase": "start",
        "label": label,
        "ok": False,
        "dph_selected": None,
        "authorization": "import-diag-only",
        "eval": False,
        "prefetch_weights": False,
        "started_est": time.strftime("%Y-%m-%d %H:%M EST"),
    }
    _write(out_path, report)

    vast = VastAI(api_key=api_key, retry=1)
    instance_id = None
    creation_attempted = False
    exit_code = 1

    def interrupt(_s, _f):
        raise KeyboardInterrupt

    handled = [signal.SIGINT, signal.SIGTERM] + (
        [signal.SIGHUP] if hasattr(signal, "SIGHUP") else []
    )
    prev = {s: signal.signal(s, interrupt) for s in handled}
    ssh_dir = tempfile.TemporaryDirectory(prefix="qwen8b-vllm-diag-ssh-")
    try:
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
            "TCPKeepAlive=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
        ]
        query = V._vast_query(DISK_GB)
        print(f"Searching Vast for {query} ...", flush=True)
        offers = (
            vast.search_offers(
                query=query,
                type="on-demand",
                order="dph_total",
                storage=float(DISK_GB),
                limit=10,
            )
            or []
        )
        chosen = _eligible(offers)
        cheapest = min(float(o["dph_total"]) for o in chosen)
        print(
            f"{len(chosen)} single 24GB 4090 offers <= ${MAX_DPH}/hr; "
            f"cheapest ${cheapest:.4f}/hr (prior ${PRIOR_DPH:.4f}/hr)",
            flush=True,
        )
        # The helper returns a contract ID, not a selected offer/price binding.
        # Keep dph_selected unknown, including after a more expensive fallback.
        report["dph_cheapest_eligible"] = cheapest
        report["phase"] = "creating"
        report["create_intent"] = {"label": label, "state": "prepared"}
        # A successful synchronous checkpoint precedes dispatch. This is not a
        # crash-durable journal, and prepared intent does not prove a sent call.
        _write(out_path, report)
        creation_attempted = True
        instance_id = S.create_from_offers(
            vast,
            chosen,
            disk_gb=DISK_GB,
            label=label,
        )
        report["instance_id"] = instance_id
        report["phase"] = "boot"
        _write(out_path, report)
        print(f"Created instance: {instance_id}", flush=True)
        H.attach_key(vast, instance_id, ssh_key_path)
        host, port = H.wait_for_ssh(vast, instance_id, options, ssh_key_path)
        ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
        V._verify_host_driver(ssh)
        report["ssh"] = f"{host}:{port}"
        print("Bootstrapping python3/boto3 (no ffmpeg, no model pull)...", flush=True)
        S.run([*ssh, "bash -s"], input=BOOTSTRAP, timeout=15 * 60)
        print(
            "Restoring versioned vLLM packed env from R2 (no GPU pip, no weights)...",
            flush=True,
        )
        packed_python = restore_env(
            ssh_options=options,
            host=host,
            port=port,
            name=ENV_NAME,
            expected_env=env_spec,
            remote_dir=REMOTE,
            r2_env=r2_env,
            python="python3",
        )
        report["python"] = packed_python
        report["activate"] = str(Path(packed_python).with_name("activate"))
        report["phase"] = "verify"
        _write(out_path, report)

        host_info = _ssh_run(
            ssh,
            (
                "nvidia-smi; echo '---'; "
                "nvidia-smi --query-gpu=driver_version,name,compute_cap "
                "--format=csv; echo '---'; echo CONDA_PREFIX=${CONDA_PREFIX-}; "
                "echo PATH=$PATH; command -v python; command -v python3; "
                "python3 -c 'import sys; print(sys.executable, sys.version)'; "
                "ls -l /opt/env/qwen8b-vllm/bin/python "
                "/opt/env/qwen8b-vllm/bin/activate"
            ),
            timeout=60,
        )
        report["host"] = host_info

        old = _ssh_run(ssh, _old_verify_remote(packed_python), timeout=180)
        report["old_verify"] = old
        print(f"OLD verify rc={old['rc']}", flush=True)
        print(old["stdout"] or old["stderr"], flush=True)

        new = _ssh_run(ssh, _new_verify_remote(packed_python), timeout=180)
        report["new_verify"] = new
        print(f"NEW verify rc={new['rc']}", flush=True)
        print(new["stdout"] or new["stderr"], flush=True)

        python = V._packed_python(packed_python)
        cuda = _ssh_run(
            ssh, _isolated_probe_remote(packed_python, CUDA_PROBE), timeout=180
        )
        report["cuda_probe"] = cuda
        print(f"CUDA probe rc={cuda['rc']}", flush=True)
        print(cuda["stdout"] or cuda["stderr"], flush=True)

        site_packages = Path(python).parent.parent / "lib/python3.12/site-packages"
        libs = _ssh_run(
            ssh,
            (
                f"set +u; PYLIB={shlex.quote(str(site_packages))}; "
                "echo '--- libcudart ---'; "
                "find /opt/env/qwen8b-vllm -name 'libcudart.so*' | head; "
                "echo '--- libcuda ---'; "
                "ldconfig -p 2>/dev/null | grep -i libcuda || true; "
                "ls -l /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null || true; "
                "echo '--- ldd libtorch_cuda ---'; "
                "ldd $PYLIB/torch/lib/libtorch_cuda.so 2>/dev/null | "
                "grep -E 'libcuda|libcudart|not found' | head"
            ),
            timeout=60,
        )
        report["libs"] = libs

        report["phase"] = "done"
        report["checks_passed"] = all(probe["rc"] == 0 for probe in (old, new, cuda))
        report["readiness_scope"] = (
            "deep imports and CUDA device visibility; no kernel or model inference"
        )
        exit_code = 0 if report["checks_passed"] else 1
        _write(out_path, report)
    except KeyboardInterrupt:
        exit_code = 130
        report["phase"] = "interrupted"
        try:
            _write(out_path, report)
        except OSError as error:
            print(f"Could not persist interruption report: {error}", file=sys.stderr)
    except Exception as error:
        exit_code = 1
        report["phase"] = "failed"
        report["error"] = str(error)
        try:
            _write(out_path, report)
        except OSError as report_error:
            print(f"Could not persist failure report: {report_error}", file=sys.stderr)
        print(f"diag failed: {error}", file=sys.stderr, flush=True)
    finally:
        for s in handled:
            signal.signal(s, signal.SIG_IGN)
        if creation_attempted and instance_id is None:
            try:
                matches = [i for i in vast.show_instances() if i.get("label") == label]
                if len(matches) == 1:
                    instance_id = int(matches[0]["id"])
                    report["instance_id"] = instance_id
            except Exception:
                pass
        if instance_id is not None:
            try:
                H._release_instance_checked(vast, instance_id)
                report["released"] = True
            except Exception as error:
                report["released"] = False
                exit_code = exit_code or 1
                report["release_error"] = str(error)
                print(
                    f"RELEASE FAILED for instance {instance_id}: {error}\n"
                    "Destroy it in the Vast console to stop billing.",
                    file=sys.stderr,
                    flush=True,
                )
        try:
            rows = vast.show_instances()
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or row.get("id") is None for row in rows
            ):
                raise RuntimeError("Invalid live-instance response")
            report["live_after"] = [
                {"id": row["id"], "label": row.get("label")} for row in rows
            ]
            if instance_id is not None and any(
                str(row["id"]) == str(instance_id) for row in rows
            ):
                report["released"] = False
                report["release_error"] = "Owned instance remains listed after cleanup"
                exit_code = exit_code or 1
        except Exception as error:
            report["live_after"] = None
            report["live_check_error"] = str(error)
            exit_code = exit_code or 1
        report["ok"] = exit_code == 0 and report.get("released") is True
        report["exit_code"] = exit_code
        report["finished_est"] = time.strftime("%Y-%m-%d %H:%M EST")
        try:
            _write(out_path, report)
        except OSError as error:
            print(
                f"Could not persist final diagnostic report: {error}", file=sys.stderr
            )
            exit_code = exit_code or 1
        finally:
            try:
                ssh_dir.cleanup()
            finally:
                for s, h in prev.items():
                    signal.signal(s, h)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
