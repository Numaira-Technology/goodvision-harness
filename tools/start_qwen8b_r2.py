"""Rent one Vast 4090, restore the qwen8b packed env + weights from R2, serve it.

Attach the SSH key immediately after create (team API keys do not inject it at
boot). Do not sit out a 20-minute Waiting-for-SSH hang on Permission denied —
re-attach once, then fail that machine. Always release this labelled instance.
"""

from __future__ import annotations

import argparse
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
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from tools import registry
from tools import start_sam as S
from tools.cloud_env import restore_env
from tools.cloud_prefetch import prefetch_models

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
REMOTE = "/workspace/qwen8b-r2"
REPO = f"{REMOTE}/repo"
CACHE = f"{REMOTE}/cache"
LOCAL_PORT = 8000
R2_KEYS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
BOOTSTRAP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update
apt-get install -y --no-install-recommends python3 python3-pip ffmpeg ca-certificates
python3 -m pip install --break-system-packages --no-cache-dir \\
  boto3==1.43.123 PyYAML==6.0.2
mkdir -p {REPO} {CACHE}
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


def attach_key(vast, instance_id: int, key_path: Path) -> None:
    """Inject our public key at minute 0 so it propagates during the SSH wait."""
    res = vast.attach_ssh(instance_id=instance_id, ssh_key=_public_key(key_path))
    msg = res.get("msg") if isinstance(res, dict) else res
    print(f"Attached SSH key to instance {instance_id}: {msg}", flush=True)


def wait_for_ssh(
    vast,
    instance_id: int,
    options: list[str],
    key_path: Path,
    *,
    deny_budget=90,
    boot_budget=8 * 60,
) -> tuple[str, int]:
    """Fail a publickey denial quickly instead of burning the 20-minute poll."""
    boot_deadline = time.monotonic() + boot_budget
    deny_deadline = None
    reattached = False
    while time.monotonic() < boot_deadline:
        instance = vast.show_instance(id=instance_id) or {}
        if instance.get("actual_status") in {"offline", "exited", "stopped", "frozen"}:
            raise RuntimeError(
                "Instance stopped before SSH was ready: "
                f"{instance.get('status_msg', '')}"
            )
        targets = []
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
        ssh_ports = (instance.get("ports") or {}).get("22/tcp", [])
        if instance.get("public_ipaddr") and ssh_ports:
            targets.append((instance["public_ipaddr"], ssh_ports[0]["HostPort"]))
        if host and port and (host, port) not in targets:
            targets.append((host, port))
        denied = False
        for host, port in targets:
            command = ["ssh", *options, "-p", str(port), f"root@{host}", "true"]
            try:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=15,
                )
            except subprocess.TimeoutExpired:
                continue
            if result.returncode == 0:
                print(f"SSH connected: {host}:{port}", flush=True)
                return str(host), int(port)
            err = (result.stderr or "").lower()
            if "permission denied" in err and "publickey" in err:
                denied = True
        if denied:
            if deny_deadline is None:
                deny_deadline = time.monotonic() + deny_budget
                print(
                    "SSH offered our key but the box rejected it; re-attaching.",
                    flush=True,
                )
                if not reattached:
                    attach_key(vast, instance_id, key_path)
                    reattached = True
            if time.monotonic() >= deny_deadline:
                raise RuntimeError(
                    "SSH publickey denied after key attach; "
                    "not waiting out the 20-minute poll."
                )
        time.sleep(5)
    raise RuntimeError("SSH did not become available before the boot budget elapsed.")


def _vast_query(disk_gb: int) -> str:
    gpu = os.environ.get("QWEN8B_GPU", "RTX_4090")
    return (
        f"gpu_name={gpu} num_gpus=1 cuda_vers>=12.8 gpu_ram>=23 "
        f"disk_space>={disk_gb} inet_down>=200 reliability>0.95 rentable=true "
        "verified=true datacenter=true"
    )


def _upload_repo(ssh):
    """Stream src/ and tools/ over SSH without requiring rsync on the controller."""
    S.run([*ssh, f"mkdir -p {REPO}"])
    remote = " ".join(shlex.quote(part) for part in ssh)
    command = (
        "tar czf - --exclude=__pycache__ --exclude=*.pyc --exclude=.venv "
        f"-C {shlex.quote(str(ROOT))} src tools | "
        f"{remote} tar xzf - -C {shlex.quote(REPO)}"
    )
    result = subprocess.run(command, shell=True, timeout=600)
    if result.returncode:
        raise RuntimeError("repo upload failed")


def _posix_prefix(value: str) -> str | None:
    """Return a leading-slash registry path unchanged, or None for a host path."""
    text = str(value)
    if text.startswith("/") and not text.startswith("//"):
        return text.rstrip("/")
    return None


def _packed_python(activate: str) -> str:
    """Absolute packed interpreter. Never PATH python or `source activate`.

    A registry prefix such as ``/opt/env/...`` must stay a POSIX string.
    pathlib rewrites a leading slash to the current drive root on Windows.
    """
    text = _posix_prefix(activate)
    if text is not None:
        python = text.rsplit("/", 1)[0] + "/python"
    else:
        python = str(Path(activate).parent / "python")
    name = python.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name != "python":
        raise RuntimeError(f"packed interpreter path is unsafe: {python}")
    return python


def _unpack_to(activate: str) -> str:
    text = _posix_prefix(activate)
    if text is not None:
        return text.rsplit("/", 1)[0].rsplit("/", 1)[0]
    return str(Path(activate).parent.parent)


def _prefix_guard(unpack_to: str) -> str:
    return (
        "import os, sys, site\n"
        f"want = os.path.realpath({unpack_to!r})\n"
        "got = os.path.realpath(sys.prefix)\n"
        "if got != want:\n"
        "    raise SystemExit(f'sys.prefix {got!r} != restored env {want!r}')\n"
        "if site.ENABLE_USER_SITE:\n"
        "    raise SystemExit('user site must be disabled (PYTHONNOUSERSITE)')\n"
    )


def _serve_remote(activate: str) -> str:
    python = _packed_python(activate)
    unpack = _unpack_to(activate)
    code = _prefix_guard(unpack) + (
        "import runpy, sys\n"
        "sys.argv = ['tools.qwen8b.serve', '--cache-dir', "
        f"{CACHE!r}, '--host', '127.0.0.1', '--port', '8000']\n"
        "runpy.run_module('tools.qwen8b.serve', run_name='__main__')\n"
    )
    return (
        f"set -eo pipefail; "
        f"if [ ! -x {shlex.quote(python)} ]; then "
        f"echo packed interpreter missing: {shlex.quote(python)} >&2; exit 1; fi; "
        f"cd {shlex.quote(REPO)} && exec env PYTHONNOUSERSITE=1 "
        f"PYTHONPATH={shlex.quote(REPO)} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"{shlex.quote(python)} -s -c {shlex.quote(code)}"
    )


def _server_process(ssh, activate: str):
    repo, rev = registry.weights_of("qwen8b")
    remote = _serve_remote(activate)
    print(
        f"Starting qwen8b serve + SSH tunnel on :{LOCAL_PORT} ({repo}@{rev[:12]})...",
        flush=True,
    )
    return subprocess.Popen(
        [
            *ssh[:-1],
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:8000",
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
                f"qwen8b/SSH exited with status {process.returncode}; see logs above."
            )
        try:
            with opener.open(f"http://127.0.0.1:{LOCAL_PORT}/health", timeout=4) as r:
                state = json.load(r)
            if state.get("ready"):
                print(
                    f"qwen8b ready: {state.get('model')} on {state.get('gpu')}",
                    flush=True,
                )
                return state
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError("qwen8b server did not become ready in time; see logs above.")


def _verify_eval_bundle(bundle: Path, phase: str, env_spec=None) -> str:
    """Check the frozen HF profile and requests before any rental or inference."""
    from tools.qwen8b.prepare_vllm_bundle import HOST, _plans
    from tools.unified_benchmark.preflight import model_configs
    from tools.unified_benchmark.workflow import verify_bundle

    if phase not in ("validation", "test", "both"):
        raise ValueError("unsupported evaluation phase")
    bundle_sha = verify_bundle(bundle)
    hashes = json.loads((bundle / "bundle-files-sha256.json").read_text())
    # Frozen HF bundles contain both phases; validate both even for a test-only run.
    config = model_configs([HOST])[HOST]
    env_spec = registry.env_of("qwen8b") if env_spec is None else env_spec
    registry.validate_env(env_spec)
    if config.get("env_artifact") != {
        key: env_spec[key] for key in ("r2_key", "sha256", "size")
    }:
        raise ValueError("HF profile and selected environment artifact differ")
    _plans(bundle, HOST, config, hashes)
    return bundle_sha


def _require_new_eval_output(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError(f"evaluation output already exists: {output}")


def _release_instance_checked(vast, instance_id: int) -> None:
    """Require destruction acknowledgement and absence of this owned instance."""
    S.release_instance(vast, instance_id)
    rows = vast.show_instances()
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or row.get("id") is None for row in rows
    ):
        raise RuntimeError("Cannot confirm instance absence after destruction")
    if any(str(row["id"]) == str(instance_id) for row in rows):
        raise RuntimeError(f"Instance {instance_id} remains listed after destruction")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--disk-gb", type=int, default=int(os.environ.get("QWEN8B_DISK_GB", "80"))
    )
    ap.add_argument("--resume-instance-id", type=int)
    ap.add_argument("--status-file", type=Path)
    ap.add_argument(
        "--eval-bundle",
        type=Path,
        help="run tools.qwen8b.eval_local against the tunnel, then release",
    )
    ap.add_argument("--eval-output", type=Path)
    ap.add_argument(
        "--eval-phase", choices=("validation", "test", "both"), default="both"
    )
    ap.add_argument(
        "--authorization-note", default="Approved self-hosted qwen8b R2 4090 run"
    )
    ap.add_argument(
        "--hold",
        action="store_true",
        help="keep the tunnel up until Ctrl+C (default when --eval-bundle is omitted)",
    )
    args = ap.parse_args()
    if args.eval_bundle and not args.eval_output:
        ap.error("--eval-output is required with --eval-bundle")

    env_spec = copy.deepcopy(registry.env_of("qwen8b"))
    registry.validate_env(env_spec)
    bundle_sha = None
    if args.eval_bundle:
        _require_new_eval_output(args.eval_output)
        bundle_sha = _verify_eval_bundle(args.eval_bundle, args.eval_phase, env_spec)

    from dotenv import load_dotenv
    from vastai import VastAI

    load_dotenv(ROOT / ".env", override=False)
    S.IMAGE = IMAGE
    api_key = S.required("VAST_API_KEY")
    ssh_key_path = Path(S.required("SAM_SSH_KEY")).expanduser().resolve()
    if not ssh_key_path.is_file():
        raise RuntimeError(f"SSH key not found: {ssh_key_path}")
    r2_env = {k: os.environ.get(k, "") for k in R2_KEYS}
    if not all(r2_env.values()):
        raise RuntimeError(
            "Set R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in .env"
        )

    vast = VastAI(api_key=api_key, retry=1)
    instance_id = None
    worker = None
    label = f"goodvision-qwen8b-{uuid4().hex[:12]}"
    exit_code = 1
    creation_attempted = False
    repo, rev = registry.weights_of("qwen8b")

    def interrupt(_s, _f):
        raise KeyboardInterrupt

    handled = [signal.SIGINT, signal.SIGTERM] + (
        [signal.SIGHUP] if hasattr(signal, "SIGHUP") else []
    )
    prev = {s: signal.signal(s, interrupt) for s in handled}
    ssh_dir = tempfile.TemporaryDirectory(prefix="qwen8b-ssh-")
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
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if args.resume_instance_id:
            inst = vast.show_instance(id=args.resume_instance_id) or {}
            if not str(inst.get("label", "")).startswith("goodvision-qwen8b-"):
                raise RuntimeError("Resume requires an existing qwen8b-r2 instance.")
            instance_id = args.resume_instance_id
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
            instance_id = S.create_from_offers(
                vast, offers, disk_gb=args.disk_gb, label=label
            )
            print(f"Created instance: {instance_id}", flush=True)

        attach_key(vast, instance_id, ssh_key_path)
        print("Waiting for SSH (publickey denial fails fast)...", flush=True)
        host, port = wait_for_ssh(vast, instance_id, options, ssh_key_path)
        ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
        print(
            "Bootstrapping python3/boto3/ffmpeg (no flash-attn compile)...", flush=True
        )
        S.run([*ssh, "bash -s"], input=BOOTSTRAP, timeout=15 * 60)
        print("Uploading repo...", flush=True)
        _upload_repo(ssh)
        print("Restoring qwen8b packed env from R2...", flush=True)
        activate = restore_env(
            ssh_options=options,
            host=host,
            port=port,
            name="qwen8b",
            expected_env=env_spec,
            remote_dir=REPO,
            r2_env=r2_env,
            python="python3",
        )
        print(f"env python: {_packed_python(activate)}", flush=True)
        print("Prefetching qwen8b weights from R2...", flush=True)
        prefetch_models(
            ssh_options=options,
            host=host,
            port=port,
            remote_dir=REPO,
            models=[(repo, rev)],
            cache_dir=CACHE,
            r2_env=r2_env,
            python="python3",
        )
        worker = _server_process(ssh, activate)
        state = _wait_ready(worker)
        print(
            f"\nqwen8b is live on http://127.0.0.1:{LOCAL_PORT}/v1/chat/completions\n"
            f"Eval: PYTHONPATH=. python -m tools.qwen8b.eval_local --bundle <bundle> "
            f"--endpoint http://127.0.0.1:{LOCAL_PORT} --output outputs/qwen8b-r2-host "
            f"--authorization-note {args.authorization_note!r} --execute\n"
            f"Instance {instance_id} is billing; Ctrl+C here releases it.\n",
            flush=True,
        )
        if args.status_file:
            args.status_file.write_text(
                f"ready instance={instance_id} host={host} port={port} "
                f"gpu={state.get('gpu', '')}\n"
            )
        if args.eval_bundle:
            _require_new_eval_output(args.eval_output)
            if (
                _verify_eval_bundle(args.eval_bundle, args.eval_phase, env_spec)
                != bundle_sha
            ):
                raise ValueError("evaluation bundle changed during server preparation")
            from tools.qwen8b.eval_local import main as eval_main

            print(f"Running eval_local phase={args.eval_phase} ...", flush=True)
            eval_code = eval_main(
                [
                    "--bundle",
                    str(args.eval_bundle),
                    "--output",
                    str(args.eval_output),
                    "--endpoint",
                    f"http://127.0.0.1:{LOCAL_PORT}",
                    "--phase",
                    args.eval_phase,
                    "--authorization-note",
                    args.authorization_note,
                    "--execute",
                ]
            )
            if eval_code:
                raise RuntimeError(f"eval_local exited {eval_code}")
            exit_code = 0
        else:
            print(
                "Holding tunnel until interrupt (always releases on exit).", flush=True
            )
            worker.wait()
            raise RuntimeError(f"qwen8b/SSH exited with status {worker.returncode}.")
    except KeyboardInterrupt:
        print("\nStopping qwen8b GPU session...", flush=True)
        exit_code = 130
    except Exception as error:
        print(f"qwen8b-r2 start failed: {error}", file=sys.stderr, flush=True)
    finally:
        for s in handled:
            signal.signal(s, signal.SIG_IGN)
        try:
            S.stop_process(worker)
        except OSError:
            pass
        if creation_attempted and instance_id is None:
            try:
                matches = [i for i in vast.show_instances() if i.get("label") == label]
                if len(matches) == 1:
                    instance_id = int(matches[0]["id"])
            except Exception:
                pass
        if instance_id is not None:
            try:
                _release_instance_checked(vast, instance_id)
            except Exception as error:
                exit_code = exit_code or 1
                if args.status_file:
                    try:
                        args.status_file.write_text(
                            f"release-unconfirmed instance={instance_id} "
                            f"exit={exit_code}\n{error}\n"
                        )
                    except OSError as status_error:
                        print(
                            f"Could not write release status: {status_error}",
                            file=sys.stderr,
                        )
                print(
                    f"RELEASE FAILED for instance {instance_id}: {error}\n"
                    "Destroy it in the Vast console to stop billing.",
                    file=sys.stderr,
                    flush=True,
                )
        try:
            ssh_dir.cleanup()
        finally:
            for s, h in prev.items():
                signal.signal(s, h)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
