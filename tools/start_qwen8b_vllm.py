"""Rent one 4090, restore the vLLM env from R2, serve, official @12 eval, release.

The conda-pack is built on a CPU VM and uploaded to a versioned R2 key first.
This launcher does not pip-install vLLM on the GPU. Exam protocol is unchanged:
12 client slots via tools.qwen8b.eval_local, frozen JPEGs, prompts, parser,
temperature 0, max_tokens 30. Always release the instance.
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
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from tools import registry
from tools import start_qwen8b_r2 as H
from tools import start_sam as S
from tools.cloud_env import restore_env
from tools.cloud_prefetch import prefetch_models
from tools.qwen8b.serve_vllm import DEFAULTS, MODEL_ID, REVISION

ENV_NAME = "qwen8b-vllm"
HF_ENV_KEY = "envs/qwen8b/cu128-py312-torch280-ada.tar.gz"
MAX_DPH = 1.85  # refuse silently upgrading past ~1.5x the prior $1.2296/hr
STATUS_DEFAULT = Path("outputs/qwen8b-r2-vllm/STATUS.md")


def _status(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    art = Path("/opt/cursor/artifacts/qwen8b-r2-vllm-STATUS.md")
    try:
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text(text)
    except OSError:
        pass


def _eligible_offers(offers: list[dict]) -> list[dict]:
    usable = [o for o in offers if int(o.get("num_gpus") or 1) == 1
              and float(o["dph_total"]) <= MAX_DPH]
    if not offers:
        raise RuntimeError("No eligible RTX 4090 / CUDA 12.8 offer available.")
    if not usable:
        cheapest = min(float(o["dph_total"]) for o in offers)
        raise RuntimeError(
            f"No single-GPU 4090 at <= ${MAX_DPH}/hr (cheapest ${cheapest:.4f}/hr); "
            "not silently renting a much more expensive or multi-GPU card."
        )
    return usable


def _server_process(ssh, activate: str, snapshot: str):
    cfg = DEFAULTS
    limit = json.dumps(cfg["limit_mm_per_prompt"], separators=(",", ":"))
    eager = "--enforce-eager" if cfg.get("enforce_eager") else ""
    remote = (
        f"set -eo pipefail; set +u; source {shlex.quote(activate)}; "
        f"cd {H.REPO} && exec env PYTHONPATH={H.REPO} HF_HUB_OFFLINE=1 "
        f"TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {shlex.quote(snapshot)} "
        f"--served-model-name {MODEL_ID} "
        f"--host 127.0.0.1 --port 8000 --dtype bfloat16 "
        f"--max-model-len {int(cfg['max_model_len'])} "
        f"--max-num-seqs 12 "
        f"--max-num-batched-tokens {int(cfg['max_num_batched_tokens'])} "
        f"--gpu-memory-utilization {cfg['gpu_memory_utilization']} "
        f"--limit-mm-per-prompt {shlex.quote(limit)} "
        f"--tensor-parallel-size 1 --generation-config vllm {eager}"
    )
    print("Starting vLLM OpenAI server + SSH tunnel (12 client slots stay in eval_local)...",
          flush=True)
    return subprocess.Popen(
        [*ssh[:-1], "-o", "ExitOnForwardFailure=yes",
         "-L", f"127.0.0.1:{H.LOCAL_PORT}:127.0.0.1:8000", ssh[-1], remote],
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def _wait_ready(process, timeout=25 * 60):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM/SSH exited with status {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{H.LOCAL_PORT}/v1/models", timeout=4) as r:
                body = json.load(r)
            names = [m.get("id") for m in body.get("data") or []]
            if MODEL_ID in names or names:
                print(f"vLLM ready models={names}", flush=True)
                return body
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(3)
    raise RuntimeError("vLLM did not become ready in time")


def _smoke(opener):
    """One official-shaped request (JPEG data URI, max_tokens=30, temperature 0)."""
    url = f"http://127.0.0.1:{H.LOCAL_PORT}/v1/chat/completions"
    # 1x1 JPEG is only connectivity; official eval uses frozen 720px frames.
    tiny = (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAd"
        "Hx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3"
        "Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAAEAAQMBEQACEQEDEQH/"
        "xAAbAAABBQEBAAAAAAAAAAAAAAADAAIEBQYBB//EABQBAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEA"
        "AhEDEQA/AKpAAf/Z"
    )
    body = {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 30,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Is any fighting visible in the current image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + tiny}},
        ]}],
    }
    req = Request(url, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=120) as resp:
        payload = json.load(resp)
    model = payload.get("model")
    if model and model != MODEL_ID:
        raise RuntimeError(f"smoke served model mismatch: {model}")
    print("vLLM smoke ok", flush=True)
    return payload


def _remote_snapshot(ssh) -> str:
    cmd = (
        f"cd {H.REPO} && PYTHONPATH={H.REPO} python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "from tools.model_source import snapshot_dir\n"
        f"print(snapshot_dir(Path({H.CACHE!r}), 'Qwen/Qwen3-VL-8B-Instruct', {REVISION!r}))\n"
        "PY"
    )
    out = subprocess.check_output([*ssh, cmd], text=True, timeout=60)
    return out.strip().splitlines()[-1]


def _require_r2_pack(env_spec: dict) -> str:
    """Refuse to rent a GPU until the versioned vLLM tarball is on R2."""
    if env_spec.get("r2_key") == HF_ENV_KEY:
        raise RuntimeError("qwen8b-vllm must not reuse the HF env R2 key")
    sha = str(env_spec.get("sha256") or "").strip()
    if not sha:
        raise RuntimeError(
            "qwen8b-vllm R2 pack is missing (env.sha256 empty). "
            "Pack CUDA wheels on the CPU VM with "
            "`PYTHONPATH=. python -m tools.build_env --model qwen8b-vllm "
            "--requirements tools/qwen8b_vllm/requirements.txt` "
            "and fill env.sha256/size. The 4090 only restores "
            "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz; "
            "it does not pip-install vLLM."
        )
    return sha


def _verify_restored_env(ssh, activate: str) -> None:
    """Import check only. Missing modules are a pack bug, not a GPU pip job.

    Do not `set -u` around conda-pack activate (it reads possibly-unset
    ``PS1`` / ``CONDA_PREFIX``). Do not wrap a heredoc in ``bash -lc`` over
    SSH — that hid the traceback on instance 59781901 (exit 1, no stdout).
    """
    remote = (
        f"set +u; source {shlex.quote(activate)}; "
        "python -c " + shlex.quote(
            "import vllm, torch\n"
            "print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
            "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
            "assert torch.version.cuda, 'restored torch is not a CUDA wheel'\n"
        )
    )
    try:
        out = subprocess.check_output(
            [*ssh, remote],
            text=True, timeout=120, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.output or "").strip() or str(error)
        raise RuntimeError(
            f"restored env import failed (exit {error.returncode}); "
            f"not pip-installing vLLM on the GPU:\n{detail}"
        ) from error
    print(out, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disk-gb", type=int, default=int(os.environ.get("QWEN8B_DISK_GB", "80")))
    ap.add_argument("--resume-instance-id", type=int)
    ap.add_argument("--status-file", type=Path, default=STATUS_DEFAULT)
    ap.add_argument("--eval-bundle", type=Path, required=True)
    ap.add_argument("--eval-output", type=Path, required=True)
    ap.add_argument("--eval-phase", choices=("validation", "test", "both"), default="both")
    ap.add_argument("--authorization-note",
                    default="Approved self-hosted qwen8b R2 vLLM 4090 run")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    from dotenv import load_dotenv
    from vastai import VastAI

    load_dotenv(H.ROOT / ".env", override=False)
    S.IMAGE = H.IMAGE
    api_key = S.required("VAST_API_KEY")
    ssh_key_path = Path(S.required("SAM_SSH_KEY")).expanduser().resolve()
    if not ssh_key_path.is_file():
        raise RuntimeError(f"SSH key not found: {ssh_key_path}")
    r2_env = {k: os.environ.get(k, "") for k in H.R2_KEYS}
    if not all(r2_env.values()):
        raise RuntimeError("Set R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")

    env_spec = registry.env_of(ENV_NAME)
    _require_r2_pack(env_spec)

    vast = VastAI(api_key=api_key, retry=1)
    instance_id = None
    worker = None
    last_error = ""
    label = f"goodvision-qwen8b-vllm-{uuid4().hex[:12]}"
    exit_code = 1
    creation_attempted = False
    repo, rev = registry.weights_of(ENV_NAME)
    started = time.strftime("%Y-%m-%d %H:%M EST")

    def interrupt(_s, _f):
        raise KeyboardInterrupt

    handled = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
    prev = {s: signal.signal(s, interrupt) for s in handled}
    ssh_dir = tempfile.TemporaryDirectory(prefix="qwen8b-vllm-ssh-")
    try:
        _status(args.status_file, f"# qwen8b_r2_vllm\nUpdated: {started}\nPhase: rent\n")
        key = S.prepare_ssh_key(ssh_key_path, Path(ssh_dir.name))
        options = [
            "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
            "-o", f"UserKnownHostsFile={Path(ssh_dir.name) / 'known_hosts'}",
            "-o", "TCPKeepAlive=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=20",
        ]
        if args.resume_instance_id:
            inst = vast.show_instance(id=args.resume_instance_id) or {}
            if not str(inst.get("label", "")).startswith("goodvision-qwen8b"):
                raise RuntimeError("Resume requires an existing qwen8b instance.")
            instance_id = args.resume_instance_id
            print(f"Recovering instance {instance_id}; no new rental.", flush=True)
        else:
            print(f"Searching Vast for {H._vast_query(args.disk_gb)} ...", flush=True)
            offers = vast.search_offers(query=H._vast_query(args.disk_gb),
                                        type="on-demand", order="dph_total",
                                        storage=float(args.disk_gb), limit=10)
            if not offers:
                raise RuntimeError("No eligible RTX 4090 / CUDA 12.8 offer available.")
            chosen = _eligible_offers(offers)
            cheapest = min(float(o["dph_total"]) for o in chosen)
            print(
                f"{len(chosen)} single-GPU 4090 offers <= ${MAX_DPH}/hr; "
                f"cheapest ${cheapest:.4f}/hr",
                flush=True,
            )
            creation_attempted = True
            instance_id = S.create_from_offers(vast, chosen, disk_gb=args.disk_gb, label=label)
            print(f"Created instance: {instance_id}", flush=True)

        _status(args.status_file,
                f"# qwen8b_r2_vllm\nUpdated: {started}\nPhase: boot\nInstance: {instance_id}\n")
        H.attach_key(vast, instance_id, ssh_key_path)
        host, port = H.wait_for_ssh(vast, instance_id, options, ssh_key_path)
        ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
        print("Bootstrapping python3/boto3/ffmpeg...", flush=True)
        S.run([*ssh, "bash -s"], input=H.BOOTSTRAP, timeout=15 * 60)
        print("Uploading repo...", flush=True)
        H._upload_repo(ssh)
        print("Prefetching pinned weights from R2 (same revision as HF run)...", flush=True)
        prefetch_models(
            ssh_options=options, host=host, port=port, remote_dir=H.REPO,
            models=[(repo, rev)], cache_dir=H.CACHE, r2_env=r2_env, python="python3",
        )
        print("Restoring versioned vLLM packed env from R2 (no GPU pip)...", flush=True)
        activate = restore_env(
            ssh_options=options, host=host, port=port, name=ENV_NAME,
            remote_dir=H.REPO, r2_env=r2_env, python="python3",
        )
        _verify_restored_env(ssh, activate)

        snapshot = _remote_snapshot(ssh)
        worker = _server_process(ssh, activate, snapshot)
        _wait_ready(worker)
        opener = build_opener(ProxyHandler({}))
        _smoke(opener)

        _status(args.status_file,
                f"# qwen8b_r2_vllm\nPhase: official-eval @12 slots\nInstance: {instance_id}\n"
                "Client concurrency is 12 (eval_local / contract RULES). "
                "Not lowering slots on OOM.\n")
        from tools.qwen8b.eval_local import main as eval_main
        print("Official eval_local phase=%s model=qwen8b_r2_vllm slots=12 ..." % args.eval_phase,
              flush=True)
        eval_code = eval_main([
            "--bundle", str(args.eval_bundle),
            "--output", str(args.eval_output),
            "--endpoint", f"http://127.0.0.1:{H.LOCAL_PORT}",
            "--model", "qwen8b_r2_vllm",
            "--phase", args.eval_phase,
            "--authorization-note", args.authorization_note,
            "--execute",
        ])
        if eval_code:
            raise RuntimeError(
                f"official @12 eval_local exited {eval_code}; not retrying with fewer clients"
            )
        exit_code = 0
    except KeyboardInterrupt:
        print("\nStopping vLLM GPU session...", flush=True)
        exit_code = 130
    except Exception as error:
        last_error = str(error)
        print(f"qwen8b-vllm start failed: {error}", file=sys.stderr, flush=True)
        _status(args.status_file, f"# qwen8b_r2_vllm\nPhase: failed\nError: {error}\n")
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
                S.release_instance(vast, instance_id)
                err_line = f"Error: {last_error}\n" if last_error else ""
                _status(args.status_file,
                        f"# qwen8b_r2_vllm\nPhase: released\nInstance: {instance_id}\n"
                        f"exit={exit_code}\n{err_line}")
            except Exception as error:
                print(f"RELEASE FAILED for instance {instance_id}: {error}\n"
                      "Destroy it in the Vast console to stop billing.",
                      file=sys.stderr, flush=True)
        ssh_dir.cleanup()
        for s, h in prev.items():
            signal.signal(s, h)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
