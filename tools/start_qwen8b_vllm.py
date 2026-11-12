"""Rent one 4090, restore the vLLM env from R2, serve, official @12 eval, release.

The conda-pack is built on a CPU VM and uploaded to a versioned R2 key first.
This launcher does not pip-install vLLM on the GPU. Exam protocol is unchanged:
12 client slots via tools.qwen8b.eval_local, frozen JPEGs, prompts, parser,
temperature 0, max_tokens 30. Always release the instance.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
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
OLD_VLLM_KEY = "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz"
EXPECTED_R2_KEY = "envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz"
EXPECTED_SHA256 = "59314d00783b6698ef83cbfbd2e028c2f094d76c9d3383f5a37c3fdf3687b95b"
EXPECTED_SIZE = 4252375781
# Mixed conda-forge setuptools 84.2.1 + pip 80.10.2; conda-pack
# --ignore-missing-files dropped compilers.errors / compilers.platform.
# Do not rent a GPU against this object; do not overwrite it.
INCOMPLETE_VLLM_KEY = "envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz"
MAX_DPH = 1.85  # refuse silently upgrading past ~1.5x the prior $1.2296/hr
STATUS_DEFAULT = Path("outputs/qwen8b-r2-vllm/STATUS.md")
MIN_CUDA = 13.0
MIN_DRIVER_BRANCH = 580


def _vast_query(disk_gb: int) -> str:
    """The cu130 pack has a different host requirement from the HF cu128 pack."""
    gpu = os.environ.get("QWEN8B_GPU", "RTX_4090")
    return (
        f"gpu_name={gpu} num_gpus=1 cuda_max_good>={MIN_CUDA} gpu_ram>=23 "
        f"disk_space>={disk_gb} inet_down>=200 reliability>0.95 rentable=true "
        "verified=true datacenter=true"
    )


def _cuda_compatible(host: dict) -> bool:
    """Require the advertised CUDA capability and driver; missing data fails closed."""
    try:
        cuda = float(host["cuda_max_good"])
        driver = int(str(host["driver_version"]).split(".")[0])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(cuda) and cuda >= MIN_CUDA and driver >= MIN_DRIVER_BRANCH


def _verify_host_driver(ssh) -> None:
    """Check actual host compatibility before bootstrap or multi-GB downloads."""
    out = subprocess.check_output(
        [*ssh, "nvidia-smi --query-gpu=driver_version --format=csv,noheader"],
        text=True,
        timeout=30,
        stderr=subprocess.STDOUT,
    )
    drivers = [line.strip() for line in out.splitlines() if line.strip()]
    try:
        compatible = (
            len(drivers) == 1 and int(drivers[0].split(".")[0]) >= MIN_DRIVER_BRANCH
        )
    except ValueError:
        compatible = False
    if not compatible:
        raise RuntimeError(
            f"cu130 requires one GPU with NVIDIA driver >=580; observed {drivers}"
        )


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
    usable = [
        o
        for o in offers
        if int(o.get("num_gpus") or 1) == 1
        and float(o["dph_total"]) <= MAX_DPH
        and _cuda_compatible(o)
    ]
    if not offers:
        raise RuntimeError("No eligible RTX 4090 / CUDA 13.0 offer available.")
    if not usable:
        cheapest = min(float(o["dph_total"]) for o in offers)
        raise RuntimeError(
            f"No CUDA 13.0 / driver >=580 single-GPU 4090 at <= ${MAX_DPH}/hr "
            f"(cheapest ${cheapest:.4f}/hr); "
            "not silently renting a much more expensive or multi-GPU card."
        )
    return usable


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


def _vllm_serve_remote(activate: str, snapshot: str) -> str:
    """Packed python -s, no repo PYTHONPATH. Exam knobs stay 12 / bf16 / eager."""
    cfg = DEFAULTS
    python = _packed_python(activate)
    unpack = _unpack_to(activate)
    limit = json.dumps(cfg["limit_mm_per_prompt"], separators=(",", ":"))
    module_argv = [
        "vllm.entrypoints.openai.api_server",
        "--model",
        snapshot,
        "--served-model-name",
        MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(int(cfg["max_model_len"])),
        "--max-num-seqs",
        "12",
        "--max-num-batched-tokens",
        str(int(cfg["max_num_batched_tokens"])),
        "--gpu-memory-utilization",
        str(cfg["gpu_memory_utilization"]),
        "--limit-mm-per-prompt",
        limit,
        "--tensor-parallel-size",
        "1",
        "--generation-config",
        "vllm",
    ]
    if cfg.get("enforce_eager"):
        module_argv.append("--enforce-eager")
    code = _prefix_guard(unpack) + (
        "import runpy, sys\n"
        f"sys.argv = {module_argv!r}\n"
        "runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')\n"
    )
    return (
        f"set -eo pipefail; "
        f"if [ ! -x {shlex.quote(python)} ]; then "
        f"echo packed interpreter missing: {shlex.quote(python)} >&2; exit 1; fi; "
        f"cd /tmp && exec env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 "
        f"TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "
        f"{shlex.quote(python)} -s -c {shlex.quote(code)}"
    )


def _server_process(ssh, activate: str, snapshot: str, log_path: Path | None = None):
    remote = _vllm_serve_remote(activate, snapshot)
    print(
        "Starting vLLM OpenAI server + SSH tunnel "
        "(12 client slots stay in eval_local)...",
        flush=True,
    )
    logf = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Never replace retained evidence. POSIX O_EXCL already refuses any
        # existing path, including a dangling symlink; Windows follows a
        # dangling link and creates its target, so check the link itself too.
        if log_path.is_symlink() or log_path.exists():
            raise FileExistsError(f"server log already exists: {log_path}")
        logf = open(log_path, "x", buffering=1)
    try:
        proc = subprocess.Popen(
            [
                *ssh[:-1],
                "-o",
                "ExitOnForwardFailure=yes",
                "-L",
                f"127.0.0.1:{H.LOCAL_PORT}:127.0.0.1:8000",
                ssh[-1],
                remote,
            ],
            stdin=subprocess.DEVNULL,
            stdout=logf or None,
            stderr=subprocess.STDOUT if logf else None,
            start_new_session=True,
        )
    except BaseException:
        if logf is not None:
            logf.close()
        raise
    proc._log_file = logf  # type: ignore[attr-defined]
    return proc


def _wait_ready(process, timeout=25 * 60):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM/SSH exited with status {process.returncode}")
        try:
            with opener.open(
                f"http://127.0.0.1:{H.LOCAL_PORT}/v1/models", timeout=4
            ) as r:
                body = json.load(r)
            names = [m.get("id") for m in body.get("data") or []]
            if MODEL_ID in names or names:
                print(f"vLLM ready models={names}", flush=True)
                return body
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(3)
    raise RuntimeError("vLLM did not become ready in time")


def _verify_eval_bundle(bundle: Path, phase: str, env_spec: dict) -> str:
    """Offline identity/request verification before renting or sending any inference."""
    from tools.unified_benchmark.adapters import request_body
    from tools.unified_benchmark.contract import RULES, VERSION, digest
    from tools.unified_benchmark.preflight import model_configs
    from tools.unified_benchmark.workflow import verify_bundle

    bundle_sha = verify_bundle(bundle)
    hashes = json.loads((bundle / "bundle-files-sha256.json").read_text())
    config_name = "configs/qwen8b_r2_vllm.json"
    if config_name not in hashes:
        raise ValueError("vLLM config is not sealed in the bundle")
    config = json.loads((bundle / config_name).read_text())
    expected = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    if config != expected:
        raise ValueError(
            "vLLM profile differs; create a new bundle for the reviewed ada2 profile"
        )
    identity = {k: env_spec[k] for k in ("r2_key", "sha256", "size")}
    if config.get("env_artifact") != identity:
        raise ValueError(
            "vLLM profile pack identity differs from the restored registry artifact"
        )
    if phase not in ("validation", "test", "both"):
        raise ValueError("unsupported evaluation phase")
    for split in ("validation", "test") if phase == "both" else (phase,):
        name = f"plans/qwen8b_r2_vllm-{split}.json"
        if name not in hashes:
            raise ValueError(f"vLLM {split} plan is not sealed in the bundle")
        plan = json.loads((bundle / name).read_text())
        if (
            plan.get("protocol") != VERSION
            or plan.get("split") != split
            or plan.get("config_sha256") != digest(config)
            or plan.get("rules_sha256") != digest(RULES)
        ):
            raise ValueError(f"vLLM {split} plan identity differs")
        units = plan.get("units") or []
        if not units or len({u["id"] for u in units}) != len(units):
            raise ValueError(f"vLLM {split} plan requires unique request units")
        for unit in units:
            _, body = request_body(config, unit, bundle)
            if digest(body) != unit["request_sha256"]:
                raise ValueError(
                    f"vLLM {split} request fingerprint differs: {unit['id']}"
                )
    return bundle_sha


def _smoke_body() -> dict:
    """A separate synthetic 720px probe, never a scored validation/test request."""
    from PIL import Image

    jpeg = io.BytesIO()
    Image.new("RGB", (720, 404), (32, 64, 96)).save(jpeg, format="JPEG", quality=90)
    return {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 30,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Is a red square visible? Answer only with compact JSON: "
                            '{"answer": true} or {"answer": false}.'
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(jpeg.getvalue()).decode("ascii")
                        },
                    },
                ],
            }
        ],
    }


def _smoke(opener, report_path: Path):
    """Retain a separate diagnostic request/receipt, excluded from scored ledgers."""
    from tools.unified_benchmark.adapters import parse_response
    from tools.unified_benchmark.api_run import atomic
    from tools.unified_benchmark.contract import digest, write_new

    url = f"http://127.0.0.1:{H.LOCAL_PORT}/v1/chat/completions"
    body = _smoke_body()
    record = {
        "purpose": "synthetic_transport_smoke",
        "included_in_evaluation": False,
        "request": body,
        "request_sha256": digest(body),
        "state": "started",
        "started_at_unix": time.time(),
    }
    # Refuse an existing diagnostic before sending anything, retaining its evidence.
    write_new(report_path, record)
    req = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(req, timeout=120) as resp:
            raw = resp.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("oversized smoke response")
        record["response_body"] = raw.decode("utf-8", "replace")
        payload = json.loads(raw)
        digest(payload)  # Reject non-finite JSON before storing a structured receipt.
        record["response"] = payload
        if not isinstance(payload, dict) or payload.get("model") != MODEL_ID:
            raise ValueError("smoke served model identity missing or mismatched")
        parse_response(
            {"backend": "language", "answer_parser": "boolean-string-v1"},
            {"targets": ["synthetic_smoke"]},
            payload,
        )
        record["state"] = "completed"
    except URLError as error:
        detail = ""
        raw = getattr(error, "read", None)
        if callable(raw):
            try:
                detail = raw(2000).decode("utf-8", "replace")
            except OSError:
                detail = ""
        code = getattr(error, "code", None)
        record.update(state="failed", error=f"HTTP {code}: {detail or error}")
        raise RuntimeError(f"smoke HTTP {code}: {detail or error}") from error
    except Exception as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at_unix"] = time.time()
        atomic(report_path, record)
    print("vLLM synthetic smoke ok (excluded from evaluation)", flush=True)
    return payload


def _remote_snapshot(ssh) -> str:
    cmd = (
        f"cd {H.REPO} && PYTHONPATH={H.REPO} python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "from tools.model_source import snapshot_dir\n"
        f"print(snapshot_dir(Path({H.CACHE!r}), "
        f"'Qwen/Qwen3-VL-8B-Instruct', {REVISION!r}))\n"
        "PY"
    )
    out = subprocess.check_output([*ssh, cmd], text=True, timeout=60)
    return out.strip().splitlines()[-1]


def _require_r2_pack(env_spec: dict) -> str:
    """Validate local declarations; remote object availability is not checked."""
    key = env_spec.get("r2_key")
    if key == HF_ENV_KEY:
        raise RuntimeError("qwen8b-vllm must not reuse the HF env R2 key")
    if key == OLD_VLLM_KEY:
        raise RuntimeError(
            "qwen8b-vllm must not restore the superseded cu128 pack "
            f"{OLD_VLLM_KEY}; use a complete successor, not GPU file patches"
        )
    if key == INCOMPLETE_VLLM_KEY:
        raise RuntimeError(
            "qwen8b-vllm must not restore incomplete "
            f"{INCOMPLETE_VLLM_KEY} (conda setuptools 84.2.1 mixed with pip "
            "80.10.2; distutils compilers.errors/platform missing from the "
            "tarball). Pack a new versioned R2 object on CPU; do not write "
            "omitted files on the GPU and do not overwrite this key"
        )
    if key != EXPECTED_R2_KEY:
        raise RuntimeError(f"qwen8b-vllm r2_key must be {EXPECTED_R2_KEY}, got {key}")
    sha = str(env_spec.get("sha256") or "").strip()
    if not sha:
        raise RuntimeError(
            "qwen8b-vllm R2 pack is missing (env.sha256 empty). "
            "Review a new versioned key before rebuilding; published ada2 is frozen. "
            "Build CUDA wheels locally on the CPU VM with "
            "`PYTHONPATH=. python -m tools.build_env --model qwen8b-vllm "
            "--requirements tools/qwen8b_vllm/requirements.txt --skip-upload` "
            "and review successor identity/evidence separately. The 4090 only restores "
            f"{env_spec.get('r2_key') or 'the versioned qwen8b-vllm R2 key'}; "
            "it does not pip-install vLLM."
        )
    if sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"qwen8b-vllm sha256 mismatch: registry {sha} != {EXPECTED_SHA256}"
        )
    if int(env_spec.get("size") or 0) != EXPECTED_SIZE:
        raise RuntimeError(
            f"qwen8b-vllm size mismatch: registry {env_spec.get('size')} "
            f"!= {EXPECTED_SIZE}"
        )
    return sha


def _verify_restored_env(ssh, activate: str) -> None:
    """Import check only. Missing modules are a pack bug, not a GPU pip job.

    Call the packed interpreter by absolute path with -s and PYTHONNOUSERSITE=1.
    The launched process itself rejects a mismatched sys.prefix. No conda
    activate, no PATH python, no second-chance source fallback.
    """
    python = _packed_python(activate)
    unpack = _unpack_to(activate)
    probe = _prefix_guard(unpack) + (
        "print('executable', sys.executable)\n"
        "print('prefix', sys.prefix)\n"
        "import vllm, torch, typing_extensions, filelock, tqdm\n"
        "print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
        "assert torch.version.cuda, 'restored torch is not a CUDA wheel'\n"
        "print('cuda_available', torch.cuda.is_available())\n"
        "assert torch.cuda.is_available(), 'CUDA not available after restore'\n"
        "print('device', torch.cuda.get_device_name(0))\n"
        "from torch.utils.cpp_extension import load_inline\n"
        "import vllm.entrypoints.openai.api_server as _api\n"
        "print('api_server', _api.__file__)\n"
    )
    smi = (
        "(command -v nvidia-smi >/dev/null && "
        "nvidia-smi --query-gpu=driver_version,name,memory.used,memory.total "
        "--format=csv,noheader) || echo nvidia-smi=n/a"
    )
    remote = (
        f"set -eo pipefail; "
        f"if [ ! -x {shlex.quote(python)} ]; then "
        f"echo packed interpreter missing: {shlex.quote(python)} >&2; exit 1; fi; "
        f"{smi}; PYTHONNOUSERSITE=1 {shlex.quote(python)} -s -c {shlex.quote(probe)}"
    )
    try:
        out = subprocess.check_output(
            [*ssh, remote],
            text=True,
            timeout=180,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.output or "").strip() or str(error)
        print(f"verify failed:\n{detail}", flush=True)
        if "No module named" in detail:
            raise RuntimeError(
                f"restored env import failed (exit {error.returncode}); "
                f"not pip-installing vLLM on the GPU:\n{detail}"
            ) from error
        raise RuntimeError(
            "restored env import/CUDA failed; not pip-installing vLLM on the GPU:\n"
            + detail
        ) from error
    print(out, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--disk-gb", type=int, default=int(os.environ.get("QWEN8B_DISK_GB", "80"))
    )
    ap.add_argument("--resume-instance-id", type=int)
    ap.add_argument("--status-file", type=Path, default=STATUS_DEFAULT)
    ap.add_argument("--eval-bundle", type=Path, required=True)
    ap.add_argument("--eval-output", type=Path, required=True)
    ap.add_argument(
        "--eval-phase", choices=("validation", "test", "both"), default="both"
    )
    ap.add_argument(
        "--authorization-note", default="Approved self-hosted qwen8b R2 vLLM 4090 run"
    )
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    env_spec = copy.deepcopy(registry.env_of(ENV_NAME))
    pack_sha = _require_r2_pack(env_spec)
    bundle_sha = _verify_eval_bundle(args.eval_bundle, args.eval_phase, env_spec)
    H._require_new_eval_output(args.eval_output)

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
        raise RuntimeError(
            "Set R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
        )

    print(
        f"Restoring R2 pack {env_spec['r2_key']} sha256={pack_sha} "
        f"size={env_spec.get('size')} (old cu128 pack forbidden)",
        flush=True,
    )

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

    handled = [signal.SIGINT, signal.SIGTERM] + (
        [signal.SIGHUP] if hasattr(signal, "SIGHUP") else []
    )
    prev = {s: signal.signal(s, interrupt) for s in handled}
    ssh_dir = tempfile.TemporaryDirectory(prefix="qwen8b-vllm-ssh-")
    try:
        _status(
            args.status_file, f"# qwen8b_r2_vllm\nUpdated: {started}\nPhase: rent\n"
        )
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
        if args.resume_instance_id:
            inst = vast.show_instance(id=args.resume_instance_id) or {}
            if not str(inst.get("label", "")).startswith("goodvision-qwen8b"):
                raise RuntimeError("Resume requires an existing qwen8b instance.")
            if not _cuda_compatible(inst):
                raise RuntimeError(
                    "Resume requires verified CUDA >=13.0 / driver >=580 metadata; "
                    "the existing instance was not modified or released."
                )
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
                raise RuntimeError("No eligible RTX 4090 / CUDA 13.0 offer available.")
            chosen = _eligible_offers(offers)
            cheapest = min(float(o["dph_total"]) for o in chosen)
            print(
                f"{len(chosen)} single-GPU 4090 offers <= ${MAX_DPH}/hr; "
                f"cheapest ${cheapest:.4f}/hr",
                flush=True,
            )
            creation_attempted = True
            instance_id = S.create_from_offers(
                vast, chosen, disk_gb=args.disk_gb, label=label
            )
            print(f"Created instance: {instance_id}", flush=True)

        _status(
            args.status_file,
            f"# qwen8b_r2_vllm\nUpdated: {started}\nPhase: boot\n"
            f"Instance: {instance_id}\n",
        )
        H.attach_key(vast, instance_id, ssh_key_path)
        host, port = H.wait_for_ssh(vast, instance_id, options, ssh_key_path)
        ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
        _verify_host_driver(ssh)
        print("Bootstrapping python3/boto3/ffmpeg...", flush=True)
        S.run([*ssh, "bash -s"], input=H.BOOTSTRAP, timeout=15 * 60)
        print("Uploading repo...", flush=True)
        H._upload_repo(ssh)
        print(
            "Prefetching pinned weights from R2 (same revision as HF run)...",
            flush=True,
        )
        prefetch_models(
            ssh_options=options,
            host=host,
            port=port,
            remote_dir=H.REPO,
            models=[(repo, rev)],
            cache_dir=H.CACHE,
            r2_env=r2_env,
            python="python3",
        )
        print("Restoring versioned vLLM packed env from R2 (no GPU pip)...", flush=True)
        activate = restore_env(
            ssh_options=options,
            host=host,
            port=port,
            name=ENV_NAME,
            expected_env=env_spec,
            remote_dir=H.REPO,
            r2_env=r2_env,
            python="python3",
        )
        _verify_restored_env(ssh, activate)
        smi_ready = subprocess.check_output(
            [
                *ssh,
                "nvidia-smi --query-gpu=driver_version,name,memory.used,memory.total "
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
            stderr=subprocess.STDOUT,
        )
        print("nvidia-smi after restore:\n" + smi_ready, flush=True)

        snapshot = _remote_snapshot(ssh)
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        vllm_log = args.eval_output.parent / f"vllm-server-{instance_id}.log"
        worker = _server_process(ssh, activate, snapshot, vllm_log)
        _wait_ready(worker)
        H._require_new_eval_output(args.eval_output)
        smi_serve = subprocess.check_output(
            [*ssh, "nvidia-smi"],
            text=True,
            timeout=30,
            stderr=subprocess.STDOUT,
        )
        print("nvidia-smi after vLLM ready:\n" + smi_serve, flush=True)
        if (
            _verify_eval_bundle(args.eval_bundle, args.eval_phase, env_spec)
            != bundle_sha
        ):
            raise ValueError("evaluation bundle changed while preparing the server")
        opener = build_opener(ProxyHandler({}))
        _smoke(opener, args.eval_output.parent / f"vllm-smoke-{instance_id}.json")

        _status(
            args.status_file,
            "# qwen8b_r2_vllm\nPhase: official-eval @12 slots\n"
            f"Instance: {instance_id}\n"
            "Client concurrency is 12 (eval_local / contract RULES). "
            "Not lowering slots on OOM.\n",
        )
        from tools.qwen8b.eval_local import main as eval_main

        print(
            f"Official eval_local phase={args.eval_phase} "
            "model=qwen8b_r2_vllm slots=12 ...",
            flush=True,
        )
        eval_code = eval_main(
            [
                "--bundle",
                str(args.eval_bundle),
                "--output",
                str(args.eval_output),
                "--endpoint",
                f"http://127.0.0.1:{H.LOCAL_PORT}",
                "--model",
                "qwen8b_r2_vllm",
                "--phase",
                args.eval_phase,
                "--authorization-note",
                args.authorization_note,
                "--execute",
            ]
        )
        if eval_code:
            raise RuntimeError(
                f"official @12 eval_local exited {eval_code}; "
                "not retrying with fewer clients"
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
        logf = getattr(worker, "_log_file", None) if worker is not None else None
        if logf is not None:
            try:
                logf.close()
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
                H._release_instance_checked(vast, instance_id)
                err_line = f"Error: {last_error}\n" if last_error else ""
                _status(
                    args.status_file,
                    f"# qwen8b_r2_vllm\nPhase: released\nInstance: {instance_id}\n"
                    f"exit={exit_code}\n{err_line}",
                )
            except Exception as error:
                exit_code = exit_code or 1
                try:
                    _status(
                        args.status_file,
                        f"# qwen8b_r2_vllm\nPhase: release-unconfirmed\n"
                        f"Instance: {instance_id}\nError: {error}\nexit={exit_code}\n",
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
