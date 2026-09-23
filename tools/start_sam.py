"""Manage a remote SAM GPU pool and SSH tunnel for the Compose controller."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

if __package__:
    from .cloud_gpu_settings import CloudGPUSettings
else:
    try:
        from cloud_gpu_settings import CloudGPUSettings
    except ModuleNotFoundError:
        from tools.cloud_gpu_settings import CloudGPUSettings

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/goodvision-sam"
IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
SAM_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
# Runs through Vast before the controller needs SSH. Putting this in SETUP
# would be too late when sshd rejects the injected authorized_keys permissions.
# Preserve the existing keys; never replace them or disable StrictModes.
SSH_ONSTART = """set -eu
chown root:root /root/.ssh /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
"""
WORKER_FILES = (
    "app.py", "sam31.py", "continuous.py", "streaming_state.py",
    "perception_routes.py", "sam_live.py", "multistream.py", "gpu_memory.py", "cloud_pool.py",
    "h264_transport.py", "requirements.txt",
)
WORKER_ROUTES = ("sam_live",)
WORKER_ENV = {
    "SAM31_ENABLED": "true",
    "SAM31_SESSION_ENABLED": "false",
    "SAM31_LIVE_ENABLED": "true",
    "SAM_SCENE_IMAGE_ENABLED": "false",
    "SAM31_SESSION_H264_ENABLED": "true",
    "SAM31_SESSION_HOTSTART_DELAY": "5",
    "SAM31_SESSION_POSTPROCESS_BATCH_SIZE": "1",
    "SAM31_COMPILE_MODE": "off",
    "SAM31_INFERENCE_MAX_SIDE": "640",
    "SAM31_CHECKPOINT_PATH": f"{REMOTE}/models/sam3.1_multiplex.pt",
    "HF_HUB_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
}

# Validate the installed versions and pinned SAM source before skipping setup.
# A directory alone does not prove that a previous installation completed.
RUNTIME_CHECK = f"""import json, sys
from importlib import metadata
from pathlib import Path
assert sys.version_info[:2] == (3, 12), 'Python 3.12 required'
assert metadata.version('torch') == '2.10.0+cu126', 'PyTorch CUDA build mismatch'
assert metadata.version('torchvision') == '0.25.0+cu126', 'torchvision build mismatch'
for line in Path('requirements.txt').read_text().splitlines():
    line = line.split('#', 1)[0].strip()
    if line:
        name, expected = line.split('==')
        assert metadata.version(name) == expected, f'Runtime dependency mismatch: {{name}}'
source = json.loads(metadata.distribution('sam3').read_text('direct_url.json') or '{{}}')
assert source.get('vcs_info', {{}}).get('commit_id') == {SAM_COMMIT!r}, 'SAM source mismatch'
from sam3.model_builder import build_sam3_multiplex_video_predictor
import app, sam_live
"""

# Execute inside the rented container, without nesting another Docker daemon.
# Keep credentials out of this script and out of remote command arguments.
SETUP = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd {REMOTE}/worker
if [ -x {REMOTE}/env/bin/python ] && {REMOTE}/env/bin/python -c {shlex.quote(RUNTIME_CHECK)}; then
  echo 'Reusing verified SAM runtime; installation skipped.'
  exit 0
fi
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git build-essential libgomp1 libglib2.0-0
cd {REMOTE}
if [ ! -x {REMOTE}/conda/bin/conda ]; then
  curl --fail --location --retry 3 \\
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \\
  -o miniforge.sh
  bash miniforge.sh -b -p {REMOTE}/conda
fi
if [ ! -x {REMOTE}/env/bin/python ]; then
  {REMOTE}/conda/bin/conda create -y -p {REMOTE}/env python=3.12 pip
fi
{REMOTE}/env/bin/python -m pip install --no-cache-dir \\
  torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
{REMOTE}/env/bin/python -m pip install --no-cache-dir -r worker/requirements.txt
{REMOTE}/env/bin/python -m pip install --no-cache-dir \\
  git+https://github.com/facebookresearch/sam3.git@{SAM_COMMIT}
# Check the standalone upload before downloading weights or starting the tunnel.
cd {REMOTE}/worker
{REMOTE}/env/bin/python -c {shlex.quote(RUNTIME_CHECK)}
"""


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the repository .env or your terminal.")
    return value


class CreationRejected(RuntimeError):
    """Vast explicitly refused creation; no contract needs reconciliation."""


def creation_diagnostic(payload, *, status: int | None = None) -> str:
    """Keep the provider's error fields without logging requests or credentials."""
    fields = []

    def collect(value, depth=0):
        if not isinstance(value, dict) or depth > 2:
            return
        for name in ("code", "error_code", "error", "message", "msg", "detail"):
            item = value.get(name)
            if isinstance(item, dict):
                collect(item, depth + 1)
            elif isinstance(item, (str, int)) and not isinstance(item, bool):
                fields.append(f"{name}={item}")

    collect(payload)
    detail = "; ".join(fields) or "No structured error details returned"
    # Redact before truncating so a clipped credential cannot escape replacement.
    for name in ("VAST_API_KEY", "HF_TOKEN", "SAM_SSH_KEY"):
        secret = os.environ.get(name, "")
        if secret:
            detail = detail.replace(secret, "[redacted]")
    detail = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|$)",
        "[redacted private key]", detail, flags=re.DOTALL,
    )
    detail = re.sub(
        r"(?i)\b(?:bearer\s+|hf_)[a-z0-9._~+/=-]+", "[redacted]", detail,
    )
    detail = re.sub(
        r"(?i)\b(api[_ -]?key|token|password|secret|authorization)\s*[:=]\s*"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1=[redacted]", detail,
    )
    detail = " ".join(detail.split())[:1000]
    return f"HTTP {status}: {detail}" if status is not None else detail


def offer_was_unavailable(payload) -> bool:
    """Only recognize explicit capacity/offer refusals, never generic HTTP 400s."""
    if not isinstance(payload, dict):
        return False
    codes = {
        "no_such_ask", "no_such_offer", "ask_not_found", "offer_not_found",
        "ask_unavailable", "offer_unavailable", "machine_unavailable",
        "offer_already_rented", "ask_already_rented", "offer_taken",
    }
    for name in ("code", "error_code"):
        value = payload.get(name)
        if isinstance(value, str) and value.lower() not in codes:
            return False
    error = payload.get("error")
    if (
        isinstance(error, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", error)
        and error.lower() not in codes
    ):
        return False
    if isinstance(payload.get("error"), dict):
        return offer_was_unavailable(payload["error"])
    for name in ("code", "error_code", "error"):
        value = payload.get(name)
        if isinstance(value, str) and value.lower() in codes:
            return True
    # Keep the unavailable phrase adjacent to its subject: "image not found"
    # or "invalid parameters for offer 123" must not trigger another rental.
    message = creation_diagnostic(payload).lower()
    return re.search(
        r"\b(?:offer|ask|machine)(?:\s+(?:id\s+)?#?\d+)?\s+"
        r"(?:(?:is|was|has been)\s+)?(?:not found|does not exist|no longer exists|"
        r"not available|unavailable|no longer available|already (?:rented|taken))\b",
        message,
    ) is not None


def create_from_offers(
    vast, offers: list[dict], *, disk_gb: int, label: str,
    attempted_machines: set[str] | None = None,
) -> int:
    """Share a three-machine budget across creation and startup failures."""
    seen_machines = attempted_machines if attempted_machines is not None else set()
    seen_offers = set()
    attempts = len(seen_machines)
    for offer in sorted(offers, key=lambda item: float(item["dph_total"])):
        offer_id = int(offer["id"])
        # Without a machine identity we cannot prove that another unknown offer
        # is a different host, so try at most one such candidate.
        machine = str(offer.get("machine_id") or "unknown")
        if offer_id in seen_offers or machine in seen_machines:
            continue
        if attempts == 3:
            break
        seen_offers.add(offer_id)
        seen_machines.add(machine)
        attempts += 1
        print(
            f"Selected offer: {offer_id}; location: {offer.get('geolocation') or 'unknown'}; "
            f"estimated total hourly price: "
            f"${float(offer['dph_total']):.4f}; candidate {attempts}/3", flush=True,
        )
        print(f"Creating instance with label: {label}", flush=True)
        status = None
        try:
            created = vast.create_instance(
                id=offer_id, image=IMAGE, disk=disk_gb,
                label=label, ssh=True, direct=True, cancel_unavail=True,
                onstart_cmd=SSH_ONSTART,
            )
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            try:
                created = response.json() if response is not None else None
            except (ValueError, TypeError):
                created = None
            detail = creation_diagnostic(created, status=status)
            # A timeout, server error or lost response may hide a successful
            # creation. Stop here; main reconciles this unique label before exit.
            if not isinstance(status, int) or not 400 <= status < 500 or status == 408:
                raise RuntimeError(
                    f"Creation outcome unknown for offer {offer_id} "
                    f"({type(error).__name__}); {detail}"
                ) from None
        else:
            contract = (
                created.get("new_contract") if isinstance(created, dict) else None
            )
            if (
                isinstance(created, dict) and created.get("success") is True
                and isinstance(contract, int) and not isinstance(contract, bool)
                and contract > 0
            ):
                return contract
            if not isinstance(created, dict) or created.get("success") is not False:
                raise RuntimeError(
                    f"Creation outcome unknown for offer {offer_id}: "
                    "invalid creation response; "
                    + creation_diagnostic(created)
                )
        detail = creation_diagnostic(created, status=status)
        if isinstance(created, dict) and created.get("new_contract"):
            raise RuntimeError(
                f"Creation outcome unknown for offer {offer_id}: "
                f"refusal includes a contract; {detail}"
            )
        if status in {None, 400, 404, 409, 410} and offer_was_unavailable(created):
            print(
                f"Offer {offer_id} was refused: {detail}; skipping this machine.",
                flush=True,
            )
            continue
        raise CreationRejected(
            f"Creation refused for offer {offer_id}; {detail}. No instance created."
        )
    raise CreationRejected(
        f"No instance created: no untried offers remain after {attempts} machine attempts "
        "(at most 3 machines per launch)."
    )


def prepare_ssh_key(source: Path, directory: Path) -> Path:
    """Copy a host-mounted key with private permissions and validate before renting."""
    key = directory / "identity"
    # Windows Docker bind mounts may report 0777. Create a 0600 copy in the
    # controller's private temporary directory without changing the host file.
    descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as target, source.open("rb") as original:
        shutil.copyfileobj(original, target)
    key.chmod(0o600)
    result = subprocess.run(
        ["ssh-keygen", "-y", "-P", "", "-f", str(key)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=10,
    )
    if result.returncode:
        raise RuntimeError(
            "SAM_SSH_KEY must be a valid private key usable without a passphrase; "
            "SSH key validation failed before renting a GPU."
        )
    return key


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
        targets = []
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
        ssh_ports = (instance.get("ports") or {}).get("22/tcp", [])
        if instance.get("public_ipaddr") and ssh_ports:
            targets.append((instance["public_ipaddr"], ssh_ports[0]["HostPort"]))
        if host and port and (host, port) not in targets:
            targets.append((host, port))
        # Prefer direct access, but a blocked direct port must not prevent use
        # of the ordinary Vast SSH gateway with the same configured key.
        for host, port in targets:
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
            for route in WORKER_ROUTES:
                with opener.open(
                    f"http://127.0.0.1:{port}/health/perception/{route}", timeout=3,
                ) as response:
                    state = json.load(response)
                if not all(state.get(name) for name in ("enabled", "ready", "gpu_ready")):
                    break
            else:
                return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError("SAM continuous analysis did not become ready within 10 minutes; see logs above.")


def release_instance(vast, instance_id: int) -> None:
    """Require confirmed cleanup before another paid rental can be created."""
    print(f"Destroying instance: {instance_id}", flush=True)
    previous_timeout = vast.client.timeout
    vast.client.timeout = (5, 20)
    try:
        result = vast.destroy_instance(id=instance_id)
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError("Vast did not confirm instance destruction.")
    finally:
        vast.client.timeout = previous_timeout
    print(f"Released instance: {instance_id}", flush=True)


def start_worker(vast, instance_id: int, options: list[str], *, hf_token: str,
                 sam_port: int, settings: CloudGPUSettings | None = None):
    settings = settings or CloudGPUSettings()
    print("Waiting for SSH...", flush=True)
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
    print("Checking cached SAM 3.1 weights; downloading if needed...", flush=True)
    download = (
        "import sys; from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id='facebook/sam3.1', filename='sam3.1_multiplex.pt', "
        f"local_dir='{REMOTE}/models', token=sys.stdin.read().strip())"
    )
    run([
        *ssh, f"HF_HUB_OFFLINE=0 {REMOTE}/env/bin/python -c {shlex.quote(download)}",
    ], input=hf_token, timeout=30 * 60)
    environment = dict(WORKER_ENV, SAM_GPU_COUNT=str(settings.gpu_count),
                       SAM31_LIVE_CAMERAS=str(settings.streams))
    worker_environment = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in environment.items()
    )
    worker_command = (
        f"cd {REMOTE}/worker && exec env "
        f"{worker_environment} "
        f"{REMOTE}/env/bin/python -m uvicorn cloud_pool:app --host 127.0.0.1 --port 8092 --workers 1"
    )
    print("Starting SAM and its SSH tunnel; waiting for model loading (logs follow)...", flush=True)
    return subprocess.Popen([
        *ssh[:-1], "-o", "ExitOnForwardFailure=yes",
        # Compose keeps this port inside its network; it is not published.
        "-L", f"0.0.0.0:{sam_port}:127.0.0.1:8092",
        ssh[-1], worker_command,
    ], stdin=subprocess.DEVNULL, start_new_session=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-key", type=Path, help="SSH private key; defaults to SAM_SSH_KEY.")
    parser.add_argument("--disk-gb", type=int, help="Rented disk size; defaults to SAM_DISK_GB or 50 GB.")
    parser.add_argument("--gpu-name", help="Marketplace GPU name, or A100 for the A100 family.")
    parser.add_argument("--gpu-count", type=int, help="Exact rented GPU count; defaults to SAM_GPU_COUNT or 1.")
    parser.add_argument("--streams", type=int, help="Video lanes; defaults to SAM31_LIVE_CAMERAS or 1, without a product cap.")
    parser.add_argument("--status-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sam-port", type=int, default=8092, help="SSH forwarding port inside the controller (default: 8092).")
    parser.add_argument(
        "--resume-instance-id", type=int,
        help="Recover an existing GoodVision instance without renting another. "
        "Failure or normal shutdown releases it, as with a new rental.",
    )
    args = parser.parse_args()

    vast = None
    instance_id = None
    creation_attempted = False
    worker = None
    offers = None
    attempted_machines: set[str] = set()
    ssh_directory = None
    exit_code = 1
    label = f"goodvision-sam-{uuid4().hex[:12]}"
    cleanup_confirmed = True
    failure = None

    def report(state, *, clean=False):
        if args.status_file is not None:
            temporary = args.status_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"state": state, "cleanup_confirmed": clean,
                                            "instance_id": instance_id, "label": label,
                                            "error": failure}), encoding="utf-8")
            temporary.replace(args.status_file)

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
        defaults = CloudGPUSettings.from_environment().public()
        settings = CloudGPUSettings(**{name: getattr(args, name) if getattr(args, name) is not None
                                      else value for name, value in defaults.items()})
        args.disk_gb = settings.disk_gb
        gpu_name = settings.gpu_name
        for program in ("ssh", "scp", "ssh-keygen"):
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

        ssh_directory = tempfile.TemporaryDirectory(prefix="goodvision-sam-ssh-")
        ssh_key = prepare_ssh_key(ssh_key, Path(ssh_directory.name))
        # vastai 1.5.6 counts attempts, so retry=1 sends exactly one request.
        vast = VastAI(api_key=api_key, retry=1)
        if args.resume_instance_id is not None:
            if args.resume_instance_id <= 0:
                raise RuntimeError("Resume instance ID must be positive.")
            existing = vast.show_instance(id=args.resume_instance_id) or {}
            if (
                existing.get("id") != args.resume_instance_id
                or not str(existing.get("label", "")).startswith("goodvision-sam-")
                or existing.get("actual_status") != "running"
            ):
                raise RuntimeError("Resume requires an existing running GoodVision SAM instance.")
            instance_id = args.resume_instance_id
            cleanup_confirmed = False
            report("starting")
            print(f"Recovering existing instance: {instance_id}; no new rental.", flush=True)
        else:
            print(
                f"Searching for {settings.gpu_count} on-demand {gpu_name} GPUs "
                "(CUDA >= 12.6; mainland China excluded)...", flush=True,
            )
            offers = vast.search_offers(
                query=settings.query(),
                type="on-demand", order="dph_total", storage=float(args.disk_gb), limit=10,
            )
            if not offers:
                raise RuntimeError(
                    f"No eligible {settings.gpu_count} x {gpu_name} offer outside mainland China is available. "
                    "No instance was created."
                )
        options = [
            "-i", str(ssh_key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
            "-o", f"UserKnownHostsFile={Path(ssh_directory.name) / 'known_hosts'}",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        ]
        while True:
            if instance_id is None:
                creation_attempted = True
                cleanup_confirmed = False
                report("starting")
                try:
                    instance_id = create_from_offers(
                        vast, offers, disk_gb=args.disk_gb, label=label,
                        attempted_machines=attempted_machines,
                    )
                except CreationRejected:
                    creation_attempted = False
                    cleanup_confirmed = True
                    raise
                print(f"Created instance: {instance_id}", flush=True)
                report("starting")
            try:
                worker = start_worker(
                    vast, instance_id, options, hf_token=hf_token, sam_port=args.sam_port,
                    settings=settings,
                )
                wait_for_worker(worker, args.sam_port)
            except Exception as error:
                # Explicit resume never creates a rental; signals also bypass
                # this handler. Once ready, a live session is not replaced.
                if offers is None or len(attempted_machines) >= 3:
                    raise
                print(
                    f"Startup failed on instance {instance_id}: "
                    f"{creation_diagnostic({'message': str(error)})}; "
                    "releasing it before trying a different machine.", flush=True,
                )
                stop_process(worker)
                worker = None
                release_instance(vast, instance_id)
                cleanup_confirmed = True
                instance_id = None
                creation_attempted = False
                label = f"goodvision-sam-{uuid4().hex[:12]}"
                continue
            break
        report("ready")
        print(
            "SAM is ready: sam_live passed its GPU health check.\n"
            f"Perception routes: http://sam-remote:{args.sam_port}/v1/perception/routes\n"
            "Stop the Compose application to release this GPU. Closing the browser does not release it.",
            flush=True,
        )
        worker.wait()
        raise RuntimeError(f"SAM/SSH exited with status {worker.returncode}; releasing the instance.")
    except KeyboardInterrupt:
        print("\nStopping SAM session...", flush=True)
        exit_code = 130
    except Exception as error:
        failure = creation_diagnostic({"message": str(error)})
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
                detail = creation_diagnostic({"message": str(error)})
                print(
                    f"Could not look up the unconfirmed creation: {detail}",
                    file=sys.stderr, flush=True,
                )
            if instance_id is None:
                print(
                    f"Creation remains unconfirmed. Check Vast Instances for label {label} before retrying.\n"
                    "An instance may appear after this lookup. Destroy that instance in Vast Instances.",
                    file=sys.stderr, flush=True,
                )
        if instance_id is not None:
            try:
                release_instance(vast, instance_id)
                cleanup_confirmed = True
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
        report(("error" if failure else "stopped") if cleanup_confirmed else "cleanup_required", clean=cleanup_confirmed)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
