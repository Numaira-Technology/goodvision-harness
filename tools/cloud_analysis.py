"""Install, persist and tunnel the full analysis runtime alongside rented GPUs."""

from __future__ import annotations

import json
import os
import shlex
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


def enabled():
    return os.environ.get("SAM_ANALYSIS_ENABLED", "false").lower() == "true"


def state_directory():
    return Path(os.environ.get("SAM_CONTROL_DIRECTORY", "/data/cloud-gpu"))


def runtime_environment(environment):
    """Only application settings cross to the GPU host, never rental/SSH keys."""
    prefixes = (
        "INFERENCE_",
        "FACE_",
        "SCENE_",
        "SAM31_",
        "LOCAL_CAMERA_",
    )
    names = {
        "AICLUB_API_KEY",
        "OPENROUTER_API_KEY",
        "DASHSCOPE_API_KEY",
        "HARNESS_DRY_RUN",
        "HARNESS_STREAM_URL",
        "RTSP_URL",
        "HARNESS_WEBHOOK_SECRET",
        "RESULT_WEBHOOK_URL",
        "LOCAL_TRACKING_FPS",
        "LOCAL_INFERENCE_API_KEY",
        "LOCAL_INFERENCE_MAX_TOKENS",
        "HARNESS_INFERENCE_PROFILE",
    }
    return {
        key: value
        for key, value in environment.items()
        if key in names or key.startswith(prefixes)
    }


def prepare(root, remote, ssh, scp, run):
    """Upload reviewed application files and last stopped snapshot over SSH."""
    directory = state_directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def source_filter(member):
        if member.issym() or member.islnk() or "__pycache__" in member.name:
            return None
        return member

    with tempfile.TemporaryDirectory(prefix="analysis-upload-") as temporary:
        bundle = Path(temporary) / "source.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            names = [
                "src",
                "pyproject.toml",
                "config.yaml",
                "tools/gpu_analysis_host.py",
            ]
            if (root / "memory").is_dir():
                names.append("memory")
            for name in names:
                archive.add(root / name, arcname=name, filter=source_filter)
        environment = Path(temporary) / "environment.json"
        environment.write_text(json.dumps(runtime_environment(os.environ)), "utf-8")
        environment.chmod(0o600)
        run([*ssh, f"mkdir -p {remote}/source {remote}/data && chmod 700 {remote}"])
        run([*scp, str(bundle), str(environment), f"{ssh[-1]}:{remote}/"])
        snapshot = directory / "analysis-data.tar.gz"
        if not snapshot.exists():
            # Import existing per-camera settings/galleries exactly once.
            snapshot = Path(temporary) / "legacy-data.tar.gz"
            with tarfile.open(snapshot, "w:gz") as archive:
                legacy = Path("/legacy")
                lanes = {
                    lane.name: lane
                    for lane in sorted(legacy.glob("camera-*"))
                    if lane.is_dir()
                }
                primary = lanes.get("camera-01")
                default = legacy / "default"
                # The launcher creates empty per-camera volumes even for an
                # older single-camera installation whose data is in face-data.
                # Empty lane 1 must not hide that gallery, settings or keys;
                # already populated per-camera state remains authoritative.
                if (
                    default.is_dir()
                    and any(default.iterdir())
                    and (primary is None or not any(primary.iterdir()))
                ):
                    lanes["camera-01"] = default
                for name, lane in sorted(lanes.items()):
                    archive.add(
                        lane, arcname=f"data/{name}", filter=source_filter
                    )
        if snapshot.exists():
            run(
                [*scp, str(snapshot), f"{ssh[-1]}:{remote}/restore.tar.gz"], timeout=120
            )
        run(
            [*ssh, "bash -s"],
            input=f"""set -euo pipefail
cd {remote}
chmod 600 environment.json
tar -xzf source.tar.gz -C source
if [ -f restore.tar.gz ] && [ -z "$(ls -A data)" ]; then
  tar -xzf restore.tar.gz
  rm restore.tar.gz
fi
apt-get update
apt-get install -y --no-install-recommends ffmpeg libgl1 libgomp1 libglib2.0-0
if [ ! -x analysis-env/bin/python ]; then
  env/bin/python -m venv analysis-env
fi
analysis-env/bin/python -m pip install --no-cache-dir './source[face,webrtc,vlm,video]'
analysis-env/bin/python -m pip uninstall -y opencv-python opencv-python-headless
analysis-env/bin/python -m pip install --no-cache-dir --no-deps \\
  opencv-python-headless==4.14.0.94
""",
            timeout=30 * 60,
        )


def worker_command(remote, environment, sam_command):
    rendered = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
    # A separate interpreter prevents Harness/face dependencies from changing
    # the pinned SAM dependency set. Both execute on this same GPU host.
    script = f"""set -euo pipefail
{sam_command} &
sam_pid=$!
env {rendered} GPU_ANALYSIS_ROOT={remote} \\
  {remote}/analysis-env/bin/python {remote}/source/tools/gpu_analysis_host.py &
analysis_pid=$!
trap 'kill "$sam_pid" "$analysis_pid" 2>/dev/null || true; wait || true' EXIT
wait -n "$sam_pid" "$analysis_pid"
"""
    return "bash -c " + shlex.quote(script)


def wait_ready(process):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("GPU host analysis process exited")
        try:
            with opener.open("http://127.0.0.1:8095/health", timeout=5) as response:
                if json.load(response).get("ready") is True:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("GPU host analysis did not become ready")


def backup(process, remote, run):
    """Drain all writers, then atomically replace the controller's durable copy.

    Failure must leave the instance intact for recovery, rather than destroy
    the only copy of an enrolled gallery or captured evidence.
    """
    ssh, scp = process.analysis_connection
    # Use a fresh SSH command so backup remains possible after a tunnel failure.
    code = (
        "import json,urllib.request; "
        "r=urllib.request.urlopen(urllib.request.Request("
        "'http://127.0.0.1:8095/stop',data=b'',method='POST'),timeout=110); "
        "assert json.load(r)['stopped'] is True"
    )
    run([*ssh, f"{remote}/analysis-env/bin/python -c {shlex.quote(code)}"], timeout=115)
    run([*ssh, f"tar -czf {remote}/analysis-data.tar.gz -C {remote} data"], timeout=120)
    destination = state_directory() / "analysis-data.tar.gz"
    partial = destination.with_suffix(".partial")
    run([*scp, f"{ssh[-1]}:{remote}/analysis-data.tar.gz", str(partial)], timeout=120)
    # Read every member before accepting the transfer (including gzip checksum).
    with tarfile.open(partial, "r:gz") as archive:
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as source:
                    while source.read(1024 * 1024):
                        pass
    partial.chmod(0o600)
    partial.replace(destination)
