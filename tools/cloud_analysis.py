"""Install, persist and tunnel the full analysis runtime alongside rented GPUs."""

from __future__ import annotations

import gzip
import json
import os
import shlex
import subprocess
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
        "HARNESS_RELAY_KEY",
        "HARNESS_STREAM_URL",
        "RTSP_URL",
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "HARNESS_WEBHOOK_SECRET",
        "RESULT_WEBHOOK_URL",
        "LOCAL_TRACKING_FPS",
        "LOCAL_INFERENCE_API_KEY",
        "LOCAL_INFERENCE_MAX_TOKENS",
        "HARNESS_INFERENCE_PROFILE",
    }
    selected = {
        key: value
        for key, value in environment.items()
        if key in names or key.startswith(prefixes)
    }
    # Match service.main's legacy alias precedence, including an explicitly
    # empty alias; send only the canonical setting to the GPU analysis lanes.
    selected["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = environment.get(
        "VIDEO_CAPTURE_OPTIONS",
        environment.get(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000",
        ),
    )
    return selected


def prepare(root, remote, ssh, scp, run):
    """Upload reviewed application files and last stopped snapshot over SSH."""
    directory = state_directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def source_filter(member):
        if member.issym() or member.islnk() or "__pycache__" in member.name:
            return None
        return member

    with tempfile.TemporaryDirectory(
        prefix="analysis-upload-", dir=directory,
    ) as temporary:
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
        run(
            [*scp, str(bundle), str(environment), f"{ssh[-1]}:{remote}/"],
            timeout=operation_timeout(
                bundle.stat().st_size + environment.stat().st_size,
            ),
        )
        face_root = Path(
            os.environ.get("FACE_MODEL_ROOT") or root / ".models/insightface"
        ).expanduser()
        face_bundle = Path(temporary) / "face-models.tar.gz"
        if face_root.is_dir() and any(face_root.iterdir()):
            with tarfile.open(face_bundle, "w:gz") as archive:
                archive.add(face_root, arcname="models/face", filter=source_filter)
            run(
                [*scp, str(face_bundle), f"{ssh[-1]}:{remote}/face-models.tar.gz"],
                timeout=operation_timeout(face_bundle.stat().st_size),
            )
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
                [*scp, str(snapshot), f"{ssh[-1]}:{remote}/restore.tar.gz"],
                timeout=operation_timeout(snapshot.stat().st_size)
            )
        run(
            [*ssh, "bash -s"],
            input=f"""set -euo pipefail
cd {remote}
chmod 600 environment.json
tar -xzf source.tar.gz -C source
{remote}/env/bin/python - <<'RESTORE_ANALYSIS'
from pathlib import Path
import tarfile
root = Path({remote!r})
data_empty = not any((root / "data").iterdir())
for filename, supplied in (("restore.tar.gz", False), ("face-models.tar.gz", True)):
    path = root / filename
    if not path.exists():
        continue
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name.rstrip("/")
            is_face = name == "models/face" or name.startswith("models/face/")
            is_data = name == "data" or name.startswith("data/")
            if not is_face and not (not supplied and data_empty and is_data):
                continue
            if member.issym() or member.islnk():
                continue
            destination = root / name
            if is_face and destination.exists() and (not supplied or not data_empty):
                continue
            archive.extract(member, root, filter="data")
    path.unlink()
RESTORE_ANALYSIS
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
            timeout=30 * 60 + operation_timeout(sum(
                path.stat().st_size
                for path in (snapshot, face_bundle)
                if path.exists()
            )),
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


def operation_timeout(size_bytes):
    """Allow setup plus transfer/compression at a conservative 1 MiB/s.

    The budget grows with retention, without a fixed total-duration cap.
    """
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("snapshot size must be a nonnegative byte count")
    rate = int(os.environ.get("SAM_ANALYSIS_BACKUP_BYTES_PER_SECOND", str(1024**2)))
    if rate <= 0:
        raise ValueError("SAM_ANALYSIS_BACKUP_BYTES_PER_SECOND must be positive")
    return 120 + (size_bytes + rate - 1) // rate


def remote_snapshot_size(ssh, remote, *, archive=False):
    """Read byte counts over SSH, independently of the camera tunnel."""
    if archive:
        code = (
            "from pathlib import Path; "
            f"print((Path({remote!r}) / 'analysis-data.tar.gz').stat().st_size)"
        )
    else:
        code = (
            "from pathlib import Path; import subprocess; "
            f"root=Path({remote!r}); "
            "paths=[root/'data']; "
            "paths += [root/'models/face'] if (root/'models/face').is_dir() else []; "
            "print(sum(int(subprocess.check_output(['du','-sb',str(path)])"
            ".split()[0]) for path in paths))"
        )
    output = subprocess.check_output(
        [*ssh, f"{shlex.quote(remote)}/analysis-env/bin/python -c {shlex.quote(code)}"],
        text=True, timeout=120,
    )
    size = int(output.strip())
    if size < 0:
        raise ValueError("invalid remote snapshot size")
    return size


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
    uncompressed = remote_snapshot_size(ssh, remote)
    archive_command = (
        f"cd {shlex.quote(remote)} && "
        "if [ -d models/face ]; then "
        "tar -czf analysis-data.tar.gz data models/face; "
        "else tar -czf analysis-data.tar.gz data; fi"
    )
    run([*ssh, archive_command], timeout=operation_timeout(uncompressed))
    compressed = remote_snapshot_size(ssh, remote, archive=True)
    destination = state_directory() / "analysis-data.tar.gz"
    partial = destination.with_suffix(".partial")
    run(
        [*scp, f"{ssh[-1]}:{remote}/analysis-data.tar.gz", str(partial)],
        timeout=operation_timeout(compressed),
    )
    # Tar iteration can stop at its end marker before gzip checks the trailer.
    # Read every member, then drain padding to EOF to validate CRC and ISIZE
    # before accepting the transfer or replacing the last complete snapshot.
    with gzip.open(partial, "rb") as payload:
        with tarfile.open(fileobj=payload, mode="r:") as archive:
            for member in archive:
                if member.isfile():
                    with archive.extractfile(member) as source:
                        while source.read(1024 * 1024):
                            pass
        while payload.read(1024 * 1024):
            pass
    partial.chmod(0o600)
    partial.replace(destination)
