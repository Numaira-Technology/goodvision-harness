"""Deploy cloud CPU analysis beside SAM; keep capture and presentation in the browser.

The CPU application uses its own environment so installing its dependencies cannot
change the pinned SAM runtime. Only an explicit inference configuration allowlist
crosses SSH; cloud-account and model-download credentials are never bundled.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

REMOTE = "/workspace/goodvision-sam/analysis"
REMOTE_PORT = 8200
ENVIRONMENT_KEYS = frozenset(
    {
        "OPENROUTER_API_KEY",
        "DASHSCOPE_API_KEY",
        "AICLUB_API_KEY",
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "INFERENCE_API_KEY",
        "SCENE_VLM_OUTPUT_MODE",
        "SCENE_INDEPENDENT_VLM",
        "SCENE_DIAGNOSTICS_ENABLED",
        "SCENE_OBSERVATION_FPS",
        "SCENE_MAX_IN_FLIGHT",
        "SCENE_MAX_IMAGE_SIDE",
        "SCENE_MAX_TOKENS",
        "SCENE_JPEG_QUALITY",
        "SCENE_TIMEOUT_SECONDS",
        "LOCAL_TRACKING_FPS",
        "FACE_SAMPLE_FPS",
        "LOCAL_CAMERA_TARGET_FPS",
    }
)


def validate_ports(streams: int, local_port: int, sam_port: int) -> None:
    if streams < 1 or not 1 <= local_port <= 65536 - streams:
        raise ValueError("Hybrid console port range must fit within 1..65535")
    if REMOTE_PORT + streams > 65536:
        raise ValueError("Too many hybrid CPU lanes for the private port range")
    if local_port <= sam_port < local_port + streams:
        raise ValueError("Hybrid console ports overlap the SAM tunnel port")


def inference_environment(source) -> dict[str, str]:
    return {key: str(source[key]) for key in ENVIRONMENT_KEYS if source.get(key)}


def build_bundle(root: Path, destination: Path) -> None:
    files = [
        root / name
        for name in ("pyproject.toml", "config.yaml", "tools/hybrid_cloud.py")
    ]
    suffixes = {
        ".py",
        ".yaml",
        ".js",
        ".html",
        ".css",
        ".wasm",
        ".txt",
        ".otf",
        ".ttf",
        ".json",
    }
    files += [
        p
        for p in (root / "src/goodvision_harness").rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
        and p.suffix in suffixes
    ]
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(files):
            if path.is_symlink():
                raise ValueError("Hybrid source bundle must not follow symbolic links")
            archive.add(
                path, arcname=path.relative_to(root).as_posix(), recursive=False
            )


def deploy(root: Path, ssh: list[str], run, environment) -> None:
    """Use binary stdin for source and a private JSON file for inference secrets."""
    run([*ssh, f"mkdir -p {REMOTE}/source {REMOTE}/data"])
    with tempfile.TemporaryDirectory(prefix="goodvision-hybrid-") as directory:
        bundle = Path(directory) / "source.tar.gz"
        build_bundle(root, bundle)
        with bundle.open("rb") as source:
            subprocess.run(
                [*ssh, f"tar -xzf - -C {REMOTE}/source"],
                stdin=source,
                check=True,
                timeout=600,
            )
    install = (
        f"{REMOTE}/env/bin/python -m pip install"
        " --disable-pip-version-check --no-cache-dir "
        f"'{REMOTE}/source[video,vlm,face]'"
    )
    setup = f"""set -euo pipefail
if [ ! -x {REMOTE}/env/bin/python ]; then
  /workspace/goodvision-sam/env/bin/python -m venv {REMOTE}/env
fi
{install}
"""
    run([*ssh, "bash -s"], input=setup, timeout=1800)
    # The token is absent from argv/log output and is deleted before serving.
    writer = (
        "import os,sys; "
        f"p={REMOTE + '/runtime.json'!r}; "
        "fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600); "
        "os.fchmod(fd,0o600); "
        "f=os.fdopen(fd,'wb'); f.write(sys.stdin.buffer.read()); f.close()"
    )
    run(
        [*ssh, f"{REMOTE}/env/bin/python -c {shlex.quote(writer)}"],
        input=json.dumps(inference_environment(environment)),
    )


def launch_command(sam_command: str, streams: int) -> str:
    return (
        f"exec {REMOTE}/env/bin/python {REMOTE}/source/tools/hybrid_cloud.py "
        f"serve --streams {streams} --sam-command {shlex.quote(sam_command)}"
    )


def forwards(streams: int, local_port: int) -> list[str]:
    return [
        part
        for i in range(streams)
        for part in ("-L", f"127.0.0.1:{local_port + i}:127.0.0.1:{REMOTE_PORT + i}")
    ]


def lane_environment(base, settings, root: Path, data: Path, lane: int, streams: int):
    env = {**base, **inference_environment(settings)}
    env.update(
        {
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONUNBUFFERED": "1",
            "HARNESS_CONFIG": str(root / "config.yaml"),
            "HARNESS_DEPLOYMENT_SETTINGS_PATH": str(data / "settings.json"),
            "EVENT_EVIDENCE_DIRECTORY": str(data / "events"),
            "HARNESS_AUTOSTART": "false",
            "HARNESS_DRY_RUN": "false",
            # Each console owns one independent camera; SAM routing retains lane ID.
            "HARNESS_FLEET_CAPACITY": "1",
            "HARNESS_FLEET_CAMERA_INDEX": "1",
            "HARNESS_SOURCE_ID": f"camera-{lane:02d}",
            "TRACKING_EXECUTION_MODE": "legacy",
            "TRACKING_SERVER_PROTOCOL": "legacy",
            "TRACKING_SERVER_URL": "http://127.0.0.1:8092",
            "TRACKING_SERVER_CAPACITY": str(streams),
            "LOCAL_GPU_AVAILABLE": "false",
            "LOCAL_INFERENCE_ENABLED": "false",
            "LOCAL_TRACKING_ENABLED": "false",
            "CUDA_VISIBLE_DEVICES": "",
            "FACE_ACCELERATOR": "cpu",
            "PERSON_TRACKING_ENABLED": "true",
            "RICH_LABEL_ENABLED": "true",
            "RICH_LABEL_BACKEND": "sam31_stream",
            "PERCEPTION_METHOD": "sam_live",
            "RICH_LABEL_ENDPOINT": f"http://127.0.0.1:8092/v1/cameras/camera-{lane:02d}/perception/sam_live/sessions",
            "SCENE_ENGINE_MODE": "active",
            "SAM31_QUEUE_POLICY": "latest_only",
            # Browser JPEG is uploaded once. Do not transcode to H264 on the client;
            # the two cloud processes exchange JPEGs over loopback.
            "SAM31_TRANSPORT_CODEC": "jpeg",
            "SAM31_TRANSPORT_MAX_SIDE": "512",
            "LOCAL_CAMERA_DISPLAY_ENABLED": "true",
            "LOCAL_CAMERA_DISPLAY_DELAY_SECONDS": "2",
            "LOCAL_TRACKING_FPS": settings.get("LOCAL_TRACKING_FPS", "1.1"),
            "LOCAL_CAMERA_TARGET_FPS": settings.get("LOCAL_CAMERA_TARGET_FPS", "15"),
        }
    )
    return env


def wait_ready(process, streams: int, local_port: int, timeout=120):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Hybrid SSH supervisor exited before CPU lanes became ready"
            )
        try:
            for i in range(streams):
                with opener.open(
                    f"http://127.0.0.1:{local_port + i}/v1/ui/info", timeout=5
                ) as response:
                    info = json.load(response)
                if not info.get("local_camera_display", {}).get("enabled"):
                    raise ValueError("Browser contour display is not enabled")
            return
        except (OSError, ValueError):
            time.sleep(1)
    raise RuntimeError("Cloud CPU analysis consoles did not become ready")


def serve(streams: int, sam_command: str):
    validate_ports(streams, REMOTE_PORT, 8092)
    root, data_root = Path(REMOTE) / "source", Path(REMOTE) / "data"
    settings_file = Path(REMOTE) / "runtime.json"
    settings = inference_environment(
        json.loads(settings_file.read_text(encoding="utf-8"))
    )
    settings_file.unlink()
    processes, logs = [], []

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    for item in signals:
        signal.signal(item, stop)
    try:
        processes.append(
            subprocess.Popen(["bash", "-c", sam_command], start_new_session=True)
        )
        for lane in range(1, streams + 1):
            data = data_root / f"camera-{lane:02d}"
            data.mkdir(parents=True, exist_ok=True)
            log = (data / "service.log").open("ab", buffering=0)
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "goodvision_harness.service:create_app",
                        "--factory",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(REMOTE_PORT + lane - 1),
                        "--no-access-log",
                    ],
                    cwd=data,
                    env=lane_environment(
                        os.environ, settings, root, data, lane, streams
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        print(
            f"Hybrid supervisor started: {streams} CPU lanes; SAM on GPU; "
            "browser presentation stays local.",
            flush=True,
        )
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A hybrid child exited; stopping all lanes and SAM")
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        for item in signals:
            signal.signal(item, signal.SIG_IGN)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve"])
    parser.add_argument("--streams", type=int, required=True)
    parser.add_argument("--sam-command", required=True)
    args = parser.parse_args()
    raise SystemExit(serve(args.streams, args.sam_command))
