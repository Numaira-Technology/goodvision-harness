"""Gateway deployment retains RTSP capture policy for local video pulling."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from compose_support import isolated_docker_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "rtsp_transport;tcp|timeout;5000000"
OVERRIDE = "rtsp_transport;tcp|timeout;100000"


def render_gateway(tmp_path, filename, options):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is required for actual deployment rendering")
    directory = tmp_path / Path(filename).stem
    directory.mkdir(parents=True, exist_ok=True)
    key = directory / "test-ssh-key"
    key.write_text("fixture key, never used to connect", encoding="utf-8")
    env_file = directory / "settings.env"
    values = {
        "VAST_API_KEY": "fixture-rental-key",
        "HF_TOKEN": "fixture-model-token",
        "SAM_SSH_KEY": key.as_posix(),
        "HARNESS_ANALYSIS_URL": "http://fixture-analysis:8080",
    }
    if options is not None:
        values["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    env_file.write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )
    docker_config = directory / "docker-config"
    docker_config.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    environment["DOCKER_CONFIG"] = isolated_docker_config(docker_config)
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "--env-file",
            str(env_file),
            "-f",
            str(ROOT / filename),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["services"]["harness"]["environment"]


@pytest.mark.parametrize(
    "filename",
    [
        "compose.analysis-gateway.yaml",
        "compose.cloud-gpu.yaml",
    ],
)
@pytest.mark.parametrize("options", [None, OVERRIDE])
def test_standalone_and_cloud_gateway_keep_default_and_operator_rtsp_options(
    tmp_path,
    filename,
    options,
):
    environment = render_gateway(tmp_path, filename, options)
    assert environment["HARNESS_VIDEO_PULL_LOCATION"] == "local"
    assert environment["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == (options or DEFAULT)


@pytest.mark.parametrize("options", [None, OVERRIDE])
def test_rendered_options_are_read_by_the_native_rtsp_decoder(tmp_path, options):
    environment = render_gateway(tmp_path, "compose.analysis-gateway.yaml", options)
    selected = environment["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    child_environment = dict(os.environ)
    child_environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "OPENCV_FFMPEG_CAPTURE_OPTIONS": selected,
            "OPENCV_LOG_LEVEL": "DEBUG",
            "HARNESS_STREAM_DECODER": "opencv",
        }
    )
    # The loopback probe deliberately has no video service. It exercises the
    # actual OpenCV/FFmpeg open through NetworkStreamAdapter, without cameras,
    # model calls or depending on an available RTSP server.
    program = """from goodvision_harness.stream_adapter import (
    NetworkStreamAdapter, StreamAdapterError,
)
adapter = NetworkStreamAdapter('rtsp://127.0.0.1:1/fixture', reconnect_attempts=0)
try:
    next(adapter.frames())
except StreamAdapterError:
    pass
finally:
    adapter.stop()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=child_environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    logs = result.stdout + result.stderr
    assert "using capture options from environment: " + selected in logs
