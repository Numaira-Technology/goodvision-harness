"""GPU direct pull retains camera transport settings through the real deployment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from compose_support import isolated_docker_config

from tools.cloud_analysis import runtime_environment
from tools.gpu_analysis_host import lane_environment
from tools.relay_bootstrap import controller_environment

ROOT = Path(__file__).resolve().parents[1]
KEY = "rtsp-test-auth-key-0123456789abcdef"
DEFAULT = "rtsp_transport;tcp|timeout;5000000"
OVERRIDE = "rtsp_transport;tcp|timeout;100000"
ALIAS = "rtsp_transport;tcp|timeout;200000"
CASES = [
    ({}, DEFAULT),
    ({"OPENCV_FFMPEG_CAPTURE_OPTIONS": OVERRIDE}, OVERRIDE),
    (
        {"OPENCV_FFMPEG_CAPTURE_OPTIONS": OVERRIDE, "VIDEO_CAPTURE_OPTIONS": ALIAS},
        ALIAS,
    ),
    ({"OPENCV_FFMPEG_CAPTURE_OPTIONS": OVERRIDE, "VIDEO_CAPTURE_OPTIONS": ""}, ""),
]


def rendered_controller(tmp_path, recipe, options):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is required for actual deployment rendering")
    directory = tmp_path / recipe
    directory.mkdir()
    ssh = directory / "fixture-ssh"
    ssh.write_text("fixture key, never used", "utf-8")
    values = {
        "VAST_API_KEY": "fixture-rental-key",
        "HF_TOKEN": "fixture-model-token",
        "SAM_SSH_KEY": ssh.as_posix(),
        "HARNESS_RELAY_KEY": KEY,
        **options,
    }
    filename = directory / "settings.env"
    filename.write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        "utf-8",
    )
    config = directory / "docker-config"
    config.mkdir()
    environment = dict(os.environ)
    for name in (*values, "OPENCV_FFMPEG_CAPTURE_OPTIONS", "VIDEO_CAPTURE_OPTIONS"):
        environment.pop(name, None)
    environment["DOCKER_CONFIG"] = isolated_docker_config(config)
    files = (
        ["compose.cloud-gpu.yaml"]
        if recipe == "cloud"
        else [
            "compose.yaml",
            "compose.sam31-remote.yaml",
        ]
    )
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(filename),
    ]
    for path in files:
        command += ["-f", str(ROOT / path)]
    result = subprocess.run(
        [*command, "config", "--format", "json"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["services"]["sam-remote"]["environment"]


def native_rtsp_probe(tmp_path, gpu_environment, expected):
    environment = dict(os.environ)
    environment.pop("VIDEO_CAPTURE_OPTIONS", None)
    environment.update(gpu_environment)
    environment.update(
        PYTHONPATH=str(ROOT / "src"),
        OPENCV_LOG_LEVEL="DEBUG",
        HARNESS_STREAM_DECODER="opencv",
    )
    # All source/decoder settings above come from the actual GPU lane mapping.
    # Reject the first RTSP request locally; no external camera or models exist.
    program = r"""import socket
import threading
import cv2
from goodvision_harness.stream_adapter import NetworkStreamAdapter, StreamAdapterError

native_capture = cv2.VideoCapture
cv2.VideoCapture = lambda source: native_capture(source, cv2.CAP_FFMPEG)
requests = []
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(2)
    def reject():
        with listener.accept()[0] as connection:
            connection.settimeout(2)
            requests.append(connection.recv(8192))
            connection.sendall(
                b'RTSP/1.0 400 Bad Request\r\nCSeq: 1\r\n'
                b'Content-Length: 0\r\n\r\n'
            )
    peer = threading.Thread(target=reject, daemon=True)
    peer.start()
    adapter = NetworkStreamAdapter(
        f'rtsp://127.0.0.1:{listener.getsockname()[1]}/fixture', reconnect_attempts=0,
    )
    try:
        next(adapter.frames())
    except StreamAdapterError:
        pass
    else:
        raise AssertionError('fixture must reject RTSP')
    finally:
        adapter.stop()
        peer.join(3)
    assert not peer.is_alive()
    assert requests and b'RTSP/1.0' in requests[0]
"""
    result = subprocess.run(
        [sys.executable, "-u", "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "using capture options from environment: " + expected in (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize("recipe", ["cloud", "plain"])
@pytest.mark.parametrize(
    "options,expected", CASES, ids=["default", "override", "alias", "empty-alias"]
)
def test_compose_controller_upload_and_gpu_decoder_preserve_rtsp_policy(
    tmp_path,
    recipe,
    options,
    expected,
):
    rendered = rendered_controller(tmp_path, recipe, options)
    assert rendered.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == expected
    private = controller_environment(rendered, tmp_path / "controller")
    uploaded = runtime_environment(private)
    assert uploaded["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == expected
    assert "VIDEO_CAPTURE_OPTIONS" not in uploaded
    assert not {"VAST_API_KEY", "HF_TOKEN", "SAM_SSH_KEY"} & uploaded.keys()
    gpu = lane_environment(uploaded, tmp_path / "gpu", 1, 1)
    assert gpu["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == expected
    if expected:
        native_rtsp_probe(tmp_path, gpu, expected)


@pytest.mark.parametrize(
    "options,expected", CASES, ids=["default", "override", "alias", "empty-alias"]
)
def test_native_full_analysis_uses_same_rtsp_policy_without_compose(
    tmp_path,
    options,
    expected,
):
    private = controller_environment(
        {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY": KEY, **options},
        tmp_path / "controller",
    )
    uploaded = runtime_environment(private)
    assert uploaded.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == expected
    assert "VIDEO_CAPTURE_OPTIONS" not in uploaded
    gpu = lane_environment(uploaded, tmp_path / "gpu", 1, 1)
    assert gpu["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == expected
    if expected:
        native_rtsp_probe(tmp_path, gpu, expected)
