"""Compose-to-runtime queue configuration, including the authenticated Contour route."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from compose_support import isolated_docker_config

from goodvision_harness.config import load_config
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.perception import (
    ContourSessionHTTPClient,
    create_tracking_client,
    create_tracking_coordinator,
    processing_status,
)
from goodvision_harness.service_locations import ServiceLocations
from tools import start_modular as launcher

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    for key in launcher.SETTINGS:
        monkeypatch.delenv(key, raising=False)


def args(*options):
    return launcher.parser().parse_args(["main", *options])


def test_default_plan_does_not_override_queue_or_sampling_defaults():
    env = launcher.main_environment({}, args())
    assert not launcher.QUEUE_SETTINGS.intersection(env)
    assert "PERCEPTION_METHOD" not in env
    assert "LOCAL_TRACKING_FPS" not in env


def test_queue_switch_preserves_custom_window_and_does_not_change_fps():
    settings = {
        "SAM31_QUEUE_POLICY": "durable_ordered",
        "SAM31_REALTIME_PENDING_FRAMES": "8",
        "SAM31_REALTIME_MAX_AGE_SECONDS": "3",
        "SAM31_REALTIME_APPEND_FRAMES": "2",
        "SAM31_RESULT_DEADLINE_SECONDS": "5",
        "LOCAL_TRACKING_FPS": "3",
    }
    env = launcher.main_environment(
        settings,
        args("--sam-queue-policy", "realtime_latest"),
    )
    assert env["SAM31_QUEUE_POLICY"] == "realtime_latest"
    assert env["PERCEPTION_METHOD"] == "sam_live"
    for key in launcher.QUEUE_SETTINGS - {"SAM31_QUEUE_POLICY"}:
        assert env[key] == settings[key]
    # Compose still reads the original FPS from .env/the process environment.
    assert "LOCAL_TRACKING_FPS" not in env
    rollback = launcher.main_environment(
        {**settings, "SAM31_QUEUE_POLICY": "realtime_latest"},
        args("--sam-queue-policy", "durable_ordered", "--tracking-fps", "1"),
    )
    assert rollback["SAM31_QUEUE_POLICY"] == "durable_ordered"
    assert rollback["LOCAL_TRACKING_FPS"] == "1.0"


@pytest.mark.parametrize(
    "settings",
    [
        {"SAM31_QUEUE_POLICY": "unknown"},
        {"SAM31_QUEUE_POLICY": "realtime_latest", "PERCEPTION_METHOD": "sam3_batch"},
        {"SAM31_REALTIME_APPEND_FRAMES": "3"},
        {"SAM31_REALTIME_PENDING_FRAMES": "0"},
        {"SAM31_REALTIME_PENDING_FRAMES": "9"},
        {"SAM31_REALTIME_PENDING_FRAMES": "9" * 400},
        {"SAM31_REALTIME_PENDING_FRAMES": "2.5"},
        {"SAM31_REALTIME_MAX_AGE_SECONDS": "nan"},
        {"SAM31_REALTIME_MAX_AGE_SECONDS": "inf"},
        {"SAM31_REALTIME_MAX_AGE_SECONDS": "0"},
        {"SAM31_RESULT_DEADLINE_SECONDS": "31"},
        {"SAM31_RESULT_DEADLINE_SECONDS": "no"},
    ],
)
def test_invalid_queue_settings_fail_before_container_start(settings):
    with pytest.raises(launcher.LaunchError):
        launcher.main_environment(settings, args())


@pytest.mark.parametrize("fps", ["nan", "inf", "0", "0.49", "31"])
def test_invalid_cli_fps_rejected(fps):
    with pytest.raises(launcher.LaunchError):
        launcher.main_environment({}, args("--tracking-fps", fps))


def test_dry_run_shows_effective_contour_plan_without_keys_or_writes(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    settings = tmp_path / "test.env"
    settings.write_text(
        "TRACKING_SERVER_PROTOCOL=contour\n"
        "SAM31_QUEUE_POLICY=durable_ordered\n"
        "SAM31_REALTIME_PENDING_FRAMES=4\n"
        "INFERENCE_API_KEY=not-for-the-plan\n",
    )
    run = Mock(side_effect=AssertionError("dry-run cannot start a process"))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert (
        launcher.main(
            [
                "main",
                "--streams",
                "12",
                "--tracking-server",
                "http://node:8093",
                "--sam-queue-policy",
                "realtime_latest",
                "--tracking-fps",
                "1",
                "--env-file",
                str(settings),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "not-for-the-plan" not in output
    env = json.loads(output)["environment"]
    assert env["TRACKING_SERVER_PROTOCOL"] == "contour"
    assert env["SAM31_QUEUE_POLICY"] == "realtime_latest"
    assert env["SAM31_REALTIME_PENDING_FRAMES"] == "4"
    assert env["LOCAL_TRACKING_FPS"] == "1.0"
    assert not (tmp_path / ".goodvision").exists()
    run.assert_not_called()


@pytest.mark.parametrize("policy", [None, "durable_ordered", "realtime_latest"])
def test_twelve_rendered_contour_lanes_load_actual_config(
    tmp_path,
    monkeypatch,
    policy,
):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose not installed")
    settings = {
        "TRACKING_SERVER_PROTOCOL": "contour",
        "LOCAL_TRACKING_FPS": "1",
    }
    if policy is not None:
        settings.update(
            SAM31_QUEUE_POLICY=policy,
            SAM31_REALTIME_APPEND_FRAMES="2",
            SAM31_REALTIME_PENDING_FRAMES="4",
            SAM31_REALTIME_MAX_AGE_SECONDS="3",
            SAM31_RESULT_DEADLINE_SECONDS="5",
        )
    env_file = tmp_path / "test.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in settings.items()))
    env = launcher.main_environment(
        settings,
        args(
            "--streams",
            "12",
            "--tracking-server",
            "http://node:8093",
        ),
    )
    overlay = tmp_path / "fleet.yaml"
    overlay.write_text(launcher.application_fleet(12))
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-f",
            str(ROOT / "compose.main-server.yaml"),
            "-f",
            str(overlay),
            "config",
            "--format",
            "json",
        ],
        env={**os.environ, **env, "DOCKER_CONFIG": isolated_docker_config(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    assert len(services) == 12
    for index in range(1, 13):
        name = "harness" if index == 1 else f"harness-{index:02d}"
        rendered = services[name]["environment"]
        for key in launcher.QUEUE_SETTINGS | {"PERCEPTION_METHOD"}:
            assert rendered[key] == env.get(key, "")
            monkeypatch.setenv(key, rendered[key])
        assert rendered["LOCAL_TRACKING_FPS"] == "1"
        assert rendered["TRACKING_SERVER_PROTOCOL"] == "contour"
        base = load_config(ROOT / "config.yaml")
        configured = ServiceLocations(
            tracking_mode="server",
            tracking_server_url="http://node:8093",
            tracking_server_capacity=12,
            tracking_server_protocol="contour",
            tracking_sample_fps=1,
        ).configure(base, LocalGPUDeployment(), camera_index=index)
        rich = configured.rich_labeling
        assert rich.endpoint == (
            f"http://node:8093/v1/cameras/camera-{index:02d}/contour/sessions"
        )
        assert rich.video_queue_policy == (policy or "durable_ordered")
        assert rich.max_fps == 1
        assert rich.processing_method == "sam_live"
        if policy is not None:
            assert rich.video_realtime_pending_frames == 4
            assert rich.video_realtime_append_frames == 2
            assert rich.video_realtime_max_age_seconds == 3
            assert rich.video_result_deadline_seconds == 5
        # This creates the actual Contour client/coordinator, but sends no HTTP.
        monkeypatch.setenv("GOODVISION_SAM_SPOOL_DIRECTORY", str(tmp_path / name))
        client = create_tracking_client(rich, api_key="fixture-node-key")
        assert isinstance(client, ContourSessionHTTPClient)
        coordinator = create_tracking_coordinator(client, rich, None)
        try:
            status = processing_status(coordinator, "sam_live")
            assert status["queue_policy"] == (policy or "durable_ordered")
            if policy == "realtime_latest":
                assert status["pending_capacity"] == 4
                assert status["realtime_max_age_seconds"] == 3
                assert status["effective_max_append_frames"] == 2
                assert status["result_deadline_seconds"] == 5
            else:
                assert status["realtime_max_age_seconds"] is None
            assert status["session_admitted_frames"] == 0
            assert status["admitted_deadline_coverage"] is None
        finally:
            coordinator.close()
            client.close()


def test_no_coordinator_does_not_invent_applied_queue_configuration():
    status = processing_status(None, "sam_live")
    assert status["queue_policy"] is None
    assert status["pending_capacity"] is None
    assert status["realtime_max_age_seconds"] is None
    assert status["admitted_deadline_coverage"] is None
