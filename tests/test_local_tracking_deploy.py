"""Exercise the SAM-only launch path without NVIDIA, weights or container writes."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from compose_support import isolated_docker_config

from tools import start_local_gpu as launcher

ROOT = Path(__file__).resolve().parents[1]
DEVICES = [
    {"index": index, "name": "NVIDIA test GPU", "memory_mib": 24000}
    for index in (2, 5)
]


def test_tracking_plan_uses_first_detected_gpu_and_ignores_vlm_settings(tmp_path):
    plan = launcher.plan_tracking_environment(DEVICES, {
        "LOCAL_INFERENCE_ENABLED": "true",
        "LOCAL_INFERENCE_MODEL_DIRECTORY": "/missing/vlm",
        "LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION": "invalid",
        "INFERENCE_API_KEY": "secret",
    }, root=tmp_path)
    assert plan["LOCAL_TRACKING_GPU_ID"] == "2"
    assert plan["LOCAL_INFERENCE_AVAILABLE"] == "false"
    assert plan["LOCAL_TRACKING_AVAILABLE"] == "true"
    assert plan["INFERENCE_EXECUTION_MODE"] == "cloud"
    assert plan["TRACKING_EXECUTION_MODE"] == "local"
    assert "secret" not in json.dumps(plan)
    assert "LOCAL_INFERENCE_MODEL_DIRECTORY" not in plan
    plan = launcher.plan_tracking_environment(DEVICES, {
        "LOCAL_TRACKING_GPU_ID": "5", "INFERENCE_EXECUTION_MODE": "server",
        "INFERENCE_SERVER_URL": "http://vlm:8000",
    }, root=tmp_path)
    assert plan["LOCAL_TRACKING_GPU_ID"] == "5"
    assert plan["INFERENCE_SERVER_URL"] == "http://vlm:8000"


@pytest.mark.parametrize("settings", [
    {"LOCAL_TRACKING_GPU_ID": "0"}, {"LOCAL_TRACKING_GPU_ID": "-1"},
    {"LOCAL_TRACKING_GPU_ID": "bad"}, {"LOCAL_TRACKING_FPS": "nan"},
    {"LOCAL_TRACKING_FPS": "0"}, {"LOCAL_TRACKING_TORCH_CUDA": "cu121"},
    {"INFERENCE_EXECUTION_MODE": "local"}, {"INFERENCE_EXECUTION_MODE": "server"},
    {"INFERENCE_SERVER_URL": "https://user:private@vlm.example"},
    {"INFERENCE_SERVER_URL": "https://vlm.example?token=private"},
])
def test_invalid_tracking_setup_is_rejected(settings, tmp_path):
    with pytest.raises(launcher.LaunchError):
        launcher.plan_tracking_environment(DEVICES, settings, root=tmp_path)


def test_reuses_contour_or_legacy_checkpoint_and_honors_explicit_path(tmp_path):
    legacy = tmp_path / ".models/sam3.1/sam3.1_multiplex.pt"
    contour = tmp_path / "workers/contour/models/contour.pt"
    for checkpoint in (legacy, contour):
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint fixture")
        plan = launcher.plan_tracking_environment(DEVICES, {}, root=tmp_path)
        assert Path(plan["SAM31_CHECKPOINT_FILE"]) == checkpoint
        launcher.validate_tracking_checkpoint(plan)
    plan = launcher.plan_tracking_environment(DEVICES, {
        "SAM31_CHECKPOINT_FILE": str(legacy.relative_to(tmp_path)),
    }, root=tmp_path)
    assert Path(plan["SAM31_CHECKPOINT_FILE"]) == legacy
    for content in (b"", b"version https://git-lfs.github.com/spec/v1\n"):
        legacy.write_bytes(content)
        with pytest.raises(launcher.LaunchError):
            launcher.validate_tracking_checkpoint(plan)
    legacy.unlink()
    with pytest.raises(launcher.LaunchError, match="Missing SAM checkpoint"):
        launcher.validate_tracking_checkpoint(plan)


@pytest.fixture
def launch_rig(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "detect_nvidia_devices", lambda: DEVICES)
    for key in launcher.TRACKING_SETTINGS:
        monkeypatch.delenv(key, raising=False)
    checkpoint = tmp_path / "sam.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    return SimpleNamespace(calls=calls, checkpoint=checkpoint, root=tmp_path)


def test_default_launch_checks_cuda_before_start_and_forwards_cloud_key(
    launch_rig, monkeypatch,
):
    monkeypatch.setenv("INFERENCE_API_KEY", "cloud-key")
    assert launcher.main(["--checkpoint", str(launch_rig.checkpoint)]) == 0
    build, preflight, storage, start = launch_rig.calls
    assert build[0][-3:] == ["build", "harness", "local-tracking"]
    assert preflight[0][-4:] == [
        "local-tracking", "python", "-c", launcher.runtime_check("local-tracking"),
    ]
    assert storage[0][-4:] == [
        "harness", "python", "-c", launcher.runtime_check("harness"),
    ]
    assert preflight[1]["timeout"] == 120
    assert start[0][-2:] == ["harness", "local-tracking"]
    assert "--wait" in start[0]
    assert "--no-build" in start[0]
    assert start[0][start[0].index("--pull") + 1] == "never"
    assert "local-inference" not in start[0]
    assert start[1]["env"]["INFERENCE_API_KEY"] == "cloud-key"
    assert start[1]["env"]["LOCAL_TRACKING_GPU_ID"] == "2"


@pytest.mark.parametrize("failure", ["cuda", "timeout", "build", "storage"])
def test_failed_preparation_does_not_start_application(
    launch_rig, monkeypatch, failure,
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if failure == "storage" and len(calls) == 1:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    args = [] if failure == "build" else ["--no-build"]
    assert launcher.main([
        *args, "--checkpoint", str(launch_rig.checkpoint),
    ]) == 2
    assert len(calls) == (2 if failure == "storage" else 1)
    assert not any("up" in command for command in calls)


def test_dry_run_is_credential_free_and_has_no_writes(launch_rig, capsys, monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "never-print")
    assert launcher.main([
        "--dry-run", "--gpu-count", "2", "--streams", "3",
    ]) == 0
    output = capsys.readouterr().out
    plan = json.loads(output)
    assert plan["mode"] == "tracking"
    assert plan["environment"]["LOCAL_TRACKING_GPU_ID"] == "0"
    assert "never-print" not in output
    assert not launch_rig.calls
    assert not (launch_rig.root / ".goodvision").exists()
    assert "compose.local-tracking.yaml" in plan["fleet_overlay"]
    assert [cmd[-4] for cmd in plan["preflight_commands"]] == [
        "local-tracking", "harness", "harness-02", "harness-03",
    ]


@pytest.mark.parametrize("service", [
    "harness", "local-tracking", "local-vlm-processing",
])
def test_preflight_rejects_old_images_and_checks_writable_runtime_paths(
    tmp_path, monkeypatch, service,
):
    source = ROOT / ("workers/sam_scene" if service == "local-tracking"
                     else "src/goodvision_harness")
    image = tmp_path / "image"
    image.mkdir()
    for name in launcher.RUNTIME_FILES[service]:
        # Match Linux-built images when the source checkout uses Windows CRLF.
        target = image / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes().replace(b"\r\n", b"\n"))
    monkeypatch.setattr(launcher, "CUDA_CHECK", "")
    script = launcher.runtime_check(service)
    if service == "local-tracking":
        # Substitute only the container root; execute the complete emitted probe.
        script = script.replace("Path('/engine')", f"Path({str(image)!r})")
        for key in (
            "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR",
            "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH",
        ):
            monkeypatch.setenv(key, str(tmp_path / "cache" / key))
        expected = tmp_path / "cache" / "TRITON_CACHE_DIR"
    else:
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: SimpleNamespace(
            origin=str(image / "__init__.py"),
        ))
        monkeypatch.setenv(
            "HARNESS_DEPLOYMENT_SETTINGS_PATH", str(tmp_path / "data/settings.json"),
        )
        monkeypatch.setenv(
            "GOODVISION_SAM_SPOOL_DIRECTORY", str(tmp_path / "data/sam-spool"),
        )
        monkeypatch.delenv("GOODVISION_TRACKING_SPOOL_DIRECTORY", raising=False)
        expected = (None if service == "local-vlm-processing"
                    else tmp_path / "data/sam-spool")
    exec(script, {})
    if expected is not None:
        assert expected.is_dir()
        assert list(expected.iterdir()) == []  # The write probe cleans up after itself.
    else:
        assert not (tmp_path / "data").exists()
    # Delegating shims stay identical while an actual scene implementation is stale.
    target = image / (launcher.RUNTIME_FILES[service][0]
                      if service == "local-tracking" else "runtime/scene.py")
    original = target.read_bytes()
    target.write_bytes(b"# old image with the same tag\n")
    with pytest.raises(AssertionError, match="Outdated"):
        exec(script, {})
    target.write_bytes(original)
    if expected is not None:
        expected.rmdir()
        expected.write_text("a file cannot be used as a writable cache directory")
        with pytest.raises(OSError):
            exec(script, {})


def test_full_local_model_deployment_remains_explicit(launch_rig, capsys):
    assert launcher.main([
        "--dry-run", "--gpu-count", "2", "--with-local-inference",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "full"
    assert "local-inference" in plan["command"]
    assert plan["environment"]["LOCAL_TRACKING_GPU_ID"] == "1"
    assert plan["environment"]["LOCAL_INFERENCE_ENABLED"] == "true"
    assert [cmd[-4] for cmd in plan["preflight_commands"]] == [
        "local-tracking", "local-vlm-processing", "harness",
    ]
    assert not launch_rig.calls


@pytest.mark.skipif(not shutil.which("docker"), reason="Docker Compose unavailable")
@pytest.mark.parametrize("streams", [1, 3])
def test_compose_reuses_contour_engine_and_keeps_workers_private(tmp_path, streams):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    environment = dict(os.environ)
    environment.update(launcher.plan_tracking_environment(DEVICES, {}))
    environment["HARNESS_FLEET_CAPACITY"] = str(streams)
    environment["DOCKER_CONFIG"] = isolated_docker_config(tmp_path)
    fleet_file = None
    if streams > 1:
        fleet_file = tmp_path / "fleet.yaml"
        fleet_file.write_text(launcher.fleet_overlay(
            streams, with_local_inference=False,
        ), encoding="utf-8")
    command = launcher.compose_command(
        root=ROOT, env_file=env_file, with_paddle=False, build=False,
        streams=streams, fleet_file=fleet_file, with_local_inference=False,
    )
    result = subprocess.run(
        [*command[:command.index("up")], "config", "--format", "json"],
        env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    assert len(services) == streams + 1
    worker = services["local-tracking"]
    assert worker["image"] == "goodvision-contour-runtime:0.1.0"
    assert worker["build"]["dockerfile"] == "contour/Dockerfile.engine"
    assert worker["build"]["context"].replace("\\", "/").endswith("/workers")
    assert worker["environment"]["SAM31_FIXED_PERSON_TEXT"] == "true"
    assert worker["environment"]["SAM31_LIVE_CAMERAS"] == str(streams)
    assert "gpu_ready" in worker["healthcheck"]["test"][-1]
    assert not worker.get("ports")
    assert worker["read_only"] is True
    assert worker["tmpfs"] == ["/tmp:size=2g"]
    assert worker["stop_grace_period"] == "1m30s"
    assert worker["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "device_ids": ["2"], "capabilities": ["gpu"]},
    ]
    assert len(worker["volumes"]) == 1
    assert worker["volumes"][0]["target"] == "/models/contour.pt"
    for index in range(1, streams + 1):
        name = "harness" if index == 1 else f"harness-{index:02d}"
        service = services[name]
        assert len(service.get("ports", [])) == int(index == 1)
        assert "devices" not in service.get("deploy", {}).get("resources", {}).get(
            "reservations", {},
        )
        assert service["depends_on"]["local-tracking"]["condition"] == "service_healthy"
        env = service["environment"]
        mounts = {volume["target"]: volume for volume in service["volumes"]}
        assert service["read_only"] is True
        assert env["GOODVISION_SAM_SPOOL_DIRECTORY"] == "/data/sam-spool"
        assert env["HARNESS_DEPLOYMENT_SETTINGS_PATH"] == (
            "/data/deployment-settings.json"
        )
        assert mounts["/data"]["type"] == "volume"
        assert mounts["/data"].get("read_only", False) is False
        assert mounts["/data"]["source"] == (
            "face-data" if streams == 1 else f"camera-{index:02d}-data"
        )
        assert env["LOCAL_INFERENCE_AVAILABLE"] == "false"
        assert env["TRACKING_EXECUTION_MODE"] == "local"
        if streams > 1:
            assert f"/cameras/camera-{index:02d}/" in env["LOCAL_TRACKING_ENDPOINT"]
