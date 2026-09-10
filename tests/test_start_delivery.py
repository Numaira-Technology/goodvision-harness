from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "delivery_launcher", ROOT / "tools/start_delivery.py"
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


@pytest.fixture
def delivery(tmp_path: Path) -> Path:
    for source, target in (
        ("compose.full-gpu.delivery.yaml", "compose.yaml"),
        ("compose.full-gpu.accelerated.yaml", "compose.gpu.yaml"),
        ("config.delivery.yaml", "config.delivery.yaml"),
    ):
        (tmp_path / target).write_bytes((ROOT / source).read_bytes())
    return tmp_path


def completed(
    code: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout, stderr)


def gpu_response(
    available: bool, reason: str = "no_cuda_device"
) -> subprocess.CompletedProcess:
    return completed(
        0 if available else 3,
        launcher.PROBE_MARKER + json.dumps({"available": available, "reason": reason}),
    )


def fake_docker(
    monkeypatch: pytest.MonkeyPatch, responses: list[subprocess.CompletedProcess]
) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(list(command))
        assert kwargs.get("cwd")
        assert responses, f"Unexpected Docker call: {command}"
        return responses.pop(0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("available", [True, False])
def test_auto_probes_latest_image_and_selects_only_corresponding_services(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
) -> None:
    calls = fake_docker(
        monkeypatch, [completed(), completed(), completed(), gpu_response(available)]
    )
    plan = launcher.make_plan(delivery)

    assert plan.mode == ("gpu" if available else "cpu")
    assert calls[0][1] == "info"
    assert calls[2][1:5] == [
        "image",
        "inspect",
        launcher.MAIN_IMAGE,
        launcher.RICH_IMAGE,
    ]
    assert calls[3][1:5] == ["run", "--rm", "--pull=never", "--gpus"]
    assert launcher.MAIN_IMAGE in calls[3]
    assert "--entrypoint" in calls[3] and "python" in calls[3]
    if available:
        assert len(plan.commands) == 1
        assert str(delivery / "compose.gpu.yaml") in plan.commands[0]
        assert plan.commands[0][-4:] == ("up", "-d", "harness", "rich-label-worker")
    else:
        assert plan.commands[0][-2:] == ("stop", "rich-label-worker")
        assert plan.commands[1][-3:] == ("up", "-d", "harness")
        assert all(
            str(delivery / "compose.gpu.yaml") not in command
            for command in plan.commands
        )
        assert all(
            "down" not in command and "-v" not in command for command in plan.commands
        )


def test_cpu_mode_checks_both_images_without_probing_gpu(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_docker(monkeypatch, [completed(), completed(), completed()])
    assert launcher.make_plan(delivery, mode="cpu").mode == "cpu"
    assert len(calls) == 3 and all("run" not in call for call in calls)


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_explicit_dry_run_prints_commands_without_starting_or_probing(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    calls = fake_docker(monkeypatch, [])
    assert (
        launcher.main(["--directory", str(delivery), "--mode", mode, "--dry-run"]) == 0
    )
    output = capsys.readouterr().out
    assert f"Selected mode: {mode}" in output and "docker compose" in output
    assert not calls


def test_auto_dry_run_probes_but_does_not_start(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_docker(
        monkeypatch, [completed(), completed(), completed(), gpu_response(True)]
    )
    assert launcher.main(["--directory", str(delivery), "--dry-run"]) == 0
    assert len(calls) == 4 and all("up" not in command for command in calls)


def test_daemon_failure_is_never_treated_as_no_gpu(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_docker(
        monkeypatch, [completed(1, stderr="Cannot connect to Docker daemon")]
    )
    with pytest.raises(launcher.DeliveryError, match="daemon is unavailable"):
        launcher.make_plan(delivery)
    assert len(calls) == 1


def test_missing_current_images_never_selects_older_images(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_docker(
        monkeypatch, [completed(), completed(), completed(1), completed()]
    )
    with pytest.raises(launcher.DeliveryError, match="Both current delivery images"):
        launcher.make_plan(delivery)
    assert all("pull" not in call and "run" not in call for call in calls)


@pytest.mark.parametrize(
    "error",
    [
        "permission denied",
        "nvidia-container-cli: initialization error: driver/library version mismatch",
        'could not select device driver "" with capabilities: [[gpu]]',
    ],
)
def test_unknown_or_broken_gpu_runtime_is_a_failure(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    monkeypatch.setattr(launcher, "docker_uses_local_host", lambda _: False)
    fake_docker(
        monkeypatch,
        [
            completed(),
            completed(),
            completed(),
            completed(125, stderr=error),
            completed(),
        ],
    )
    with pytest.raises(launcher.DeliveryError, match="CPU mode was not selected"):
        launcher.make_plan(delivery)


@pytest.mark.parametrize("hardware", [False, True, None])
def test_missing_gpu_runtime_requires_confirmed_absent_local_hardware(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
    hardware: bool | None,
) -> None:
    fake_docker(
        monkeypatch,
        [
            completed(),
            completed(),
            completed(),
            completed(
                125,
                stderr='could not select device driver "" with capabilities: [[gpu]]',
            ),
            completed(),
        ],
    )
    monkeypatch.setattr(launcher, "docker_uses_local_host", lambda _: True)
    monkeypatch.setattr(launcher, "host_has_nvidia_gpu", lambda: hardware)
    if hardware is False:
        assert launcher.make_plan(delivery).mode == "cpu"
    else:
        with pytest.raises(launcher.DeliveryError, match="CPU mode was not selected"):
            launcher.make_plan(delivery)


@pytest.mark.parametrize(
    "vendor,expected", [("0x8086", False), ("0x10de", True), (None, None)]
)
def test_linux_hardware_inventory_distinguishes_absence_from_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vendor: str | None,
    expected: bool | None,
) -> None:
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher, "PCI_DEVICE_DIRECTORY", tmp_path)
    if vendor:
        device = tmp_path / "pci-device-0"
        device.mkdir()
        (device / "vendor").write_text(vendor)
    assert launcher.host_has_nvidia_gpu() is expected


@pytest.mark.parametrize(
    "payload,expected",
    [("true", True), ("false", False), ('"false"', None), ("invalid", None)],
)
def test_windows_hardware_inventory_requires_a_boolean(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    expected: bool | None,
) -> None:
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "powershell")
    calls = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        return completed(stdout=payload)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.host_has_nvidia_gpu() is expected
    assert "Get-CimInstance Win32_VideoController" in calls[0][-1]


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("unix:///var/run/docker.sock", True),
        ("npipe:////./pipe/docker_engine", True),
        ("ssh://remote-server", False),
        ("tcp://remote-server:2376", False),
    ],
)
def test_hardware_inventory_is_never_used_to_infer_a_remote_daemon_gpu(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    expected: bool,
) -> None:
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", endpoint)
    assert launcher.docker_uses_local_host(delivery) is expected


def test_named_remote_context_takes_precedence_over_local_host_environment(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    calls = fake_docker(monkeypatch, [completed(stdout='"ssh://remote-server"')])
    assert launcher.docker_uses_local_host(delivery) is False
    assert calls[0][1:4] == ["context", "inspect", "remote"]


@pytest.mark.parametrize(
    "detail,reason",
    [
        (
            "libcuda.so.1: cannot open shared object file: No such file or directory",
            "no_cuda_driver",
        ),
        ("libcuda.so.1: invalid ELF header", "cuda_driver_load_failed"),
        (
            "libdependency.so.1: cannot open shared object file: "
            "No such file or directory",
            "cuda_driver_load_failed",
        ),
    ],
)
def test_actual_cuda_probe_only_treats_missing_target_library_as_absent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    detail: str,
    reason: str,
) -> None:
    import ctypes

    def load(_name: str) -> None:
        raise OSError(detail)

    monkeypatch.setattr(ctypes, "CDLL", load)
    with pytest.raises(SystemExit) as raised:
        exec(launcher.CUDA_PROBE, {})
    assert raised.value.code == 3
    output = capsys.readouterr().out
    assert json.loads(output.removeprefix(launcher.PROBE_MARKER))["reason"] == reason


def test_daemon_lost_during_gpu_probe_does_not_start_cpu(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_docker(
        monkeypatch,
        [completed(), completed(), completed(), completed(125), completed(1)],
    )
    with pytest.raises(launcher.DeliveryError, match="daemon is unavailable"):
        launcher.make_plan(delivery)


@pytest.mark.parametrize(
    "reason", ["cuda_initialization_failed", "cuda_device_query_failed"]
)
def test_present_but_broken_cuda_does_not_select_cpu(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    fake_docker(
        monkeypatch,
        [completed(), completed(), completed(), gpu_response(False, reason)],
    )
    with pytest.raises(launcher.DeliveryError, match="CUDA is present"):
        launcher.make_plan(delivery)


def test_explicit_gpu_mode_cannot_silently_become_cpu(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_docker(
        monkeypatch, [completed(), completed(), completed(), gpu_response(False)]
    )
    with pytest.raises(launcher.DeliveryError, match="GPU mode was requested"):
        launcher.make_plan(delivery, mode="gpu")


def test_compose_failure_stops_without_alternative_mode(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_docker(
        monkeypatch, [completed(), completed(), completed(), completed(1)]
    )
    assert launcher.main(["--directory", str(delivery), "--mode", "cpu"]) == 1
    assert calls[-1][-2:] == ["stop", "rich-label-worker"]
    assert len(calls) == 4


def test_root_and_tools_launcher_locations_resolve_same_delivery_directory(
    delivery: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for script in (
        delivery / "start_delivery.py",
        delivery / "tools/start_delivery.py",
    ):
        monkeypatch.setattr(launcher, "__file__", str(script))
        assert launcher.default_directory() == delivery


def test_custom_compose_paths_are_used_exactly(
    delivery: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (delivery / "compose.yaml").rename(delivery / "custom.yaml")
    (delivery / "compose.gpu.yaml").rename(delivery / "acceleration.yaml")
    plan = launcher.make_plan(
        delivery,
        mode="gpu",
        dry_run=True,
        compose_file="custom.yaml",
        gpu_compose_file="acceleration.yaml",
    )
    assert str(delivery / "custom.yaml") in plan.commands[0]
    assert str(delivery / "acceleration.yaml") in plan.commands[0]


def test_latest_complete_images_remain_bundled_but_cpu_has_no_gpu_dependency() -> None:
    base = yaml.safe_load((ROOT / "compose.full-gpu.delivery.yaml").read_text())
    overlay = yaml.safe_load((ROOT / "compose.full-gpu.accelerated.yaml").read_text())
    harness, rich = base["services"]["harness"], base["services"]["rich-label-worker"]
    assert harness["image"] == launcher.MAIN_IMAGE
    assert rich["image"] == launcher.RICH_IMAGE
    assert harness["environment"]["FACE_ACCELERATOR"] == "cpu"
    assert harness["environment"]["RICH_LABEL_ENABLED"] == "false"
    assert harness["environment"]["HARNESS_AUTOSTART"] == "false"
    assert "env_file" not in harness
    assert "depends_on" not in harness and "deploy" not in harness
    assert rich["profiles"] == ["gpu"]
    assert "./config.delivery.yaml:/app/config.yaml:ro" in harness["volumes"]
    for service in overlay["services"].values():
        assert service["deploy"]["resources"]["reservations"]["devices"][0][
            "capabilities"
        ] == ["gpu"]
    assert overlay["services"]["harness"]["environment"] == {
        "FACE_ACCELERATOR": "gpu",
        "RICH_LABEL_ENABLED": "true",
    }
