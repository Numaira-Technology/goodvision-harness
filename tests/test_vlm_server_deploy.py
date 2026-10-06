"""Model server plans select CPU delegation without probing application GPUs."""

import json
from types import SimpleNamespace

import pytest

from tools import start_local_gpu as local
from tools import start_modular as modular


def test_server_protocol_defaults_environment_and_cli_override():
    args = modular.parser().parse_args(
        ["main", "--inference-server", "http://model:8000"]
    )
    assert (
        modular.main_environment({}, args)["INFERENCE_SERVER_PROTOCOL"] == "compatible"
    )
    settings = {"INFERENCE_SERVER_PROTOCOL": "goodvision"}
    assert (
        modular.main_environment(settings, args)["INFERENCE_SERVER_PROTOCOL"]
        == "goodvision"
    )
    args = modular.parser().parse_args(
        [
            "main",
            "--inference-server",
            "http://model:8000",
            "--inference-protocol",
            "compatible",
        ]
    )
    assert (
        modular.main_environment(settings, args)["INFERENCE_SERVER_PROTOCOL"]
        == "compatible"
    )
    with pytest.raises(modular.LaunchError):
        modular.main_environment(
            {"INFERENCE_SERVER_PROTOCOL": "invalid"},
            modular.parser().parse_args(["main"]),
        )


def test_standalone_worker_starts_both_services_and_no_build_is_explicit():
    command = modular.compose_command(
        "vlm",
        env_file=None,
        streams=1,
        fleet_file=None,
        build=False,
    )
    assert "--no-build" in command and "--build" not in command
    assert "--wait" in command
    assert command[-2:] == ["vlm-server", "vlm-processing"]


@pytest.mark.parametrize("preflight_status", [0, 1])
def test_standalone_prebuilt_companion_is_checked_before_start(
    monkeypatch,
    capsys,
    preflight_status,
):
    monkeypatch.setattr(modular.local, "read_local_settings", lambda *_, **__: {})
    monkeypatch.setattr(
        modular.local,
        "detect_nvidia_devices",
        lambda: [
            {"index": 0, "name": "NVIDIA", "memory_mib": 46068},
        ],
    )
    monkeypatch.setattr(modular.local, "validate_weights", lambda *_, **__: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=preflight_status if len(calls) == 1 else 0)

    monkeypatch.setattr(modular.subprocess, "run", run)
    status = modular.main(["vlm", "--no-build"])
    assert calls[0][-4:-2] == ["vlm-processing", "python"]
    assert "vlm_server.py" in calls[0][-1]
    assert "--no-deps" in calls[0]
    if preflight_status:
        assert status == 2 and len(calls) == 1
        assert "matching this source" in capsys.readouterr().err
    else:
        assert status == 0 and len(calls) == 2
        assert calls[1][-2:] == ["vlm-server", "vlm-processing"]


def test_managed_processing_endpoint_and_secrets_remain_independent():
    devices = [{"index": 0, "name": "NVIDIA", "memory_mib": 46068}]
    plan = local.plan_environment(
        devices,
        {
            "LOCAL_INFERENCE_API_KEY": "local-secret",
            "INFERENCE_API_KEY": "cloud-secret",
        },
    )
    assert (
        plan["LOCAL_INFERENCE_INSPECTION_ENDPOINT"]
        == "http://local-vlm-processing:8080/v1/inspections"
    )
    assert "local-secret" not in json.dumps(plan)
    assert "cloud-secret" not in json.dumps(plan)
    command = local.compose_command(
        root=local.ROOT,
        env_file=None,
        with_paddle=False,
        build=False,
    )
    assert "local-vlm-processing" in command and "--no-build" in command
    # Online builds can still acquire the pinned upstream image on first use.
    assert "--pull" not in command
    check = local.runtime_check("local-vlm-processing")
    assert "vlm_offload.py" in check and "vlm_server.py" in check
    assert "Persistent deployment settings path is required" not in check
