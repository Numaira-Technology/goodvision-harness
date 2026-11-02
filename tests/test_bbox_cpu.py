"""CPU readiness must attest a successful YOLO warmup, never pretend CUDA."""

import httpx
import pytest

from goodvision_harness.rich_labeling import RichLabelHTTPClient
from tools.build_gpu_runtime import build_commands
from workers.detector_rich_label.app import DetectorEngine
from workers.detector_rich_label.detectors import YoloDetector


def test_cpu_warmup_and_failure_are_reported_truthfully(monkeypatch):
    detector = YoloDetector("yolo26-s", "unused.pt", "cpu", 0.35)
    monkeypatch.setattr(detector, "detect", lambda image: ())
    detector.warmup()
    assert detector.health() == {
        "ready": True, "gpu_ready": False, "provider_ready": True,
        "device": "cpu", "model": "yolo26-s", "runtime": "ultralytics",
    }
    engine = DetectorEngine()
    monkeypatch.setattr(engine, "_get_detector", lambda key: detector)
    assert engine.health("yolo26-s")["ready"]

    def fail(image):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(detector, "detect", fail)
    engine._warmed_models.clear()
    assert engine.health("yolo26-s")["ready"] is False
    assert detector.health()["ready"] is False


def test_client_accepts_explicit_cpu_yolo_but_rejects_false_or_mismatched_health():
    payload = {"ready": True, "gpu_ready": False, "provider_ready": True,
               "model": "yolo26-s", "device": "cpu", "runtime": "ultralytics"}
    with RichLabelHTTPClient(
        "http://worker/v1/labels", mode="tracks",
        headers={"X-GoodVision-Detector-Model": "yolo26-s"},
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        ),
    ) as client:
        assert client.gpu_ready()  # Legacy method means inference capability.
        for key, invalid in (("ready", False), ("device", "cuda"),
                             ("model", "yolov8-m"), ("runtime", "dummy"),
                             ("model", []), ("model", "dummy"),
                             ("provider_ready", False)):
            original = payload[key]
            payload[key] = invalid
            assert not client.tracking_ready()
            payload[key] = original


def test_shared_runtime_builds_each_compatible_torch_layer_once():
    commands = build_commands(["contour", "bbox", "system1"])
    assert len(commands) == 2
    assert "goodvision-gpu-runtime:torch2.10-cu128" in commands[0]
    assert "goodvision-gpu-runtime:torch2.8-cu128" in commands[1]


@pytest.mark.parametrize("device", ["cpu", "cu128"])
@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_setup_replaces_a_different_existing_torch_build(
    tmp_path, monkeypatch, device, platform
):
    from tools import setup_bbox

    calls = []
    monkeypatch.setattr(setup_bbox.sys, "platform", platform)
    monkeypatch.setattr(setup_bbox, "ROOT", tmp_path)
    monkeypatch.setattr(setup_bbox.sys, "argv", ["setup_bbox.py", "--device", device])
    environment = tmp_path / ".venv-module-bbox"
    python = environment / (
        "Scripts/python.exe" if setup_bbox.os.name == "nt" else "bin/python"
    )
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(
        setup_bbox.subprocess, "run",
        lambda command, **options: calls.append((command, options)),
    )
    setup_bbox.main()
    command = calls[0][0]
    # A bare ==2.10.0 would accept an already-installed CPU wheel even when
    # switching to CUDA. The local version must select the requested build.
    assert f"torch==2.10.0+{device}" in command
    assert f"torchvision==0.25.0+{device}" in command
    assert command[-1] == f"https://download.pytorch.org/whl/{device}"
    assert all(command[0] == str(python) for command, options in calls)
    assert all(options["check"] for command, options in calls)


def test_macos_cpu_setup_uses_the_published_wheels(tmp_path, monkeypatch):
    from tools import setup_bbox

    calls = []
    monkeypatch.setattr(setup_bbox, "ROOT", tmp_path)
    monkeypatch.setattr(setup_bbox.sys, "platform", "darwin")
    monkeypatch.setattr(setup_bbox.sys, "argv", ["setup_bbox.py", "--device", "cpu"])
    monkeypatch.setattr(
        setup_bbox.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )
    setup_bbox.main()
    command = calls[1]  # First creates the isolated environment.
    assert command[4:] == ["torch==2.10.0", "torchvision==0.25.0"]


def test_macos_rejects_cuda_before_creating_an_environment(tmp_path, monkeypatch):
    from tools import setup_bbox

    monkeypatch.setattr(setup_bbox, "ROOT", tmp_path)
    monkeypatch.setattr(setup_bbox.sys, "platform", "darwin")
    monkeypatch.setattr(setup_bbox.sys, "argv", ["setup_bbox.py", "--device", "cu128"])
    monkeypatch.setattr(
        setup_bbox.subprocess, "run", lambda *a, **k: pytest.fail("invalid install")
    )
    with pytest.raises(SystemExit) as exc:
        setup_bbox.main()
    assert exc.value.code == 2
    assert not (tmp_path / ".venv-module-bbox").exists()
