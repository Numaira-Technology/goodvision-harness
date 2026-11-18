"""CUDA 13 admission and cleanup using synthetic hosts; no cloud or GPU calls."""

import subprocess
import sys

import pytest

from test_qwen8b_release import fake_runtime
from tools import start_qwen8b_vllm as launcher


def offer(**changes):
    return {
        "id": 1, "num_gpus": 1, "gpu_ram": 24564, "dph_total": 1.20,
        "cuda_max_good": 13.0, "driver_version": "580.65.06", **changes,
    }


@pytest.mark.parametrize("changes", [
    {"cuda_max_good": 12.8}, {"driver_version": "570.124.06"},
    {"cuda_max_good": None}, {"cuda_max_good": float("nan")},
    {"cuda_max_good": float("inf")}, {"driver_version": "unknown"},
    {"driver_version": None},
])
def test_incompatible_or_unknown_offer_is_rejected(changes):
    with pytest.raises(RuntimeError, match="CUDA 13.0"):
        launcher._eligible_offers([offer(**changes)])


@pytest.mark.parametrize("missing", ["cuda_max_good", "driver_version"])
def test_missing_compatibility_metadata_is_rejected(missing):
    host = offer()
    del host[missing]
    with pytest.raises(RuntimeError, match="CUDA 13.0"):
        launcher._eligible_offers([host])


def test_cuda13_selection_preserves_hf_cuda12_query():
    assert "cuda_max_good>=13.0" in launcher._vast_query(80)
    assert "cuda_vers>=12.8" in launcher.H._vast_query(80)
    compatible = offer(id=2)
    assert launcher._eligible_offers([offer(cuda_max_good=12.8), compatible]) == [compatible]


@pytest.mark.parametrize("output,valid", [
    ("580.65.06\n", True), ("590.1\n", True),
    ("570.124.06\n", False), ("\n", False), ("unknown\n", False),
    ("580.65.06\n580.65.06\n", False),
])
def test_actual_driver_requires_one_compatible_gpu(monkeypatch, output, valid):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: output)
    if valid:
        launcher._verify_host_driver(["synthetic-ssh"])
    else:
        with pytest.raises(RuntimeError, match="driver >=580"):
            launcher._verify_host_driver(["synthetic-ssh"])


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_main_checks_driver_before_bootstrap_and_downloads(tmp_path, monkeypatch, resume):
    real_guard = launcher._verify_host_driver
    vast, _, _ = fake_runtime(monkeypatch, tmp_path, launcher, None)
    events = []
    compatible = offer(id=2)
    if resume:
        monkeypatch.setattr(vast, "show_instance", lambda **k: compatible | {
            "label": "goodvision-qwen8b-vllm-synthetic",
        }, raising=False)
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume-instance-id", "123"])
        def no_rental(*a, **k):
            pytest.fail("compatible resume must not search or rent")
        monkeypatch.setattr(vast, "search_offers", no_rental)
        monkeypatch.setattr(launcher.S, "create_from_offers", no_rental)
    else:
        def search(**kwargs):
            assert "cuda_max_good>=13.0" in kwargs["query"]
            return [offer(cuda_max_good=12.8, dph_total=0.5), compatible]
        def create(_vast, selected, **kwargs):
            assert selected == [compatible]
            vast.alive = True
            events.append("rent")
            return 123
        monkeypatch.setattr(vast, "search_offers", search)
        monkeypatch.setattr(launcher.S, "create_from_offers", create)
    monkeypatch.setattr(launcher, "_verify_host_driver", real_guard)
    admission = "nvidia-smi --query-gpu=driver_version --format=csv,noheader"

    def driver(argv, **kwargs):
        command = argv[-1]
        # The admission probe is the only call that may gate bootstrap.
        # Later nvidia-smi probes record restore/serve state and are not it.
        if command == admission:
            events.append("driver")
            return "580.65.06\n"
        if "nvidia-smi" in command:
            return "580.65.06, NVIDIA GeForce RTX 4090, 0 MiB, 24564 MiB\n"
        raise AssertionError(f"unexpected check_output before eval: {command}")

    monkeypatch.setattr(launcher.subprocess, "check_output", driver)
    monkeypatch.setattr(launcher.S, "run", lambda *a, **k: events.append("bootstrap"))
    monkeypatch.setattr(launcher, "prefetch_models", lambda **k: events.append("weights"))
    assert launcher.main() == 0
    assert events.index("driver") < events.index("bootstrap") < events.index("weights")
    assert vast.destroyed == [123]
    if not resume:
        assert events.index("rent") < events.index("driver")


@pytest.mark.parametrize("metadata", [
    offer(cuda_max_good=12.8), offer(driver_version="570.124.06"), {},
])
def test_rejected_resume_is_not_modified_or_released(tmp_path, monkeypatch, metadata):
    vast, _, _ = fake_runtime(monkeypatch, tmp_path, launcher, None)
    monkeypatch.setattr(vast, "show_instance", lambda **k: metadata | {
        "label": "goodvision-qwen8b-vllm-synthetic",
    }, raising=False)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume-instance-id", "123"])
    def forbidden(*a, **k):
        pytest.fail("rejected resume reached a cloud/SSH operation")
    monkeypatch.setattr(vast, "search_offers", forbidden)
    monkeypatch.setattr(launcher.S, "create_from_offers", forbidden)
    monkeypatch.setattr(launcher.H, "attach_key", forbidden)
    monkeypatch.setattr(launcher.S, "run", forbidden)
    monkeypatch.setattr(launcher, "prefetch_models", forbidden)
    monkeypatch.setattr(launcher, "restore_env", forbidden)
    assert launcher.main() == 1
    assert vast.destroyed == []


@pytest.mark.parametrize("fault", ["old-driver", "unknown", "timeout", "ssh-error"])
def test_failed_actual_driver_probe_releases_fresh_instance_before_downloads(
    tmp_path, monkeypatch, fault,
):
    real_guard = launcher._verify_host_driver
    vast, _, _ = fake_runtime(monkeypatch, tmp_path, launcher, None)
    monkeypatch.setattr(launcher, "_verify_host_driver", real_guard)
    def driver(*a, **k):
        if fault == "timeout":
            raise subprocess.TimeoutExpired(["synthetic-ssh"], 30)
        if fault == "ssh-error":
            raise OSError("synthetic SSH failure")
        return "570.124.06\n" if fault == "old-driver" else "unknown\n"
    monkeypatch.setattr(launcher.subprocess, "check_output", driver)
    def forbidden(*a, **k):
        pytest.fail("failed actual driver probe reached bootstrap/downloads")
    monkeypatch.setattr(launcher.S, "run", forbidden)
    monkeypatch.setattr(launcher.H, "_upload_repo", forbidden)
    monkeypatch.setattr(launcher, "prefetch_models", forbidden)
    monkeypatch.setattr(launcher, "restore_env", forbidden)
    assert launcher.main() == 1
    assert vast.destroyed == [123] and not vast.alive
