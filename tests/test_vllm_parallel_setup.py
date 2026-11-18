"""Parallel setup timing must preserve overlap and fail-fast cleanup."""

import threading
from types import SimpleNamespace

import pytest

from tools import start_qwen8b_vllm as launcher


def test_setup_overlaps_and_records_each_duration_on_caller(monkeypatch):
    caller = threading.get_ident()
    clock = threading.local()
    barrier = threading.Barrier(2)
    records = {}
    monkeypatch.setattr(
        launcher, "time", SimpleNamespace(monotonic=lambda: getattr(clock, "now", 0))
    )

    def step(seconds, result):
        barrier.wait(timeout=3)
        clock.now = seconds
        return result

    def record(name, seconds):
        assert threading.get_ident() == caller
        records[name] = seconds

    monkeypatch.setattr(launcher, "_record_setup", record)
    assert launcher._run_setup_steps(
        weights=lambda: step(3, "weights"), env=lambda: step(7, "/env/bin/activate")
    ) == {"weights": "weights", "env": "/env/bin/activate"}
    assert records == {"weight_download_seconds": 3, "env_restore_seconds": 7}


def test_setup_failure_does_not_wait_or_write_after_cleanup(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    caller = threading.get_ident()
    records = []

    def weights():
        started.set()
        try:
            assert release.wait(timeout=3)
        finally:
            finished.set()

    def env():
        assert started.wait(timeout=3)
        raise ValueError("restore failed")

    def record(name, seconds):
        assert threading.get_ident() == caller
        assert seconds >= 0
        records.append(name)

    monkeypatch.setattr(launcher, "_record_setup", record)
    try:
        with pytest.raises(ValueError, match="restore failed"):
            launcher._run_setup_steps(weights=weights, env=env)
        assert not finished.is_set()
        assert records == ["env_restore_seconds"]
    finally:
        release.set()
        assert finished.wait(timeout=3)
    assert records == ["env_restore_seconds"]


def test_setup_logging_failure_preserves_result(monkeypatch):
    from tools import vast_run_log

    def broken_log(*args):
        raise OSError("synthetic log failure")

    monkeypatch.setattr(vast_run_log, "record_setup", broken_log)
    assert launcher._run_setup_steps(weights=lambda: "weights", env=lambda: "env") == {
        "weights": "weights", "env": "env"
    }
