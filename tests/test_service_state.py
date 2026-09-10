from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.privacy import (
    PUBLIC_ANALYSIS_ERROR,
    PUBLIC_STREAM_ERROR,
    PUBLIC_STREAM_SUPPORT_ERROR,
)
from goodvision_harness.service import HarnessRuntime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime() -> Iterator[HarnessRuntime]:
    instance = HarnessRuntime(load_config(ROOT / "config.yaml"))
    try:
        yield instance
    finally:
        instance.stop()


def test_pipeline_recovery_clears_runtime_error(runtime: HarnessRuntime) -> None:
    runtime._on_pipeline_state({
        "revision": 1, "status": "live", "last_error": PUBLIC_ANALYSIS_ERROR,
        "branch_errors": {"safety": "timeout"},
    })
    assert runtime.status()["last_error"] == PUBLIC_ANALYSIS_ERROR

    runtime._on_pipeline_state({
        "revision": 2, "status": "live", "last_error": None, "branch_errors": {},
    })

    status = runtime.status()
    assert status["last_error"] is None
    assert status["pipeline"]["last_error"] is None
    assert status["pipeline"]["branch_errors"] == {}


@pytest.mark.parametrize("late_revision", [1, 2])
def test_old_or_duplicate_state_cannot_hide_active_error(
    runtime: HarnessRuntime, late_revision: int,
) -> None:
    runtime._on_pipeline_state({
        "revision": 2, "status": "live", "last_error": PUBLIC_ANALYSIS_ERROR,
        "branch_errors": {"safety": "timeout"},
    })

    runtime._on_pipeline_state({
        "revision": late_revision, "status": "live", "last_error": None,
        "branch_errors": {},
    })

    status = runtime.status()
    assert status["last_error"] == PUBLIC_ANALYSIS_ERROR
    assert status["pipeline"]["revision"] == 2
    assert status["pipeline"]["branch_errors"] == {"safety": "timeout"}


def test_old_state_cannot_reintroduce_recovered_error(runtime: HarnessRuntime) -> None:
    runtime._on_pipeline_state({
        "revision": 3, "status": "live", "last_error": None,
    })

    runtime._on_pipeline_state({
        "revision": 2, "status": "live", "last_error": PUBLIC_ANALYSIS_ERROR,
    })

    assert runtime.status()["last_error"] is None
    assert runtime.status()["pipeline"]["revision"] == 3


def test_missing_error_field_does_not_claim_recovery(runtime: HarnessRuntime) -> None:
    runtime._on_pipeline_state({
        "status": "warming", "last_error": PUBLIC_ANALYSIS_ERROR,
    })

    runtime._on_pipeline_state({"status": "live"})

    assert runtime.status()["last_error"] == PUBLIC_ANALYSIS_ERROR


def test_new_session_accepts_fresh_state_revision(runtime: HarnessRuntime) -> None:
    runtime._on_pipeline_state({
        "revision": 10, "status": "live", "last_error": PUBLIC_ANALYSIS_ERROR,
    })
    runtime._update_state(pipeline=None, last_error=None)

    runtime._on_pipeline_state({
        "revision": 1, "status": "warming", "last_error": None,
    })

    assert runtime.status()["last_error"] is None
    assert runtime.status()["pipeline"]["revision"] == 1


@pytest.mark.parametrize("error", [PUBLIC_STREAM_ERROR, PUBLIC_STREAM_SUPPORT_ERROR])
def test_stream_failure_remains_actionable_in_status(runtime, error) -> None:
    runtime._on_end("error", error)

    status = runtime.status()
    assert status["pipeline_state"] == "error"
    assert status["last_error"] == error


def test_session_failure_still_hides_unknown_exception_text(runtime) -> None:
    runtime._on_end("error", "http://private-source/?token=secret")

    assert runtime.status()["last_error"] == PUBLIC_ANALYSIS_ERROR
