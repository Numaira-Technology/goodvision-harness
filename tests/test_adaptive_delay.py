from __future__ import annotations

from pathlib import Path

from goodvision_harness.adaptive_delay import LatencyTracker
from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
)
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.service import HarnessRuntime

ROOT = Path(__file__).resolve().parents[1]


def test_tracker_waits_for_enough_samples_then_returns_p95_plus_headroom() -> None:
    tracker = LatencyTracker(window_size=32, min_samples=8, headroom=0.2)
    for _ in range(7):
        tracker.observe(0.7)
    assert tracker.recommended_delay() is None
    tracker.observe(0.7)
    assert tracker.recommended_delay() == 0.9


def test_tracker_clamps_and_ignores_tiny_changes() -> None:
    tracker = LatencyTracker(
        min_samples=8, min_delay=0.4, max_delay=8.0, headroom=0.2, min_change=0.1,
    )
    for _ in range(8):
        tracker.observe(0.05)
    assert tracker.recommended_delay() == 0.4
    tracker.observe(0.06)
    assert tracker.recommended_delay() is None
    for _ in range(32):
        tracker.observe(12.0)
    assert tracker.recommended_delay() == 8.0


def _safety_result(latency: float, *, kind: str = "safety_events") -> AnalysisResult:
    return AnalysisResult(
        "msg",
        "camera",
        "session",
        1,
        "2026-09-17T00:00:00Z",
        FrameReference(1, 1.0, "2026-09-17T00:00:00Z", 320, 240),
        (EventJudgment("smoking", False, 0.9, "Clear"),),
        (),
        "test",
        latency,
        analysis_kind=kind,
    )


def test_runtime_adapts_display_delay_from_safety_latency(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    try:
        original = runtime._delayed_overlay.delay_seconds
        assert original == 2.5
        for _ in range(8):
            runtime._on_timeline_result(_safety_result(0.6))
        assert runtime._delayed_overlay.delay_seconds == 0.8
        assert runtime._display_settings.delay_seconds == 0.8
    finally:
        runtime.stop()


def test_user_buffer_override_blocks_adaptive_delay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"), deployment_store=store)
    try:
        runtime.update_deployment_settings({"buffer_delay_seconds": 3.5})
        for _ in range(8):
            runtime._on_timeline_result(_safety_result(0.6))
        assert runtime._delayed_overlay.delay_seconds == 3.5
    finally:
        runtime.stop()


def test_face_results_do_not_drive_adaptive_delay(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    try:
        for _ in range(8):
            runtime._on_timeline_result(
                _safety_result(0.05, kind="face_recognition"),
            )
        assert runtime._delayed_overlay.delay_seconds == 2.5
    finally:
        runtime.stop()
