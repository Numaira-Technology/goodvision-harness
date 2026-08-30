from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_windows_hybrid_exposes_only_cpu_rich_worker_on_loopback() -> None:
    overlay_path = ROOT / "compose.rich-cpu-host.yaml"
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))

    worker = overlay["services"]["rich-label-worker"]
    assert worker["environment"]["RICH_DEVICE"] == "cpu"
    assert worker["ports"] == [
        "127.0.0.1:${RICH_HOST_PORT:-8091}:8091"
    ]
    assert "harness" not in overlay["services"]


def test_windows_hybrid_script_requires_directml_and_full_face_schedule() -> None:
    script = (ROOT / "tools" / "start_windows_hybrid_demo.ps1").read_text(
        encoding="utf-8"
    )

    assert 'FACE_PROVIDER = "DmlExecutionProvider"' in script
    assert 'FACE_LIVE_SCHEDULER = "full"' in script
    assert 'RICH_LABEL_ENABLED = "true"' in script
    assert 'HARNESS_DRY_RUN = "false"' in script
    assert 'HARNESS_AUTOSTART = "false"' in script
    assert "DmlExecutionProvider is unavailable" in script
    assert '$upArgs += "rich-label-worker"' in script
    assert "& docker @composeArgs @upArgs" in script
    assert '$upArgs += "--no-build"' in script


def test_windows_hybrid_instructions_keep_runtime_packages_isolated() -> None:
    instructions = (ROOT / "WINDOWS_HYBRID.md").read_text(encoding="utf-8")

    assert ".venv-directml" in instructions
    assert ".[dev,face-directml,webrtc]" in instructions
    assert "onnxruntime-directml" in instructions
    assert "DmlExecutionProvider" in instructions
    assert "FACE_LIVE_SCHEDULER=full" in instructions
