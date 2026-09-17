"""Behavioral coverage for the JavaScript classification display state."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script_name", [
        "dashboard_classification.cjs", "dashboard_feed.cjs",
        "dashboard_buffer_delay.cjs",
        "dashboard_camera_layout.cjs",
        "dashboard_routes.cjs",
        "dashboard_diagnostics.cjs",
    ]
)
def test_dashboard_handlers(script_name: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute dashboard DOM regression tests")
    script = Path(__file__).with_name(script_name)
    completed = subprocess.run(
        [node, "--test", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
