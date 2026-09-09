"""Behavioral coverage for the JavaScript classification display state."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_classification_handlers() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute dashboard DOM regression tests")
    script = Path(__file__).with_name("dashboard_classification.cjs")
    completed = subprocess.run(
        [node, "--test", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
