"""Behavioral coverage for local GPU choices in the deployment console."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_local_gpu_controls() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute dashboard DOM regression tests")
    completed = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("dashboard_local_gpu.cjs"))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
