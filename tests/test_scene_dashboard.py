"""Behavior tests for the shipped continuous-entity/event dashboard."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_scene_dashboard_handlers() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard DOM tests")
    completed = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("scene_dashboard.cjs"))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
