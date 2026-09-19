"""Exercise multi-camera console controls and bounded preview transport."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_fleet_console() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute dashboard regression tests")
    completed = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("fleet_console.cjs"))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
