"""Compatibility of legacy imports and installed-package replay entry points."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALIASES = (
    ("goodvision_harness.scene_runtime", "goodvision_harness.runtime.scene"),
    ("goodvision_harness.scene_evidence", "goodvision_harness.runtime.scheduling"),
    ("goodvision_harness.scene_state", "goodvision_harness.events.state_machine"),
    ("goodvision_harness.scene_alerts", "goodvision_harness.actions.alerts"),
    ("goodvision_harness.scene_results", "goodvision_harness.actions.results"),
    ("tools.scene_replay", "goodvision_harness.evaluation.replay"),
)


@pytest.mark.parametrize("canonical_first", [False, True])
def test_legacy_imports_keep_identity_and_global_patches_in_fresh_interpreter(
    canonical_first,
):
    script = f"""
import importlib
pairs = {ALIASES!r}
for legacy, canonical in pairs:
    ordered = (canonical, legacy) if {canonical_first!r} else (legacy, canonical)
    first, second = (importlib.import_module(name) for name in ordered)
    assert first is second, (legacy, canonical)
legacy = importlib.import_module('goodvision_harness.scene_runtime')
canonical = importlib.import_module('goodvision_harness.runtime.scene')
legacy.monotonic = lambda: 42.0
assert canonical.SceneRuntime.__init__.__globals__['monotonic']() == 42.0
"""
    environment = dict(
        os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_installed_package_and_legacy_replay_commands_match(tmp_path):
    records = tmp_path / "observations.jsonl"
    records.write_text(
        json.dumps(
            {"type": "session", "stream_session_id": "test", "source_id": "camera"}
        )
        + "\n"
        + json.dumps({"type": "tick", "timestamp_seconds": 1})
        + "\n",
        encoding="utf-8",
    )
    environment = dict(
        os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT)
    )
    commands = (
        [sys.executable, str(ROOT / "tools/scene_replay.py"), str(records)],
        [sys.executable, "-m", "tools.scene_replay", str(records)],
        [sys.executable, "-m", "goodvision_harness.evaluation.replay", str(records)],
    )
    outputs = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        outputs.append([json.loads(line) for line in result.stdout.splitlines()])
    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[0]) == 2
    assert outputs[0][-1]["scene"]["timestamp_seconds"] == 1
