"""Execute generated sweep shells against local stubs, never rent a GPU."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path("docs/validation/evidence/videojev-compile-batch-20261122")
LAUNCHERS = ("start_sweep.py", "start_sweep_retest.py", "start_sweep_live05.py")


@pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash"), reason="POSIX GPU launcher",
)
@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("mode", ["start-fails", "not-ready", "sweep-fails", "success"])
def test_generated_launcher_preserves_failure_after_cleanup(tmp_path, launcher, mode):
    # Extract only the pure generator. Importing main would require provider
    # configuration; no provider SDK, credentials or real service is used here.
    path = ROOT / launcher
    function = next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "sweep_script"
    )
    (tmp_path / "repo").mkdir()
    bindir = tmp_path / "env/bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.write_text('''#!/bin/sh
case "$*" in
  *goodvision_harness.videojev_service*)
    [ "$SWEEP_TEST_MODE" = start-fails ] && exit 1
    exec /bin/sleep 60 ;;
  *tools.videojev_probe.gpu_sweep*)
    echo attempt >> "$SWEEP_TEST_ATTEMPTS"
    [ "$SWEEP_TEST_MODE" = sweep-fails ] && exit 1
    exit 0 ;;
  *)
    case "$SWEEP_TEST_MODE" in start-fails|not-ready) exit 1;; esac
    exit 0 ;;
esac
''')
    python.chmod(0o700)
    context = {"BASE": str(tmp_path), "REPO": str(tmp_path / "repo"),
               "CACHE": str(tmp_path / "cache"), "RBUNDLE": str(tmp_path / "bundle"),
               "ROUT": str(tmp_path / "out"),
               "CONFIGS": (("base", "bf16", "off", 4), ("candidate", "fp8", "mlp", 4))}
    code = compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec")
    exec(code, context)
    stubs = '''seq() { printf '1\\n'; }
sleep() { :; }
curl() { printf '{"ready":true}'; }
'''
    attempts = tmp_path / "attempts"
    result = subprocess.run(
        [shutil.which("bash"), "-s"], input=stubs + context["sweep_script"]("offline"),
        text=True, capture_output=True, timeout=10,
        env={
            **os.environ, "SWEEP_TEST_MODE": mode, "SWEEP_TEST_ATTEMPTS": str(attempts),
        },
    )
    expected_code = 0 if mode == "success" else 1
    assert result.returncode == expected_code, result.stdout + result.stderr
    # A failure does not skip collection.
    assert "candidate: starting service" in result.stdout
    assert "offline: done" in result.stdout
    if mode in ("sweep-fails", "success"):
        assert attempts.read_text().splitlines() == ["attempt", "attempt"]
    else:
        assert not attempts.exists()
