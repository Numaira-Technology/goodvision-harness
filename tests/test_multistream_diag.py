"""Exercise the diagnostic runner with process/HTTP doubles, no services."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="Unix diagnostic shell runner",
)


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


@pytest.mark.parametrize(
    "start_mode,status_mode,state,success",
    [(True, True, "running", True),
     (False, True, "running", False),
     (True, False, "running", False),
     (True, True, "error", False)],
)
def test_sam_off_locks_dry_run_and_rejects_wrong_running_mode(
    tmp_path, start_mode, status_mode, state, success,
):
    diag = tmp_path / "diag"
    diag.mkdir()
    shutil.copyfile(ROOT / "diag/run_multistream_diag.sh", diag / "run.sh")
    (diag / "parse_pipeline_log.py").write_text("pass\n")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"offline process double")
    commands = tmp_path / "commands"
    commands.mkdir()
    for command in ["mediamtx", "ffmpeg", "lsof", "sleep"]:
        executable(commands / command, "pass\n")
    capture = tmp_path / "capture.json"
    python = commands / "harness-python"
    executable(python, '''
import json, os, sys
from pathlib import Path
if sys.argv[1:3] == ["-m", "goodvision_harness.service"]:
    names = ["HARNESS_DRY_RUN", "SCENE_ENGINE_MODE", "SCENE_INDEPENDENT_VLM",
             "RICH_LABEL_ENABLED", "PERSON_TRACKING_ENABLED", "PERCEPTION_METHOD",
             "HARNESS_DEPLOYMENT_SETTINGS_PATH"]
    captured = {n: os.environ.get(n) for n in names}
    Path(os.environ["CAPTURE"]).write_text(json.dumps(captured))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    executable(commands / "curl", '''
import json, os, sys
from pathlib import Path
if any("/sessions/start" in arg for arg in sys.argv):
    payload = json.loads(sys.argv[sys.argv.index("-d") + 1])
    Path(os.environ["PAYLOAD"]).write_text(json.dumps(payload))
    response = {"type": "session", "dry_run": os.environ["START_MODE"] == "True"}
    print(json.dumps(response))
else:
    print(json.dumps({"connected": True, "dry_run": os.environ["STATUS_MODE"] == "True",
                      "pipeline_state": os.environ["STATE"]}))
''')
    environment = {
        **os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "VIDEO": str(video), "STREAMS": "1", "SECS": "0", "SAM": "off",
        "PY": str(python), "CAPTURE": str(capture),
        "PAYLOAD": str(tmp_path / "payload.json"),
        "START_MODE": str(start_mode), "STATUS_MODE": str(status_mode), "STATE": state,
        # Prove inherited model settings and deployment defaults are isolated.
        "HARNESS_DRY_RUN": "false", "SCENE_ENGINE_MODE": "active",
        "SCENE_INDEPENDENT_VLM": "true", "RICH_LABEL_ENABLED": "true",
        "PERSON_TRACKING_ENABLED": "true", "PERCEPTION_METHOD": "sam_live",
        "HARNESS_DEPLOYMENT_SETTINGS_PATH": str(tmp_path / "existing-settings.json"),
    }
    result = subprocess.run(
        ["bash", str(diag / "run.sh")], env=environment,
        capture_output=True, text=True, timeout=20,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert ("running 1 stream(s) for" in result.stdout) is success
    assert json.loads((tmp_path / "payload.json").read_text())["dry_run"] is True
    actual = json.loads(capture.read_text())
    assert actual["HARNESS_DRY_RUN"] == "true"
    assert actual["SCENE_ENGINE_MODE"] == "disabled"
    for name in [
        "SCENE_INDEPENDENT_VLM", "RICH_LABEL_ENABLED", "PERSON_TRACKING_ENABLED",
    ]:
        assert actual[name] == "false"
    assert actual["PERCEPTION_METHOD"] == ""
    settings_path = "HARNESS_DEPLOYMENT_SETTINGS_PATH"
    assert actual[settings_path] != environment[settings_path]
