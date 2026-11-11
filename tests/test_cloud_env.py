"""Remote env restore wiring: creds over stdin only; activate path parsed back."""

import json

from tools import cloud_env as ce

R2_ENV = {
    "R2_ENDPOINT": "https://acc.r2.cloudflarestorage.com",
    "R2_BUCKET": "goodvision-models",
    "R2_ACCESS_KEY_ID": "akid-public",
    "R2_SECRET_ACCESS_KEY": "s3cr3t-value",
}


def test_remote_command_sources_stdin_and_restores():
    cmd = ce.remote_command("videojev", remote_dir="/w")
    assert ". /dev/stdin" in cmd
    assert "-m tools.env_source --model videojev" in cmd
    assert "PYTHONPATH=/w" in cmd


def test_restore_env_carries_secret_only_on_stdin():
    calls = []

    def fake_uploader(ssh_options, host, port, remote_dir):
        calls.append(("upload", remote_dir))

    def fake_runner(command, *, input=None, timeout=1800):
        calls.append(("run", command, input))
        # no secret value may appear in argv
        joined = " ".join(command)
        assert "s3cr3t-value" not in joined
        assert "akid-public" not in joined
        return json.dumps({"name": "videojev", "source": "r2",
                           "activate": "/opt/env/videojev/bin/activate"}) + "\n"

    activate = ce.restore_env(
        ssh_options=["-i", "key"], host="h", port=22, name="videojev",
        remote_dir="/w", r2_env=R2_ENV, runner=fake_runner, uploader=fake_uploader,
    )
    assert activate == "/opt/env/videojev/bin/activate"
    # the secret travelled on stdin (the blob), not argv
    run_call = next(c for c in calls if c[0] == "run")
    assert "export R2_SECRET_ACCESS_KEY=" in run_call[2]
    assert "s3cr3t-value" in run_call[2]


def test_restore_env_raises_without_activate_line():
    def fake_runner(command, *, input=None, timeout=1800):
        return "some log line\nno json here\n"

    import pytest

    with pytest.raises(RuntimeError, match="activate path"):
        ce.restore_env(ssh_options=[], host="h", port=22, name="videojev",
                       r2_env=R2_ENV, runner=fake_runner, uploader=lambda *a: None)
