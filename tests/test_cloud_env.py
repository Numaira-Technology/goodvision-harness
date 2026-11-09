"""Offline SSH wiring: synthetic readers only, no real SSH hosts or credentials."""

import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

from tools import cloud_env as ce

R2_ENV = {
    "R2_ENDPOINT": "https://example.invalid",
    "R2_BUCKET": "synthetic",
    "R2_ACCESS_KEY_ID": "fake-akid",
    "R2_SECRET_ACCESS_KEY": "fake-secret",
}
ACTIVATE = "/opt/env/videojev/bin/activate"
PYTHON = "/opt/env/videojev/bin/python"


def test_restore_env_carries_secret_only_on_stdin_and_cleans():
    uploads, cleanups = [], []

    def runner(command, *, input=None, timeout=1800):
        assert all(value not in " ".join(command) for value in R2_ENV.values())
        assert "export R2_SECRET_ACCESS_KEY=" in input
        assert "fake-secret" in input
        return json.dumps(
            {
                "name": "videojev",
                "source": "r2",
                "activate": ACTIVATE,
                "python": PYTHON,
            }
        )

    result = ce.restore_env(
        ssh_options=["-i", "key"],
        host="h",
        port=22,
        name="videojev",
        remote_dir="/w",
        r2_env=R2_ENV,
        runner=runner,
        uploader=lambda *a: uploads.append(a),
        cleaner=lambda *a: cleanups.append(a),
    )
    assert result == PYTHON
    assert uploads == cleanups
    assert uploads[0][-1].startswith("/w/.env-reader-")


@pytest.mark.parametrize(
    "output",
    [
        "no json",
        "[]",
        '{"activate":123}',
        json.dumps(
            {
                "name": "other",
                "source": "r2",
                "activate": ACTIVATE,
                "python": PYTHON,
            }
        ),
        json.dumps(
            {
                "name": "videojev",
                "source": "r2",
                "activate": "/tmp/wrong",
                "python": PYTHON,
            }
        ),
        json.dumps(
            {
                "name": "videojev",
                "source": "failed",
                "activate": ACTIVATE,
                "python": PYTHON,
            }
        ),
        json.dumps(
            {
                "name": "videojev",
                "source": "r2",
                "activate": ACTIVATE,
            }
        ),
        json.dumps(
            {
                "name": "videojev",
                "source": "r2",
                "activate": ACTIVATE,
                "python": "/usr/bin/python3",
            }
        ),
    ],
)
def test_restore_env_requires_matching_result_and_cleans(output):
    cleanup = []
    with pytest.raises(RuntimeError, match="packed python path"):
        ce.restore_env(
            ssh_options=[],
            host="h",
            port=22,
            name="videojev",
            r2_env=R2_ENV,
            runner=lambda *a, **k: output,
            uploader=lambda *a: None,
            cleaner=lambda *a: cleanup.append(a),
        )
    assert len(cleanup) == 1


def test_credentials_must_be_complete_before_upload():
    with pytest.raises(ValueError, match="four"):
        ce.restore_env(
            ssh_options=[],
            host="h",
            port=22,
            name="videojev",
            r2_env={},
            uploader=lambda *a: pytest.fail("uploaded without credentials"),
        )


@pytest.mark.parametrize("phase", ["upload", "run"])
def test_cleanup_on_failure(phase):
    cleanup = []

    def fail(*a, **kw):
        raise subprocess.TimeoutExpired("synthetic", 1)

    with pytest.raises(subprocess.TimeoutExpired):
        ce.restore_env(
            ssh_options=[],
            host="h",
            port=22,
            name="videojev",
            r2_env=R2_ENV,
            uploader=fail if phase == "upload" else lambda *a: None,
            runner=fail,
            cleaner=lambda *a: cleanup.append(a),
        )
    assert len(cleanup) == 1


@pytest.mark.skipif(os.name != "posix", reason="remote POSIX shell")
def test_remote_command_executes_literal_paths_and_credentials(tmp_path):
    remote = tmp_path / "reader's $(literal); directory"
    (remote / "tools").mkdir(parents=True)
    (remote / "tools/__init__.py").write_text("")
    (remote / "tools/env_source.py").write_text(
        "import json,os,sys\n"
        "print(json.dumps([sys.argv[1:],os.environ['R2_SECRET_ACCESS_KEY']]))"
    )
    secret = "quote' dollar$ semicolon; newline\nvalue"
    cmd = ce.remote_command(
        "name'; $(literal)", python=sys.executable, remote_dir=str(remote)
    )
    result = subprocess.run(
        ["sh", "-c", cmd],
        input=ce.credential_blob({"R2_SECRET_ACCESS_KEY": secret}),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [["--model", "name'; $(literal)"], secret]
    assert secret not in cmd and not result.stderr


def test_uploader_uses_tar_stdin_and_quoted_remote_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    remote = "/workspace/reader's $(literal); directory"
    ce._default_uploader([], "h", 22, remote)
    argv, kw = calls[0]
    assert argv[0] == "ssh" and "scp" not in argv
    import shlex

    assert shlex.quote(remote + "/tools") in argv[-1]
    with tarfile.open(fileobj=io.BytesIO(kw["input"])) as tf:
        assert set(tf.getnames()) == {*ce.UPLOAD, "__init__.py"}
    ce._default_cleaner([], "h", 22, remote)
    assert calls[1][0][-1] == f"rm -rf -- {shlex.quote(remote)}"
