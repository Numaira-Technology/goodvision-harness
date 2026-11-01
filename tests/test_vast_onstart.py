"""SSH recovery must tighten permissions without replacing keys or touching data."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/vast_vlm/onstart-ssh.sh"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Bash startup script")


def invoke(tmp_path, home):
    # chown needs root on the real server; use real chmod on temporary files here.
    commands = tmp_path / "commands"
    commands.mkdir(exist_ok=True)
    chown = commands / "chown"
    chown.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CHOWN_LOG"\n')
    chown.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{commands}:{os.environ['PATH']}",
        CHOWN_LOG=str(tmp_path / "chown.log"),
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; repair_ssh_permissions "$2"',
            "test",
            str(SCRIPT),
            str(home),
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def fixture_home(tmp_path):
    home = tmp_path / "home"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    keys = ssh / "authorized_keys"
    keys.write_bytes(b"ssh-ed25519 first-key owner\nssh-ed25519 second-key colleague\n")
    return home, ssh, keys


def test_repair_preserves_all_keys_and_unrelated_data_and_is_repeatable(tmp_path):
    home, ssh, keys = fixture_home(tmp_path)
    original = keys.read_bytes()
    data = home / "existing-data"
    data.write_bytes(b"must survive")
    home.chmod(0o777)
    ssh.chmod(0o777)
    keys.chmod(0o666)
    for _ in range(2):
        result = invoke(tmp_path, home)
        assert result.returncode == 0, result.stderr
        assert keys.read_bytes() == original
        assert data.read_bytes() == b"must survive"
        assert home.stat().st_mode & 0o022 == 0
        assert ssh.stat().st_mode & 0o777 == 0o700
        assert keys.stat().st_mode & 0o777 == 0o600
    args = (tmp_path / "chown.log").read_text().splitlines()
    assert args == ["root:root", str(home), str(ssh), str(keys)] * 2


@pytest.mark.parametrize(
    "bad", ["missing", "empty", "key_link", "ssh_link", "home_link"]
)
def test_unsafe_layout_refused_before_changes(tmp_path, bad):
    home, ssh, keys = fixture_home(tmp_path)
    if bad == "missing":
        keys.unlink()
    elif bad == "empty":
        keys.write_bytes(b"")
    elif bad == "key_link":
        outside = tmp_path / "outside"
        keys.rename(outside)
        keys.symlink_to(outside)
    elif bad == "ssh_link":
        outside = tmp_path / "outside"
        ssh.rename(outside)
        ssh.symlink_to(outside, target_is_directory=True)
    else:
        outside = tmp_path / "outside"
        home.rename(outside)
        home.symlink_to(outside, target_is_directory=True)
    result = invoke(tmp_path, home)
    assert result.returncode != 0
    assert not (tmp_path / "chown.log").exists()
