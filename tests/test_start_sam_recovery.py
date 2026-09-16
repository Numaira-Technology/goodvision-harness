"""Reuse complete environments while retaining normal failed-instance cleanup."""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sam_recovery_launcher", Path(__file__).parents[1] / "tools/start_sam.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def recovery(monkeypatch, tmp_path):
    client = Mock()
    client.show_instance.return_value = {
        "id": 42, "label": "goodvision-sam-test", "actual_status": "running",
    }
    client.destroy_instance.return_value = {"success": True}
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(
        VastAI=Mock(return_value=client),
    ))
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", Mock())
    key = tmp_path / "key"
    key.write_text("test-only-key")
    for name, value in {
        "VAST_API_KEY": "unused", "HF_TOKEN": "unused", "SAM_SSH_KEY": str(key),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["start_sam.py", "--resume-instance-id", "42"])
    monkeypatch.setattr(launcher.shutil, "which", lambda name: name)
    monkeypatch.setattr(launcher, "prepare_ssh_key", Mock(return_value=key))
    monkeypatch.setattr(launcher, "run", Mock())
    monkeypatch.setattr(launcher, "stop_process", Mock())
    monkeypatch.setattr(launcher, "wait_for_ssh", Mock(return_value=("host", 22)))
    monkeypatch.setattr(launcher, "wait_for_worker", Mock())
    worker = Mock()
    worker.wait.return_value = 0
    worker.returncode = 0
    monkeypatch.setattr(launcher.subprocess, "Popen", Mock(return_value=worker))
    return client


def test_failed_recovery_releases_instance_for_normal_replacement(recovery, capsys):
    launcher.wait_for_ssh.side_effect = RuntimeError("not ready")
    assert launcher.main() == 1
    recovery.search_offers.assert_not_called()
    recovery.create_instance.assert_not_called()
    recovery.destroy_instance.assert_called_once_with(id=42)
    assert "Released instance: 42" in capsys.readouterr().out


def test_ready_recovery_resumes_normal_release_with_original_ssh_options(recovery):
    assert launcher.main() == 1  # The simulated worker exits after readiness.
    recovery.search_offers.assert_not_called()
    recovery.create_instance.assert_not_called()
    recovery.destroy_instance.assert_called_once_with(id=42)
    options = launcher.wait_for_ssh.call_args.args[2]
    assert "IdentitiesOnly=yes" in options
    assert "BatchMode=yes" in options
    assert not any("KexAlgorithms" in option for option in options)


def test_new_instance_gets_ssh_permissions_before_first_connection(
    recovery, monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["start_sam.py"])
    recovery.search_offers.return_value = [{"id": 10, "dph_total": 0.5}]
    recovery.create_instance.return_value = {"new_contract": 42}
    assert launcher.main() == 1  # The simulated worker exits after readiness.
    onstart = recovery.create_instance.call_args.kwargs["onstart_cmd"]
    assert onstart == launcher.SSH_ONSTART
    recovery.update_instance.assert_not_called()
    recovery.recycle_instance.assert_not_called()
    recovery.destroy_instance.assert_called_once_with(id=42)


@pytest.mark.skipif(
    os.name != "posix" or getattr(os, "geteuid", lambda: -1)() != 0,
    reason="Linux root permissions, as used by the cloud container",
)
def test_onstart_fixes_modes_without_replacing_keys(tmp_path):
    keys = tmp_path / "authorized_keys"
    original = b"ssh-ed25519 existing-public-key\n"
    keys.write_bytes(original)
    keys.chmod(0o777)
    tmp_path.chmod(0o777)
    script = launcher.SSH_ONSTART.replace("/root/.ssh", shlex.quote(str(tmp_path)))
    subprocess.run(["sh", "-eu"], input=script, text=True, check=True)
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert keys.stat().st_mode & 0o777 == 0o600
    assert keys.read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="Remote setup runs in Linux")
@pytest.mark.parametrize("complete", [True, False])
def test_setup_reuses_existing_conda_and_environment(tmp_path, complete):
    remote = tmp_path / "remote"
    commands = tmp_path / "commands"
    commands.mkdir()
    (remote / "worker").mkdir(parents=True)
    marker = tmp_path / "python-runs"

    def executable(path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n" + contents)
        path.chmod(0o700)

    executable(commands / "apt-get", "exit 0\n")
    executable(commands / "curl", "exit 31\n")
    executable(remote / "conda/bin/conda", "exit 32\n")
    quoted_marker = shlex.quote(str(marker))
    first_failure = (
        f"if [ ! -f {quoted_marker} ]; then echo incomplete >> {quoted_marker}; "
        "exit 1; fi\n"
    ) if not complete else ""
    executable(
        remote / "env/bin/python", first_failure + f"echo ok >> {quoted_marker}\n",
    )
    script = launcher.SETUP.replace(launcher.REMOTE, str(remote))
    environment = {
        **os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
    }
    for _ in range(2):
        subprocess.run(
            ["bash", "-s"], input=script, text=True, env=environment, check=True,
        )
    # Complete environments only need validation. Incomplete environments get
    # three package-install steps and a final check before the second reuse.
    assert len(marker.read_text().splitlines()) == (2 if complete else 6)


@pytest.mark.parametrize("valid", [True, False])
def test_existing_hybrid_weights_are_verified_before_downloading(
    recovery, monkeypatch, valid,
):
    import urllib.request
    assert launcher.main() == 1
    command = next(
        call.args[0][-1] for call in launcher.run.call_args_list
        if "from hybrid import" in call.args[0][-1]
    )
    verification = Mock(side_effect=[None] if valid else [RuntimeError("bad"), None])
    monkeypatch.setitem(sys.modules, "hybrid", SimpleNamespace(
        CHECKPOINT_URL="https://example.invalid/test-weights",
        verify_checkpoint=verification,
    ))
    download = Mock()
    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    exec(shlex.split(command)[-1], {})
    assert verification.call_count == (1 if valid else 2)
    assert download.call_count == (0 if valid else 1)


@pytest.mark.parametrize("field,value", [
    ("id", 99), ("label", "another-project"), ("actual_status", "stopped"),
])
def test_recovery_refuses_other_or_nonrunning_instances(recovery, field, value):
    recovery.show_instance.return_value[field] = value
    assert launcher.main() == 1
    recovery.create_instance.assert_not_called()
    recovery.destroy_instance.assert_not_called()
    launcher.wait_for_ssh.assert_not_called()


def test_direct_timeout_tries_gateway_with_the_same_key(monkeypatch):
    client = Mock()
    client.show_instance.return_value = {
        "actual_status": "running", "public_ipaddr": "192.0.2.1",
        "ports": {"22/tcp": [{"HostPort": "20030"}]},
        "ssh_host": "ssh.example.test", "ssh_port": 11302,
    }
    probe = Mock(side_effect=[
        subprocess.TimeoutExpired("ssh", 15), SimpleNamespace(returncode=0),
    ])
    monkeypatch.setattr(launcher.subprocess, "run", probe)
    options = ["-i", "/private/key", "-o", "IdentitiesOnly=yes"]
    assert launcher.wait_for_ssh(client, 42, options) == ("ssh.example.test", 11302)
    calls = [call.args[0] for call in probe.call_args_list]
    assert calls[0] == ["ssh", *options, "-p", "20030", "root@192.0.2.1", "true"]
    assert calls[1] == [
        "ssh", *options, "-p", "11302", "root@ssh.example.test", "true",
    ]
