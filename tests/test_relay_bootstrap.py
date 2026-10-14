"""Offline key lifecycle and launcher checks; all credentials are dummy fixtures."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import relay_bootstrap, start_modular, start_sam
from tools.cloud_analysis import runtime_environment
from tools.cloud_gpu_controller import CloudController
from tools.gpu_analysis_host import lane_environment

KEY = "0123456789abcdef" * 4
OTHER_KEY = "fedcba9876543210" * 4


@pytest.fixture
def native(monkeypatch):
    # Do not change any real ownership, even in root-run Linux CI.
    monkeypatch.setattr(relay_bootstrap.os, "geteuid", lambda: -1, raising=False)
    monkeypatch.setattr(relay_bootstrap.secrets, "token_hex", lambda _size: KEY)


def test_tracking_only_does_not_create_key_or_change_environment(tmp_path):
    environment = {"VAST_API_KEY": "fixture-rental", "HARNESS_RELAY_KEY": "unused"}
    assert relay_bootstrap.controller_environment(environment, tmp_path) == environment
    assert not list(tmp_path.iterdir())


def test_atomic_publish_is_complete_private_and_restart_reuses_key(
    native,
    tmp_path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "private" / "key"
    environment = {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(path)}
    link = relay_bootstrap.os.link
    observed = []

    def inspect_before_publish(staging, destination):
        assert not destination.exists()
        assert Path(staging).read_text("utf-8") == KEY
        if os.name == "posix":
            assert Path(staging).stat().st_mode & 0o777 == 0o600
        observed.append(True)
        link(staging, destination)

    monkeypatch.setattr(relay_bootstrap.os, "link", inspect_before_publish)
    first = relay_bootstrap.controller_environment(environment, tmp_path / "settings")
    inode = path.stat().st_ino
    second = relay_bootstrap.controller_environment(environment, tmp_path / "settings")
    assert first["HARNESS_RELAY_KEY"] == second["HARNESS_RELAY_KEY"] == KEY
    assert path.stat().st_ino == inode
    assert observed == [True]
    assert not list(path.parent.glob(".relay-key-*"))
    if os.name == "posix":
        assert path.parent.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
    assert capsys.readouterr() == ("", "")


def test_concurrent_controllers_use_one_complete_winner_key(
    native,
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "shared" / "key"
    barrier = Barrier(2)
    values = iter((KEY, OTHER_KEY))
    monkeypatch.setattr(
        relay_bootstrap.secrets, "token_hex", lambda _size: next(values)
    )
    link = relay_bootstrap.os.link

    def concurrent_publish(source, destination):
        barrier.wait(timeout=5)
        link(source, destination)

    monkeypatch.setattr(relay_bootstrap.os, "link", concurrent_publish)
    environment = {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(path)}
    with ThreadPoolExecutor(max_workers=2) as controllers:
        futures = [
            controllers.submit(
                relay_bootstrap.controller_environment, environment, tmp_path
            )
            for _ in range(2)
        ]
        keys = [future.result(timeout=5)["HARNESS_RELAY_KEY"] for future in futures]
    assert keys[0] == keys[1] == path.read_text("utf-8")
    assert keys[0] in (KEY, OTHER_KEY)
    assert not list(path.parent.glob(".relay-key-*"))


@pytest.mark.parametrize(
    "content", [b"fixture-private-short-value", b"x" * 1024, b"\xff"]
)
def test_invalid_existing_key_is_not_replaced_and_error_does_not_echo_value(
    native,
    tmp_path,
    content,
):
    path = tmp_path / "key"
    path.write_bytes(content)
    environment = {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(path)}
    with pytest.raises(ValueError, match="unavailable or invalid") as error:
        relay_bootstrap.controller_environment(environment, tmp_path)
    assert path.read_bytes() == content
    assert content.decode("utf-8", errors="replace") not in str(error.value)
    assert not list(tmp_path.glob(".relay-key-*"))


def test_existing_operator_file_and_parent_permissions_are_unchanged(native, tmp_path):
    parent = tmp_path / "operator"
    parent.mkdir()
    parent.chmod(0o755)
    path = parent / "key"
    path.write_text(KEY + "\n", "utf-8")
    path.chmod(0o400)
    before = (parent.stat().st_mode, path.stat().st_mode)
    result = relay_bootstrap.controller_environment(
        {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(path)},
        tmp_path,
    )
    assert result["HARNESS_RELAY_KEY"] == KEY
    assert (parent.stat().st_mode, path.stat().st_mode) == before
    path.chmod(0o600)


def test_missing_key_does_not_change_existing_generic_parent(native, tmp_path):
    parent = tmp_path / "shared-parent"
    parent.mkdir()
    parent.chmod(0o755)
    before = parent.stat().st_mode
    relay_bootstrap.controller_environment(
        {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(parent / "key")},
        tmp_path,
    )
    assert parent.stat().st_mode == before


def test_symlink_key_is_rejected_without_reading_or_replacing_target(native, tmp_path):
    target = tmp_path / "operator-key"
    target.write_text(KEY, "utf-8")
    path = tmp_path / "key"
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    with pytest.raises(ValueError, match="unavailable or invalid"):
        relay_bootstrap.controller_environment(
            {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(path)},
            tmp_path,
        )
    assert path.is_symlink() and target.read_text("utf-8") == KEY


def test_failed_publish_removes_staging_and_does_not_create_a_key(
    native,
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "private" / "key"
    monkeypatch.setattr(
        relay_bootstrap.os, "link", Mock(side_effect=OSError("fixture failure"))
    )
    with pytest.raises(OSError, match="fixture failure"):
        relay_bootstrap.controller_environment(
            {"SAM_ANALYSIS_ENABLED": "true", "HARNESS_RELAY_KEY_FILE": str(path)},
            tmp_path,
        )
    assert not path.exists() and not list(path.parent.glob(".relay-key-*"))


def test_explicit_key_takes_precedence_without_touching_bad_file(native, tmp_path):
    path = tmp_path / "key"
    path.write_text("bad", "utf-8")
    result = relay_bootstrap.controller_environment(
        {
            "SAM_ANALYSIS_ENABLED": "true",
            "HARNESS_RELAY_KEY": KEY,
            "HARNESS_RELAY_KEY_FILE": str(path),
        },
        tmp_path,
    )
    assert result["HARNESS_RELAY_KEY"] == KEY and path.read_text("utf-8") == "bad"


def test_controller_bootstrap_precedes_rental_and_key_crosses_only_analysis_whitelist(
    native,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("SAM_ANALYSIS_ENABLED", "true")
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    monkeypatch.setenv("HARNESS_RELAY_KEY_FILE", str(tmp_path / "auth" / "key"))
    monkeypatch.setenv("SAM31_LIVE_CAMERAS", "1")
    monkeypatch.setenv("VAST_API_KEY", "fixture-rental-secret")
    monkeypatch.setenv("HF_TOKEN", "fixture-download-secret")
    monkeypatch.setenv("SAM_SSH_KEY", "fixture-ssh-path")
    spawn = Mock(return_value=SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr("tools.cloud_gpu_controller.subprocess.Popen", spawn)
    controller = CloudController(tmp_path / "settings", 1)
    spawn.assert_not_called()
    assert controller.child_environment["HARNESS_RELAY_KEY"] == KEY
    assert KEY not in json.dumps(controller.status())
    controller.start()
    forwarded = runtime_environment(spawn.call_args.kwargs["env"])
    assert forwarded["HARNESS_RELAY_KEY"] == KEY
    assert (
        not {"VAST_API_KEY", "HF_TOKEN", "SAM_SSH_KEY", "HARNESS_RELAY_KEY_FILE"}
        & forwarded.keys()
    )
    lane = lane_environment(forwarded, tmp_path / "gpu", 1, 1)
    assert lane["HARNESS_RELAY_KEY"] == KEY


@pytest.mark.parametrize("source", ["environment", "env-file"])
def test_standalone_plan_does_not_print_key_or_write_it_into_overlay(
    monkeypatch,
    tmp_path,
    capsys,
    source,
):
    monkeypatch.setattr(start_modular, "ROOT", tmp_path)
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    monkeypatch.delenv("HARNESS_RELAY_KEY_FILE", raising=False)
    filename = tmp_path / "settings.env"
    filename.write_text("HARNESS_RELAY_KEY=" + KEY + "\n", "utf-8")
    args = ["main", "--analysis-server", "http://gpu:8080", "--dry-run"]
    if source == "environment":
        monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    else:
        args += ["--env-file", str(filename)]
    assert start_modular.main(args) == 0
    output = capsys.readouterr()
    assert KEY not in output.out + output.err
    assert "HARNESS_RELAY_KEY" not in json.loads(output.out)["environment"]
    assert not (tmp_path / ".goodvision").exists()


def test_standalone_local_pull_missing_key_stops_before_docker_but_gpu_pull_can_start(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(start_modular, "ROOT", tmp_path)
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    monkeypatch.delenv("HARNESS_RELAY_KEY_FILE", raising=False)
    monkeypatch.delenv("HARNESS_VIDEO_PULL_LOCATION", raising=False)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(start_modular.subprocess, "run", run)
    args = ["main", "--analysis-server", "http://gpu:8080"]
    assert start_modular.main(args) == 2
    run.assert_not_called()
    assert "matching HARNESS_RELAY_KEY" in capsys.readouterr().err
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "gpu")
    assert start_modular.main(args) == 0
    run.assert_called_once()


def test_direct_launcher_bootstraps_after_dotenv_before_any_rental(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(start_sam, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["start_sam.py"])
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setenv("SAM_CONTROL_DIRECTORY", str(tmp_path / "control"))

    def load_fixture(_path, *, override):
        assert override is False
        monkeypatch.setenv("SAM_ANALYSIS_ENABLED", "true")
        monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)

    rental = Mock(side_effect=AssertionError("a test must never create a rental"))
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=load_fixture)
    )
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(VastAI=rental))
    monkeypatch.setattr(start_sam.shutil, "which", lambda _name: "fixture-command")
    bootstrap = Mock(wraps=relay_bootstrap.controller_environment)
    monkeypatch.setattr(start_sam, "controller_environment", bootstrap)
    assert start_sam.main() == 1
    bootstrap.assert_called_once()
    assert bootstrap.call_args.args[0]["HARNESS_RELAY_KEY"] == KEY
    rental.assert_not_called()
    assert not (tmp_path / "control").exists()
    output = capsys.readouterr()
    assert KEY not in output.out + output.err
