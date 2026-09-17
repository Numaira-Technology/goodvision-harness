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
    recovery.create_instance.return_value = {"success": True, "new_contract": 42}
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


def test_sam_only_startup_does_not_upload_enable_or_download_retired_route(recovery):
    assert launcher.main() == 1
    assert launcher.WORKER_ROUTES == ("sam_live",)
    assert not any("hybrid" in name for name in launcher.WORKER_FILES)
    assert not any(name.startswith("HYBRID_") for name in launcher.WORKER_ENV)
    commands = [" ".join(call.args[0]) for call in launcher.run.call_args_list]
    assert not any("hybrid" in command for command in commands)
    assert any("sam3.1_multiplex.pt" in command for command in commands)


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


@pytest.fixture
def rental(recovery, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["start_sam.py"])
    recovery.search_offers.return_value = [
        {"id": 10, "machine_id": 100, "dph_total": 0.3},
        {"id": 11, "machine_id": 100, "dph_total": 0.31},
        {"id": 20, "machine_id": 200, "dph_total": 0.4},
        {"id": 30, "machine_id": 300, "dph_total": 0.5},
        {"id": 40, "machine_id": 400, "dph_total": 0.6},
    ]
    recovery.create_instance.return_value = {"success": True, "new_contract": 42}
    recovery.show_instances.return_value = []
    return recovery


def http_failure(status, payload):
    # Real requests responses are false-y for HTTP errors; exercise that contract.
    class Response:
        status_code = status

        def json(self):
            return payload

        def __bool__(self):
            return False

    failure = RuntimeError("request URL may contain a secret: do not log this")
    failure.response = Response()
    return failure


@pytest.mark.parametrize("stage", ["ssh", "upload", "setup", "weights", "readiness"])
def test_startup_failure_releases_before_renting_a_different_machine(
    rental, capsys, stage,
):
    rental.create_instance.side_effect = [
        {"success": True, "new_contract": 42},
        {"success": True, "new_contract": 43},
    ]
    failed = RuntimeError("simulated remote startup failure")
    if stage == "ssh":
        launcher.wait_for_ssh.side_effect = [failed, ("host", 22)]
    elif stage == "readiness":
        launcher.wait_for_worker.side_effect = [failed, None]
    else:
        failed_call = {"upload": 1, "setup": 2, "weights": 3}[stage]
        launcher.run.side_effect = [None] * failed_call + [failed] + [None] * 4
    events = Mock()
    events.attach_mock(rental.create_instance, "create")
    events.attach_mock(rental.destroy_instance, "destroy")
    assert launcher.main() == 1  # The replacement becomes ready, then exits.
    assert [(call[0], call.kwargs["id"]) for call in events.mock_calls] == [
        ("create", 10), ("destroy", 42), ("create", 20), ("destroy", 43),
    ]
    assert len({call.kwargs["label"] for call in rental.create_instance.call_args_list}) == 2
    assert "geolocation!=CN" in rental.search_offers.call_args.kwargs["query"]
    assert "SAM is ready" in capsys.readouterr().out


def test_startup_failures_stop_after_three_different_machines(rental):
    rental.create_instance.side_effect = [
        {"success": True, "new_contract": number} for number in (42, 43, 44)
    ]
    launcher.wait_for_ssh.side_effect = RuntimeError("SSH unavailable")
    assert launcher.main() == 1
    assert [call.kwargs["id"] for call in rental.create_instance.call_args_list] == [10, 20, 30]
    assert [call.kwargs["id"] for call in rental.destroy_instance.call_args_list] == [42, 43, 44]


def test_refusals_and_startup_failures_share_three_machine_limit(rental):
    rental.create_instance.side_effect = [
        {"success": False, "error": "no_such_ask"},
        {"success": True, "new_contract": 42},
        {"success": True, "new_contract": 43},
    ]
    launcher.wait_for_ssh.side_effect = RuntimeError("SSH unavailable")
    assert launcher.main() == 1
    assert [call.kwargs["id"] for call in rental.create_instance.call_args_list] == [10, 20, 30]
    assert [call.kwargs["id"] for call in rental.destroy_instance.call_args_list] == [42, 43]


@pytest.mark.parametrize("result", [{"success": False}, TimeoutError("release timed out")])
def test_unconfirmed_release_prevents_replacement(rental, result):
    launcher.wait_for_ssh.side_effect = RuntimeError("SSH unavailable")
    if isinstance(result, Exception):
        rental.destroy_instance.side_effect = result
    else:
        rental.destroy_instance.return_value = result
    assert launcher.main() == 1
    rental.create_instance.assert_called_once()
    assert rental.destroy_instance.called
    assert all(call.kwargs["id"] == 42 for call in rental.destroy_instance.call_args_list)


def test_uncertain_replacement_creation_reconciles_only_its_new_label(rental):
    rental.create_instance.side_effect = [
        {"success": True, "new_contract": 42}, TimeoutError("create response lost"),
    ]
    launcher.wait_for_ssh.side_effect = RuntimeError("SSH unavailable")
    rental.show_instances.side_effect = lambda: [
        {"id": 42, "label": rental.create_instance.call_args_list[0].kwargs["label"]},
        {"id": 43, "label": rental.create_instance.call_args.kwargs["label"]},
    ]
    assert launcher.main() == 1
    assert rental.create_instance.call_count == 2
    assert [call.kwargs["id"] for call in rental.destroy_instance.call_args_list] == [42, 43]


@pytest.mark.parametrize("stage", ["ssh", "setup", "readiness"])
def test_user_stop_during_startup_does_not_rent_a_replacement(rental, stage):
    target = {
        "ssh": launcher.wait_for_ssh, "setup": launcher.run,
        "readiness": launcher.wait_for_worker,
    }[stage]
    target.side_effect = KeyboardInterrupt()
    assert launcher.main() == 130
    rental.create_instance.assert_called_once()
    rental.destroy_instance.assert_called_once_with(id=42)


def test_failure_after_readiness_does_not_replace_the_active_session(rental):
    launcher.subprocess.Popen.return_value.returncode = 255
    assert launcher.main() == 1
    launcher.wait_for_worker.assert_called_once()
    rental.create_instance.assert_called_once()
    rental.destroy_instance.assert_called_once_with(id=42)


@pytest.mark.parametrize("refusal", [
    http_failure(400, {"success": False, "error": "no_such_ask"}),
    {"success": False, "message": "Offer 10 is already rented"},
    http_failure(409, {"error": {"code": "offer_unavailable"}}),
])
def test_offer_refusal_tries_next_machine_and_releases_successful_rental(
    rental, refusal,
):
    rental.create_instance.side_effect = [
        refusal, {"success": True, "new_contract": 42},
    ]
    assert launcher.main() == 1  # Worker exits in this fixture.
    ids = [call.kwargs["id"] for call in rental.create_instance.call_args_list]
    assert ids == [10, 20]
    labels = {call.kwargs["label"] for call in rental.create_instance.call_args_list}
    assert len(labels) == 1
    rental.show_instances.assert_not_called()
    rental.destroy_instance.assert_called_once_with(id=42)


def test_unavailable_offers_stop_after_three_machines(rental, capsys):
    rental.create_instance.return_value = {
        "success": False, "error": "offer_unavailable",
    }
    assert launcher.main() == 1
    ids = [call.kwargs["id"] for call in rental.create_instance.call_args_list]
    assert ids == [10, 20, 30]
    rental.show_instances.assert_not_called()
    rental.destroy_instance.assert_not_called()
    output = capsys.readouterr()
    assert "at most 3 machines" in output.err
    assert "unconfirmed" not in output.err
    assert "offer_unavailable" in output.out


def test_offers_with_unknown_machine_identity_are_not_repeated(rental):
    rental.search_offers.return_value = [
        {"id": 10, "dph_total": 0.3}, {"id": 20, "dph_total": 0.4},
    ]
    rental.create_instance.return_value = {"success": False, "error": "no_such_ask"}
    assert launcher.main() == 1
    rental.create_instance.assert_called_once()
    rental.show_instances.assert_not_called()


@pytest.mark.parametrize("refusal,diagnostic", [
    (http_failure(401, {"error": "unauthorized"}), "HTTP 401"),
    (http_failure(403, {"error": "offer_unavailable"}), "HTTP 403"),
    (http_failure(400, {
        "code": "insufficient_credit", "message": "Add funds",
    }), "Add funds"),
    (http_failure(400, {
        "code": "invalid_args", "message": "Offer is unavailable",
    }), "invalid_args"),
    (http_failure(400, {"error": {
        "code": "insufficient_credit", "message": "Offer is unavailable",
    }}), "insufficient_credit"),
    (http_failure(400, {
        "error": "insufficient_credit", "msg": "Machine unavailable due to funds",
    }), "insufficient_credit"),
    (http_failure(400, {
        "error": "unauthorized", "msg": "Machine unavailable for this account",
    }), "unauthorized"),
    (http_failure(400, {
        "error": "invalid_args", "msg": "Offer is unavailable with these parameters",
    }), "invalid_args"),
    (http_failure(400, {"message": "Image not found for offer 10"}), "Image not found"),
    ({"success": False, "error": "invalid_image"}, "invalid_image"),
])
def test_other_explicit_refusals_stop_without_uncertain_creation(
    rental, capsys, refusal, diagnostic,
):
    rental.create_instance.side_effect = [refusal]
    assert launcher.main() == 1
    rental.create_instance.assert_called_once()
    rental.show_instances.assert_not_called()
    rental.destroy_instance.assert_not_called()
    output = capsys.readouterr().err
    assert diagnostic in output
    assert "No instance created" in output
    assert "unconfirmed" not in output


@pytest.mark.parametrize("failure", [
    TimeoutError("unknown create result"),
    http_failure(408, {"error": "offer_unavailable"}),
    http_failure(503, {"success": False, "error": "offer_unavailable"}),
    {"success": True},
    {"new_contract": 42},
    {"success": True, "new_contract": True},
    {"success": True, "new_contract": -42},
    {"success": False, "new_contract": 42, "error": "offer_unavailable"},
])
def test_unknown_creation_reconciles_label_without_another_rental(
    rental, capsys, failure,
):
    rental.create_instance.side_effect = [failure]
    assert launcher.main() == 1
    rental.create_instance.assert_called_once()
    rental.show_instances.assert_called_once()
    rental.destroy_instance.assert_not_called()
    output = capsys.readouterr().err
    assert "Creation outcome unknown" in output
    assert "Creation remains unconfirmed" in output


def test_lost_creation_response_releases_only_matching_label(rental, capsys):
    rental.create_instance.side_effect = TimeoutError("lost response")
    rental.show_instances.side_effect = lambda: [
        {"id": 99, "label": "unrelated-instance"},
        {"id": 42, "label": rental.create_instance.call_args.kwargs["label"]},
    ]
    assert launcher.main() == 1
    rental.create_instance.assert_called_once()
    rental.destroy_instance.assert_called_once_with(id=42)
    assert "Creation remains unconfirmed" not in capsys.readouterr().err


def test_failed_reconciliation_does_not_rent_again_or_expose_credentials(
    rental, monkeypatch, capsys,
):
    monkeypatch.setenv("VAST_API_KEY", "test-api-secret")
    rental.create_instance.side_effect = TimeoutError("lost response")
    rental.show_instances.side_effect = RuntimeError(
        "HTTP 503: https://example.invalid/instances?api_key=test-api-secret",
    )
    assert launcher.main() == 1
    rental.create_instance.assert_called_once()
    rental.destroy_instance.assert_not_called()
    output = capsys.readouterr().err
    assert "Could not look up" in output
    assert "Creation remains unconfirmed" in output
    assert "test-api-secret" not in output


def test_provider_diagnostics_redact_credentials_and_omit_other_fields(
    rental, monkeypatch, capsys,
):
    monkeypatch.setenv("VAST_API_KEY", "vast-secret-value")
    monkeypatch.setenv("HF_TOKEN", "hf_secretvalue")
    rental.create_instance.side_effect = http_failure(400, {
        "error": {
            "code": "invalid_args",
            "message": (
                "Rejected vast-secret-value hf_secretvalue; "
                "Authorization: Bearer other-token; api_key=unconfigured-secret; "
                "-----BEGIN OPENSSH PRIVATE KEY-----\nprivate-key-data\n"
                "-----END OPENSSH PRIVATE KEY-----"
            ),
        },
        "request": {"api_key": "must-not-appear"},
        "access_token": "also-must-not-appear",
    })
    assert launcher.main() == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    for secret in (
        "vast-secret-value", "hf_secretvalue", "other-token", "unconfigured-secret",
        "private-key-data", "must-not-appear", "also-must-not-appear",
    ):
        assert secret not in output
    assert "invalid_args" in output
    assert "HTTP 400" in output
    assert "[redacted]" in output
