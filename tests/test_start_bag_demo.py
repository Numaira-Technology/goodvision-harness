"""Exercise the real launcher orchestration with an entirely fake Vast/SSH boundary."""

import io
import json
import signal
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tools import start_bag_demo as launch


@pytest.fixture
def rig(tmp_path, monkeypatch):
    key = tmp_path / "synthetic-key"
    key.write_text("not a real key")
    monkeypatch.setenv("VAST_API_KEY", "synthetic-key")
    monkeypatch.setenv("SAM_SSH_KEY", str(key))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(launch.socket, "socket", MagicMock())
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/synthetic/" + name)
    vast = Mock()
    vast.search_offers.return_value = [{"id": 1, "dph_total": 0.1}]
    vast.show_instances.return_value = []
    monkeypatch.setitem(
        sys.modules, "vastai", SimpleNamespace(VastAI=Mock(return_value=vast))
    )
    monkeypatch.setattr(launch.S, "prepare_ssh_key", Mock(return_value=key))
    monkeypatch.setattr(launch.S, "create_from_offers", Mock(return_value=42))
    monkeypatch.setattr(
        launch.S, "wait_for_ssh", Mock(return_value=("synthetic.invalid", 22))
    )
    monkeypatch.setattr(launch.S, "run", Mock())
    monkeypatch.setattr(launch.S, "release_instance", Mock())
    monkeypatch.setattr(launch.S, "stop_process", Mock())
    for name in ("_attach_key", "_upload_repo", "_prefetch_from_r2", "_wait_ready"):
        monkeypatch.setattr(launch, name, Mock())
    worker = Mock()
    worker.wait.side_effect = KeyboardInterrupt
    monkeypatch.setattr(launch, "_server_process", Mock(return_value=worker))
    status = tmp_path / "status.json"
    return SimpleNamespace(
        vast=vast, worker=worker, status=status, args=["--status-file", str(status)]
    )


def test_interrupt_releases_owned_instance_and_clears_ready_without_hf_token(rig):
    previous = signal.getsignal(signal.SIGINT)
    previous_image = launch.S.IMAGE
    assert launch.main(rig.args) == 130
    launch.S.release_instance.assert_called_once_with(rig.vast, 42)
    launch.S.stop_process.assert_called_once_with(rig.worker)
    launch._prefetch_from_r2.assert_called_once()
    assert launch._prefetch_from_r2.call_args.args[1] is None
    status = json.loads(rig.status.read_text())
    assert status["state"] == "stopped" and status["cleanup_confirmed"] is True
    assert status["instance_id"] == 42
    assert signal.getsignal(signal.SIGINT) == previous
    assert launch.S.IMAGE == previous_image


@pytest.mark.parametrize(
    "phase", ["_upload_repo", "_prefetch_from_r2", "_server_process", "_wait_ready"]
)
def test_every_startup_failure_releases_instance(rig, monkeypatch, phase):
    monkeypatch.setattr(
        launch, phase, Mock(side_effect=RuntimeError("synthetic-secret"))
    )
    assert launch.main(rig.args) == 1
    launch.S.release_instance.assert_called_once_with(rig.vast, 42)
    assert json.loads(rig.status.read_text())["cleanup_confirmed"] is True


def test_lost_create_response_reconciles_only_unique_label(rig):
    label = []

    def create(*args, **kwargs):
        label.append(kwargs["label"])
        raise TimeoutError("creation response lost")

    launch.S.create_from_offers.side_effect = create
    rig.vast.show_instances.side_effect = lambda: [
        {"id": 88, "label": "unrelated"},
        {"id": 42, "label": label[0]},
    ]
    assert launch.main(rig.args) == 1
    launch.S.release_instance.assert_called_once_with(rig.vast, 42)
    assert json.loads(rig.status.read_text())["cleanup_confirmed"] is True


@pytest.mark.parametrize("lookup", ["none", "failure", "ambiguous"])
def test_unconfirmed_creation_is_visible_with_recovery_label(rig, capsys, lookup):
    label = []

    def create(*args, **kwargs):
        label.append(kwargs["label"])
        raise TimeoutError("synthetic-secret")

    launch.S.create_from_offers.side_effect = create
    if lookup == "failure":
        rig.vast.show_instances.side_effect = RuntimeError("synthetic-secret")
    if lookup == "ambiguous":
        rig.vast.show_instances.side_effect = lambda: [
            {"id": 41, "label": label[0]},
            {"id": 42, "label": label[0]},
        ]
    assert launch.main(rig.args) == 1
    launch.S.release_instance.assert_not_called()
    status = json.loads(rig.status.read_text())
    assert status["state"] == "cleanup_required"
    assert status["cleanup_confirmed"] is False
    output = capsys.readouterr()
    assert label[0] in output.err and "unconfirmed" in output.err
    assert "synthetic-secret" not in output.err


def test_release_failure_remains_explicit_after_interrupt(rig, capsys):
    launch.S.release_instance.side_effect = RuntimeError("synthetic-secret")
    assert launch.main(rig.args) == 1
    assert json.loads(rig.status.read_text())["state"] == "cleanup_required"
    assert "RELEASE FAILED for instance 42" in capsys.readouterr().err


def test_worker_stop_failure_does_not_skip_release(rig):
    launch.S.stop_process.side_effect = RuntimeError("synthetic cleanup failure")
    assert launch.main(rig.args) == 130
    launch.S.release_instance.assert_called_once_with(rig.vast, 42)


def test_invalid_resume_never_destroys_other_instance(rig):
    rig.vast.show_instance.return_value = {"id": 77, "label": "someone-elses-worker"}
    assert launch.main([*rig.args, "--resume-instance-id", "77"]) == 1
    launch.S.release_instance.assert_not_called()
    launch.S.create_from_offers.assert_not_called()


def test_owned_resume_retains_original_label_and_releases(rig):
    rig.vast.show_instance.return_value = {
        "id": 77,
        "label": "goodvision-bagdemo-existing",
    }
    assert launch.main([*rig.args, "--resume-instance-id", "77"]) == 130
    launch.S.release_instance.assert_called_once_with(rig.vast, 77)
    launch.S.create_from_offers.assert_not_called()
    assert json.loads(rig.status.read_text())["label"] == "goodvision-bagdemo-existing"


def test_status_failure_precedes_rental(rig):
    assert launch.main(["--status-file", str(rig.status / "missing-parent")]) == 1
    launch.S.create_from_offers.assert_not_called()
    rig.vast.search_offers.assert_not_called()


@pytest.mark.parametrize("args", [["--disk-gb", "0"], ["--resume-instance-id", "0"]])
def test_invalid_rental_arguments_stop_before_client(rig, args):
    with pytest.raises(SystemExit):
        launch.main(args)
    rig.vast.search_offers.assert_not_called()


def test_prefetch_credentials_only_over_stdin(monkeypatch):
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "synthetic-secret-$()")
    run = Mock()
    monkeypatch.setattr(launch.S, "run", run)
    launch._prefetch_from_r2(["ssh", "synthetic.invalid"], "synthetic-hf-token")
    args, kwargs = run.call_args
    assert "synthetic-secret" not in repr(args)
    assert "synthetic-hf-token" not in repr(args)
    assert (
        "synthetic-secret" in kwargs["input"]
        and "synthetic-hf-token" in kwargs["input"]
    )


def test_upload_uses_tracked_sources_and_interrupt_safe_runner(monkeypatch):
    run = Mock()
    monkeypatch.setattr(launch.S, "run", run)
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout="src/module.py\0tools/script.py\0")),
    )
    launch._upload_repo(["ssh", "-p", "22", "root@synthetic.invalid"])
    args, kwargs = run.call_args
    assert "--files-from=-" in args[0] and "--no-links" in args[0]
    assert kwargs["input"] == "src/module.py\0tools/script.py\0"


def test_ready_rejects_wrong_scorer(monkeypatch):
    opener = Mock()
    opener.open.return_value = io.BytesIO(
        json.dumps(
            {"ready": True, "model": "wrong", "revision": launch.REVISION}
        ).encode()
    )
    monkeypatch.setattr(launch, "build_opener", lambda *a: opener)
    worker = Mock()
    worker.poll.return_value = None
    with pytest.raises(RuntimeError, match="unexpected scorer"):
        launch._wait_ready(worker)
