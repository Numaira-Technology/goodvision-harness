"""Exercise the real launcher orchestration with an entirely fake Vast/SSH boundary."""

import io
import json
import signal
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tools import start_bag_demo as launch
from tools import vast_run_log as run_log


@pytest.fixture
def rig(tmp_path, monkeypatch):
    real_attach, real_finish = run_log.attach, run_log.finish
    monkeypatch.setattr(run_log, "attach", Mock())
    monkeypatch.setattr(run_log, "finish", Mock())
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
        vast=vast,
        worker=worker,
        status=status,
        args=["--status-file", str(status)],
        attach=real_attach,
        finish=real_finish,
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


@pytest.mark.parametrize("failure,code", [(TimeoutError, 1), (KeyboardInterrupt, 130)])
def test_lost_create_response_reconciles_only_unique_label(rig, failure, code):
    label = []

    def create(*args, **kwargs):
        label.append(kwargs["label"])
        raise failure("creation response lost")

    launch.S.create_from_offers.side_effect = create
    rig.vast.show_instances.side_effect = lambda: [
        {"id": 88, "label": "unrelated"},
        {"id": 42, "label": label[0]},
    ]
    assert launch.main(rig.args) == code
    launch.S.release_instance.assert_called_once_with(rig.vast, 42)
    assert json.loads(rig.status.read_text())["cleanup_confirmed"] is True


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt])
@pytest.mark.parametrize("lookup", ["none", "failure", "ambiguous"])
def test_unconfirmed_creation_is_visible_with_recovery_label(
    rig, capsys, lookup, failure
):
    label = []

    def create(*args, **kwargs):
        label.append(kwargs["label"])
        raise failure("synthetic-secret")

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


@pytest.mark.parametrize(
    "outcome", ["interrupt", "startup_failure", "release_failure", "no_offer"]
)
def test_recorder_tracks_bag_demo_lifecycle(rig, tmp_path, monkeypatch, outcome):
    """Run the actual launcher and recorder with fake paid/SSH boundaries."""
    monkeypatch.setattr(run_log, "attach", rig.attach)
    monkeypatch.setattr(run_log, "finish", rig.finish)
    monkeypatch.setattr(run_log, "_SESSION", None)
    monkeypatch.setattr(run_log.atexit, "register", lambda fn: None)
    monkeypatch.setattr(sys, "argv", ["start_bag_demo.py", *rig.args])
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    actions = []
    rig.vast.create_instance.return_value = {"success": True, "new_contract": 42}
    rig.vast.search_offers.return_value = [{"id": 1, "machine_id": 7, "dph_total": 0.1}]

    def create(vast, offers, **kwargs):
        return vast.create_instance(offers[0]["id"], **kwargs)["new_contract"]

    def release(vast, instance_id):
        actions.append("release")
        if outcome == "release_failure":
            raise RuntimeError("synthetic release failure")

    def ready(worker):
        actions.append("ready")

    def run(command, **kwargs):
        actions.append("ssh")

    monkeypatch.setattr(launch.S, "create_from_offers", create)
    monkeypatch.setattr(launch.S, "release_instance", release)
    monkeypatch.setattr(launch.S, "run", run)
    monkeypatch.setattr(launch, "_wait_ready", ready)
    monkeypatch.setattr(run_log.RunLog, "arm_stage_hooks", lambda *args: None)
    monkeypatch.setattr(run_log.RunLog, "maybe_probe", lambda *args: None)
    monkeypatch.setattr(
        run_log.RunLog, "start_sampler", lambda self: actions.append("sample_start")
    )
    monkeypatch.setattr(
        run_log.RunLog, "stop_sampler", lambda self: actions.append("sample_stop")
    )
    temporary_directory = launch.tempfile.TemporaryDirectory

    def temporary_keys(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        cleanup = directory.cleanup

        def clean():
            actions.append("keys_removed")
            cleanup()

        directory.cleanup = clean
        return directory

    monkeypatch.setattr(launch.tempfile, "TemporaryDirectory", temporary_keys)
    if outcome == "startup_failure":
        launch._upload_repo.side_effect = RuntimeError("synthetic startup failure")
    elif outcome == "no_offer":
        rig.vast.search_offers.return_value = []
    assert launch.main(rig.args) == (130 if outcome == "interrupt" else 1)
    session = run_log._SESSION
    assert session is not None
    assert session._finished
    assert session.sample and session._expects_ready
    entry = json.loads((session.run_dir / "entry.json").read_text())
    manifest = json.loads((session.root / "manifest.json").read_text())
    assert entry["launcher"] == "tools.start_bag_demo"
    assert entry["model"] == "bag_demo_videojev"
    assert entry["score"] == "unknown"
    assert actions[-2:] == ["sample_stop", "keys_removed"]
    if outcome == "no_offer":
        rig.vast.create_instance.assert_not_called()
        assert manifest[0]["status"] == "not_rented"
        assert "release" not in actions
    else:
        assert manifest[0]["status"] == "rented"
        assert entry["instance_id"] == 42
        assert entry["hardware"]["offer"]["machine_id"] == 7
        assert actions[actions.index("release") - 1] == "sample_stop"
        assert entry["release_status"] == (
            "failed" if outcome == "release_failure" else "released"
        )
    if outcome in {"interrupt", "release_failure"}:
        assert actions.index("ready") < actions.index("sample_start")
        assert actions.index("sample_start") < actions.index("release")
    else:
        assert "sample_start" not in actions
