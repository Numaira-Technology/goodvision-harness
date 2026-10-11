"""Exercise the shipped cloud package without API calls or an SSH server."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tools.vlm_benchmark import cloud_modes, run_remote

ROOT = Path(__file__).resolve().parents[1]


def plan(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote",
            "--suite",
            "modes",
            "--plan-only",
            *extra,
        ],
    )
    # Planning must not connect or read a credential, even for --real.
    monkeypatch.setattr(
        run_remote.SSH, "connect", lambda _: pytest.fail("SSH during planning")
    )
    monkeypatch.setattr(
        run_remote.getpass, "getpass", lambda _: pytest.fail("secret during planning")
    )
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        assert run_remote.main() == 0
    return json.loads(output.getvalue().split("\n", 3)[3])


def test_modes_plan_preserves_model_and_recording_without_paid_default(
    tmp_path, monkeypatch
):
    result = plan(tmp_path, monkeypatch)
    assert result["simulate"]
    settings = result["settings"]
    assert settings["model"] == "qwen/qwen3-vl-30b-a3b-instruct"
    assert settings["modes"] == ["events", "compact", "parallel"]
    assert settings["streams"] == [1, 4, 8, 12]
    assert settings["global_limit"] == 48
    assert settings["per_stream_limit"] == 4
    assert settings["record"] and settings["monitor"]
    assert settings["monitor_port"] == 8787
    assert settings["soak_seconds"] == 0
    paid = plan(tmp_path, monkeypatch, "--real", "--global-limit", "16")
    assert not paid["simulate"] and paid["settings"]["real"]
    assert paid["settings"]["global_limit"] == 16


def test_dashscope_cloud_plan_keeps_endpoint_and_recording(tmp_path, monkeypatch):
    endpoint = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    doc = plan(tmp_path, monkeypatch, "--provider", "dashscope", "--endpoint", endpoint)
    assert doc["settings"]["backend"] == "dashscope"
    assert doc["settings"]["endpoint"] == endpoint
    assert doc["settings"]["model"] == "qwen3-vl-30b-a3b-instruct"
    assert doc["settings"]["record"]
    assert "dashscope" in doc["execution"]


def test_cloud_plan_preserves_grounding_sweep(tmp_path, monkeypatch):
    doc = plan(tmp_path, monkeypatch, "--modes", "grounding", "--streams", "12",
               "--grounding-format", "lean-v1", "--grounding-concurrency-sweep",
               "4", "6", "8", "--duration", "55.798021", "--warmup-seconds", "10",
               "--http-timeout", "20", "--no-record")
    assert doc["settings"]["grounding_concurrency_sweep"] == [4, 6, 8]
    assert doc["settings"]["http_timeout"] == 20
    assert doc["request_budget"]["total_upper"] == 5052


@pytest.mark.parametrize("backend", ["openrouter", "dashscope"])
def test_cloud_worker_installs_only_selected_key(tmp_path, monkeypatch, backend):
    from tools.vlm_benchmark import cloud, lock, modes
    from tools.vlm_benchmark.routes import KEY_ENV, mode_api_key

    doc = plan(tmp_path, monkeypatch, "--provider", backend, "--real", "--no-record")
    root = tmp_path / "app"
    root.mkdir()
    secret = tmp_path / "runtime-key.json"
    secret.write_text('{"key":"selected-test-only"}', encoding="utf-8")
    monkeypatch.setattr(cloud, "verify_bundle", lambda _: {"videos": ["video.mp4"]})
    monkeypatch.setattr(lock, "acquire", lambda _: None)
    for name in ("INFERENCE_API_KEY", "MODEL_API_KEY", *KEY_ENV.values()):
        monkeypatch.setenv(name, "stale-key")
    seen = []

    def capture(args):
        assert args.backend == backend
        assert mode_api_key(modes.configure(args).model_api) == "selected-test-only"
        assert os.environ.get(KEY_ENV[backend]) == "selected-test-only"
        assert not secret.exists()
        assert (
            "INFERENCE_API_KEY" not in os.environ and "MODEL_API_KEY" not in os.environ
        )
        assert not any(os.environ.get(v) for k, v in KEY_ENV.items() if k != backend)
        seen.append(args)

    monkeypatch.setattr(modes, "run", capture)
    assert cloud_modes.run(root, doc, tmp_path / "results", secret) == 0
    assert len(seen) == 1
    assert not any(os.environ.get(name) for name in KEY_ENV.values())
    assert "selected-test-only" not in (tmp_path / "status.json").read_text()


def test_recording_plan_preserves_lean_grounding_and_provider(tmp_path, monkeypatch):
    result = plan(
        tmp_path,
        monkeypatch,
        "--streams",
        "12",
        "--repeats",
        "1",
        "--modes",
        "events",
        "grounding",
        "parallel",
        "--grounding-format",
        "lean-v1",
        "--provider-only",
        "deepinfra",
        "--http-timeout",
        "20",
    )
    settings = result["settings"]
    assert settings["modes"] == ["events", "compact", "parallel"]
    assert settings["record"] and settings["grounding_format"] == "lean-v1"
    assert settings["provider_only"] == "deepinfra" and settings["http_timeout"] == 20


@pytest.mark.skipif(os.name == "nt", reason="cloud POSIX worker")
@pytest.mark.parametrize("backend", ["openrouter", "dashscope"])
@pytest.mark.parametrize("record", [False, True])
def test_uploaded_mode_suite_runs_and_collects_recordings(
    tmp_path, monkeypatch, record, backend
):
    if record and not os.environ.get("VLM_MONITOR_BROWSER_TESTS"):
        pytest.skip("opt-in browser integration")
    import av
    from PIL import Image

    video = tmp_path / "test input.mp4"
    with av.open(str(video), "w") as writer:
        stream = writer.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 90, "yuv420p"
        for index in range(40):
            frame = av.VideoFrame.from_image(
                Image.new("RGB", (160, 90), (index * 5, 80, 90))
            )
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    doc = plan(
        tmp_path,
        monkeypatch,
        "--streams",
        "1",
        "--provider",
        backend,
        "--duration",
        "3",
        "--repeats",
        "1",
        "--modes",
        "events",
        "--monitor-port",
        "0",
        *([] if record else ["--no-record"]),
    )
    job = tmp_path / "job"
    job.mkdir()
    run_remote.bundle(ROOT, [video], job / "bundle.tar.gz")
    app = job / "app"
    run_remote.safe_extract(job / "bundle.tar.gz", app)
    (job / "plan.json").write_text(json.dumps(doc))
    secret = job / "runtime-key.json"
    secret.write_text('{"key":"must-not-leak"}')
    env = {
        **os.environ,
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONPATH": str(app / "src") + os.pathsep + str(app),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.vlm_benchmark.cloud_modes",
            "--root",
            str(app),
            "--plan",
            str(job / "plan.json"),
            "--output",
            str(job / "results"),
        ],
        cwd=app,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((job / "results/summary.json").read_text())
    assert summary["state"] == "completed"
    assert summary["settings"]["video"] == [str(app / "media/video-01.mp4")]
    assert len(summary["jobs"]) == 3  # preflight, warmup, measurement
    assert not secret.exists()
    assert json.loads((job / "status.json").read_text())["state"] == "completed"
    subprocess.run([sys.executable, "-c", run_remote.COLLECT, str(job)], check=True)
    with tarfile.open(job / "results.tar.gz") as archive:
        names = archive.getnames()
        assert "results/report.html" in names
        assert not any("runtime-key" in name or ".env" in name for name in names)
        for member in archive.getmembers():
            if member.name.endswith((".json", ".jsonl", ".html", ".log")):
                assert b"must-not-leak" not in archive.extractfile(member).read()
        if record:
            recording = summary["jobs"][-1]["recording"]
            assert recording["complete"]
            assert "results/1-streams-events-repeat-1/recording/viewport.mp4" in names
            assert "results/1-streams-events-repeat-1/recording/final.png" in names
    downloaded = tmp_path / "downloaded"
    run_remote.safe_extract(job / "results.tar.gz", downloaded)
    assert (downloaded / "results/report.html").is_file()


def test_crash_keeps_mode_results_and_removes_secret(tmp_path, monkeypatch):
    doc = plan(tmp_path, monkeypatch)
    job = tmp_path / "job"
    job.mkdir()
    (job / "plan.json").write_text(json.dumps(doc))
    (job / "control.json").write_text(json.dumps({"pid": 999999999}))
    secret = job / "runtime-key.json"
    secret.write_text("do-not-include")
    # This helper must work under the system Python before model/browser install.
    result = subprocess.run(
        [sys.executable, "-c", run_remote.STATUS, str(job)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"state": "failed"' in result.stdout
    summary = json.loads((job / "results/summary.json").read_text())
    assert summary["settings"]["modes"] == doc["settings"]["modes"]
    assert summary["state"] == "failed"
    assert not secret.exists()


def test_failed_worker_removes_secret_and_does_not_start_measurement(
    tmp_path, monkeypatch
):
    doc = plan(tmp_path, monkeypatch, "--real")
    root = tmp_path / "run/app"
    root.mkdir(parents=True)
    # Invalid bundle must fail closed before any credential use/model request.
    (root / "bundle-manifest.json").write_text(
        '{"files":{"missing":"bad"},"videos":[]}'
    )
    secret = root.parent / "runtime-key.json"
    secret.write_text('{"key":"must-not-leak"}')
    assert cloud_modes.run(root, doc, root.parent / "results", secret) == 2
    assert not secret.exists()
    summary = json.loads((root.parent / "results/summary.json").read_text())
    assert summary["jobs"] == []
    assert "must-not-leak" not in json.dumps(summary)


def test_recording_preflight_failure_prevents_paid_calls(tmp_path, monkeypatch):
    from tools.vlm_benchmark import cloud, modes

    doc = plan(tmp_path, monkeypatch, "--real", "--streams", "12")
    root = tmp_path / "run/app"
    root.mkdir(parents=True)
    secret = root.parent / "runtime-key.json"
    secret.write_text('{"key":"must-not-leak"}')
    monkeypatch.setattr(cloud, "verify_bundle", lambda _: {"videos": ["clip.mp4"]})
    calls = []

    def recording_failure(args):
        calls.append(args)
        assert args.real is False and args.record and args.streams == [12]
        assert args.modes == ["events", "compact", "parallel"]
        raise RuntimeError("recording incomplete")

    monkeypatch.setattr(modes, "run", recording_failure)
    assert cloud_modes.run(root, doc, root.parent / "results", secret) == 2
    assert len(calls) == 1 and not secret.exists()


@pytest.mark.skipif(os.name == "nt", reason="Linux bootstrap and POSIX lock")
def test_bootstrap_installs_browser_and_starts_modes_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.vlm_benchmark import bootstrap, cloud

    doc = plan(tmp_path, monkeypatch, "--real")
    root = tmp_path / "job/app"
    root.mkdir(parents=True)
    (root.parent / "plan.json").write_text(json.dumps(doc))
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--root", str(root)])
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        bootstrap.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3)
    )
    monkeypatch.setattr(cloud, "verify_bundle", lambda _: {})
    monkeypatch.setattr(bootstrap.os, "chdir", lambda _: None)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0),
    )

    def execute(_python, command):
        calls.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(bootstrap.os, "execv", execute)
    with pytest.raises(SystemExit):
        bootstrap.main()
    assert any("playwright" in call and "--with-deps" in call for call in calls)
    assert any("fonts-noto-cjk" in call for call in calls)
    assert "tools.vlm_benchmark.cloud_modes" in calls[-1]
    assert calls[-1][-2:] == ["--secret", str(root.parent / "runtime-key.json")]


@pytest.mark.parametrize("dns", ["system", "tcp"])
def test_managed_launch_is_one_shot_and_controls_only_its_own_service(
    tmp_path, monkeypatch, dns
):
    config_dir = tmp_path / "conf.d"
    config_dir.mkdir()
    run = tmp_path / "20261008T190000Z-abcdef01"
    (run / "app").mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"dns_transport": dns}))
    monkeypatch.setattr(
        cloud_modes,
        "Path",
        lambda path: (
            config_dir if str(path) == "/etc/supervisor/conf.d" else Path(path)
        ),
    )
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append(command))
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kw: "1234\n")
    result = cloud_modes.launch_managed(run)
    name = "gv-vlm-" + run.name
    assert result == {"pid": 1234, "root": str(run / "app"), "service": name}
    assert calls == [
        ["supervisorctl", "reread"],
        ["supervisorctl", "update", name],
        ["supervisorctl", "start", name],
    ]
    config = (config_dir / (name + ".conf")).read_text()
    assert "autorestart=false" in config and "autostart=false" in config
    assert "startretries=0" in config
    assert "stopasgroup=true" in config and "killasgroup=true" in config
    assert "tools.vlm_benchmark.bootstrap" in config
    assert ('environment=RES_OPTIONS="use-vc"' in config) == (dns == "tcp")


@pytest.mark.parametrize("same_process", [True, False])
def test_status_uses_process_identity_during_exec_transition(
    tmp_path, monkeypatch, same_process
):
    doc = plan(tmp_path, monkeypatch)
    job = tmp_path / "job"
    job.mkdir()
    (job / "plan.json").write_text(json.dumps(doc))
    (job / "control.json").write_text(
        json.dumps({"pid": 1234, "process_start_ticks": "123"})
    )
    (job / "status.json").write_text('{"state":"installing"}')
    secret = job / "runtime-key.json"
    secret.write_text("test-only")
    stat = "1234 (python) " + " ".join(
        ["S"] + ["0"] * 18 + ["123" if same_process else "456"]
    )

    original_text, original_bytes = Path.read_text, Path.read_bytes

    def read_text(path, *args, **kwargs):
        if str(path).replace("\\", "/") == "/proc/1234/stat":
            return stat
        return original_text(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        if str(path).replace("\\", "/") == "/proc/1234/cmdline":
            return b""
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", ["status", str(job)])
    # Replacing pathlib.Path itself breaks Python 3.11's constructor dispatch,
    # including pytest's own error reporting. Mock only the two proc reads.
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    exec(run_remote.STATUS, {})
    assert secret.exists() is same_process
    assert json.loads((job / "status.json").read_text())["state"] == (
        "installing" if same_process else "failed"
    )


def execute_remote(code, run, monkeypatch):
    """Execute the exact SSH payload without an SSH server or paid requests."""
    monkeypatch.setattr(sys, "argv", ["remote", str(run)])
    old_mask = os.umask(0o077)
    try:
        exec(code, {})
    finally:
        os.umask(old_mask)


@pytest.mark.parametrize("failure_at", ["config", "reread", "update", "start", "pid"])
def test_partial_managed_launch_can_be_recovered_and_stopped(
    tmp_path, monkeypatch, capsys, failure_at
):
    import io
    from types import SimpleNamespace

    doc = plan(tmp_path, monkeypatch, "--real")
    run = tmp_path.resolve() / "20261008T190000Z-abcdef01"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps(doc))
    with tarfile.open(run / "bundle.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("stub.txt")
        entry.size = 4
        archive.addfile(entry, io.BytesIO(b"test"))
    config_dir = tmp_path / "conf.d"
    if failure_at != "config":
        config_dir.mkdir()
    monkeypatch.setattr(
        cloud_modes,
        "Path",
        lambda value: (
            config_dir if str(value) == "/etc/supervisor/conf.d" else Path(value)
        ),
    )
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/supervisorctl")
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"key":"test-key-never-print"}'))
    active = False
    calls = []

    def supervisor(command, **kwargs):
        nonlocal active
        calls.append(command)
        operation = command[1]
        if operation == "start":
            active = True  # A start RPC may take effect before its client times out.
        if operation == failure_at:
            raise subprocess.TimeoutExpired(command, 45)
        if operation == "stop":
            active = False
        return SimpleNamespace(returncode=0, stdout="")

    def pid(command, **kwargs):
        calls.append(command)
        if failure_at == "pid":
            raise subprocess.TimeoutExpired(command, 15)
        return "1234\n"

    monkeypatch.setattr(subprocess, "run", supervisor)
    monkeypatch.setattr(subprocess, "check_output", pid)
    with pytest.raises(SystemExit) as error:
        execute_remote(run_remote.LAUNCH, run, monkeypatch)
    assert error.value.code == 2
    assert not active
    assert not (run / "runtime-key.json").exists()
    control = json.loads((run / "control.json").read_text())
    assert control["service"] == "gv-vlm-" + run.name
    assert control["pid"] is None
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "failed" and not status["cleanup_required"]
    assert json.loads((run / "results/summary.json").read_text())["state"] == "failed"
    assert (run / "results/report.html").is_file()
    assert ["supervisorctl", "stop", control["service"]] in calls
    # A late worker status update must not turn a failed launch into success.
    (run / "status.json").write_text('{"state":"completed"}')
    with pytest.raises(SystemExit) as error:
        execute_remote(run_remote.STATUS, run, monkeypatch)
    assert error.value.code == 0
    response = capsys.readouterr().out
    assert '"state": "failed"' in response
    assert "test-key-never-print" not in response
    execute_remote(run_remote.STOP, run, monkeypatch)
    assert not json.loads((run / "launch-failure.json").read_text())["cleanup_required"]
    execute_remote(run_remote.COLLECT, run, monkeypatch)
    with tarfile.open(run / "results.tar.gz") as archive:
        assert "launch-failure.json" in archive.getnames()
        assert "runtime-key.json" not in archive.getnames()


def test_failed_cleanup_is_explicit_and_stop_can_retry(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from tools.vlm_benchmark.lifecycle import record_launch_failure

    doc = plan(tmp_path, monkeypatch)
    run = tmp_path / "20261008T190000Z-abcdef01"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps(doc))
    control = {"service": "gv-vlm-" + run.name, "pid": None}
    (run / "control.json").write_text(json.dumps(control))
    (run / "runtime-key.json").write_text('{"key":"test-only"}')

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 45)

    monkeypatch.setattr(subprocess, "run", unavailable)
    data = record_launch_failure(run, control, "launch_failed:TimeoutExpired")
    assert data["cleanup_required"]
    assert not (run / "runtime-key.json").exists()
    ssh = SimpleNamespace(call=lambda *args: "VLM_STATUS=" + json.dumps(data))
    with pytest.raises(RuntimeError, match="cloud cleanup required"):
        run_remote.watch(ssh, str(run))
    assert "stop is not confirmed" in capsys.readouterr().err
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    execute_remote(run_remote.STOP, run, monkeypatch)
    assert not json.loads((run / "launch-failure.json").read_text())["cleanup_required"]


@pytest.mark.parametrize(
    "message", ["not running", "no such process", "connection refused"]
)
def test_legacy_launch_without_control_recovers_conservatively(
    tmp_path, monkeypatch, message
):
    from types import SimpleNamespace

    doc = plan(tmp_path, monkeypatch)
    run = tmp_path / "20261008T190000Z-abcdef01"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps(doc))
    (run / "runtime-key.json").write_text('{"key":"test-only"}')
    service = "gv-vlm-" + run.name
    calls = []

    def supervisor(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=2, stdout=f"{service}: ERROR ({message})\n")

    monkeypatch.setattr(subprocess, "run", supervisor)
    with pytest.raises(SystemExit) as error:
        execute_remote(run_remote.STATUS, run, monkeypatch)
    assert error.value.code == 0
    status = json.loads((run / "launch-failure.json").read_text())
    assert status["cleanup_required"] is (message == "connection refused")
    assert calls == [["supervisorctl", "stop", service]]
    assert not (run / "runtime-key.json").exists()
    assert (run / "results/report.html").is_file()


def test_stop_rejects_a_different_managed_service(tmp_path, monkeypatch):
    from tools.vlm_benchmark.lifecycle import stop_control

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("unrelated stop")
    )
    with pytest.raises(RuntimeError, match="unexpected managed service"):
        stop_control(tmp_path, {"service": "another-job"})


def test_stop_during_registration_prevents_later_start(tmp_path, monkeypatch):
    config_dir = tmp_path / "conf.d"
    config_dir.mkdir()
    run = tmp_path / "20261008T190000Z-abcdef01"
    (run / "app").mkdir(parents=True)
    monkeypatch.setattr(
        cloud_modes,
        "Path",
        lambda value: (
            config_dir if str(value) == "/etc/supervisor/conf.d" else Path(value)
        ),
    )
    calls = []

    def registering(command, **kwargs):
        calls.append(command)
        (run / "launch-failure.json").write_text('{"cleanup_required":true}')

    monkeypatch.setattr(subprocess, "run", registering)
    with pytest.raises(RuntimeError, match="cancelled"):
        cloud_modes.launch_managed(run)
    assert calls == [["supervisorctl", "reread"]]


@pytest.mark.parametrize("same_process", [True, False])
def test_direct_stop_uses_birth_identity_and_waits_for_exit(
    tmp_path, monkeypatch, same_process
):
    from types import SimpleNamespace

    from tools.vlm_benchmark import lifecycle

    run = tmp_path
    control = {"pid": 1234, "process_start_ticks": "123"}
    killed = []

    def read_stat():
        return "1234 (python) " + " ".join(
            ["Z" if killed else "S"] + ["0"] * 18 + ["123" if same_process else "456"]
        )

    monkeypatch.setattr(
        lifecycle, "Path", lambda _: SimpleNamespace(read_text=read_stat)
    )
    monkeypatch.setattr(os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(
        os, "killpg", lambda pid, sig: killed.append((pid, sig)), raising=False
    )
    lifecycle.stop_control(run, control)
    assert bool(killed) is same_process


def test_direct_launch_error_removes_key_and_saves_failure(tmp_path, monkeypatch):
    import io

    run = tmp_path.resolve() / "job"
    run.mkdir()
    doc = plan(tmp_path, monkeypatch)
    (run / "plan.json").write_text(json.dumps(doc))
    with tarfile.open(run / "bundle.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("stub.txt")
        archive.addfile(entry)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"key":"test-only"}'))

    def fail(*args, **kwargs):
        raise OSError("could not create child")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(SystemExit) as error:
        execute_remote(run_remote.LAUNCH, run, monkeypatch)
    assert error.value.code == 2
    assert not (run / "runtime-key.json").exists()
    status = json.loads((run / "launch-failure.json").read_text())
    assert status["state"] == "failed" and not status["cleanup_required"]
    execute_remote(run_remote.STOP, run, monkeypatch)
