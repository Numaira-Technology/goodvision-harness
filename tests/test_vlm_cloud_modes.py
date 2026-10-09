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


@pytest.mark.skipif(os.name == "nt", reason="cloud POSIX worker")
@pytest.mark.parametrize("record", [False, True])
def test_uploaded_mode_suite_runs_and_collects_recordings(
    tmp_path, monkeypatch, record
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


def test_managed_launch_is_one_shot_and_controls_only_its_own_service(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "conf.d"
    config_dir.mkdir()
    run = tmp_path / "20261008T190000Z-abcdef01"
    (run / "app").mkdir(parents=True)
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


@pytest.mark.parametrize("same_process", [True, False])
def test_status_uses_process_identity_during_exec_transition(
    tmp_path, monkeypatch, same_process
):
    import pathlib
    from types import SimpleNamespace

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

    def paths(value):
        if str(value) == "/proc/1234/stat":
            return SimpleNamespace(read_text=lambda: stat)
        if str(value) == "/proc/1234/cmdline":
            return SimpleNamespace(read_bytes=lambda: b"")
        return type(tmp_path)(value)

    monkeypatch.setattr(sys, "argv", ["status", str(job)])
    monkeypatch.setattr(pathlib, "Path", paths)
    exec(run_remote.STATUS, {})
    assert secret.exists() is same_process
    assert json.loads((job / "status.json").read_text())["state"] == (
        "installing" if same_process else "failed"
    )
