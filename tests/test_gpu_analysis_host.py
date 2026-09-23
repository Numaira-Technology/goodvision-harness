"""GPU-host CPU analysis isolation, credential boundaries and safe shutdown."""

import json
from pathlib import Path
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from goodvision_harness.fleet import FleetController
from tools import cloud_analysis, gpu_analysis_host, start_modular


def test_remote_cpu_lanes_have_isolated_state_and_local_gpu_routes(
    tmp_path, monkeypatch
):
    first = gpu_analysis_host.lane_environment(
        {"HARNESS_STREAM_URL": "rtsp://first", "LOCAL_CAMERA_01_STREAM_URL": ""},
        tmp_path,
        1,
        37,
    )
    last = gpu_analysis_host.lane_environment(
        {"HARNESS_STREAM_URL": "rtsp://first"}, tmp_path, 37, 37
    )
    assert first["FACE_PROVIDER"] == last["FACE_PROVIDER"] == "CPUExecutionProvider"
    assert first["FACE_GALLERY_PATH"] != last["FACE_GALLERY_PATH"]
    assert first["EVENT_EVIDENCE_DIRECTORY"] != last["EVENT_EVIDENCE_DIRECTORY"]
    assert last["HARNESS_STREAM_URL"] == ""
    assert first["HARNESS_STREAM_URL"] == "rtsp://first"
    assert last["CLOUD_TRACKING_URL"] == "http://127.0.0.1:8092"
    assert last["HARNESS_ANALYSIS_URL"] == ""
    for key, value in first.items():
        monkeypatch.setenv(key, value)
    fleet = FleetController(FastAPI())
    assert fleet._url(37, "/v1/status") == "http://127.0.0.1:10036/v1/status"
    assert fleet._url(1, "/v1/status") == "http://fleet-self/v1/status"


def test_host_stop_drains_every_process_before_snapshot_can_be_taken(
    tmp_path, monkeypatch
):
    processes = [Mock() for _ in range(3)]
    actions = []
    for index, process in enumerate(processes):
        process.poll.return_value = None
        process.terminate.side_effect = lambda i=index: actions.append(("stop", i))
        process.wait.side_effect = lambda timeout, i=index: actions.append(("wait", i))
    spawn = Mock(side_effect=processes)
    monkeypatch.setattr(gpu_analysis_host.subprocess, "Popen", spawn)
    with TestClient(gpu_analysis_host.create_app(tmp_path, 3, {})) as client:
        assert client.post("/stop").json() == {"stopped": True}
        assert client.get("/health").json()["ready"] is False
        assert actions[:3] == [("stop", 0), ("stop", 1), ("stop", 2)]
        assert len(actions) == 6
        environments = [call.kwargs["env"] for call in spawn.call_args_list]
        assert len({env["HARNESS_PORT"] for env in environments}) == 3


def test_cloud_upload_never_transmits_rental_credentials():
    settings = cloud_analysis.runtime_environment(
        {
            "VAST_API_KEY": "paid-rental",
            "HF_TOKEN": "download-only",
            "SAM_SSH_KEY": "private-path",
            "INFERENCE_API_KEY": "model-key",
            "HARNESS_WEBHOOK_SECRET": "delivery-key",
            "HARNESS_ANALYSIS_URL": "http://gateway",
            "FACE_INFERENCE_THREADS": "4",
        }
    )
    assert set(settings) == {
        "INFERENCE_API_KEY",
        "HARNESS_WEBHOOK_SECRET",
        "FACE_INFERENCE_THREADS",
    }


def test_standalone_gpu_host_analysis_starts_all_lanes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(start_modular, "ROOT", tmp_path)
    monkeypatch.setattr(
        start_modular.local, "read_local_settings", lambda *args, **kwargs: {}
    )
    assert (
        start_modular.main(
            [
                "analysis",
                "--streams",
                "3",
                "--tracking-server",
                "http://gpu:8092",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["command"][-3:] == ["harness", "harness-02", "harness-03"]
    assert "HARNESS_ANALYSIS_URL" not in plan["environment"]
    assert plan["analysis_location"] == "this_host"
    assert "camera-03-data:/data" in plan["fleet_overlay"]
    assert not list(tmp_path.iterdir())


def test_backup_only_replaces_last_snapshot_after_successful_complete_download(
    tmp_path, monkeypatch
):
    import tarfile

    monkeypatch.setenv("SAM_CONTROL_DIRECTORY", str(tmp_path))
    previous = tmp_path / "analysis-data.tar.gz"
    previous.write_bytes(b"previous-backup")
    process = Mock(analysis_connection=(["ssh", "root@gpu"], ["scp"]))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "scp":
            with tarfile.open(command[-1], "w:gz") as archive:
                archive.addfile(tarfile.TarInfo("data"))

    cloud_analysis.backup(process, "/workspace/analysis", run)
    assert "/stop" in calls[0][-1]
    assert calls[1][-1].startswith("tar -czf")
    assert previous.read_bytes() != b"previous-backup"
    saved = previous.read_bytes()

    def fail_download(command, **kwargs):
        if command[0] == "scp":
            Path(command[-1]).write_bytes(b"incomplete")

    import pytest

    with pytest.raises(tarfile.ReadError):
        cloud_analysis.backup(process, "/workspace/analysis", fail_download)
    assert previous.read_bytes() == saved
