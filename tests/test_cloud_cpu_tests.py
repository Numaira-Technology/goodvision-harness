"""Acceptance logic and cloud configuration; no real GPU/API/screen recording."""

import copy
import hashlib
import json
from argparse import Namespace

import pytest
from fastapi.testclient import TestClient

from tools.cloud_cpu_tests import report, server
from tools.cloud_cpu_tests.run import PLAN_DIR, load_plan, schedule


def plan(number):
    name = "test_1_recording" if number == 1 else "test_2_dual_stream"
    return load_plan(PLAN_DIR / (name + ".json"))


def test_balanced_recording_and_concurrency_schedules():
    for number in (1, 2):
        value = plan(number)
        jobs = list(schedule(value))
        assert len(jobs) == 16
        assert jobs[0]["name"] == jobs[3]["name"]
        assert jobs[1]["name"] == jobs[2]["name"]
        for delay in (0.5, 1.0, 1.5, 2.0):
            assert len([job for job in jobs if job["delay"] == delay]) == 4
    assert all(not job["recording"] for job in schedule(plan(2)))


def test_plan_refuses_an_uncontrolled_extra_condition(tmp_path):
    value = plan(1)
    value["conditions"][1]["streams"] = 2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="isolate"):
        load_plan(path)


def paired_trials(number):
    value = plan(number)
    value["delays_seconds"] = [0.5]
    trials = []
    for job in schedule(value):
        lane = {
            "coverage": 0.75,
            "presentation_fps": 15,
            "sam_append_seconds": {"p95": 0.3},
            "invariant": {"same": True},
        }
        trials.append(
            {
                "job": job,
                "valid": True,
                "lanes": [dict(lane) for _ in range(job["streams"])],
                "overlap_ratio": 0.99,
                "start_skew_seconds": 0.01,
            }
        )
    return value, trials


def test_recording_equal_low_coverage_is_not_a_70_percent_pass():
    value, trials = paired_trials(1)
    for trial in trials:
        trial["lanes"][0]["coverage"] = 0.05
    result = report.comparisons(value, trials)[0]
    assert result["repeated_acceptance"]
    assert all(not p["each_changed_lane_coverage_pass"] for p in result["pairs"])


def test_recording_regression_uses_cloud_time_not_only_display_latency():
    value, trials = paired_trials(1)
    next(t for t in trials if t["job"]["recording"])["lanes"][0][
        "sam_append_seconds"
    ] = {"p95": 0.5}
    assert not report.comparisons(value, trials)[0]["repeated_acceptance"]


@pytest.mark.parametrize(
    "failure", ["one_lane", "overlap", "skew", "config", "invalid", "duplicate"]
)
def test_dual_cannot_hide_failed_lane_or_sequential_playback(failure):
    value, trials = paired_trials(2)
    changed = next(t for t in trials if t["job"]["streams"] == 2)
    if failure == "one_lane":
        changed["lanes"][0]["coverage"] = 1.0
        changed["lanes"][1]["coverage"] = 0.5  # Mean passes; this lane does not.
    elif failure == "overlap":
        changed["overlap_ratio"] = 0.1
    elif failure == "skew":
        changed["start_skew_seconds"] = 1
    elif failure == "config":
        changed["lanes"][1]["invariant"] = {"other": True}
    elif failure == "invalid":
        changed["valid"] = False
    else:
        trials.append(copy.deepcopy(changed))
    assert not report.comparisons(value, trials)[0]["repeated_acceptance"]


def test_both_repeated_dual_lanes_must_pass():
    value, trials = paired_trials(2)
    assert report.comparisons(value, trials)[0]["repeated_acceptance"]
    assert not report.comparisons(value, trials[:2])[0]["repeated_acceptance"]


@pytest.mark.parametrize("serve_video", [False, True])
def test_cloud_service_uses_real_sam_client_and_changes_only_buffer(
    tmp_path, monkeypatch, serve_video
):
    from goodvision_harness.perception import PerceptionSessionHTTPClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-not-a-real-key")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("LOCAL_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("LOCAL_TRACKING_ENABLED", "false")
    monkeypatch.setenv("INFERENCE_EXECUTION_MODE", "cloud")
    monkeypatch.setenv("TRACKING_EXECUTION_MODE", "legacy")
    monkeypatch.setattr(
        server, "gpu_snapshot", lambda: "gpu-test,RTX 4090,24576,0,0,450,0,2500,TEST"
    )
    monkeypatch.setattr(server.HarnessRuntime, "start", lambda self: None)
    monkeypatch.setattr(server.HarnessRuntime, "stop", lambda self: None)
    video = tmp_path / "video.mp4"
    if serve_video:
        video.write_bytes(b"example video bytes")
    args = Namespace(
        output=tmp_path,
        lane=2,
        sam_endpoint="http://127.0.0.1:8092",
        face_model_root=tmp_path / "models",
        allow_face_download=False,
        video=video if serve_video else None,
    )
    app = server.build_app(args)
    with TestClient(app) as client:
        before = client.get("/benchmark/config").json()
        changed = client.post("/benchmark/trial", json={"id": "trial-1", "delay": 1.0})
        assert changed.status_code == 200, changed.text
        after = changed.json()
        if serve_video:
            assert (
                after["host"]["video_sha256"]
                == hashlib.sha256(video.read_bytes()).hexdigest()
            )
            response = client.get(
                "/experiment-video/original", headers={"range": "bytes=0-6"}
            )
            assert response.status_code == 206
            assert response.content == b"example"
        else:
            assert after["host"]["video_sha256"] is None
            assert client.get("/experiment-video/original").status_code == 404
        assert after["fixed_main"]["effective_display_delay_seconds"] == 1.0
        browser = client.get("/v1/ui/info").json()["local_camera_display"]
        assert browser["enabled"] is True
        assert browser["delay_seconds"] == 1.0
        assert report.invariant(before) == report.invariant(after)
        assert after["host"]["sam_endpoint"].endswith(
            "camera-02/perception/sam_live/sessions"
        )
        runtime = app.state.runtime
        assert runtime.config.model_api.provider == "openrouter"
        assert runtime.config.model_api.api_key_env == "OPENROUTER_API_KEY"
        assert runtime.config.inference.model == "qwen/qwen3.5-35b-a3b:nitro"
        tracking = runtime._create_tracking_client(runtime.config.rich_labeling)
        assert type(tracking) is PerceptionSessionHTTPClient
        assert tracking.native_sam_video_tracking
        tracking.close()
        assert runtime.config.face.provider == "CPUExecutionProvider"
        assert (
            client.post("/benchmark/trial", json={"id": "bad", "delay": -1}).status_code
            == 400
        )
        monkeypatch.setattr(runtime, "_processing_lifecycle_busy", lambda: True)
        assert (
            client.post("/benchmark/trial", json={"id": "busy", "delay": 2}).status_code
            == 409
        )


def test_recording_and_source_timing_metrics_are_not_invented(tmp_path):
    # Insufficient evidence must fail instead of fabricating a fast GPU time.
    job = {"id": "x", "delay": 0.5, "video_sha256": "same"}
    lane = tmp_path / "lane-1"
    lane.mkdir()
    (lane / "browser-report.json").write_text(
        json.dumps(
            {
                "completed": True,
                "drained": True,
                "errors": [],
                "video_sha256": "same",
                "config": {"config": {"delay_seconds": 0.5}, "video_duration": 2},
                "final": {
                    "distinct_presented_frames": 2,
                    "continuous_contour_frames": 1,
                    "complete_contour_frames": 1,
                },
                "play_started_at": 1000,
                "source_ended_at": 3000,
                "benchmark": {"fixed_main": {"effective_display_delay_seconds": 0.5}},
            }
        )
    )
    (lane / "frames.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "present",
                    "sequence": index,
                    "at": 1000 + index * 1000,
                    "counted_continuous": bool(index),
                    "counted_complete": bool(index),
                    "latency_seconds": 0.5,
                }
            )
            for index in range(2)
        )
    )
    (lane / "ws.jsonl").write_text(
        json.dumps({"payload": {"type": "session", "stream_session_id": "current"}})
    )
    info = {
        "trial": {"id": "x", "delay": 0.5},
        "timings": [
            {
                "event": "sam_coordinator_timing",
                "stream_session_id": "old",
                "outcome": "ok",
                "during_capture": True,
                "output_frames": 1,
                "append_seconds": 0.01,
            }
        ],
        "telemetry": [],
        "fixed_main": {"effective_display_delay_seconds": 0.5},
        "host": {
            "hostname": "cloud",
            "gpu_at_start": "uuid,4090,24,0,0,450,0,2,driver",
            "cpu_count": 8,
            "ram_bytes": 64,
            "config_sha256": "a",
            "harness_source_sha256": "b",
            "worker_source_sha256": {},
        },
    }
    (tmp_path / "cloud-evidence-1.json").write_text(json.dumps(info))
    result = report.lane_summary(tmp_path, 1, job)
    assert result["coverage"] == 0.5
    assert not result["valid"]
    assert result["sam_append_seconds"]["p95"] is None


def test_upload_backpressure_is_a_valid_measured_failure(tmp_path):
    job = {"id": "x", "delay": 1.5, "video_sha256": "same"}
    lane = tmp_path / "lane-1"
    lane.mkdir()
    browser = {
        "completed": True,
        "drained": True,
        "errors": [{"kind": "failure", "message": report.UPLOAD_BACKPRESSURE_MESSAGE}],
        "video_sha256": "same",
        "config": {"config": {"delay_seconds": 1.5}, "video_duration": 2},
        "final": {
            "distinct_presented_frames": 2,
            "continuous_contour_frames": 0,
            "complete_contour_frames": 0,
        },
        "play_started_at": 1000,
        "source_ended_at": 3000,
        "benchmark": {"fixed_main": {"effective_display_delay_seconds": 1.5}},
    }
    (lane / "browser-report.json").write_text(json.dumps(browser))
    (lane / "frames.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"kind": "capture", "at": 1000, "sequence": 1},
                *[
                    {
                        "kind": "present",
                        "at": 1000 + i * 1000,
                        "sequence": i + 1,
                        "counted_continuous": False,
                        "counted_complete": False,
                        "latency_seconds": 1.5,
                    }
                    for i in range(2)
                ],
            ]
        )
    )
    (lane / "ws.jsonl").write_text(
        json.dumps({"payload": {"type": "session", "stream_session_id": "current"}})
    )
    evidence = {
        "trial": {"id": "x", "delay": 1.5},
        "timings": [
            {
                "event": "sam_coordinator_timing",
                "stream_session_id": "current",
                "outcome": "ok",
                "during_capture": True,
                "output_frames": 1,
                "append_seconds": 0.3,
                "source_to_result": [{"seconds": 0.4}],
            }
        ],
        "telemetry": [
            {
                "trial_id": "x",
                "process_cpu_percent": 1,
                "system_cpu_percent": 2,
                "available_ram_bytes": 3,
                "swap_used_bytes": 0,
            }
        ],
        "fixed_main": {"effective_display_delay_seconds": 1.5},
        "host": {
            "hostname": "cloud",
            "gpu_at_start": "uuid,4090,24,0,0,450,0,2,driver",
            "cpu_count": 8,
            "ram_bytes": 64,
            "config_sha256": "a",
            "harness_source_sha256": "b",
            "worker_source_sha256": {},
        },
    }
    (tmp_path / "cloud-evidence-1.json").write_text(json.dumps(evidence))
    result = report.lane_summary(tmp_path, 1, job)
    assert result["valid"]
    assert result["coverage"] == 0
    assert result["analysis_backpressure_warnings"] == 1
    browser["errors"] = [
        {
            "kind": "failure",
            "message": report.SERVER_QUEUE_BACKPRESSURE_MESSAGE,
        }
    ]
    (lane / "browser-report.json").write_text(json.dumps(browser))
    assert report.lane_summary(tmp_path, 1, job)["valid"]
    browser["errors"] = [{"kind": "fatal", "message": "Browser crashed"}]
    (lane / "browser-report.json").write_text(json.dumps(browser))
    assert not report.lane_summary(tmp_path, 1, job)["valid"]


def test_repeat_on_a_different_machine_is_not_a_repeated_pass():
    value, trials = paired_trials(2)
    for trial in trials:
        if trial["job"]["repeat"] == 2:
            for lane in trial["lanes"]:
                lane["invariant"] = {"other_gpu": True}
    assert not report.comparisons(value, trials)[0]["repeated_acceptance"]
