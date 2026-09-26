"""Candidate protocol/source integrity and acceptance; no real GPU/API claims."""

import io
import json
import os

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.video_tracking import VideoFrame
from tools.seg_pipeline import worker
from tools.seg_pipeline.report import summarize_trial, write_report
from tools.seg_pipeline.service import (
    CandidateClient,
    CandidateRuntime,
    app_for_job,
    benchmark_config,
    fixed_protocol,
    main_defaults,
)


def test_fixed_main_ignores_local_overrides_and_preserves_pipeline(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DISPLAY_SAM_LIVE_DELAY_SECONDS", "9")
    monkeypatch.setenv("LOCAL_TRACKING_FPS", "12")
    monkeypatch.setenv("RICH_LABELING_VIDEO_TRANSPORT_CODEC", "h264")
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    base = main_defaults("openrouter")
    job = {
        "output": str(tmp_path),
        "endpoint": "http://127.0.0.1:8092",
        "face_model_root": str(tmp_path / "models"),
        "allow_face_download": False,
    }
    config = benchmark_config(job)
    assert config.display == base.display
    assert config.video == base.video
    assert config.model_api == base.model_api
    assert config.inference == base.inference
    assert config.model_api.provider == "openrouter"
    assert config.model_api.api_key_env == "OPENROUTER_API_KEY"
    assert config.inference.model == "qwen/qwen3.5-35b-a3b:nitro"
    protocol = fixed_protocol(config)
    assert protocol["effective_display_delay_seconds"] == 0
    assert protocol["tracking"]["max_fps"] == 2
    assert protocol["tracking"]["video_transport_codec"] == "jpeg"
    assert protocol["face"]["provider"] == "CPUExecutionProvider"
    assert protocol["face"]["sample_fps"] == base.face.sample_fps
    assert os.environ["LOCAL_TRACKING_FPS"] == "12"
    assert os.environ["INFERENCE_PROVIDER"] == "aiclub"
    changed = protocol | {"effective_display_delay_seconds": 1.5}
    with pytest.raises(ValueError, match="configuration changed"):
        app_for_job(job | {"fixed_main": changed})


@pytest.mark.parametrize("delay", [0, 0.5, 1, 1.5, 2])
def test_browser_route_uses_frozen_delay_before_analysis(tmp_path, monkeypatch, delay):
    # Exercise the real UI endpoint without loading a face/GPU model or calling
    # a remote API. A stale browser environment must not override the job.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("LOCAL_CAMERA_DISPLAY_ENABLED", "false")
    monkeypatch.setenv("LOCAL_CAMERA_DISPLAY_DELAY_SECONDS", "6")
    monkeypatch.setattr(CandidateClient, "gpu_ready", lambda self: False)
    monkeypatch.setattr(
        CandidateRuntime, "_ensure_rich_readiness_probe", lambda self: None
    )
    job = {
        "output": str(tmp_path),
        "endpoint": "http://127.0.0.1:8092",
        "face_model_root": str(tmp_path / "models"),
        "allow_face_download": False,
        "display_delay_seconds": delay,
    }
    protocol = fixed_protocol(benchmark_config(job))
    with TestClient(app_for_job(job | {"fixed_main": protocol})) as client:
        response = client.get("/v1/ui/info")
        assert response.status_code == 200
        display = response.json()["local_camera_display"]
        assert display["enabled"] is True
        assert display["delay_seconds"] == protocol["effective_display_delay_seconds"]
        assert display["delay_seconds"] == delay
        assert display["analysis_sample_fps"] == protocol["face"]["sample_fps"]
        assert display["motion_backend"] == "dense"
        assert client.app.state.runtime._session is None


class DetectorDouble:
    device = "cpu"

    def predict_scored(self, bgr):
        masks = np.zeros((1, *bgr.shape[:2]), bool)
        masks[:, 5:25, 10:25] = True
        return masks, np.array([0.87])


class TinyDetectorDouble:
    device = "cpu"

    def predict_scored(self, bgr):
        masks = np.zeros((1, *bgr.shape[:2]), bool)
        masks[:, 5, 10] = True
        return masks, np.array([0.87])


@pytest.mark.parametrize("codec", ["jpeg", "h264"])
def test_candidate_real_protocol_and_scene_accept_exact_source(codec, monkeypatch):
    monkeypatch.setattr(worker, "synchronize", lambda _: None)
    # GPU flag is an explicit protocol double, never a real device assertion.
    backend = worker.CandidateBackend(DetectorDouble(), verified_cuda=True)
    app = worker.create_app(backend, {"model": "TEST_DOUBLE"})
    with TestClient(app) as gateway:

        def transport(request):
            result = gateway.request(
                request.method, request.url.path, content=request.content
            )
            return httpx.Response(result.status_code, content=result.content)

        client = CandidateClient(
            "http://test",
            transport=httpx.MockTransport(transport),
            transport_codec=codec,
            timeout_seconds=5,
        )
        scene = SceneRuntime(
            "camera",
            "test",
            mode="active",
            tracking_mode="batch",
            native_memory_scope="session",
        )
        registry = PersonRegistry("camera", "test")
        stream = io.BytesIO()
        Image.new("RGB", (64, 40), "gray").save(stream, "JPEG")
        jpeg = stream.getvalue()
        try:
            assert client.ready_for_new_session()
            client.open_session("camera")
            ids = []
            for index in range(3):
                frame = VideoFrame(index * 7, index * 0.5, jpeg)
                scene.observe_frame(jpeg, frame.timestamp_seconds, frame.frame_sequence)
                result = client.append((frame,), stream_session_id="camera")
                snap = result.frames[0]
                person = snap.objects[0]
                assert person.confidence == pytest.approx(0.87)
                assert person.attributes["native_track"]["backend"] == "seg_benchmark"
                assert (
                    person.attributes["mask_contours"]["source_frame_sequence"]
                    == index * 7
                )
                assert person.attributes["tracking_confirmation"] == (
                    "provisional" if index < 2 else "confirmed"
                )
                ids.append(person.track_id)
                registry.observe_snapshot(snap)
                assert scene.observe_tracking_batch(
                    (snap,),
                    completed_frame_sequence=frame.frame_sequence,
                    completed_timestamp_seconds=frame.timestamp_seconds,
                )
            assert len(set(ids)) == 1
            assert registry.native_tracking_stats()["tracks"] == 1
            eof = client.append((), stream_session_id="camera", is_last_batch=True)
            assert eof.frames == () and eof.state == "closed"
            assert [row["source_sequence"] for row in backend.timings] == [0, 7, 14]
        finally:
            client.close()
            scene.close()


def test_candidate_omits_tiny_masks_before_confirmation_metadata(monkeypatch):
    monkeypatch.setattr(worker, "synchronize", lambda _: None)
    backend = worker.CandidateBackend(TinyDetectorDouble(), verified_cuda=True)
    with TestClient(worker.create_app(backend, {"model": "TEST_DOUBLE"})) as gateway:

        def transport(request):
            result = gateway.request(
                request.method, request.url.path, content=request.content
            )
            return httpx.Response(result.status_code, content=result.content)

        client = CandidateClient(
            "http://test",
            transport=httpx.MockTransport(transport),
            timeout_seconds=5,
        )
        stream = io.BytesIO()
        Image.new("RGB", (64, 40), "gray").save(stream, "JPEG")
        try:
            client.open_session("camera")
            result = client.append(
                (VideoFrame(0, 0, stream.getvalue()),),
                stream_session_id="camera",
            )
            assert result.frames[0].objects == ()
            assert backend.timings[-1]["people"] == 0
        finally:
            client.close()


def test_cpu_double_cannot_claim_gpu_ready(monkeypatch):
    monkeypatch.setattr(worker, "synchronize", lambda _: None)
    with TestClient(
        worker.create_app(worker.CandidateBackend(DetectorDouble()), {})
    ) as client:
        response = client.get("/health/perception/seg_benchmark")
        assert response.status_code == 503
        assert response.json()["gpu_ready"] is False


def test_tracker_expires_by_source_time_and_never_emits_missing_masks():
    masks, _ = DetectorDouble().predict_scored(np.zeros((40, 64, 3)))
    tracker = worker.MaskTracker()
    assert tracker.update(masks, 0)[0].tolist() == [0]
    assert tracker.update(np.zeros((0, 40, 64), bool), 1)[0].tolist() == []
    assert tracker.update(masks, 2)[0].tolist() == [1]


def report_fixture(directory):
    def write(name, data):
        (directory / name).write_text(json.dumps(data))

    write(
        "job.json",
        {
            "model": "yolov8-n",
            "delay": 0,
            "repeat": 1,
            "sample_fps": 2,
            "queue_policy": "latest_only",
            "video_sha256": "same",
            "fixed_main": {"effective_display_delay_seconds": 0},
        },
    )
    write(
        "browser-report.json",
        {
            "completed": True,
            "drained": True,
            "errors": [],
            "video_sha256": "same",
            "play_started_at": 1000,
            "source_ended_at": 5000,
            "config": {
                "config": {"delay_seconds": 0, "target_fps": 1},
                "video_duration": 4,
            },
            "benchmark": {"fixed_main": {"effective_display_delay_seconds": 0}},
            "final": {
                "captured_frames": 4,
                "distinct_presented_frames": 4,
                "continuous_contour_frames": 3,
                "complete_contour_frames": 2,
                "display_buffer_expired": 0,
                "display_catchup_expired": 0,
                "rejected_source_frames": 0,
            },
        },
    )
    rows = []
    for i in range(4):
        rows.extend(
            [
                {
                    "kind": "capture",
                    "at": 1000 + i * 1000,
                    "sequence": i,
                    "pts": i,
                },
                {
                    "kind": "present",
                    "at": 1000 + i * 1000,
                    "sequence": i,
                    "pts": i,
                    "latency_seconds": 1.6,
                    "counted_continuous": i > 0,
                    "counted_complete": i > 1,
                },
            ]
        )
    (directory / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (directory / "ws.jsonl").write_text(
        json.dumps({"payload": {"type": "session", "stream_session_id": "test"}}) + "\n"
    )
    write(
        "worker-timings.json",
        {
            "frames": [
                {"stream_session_id": "test", "inference_seconds": 0.02},
                {"stream_session_id": "old-session", "inference_seconds": 100},
            ]
        },
    )


def test_report_counts_startup_and_rejects_counter_disagreement(tmp_path):
    report_fixture(tmp_path)
    result = summarize_trial(tmp_path)
    assert result["coverage"] == 0.75 and result["strict_coverage"] == 0.5
    assert result["coverage_pass"]
    assert result["model_inference_seconds"]["p95"] == 0.02
    assert not result["segmentation_accuracy_measured"]
    path = tmp_path / "browser-report.json"
    value = json.loads(path.read_text())
    value["final"]["continuous_contour_frames"] = 4
    path.write_text(json.dumps(value))
    assert not summarize_trial(tmp_path)["coverage_pass"]


def test_report_rejects_single_presented_frame_for_complete_video(tmp_path):
    report_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "frames.jsonl").read_text().splitlines()
    ]
    rows = [row for row in rows if row["sequence"] == 0]
    present = next(row for row in rows if row["kind"] == "present")
    present["counted_continuous"] = True
    present["counted_complete"] = True
    (tmp_path / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    path = tmp_path / "browser-report.json"
    value = json.loads(path.read_text())
    value["final"].update(
        captured_frames=1,
        distinct_presented_frames=1,
        continuous_contour_frames=1,
        complete_contour_frames=1,
    )
    path.write_text(json.dumps(value))
    result = summarize_trial(tmp_path)
    assert result["coverage"] == 1
    assert not result["valid"]
    assert not result["coverage_pass"]
    assert any("full video" in problem for problem in result["problems"])


def test_repeated_pass_requires_unique_trials_under_same_fixed_main(tmp_path):
    protocol = {"effective_display_delay_seconds": 0, "tracking": {"max_fps": 2}}
    manifest = {"model": "yolov8-n", "repeats": 2, "fixed_main": protocol}
    result = {
        "model": "yolov8-n",
        "configured_delay_seconds": 0,
        "repeat": 1,
        "coverage_pass": True,
        "coverage": 0.75,
        "fixed_main": protocol,
    }
    baseline = {"recommendation": {"lowest_repeated_passing_tested_delay_seconds": 5}}
    summary = write_report(tmp_path, manifest, [result], baseline)
    assert not summary["repeated_coverage_pass"]
    summary = write_report(
        tmp_path, manifest, [result, result | {"repeat": 2}], baseline
    )
    assert summary["repeated_coverage_pass"]
    assert summary["same_host_same_version_comparison"] is False
    assert not write_report(tmp_path, manifest, [result, result], baseline)[
        "repeated_coverage_pass"
    ]
    other_config = result | {"repeat": 2, "configured_delay_seconds": 1.5}
    assert not write_report(tmp_path, manifest, [result, other_config], baseline)[
        "repeated_coverage_pass"
    ]
