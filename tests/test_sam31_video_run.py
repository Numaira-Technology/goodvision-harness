"""Synthetic source video plus MockTransport; never runs a GPU or paid model."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import (
    RichLabelObject,
    RichLabelProtocolError,
    RichLabelSnapshot,
)
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.video_tracking import VideoFrame, VideoTrackingHTTPClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sam31_video_run_tests", ROOT / "tools" / "sam31_video_run.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    video = tmp_path / "private-video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 120))
    assert writer.isOpened()
    try:
        for sequence in range(10):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(
                frame, (20 + sequence, 12), (70 + sequence, 110), (120, 120, 120), -1
            )
            cv2.rectangle(frame, (100, 75), (150, 110), (sequence * 20, 60, 120), -1)
            writer.write(frame)
    finally:
        writer.release()
    settings = SimpleNamespace(
        ready=True,
        reuse_epoch=False,
        corrupt_source=False,
        people=True,
        empty_sequences=set(),
        requests=[],
        clients=[],
    )

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "ready": settings.ready,
                    "gpu_ready": settings.ready,
                    "backend": "sam3.1_video",
                    "native_sam_video_tracking": True,
                    "memory_scope": "batch",
                },
            )
        incoming = json.loads(request.content)
        settings.requests.append(incoming)
        epoch = (
            "epoch-fixed" if settings.reuse_epoch else f"epoch-{len(settings.requests)}"
        )
        frames = []
        for frame in incoming["frames"]:
            jpeg = base64.b64decode(frame["jpeg_base64"], validate=True)
            assert jpeg.startswith(b"\xff\xd8")
            seq, at = frame["frame_sequence"], frame["timestamp_seconds"]
            frames.append(
                {
                    "stream_session_id": incoming["stream_session_id"],
                    "frame_sequence": seq,
                    "timestamp_seconds": at,
                    "objects": [
                        {
                            "track_id": f"native-{epoch}-1",
                            "label": "person",
                            "bbox": {"x1": 0.1, "y1": 0.05, "x2": 0.5, "y2": 0.95},
                            "confidence": 0.99,
                            "observation_state": "observed",
                            "last_observed_timestamp_seconds": at,
                            "last_observed_frame_sequence": seq,
                            "attributes": {
                                "native_track": {
                                    "backend": "sam3.1_video",
                                    "epoch": epoch,
                                    "object_id": "1",
                                }
                            },
                        }
                    ]
                    if settings.people and seq not in settings.empty_sequences
                    else [],
                }
            )
        response = {
            "stream_session_id": incoming["stream_session_id"],
            "batch_id": incoming["batch_id"],
            "predictor_epoch": epoch,
            "backend": "sam3.1_video",
            "memory_scope": "batch",
            "frames": frames,
            "completed_frame_sequence": frames[-1]["frame_sequence"],
            "completed_timestamp_seconds": frames[-1]["timestamp_seconds"],
        }
        if settings.corrupt_source:
            response["frames"][0]["frame_sequence"] += 1000
        return httpx.Response(200, json=response)

    def client(endpoint, **kwargs):
        value = VideoTrackingHTTPClient(
            endpoint, transport=httpx.MockTransport(handler), **kwargs
        )
        settings.clients.append(value)
        return value

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "default validation must not load model configuration or credentials"
        )

    monkeypatch.setattr(runner, "VideoTrackingHTTPClient", client)
    monkeypatch.setattr(runner, "load_config", forbidden)
    monkeypatch.setattr(runner, "TemporalVLMInspector", forbidden)
    monkeypatch.setattr(runner, "environment_api_key", forbidden)
    args = SimpleNamespace(
        video=str(video),
        output=str(tmp_path / "output"),
        sam_endpoint="http://mock-worker:8092/v1/video/batches",
        config="unused.yaml",
        events="smoking",
        fps=4,
        max_seconds=60,
        batch_frames=2,
        timeout_seconds=75,
        max_requests=1,
        with_inference=False,
    )
    return args, settings


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_video_sampling_tail_flush_and_native_source_mapping_without_vlm(
    environment,
):
    args, settings = environment
    report = runner.run(args)
    assert report["samples"] == 5
    assert report["batches"] == 3
    assert [len(request["frames"]) for request in settings.requests] == [2, 2, 1]
    frames = [frame for request in settings.requests for frame in request["frames"]]
    assert [frame["frame_sequence"] for frame in frames] == [0, 2, 4, 6, 8]
    assert [frame["timestamp_seconds"] for frame in frames] == [0, 0.25, 0.5, 0.75, 1]
    output = Path(args.output)
    tracks = records(output / "tracks.jsonl")
    canonical_ids = [row["canonical"]["objects"][0]["track_id"] for row in tracks]
    assert canonical_ids[0] == canonical_ids[1]
    assert canonical_ids[2] == canonical_ids[3]
    assert len(set(canonical_ids)) == 3
    scene_rows = records(output / "scene.jsonl")
    assert len([row for row in scene_rows if row["kind"] == "source_frame"]) == 5
    assert len([row for row in scene_rows if row["kind"] == "batch_completed"]) == 3
    assert scene_rows[-1]["runtime"]["tracking_completed_frame_sequence"] == 8
    assert report["model_enabled"] is False
    assert report["model_requests"] == 0
    assert report["memory_scope"] == "batch"
    assert report["continuous_camera_memory"] is False
    assert report["accuracy_measured"] is False
    assert report["identity_switches"] is None
    assert report["processing_latency_seconds"]["mean"] >= 0
    assert json.loads((output / "report.json").read_text()) == json.loads(
        json.dumps(report)
    )
    assert all(client._client.is_closed for client in settings.clients)
    assert "jpeg_base64" not in (output / "tracks.jsonl").read_text()


def test_empty_video_observations_still_commit_completion_watermark(environment):
    args, settings = environment
    settings.people = False
    report = runner.run(args)
    assert report["native_object_instances_across_batches"] == 0
    assert report["final_runtime"]["tracking_completed_frame_sequence"] == 8
    assert report["final_runtime"]["scene"]["entities"] == []


def test_existing_output_is_rejected_before_any_worker_call(environment):
    args, settings = environment
    output = Path(args.output)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing output")
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert not settings.clients
    assert sentinel.read_text() == "existing output"


def test_unready_gpu_fails_without_creating_output(environment):
    args, settings = environment
    settings.ready = False
    with pytest.raises(RuntimeError, match="GPU-ready"):
        runner.run(args)
    assert not settings.requests
    assert not Path(args.output).exists()
    assert settings.clients[0]._client.is_closed


def test_empty_recording_cannot_produce_a_success_report(environment):
    args, settings = environment
    empty = Path(args.video).with_name("empty.avi")
    writer = cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 120))
    assert writer.isOpened()
    writer.release()
    args.video = str(empty)
    with pytest.raises(ValueError, match="no sampled frames"):
        runner.run(args)
    assert not settings.requests
    assert not (Path(args.output) / "report.json").exists()
    assert all(client._client.is_closed for client in settings.clients)


def test_source_corruption_is_rejected_before_batch_changes_scene(environment):
    args, settings = environment
    settings.corrupt_source = True
    with pytest.raises(RichLabelProtocolError):
        runner.run(args)
    assert (Path(args.output) / "tracks.jsonl").read_text() == ""
    assert not (Path(args.output) / "report.json").exists()
    assert not (Path(args.output) / "annotated.mp4").exists()
    assert (Path(args.output) / "annotations.jsonl").read_text() == ""
    assert settings.clients[0]._client.is_closed


def test_reused_batch_epoch_is_not_reported_as_continuous_memory(environment):
    args, settings = environment
    settings.reuse_epoch = True
    with pytest.raises(ValueError, match="reused"):
        runner.run(args)
    assert len(records(Path(args.output) / "batches.jsonl")) == 1
    assert not (Path(args.output) / "report.json").exists()


@pytest.mark.parametrize("fails", [False, True])
def test_explicit_vlm_budget_is_strict_and_failures_are_counted(
    environment, monkeypatch, fails
):
    args, _ = environment
    args.with_inference = True
    args.batch_frames = 16

    class Inspector:
        calls = 0
        closed = False

        def inspect(self, job):
            self.calls += 1
            if fails:
                raise RuntimeError("private credential material")
            subject = job.request.subject_ids or (job.subject_boxes[-1][1],)
            return (TemporalVerdict("smoking", subject, True, 0.9),)

        def stats(self):
            return {"calls": self.calls}

        def close(self):
            self.closed = True

    inspector = Inspector()
    monkeypatch.setattr(runner, "_make_inspector", lambda _args: inspector)
    report = runner.run(args)
    assert report["model_requests"] == inspector.calls == 1
    assert report["model_failures"] == int(fails)
    assert report["ok"] is not fails
    assert report["budget_exhausted"] is True
    assert inspector.closed
    assert "private credential" not in json.dumps(report)


def test_two_independent_windows_in_one_native_batch_confirm_an_event(
    environment, monkeypatch
):
    args, settings = environment
    video = Path(args.video).with_name("longer.avi")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 120))
    assert writer.isOpened()
    try:
        for _ in range(32):
            writer.write(np.full((120, 160, 3), 100, dtype=np.uint8))
    finally:
        writer.release()
    args.video = str(video)
    args.batch_frames = 16
    args.with_inference = True
    args.max_requests = 2

    class Inspector:
        def __init__(self):
            self.jobs = []

        def inspect(self, job):
            self.jobs.append(job)
            subject = (job.subject_boxes[-1][1],)
            return (TemporalVerdict("smoking", subject, True, 0.9),)

        def stats(self):
            return {"calls": len(self.jobs)}

        def close(self):
            pass

    inspector = Inspector()
    monkeypatch.setattr(runner, "_make_inspector", lambda _args: inspector)
    report = runner.run(args)
    assert report["batches"] == len(settings.requests) == 1
    assert report["model_requests"] == len(inspector.jobs) == 2
    first = {frame.frame_sequence for frame in inspector.jobs[0].frames}
    second = {frame.frame_sequence for frame in inspector.jobs[1].frames}
    assert first.isdisjoint(second)
    assert max(first) < min(second)
    events = report["final_runtime"]["scene"]["events"]
    assert len(events) == 1
    assert events[0]["phase"] == "confirmed"
    assert events[0]["positive_observations"] == 2
    assert report["accepted_verdicts"] == 2
    assert report["accuracy_measured"] is False


def test_cli_defaults_to_no_vlm_and_does_not_print_private_source_paths(
    environment, monkeypatch, capsys
):
    args, _ = environment
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam31_video_run.py",
            args.video,
            "--sam-endpoint",
            args.sam_endpoint,
            "--output",
            args.output,
        ],
    )
    assert runner.main() == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["model_requests"] == 0
    assert report["backend"] == "sam3.1_video"
    assert report["memory_scope"] == "batch"
    assert "private-video" not in output
    assert args.sam_endpoint not in output


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--fps", "nan"),
        ("--max-seconds", "inf"),
        ("--batch-frames", "17"),
        ("--max-requests", "101"),
        ("--timeout-seconds", "0"),
    ],
)
def test_cli_rejects_unbounded_work_before_external_calls(
    environment, monkeypatch, flag, value
):
    args, settings = environment
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam31_video_run.py",
            args.video,
            "--sam-endpoint",
            args.sam_endpoint,
            "--output",
            args.output,
            flag,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not settings.clients


def test_video_and_annotation_map_preserve_actual_source_frames_and_batch_boundaries(
    environment,
):
    args, settings = environment
    report = runner.run(args)
    output = Path(args.output)
    annotations = records(output / "annotations.jsonl")
    assert [row["source_frame_sequence"] for row in annotations] == [0, 2, 4, 6, 8]
    assert [row["source_timestamp_seconds"] for row in annotations] == [
        0,
        0.25,
        0.5,
        0.75,
        1,
    ]
    assert [row["output_frame_sequence"] for row in annotations] == list(range(5))
    assert [row["output_timestamp_seconds"] for row in annotations] == [
        0,
        0.25,
        0.5,
        0.75,
        1,
    ]
    assert [row["batch_number"] for row in annotations] == [1, 1, 2, 2, 3]
    assert [row["batch_start"] for row in annotations] == [
        True,
        False,
        True,
        False,
        True,
    ]
    tracks = records(output / "tracks.jsonl")
    for annotation, track in zip(annotations, tracks, strict=True):
        assert annotation["subject_ids"] == [
            track["canonical"]["objects"][0]["track_id"]
        ]
        assert annotation["native_object_ids"] == ["1"]
        assert annotation["predictor_epoch"] == track["predictor_epoch"]
    source_frames = [
        frame for request in settings.requests for frame in request["frames"]
    ]
    video = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert video.isOpened()
    assert video.get(cv2.CAP_PROP_FPS) == 4
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    try:
        for source in source_frames:
            ok, image = video.read()
            assert ok
            expected = cv2.imdecode(
                np.frombuffer(base64.b64decode(source["jpeg_base64"]), np.uint8),
                cv2.IMREAD_COLOR,
            )
            # This per-frame color patch is outside all boxes and captions. If a
            # late batch rendered every result over its last JPEG, this fails.
            source_patch = expected[82:102, 112:140].astype(float)
            video_patch = image[76 + 82 : 76 + 102, 112:140].astype(float)
            assert (
                np.abs(
                    video_patch.mean(axis=(0, 1)) - source_patch.mean(axis=(0, 1))
                ).max()
                < 8
            )
        assert video.read()[0] is False
    finally:
        video.release()
    assert report["annotated_video"]["frames"] == report["samples"] == 5
    assert report["annotated_video"]["cross_batch_identity_continuity"] is False


def test_empty_later_sources_do_not_reuse_previous_person_boxes(environment):
    args, settings = environment
    settings.empty_sequences = {4, 6, 8}
    report = runner.run(args)
    annotations = records(Path(args.output) / "annotations.jsonl")
    assert annotations[0]["subject_ids"]
    assert all(row["subject_ids"] == [] for row in annotations[2:])
    assert all(row["native_object_ids"] == [] for row in annotations[2:])
    assert report["final_runtime"]["scene"]["entities"]  # Scene history still exists.
    video = cv2.VideoCapture(str(Path(args.output) / "annotated.mp4"))
    try:
        images = []
        while True:
            ok, image = video.read()
            if not ok:
                break
            images.append(image)
        # x=16 is the prior bbox edge; no green border on an empty source frame.
        assert images[0][76 + 60, 16].max() > 100
        assert images[2][76 + 60, 16].max() < 30
    finally:
        video.release()


def test_rendered_boxes_use_that_frames_geometry_and_show_native_and_canonical_ids(
    monkeypatch,
):
    captions = []
    put_text = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        captions.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(runner.cv2, "putText", capture_text)
    rendered = []
    for sequence, x1 in ((0, 0.1), (1, 0.6)):
        ok, encoded = cv2.imencode(".jpg", np.full((120, 160, 3), 40, dtype=np.uint8))
        assert ok
        frame = VideoFrame(sequence, sequence / 4, encoded.tobytes())
        snapshot = RichLabelSnapshot(
            "session",
            sequence,
            sequence / 4,
            (
                RichLabelObject(
                    "subject-example-17",
                    "person",
                    NormalizedBBox(x1, 0.1, x1 + 0.2, 0.9),
                    attributes={
                        "native_track": {
                            "backend": "sam3.1_video",
                            "epoch": "epoch-b",
                            "object_id": "3",
                        }
                    },
                    observation_state="observed",
                    last_observed_timestamp_seconds=sequence / 4,
                    last_observed_frame_sequence=sequence,
                ),
            ),
        )
        rendered.append(
            runner.render_native_source_frame(
                frame,
                snapshot,
                batch_number=2,
                predictor_epoch="epoch-b",
                batch_start=sequence == 0,
            )
        )
    assert rendered[0][76 + 60, 16].max() > 100
    assert rendered[0][76 + 60, 96].max() == 40
    assert rendered[1][76 + 60, 16].max() == 40
    assert rendered[1][76 + 60, 96].max() > 100
    assert "subject-example-17" in captions
    assert "Native 3" in captions
    assert any("NEW BATCH 2" in text for text in captions)
    assert any("BATCH 2" in text and "NEW" not in text for text in captions)
    assert any("cross-batch IDs are separate" in text for text in captions)
    mismatched = VideoFrame(2, 0.5, frame.jpeg_bytes)
    with pytest.raises(ValueError, match="does not match source"):
        runner.render_native_source_frame(
            mismatched,
            snapshot,
            batch_number=2,
            predictor_epoch="epoch-b",
            batch_start=False,
        )
