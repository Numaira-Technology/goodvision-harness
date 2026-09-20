"""Finite recording review with protocol doubles; no real model or GPU calls."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.video_tracking_session import VideoSessionTrackingHTTPClient


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    return selected


ROOT = Path(__file__).resolve().parents[1]
runner = module("continuous_recording_runner", ROOT / "tools" / "sam31_session_run.py")
doubles = module(
    "continuous_protocol_double",
    Path(__file__).with_name("test_video_tracking_session.py"),
)


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def create(count=26):
        video = tmp_path / "source.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (320, 240)
        )
        assert writer.isOpened()
        for sequence in range(count):
            image = np.full((240, 320, 3), (sequence * 5 % 200, 60, 120), np.uint8)
            cv2.rectangle(image, (60, 60), (140, 210), (140, 140, 140), -1)
            writer.write(image)
        writer.release()
        server = doubles.SessionServer()
        clients = []

        def client(endpoint, **kwargs):
            selected = VideoSessionTrackingHTTPClient(
                endpoint, transport=httpx.MockTransport(server), **kwargs
            )
            clients.append(selected)
            return selected

        monkeypatch.setattr(runner, "VideoSessionTrackingHTTPClient", client)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("no-inference must not load model credentials")

        monkeypatch.setattr(runner._helpers, "load_config", forbidden)
        args = SimpleNamespace(
            video=str(video),
            output=str(tmp_path / "output"),
            sam_endpoint="http://mock.test/v1/video/sessions",
            config="unused.yaml",
            events="smoking",
            fps=4,
            batch_frames=16,
            max_seconds=60,
            timeout_seconds=2,
            max_requests=2,
            with_inference=False,
            transport_max_side=None,
            transport_jpeg_quality=None,
        )
        return args, server, clients

    return create


@pytest.mark.parametrize(
    "count,append_sizes,emitted_sizes",
    [(26, [16, 10], [2, 24]), (32, [16, 16, 0], [2, 16, 14])],
)
def test_delayed_recording_outputs_every_original_once_with_stable_subject(
    setup, count, append_sizes, emitted_sizes
):
    args, server, clients = setup(count)
    result = runner.run(args)
    output = Path(args.output)
    assert result["ok"] and result["samples"] == count
    assert [len(request["frames"]) for request in server.requests] == append_sizes
    assert [
        len(row["emitted_frame_sequences"]) for row in records(output / "appends.jsonl")
    ] == emitted_sizes
    tracks = records(output / "tracks.jsonl")
    assert [row["raw"]["frame_sequence"] for row in tracks] == list(range(count))
    assert (
        len({obj["track_id"] for row in tracks for obj in row["canonical"]["objects"]})
        == 1
    )
    assert len({row["predictor_epoch"] for row in tracks}) == 1
    annotation = records(output / "annotations.jsonl")
    assert [row["source_frame_sequence"] for row in annotation] == list(range(count))
    assert [row["source_timestamp_seconds"] for row in annotation] == [
        index / 4 for index in range(count)
    ]
    assert sum(row["session_start"] for row in annotation) == 1
    assert all(row["memory_scope"] == "session" for row in annotation)
    assert all("batch_start" not in row for row in annotation)
    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) == count
    ok, image = capture.read()
    capture.release()
    assert ok and image.shape[:2] == (316, 680)
    assert result["final_watermarks"]["emitted_frame_sequence"] == count - 1
    assert (
        result["pending_frame_count"]
        == result["pending_originals"]["remaining_frames"]
        == 0
    )
    assert result["pending_originals"]["peak_frames"] <= 30
    assert result["continuous_camera_memory"] is True
    assert result["mask_accuracy_measured"] is False
    assert result["model_requests"] == 0
    assert clients[0]._client.is_closed
    assert len(server.deletes) == 1
    assert "jpeg_base64" not in (output / "tracks.jsonl").read_text()


def test_one_frame_final_is_a_real_source_not_repeated_warmup(setup):
    args, server, _clients = setup(1)
    result = runner.run(args)
    assert result["samples"] == result["annotated_video"]["frames"] == 1
    assert len(server.requests) == 1 and server.requests[0]["is_last_batch"] is True
    assert len(server.requests[0]["frames"]) == 1
    assert result["continuous_camera_memory"] is False
    assert result["cross_append_identity_continuity"] is False


def test_explicit_transport_changes_only_model_copy_not_annotation_original(
    setup, monkeypatch
):
    args, server, _clients = setup(26)
    args.transport_max_side, args.transport_jpeg_quality = 160, 70
    originals = []
    real_render = runner.render_session_source_frame

    def capture_original(frame, snapshot, **kwargs):
        originals.append((frame.frame_sequence, frame.jpeg_bytes))
        return real_render(frame, snapshot, **kwargs)

    monkeypatch.setattr(runner, "render_session_source_frame", capture_original)
    report = runner.run(args)
    assert tuple(report["transport"]["source_dimensions"]) == (320, 240)
    assert tuple(report["transport"]["transport_dimensions"]) == (160, 120)
    assert (
        report["transport"]["transport_jpeg_bytes"]
        < report["transport"]["original_jpeg_bytes"]
    )
    sources = dict(originals)
    for request in server.requests:
        for record in request["frames"]:
            jpeg = base64.b64decode(record["jpeg_base64"])
            assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[
                :2
            ] == (120, 160)
            original = sources[record["frame_sequence"]]
            assert cv2.imdecode(
                np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR
            ).shape[:2] == (240, 320)
            assert jpeg != original


def test_default_transport_preserves_original_jpeg_bytes(setup, monkeypatch):
    args, server, _clients = setup(1)
    originals = []
    real_render = runner.render_session_source_frame

    def capture_original(frame, snapshot, **kwargs):
        originals.append(frame.jpeg_bytes)
        return real_render(frame, snapshot, **kwargs)

    monkeypatch.setattr(runner, "render_session_source_frame", capture_original)
    result = runner.run(args)
    assert result["transport"]["enabled"] is False
    assert originals == [
        base64.b64decode(server.requests[0]["frames"][0]["jpeg_base64"])
    ]


def test_existing_output_and_invalid_setting_do_not_contact_worker(setup):
    args, server, clients = setup()
    Path(args.output).mkdir()
    sentinel = Path(args.output) / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert sentinel.read_text() == "keep" and clients == []
    args.output += "-other"
    args.transport_jpeg_quality = 101
    with pytest.raises(ValueError):
        runner.run(args)
    assert server.requests == [] and clients == []


def test_protocol_error_aborts_and_discards_session_without_false_success(setup):
    args, server, _clients = setup()
    server.mutate = lambda payload: payload.update(
        predictor_epoch="private-wrong-epoch"
    )
    with pytest.raises(RichLabelProtocolError):
        runner.run(args)
    output = Path(args.output)
    assert not (output / "report.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["continuous_session_aborted"] is True
    assert failure["annotated_frames"] == 0
    assert "private" not in (output / "failure.json").read_text()
    assert len(server.deletes) == 1


@pytest.mark.parametrize("budget", [1, 2])
def test_explicit_event_model_budget_keeps_source_confirmations_independent(
    setup, monkeypatch, budget
):
    args, _server, _clients = setup(32)
    args.with_inference = True
    args.max_requests = budget
    calls = []

    class Inspector:
        def inspect(self, job):
            calls.append(job)
            subjects = tuple(
                sorted({subject for _seq, subject, _box in job.subject_boxes})
            )
            return (TemporalVerdict("smoking", subjects, True, 0.95),)

        def stats(self):
            return {"requests": len(calls)}

        def close(self):
            pass

    monkeypatch.setattr(runner, "_make_inspector", lambda _args: Inspector())
    report = runner.run(args)
    assert report["model_requests"] == len(calls) == budget
    assert report["model_failures"] == 0
    events = report["final_runtime"]["scene"]["events"]
    assert any(
        event["phase"] == ("confirmed" if budget == 2 else "suspected")
        for event in events
    )
    if budget == 2:
        first = {frame.frame_sequence for frame in calls[0].frames}
        second = {frame.frame_sequence for frame in calls[1].frames}
        assert first.isdisjoint(second)
