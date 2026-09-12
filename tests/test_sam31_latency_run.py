"""Clock/EOF statistics and source previews using protocol doubles only."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    return selected


ROOT = Path(__file__).resolve().parents[1]
runner = module("latency_runner", ROOT / "tools/sam31_latency_run.py")
doubles = module(
    "session_double", Path(__file__).with_name("test_video_tracking_session.py")
)


def video(tmp_path, count=3):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (160, 120))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((120, 160, 3), (20 + index * 40, 50, 90), np.uint8))
    writer.release()
    return path


def test_live_lag_uses_periodic_samples_and_excludes_eof_zero():
    rows = [
        {"kind": "clock", "during_capture": True, "tracking_lag_seconds": lag}
        for lag in (None, 4, 8)
    ]
    rows += [
        {"kind": "capture", "during_capture": True, "tracking_lag_seconds": 0}
    ] * 100
    rows += [
        {"kind": "clock", "during_capture": False, "tracking_lag_seconds": 0}
    ] * 100
    tracks = [
        {
            "native": {"frame_sequence": 1},
            "during_capture": True,
            "source_to_result_seconds": 9,
        },
        {
            "native": {"frame_sequence": 3},
            "during_capture": False,
            "source_to_result_seconds": 12,
        },
    ]
    summary = runner.summarize_observations(rows, tracks, 4)
    assert summary["live_tracking_lag_seconds"] == {
        "samples": 2,
        "peak": 8,
        "p50": 6.0,
        "p95": pytest.approx(7.8),
    }
    assert summary["live_lag_sampling"]["samples_without_output"] == 1
    assert summary["tracking_coverage"]["unprocessed_source_frames"] == 2
    assert summary["tracking_coverage"]["completed_after_capture"] == 1
    assert summary["source_to_result_seconds"]["after_capture"]["p50"] == 12


@pytest.mark.parametrize(
    "fps,duration",
    [(True, 1), (4, True), (float("nan"), 1), (4, float("inf")), (0.49, 1), (4, 61)],
)
def test_source_rejects_unbounded_numeric_inputs(fps, duration):
    with pytest.raises(ValueError):
        runner.load_source("unused", fps=fps, max_seconds=duration)


def test_short_eof_with_real_coordinator_closes_client_and_annotations(
    tmp_path, monkeypatch
):
    source = video(tmp_path, count=2)
    server = doubles.SessionServer()
    client_class = runner.MeasuredClient
    clients = []

    def client(endpoint, **kwargs):
        selected = client_class(
            endpoint, transport=httpx.MockTransport(server), **kwargs
        )
        clients.append(selected)
        return selected

    monkeypatch.setattr(runner, "MeasuredClient", client)
    args = SimpleNamespace(
        video=str(source),
        output=str(tmp_path / "run"),
        fps=4,
        max_seconds=0.5,
        sam_endpoint="http://mock.test/v1/video/sessions",
        batch_frames=4,
        batch_seconds=0.5,
        transport_max_side=128,
        transport_jpeg_quality=65,
        annotation=True,
    )
    result = runner.run(args)
    assert result["ok"] and result["finished_idle"]
    assert result["capture_frames"] == 2
    assert result["tracking_coverage"]["all_capture_frames_processed"]
    assert result["tracking_coverage"]["completed_after_capture"] == 2
    assert result["live_tracking_lag_seconds"]["samples"] == 0
    assert result["eof_drain_seconds"] > 0
    assert result["source_evidence_unchanged"]
    assert result["annotation"]["rendered_after_measurement"]
    assert result["annotation"]["frames"] == 2
    assert result["requests"][0]["final"] and result["requests"][0]["succeeded"]
    assert clients[0]._client.is_closed
    saved = json.loads((Path(args.output) / "report.json").read_text())
    assert saved["elapsed_seconds"] == result["elapsed_seconds"]


def test_unprocessed_preview_keeps_every_source_and_never_copies_boxes(tmp_path):
    source = video(tmp_path, count=3)
    frames, mapping, fps = runner.load_source(source, fps=4, max_seconds=1)
    output = tmp_path / "preview"
    output.mkdir()
    result = runner.render_annotations(frames, mapping, [], output=output, fps=fps)
    assert result["frames"] == result["not_processed_frames"] == 3
    rows = [
        json.loads(line)
        for line in (output / "annotation-map.jsonl").read_text().splitlines()
    ]
    assert [row["source_frame_sequence"] for row in rows] == [0, 1, 2]
    assert all(
        row["state"] == "not_processed" and not row["subject_ids"] for row in rows
    )
    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
    assert capture.get(cv2.CAP_PROP_FPS) == 4
    capture.release()


def test_annotation_rejects_changed_source_evidence_before_writing(tmp_path):
    source = video(tmp_path, count=1)
    frames, mapping, fps = runner.load_source(source, fps=4, max_seconds=1)
    mapping[0]["source_sha256"] = "wrong"
    output = tmp_path / "preview"
    output.mkdir()
    with pytest.raises(ValueError, match="source hash"):
        runner.render_annotations(frames, mapping, [], output=output, fps=fps)
    assert not (output / "annotated.mp4").exists()
