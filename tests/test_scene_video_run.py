"""Replay a real tiny recording without model traffic or claims of SAM validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scene_video_run_test_module", ROOT / "tools" / "scene_video_run.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "recorded-track dry validation must not call models or SAM"
        )

    monkeypatch.setattr(runner, "TemporalVLMInspector", forbidden)
    monkeypatch.setattr(runner, "RichLabelHTTPClient", forbidden)
    video = tmp_path / "private-recording.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 120))
    assert writer.isOpened(), "test environment needs the normal OpenCV MJPEG encoder"
    records = []
    try:
        for sequence in range(8):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(
                frame, (20 + sequence, 15), (60 + sequence, 108), (120, 120, 120), -1
            )
            writer.write(frame)
            records.append(
                RichLabelSnapshot(
                    "recorded-session",
                    sequence,
                    sequence / 8,
                    (
                        RichLabelObject(
                            "recorded-person",
                            "person",
                            NormalizedBBox(0.1, 0.1, 0.5, 0.95),
                            0.99,
                        ),
                    ),
                ).as_dict()
            )
    finally:
        writer.release()
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    return SimpleNamespace(
        video=str(video),
        config=str(ROOT / "config.yaml"),
        output=str(tmp_path / "output"),
        recorded_tracks=str(tracks),
        sam_endpoint=None,
        no_inference=True,
        events="smoking",
        fps=4,
        max_seconds=3,
        max_requests=4,
    )


def test_recorded_no_inference_produces_scene_json_and_annotated_video(inputs):
    report = runner.run(inputs)
    output = Path(inputs.output)
    stored = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(report)) == stored
    assert report["perception_source"] == "recorded_tracks"
    assert report["sam_validated"] is False
    assert report["accuracy_measured"] is False
    assert report["model_enabled"] is False
    assert report["model_requests"] == report["model_failures"] == 0
    assert report["samples"] == 4
    states = [
        json.loads(line)
        for line in (output / "scene.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(states) == 4
    entities = states[-1]["scene"]["entities"]
    assert len(entities) == 1
    assert entities[0]["subject_id"] != "recorded-person"
    assert len({state["scene"]["entities"][0]["subject_id"] for state in states}) == 1
    assert all(state["scene"]["events"] == [] for state in states)
    assert all(
        "jpeg" not in str(state) and "base64" not in str(state) for state in states
    )
    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        ok, first = capture.read()
        assert ok and first.shape[:2] == (120, 160)
        assert np.any(first[:, :, 1] > first[:, :, 2] + 40)
    finally:
        capture.release()


def test_output_directory_is_never_overwritten(inputs):
    runner.run(inputs)
    output = Path(inputs.output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        runner.run(inputs)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_recorded_timestamp_must_match_actual_source_frame(inputs):
    path = Path(inputs.recorded_tracks)
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["timestamp_seconds"] = 100
    path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="timestamp"):
        runner.run(inputs)
    assert not (Path(inputs.output) / "report.json").exists()


def test_duplicate_recorded_source_sequence_is_rejected_before_output(inputs):
    path = Path(inputs.recorded_tracks)
    content = path.read_text(encoding="utf-8")
    path.write_text(content + "\n" + content.splitlines()[0], encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        runner.run(inputs)
    assert not Path(inputs.output).exists()


def test_cli_no_inference_path_is_usable_and_does_not_leak_source_names(
    inputs, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scene_video_run.py",
            inputs.video,
            "--config",
            inputs.config,
            "--recorded-tracks",
            inputs.recorded_tracks,
            "--no-inference",
            "--output",
            inputs.output,
            "--fps",
            "4",
        ],
    )
    assert runner.main() == 0
    summary = capsys.readouterr().out
    assert json.loads(summary)["perception_source"] == "recorded_tracks"
    assert "private-recording" not in summary
    assert "recorded-person" not in summary


@pytest.mark.parametrize(
    "flag,value", [("--fps", "nan"), ("--max-seconds", "inf"), ("--max-requests", "0")]
)
def test_cli_rejects_unbounded_work(inputs, monkeypatch, flag, value):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scene_video_run.py",
            inputs.video,
            "--recorded-tracks",
            inputs.recorded_tracks,
            "--no-inference",
            "--output",
            inputs.output,
            flag,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not Path(inputs.output).exists()
