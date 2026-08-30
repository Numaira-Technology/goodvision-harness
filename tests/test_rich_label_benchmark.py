from __future__ import annotations

from pathlib import Path

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import RichLabelObject
from tools.rich_label_benchmark import (
    TrackMetrics,
    _safe_endpoint,
    advance_sample_deadline,
    build_parser,
    percentile,
    summarize_latencies,
)


def _object(track_id: str, label: str = "person") -> RichLabelObject:
    return RichLabelObject(
        track_id=track_id,
        label=label,
        bbox=NormalizedBBox(0.1, 0.1, 0.5, 0.8),
        confidence=0.9,
    )


def test_percentile_uses_linear_interpolation_without_mutating_input() -> None:
    values = [40.0, 10.0, 30.0, 20.0]

    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.95) == pytest.approx(38.5)
    assert percentile(values, 0.99) == pytest.approx(39.7)
    assert percentile(values, 1.0) == 40.0
    assert values == [40.0, 10.0, 30.0, 20.0]


def test_latency_summary_is_stable_for_empty_and_populated_samples() -> None:
    assert summarize_latencies([]) == {
        "count": 0,
        "min": None,
        "mean": None,
        "p50": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    assert summarize_latencies([10.0, 20.0, 30.0, 40.0]) == {
        "count": 4,
        "min": 10.0,
        "mean": 25.0,
        "p50": 25.0,
        "p95": 38.5,
        "p99": 39.7,
        "max": 40.0,
    }


@pytest.mark.parametrize(
    "values",
    [
        [-1.0],
        [float("nan")],
        [float("inf")],
    ],
)
def test_latency_summary_rejects_invalid_samples(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize_latencies(values)


def test_sample_deadline_selects_source_frames_without_drift() -> None:
    deadline = 0.0
    selected: list[int] = []
    for frame_sequence in range(13):
        due, deadline = advance_sample_deadline(
            frame_sequence / 30.0,
            deadline,
            7.5,
        )
        if due:
            selected.append(frame_sequence)

    assert selected == [0, 4, 8, 12]
    assert deadline == pytest.approx(16 / 30)


def test_sample_deadline_tolerance_never_repeats_a_deadline() -> None:
    deadline = 0.0
    selected: list[int] = []
    for frame_sequence in range(795):
        due, deadline = advance_sample_deadline(
            frame_sequence / 10.0,
            deadline,
            7.5,
        )
        if due:
            selected.append(frame_sequence)

    assert len(selected) == 596
    assert selected[0] == 0
    assert selected[-1] == 794
    assert deadline == pytest.approx(596 / 7.5)


def test_track_metrics_count_rendered_duration_and_contiguous_runs() -> None:
    tracks = TrackMetrics()
    person = _object("P0001")
    bottle = _object("O0002", "bottle")

    tracks.observe_detection((person,))
    tracks.observe_rendered_frame(10, 1.0, (person,))
    tracks.observe_detection((person, bottle))
    tracks.observe_rendered_frame(11, 1.1, (person, bottle))
    tracks.observe_rendered_frame(12, 1.2, (bottle,))
    tracks.observe_detection((person,))
    tracks.observe_rendered_frame(14, 1.4, (person,))

    result = tracks.as_dict()
    assert list(result) == ["O0002", "P0001"]
    assert result["P0001"] == {
        "label": "person",
        "duration_frames": 3,
        "longest_contiguous_frames": 2,
        "detector_observations": 3,
        "first_frame_sequence": 10,
        "last_frame_sequence": 14,
        "first_timestamp_seconds": 1.0,
        "last_timestamp_seconds": 1.4,
    }
    assert result["O0002"]["duration_frames"] == 2
    assert result["O0002"]["longest_contiguous_frames"] == 2
    assert result["O0002"]["detector_observations"] == 1


def test_parser_exposes_reproducible_defaults() -> None:
    arguments = build_parser().parse_args(
        ["source.mp4", "--output-dir", "benchmark-output"]
    )

    assert arguments.input_video == Path("source.mp4")
    assert arguments.output_dir == Path("benchmark-output")
    assert arguments.endpoint == "http://127.0.0.1:8091/v1/labels"
    assert arguments.sample_fps == 7.5
    assert arguments.timeout_seconds == 30.0
    assert arguments.max_image_side == 720
    assert arguments.jpeg_quality == 70
    assert arguments.max_frames is None
    assert arguments.fourcc == "mp4v"
    assert arguments.overwrite is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["video.mp4"],
        ["video.mp4", "--output-dir", "out", "--sample-fps", "0"],
        ["video.mp4", "--output-dir", "out", "--timeout-seconds", "nan"],
        ["video.mp4", "--output-dir", "out", "--jpeg-quality", "101"],
        ["video.mp4", "--output-dir", "out", "--max-frames", "-1"],
        ["video.mp4", "--output-dir", "out", "--endpoint", "file:///tmp/x"],
        ["video.mp4", "--output-dir", "out", "--fourcc", "too-long"],
        ["video.mp4", "--output-dir", "out", "--session-id", " bad "],
    ],
)
def test_parser_rejects_invalid_parameters(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_metrics_endpoint_removes_credentials_query_and_fragment() -> None:
    assert _safe_endpoint("https://example.test:8443/v1/labels?token=x#debug") == (
        "https://example.test:8443/v1/labels"
    )
