"""Selection must preserve the frame on which detector identities were observed."""

import io

import pytest
import test_vlm_detector_preview as detector
from PIL import Image

from tools.vlm_benchmark.monitor import LiveMonitor

detector_monitor = detector.monitor
worker = detector.worker


class Frames(list):
    media_seconds = (0.0, 0.4, 0.8, 1.2)
    media_duration = 1.6


def jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def frames():
    return Frames([jpeg(color) for color in ("red", "blue", "yellow", "white")])


def test_source_lookup_uses_actual_nonuniform_pts_and_rounding_tolerance():
    monitor = object.__new__(LiveMonitor)
    source = frames()
    source.media_seconds = (0.0, 0.7, 1.0, 1.3)
    monitor.framesets = [source]
    key = ("viewer", 0)
    assert monitor._hires_frame(key, 0.7, 0) == source[1]
    assert monitor._hires_frame(key, 0.70005, 0) == source[1]
    assert monitor._hires_frame(key, 0.69995, 0) == source[1]
    assert monitor._hires_frame(key, 0.7002, 0) is None
    assert monitor._hires_frame(key, 0.6, 0) is None
    assert monitor._hires_frame(key, 0.71, 0) is None


@pytest.mark.parametrize(
    "at,epoch,expected",
    [
        (0.4, 0, 1),
        (2.0, 1, 1),
        (3.6, 2, 1),
        (1.55, 0, None),  # Tail cannot wrap to the next loop's first frame.
        (1.6, 0, None),
        (1.6, 1, 0),
        (1.59, 1, None),
        (3.2, 1, None),
        (-0.1, 0, None),
        (float("nan"), 0, None),
        (float("inf"), 0, None),
    ],
)
def test_source_lookup_preserves_loop_and_tail_boundaries(at, epoch, expected):
    monitor = object.__new__(LiveMonitor)
    source = frames()
    monitor.framesets = [source]
    actual = monitor._hires_frame(("viewer", 0), at, epoch)
    assert actual == (source[expected] if expected is not None else None)


@pytest.mark.parametrize("metadata", ["missing", "incomplete", "duration"])
def test_missing_exact_source_metadata_cannot_infer_a_sample(metadata):
    monitor = object.__new__(LiveMonitor)
    source = frames()
    if metadata == "missing":
        source = list(source)
    elif metadata == "incomplete":
        source.media_seconds = (0.0, 0.4)
    else:
        source.media_duration = None
    monitor.framesets = [source]
    assert monitor._hires_frame(("viewer", 0), 0.4, 0) is None


@pytest.mark.parametrize(
    "at,epoch,expected",
    [(0.4, 0, 1), (2.0, 1, 1), (0.1, 0, None), (1.55, 0, None)],
)
def test_selection_receives_source_aligned_image_or_original_capture(
    detector_monitor, worker, monkeypatch, at, epoch, expected
):
    monitor = detector_monitor
    source = frames()
    monitor.framesets = [source]
    monitor._select_events = True
    captured = jpeg("green")  # Distinct from every stored source frame.
    selections = []
    monkeypatch.setattr(monitor, "_maybe_select", lambda *args: selections.append(args))
    result = monitor._yolo_track(
        ("viewer", 0),
        at,
        captured,
        [],
        False,
        stage=monitor.state["stage"],
        epoch=epoch,
        evidence_at=100.0,  # Arrival clock must affect TTL, never source lookup.
    )
    assert result["status"] == "tracking"
    assert len(selections) == 1
    assert selections[0][3] == 100.0
    assert selections[0][4] == (
        source[expected] if expected is not None else captured
    )
    assert selections[0][5] == [
        ("actor", (0.1, 0.1, 0.4, 0.9)),
        ("bystander", (0.6, 0.1, 0.9, 0.9)),
    ]
