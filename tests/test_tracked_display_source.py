"""Delayed presentation reads exact completed originals without changing clocks."""

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime

BOX = NormalizedBBox(0.1, 0.1, 0.5, 0.9)
NATIVE = {"backend": "sam3.1_video", "epoch": "e", "object_id": "0"}


def runtime():
    scene = SceneRuntime(
        "s", "camera", tracking_mode="batch", mode="active", event_labels=()
    )
    for sequence, at in ((0, 0.0), (3, 1.0), (8, 2.0)):
        scene.observe_frame(f"source-{sequence}".encode(), at, sequence)
    frames = tuple(
        RichLabelSnapshot(
            "s",
            sequence,
            at,
            (
                RichLabelObject(
                    "person", "person", BOX, attributes={"native_track": NATIVE}
                ),
            )
            if sequence != 8
            else (),
        )
        for sequence, at in ((0, 0.0), (3, 1.0), (8, 2.0))
    )
    assert scene.observe_tracking_batch(
        frames, completed_timestamp_seconds=2.0, completed_frame_sequence=8
    )
    scene.observe_frame(b"future", 5.0, 100)
    return scene


def test_display_selection_exact_source_and_monotonic_sequence_without_clock_rewind():
    scene = runtime()
    before = scene.snapshot()
    frame, view = scene.tracked_display_frame(before_timestamp_seconds=1.5)
    assert (frame.frame_sequence, frame.timestamp_seconds, frame.jpeg_bytes) == (
        3,
        1.0,
        b"source-3",
    )
    assert view["timestamp_seconds"] == 1.0
    assert view["capture_timestamp_seconds"] == 5.0
    assert view["tracking_completed_timestamp_seconds"] == 2.0
    assert view["entities"][0]["bbox"] == BOX.as_dict()
    assert (
        scene.tracked_display_frame(
            before_timestamp_seconds=1.5, after_frame_sequence=3
        )
        is None
    )
    assert scene.snapshot() == before


def test_empty_completed_source_does_not_borrow_a_previous_person_box():
    scene = runtime()
    frame, view = scene.tracked_display_frame(before_timestamp_seconds=5.0)
    assert frame.jpeg_bytes == b"source-8"
    assert all(item["observation_state"] == "occluded" for item in view["entities"])


def test_expired_or_closed_completed_source_is_unavailable():
    scene = runtime()
    scene.observe_frame(b"future", 20.0, 200)
    assert scene.tracked_display_frame(before_timestamp_seconds=20.0) is None
    scene.close()
    assert scene.tracked_display_frame(before_timestamp_seconds=20.0) is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_display_timestamp_is_rejected(value):
    with pytest.raises(ValueError):
        runtime().tracked_display_frame(before_timestamp_seconds=value)
