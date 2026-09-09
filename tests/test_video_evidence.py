from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from goodvision_harness.video_evidence import (
    EvidenceFrame,
    FrameRing,
    encode_staged_clip,
    stage_clip,
)


def test_ring_bounds_bytes_and_never_splices_an_outage() -> None:
    ring = FrameRing(90, 100)
    for index in range(12):
        ring.append(EvidenceFrame(b"x" * 10, index / 5, index))
    assert ring.bytes_used == 100
    assert ring.evicted == 2
    ring.append(EvidenceFrame(b"x" * 10, 8, 40))
    frames, reason = ring.select(0, 9, 8, 5)
    assert [f.timestamp_seconds for f in frames] == [8]
    assert reason == "source_gap"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="video encoder not installed"
)
def test_real_mp4_has_three_seconds_five_fps_and_even_dimensions(
    tmp_path: Path,
) -> None:
    frames = []
    for index in range(15):
        image = np.full((91, 161, 3), index * 10, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        frames.append(EvidenceFrame(encoded.tobytes(), index / 5, index))
    manifest = stage_clip(
        tmp_path / "stage",
        frames,
        {
            "event_id": "test",
            "revision": 1,
            "fps": 5,
            "requested_start_seconds": 0,
            "requested_end_seconds": 3,
            "partial_reason": None,
        },
    )
    destination = tmp_path / "video.mp4"
    metadata = encode_staged_clip(manifest, destination)
    assert metadata["duration_seconds"] == 3
    assert metadata["fps"] == 5
    assert metadata["frame_count"] == 15
    assert metadata["width"] == 160 and metadata["height"] == 90
    assert destination.read_bytes()[4:8] == b"ftyp"
    capture = cv2.VideoCapture(str(destination))
    try:
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, 14)
        ok, frame = capture.read()
        assert ok and frame.mean() > 100
    finally:
        capture.release()
