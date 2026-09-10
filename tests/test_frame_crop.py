from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import pytest
from goodvision_harness.config import load_config
from goodvision_harness.frame_crop import FrameCropSelector
from goodvision_harness.grounding import DryRunGroundingClient
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.video_to_stream import LiveFrame

ROOT = Path(__file__).resolve().parents[1]


def textured_view():
    return np.full((320, 480, 3), (40, 90, 160), np.uint8)


def stereo_view():
    return np.hstack((textured_view(), np.full((320, 480, 3), 230, np.uint8)))


@pytest.mark.parametrize("mode,start,end", [
    ("none", 0, 960), ("left_half", 0, 480), ("right_half", 480, 960),
    ("auto", 0, 960),
])
def test_manual_modes_publish_first_frame_without_guessing(mode, start, end):
    image = stereo_view()
    selector = FrameCropSelector(mode)
    frames = [LiveFrame(i, i * 0.2, image) for i in range(8)]
    output = list(selector.frames(iter(frames), stopped=lambda: False))
    assert [f.index for f in output] == list(range(8))
    assert [f.timestamp_seconds for f in output] == [f.timestamp_seconds for f in frames]
    assert selector.probes == 0
    for frame in output:
        np.testing.assert_array_equal(frame.bgr, image[:, start:end])
    if mode == "auto":
        assert selector.status()["reason"] == "legacy_auto_disabled"
        assert selector.effective == "none"


def test_manual_crop_honors_stop_without_publishing_more_frames():
    stopped = False
    def frames():
        nonlocal stopped
        yield LiveFrame(0, 0.0, stereo_view())
        stopped = True
        yield LiveFrame(1, 0.2, stereo_view())
    output = list(FrameCropSelector("left_half").frames(frames(), stopped=lambda: stopped))
    assert [f.index for f in output] == [0]


@pytest.mark.parametrize("stereo", [False, True])
def test_manual_selection_precedes_display_inference_and_evidence(stereo):
    config = load_config(ROOT / "config.yaml")
    image = stereo_view() if stereo else textured_view()
    config = replace(
        config, stream=replace(config.stream, frame_crop="left_half" if stereo else "none"),
        video=replace(config.video, max_image_side=960),
        display=replace(config.display, max_image_side=960),
    )

    class Source:
        realtime = True

        def frames(self):
            return (LiveFrame(i, i * 0.2, image) for i in range(9))

    class Archive:
        def submit_frame(self, jpeg, timestamp, index, session_id):
            evidence.append((jpeg, timestamp, index))

    displayed, evidence = [], []
    session = StreamingSession(
        config, DryRunGroundingClient(config.inference),
        SessionCallbacks(on_display_frame=lambda jpg, ts, idx: displayed.append((jpg, ts, idx))),
        mode="stream", video_stream=Source(), archive=Archive(),
    )
    samples = list(session._sampled_frames())
    expected_width = image.shape[1] // 2 if stereo else image.shape[1]
    assert samples and all(s.original_dimensions == (expected_width, 320) for s in samples)
    assert displayed == evidence
    for jpeg, _, _ in displayed:
        assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[1] == expected_width
