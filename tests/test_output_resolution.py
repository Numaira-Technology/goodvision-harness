"""Preserve source detail through view selection and independent outputs."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from goodvision_harness import pipeline
from goodvision_harness.config import load_config
from goodvision_harness.grounding import DryRunGroundingClient
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.video_to_stream import LiveFrame, encode_jpeg

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("crop,width,height,selected_width", [
    ("left_half", 2560, 720, 1280),
    ("right_half", 3840, 1080, 1920),
    ("none", 1920, 1080, 1920),
    ("none", 640, 480, 640),
])
def test_native_display_crops_before_every_encode(
    monkeypatch, crop, width, height, selected_width,
):
    config = load_config(ROOT / "config.yaml")
    config = replace(config, stream=replace(config.stream, frame_crop=crop))
    image = np.zeros((height, width, 3), np.uint8)
    # One-pixel stripes expose detail lost by resizing the stereo pair first.
    image[:, ::2] = 255
    encoded_shapes = []

    def checked_encode(bgr, **kwargs):
        encoded_shapes.append(bgr.shape[:2])
        assert bgr.shape[:2] == (height, selected_width)
        assert np.shares_memory(bgr, image)
        return encode_jpeg(bgr, **kwargs)

    monkeypatch.setattr(pipeline, "encode_jpeg", checked_encode)

    class Source:
        realtime = True

        def frames(self):
            yield LiveFrame(0, 0.0, image)

    displayed = []
    session = StreamingSession(
        config, DryRunGroundingClient(config.inference),
        SessionCallbacks(
            on_display_frame=lambda jpeg, *_args: displayed.append(jpeg),
        ),
        mode="stream", video_stream=Source(),
    )
    sampled = next(session._sampled_frames())
    rendered = cv2.imdecode(np.frombuffer(displayed[0], np.uint8), cv2.IMREAD_COLOR)
    assert encoded_shapes == [(height, selected_width)] * 2
    assert rendered.shape[:2] == (height, selected_width)
    assert sampled.original_dimensions == (selected_width, height)
    assert max(sampled.width, sampled.height) <= config.video.max_image_side
    # Original stripes survive on the native output, including 1080p sources.
    assert float(rendered[:, 1::2].mean()) < 10
    assert float(rendered[:, ::2].mean()) > 245


@pytest.mark.parametrize("side,expected", [(None, (1920, 1080)), (1280, (1280, 720))])
def test_display_optional_cap_preserves_aspect_ratio(tmp_path, side, expected):
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["display"]["max_image_side"] = side
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    jpeg, width, height = encode_jpeg(
        np.zeros((1080, 1920, 3), np.uint8),
        max_image_side=config.display.max_image_side,
        quality=config.display.jpeg_quality,
    )
    assert (width, height) == expected
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape[:2] == expected[::-1]


@pytest.mark.parametrize("side", [0, -1, True, "source", 1280.5])
def test_display_cap_still_rejects_invalid_limits(tmp_path, side):
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["display"]["max_image_side"] = side
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=r"display\.max_image_side"):
        load_config(path)
