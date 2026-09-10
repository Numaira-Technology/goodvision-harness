from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.frame_crop import FrameCropSelector, stereo_pair_evidence
from goodvision_harness.grounding import DryRunGroundingClient
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.video_to_stream import LiveFrame

ROOT = Path(__file__).resolve().parents[1]


def textured_view(seed=17):
    rng = np.random.default_rng(seed)
    image = np.full((320, 480, 3), 35, dtype=np.uint8)
    for _ in range(250):
        center = tuple(map(int, rng.integers((18, 18), (462, 302))))
        color = tuple(map(int, rng.integers(50, 255, 3)))
        cv2.circle(image, center, int(rng.integers(2, 12)), color, -1)
    return image


def stereo_view(*, brightness=1.0, disparity=-9):
    left = textured_view()
    right = cv2.warpAffine(
        left, np.float32([[1, 0, disparity], [0, 1, 1]]), (480, 320),
    )
    right = np.clip(right.astype(float) * brightness, 0, 255).astype(np.uint8)
    return np.hstack((left, right))


@pytest.mark.parametrize("brightness", [0.7, 1.0, 1.2])
@pytest.mark.parametrize("disparity", [-18, -6, 5])
def test_detects_distributed_stereo_matches_despite_exposure_and_disparity(
    brightness, disparity,
):
    assert stereo_pair_evidence(stereo_view(
        brightness=brightness, disparity=disparity,
    )) is True


@pytest.mark.parametrize("kind", [
    "mono", "panorama", "mirror", "dark", "blank", "checkerboard", "stripes",
    "shifted_jpeg_checkerboard",
    "shared_logo", "vertical_pair", "odd_width", "identical_halves",
])
def test_does_not_crop_without_evidence_of_two_whole_horizontal_views(kind):
    left, right = textured_view(17), textured_view(21)
    if kind == "mono":
        image = left
    elif kind == "panorama":
        image = np.hstack((left, right))
    elif kind == "mirror":
        image = np.hstack((left, left[:, ::-1]))
    elif kind == "dark":
        image = stereo_view() // 30
    elif kind == "blank":
        image = np.full((320, 960, 3), 128, np.uint8)
    elif kind in {"checkerboard", "shifted_jpeg_checkerboard", "stripes"}:
        indices = np.indices((320, 480))
        pattern = (
            indices.sum(axis=0) // 16 if kind == "stripes"
            else (indices // 24).sum(axis=0)
        )
        checker = (pattern % 2 * 255).astype(np.uint8)
        other = np.roll(checker, 7, axis=1) if kind.startswith("shifted") else checker
        image = cv2.cvtColor(np.hstack((checker, other)), cv2.COLOR_GRAY2BGR)
        if kind.startswith("shifted"):
            _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    elif kind == "shared_logo":
        right[80:180, 100:240] = left[80:180, 100:240]
        image = np.hstack((left, right))
    elif kind == "vertical_pair":
        image = np.vstack((left, left))
    elif kind == "identical_halves":
        image = np.hstack((left, left))
    else:
        image = stereo_view()[:, :-1]
    selector = FrameCropSelector("auto")
    output = list(selector.frames(
        (LiveFrame(i, i * 0.2, image) for i in range(9)), stopped=lambda: False,
    ))
    assert selector.effective == "none"
    assert output and all(frame.bgr.shape == image.shape for frame in output)


def test_requires_three_frames_and_locks_coordinate_system_for_session():
    stereo, mono = stereo_view(), textured_view()
    frames = [LiveFrame(i, i * 0.2, stereo if i < 3 else mono) for i in range(5)]
    selector = FrameCropSelector("auto")
    output = list(selector.frames(iter(frames), stopped=lambda: False))
    assert output[0].index == 2
    assert output[0].timestamp_seconds == 0.4
    assert selector.status() == {
        "requested": "auto", "effective": "left_half",
        "reason": "stereo_confirmed", "probed_frames": 3,
    }
    # Later scene changes cannot switch full/half geometry under pending boxes.
    assert output[-1].bgr.shape[1] == mono.shape[1] // 2
    fresh_session = FrameCropSelector("auto")
    assert list(fresh_session.frames(
        iter([LiveFrame(0, 0.0, mono)]), stopped=lambda: False,
    ))[0].bgr.shape == mono.shape


def test_one_stereo_frame_is_not_enough_and_probe_memory_is_bounded():
    image = stereo_view()
    selector = FrameCropSelector("auto")
    output = list(selector.frames(
        iter([LiveFrame(0, 0.0, image)]), stopped=lambda: False,
    ))
    assert output[0].bgr is image
    assert selector.effective == "none"
    selector = FrameCropSelector("auto")
    # Even a broken source clock cannot hold startup indefinitely.
    frames = (LiveFrame(i, 0.0, image) for i in range(25))
    output = list(selector.frames(frames, stopped=lambda: False))
    assert output[0].index == 19
    assert len(output) == 6
    assert selector.effective == "none"


@pytest.mark.parametrize("mode,width", [("none", 960), ("left_half", 480), ("right_half", 480)])
def test_explicit_modes_are_immediate_and_authoritative(mode, width):
    image = stereo_view()
    selector = FrameCropSelector(mode)
    output = list(selector.frames(iter([LiveFrame(7, 0.0, image)]), stopped=lambda: False))
    assert len(output) == 1 and output[0].index == 7
    assert output[0].bgr.shape[1] == width
    assert selector.probes == 0


def test_stopping_during_probe_does_not_publish_retained_frame():
    stopped = False

    def frames():
        nonlocal stopped
        yield LiveFrame(0, 0.0, stereo_view())
        stopped = True

    assert list(FrameCropSelector("auto").frames(frames(), stopped=lambda: stopped)) == []


@pytest.mark.parametrize("stereo", [False, True])
def test_auto_selection_precedes_display_inference_and_evidence(stereo):
    config = load_config(ROOT / "config.yaml")
    image = stereo_view() if stereo else textured_view()
    config = replace(
        config, video=replace(config.video, max_image_side=960),
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
