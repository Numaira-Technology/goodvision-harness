"""Mixed safety + stranger sessions keep independent sampling and windows.

Regression: selecting ``stranger`` used to replace the safety sampling rate and
force one-frame safety windows with the face settings. No model is called.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundedEventClassification,
    GroundingResponse,
    ModelObject,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.service import HarnessRuntime
from goodvision_harness.sliding_window import (
    BranchCadence,
    BranchSampling,
    BranchWindowBuilder,
    SampledFrame,
    iter_sliding_windows,
)

ROOT = Path(__file__).resolve().parents[1]


def _jpeg(value: int = 0) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((36, 64, 3), value % 256, np.uint8))
    assert ok
    return encoded.tobytes()


def _frames(timestamps):
    return [
        SampledFrame(index, index * 3, ts, _jpeg(index), 64, 36)
        for index, ts in enumerate(timestamps)
    ]


def test_same_rate_branch_keeps_every_frame_and_index():
    frames = _frames([i * 0.4 for i in range(6)])
    builder = BranchWindowBuilder(BranchCadence(2.5), base_fps=2.5)
    windows = [w for w in map(builder.push, frames) if w is not None]
    assert [w.index for w in windows] == list(range(6))
    assert [w.newest_frame for w in windows] == frames
    assert builder.skipped_frames == 0


@pytest.mark.parametrize(("size", "stride"), [(4, 2), (1, 3), (2, 3), (3, 1)])
def test_branch_windows_match_shared_sliding_window_semantics(size, stride):
    frames = _frames([i * 0.2 for i in range(13)])
    builder = BranchWindowBuilder(BranchCadence(5.0, size, stride), base_fps=5.0)
    built = [w for w in map(builder.push, frames) if w is not None]
    expected = list(
        iter_sliding_windows(frames, size_frames=size, stride_frames=stride)
    )
    assert [(w.index, w.frames) for w in built] == [
        (w.index, w.frames) for w in expected
    ]


def test_slower_branch_tolerates_arrival_jitter_without_halving():
    jitter = [0.0, 0.03, -0.03, 0.02, -0.02, 0.0, 0.03, -0.03, 0.01, -0.01]
    timestamps = [max(0.0, i * 0.2 + jitter[i % 10]) for i in range(20)]
    builder = BranchWindowBuilder(BranchCadence(2.5), base_fps=5.0)
    windows = [w for w in map(builder.push, _frames(timestamps)) if w is not None]
    assert len(windows) == 10
    # Re-indexed per branch so windows stay contiguous.
    assert [w.newest_frame.sample_index for w in windows] == list(range(10))
    gaps = np.diff([w.end_timestamp for w in windows])
    assert gaps.max() < 0.61


def test_branch_sampling_rejects_base_slower_than_a_branch():
    with pytest.raises(ValueError):
        BranchSampling(2.5, BranchCadence(5.0), BranchCadence(2.5))
    with pytest.raises(ValueError):
        BranchCadence(0)


def _decoupled_config():
    config = load_config(ROOT / "config.yaml")
    return replace(
        config,
        video=replace(config.video, fps=5.0),
        sliding_window=replace(config.sliding_window, size_frames=3, stride_frames=1),
        face=replace(config.face, sample_fps=2.5),
    )


def test_selecting_stranger_keeps_safety_rate_and_window(monkeypatch):
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    runtime = HarnessRuntime(_decoupled_config())
    try:
        runtime.start_camera(
            analysis_mode="multi",
            features=["smoking", "stranger"],
            dry_run=True,
            owner="branch-sampling-test",
        )
        session = runtime._session
        assert session is not None
        sampling = session.config.branch_sampling
        assert sampling == BranchSampling(
            5.0, BranchCadence(5.0, 3, 1), BranchCadence(2.5, 1, 1)
        )
        # Camera/stream sampling runs at the faster branch; face resolution
        # stays the source for both branches (safety re-encodes it smaller).
        assert session.config.video.fps == 5.0
        assert session.config.video.max_image_side == 1280
        assert session.branch_sampling_status()["safety"]["size_frames"] == 3
        info = runtime.ui_info()["multi"]
        assert info["sample_fps"] == 5.0
        assert info["safety_sample_fps"] == 5.0
        assert info["face_sample_fps"] == 2.5
        assert info["safety_window"] == {"size_frames": 3, "stride_frames": 1}
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    ("features", "expected_fps", "expected_size"),
    [(["stranger"], 2.5, 1), (["smoking"], 5.0, 3)],
)
def test_single_branch_sessions_keep_their_existing_sampling(
    monkeypatch, features, expected_fps, expected_size,
):
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    runtime = HarnessRuntime(_decoupled_config())
    try:
        runtime.start_camera(
            analysis_mode="multi", features=features, dry_run=True, owner="single",
        )
        session = runtime._session
        assert session is not None
        assert session.config.branch_sampling is None
        assert session.config.video.fps == expected_fps
        assert session.config.sliding_window.size_frames == expected_size
    finally:
        runtime.stop()


class _Safety:
    def __init__(self, settings):
        self.settings = settings
        self.sizes: list[int] = []
        self.lock = threading.Lock()

    def classify_window(self, window):
        with self.lock:
            self.sizes.append(len(window.frames))
        return GroundingResponse(
            window,
            MultiGroundedClassification(
                (GroundedEventClassification("smoking", False, 0.9, "none"),)
            ),
            0.0,
        )


class _Face:
    def __init__(self, settings):
        self.settings = settings
        self.timestamps: list[float] = []
        self.lock = threading.Lock()

    def classify_window(self, window):
        with self.lock:
            assert len(window.frames) == 1
            self.timestamps.append(window.end_timestamp)
        objects = (
            ModelObject(
                "unknown_person",
                NormalizedBBox(0.2, 0.1, 0.4, 0.4),
                attributes={"status": "unknown"},
            ),
        )
        return GroundingResponse(
            window,
            GroundedClassification(
                True, 0.9, "face", objects,
                event_object_indexes=(0,), face_observations=objects,
            ),
            0.0,
        )


def test_live_pipeline_sends_multi_frame_safety_and_half_rate_faces():
    config = _decoupled_config()
    config = replace(
        config,
        branch_sampling=BranchSampling(
            5.0, BranchCadence(5.0, 3, 1), BranchCadence(2.5, 1, 1)
        ),
    )
    safety, face = _Safety(config.inference), _Face(config.inference)
    client = SelectedFeaturesGroundingClient(
        safety, face, safety_event_labels=("smoking",), face_live_scheduler="full",
    )
    ended = threading.Event()
    session = StreamingSession.for_camera(
        config, client, SessionCallbacks(on_end=lambda *_: ended.set()),
    )
    session.start()
    for index in range(10):
        assert session.push_camera_frame(
            _jpeg(index), width=64, height=36,
            timestamp_seconds=index * 0.2, source_frame_index=index,
        )
        time.sleep(0.01)
    assert session.camera_source is not None
    session.camera_source.close()
    assert ended.wait(5), "session did not finish"
    session.join(5)
    # Ten frames at 5 FPS: the first full 3-frame window is at frame 2.
    assert safety.sizes == [3] * 8
    assert face.timestamps == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6])
    status = session.branch_sampling_status()
    assert status["stranger"]["selected_frames"] == 5
    assert status["stranger"]["skipped_frames"] == 5
