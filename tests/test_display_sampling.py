"""Display cadence must not change model samples or queue busy inference.

Sources and model responses are local fakes; sampling, JPEG encoding, branch
scheduling and result delivery use the real pipeline. Waits are watchdogs only.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import cv2
import numpy as np
import pytest

from goodvision_harness import pipeline
from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundedEventClassification,
    GroundingResponse,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.video_to_stream import LiveFrame


class _Model:
    def __init__(self, settings, label, release=None):
        self.settings = settings
        self.label = label
        self.release = release
        self.entered = threading.Event()
        self.calls = []

    def classify_window(self, window):
        frame = window.newest_frame
        self.calls.append((
            frame.source_frame_index, frame.timestamp_seconds, frame.jpeg_bytes,
        ))
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5), "test did not release the model"
        classification = (
            GroundedClassification(False, 0.9, "No stranger.", ())
            if self.label == "stranger"
            else MultiGroundedClassification((
                GroundedEventClassification(
                    "smoking", False, 0.9, "No smoking.", (),
                ),
            ))
        )
        return GroundingResponse(window, classification, 0.001)


class _NativeSource:
    """Two seconds of distinct native 30 FPS frames, with display decimation."""

    def __init__(self, fps, *, realtime=False, started_models=()):
        self.fps = fps
        self.realtime = realtime
        self.started_models = started_models
        self.consumed = threading.Event()
        self.stopped = False

    def frames(self):
        for index in range(0, 60, round(30 / self.fps)):
            if self.stopped:
                break
            yield LiveFrame(
                index, index / 30,
                np.full((48, 64, 3), index * 3, np.uint8),
            )
            if index == 0:
                for entered in self.started_models:
                    assert entered.wait(5), "first model request did not start"
        self.consumed.set()

    def stop(self):
        self.stopped = True


def _config(display_fps, inference_fps):
    config = load_config()
    return replace(
        config,
        display=replace(config.display, fps=display_fps),
        video=replace(config.video, fps=inference_fps, max_sampled_frames=None),
        face=replace(config.face, sample_fps=inference_fps),
        stream=replace(config.stream, frame_crop="none"),
        sliding_window=replace(
            config.sliding_window,
            size_frames=1, stride_frames=1, include_partial_final_window=False,
        ),
        runtime=replace(config.runtime, max_concurrent_requests=1),
    )


def _client(config, release=None):
    safety = _Model(config.inference, "smoking", release)
    face = _Model(config.inference, "stranger", release)
    return safety, face, SelectedFeaturesGroundingClient(
        safety, face, safety_event_labels=("smoking",),
    )


@pytest.mark.parametrize("source_mode", ["file", "stream"])
@pytest.mark.parametrize("inference_fps", [2.5, 7.5])
def test_30fps_uses_original_frames_without_adding_llm_or_face_samples(
    monkeypatch, source_mode, inference_fps,
):
    runs = []
    for display_fps in (15, 30):
        config = _config(display_fps, inference_fps)
        safety, face, client = _client(config)
        sources, shown, results, endings = [], [], [], []
        finished = threading.Event()

        def source(*_args, sources=sources, **kwargs):
            fps = kwargs.get("display_fps", kwargs.get("output_fps"))
            created = _NativeSource(fps)
            sources.append(created)
            return created

        def on_end(reason, error, endings=endings, finished=finished):
            endings.append((reason, error))
            finished.set()

        monkeypatch.setattr(pipeline, "VideoFileStream", source)
        monkeypatch.setattr(pipeline, "NetworkStreamAdapter", source)
        callbacks = SessionCallbacks(
            on_display_frame=lambda *args, shown=shown: shown.append(args),
            on_analysis_result=results.append,
            on_end=on_end,
        )
        if source_mode == "file":
            session = StreamingSession.for_file(
                config, client, callbacks, video_path="unused.mp4",
            )
        else:
            session = StreamingSession.for_stream(
                config, client, callbacks, stream_url="rtsp://unused.example/live",
            )
        try:
            session.start()
            assert finished.wait(5), "session did not finish within watchdog"
            assert endings == [(
                "eof" if source_mode == "file" else "disconnected", None,
            )]
            assert sources[0].fps == display_fps
            assert len(shown) == 2 * display_fps
            assert len(results) == 4 * inference_fps
            assert len(safety.calls) == len(face.calls) == 2 * inference_fps
            expected_indices = list(range(0, 60, round(30 / inference_fps)))
            assert [call[0] for call in safety.calls] == expected_indices
            assert [call[0] for call in face.calls] == expected_indices
            assert [entry[2] for entry in shown] == list(
                range(0, 60, round(30 / display_fps)),
            )
            for jpeg, timestamp, index in shown:
                pixels = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
                assert pixels.shape == (48, 64, 3)
                assert abs(float(pixels.mean()) - index * 3) <= 1
                assert timestamp == index / 30
            runs.append((shown, safety.calls, face.calls))
        finally:
            session.stop()
            session.join(5)

    old_display, old_safety, old_face = runs[0]
    new_display, new_safety, new_face = runs[1]
    assert old_safety == new_safety  # Same source IDs, timestamps and JPEG bytes.
    assert old_face == new_face
    assert new_display[::2] == old_display
    assert len({entry[0] for entry in new_display}) == 60


def test_30fps_display_continues_while_both_model_branches_are_busy():
    config = _config(30, 2.5)
    release = threading.Event()
    safety, face, client = _client(config, release)
    source = _NativeSource(
        30, realtime=True, started_models=(safety.entered, face.entered),
    )
    shown, results, endings = [], [], []
    finished = threading.Event()

    def on_end(reason, error):
        endings.append((reason, error))
        finished.set()

    session = StreamingSession(
        config, client,
        SessionCallbacks(
            on_display_frame=lambda *args: shown.append(args),
            on_analysis_result=results.append, on_end=on_end,
        ),
        mode="stream", video_stream=source,
    )
    try:
        session.start()
        assert source.consumed.wait(5), "display source waited for model completion"
        assert len(shown) == 60
        assert [entry[2] for entry in shown] == list(range(60))
        assert results == []
        assert len(safety.calls) == len(face.calls) == 1
        assert len(session._pending) == len(session._face_pending) == 1
        release.set()
        assert finished.wait(5), "session did not finish after model release"
        assert endings == [("disconnected", None)]
        assert len(results) == 2
        assert len(safety.calls) == len(face.calls) == 1
    finally:
        release.set()
        session.stop()
        session.join(5)


@pytest.mark.parametrize("inference_fps", [2.5, 7.5])
def test_real_30fps_video_adds_native_display_frames_without_changing_samples(
    tmp_path, inference_fps,
):
    video_path = tmp_path / "native-30fps.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48),
    )
    assert writer.isOpened(), "local OpenCV MJPEG video encoder is unavailable"
    try:
        for index in range(60):
            writer.write(np.full((48, 64, 3), index * 3, np.uint8))
    finally:
        writer.release()

    runs = []
    for display_fps in (15, 30):
        config = _config(display_fps, inference_fps)
        _, _, client = _client(config)
        shown = []
        session = StreamingSession.for_file(
            config, client,
            SessionCallbacks(
                on_display_frame=lambda *args, shown=shown: shown.append(args),
            ),
            video_path=video_path, realtime=False,
        )
        try:
            samples = list(session._sampled_frames())
        finally:
            session.stop()
        assert session.video_stream.info.source_fps == 30
        assert len(shown) == display_fps * 2
        assert len(samples) == inference_fps * 2
        timestamps = [index / 30 for index in range(
            0, 60, round(30 / inference_fps),
        )]
        assert [frame.timestamp_seconds for frame in samples] == timestamps
        for jpeg, timestamp, _emitted_index in shown:
            native_index = round(timestamp * 30)
            pixels = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
            assert abs(float(pixels.mean()) - native_index * 3) <= 3
        # VideoFileStream IDs count emitted frames, so compare the actual
        # source timestamps and encoded pixels rather than those ordinal IDs.
        runs.append((
            [(timestamp, jpeg) for jpeg, timestamp, _ in shown],
            [(frame.timestamp_seconds, frame.jpeg_bytes) for frame in samples],
        ))

    assert runs[0][1] == runs[1][1]
    assert runs[1][0][::2] == runs[0][0]
    assert len({jpeg for _, jpeg in runs[1][0]}) == 60
