"""Evidence receives source frames independently of model sampling cadence."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from goodvision_harness import pipeline
from goodvision_harness.config import load_config
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.video_to_stream import LiveFrame


@pytest.mark.parametrize("source_mode", ["file", "stream"])
@pytest.mark.parametrize("archive_enabled", [False, True])
def test_evidence_source_cadence_does_not_increase_model_samples(
    monkeypatch, source_mode, archive_enabled
):
    config = load_config()
    config = replace(
        config,
        display=replace(config.display, fps=1.0),
        video=replace(config.video, fps=1.0, max_sampled_frames=None),
        events=replace(config.events, fps=5.0),
        stream=replace(config.stream, frame_crop="none"),
    )
    recorded = {}
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)

    def source(*args, **kwargs):
        fps = kwargs.get("display_fps", kwargs.get("output_fps"))
        recorded["fps"] = fps

        def frames():
            for index in range(int(2 * fps) + 1):
                yield LiveFrame(index, index / fps, pixels)

        return SimpleNamespace(realtime=False, frames=frames)

    monkeypatch.setattr(pipeline, "VideoFileStream", source)
    monkeypatch.setattr(pipeline, "NetworkStreamAdapter", source)
    archived = []
    shown = []
    archive = (
        SimpleNamespace(submit_frame=lambda *args: archived.append(args))
        if archive_enabled
        else None
    )
    client = SimpleNamespace(settings=config.inference)
    callbacks = SessionCallbacks(on_display_frame=lambda *args: shown.append(args))
    if source_mode == "file":
        session = StreamingSession.for_file(
            config, client, callbacks, video_path="unused.mp4", archive=archive
        )
    else:
        session = StreamingSession.for_stream(
            config,
            client,
            callbacks,
            stream_url="rtsp://camera.example/live",
            archive=archive,
        )

    samples = list(session._sampled_frames())
    assert recorded["fps"] == (5.0 if archive_enabled else 1.0)
    assert [frame.timestamp_seconds for frame in samples] == [0.0, 1.0, 2.0]
    assert len(shown) == (11 if archive_enabled else 3)
    assert len(archived) == (11 if archive_enabled else 0)
    if archive_enabled:
        assert [frame[1] for frame in archived] == [index / 5 for index in range(11)]
        assert all(frame[3] == session.stream_session_id for frame in archived)
