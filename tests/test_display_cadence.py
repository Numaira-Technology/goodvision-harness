from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from goodvision_harness import stream_adapter, video_to_stream
from goodvision_harness.config import DisplaySettings, load_config
from goodvision_harness.stream_adapter import NetworkStreamAdapter
from goodvision_harness.video_to_stream import VideoFileStream, VideoInfo

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("protocol", ["rtsp", "webrtc"])
@pytest.mark.parametrize("source_fps", [30, 60])
def test_network_display_preserves_cadence_despite_arrival_jitter(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    source_fps: int,
) -> None:
    arrivals = [
        1000.0 + index / source_fps + (0.002 if index % 2 else 0.0)
        for index in range(source_fps)
    ]
    frames, adapter = _network_frames(monkeypatch, protocol, arrivals)

    assert adapter.output_fps == 30
    assert len(frames) == 30
    assert [frame.bgr for frame in frames] == list(
        range(0, source_fps, source_fps // 30)
    )
    assert frames[-1].timestamp_seconds == pytest.approx(
        arrivals[-source_fps // 30] - arrivals[0]
    )


@pytest.mark.parametrize("protocol", ["rtsp", "webrtc"])
def test_network_source_resumes_immediately_after_gap_without_catchup(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
) -> None:
    arrivals = [1000.0, 1000.034, 1010.0, 1010.001, 1010.033, 1010.067]
    frames, adapter = _network_frames(monkeypatch, protocol, arrivals)

    assert [frame.bgr for frame in frames] == [0, 1, 2, 4, 5]
    assert frames[2].timestamp_seconds == 10.0
    assert adapter.dropped_frames == 1


def _network_frames(monkeypatch, protocol: str, arrivals: list[float]):
    url = "rtsp://camera/live" if protocol == "rtsp" else "webrtc+http://camera/whep"
    adapter = NetworkStreamAdapter(url)
    clock = iter(arrivals)
    monkeypatch.setattr(
        stream_adapter, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    images = iter(range(len(arrivals)))

    if protocol == "rtsp":

        class Capture:
            def isOpened(self):
                return True

            def set(self, *_):
                return True

            def release(self):
                pass

            def read(self):
                try:
                    return True, next(images)
                except StopIteration:
                    adapter.stop()
                    return False, None

        monkeypatch.setattr(stream_adapter.cv2, "VideoCapture", lambda _: Capture())
        return list(adapter.frames()), adapter

    class EndOfStream(Exception):
        pass

    class Track:
        async def recv(self):
            try:
                image = next(images)
            except StopIteration as exc:
                raise EndOfStream from exc
            return SimpleNamespace(to_ndarray=lambda **_: image)

    frames = []
    monkeypatch.setattr(adapter, "_put_latest", frames.append)
    asyncio.run(
        adapter._pump_track(
            Track(), SimpleNamespace(connectionState="connected"), EndOfStream
        )
    )
    return frames, adapter


def test_file_source_uses_original_frames_at_default_30fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "input.mp4"
    path.touch()
    images = [object() for _ in range(60)]
    source = iter(images)
    monkeypatch.setattr(
        video_to_stream, "probe_video", lambda _: VideoInfo(1, 60, 60, 8, 8)
    )
    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (
            (True, next(source)) if source.__length_hint__() else (False, None)
        ),
        release=lambda: None,
    )
    monkeypatch.setattr(video_to_stream.cv2, "VideoCapture", lambda _: capture)

    frames = list(VideoFileStream(path, realtime=False).frames())

    assert [frame.bgr for frame in frames] == images[::2]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx(
        [index / 30 for index in range(30)]
    )


@pytest.mark.parametrize(
    ("path", "buffer_delay"),
    [
        (ROOT / "config.yaml", 2.5),
        (ROOT / "src/goodvision_harness/config.yaml", 2.5),
        (ROOT / "config.delivery.yaml", 2.5),
        (ROOT / "config.vlm-rich.yaml", 10.0),
    ],
)
def test_display_defaults_leave_inference_and_buffer_delay_unchanged(
    path: Path,
    buffer_delay: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    raw = yaml.safe_load(path.read_text())
    assert raw["display"]["fps"] == 30
    assert raw["video"]["fps"] == raw["face"]["sample_fps"] == 2.5
    assert raw["display"]["delay_seconds"] == buffer_delay
    del raw["display"]["fps"]
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(raw))

    assert load_config(custom).display.fps == DisplaySettings().fps == 30
