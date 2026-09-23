from types import SimpleNamespace

import numpy as np
import pytest

from goodvision_harness.stream_adapter import NetworkStreamAdapter
from goodvision_harness.stream_decode import DecodeSupportError


def test_ffmpeg_adapter_preserves_source_clock_and_shuts_down_owner(monkeypatch):
    from goodvision_harness import stream_adapter

    image = np.zeros((8, 12, 3), dtype=np.uint8)
    stopped = []

    class Receiver:
        decode_errors = 0

        def __init__(self, *args):
            self.metadata = {"color_transfer": "smpte2084"}

        def frames(self):
            for i in range(4):
                yield SimpleNamespace(
                    pts=90000 + i * 6000,
                    time_base=1 / 90000,
                    is_corrupt=False,
                    key_frame=i == 0,
                    width=12,
                    height=8,
                    to_ndarray=lambda **kwargs: image,
                )

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(stream_adapter, "FFmpegReceiver", Receiver)
    adapter = NetworkStreamAdapter("rtsp://example.test/live", decoder_backend="ffmpeg")
    iterator = adapter.frames()
    frames = [next(iterator) for _ in range(4)]
    adapter.stop()
    iterator.close()
    assert stopped
    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    assert frames[-1].timestamp_seconds - frames[0].timestamp_seconds == pytest.approx(
        0.2
    )
    assert adapter._diagnostics._state["normalized_to_sdr"] is True


def test_unavailable_decoder_fails_explicitly_without_silent_hdr_fallback(monkeypatch):
    from goodvision_harness import stream_adapter

    class Missing:
        def __init__(self, *args):
            pass

        def frames(self):
            raise DecodeSupportError("missing binary")

        def stop(self):
            pass

    monkeypatch.setattr(stream_adapter, "FFmpegReceiver", Missing)
    adapter = NetworkStreamAdapter("rtsp://example.test/live", decoder_backend="ffmpeg")
    with pytest.raises(stream_adapter.StreamAdapterError) as caught:
        next(adapter.frames())
    assert caught.value.code == "missing_support"
