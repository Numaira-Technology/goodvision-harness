from __future__ import annotations

import asyncio
import sys

import pytest

from goodvision_harness.privacy import PUBLIC_STREAM_SUPPORT_ERROR
from goodvision_harness.stream_adapter import (
    NetworkStreamAdapter,
    StreamAdapterError,
    parse_stream_source,
    _END,
)

ROBOT_WHEP_URL = (
    "webrtc+http://192.168.20.199:1985/rtc/v1/whep/"
    "?app=live&stream=livestream"
)


def test_robot_whep_url_maps_to_http_transport() -> None:
    source = parse_stream_source(ROBOT_WHEP_URL)

    assert source.protocol == "webrtc"
    assert source.host == "192.168.20.199"
    assert source.source_url == ROBOT_WHEP_URL
    assert source.transport_url == (
        "http://192.168.20.199:1985/rtc/v1/whep/"
        "?app=live&stream=livestream"
    )


def test_plain_whep_http_url_requires_explicit_webrtc_prefix() -> None:
    with pytest.raises(ValueError, match="webrtc\\+https"):
        parse_stream_source(
            "http://192.168.20.199:1985/rtc/v1/whep/"
            "?app=live&stream=livestream"
        )


def test_webrtc_stop_unblocks_a_waiting_frame_consumer() -> None:
    adapter = NetworkStreamAdapter(ROBOT_WHEP_URL)

    adapter.stop()

    assert adapter.stopped
    assert adapter._queue.get(timeout=0.2) is _END


def test_missing_video_dependency_has_a_safe_actionable_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "aiortc", None)
    adapter = NetworkStreamAdapter(ROBOT_WHEP_URL)

    with pytest.raises(StreamAdapterError) as caught:
        asyncio.run(adapter._receive_whep())

    assert caught.value.code == "missing_support"
    assert caught.value.public_message == PUBLIC_STREAM_SUPPORT_ERROR
