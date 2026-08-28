from __future__ import annotations

import pytest
from goodvision_harness.stream_adapter import parse_stream_source

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
