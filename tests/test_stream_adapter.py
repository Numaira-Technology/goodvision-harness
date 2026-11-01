from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from goodvision_harness.privacy import (
    PUBLIC_STREAM_SUPPORT_ERROR,
    STREAM_ERROR_MESSAGES,
)
from goodvision_harness.stream_adapter import (
    NetworkStreamAdapter,
    StreamAdapterError,
    parse_stream_source,
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


def test_missing_video_dependency_has_a_safe_actionable_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "aiortc", None)
    adapter = NetworkStreamAdapter(ROBOT_WHEP_URL)

    with pytest.raises(StreamAdapterError) as caught:
        asyncio.run(adapter._receive_whep())

    assert caught.value.code == "missing_support"
    assert caught.value.public_message == PUBLIC_STREAM_SUPPORT_ERROR


@pytest.mark.parametrize("failure,code", [
    (TimeoutError("private-source?token=secret"), "signaling_timeout"),
    (ConnectionError("private-source?token=secret"), "signaling_unreachable"),
    (401, "source_rejected"), (404, "source_rejected"),
    ("missing_location", "invalid_response"), ("html", "invalid_response"),
])
def test_whep_signaling_failures_have_safe_actionable_reasons(failure, code):
    import aiohttp

    class Response:
        status = failure if isinstance(failure, int) else 201
        url = "http://private-source/session?token=secret"

        def __init__(self):
            self.headers = {
                "Content-Type": "text/html" if failure == "html" else "application/sdp"
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def post(*args, **kwargs):
        if isinstance(failure, TimeoutError):
            raise failure
        if isinstance(failure, ConnectionError):
            raise aiohttp.ClientConnectionError(str(failure))
        return Response()

    adapter = NetworkStreamAdapter(ROBOT_WHEP_URL)
    with pytest.raises(StreamAdapterError) as caught:
        asyncio.run(adapter._open_whep_session(SimpleNamespace(post=post), "offer"))
    assert caught.value.code == code
    assert caught.value.public_message == STREAM_ERROR_MESSAGES[code]
    assert "secret" not in str(caught.value)
    assert "private-source" not in str(caught.value)
