"""Camera routing must retain lane identity for every tracking operation."""

import httpx
import pytest

from goodvision_harness.perception import PerceptionSessionHTTPClient


@pytest.mark.parametrize(
    "camera",
    [
        "camera-01", "camera-06", "camera-09", "camera-10", "camera-12",
        "camera-13", "camera-16", "camera-24", "camera-99", "camera-100",
    ],
)
def test_camera_lane_prefix_is_preserved(camera):
    client = PerceptionSessionHTTPClient(
        f"http://worker:8092/v1/cameras/{camera}/perception/sam_live/sessions",
        method="sam_live",
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    )
    try:
        assert client.endpoint == (
            f"http://worker:8092/v1/cameras/{camera}/perception/sam_live/sessions"
        )
        assert client._health == (
            f"http://worker:8092/health/cameras/{camera}/perception/sam_live"
        )
    finally:
        client.close()


@pytest.mark.parametrize(
    "camera",
    [
        "camera-00", "camera-0", "camera--1", "camera-1.5", "other", "camera-1",
        "camera-001", "camera-010", "camera-0100", "camera-+13", "camera-13extra",
    ],
)
def test_invalid_lane_does_not_fall_back_to_shared_legacy_route(camera):
    with pytest.raises(ValueError, match="camera"):
        PerceptionSessionHTTPClient(
            f"http://worker/v1/cameras/{camera}/perception/sam_live/sessions",
            method="sam_live",
        )
