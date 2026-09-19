"""Upload copies preserve original evidence and remain inside request admission."""

import base64
import io
import json
import time

import httpx
import pytest
from PIL import Image
from test_video_tracking import payload
from test_video_tracking_session import SessionServer

from goodvision_harness.video_tracking import VideoFrame, VideoTrackingHTTPClient
from goodvision_harness.video_tracking_session import VideoSessionTrackingHTTPClient


def jpeg(size=(1024, 640)):
    buffer = io.BytesIO()
    with Image.effect_noise(size, 75).convert("RGB") as image:
        image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def client_for(inputs, seen, **options):
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=payload(inputs, batch=body["batch_id"]))

    return VideoTrackingHTTPClient(
        "http://test/v1/video/batches",
        transport=httpx.MockTransport(handler),
        **options,
    )


def test_upload_resize_compresses_each_frame_once_and_keeps_source_geometry(
    monkeypatch,
):
    original = jpeg()
    inputs = (VideoFrame(90, 8.5, original), VideoFrame(94, 8.75, original))
    seen, saves = [], []
    save = Image.Image.save

    def record_save(self, *args, **kwargs):
        saves.append(self.size)
        return save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", record_save)
    client = client_for(inputs, seen, transport_max_side=512, transport_jpeg_quality=65)
    try:
        result = client.infer_batch(inputs, stream_session_id="camera")
        assert saves == [(512, 320), (512, 320)]
        assert all(frame.jpeg_bytes is original for frame in inputs)
        assert [(f.frame_sequence, f.timestamp_seconds) for f in result.frames] == [
            (90, 8.5),
            (94, 8.75),
        ]
        assert result.frames[0].objects[0].bbox.x1 == 0.1
        encoded = [base64.b64decode(f["jpeg_base64"]) for f in seen[0]["frames"]]
        assert all(
            Image.open(io.BytesIO(value)).size == (512, 320) for value in encoded
        )
        metrics = client.transport_stats()
        assert (
            metrics["requests"] == 1
            and metrics["frames"] == metrics["last_frames"] == 2
        )
        assert metrics["source_jpeg_bytes"] == 2 * len(original)
        assert metrics["transport_jpeg_bytes"] == sum(map(len, encoded)) < len(original)
        assert metrics["compression_seconds"] == metrics["last_compression_seconds"] > 0
    finally:
        client.close()


@pytest.mark.parametrize("options", [{}, {"transport_max_side": 512}])
def test_disabled_or_already_fitting_upload_is_byte_identical(options):
    original = jpeg((256, 160))
    inputs = (VideoFrame(1, 0.25, original),)
    seen = []
    client = client_for(inputs, seen, **options)
    try:
        client.infer_batch(inputs, stream_session_id="camera")
        assert base64.b64decode(seen[0]["frames"][0]["jpeg_base64"]) == original
        assert (
            client.transport_stats()["source_jpeg_bytes"]
            == client.transport_stats()["transport_jpeg_bytes"]
        )
    finally:
        client.close()


def test_quality_only_never_upscales_or_changes_aspect():
    inputs = (VideoFrame(1, 0, jpeg((257, 161))),)
    seen = []
    client = client_for(inputs, seen, transport_jpeg_quality=55)
    try:
        client.infer_batch(inputs, stream_session_id="camera")
        wire = base64.b64decode(seen[0]["frames"][0]["jpeg_base64"])
        assert Image.open(io.BytesIO(wire)).size == (257, 161)
        assert wire != inputs[0].jpeg_bytes
    finally:
        client.close()


def test_session_worker_byte_budget_applies_to_upload_copy_and_source_is_unchanged():
    original = jpeg()
    assert len(original) > 200_000
    server = SessionServer()
    server.settings.update(
        max_jpeg_bytes=200_000, max_total_jpeg_bytes=200_000, max_request_bytes=280_000
    )
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        transport_max_side=512,
        transport_jpeg_quality=65,
    )
    frame = VideoFrame(90, 8.5, original)
    try:
        client.open_session("camera")
        result = client.append((frame,), stream_session_id="camera", is_last_batch=True)
        assert frame.jpeg_bytes is original
        assert result.frames[0].frame_sequence == 90
        assert result.frames[0].timestamp_seconds == 8.5
        assert (
            len(base64.b64decode(server.requests[0]["frames"][0]["jpeg_base64"]))
            < 200_000
        )
        assert client.stats()["transport"]["source_jpeg_bytes"] == len(original)
    finally:
        client.close()


def test_invalid_upload_image_does_not_advance_or_poison_native_session():
    server = SessionServer()
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        transport_max_side=512,
        transport_jpeg_quality=65,
    )
    try:
        client.open_session("camera")
        with pytest.raises(OSError):
            client.append(
                (VideoFrame(1, 0.25, b"invalid"),),
                stream_session_id="camera",
                is_last_batch=True,
            )
        assert not server.requests
        assert client.stats()["state"] == "active"
        assert client.stats()["accepted_frames"] == 0
        frame = VideoFrame(1, 0.25, jpeg((256, 160)))
        result = client.append((frame,), stream_session_id="camera", is_last_batch=True)
        assert result.accepted_frame_sequence == 1 and len(server.creates) == 1
    finally:
        client.close()


def test_oversized_decoded_source_is_rejected_before_decode_or_network(monkeypatch):
    buffer = io.BytesIO()
    with Image.new("RGB", (4000, 2160)) as image:
        image.save(buffer, "JPEG")
    inputs = (VideoFrame(0, 0, buffer.getvalue()),)
    seen = []
    client = client_for(inputs, seen, transport_max_side=512)
    monkeypatch.setattr(
        Image.Image, "convert", lambda *a, **k: pytest.fail("decoded oversized image")
    )
    try:
        with pytest.raises(ValueError, match="decoded limits"):
            client.infer_batch(inputs, stream_session_id="camera")
        assert seen == []
    finally:
        client.close()


def test_slow_compression_cannot_send_after_request_deadline(monkeypatch):
    inputs = (VideoFrame(0, 0, jpeg((256, 160))),)
    seen = []
    save = Image.Image.save

    def slow_save(self, *args, **kwargs):
        time.sleep(0.08)
        return save(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", slow_save)
    client = client_for(inputs, seen, transport_jpeg_quality=55, timeout_seconds=0.02)
    try:
        with pytest.raises(httpx.TimeoutException):
            client.infer_batch(inputs, stream_session_id="camera")
        with pytest.raises(httpx.PoolTimeout):
            client.infer_batch(inputs, stream_session_id="camera")
        time.sleep(0.12)
        assert seen == []
    finally:
        client.close()


@pytest.mark.parametrize(
    "options",
    [
        {"transport_max_side": True},
        {"transport_max_side": 127},
        {"transport_max_side": 1281},
        {"transport_jpeg_quality": 0},
        {"transport_jpeg_quality": 101},
        {"transport_jpeg_quality": 55.0},
    ],
)
def test_explicit_transport_settings_must_be_bounded_integers(options):
    with pytest.raises(ValueError):
        VideoTrackingHTTPClient("http://test/v1/video/batches", **options)
