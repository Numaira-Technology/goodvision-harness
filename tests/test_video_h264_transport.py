"""Real local CPU codec, source provenance and bounded failure tests; no GPU."""

import base64
import io
import time
from dataclasses import replace
from threading import Event

import httpx
import numpy as np
import pytest
from PIL import Image
from test_sam31_continuous_worker import engine
from test_video_tracking_session import SessionServer

from goodvision_harness import video_transport
from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.video_tracking import VideoFrame
from goodvision_harness.video_tracking_session import VideoSessionTrackingHTTPClient
from workers.sam_scene.h264_transport import (
    H264TransportError,
    decode_h264_batch,
    validate_h264,
)

pytest.importorskip("av")
LIMITS = dict(max_image_pixels=1280 * 1280, max_total_image_pixels=16 * 1280 * 1280)


def sources(count=16, *, size=(160, 96), start=0):
    result = []
    for index in range(start, start + count):
        buffer = io.BytesIO()
        with Image.new(
            "RGB", size, (index * 11 % 220, 35, 200 - index * 7 % 180)
        ) as image:
            image.save(buffer, "JPEG", quality=90)
        result.append(
            VideoFrame(100 + index * 3, 9.25 + index * 0.5, buffer.getvalue())
        )
    return tuple(result)


@pytest.mark.parametrize("count", [1, 3, 16])
def test_independent_h264_roundtrip_keeps_source_order_and_full_dimensions(count):
    original = sources(count, size=(768, 576))
    before = tuple(frame.jpeg_bytes for frame in original)
    encoded = video_transport.encode_h264_batch(original)
    header = validate_h264(encoded.data, expected_frames=count, **LIMITS)
    assert (header.width, header.height, header.reference_frames) == (512, 384, 1)
    decoded = decode_h264_batch(encoded.data, expected_frames=count, **LIMITS)
    try:
        assert len(decoded) == count and all(
            image.size == (512, 384) for image in decoded
        )
        for index, image in enumerate(decoded):
            actual = np.asarray(image).mean(axis=(0, 1))
            assert np.max(np.abs(actual - [index * 11, 35, 200 - index * 7])) < 6
        assert tuple(frame.jpeg_bytes for frame in original) == before
        assert [frame.timestamp_seconds for frame in original] == [
            9.25 + i * 0.5 for i in range(count)
        ]
    finally:
        for image in decoded:
            image.close()


def test_odd_source_dimensions_are_resized_without_crop_to_even_chroma_grid():
    encoded = video_transport.encode_h264_batch(
        sources(1, size=(769, 577)), max_side=512
    )
    assert (encoded.width, encoded.height) == (512, 384)
    assert validate_h264(encoded.data, expected_frames=1, **LIMITS).width == 512


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data[:20],
        lambda data: data[:-40],
        lambda data: data + b"\0\0\0\1\x09\x30",
        lambda data: data + data,
        lambda data: b"\0\0\0\x18ftypmp42" + data,
    ],
)
def test_truncation_extra_tail_and_containers_are_rejected(mutation):
    encoded = video_transport.encode_h264_batch(sources()).data
    with pytest.raises(H264TransportError):
        decode_h264_batch(mutation(encoded), expected_frames=16, **LIMITS)


def test_second_parameter_set_resolution_change_is_rejected_before_av(monkeypatch):
    first = video_transport.encode_h264_batch(sources(1)).data
    second = video_transport.encode_h264_batch(sources(1, size=(320, 192))).data
    import av

    monkeypatch.setattr(
        av, "open", lambda *a, **kw: pytest.fail("invalid headers reached AV")
    )
    with pytest.raises(H264TransportError):
        decode_h264_batch(first + second, expected_frames=2, **LIMITS)


@pytest.mark.parametrize(
    "bounds",
    [
        {"expected_frames": 0},
        {"expected_frames": 17},
        {"expected_frames": True},
        {"max_image_pixels": 100},
        {"max_total_image_pixels": 100},
        {"max_encoded_bytes": 10},
    ],
)
def test_pixel_frame_and_encoded_bounds_precede_decode(monkeypatch, bounds):
    encoded = video_transport.encode_h264_batch(sources()).data
    import av

    monkeypatch.setattr(
        av, "open", lambda *a, **kw: pytest.fail("unbounded headers reached AV")
    )
    with pytest.raises(H264TransportError):
        decode_h264_batch(encoded, **({"expected_frames": 16} | LIMITS | bounds))


def test_decode_cancellation_closes_partial_pil_images(monkeypatch):
    encoded = video_transport.encode_h264_batch(sources()).data
    closed = []
    close = Image.Image.close
    checks = 0

    def record_close(self):
        closed.append(self)
        close(self)

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 4:
            raise TimeoutError("cancelled")

    monkeypatch.setattr(Image.Image, "close", record_close)
    with pytest.raises(TimeoutError):
        decode_h264_batch(encoded, expected_frames=16, check=cancel, **LIMITS)
    assert len(closed) == 2


def test_client_sends_only_h264_plus_original_source_metadata_without_jpeg_reencode(
    monkeypatch,
):
    server = SessionServer()
    server.settings["h264_transport"] = 1
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        transport_codec="h264",
        transport_jpeg_quality=55,
    )
    original = sources()
    monkeypatch.setattr(
        client,
        "_transport_frames",
        lambda *a, **kw: pytest.fail("double lossy JPEG encode"),
    )
    try:
        assert client.gpu_ready()
        client.open_session("camera")
        result = client.append(original, stream_session_id="camera", is_last_batch=True)
        wire = server.requests[0]
        assert set(wire) == {
            "stream_session_id",
            "request_id",
            "is_last_batch",
            "frames",
            "video",
        }
        assert all(
            set(row) == {"frame_sequence", "timestamp_seconds"}
            for row in wire["frames"]
        )
        assert wire["video"]["format"] == "h264_annexb"
        assert [(f.frame_sequence, f.timestamp_seconds) for f in result.frames] == [
            (f.frame_sequence, f.timestamp_seconds) for f in original
        ]
        data = base64.b64decode(wire["video"]["data_base64"])
        assert validate_h264(data, expected_frames=16, **LIMITS).frame_count == 16
        metrics = client.transport_stats()
        assert metrics["codec"] == "h264"
        assert (
            metrics["transport_bytes"] == len(data)
            and metrics["transport_jpeg_bytes"] == 0
        )
        assert metrics["source_jpeg_bytes"] == sum(len(f.jpeg_bytes) for f in original)
        assert metrics["compression_seconds"] >= metrics["encode_seconds"] > 0
    finally:
        client.close()


def test_h264_requires_worker_capability_and_keeps_deletable_session_handle():
    server = SessionServer()
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        transport_codec="h264",
    )
    try:
        assert not client.gpu_ready()
        with pytest.raises(RichLabelProtocolError, match="advertise H264"):
            client.open_session("camera")
        assert client.stats()["state"] == "invalid"
        assert not server.requests
    finally:
        client.close()
    assert server.deletes == ["video-1"]


def test_encoding_timeout_keeps_actual_slot_occupied_and_sends_nothing(monkeypatch):
    server = SessionServer()
    server.settings["h264_transport"] = 1
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        transport_codec="h264",
        timeout_seconds=0.04,
    )
    entered, release = Event(), Event()
    encode = video_transport.encode_h264_batch

    def slow_encode(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return encode(*args, **kwargs)

    monkeypatch.setattr(video_transport, "encode_h264_batch", slow_encode)
    try:
        client.open_session("camera")
        with pytest.raises(httpx.TimeoutException):
            client.append(sources(1), stream_session_id="camera", is_last_batch=True)
        assert entered.is_set()
        with pytest.raises(httpx.PoolTimeout):
            client.append(sources(1), stream_session_id="camera", is_last_batch=True)
        release.set()
        time.sleep(0.08)
        assert not server.requests and client.stats()["accepted_frames"] == 0
    finally:
        release.set()
        client.close()


def test_close_retains_remote_token_until_timed_out_encoder_unwinds(monkeypatch):
    import threading

    server = SessionServer()
    server.settings["h264_transport"] = 1
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        transport_codec="h264",
        timeout_seconds=0.04,
    )
    entered, release = Event(), Event()
    encode = video_transport.encode_h264_batch

    def slow_encode(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return encode(*args, **kwargs)

    monkeypatch.setattr(video_transport, "encode_h264_batch", slow_encode)
    closer = None
    try:
        client.open_session("camera")
        with pytest.raises(httpx.TimeoutException):
            client.append(sources(1), stream_session_id="camera", is_last_batch=True)
        assert entered.is_set()
        with pytest.raises(httpx.PoolTimeout):
            client.discard_session()
        assert client.stats()["cleanup_pending"]
        assert client._native_session.video_session_id == "video-1"
        closer = threading.Thread(target=client.close)
        closer.start()
        time.sleep(0.02)
        assert not server.requests and not server.deletes
        release.set()
        closer.join(1)
        assert not closer.is_alive()
        assert server.deletes == ["video-1"]
        assert client._native_session is None
    finally:
        release.set()
        client.close()
        if closer:
            closer.join(1)


@pytest.mark.parametrize("count", [3, 16])
def test_real_codec_and_real_worker_route_preserve_delayed_sources_and_empty_eof(count):
    import threading

    from fastapi import FastAPI

    from workers.sam_scene.app import decode_jpeg
    from workers.sam_scene.continuous import register_session_routes

    selected = engine(h264_enabled=True, hotstart_delay=5)
    application = FastAPI()
    register_session_routes(application, selected, threading.Lock(), decode_jpeg)
    client = VideoSessionTrackingHTTPClient(
        "http://test/v1/video/sessions",
        transport=httpx.ASGITransport(app=application),
        transport_codec="h264",
    )
    original = sources(count)
    try:
        client.open_session("camera")
        first = client.append(
            original, stream_session_id="camera", is_last_batch=count < 16
        )
        results = [first]
        if count == 16:
            results.append(
                client.append((), stream_session_id="camera", is_last_batch=True)
            )
        assert [
            (f.frame_sequence, f.timestamp_seconds) for r in results for f in r.frames
        ] == [(f.frame_sequence, f.timestamp_seconds) for f in original]
        assert results[-1].state == "closed"
        assert selected.backend.calls == (2 if count == 16 else 1)
    finally:
        client.close()
        selected.close()


@pytest.mark.parametrize(
    "options",
    [{"max_side": 127}, {"max_side": 1281}, {"crf": 17}, {"crf": 36}, {"crf": True}],
)
def test_h264_encoder_settings_are_bounded(options):
    with pytest.raises(ValueError):
        video_transport.encode_h264_batch(sources(1), **options)


def test_source_resolution_changes_fail_before_any_upload():
    original = (
        *sources(1),
        replace(
            sources(1, size=(320, 192))[0], frame_sequence=200, timestamp_seconds=11
        ),
    )
    with pytest.raises(ValueError, match="resolution changed"):
        video_transport.encode_h264_batch(original)
