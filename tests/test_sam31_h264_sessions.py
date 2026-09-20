"""Continuous H264 wire/source/cancellation contracts with explicit codec doubles."""

import base64
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from test_sam31_continuous_worker import append, create, engine, payload

from workers.sam_scene import continuous
from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import SessionSettings, parse_append
from workers.sam_scene.h264_transport import H264TransportError
from workers.sam_scene.sam31 import VideoSettings


def wire(count=16, *, start=0, request_id="h0", final=False, data=b"codec-double"):
    raw = json.loads(payload(count, start=start, request_id=request_id, final=final))
    for frame in raw["frames"]:
        del frame["jpeg_base64"]
    raw["video"] = {
        "format": "h264_annexb",
        "data_base64": base64.b64encode(data).decode(),
    }
    return json.dumps(raw).encode()


@pytest.fixture
def codec(monkeypatch):
    state = SimpleNamespace(calls=[], images=[], available_calls=0, callback=None)

    def available():
        state.available_calls += 1

    def decode(data, **kwargs):
        state.calls.append((data, kwargs))
        kwargs["check"]()
        if state.callback is not None:
            return state.callback(data, **kwargs)
        images = [Image.new("RGB", (64, 40)) for _ in range(kwargs["expected_frames"])]
        state.images.extend(images)
        return images

    module = SimpleNamespace(
        ensure_h264_decoder_available=available,
        decode_h264_batch=decode,
        H264TransportError=H264TransportError,
    )
    state.module = module
    monkeypatch.setattr(continuous, "_h264_transport", lambda: module)
    return state


def test_h264_is_explicit_and_warmup_checks_codec_without_attesting_gpu(
    codec, monkeypatch
):
    assert not SessionSettings().h264_enabled
    monkeypatch.setenv("SAM31_SESSION_H264_ENABLED", "true")
    assert SessionSettings.from_env().h264_enabled
    selected = engine(h264_enabled=True)
    assert codec.available_calls == 1
    assert selected.ready and not selected.gpu_ready
    assert selected.limits()["h264_transport"] == 1
    assert engine().limits()["h264_transport"] == 0
    with pytest.raises(HTTPException, match="disabled"):
        parse_append(wire(), VideoSettings())
    for invalid in (1, "true", None):
        with pytest.raises(ValueError):
            SessionSettings(h264_enabled=invalid)


def test_missing_decoder_fails_initialization(codec):
    def unavailable():
        raise ImportError("private codec path")

    codec.module.ensure_h264_decoder_available = unavailable
    with pytest.raises(ImportError):
        engine(h264_enabled=True)


def test_h264_keeps_exact_source_prefix_and_empty_eof(codec):
    selected = engine(h264_enabled=True, hotstart_delay=5)
    token = create(selected)
    first = append(selected, token, wire())
    second = append(selected, token, wire(3, start=16, request_id="h1"))
    eof = append(selected, token, payload(0, request_id="eof", final=True))
    snapshots = first["frames"] + second["frames"] + eof["frames"]
    assert [item["frame_sequence"] for item in snapshots] == list(range(0, 38, 2))
    assert [item["timestamp_seconds"] for item in snapshots] == [
        i / 4 for i in range(19)
    ]
    assert {item["stream_session_id"] for item in snapshots} == {"camera"}
    assert (
        first["predictor_epoch"] == second["predictor_epoch"] == eof["predictor_epoch"]
    )
    assert eof["state"] == "closed" and eof["pending_frame_count"] == 0
    assert len(codec.calls) == 2  # Empty final never encodes or decodes fake frames.
    assert codec.calls[0][1]["max_side"] == 1280
    assert codec.calls[0][1]["max_encoded_bytes"] == 8 * 1024 * 1024
    assert selected.session.sources == {}
    for image in codec.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_replay_hash_covers_codec_bytes_metadata_and_final(codec):
    selected = engine(h264_enabled=True)
    token = create(selected)
    original = wire()
    first = append(selected, token, original)
    # JSON whitespace/key order differences retain the same canonical wire hash.
    replay = append(
        selected, token, json.dumps(json.loads(original), sort_keys=True).encode()
    )
    assert replay == first and selected.backend.calls == 1 and len(codec.calls) == 1
    for changed in (
        wire(data=b"different"),
        wire(start=1),
        wire(final=True),
        payload(),
    ):
        if changed == payload():
            raw = json.loads(changed)
            raw["request_id"] = "h0"
            changed = json.dumps(raw).encode()
        with pytest.raises(HTTPException) as error:
            append(selected, token, changed)
        assert error.value.status_code == 409
    assert selected.backend.calls == 1 and len(codec.calls) == 1


@pytest.mark.parametrize(
    "change,status",
    [
        (lambda x: x["video"].update(format="mp4"), 400),
        (lambda x: x["video"].update(extra=1), 400),
        (lambda x: x["video"].update(data_base64="%%%"), 400),
        (lambda x: x["video"].update(data_base64="☃"), 400),
        (lambda x: x["video"].update(data_base64=True), 400),
        (lambda x: x["frames"][0].update(jpeg_base64="AA=="), 400),
        (lambda x: x["frames"][0].update(frame_sequence=True), 400),
        (lambda x: x["frames"][0].update(timestamp_seconds=True), 400),
        (lambda x: x["frames"][0].update(timestamp_seconds=10**400), 400),
        (lambda x: x["frames"][1].update(timestamp_seconds=0), 409),
        (lambda x: x.update(frames=[]), 413),
        (lambda x: x.update(frames=None), 400),
        (lambda x: x.update(frames=x["frames"] * 2), 413),
    ],
)
def test_h264_wire_strict_validation(change, status):
    raw = json.loads(wire())
    change(raw)
    with pytest.raises(HTTPException) as error:
        parse_append(json.dumps(raw).encode(), VideoSettings(), h264_enabled=True)
    assert error.value.status_code == status


def test_h264_rejects_duplicate_fields_and_bytes_before_decode(monkeypatch):
    duplicate = wire().replace(
        b'"format": "h264_annexb"', b'"format":"h264_annexb","format":"h264_annexb"'
    )
    with pytest.raises(HTTPException) as error:
        parse_append(duplicate, VideoSettings(), h264_enabled=True)
    assert error.value.status_code == 400

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("oversized base64 reached decoder")

    oversized = wire(data=b"12345")
    monkeypatch.setattr(continuous.base64, "b64decode", forbidden_decode)
    with pytest.raises(HTTPException) as error:
        parse_append(
            oversized,
            replace(VideoSettings(), max_total_jpeg_bytes=3),
            h264_enabled=True,
        )
    assert error.value.status_code == 413


@pytest.mark.parametrize("status", [400, 413])
def test_decoder_error_preserves_empty_session_and_private_text(codec, caplog, status):
    def fail(*args, **kwargs):
        raise H264TransportError("private codec path and source", status)

    selected = engine(h264_enabled=True)
    token = create(selected)
    codec.callback = fail
    with pytest.raises(HTTPException) as error:
        append(selected, token, wire())
    assert error.value.status_code == status
    assert "private" not in str(error.value) + caplog.text
    assert selected.session.source_size is None and selected.session.sources == {}
    assert selected.session.total_frames == 0 and selected.backend.calls == 0
    assert selected.session.last_response is None and selected.ready
    codec.callback = None
    assert append(selected, token, wire())["accepted_frame_sequence"] == 30


def test_cancel_during_decode_never_reaches_native_or_replay(codec):
    selected = engine(h264_enabled=True)
    token = create(selected)

    def cancelled(*args, **kwargs):
        selected.invalidate(token)
        kwargs["check"]()

    codec.callback = cancelled
    with pytest.raises(TimeoutError):
        append(selected, token, wire())
    assert selected.backend.calls == 0 and selected.session.native_state is None
    assert selected.session.source_size is None and selected.session.sources == {}
    assert selected.session.last_response is None and selected.ready
    assert selected.session.status == "invalidated"


def test_deadline_during_decode_never_begins_native(codec, monkeypatch):
    selected = engine(h264_enabled=True)
    token = create(selected)
    clock = [time.monotonic()]
    monkeypatch.setattr(continuous.time, "monotonic", lambda: clock[0])

    def slow(*args, **kwargs):
        images = [Image.new("RGB", (64, 40)) for _ in range(kwargs["expected_frames"])]
        codec.images.extend(images)
        clock[0] += 10
        return images

    codec.callback = slow
    with pytest.raises(TimeoutError):
        append(selected, token, wire(), deadline=clock[0] + 1)
    assert selected.backend.calls == 0 and selected.session.total_frames == 0
    assert selected.session.sources == {} and selected.session.source_size is None
    for image in codec.images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_http_health_and_codec_request_share_normal_routes(codec):
    selected = engine(h264_enabled=True)
    app = FastAPI()
    continuous.register_session_routes(app, selected, threading.Lock(), decode_jpeg)
    with TestClient(app) as client:
        health = client.get("/health/video/session")
        assert health.status_code == 503  # Native inference is a double.
        assert health.json()["limits"]["h264_transport"] == 1
        opened = client.post("/v1/video/sessions", json={"stream_session_id": "camera"})
        assert opened.status_code == 200
        token = opened.json()["video_session_id"]
        response = client.post(
            f"/v1/video/sessions/{token}/append", content=wire(1, final=True)
        )
        assert response.status_code == 200
        assert response.json()["frames"][0]["frame_sequence"] == 0
        assert response.json()["state"] == "closed"
