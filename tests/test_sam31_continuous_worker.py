"""Session lifetime/protocol/failure tests with doubles; no GPU attestation."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import (
    SessionEngine,
    SessionSettings,
    parse_append,
    register_session_routes,
)
from workers.sam_scene.sam31 import VideoSettings
from workers.sam_scene.streaming_state import finalize_pending


def payload(count=16, *, start=0, request_id="r0", final=False, size=(64, 40)):
    stream = io.BytesIO()
    with Image.new("RGB", size) as image:
        image.save(stream, "JPEG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    return json.dumps(
        {
            "stream_session_id": "camera",
            "request_id": request_id,
            "is_last_batch": final,
            "frames": [
                {
                    "frame_sequence": i * 2,
                    "timestamp_seconds": i * 0.25,
                    "jpeg_base64": encoded,
                }
                for i in range(start, start + count)
            ],
        }
    ).encode()


class Backend:
    gpu_ready = True

    def __init__(self, hotstart_delay=15):
        self.calls = 0
        self.hotstart_delay = hotstart_delay
        self.object_id = 7
        self.entered = threading.Event()
        self.release = None
        self.fail = False

    def warmup(self):
        pass

    def run(self, session, images, *, final):
        self.calls += 1
        self.entered.set()
        session.native_state = {"test_double": True}
        if self.release is not None:
            assert self.release.wait(5)
        if self.fail:
            raise RuntimeError("native failed")
        end = session.total_frames + len(images)
        stop = end if final else max(0, end - (self.hotstart_delay - 1))
        for index in range(session.emitted_index + 1, stop):
            width, height = session.output_size
            mask = np.zeros((1, height, width), dtype=bool)
            mask[:, 4:20, 8:24] = True
            yield (
                index,
                {
                    "out_obj_ids": np.array([self.object_id]),
                    "out_probs": np.array([0.9]),
                    "out_binary_masks": mask,
                },
            )

    def prune(self, session):
        assert len(session.sources) <= self.hotstart_delay - 1

    def discard(self, session):
        session.native_state = None

    def close(self):
        pass


def engine(**settings):
    video = SimpleNamespace(
        settings=replace(VideoSettings(), enabled=True), gpu_ready=True
    )
    session_settings = replace(SessionSettings(), enabled=True, **settings)
    result = SessionEngine(
        video, session_settings, Backend(session_settings.hotstart_delay)
    )
    result.warmup()
    return result


def create(selected):
    return selected.create(b'{"stream_session_id":"camera"}')["video_session_id"]


def append(selected, token, body, deadline=None):
    return selected.append(
        token, body, deadline=deadline or time.monotonic() + 5, decode_jpeg=decode_jpeg
    )


def test_continuous_epoch_prior_request_prefix_and_empty_eof():
    selected = engine()
    token = create(selected)
    first = append(selected, token, payload())
    assert [frame["frame_sequence"] for frame in first["frames"]] == [0, 2]
    assert first["accepted_frame_sequence"] == first["processed_frame_sequence"] == 30
    assert first["pending_frame_count"] == 14
    second = append(selected, token, payload(3, start=16, request_id="r1"))
    assert [frame["frame_sequence"] for frame in second["frames"]] == [4, 6, 8]
    assert second["predictor_epoch"] == first["predictor_epoch"]
    eof = append(selected, token, payload(0, start=19, request_id="eof", final=True))
    assert [frame["frame_sequence"] for frame in eof["frames"]] == list(
        range(10, 38, 2)
    )
    assert eof["accepted_frame_sequence"] == eof["emitted_frame_sequence"] == 36
    assert eof["pending_frame_count"] == 0 and eof["state"] == "closed"
    assert not selected.has_active_session
    assert selected.session.sources == {}
    assert selected.session.native_state is None


def test_exact_retry_is_idempotent_and_response_cache_is_isolated():
    selected = engine()
    token = create(selected)
    body = payload()
    first = append(selected, token, body)
    original_epoch = first["predictor_epoch"]
    first["frames"].clear()
    retried = append(selected, token, body)
    assert len(retried["frames"]) == 2 and retried["predictor_epoch"] == original_epoch
    assert selected.backend.calls == 1
    with pytest.raises(HTTPException) as error:
        append(selected, token, payload(request_id="r0", final=True))
    assert error.value.status_code == 409
    with pytest.raises(HTTPException):
        append(selected, token, payload(request_id="different"))
    assert selected.backend.calls == 1


@pytest.mark.parametrize("count", [1, 3, 16])
def test_first_final_uses_only_arrived_frames(count):
    selected = engine()
    result = append(selected, create(selected), payload(count, final=True))
    assert len(result["frames"]) == count and result["pending_frame_count"] == 0
    assert result["state"] == "closed"


@pytest.mark.parametrize("count,final", [(1, False), (15, False), (0, True)])
def test_insufficient_startup_does_not_advance(count, final):
    selected = engine()
    with pytest.raises(HTTPException):
        append(selected, create(selected), payload(count, final=final))
    assert selected.backend.calls == 0


def test_sources_and_native_ids_do_not_grow_without_limit():
    selected = engine(max_session_frames=64)
    token = create(selected)
    for start in range(0, 64, 16):
        append(selected, token, payload(16, start=start, request_id=str(start)))
        assert len(selected.session.sources) == 14
        assert all(not frame.jpeg_bytes for frame in selected.session.sources.values())
    with pytest.raises(HTTPException, match="reset_required"):
        append(selected, token, payload(1, start=64, request_id="overflow"))
    assert not selected.has_active_session
    selected = engine()
    selected.backend.object_id = 2048
    with pytest.raises(HTTPException, match="reset_required"):
        append(selected, create(selected), payload())
    assert not selected.has_active_session


def test_ttl_delete_and_resolution_change_release_native_state():
    selected = engine(idle_seconds=5)
    token = create(selected)
    append(selected, token, payload())
    with pytest.raises(HTTPException, match="active_session_exists"):
        create(selected)
    deleted = selected.delete(token)
    assert deleted["discarded_pending_frames"] == 14
    token = create(selected)
    append(selected, token, payload())
    with pytest.raises(HTTPException, match="reset_required"):
        append(selected, token, payload(1, start=16, request_id="size", size=(80, 40)))
    token = create(selected)
    append(selected, token, payload())
    selected.session.touched -= 6
    selected.expire()
    assert selected.session is None


def test_native_failure_invalidates_and_quarantines_readiness():
    selected = engine()
    selected.backend.fail = True
    with pytest.raises(RuntimeError, match="native failed"):
        append(selected, create(selected), payload())
    assert not selected.ready and not selected.has_active_session
    assert selected.session.sources == {} and selected.session.last_response is None


def test_health_double_never_attests_real_cuda():
    selected = engine()
    app = FastAPI()
    register_session_routes(app, selected, threading.Lock(), decode_jpeg)
    with TestClient(app) as client:
        result = client.get("/health/video/session")
        assert result.status_code == 503
        assert result.json()["ready"] is result.json()["gpu_ready"] is False
        assert (
            client.post(
                "/v1/video/sessions", json={"stream_session_id": "camera"}
            ).status_code
            == 200
        )


def test_http_timeout_keeps_single_gpu_slot_until_native_cleanup():
    selected = engine()
    selected.video.settings = replace(
        selected.video.settings, request_timeout_seconds=0.05
    )
    selected.backend.release = threading.Event()
    app, lock = FastAPI(), threading.Lock()
    register_session_routes(app, selected, lock, decode_jpeg)
    try:
        with TestClient(app) as client:
            token = client.post(
                "/v1/video/sessions", json={"stream_session_id": "camera"}
            ).json()["video_session_id"]
            result = client.post(
                f"/v1/video/sessions/{token}/append", content=payload()
            )
            assert result.status_code == 504 and selected.backend.entered.is_set()
            assert (
                client.post(
                    "/v1/video/sessions", json={"stream_session_id": "camera"}
                ).status_code
                == 429
            )
            selected.backend.release.set()
            limit = time.monotonic() + 2
            while lock.locked() and time.monotonic() < limit:
                time.sleep(0.005)
            assert not lock.locked() and not selected.has_active_session
    finally:
        selected.backend.release.set()


def test_strict_json_and_empty_final_fields():
    settings = VideoSettings()
    with pytest.raises(HTTPException):
        parse_append(b'{"stream_session_id":"a","stream_session_id":"b"}', settings)
    wrong = json.loads(payload())
    wrong["is_last_batch"] = 1
    with pytest.raises(HTTPException):
        parse_append(json.dumps(wrong).encode(), settings)


def test_failed_initial_jpeg_decode_does_not_commit_source_dimensions():
    selected = engine()
    token = create(selected)
    invalid = json.loads(payload())
    invalid["frames"][1]["jpeg_base64"] = base64.b64encode(b"not a JPEG").decode()
    with pytest.raises(HTTPException) as error:
        append(selected, token, json.dumps(invalid).encode())
    assert error.value.status_code == 400
    assert selected.session.source_size is selected.session.output_size is None
    assert selected.session.total_frames == 0 and selected.session.sources == {}
    result = append(selected, token, payload(size=(80, 40), request_id="valid"))
    assert result["processed_frame_sequence"] == 30


def test_create_timeout_retires_only_its_unreturned_session(monkeypatch):
    selected = engine()
    selected.video.settings = replace(
        selected.video.settings, request_timeout_seconds=0.05
    )
    original = selected.create
    release, entered = threading.Event(), threading.Event()
    first = True

    def delayed(body):
        nonlocal first
        result = original(body)
        if first:
            first = False
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(selected, "create", delayed)
    app, lock = FastAPI(), threading.Lock()
    register_session_routes(app, selected, lock, decode_jpeg)
    try:
        with TestClient(app) as client:
            result = client.post(
                "/v1/video/sessions", json={"stream_session_id": "camera"}
            )
            assert result.status_code == 504 and entered.is_set()
            old_token = selected.session.video_session_id
            release.set()
            limit = time.monotonic() + 2
            while (
                lock.locked() or selected.has_active_session
            ) and time.monotonic() < limit:
                time.sleep(0.005)
            assert not lock.locked() and not selected.has_active_session
            result = client.post(
                "/v1/video/sessions", json={"stream_session_id": "camera"}
            )
            assert result.status_code == 200
            assert result.json()["video_session_id"] != old_token
            selected.invalidate(old_token)
            assert selected.has_active_session and not selected.session.cancel.is_set()
    finally:
        release.set()


def test_arbitrary_backend_error_is_not_exposed_in_logs(caplog):
    selected = engine()

    def fail(*args, **kwargs):
        raise RuntimeError("private-source-data-do-not-log")

    selected.backend.run = fail
    app = FastAPI()
    register_session_routes(app, selected, threading.Lock(), decode_jpeg)
    with TestClient(app) as client:
        token = client.post(
            "/v1/video/sessions", json={"stream_session_id": "camera"}
        ).json()["video_session_id"]
        result = client.post(f"/v1/video/sessions/{token}/append", content=payload())
    assert result.status_code == 503
    assert "private-source-data-do-not-log" not in caplog.text
    assert "native session operation failed" in caplog.text


def test_cancellation_at_response_commit_cannot_publish_replay_cache():
    selected = engine()
    token = create(selected)

    class CommitLock:
        def __init__(self):
            self.lock = threading.RLock()
            self.cancel_next = False

        def __enter__(self):
            self.lock.acquire()
            if self.cancel_next:
                selected.session.cancel.set()
            return self

        def __exit__(self, *args):
            self.lock.release()

    gate = CommitLock()
    selected._mutex = gate
    selected.backend.prune = lambda _session: setattr(gate, "cancel_next", True)
    with pytest.raises(TimeoutError):
        append(selected, token, payload())
    assert selected.session.last_response is None
    assert selected.session.native_state is None
    assert selected.session.status == "invalidated"


def test_empty_finalizer_uses_exact_confirmation_horizon_without_new_inference():
    calls = []
    selected = SimpleNamespace(
        world_size=1,
        postprocess_batch_size=1,
        masklet_confirmation_consecutive_det_thresh=3,
    )
    selected._postprocess_output = (
        lambda state, output, removed, suppressed, unconfirmed: (
            calls.append((output["i"], unconfirmed)) or output
        )
    )
    state = {
        "num_frames": 16,
        "generator_state": {
            "hotstart_buffer": [
                (i, {"i": i, "suppressed_obj_ids": set()}) for i in range(2, 16)
            ],
            "postprocess_yield_list": [],
            "hotstart_removed_obj_ids": set(),
            "unconfirmed_obj_ids_per_frame": {i: [i] for i in range(16)},
        },
    }
    assert [index for index, _ in finalize_pending(selected, state)] == list(
        range(2, 16)
    )
    assert calls[-3:] == [(13, [15]), (14, [15]), (15, [15])]
    assert state["generator_state"]["hotstart_buffer"] == []
