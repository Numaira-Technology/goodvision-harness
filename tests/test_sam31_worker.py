"""Native-video contract/API doubles only; no real CUDA or model weights used."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image

from goodvision_harness.rich_labeling import RichLabelSnapshot
from workers.sam_scene.app import SceneEngine, create_app, decode_jpeg
from workers.sam_scene.sam31 import (
    BACKEND,
    SAM3_COMMIT,
    NativeSam31,
    SourceFrame,
    VideoBatch,
    VideoEngine,
    VideoSettings,
    native_snapshot,
    register_video_routes,
    strict_checkpoint_state,
    verify_installed_source,
)


def jpeg(size=(80, 48), format="JPEG"):
    stream = io.BytesIO()
    with Image.new("RGB", size, (30, 80, 120)) as image:
        image.save(stream, format=format)
    return stream.getvalue()


def batch(count=3, *, size=(80, 48), sequence=10, timestamp=2.0):
    encoded = base64.b64encode(jpeg(size)).decode("ascii")
    return {
        "stream_session_id": "session-a",
        "batch_id": "batch-a",
        "frames": [
            {
                "frame_sequence": sequence + i * 2,
                "timestamp_seconds": timestamp + i * 0.25,
                "jpeg_base64": encoded,
            }
            for i in range(count)
        ],
    }


def native_output(size, ids=(7,), empty=False):
    width, height = size
    masks = np.zeros((len(ids), height, width), dtype=bool)
    if not empty:
        masks[:, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = True
    return {
        "out_obj_ids": np.array(ids, dtype=np.int64),
        "out_probs": np.full(len(ids), 0.85, dtype=np.float32),
        "out_binary_masks": masks,
    }


class BackendDouble:
    gpu_ready = True  # Injection must never be allowed to attest GPU readiness.

    def __init__(self):
        self.calls = 0
        self.closed = False
        self.released = 0
        self.sizes = []

    def warmup(self):
        pass

    def infer(self, images):
        self.calls += 1
        self.sizes = [image.size for image in images]
        try:
            for index, image in enumerate(images):
                # A visible native ID persists within this batch. An empty mask
                # is intentionally present to verify it is never made visible.
                yield index, native_output(image.size, empty=index == 1)
        finally:
            self.released += 1

    def close(self):
        self.closed = True


def make_engine(backend=None, **settings):
    engine = VideoEngine(
        replace(VideoSettings(), enabled=True, **settings),
        backend=backend or BackendDouble(),
    )
    engine.warmup()
    return engine


def run(engine, data=None, **kwargs):
    return engine.run(
        json.dumps(data or batch()).encode(),
        deadline=kwargs.get("deadline", time.monotonic() + 5),
        decode_jpeg=kwargs.get("decoder", decode_jpeg),
    )


@pytest.fixture
def http_factory(monkeypatch):
    monkeypatch.setenv("SAM_SCENE_IMAGE_ENABLED", "false")

    def create(engine):
        return TestClient(create_app(video_engine=engine))

    return create


def route_request(engine, receive, *, admission=None, authorization=b"private-token"):
    """Exercise real streaming Request reads without client-side body buffering."""
    app = FastAPI()
    lock = admission if admission is not None else threading.Lock()
    register_video_routes(app, engine, lock, decode_jpeg)
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/v1/video/batches"
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/video/batches",
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", authorization),
            ],
        },
        receive,
    )
    return endpoint(request)


def test_route_phase_timing_measures_body_and_cleanup_without_payload(caplog):
    class CleanupDelay(BackendDouble):
        def infer(self, images):
            try:
                yield from super().infer(images)
            finally:
                time.sleep(0.03)

    engine = make_engine(CleanupDelay())
    data = batch(1)
    data["stream_session_id"] = "private-session"
    data["batch_id"] = "private-batch"
    payload = json.dumps(data).encode()
    chunks = [payload[:100], payload[100:]]

    async def receive():
        await asyncio.sleep(0.02)
        return {
            "type": "http.request",
            "body": chunks.pop(0),
            "more_body": bool(chunks),
        }

    result = asyncio.run(route_request(engine, receive))
    assert result["batch_id"] == "private-batch"
    assert engine.backend.released == 1
    records = [
        record
        for record in caplog.records
        if record.name == "goodvision.sam31" and "batch completed" in record.msg
    ]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO  # Visible under normal root WARNING.
    received_bytes, body_seconds, inference_seconds, total_seconds = record.args
    assert received_bytes == len(payload)
    assert body_seconds >= 0.03
    assert inference_seconds >= 0.025  # Includes the native generator's cleanup.
    assert total_seconds >= body_seconds + inference_seconds
    assert "body_read_seconds=" in record.getMessage()
    assert "inference_cleanup_seconds=" in record.getMessage()
    for private in (
        "private-session",
        "private-batch",
        "private-token",
        data["frames"][0]["jpeg_base64"],
    ):
        assert private not in caplog.text


def test_route_fallback_log_yields_to_later_parent_handler(caplog, capsys, monkeypatch):
    parent = logging.Logger("isolated-worker-parent", logging.WARNING)
    worker = logging.Logger("isolated-worker")
    worker.parent = parent
    monkeypatch.setattr("workers.sam_scene.sam31.LOGGER", worker)
    register_video_routes(FastAPI(), make_engine(), threading.Lock(), decode_jpeg)
    worker.info("before parent setup")
    assert "before parent setup" in capsys.readouterr().err
    parent.addHandler(caplog.handler)
    worker.info("after parent setup")
    assert "after parent setup" not in capsys.readouterr().err
    assert [record.getMessage() for record in caplog.records] == ["after parent setup"]
    assert parent.level == logging.WARNING


def test_route_receiving_timeout_logs_partial_bytes_without_starting_gpu(caplog):
    engine = make_engine(request_timeout_seconds=0.03)
    admission = threading.Lock()
    calls = 0

    async def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": b"private", "more_body": True}
        await asyncio.sleep(1)
        return {"type": "http.request", "body": b"-body", "more_body": False}

    with pytest.raises(HTTPException) as error:
        asyncio.run(route_request(engine, receive, admission=admission))
    assert error.value.status_code == 504
    assert not admission.locked()
    assert engine.backend.calls == 0
    records = [record for record in caplog.records if "batch timed out" in record.msg]
    assert len(records) == 1
    stage, elapsed, received = records[0].args
    assert records[0].levelno == logging.WARNING
    assert stage == "receiving" and elapsed >= 0.025 and received == 7
    assert "private" not in caplog.text
    assert not any("batch completed" in record.msg for record in caplog.records)


def test_route_backend_exception_text_never_reaches_worker_logs(caplog, monkeypatch):
    engine = make_engine()

    def fail(*_args, **_kwargs):
        raise RuntimeError("private-session private-token private-image-payload")

    monkeypatch.setattr(engine, "run", fail)

    async def receive():
        return {"type": "http.request", "body": b"private-body", "more_body": False}

    with pytest.raises(HTTPException) as error:
        asyncio.run(route_request(engine, receive))
    assert error.value.status_code == 503
    assert "Native video batch failed" in caplog.text
    assert "private" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_native_id_metadata_source_frames_and_empty_masks():
    engine = make_engine()
    result = run(engine)
    assert result["backend"] == BACKEND
    assert result["memory_scope"] == "batch"
    assert result["batch_id"] == "batch-a"
    assert result["completed_frame_sequence"] == 14
    assert result["completed_timestamp_seconds"] == 2.5
    snapshots = [RichLabelSnapshot.from_dict(row) for row in result["frames"]]
    assert [row.frame_sequence for row in snapshots] == [10, 12, 14]
    assert [row.timestamp_seconds for row in snapshots] == [2, 2.25, 2.5]
    assert snapshots[1].objects == ()
    first, last = snapshots[0].objects[0], snapshots[2].objects[0]
    assert first.track_id == last.track_id
    assert first.attributes["native_track"] == {
        "backend": BACKEND,
        "epoch": result["predictor_epoch"],
        "object_id": "7",
    }
    assert first.observation_state == "observed"
    assert first.last_observed_frame_sequence == 10
    assert first.last_observed_timestamp_seconds == 2
    assert last.last_observed_frame_sequence == 14
    assert result["frames"][0]["objects"][0]["bbox"] == {
        "x1": 0.25,
        "y1": 0.25,
        "x2": 0.75,
        "y2": 0.75,
    }
    assert engine.backend.released == 1


def _snapshot_for_masks(masks, *, image_size=None):
    count, height, width = masks.shape
    output = {
        "out_obj_ids": np.arange(count, dtype=np.int64),
        "out_probs": np.full(count, 0.95, dtype=np.float32),
        "out_binary_masks": masks,
    }
    return native_snapshot(
        output,
        batch=VideoBatch("session", "batch", (SourceFrame(23, 2.3, b""),)),
        index=0,
        epoch="original-epoch",
        image_size=image_size if image_size is not None else (width, height),
        max_objects=16,
    )


def test_distant_single_pixels_do_not_expand_native_person_bbox_or_mutate_masks():
    masks = np.zeros((1, 540, 720), dtype=bool)
    masks[0, 200:260, 300:322] = True
    masks[0, 0, 0] = masks[0, 539, 719] = True
    original = masks.copy()
    masks.flags.writeable = False
    result = _snapshot_for_masks(masks)
    item = result["objects"][0]
    assert item["bbox"] == {
        "x1": 300 / 720,
        "y1": 200 / 540,
        "x2": 322 / 720,
        "y2": 260 / 540,
    }
    assert item["attributes"]["mask_geometry"] == {
        "max_ignored_component_area_pixels": 4,
        "ignored_component_count": 2,
        "ignored_component_pixels": 2,
    }
    assert item["attributes"]["native_track"] == {
        "backend": BACKEND,
        "epoch": "original-epoch",
        "object_id": "0",
    }
    assert item["observation_state"] == "observed"
    assert item["last_observed_frame_sequence"] == 23
    assert item["last_observed_timestamp_seconds"] == 2.3
    np.testing.assert_array_equal(masks, original)


def test_all_larger_disconnected_body_components_remain_in_native_bbox():
    masks = np.zeros((1, 540, 720), dtype=bool)
    masks[0, 100:207, 200:206] = True  # 642-pixel body component.
    masks[0, 190:213, 210:217] = True  # 161-pixel separate visible body part.
    masks[0, 400, 700] = True
    assert masks.sum() == 804
    item = _snapshot_for_masks(masks)["objects"][0]
    assert item["bbox"] == {
        "x1": 200 / 720,
        "y1": 100 / 540,
        "x2": 217 / 720,
        "y2": 213 / 540,
    }
    assert item["attributes"]["mask_geometry"]["ignored_component_pixels"] == 1


@pytest.mark.parametrize("pixels", [1, 2, 3, 4])
def test_only_tiny_components_never_create_observed_person(pixels):
    masks = np.zeros((1, 48, 80), dtype=bool)
    masks[0, 8, 10 : 10 + pixels] = True
    masks[0, 30, 60 : 60 + pixels] = True
    assert _snapshot_for_masks(masks)["objects"] == []


@pytest.mark.parametrize("size", [(9, 7), (80, 48), (720, 540)])
def test_absolute_pixel_threshold_and_exclusive_bounds_use_actual_mask_size(size):
    width, height = size
    masks = np.zeros((2, height, width), dtype=bool)
    masks[0, 0:2, 0:2] = True  # Four pixels are ignored at every resolution.
    masks[1, height - 2 : height, width - 3 : width] = True  # Six pixels retained.
    items = _snapshot_for_masks(masks)["objects"]
    assert len(items) == 1
    assert items[0]["attributes"]["native_track"]["object_id"] == "1"
    assert items[0]["bbox"] == {
        "x1": (width - 3) / width,
        "y1": (height - 2) / height,
        "x2": 1.0,
        "y2": 1.0,
    }


def test_diagonally_connected_five_pixel_component_is_retained():
    masks = np.zeros((1, 12, 16), dtype=bool)
    for index in range(5):
        masks[0, index + 2, index + 3] = True
    item = _snapshot_for_masks(masks)["objects"][0]
    assert item["bbox"] == {
        "x1": 3 / 16,
        "y1": 2 / 12,
        "x2": 8 / 16,
        "y2": 7 / 12,
    }
    assert item["attributes"]["mask_geometry"]["ignored_component_count"] == 0


@pytest.mark.parametrize(
    "image_size", [(0, 12), (16, 0), (True, 12), (16.0, 12), (12, 16)]
)
def test_invalid_original_mask_dimensions_rejected_before_component_analysis(
    image_size,
):
    masks = np.ones((1, 12, 16), dtype=bool)
    with pytest.raises(RuntimeError, match="contract"):
        _snapshot_for_masks(masks, image_size=image_size)


def test_every_batch_gets_new_epoch_even_native_id_and_batch_id_repeat():
    engine = make_engine()
    first, second = run(engine), run(engine)
    assert first["predictor_epoch"] != second["predictor_epoch"]
    assert (
        first["frames"][0]["objects"][0]["track_id"]
        != second["frames"][0]["objects"][0]["track_id"]
    )
    assert not hasattr(engine, "sessions")


def test_one_frame_flush_is_valid_and_does_not_invent_extra_frames():
    result = run(make_engine(), batch(1))
    assert len(result["frames"]) == 1
    assert result["completed_frame_sequence"] == 10


def test_gpu_health_cannot_be_attested_by_backend_double(http_factory):
    with http_factory(make_engine()) as client:
        response = client.get("/health/video")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False and data["gpu_ready"] is False
        assert data["native_sam_video_tracking"] is True
        assert data["memory_scope"] == "batch"
        assert data["limits"]["max_frames"] == 16
        assert data["limits"]["max_inflight_requests"] == 1
        assert client.post("/v1/video/batches", json=batch()).status_code == 200


def test_missing_checkpoint_never_imports_sam_or_attests_readiness(
    tmp_path, http_factory
):
    engine = VideoEngine(
        replace(
            VideoSettings(), enabled=True, checkpoint_path=str(tmp_path / "missing.pt")
        )
    )
    with http_factory(engine) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/video")
        assert response.status_code == 503
        assert response.json()["error_code"] == "model_initialization_failed"
        assert client.post("/v1/video/batches", json=batch()).status_code == 503


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "duplicate",
        "boolean_sequence",
        "negative_sequence",
        "huge_sequence",
        "boolean_timestamp",
        "nonfinite",
        "unknown_frame",
        "base64",
        "empty_id",
    ],
)
def test_bad_json_or_source_metadata_never_reaches_native_model(mutation, http_factory):
    engine = make_engine()
    data = batch()
    if mutation == "extra":
        data["prompt"] = "anything"
    elif mutation == "duplicate":
        pass
    elif mutation == "boolean_sequence":
        data["frames"][0]["frame_sequence"] = True
    elif mutation == "negative_sequence":
        data["frames"][0]["frame_sequence"] = -1
    elif mutation == "huge_sequence":
        data["frames"][0]["frame_sequence"] = 2**64
    elif mutation == "boolean_timestamp":
        data["frames"][0]["timestamp_seconds"] = True
    elif mutation == "nonfinite":
        data["frames"][0]["timestamp_seconds"] = float("nan")
    elif mutation == "unknown_frame":
        data["frames"][0]["fake"] = 1
    elif mutation == "base64":
        data["frames"][0]["jpeg_base64"] = "not base64!"
    elif mutation == "empty_id":
        data["batch_id"] = ""
    payload = json.dumps(data)
    if mutation == "duplicate":
        payload = payload.replace('"batch_id":', '"batch_id":"other","batch_id":')
    with http_factory(engine) as client:
        response = client.post(
            "/v1/video/batches",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400
    assert engine.backend.calls == 0


@pytest.mark.parametrize("field", ["frame_sequence", "timestamp_seconds"])
def test_duplicate_or_reverse_source_order_rejected(field, http_factory):
    engine = make_engine()
    data = batch()
    data["frames"][1][field] = data["frames"][0][field]
    with http_factory(engine) as client:
        assert client.post("/v1/video/batches", json=data).status_code == 409
    assert engine.backend.calls == 0


@pytest.mark.parametrize(
    "setting,value,expected",
    [
        ("max_frames", 2, 413),
        ("max_request_bytes", 20, 413),
        ("max_jpeg_bytes", 20, 413),
        ("max_total_jpeg_bytes", 1000, 413),
        ("max_image_pixels", 100, 413),
        ("max_total_image_pixels", 5000, 413),
    ],
)
def test_http_resource_budgets_before_gpu(setting, value, expected, http_factory):
    engine = make_engine(**{setting: value})
    with http_factory(engine) as client:
        assert client.post("/v1/video/batches", json=batch()).status_code == expected
    assert engine.backend.calls == 0


def test_chunked_upload_is_bounded_without_content_length(http_factory):
    engine = make_engine(max_request_bytes=100)
    with http_factory(engine) as client:
        response = client.post(
            "/v1/video/batches",
            content=iter([b"x" * 80, b"y" * 80]),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413


def test_invalid_jpeg_and_mixed_dimensions_fail_before_native():
    engine = make_engine()
    data = batch(2)
    data["frames"][1]["jpeg_base64"] = base64.b64encode(jpeg((40, 24))).decode()
    with pytest.raises(HTTPException) as error:
        run(engine, data)
    assert error.value.status_code == 400
    data["frames"][1]["jpeg_base64"] = base64.b64encode(jpeg(format="PNG")).decode()
    with pytest.raises(HTTPException):
        run(engine, data)
    assert engine.backend.calls == 0


def test_decode_shrinks_each_image_before_decoding_next_and_closes_all():
    engine = make_engine(inference_max_side=40)
    decoded = []

    def decoder(payload, settings):
        if decoded:
            assert decoded[-1].size == (40, 24)
        image = decode_jpeg(payload, settings)
        decoded.append(image)
        return image

    run(engine, decoder=decoder)
    assert engine.backend.sizes == [(40, 24)] * 3
    for image in decoded:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "backwards", "bad_masks", "nan_score", "duplicate_id"],
)
def test_native_contract_failure_discards_batch_and_quarantines_model(failure):
    class Broken(BackendDouble):
        def infer(self, images):
            try:
                if failure == "missing":
                    return
                for index, image in enumerate(images):
                    output = native_output(image.size)
                    if failure == "duplicate" and index == 1:
                        index = 0
                    if failure == "backwards":
                        index = 2 - index
                    if failure == "bad_masks":
                        output["out_binary_masks"] = np.zeros((1, 2, 2), dtype=bool)
                    if failure == "nan_score":
                        output["out_probs"][0] = float("nan")
                    if failure == "duplicate_id":
                        output = native_output(image.size, ids=(7, 7))
                    yield index, output
            finally:
                self.released += 1

    engine = make_engine(Broken())
    with pytest.raises(RuntimeError):
        run(engine)
    assert engine.ready is False
    assert engine.error_code == "native_inference_failed"
    assert engine.backend.released == 1


def test_timeout_keeps_shared_gpu_slot_until_actual_native_work_finishes(
    http_factory, caplog
):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class Slow(BackendDouble):
        def infer(self, images):
            entered.set()
            try:
                release.wait(3)
                yield 0, native_output(images[0].size)
            finally:
                finished.set()

    engine = make_engine(Slow(), request_timeout_seconds=0.05)
    with http_factory(engine) as client:
        with ThreadPoolExecutor() as pool:
            request = pool.submit(client.post, "/v1/video/batches", json=batch(1))
            assert entered.wait(1)
            assert request.result(timeout=2).status_code == 504
            timeouts = [
                record for record in caplog.records if "batch timed out" in record.msg
            ]
            assert len(timeouts) == 1
            stage, elapsed, received = timeouts[0].args
            assert stage == "inference" and elapsed >= 0.04 and received > len(jpeg())
            assert timeouts[0].levelno == logging.WARNING
            assert client.post("/v1/video/batches", json=batch(1)).status_code == 429
            release.set()
            assert finished.wait(1)
    assert engine.backend.closed
    assert "session-a" not in caplog.text and "batch-a" not in caplog.text
    assert not any("batch completed" in record.msg for record in caplog.records)


def test_image_endpoint_shares_native_batch_gpu_admission(monkeypatch):
    monkeypatch.setenv("SAM_SCENE_IMAGE_ENABLED", "false")
    entered, release = threading.Event(), threading.Event()

    class Slow(BackendDouble):
        def infer(self, images):
            entered.set()
            release.wait(2)
            yield 0, native_output(images[0].size)

    video = make_engine(Slow())
    image = SceneEngine(
        detector=SimpleNamespace(
            warmup=lambda: None, detect=lambda _: (), close=lambda: None
        )
    )
    image.warmup()
    with TestClient(create_app(engine=image, video_engine=video)) as client:
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(client.post, "/v1/video/batches", json=batch(1))
            assert entered.wait(1)
            response = client.post(
                "/v1/tracks",
                content=jpeg(),
                headers={
                    "Content-Type": "image/jpeg",
                    "X-GoodVision-Stream-Session-Id": "old-session",
                    "X-GoodVision-Frame-Sequence": "1",
                    "X-GoodVision-Timestamp-Seconds": "1",
                },
            )
            assert response.status_code == 429
            release.set()
            assert pending.result(timeout=2).status_code == 200


def test_deadline_discards_late_results_and_closes_native_generator():
    engine = make_engine()

    class ClockBackend(BackendDouble):
        def infer(self, images):
            try:
                yield 0, native_output(images[0].size)
                time.sleep(0.03)
                yield 1, native_output(images[1].size)
            finally:
                self.released += 1

    engine.backend = ClockBackend()
    with pytest.raises(TimeoutError):
        run(engine, deadline=time.monotonic() + 0.02)
    assert engine.backend.released == 1
    assert engine.ready is True  # A cleanly released deadline is not GPU failure.


def test_wrong_source_installation_rejected(monkeypatch):
    monkeypatch.setattr(
        "workers.sam_scene.sam31.metadata.distribution",
        lambda _: SimpleNamespace(
            read_text=lambda _: json.dumps({"vcs_info": {"commit_id": "wrong"}})
        ),
    )
    with pytest.raises(RuntimeError, match="source commit"):
        verify_installed_source()


def test_native_adapter_uses_official_finite_video_api_and_strict_weights(
    monkeypatch, tmp_path
):
    """Runs only Python API doubles, never imports real torch/SAM or uses CUDA."""
    import workers.sam_scene.sam31 as module

    calls = {}
    autocast_stack = []
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"api-double")

    class TensorDouble:
        shape = (2,)

        def is_floating_point(self):
            return False

    expected = {"detector.weight": TensorDouble(), "tracker.weight": TensorDouble()}

    class BuilderAutocast:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            autocast_stack.append(self.owner)
            return self

        def __exit__(self, *_args):
            assert autocast_stack.pop() == self.owner
            calls.setdefault("builder_autocast_exits", []).append(self.owner)

    class ModelDouble:
        def __init__(self):
            self.tracker = SimpleNamespace(bf16_context=BuilderAutocast("tracker"))
            self.tracker.bf16_context.__enter__()

        def state_dict(self):
            return expected

        def load_state_dict(self, state, *, strict):
            calls["strict"] = strict
            assert state == expected

        def init_state(
            self, *, resource_path, offload_video_to_cpu, async_loading_frames
        ):
            calls["init"] = (offload_video_to_cpu, async_loading_frames)
            return {"images": resource_path}

        def add_prompt(self, *, inference_state, frame_idx, text_str):
            calls["prompt"] = (frame_idx, text_str)

        def propagate_in_video(
            self, *, inference_state, start_frame_idx, max_frame_num_to_track, reverse
        ):
            calls["propagate"] = (start_frame_idx, max_frame_num_to_track, reverse)
            assert self.use_batched_grounding is False
            assert self.postprocess_batch_size == 1
            for index, image in enumerate(inference_state["images"]):
                yield index, native_output(image.size, ids=())

        def reset_state(self, state):
            calls["reset"] = calls.get("reset", 0) + 1

    def build(**kwargs):
        calls["build"] = kwargs
        model = ModelDouble()
        context = BuilderAutocast("predictor")
        context.__enter__()
        return SimpleNamespace(model=model, bf16_context=context)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.update(synchronized=True),
        empty_cache=lambda: None,
    )
    torch.inference_mode = nullcontext
    torch.autocast = lambda **_: nullcontext()
    torch.bfloat16 = "bf16"

    def load(path, *, map_location, weights_only):
        calls["load"] = (path, map_location, weights_only)
        return {"model": expected}

    torch.load = load
    sam = ModuleType("sam3")
    builder = ModuleType("sam3.model_builder")
    builder.build_sam3_multiplex_video_predictor = build
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sam3", sam)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)
    monkeypatch.setattr(module, "verify_installed_source", lambda: None)
    backend = NativeSam31(
        replace(VideoSettings(), enabled=True, checkpoint_path=str(checkpoint))
    )
    backend.warmup()
    assert autocast_stack == []
    assert calls["builder_autocast_exits"] == ["predictor", "tracker"]
    assert backend.predictor.bf16_context is None
    assert backend.predictor.model.tracker.bf16_context is None
    assert calls["strict"] is True
    assert calls["load"] == (str(checkpoint), "cpu", True)
    assert calls["build"] == {
        "checkpoint_path": str(checkpoint),
        "max_num_objects": 16,
        "multiplex_count": 16,
        "use_fa3": False,
        "use_rope_real": False,
        "compile": False,
        "warm_up": False,
        "async_loading_frames": False,
    }
    assert calls["init"] == (True, False)
    assert calls["prompt"] == (0, "person")
    assert calls["propagate"] == (0, 3, False)
    assert calls["reset"] == 1 and calls["synchronized"]
    backend.close()
    assert calls["builder_autocast_exits"] == ["predictor", "tracker"]


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape", "remap_collision"])
def test_merged_checkpoint_validation_never_swallows_damage(kind):
    tensor = SimpleNamespace(shape=(2,))
    expected = {"detector.weight": tensor, "tracker.weight": tensor}
    state = dict(expected)
    if kind == "missing":
        del state["tracker.weight"]
    elif kind == "unexpected":
        state["something_else"] = tensor
    elif kind == "shape":
        state["tracker.weight"] = SimpleNamespace(shape=(3,))
    else:
        state["sam3_model.weight"] = tensor
    with pytest.raises(RuntimeError):
        strict_checkpoint_state(state, expected)


def test_explicit_official_checkpoint_prefix_remapping_and_pin():
    tensor = SimpleNamespace(shape=(2,))
    assert strict_checkpoint_state(
        {"sam3_model.weight": tensor, "sam2_predictor.weight": tensor},
        {"detector.weight": tensor, "tracker.weight": tensor},
    ) == {"detector.weight": tensor, "tracker.weight": tensor}
    assert SAM3_COMMIT == "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"


@pytest.mark.parametrize(
    "settings",
    [
        {"max_frames": 17},
        {"max_objects": 17},
        {"inference_max_side": 2048},
        {"request_timeout_seconds": float("inf")},
    ],
)
def test_limits_cannot_be_configured_past_hard_ceiling(settings):
    with pytest.raises(ValueError):
        replace(VideoSettings(), **settings)
