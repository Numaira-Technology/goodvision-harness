"""Video provenance, transport bounds and asynchronous lifecycle; no GPU claims."""

from __future__ import annotations

import copy
import json
from threading import Event
from uuid import uuid4

import httpx
import pytest

from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.video_tracking import (
    MAX_BATCH_JPEG_BYTES,
    MAX_JPEG_BYTES,
    MAX_RESPONSE_BYTES,
    VideoFrame,
    VideoTrackingBatch,
    VideoTrackingCoordinator,
    VideoTrackingHTTPClient,
)


def frame(sequence=0, at=0):
    return VideoFrame(sequence, float(at), b"jpeg")


def payload(frames, session="camera", batch="batch", epoch="epoch", *, people=True):
    return {
        "stream_session_id": session,
        "batch_id": batch,
        "predictor_epoch": epoch,
        "backend": "sam3.1_video",
        "memory_scope": "batch",
        "completed_frame_sequence": frames[-1].frame_sequence,
        "completed_timestamp_seconds": frames[-1].timestamp_seconds,
        "frames": [
            {
                "stream_session_id": session,
                "frame_sequence": item.frame_sequence,
                "timestamp_seconds": item.timestamp_seconds,
                "objects": [
                    {
                        "track_id": f"sam31-{epoch}-1",
                        "label": "person",
                        "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
                        "confidence": 0.9,
                        "observation_state": "observed",
                        "last_observed_timestamp_seconds": item.timestamp_seconds,
                        "last_observed_frame_sequence": item.frame_sequence,
                        "attributes": {
                            "native_track": {
                                "backend": "sam3.1_video",
                                "epoch": epoch,
                                "object_id": "1",
                            }
                        },
                    }
                ]
                if people
                else [],
            }
            for item in frames
        ],
    }


def parse(value, frames):
    return VideoTrackingBatch.parse(
        value,
        stream_session_id="camera",
        batch_id="batch",
        inputs=frames,
    )


def test_complete_ordered_video_provenance_and_empty_frames():
    inputs = (frame(), frame(30, 0.25))
    result = parse(payload(inputs), inputs)
    assert [item.frame_sequence for item in result.frames] == [0, 30]
    assert result.frames[1].objects[0].attributes["native_track"]["object_id"] == "1"
    assert result.completed_timestamp_seconds == 0.25
    assert not parse(payload(inputs, people=False), inputs).frames[-1].objects


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(stream_session_id="other"),
        lambda p: p.update(batch_id="other"),
        lambda p: p.update(memory_scope="unbounded"),
        lambda p: p.update(backend="sam3_image_motion"),
        lambda p: p.update(completed_frame_sequence=True),
        lambda p: p.update(completed_timestamp_seconds=900),
        lambda p: p["frames"].pop(),
        lambda p: p["frames"].reverse(),
        lambda p: p["frames"][0].update(timestamp_seconds=0.01),
        lambda p: p["frames"][0]["objects"][0].update(observation_state="predicted"),
        lambda p: p["frames"][0]["objects"][0].update(last_observed_frame_sequence=7),
        lambda p: p["frames"][0]["objects"][0]["attributes"]["native_track"].update(
            epoch="other"
        ),
        lambda p: p["frames"][0]["objects"][0]["attributes"].clear(),
        lambda p: p["frames"][0]["objects"].append(
            copy.deepcopy(p["frames"][0]["objects"][0])
        ),
    ],
)
def test_entire_invalid_batch_is_rejected(mutation):
    inputs = (frame(), frame(1, 0.25))
    value = payload(inputs)
    mutation(value)
    with pytest.raises(RichLabelProtocolError):
        parse(value, inputs)


def test_transport_preserves_sparse_source_indices_and_validates_batch_id():
    seen = []

    def handler(request):
        value = json.loads(request.content)
        seen.append(value)
        frames = tuple(
            frame(x["frame_sequence"], x["timestamp_seconds"]) for x in value["frames"]
        )
        return httpx.Response(
            200, json=payload(frames, value["stream_session_id"], value["batch_id"])
        )

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches", transport=httpx.MockTransport(handler)
    )
    result = client.infer_batch(
        (frame(100, 1), frame(105, 1.25)), stream_session_id="camera"
    )
    assert result.completed_frame_sequence == 105
    assert seen[0]["frames"][0]["jpeg_base64"] == "anBlZw=="
    client.close()


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"gpu_ready": False},
        {"ready": 1},
        {"backend": "sam3_image_motion"},
        {"memory_scope": "continuous"},
    ],
)
def test_health_requires_native_backend_and_real_gpu_readiness(change):
    value = {
        "ready": True,
        "gpu_ready": True,
        "backend": "sam3.1_video",
        "native_sam_video_tracking": True,
        "memory_scope": "batch",
        **change,
    }
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=value)

    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches?secret=ignored",
        transport=httpx.MockTransport(handler),
    )
    assert client.gpu_ready() is (not change)
    assert urls == ["http://worker/health/video"]
    client.close()


def test_response_is_byte_bounded():
    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b" " * (MAX_RESPONSE_BYTES + 1))
        ),
    )
    with pytest.raises(RichLabelProtocolError, match="byte limit"):
        client.infer_batch((frame(),), stream_session_id="camera")
    client.close()


def test_input_order_count_and_byte_limits_before_network():
    client = VideoTrackingHTTPClient(
        "http://worker/v1/video/batches",
        transport=httpx.MockTransport(
            lambda _: pytest.fail("invalid input reached network")
        ),
    )
    for inputs in (
        (),
        tuple(frame(i, i) for i in range(17)),
        (frame(1, 1), frame(2, 1)),
        (frame(2, 1), frame(1, 2)),
    ):
        with pytest.raises(ValueError):
            client.infer_batch(inputs, stream_session_id="camera")
    with pytest.raises(ValueError, match="JPEG exceeds"):
        VideoFrame(0, 0, b"x" * (MAX_JPEG_BYTES + 1))
    size = MAX_BATCH_JPEG_BYTES // 3 + 1
    with pytest.raises(ValueError, match="batch exceeds"):
        client.infer_batch(
            tuple(VideoFrame(i, i, b"x" * size) for i in range(3)),
            stream_session_id="camera",
        )
    client.close()


class BatchClient:
    def __init__(self, *, blocked=False, fails=False):
        self.started, self.release, self.closed = Event(), Event(), Event()
        self.requests = []
        self.fails = fails
        if not blocked:
            self.release.set()

    def infer_batch(self, frames, *, stream_session_id):
        self.requests.append(tuple(frames))
        self.started.set()
        assert self.release.wait(5)
        if self.fails:
            raise RuntimeError("secret must not enter public diagnostics")
        value = payload(frames, stream_session_id, "batch", uuid4().hex)
        return VideoTrackingBatch.parse(
            value, stream_session_id=stream_session_id, batch_id="batch", inputs=frames
        )

    def close(self):
        self.closed.set()


def submit(coordinator, index, *, session="camera"):
    return coordinator.submit(
        b"jpeg",
        stream_session_id=session,
        frame_sequence=index,
        timestamp_seconds=index / 4,
    )


def test_async_batch_commit_is_ordered_and_partial_flush_works():
    client, results = BatchClient(), []
    coordinator = VideoTrackingCoordinator(client, on_batch=results.append)
    try:
        for i in range(4):
            submit(coordinator, i)
        assert not client.started.is_set()
        coordinator.flush()
        assert coordinator.wait_until_idle()
        assert len(results) == 1
        assert [x.frame_sequence for x in results[0].frames] == list(range(4))
        assert coordinator.latest("camera").frame_sequence == 3
        assert coordinator.latest("other") is None
        assert coordinator.stats()["tracking_lag_seconds"] == 0
        assert not coordinator.select("camera", 10).objects
    finally:
        coordinator.close()
    assert client.closed.is_set()


def test_native_display_precedes_blocked_scene_commit():
    release, entered = Event(), Event()
    native = []

    def commit(_):
        entered.set()
        assert release.wait(5)

    coordinator = VideoTrackingCoordinator(BatchClient(), on_batch=commit)
    coordinator.on_native_batch = lambda result, generation: native.append(result)
    try:
        submit(coordinator, 0)
        coordinator.flush()
        assert entered.wait(2)
        assert len(native) == 1
        assert native[0].frames[0].frame_sequence == 0
    finally:
        release.set()
        coordinator.close()


def test_queue_is_bounded_and_reports_backpressure_while_gpu_is_busy():
    client = BatchClient(blocked=True)
    coordinator = VideoTrackingCoordinator(client, batch_frames=2)
    try:
        submit(coordinator, 0)
        submit(coordinator, 1)
        assert client.started.wait(2)
        for i in range(2, 10):
            submit(coordinator, i)
        stats = coordinator.stats()
        assert stats["pending_frames"] == 2
        assert stats["pending_bytes"] == 8
        assert stats["dropped_backpressure"] == 6
        client.release.set()
        assert coordinator.wait_until_idle()
        assert [[x.frame_sequence for x in group] for group in client.requests] == [
            [0, 1],
            [8, 9],
        ]
    finally:
        client.release.set()
        coordinator.close()


def test_reset_discards_late_batch_and_does_not_close_busy_client():
    client, results = BatchClient(blocked=True), []
    coordinator = VideoTrackingCoordinator(
        client, batch_frames=1, on_batch=results.append
    )
    submit(coordinator, 0)
    assert client.started.wait(2)
    coordinator.reset("new-camera")
    assert not submit(coordinator, 1)
    assert submit(coordinator, 0, session="new-camera")
    client.release.set()
    assert coordinator.wait_until_idle()
    assert [x.stream_session_id for x in results] == ["new-camera"]
    assert coordinator.stats()["stale_results"] == 1
    coordinator.close()


def test_source_sampling_and_rejected_order_are_explicit():
    coordinator = VideoTrackingCoordinator(BatchClient())
    try:
        assert submit(coordinator, 2)
        assert not submit(coordinator, 1)
        assert not coordinator.submit(
            b"jpeg",
            stream_session_id="camera",
            frame_sequence=3,
            timestamp_seconds=0.51,
        )
        assert coordinator.stats()["rejected_order"] == 1
        assert coordinator.stats()["sampled_out"] == 1
    finally:
        coordinator.close()


def test_errors_do_not_publish_history_or_sensitive_remote_text():
    coordinator = VideoTrackingCoordinator(BatchClient(fails=True), batch_frames=1)
    try:
        submit(coordinator, 0)
        assert coordinator.wait_until_idle()
        assert coordinator.stats()["failed"] == 1
        assert coordinator.stats()["last_error"] == "RuntimeError"
        assert coordinator.latest() is None
    finally:
        coordinator.close()


def test_close_returns_promptly_while_late_work_owns_resources():
    client, results = BatchClient(blocked=True), []
    coordinator = VideoTrackingCoordinator(
        client, batch_frames=1, on_batch=results.append
    )
    submit(coordinator, 0)
    assert client.started.wait(2)
    coordinator.close(timeout_seconds=0.01)
    assert not client.closed.is_set()
    client.release.set()
    assert client.closed.wait(2)
    assert results == []
