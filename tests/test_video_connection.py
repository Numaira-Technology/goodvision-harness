from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness import video_connection as module
from goodvision_harness.privacy import (
    PUBLIC_STREAM_ERROR,
    PUBLIC_STREAM_SUPPORT_ERROR,
)
from goodvision_harness.stream_adapter import StreamAdapterError, parse_stream_source
from goodvision_harness.video_to_stream import LiveFrame

_END = object()


class ControlledAdapter:
    def __init__(self, stream_url, **kwargs):
        self.source = parse_stream_source(stream_url)
        self.options = kwargs
        self.items = queue.Queue()
        self.stop_requested = threading.Event()
        self.frames_calls = 0
        self.processed_frames = 0

    def frames(self):
        self.frames_calls += 1
        while not self.stop_requested.is_set():
            item = self.items.get(timeout=3)
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item
            self.processed_frames += 1

    def stop(self):
        self.stop_requested.set()
        self.items.put(_END)


@pytest.fixture
def connections(monkeypatch):
    adapters = []
    connections = []

    def adapter_factory(url, **kwargs):
        adapter = ControlledAdapter(url, **kwargs)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(module, "NetworkStreamAdapter", adapter_factory)

    def create(**kwargs):
        connection = module.VideoConnection(
            "rtsp://private-user:private-password@camera.local/private-path",
            output_fps=15,
            **kwargs,
        )
        connections.append(connection)
        return connection, adapters[-1]

    yield create, adapters
    for connection in connections:
        connection.stop(timeout=1)
        assert not connection.running


def frame(index, *, image=None):
    return LiveFrame(
        index=index,
        timestamp_seconds=index / 15,
        bgr=np.full((20, 40, 3), index % 255, dtype=np.uint8)
        if image is None else image,
    )


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("video connection did not reach the expected state")
        time.sleep(0.002)


def publish(connection, adapter, live_frame):
    expected = connection._sequence + 1
    adapter.items.put(live_frame)
    wait_for(
        lambda: connection._sequence >= expected
        and adapter.processed_frames >= expected
    )


def test_connection_decodes_once_and_analysis_detaches_without_stopping_video(
    connections,
):
    create, adapters = connections
    connection, adapter = create(whep_headers={"Authorization": "private-token"})
    assert connection.status()["connection_state"] == "disconnected"
    connection.start()
    connection.start()
    assert connection.status()["connection_state"] == "connecting"
    publish(connection, adapter, frame(1))
    wait_for(lambda: connection.latest_jpeg() is not None)
    assert connection.status()["connected"]
    first = connection.subscribe()
    second = connection.subscribe()
    current = frame(2)
    publish(connection, adapter, current)
    assert next(first.frames()) is current
    second_frames = second.frames()
    assert next(second_frames) is current
    first.stop()
    assert not adapter.stop_requested.is_set()
    assert connection.running
    later = frame(3)
    publish(connection, adapter, later)
    assert next(second_frames) is later
    second.stop()
    assert list(second_frames) == []
    assert len(adapters) == adapter.frames_calls == 1
    assert adapter.options["whep_headers"] == {"Authorization": "private-token"}
    assert "private" not in str(connection.status())


def test_new_session_waits_for_a_new_frame_and_preserves_source_timing(connections):
    create, _ = connections
    connection, adapter = create()
    connection.start()
    publish(connection, adapter, frame(150))
    subscription = connection.subscribe()
    result = []
    ready = threading.Event()

    def consume_one():
        ready.set()
        result.append(next(subscription.frames()))

    thread = threading.Thread(target=consume_one)
    thread.start()
    assert ready.wait(1)
    assert not result
    expected = frame(151)
    publish(connection, adapter, expected)
    thread.join(1)
    assert not thread.is_alive()
    assert result == [expected]
    assert result[0].index == 151
    assert result[0].timestamp_seconds == 151 / 15


def test_slow_analysis_keeps_only_latest_frame(connections):
    create, _ = connections
    connection, adapter = create()
    connection.start()
    subscription = connection.subscribe()
    for index in range(1, 51):
        publish(connection, adapter, frame(index))
    frames = subscription.frames()
    assert next(frames).index == 50
    assert subscription.dropped_frames == 49
    assert connection._latest.index == 50
    publish(connection, adapter, frame(51))
    assert next(frames).index == 51
    subscription.stop()


def test_stopping_subscriber_wakes_waiting_reader_without_disconnect(connections):
    create, _ = connections
    connection, _ = create()
    connection.start()
    subscription = connection.subscribe()
    result = []
    thread = threading.Thread(target=lambda: result.extend(subscription.frames()))
    thread.start()
    subscription.stop()
    thread.join(1)
    assert not thread.is_alive()
    assert result == []
    assert connection.running
    assert connection.status()["connection_state"] == "connecting"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("rtsp://secret:password@private-host"), PUBLIC_STREAM_ERROR),
        (
            StreamAdapterError("private-dependency", code="missing_support"),
            PUBLIC_STREAM_SUPPORT_ERROR,
        ),
        (_END, PUBLIC_STREAM_ERROR),
    ],
)
def test_failed_source_wakes_subscribers_and_exposes_only_safe_error(
    connections, error, message,
):
    create, _ = connections
    connection, adapter = create()
    connection.start()
    subscription = connection.subscribe()
    adapter.items.put(error)
    wait_for(lambda: not connection.running)
    assert connection.status()["connection_state"] == "error"
    assert connection.status()["connection_error"] == message
    assert not connection.status()["connected"]
    with pytest.raises(StreamAdapterError, match=message):
        next(subscription.frames())
    with pytest.raises(StreamAdapterError, match=message):
        connection.subscribe()
    assert "private" not in str(connection.status())


def test_preview_crops_and_limits_size_but_subscriber_receives_raw_frame(connections):
    create, _ = connections
    connection, adapter = create(frame_crop="right_half", preview_max_side=2000)
    connection.start()
    subscription = connection.subscribe()
    image = np.zeros((1200, 2400, 3), dtype=np.uint8)
    image[:, 1200:, 2] = 240
    original = frame(0, image=image)
    publish(connection, adapter, original)
    wait_for(lambda: connection.latest_jpeg() is not None)
    decoded = cv2.imdecode(
        np.frombuffer(connection.latest_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    assert decoded.shape == (1200, 1200, 3)
    assert decoded[:, :, 2].mean() > 230
    assert next(subscription.frames()) is original
    assert original.bgr.shape == (1200, 2400, 3)


def test_preview_keeps_source_cadence_without_the_old_four_fps_limit(
    connections, monkeypatch,
):
    create, _ = connections
    clock = [1.0]
    encoded = []
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def encode(image, **kwargs):
        encoded.append(int(image[0, 0, 0]))
        return b"preview", 40, 20

    monkeypatch.setattr(module, "encode_jpeg", encode)
    connection, adapter = create()
    connection.start()
    publish(connection, adapter, frame(0))
    wait_for(lambda: len(encoded) == 1)
    for index in range(1, 20):
        clock[0] = 1.0 + index / 100
        publish(connection, adapter, frame(index))
        wait_for(lambda: len(encoded) == index + 1)
    assert encoded == list(range(20))
    clock[0] = 1.26
    publish(connection, adapter, frame(20))
    wait_for(lambda: len(encoded) == 21)
    assert encoded == list(range(21))
    assert connection._sequence == 21


def test_slow_display_encoder_cannot_block_capture_or_analysis(connections, monkeypatch):
    create, _ = connections
    entered, release = threading.Event(), threading.Event()
    def encode(image, **kwargs):
        entered.set()
        release.wait(2)
        return b"jpeg", 40, 20
    monkeypatch.setattr(module, "encode_jpeg", encode)
    connection, adapter = create()
    connection.start()
    subscription = connection.subscribe()
    try:
        publish(connection, adapter, frame(1))
        assert entered.wait(1)
        publish(connection, adapter, frame(2))
        assert next(subscription.frames()).index == 2
        assert connection.latest_jpeg() is None
    finally:
        release.set()


def test_display_cursors_preserve_raw_source_time_and_do_not_replay_a_backlog(connections):
    create, _ = connections
    connection, adapter = create()
    connection.start()
    for index in range(20):
        publish(connection, adapter, frame(index))
        wait_for(lambda: connection.display_frames() and connection.display_frames()[-1][0] == index)
    assert connection.display_frames()[0][0] == 19
    batch = connection.display_frames(0)
    assert len(batch) <= 8
    assert batch[0][1] >= 19 / 15 - .5
    assert all(item[1] == item[0] / 15 for item in batch)
    assert connection.display_frames(19) == ()


def test_blocked_native_reader_stays_disconnecting_until_owner_returns(monkeypatch):
    release = threading.Event()

    class BlockedAdapter(ControlledAdapter):
        def frames(self):
            release.wait(3)
            return
            yield  # pragma: no cover

    monkeypatch.setattr(module, "NetworkStreamAdapter", BlockedAdapter)
    connection = module.VideoConnection("rtsp://camera.local/live", output_fps=15)
    connection.start()
    subscription = connection.subscribe()
    try:
        connection.stop(timeout=0)
        assert connection.running
        assert connection.status()["connection_state"] == "disconnecting"
        assert list(subscription.frames()) == []
        assert connection.latest_jpeg() is None
        with pytest.raises(RuntimeError, match="reconnect"):
            connection.start()
    finally:
        release.set()
        connection.stop(timeout=1)
    assert not connection.running
    assert connection.status()["connection_state"] == "disconnected"


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan")])
def test_connection_rejects_invalid_rate_before_creating_adapter(fps):
    with pytest.raises(ValueError, match="output_fps"):
        module.VideoConnection("rtsp://camera.local/live", output_fps=fps)


def test_four_second_viewer_can_recover_two_second_gap_without_missing_frames(connections):
    create, _ = connections
    connection, _ = create()
    connection._display_frames.extend((i, i/15, 40, 20, b"jpeg") for i in range(90))
    # A slow 1.4-second request used to discard almost all of these frames.
    frames = connection.display_frames(59, max_age_seconds=4.5, limit=64)
    assert [frame[0] for frame in frames] == list(range(60, 90))
    expired = connection.display_frames(0, max_age_seconds=4.5, limit=64)
    assert all(frame[1] >= 89/15-4.5 for frame in expired)
    assert len(expired) <= 64
