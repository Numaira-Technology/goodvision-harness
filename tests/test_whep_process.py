from __future__ import annotations

import multiprocessing
import os
import time

import numpy as np
import pytest

from goodvision_harness.stream_adapter import StreamAdapterError
from goodvision_harness.video_to_stream import LiveFrame
from goodvision_harness.whep_process import SharedFrameSlot, WhepProcessReceiver


def picture(index, shape=(6, 8, 3)):
    return LiveFrame(
        index=index,
        timestamp_seconds=index / 30,
        bgr=np.full(shape, index % 256, np.uint8),
    )


def finite_worker(options, slot, stop, result):
    slot.publish(picture(os.getpid(), (4, 6, 3)))
    options["ready"].set()
    options["release"].wait(5)
    for index in range(20):
        slot.publish(picture(index, (8, 10, 3)))
    result.send(("end", None))
    result.close()


def waiting_worker(options, slot, stop, result):
    options["ready"].set()
    stop.wait(5)
    result.send(("end", None))
    result.close()


def failed_worker(options, slot, stop, result):
    result.send(("error", "missing_support"))
    result.close()


def crashed_worker(options, slot, stop, result):
    os._exit(7)


def stuck_worker(options, slot, stop, result):
    slot.publish(picture(1))
    time.sleep(60)


def test_slot_keeps_latest_owned_image_and_supports_resolution_changes():
    slot = SharedFrameSlot(multiprocessing.get_context("spawn"), 2048)
    source = picture(1)
    assert slot.publish(source)
    source.bgr[:] = 99
    first = slot.take(0)
    assert np.all(first.bgr == 1)
    slot.publish(picture(2, (10, 12, 3)))
    slot.publish(picture(3, (12, 10, 3)))
    last = slot.take(0)
    assert last.index == 3 and last.timestamp_seconds == 0.1
    assert last.bgr.shape == (12, 10, 3) and np.all(last.bgr == 3)
    assert np.all(first.bgr == 1)  # Producer cannot mutate an already yielded frame.
    assert slot.overwritten.value == 1
    assert slot.take(0) is None


def test_slow_consumer_never_blocks_publishing_and_capacity_is_enforced():
    slot = SharedFrameSlot(multiprocessing.get_context("spawn"), 1024)
    with slot.lock:
        assert slot.publish(picture(1)) is False
    assert slot.busy.value == 1
    assert slot.header[0] == 0
    with pytest.raises(ValueError, match="capacity"):
        slot.publish(picture(1, (100, 100, 3)))
    assert slot.publish(picture(2))


def test_spawned_receiver_latest_frame_and_clean_recreation():
    context = multiprocessing.get_context("spawn")
    for _ in range(2):
        ready, release = context.Event(), context.Event()
        receiver = WhepProcessReceiver(
            {"ready": ready, "release": release}, capacity=2048, worker=finite_worker
        )
        frames = receiver.frames()
        try:
            first = next(frames)
            assert ready.wait(5)
            assert first.index == receiver.pid != os.getpid()
            receiver._process.join(0.05)
            release.set()
            receiver._process.join(10)
            assert not receiver._process.is_alive()
            # All later images replace one slot while this consumer is paused.
            rest = list(frames)
            assert len(rest) == 1 and rest[0].index == 19
            assert np.all(first.bgr == first.index % 256)
            assert receiver.dropped_frames == 19
        finally:
            release.set()
            frames.close()
            receiver.close()
        assert receiver._closed


@pytest.mark.parametrize(
    "worker,code",
    [(failed_worker, "missing_support"), (crashed_worker, "connection_failed")],
)
def test_child_failure_ends_consumer_and_preserves_safe_error_code(worker, code):
    receiver = WhepProcessReceiver({}, capacity=1024, worker=worker)
    with pytest.raises(StreamAdapterError) as caught:
        list(receiver.frames())
    assert caught.value.code == code
    assert receiver._closed


def test_stop_before_start_allocates_no_process():
    receiver = WhepProcessReceiver({}, capacity=1024, worker=crashed_worker)
    receiver.stop()
    assert list(receiver.frames()) == []
    assert not receiver._started
    receiver.close()


def test_stop_during_startup_and_forced_cleanup_are_bounded():
    import threading

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    receiver = WhepProcessReceiver(
        {"ready": ready}, capacity=1024, worker=waiting_worker
    )
    output = []
    consumer = threading.Thread(target=lambda: output.extend(receiver.frames()))
    consumer.start()
    try:
        assert ready.wait(10)
        receiver.stop()
        consumer.join(5)
        assert not consumer.is_alive() and not output
        assert receiver._closed
    finally:
        receiver.close()

    receiver = WhepProcessReceiver(
        {}, capacity=1024, worker=stuck_worker, shutdown_timeout=0.05
    )
    stream = receiver.frames()
    try:
        assert next(stream).index == 1
        started = time.monotonic()
        receiver.stop()
        assert list(stream) == []
        assert time.monotonic() - started < 3
        assert receiver._closed
    finally:
        stream.close()
        receiver.close()
