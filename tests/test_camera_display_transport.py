"""Display delivery retains source facts independently of admission and scene work."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_camera_display_protocol import RUN, batch, bus, event, header, scene, source

from goodvision_harness.camera_display import CameraSocketChannel


def test_incremental_geometry_is_idempotent_and_events_remain_full():
    display = bus()
    first = display.delta(None, stream_session_id=RUN)
    assert display.delta(first["metadata_revision"], stream_session_id=RUN) is None
    display.observe_scene(scene(2, event()))
    display.observe_batch(batch(source(1, 1.0)))
    update = display.delta(first["metadata_revision"], stream_session_id=RUN)
    assert [a["source_sequence"] for a in update["anchors"]] == [1]
    assert update["events"][0]["valid_from_timestamp"] == 2
    display.observe_batch(batch(source(1, 1.0)))
    assert display.delta(update["metadata_revision"], stream_session_id=RUN) is None
    empty = SimpleNamespace(frame_sequence=2, timestamp_seconds=2.0, objects=())
    display.observe_batch(batch(empty))
    tail = display.delta(update["metadata_revision"], stream_session_id=RUN)
    assert tail["anchors"] == [
        {
            "source_sequence": 2,
            "media_timestamp": 2.0,
            "epoch": "epoch-one",
            "subjects": [],
        }
    ]
    assert tail["events"] == update["events"]
    assert tail["event_snapshot_revision"] == update["event_snapshot_revision"]


def test_native_token_epoch_and_source_order_barriers_are_atomic():
    display = bus()
    display.bind_native_run(RUN, (1, 2))
    assert not display.observe_batch(batch(source(1, 1)), native_token=(1, 1))
    assert not display.observe_batch(batch(source(1, 1)))
    assert display.observe_batch(batch(source(2, 2)), native_token=(1, 2))
    revision = display.delta(None, stream_session_id=RUN)["metadata_revision"]
    display.observe_batch(batch(source(1, 1)), native_token=(1, 2))
    assert display.delta(revision, stream_session_id=RUN) is None
    next_epoch = batch(
        SimpleNamespace(frame_sequence=3, timestamp_seconds=3, objects=())
    )
    next_epoch.predictor_epoch = "epoch-two"
    display.observe_batch(next_epoch, native_token=(1, 2))
    assert not display.observe_batch(batch(source(4, 4)), native_token=(1, 2))
    display.reset(RUN, native_token=(2, 3))
    display.accept_source(header())
    assert not display.observe_batch(batch(source(5, 5)), native_token=(1, 2))
    assert display.delta(None, stream_session_id=RUN)["metadata_revision"] > revision
    assert display.snapshot()["anchors"] == []


class Socket:
    def __init__(self):
        self.messages = []
        self.active = 0
        self.closed = False

    async def send_text(self, encoded):
        payload = json.loads(encoded)
        self.active += 1
        assert self.active == 1
        await asyncio.sleep(0)
        self.messages.append(payload)
        self.active -= 1

    async def close(self, **kwargs):
        self.closed = True


async def until(predicate):
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("display task did not progress")


def test_single_writer_prioritizes_ack_and_publishes_tail_without_upload():
    async def run():
        display, socket = bus(), Socket()
        channel = CameraSocketChannel(socket, display, interval=0.005)
        channel.enable(RUN)
        ack = {"type": "camera_frame_ack", "accepted": True}
        future = asyncio.get_running_loop().create_future()
        channel.controls.append((ack, future))
        channel.pending = {
            "type": "camera_display_metadata",
            **display.delta(None, stream_session_id=RUN),
        }
        channel.wake.set()
        channel.start()
        try:
            await future
            await until(lambda: len(socket.messages) == 2)
            assert socket.messages[0] == ack
            await asyncio.sleep(0.03)
            assert (
                len(socket.messages) == 2
            )  # No repeated full history polling messages.
            await channel.send({"type": "ended"})
            display.observe_batch(batch(source(1, 1.0)))
            await until(lambda: len(socket.messages) == 4)
            assert socket.messages[-1]["anchors"][0]["source_sequence"] == 1
            assert socket.messages[-1]["type"] == "camera_display_metadata"
        finally:
            await channel.close()
        assert channel.writer.done() and channel.producer.done()

    asyncio.run(run())


def test_frame_ack_does_not_wait_for_shared_thread_pool():
    async def run():
        socket = Socket()
        channel = CameraSocketChannel(socket, bus())
        channel.start()
        try:
            with patch("goodvision_harness.camera_display.asyncio.to_thread",
                       side_effect=AssertionError("ACK used shared executor")):
                await channel.send({"type": "camera_frame_ack", "accepted": True})
            assert socket.messages == [
                {"type": "camera_frame_ack", "accepted": True}
            ]
        finally:
            await channel.close()

    asyncio.run(run())


def test_queued_old_run_geometry_cannot_cross_enable_new_run():
    async def run():
        display, socket = bus(), Socket()
        channel = CameraSocketChannel(socket, display, interval=0.005)
        channel.enable(RUN)
        channel.pending = {
            "type": "camera_display_metadata",
            **display.delta(None, stream_session_id=RUN),
        }
        display.reset("new")
        display.accept_source({**header(), "stream_session_id": "new"})
        channel.enable("new")
        channel.start()
        try:
            await until(lambda: bool(socket.messages))
            assert {m["stream_session_id"] for m in socket.messages} == {"new"}
        finally:
            await channel.close()

    asyncio.run(run())


def test_producer_failure_closes_transport_instead_of_silently_stopping_updates():
    class FailedMetadata:
        def delta(self, *args, **kwargs):
            raise RuntimeError("test failure")

    async def run():
        socket = Socket()
        channel = CameraSocketChannel(socket, FailedMetadata(), interval=0.005)
        channel.enable(RUN)
        channel.start()
        try:
            await until(lambda: socket.closed)
            assert channel.closed
            with pytest.raises(RuntimeError, match="closed"):
                await channel.send({"type": "camera_frame_ack"})
        finally:
            await channel.close()

    asyncio.run(run())


def test_oversize_transport_message_is_explicitly_bounded():
    with pytest.raises(ValueError, match="capacity"):
        CameraSocketChannel._encode({"data": "x" * (1024 * 1024)})
