from __future__ import annotations

import asyncio
import socket
import threading
from fractions import Fraction
from types import SimpleNamespace

import pytest

from goodvision_harness.stream_adapter import NetworkStreamAdapter


class Frame:
    time_base = Fraction(1, 30)

    def __init__(self, pts=0):
        self.pts = pts
        self.image = object()
        self.converted = False

    def to_ndarray(self, *, format):
        assert format == "bgr24"
        self.converted = True
        return self.image


class BlockedFrame(Frame):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def to_ndarray(self, *, format):
        self.entered.set()
        # A watchdog lets the old blocking implementation fail, not hang pytest.
        self.release.wait(timeout=2)
        self.finished.set()
        return super().to_ndarray(format=format)


class Track:
    def __init__(self, *frames, end=False):
        self.frames = iter(frames)
        self.end = end
        self.receive_cancelled = False

    async def recv(self):
        frame = next(self.frames, None)
        if frame is not None:
            return frame
        if self.end:
            raise EOFError
        try:
            await asyncio.Future()
        finally:
            self.receive_cancelled = True


def adapter():
    return NetworkStreamAdapter("webrtc+http://127.0.0.1/whep", output_fps=15)


def pump(receiver, track):
    return receiver._pump_track(
        track, SimpleNamespace(connectionState="connected"), EOFError,
    )


def test_udp_receives_while_color_conversion_is_blocked():
    async def scenario():
        frame = BlockedFrame()
        receiver = adapter()
        loop = asyncio.get_running_loop()
        delivered = loop.create_future()

        class DatagramReceiver(asyncio.DatagramProtocol):
            def datagram_received(self, data, address):
                if not delivered.done():
                    delivered.set_result((data, frame.finished.is_set()))
                frame.release.set()

        transport, _ = await loop.create_datagram_endpoint(
            DatagramReceiver, local_addr=("127.0.0.1", 0),
        )
        address = transport.get_extra_info("sockname")

        def send_during_conversion():
            if frame.entered.wait(timeout=3):
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(b"video-packet", address)

        sender = threading.Thread(target=send_during_conversion, daemon=True)
        sender.start()
        task = asyncio.create_task(pump(receiver, Track(frame)))
        try:
            data, conversion_finished = await asyncio.wait_for(delivered, timeout=4)
            assert data == b"video-packet"
            assert not conversion_finished, "UDP reception waited for frame conversion"
        finally:
            frame.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            transport.close()
            sender.join(timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["stop", "cancel"])
def test_shutdown_during_conversion_does_not_publish_a_late_frame(action):
    async def scenario():
        frame = BlockedFrame()
        receiver = adapter()
        track = Track(frame)
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(pump(receiver, track))

        def shutdown():
            if action == "stop":
                receiver.stop()
            else:
                task.cancel()
            frame.release.set()

        def schedule_shutdown():
            if frame.entered.wait(timeout=3):
                loop.call_soon_threadsafe(shutdown)

        stopper = threading.Thread(target=schedule_shutdown, daemon=True)
        stopper.start()
        try:
            if action == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=4)
            else:
                await asyncio.wait_for(task, timeout=4)
            assert receiver._queue.empty()
            assert track.receive_cancelled
        finally:
            frame.release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            stopper.join(timeout=1)

    asyncio.run(scenario())


def test_conversion_preserves_source_sampling_order_and_bounded_handoff():
    receiver = adapter()
    frames = [Frame(index) for index in range(8)]

    asyncio.run(pump(receiver, Track(*frames, end=True)))

    assert [frame.pts for frame in frames if frame.converted] == [0, 2, 4, 6]
    assert receiver._queue.qsize() == 2
    output = [receiver._queue.get_nowait(), receiver._queue.get_nowait()]
    assert [frame.index for frame in output] == [2, 3]
    assert [frame.timestamp_seconds for frame in output] == pytest.approx([4/30, 6/30])
    assert [frame.bgr for frame in output] == [frames[4].image, frames[6].image]


def test_conversion_failure_cleans_up_pending_receive():
    class BrokenFrame(Frame):
        def to_ndarray(self, *, format):
            raise ValueError("conversion failed")

    receiver = adapter()
    track = Track(BrokenFrame())
    with pytest.raises(ValueError, match="conversion failed"):
        asyncio.run(pump(receiver, track))
    assert receiver._queue.empty()
    assert track.receive_cancelled
