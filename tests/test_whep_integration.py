"""Real WHEP/ICE/SRTP/H264 across spawned processes, no external video source."""

import asyncio
import json
import multiprocessing
import os
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aiortc")
import aioice.ice
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import CodecContext, Packet, VideoFrame

from goodvision_harness.stream_adapter import NetworkStreamAdapter


class PatternTrack(VideoStreamTrack):
    def __init__(self, packets):
        super().__init__()
        self.index = 0
        self.started = time.monotonic()
        self.packets = packets

    async def recv(self):
        await asyncio.sleep(max(0, self.started + self.index / 30 - time.monotonic()))
        frame = Packet(self.packets[self.index % len(self.packets)])
        frame.pts = frame.dts = self.index * 3000
        frame.time_base = Fraction(1, 90000)
        self.index += 1
        return frame


async def server_main(control, stop):
    peers = set()
    # Pre-encode before the test interval, so software encoding cannot masquerade
    # as a receiver FPS bottleneck under load.
    encoder = CodecContext.create("libx264", "w")
    encoder.width, encoder.height = 2560, 720
    encoder.pix_fmt = "yuv420p"
    encoder.time_base, encoder.framerate = Fraction(1, 30), Fraction(30, 1)
    encoder.bit_rate = 2_500_000
    encoder.thread_count = 2
    encoder.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "x264-params": "keyint=60:min-keyint=60:scenecut=0",
    }
    rng = np.random.default_rng(42)
    base = np.repeat(
        np.repeat(rng.integers(0, 256, (45, 160, 3), dtype=np.uint8), 16, axis=0),
        16,
        axis=1,
    )
    packets = []
    for index in range(60):
        frame = VideoFrame.from_ndarray(
            np.roll(base, index * 3, axis=1), format="bgr24"
        )
        frame.pts, frame.time_base = index, Fraction(1, 30)
        packets.extend(bytes(packet) for packet in encoder.encode(frame))
    assert len(packets) == 60

    async def offer(request):
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peers.add(pc)
        pc.addTrack(PatternTrack(packets))
        for transceiver in pc.getTransceivers():
            transceiver.setCodecPreferences(
                [
                    c
                    for c in RTCRtpSender.getCapabilities("video").codecs
                    if c.mimeType == "video/H264"
                ]
            )
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=await request.text(), type="offer")
        )
        await pc.setLocalDescription(await pc.createAnswer())
        assert pc.getTransceivers()[0]._codecs[0].mimeType == "video/H264"
        return web.Response(
            status=201,
            text=pc.localDescription.sdp,
            content_type="application/sdp",
            headers={"Location": "/session"},
        )

    async def remove(request):
        for pc in list(peers):
            await pc.close()
            peers.discard(pc)
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/whep", offer)
    app.router.add_delete("/session", remove)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    control.send(site._server.sockets[0].getsockname()[1])
    try:
        while not stop.is_set():
            await asyncio.sleep(0.05)
    finally:
        for pc in peers:
            await pc.close()
        await runner.cleanup()
        control.close()


def serve(control, stop):
    # --network none has no non-loopback ICE candidates. Test only: advertise
    # loopback on both peers, so actual ICE/DTLS/SRTP remain exercised offline.
    aioice.ice.get_host_addresses = lambda **kwargs: ["127.0.0.1"]
    asyncio.run(server_main(control, stop))


def receive_on_loopback(options, slot, stop, result):
    from goodvision_harness.whep_process import _worker

    aioice.ice.get_host_addresses = lambda **kwargs: ["127.0.0.1"]
    _worker(options, slot, stop, result)


def computation(stop):
    obj = [{"id": n, "points": list(range(100))} for n in range(100)]
    while not stop.is_set():
        json.loads(json.dumps(obj))


def udp_drops():
    path = Path("/proc/net/snmp")
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    for keys, values in zip(lines[::2], lines[1::2], strict=True):
        if keys.startswith("Udp:"):
            return dict(
                zip(keys.split()[1:], map(int, values.split()[1:]), strict=True)
            )["RcvbufErrors"]


def test_real_whep_survives_parent_compute_and_slow_frame_consumption(monkeypatch):
    from goodvision_harness import whep_process

    receiver_class = whep_process.WhepProcessReceiver
    monkeypatch.setattr(
        whep_process,
        "WhepProcessReceiver",
        lambda options: receiver_class(
            options,
            worker=receive_on_loopback,
        ),
    )
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    stopped = context.Event()
    server = context.Process(target=serve, args=(writer, stopped))
    server.start()
    assert reader.poll(30)
    port = reader.recv()
    adapter = NetworkStreamAdapter(
        f"webrtc+http://127.0.0.1:{port}/whep", output_fps=15
    )
    done = threading.Event()
    workers = []
    stream = adapter.frames()
    duration = float(os.environ.get("WHEP_TEST_STRESS_SECONDS", "6"))
    watchdog = threading.Timer(duration + 40, adapter.stop)
    watchdog.start()
    try:
        first = next(stream)
        assert first.bgr.shape == (720, 2560, 3)
        child = adapter._whep_receiver
        before = udp_drops()
        workers = [threading.Thread(target=computation, args=(done,)) for _ in range(4)]
        for worker in workers:
            worker.start()
        # Exercise handoff synchronization while the consumer itself contends
        # for the GIL, then stop consuming altogether. Neither may stall RTP.
        active_deadline = time.monotonic() + 3
        active_frames = 0
        while time.monotonic() < active_deadline:
            assert next(stream).bgr.shape == first.bgr.shape
            active_frames += 1
        assert active_frames >= 5
        started_version = child._slot.header[0]
        done.wait(duration)
        done.set()
        for worker in workers:
            worker.join(5)
        after = udp_drops()
        latest = next(stream)
        published = child._slot.header[0] - started_version
        assert published >= duration * 10, (
            f"receiver stalled: only {published} frames in {duration}s"
        )
        assert latest.index > first.index + 30
        assert child.dropped_frames > 0  # Whole decoded frames, not UDP packets.
        if before is not None:
            assert after == before, (
                f"kernel UDP overflow under parent load: {after - before}"
            )
        print(
            json.dumps(
                {
                    "shared_publications": published,
                    "stress_seconds": duration,
                    "active_consumer_frames": active_frames,
                    "full_frames_skipped": child.dropped_frames,
                    "udp_overflow_delta": None if before is None else after - before,
                }
            )
        )
        adapter.stop()
        assert list(stream) == []
        assert child._closed
    finally:
        watchdog.cancel()
        done.set()
        adapter.stop()
        stream.close()
        for worker in workers:
            worker.join(5)
        stopped.set()
        server.join(5)
        if server.is_alive():
            server.terminate()
            server.join(2)
        reader.close()
        writer.close()
