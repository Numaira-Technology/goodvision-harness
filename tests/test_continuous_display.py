import io
import time
from threading import Event

import pytest
from PIL import Image

from goodvision_harness import continuous_display
from goodvision_harness.continuous_display import ContinuousTrackedSceneDisplayWorker
from goodvision_harness.result_bus import FramePacket


def packet(sequence):
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (sequence % 256, 0, 0)).save(out, "JPEG")
    return FramePacket(out.getvalue(), sequence / 15, sequence, "test")


class Scene:
    stream_session_id = "test-session"

    def __init__(self):
        self.requested = []
        self.events = []

    def display_snapshot(self, timestamp):
        self.requested.append(timestamp)
        return {"state": "running", "entities": [], "events": self.events}


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("continuous display did not make progress")
        time.sleep(0.005)


def test_unsampled_capture_frames_continue_without_waiting_for_model():
    scene, published = Scene(), []
    worker = ContinuousTrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    worker.start()
    try:
        for index in range(6):
            worker.submit(packet(index), retain_source=False)
            wait_for(lambda expected=index + 1: len(published) == expected)
        assert [p.frame_sequence for p in published] == list(range(6))
        assert scene.requested == [index / 15 for index in range(6)]
        assert worker.stats()["distinct_source_frames"] == 6
        assert worker.wait_until_idle(0.1)
    finally:
        worker.close()


def test_delay_is_honored_and_late_events_repaint_same_source():
    scene, published = Scene(), []
    worker = ContinuousTrackedSceneDisplayWorker(
        scene, published.append, max_fps=60, delay_seconds=0.15
    )
    worker.start()
    try:
        start = time.monotonic()
        worker.submit(packet(1), retain_source=False)
        assert not worker.wait_until_idle(0.05)
        assert not published
        wait_for(lambda: len(published) == 1)
        assert time.monotonic() - start >= 0.15
        worker.notify_events()
        wait_for(lambda: len(published) == 2)
        assert published[0].frame_sequence == published[1].frame_sequence == 1
        assert worker.stats()["same_source_repaints"] == 1
    finally:
        worker.close()


def test_buffer_is_bounded_and_slow_render_does_not_block_capture(monkeypatch):
    entered, release = Event(), Event()

    def render(jpeg, *args, **kwargs):
        entered.set()
        release.wait(3)
        return jpeg

    monkeypatch.setattr(continuous_display, "render_scene_jpeg", render)
    published = []
    worker = ContinuousTrackedSceneDisplayWorker(
        Scene(), published.append, max_fps=60, max_buffer_bytes=3000
    )
    worker.start()
    try:
        worker.submit(packet(0))
        assert entered.wait(2)
        for index in range(1, 30):
            worker.submit(packet(index), retain_source=False)
        assert worker.stats()["pending_bytes"] <= 3000
        assert worker.stats()["discarded_frames"] > 0
        worker.close(0.01)
        release.set()
        worker.close(2)
        assert not published
        assert worker.stats()["pending_bytes"] == 0
    finally:
        release.set()
        worker.close()


def test_byte_pressure_preserves_frames_until_minimum_delay():
    published = []
    worker = ContinuousTrackedSceneDisplayWorker(
        Scene(), published.append, max_fps=60, delay_seconds=0.15, max_buffer_bytes=1500
    )
    worker.start()
    try:
        for index in range(20):
            worker.submit(packet(index), retain_source=False)
            time.sleep(0.01)
        wait_for(lambda: bool(published))
        assert published[0].stream_delay_seconds >= 0.15
        assert worker.stats()["discarded_frames"] > 0
    finally:
        worker.close()
