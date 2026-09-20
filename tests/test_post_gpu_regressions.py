"""Regressions from the 5060 Ti experiment, independent of GPU throughput."""

from concurrent.futures import Future
from threading import Event, Thread

from test_face_observation_pipeline import wait_for, window
from test_tracked_scene_delivery import capture, packet, runtime, tracked

from goodvision_harness.face_admission import FaceAdmissionQueue
from goodvision_harness.scene_delivery import TrackedSceneDisplayWorker


def test_face_capacity_covers_inference_and_all_ordered_delivery_callbacks():
    completions = []

    def admit(_window, *, branch):
        delivered = Future()
        completions.append(delivered)
        return delivered

    queue = FaceAdmissionQueue(admit, capacity=2)
    closed = Event()
    try:
        assert queue.offer(window(0), branch=None)
        assert queue.offer(window(1), branch=None)
        wait_for(lambda: len(completions) == 2)
        # Both have passed durable admission; inference/identity delivery is slow.
        for n in range(2, 50):
            assert not queue.offer(window(n), branch=None)
        assert queue.stats()["unfinished_observations"] == 2
        completions[0].set_result(None)
        assert queue.offer(window(50), branch=None)
        wait_for(lambda: len(completions) == 3)
        closer = Thread(target=lambda: (queue.close(), closed.set()))
        closer.start()
        assert not closed.wait(0.05)
        completions[1].set_result(None)
        completions[2].set_result(None)
        assert closed.wait(2.0)
        closer.join(1.0)
        assert queue.stats()["unfinished_observations"] == 0
    finally:
        for completion in completions:
            if not completion.done():
                completion.set_result(None)
        queue.close()


def test_expired_tail_above_final_tracking_watermark_does_not_stall_stop():
    scene, published = runtime(), []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60)
    completed = packet(10, 1.0)
    tails = (packet(11, 1.5), packet(12, 2.0), packet(13, 2.5))
    capture(scene, worker, completed)
    tracked(scene, (completed,))
    for source in tails:
        capture(scene, worker, source)
    worker.start()
    try:
        wait_for(lambda: len(published) == 1)
        assert not worker.wait_until_idle(0.02)
        assert worker.stats()["pending_frames"] == 3
        # Bytes remain present; the completed watermark will never advance.
        assert (
            scene.unavailable_display_sources(
                tuple((s.frame_sequence, s.media_timestamp) for s in tails)
            )
            == ()
        )
        scene.finish_tracking_sources()
        worker.notify_tracking()
        assert worker.wait_until_idle(0.5)
        assert worker.stats()["pending_frames"] == 0
        assert worker.stats()["unavailable_source_frames"] == 3
        assert len(published) == 1
        assert (
            scene.snapshot()["state"] == "running"
        )  # identity/evidence not closed to escape
        assert scene.snapshot()["tracking_completed_frame_sequence"] == 10
    finally:
        worker.close()
        scene.close()
