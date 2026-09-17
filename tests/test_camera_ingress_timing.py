"""Camera admission stages remain independent from inference and drain clocks."""

import json
import time
from threading import RLock
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_sam_tracking_pipeline import setup_pipeline
from test_video_tracking_session import frames, submit, wait_for

from goodvision_harness import service
from goodvision_harness.camera_display import CameraDisplayMetadata
from goodvision_harness.service import HarnessRuntime


def runtime_for_ingress():
    # Keep the real admission/generation checks, without starting any models.
    runtime = object.__new__(HarnessRuntime)
    runtime.config = SimpleNamespace(
        rich_labeling=SimpleNamespace(backend="sam31_stream")
    )
    runtime._camera_max_image_side = 1280
    runtime._display_settings = SimpleNamespace(max_image_side=None, jpeg_quality=85)
    runtime._control_lock = RLock()
    runtime._state_lock = RLock()
    runtime._state = {"pipeline_state": "running"}
    runtime._session_mode = "camera"
    runtime._camera_owner = "private-owner"
    runtime._scene = SimpleNamespace(mode="active")
    runtime._generation = 7
    runtime._camera_next_sample_at = 0
    runtime._camera_sample_period = 0.4
    runtime._camera_frame_sequence = 0
    runtime._session_started_monotonic = time.monotonic()
    runtime._last_camera_receipt = None
    runtime._camera_display_metadata = CameraDisplayMetadata()
    runtime._camera_display_metadata.reset("ingress-run")
    queued, displayed = [], []
    runtime._session = SimpleNamespace(
        stream_session_id="ingress-run", running=True,
        push_camera_frame=lambda jpeg, **stamp: queued.append((jpeg, stamp)) or True,
    )
    runtime._on_display_frame = lambda *args: displayed.append(args)
    return runtime, queued, displayed


def source():
    ok, jpeg = cv2.imencode(".jpg", np.zeros((10, 20, 3), dtype=np.uint8))
    assert ok
    return jpeg.tobytes()


def push(runtime, jpeg, *, modern=True):
    return runtime.push_camera_frame(
        jpeg, width=20, height=10, owner="private-owner",
        **({"stream_session_id": "ingress-run", "generation": 3,
            "crop_version": 0, "source_sequence": 42, "media_timestamp": 1.25}
           if modern else {}),
    )


@pytest.mark.parametrize(
    "local,modern,side,backend,reused",
    [(True, True, None, "sam31_stream", True),
     (False, True, None, "sam31_stream", False),
     (True, False, None, "sam31_stream", False),
     (True, True, 960, "sam31_stream", False),
     (True, True, None, "vlm", False)],
)
def test_only_local_gvc2_unresized_sources_bypass_second_jpeg_encode(
    local, modern, side, backend, reused, monkeypatch,
):
    runtime, queued, displayed = runtime_for_ingress()
    runtime.config.rich_labeling.backend = backend
    runtime._display_settings.max_image_side = side
    monkeypatch.setenv("LOCAL_CAMERA_DISPLAY_ENABLED", str(local).lower())
    encoded, logs = [], []
    monkeypatch.setattr(
        service, "encode_jpeg",
        lambda image, **options: encoded.append(options) or (b"display-copy", 20, 10),
    )
    monkeypatch.setattr(
        service, "log_timing", lambda name, **fields: logs.append(fields),
    )
    jpeg = source()
    assert push(runtime, jpeg, modern=modern)
    assert len(queued) == len(displayed) == 1
    assert queued[0][0] is jpeg  # face/source keeps the incoming immutable JPEG
    assert displayed[0][0] == (jpeg if reused else b"display-copy")
    assert len(encoded) == (0 if reused else 1)
    assert logs[0]["display_source_reused"] is reused
    assert logs[0]["outcome"] == "accepted"


def test_ingress_timing_attributes_cost_without_logging_content(monkeypatch):
    runtime, _, _ = runtime_for_ingress()
    clock, logs = [100.0], []
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setenv("LOCAL_CAMERA_DISPLAY_ENABLED", "false")
    monkeypatch.setattr(
        service, "log_timing", lambda name, **fields: logs.append(fields),
    )

    def take(seconds, result):
        clock[0] += seconds
        return result

    monkeypatch.setattr(
        service.cv2, "imdecode", lambda *_: take(.2, np.zeros((10, 20, 3))),
    )
    monkeypatch.setattr(
        service, "encode_jpeg", lambda *_, **__: take(.3, (b"private-copy", 20, 10)),
    )
    runtime._session.push_camera_frame = lambda *_, **__: take(.4, True)
    runtime._on_display_frame = lambda *_: take(.5, None)
    assert push(runtime, b"private-input")
    row, = logs
    for key, seconds in {
        "decode_seconds": .2, "display_encode_seconds": .3,
        "source_queue_seconds": .4, "display_callback_seconds": .5,
        "control_lock_wait_seconds": 0, "display_callback_state_wait_seconds": 0,
        "total_seconds": 1.4,
    }.items():
        assert row[key] == pytest.approx(seconds)
    assert row["source_sequence"] == 42 and row["stream_session_id"] == "ingress-run"
    assert "private" not in json.dumps(row)
    assert not any(key.startswith("_") for key in row)


def test_duplicate_ack_and_rejected_capacity_do_not_offer_sam_twice(monkeypatch):
    runtime, queued, displayed = runtime_for_ingress()
    monkeypatch.setenv("LOCAL_CAMERA_DISPLAY_ENABLED", "true")
    logs = []
    monkeypatch.setattr(
        service, "log_timing", lambda name, **fields: logs.append(fields),
    )
    jpeg = source()
    assert push(runtime, jpeg)
    assert push(runtime, jpeg)
    assert len(queued) == len(displayed) == 1
    assert logs[-1]["outcome"] == "duplicate"
    runtime._session.push_camera_frame = lambda *_, **__: False
    assert not runtime.push_camera_frame(
        jpeg, width=20, height=10, owner="private-owner",
        stream_session_id="ingress-run", generation=3, crop_version=0,
        source_sequence=43, media_timestamp=2,
    )
    assert len(displayed) == 1
    assert logs[-1]["outcome"] == "rejected"


def test_invalid_image_keeps_safe_partial_decode_timing(monkeypatch):
    runtime, queued, displayed = runtime_for_ingress()
    logs = []
    monkeypatch.setattr(
        service, "log_timing", lambda name, **fields: logs.append(fields),
    )
    with pytest.raises(ValueError):
        push(runtime, b"private-invalid-image")
    assert queued == displayed == []
    assert logs[0]["outcome"] == "error"
    assert logs[0]["error_type"] == "ValueError"
    assert logs[0]["decode_seconds"] >= 0
    assert "private" not in json.dumps(logs)


def test_camera_stop_freezes_capture_clock_before_slow_face_drain(tmp_path):
    runtime, _, _ = runtime_for_ingress()
    received = []
    server, _, coordinator = setup_pipeline(tmp_path, received.append)
    runtime._rich_labels = coordinator
    runtime._update_state = lambda **values: runtime._state.update(values)
    runtime._session.stream_session_id = "camera"
    stopped = []

    def stop(*, drain):
        assert drain and not coordinator._flush
        assert coordinator._capture_finished_at is not None
        stopped.append(coordinator._capture_finished_at)

    runtime._session.stop = stop
    runtime._session.join = lambda **_: time.sleep(.06)  # Face still draining
    try:
        submit(coordinator, frames(1, 1))
        wait_for(lambda: coordinator.stats()["processed_frames"] == 1)
        server.entered.clear()
        server.release.clear()
        submit(coordinator, frames(2, 1))
        assert server.entered.wait(2)
        assert runtime.stop_session(camera_owner="private-owner")
        frozen = coordinator.stats()["capture_elapsed_seconds"]
        assert runtime.stop_session(camera_owner="private-owner")
        assert stopped[0] == stopped[1]
        assert coordinator.stats()["capture_elapsed_seconds"] == frozen
        # Marking capture ended is not closing/flushing: accepted work delayed
        # in an upstream queue can still arrive and must not silently disappear.
        submit(coordinator, frames(3, 1))
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(3)
        stats = coordinator.stats()
        assert stats["processed_frames"] == 3
        assert stats["completed_during_capture"] == 1
        assert stats["completed_after_capture"] == 2
        assert stats["capture_elapsed_seconds"] == frozen
        old_generation = coordinator._generation
        coordinator.reset("camera")
        assert not coordinator.mark_capture_finished(
            stream_session_id="camera", generation=old_generation,
        )
        assert coordinator._capture_finished_at is None
    finally:
        server.release.set()
        coordinator.close()
