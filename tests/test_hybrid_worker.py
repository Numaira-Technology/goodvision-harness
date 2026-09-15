"""Hybrid scheduler/session contracts using explicit CPU doubles, no GPU claims."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from workers.sam_scene.continuous import Session, SessionSettings
from workers.sam_scene.hybrid import (
    HybridContinuation,
    HybridSettings,
    PriorityGpuGate,
    verify_checkpoint,
)
from workers.sam_scene.hybrid_tracker import PersonDetection
from workers.sam_scene.sam31 import SourceFrame, VideoBatch, VideoSettings


class Detector:
    gpu_ready = True  # A double must not turn production attestation on.

    def __init__(self):
        self.calls = []
        self.closed = False

    def warmup(self):
        pass

    def detect(self, image):
        self.calls.append(image.getpixel((0, 0)))
        return (PersonDetection((0.1, 0.1, 0.4, 0.9), 0.9),)

    def close(self):
        self.closed = True


def backend(calibrator=None):
    return HybridContinuation(
        SimpleNamespace(settings=VideoSettings(), gpu_ready=True),
        replace(SessionSettings(), enabled=True),
        settings=HybridSettings(calibration_seconds=0.5),
        detector=Detector(),
        calibrator=calibrator
        or (lambda image, source: (PersonDetection((0.11, 0.1, 0.4, 0.9), 0.9),)),
    )


def session(epoch="epoch"):
    return Session(epoch, "camera", epoch, time.monotonic())


def run(selected, state, *, count=1, start=0):
    images = [
        Image.new("RGB", (40, 24), (start + index, 1, 2)) for index in range(count)
    ]
    for index in range(start, start + count):
        state.sources[index] = SourceFrame(index, index * 0.5, b"")
    state.total_frames = start
    try:
        return list(selected.run(state, images, final=False))
    finally:
        for image in images:
            image.close()


def wait_completed(selected, state, count=1):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if selected._diagnostics(state.native_state)["calibrations_completed"] >= count:
            return
        time.sleep(0.005)
    pytest.fail("calibration did not complete")


def test_real_detector_contract_no_synthetic_mask_or_attestation():
    selected = backend()
    state = session()
    selected.warmup()
    try:
        output = run(selected, state)
        assert selected.gpu_ready is False
        assert [index for index, _ in output] == [0]
        snapshot = selected.serialize_snapshot(
            output[0][1],
            batch=VideoBatch("camera", "r0", (state.sources[0],)),
            index=0,
            epoch=state.predictor_epoch,
            image_size=(40, 24),
            max_objects=16,
        )
        obj = snapshot["objects"][0]
        assert obj["attributes"]["native_track"]["backend"] == "person_hybrid"
        assert obj["attributes"]["person_detection"]["geometry"] == "box"
        assert "mask_geometry" not in obj["attributes"]
        assert obj["observation_state"] == "observed"
        assert snapshot["frame_sequence"] == 0
        metadata = selected.route_metadata(state, [snapshot])
        assert metadata["observations"] == [
            {"frame_sequence": 0, "provisional_object_ids": ["0"]}
        ]
        wait_completed(selected, state)
        assert selected._diagnostics(state.native_state)["calibration_confirmed"] == 1
    finally:
        selected.discard(state)
        selected.close()


def test_slow_calibration_does_not_block_current_result_and_discard_drains():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def calibrate(image, source):
        entered.set()
        assert release.wait(3)
        return ()

    selected, state = backend(calibrate), session()
    selected.warmup()
    try:
        result = run(selected, state)
        assert len(result) == 1
        assert entered.wait(2)
        captured = state.native_state

        def discard():
            selected.discard(state)
            finished.set()

        thread = threading.Thread(target=discard)
        thread.start()
        assert not finished.wait(0.05)
        release.set()
        assert finished.wait(3)
        thread.join()
        assert state.native_state is None
        assert captured.calibrations_completed == 0
        assert captured.closed
    finally:
        release.set()
        selected.close()


def test_latest_pending_slot_replaces_old_frames_and_releases_images():
    selected, state = backend(), session()
    # Deliberately do not start the consumer. Scheduling remains bounded.
    try:
        for start in range(4):
            run(selected, state, start=start)
        diagnostics = selected._diagnostics(state.native_state)
        assert diagnostics["calibrations_scheduled"] == 4
        assert diagnostics["calibrations_replaced"] == 3
        assert selected._pending.source.frame_sequence == 3
        pending_image = selected._pending.image
        selected.discard(state)
        assert selected._pending is None
        with pytest.raises(ValueError):
            pending_image.getpixel((0, 0))
    finally:
        selected.close()


def test_retired_session_cannot_affect_new_epoch():
    selected = backend()
    selected.warmup()
    first = session("first")
    try:
        run(selected, first)
        wait_completed(selected, first)
        selected.discard(first)
        second = session("second")
        output = run(selected, second)
        assert output[0][1][0].object_id == 0
        assert output[0][1][0].calibration is None
        assert second.native_state.epoch == "second"
        selected.discard(second)
    finally:
        selected.close()


def test_calibration_failure_visible_and_does_not_claim_success():
    entered = threading.Event()

    def fail(image, source):
        entered.set()
        raise RuntimeError("test failure")

    selected, state = backend(fail), session()
    selected.warmup()
    try:
        run(selected, state)
        assert entered.wait(3)
        deadline = time.monotonic() + 3
        while (
            not state.native_state.calibration_failures and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert state.native_state.calibration_failures == 1
        assert state.native_state.calibrations_completed == 0
        assert selected.gpu_ready is False
    finally:
        selected.discard(state)
        selected.close()


def test_priority_gate_serializes_and_prioritizes_queued_foreground():
    gate = PriorityGpuGate()
    order, errors = [], []

    def use(name, foreground):
        try:
            with gate.use(foreground=foreground, cancelled=lambda: False):
                order.append(name)
        except Exception as error:
            errors.append(error)

    with gate.use(foreground=True, cancelled=lambda: False):
        background = threading.Thread(target=use, args=("background", False))
        foreground = threading.Thread(target=use, args=("foreground", True))
        background.start()
        foreground.start()
        deadline = time.monotonic() + 3
        while not gate.foreground_waiters and time.monotonic() < deadline:
            time.sleep(0.005)
        assert gate.foreground_waiters == 1
        assert order == []
    foreground.join(3)
    background.join(3)
    assert order == ["foreground", "background"] and errors == []


def test_cancelled_gate_does_not_enter():
    gate = PriorityGpuGate()
    with (
        pytest.raises(InterruptedError),
        gate.use(foreground=False, cancelled=lambda: True),
    ):
        pytest.fail("cancelled calibration entered GPU")
    assert not gate.active


def test_no_download_and_wrong_checkpoint_hash_rejected(tmp_path):
    missing = tmp_path / "missing.pth"
    with pytest.raises(RuntimeError, match="missing"):
        verify_checkpoint(missing)
    missing.write_bytes(b"not approved pretrained weights")
    with pytest.raises(RuntimeError, match="official weights"):
        verify_checkpoint(missing)


@pytest.mark.parametrize("malformed", [False, True])
def test_sam_calibration_uses_prompt_frame_output_without_video_confirmation(malformed):
    """A video propagation double intentionally suppresses unconfirmed masklets."""
    calls, native_state = [], {"one_source": True}

    class Model:
        def init_state(self, **kwargs):
            calls.append("init")
            assert len(kwargs["resource_path"]) == 1
            return native_state

        def add_prompt(self, **kwargs):
            calls.append("prompt")
            assert kwargs["inference_state"] is native_state
            assert kwargs["frame_idx"] == 0 and kwargs["text_str"] == "person"
            mask = np.zeros((1, 24, 40), dtype=bool)
            mask[:, 2:22, 4:16] = True
            return (1 if malformed else 0), {
                "out_obj_ids": np.array([0]),
                "out_probs": np.array([0.9]),
                "out_binary_masks": mask,
            }

        def propagate_in_video(self, **kwargs):
            pytest.fail(
                "single-source calibration must not wait for video confirmation"
            )

        def reset_state(self, state):
            calls.append("reset")
            assert state is native_state

    selected = backend()
    selected.video.backend = SimpleNamespace(
        predictor=SimpleNamespace(model=Model()),
        _torch=SimpleNamespace(
            inference_mode=nullcontext,
            autocast=lambda **kwargs: nullcontext(),
            bfloat16="bfloat16",
            cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")),
        ),
    )
    with Image.new("RGB", (40, 24)) as image:
        if malformed:
            with pytest.raises(RuntimeError, match="source ordering"):
                selected._sam_calibrate(image, SourceFrame(12, 3.0, b""))
        else:
            detected = selected._sam_calibrate(image, SourceFrame(12, 3.0, b""))
            assert len(detected) == 1 and detected[0].confidence == 0.9
            assert detected[0].bbox == (0.1, 2 / 24, 0.4, 22 / 24)
    assert calls == ["init", "prompt", "reset", "sync"]
    assert native_state == {}
    selected.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_threshold": float("nan")},
        {"low_score_threshold": 0.45},
        {"calibration_seconds": 0.1},
        {"track_ttl_seconds": 6},
        {"checkpoint_path": ""},
    ],
)
def test_invalid_settings_fail_closed(kwargs):
    with pytest.raises(ValueError):
        HybridSettings(**kwargs)
