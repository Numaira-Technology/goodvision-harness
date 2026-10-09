"""CPU/model doubles exercise ordering and durable admission; no model is run."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import goodvision_harness.face_recognition as face_module
from goodvision_harness.config import load_config
from goodvision_harness.face_observation_store import (
    FaceObservationCapacityError,
    FaceObservationStore,
)
from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
)
from goodvision_harness.grounding import SelectedFeaturesGroundingClient
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from goodvision_harness.video_to_stream import PushSampledSource


def window(index):
    ok, image = cv2.imencode(".jpg", np.full((96, 128, 3), index, np.uint8))
    assert ok
    return FrameWindow(
        index,
        (
            SampledFrame(
                index,
                index,
                index * 0.4,
                image.tobytes(),
                128,
                96,
            ),
        ),
    )


def wait_for(predicate):
    deadline = time.monotonic() + 8
    while not predicate():
        assert time.monotonic() < deadline, "test watchdog expired"
        threading.Event().wait(0.005)


def runtime(tmp_path, monkeypatch, *, gate=None):
    loaded, calls, finished = [], [], []

    class Model:
        def __init__(self):
            self.busy = False
            loaded.append(self)

        def get(self, image):
            assert not self.busy, "a model instance was concurrently shared"
            self.busy = True
            index = int(image[0, 0, 0])
            calls.append(index)
            try:
                if gate is not None and index == 0:
                    assert gate.wait(8), "test must release first observation"
                finished.append(index)
                return [
                    SimpleNamespace(
                        bbox=np.asarray([15, 10, 90, 85], np.float32),
                        normed_embedding=np.asarray([1.0, 0.0], np.float32),
                        det_score=0.95,
                    )
                ]
            finally:
                self.busy = False

    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _: Model())
    config = FaceRecognizerConfig(
        model_root=tmp_path / "models",
        gallery_path=tmp_path / "gallery.json",
        min_face_size=30,
        inference_instances=2,
        inference_threads=2,
    )
    value = FaceRecognitionRuntime(config)
    return value, loaded, calls, finished


def test_parallel_inference_commits_identity_and_facts_in_order(
    tmp_path,
    monkeypatch,
):
    gate = threading.Event()
    engine, loaded, calls, finished = runtime(tmp_path, monkeypatch, gate=gate)
    face = FaceRecognitionSession(
        engine, observation_store_path=tmp_path / "facts.db.sqlite"
    )
    order = []
    try:
        futures = [
            face.submit_window(
                window(index),
                on_admitted=lambda future, index=index: future.add_done_callback(
                    lambda done: order.append(done.result().window.index),
                ),
            )
            for index in range(16)
        ]
        # Submission order does not guarantee inference entry order. Wait for
        # both the blocked first call and the completed second call before
        # checking that they required separate model instances.
        wait_for(lambda: 0 in calls and 1 in finished)
        assert not futures[0].done() and not futures[1].done()
        assert not futures[0].cancel()
        assert len(loaded) == 2
        assert face._worker.store._connection is not None
        gate.set()
        face.close(wait=True)
        assert face._worker.store._connection is None
        assert order == list(range(16))
        assert sorted(calls) == list(range(16))
        records = face._worker.store.records()
        assert len(records) == 16
        assert all(
            row["state"] == "committed" and row["window"] is None for row in records
        )
        assert all(
            json.loads(row["raw"])["objects"][0]["identity_embedding"] == [1, 0]
            for row in records
        )
        assert all(json.loads(row["online"])["success"] for row in records)
        assert face.stats()["completed"] == 16 and face.stats()["backlog"] == 0
        assert face.stats()["ordered_commit_wait_seconds"] >= 0
    finally:
        gate.set()
        face.close(wait=True)


def test_nonblocking_close_keeps_connection_until_admitted_face_work_drains(
    tmp_path, monkeypatch,
):
    gate = threading.Event()
    engine, _, _, finished = runtime(tmp_path, monkeypatch, gate=gate)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "private")
    try:
        first = face.submit_window(window(0))
        second = face.submit_window(window(1))
        wait_for(lambda: 1 in finished)
        face.close(wait=False)
        assert face._worker.store._connection is not None
        assert not first.done() and not second.done()
        gate.set()
        face.close(wait=True)
        assert face._worker.store._connection is None
        assert first.result().classification is not None
        assert second.result().classification is not None
        assert [row["state"] for row in face._worker.store.records()] == [
            "committed", "committed",
        ]
    finally:
        gate.set()
        face.close(wait=True)


@pytest.mark.parametrize("wrapped", [False, True])
def test_camera_full_32_queue_drain_and_slow_face_burst_never_drop_admitted_frames(
    tmp_path,
    monkeypatch,
    wrapped,
):
    gate = threading.Event()
    engine, _, calls, _ = runtime(tmp_path, monkeypatch, gate=gate)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "private")
    config = load_config()
    config = replace(
        config,
        sliding_window=replace(
            config.sliding_window,
            size_frames=1,
            stride_frames=1,
        ),
    )
    client = face
    if wrapped:
        client = SelectedFeaturesGroundingClient(
            None,
            face,
            safety_event_labels=(),
        )
    delivered, ends = [], []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_analysis_result=lambda result: delivered.append(
                result.frame.source_frame_index
            ),
            on_end=lambda reason, error: ends.append((reason, error, face.stats())),
        ),
    )
    for index in range(32):
        frame = window(index).newest_frame
        assert session.push_camera_frame(
            frame.jpeg_bytes,
            width=128,
            height=96,
            source_frame_index=index,
            timestamp_seconds=index * 0.4,
        )
    assert not session.push_camera_frame(
        window(32).newest_frame.jpeg_bytes, width=128, height=96
    )
    assert session.camera_source.capacity_failures == 1
    assert session.camera_source.last_rejection == "capacity_fault"
    session.stop(drain=True)
    session.start()
    try:
        wait_for(lambda: face.stats()["submitted"] == 32)
        assert session.running and not ends
        gate.set()
        session.join(8)
        assert not session.running
        assert sorted(calls) == list(range(32))
        assert delivered == list(range(32))
        assert ends[0][:2] == ("stopped", None)
        assert ends[0][2]["completed"] == 32
        assert len(face._worker.store.records()) == 32
        assert all(
            row["raw"] and row["online"] and row["state"] == "committed"
            for row in face._worker.store.records()
        )
    finally:
        gate.set()
        session.stop(drain=True)
        session.join(8)


def test_source_close_preserves_accepted_frames_and_rejected_retry_keeps_index():
    source = PushSampledSource(max_queue=2)
    assert source.push_jpeg(b"a", width=1, height=1, source_frame_index=1)
    assert source.push_jpeg(b"b", width=1, height=1, source_frame_index=2)
    assert not source.push_jpeg(b"c", width=1, height=1, source_frame_index=3)
    frames = source.frames()
    assert next(frames).sample_index == 0
    assert source.push_jpeg(b"c", width=1, height=1, source_frame_index=3)
    source.close()
    assert [(frame.sample_index, frame.source_frame_index) for frame in frames] == [
        (1, 2),
        (2, 3),
    ]


def test_private_store_capacity_is_explicit_and_preserves_prior_admission(tmp_path):
    store = FaceObservationStore(tmp_path / "private.sqlite", max_bytes=3 * 1024**2)
    gallery = {"version": "v1", "entries": []}
    store.admit(1, "run", window(1), gallery)
    store.max_bytes = 1
    with pytest.raises(FaceObservationCapacityError):
        store.admit(2, "run", window(2), gallery)
    assert [row["source_sequence"] for row in store.records()] == [1]
    assert store.load(1)[0].newest_frame.jpeg_bytes == window(1).newest_frame.jpeg_bytes


def test_gallery_change_during_inference_rematches_without_losing_quality(
    tmp_path, monkeypatch
):
    engine, *_ = runtime(tmp_path, monkeypatch)
    engine.gallery.enroll("person", "Person", [1, 0])
    observation = engine.observe_window(window(1))
    assert observation.objects[0].attributes["status"] == "known"
    engine.gallery.delete("person")
    result = engine.complete_observation(observation)
    assert result.classification.face_observations[0].attributes["status"] == "unknown"
    assert "person_id" not in result.classification.face_observations[0].attributes


def test_failed_decode_has_private_source_and_failure_fact_and_next_frame_continues(
    tmp_path,
    monkeypatch,
):
    engine, *_ = runtime(tmp_path, monkeypatch)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "private")
    bad = replace(
        window(0), frames=(replace(window(0).newest_frame, jpeg_bytes=b"bad"),)
    )
    first, second = face.submit_window(bad), face.submit_window(window(1))
    face.close(wait=True)
    assert first.result().classification is None
    assert second.result().classification is not None
    rows = face._worker.store.records()
    assert rows[0]["state"] == "failed" and rows[0]["window"] is not None
    assert rows[0]["error_type"] == "FaceDecodeError"
    assert rows[1]["state"] == "committed"
    assert face.stats()["failed"] == 1


def test_ort_threads_are_applied_to_each_actual_selected_session(tmp_path):
    created = []

    class Session:
        def __init__(self, path, *, sess_options, providers):
            self.path, self.options, self.providers = path, sess_options, providers
            created.append(self)

        def get_session_options(self):
            return self.options

    model = SimpleNamespace(
        models={
            name: SimpleNamespace(
                model_file=f"{name}.onnx",
                session=object(),
            )
            for name in ("detection", "recognition")
        }
    )
    engine = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session)
    config = FaceRecognizerConfig(
        gallery_path=tmp_path / "gallery", inference_threads=3
    )
    face_module._configure_face_session_threads(model, engine, config)
    assert len(created) == 2
    assert all(
        session.options.intra_op_num_threads == 3
        and session.options.inter_op_num_threads == 1
        for session in created
    )
    assert all(
        model.models[name].session.path == f"{name}.onnx" for name in model.models
    )


def test_network_face_disk_admission_cannot_stall_source_or_sam_callbacks(tmp_path, monkeypatch):
    from goodvision_harness.video_to_stream import LiveFrame
    gate, entered, captured = threading.Event(), threading.Event(), []
    engine, _, calls, _ = runtime(tmp_path, monkeypatch)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "facts.sqlite")
    original = face.submit_window
    def slow_admit(*args, **kwargs):
        entered.set()
        assert gate.wait(8)
        return original(*args, **kwargs)
    monkeypatch.setattr(face, "submit_window", slow_admit)
    class Source:
        realtime = True
        def frames(self):
            for index in range(8):
                if index == 1:
                    assert entered.wait(2)
                yield LiveFrame(index, index/15, np.full((96, 128, 3), index, np.uint8))
        def stop(self):
            pass
    config = load_config()
    config = replace(config, video=replace(config.video, fps=15),
                     sliding_window=replace(config.sliding_window, size_frames=1, stride_frames=1))
    delivered, ends = [], []
    session = StreamingSession.for_stream(config, face, SessionCallbacks(
        on_display_frame=lambda jpeg, stamp, sequence: captured.append(sequence),
        on_analysis_result=lambda result: delivered.append(result.frame.source_frame_index),
        on_end=lambda reason, error: ends.append((reason, error)),
    ), stream_url="rtsp://test/source", video_stream=Source())
    try:
        session.start()
        assert entered.wait(2)
        wait_for(lambda: len(captured) == 8)
        assert session.running and not delivered
        assert session.organizer.snapshot()["counts"]["dropped"] == 6
        gate.set()
        session.join(8)
        assert not session.running
        assert delivered == calls == [0, 1]
        assert all(row["state"] == "committed" for row in face._worker.store.records())
        assert ends == [("disconnected", None)]
    finally:
        gate.set(); session.stop(drain=True); session.join(8)


def test_network_admission_failure_is_reported_even_at_end_of_capture():
    from goodvision_harness.face_admission import FaceAdmissionQueue
    def fail(*args, **kwargs):
        raise OSError("storage failure")
    admission = FaceAdmissionQueue(fail)
    assert admission.offer(window(0), branch=None)
    with pytest.raises(RuntimeError, match="live face admission failed"):
        admission.close()


def test_live_face_candidate_bytes_are_bounded_before_durable_admission():
    from goodvision_harness.face_admission import FaceAdmissionQueue
    seen = []
    admission = FaceAdmissionQueue(lambda *a, **k: seen.append(a), max_bytes=1)
    assert not admission.offer(window(0), branch=None)
    admission.close()
    assert not seen
