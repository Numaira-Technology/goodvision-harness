"""Local face and remote events must progress independently on exact source frames.

Thread waits are deadlock watchdogs, not production inference latency claims.
No real model, camera, network service or webhook is used here.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.face_recognition import FaceRecognitionSession
from goodvision_harness.face_tracking import FaceIdentityTracker
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundedEventClassification,
    GroundingResponse,
    ModelObject,
    MultiGroundedClassification,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.privacy import PUBLIC_ANALYSIS_ERROR
from goodvision_harness.result_bus import ResultBus
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def _completion_timings(caplog):
    return [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("analysis_branch_timing ")
    ]


@pytest.mark.parametrize("branch", ["stranger", "safety"])
@pytest.mark.parametrize("fails", [False, True])
def test_branch_completion_timing_counts_requests_not_context_or_events(
    caplog, branch, fails,
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with _session() as (session, _pool, safety, face, *_outputs):
        session.stream_session_id = "rtsp://user:private@camera"
        frames = (_window(0).newest_frame, _window(1).newest_frame)
        window = FrameWindow(1, frames)
        future = Future()
        if fails:
            future.set_exception(RuntimeError("PRIVATE exception response"))
        else:
            detector = face if branch == "stranger" else safety
            future.set_result(detector.classify_window(window))
        session._apply_window_future(future, window, branch=branch)
        record, = _completion_timings(caplog)
        assert record["branch"] == (
            "insightface" if branch == "stranger" else "vlm_frame"
        )
        assert record["context_frame_sequences"] == [0, 10]
        assert record["source_frame_sequence"] == 10
        assert record["source_timestamp_seconds"] == 0.4
        assert record["completed_requests"] == 1
        assert record["completed_observations"] == int(not fails)
        assert record["success"] is not fails
        assert record["model_seconds"] == (None if fails else 0.01)
        assert record["error_type"] == ("RuntimeError" if fails else None)
        assert record["stream_session_id"] is None
        assert all(
            text not in str(record) for text in ("PRIVATE", "private", "Face evidence")
        )


@pytest.mark.parametrize("fails", [False, True])
def test_face_only_completion_timing_is_insightface_without_independent_branch(
    caplog, fails,
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    config = load_config()
    client = FaceRecognitionSession(SimpleNamespace(settings=config.inference))
    session = StreamingSession.for_camera(config, client, SessionCallbacks())
    window, future = _window(0), Future()
    if fails:
        future.set_exception(ValueError("PRIVATE face input"))
    else:
        future.set_result(Face(config.inference).classify_window(window))
    session._apply_window_future(future, window)
    record, = _completion_timings(caplog)
    assert record["branch"] == "insightface"
    assert record["success"] is not fails
    assert record["completed_requests"] == 1
    assert "PRIVATE" not in str(record)


def test_completed_face_response_error_keeps_measured_time_without_message(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with _session() as (session, _pool, _safety, _face, *_outputs):
        window, future = _window(0), Future()
        future.set_result(GroundingResponse(
            window, None, 0.8, error="PRIVATE recognizer diagnostics",
        ))
        session._apply_window_future(future, window, branch="stranger")
        record, = _completion_timings(caplog)
        assert record["branch"] == "insightface"
        assert record["success"] is False
        assert record["model_seconds"] == 0.8
        assert record["completed_observations"] == 0
        assert record["error_type"] == "ResponseError"
        assert "PRIVATE" not in str(record)


def _window(index):
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), (index * 30) % 256, np.uint8))
    assert ok
    return FrameWindow(
        index, (SampledFrame(index, index * 10, index * 0.4, jpeg.tobytes(), 64, 48),)
    )


def _wait(predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline, "branch did not finish within watchdog"
        threading.Event().wait(0.005)


class Safety:
    def __init__(self, settings, gate=None):
        self.settings = settings
        self.gate = gate
        self.entered = threading.Event()
        self.calls = []

    def classify_window(self, window):
        self.calls.append(window.index)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(4), "test did not release safety"
        positive = window.index == 0
        objects = (
            (ModelObject("cigarette", NormalizedBBox(0.3, 0.3, 0.35, 0.35)),)
            if positive
            else ()
        )
        return GroundingResponse(
            window,
            MultiGroundedClassification(
                (
                    GroundedEventClassification(
                        "smoking", positive, 0.9, "Safety evidence", objects
                    ),
                )
            ),
            0.01,
        )


class Face:
    def __init__(self, settings, gate=None, confirm=False):
        self.settings = settings
        self.gate = gate
        self.confirm = confirm
        self.entered = threading.Event()
        self.calls = []
        self.tracker = FaceIdentityTracker()

    def classify_window(self, window):
        self.calls.append(window.index)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(4), "test did not release face"
        objects = (
            ModelObject(
                "unknown_person",
                NormalizedBBox(0.2, 0.1, 0.4, 0.4),
                attributes={"status": "unknown"},
            ),
        )
        if self.confirm:
            objects = self.tracker.resolve(
                objects,
                timestamp=window.end_timestamp,
                frame_sequence=window.newest_frame.source_frame_index,
                people={},
            )
        positive = objects[0].attributes["status"] == "unknown"
        return GroundingResponse(
            window,
            GroundedClassification(
                positive,
                0.9,
                "Face evidence",
                objects,
                event_object_indexes=(0,) if positive else (),
                face_observations=objects,
            ),
            0.01,
        )


@contextmanager
def _session(*, safety_gate=None, face_gate=None, confirm=False):
    config = load_config()
    config = replace(config, runtime=replace(config.runtime, max_concurrent_requests=1))
    safety = Safety(config.inference, safety_gate)
    face = Face(config.inference, face_gate, confirm)
    client = SelectedFeaturesGroundingClient(
        safety,
        face,
        safety_event_labels=("smoking",),
    )
    live, exact, timeline, states = [], [], [], []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_analysis_result=live.append,
            on_analysis_frame=lambda result, jpeg: exact.append((result, jpeg)),
            on_timeline_result=timeline.append,
            on_state=states.append,
        ),
    )
    with (
        ThreadPoolExecutor(max_workers=1) as safety_pool,
        ThreadPoolExecutor(max_workers=1) as face_pool,
    ):
        session._face_executor = face_pool
        try:
            yield session, safety_pool, safety, face, live, exact, timeline, states
        finally:
            if safety_gate is not None:
                safety_gate.set()
            if face_gate is not None:
                face_gate.set()
            _wait(lambda: not session._has_pending())


def _feature(rows, label):
    return [(r, j) for r in rows for j in r.judgments if j.event_label == label]


def test_graceful_capture_stop_preserves_inflight_face_result_before_end():
    config = load_config()
    config = replace(config, sliding_window=replace(
        config.sliding_window, size_frames=1, stride_frames=1,
    ))
    gate = threading.Event()
    face = Face(config.inference, gate)
    client = SelectedFeaturesGroundingClient(
        Safety(config.inference), face, safety_event_labels=("smoking",),
    )
    delivered, ended = [], threading.Event()
    session = StreamingSession.for_camera(
        config, client,
        SessionCallbacks(
            on_timeline_result=delivered.append,
            on_end=lambda reason, error: ended.set(),
        ),
    )
    frame = _window(0).newest_frame
    session.start()
    try:
        assert session.push_camera_frame(
            frame.jpeg_bytes, width=64, height=48, timestamp_seconds=0,
        )
        assert face.entered.wait(2)
        session.stop(drain=True)
        assert not ended.wait(0.05)
        assert session.running
        assert not _feature(delivered, "stranger")
        gate.set()
        assert ended.wait(2)
        session.join(2)
        assert not session.running
        assert len(_feature(delivered, "stranger")) == 1
    finally:
        gate.set()
        session.stop(drain=True)
        session.join(3)


def test_face_public_result_does_not_wait_for_remote_safety():
    gate = threading.Event()
    with _session(safety_gate=gate) as (
        session,
        pool,
        safety,
        _,
        live,
        exact,
        _,
        _states,
    ):
        window = _window(0)
        session._schedule_window(window, pool)
        assert safety.entered.wait(4)
        _wait(lambda: len(_feature(live, "stranger")) == 1)
        assert not _feature(live, "smoking")
        assert live[0].missing_event_labels == ("smoking",)
        assert live[0].frame.source_frame_index == 0
        assert exact[0][1] == window.newest_frame.jpeg_bytes
        gate.set()
        _wait(lambda: not session._has_pending())
        assert _feature(live, "smoking")[0][1].detected is True


def test_smoking_negative_is_immediate_while_face_is_busy_then_face_is_not_lost():
    gate = threading.Event()
    with _session(face_gate=gate) as (session, pool, _, face, live, exact, _, states):
        session._schedule_window(_window(0), pool)
        assert face.entered.wait(4)
        _wait(lambda: not session._pending)
        session._schedule_window(_window(1), pool)
        _wait(lambda: not session._pending)
        assert [j.detected for _, j in _feature(live, "smoking")] == [True, False]
        assert not _feature(live, "stranger")
        assert face.calls == [0]  # CPU remains single-flight; no stale queue.
        gate.set()
        _wait(lambda: not session._has_pending())
        stranger_result, stranger = _feature(live, "stranger")[0]
        assert stranger.detected is True
        assert stranger_result.frame.timestamp_seconds == 0
        assert stranger_result.missing_event_labels == ("smoking",)
        assert all(j.event_label == "stranger" for j in stranger_result.judgments)
        assert exact[-1][0].message_id == stranger_result.message_id
        assert exact[-1][1] == _window(0).newest_frame.jpeg_bytes
        assert states[-1]["branches"]["stranger"]["counts"]["dropped"] == 1


def test_busy_llm_does_not_delay_the_first_stranger_frame():
    gate = threading.Event()
    with _session(safety_gate=gate, confirm=True) as (
        session,
        pool,
        safety,
        face,
        live,
        _,
        _,
        states,
    ):
        for index in range(3):
            session._schedule_window(_window(index), pool)
            _wait(lambda: not session._face_pending)
            assert len(_feature(live, "stranger")) == index + 1
            assert _feature(live, "stranger")[-1][1].detected
        assert safety.entered.wait(4)
        assert safety.calls == [0]
        assert face.calls == [0, 1, 2]
        assert [j.detected for _, j in _feature(live, "stranger")] == [
            True,
            True,
            True,
        ]
        assert states[-1]["branches"]["safety"]["counts"]["dropped"] == 2
        assert states[-1]["branches"]["stranger"]["counts"]["submitted"] == 3
        assert not _feature(live, "smoking")


def _partial_future(index, label, positive):
    window = _window(index)
    objects = (
        (
            ModelObject(
                "unknown_person" if label == "stranger" else "cigarette",
                NormalizedBBox(0.2, 0.2, 0.4, 0.4),
            ),
        )
        if positive
        else ()
    )
    future = Future()
    future.set_result(
        GroundingResponse(
            window,
            MultiGroundedClassification(
                (
                    GroundedEventClassification(
                        label, positive, 0.9, "Observed", objects
                    ),
                ),
                missing_event_labels=(
                    "smoking" if label == "stranger" else "stranger",
                ),
                analysis_kind="selected_features",
            ),
            0.01,
            completed_branches=("stranger" if label == "stranger" else "safety",),
        )
    )
    return future, window


@pytest.mark.parametrize("recovered", (False, True))
def test_classification_diagnostics_stay_in_branch_status_on_original_source_frame(
    recovered,
):
    with _session() as (session, _, _, _, live, exact, timeline, states):
        original, window = _partial_future(2, "smoking", True)
        diagnostics = {
            "category": (
                "single_event_envelope" if recovered else "invalid_envelope"
            ),
            "finish_reason": "stop",
            "recovered": recovered,
            "response_chars": 256,
            "completion_tokens": 80,
        }
        response = replace(
            original.result(),
            classification=original.result().classification if recovered else None,
            classification_diagnostics=diagnostics,
            error=None if recovered else PUBLIC_ANALYSIS_ERROR,
            completed_branches=("safety",) if recovered else (),
            branch_errors=() if recovered else (("safety", "invalid_response"),),
        )
        future = Future()
        future.set_result(response)
        session._apply_window_future(future, window, branch="safety")

        completion = states[-1]["branches"]["safety"]["last_completion"]
        assert completion["classification"] == diagnostics
        assert completion["classification"] is not diagnostics
        assert completion["source_frame_index"] == 20
        assert completion["source_timestamp_seconds"] == window.end_timestamp
        assert completion["error"] == response.error
        assert "raw_text" not in json.dumps(completion)

        if recovered:
            assert len(live) == len(exact) == len(timeline) == 1
            assert live[0].judgments[0].detected is True
            assert live[0].frame.source_frame_index == 20
            assert exact[0][1] == window.newest_frame.jpeg_bytes
            payload = json.dumps(live[0].as_dict())
            assert "classification" not in payload
            assert diagnostics["category"] not in payload
            assert "response_chars" not in payload
            assert "completion_tokens" not in payload
            assert "raw_text" not in payload
        else:
            assert live == exact == timeline == []


@pytest.mark.parametrize(
    "order", list(itertools.permutations(("face0", "smoke0", "smoke1")))
)
def test_every_interleaving_preserves_latest_of_each_branch_with_original_frames(order):
    with _session() as (session, _, _, _, live, exact, timeline, _states):
        responses = {
            "face0": _partial_future(0, "stranger", True),
            "smoke0": _partial_future(0, "smoking", True),
            "smoke1": _partial_future(1, "smoking", False),
        }
        for name in order:
            future, window = responses[name]
            session._apply_window_future(
                future, window, branch="stranger" if name == "face0" else "safety"
            )
        assert _feature(live, "stranger")[0][1].detected is True
        assert _feature(live, "stranger")[0][0].frame.source_frame_index == 0
        smoking = max(
            _feature(live, "smoking"), key=lambda row: row[0].frame.timestamp_seconds
        )
        assert smoking[1].detected is False
        assert smoking[0].frame.source_frame_index == 10
        assert len(_feature(live, "smoking")) == 2
        assert len(timeline) == 3
        assert [r.sequence for r in live] == list(range(1, len(live) + 1))
        for result, jpeg in exact:
            assert (
                jpeg
                == _window(
                    result.frame.source_frame_index // 10
                ).newest_frame.jpeg_bytes
            )


@pytest.mark.parametrize(
    "label,branch", [("smoking", "safety"), ("stranger", "stranger")]
)
def test_old_positive_notifies_without_replacing_newer_negative_in_same_branch(
    label, branch,
):
    with _session() as (session, _, _, _, live, exact, timeline, _states):
        for index, positive in [(1, False), (0, True)]:
            future, window = _partial_future(index, label, positive)
            session._apply_window_future(future, window, branch=branch)
        assert len(live) == 2
        assert live[0].judgments[0].detected is False
        assert live[1].judgments[0].detected is True
        assert [result.sequence for result in live] == [1, 2]
        assert [result.frame.source_frame_index for result, _ in exact] == [10]
        assert len(timeline) == 2
        assert timeline[-1].message_id == live[-1].message_id
        bus = ResultBus()
        for result in live:
            bus.publish(result.as_dict())
        assert bus.latest(feature=label).payload == live[0].as_dict()
        organizer = (
            session._face_organizer if branch == "stranger" else session.organizer
        )
        assert organizer.snapshot()["decision"]["detected"] is False


@pytest.mark.parametrize(
    "label,branch", [("smoking", "safety"), ("stranger", "stranger")]
)
def test_source_history_keeps_late_negatives_for_ten_frame_clear_confirmation(
    label, branch,
):
    with _session() as (session, _, _, _, live, exact, timeline, _states):
        # Frame 10 finishes first. Frames 1 through 9 must still reach consumers
        # so their ten-negative confirmation has exactly the private evidence.
        for index in (0, 10, *range(1, 9)):
            future, window = _partial_future(index, label, index == 0)
            session._apply_window_future(future, window, branch=branch)
        assert sum(not result.judgments[0].detected for result in live) == 9
        future, window = _partial_future(9, label, False)
        session._apply_window_future(future, window, branch=branch)
        assert sum(not result.judgments[0].detected for result in live) == 10
        assert live == timeline
        assert [result.sequence for result in live] == list(range(1, 12))
        assert [result.frame.source_frame_index for result, _ in exact] == [0, 100]
        bus = ResultBus()
        for result in live:
            bus.publish(result.as_dict())
        latest = bus.latest(feature=label).payload
        assert latest["frame"]["source_frame_index"] == 100
        assert latest["judgments"][0]["detected"] is False
        source_order = sorted(
            bus.records_after(0),
            key=lambda record: record.payload["frame"]["source_frame_index"],
        )
        assert [row.payload["judgments"][0]["detected"] for row in source_order] == [
            True, *([False] * 10),
        ]


def test_stopped_camera_does_not_publish_late_results_from_either_branch():
    safety_gate, face_gate = threading.Event(), threading.Event()
    config = load_config()
    safety, face = (
        Safety(config.inference, safety_gate),
        Face(config.inference, face_gate),
    )
    client = SelectedFeaturesGroundingClient(
        safety, face, safety_event_labels=("smoking",)
    )
    live = []
    session = StreamingSession.for_camera(
        config, client, SessionCallbacks(on_analysis_result=live.append)
    )
    try:
        session.start()
        frame = _window(0).newest_frame
        assert session.push_camera_frame(
            frame.jpeg_bytes,
            width=64,
            height=48,
            timestamp_seconds=0,
            source_frame_index=0,
        )
        assert safety.entered.wait(4) and face.entered.wait(4)
        session.stop()
        session.join(2)
        assert not session.running
        safety_gate.set()
        face_gate.set()
        _wait(lambda: not session._has_pending())
        assert live == []
    finally:
        safety_gate.set()
        face_gate.set()
        session.stop()
        session.join(2)


@pytest.mark.parametrize("branch", ["safety", "stranger"])
def test_unexpected_branch_exception_recovers_without_clearing_other_branch(branch):
    with _session() as (
        session,
        _pool,
        _safety,
        _face,
        live,
        _exact,
        _timeline,
        states,
    ):
        failed = Future()
        failed.set_exception(RuntimeError("test engine error"))
        session._apply_window_future(failed, _window(0), branch=branch)
        assert states[-1]["branches"][branch]["last_error"] is not None
        other = "stranger" if branch == "safety" else "safety"
        other_label = "stranger" if other == "stranger" else "smoking"
        response, window = _partial_future(0, other_label, True)
        session._apply_window_future(response, window, branch=other)
        assert _feature(live, other_label)[-1][1].detected is True
        label = "stranger" if branch == "stranger" else "smoking"
        response, window = _partial_future(1, label, False)
        session._apply_window_future(response, window, branch=branch)
        assert states[-1]["branches"][branch]["last_error"] is None
        assert _feature(live, label)[-1][1].detected is False
        assert _feature(live, other_label)[-1][1].detected is True


def test_person_reassociation_preserves_the_original_archive_branch_window(monkeypatch):
    class Archive:
        def __init__(self):
            self.results = []
            self.windows = []

        def submit_result(self, result, *, window_id=None):
            self.results.append((result.message_id, window_id))
            return True

        def note_window(self, _session, window_id, _timestamp, **kwargs):
            self.windows.append(
                (window_id, kwargs["expected_labels"], kwargs["outcome"])
            )

    with _session() as (
        session,
        _pool,
        _safety,
        _face,
        live,
        _exact,
        _timeline,
        _states,
    ):
        archive = Archive()
        session.archive = archive
        future, window = _partial_future(0, "stranger", True)
        session._archive_window(window, "admitted", branch="stranger")
        session._apply_window_future(future, window, branch="stranger")
        original = live[0]
        monkeypatch.setattr(
            session.person_registry, "drain_reassociated_results", lambda: [original]
        )
        future, window = _partial_future(0, "smoking", True)
        session._archive_window(window, "admitted", branch="safety")
        session._apply_window_future(future, window, branch="safety")
        assert (
            archive.results[0]
            == archive.results[-1]
            == (original.message_id, "stranger:0")
        )
        assert archive.results[1][1] == "safety:0"
        assert archive.windows == [
            ("stranger:0", ("stranger",), "admitted"),
            ("stranger:0", ("stranger",), "completed"),
            ("safety:0", ("smoking",), "admitted"),
            ("safety:0", ("smoking",), "completed"),
        ]


def test_person_reassociation_updates_private_display_without_archive(monkeypatch):
    with _session() as (
        session, _pool, _safety, _face, live, exact, timeline, _states,
    ):
        future, window = _partial_future(0, "smoking", True)
        session._apply_window_future(future, window, branch="safety")
        original = live[0]
        corrected = replace(original, objects=tuple(
            replace(item, attributes={"subject_ids": ["subject-alice"]})
            for item in original.objects
        ))
        monkeypatch.setattr(
            session.person_registry, "drain_reassociated_results", lambda: [corrected],
        )
        future, window = _partial_future(1, "stranger", False)
        session._apply_window_future(future, window, branch="stranger")
        assert timeline[-1] == corrected
        assert len(timeline) == 3
        assert len(live) == len(exact) == 2
        assert live[0] == original
        assert not live[0].objects[0].attributes.get("subject_ids")


@pytest.mark.parametrize(
    "callback",
    [
        "on_event",
        "on_state",
        "on_timeline_result",
        "on_analysis_frame",
    ],
)
def test_stop_during_delivery_does_not_publish_the_remaining_callbacks(callback):
    entered, release = threading.Event(), threading.Event()
    with _session() as (
        session,
        _pool,
        _safety,
        _face,
        live,
        _exact,
        _timeline,
        _states,
    ):

        def block(*_args):
            entered.set()
            assert release.wait(4)

        setattr(session.callbacks, callback, block)
        future, window = _partial_future(0, "stranger", True)
        worker = threading.Thread(
            target=lambda: session._apply_window_future(
                future, window, branch="stranger"
            ),
            daemon=True,
        )
        try:
            worker.start()
            assert entered.wait(4)
            session.stop()
            release.set()
            worker.join(4)
            assert not worker.is_alive()
            assert live == []
        finally:
            release.set()
            worker.join(4)
