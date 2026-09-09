"""Local face and remote events must progress independently on exact source frames.

Thread waits are deadlock watchdogs, not production inference latency claims.
No real model, camera, network service or webhook is used here.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
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
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def _window(index):
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), index * 30, np.uint8))
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


def test_llm_capacity_does_not_prevent_three_reliable_face_confirmations():
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
        assert safety.entered.wait(4)
        assert safety.calls == [0]
        assert face.calls == [0, 1, 2]
        assert [j.detected for _, j in _feature(live, "stranger")] == [
            False,
            False,
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
        assert _feature(live, "smoking")[-1][1].detected is False
        assert _feature(live, "smoking")[-1][0].frame.source_frame_index == 10
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
def test_old_positive_never_replaces_newer_negative_in_the_same_branch(label, branch):
    with _session() as (session, _, _, _, live, _, timeline, _states):
        for index, positive in [(1, False), (0, True)]:
            future, window = _partial_future(index, label, positive)
            session._apply_window_future(future, window, branch=branch)
        assert len(live) == 1
        assert live[0].judgments[0].detected is False
        assert len(timeline) == 2


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
