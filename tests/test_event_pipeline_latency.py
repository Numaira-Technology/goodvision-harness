"""Completed inference must reach live callbacks independently of archive/GPU work.

Thread-event waits below are deadlock watchdogs, not latency requirements. The
small benchmark measures local completed-Future wiring, excluding inference,
model tokens, network calls, and frame preparation.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundingResponse,
    ModelActor,
    ModelObject,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.rich_labeling import RichLabelCoordinator, RichLabelSnapshot
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


@pytest.fixture
def jpeg():
    ok, encoded = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))
    assert ok
    return encoded.tobytes()


def _session(archive=None, *, result_callback=None):
    config = load_config()
    live, timeline, exact, legacy = [], [], [], []

    def on_result(result):
        live.append(result)
        if result_callback:
            result_callback(result)

    session = StreamingSession.for_camera(
        config,
        SimpleNamespace(settings=config.inference, event_labels=("smoking",)),
        SessionCallbacks(
            on_analysis_result=on_result,
            on_timeline_result=timeline.append,
            on_analysis_frame=lambda result, frame: exact.append((result, frame)),
            on_vision_result=legacy.append,
        ),
        archive=archive,
    )
    if archive:
        archive.start_session(session.stream_session_id, session.source_id)
    return session, live, timeline, exact, legacy


def _response(index, timestamp, positive, jpeg, *, with_actors=True):
    window = FrameWindow(
        index=index,
        frames=(
            SampledFrame(
                sample_index=index,
                source_frame_index=index * 10,
                timestamp_seconds=timestamp,
                jpeg_bytes=jpeg,
                width=160,
                height=90,
                observed_at="2026-09-09T00:00:00+00:00",
            ),
        ),
    )
    objects = (
        (
            ModelObject(
                label="cigarette",
                bbox=NormalizedBBox(0.3, 0.3, 0.34, 0.35),
                confidence=0.9,
                actor_ids=("actor-1",) if with_actors else (),
            ),
        )
        if positive
        else ()
    )
    future = Future()
    future.set_result(
        GroundingResponse(
            window=window,
            classification=GroundedClassification(
                detected=positive,
                confidence=0.9,
                description="Visible smoking" if positive else "No smoking",
                objects=objects,
                actors=(
                    (ModelActor("actor-1", NormalizedBBox(0.1, 0.1, 0.7, 0.95)),)
                    if with_actors
                    else ()
                ),
            ),
            latency_seconds=0.123,
        )
    )
    return future, window


def _apply_all(session, responses):
    for future, window in responses:
        session._archive_window(window, "admitted")
        session._apply_window_future(future, window)


def _apply_while_blocked(session, responses):
    finished = threading.Event()
    failures = []

    def apply():
        try:
            _apply_all(session, responses)
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=apply, daemon=True)
    thread.start()
    return thread, finished, failures


def _assert_live_switches(live, exact, legacy, jpeg):
    assert [item.judgments[0].detected for item in live] == [True, False, True]
    assert [item.detected for item in legacy] == [True, False, True]
    assert [item.frame.timestamp_seconds for item in live] == [1.0, 1.4, 2.0]
    assert [item.message_id for item, _ in exact] == [item.message_id for item in live]
    assert all(frame == jpeg for _, frame in exact)
    assert (
        live[0].objects[0].attributes["subject_id"]
        == (live[2].objects[0].attributes["subject_id"])
    )


def test_archive_io_lock_does_not_delay_immediate_positive_or_negative(tmp_path, jpeg):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    session, live, _, exact, legacy = _session(archive)
    thread = None
    try:
        assert archive.flush()
        responses = [
            _response(i, t, p, jpeg)
            for i, (t, p) in enumerate(((1.0, True), (1.4, False), (2.0, True)))
        ]
        # The owner uses this lock for SQLite, replay and evidence staging. A
        # different thread must complete all callbacks while it is unavailable.
        with archive._lock:
            thread, finished, failures = _apply_while_blocked(session, responses)
            assert finished.wait(2), "archive I/O serialized live completion"
            assert not failures
            _assert_live_switches(live, exact, legacy, jpeg)
    finally:
        if thread:
            thread.join(timeout=2)
        archive.close()


def test_encoding_does_not_delay_immediate_positive_or_negative(tmp_path, jpeg):
    entered, release = threading.Event(), threading.Event()

    def blocked_encoder(manifest, destination, **kwargs):
        entered.set()
        assert release.wait(5)
        destination.write_bytes(b"test-video")
        return {
            "fps": 5.0,
            "frame_count": 15,
            "start_seconds": 0.0,
            "end_seconds": 3.0,
            "partial": False,
        }

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=blocked_encoder,
    )
    session, live, _, exact, legacy = _session(archive)
    thread = None
    try:
        for index in range(21):
            assert archive.submit_frame(
                jpeg, index / 5, index, session.stream_session_id
            )
        _apply_all(session, [_response(0, 1.0, True, jpeg)])
        assert entered.wait(2), "first positive did not request evidence encoding"
        thread, finished, failures = _apply_while_blocked(
            session,
            [
                _response(1, 1.4, False, jpeg),
                _response(2, 2.0, True, jpeg),
            ],
        )
        assert finished.wait(2), "encoding serialized live completion"
        assert not failures
        _assert_live_switches(live, exact, legacy, jpeg)
        assert not release.is_set()
    finally:
        release.set()
        if thread:
            thread.join(timeout=2)
        archive.close()


def test_slow_gpu_tracking_does_not_delay_actor_assisted_live_results(tmp_path, jpeg):
    entered, release = threading.Event(), threading.Event()

    class SlowTrackingClient:
        def infer(
            self, jpeg_bytes, *, stream_session_id, frame_sequence, timestamp_seconds
        ):
            entered.set()
            assert release.wait(5)
            return RichLabelSnapshot(
                stream_session_id, frame_sequence, timestamp_seconds
            )

        def close(self):
            pass

    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    session, live, _, exact, legacy = _session(archive)
    coordinator = RichLabelCoordinator(
        SlowTrackingClient(),
        on_snapshot=session.person_registry.observe_snapshot,
    )
    thread = None
    try:
        coordinator.reset(session.stream_session_id)
        assert coordinator.submit(
            jpeg,
            stream_session_id=session.stream_session_id,
            frame_sequence=30,
            timestamp_seconds=2.1,
        )
        assert entered.wait(2)
        thread, finished, failures = _apply_while_blocked(
            session,
            [
                _response(0, 1.0, True, jpeg),
                _response(1, 1.4, False, jpeg),
                _response(2, 2.0, True, jpeg),
            ],
        )
        assert finished.wait(2), "live callback waited for GPU tracking"
        assert not failures
        _assert_live_switches(live, exact, legacy, jpeg)
        assert coordinator.history() == ()
    finally:
        release.set()
        if thread:
            thread.join(timeout=2)
        coordinator.close()
        archive.close()


def test_older_positive_reaches_archive_without_replacing_newer_live_negative(
    tmp_path, jpeg
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    session, live, timeline, exact, legacy = _session(archive)
    try:
        older = _response(0, 1.0, True, jpeg)
        newer = _response(1, 1.4, False, jpeg)
        for _, window in (older, newer):
            session._archive_window(window, "admitted")
        for future, window in (newer, older):
            session._apply_window_future(future, window)
        assert [item.judgments[0].detected for item in live] == [False]
        assert [item.detected for item in legacy] == [False]
        assert [item.frame.timestamp_seconds for item, _ in exact] == [1.4]
        assert [item.judgments[0].detected for item in timeline] == [False, True]
        assert archive.flush()
        archived = archive.list_events(message_id=timeline[-1].message_id)["events"]
        assert len(archived) == 1
        assert archived[0]["start_seconds"] == 1.0
    finally:
        archive.close()


def test_completed_future_wiring_benchmark_reports_first_and_p95(tmp_path, jpeg):
    # No timing threshold: this is an inspectable baseline, not a performance
    # promise or an estimate of total model latency on production hardware.
    measurements = {}
    for variant, enabled, with_actors in (
        ("no_archive_no_actors", False, False),
        ("no_archive_with_actors", False, True),
        ("archive_with_actors", True, True),
    ):
        archive = (
            EventArchive(EventArchiveSettings(directory=tmp_path / "archive"))
            if enabled
            else None
        )
        callback_ms, elapsed_ms = [], []
        started = [0.0]
        session, live, _, _, _ = _session(
            archive,
            result_callback=lambda result, callback_ms=callback_ms, started=started: (
                callback_ms.append((time.perf_counter() - started[0]) * 1000)
            ),
        )
        responses = [
            _response(i, 1.0 + i * 0.4, i % 3 != 1, jpeg, with_actors=with_actors)
            for i in range(120)
        ]
        try:
            if archive:
                assert archive.flush()
            for future, window in responses:
                session._archive_window(window, "admitted")
                started[0] = time.perf_counter()
                session._apply_window_future(future, window)
                elapsed_ms.append((time.perf_counter() - started[0]) * 1000)
            assert len(live) == len(responses)
            if with_actors:
                subjects = {
                    item.objects[0].attributes["subject_id"]
                    for item in live
                    if item.objects
                }
                assert len(subjects) == 1
            if archive:
                assert archive.flush()
                assert archive.status()["dropped"] == 0
            measurements[variant] = {
                "count": len(callback_ms),
                "first_callback_ms": callback_ms[0],
                "p50_callback_ms": float(np.percentile(callback_ms, 50)),
                "p95_callback_ms": float(np.percentile(callback_ms, 95)),
                "p95_apply_ms": float(np.percentile(elapsed_ms, 95)),
            }
        finally:
            if archive:
                archive.close()
    for record in measurements.values():
        assert all(math.isfinite(value) and value >= 0 for value in record.values())
    report = {
        "scope": (
            "local completed Future -> live callback / apply return, "
            "comparing absent/present precomputed actor observations and archive"
        ),
        "excludes": [
            "model inference",
            "model tokens",
            "network",
            "frame preparation",
            "video encoding",
        ],
        "conditions": (
            "120 synthetic observations per case, archive owner running, "
            "no timing pass/fail threshold"
        ),
        "limitation": (
            "Actor variants isolate local association cost. Extra model output "
            "tokens and their effect on inference latency are not measured."
        ),
        "measurements": measurements,
    }
    path = tmp_path / "pipeline-latency.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPipeline wiring benchmark: {path}\n{json.dumps(report, indent=2)}")
