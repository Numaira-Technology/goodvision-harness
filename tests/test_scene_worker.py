from __future__ import annotations

import io
import time
from dataclasses import replace
from threading import Event, RLock, Thread

import pytest
from PIL import Image

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_runtime import (
    SceneRuntime,
    TemporalInspection,
    TemporalVerdict,
)
from goodvision_harness.scene_worker import SceneInspectionWorker


def _job(request_id="request"):
    with Image.new("RGB", (48, 32), "red") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        jpeg = buffer.getvalue()
    box = NormalizedBBox(0.1, 0.1, 0.6, 0.9)
    return TemporalInspection(
        "session",
        "camera",
        InspectionRequest(request_id, ("person-1",), "new_person", ("smoking",), 0, 1),
        (EvidenceFrame("session", 0, 0, jpeg), EvidenceFrame("session", 1, 1, jpeg)),
        ((0, "person-1", box), (1, "person-1", box)),
    )


class Runtime:
    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.claims = set()
        self.completions = []
        self.closed = False
        self.lock = RLock()

    def take_inspections(self, limit=None):
        with self.lock:
            if self.closed:
                return ()
            jobs = self.jobs[:limit]
            self.jobs = self.jobs[len(jobs) :]
            self.claims.update(job.request.request_id for job in jobs)
            return tuple(jobs)

    def inspection_active(self, request_id):
        with self.lock:
            return not self.closed and request_id in self.claims

    def complete_inspection(
        self, request_id, verdicts, *, success=True, completion_counts=None
    ):
        with self.lock:
            if not self.inspection_active(request_id):
                return False
            self.claims.remove(request_id)
            self.completions.append((request_id, verdicts, success))
            if completion_counts is not None:
                completion_counts.update(
                    accepted_verdicts=len(verdicts), rejected_verdicts=0
                )
            return True

    def close(self):
        with self.lock:
            self.closed = True
            self.jobs.clear()
            self.claims.clear()


class Inspector:
    def __init__(self, *, verdicts=None, error=None, block=False):
        self.verdicts = (
            (TemporalVerdict("smoking", ("person-1",), True, 0.9),)
            if verdicts is None
            else verdicts
        )
        self.error = error
        self.started = Event()
        self.release = Event()
        if not block:
            self.release.set()
        self.closed = Event()
        self.calls = []

    def inspect(self, job):
        self.calls.append(job)
        self.started.set()
        assert self.release.wait(5), "test did not release fake model"
        if self.error:
            raise self.error
        return self.verdicts

    def close(self):
        self.closed.set()


def _await(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("worker did not reach expected condition")
        time.sleep(0.005)


def test_start_and_source_frames_do_not_wait_for_model_and_stop_is_bounded():
    job = _job()
    runtime = SceneRuntime("session", "camera")
    for frame in job.frames:
        runtime.scene.observe_person(
            "person-1",
            timestamp_seconds=frame.timestamp_seconds,
            frame_sequence=frame.frame_sequence,
            bbox=job.subject_boxes[0][2],
            observation_state="observed",
        )
        runtime.observe_frame(
            frame.jpeg_bytes, frame.timestamp_seconds, frame.frame_sequence
        )
    inspector = Inspector(block=True)
    updated = Event()
    worker = SceneInspectionWorker(runtime, inspector, on_update=updated.set)
    started = time.monotonic()
    worker.start()
    assert time.monotonic() - started < 0.2
    assert inspector.started.wait(2)
    reader = Thread(
        target=lambda: runtime.observe_frame(job.frames[0].jpeg_bytes, 1.5, 2)
    )
    reader.start()
    reader.join(timeout=0.5)
    assert not reader.is_alive()
    assert worker.stats()["active_requests"] == 1
    started = time.monotonic()
    worker.close(timeout_seconds=0.02)
    assert time.monotonic() - started < 0.3
    assert not inspector.closed.is_set()
    assert worker.stats()["state"] == "stopping"
    runtime.close()
    inspector.release.set()
    assert inspector.closed.wait(2)
    assert not updated.is_set()
    assert runtime.snapshot()["stats"]["completed_inspections"] == 0
    assert worker.stats()["discarded_requests"] == 1


def test_claim_one_and_publish_each_completion_without_waiting_for_all():
    runtime = Runtime((_job("first"), _job("second")))
    inspector = Inspector(block=True)
    updates = []
    worker = SceneInspectionWorker(
        runtime, inspector, on_update=lambda: updates.append(1)
    )
    worker.start()
    try:
        assert inspector.started.wait(2)
        assert runtime.claims == {"first"}
        assert len(runtime.jobs) == 1
        assert worker.stats()["max_in_flight"] == 1
        inspector.release.set()
        _await(lambda: len(updates) == 2)
        assert [entry[0] for entry in runtime.completions] == ["first", "second"]
        assert worker.stats()["accepted_verdicts"] == 2
    finally:
        inspector.release.set()
        worker.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": RuntimeError("SECRET_KEY")},
        {"verdicts": (TemporalVerdict("smoking", ("person-1",), True),) * 2},
        {"verdicts": (TemporalVerdict("smoking", ("other",), True),)},
        {"verdicts": (TemporalVerdict("smoking", ("person-1",), "false"),)},
    ],
)
def test_failure_never_submits_negative_or_partial_evidence(kwargs, capsys):
    runtime = Runtime((_job(),))
    worker = SceneInspectionWorker(runtime, Inspector(**kwargs))
    worker.start()
    try:
        _await(lambda: worker.stats()["failed_requests"] == 1)
        assert runtime.completions == [("request", (), False)]
        assert worker.stats()["accepted_verdicts"] == 0
        assert "SECRET" not in str(worker.stats())
        assert not capsys.readouterr().err
    finally:
        worker.close()


def test_failed_inspection_preserves_an_existing_event():
    job = _job()
    runtime = SceneRuntime("session", "camera")
    for frame in job.frames:
        runtime.scene.observe_person(
            "person-1",
            timestamp_seconds=frame.timestamp_seconds,
            frame_sequence=frame.frame_sequence,
            bbox=job.subject_boxes[0][2],
            observation_state="observed",
        )
        runtime.observe_frame(
            frame.jpeg_bytes, frame.timestamp_seconds, frame.frame_sequence
        )
        runtime.scene.observe_event(
            "smoking",
            ("person-1",),
            timestamp_seconds=frame.timestamp_seconds,
            frame_sequence=frame.frame_sequence,
            detected=True,
            confidence=0.9,
            evidence_id=f"prior-{frame.frame_sequence}",
        )
    before = runtime.snapshot()["scene"]["events"]
    assert before[0]["phase"] == "confirmed"
    worker = SceneInspectionWorker(
        runtime, Inspector(error=RuntimeError("transport failed"))
    )
    worker.start()
    try:
        _await(lambda: worker.stats()["failed_requests"] > 0)
        assert runtime.snapshot()["scene"]["events"] == before
    finally:
        worker.close()


def test_empty_verdict_response_is_a_successful_inspection():
    runtime = Runtime((_job(),))
    worker = SceneInspectionWorker(runtime, Inspector(verdicts=()))
    worker.start()
    try:
        _await(lambda: worker.stats()["succeeded_requests"] == 1)
        assert runtime.completions == [("request", (), True)]
        assert worker.stats()["accepted_verdicts"] == 0
    finally:
        worker.close()


def test_expired_claim_is_skipped_before_any_model_call():
    class ExpiredRuntime(Runtime):
        def take_inspections(self, limit=None):
            jobs = super().take_inspections(limit)
            self.claims.clear()
            return jobs

    runtime, inspector = ExpiredRuntime((_job(),)), Inspector()
    worker = SceneInspectionWorker(runtime, inspector)
    worker.start()
    try:
        _await(lambda: worker.stats()["discarded_requests"] == 1)
        assert not inspector.calls
    finally:
        worker.close()


def test_expired_session_discards_response_without_claiming_the_next_job():
    runtime = Runtime((_job("first"), _job("second")))
    inspector = Inspector(block=True)
    updated = Event()
    worker = SceneInspectionWorker(runtime, inspector, on_update=updated.set)
    worker.start()
    try:
        assert inspector.started.wait(2)
        runtime.close()
        inspector.release.set()
        _await(lambda: worker.stats()["discarded_requests"] == 1)
        assert len(inspector.calls) == 1
        assert not runtime.completions
        assert not updated.is_set()
    finally:
        inspector.release.set()
        worker.close()


def test_invalid_jpeg_is_failed_before_inspector_is_called():
    job = _job()
    job = replace(
        job, frames=(replace(job.frames[0], jpeg_bytes=b"bad"), job.frames[1])
    )
    runtime, inspector = Runtime((job,)), Inspector()
    worker = SceneInspectionWorker(runtime, inspector)
    worker.start()
    try:
        _await(lambda: worker.stats()["failed_requests"] == 1)
        assert not inspector.calls
    finally:
        worker.close()


def test_callback_error_does_not_stop_next_inspection_or_expose_exception():
    def update():
        raise RuntimeError("SECRET callback")

    runtime = Runtime((_job("first"), _job("second")))
    worker = SceneInspectionWorker(runtime, Inspector(), on_update=update)
    worker.start()
    try:
        _await(lambda: worker.stats()["callback_failures"] == 2)
        assert worker.stats()["succeeded_requests"] == 2
        assert worker.stats()["worker_failures"] == 0
        assert "SECRET" not in str(worker.stats())
    finally:
        worker.close()


def test_wake_resumes_idle_worker_and_start_is_idempotent():
    runtime, inspector = Runtime(()), Inspector()
    worker = SceneInspectionWorker(runtime, inspector, poll_seconds=60)
    worker.start()
    worker.start()
    try:
        time.sleep(0.02)
        runtime.jobs.append(_job())
        worker.wake()
        assert inspector.started.wait(1)
        _await(lambda: worker.stats()["succeeded_requests"] == 1)
        assert len(inspector.calls) == 1
    finally:
        worker.close()


def test_closing_unstarted_worker_releases_client_without_starting_model():
    inspector = Inspector()
    worker = SceneInspectionWorker(Runtime((_job(),)), inspector)
    assert worker.stats()["state"] == "ready"
    worker.close()
    worker.close()
    worker.start()
    worker.wake()
    assert inspector.closed.is_set()
    assert not inspector.calls
    assert worker.stats()["state"] == "stopped"


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_invalid_shutdown_budget_is_rejected(timeout):
    worker = SceneInspectionWorker(Runtime(()), Inspector())
    with pytest.raises(ValueError):
        worker.close(timeout)
    worker.close()
