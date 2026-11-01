from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import replace
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest
from PIL import Image

from goodvision_harness import scene_worker
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


def test_delegated_worker_skips_pixel_decode_but_checks_job_metadata(monkeypatch):
    def forbidden(_jpeg):
        raise AssertionError("pixel validation belongs to the companion")

    monkeypatch.setattr(scene_worker, "_read_image", forbidden)
    valid = _job("valid")
    invalid = replace(_job("invalid"), stream_session_id="wrong-session")
    runtime, inspector = Runtime((valid, invalid)), Inspector()
    worker = SceneInspectionWorker(
        runtime, inspector, validate_source_images=False,
    )
    worker.start()
    try:
        _await(lambda: len(runtime.completions) == 2)
        assert inspector.calls == [valid]
        assert runtime.completions[0][2] is True
        assert runtime.completions[1][2] is False
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


def _timing_records(caplog):
    return [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("vlm_inspection_timing ")
    ]


def test_temporal_timing_separates_stages_and_counts_one_context_observation(
    monkeypatch, caplog,
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    now = [100.0]
    monkeypatch.setattr(scene_worker, "time", SimpleNamespace(monotonic=lambda: now[0]))
    job = _job()
    job = replace(job, request=replace(job.request, event_labels=("smoking", "fall")))
    runtime = Runtime((job,))
    inspector = Inspector(verdicts=(
        TemporalVerdict("smoking", ("person-1",), True, 0.9),
        TemporalVerdict("fall", ("person-1",), False, 0.9),
    ))

    def timed(function, seconds):
        def call(*args, **kwargs):
            now[0] += seconds
            return function(*args, **kwargs)
        return call

    monkeypatch.setattr(
        scene_worker, "_validate_job", timed(scene_worker._validate_job, 0.1),
    )
    monkeypatch.setattr(inspector, "inspect", timed(inspector.inspect, 0.2))
    monkeypatch.setattr(
        scene_worker, "_validate_verdicts", timed(scene_worker._validate_verdicts, 0.3),
    )
    monkeypatch.setattr(
        runtime, "complete_inspection", timed(runtime.complete_inspection, 0.4),
    )
    worker = SceneInspectionWorker(
        runtime, inspector, on_update=timed(lambda: None, 0.5),
    )
    worker._inspect_one(runtime.take_inspections()[0])
    record, = _timing_records(caplog)
    assert record["context_frame_sequences"] == [0, 1]
    assert record["context_frames"] == 2
    assert record["observation_end_frame_sequence"] == 1
    assert record["observation_end_timestamp_seconds"] == 1
    assert record["completed_observations"] == record["model_attempts"] == 1
    assert record["succeeded_requests"] == 1
    assert record["accepted_verdicts"] == 2
    assert record["submitted_verdicts"] == 2
    assert record["failed_requests"] == record["discarded_requests"] == 0
    assert [record[name] for name in (
        "prepare_seconds", "model_seconds", "verdict_validation_seconds",
        "complete_seconds", "callback_seconds", "total_seconds",
    )] == [0.1, 0.2, 0.3, 0.4, 0.5, 1.5]
    assert "person-1" not in str(record)
    worker.close()


@pytest.mark.parametrize("discard", [False, True])
def test_failed_or_discarded_temporal_call_keeps_model_timing_without_private_data(
    monkeypatch, caplog, discard,
):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    now = [10.0]
    monkeypatch.setattr(scene_worker, "time", SimpleNamespace(monotonic=lambda: now[0]))
    job = _job("private@request")
    job = replace(
        job, stream_session_id="rtsp://user:password@camera",
        frames=tuple(replace(frame, stream_session_id="rtsp://user:password@camera")
                     for frame in job.frames),
    )
    runtime = Runtime((job,))
    inspector = Inspector()

    def inspect(_job):
        now[0] += 0.75
        if discard:
            runtime.close()
            return inspector.verdicts
        raise RuntimeError("SECRET model response user:password")

    monkeypatch.setattr(inspector, "inspect", inspect)
    worker = SceneInspectionWorker(runtime, inspector)
    worker._inspect_one(runtime.take_inspections()[0])
    record, = _timing_records(caplog)
    assert record["model_seconds"] == 0.75
    assert record["model_attempts"] == 1
    assert record["completed_observations"] == 0
    assert record["discarded_requests"] == int(discard)
    assert record["failed_requests"] == int(not discard)
    assert record["stream_session_id"] is record["request_id"] is None
    if not discard:
        assert record["error_type"] == "RuntimeError"
    assert all(value not in str(record) for value in ("SECRET", "password", "person-1"))
    worker.close()


def test_parallel_clients_infer_together_but_commit_in_source_order(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    runtime = Runtime(tuple(_job(f"job-{index}") for index in range(4)))
    clients = [Inspector(block=True) for _ in range(4)]
    remaining = iter(clients[1:])
    worker = SceneInspectionWorker(
        runtime, clients[0], max_in_flight=4,
        inspector_factory=lambda: next(remaining),
    )
    worker.start()
    try:
        assert all(client.started.wait(2) for client in clients)
        assert worker.stats()["active_requests"] == 4
        for client in clients[1:]:
            client.release.set()
        _await(lambda: worker.stats()["active_requests"] == 1)
        assert runtime.completions == []
        clients[0].release.set()
        _await(lambda: worker.stats()["succeeded_requests"] == 4)
        assert [entry[0] for entry in runtime.completions] == [
            f"job-{index}" for index in range(4)
        ]
        _await(lambda: len(_timing_records(caplog)) == 4)
        assert all(record["completed_observations"] == 1
                   for record in _timing_records(caplog))
        assert sum(
            record["accepted_verdicts"] for record in _timing_records(caplog)
        ) == 4
    finally:
        for client in clients:
            client.release.set()
        worker.close()
    assert all(client.closed.is_set() for client in clients)


def test_expired_parallel_head_does_not_block_later_success_or_apply_late_reply(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    runtime = Runtime((_job("slow"), _job("ready")))
    slow, ready = Inspector(block=True), Inspector()
    worker = SceneInspectionWorker(
        runtime, slow, max_in_flight=2, inspector_factory=lambda: ready,
    )
    worker.start()
    try:
        assert slow.started.wait(2) and ready.started.wait(2)
        with runtime.lock:
            runtime.claims.remove("slow")
        worker.wake()
        _await(lambda: worker.stats()["succeeded_requests"] == 1)
        assert [entry[0] for entry in runtime.completions] == ["ready"]
        assert worker.stats()["discarded_requests"] == 1
        slow.release.set()
        _await(lambda: worker.stats()["active_requests"] == 0)
        _await(lambda: any(record.getMessage().startswith("vlm_late_result_timing ")
                           for record in caplog.records))
        assert worker.stats()["discarded_requests"] == 1
        assert len(runtime.completions) == 1
        expired, = [record for record in _timing_records(caplog)
                    if record["request_id"] == "slow"]
        assert expired["verdict_response_state"] == "unavailable"
        assert expired["event_label_coverage"] == {
            "requested": 1, "reported": None, "omitted": None,
        }
        late, = [json.loads(record.getMessage().split(" ", 1)[1])
                 for record in caplog.records
                 if record.getMessage().startswith("vlm_late_result_timing ")]
        assert late["request_id"] == "slow"
        assert late["completed_observations"] == 0
        assert late["observation_end_frame_sequence"] == 1
        assert late["verdict_response_state"] == "validated"
        assert late["event_verdict_counts"]["smoking"]["true"] == 1
    finally:
        slow.release.set()
        worker.close()


def test_parallel_failure_commits_unknown_then_releases_later_result():
    runtime = Runtime((_job("failure"), _job("success")))
    failed = Inspector(error=RuntimeError("private"), block=True)
    ready = Inspector()
    worker = SceneInspectionWorker(
        runtime, failed, max_in_flight=2, inspector_factory=lambda: ready,
    )
    worker.start()
    try:
        assert failed.started.wait(2) and ready.started.wait(2)
        failed.release.set()
        _await(lambda: worker.stats()["succeeded_requests"] == 1)
        assert runtime.completions[0] == ("failure", (), False)
        assert runtime.completions[1][0] == "success"
        assert worker.stats()["failed_requests"] == 1
    finally:
        failed.release.set()
        worker.close()


def test_parallel_stop_is_bounded_and_discards_all_session_results():
    runtime = Runtime((_job("a"), _job("b")))
    clients = [Inspector(block=True), Inspector(block=True)]
    worker = SceneInspectionWorker(
        runtime, clients[0], max_in_flight=2,
        inspector_factory=lambda: clients[1],
    )
    worker.start()
    assert all(client.started.wait(2) for client in clients)
    started = time.monotonic()
    worker.close(timeout_seconds=0.01)
    assert time.monotonic() - started < 0.2
    assert worker.stats()["state"] == "stopping"
    runtime.close()
    for client in clients:
        client.release.set()
    worker.close()
    assert worker.stats()["discarded_requests"] == 2
    assert all(client.closed.is_set() for client in clients)
    assert runtime.completions == []


@pytest.mark.parametrize("maximum", [0, 5, True, 1.5])
def test_parallel_capacity_is_explicit_and_bounded(maximum):
    with pytest.raises(ValueError, match="max_in_flight"):
        SceneInspectionWorker(Runtime(()), Inspector(), max_in_flight=maximum)


def test_verdict_timing_counts_labels_and_groups_without_logging_people(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    job = _job()
    people = ("person-1", "PRIVATE_PERSON_2", "PRIVATE_PERSON_3")
    labels = ("fight", "smoking", "fall", "weapon")
    job = replace(
        job,
        request=replace(job.request, subject_ids=people, event_labels=labels),
        subject_boxes=tuple(
            (sequence, subject, job.subject_boxes[0][2])
            for sequence in (0, 1) for subject in people
        ),
    )
    verdicts = (
        TemporalVerdict("smoking", (people[0],), True, .8),
        TemporalVerdict("smoking", (people[1],), False, .9),
        TemporalVerdict("smoking", (people[2],), None),
        TemporalVerdict("fall", (people[0],), False, .9),
        TemporalVerdict("fight", people[:2], None),
    )
    runtime = Runtime((job,))
    worker = SceneInspectionWorker(runtime, Inspector(verdicts=verdicts))
    worker._inspect_one(runtime.take_inspections()[0])
    record, = _timing_records(caplog)
    assert record["verdict_response_state"] == "validated"
    assert record["event_label_coverage"] == {
        "requested": 4, "reported": 3, "omitted": 1,
    }
    assert record["event_verdict_counts"] == {
        "smoking": {"true": 1, "false": 1, "unknown": 1, "omitted": 0},
        "fall": {"true": 0, "false": 1, "unknown": 0, "omitted": 0},
        "fight": {"true": 0, "false": 0, "unknown": 1, "omitted": 0},
        "weapon": {"true": 0, "false": 0, "unknown": 0, "omitted": 1},
    }
    assert record["submitted_verdicts"] == 5
    assert runtime.completions == [("request", verdicts, True)]
    assert all(person not in str(record) for person in people)
    assert "jpeg" not in str(record).lower()
    worker.close()


@pytest.mark.parametrize("failure", (False, True))
def test_empty_and_failed_responses_have_distinct_verdict_coverage(caplog, failure):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    job = _job()
    runtime = Runtime((job,))
    inspector = Inspector(
        verdicts=(), error=RuntimeError("SECRET provider text") if failure else None,
    )
    worker = SceneInspectionWorker(runtime, inspector)
    worker._inspect_one(runtime.take_inspections()[0])
    record, = _timing_records(caplog)
    assert record["verdict_response_state"] == (
        "failed" if failure else "validated_empty"
    )
    assert record["event_label_coverage"] == {
        "requested": 1, "reported": None if failure else 0,
        "omitted": None if failure else 1,
    }
    assert record["event_verdict_counts"]["smoking"] == {
        "true": None if failure else 0, "false": None if failure else 0,
        "unknown": None if failure else 0, "omitted": None if failure else 1,
    }
    assert runtime.completions == [("request", (), not failure)]
    assert "SECRET" not in str(record)
    worker.close()


def test_invalid_partial_response_does_not_log_unvalidated_positive(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    job = _job()
    runtime = Runtime((job,))
    worker = SceneInspectionWorker(runtime, Inspector(verdicts=(
        TemporalVerdict("smoking", ("person-1",), True),
        TemporalVerdict("SECRET_EVENT", ("PRIVATE_PERSON",), True),
    )))
    worker._inspect_one(runtime.take_inspections()[0])
    record, = _timing_records(caplog)
    assert record["verdict_response_state"] == "failed"
    assert record["event_verdict_counts"]["smoking"]["true"] is None
    assert record["event_label_coverage"]["reported"] is None
    assert runtime.completions == [("request", (), False)]
    assert all(value not in str(record) for value in ("SECRET", "PRIVATE", "person-1"))
    worker.close()


def test_parallel_requires_distinct_client_instances():
    client = Inspector()
    with pytest.raises(ValueError, match="independent client"):
        SceneInspectionWorker(Runtime(()), client, max_in_flight=2)
    worker = SceneInspectionWorker(
        Runtime((_job(),)), client, max_in_flight=2, inspector_factory=lambda: client,
    )
    worker.start()
    assert client.closed.wait(2)
    assert worker.stats()["worker_failures"] == 1
    assert client.calls == []
    worker.close()


@pytest.mark.parametrize(
    "code,status", [("http_error", 429), ("invalid_response", None)],
)
def test_model_failure_logs_safe_code_and_status_without_response(caplog, code, status):
    from goodvision_harness.model_api import ModelAPIError

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    job = _job("bounded-request")
    runtime = Runtime((job,))
    inspector = Inspector()

    def fail(_job):
        raise ModelAPIError(
            "SECRET provider body api-key", code=code, status_code=status,
        )

    inspector.inspect = fail
    worker = SceneInspectionWorker(runtime, inspector)
    worker._inspect_one(runtime.take_inspections()[0])
    record, = _timing_records(caplog)
    assert record["error_type"] == "ModelAPIError"
    assert record["error_code"] == code
    assert record["http_status"] == status
    assert record["completed_observations"] == 0
    assert record["failed_requests"] == 1
    assert not any(
        text in str(record) for text in ("SECRET", "api-key", "provider body")
    )
    worker.close()
