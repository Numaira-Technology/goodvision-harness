"""Detector audit I/O must not run on or hold the paced source caller."""

import io
import json
import threading
import time
from pathlib import Path

import pytest

from tools.benchmark_runtime import person_detector
from tools.benchmark_runtime.person_detector import DetectorBroker


class PausedJournal:
    def __init__(self, *, failure=None):
        self.entered, self.release = threading.Event(), threading.Event()
        self.rows, self.writers = [], []
        self.closed = False
        self.failure = failure
        self.flushes = 0

    def write(self, value):
        assert not self.closed
        self.writers.append(threading.get_ident())
        if self.failure == "write":
            raise OSError("synthetic audit write failure")
        self.rows.append(json.loads(value))
        return len(value) - 1 if self.failure == "short-write" else len(value)

    def flush(self):
        assert not self.closed
        self.flushes += 1
        self.entered.set()
        assert self.release.wait(10), "test did not release audit writer"
        if self.failure == "flush":
            raise OSError("synthetic audit flush failure")

    def close(self):
        self.closed = True
        if self.failure == "close":
            raise OSError("synthetic audit close failure")


def background(action, *, name=None):
    entered, done = threading.Event(), threading.Event()
    results, errors = [], []

    def run():
        entered.set()
        try:
            results.append(action())
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=run, name=name)
    thread.start()
    assert entered.wait(1)
    return thread, done, results, errors


@pytest.mark.parametrize("as_iterator", [False, True])
def test_blocked_audit_allows_source_but_not_premature_lookup_or_drain(as_iterator):
    journal = PausedJournal()
    threads = []
    with DetectorBroker(
        "http://detector.invalid", transport=lambda row, image: []
    ) as broker:
        broker.log = journal
        broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
        assert journal.entered.wait(1)
        try:
            lookup = background(
                lambda: broker.lookup("camera", iter([0]) if as_iterator else [0])
            )
            drain = background(broker.drain)
            source = background(
                lambda: broker.submit(0, "camera", 1, 0.4, b"synthetic-image")
            )
            threads = [lookup, drain, source]
            # Causal event ordering, not a machine-throughput measurement: the
            # source must finish while the journal is still explicitly blocked.
            assert source[1].wait(1) and not source[3]
            assert not journal.release.is_set()
            assert not lookup[1].is_set() and not drain[1].is_set()
        finally:
            journal.release.set()
            for thread, *_ in threads:
                thread.join(3)
        assert all(done.is_set() and not errors for _, done, _, errors in threads)
        assert broker.drain() == {"submitted": 2, "ready": 2}
        assert [row["frame_sequence"] for row in journal.rows] == [0, 1]
        assert set(journal.writers) == {broker.thread.ident}
    assert journal.closed


def test_queue_full_source_defers_immutable_audit_without_rolling_latest_back():
    journal = PausedJournal()
    entered, release = threading.Event(), threading.Event()
    source = None

    def detector(row, image):
        entered.set()
        assert release.wait(10)
        return []

    with DetectorBroker(
        "http://detector.invalid", transport=detector, capacity=1
    ) as broker:
        broker.log = journal
        broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
        assert entered.wait(1)
        broker.submit(0, "camera", 1, 0.4, b"synthetic-image")
        try:
            source = background(
                lambda: broker.submit(0, "camera", 2, 0.8, b"synthetic-image")
            )
            assert source[1].wait(1) and not source[3]
            assert not journal.entered.is_set()
            assert broker.counts["queue_full"] == 1
            assert broker.latest[0]["frame_sequence"] == 2
            # Audit snapshots must not alias subsequently mutable cache data.
            broker.cache[("camera", 2)]["people"].append({"changed": True})
            release.set()
            assert journal.entered.wait(1)
        finally:
            journal.release.set()
            release.set()
            if source is not None:
                source[0].join(3)
        assert broker.drain() == {"submitted": 3, "queue_full": 1, "ready": 2}
        assert [(r["frame_sequence"], r["state"]) for r in journal.rows] == [
            (2, "queue_full"),
            (0, "ready"),
            (1, "ready"),
        ]
        assert journal.rows[0]["people"] == []
        assert broker.latest[0]["frame_sequence"] == 2
        assert broker.latest[0]["state"] == "queue_full"
        assert set(journal.writers) == {broker.thread.ident}
    assert journal.closed


@pytest.mark.parametrize("failure", ["write", "short-write", "flush"])
def test_audit_error_is_sticky_without_retry_or_successful_drain(failure):
    journal = PausedJournal(failure=failure)
    journal.release.set()
    broker = DetectorBroker("http://detector.invalid", transport=lambda row, image: [])
    broker.__enter__()
    broker.log = journal
    broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
    broker.thread.join(3)
    assert not broker.thread.is_alive()
    try:
        for action in (
            lambda: broker.submit(0, "camera", 1, 0.4, b"synthetic-image"),
            lambda: broker.lookup("camera", [0]),
            broker.drain,
        ):
            with pytest.raises(RuntimeError, match="detector broker failed") as error:
                action()
            assert isinstance(error.value.__cause__, OSError)
            assert error.value.__cause__ is broker._error
        assert len(journal.writers) == 1  # An uncertain/partial write is not retried.
        assert len(journal.rows) == (0 if failure == "write" else 1)
        assert broker._audit_pending == {("camera", 0)}
        assert journal.closed
    finally:
        with pytest.raises(RuntimeError, match="detector broker failed"):
            broker.__exit__(None, None, None)
    assert not broker.server_thread.is_alive()


def test_audit_capacity_counts_inflight_record_and_fails_closed():
    journal = PausedJournal()
    broker = DetectorBroker(
        "http://detector.invalid", transport=lambda row, image: [], capacity=1
    )
    broker._audit_limit = 2
    broker.__enter__()
    broker.log = journal
    try:
        broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
        assert journal.entered.wait(1)
        broker.submit(0, "camera", 1, 0.4, b"synthetic-image")
        broker.submit(0, "camera", 2, 0.8, b"synthetic-image")
        assert len(broker._audit) == 1  # The other record is already in flight.
        assert len(broker._audit_pending) == 2
        with pytest.raises(
            RuntimeError, match="audit unavailable or capacity exhausted"
        ):
            broker.submit(0, "camera", 3, 1.2, b"synthetic-image")
        assert len(broker._audit_pending) == 2
        with pytest.raises(RuntimeError, match="detector broker failed"):
            broker.drain()
    finally:
        journal.release.set()
        with pytest.raises(RuntimeError, match="detector broker failed"):
            broker.__exit__(None, None, None)
    assert not broker.thread.is_alive() and not broker.server_thread.is_alive()
    assert journal.closed
    # Both accepted audit records survive overflow; the invocation fails rather
    # than pretending it also recorded the frames that exceeded the bound.
    assert [(r["frame_sequence"], r["state"]) for r in journal.rows[:2]] == [
        (0, "ready"),
        (2, "queue_full"),
    ]
    assert not broker._audit_pending


def test_shutdown_records_queued_cancellation_before_writer_exits():
    journal = PausedJournal()
    entered, release = threading.Event(), threading.Event()

    def detector(row, image):
        entered.set()
        assert release.wait(10)
        return []

    broker = DetectorBroker("http://detector.invalid", transport=detector, capacity=2)
    broker.__enter__()
    broker.log = journal
    closer = None
    try:
        broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
        assert entered.wait(1)
        broker.submit(0, "camera", 1, 0.4, b"synthetic-image")
        broker.submit(0, "camera", 2, 0.8, b"synthetic-image")
        closer = background(lambda: broker.__exit__(None, None, None))
        with broker.condition:
            assert broker.condition.wait_for(lambda: broker.counts["cancelled"] == 2, 1)
        assert not closer[1].is_set()
        release.set()
        assert journal.entered.wait(1)
        assert not closer[1].is_set() and not journal.closed
    finally:
        release.set()
        journal.release.set()
        if closer:
            closer[0].join(5)
        else:
            broker.__exit__(None, None, None)
    assert closer[1].is_set() and not closer[3]
    assert not broker.thread.is_alive() and not broker.server_thread.is_alive()
    assert journal.closed and not broker._audit_pending
    assert [(r["frame_sequence"], r["state"]) for r in journal.rows] == [
        (1, "cancelled"),
        (2, "cancelled"),
        (0, "ready"),
    ]
    assert broker.latest[0]["frame_sequence"] == 2
    assert broker.drain() == {"submitted": 3, "cancelled": 2, "ready": 1}


def test_stalled_audit_shutdown_is_bounded_and_does_not_close_live_writer():
    journal = PausedJournal()
    broker = DetectorBroker("http://detector.invalid", transport=lambda row, image: [])
    broker.__enter__()
    broker.log = journal
    try:
        broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
        assert journal.entered.wait(1)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="detector worker did not stop"):
            broker.__exit__(None, None, None)
        assert time.monotonic() - started < 6
        assert broker.thread.is_alive() and not journal.closed
        assert not broker.server_thread.is_alive()
        assert broker._audit_pending == {("camera", 0)}
    finally:
        journal.release.set()
        broker.thread.join(3)
    assert not broker.thread.is_alive() and journal.closed
    assert len(journal.rows) == 1 and not broker._audit_pending
    with pytest.raises(RuntimeError, match="detector broker failed"):
        broker.drain()


def test_audit_close_failure_is_reported_by_shutdown():
    journal = PausedJournal(failure="close")
    journal.release.set()
    broker = DetectorBroker("http://detector.invalid", transport=lambda row, image: [])
    broker.__enter__()
    broker.log = journal
    broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
    assert broker.drain() == {"submitted": 1, "ready": 1}
    with pytest.raises(RuntimeError, match="detector broker failed") as error:
        broker.__exit__(None, None, None)
    assert str(error.value.__cause__) == "synthetic audit close failure"
    assert not broker.thread.is_alive() and not broker.server_thread.is_alive()
    assert journal.closed


def test_output_disabled_preserves_counts_and_lookup():
    with DetectorBroker(
        "http://detector.invalid", transport=lambda row, image: []
    ) as broker:
        for sequence in range(3):
            broker.submit(0, "camera", sequence, sequence * 0.4, b"synthetic-image")
        assert broker.drain() == {"submitted": 3, "ready": 3}
        assert [
            r["frame_sequence"] for r in broker.lookup("camera", iter(range(3)))
        ] == [0, 1, 2]
        assert not broker._audit_pending and not broker._audit and broker.log is None
    assert not broker.thread.is_alive() and not broker.server_thread.is_alive()


def test_drain_rechecks_failure_after_last_audit_acknowledgment():
    journal = PausedJournal(failure="close")
    waiting = threading.Event()
    broker = DetectorBroker("http://detector.invalid", transport=lambda row, image: [])

    class OrderedCondition(threading.Condition):
        def wait(self, timeout=None):
            if threading.current_thread().name != "audit-drain":
                return super().wait(timeout)
            waiting.set()
            result = super().wait(timeout)
            # A legal scheduler order: the writer fails close before the
            # notified drain waiter gets its turn to inspect balanced counts.
            while broker._error is None:
                assert super().wait(3), "close failure was not delivered"
            return result

    broker.condition = OrderedCondition()
    broker.__enter__()
    broker.log = journal
    drainer = closer = None
    try:
        broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
        assert journal.entered.wait(1)
        drainer = background(broker.drain, name="audit-drain")
        assert waiting.wait(1)
        closer = background(lambda: broker.__exit__(None, None, None))
        assert broker.stop.wait(1)
    finally:
        journal.release.set()
        for task in (drainer, closer):
            if task:
                task[0].join(5)
        if closer is None:
            with pytest.raises(RuntimeError, match="detector broker failed"):
                broker.__exit__(None, None, None)
    for task in (drainer, closer):
        assert task[1].is_set() and not task[2] and len(task[3]) == 1
        assert str(task[3][0]) == "detector broker failed"
        assert str(task[3][0].__cause__) == "synthetic audit close failure"
    assert not broker.thread.is_alive() and not broker.server_thread.is_alive()


@pytest.mark.parametrize("failure", ["partial-raise", "short-write", "no-write"])
def test_real_file_failed_write_is_not_replayed_during_close(
    tmp_path, monkeypatch, failure
):
    target = tmp_path / "detector.jsonl"
    opened, writes = [], []
    original_open = Path.open

    class FaultyFile(io.FileIO):
        def write(self, value):
            value = bytes(value)
            writes.append(value)
            if len(writes) == 1:
                if failure == "no-write":
                    return None
                count = super().write(value[: len(value) // 2])
                if failure == "partial-raise":
                    raise OSError("partial write acknowledgment failed")
                return count
            return super().write(value)

    def open_audit(path, mode="r", buffering=-1, *args, **kwargs):
        if path == target and mode in {"w", "wb"}:
            opened.append((mode, buffering))
            raw = FaultyFile(path, "wb")
            if mode == "wb" and buffering == 0:
                return raw
            # Model the prior real buffered path so this regression also proves
            # why a normal TextIOWrapper.close() can retry uncertain writes.
            return io.TextIOWrapper(io.BufferedWriter(raw), encoding="utf-8")
        return original_open(path, mode, buffering, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_audit)
    broker = DetectorBroker(
        "http://detector.invalid", tmp_path, transport=lambda row, image: []
    )
    broker.__enter__()
    broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
    broker.thread.join(3)
    try:
        with pytest.raises(RuntimeError, match="detector broker failed"):
            broker.drain()
    finally:
        with pytest.raises(RuntimeError, match="detector broker failed"):
            broker.__exit__(None, None, None)
    assert opened == [("wb", 0)]
    assert len(writes) == 1
    assert target.read_bytes() == (
        b"" if failure == "no-write" else writes[0][: len(writes[0]) // 2]
    )
    assert broker.log.closed and not broker.thread.is_alive()
    assert not broker.server_thread.is_alive()


@pytest.mark.parametrize("stage", ["event-loop", "http-client"])
def test_worker_initialization_failure_is_reported_and_closes_resources(
    tmp_path, monkeypatch, stage
):
    error = OSError("synthetic initialization failure")

    def fail(*args, **kwargs):
        raise error

    if stage == "event-loop":
        monkeypatch.setattr(person_detector.asyncio, "new_event_loop", fail)
    else:
        monkeypatch.setattr(person_detector.httpx, "AsyncClient", fail)
    broker = DetectorBroker(
        "http://detector.invalid", tmp_path, transport=lambda row, image: []
    )
    broker.__enter__()
    broker.thread.join(3)
    try:
        for action in (
            broker.drain,
            lambda: broker.submit(0, "camera", 0, 0.0, b"synthetic-image"),
        ):
            with pytest.raises(RuntimeError, match="detector broker failed") as caught:
                action()
            assert caught.value.__cause__ is error
    finally:
        with pytest.raises(RuntimeError, match="detector broker failed") as caught:
            broker.__exit__(None, None, None)
        assert caught.value.__cause__ is error
    assert broker.log.closed and not broker.thread.is_alive()
    assert not broker.server_thread.is_alive()
    assert broker.loop is None or broker.loop.is_closed()
    assert (tmp_path / "detector.jsonl").read_bytes() == b""


def test_shutdown_cannot_report_success_with_unflushed_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(DetectorBroker, "_work", lambda self: None)
    broker = DetectorBroker(
        "http://detector.invalid", tmp_path, transport=lambda row, image: []
    )
    broker.__enter__()
    broker.thread.join(3)
    broker.submit(0, "camera", 0, 0.0, b"synthetic-image")
    with pytest.raises(RuntimeError, match="detector broker failed") as error:
        broker.__exit__(None, None, None)
    assert str(error.value.__cause__) == "detector audit incomplete"
    assert broker._audit_pending == {("camera", 0)}
    assert broker.log.closed and not broker.thread.is_alive()
    assert not broker.server_thread.is_alive()
