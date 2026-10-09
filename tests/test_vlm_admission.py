"""Exercise shared limits with real processes, crashes and bounded waiting."""

import multiprocessing as mp
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from threading import Event, Thread, current_thread

import pytest

from goodvision_harness.vlm_admission import (
    AdmissionError,
    AdmissionSettings,
    SharedAdmission,
)


def settings(tmp_path, **kwargs):
    return AdmissionSettings(str(tmp_path / "admission.sqlite"), **kwargs)


def participant(config, stream, start, ready, results):
    gate = SharedAdmission(config)
    ready.put(stream)
    try:
        assert start.wait(30), "parent did not release the startup barrier"
        completed = expired = 0
        until = time.monotonic() + 30
        # This verifies shared limits and eventual progress, not a CI machine's
        # latency SLA. Expired evidence must stay expired; offer a fresh window
        # within a bounded overall budget when a busy runner misses its deadline.
        while completed < 3:
            assert time.monotonic() < until, (
                f"no progress after {completed} grants / {expired} expired windows; "
                f"{gate.snapshot()}"
            )
            try:
                with gate.slot(
                    stream, stop=Event(), source_deadline=time.monotonic() + 3
                ):
                    begin = time.monotonic()
                    time.sleep(0.08)
                    results.put((stream, begin, time.monotonic()))
                    completed += 1
            except AdmissionError as exc:
                if exc.code != "admission_timeout":
                    raise
                expired += 1
    except Exception as exc:
        results.put((stream, None, f"{type(exc).__name__}: {exc}"))
        raise


def test_twelve_processes_share_concurrency_rate_and_all_make_progress(tmp_path):
    config = settings(
        tmp_path,
        global_limit=3,
        per_stream_limit=1,
        requests_per_second=80,
        queue_timeout_seconds=3,
    )
    gate = SharedAdmission(config)
    ctx = mp.get_context("spawn")
    start, ready, results = ctx.Event(), ctx.Queue(), ctx.Queue()
    workers = [
        ctx.Process(target=participant, args=(config, str(i), start, ready, results))
        for i in range(12)
    ]
    try:
        for process in workers:
            process.start()
        assert len({ready.get(timeout=20) for _ in workers}) == 12
        start.set()
        rows = []
        for _ in range(36):
            row = results.get(timeout=40)
            assert row[1] is not None, f"participant {row[0]} failed: {row[2]}"
            rows.append(row)
        for process in workers:
            process.join(5)
            assert process.exitcode == 0
        edges = sorted(
            [(begin, 1) for _, begin, _ in rows] + [(end, -1) for _, _, end in rows]
        )
        active = peak = 0
        for _, delta in edges:
            active += delta
            peak = max(peak, active)
        assert 1 <= peak <= 3
        for stream in map(str, range(12)):
            own = sorted((begin, end) for s, begin, end in rows if s == stream)
            assert len(own) == 3
            assert all(a[1] <= b[0] for a, b in pairwise(own))
        assert gate.snapshot()["active"] == 0
        assert gate.snapshot()["waiting"] == 0
        assert gate.snapshot()["peak_active"] <= 3
        assert gate.snapshot()["grants"] == 36
    finally:
        start.set()
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join(5)
        for queue in (ready, results):
            queue.close()
            queue.join_thread()


def test_shared_rate_and_release_on_error(tmp_path):
    gate = SharedAdmission(settings(tmp_path, requests_per_second=10))
    starts = []
    for i in range(4):
        with pytest.raises(RuntimeError, match="model failed"):
            with gate.slot(str(i), stop=Event(), source_deadline=time.monotonic() + 3):
                starts.append(time.monotonic())
                raise RuntimeError("model failed")
    assert starts[-1] - starts[0] >= 0.29
    assert gate.snapshot()["active"] == 0


def test_blocked_waiter_does_not_repeatedly_take_the_writer_lock(tmp_path, monkeypatch):
    gate = SharedAdmission(settings(tmp_path, global_limit=1, per_stream_limit=1))
    original = gate._transaction
    repeated_write = Event()
    calls = []
    errors = []

    @contextmanager
    def transaction(*args, **kwargs):
        if current_thread() is worker:
            calls.append(1)
            if len(calls) >= 3:
                repeated_write.set()
        with original(*args, **kwargs) as db:
            yield db

    def wait():
        try:
            with gate.slot(
                "waiting", stop=Event(), source_deadline=time.monotonic() + 3
            ):
                pass
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=wait)
    monkeypatch.setattr(gate, "_transaction", transaction)
    try:
        with gate.slot("holder", stop=Event(), source_deadline=time.monotonic() + 3):
            worker.start()
            deadline = time.monotonic() + 1
            while not gate.snapshot()["waiting"]:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            # Register once, then poll without contending with grants/releases.
            assert not repeated_write.wait(0.15)
        worker.join(3)
        assert not worker.is_alive()
        assert not errors
        assert gate.snapshot()["grants"] == 2
    finally:
        if worker.ident is not None:
            worker.join(3)


def test_new_waiting_evidence_supersedes_old_without_exceeding_queue(tmp_path):
    gate = SharedAdmission(settings(tmp_path, global_limit=1, per_stream_limit=1))
    errors, ran = [], []

    def wait(name):
        try:
            with gate.slot(
                "same-source", stop=Event(), source_deadline=time.monotonic() + 3
            ):
                ran.append(name)
        except AdmissionError as exc:
            errors.append((name, exc.code))

    with gate.slot("blocker", stop=Event(), source_deadline=time.monotonic() + 3):
        old = Thread(target=wait, args=("old",))
        new = Thread(target=wait, args=("new",))
        old.start()
        deadline = time.monotonic() + 1
        while not gate.snapshot()["waiting"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        new.start()
        old.join(1)
        assert errors == [("old", "admission_superseded")]
        assert gate.snapshot()["waiting"] == 1
    new.join(2)
    assert ran == ["new"]


def test_cancel_and_timeout_do_not_start_work_or_leave_tickets(tmp_path):
    gate = SharedAdmission(
        settings(
            tmp_path, global_limit=1, per_stream_limit=1, queue_timeout_seconds=0.05
        )
    )
    with gate.slot("holder", stop=Event(), source_deadline=time.monotonic() + 3):
        with pytest.raises(AdmissionError, match="admission_timeout"):
            with gate.slot("other", stop=Event(), source_deadline=time.monotonic() + 3):
                pytest.fail("over capacity")
        stopped = Event()
        stopped.set()
        with pytest.raises(AdmissionError, match="admission_cancelled"):
            with gate.slot(
                "stopped", stop=stopped, source_deadline=time.monotonic() + 3
            ):
                pytest.fail("cancelled work ran")
        assert gate.snapshot()["waiting"] == 0


def abandon(config, ready):
    gate = SharedAdmission(config)
    with gate.slot("crashed", stop=Event(), source_deadline=time.monotonic() + 3):
        ready.set()
        time.sleep(30)


def test_crash_lease_is_conservative_and_expiry_reclaims_capacity(tmp_path):
    config = settings(tmp_path, global_limit=1, per_stream_limit=1)
    gate = SharedAdmission(config)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=abandon, args=(config, ready))
    process.start()
    try:
        assert ready.wait(10)
        process.terminate()
        process.join(5)
        assert gate.snapshot()["active"] == 1
        # Advance just the crashed lease instead of making a test sleep 60 s.
        with sqlite3.connect(config.database) as db:
            db.execute("UPDATE tickets SET expires=?", (time.monotonic() - 1,))
        with gate.slot(
            "replacement", stop=Event(), source_deadline=time.monotonic() + 3
        ):
            assert gate.snapshot()["active"] == 1
        assert gate.snapshot()["active"] == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)


def test_already_waiting_process_notices_expired_crash_lease(tmp_path, monkeypatch):
    config = settings(tmp_path, global_limit=1, per_stream_limit=1)
    gate = SharedAdmission(config)
    with sqlite3.connect(config.database) as db:
        db.execute(
            "INSERT INTO tickets(id,stream,active,created,expires) "
            "VALUES('crashed','crashed',1,?,?)",
            (time.monotonic(), time.monotonic() + 60),
        )
    read_waiting, admitted, stop = Event(), Event(), Event()
    original = gate._may_acquire
    errors = []

    def peek(token):
        possible = original(token)
        if not possible:
            read_waiting.set()
        return possible

    def wait():
        try:
            with gate.slot("waiting", stop=stop, source_deadline=time.monotonic() + 3):
                admitted.set()
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(gate, "_may_acquire", peek)
    worker = Thread(target=wait)
    worker.start()
    try:
        assert read_waiting.wait(1)
        assert not admitted.is_set()
        # No new registration or snapshot/reaper is needed to notice expiry.
        with sqlite3.connect(config.database) as db:
            db.execute(
                "UPDATE tickets SET expires=? WHERE id='crashed'",
                (time.monotonic() - 1,),
            )
        assert admitted.wait(2)
        worker.join(2)
        assert not worker.is_alive()
        assert not errors
        assert gate.snapshot()["active"] == 0
        assert gate.snapshot()["waiting"] == 0
    finally:
        stop.set()
        worker.join(3)


def test_policy_mismatch_fails_closed(tmp_path):
    config = settings(tmp_path)
    SharedAdmission(config)
    with pytest.raises(ValueError, match="policy mismatch"):
        SharedAdmission(replace(config, global_limit=24))


def wait_until(predicate, timeout=2):
    until = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < until, "condition did not become true"
        time.sleep(0.005)


def test_parallel_waiters_only_replace_their_own_lane(tmp_path):
    gate = SharedAdmission(
        settings(
            tmp_path,
            global_limit=1,
            per_stream_limit=1,
            queue_timeout_seconds=3,
        )
    )
    stopped = Event()
    ran, errors = [], []

    def wait(name, lane):
        try:
            with gate.slot(
                "camera", lane=lane, stop=stopped, source_deadline=time.monotonic() + 3
            ):
                ran.append(name)
        except AdmissionError as exc:
            errors.append((name, exc.code))

    workers = [
        Thread(target=wait, args=(name, lane))
        for name, lane in (
            ("events-old", "events"),
            ("grounding", "compact"),
            ("events-new", "events"),
        )
    ]
    try:
        with gate.slot("blocker", stop=stopped, source_deadline=time.monotonic() + 3):
            workers[0].start()
            wait_until(lambda: gate.snapshot()["waiting"] == 1)
            workers[1].start()
            wait_until(lambda: gate.snapshot()["waiting"] == 2 or errors)
            assert not errors, "peer lane replaced existing waiter"
            workers[2].start()
            wait_until(lambda: errors)
            assert errors == [("events-old", "admission_superseded")]
            assert gate.snapshot()["waiting"] == 2
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()
        assert set(ran) == {"grounding", "events-new"}
        assert gate.snapshot()["grants"] == 3
        assert gate.snapshot()["peak_active"] == 1
        assert gate.snapshot()["waiting"] == gate.snapshot()["active"] == 0
    finally:
        stopped.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(3)


def test_parallel_lanes_still_share_per_stream_limit(tmp_path):
    gate = SharedAdmission(
        settings(
            tmp_path,
            global_limit=8,
            per_stream_limit=1,
            queue_timeout_seconds=3,
        )
    )
    admitted, stopped = Event(), Event()
    errors = []

    def wait():
        try:
            with gate.slot(
                "camera",
                lane="compact",
                stop=stopped,
                source_deadline=time.monotonic() + 3,
            ):
                admitted.set()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=wait)
    try:
        with gate.slot(
            "camera", lane="events", stop=stopped, source_deadline=time.monotonic() + 3
        ):
            worker.start()
            wait_until(lambda: gate.snapshot()["waiting"] == 1)
            assert not admitted.wait(0.05)
            # Another source may still use the spare GLOBAL capacity.
            with gate.slot(
                "other-camera",
                lane="events",
                stop=stopped,
                source_deadline=time.monotonic() + 3,
            ):
                assert gate.snapshot()["active"] == 2
        assert admitted.wait(2)
        worker.join(3)
        assert not errors
    finally:
        stopped.set()
        if worker.ident is not None:
            worker.join(3)


def test_legacy_database_migration_preserves_active_lease(tmp_path):
    config = settings(tmp_path, global_limit=1, per_stream_limit=1)
    with sqlite3.connect(config.database) as db:
        db.execute(
            "CREATE TABLE tickets (id TEXT PRIMARY KEY, stream TEXT, "
            "active INTEGER, created REAL, expires REAL)"
        )
        db.execute(
            "INSERT INTO tickets VALUES('old','camera',1,?,?)",
            (time.monotonic(), time.monotonic() + 60),
        )
    gate = SharedAdmission(config)
    SharedAdmission(config)  # Migration is idempotent across processes/restarts.
    assert gate.snapshot()["active"] == 1
    with sqlite3.connect(config.database) as db:
        assert db.execute("SELECT lane FROM tickets WHERE id='old'").fetchone() == (
            "default",
        )
    with pytest.raises(AdmissionError, match="admission_timeout"):
        with gate.slot(
            "camera",
            lane="compact",
            stop=Event(),
            source_deadline=time.monotonic() + 0.03,
        ):
            pytest.fail("old capacity lease was ignored")


def test_persisted_previous_monotonic_epoch_does_not_stall_restart(tmp_path):
    config = settings(tmp_path)
    SharedAdmission(config)
    with sqlite3.connect(config.database) as db:
        db.execute(
            "INSERT INTO tickets(id,stream,active,created,expires) "
            "VALUES('old','old',1,?,?)",
            (time.monotonic() + 10000, time.monotonic() + 10100),
        )
        db.execute("UPDATE policy SET next_grant=?", (time.monotonic() + 10000,))
    gate = SharedAdmission(config)
    with gate.slot("new", stop=Event(), source_deadline=time.monotonic() + 3):
        assert gate.snapshot()["active"] == 1
