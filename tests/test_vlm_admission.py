"""Exercise shared limits with real processes, crashes and bounded waiting."""

import multiprocessing as mp
import os
import sqlite3
import time
from contextlib import ExitStack, closing, contextmanager
from dataclasses import replace
from itertools import pairwise
from threading import BoundedSemaphore, Event, Thread, current_thread
from types import SimpleNamespace

import pytest

import goodvision_harness.vlm_admission as admission
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
    for i in range(4):
        with pytest.raises(RuntimeError, match="model failed"):
            with gate.slot(str(i), stop=Event(), source_deadline=time.monotonic() + 3):
                raise RuntimeError("model failed")
    # The rate limit spaces committed grants. Thread startup or scheduling after
    # a grant can delay entry to the body and compress the observed body interval.
    with closing(sqlite3.connect(gate.settings.database)) as db:
        grants = [
            row[0]
            for row in db.execute("SELECT last_grant FROM streams ORDER BY last_grant")
        ]
    assert len(grants) == 4
    assert all(b >= a + 0.1 for a, b in pairwise(grants))
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


class ReleaseConflict:
    """Deterministic release failure, with separately gated cleanup recovery."""

    def __init__(self, gate):
        self.original = gate._transaction
        self.armed = False
        self.failures = 0
        self.cleanup_entered = Event()
        self.allow_cleanup = Event()
        self.worker = None

    @contextmanager
    def transaction(self, *args, **kwargs):
        if current_thread().name == "vlm-admission-cleanup":
            self.worker = current_thread()
            self.cleanup_entered.set()
            if not self.allow_cleanup.wait(30):
                raise AssertionError("test did not release cleanup barrier")
        elif self.armed and self.failures < 3:
            self.failures += 1
            raise sqlite3.OperationalError("database is locked")
        with self.original(*args, **kwargs) as db:
            yield db

    def finish(self):
        self.allow_cleanup.set()
        if self.worker is not None:
            self.worker.join(3)
            assert not self.worker.is_alive(), "cleanup worker survived its test"


def admission_tickets(config):
    with closing(sqlite3.connect(config.database)) as db:
        return db.execute(
            "SELECT id,stream,active,expires FROM tickets ORDER BY stream,id"
        ).fetchall()


@pytest.mark.parametrize("body_error", [False, True])
def test_completed_release_recovers_without_another_slot(
    tmp_path, monkeypatch, body_error
):
    config = settings(tmp_path, global_limit=1, per_stream_limit=1)
    gate = SharedAdmission(config)
    conflict = ReleaseConflict(gate)
    monkeypatch.setattr(gate, "_transaction", conflict.transaction)

    def complete():
        with gate.slot("finished", stop=Event(), source_deadline=time.monotonic() + 3):
            token = admission_tickets(config)[0][0]
            conflict.armed = True
            if body_error:
                raise ValueError("original body failure")
            return token

    try:
        if body_error:
            with pytest.raises(ValueError, match="original body failure"):
                complete()
        else:
            assert complete() == admission_tickets(config)[0][0]
        assert conflict.failures == 3
        assert conflict.cleanup_entered.wait(2), "completed release was abandoned"
        assert len(admission_tickets(config)) == 1
        conflict.finish()
        assert admission_tickets(config) == []
        # Fresh evidence can use the recovered capacity under the original policy.
        with gate.slot("finished", stop=Event(), source_deadline=time.monotonic() + 3):
            assert gate.snapshot()["active"] == 1
        assert gate.snapshot()["grants"] == 2
    finally:
        conflict.finish()


def test_completed_cleanup_preserves_another_live_lane(tmp_path, monkeypatch):
    config = settings(
        tmp_path, global_limit=2, per_stream_limit=2, requests_per_second=80
    )
    gate = SharedAdmission(config)
    conflict = ReleaseConflict(gate)
    monkeypatch.setattr(gate, "_transaction", conflict.transaction)
    try:
        with gate.slot(
            "same", lane="compact", stop=Event(), source_deadline=time.monotonic() + 3
        ):
            live_token = admission_tickets(config)[0][0]
            with gate.slot(
                "same",
                lane="events",
                stop=Event(),
                source_deadline=time.monotonic() + 3,
            ):
                conflict.armed = True
            assert conflict.cleanup_entered.wait(2), "completed release was abandoned"
            assert len(admission_tickets(config)) == 2
            conflict.finish()
            rows = admission_tickets(config)
            assert len(rows) == 1
            assert rows[0][0] == live_token
            assert rows[0][2] == 1
            assert rows[0][3] > time.monotonic()
        assert admission_tickets(config) == []
    finally:
        conflict.finish()


def final_release_participant(config, body_ready, finish_body, returned):
    gate = SharedAdmission(config)
    with gate.slot("final", stop=Event(), source_deadline=time.monotonic() + 3):
        body_ready.set()
        assert finish_body.wait(3), "test did not finish synthetic body"
    returned.set()
    # No further gate call. Normal process exit must retain deferred cleanup.


def test_final_release_recovers_before_normal_process_exit(tmp_path):
    config = settings(tmp_path, global_limit=1, per_stream_limit=1)
    SharedAdmission(config)
    ctx = mp.get_context("spawn")
    body_ready, finish_body, returned = ctx.Event(), ctx.Event(), ctx.Event()
    process = ctx.Process(
        target=final_release_participant,
        args=(config, body_ready, finish_body, returned),
    )
    process.start()
    try:
        assert body_ready.wait(3)
        with closing(sqlite3.connect(config.database, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            finish_body.set()
            assert returned.wait(3), "cleanup delayed the original body result"
            process.join(0.2)
            assert process.is_alive(), "final completed release was abandoned on exit"
            assert (
                len(db.execute("SELECT id FROM tickets WHERE active=1").fetchall()) == 1
            )
            db.commit()
        process.join(3)
        assert process.exitcode == 0
        assert admission_tickets(config) == []
    finally:
        finish_body.set()
        if process.is_alive():
            process.terminate()
        process.join(3)
        assert not process.is_alive()


def cleanup_budgets(monkeypatch, tokens=2, workers=2):
    budgets = BoundedSemaphore(tokens), BoundedSemaphore(workers)
    monkeypatch.setattr(admission, "_cleanup_tokens", budgets[0])
    monkeypatch.setattr(admission, "_cleanup_workers", budgets[1])
    return budgets


def assert_permits(budget, count):
    held = 0
    try:
        for _ in range(count):
            assert budget.acquire(blocking=False), "cleanup permit leaked"
            held += 1
        assert not budget.acquire(blocking=False), "cleanup permit released twice"
    finally:
        for _ in range(held):
            budget.release()


@pytest.mark.parametrize("limited", ["tokens", "workers"])
def test_cleanup_total_budgets_cover_many_gate_instances(
    tmp_path, monkeypatch, limited
):
    counts = (2, 3) if limited == "tokens" else (3, 2)
    budgets = cleanup_budgets(monkeypatch, *counts)
    gates, conflicts = [], []
    try:
        for i in range(3):
            gate = SharedAdmission(settings(tmp_path / str(i)))
            gates.append(gate)
            conflict = ReleaseConflict(gate)
            conflicts.append(conflict)
            monkeypatch.setattr(gate, "_transaction", conflict.transaction)
            with gate.slot(str(i), stop=Event(), source_deadline=time.monotonic() + 3):
                conflict.armed = True
            if i < 2:
                assert conflict.cleanup_entered.wait(2)
        assert sum(len(g._pending_releases) for g in gates) == 2
        assert gates[2]._pending_releases == {}
        assert gates[2]._cleanup_thread is None
        assert len(admission_tickets(gates[2].settings)) == 1
        for conflict in conflicts:
            conflict.finish()
        assert admission_tickets(gates[0].settings) == []
        assert admission_tickets(gates[1].settings) == []
        # Budget exhaustion is conservative, not permission to release a lease.
        assert admission_tickets(gates[2].settings)[0][2] == 1
        for budget, count in zip(budgets, counts, strict=True):
            assert_permits(budget, count)
    finally:
        for conflict in conflicts:
            conflict.finish()


@pytest.mark.parametrize("failure", ["construct", "start", "started"])
def test_cleanup_start_failure_churn_preserves_results_and_budgets(
    tmp_path, monkeypatch, failure
):
    budgets = cleanup_budgets(monkeypatch)
    original_thread = admission.Thread
    workers = []

    class StartErrorThread(original_thread):
        def start(self):
            if failure == "started":
                super().start()
            raise RuntimeError("synthetic cleanup start error")

    def thread(*args, **kwargs):
        if kwargs.get("name") == "vlm-admission-cleanup":
            if failure == "construct":
                raise RuntimeError("synthetic cleanup construction error")
            worker = StartErrorThread(*args, **kwargs)
            workers.append(worker)
            return worker
        return original_thread(*args, **kwargs)

    def broken_warning(*args, **kwargs):
        raise RuntimeError("synthetic logging failure")

    monkeypatch.setattr(admission, "Thread", thread)
    monkeypatch.setattr(
        admission,
        "logging",
        SimpleNamespace(getLogger=lambda _: SimpleNamespace(warning=broken_warning)),
    )
    gates = []
    try:
        for i in range(8):
            gate = SharedAdmission(settings(tmp_path / str(i)))
            gates.append(gate)
            conflict = ReleaseConflict(gate)
            conflict.allow_cleanup.set()
            monkeypatch.setattr(gate, "_transaction", conflict.transaction)
            try:
                with gate.slot(
                    str(i), stop=Event(), source_deadline=time.monotonic() + 3
                ):
                    conflict.armed = True
                    if i % 2:
                        raise ValueError("body error survives cleanup")
            except ValueError as exc:
                assert str(exc) == "body error survives cleanup"
            if workers and workers[-1].ident is not None:
                workers[-1].join(3)
                assert not workers[-1].is_alive()
            assert gate._pending_releases == {}
            assert gate._cleanup_thread is None
            assert bool(admission_tickets(gate.settings)) == (failure != "started")
            for budget in budgets:
                assert_permits(budget, 2)
        assert all(g._pending_releases == {} for g in gates)
    finally:
        for worker in workers:
            if worker.ident is not None:
                worker.join(3)
                assert not worker.is_alive()


@pytest.mark.parametrize("committed", [False, True])
def test_cleanup_retries_rollback_and_uncertain_commit(
    tmp_path, monkeypatch, committed
):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(settings(tmp_path, global_limit=1, per_stream_limit=1))
    original = gate._transaction
    retry, resume = Event(), Event()
    attempts = []
    workers = []

    @contextmanager
    def transaction(*args, **kwargs):
        cleanup = current_thread().name == "vlm-admission-cleanup"
        if cleanup:
            workers[:] = [current_thread()]
            attempts.append(1)
            if len(attempts) == 2:
                retry.set()
                assert resume.wait(3)
        with original(*args, **kwargs) as db:
            yield db
            if cleanup and len(attempts) == 1 and not committed:
                raise sqlite3.OperationalError("rollback the deletion")
        if cleanup and len(attempts) == 1 and committed:
            raise sqlite3.OperationalError("commit acknowledgement uncertain")

    monkeypatch.setattr(gate, "_transaction", transaction)
    monkeypatch.setattr(gate, "_remove", lambda token: False)
    try:
        with gate.slot("done", stop=Event(), source_deadline=time.monotonic() + 3):
            token = admission_tickets(gate.settings)[0][0]
        assert retry.wait(2)
        assert token in gate._pending_releases
        assert bool(admission_tickets(gate.settings)) is not committed
        resume.set()
        workers[0].join(3)
        assert not workers[0].is_alive()
        assert len(attempts) == 2
        assert gate._pending_releases == {}
        assert admission_tickets(gate.settings) == []
        for budget in budgets:
            assert_permits(budget, 2)
    finally:
        resume.set()
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()


def test_cleanup_absolute_horizon_preserves_database_lease(tmp_path, monkeypatch):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(settings(tmp_path, global_limit=1, per_stream_limit=1))
    original = gate._transaction
    real_time = time.monotonic
    clock = {"now": real_time()}
    workers, deadlines, attempts = [], [], []
    monkeypatch.setattr(
        admission,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"], sleep=time.sleep),
    )

    @contextmanager
    def transaction(*args, **kwargs):
        if current_thread().name == "vlm-admission-cleanup":
            workers[:] = [current_thread()]
            attempts.append(1)
            deadlines.extend(gate._pending_releases.values())
            clock["now"] += 61
            raise sqlite3.OperationalError("persistent database failure")
        with original(*args, **kwargs) as db:
            yield db

    monkeypatch.setattr(gate, "_transaction", transaction)
    monkeypatch.setattr(gate, "_remove", lambda token: False)
    with gate.slot("done", stop=Event(), source_deadline=clock["now"] + 3):
        original_row = admission_tickets(gate.settings)[0]
        completed = clock["now"]
    wait_until(lambda: bool(workers))
    workers[0].join(3)
    assert not workers[0].is_alive()
    assert len(attempts) == 1
    assert deadlines == [completed + 60]
    assert gate._pending_releases == {}
    assert gate._cleanup_thread is None
    assert admission_tickets(gate.settings) == [original_row]
    assert original_row[3] > real_time()
    for budget in budgets:
        assert_permits(budget, 2)


def test_late_cleanup_finalizer_does_not_discard_replacement_queue(
    tmp_path, monkeypatch
):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(settings(tmp_path, global_limit=2, per_stream_limit=1))
    original = gate._transaction
    real_time = time.monotonic
    first_attempt, expire, handoff, finalize, second_attempt, recover = (
        Event() for _ in range(6)
    )
    first, second = [], []
    clock = {"advance": False}

    def now():
        extra = 61 if first and current_thread() is first[0] and clock["advance"] else 0
        return real_time() + extra

    def warning(message):
        if "horizon exhausted" in message:
            handoff.set()
            assert finalize.wait(3)

    @contextmanager
    def transaction(*args, **kwargs):
        if current_thread().name == "vlm-admission-cleanup":
            if not first:
                first.append(current_thread())
                first_attempt.set()
                assert expire.wait(3)
                raise sqlite3.OperationalError("expire first cleanup")
            second[:] = [current_thread()]
            second_attempt.set()
            assert recover.wait(3)
        with original(*args, **kwargs) as db:
            yield db

    monkeypatch.setattr(
        admission, "time", SimpleNamespace(monotonic=now, sleep=time.sleep)
    )
    monkeypatch.setattr(admission, "_cleanup_warning", warning)
    monkeypatch.setattr(gate, "_transaction", transaction)
    monkeypatch.setattr(gate, "_remove", lambda token: False)
    try:
        with gate.slot("old", stop=Event(), source_deadline=real_time() + 3):
            pass
        assert first_attempt.wait(2)
        clock["advance"] = True
        expire.set()
        assert handoff.wait(2)
        assert gate._cleanup_thread is None
        with gate.slot("new", stop=Event(), source_deadline=real_time() + 3):
            new_token = next(
                r[0] for r in admission_tickets(gate.settings) if r[1] == "new"
            )
        assert second_attempt.wait(2)
        assert first[0] is not second[0]
        assert not budgets[1].acquire(blocking=False), "old finalizer lost its permit"
        finalize.set()
        first[0].join(3)
        assert not first[0].is_alive()
        assert gate._cleanup_thread is second[0]
        assert new_token in gate._pending_releases
        recover.set()
        second[0].join(3)
        assert not second[0].is_alive()
        assert [row[1] for row in admission_tickets(gate.settings)] == ["old"]
        assert gate._pending_releases == {}
        for budget in budgets:
            assert_permits(budget, 2)
    finally:
        expire.set()
        finalize.set()
        recover.set()
        for worker in first + second:
            worker.join(3)
            assert not worker.is_alive()


def inherited_cleanup_child(gate, inherited_slot, connection):
    try:
        # Must not finalize the parent's acquired context or touch its locks.
        inherited_slot.__exit__(None, None, None)
        gate._transaction = SharedAdmission._transaction.__get__(gate)
        gate._remove = lambda token: False
        with gate.slot("child", stop=Event(), source_deadline=time.monotonic() + 3):
            assert gate._cleanup_pid == os.getpid()
            assert gate._pending_releases == {}
            assert gate._cleanup_thread is None
        worker = gate._cleanup_thread
        if worker is not None:
            worker.join(3)
            assert not worker.is_alive()
        assert gate._pending_releases == {}
        connection.send(("ok", admission_tickets(gate.settings)))
    except BaseException as exc:
        connection.send(("error", repr(exc)))
        raise
    finally:
        connection.close()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork")
def test_fork_resets_cleanup_state_without_releasing_parent_work(tmp_path, monkeypatch):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(settings(tmp_path, global_limit=3, per_stream_limit=1))
    conflict = ReleaseConflict(gate)
    monkeypatch.setattr(gate, "_transaction", conflict.transaction)
    ctx = mp.get_context("fork")
    receive, send = ctx.Pipe(duplex=False)
    process = None
    borrowed = []
    live = None
    try:
        with gate.slot("completed", stop=Event(), source_deadline=time.monotonic() + 3):
            conflict.armed = True
        assert conflict.cleanup_entered.wait(2)
        live = gate.slot(
            "parent-live", stop=Event(), source_deadline=time.monotonic() + 3
        )
        live.__enter__()
        parent_rows = admission_tickets(gate.settings)
        for budget in budgets:
            assert budget.acquire(blocking=False)
            borrowed.append(budget)
        # The child must neither acquire these inherited locks nor use the
        # parent's exhausted budgets, pending IDs, or thread references.
        with gate._cleanup_condition, admission._cleanup_reset_lock:
            process = ctx.Process(
                target=inherited_cleanup_child, args=(gate, live, send)
            )
            process.start()
            send.close()
            assert receive.poll(3), "child waited on inherited cleanup state"
            outcome, rows = receive.recv()
            assert outcome == "ok", rows
        process.join(3)
        assert process.exitcode == 0
        assert rows == parent_rows
        assert admission_tickets(gate.settings) == parent_rows
        assert len(gate._pending_releases) == 1
        for budget in borrowed:
            budget.release()
        borrowed.clear()
        live.__exit__(None, None, None)
        live = None
        conflict.finish()
        assert admission_tickets(gate.settings) == []
        for budget in budgets:
            assert_permits(budget, 2)
    finally:
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(3)
            assert not process.is_alive()
        for budget in borrowed:
            budget.release()
        if live is not None:
            live.__exit__(None, None, None)
        conflict.finish()
        receive.close()
        send.close()


def test_forced_exit_keeps_final_release_lease_conservative(tmp_path):
    config = settings(tmp_path, global_limit=1, per_stream_limit=1)
    SharedAdmission(config)
    ctx = mp.get_context("spawn")
    body_ready, finish_body, returned = ctx.Event(), ctx.Event(), ctx.Event()
    process = ctx.Process(
        target=final_release_participant,
        args=(config, body_ready, finish_body, returned),
    )
    process.start()
    try:
        assert body_ready.wait(3)
        with closing(sqlite3.connect(config.database, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            before = db.execute("SELECT id,active,expires FROM tickets").fetchall()
            finish_body.set()
            assert returned.wait(3)
            assert process.is_alive()
            process.terminate()
            process.join(3)
            assert not process.is_alive()
            assert (
                db.execute("SELECT id,active,expires FROM tickets").fetchall() == before
            )
            assert before[0][1] == 1 and before[0][2] > time.monotonic()
            db.commit()
        assert admission_tickets(config)[0][2] == 1
    finally:
        finish_body.set()
        if process.is_alive():
            process.terminate()
        process.join(3)
        assert not process.is_alive()


def test_parallel_completed_lanes_share_cleanup_and_unblock_waiter(
    tmp_path, monkeypatch
):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(
        settings(tmp_path, global_limit=3, per_stream_limit=3, requests_per_second=80)
    )
    conflict = ReleaseConflict(gate)
    original_remove = gate._remove
    monkeypatch.setattr(gate, "_transaction", conflict.transaction)
    entered, finish, stop = Event(), Event(), Event()
    errors = []

    def waiting_lane():
        try:
            with gate.slot(
                "same", lane="full", stop=stop, source_deadline=time.monotonic() + 3
            ):
                entered.set()
                assert finish.wait(3)
        except BaseException as exc:
            errors.append(exc)

    waiter = Thread(target=waiting_lane)
    try:
        with gate.slot(
            "same", lane="default", stop=Event(), source_deadline=time.monotonic() + 3
        ):
            live_token = admission_tickets(gate.settings)[0][0]
            with ExitStack() as completed:
                for lane in ("events", "compact"):
                    completed.enter_context(
                        gate.slot(
                            "same",
                            lane=lane,
                            stop=Event(),
                            source_deadline=time.monotonic() + 3,
                        )
                    )
                monkeypatch.setattr(gate, "_remove", lambda token: False)
            assert conflict.cleanup_entered.wait(2)
            assert len(gate._pending_releases) == 2
            assert gate._cleanup_thread is conflict.worker
            assert_permits(budgets[1], 1)  # One worker, not a worker per token.
            waiter.start()
            wait_until(lambda: gate.snapshot()["waiting"] == 1)
            assert not entered.is_set()
            monkeypatch.setattr(gate, "_remove", original_remove)
            conflict.finish()
            assert entered.wait(2)
            rows = admission_tickets(gate.settings)
            assert len(rows) == 2 and all(row[2] == 1 for row in rows)
            assert live_token in {row[0] for row in rows}
            assert gate.snapshot()["grants"] == 4
            finish.set()
            waiter.join(3)
            assert not waiter.is_alive()
            assert not errors
        assert admission_tickets(gate.settings) == []
        for budget in budgets:
            assert_permits(budget, 2)
    finally:
        monkeypatch.setattr(gate, "_remove", original_remove)
        stop.set()
        finish.set()
        conflict.finish()
        if waiter.ident is not None:
            waiter.join(3)
            assert not waiter.is_alive()


def test_waiting_timeout_and_cancel_never_schedule_active_cleanup(
    tmp_path, monkeypatch
):
    gate = SharedAdmission(
        settings(
            tmp_path, global_limit=1, per_stream_limit=1, queue_timeout_seconds=0.03
        )
    )
    original_remove = gate._remove
    deferred = []
    monkeypatch.setattr(
        gate, "_defer_completed_release", lambda *args: deferred.append(args)
    )
    with gate.slot("live", stop=Event(), source_deadline=time.monotonic() + 3):
        monkeypatch.setattr(gate, "_remove", lambda token: False)
        try:
            with pytest.raises(AdmissionError, match="admission_timeout"):
                with gate.slot(
                    "waiting", stop=Event(), source_deadline=time.monotonic() + 3
                ):
                    pytest.fail("expired request ran")
            stop = Event()
            stop.set()
            with pytest.raises(AdmissionError, match="admission_cancelled"):
                with gate.slot(
                    "cancelled", stop=stop, source_deadline=time.monotonic() + 3
                ):
                    pytest.fail("cancelled request ran")
            assert deferred == []
            assert gate._pending_releases == {}
            assert gate._cleanup_thread is None
            assert gate.snapshot()["active"] == 1
            assert gate.snapshot()["waiting"] == 0
        finally:
            monkeypatch.setattr(gate, "_remove", original_remove)
    assert gate.snapshot()["active"] == 0


def test_late_heartbeat_cannot_resurrect_completed_cleaned_token(tmp_path, monkeypatch):
    gate = SharedAdmission(settings(tmp_path, global_limit=1, per_stream_limit=1))
    conflict = ReleaseConflict(gate)
    original = gate._transaction
    renewing, resume_renewal = Event(), Event()
    heartbeat = []

    class EarlyHeartbeat:
        def __init__(self):
            self.stopped = Event()
            self.first = True

        def set(self):
            self.stopped.set()

        def wait(self, timeout):
            if self.first:
                self.first = False
                return False
            return self.stopped.wait(timeout)

    @contextmanager
    def transaction(*args, **kwargs):
        if current_thread().name == "vlm-admission-lease":
            heartbeat[:] = [current_thread()]
            renewing.set()
            assert resume_renewal.wait(3)
            with original(*args, **kwargs) as db:
                yield db
        else:
            with conflict.transaction(*args, **kwargs) as db:
                yield db

    monkeypatch.setattr(admission, "Event", EarlyHeartbeat)
    monkeypatch.setattr(gate, "_transaction", transaction)
    try:
        with gate.slot("done", stop=Event(), source_deadline=time.monotonic() + 3):
            assert renewing.wait(2)
            conflict.armed = True
        assert conflict.cleanup_entered.wait(2)
        assert heartbeat[0].is_alive()
        conflict.finish()
        assert admission_tickets(gate.settings) == []
        resume_renewal.set()
        heartbeat[0].join(3)
        assert not heartbeat[0].is_alive()
        assert admission_tickets(gate.settings) == []
    finally:
        resume_renewal.set()
        conflict.finish()
        for worker in heartbeat:
            worker.join(3)
            assert not worker.is_alive()


def test_unexpected_cleanup_failure_is_bounded_and_conservative(tmp_path, monkeypatch):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(settings(tmp_path, global_limit=1, per_stream_limit=1))
    original = gate._transaction
    workers = []

    @contextmanager
    def transaction(*args, **kwargs):
        if current_thread().name == "vlm-admission-cleanup":
            workers[:] = [current_thread()]
            raise sqlite3.ProgrammingError("unexpected cleanup failure")
        with original(*args, **kwargs) as db:
            yield db

    monkeypatch.setattr(gate, "_transaction", transaction)
    monkeypatch.setattr(gate, "_remove", lambda token: False)
    with gate.slot("done", stop=Event(), source_deadline=time.monotonic() + 3):
        original_row = admission_tickets(gate.settings)[0]
    wait_until(lambda: bool(workers))
    workers[0].join(3)
    assert not workers[0].is_alive()
    assert gate._pending_releases == {}
    assert gate._cleanup_thread is None
    assert admission_tickets(gate.settings) == [original_row]
    for budget in budgets:
        assert_permits(budget, 2)


@pytest.mark.parametrize("existing_worker", [False, True])
def test_cleanup_enqueue_failure_returns_only_owned_permits(
    tmp_path, monkeypatch, existing_worker
):
    budgets = cleanup_budgets(monkeypatch)
    gate = SharedAdmission(settings(tmp_path, global_limit=2, per_stream_limit=1))
    conflict = ReleaseConflict(gate)
    monkeypatch.setattr(gate, "_transaction", conflict.transaction)
    monkeypatch.setattr(gate, "_remove", lambda token: False)

    class RejectNewToken(dict):
        def __setitem__(self, key, value):
            raise MemoryError("synthetic bounded bookkeeping failure")

    try:
        if existing_worker:
            with gate.slot("first", stop=Event(), source_deadline=time.monotonic() + 3):
                pass
            assert conflict.cleanup_entered.wait(2)
        with gate._cleanup_condition:
            gate._pending_releases = RejectNewToken(gate._pending_releases)
        with gate.slot("rejected", stop=Event(), source_deadline=time.monotonic() + 3):
            pass
        assert len(gate._pending_releases) == int(existing_worker)
        conflict.finish()
        assert gate._pending_releases == {}
        assert gate._cleanup_thread is None
        assert [row[1] for row in admission_tickets(gate.settings)] == ["rejected"]
        for budget in budgets:
            assert_permits(budget, 2)
    finally:
        conflict.finish()
