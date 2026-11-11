"""Opt-in fair VLM admission across processes on one shared local filesystem.

Only opaque IDs and counters enter SQLite, never credentials or image data.
All participating workers must use the SAME database file (a shared local
volume for containers). This is not a distributed/NFS lock or a GPU scheduler.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Condition, Event, Lock, Thread, current_thread
from uuid import uuid4

# Cleanup resources, not admission capacity. Bound totals across gate instances
# and databases; exhausted bookkeeping leaves the original lease conservative.
_CLEANUP_LIMIT = 128


def _reset_cleanup_budgets():
    global _cleanup_tokens, _cleanup_workers, _cleanup_reset_lock
    _cleanup_tokens = BoundedSemaphore(_CLEANUP_LIMIT)
    _cleanup_workers = BoundedSemaphore(_CLEANUP_LIMIT)
    _cleanup_reset_lock = Lock()


_reset_cleanup_budgets()
if hasattr(os, "register_at_fork"):
    # One module callback, never one bound callback retaining every gate.
    os.register_at_fork(after_in_child=_reset_cleanup_budgets)


def _cleanup_warning(message):
    try:
        logging.getLogger(__name__).warning(message)
    except Exception:
        # A broken logging handler must not replace a model result/exception.
        pass


class AdmissionError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class AdmissionSettings:
    database: str
    global_limit: int = 16
    per_stream_limit: int = 2
    requests_per_second: float = 20.0
    queue_timeout_seconds: float = 1.0

    def __post_init__(self):
        if not self.database or not Path(self.database).is_absolute():
            raise ValueError("VLM admission requires an absolute shared database path")
        if type(self.global_limit) is not int or not 1 <= self.global_limit <= 128:
            raise ValueError("VLM global concurrency must be in 1..128")
        if type(
            self.per_stream_limit
        ) is not int or not 1 <= self.per_stream_limit <= min(4, self.global_limit):
            raise ValueError("VLM per-stream concurrency exceeds shared capacity")
        for value, maximum in [
            (self.requests_per_second, 200),
            (self.queue_timeout_seconds, 3),
        ]:
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError("invalid VLM admission rate or queue timeout")


class SharedAdmission:
    # Leases are renewed while inspect is running. A crashed process keeps its
    # slot conservatively for up to 60 seconds, rather than leaking it forever.
    LEASE_SECONDS = 60.0

    def __init__(self, settings: AdmissionSettings):
        self.settings = settings
        self._reset_cleanup_state()
        path = Path(settings.database)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Waiting readers must not prevent another process from releasing a
        # slot. Keep SQLite's default synchronous durability; WAL only changes
        # the journal/locking strategy on this shared local filesystem.
        with closing(sqlite3.connect(settings.database, timeout=5.0)) as db:
            db.execute("PRAGMA journal_mode=WAL")
        with self._transaction(timeout=5.0) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, "
                "global_limit INTEGER, per_stream_limit INTEGER, rps REAL, "
                "next_grant REAL DEFAULT 0, grants INTEGER DEFAULT 0, "
                "peak_active INTEGER DEFAULT 0)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, "
                "stream TEXT, active INTEGER, created REAL, expires REAL, "
                "lane TEXT NOT NULL DEFAULT 'default')"
            )
            # Upgrade persisted databases without dropping active capacity leases.
            # Stop old workers before upgrading: old binaries do not know lanes.
            columns = {row[1] for row in db.execute("PRAGMA table_info(tickets)")}
            if "lane" not in columns:
                db.execute(
                    "ALTER TABLE tickets ADD COLUMN lane "
                    "TEXT NOT NULL DEFAULT 'default'"
                )
            db.execute(
                "CREATE INDEX IF NOT EXISTS tickets_stream ON tickets(stream, active)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS streams (id TEXT PRIMARY KEY, "
                "last_grant REAL DEFAULT 0, touched REAL)"
            )
            row = db.execute(
                "SELECT global_limit, per_stream_limit, rps FROM policy WHERE id=1"
            ).fetchone()
            expected = (
                settings.global_limit,
                settings.per_stream_limit,
                settings.requests_per_second,
            )
            if row is not None and tuple(row) != expected:
                raise ValueError(
                    "VLM admission policy mismatch; use one policy per database"
                )
            db.execute(
                "INSERT OR IGNORE INTO policy(id,global_limit,per_stream_limit,rps) "
                "VALUES(1,?,?,?)",
                expected,
            )
            # A persisted monotonic timestamp can belong to a previous boot.
            # If this epoch restarted, no old worker can still own these slots.
            latest = db.execute(
                "SELECT MAX(t) FROM (SELECT MAX(created) t FROM tickets "
                "UNION ALL SELECT MAX(last_grant) t FROM streams)"
            ).fetchone()[0]
            if latest is not None and latest > time.monotonic() + 1:
                db.execute("DELETE FROM tickets")
                db.execute("DELETE FROM streams")
                db.execute("UPDATE policy SET next_grant=0 WHERE id=1")

    def _reset_cleanup_state(self):
        self._cleanup_condition = Condition()
        self._pending_releases = {}
        self._cleanup_thread = None
        self._cleanup_token_budget = _cleanup_tokens
        self._cleanup_worker_budget = _cleanup_workers
        self._cleanup_pid = os.getpid()

    def _ensure_cleanup_process(self):
        if self._cleanup_pid != os.getpid():
            # Never take an inherited gate lock: its owning thread may not
            # exist after fork. The module's child callback resets this lock.
            with _cleanup_reset_lock:
                if self._cleanup_pid != os.getpid():
                    self._reset_cleanup_state()

    @contextmanager
    def _transaction(self, timeout=0.1):
        db = sqlite3.connect(
            self.settings.database, timeout=timeout, isolation_level=None
        )
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _reap(db, now):
        db.execute("DELETE FROM tickets WHERE expires <= ?", (now,))
        db.execute(
            "DELETE FROM streams WHERE touched < ? AND id NOT IN "
            "(SELECT stream FROM tickets)",
            (now - 300,),
        )

    def snapshot(self):
        with self._transaction() as db:
            self._reap(db, time.monotonic())
            active, waiting = db.execute(
                "SELECT COALESCE(SUM(active),0), COALESCE(SUM(1-active),0) FROM tickets"
            ).fetchone()
            grants, peak = db.execute(
                "SELECT grants,peak_active FROM policy WHERE id=1"
            ).fetchone()
        return dict(
            active=active,
            waiting=waiting,
            grants=grants,
            peak_active=peak,
            global_limit=self.settings.global_limit,
            per_stream_limit=self.settings.per_stream_limit,
            requests_per_second=self.settings.requests_per_second,
        )

    def _remove(self, token):
        # A release collision must not replace a valid model result with an
        # exception. The expiring lease remains a conservative capacity hold.
        for _ in range(3):
            try:
                with self._transaction() as db:
                    db.execute("DELETE FROM tickets WHERE id=?", (token,))
                return True
            except sqlite3.OperationalError:
                time.sleep(0.01)
        return False

    def _defer_completed_release(self, token, deadline):
        warning = None
        with self._cleanup_condition:
            if token in self._pending_releases:
                return  # Never extend an existing absolute cleanup deadline.
            if time.monotonic() >= deadline:
                warning = "Completed VLM release exceeded its cleanup horizon"
            elif not self._cleanup_token_budget.acquire(blocking=False):
                warning = "Completed VLM release exceeds cleanup token budget"
            elif self._cleanup_thread is not None:
                try:
                    self._pending_releases[token] = deadline
                except Exception:
                    self._cleanup_token_budget.release()
                    raise
                self._cleanup_condition.notify_all()
            elif not self._cleanup_worker_budget.acquire(blocking=False):
                self._cleanup_token_budget.release()
                warning = "Completed VLM release exceeds cleanup worker budget"
            else:
                worker = None
                try:
                    self._pending_releases[token] = deadline
                    worker = Thread(
                        target=self._cleanup_completed,
                        name="vlm-admission-cleanup",
                        daemon=False,
                    )
                    self._cleanup_thread = worker
                    worker.start()
                except Exception:
                    # A start wrapper can report an error after actually
                    # starting. In that case the worker still owns cleanup.
                    if worker is None or worker.ident is None:
                        self._cleanup_thread = None
                        self._pending_releases.pop(token, None)
                        self._cleanup_token_budget.release()
                        self._cleanup_worker_budget.release()
                    warning = "Completed VLM release cleanup could not start normally"
        if warning:
            _cleanup_warning(warning)

    def _cleanup_completed(self):
        worker = current_thread()
        try:
            while True:
                with self._cleanup_condition:
                    now = time.monotonic()
                    expired = [
                        token
                        for token, deadline in self._pending_releases.items()
                        if deadline <= now
                    ]
                    for token in expired:
                        del self._pending_releases[token]
                        self._cleanup_token_budget.release()
                    batch = sorted(
                        self._pending_releases.items(), key=lambda item: item[1]
                    )[: self.settings.global_limit]
                    if not batch:
                        # Enqueue and this handoff share the same lock. A new
                        # enqueue starts a new worker, never a lost old queue.
                        self._cleanup_thread = None
                if expired:
                    _cleanup_warning("Completed VLM release cleanup horizon exhausted")
                if not batch:
                    return
                try:
                    remaining = batch[0][1] - time.monotonic()
                    if remaining <= 0:
                        continue
                    with self._transaction(timeout=min(0.1, remaining)) as db:
                        db.executemany(
                            "DELETE FROM tickets WHERE id=? AND active=1",
                            [(token,) for token, _ in batch],
                        )
                except sqlite3.OperationalError:
                    with self._cleanup_condition:
                        self._cleanup_condition.wait(
                            min(0.01, max(0, batch[0][1] - time.monotonic()))
                        )
                else:
                    # Only a successful transaction acknowledges deletion.
                    # Retrying after rollback/uncertain commit is idempotent.
                    with self._cleanup_condition:
                        for token, _ in batch:
                            del self._pending_releases[token]
                            self._cleanup_token_budget.release()
        except Exception:
            _cleanup_warning("Completed VLM release cleanup failed; lease retained")
        finally:
            with self._cleanup_condition:
                if self._cleanup_thread is worker:
                    # Unexpected worker failure leaves no unserviceable queue.
                    for _ in self._pending_releases:
                        self._cleanup_token_budget.release()
                    self._pending_releases.clear()
                    self._cleanup_thread = None
            # A late finalizer must not clear a replacement worker's queue.
            self._cleanup_worker_budget.release()

    def _candidate(self, db, now):
        return db.execute(
            "SELECT t.id FROM tickets t "
            "JOIN streams s ON s.id=t.stream "
            "WHERE t.active=0 AND t.expires>? AND (SELECT COUNT(*) FROM tickets a "
            "WHERE a.active=1 AND a.stream=t.stream AND a.expires>?) < ? "
            "ORDER BY s.last_grant,t.created,t.id LIMIT 1",
            (now, now, self.settings.per_stream_limit),
        ).fetchone()

    def _may_acquire(self, token):
        # This is only a read-only hint. Capacity, fairness, cancellation and
        # deadlines are checked again inside BEGIN IMMEDIATE before any grant.
        # Previously every waiter took the writer lock every 10 ms, competing
        # with the very grants/releases it needed to make progress.
        with closing(sqlite3.connect(self.settings.database, timeout=0.1)) as db:
            now = time.monotonic()
            if (
                db.execute("SELECT 1 FROM tickets WHERE id=?", (token,)).fetchone()
                is None
            ):
                return True  # Let the write path report superseded work.
            active = db.execute(
                "SELECT COUNT(*) FROM tickets WHERE active=1 AND expires>?", (now,)
            ).fetchone()[0]
            next_grant = db.execute(
                "SELECT next_grant FROM policy WHERE id=1"
            ).fetchone()[0]
            if active >= self.settings.global_limit or now < next_grant:
                return False
            candidate = self._candidate(db, now)
            return candidate is not None and candidate[0] == token

    @contextmanager
    def slot(self, stream, *, stop, source_deadline, lane="default"):
        # Only the replacement key includes the lane. Admission/fairness still
        # count all lanes against the same source and global capacity budgets.
        if lane not in {"default", "events", "compact", "full"}:
            raise ValueError("invalid VLM admission lane")
        self._ensure_cleanup_process()
        owner_pid = os.getpid()
        token = uuid4().hex
        started = time.monotonic()
        deadline = min(source_deadline, started + self.settings.queue_timeout_seconds)
        registered = False
        acquired = False
        heartbeat_stop = Event()
        heartbeat = None
        try:
            while not acquired:
                now = time.monotonic()
                if stop.is_set():
                    raise AdmissionError("admission_cancelled")
                if now >= deadline:
                    raise AdmissionError("admission_timeout")
                try:
                    if registered and not self._may_acquire(token):
                        stop.wait(0.01)
                        continue
                    with self._transaction() as db:
                        now = time.monotonic()
                        if now >= deadline:
                            raise AdmissionError("admission_timeout")
                        if stop.is_set():
                            raise AdmissionError("admission_cancelled")
                        self._reap(db, now)
                        if not registered:
                            db.execute(
                                "INSERT INTO streams(id,touched) VALUES(?,?) "
                                "ON CONFLICT(id) DO UPDATE SET "
                                "touched=excluded.touched",
                                (stream, now),
                            )
                            # One unsent observation per source AND lane. A
                            # grounding request must not replace its events peer.
                            db.execute(
                                "DELETE FROM tickets WHERE stream=? AND lane=? "
                                "AND active=0",
                                (stream, lane),
                            )
                            db.execute(
                                "INSERT INTO tickets"
                                "(id,stream,active,created,expires,lane) "
                                "VALUES(?,?,0,?,?,?)",
                                (token, stream, now, deadline, lane),
                            )
                        elif (
                            db.execute(
                                "SELECT 1 FROM tickets WHERE id=?", (token,)
                            ).fetchone()
                            is None
                        ):
                            raise AdmissionError("admission_superseded")
                        active = db.execute(
                            "SELECT COUNT(*) FROM tickets WHERE active=1"
                        ).fetchone()[0]
                        next_grant = db.execute(
                            "SELECT next_grant FROM policy WHERE id=1"
                        ).fetchone()[0]
                        candidate = self._candidate(db, now)
                        acquired = (
                            active < self.settings.global_limit
                            and now >= next_grant
                            and candidate is not None
                            and candidate[0] == token
                        )
                        if acquired:
                            db.execute(
                                "UPDATE tickets SET active=1,expires=? WHERE id=?",
                                (now + self.LEASE_SECONDS, token),
                            )
                            db.execute(
                                "UPDATE streams SET last_grant=?,touched=? WHERE id=?",
                                (now, now, stream),
                            )
                            db.execute(
                                "UPDATE policy SET next_grant=?,grants=grants+1, "
                                "peak_active=MAX(peak_active,?) WHERE id=1",
                                (
                                    now + 1 / self.settings.requests_per_second,
                                    active + 1,
                                ),
                            )
                    registered = True
                except sqlite3.OperationalError:
                    # Busy/database failure never bypasses the shared limit.
                    acquired = False
                if not acquired:
                    stop.wait(0.01)

            def renew():
                while not heartbeat_stop.wait(5.0):
                    try:
                        with self._transaction() as db:
                            db.execute(
                                "UPDATE tickets SET expires=? WHERE id=? AND active=1",
                                (time.monotonic() + self.LEASE_SECONDS, token),
                            )
                    except sqlite3.OperationalError:
                        # Keep the existing 60-second lease on a brief collision.
                        pass

            heartbeat = Thread(target=renew, name="vlm-admission-lease", daemon=True)
            heartbeat.start()
            yield time.monotonic() - started
        finally:
            # An inherited context in a forked child does not own parent work.
            if os.getpid() == owner_pid:
                completed = time.monotonic()
                heartbeat_stop.set()
                if heartbeat is not None:
                    heartbeat.join(0.3)
                if registered and not self._remove(token) and acquired:
                    # Results retain the short release path. Normal process
                    # exit waits for this non-daemon cleanup, for at most the
                    # remaining 60 s horizon; forced death keeps crash leases.
                    try:
                        self._defer_completed_release(
                            token, completed + self.LEASE_SECONDS
                        )
                    except Exception:
                        _cleanup_warning("Completed VLM release could not be scheduled")
