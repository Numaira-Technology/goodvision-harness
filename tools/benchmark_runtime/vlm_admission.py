"""Opt-in fair VLM admission across processes on one shared local filesystem.

Only opaque IDs and counters enter SQLite, never credentials or image data.
All participating workers must use the SAME database file (a shared local
volume for containers). This is not a distributed/NFS lock or a GPU scheduler.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

MAX_PER_STREAM_CONCURRENCY = 8


@dataclass
class AdmissionTrace:
    """Per-request numeric observations, not exclusive causal time attribution.

    Poll conditions may overlap. Counts are samples, not rejected requests.
    Database operation durations include Python/OS scheduling and SQLite work;
    begin_seconds is a subset of transaction_seconds, not an additional phase.
    """

    values: dict = field(default_factory=dict)

    def add(self, name, value=1):
        self.values[name] = self.values.get(name, 0) + value

    def maximum(self, name, value):
        self.values[name] = max(self.values.get(name, 0), value)

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, max(0, time.perf_counter() - started))


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
        ) is not int or not 1 <= self.per_stream_limit <= min(
            MAX_PER_STREAM_CONCURRENCY, self.global_limit
        ):
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
    RELEASE_TIMEOUT_SECONDS = 5.0

    def __init__(self, settings: AdmissionSettings):
        self.settings = settings
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
            if "pair_id" not in columns:
                db.execute(
                    "ALTER TABLE tickets ADD COLUMN pair_id TEXT NOT NULL DEFAULT ''"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS pair_grants (stream TEXT, pair_id TEXT, "
                "lane TEXT, expires REAL, PRIMARY KEY(stream,pair_id,lane))"
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
                db.execute("DELETE FROM pair_grants")
                db.execute("UPDATE policy SET next_grant=0 WHERE id=1")

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
        db.execute("DELETE FROM pair_grants WHERE expires <= ?", (now,))
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
        # Give a completed request enough time to release its slot during writer
        # contention. Three 100 ms attempts left 60-second phantom capacity holds.
        # Never replace a valid model result or its original exception with a
        # cleanup error; persistent failures remain visible and expire safely.
        deadline = time.monotonic() + self.RELEASE_TIMEOUT_SECONDS
        error_code = None
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                with self._transaction(timeout=min(0.5, remaining)) as db:
                    db.execute("DELETE FROM tickets WHERE id=?", (token,))
                return True
            except sqlite3.OperationalError as exc:
                error_code = getattr(exc, "sqlite_errorcode", None)
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        logging.getLogger(__name__).warning(
            "VLM admission release failed; capacity lease remains until expiry "
            "(sqlite_errorcode=%s)",
            error_code if type(error_code) is int else None,
        )
        return False

    def _candidate(self, db, now):
        return db.execute(
            "SELECT t.id FROM tickets t "
            "JOIN streams s ON s.id=t.stream "
            "WHERE t.active=0 AND t.expires>? AND (SELECT COUNT(*) FROM tickets a "
            "WHERE a.active=1 AND a.stream=t.stream AND a.expires>?) < ? "
            # Preserve fairness between cameras. Within a camera, finish an
            # already-started pair before starting another observation window.
            "ORDER BY s.last_grant, EXISTS(SELECT 1 FROM pair_grants p "
            "WHERE t.pair_id!='' AND p.stream=t.stream AND p.pair_id=t.pair_id "
            "AND p.lane!=t.lane AND p.expires>?) DESC,t.created,t.id LIMIT 1",
            (now, now, self.settings.per_stream_limit, now),
        ).fetchone()

    def _may_acquire(self, token, diagnostics=None):
        # This is only a read-only hint. Capacity, fairness, cancellation and
        # deadlines are checked again inside BEGIN IMMEDIATE before any grant.
        # Previously every waiter took the writer lock every 10 ms, competing
        # with the very grants/releases it needed to make progress.
        with closing(sqlite3.connect(self.settings.database, timeout=0.1)) as db:
            now = time.monotonic()
            ticket = db.execute(
                "SELECT stream FROM tickets WHERE id=?", (token,)
            ).fetchone()
            if ticket is None:
                return True  # Let the write path report superseded work.
            if diagnostics is None:
                active = db.execute(
                    "SELECT COUNT(*) FROM tickets WHERE active=1 AND expires>?", (now,)
                ).fetchone()[0]
                own_active = None
            else:
                # Same roundtrip as the normal count; no extra polling queries.
                active, own_active = db.execute(
                    "SELECT COUNT(*),COALESCE(SUM(stream=?),0) FROM tickets "
                    "WHERE active=1 AND expires>?",
                    (ticket[0], now),
                ).fetchone()
            next_grant = db.execute(
                "SELECT next_grant FROM policy WHERE id=1"
            ).fetchone()[0]
            if diagnostics is not None:
                diagnostics.add("polls")
                diagnostics.maximum("global_active_peak_observed", active)
                diagnostics.maximum("stream_active_peak_observed", own_active)
                diagnostics.add(
                    "global_full_polls",
                    int(active >= self.settings.global_limit),
                )
                diagnostics.add(
                    "stream_full_polls",
                    int(own_active >= self.settings.per_stream_limit),
                )
                diagnostics.add("rate_spacing_polls", int(now < next_grant))
            if active >= self.settings.global_limit or now < next_grant:
                return False
            candidate = self._candidate(db, now)
            if diagnostics is not None:
                diagnostics.add("candidate_checks")
                if candidate is None:
                    # Expired/ineligible work is not evidence of another
                    # camera taking priority. Keep admission behavior intact.
                    diagnostics.add("no_candidate_polls")
                elif candidate[0] != token:
                    diagnostics.add("other_candidate_polls")
                    if own_active < self.settings.per_stream_limit:
                        diagnostics.add("other_candidate_with_capacity_polls")
            return candidate is not None and candidate[0] == token

    @contextmanager
    def _observed_transaction(self, diagnostics):
        if diagnostics is None:
            with self._transaction() as db:
                yield db
            return
        with diagnostics.measure("transaction_seconds"):
            began = time.perf_counter()
            entered = False
            try:
                with self._transaction() as db:
                    entered = True
                    diagnostics.add(
                        "transaction_begin_seconds",
                        time.perf_counter() - began,
                    )
                    yield db
            finally:
                if not entered:
                    diagnostics.add(
                        "transaction_begin_seconds",
                        time.perf_counter() - began,
                    )

    @contextmanager
    def slot(
        self,
        stream,
        *,
        stop,
        source_deadline,
        lane="default",
        pair_id="",
        diagnostics=None,
    ):
        # Only the replacement key includes the lane. Admission/fairness still
        # count all lanes against the same source and global capacity budgets.
        if lane not in {"default", "events", "compact", "full"}:
            raise ValueError("invalid VLM admission lane")
        if (
            not isinstance(pair_id, str)
            or len(pair_id) > 128
            or (pair_id and lane not in {"events", "compact"})
        ):
            raise ValueError("invalid VLM admission pair")
        token = uuid4().hex
        started = time.monotonic()
        deadline = min(source_deadline, started + self.settings.queue_timeout_seconds)
        registered = False
        acquired = False
        heartbeat_stop = Event()
        heartbeat = None
        yielded_at = None

        def pause():
            if diagnostics is None:
                stop.wait(0.01)
            else:
                before = time.monotonic()
                stop.wait(0.01)
                elapsed = time.monotonic() - before
                diagnostics.add("poll_sleep_seconds", elapsed)
                diagnostics.add("poll_oversleep_seconds", max(0, elapsed - 0.01))

        try:
            while not acquired:
                now = time.monotonic()
                if stop.is_set():
                    raise AdmissionError("admission_cancelled")
                if now >= deadline:
                    raise AdmissionError("admission_timeout")
                try:
                    if registered:
                        with (
                            diagnostics.measure("read_check_seconds")
                            if diagnostics is not None
                            else nullcontext()
                        ):
                            possible = (
                                self._may_acquire(token, diagnostics=diagnostics)
                                if diagnostics is not None
                                else self._may_acquire(token)
                            )
                        if not possible:
                            pause()
                            continue
                    with self._observed_transaction(diagnostics) as db:
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
                            # Independent work retains latest-only replacement.
                            # Paired work is bounded by local worker slots and
                            # the existing queue/source deadlines; never delete
                            # its waiting mate when a newer window arrives.
                            if not pair_id:
                                db.execute(
                                    "DELETE FROM tickets WHERE stream=? AND lane=? "
                                    "AND active=0 AND pair_id=''",
                                    (stream, lane),
                                )
                            db.execute(
                                "INSERT INTO tickets"
                                "(id,stream,active,created,expires,lane,pair_id) "
                                "VALUES(?,?,0,?,?,?,?)",
                                (token, stream, now, deadline, lane, pair_id),
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
                        if diagnostics is not None:
                            diagnostics.add("write_checks")
                            diagnostics.add("write_checks_eligible", int(acquired))
                        if acquired:
                            if pair_id:
                                # Keep this marker after a fast reply releases
                                # its slot, but never beyond evidence freshness.
                                db.execute(
                                    "INSERT OR REPLACE INTO pair_grants "
                                    "(stream,pair_id,lane,expires) VALUES(?,?,?,?)",
                                    (stream, pair_id, lane, source_deadline),
                                )
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
                    if diagnostics is not None:
                        diagnostics.add("sqlite_errors")
                    acquired = False
                if not acquired:
                    pause()

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
            yielded_at = time.monotonic()
            if diagnostics is not None:
                diagnostics.add("acquired")
                diagnostics.add("acquire_wait_seconds", yielded_at - started)
            yield time.monotonic() - started
        finally:
            if diagnostics is not None:
                if yielded_at is None:
                    diagnostics.add("acquire_wait_seconds", time.monotonic() - started)
                else:
                    diagnostics.add(
                        "occupied_before_release_seconds",
                        time.monotonic() - yielded_at,
                    )
            heartbeat_stop.set()
            if heartbeat is not None:
                heartbeat.join(0.3)
            if registered:
                with (
                    diagnostics.measure("release_seconds")
                    if diagnostics is not None
                    else nullcontext()
                ):
                    released = self._remove(token)
                if diagnostics is not None:
                    diagnostics.add("release_failed", int(not released))
