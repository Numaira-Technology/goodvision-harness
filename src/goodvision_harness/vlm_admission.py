"""Opt-in fair VLM admission across processes on one shared local filesystem.

Only opaque IDs and counters enter SQLite, never credentials or image data.
All participating workers must use the SAME database file (a shared local
volume for containers). This is not a distributed/NFS lock or a GPU scheduler.
"""

from __future__ import annotations

import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4


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
        path = Path(settings.database)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction(timeout=5.0) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, "
                "global_limit INTEGER, per_stream_limit INTEGER, rps REAL, "
                "next_grant REAL DEFAULT 0, grants INTEGER DEFAULT 0, "
                "peak_active INTEGER DEFAULT 0)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, "
                "stream TEXT, active INTEGER, created REAL, expires REAL)"
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
                return
            except sqlite3.OperationalError:
                time.sleep(0.01)

    @contextmanager
    def slot(self, stream, *, stop, source_deadline):
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
                            # One unsent observation per stream. New evidence
                            # supersedes waiting work, never an in-flight call.
                            db.execute(
                                "DELETE FROM tickets WHERE stream=? AND active=0",
                                (stream,),
                            )
                            db.execute(
                                "INSERT INTO tickets VALUES(?,?,0,?,?)",
                                (token, stream, now, deadline),
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
                        candidate = db.execute(
                            "SELECT t.id FROM tickets t "
                            "JOIN streams s ON s.id=t.stream "
                            "WHERE t.active=0 AND (SELECT COUNT(*) FROM tickets a "
                            "WHERE a.active=1 AND a.stream=t.stream) < ? "
                            "ORDER BY s.last_grant,t.created,t.id LIMIT 1",
                            (self.settings.per_stream_limit,),
                        ).fetchone()
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
            heartbeat_stop.set()
            if heartbeat is not None:
                heartbeat.join(0.3)
            if registered:
                self._remove(token)
