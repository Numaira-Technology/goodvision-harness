"""Private durable SAM ingress and recovery facts, independent of model I/O.

Only a bounded active prefix is decoded in memory. Interrupted or failed work is
retained; reopening this journal does not automatically replay a native epoch.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from .video_tracking import VideoFrame


class TrackingSpoolFull(RuntimeError):
    """New work could not be durably admitted; existing work remains intact."""


class TrackingSpool:
    def __init__(
        self, path: Path, *, max_bytes: int = 256 * 1024 * 1024,
        keep_open: bool = False,
    ):
        if type(max_bytes) is not int or max_bytes < 4096:
            raise ValueError("SAM spool byte limit must be at least 4096")
        self.path, self.max_bytes = Path(path), max_bytes
        self._lock = threading.RLock()
        self._keep_open = keep_open
        self._connection = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    generation INTEGER, sequence INTEGER, timestamp REAL,
                    received REAL, session TEXT, jpeg BLOB, batch_id TEXT,
                    PRIMARY KEY (generation, sequence));
                CREATE TABLE IF NOT EXISTS batches (
                    request_id TEXT PRIMARY KEY, generation INTEGER,
                    ordinal INTEGER, state TEXT, payload BLOB,
                    result BLOB, error_type TEXT);
                CREATE TABLE IF NOT EXISTS skipped_sources (
                    generation INTEGER, sequence INTEGER, reason TEXT,
                    PRIMARY KEY (generation, sequence));
                CREATE TABLE IF NOT EXISTS lifecycle (
                    generation INTEGER PRIMARY KEY, session TEXT, state TEXT,
                    error_type TEXT);
            """)

    @contextmanager
    def _connect(self):
        # Every use, including transaction completion and close(), shares this
        # lock. Reuse avoids closing the last WAL connection after each write;
        # transactions still commit synchronously before an operation returns.
        with self._lock:
            db = self._connection
            if db is None:
                db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
                try:
                    db.execute("PRAGMA synchronous=FULL")
                    db.execute("PRAGMA wal_autocheckpoint=64")
                except BaseException:
                    db.close()
                    raise
                if self._keep_open:
                    self._connection = db
            try:
                with db:
                    yield db
            finally:
                db.row_factory = None
                if db is not self._connection:
                    db.close()

    def close(self):
        """Release the cached connection; late writes retain short transactions."""
        with self._lock:
            self._keep_open = False
            db, self._connection = self._connection, None
            if db is not None:
                db.close()

    @staticmethod
    def _usage(db):
        source = db.execute(
            "SELECT COALESCE(SUM(COALESCE(LENGTH(jpeg),0)+256),0) FROM sources"
        ).fetchone()[0]
        batches = db.execute(
            "SELECT COALESCE(SUM(COALESCE(LENGTH(payload),0)"
            "+COALESCE(LENGTH(result),0)+512),0) FROM batches"
        ).fetchone()[0]
        skipped = db.execute(
            "SELECT COUNT(*) * 128 FROM skipped_sources"
        ).fetchone()[0]
        return source + batches + skipped

    def begin(self, generation: int, session: str | None):
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO lifecycle VALUES (?,?,?,NULL)",
                (generation, session, "active"),
            )

    def admit(self, generation, session, frame, received):
        with self._lock, self._connect() as db:
            # Reserve room for prepared payloads/results, so ordinary ingress
            # cannot consume the entire journal budget behind an active request.
            reserve = min(48 * 1024 * 1024, self.max_bytes // 2)
            if self._usage(db) + len(frame.jpeg_bytes) + 256 > self.max_bytes - reserve:
                raise TrackingSpoolFull("SAM source journal capacity exhausted")
            db.execute(
                "INSERT INTO sources VALUES (?,?,?,?,?,?,NULL)",
                (
                    generation,
                    frame.frame_sequence,
                    frame.timestamp_seconds,
                    received,
                    session,
                    frame.jpeg_bytes,
                ),
            )

    def skip(self, generation, sequences, reason, *, request_id=None):
        """Atomically retire only sources proven never sent to the worker.

        Source metadata survives for audit; only this private SAM JPEG copy is
        released. A prepared batch must be retired whole, with its hash intact.
        """
        if reason not in {"superseded", "expired"} or not sequences:
            raise ValueError("invalid SAM skip decision")
        sequences = tuple(sequences)
        if len(set(sequences)) != len(sequences):
            raise ValueError("duplicate SAM skip sequence")
        with self._lock, self._connect() as db:
            if request_id is not None:
                batch = db.execute(
                    "SELECT generation,state FROM batches WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                members = db.execute(
                    "SELECT sequence FROM sources WHERE batch_id=?",
                    (request_id,),
                ).fetchall()
                if (batch != (generation, "prepared")
                        or {r[0] for r in members} != set(sequences)):
                    raise RuntimeError("cannot skip sent or partial SAM batch")
            for sequence in sequences:
                row = db.execute(
                    "SELECT batch_id,jpeg IS NOT NULL FROM sources "
                    "WHERE generation=? AND sequence=?",
                    (generation, sequence),
                ).fetchone()
                if row != (request_id, 1):
                    raise RuntimeError("cannot skip unavailable or assigned source")
            # Releasing JPEG space exceeds the fixed-size audit row cost.
            # Still check the bound so tiny test frames cannot bypass quota.
            released = sum(db.execute(
                "SELECT LENGTH(jpeg) FROM sources WHERE generation=? AND sequence=?",
                (generation, sequence),
            ).fetchone()[0] for sequence in sequences)
            if self._usage(db) - released + 128 * len(sequences) > self.max_bytes:
                raise TrackingSpoolFull("SAM skip journal capacity exhausted")
            db.executemany(
                "INSERT INTO skipped_sources VALUES (?,?,?)",
                [(generation, sequence, reason) for sequence in sequences],
            )
            db.executemany(
                "UPDATE sources SET jpeg=NULL WHERE generation=? AND sequence=?",
                [(generation, sequence) for sequence in sequences],
            )
            if request_id is not None:
                db.execute(
                    "UPDATE batches SET state='abandoned',payload=NULL "
                    "WHERE request_id=?", (request_id,),
                )

    def skipped_records(self):
        """Private audit with original source identity, never image bytes."""
        with self._lock, self._connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(
                "SELECT s.generation,s.session,s.sequence,s.timestamp,s.received,"
                "s.batch_id,k.reason FROM sources s JOIN skipped_sources k "
                "ON s.generation=k.generation AND s.sequence=k.sequence "
                "ORDER BY s.generation,s.sequence"
            )]

    def load(self, generation, sequences):
        with self._lock, self._connect() as db:
            records = [
                db.execute(
                    "SELECT sequence,timestamp,jpeg FROM sources "
                    "WHERE generation=? AND sequence=?",
                    (generation, sequence),
                ).fetchone()
                for sequence in sequences
            ]
        if any(row is None or row[2] is None for row in records):
            raise RuntimeError("SAM journal source is unavailable")
        return tuple(VideoFrame(*row) for row in records)

    def sources_for_request(self, request_id, session):
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT sequence,timestamp,jpeg FROM sources "
                "WHERE batch_id=? AND session=? ORDER BY sequence",
                (request_id, session),
            ).fetchall()
        if any(row[2] is None for row in rows):
            return ()  # Live commit already acknowledged and released these bytes.
        return tuple(VideoFrame(*row) for row in rows)

    def batch(self, request_id, generation, ordinal, sequences):
        with self._lock, self._connect() as db:
            if self._usage(db) + 512 > self.max_bytes:
                raise TrackingSpoolFull("SAM batch journal capacity exhausted")
            db.execute(
                "INSERT INTO batches (request_id,generation,ordinal,state) VALUES (?,?,?,?)",
                (request_id, generation, ordinal, "preparing"),
            )
            db.executemany(
                "UPDATE sources SET batch_id=? WHERE generation=? AND sequence=?",
                [(request_id, generation, sequence) for sequence in sequences],
            )

    def prepared(self, request_id, body):
        with self._lock, self._connect() as db:
            if self._usage(db) + len(body) > self.max_bytes:
                raise TrackingSpoolFull("SAM prepared journal capacity exhausted")
            db.execute(
                "UPDATE batches SET payload=?,state='prepared' "
                "WHERE request_id=?",
                (body, request_id),
            )

    def result(self, request_id, result):
        payload = {field.name: getattr(result, field.name) for field in fields(result)}
        payload["frames"] = [snapshot.as_dict() for snapshot in result.frames]
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        with self._lock, self._connect() as db:
            if self._usage(db) + len(body) > self.max_bytes:
                raise TrackingSpoolFull("SAM result journal capacity exhausted")
            db.execute(
                "UPDATE batches SET result=?,state='received' WHERE request_id=?",
                (body, request_id),
            )

    def mark(self, request_id, state, error_type=None):
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE batches SET state=?,error_type=? WHERE request_id=?",
                (state, error_type, request_id),
            )
            if state == "committed":
                db.execute(
                    "UPDATE sources SET jpeg=NULL WHERE batch_id=?", (request_id,)
                )
                db.execute(
                    "UPDATE batches SET payload=NULL WHERE request_id=?", (request_id,)
                )

    def finish(self, generation, state, error_type=None):
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE lifecycle SET state=?,error_type=? WHERE generation=?",
                (state, error_type, generation),
            )

    def stats(self):
        with self._lock, self._connect() as db:
            retained = db.execute(
                "SELECT COUNT(*) FROM sources WHERE jpeg IS NOT NULL"
            ).fetchone()[0]
            results = db.execute(
                "SELECT COUNT(*) FROM batches WHERE result IS NOT NULL "
                "AND state!='committed'"
            ).fetchone()[0]
            return {
                "retained_sources": retained,
                "recoverable_results": results,
                "logical_bytes": self._usage(db),
                "max_bytes": self.max_bytes,
            }

    def records(self):
        """Private recovery inspection; never included in public status/logs."""
        with self._lock, self._connect() as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM batches ORDER BY generation,ordinal"
                )
            ]
