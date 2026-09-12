"""Bounded asynchronous journal of scene event metadata and corrections.

The worker owns SQLite exclusively. Images, identity details, and arbitrary model
attributes never enter the queue or the database. This is not a video archive.
"""

from __future__ import annotations

import json
import math
import queue
import sqlite3
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

_LABELS = frozenset({"fight", "smoking", "fall", "weapon", "stranger"})
_PHASES = frozenset({"suspected", "confirmed", "ended", "retracted"})
_TEXT_LIMIT = 256


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _TEXT_LIMIT:
        raise ValueError("invalid event identifier")
    return value.strip()


def _number(value: Any, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("invalid source number")
    if integer and (type(value) is not int or value > 2**63 - 1):
        raise ValueError("invalid source sequence")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError("invalid source number") from exc
    if not finite or value < 0:
        raise ValueError("invalid source number")
    return value


class SceneArchive:
    """Queue final event states and keep a restart-safe, size-limited journal."""

    def __init__(
        self,
        directory: Path,
        *,
        retention_days: float = 7,
        max_storage_bytes: int = 64 * 1024 * 1024,
        queue_size: int = 32,
    ) -> None:
        if _number(retention_days) <= 0:
            raise ValueError("retention_days must be positive")
        if type(max_storage_bytes) is not int or max_storage_bytes < 64 * 1024:
            raise ValueError("max_storage_bytes must be at least 65536")
        if type(queue_size) is not int or not 1 <= queue_size <= 1024:
            raise ValueError("queue_size must be an integer between 1 and 1024")
        self.directory = Path(directory)
        self.retention_days = float(retention_days)
        self.max_storage_bytes = max_storage_bytes
        self._row_limit = min(10000, max_storage_bytes // 1024)
        # Leave room for table/index pages, fragmentation, and rollback journaling.
        self._payload_budget = max_storage_bytes // 8
        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue(maxsize=queue_size)
        self._lock = Lock()
        self._closing = Event()
        self._ready = Event()
        self._cache: list[dict[str, Any]] = []
        self._stats: dict[str, Any] = {
            "state": "starting",
            "submitted": 0,
            "persisted_batches": 0,
            "persisted_events": 0,
            "queue_dropped": 0,
            "invalid_snapshots": 0,
            "failures": 0,
            "capacity_evicted": 0,
            "retention_deleted": 0,
            "stored_events": 0,
            "storage_bytes": 0,
            "close_timeouts": 0,
        }
        self._thread = Thread(target=self._run, name="scene-event-journal", daemon=True)
        self._thread.start()
        self._ready.wait(1.0)

    @staticmethod
    def _record(event: Mapping, session: str, source: str) -> dict[str, Any]:
        label, phase = event.get("event_label"), event.get("phase")
        if not isinstance(label, str) or label not in _LABELS:
            raise ValueError("invalid event label")
        if not isinstance(phase, str) or phase not in _PHASES:
            raise ValueError("invalid event phase")
        subjects = event.get("subject_ids")
        if not isinstance(subjects, (list, tuple)) or not 1 <= len(subjects) <= 128:
            raise ValueError("invalid event participants")
        record: dict[str, Any] = {
            "event_id": _text(event.get("event_id")),
            "stream_session_id": session,
            "source_id": source,
            "event_label": label,
            "phase": phase,
            "subject_ids": sorted({_text(value) for value in subjects}),
        }
        for key in (
            "first_evidence_timestamp_seconds",
            "last_evidence_timestamp_seconds",
        ):
            if key in event:
                record[key] = _number(event[key])
        for key in ("frame_sequence", "positive_observations", "negative_observations"):
            if key in event:
                record[key] = _number(event[key], integer=True)
        state = event.get("observation_state")
        if state in ("observed", "uncertain"):
            record["observation_state"] = state
        if event.get("confidence") is not None:
            confidence = _number(event["confidence"])
            if confidence > 1:
                raise ValueError("invalid event confidence")
            record["confidence"] = confidence
        return record

    def submit(self, snapshot: dict) -> bool:
        """Copy only allowlisted metadata; enqueue without waiting for disk I/O."""
        if self._closing.is_set():
            return False
        try:
            scene = snapshot.get("scene", snapshot)
            if not isinstance(scene, Mapping):
                raise ValueError("invalid scene snapshot")
            session, source = (
                _text(scene.get("stream_session_id")),
                _text(scene.get("source_id")),
            )
            events, transitions = scene.get("events", ()), scene.get("transitions", ())
            if (
                not isinstance(events, (list, tuple))
                or len(events) > 1000
                or not isinstance(transitions, (list, tuple))
                or len(transitions) > 2048
            ):
                raise ValueError("oversized event snapshot")
            records = {}
            for transition in transitions:
                if (
                    not isinstance(transition, Mapping)
                    or transition.get("type") != "retract"
                ):
                    continue
                record = self._record(
                    {
                        **transition,
                        "phase": "retracted",
                        "last_evidence_timestamp_seconds": transition.get(
                            "timestamp_seconds"
                        ),
                    },
                    session,
                    source,
                )
                record["_retraction_only"] = True
                records[record["event_id"]] = record
            # Current event state supersedes historical transitions for that ID.
            for event in events:
                if not isinstance(event, Mapping):
                    raise ValueError("invalid event record")
                record = self._record(event, session, source)
                records[record["event_id"]] = record
            if len(records) > 1000:
                raise ValueError("oversized event batch")
            batch = list(records.values())
            if not batch:
                return False
            if any(
                len(json.dumps(record).encode("utf-8"))
                > min(32768, self._payload_budget)
                for record in batch
            ):
                raise ValueError("oversized event record")
        except (AttributeError, TypeError, ValueError):
            with self._lock:
                self._stats["invalid_snapshots"] += 1
            return False
        with self._lock:
            if self._closing.is_set() or self._stats["state"] == "failed":
                return False
            try:
                self._queue.put_nowait(batch)
            except queue.Full:
                self._stats["queue_dropped"] += 1
                return False
            self._stats["submitted"] += 1
            return True

    def list_events(self, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        with self._lock:
            return deepcopy(self._cache[:limit])

    def stats(self) -> dict:
        with self._lock:
            return {**self._stats, "queued": self._queue.qsize()}

    def close(self, timeout_seconds: float = 1) -> None:
        timeout = float(_number(timeout_seconds))
        if timeout > 60:
            raise ValueError("close timeout must not exceed 60 seconds")
        with self._lock:
            self._closing.set()
            if self._stats["state"] not in {"failed", "closed"}:
                self._stats["state"] = "closing"
        self._thread.join(timeout)
        if self._thread.is_alive():
            with self._lock:
                self._stats["close_timeouts"] += 1

    def _refresh(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT payload, updated_at FROM events "
            "ORDER BY updated_at DESC, rowid DESC LIMIT 1000"
        ).fetchall()
        count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        cache = [
            {**json.loads(payload), "archived_at": updated} for payload, updated in rows
        ]
        path = self.directory / "scene-events.sqlite3"
        with self._lock:
            self._cache = cache
            self._stats["stored_events"] = count
            self._stats["storage_bytes"] = path.stat().st_size

    def _retain(self, connection: sqlite3.Connection, now: float) -> int:
        return connection.execute(
            "DELETE FROM events WHERE updated_at < ?",
            (now - self.retention_days * 86400,),
        ).rowcount

    def _write(self, connection: sqlite3.Connection, records: list[dict]) -> None:
        now = time.time()
        removed = self._retain(connection, now)
        evicted = 0
        count, payload_bytes = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(CAST(payload AS BLOB))), 0) "
            "FROM events"
        ).fetchone()
        for original in records:
            record = dict(original)
            retraction = record.pop("_retraction_only", False)
            key = (record["stream_session_id"], record["event_id"])
            previous = connection.execute(
                "SELECT payload FROM events WHERE session_id = ? AND event_id = ?", key
            ).fetchone()
            old_bytes = len(previous[0].encode("utf-8")) if previous else 0
            if previous and retraction:
                record = {**json.loads(previous[0]), **record}
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            size = len(payload.encode("utf-8"))
            while (
                count + (previous is None) > self._row_limit
                or payload_bytes - old_bytes + size > self._payload_budget
            ):
                oldest = connection.execute(
                    "SELECT rowid, LENGTH(CAST(payload AS BLOB)) FROM events "
                    "WHERE NOT(session_id = ? AND event_id = ?) "
                    "ORDER BY updated_at, rowid LIMIT 1",
                    key,
                ).fetchone()
                if oldest is None:
                    raise ValueError("event exceeds archive capacity")
                connection.execute("DELETE FROM events WHERE rowid = ?", (oldest[0],))
                count, payload_bytes = count - 1, payload_bytes - oldest[1]
                evicted += 1
            connection.execute(
                "INSERT INTO events(session_id,event_id,payload,updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(session_id,event_id) "
                "DO UPDATE SET payload=excluded.payload, "
                "updated_at=excluded.updated_at",
                (*key, payload, now),
            )
            count += previous is None
            payload_bytes += size - old_bytes
        connection.commit()
        with self._lock:
            self._stats["retention_deleted"] += removed
            self._stats["capacity_evicted"] += evicted
        self._refresh(connection)

    def _run(self) -> None:
        connection = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / "scene-events.sqlite3"
            connection = sqlite3.connect(path, timeout=0.1)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            # Reserve half the storage envelope for SQLite's transient rollback journal.
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            pages = (self.max_storage_bytes - page_size) // (2 * page_size)
            actual_pages = connection.execute(
                f"PRAGMA max_page_count={pages}"
            ).fetchone()[0]
            if actual_pages > pages:
                raise ValueError(
                    "existing journal exceeds the configured storage quota"
                )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events(session_id TEXT NOT NULL, "
                "event_id TEXT NOT NULL, payload TEXT NOT NULL, "
                "updated_at REAL NOT NULL, PRIMARY KEY(session_id,event_id))"
            )
            removed = self._retain(connection, time.time())
            connection.commit()
            self._refresh(connection)
            with self._lock:
                self._stats["state"] = "running"
                self._stats["retention_deleted"] += removed
            self._ready.set()
            last_maintenance = time.monotonic()
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    records = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - last_maintenance >= 1:
                        removed = self._retain(connection, time.time())
                        connection.commit()
                        with self._lock:
                            self._stats["retention_deleted"] += removed
                        self._refresh(connection)
                        last_maintenance = time.monotonic()
                    continue
                try:
                    self._write(connection, records)
                except (OSError, sqlite3.Error, ValueError) as exc:
                    connection.rollback()
                    with self._lock:
                        self._stats["failures"] += 1
                        self._stats["last_error"] = type(exc).__name__
                else:
                    with self._lock:
                        self._stats["persisted_batches"] += 1
                        self._stats["persisted_events"] += len(records)
                finally:
                    self._queue.task_done()
            with self._lock:
                self._stats["state"] = "closed"
        except (OSError, sqlite3.Error, ValueError) as exc:
            with self._lock:
                self._stats["state"] = "failed"
                self._stats["failures"] += 1
                self._stats["last_error"] = type(exc).__name__
        finally:
            if connection is not None:
                connection.close()
            self._ready.set()
