"""Durable FIFO for archive facts, independent of expensive archive work.

JPEG sources remain available after the in-memory ring expires. A successful put
acknowledges a SQLite FULL synchronous transaction; it does not imply that event
organization or encoding has completed. Failed processing remains recoverable.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import time
from collections import deque
from pathlib import Path
from threading import Condition, RLock
from uuid import uuid4

from .video_evidence import EvidenceFrame


class ArchiveIngress:
    def __init__(self, path: Path, *, max_bytes: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._condition = Condition(RLock())
        self._closed = False
        self._closed_unfinished = 0
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self._db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS ingress (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', created REAL NOT NULL,
                run TEXT NOT NULL DEFAULT '');
            CREATE INDEX IF NOT EXISTS ingress_state ON ingress(state, id);
            CREATE INDEX IF NOT EXISTS ingress_run_state ON ingress(run, state, id);
            CREATE TABLE IF NOT EXISTS sources (
                run TEXT NOT NULL, sequence INTEGER NOT NULL, timestamp REAL NOT NULL,
                jpeg BLOB NOT NULL, PRIMARY KEY(run, sequence));
            CREATE INDEX IF NOT EXISTS source_time ON sources(run, timestamp);
        """)
        # Process-local waiters and shutdown markers are not work to replay.
        with self._db:
            self._db.execute("DELETE FROM ingress WHERE kind IN ('barrier','close')")
            self._db.execute("UPDATE ingress SET state='pending' WHERE state='failed'")
            for record_id, payload in self._db.execute(
                "SELECT id,payload FROM ingress WHERE kind='window' AND run=''"
            ).fetchall():
                self._db.execute("UPDATE ingress SET run=? WHERE id=?", (
                    json.loads(payload).get("session_id", ""), record_id,
                ))
        # Long source reads use a WAL reader, never the producer's admission lock.
        self._read_lock = RLock()
        self._reader = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self._reader.execute("PRAGMA query_only=ON")
        self._inflight = deque()
        self._volatile = {}
        self._failed = set()
        self.max_bytes = max_bytes
        self._bytes = self._db.execute(
            "SELECT COALESCE(SUM(LENGTH(jpeg)),0) FROM sources"
        ).fetchone()[0] + self._db.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(payload AS BLOB))),0) FROM ingress"
        ).fetchone()[0]

    def put(self, item: tuple, timeout: float = 0) -> None:
        kind, payload = item
        run = ""
        frame = None
        token = None
        if kind == "frame":
            run, frame = payload
            value = [run, frame.source_frame_index, frame.timestamp_seconds]
        elif kind == "barrier":
            token = uuid4().hex
            value = token
        elif kind == "encoded":
            metadata, destination, available, error = payload
            value = [metadata, str(destination), available, error]
        else:
            value = payload
        if kind in {"start", "end", "subjects"}:
            run = payload[0]
        elif kind == "result":
            run = payload[0].get("stream_session_id", "")
        elif kind == "window":
            run = payload.get("session_id", "")
        elif kind == "scene":
            run = payload.get("stream_session_id", "")
        elif kind == "encoded":
            run = payload[0].get("stream_session_id", "")
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
        added = len(encoded.encode("utf-8")) + (len(frame.jpeg) if frame else 0)
        with self._condition:
            existing = None
            if frame:
                existing = self._db.execute(
                    "SELECT timestamp,jpeg FROM sources WHERE run=? AND sequence=?",
                    (run, frame.source_frame_index),
                ).fetchone()
                if existing and existing != (frame.timestamp_seconds, frame.jpeg):
                    raise ValueError("conflicting archive source")
                if existing:
                    added -= len(frame.jpeg)
            # A full disk is an explicit admission failure, never oldest eviction.
            if (
                self._bytes + added > self.max_bytes
                and kind not in {"barrier", "close"}
            ):
                raise queue.Full("archive source storage capacity exceeded")
            with self._db:
                if frame:
                    self._db.execute(
                        "INSERT OR IGNORE INTO sources VALUES(?,?,?,?)",
                        (run, frame.source_frame_index,
                         frame.timestamp_seconds, frame.jpeg),
                    )
                self._db.execute(
                    "INSERT INTO ingress(kind,payload,created,run) VALUES(?,?,?,?)",
                    (kind, encoded, time.time(), run),
                )
            self._bytes += added
            if token:
                self._volatile[token] = payload
            self._condition.notify_all()

    def get(self, timeout: float = 0) -> tuple:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                row = self._db.execute(
                    "SELECT current.id,current.kind,current.payload "
                    "FROM ingress current "
                    "WHERE current.state='pending' AND current.id>? "
                    "AND NOT EXISTS (SELECT 1 FROM ingress failed "
                    "WHERE failed.state='failed' AND failed.run=current.run "
                    "AND current.run!='' AND failed.id<current.id) "
                    "ORDER BY current.id LIMIT 1",
                    (self._inflight[-1] if self._inflight else 0,),
                ).fetchone()
                if row:
                    record_id, kind, encoded = row
                    payload = json.loads(encoded)
                    if kind == "frame":
                        run, seq, stamp = payload
                        data = self._db.execute(
                            "SELECT jpeg FROM sources WHERE run=? AND sequence=?",
                            (run, seq),
                        ).fetchone()
                        if data is None:
                            raise RuntimeError("durable archive source missing")
                        payload = (run, EvidenceFrame(data[0], stamp, seq))
                    elif kind == "barrier":
                        payload = self._volatile.pop(payload)
                    elif kind == "encoded":
                        payload[1] = Path(payload[1])
                    self._inflight.append(record_id)
                    return kind, payload
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)

    def task_failed(self) -> None:
        with self._condition:
            if self._inflight:
                self._failed.add(self._inflight[0])

    def task_done(self) -> None:
        with self._condition:
            record_id = self._inflight.popleft()
            state = "failed" if record_id in self._failed else "done"
            self._failed.discard(record_id)
            with self._db:
                self._db.execute(
                    "UPDATE ingress SET state=? WHERE id=?", (state, record_id),
                )
            self._condition.notify_all()

    @property
    def unfinished_tasks(self) -> int:
        with self._condition:
            if self._closed:
                return self._closed_unfinished
            return self._db.execute(
                "SELECT COUNT(*) FROM ingress WHERE state IN ('pending','failed')"
            ).fetchone()[0]

    def qsize(self) -> int:
        with self._condition:
            return max(0, self.unfinished_tasks - len(self._inflight))

    def pending_for(self, run: str) -> int:
        with self._condition:
            return self._db.execute(
                "SELECT COUNT(*) FROM ingress WHERE run=? "
                "AND state IN ('pending','failed')", (run,),
            ).fetchone()[0]

    def source_latest(self, run: str) -> float:
        with self._read_lock:
            return self._reader.execute(
                "SELECT COALESCE(MAX(timestamp),0) FROM sources WHERE run=?", (run,),
            ).fetchone()[0]

    def source_window(self, run: str, start: float, end: float) -> list[EvidenceFrame]:
        with self._read_lock:
            rows = self._reader.execute(
                "SELECT jpeg,timestamp,sequence FROM sources WHERE run=? "
                "AND timestamp>=? AND timestamp<? ORDER BY timestamp",
                (run, start - 1e-6, end - 1e-6),
            ).fetchall()
            successor = self._reader.execute(
                "SELECT jpeg,timestamp,sequence FROM sources WHERE run=? "
                "AND timestamp>=? ORDER BY timestamp LIMIT 1", (run, end - 1e-6),
            ).fetchone()
            if successor and rows:
                rows.append(successor)
        return [EvidenceFrame(*row) for row in rows]

    def processed_runs(self, *, pending_only: bool = False) -> dict[str, dict]:
        """Reconstruct only acknowledged progress, never future admitted sources."""
        with self._read_lock:
            query = (
                "SELECT kind,payload,run FROM ingress WHERE state='done' "
                "AND kind IN ('start','frame','window','end') "
            )
            if pending_only:
                query += (
                    "AND run IN (SELECT DISTINCT run FROM ingress "
                    "WHERE state IN ('pending','failed') AND run!='') "
                )
            rows = self._reader.execute(query + "ORDER BY id").fetchall()
        runs = {}
        for kind, encoded, run in rows:
            if not run:
                continue
            payload = json.loads(encoded)
            current = runs.setdefault(run, {
                "source_id": "", "latest": 0.0, "ended": False,
                "end_reason": "stopped", "windows": {},
            })
            if kind == "start":
                current["source_id"] = payload[1]
            elif kind == "end":
                current.update(ended=True, end_reason=payload[1])
            elif kind == "frame":
                current["latest"] = max(current["latest"], payload[2])
            elif kind == "window":
                current["latest"] = max(current["latest"], payload["timestamp"])
                current["windows"][payload["window_id"]] = payload
        return runs

    def recoverable_runs(self) -> set[str]:
        with self._read_lock:
            return {row[0] for row in self._reader.execute(
                "SELECT DISTINCT run FROM ingress WHERE state!='done' AND run!=''"
            )}

    def prune_completed_runs(self, *, before: float, protected: set[str]) -> int:
        """Release expired source material only after every fact in its run is done."""
        with self._read_lock:
            candidates = self._reader.execute(
                "SELECT run FROM ingress WHERE run!='' GROUP BY run "
                "HAVING MAX(created)<? AND SUM(state!='done')=0 "
                "AND SUM(kind='end')>0", (before,),
            ).fetchall()
        for (run,) in candidates:
            if run in protected:
                continue
            with self._condition:
                if self._db.execute(
                    "SELECT 1 FROM ingress WHERE run=? AND state!='done' LIMIT 1",
                    (run,),
                ).fetchone():
                    continue
                # Bound each maintenance write. A multi-hour run must not hold
                # the producer lock while gigabytes of old sources are deleted.
                sources = self._db.execute(
                    "SELECT sequence,LENGTH(jpeg) FROM sources WHERE run=? LIMIT 8",
                    (run,),
                ).fetchall()
                size = sum(row[1] for row in sources)
                with self._db:
                    self._db.executemany(
                        "DELETE FROM sources WHERE run=? AND sequence=?",
                        ((run, row[0]) for row in sources),
                    )
                    if not self._db.execute(
                        "SELECT 1 FROM sources WHERE run=? LIMIT 1", (run,),
                    ).fetchone():
                        facts = self._db.execute(
                            "SELECT id,LENGTH(CAST(payload AS BLOB)) FROM ingress "
                            "WHERE run=? ORDER BY kind='end',id LIMIT 32", (run,),
                        ).fetchall()
                        size += sum(row[1] for row in facts)
                        self._db.executemany(
                            "DELETE FROM ingress WHERE id=?",
                            ((row[0],) for row in facts),
                        )
                self._bytes -= size
                if size:
                    self._db.execute("PRAGMA incremental_vacuum(32)")
                self._condition.notify_all()
                return size
        return 0

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed_unfinished = self.unfinished_tasks
            self._closed = True
            self._db.close()
        with self._read_lock:
            self._reader.close()
