"""Private, explicit JSON facts for ordered local-face work and later review.

Embeddings and source images in this store must never be copied to public
result payloads or timing logs. No pickle or live model/tracker state is stored.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from .contracts import NormalizedBBox, json_mapping_as_dict
from .grounding import ModelObject
from .sliding_window import FrameWindow, SampledFrame


class FaceObservationCapacityError(RuntimeError):
    pass


def serialize_objects(objects):
    return [
        {
            "label": item.label,
            "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
            "confidence": item.confidence,
            "attributes": json_mapping_as_dict(item.attributes),
            "actor_ids": list(item.actor_ids),
            "identity_embedding": list(item.identity_embedding),
            "identity_quality": item.identity_quality,
        }
        for item in objects
    ]


def deserialize_objects(rows):
    return tuple(
        ModelObject(
            row["label"],
            NormalizedBBox(*row["bbox"]),
            row["confidence"],
            row["attributes"],
            tuple(row["actor_ids"]),
            tuple(row["identity_embedding"]),
            row["identity_quality"],
        )
        for row in rows
    )


def _window_row(window):
    rows = []
    for frame in window.frames:
        row = {field.name: getattr(frame, field.name) for field in fields(frame)}
        for key in ("jpeg_bytes", "presentation_jpeg_bytes"):
            if row[key] is not None:
                row[key] = base64.b64encode(row[key]).decode("ascii")
        rows.append(row)
    return {"index": window.index, "is_partial": window.is_partial, "frames": rows}


def _decode_window(payload):
    frames = []
    for row in payload["frames"]:
        row = dict(row)
        for key in ("jpeg_bytes", "presentation_jpeg_bytes"):
            if row[key] is not None:
                row[key] = base64.b64decode(row[key], validate=True)
        frames.append(SampledFrame(**row))
    return FrameWindow(payload["index"], tuple(frames), payload["is_partial"])


class FaceObservationStore:
    def __init__(self, path: Path, *, max_bytes=256 * 1024**2, keep_open=False):
        self.path, self.max_bytes = Path(path), max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._keep_open = keep_open
        self._connection = None
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS galleries (
                    version TEXT PRIMARY KEY, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (
                    ticket INTEGER PRIMARY KEY, run_id TEXT, source_sequence INTEGER,
                    source_time REAL, source_metadata BLOB, window BLOB,
                    gallery_version TEXT, raw BLOB, online BLOB,
                    state TEXT, error_type TEXT);
            """)

    @contextmanager
    def _connect(self):
        # The shared lock covers both the connection and its whole transaction.
        # Keeping the last WAL connection alive avoids per-operation close-time
        # checkpoints without weakening synchronous commit or rollback.
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
        """Release the cached connection without rejecting residual durable work."""
        with self._lock:
            self._keep_open = False
            db, self._connection = self._connection, None
            if db is not None:
                db.close()

    @staticmethod
    def _encode(value):
        return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()

    def _reserve(self, db, size):
        used = db.execute(
            "SELECT COALESCE(SUM(COALESCE(LENGTH(window),0)"
            "+COALESCE(LENGTH(raw),0)+COALESCE(LENGTH(online),0)"
            "+LENGTH(source_metadata)+512),0) FROM observations"
        ).fetchone()[0]
        used += db.execute(
            "SELECT COALESCE(SUM(LENGTH(payload)),0) FROM galleries"
        ).fetchone()[0]
        if used + size > self.max_bytes:
            raise FaceObservationCapacityError(
                "private face observation capacity exhausted"
            )

    def admit(self, ticket, run_id, window, gallery):
        row = _window_row(window)
        source_metadata = {
            **row,
            "frames": [
                {
                    key: value
                    for key, value in frame.items()
                    if key not in {"jpeg_bytes", "presentation_jpeg_bytes"}
                }
                for frame in row["frames"]
            ],
        }
        body = self._encode(row)
        gallery_bytes = self._encode(gallery)
        source = self._encode(source_metadata)
        with self._lock, self._connect() as db:
            # Leave room for the two in-flight raw inference responses.
            self._reserve(
                db, len(body) + len(source) + len(gallery_bytes) + 2 * 1024**2
            )
            db.execute(
                "INSERT OR IGNORE INTO galleries VALUES (?,?)",
                (gallery["version"], gallery_bytes),
            )
            db.execute(
                "INSERT INTO observations VALUES (?,?,?,?,?,?,?,NULL,NULL,?,NULL)",
                (
                    ticket,
                    run_id,
                    window.newest_frame.source_frame_index,
                    window.end_timestamp,
                    source,
                    body,
                    gallery["version"],
                    "admitted",
                ),
            )

    def load(self, ticket):
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT window,gallery_version,raw FROM observations WHERE ticket=?",
                (ticket,),
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError("private face source unavailable")
            gallery = db.execute(
                "SELECT payload FROM galleries WHERE version=?", (row[1],)
            ).fetchone()[0]
        return (
            _decode_window(json.loads(row[0])),
            json.loads(gallery),
            json.loads(row[2]) if row[2] is not None else None,
        )

    def record_raw(self, ticket, observation):
        body = self._encode(observation)
        with self._lock, self._connect() as db:
            self._reserve(db, len(body))
            db.execute(
                "UPDATE observations SET raw=?,state='inferred' WHERE ticket=?",
                (body, ticket),
            )

    def commit(self, ticket, online):
        body = self._encode(online)
        with self._lock, self._connect() as db:
            self._reserve(db, len(body))
            db.execute(
                "UPDATE observations SET online=?,window=NULL,state='committed' "
                "WHERE ticket=?",
                (body, ticket),
            )

    def fail(self, ticket, error_type):
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE observations SET state='failed',error_type=? WHERE ticket=?",
                (error_type, ticket),
            )

    def records(self):
        """Explicit private inspection for recovery/review, never public status."""
        with self._lock, self._connect() as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute("SELECT * FROM observations ORDER BY ticket")
            ]
