"""Asynchronous person-event journal, revisable episodes, and video evidence.

The live pipeline only submits immutable observations. One owner thread orders
observations by source time and owns SQLite; a separate worker encodes clips.
"""

from __future__ import annotations

import copy
import json
import math
import queue
import shutil
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from .archive_ingress import ArchiveIngress
from .person_identity import stranger_evidence
from .video_evidence import (
    EvidenceFrame,
    FrameRing,
    encode_staged_clip,
    stage_clip,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _label(value: str) -> str:
    return "stranger" if value in {"unknown_person", "unknown_face"} else value


@dataclass(frozen=True, slots=True)
class EventArchiveSettings:
    directory: Path = Path(".goodvision/event-evidence")
    clip_seconds: float = 3.0
    pre_roll_seconds: float = 1.0
    fps: float = 5.0
    max_image_side: int = 720
    jpeg_quality: int = 70
    buffer_seconds: float = 90.0
    buffer_max_bytes: int = 64 * 1024 * 1024
    retention_days: float = 7.0
    max_storage_bytes: int = 10 * 1024**3
    revision_seconds: float = 65.0
    negative_gap_seconds: float = 2.0
    unknown_timeout_seconds: float = 5.0
    queue_size: int = 512
    encoder_timeout_seconds: float = 10.0
    ffmpeg_path: str = "ffmpeg"

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        for name in (
            "clip_seconds",
            "fps",
            "buffer_seconds",
            "retention_days",
            "revision_seconds",
            "negative_gap_seconds",
            "unknown_timeout_seconds",
            "encoder_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"events.{name} must be positive and finite")
        for name in (
            "max_image_side",
            "buffer_max_bytes",
            "max_storage_bytes",
            "queue_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"events.{name} must be a positive integer")
        if not 0 <= self.pre_roll_seconds < self.clip_seconds:
            raise ValueError("events.pre_roll_seconds must be within the clip")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("events.jpeg_quality must be between 1 and 100")
        if self.buffer_seconds < self.revision_seconds + self.clip_seconds:
            raise ValueError("events.buffer_seconds must cover revisions and a clip")


@dataclass(slots=True)
class _Session:
    source_id: str
    ring: FrameRing
    latest: float = 0.0
    last_activity: float = 0.0
    ended: bool = False
    end_reason: str = "stopped"
    windows: dict[str, dict[str, Any]] = field(default_factory=dict)
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    pins: dict[str, list[EvidenceFrame]] = field(default_factory=dict)


class EventArchive:
    """Bounded nonblocking ingress, with read snapshots safe for HTTP threads."""

    def __init__(
        self,
        settings: EventArchiveSettings | None = None,
        *,
        encoder: Callable[..., dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or EventArchiveSettings()
        self._clock = clock
        self._encoder = encoder or encode_staged_clip
        self._queue = ArchiveIngress(
            self.settings.directory / "ingress.sqlite3",
            max_bytes=max(1, self.settings.max_storage_bytes // 2),
        )
        self._recovered_runs = self._queue.processed_runs(pending_only=True)
        self._recoverable_runs = self._queue.recoverable_runs()
        self._encode_queue: queue.Queue[tuple[Path, Path] | None] = queue.Queue(32)
        self._lock = threading.RLock()
        self._ingest_lock = threading.Lock()
        self._closed = threading.Event()
        self._initialized = threading.Event()
        self._sessions: dict[str, _Session] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        self._events: dict[str, dict[str, Any]] = {}
        self._scene_events: dict[str, dict[str, Any]] = {}
        self._anchors: dict[str, tuple[int, str]] = {}
        self._aliases: dict[str, str] = {}
        self._subject_aliases: dict[tuple[str, str], str] = {}
        self._subject_identities: dict[tuple[str, str], dict[str, Any]] = {}
        self._links: dict[str, set[str]] = defaultdict(set)
        self._videos: dict[tuple[str, int], Path] = {}
        self._leases: dict[tuple[str, int], int] = {}
        self._jobs: dict[tuple[str, int], dict[str, Any]] = {}
        self._changes: deque[dict[str, Any]] = deque(maxlen=2000)
        self._epoch = uuid4().hex
        self._reset_reason = "epoch_changed"
        self._cursor = 0
        self._anchor_sequence = 0
        self._ingest_times: dict[str, float] = {}
        self._counts = {
            "submitted": 0,
            "dropped": 0,
            "late_results": 0,
            "encoding_failures": 0,
            "storage_failures": 0,
        }
        self._error: str | None = None
        self._status_snapshot: dict[str, Any] = {"state": "initializing"}
        self._db: sqlite3.Connection | None = None
        self._persisted_events: dict[str, str] = {}
        self._persisted_scene_events: dict[str, str] = {}
        self._persisted_observations: dict[str, str] = {}
        self._persisted_anchors: dict[str, tuple[int, str]] = {}
        self._persisted_aliases: dict[str, str] = {}
        self._persisted_links: dict[str, frozenset[str]] = {}
        self._queued_frame_bytes = 0
        self._storage_bytes = 0
        self._last_maintenance = self._clock()
        self._thread = threading.Thread(
            target=self._run,
            name="event-archive",
            daemon=True,
        )
        self._encoder_thread = threading.Thread(
            target=self._encode_loop,
            name="event-video-encoder",
            daemon=True,
        )
        self._encoder_thread.start()
        self._thread.start()

    def _submit(self, kind: str, payload: Any, timeout_seconds: float = 0) -> bool:
        if self._closed.is_set():
            return False
        try:
            self._queue.put((kind, payload), timeout=max(0.0, timeout_seconds))
        except queue.Full:
            with self._ingest_lock:
                self._counts["dropped"] += 1
                self._error = "archive_ingress_capacity"
            return False
        except (OSError, ValueError, sqlite3.Error):
            with self._ingest_lock:
                self._counts["storage_failures"] += 1
                self._error = "archive_ingress_storage_error"
            return False
        with self._ingest_lock:
            self._counts["submitted"] += 1
        return True

    def start_session(self, session_id: str, source_id: str = "") -> bool:
        return self._submit("start", (session_id, source_id))

    def submit_frame(
        self,
        jpeg: bytes,
        timestamp_seconds: float,
        source_frame_index: int,
        stream_session_id: str,
    ) -> bool:
        if not jpeg or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            return False
        with self._ingest_lock:
            last = self._ingest_times.get(stream_session_id, -math.inf)
            if timestamp_seconds < last + 1.0 / self.settings.fps - 1e-6:
                return True
            self._queued_frame_bytes += len(jpeg)
            self._ingest_times[stream_session_id] = timestamp_seconds
        accepted = self._submit(
            "frame",
            (
                stream_session_id,
                EvidenceFrame(
                    bytes(jpeg), float(timestamp_seconds), source_frame_index
                ),
            ),
        )
        if not accepted:
            with self._ingest_lock:
                self._queued_frame_bytes -= len(jpeg)
                self._ingest_times[stream_session_id] = last
        return accepted

    def note_window(
        self,
        stream_session_id: str,
        window_id: str | int,
        timestamp_seconds: float,
        expected_labels: tuple[str, ...] = (),
        outcome: str = "admitted",
        source_frame_index: int | None = None,
    ) -> bool:
        return self._submit(
            "window",
            {
                "session_id": stream_session_id,
                "window_id": str(window_id),
                # FrameReference.as_dict() publishes millisecond precision.
                # Use the same timeline for window matching and the watermark,
                # so rounding down cannot make the first result appear late.
                "timestamp": round(float(timestamp_seconds), 3),
                "labels": [_label(v) for v in expected_labels],
                "outcome": outcome,
                "source_frame_index": source_frame_index,
                "admitted": self._clock(),
            },
        )

    def submit_result(
        self,
        result: Any,
        *,
        actors: dict[str, Any] | None = None,
        window_id: str | int | None = None,
    ) -> bool:
        payload = (
            result.as_dict() if hasattr(result, "as_dict") else copy.deepcopy(result)
        )
        return self._submit("result", (payload, copy.deepcopy(actors or {}), window_id))

    def end_session(
        self, session_id: str, reason: str = "stopped", *, timeout_seconds: float = 1.0
    ) -> bool:
        return self._submit("end", (session_id, reason), timeout_seconds)

    def submit_scene(
        self, snapshot: dict[str, Any], *, timeout_seconds: float = 0
    ) -> bool:
        """Archive authoritative temporal episodes without re-grouping by person.

        Only safety events use this path. Stranger evidence continues to come
        from the face pipeline and retains its one-clip-per-person policy.
        Copy metadata only, never scene images, masks or identity embeddings.
        """
        scene = snapshot.get("scene", snapshot)
        if not scene or not scene.get("stream_session_id"):
            return True
        keys = (
            "event_id",
            "event_label",
            "subject_ids",
            "phase",
            "first_evidence_timestamp_seconds",
            "last_evidence_timestamp_seconds",
            "positive_observations",
            "confidence",
            "review_outcome",
        )
        records = []
        for event in scene.get("events", ()):
            if event.get("event_label") not in {"smoking", "fall", "fight", "weapon"}:
                continue
            record = {key: copy.deepcopy(event[key]) for key in keys if key in event}
            positives = [
                item["timestamp_seconds"]
                for item in event.get("history", ())
                if item.get("detected") is True
            ]
            latest_positive = event.get("last_positive_timestamp_seconds")
            if latest_positive is not None:
                record["last_positive_seconds"] = latest_positive
            elif positives:
                record["last_positive_seconds"] = max(positives)
            records.append(record)
        latest_transitions = {
            item["event_id"]: item for item in scene.get("transitions", ())
        }
        transitions = [
            {
                **{
                    key: copy.deepcopy(item[key])
                    for key in (
                        "event_id",
                        "event_label",
                        "subject_ids",
                        "timestamp_seconds",
                    )
                    if key in item
                },
                "phase": "retracted" if item.get("type") == "retract" else "ended",
            }
            for item in latest_transitions.values()
            if (item.get("type") in {"end", "retract"} or item.get("phase") == "ended")
            and item.get("event_label") in {"smoking", "fall", "fight", "weapon"}
        ]
        if not records and not transitions:
            return True
        return self._submit(
            "scene",
            {
                "stream_session_id": scene["stream_session_id"],
                "source_id": scene.get("source_id", ""),
                "events": records,
                "transitions": transitions,
            },
            timeout_seconds,
        )

    def reconcile_subjects(
        self,
        stream_session_id: str,
        aliases: dict[str, str],
        identities: dict[str, dict[str, Any]] | None = None,
        *,
        timeout_seconds: float = 0,
    ) -> bool:
        """Apply trusted person-review corrections, including finalized episodes.

        A reviewed identity does not change the source evidence or its timing, so
        it must not be rejected by the late detector-result watermark.
        """
        return self._submit(
            "subjects",
            (stream_session_id, dict(aliases), copy.deepcopy(identities or {})),
            timeout_seconds,
        )

    def list_events(
        self,
        *,
        limit: int = 50,
        before: str | None = None,
        stream_session_id: str | None = None,
        event_label: str | None = None,
        subject_id: str | None = None,
        message_id: str | None = None,
        since: str | float | None = None,
        until: str | float | None = None,
    ) -> dict[str, Any]:
        self._await_ready()
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._lock:
            values = [
                item for key, item in self._events.items() if key not in self._aliases
            ]
            if message_id is not None:
                linked = self._links.get(message_id, set())
                if self._db is not None:
                    try:
                        linked = linked | {
                            row[0]
                            for row in self._db.execute(
                                "SELECT event_id FROM links WHERE message_id=?",
                                (message_id,),
                            )
                        }
                    except sqlite3.Error:
                        self._error = "event_storage_error"
                linked = {self._aliases.get(key, key) for key in linked}
                values = [item for item in values if item["event_id"] in linked]
            if stream_session_id:
                values = [
                    v for v in values if v["stream_session_id"] == stream_session_id
                ]
            if event_label:
                values = [v for v in values if v["event_label"] == _label(event_label)]
            if subject_id:
                values = [
                    value
                    for value in values
                    if self._canonical_subject(value["stream_session_id"], subject_id)
                    in value.get("subject_ids", [value["subject_id"]])
                ]
            for value, lower in ((since, True), (until, False)):
                if value is not None:
                    if isinstance(value, (int, float)):
                        values = [
                            v
                            for v in values
                            if (
                                v["start_seconds"] >= value
                                if lower
                                else v["start_seconds"] <= value
                            )
                        ]
                    else:
                        values = [
                            v
                            for v in values
                            if (
                                v["started_at"] >= value
                                if lower
                                else v["started_at"] <= value
                            )
                        ]
            values = self._representative_events(values)
            values.sort(key=lambda v: (v["created_at"], v["event_id"]), reverse=True)
            if before:
                position = next(
                    (i for i, v in enumerate(values) if v["event_id"] == before), None
                )
                values = values[position + 1 :] if position is not None else []
            page = values[:limit]
            return {
                "events": copy.deepcopy(page),
                "count": len(page),
                "next_cursor": page[-1]["event_id"] if len(values) > limit else None,
                "epoch": self._epoch,
                "cursor": self._cursor,
            }

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        self._await_ready()
        with self._lock:
            if event_id in self._aliases:
                canonical = self._aliases[event_id]
                event = copy.deepcopy(self._events.get(canonical))
                if event is not None:
                    event.update(
                        requested_event_id=event_id, canonical_event_id=canonical
                    )
                return event
            return copy.deepcopy(self._events.get(event_id))

    def video_path(self, event_id: str, revision: int | None = None) -> Path | None:
        self._await_ready()
        with self._lock:
            if revision is None:
                event = self._events.get(self._aliases.get(event_id, event_id))
                available = (
                    event.get("evidence", {}).get("available") if event else None
                )
                if not available:
                    return None
                event_id = event["event_id"]
                revision = available["revision"]
            return self._videos.get((event_id, revision))

    def changes_after(
        self, cursor: str | int | None = None, limit: int = 100
    ) -> dict[str, Any]:
        self._await_ready()
        with self._lock:
            reset = None
            position = 0
            if cursor is not None:
                try:
                    if isinstance(cursor, str) and ":" in cursor:
                        epoch, value = cursor.rsplit(":", 1)
                        if epoch != self._epoch:
                            reset = self._reset_reason
                        position = int(value)
                    else:
                        position = int(cursor)
                except (ValueError, TypeError):
                    reset = "invalid_cursor"
                if position > self._cursor:
                    reset = "cursor_ahead"
                elif self._changes and position < self._changes[0]["sequence"] - 1:
                    reset = "history_expired"
            if reset:
                records = []
            else:
                records = [v for v in self._changes if v["sequence"] > position][:limit]
            next_position = records[-1]["sequence"] if records else self._cursor
            return {
                "epoch": self._epoch,
                "cursor": next_position,
                "reset_reason": reset,
                "changes": copy.deepcopy(records),
            }

    def acquire_video(
        self,
        event_id: str,
        revision: int | None = None,
    ) -> tuple[Path, Callable[[], None]] | None:
        """Lease a version before FileResponse opens it, including disconnects."""
        self._await_ready()
        with self._lock:
            path = self.video_path(event_id, revision)
            if path is None:
                return None
            key = next(key for key, value in self._videos.items() if value == path)
            self._leases[key] = self._leases.get(key, 0) + 1
        released = False

        def release() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                remaining = self._leases.get(key, 1) - 1
                if remaining:
                    self._leases[key] = remaining
                else:
                    self._leases.pop(key, None)

        return path, release

    def status(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return copy.deepcopy(self._status_snapshot)
        try:
            snapshot = {
                "state": "error"
                if self._error
                else ("stopped" if self._closed.is_set() else "running"),
                "error": self._error,
                "epoch": self._epoch,
                "queue_depth": self._queue.qsize(),
                "encoding_queue_depth": self._encode_queue.qsize(),
                "event_count": len(
                    self._representative_events(
                        [
                            event
                            for key, event in self._events.items()
                            if key not in self._aliases
                        ]
                    )
                ),
                "buffer_bytes": sum(s.ring.bytes_used for s in self._sessions.values()),
                "pending_evidence": len(self._jobs),
                "ingress_bytes": self._queued_frame_bytes,
                "storage_bytes": self._storage_bytes,
                "reserved_storage_bytes": sum(
                    job.get("reserved_bytes", 0) for job in self._jobs.values()
                ),
                **self._counts,
            }
            self._status_snapshot = snapshot
            return copy.deepcopy(snapshot)
        finally:
            self._lock.release()

    def _await_ready(self) -> None:
        if not self._initialized.wait(5.0):
            raise RuntimeError("event_archive_initializing")

    def flush(self, timeout: float = 10.0) -> bool:
        barrier = threading.Event()
        if not self._submit("barrier", barrier):
            return False
        return barrier.wait(timeout)

    def wait_until_idle(
        self, timeout_seconds: float = 10.0, stream_session_id: str | None = None
    ) -> bool:
        """Wait for accepted observations and all selected evidence to finish."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            with self._lock:
                # Terminal storage failure or shutdown is observable through
                # status(); a lifecycle drain must not wait forever afterward.
                if self._error or (
                    self._closed.is_set() and not self._thread.is_alive()
                ):
                    return True
                relevant = [
                    event
                    for key, event in self._events.items()
                    if (
                        stream_session_id is None
                        or event["stream_session_id"] == stream_session_id
                    )
                ]
                event_ids = {event["event_id"] for event in relevant}
                pending = any(
                    event["evidence"]["status"] in {"collecting", "queued", "encoding"}
                    for event in self._representative_events(
                        [
                            event
                            for event in relevant
                            if event["event_id"] not in self._aliases
                        ]
                    )
                ) or any(
                    stream_session_id is None or key[0] in event_ids
                    for key in self._jobs
                )
                if (
                    not pending
                    and (
                        self._queue.unfinished_tasks
                        if stream_session_id is None
                        else self._queue.pending_for(stream_session_id)
                    )
                    == 0
                ):
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def close(self, timeout: float = 15.0) -> None:
        if not self._closed.is_set():
            self._closed.set()
            try:
                self._queue.put(("close", None), timeout=min(timeout, 1.0))
            except queue.Full:
                pass  # The owner also observes the closed flag after draining ingress.
        self._thread.join(timeout)
        self._encoder_thread.join(max(0.0, timeout / 2))
        if not self._thread.is_alive() and not self._encoder_thread.is_alive():
            self._queue.close()

    def _session(self, session_id: str, source_id: str = "") -> _Session:
        if session_id not in self._sessions:
            recovered = self._recovered_runs.pop(session_id, {})
            windows = copy.deepcopy(recovered.get("windows", {}))
            for window in windows.values():
                # Recovery renews a processing budget, never source timestamps.
                window["admitted"] = self._clock()
            self._sessions[session_id] = _Session(
                source_id or recovered.get("source_id", ""),
                FrameRing(
                    self.settings.buffer_seconds,
                    max(1, self.settings.buffer_max_bytes * 3 // 4),
                ),
                last_activity=self._clock(),
                latest=recovered.get("latest", 0.0),
                ended=recovered.get("ended", False),
                end_reason=recovered.get("end_reason", "stopped"),
                windows=windows,
            )
        return self._sessions[session_id]

    def _open_store(self) -> None:
        root = self.settings.directory
        root.mkdir(parents=True, exist_ok=True)
        (root / "videos").mkdir(exist_ok=True)
        (root / "staging").mkdir(exist_ok=True)
        db = sqlite3.connect(root / "events.sqlite3", check_same_thread=False)
        db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scene_episodes (
                event_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS anchors (
                event_id TEXT PRIMARY KEY, sequence INTEGER, observation_key TEXT);
            CREATE TABLE IF NOT EXISTS observations (
                observation_key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS aliases (
                event_id TEXT PRIMARY KEY, canonical_event_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS subject_aliases (
                session_id TEXT, subject_id TEXT, canonical_subject_id TEXT,
                PRIMARY KEY(session_id,subject_id));
            CREATE TABLE IF NOT EXISTS subject_identities (
                session_id TEXT, subject_id TEXT, payload TEXT NOT NULL,
                PRIMARY KEY(session_id,subject_id));
            CREATE TABLE IF NOT EXISTS links (
                message_id TEXT, event_id TEXT, PRIMARY KEY(message_id,event_id));
            CREATE TABLE IF NOT EXISTS videos (
                event_id TEXT, revision INTEGER, path TEXT, created_at REAL,
                PRIMARY KEY(event_id,revision));
        """)
        self._db = db
        self._subject_identities.update(
            ((sid, subject), json.loads(payload))
            for sid, subject, payload in db.execute("SELECT * FROM subject_identities")
        )
        for session_id in tuple(self._recovered_runs):
            self._session(session_id)
        for key, payload in db.execute("SELECT event_id,payload FROM events"):
            self._persisted_events[key] = payload
            event = json.loads(payload)
            if (
                not event.get("finalized")
                and event.get("stream_session_id") not in self._recoverable_runs
            ):
                event.update(
                    state="ended",
                    status="ended",
                    finalized=True,
                    end_reason="interrupted",
                )
                event["end_seconds"] = (
                    event.get("end_seconds") or event["last_detected_seconds"]
                )
            self._events[key] = event
        if self._recoverable_runs:
            for key, payload in db.execute(
                "SELECT observation_key,payload FROM observations"
            ):
                observation = json.loads(payload)
                if observation.get("session_id") in self._recoverable_runs:
                    observation["admitted"] = self._clock()
                    self._observations[key] = observation
                    self._persisted_observations[key] = payload
                    message = observation.get("message_id")
                    if message:
                        self._links[message].update(
                            row[0] for row in db.execute(
                                "SELECT event_id FROM links WHERE message_id=?",
                                (message,),
                            )
                        )
        for key, payload in db.execute("SELECT event_id,payload FROM scene_episodes"):
            self._scene_events[key] = json.loads(payload)
            self._persisted_scene_events[key] = payload
        for key, sequence, anchor in db.execute("SELECT * FROM anchors"):
            self._anchors[key] = (sequence, anchor)
            self._persisted_anchors[key] = (sequence, anchor)
            self._anchor_sequence = max(self._anchor_sequence, sequence)
        self._aliases.update(db.execute("SELECT * FROM aliases"))
        self._persisted_aliases = dict(self._aliases)
        self._subject_aliases.update(
            ((session_id, subject_id), canonical)
            for session_id, subject_id, canonical in db.execute(
                "SELECT * FROM subject_aliases"
            )
        )
        # Historical message lookups use the indexed table directly; loading
        # every past observation link would defeat the live-memory bound.
        for event_id, revision, path in db.execute(
            "SELECT event_id,revision,path FROM videos"
        ):
            resolved = root / path
            if resolved.is_file() and resolved.resolve().is_relative_to(root.resolve()):
                self._videos[event_id, revision] = resolved
        for event in self._events.values():
            evidence = event.get("evidence", {})
            available = evidence.get("available")
            if (
                available
                and (event["event_id"], available["revision"]) not in self._videos
            ):
                evidence.update(
                    status="failed",
                    failure_reason="stored_video_missing",
                    available=None,
                )
        temporary_videos = [
            *(root / "videos").glob(".event-*.tmp.mp4"),
            *(root / "videos").glob(".e-*.tmp.mp4"),
        ]
        for temporary in temporary_videos:
            if temporary.resolve().is_relative_to((root / "videos").resolve()):
                temporary.unlink(missing_ok=True)
        for stage in (root / "staging").glob("event-*"):
            if (
                stage.is_dir()
                and not (stage / "manifest.json").is_file()
                and (stage.resolve().is_relative_to((root / "staging").resolve()))
            ):
                shutil.rmtree(stage)
        for manifest in sorted((root / "staging").glob("*/manifest.json")):
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                event_id, revision = metadata["event_id"], int(metadata["revision"])
                if event_id not in self._events:
                    continue
                if (event_id, revision) in self._videos:
                    shutil.rmtree(manifest.parent)
                    continue
                destination = root / "videos" / f"{event_id}-r{revision}.mp4"
                self._encode_queue.put_nowait((manifest, destination))
                self._jobs[event_id, revision] = metadata
            except (OSError, ValueError, KeyError, queue.Full):
                continue
        for event in self._events.values():
            evidence = event.get("evidence", {})
            if (
                evidence.get("status") in {"collecting", "queued", "encoding"}
                and (event["event_id"], evidence.get("revision", 0)) not in self._jobs
                and event.get("stream_session_id") not in self._recoverable_runs
            ):
                evidence.update(status="failed", failure_reason="interrupted")
        self._persist()

    def _run(self) -> None:
        try:
            with self._lock:
                self._open_store()
        except Exception:
            with self._lock:
                self._error = "event_storage_unavailable"
                self._counts["storage_failures"] += 1
        finally:
            self._initialized.set()
        while True:
            try:
                kind, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                kind, payload = "tick", None
            except (OSError, ValueError, RuntimeError, sqlite3.Error):
                with self._lock:
                    self._counts["storage_failures"] += 1
                    self._error = "archive_ingress_read_error"
                break
            if kind == "tick" and self._closed.is_set():
                kind, payload = "close_drained", None
            try:
                with self._lock:
                    if kind in {"close", "close_drained"}:
                        for session in self._sessions.values():
                            session.ended = True
                            session.end_reason = "shutdown"
                        self._rebuild()
                        self._finish_scene_events()
                        self._schedule_evidence()
                        break
                    if kind == "barrier":
                        payload.set()
                    elif kind == "start":
                        self._session(*payload)
                    elif kind == "frame":
                        session_id, frame = payload
                        with self._ingest_lock:
                            self._queued_frame_bytes = max(
                                0, self._queued_frame_bytes - len(frame.jpeg)
                            )
                        session = self._session(session_id)
                        if not session.ended:
                            session.ring.append(frame)
                            for window_id, pinned in session.pins.items():
                                window = session.windows[window_id]
                                start = (
                                    window["timestamp"] - self.settings.pre_roll_seconds
                                )
                                end = start + self.settings.clip_seconds
                                if (
                                    start <= frame.timestamp_seconds
                                    and (
                                        frame.timestamp_seconds < end
                                        or (
                                            pinned
                                            and pinned[-1].timestamp_seconds < end
                                        )
                                    )
                                    and (
                                        not pinned
                                        or frame.timestamp_seconds
                                        > pinned[-1].timestamp_seconds
                                    )
                                ):
                                    pinned.append(frame)
                            self._trim_buffers()
                            session.latest = max(
                                session.latest, frame.timestamp_seconds
                            )
                            session.last_activity = self._clock()
                            self._rebuild(session_id)
                            self._schedule_evidence(session_id)
                    elif kind == "window":
                        self._window(payload)
                    elif kind == "result":
                        self._result(*payload)
                    elif kind == "subjects":
                        self._reconcile_subjects(*payload)
                    elif kind == "scene":
                        self._scene_result(payload)
                    elif kind == "end":
                        session_id, reason = payload
                        session = self._session(session_id)
                        session.ended, session.end_reason = True, reason
                        self._rebuild(session_id)
                        self._finish_scene_events(session_id)
                        self._schedule_evidence(session_id)
                    elif kind == "encoded":
                        self._encoded(*payload)
                    if self._clock() - self._last_maintenance >= 1.0:
                        self._last_maintenance = self._clock()
                        self._expire_windows()
                        self._rebuild()
                        self._schedule_evidence()
                        self._retention()
                        self._trim_buffers()
                        self._compact_observations()
            except Exception:
                if kind not in {"tick", "close_drained"}:
                    self._queue.task_failed()
                with self._lock:
                    self._counts["storage_failures"] += 1
                    self._error = "event_storage_error"
                    failures = []
                    for event in self._events.values():
                        if event["evidence"]["status"] in {
                            "collecting",
                            "queued",
                            "encoding",
                        }:
                            event["evidence"].update(
                                status="failed",
                                failure_reason="evidence_storage_unavailable",
                            )
                            self._touch(event)
                            failures.append(event)
                    if failures:
                        self._change(failures)
            finally:
                if kind not in {"tick", "close_drained"}:
                    try:
                        self._queue.task_done()
                    except (OSError, sqlite3.Error):
                        # A failed ACK leaves the fact pending for restart.
                        with self._lock:
                            self._counts["storage_failures"] += 1
                            self._error = "archive_ingress_ack_error"
                        break
        # Pending durable jobs are safely resumed next start; allow the current
        # encoder to finish, but do not make shutdown wait for the entire queue.
        try:
            self._encode_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._db is not None:
            self._db.close()

    def _window(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        session = self._session(payload["session_id"])
        previous = session.windows.get(payload["window_id"])
        if previous is not None:
            payload["admitted"] = previous["admitted"]
            if not payload["labels"]:
                payload["labels"] = previous["labels"]
        else:
            # This is L3's revision/processing clock. Waiting durably in the
            # ingress must not make an already accepted model result "late".
            payload["source_admitted_monotonic"] = payload["admitted"]
            payload["admitted"] = self._clock()
        session.windows[payload["window_id"]] = payload
        if payload["timestamp"] > session.latest:
            session.last_activity = self._clock()
        session.latest = max(session.latest, payload["timestamp"])
        if previous is None and payload["outcome"] in {"admitted", "submitted"}:
            start = payload["timestamp"] - self.settings.pre_roll_seconds
            end = start + self.settings.clip_seconds
            session.pins[payload["window_id"]] = session.ring.window_frames(start, end)
        if payload["outcome"] in {"failed", "dropped", "cancelled"}:
            for label in payload["labels"]:
                key = f"window:{payload['session_id']}:{payload['window_id']}:{label}"
                self._observations[key] = {
                    "key": key,
                    "session_id": payload["session_id"],
                    "label": label,
                    "subject_id": "*",
                    "timestamp": payload["timestamp"],
                    "detected": None,
                    "admitted": payload["admitted"],
                    "message_id": "",
                    "object_id": "",
                }
        self._rebuild(payload["session_id"])
        if payload["outcome"] in {"completed", "failed", "dropped", "cancelled"}:
            session.pins.pop(payload["window_id"], None)

    def _expire_windows(self) -> None:
        for session_id, session in self._sessions.items():
            for window_id, window in session.windows.items():
                if window["outcome"] in {"admitted", "submitted"} and (
                    self._clock() - window["admitted"] > self.settings.revision_seconds
                ):
                    window["outcome"] = "failed"
                    session.pins.pop(window_id, None)
                    for label in window["labels"]:
                        key = f"window:{session_id}:{window_id}:{label}"
                        self._observations[key] = {
                            "key": key,
                            "session_id": session_id,
                            "label": label,
                            "subject_id": "*",
                            "timestamp": window["timestamp"],
                            "detected": None,
                            "admitted": window["admitted"],
                            "message_id": "",
                            "object_id": "",
                        }

    def _trim_buffers(self) -> None:
        for session_id, session in self._sessions.items():
            events = [
                v for v in self._events.values() if v["stream_session_id"] == session_id
            ]
            if (
                session.ended
                and all(v["finalized"] for v in events)
                and not any(v["evidence"]["status"] == "collecting" for v in events)
            ):
                session.ring.frames.clear()
                session.ring.bytes_used = 0
                session.targets.clear()
                session.windows.clear()
                session.pins.clear()
        total = sum(session.ring.bytes_used for session in self._sessions.values())
        budget = max(1, self.settings.buffer_max_bytes * 3 // 4)
        for session in self._sessions.values():
            while total > budget and session.ring.frames:
                size = len(session.ring.frames.popleft().jpeg)
                session.ring.bytes_used -= size
                total -= size
        pinned_frames = {
            id(frame): frame
            for session in self._sessions.values()
            for pinned in session.pins.values()
            for frame in pinned
        }
        resident = {
            id(frame): frame
            for session in self._sessions.values()
            for frame in session.ring.frames
        }
        resident.update(pinned_frames)
        used = sum(len(frame.jpeg) for frame in resident.values())
        if used > budget:
            for session in self._sessions.values():
                while session.ring.frames and used > budget:
                    frame = session.ring.frames.popleft()
                    session.ring.bytes_used -= len(frame.jpeg)
                    if id(frame) not in pinned_frames:
                        used -= len(frame.jpeg)
            if used > budget:
                # Too many simultaneous windows cannot defeat the global bound.
                # Drop oldest pins explicitly; their eventual evidence reports
                # source_frames_unavailable if the ordinary ring also expired.
                for session in self._sessions.values():
                    for window_id in list(session.pins):
                        session.pins.pop(window_id)
                        resident = {
                            id(frame): frame
                            for value in self._sessions.values()
                            for pinned in value.pins.values()
                            for frame in pinned
                        }
                        used = sum(len(frame.jpeg) for frame in resident.values())
                        if used <= budget:
                            return

    def _target_subject(
        self,
        session: _Session,
        session_id: str,
        label: str,
        item: dict[str, Any],
        timestamp: float,
        used: set[str],
    ) -> tuple[str, str]:
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            return "scene", f"scene:{session_id}:{label}"

        def overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
            try:
                intersection = max(
                    0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"])
                ) * max(
                    0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"])
                )
                area = (left["x2"] - left["x1"]) * (left["y2"] - left["y1"])
                area += (right["x2"] - right["x1"]) * (right["y2"] - right["y1"])
                return intersection / max(area - intersection, 1e-12)
            except (KeyError, TypeError):
                return 0.0

        candidates = [
            key
            for key, value in session.targets.items()
            if value["label"] == label
            and key not in used
            and abs(timestamp - value["timestamp"]) <= 3.0
            and overlap(bbox, value["bbox"]) >= 0.3
        ]
        subject = candidates[0] if len(candidates) == 1 else f"target-{uuid4().hex}"
        existing = session.targets.get(subject)
        if existing is None or timestamp >= existing["timestamp"]:
            session.targets[subject] = {
                "label": label,
                "bbox": bbox,
                "timestamp": timestamp,
            }
        used.add(subject)
        return "target", subject

    def _compact_observations(self) -> None:
        """Keep the revisable interval plus compact proven episode prefixes."""
        removed: set[str] = set()
        anchor_keys = {value[1] for value in self._anchors.values()}
        for session_id, session in self._sessions.items():
            pending = [
                window["timestamp"]
                for window in session.windows.values()
                if window["outcome"] in {"admitted", "submitted"}
                and self._clock() - window["admitted"] <= self.settings.revision_seconds
            ]
            cutoff = (
                min(pending)
                if pending
                else session.latest - self.settings.unknown_timeout_seconds
            )
            revisable = [
                window["timestamp"]
                for window in session.windows.values()
                if self._clock() - window["admitted"] <= self.settings.revision_seconds
                and any(
                    not event["finalized"]
                    and event["stream_session_id"] == session_id
                    and event["start_seconds"]
                    <= window["timestamp"]
                    <= event["last_detected_seconds"]
                    for event in self._events.values()
                )
            ]
            if revisable:
                cutoff = min(cutoff, min(revisable))
            if not session.windows:
                mutable = [
                    point["timestamp"]
                    for point in self._observations.values()
                    if point["session_id"] == session_id
                    and self._clock() - point["admitted"]
                    <= self.settings.revision_seconds
                ]
                if mutable:
                    cutoff = min(cutoff, min(mutable))
            events = [
                event
                for key, event in self._events.items()
                if key not in self._aliases and event["stream_session_id"] == session_id
            ]
            for event in events:
                old = sorted(
                    (
                        point
                        for point in self._observations.values()
                        if point["session_id"] == session_id
                        and point["subject_id"] == event["subject_id"]
                        and point["label"] == event["event_label"]
                        and point["detected"] is True
                        and event["start_seconds"]
                        <= point["timestamp"]
                        <= event["last_detected_seconds"]
                        and point["timestamp"] < cutoff
                    ),
                    key=lambda p: p["timestamp"],
                )
                if len(old) < 3:
                    continue
                keep = [
                    point
                    for point in old
                    if point["key"] in anchor_keys or point is old[-1]
                ]
                if old[0] not in keep:
                    keep.insert(0, old[0])
                keep.sort(key=lambda p: p["timestamp"])
                keep[0]["weight"] = sum(point.get("weight", 1) for point in old) - sum(
                    point.get("weight", 1) for point in keep[1:]
                )
                for left, right in pairwise(keep):
                    right["continues_from"] = left["key"]
                if (
                    event["finalized"]
                    and old[-1]["timestamp"] == event["last_detected_seconds"]
                ):
                    keep[-1]["closed_reason"] = event["end_reason"]
                    keep[-1]["closed_end"] = event["end_seconds"]
                retained = {point["key"] for point in keep}
                removed.update(
                    point["key"] for point in old if point["key"] not in retained
                )
            for point in self._observations.values():
                if (
                    point["session_id"] != session_id
                    or point["subject_id"] != "*"
                    or point["timestamp"] >= cutoff
                ):
                    continue
                preserves_boundary = any(
                    event["event_label"] == point["label"]
                    and event["end_reason"] == "negative"
                    and event["end_seconds"]
                    <= point["timestamp"]
                    <= event["end_seconds"] + self.settings.negative_gap_seconds
                    for event in events
                )
                if not preserves_boundary:
                    removed.add(point["key"])
            for window_id, window in list(session.windows.items()):
                if window["timestamp"] < cutoff and window["outcome"] not in {
                    "admitted",
                    "submitted",
                }:
                    session.windows.pop(window_id)
                    session.pins.pop(window_id, None)
            session.targets = {
                key: target
                for key, target in session.targets.items()
                if target["timestamp"] >= session.latest - 3.0
            }
        if not removed:
            return
        for key in removed:
            self._observations.pop(key, None)
            self._persisted_observations.pop(key, None)
        retained_messages = {
            point["message_id"] for point in self._observations.values()
        }
        for message in list(self._links):
            if message not in retained_messages:
                self._links.pop(message, None)
                self._persisted_links.pop(message, None)
        if self._db is not None:
            with self._db:
                self._db.executemany(
                    "DELETE FROM observations WHERE observation_key=?",
                    ((key,) for key in removed),
                )
        self._persist()

    def _result(
        self, result: dict[str, Any], actors: dict[str, Any], window_id: Any
    ) -> None:
        session_id = result["stream_session_id"]
        timestamp = float(result["frame"]["timestamp_seconds"])
        session = self._session(session_id, result.get("source_id", ""))
        explicit_window = (
            session.windows.get(str(window_id)) if window_id is not None else None
        )
        window = explicit_window
        if window is None:
            window = next(
                (
                    v
                    for v in session.windows.values()
                    if (abs(v["timestamp"] - timestamp) < 1e-5)
                ),
                None,
            )
        admitted = window["admitted"] if window else self._clock()
        previous_points = [
            point
            for point in self._observations.values()
            if point["message_id"] == result["message_id"]
        ]
        if previous_points:
            admitted = min(point["admitted"] for point in previous_points)
            if any(
                self._events.get(event_id, {}).get("finalized")
                for event_id in self._links.get(result["message_id"], set())
            ):
                self._counts["late_results"] += 1
                return
        elif (
            self._db is not None
            and self._db.execute(
                "SELECT 1 FROM links WHERE message_id=? LIMIT 1",
                (result["message_id"],),
            ).fetchone()
        ):
            # Its source observation was already compacted beyond the revision
            # watermark; a late person-association callback must not reopen it.
            self._counts["late_results"] += 1
            return
        mutable_windows = [
            w["timestamp"]
            for w in session.windows.values()
            if self._clock() - w["admitted"] <= self.settings.revision_seconds
        ]
        watermark = (
            min(mutable_windows)
            if mutable_windows
            else session.latest - self.settings.revision_seconds
        )
        if (
            (self._clock() - admitted > self.settings.revision_seconds)
            or timestamp < watermark
            or (session.ended and window is None)
        ):
            self._counts["late_results"] += 1
            return
        if window:
            window["seen_result"] = True
            prefix = f"window:{session_id}:{window['window_id']}:"
            for key in list(self._observations):
                if key.startswith(prefix):
                    del self._observations[key]
        if timestamp >= session.latest:
            session.last_activity = self._clock()
        session.latest = max(session.latest, timestamp)
        objects = {item["object_id"]: item for item in result.get("objects", [])}
        message_id = result["message_id"]
        # A revision replaces this source observation, including positives
        # withdrawn after identity resolution.
        for point in previous_points:
            self._observations.pop(point["key"], None)
            self._persisted_observations.pop(point["key"], None)
        if previous_points and self._db is not None:
            with self._db:
                self._db.executemany(
                    "DELETE FROM observations WHERE observation_key=?",
                    ((point["key"],) for point in previous_points),
                )
        for judgment in result.get("judgments", []):
            label = _label(judgment["event_label"])
            if judgment["detected"]:
                used_targets: set[str] = set()
                unknown_key = f"{message_id}:unmentioned:{label}"
                self._observations[unknown_key] = {
                    "key": unknown_key,
                    "message_id": message_id,
                    "object_id": "",
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "admitted": admitted,
                    "label": label,
                    "subject_id": "*",
                    "detected": None,
                }
                emitted_subjects: set[str] = set()
                for object_id in judgment.get("object_ids", []):
                    item = objects.get(object_id, {})
                    attributes = {
                        **item.get("attributes", {}),
                        **actors.get(object_id, {}),
                    }
                    if (
                        label == "stranger"
                        and (
                            "status" in attributes
                            or "face_observation_state" in attributes
                        )
                        and stranger_evidence(attributes) is not True
                    ):
                        continue
                    subject_type = attributes.get("subject_type", "person")
                    subject_id = attributes.get("subject_id")
                    subject_ids = sorted(
                        set(
                            self._canonical_subject(session_id, str(value))
                            for value in attributes.get("subject_ids", [])
                            if value
                        )
                    )
                    if subject_id:
                        subject_id = self._canonical_subject(
                            session_id, str(subject_id)
                        )
                    if label == "fight" and len(subject_ids) > 1:
                        subject_id = "group:" + ":".join(subject_ids)
                        subject_type = "group"
                    if not subject_id:
                        subject_type, subject_id = self._target_subject(
                            session,
                            session_id,
                            label,
                            item,
                            timestamp,
                            used_targets,
                        )
                    if not subject_ids:
                        subject_ids = [str(subject_id)]
                    if str(subject_id) in emitted_subjects:
                        continue
                    emitted_subjects.add(str(subject_id))
                    key = f"{message_id}:{object_id}:{label}"
                    self._observations[key] = {
                        "key": key,
                        "message_id": message_id,
                        "object_id": object_id,
                        "session_id": session_id,
                        "source_id": result.get("source_id", ""),
                        "timestamp": timestamp,
                        "admitted": admitted,
                        "observed_at": result["frame"].get(
                            "observed_at", result.get("emitted_at", _utc())
                        ),
                        "label": label,
                        "subject_id": str(subject_id),
                        "subject_ids": subject_ids,
                        "subject_type": subject_type,
                        "detected": True,
                        "attributes": attributes,
                        "bbox": item.get("bbox"),
                        "confidence": judgment.get("confidence", 0.0),
                        "description": judgment.get("description", ""),
                    }
            else:
                key = f"{message_id}:negative:{label}"
                self._observations[key] = {
                    "key": key,
                    "message_id": message_id,
                    "object_id": "",
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "admitted": admitted,
                    "label": label,
                    "subject_id": "*",
                    "detected": False,
                }
        for value in result.get("missing_event_labels", []):
            label = _label(value)
            # A partial branch response can list the other branch as missing.
            # Its independently registered request owns that uncertainty; do
            # not let it erase a real negative from the other branch. Only an
            # explicit known window establishes this scope. Timestamp fallback
            # is ambiguous when both branches sampled the same source frame.
            if (
                explicit_window is not None
                and explicit_window["labels"]
                and label not in explicit_window["labels"]
            ):
                continue
            key = f"{message_id}:unknown:{label}"
            self._observations[key] = {
                "key": key,
                "message_id": message_id,
                "object_id": "",
                "session_id": session_id,
                "timestamp": timestamp,
                "admitted": admitted,
                "label": label,
                "subject_id": "*",
                "detected": None,
            }
        identities = {}
        for item in objects.values():
            attributes = item.get("attributes", {})
            subject = attributes.get("subject_id")
            if (
                subject
                and attributes.get("status") == "known"
                and attributes.get("person_id")
                and (
                    attributes.get("identity_source") == "face_match"
                    or (
                        attributes.get("identity_source") == "track_history"
                        and isinstance(
                            attributes.get("identity_confirmed_at"), (int, float)
                        )
                        and 0 <= attributes["identity_confirmed_at"] <= timestamp
                    )
                )
                and attributes.get("face_observation_state") == "observed"
            ):
                identities[subject] = {
                    key: attributes[key]
                    for key in ("status", "person_id", "name", "identity_source")
                    if key in attributes
                }
                # Resolution through this source time, including continuity
                # proven by the registry, not necessarily a new gallery match.
                identities[subject]["confirmed_at"] = timestamp
        if identities:
            self._reconcile_subjects(session_id, {}, identities)
        else:
            self._rebuild(session_id)
            self._schedule_evidence(session_id)
            self._persist()

    @staticmethod
    def _scene_event_id(session_id: str, event_id: str) -> str:
        # Preserve the public event/v1 UUID shape and stable source-episode
        # identity across restarts without coupling it to mutable participants.
        key = json.dumps(["goodvision-scene", session_id, event_id])
        return f"event-{uuid5(NAMESPACE_URL, key).hex}"

    def _scene_result(self, payload: dict[str, Any]) -> None:
        sid = payload["stream_session_id"]
        session = self._session(sid, payload["source_id"])
        current = {item["event_id"]: item for item in payload["events"]}
        records = {
            item["event_id"]: item
            for item in payload["transitions"]
            if item["event_id"] not in current
        }
        records.update(current)
        changed = []
        for source_id, record in records.items():
            event_id = self._scene_event_id(sid, source_id)
            previous = self._events.get(event_id)
            first = record.get("first_evidence_timestamp_seconds")
            if first is None:
                if previous is None:
                    continue
                first = previous["start_seconds"]
            last = record.get(
                "last_evidence_timestamp_seconds",
                record.get("timestamp_seconds", first),
            )
            subjects = sorted(
                {
                    self._canonical_subject(sid, subject)
                    for subject in record.get("subject_ids", ())
                }
            )
            if not subjects:
                continue
            phase = record["phase"]
            dismissed = (
                phase == "retracted" or record.get("review_outcome") == "unconfirmed"
            )
            finalized = session.ended or bool(previous and previous["finalized"])
            ended = phase == "ended" or dismissed or finalized
            evidence = (
                copy.deepcopy(previous["evidence"])
                if previous
                else {
                    "revision": 0,
                    "status": "collecting",
                    "available": None,
                }
            )
            requested = {
                "start_seconds": first - self.settings.pre_roll_seconds,
                "end_seconds": first
                + self.settings.clip_seconds
                - self.settings.pre_roll_seconds,
            }
            if dismissed:
                # Invalidate in-flight encoding too: it must not revive a
                # withdrawn positive when its completion arrives later.
                if evidence.get("failure_reason") != "event_" + (
                    "retracted" if phase == "retracted" else "unconfirmed"
                ):
                    evidence["revision"] += 1
                evidence.update(
                    status="expired",
                    available=None,
                    requested_window=requested,
                    failure_reason="event_"
                    + ("retracted" if phase == "retracted" else "unconfirmed"),
                )
            elif evidence.get("requested_window") != requested or evidence.get(
                "failure_reason"
            ) in {"event_retracted", "event_unconfirmed"}:
                evidence.update(
                    revision=evidence["revision"] + 1,
                    status="collecting",
                    requested_window=requested,
                    failure_reason=None,
                )
            positive = record.get("last_positive_seconds")
            if positive is None:
                positive = (
                    last
                    if phase in {"suspected", "confirmed"}
                    else previous["last_detected_seconds"]
                    if previous
                    else first
                )
            now = _utc()
            event = {
                "type": "event",
                "schema_version": "1",
                "event_id": event_id,
                "revision": previous["revision"] if previous else 0,
                "stream_session_id": sid,
                "source_id": session.source_id,
                "subject_id": subjects[0]
                if len(subjects) == 1
                else "group:" + ":".join(subjects),
                "subject_ids": subjects,
                "subject_type": "person" if len(subjects) == 1 else "group",
                "event_label": record["event_label"],
                "state": "ended" if ended else "active",
                "status": "ended" if ended else "active",
                "finalized": finalized,
                "start_seconds": first,
                "last_detected_seconds": max(first, positive),
                "end_seconds": (
                    previous["end_seconds"]
                    if previous
                    and previous["finalized"]
                    and phase != "ended"
                    and not dismissed
                    else max(first, last)
                    if ended
                    else None
                ),
                "end_reason": (
                    "retracted"
                    if phase == "retracted"
                    else "unconfirmed"
                    if dismissed
                    else "negative"
                    if phase == "ended"
                    else previous["end_reason"]
                    if previous and previous["finalized"]
                    else session.end_reason
                    if session.ended
                    else None
                ),
                "started_at": previous["started_at"] if previous else now,
                "created_at": previous["created_at"] if previous else now,
                "updated_at": previous["updated_at"] if previous else now,
                "observation_count": max(
                    1,
                    record.get("positive_observations", 0),
                    previous["observation_count"] if previous else 0,
                ),
                "confidence": record["confidence"]
                if record.get("confidence") is not None
                else previous["confidence"] if previous else 0.0,
                "description": previous["description"] if previous else "",
                "identity_state": "group"
                if len(subjects) > 1
                else previous["identity_state"]
                if previous
                else "unknown",
                "person_id": previous["person_id"] if previous else None,
                "person_name": previous["person_name"] if previous else None,
                "evidence": evidence,
            }
            self._scene_events[event_id] = {"scene_event_id": source_id, "phase": phase}
            self._correct_event_identity(event)
            if previous != event:
                self._touch(event)
                self._events[event_id] = event
                changed.append(event)
        self._persist()
        if changed:
            self._change(changed)
        self._schedule_evidence(sid)

    def _finish_scene_events(self, session_id: str | None = None) -> None:
        changed = []
        for event_id in self._scene_events:
            event = self._events.get(event_id)
            if event is None or event["finalized"]:
                continue
            sid = event["stream_session_id"]
            if session_id is not None and sid != session_id:
                continue
            session = self._sessions.get(sid)
            if session is None or not session.ended:
                continue
            if event["state"] != "ended":
                event.update(
                    state="ended",
                    status="ended",
                    end_reason=session.end_reason,
                    end_seconds=max(event["last_detected_seconds"], session.latest),
                )
            event["finalized"] = True
            self._touch(event)
            changed.append(event)
        if changed:
            self._persist()
            self._change(changed)

    def _segments(
        self, points: list[dict[str, Any]], latest: float
    ) -> list[dict[str, Any]]:
        # A positive wins over another branch's negative at the same source time.
        by_time: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            by_time[point["timestamp"]].append(point)
        segments: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        negative_start: float | None = None
        for timestamp, group in sorted(by_time.items()):
            positives = [p for p in group if p["detected"] is True]
            continuation = bool(
                current
                and any(
                    point.get("continues_from")
                    in {p["key"] for p in current["positives"]}
                    for point in positives
                )
            )
            if (
                current
                and not current["reason"]
                and not continuation
                and (
                    timestamp - current["positives"][-1]["timestamp"]
                    >= self.settings.unknown_timeout_seconds
                )
            ):
                current.update(
                    reason="timeout",
                    end=current["positives"][-1]["timestamp"]
                    + self.settings.unknown_timeout_seconds,
                    decision=current["positives"][-1]["timestamp"]
                    + self.settings.unknown_timeout_seconds,
                )
            if positives:
                if current is None or current["reason"]:
                    current = {
                        "positives": [],
                        "reason": None,
                        "end": None,
                        "decision": None,
                    }
                    segments.append(current)
                current["positives"].extend(sorted(positives, key=lambda p: p["key"]))
                negative_start = None
                closed = next(
                    (point for point in positives if point.get("closed_reason")), None
                )
                if closed:
                    current.update(
                        reason=closed["closed_reason"],
                        end=closed["closed_end"],
                        decision=closed["closed_end"],
                    )
            elif current and not current["reason"]:
                if any(p["detected"] is None for p in group):
                    negative_start = None
                elif negative_start is None:
                    negative_start = timestamp
                elif timestamp - negative_start >= self.settings.negative_gap_seconds:
                    current.update(
                        reason="negative", end=negative_start, decision=timestamp
                    )
        if (
            current
            and not current["reason"]
            and (
                latest - current["positives"][-1]["timestamp"]
                >= self.settings.unknown_timeout_seconds
            )
        ):
            current.update(
                reason="timeout",
                end=current["positives"][-1]["timestamp"]
                + self.settings.unknown_timeout_seconds,
                decision=current["positives"][-1]["timestamp"]
                + self.settings.unknown_timeout_seconds,
            )
        return segments

    def _rebuild(self, session_id: str | None = None) -> None:
        changed: list[dict[str, Any]] = []
        dirty = False
        aliases_before = dict(self._aliases)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        global_points: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for point in self._observations.values():
            sid = point["session_id"]
            if session_id is not None and sid != session_id:
                continue
            if point["subject_id"] == "*":
                global_points[sid, point["label"]].append(point)
            else:
                groups[sid, point["subject_id"], point["label"]].append(point)
        affected_messages = {
            p["message_id"]
            for p in self._observations.values()
            if session_id is None or p["session_id"] == session_id
        }
        previous_event_ids = {
            event_id
            for message_id in affected_messages
            for event_id in self._links.get(message_id, ())
        }
        rebuilt_event_ids = set()
        for message_id in affected_messages:
            self._links.pop(message_id, None)
        for (sid, subject, label), positives in groups.items():
            session = self._sessions[sid]
            latest = session.latest
            if not session.ended:
                latest += max(0.0, self._clock() - session.last_activity)
            points = positives + global_points[sid, label]
            identity = self._subject_identities.get((sid, subject))
            if label == "stranger" and identity:
                # Keep corrected past observations separate from new unmatched
                # faces. A gallery match is not permission to ignore the future.
                boundary = identity["confirmed_at"]
                segments = self._segments(
                    [p for p in points if p["timestamp"] <= boundary], latest
                ) + self._segments(
                    [p for p in points if p["timestamp"] > boundary], latest
                )
            else:
                segments = self._segments(points, latest)
            for segment in segments:
                members = segment["positives"]
                keys = {p["key"] for p in members}
                anchored = sorted(
                    (order, event_id)
                    for event_id, (order, key) in self._anchors.items()
                    if key in keys
                )
                if not anchored:
                    event_id = f"event-{uuid4().hex}"
                    self._anchor_sequence += 1
                    self._anchors[event_id] = (self._anchor_sequence, members[0]["key"])
                    anchored = [(self._anchor_sequence, event_id)]
                elif label == "stranger":
                    # A reviewed merge can join an unavailable early episode to
                    # an already playable later one. Keep the playable owner.
                    anchored.sort(
                        key=lambda entry: (
                            self._evidence_rank(self._events[entry[1]])
                            if self._events.get(entry[1], {})
                            .get("evidence", {})
                            .get("available")
                            else (5, entry[0], entry[1])
                        )
                    )
                event_id = anchored[0][1]
                self._aliases.pop(event_id, None)
                for _, alias in anchored[1:]:
                    self._aliases[alias] = event_id
                first, last = members[0], members[-1]
                previous = self._events.get(event_id)
                if previous and previous.get("finalized"):
                    for point in members:
                        self._links[point["message_id"]].add(event_id)
                    continue
                ended = bool(segment["reason"] or session.ended)
                revision_until = (
                    max(p["admitted"] for p in members) + self.settings.revision_seconds
                )
                relevant_windows = [
                    w
                    for w in session.windows.values()
                    if w["timestamp"]
                    <= (
                        segment["decision"]
                        if segment["decision"] is not None
                        else latest
                    )
                    and (not w["labels"] or label in w["labels"])
                ]
                if relevant_windows:
                    finalized = ended and all(
                        w["outcome"] in {"completed", "failed", "dropped", "cancelled"}
                        or self._clock() - w["admitted"]
                        > self.settings.revision_seconds
                        for w in relevant_windows
                    )
                else:
                    finalized = ended and self._clock() > revision_until
                reason = segment["reason"] or (
                    session.end_reason if session.ended else None
                )
                attributes = last.get("attributes", {})
                identity = attributes.get("subject_identity", {})
                if not isinstance(identity, dict):
                    identity = {}
                evidence = (
                    copy.deepcopy(previous["evidence"])
                    if previous
                    else {
                        "revision": 0,
                        "status": "collecting",
                        "available": None,
                    }
                )
                start = first["timestamp"] - self.settings.pre_roll_seconds
                requested = {
                    "start_seconds": start,
                    "end_seconds": first["timestamp"]
                    + self.settings.clip_seconds
                    - self.settings.pre_roll_seconds,
                }
                if (
                    label == "stranger"
                    and evidence.get("requested_window")
                    and (
                        evidence.get("available")
                        or evidence.get("status") == "encoding"
                    )
                ):
                    # Identity evidence needs one usable clip; an earlier
                    # observation does not require another immutable revision.
                    requested = evidence["requested_window"]
                if evidence.get("requested_window") != requested:
                    evidence.update(
                        revision=evidence["revision"] + 1,
                        status="collecting",
                        requested_window=requested,
                        failure_reason=None,
                    )
                event = {
                    "type": "event",
                    "schema_version": "1",
                    "event_id": event_id,
                    "revision": previous["revision"] if previous else 0,
                    "stream_session_id": sid,
                    "source_id": first.get("source_id", session.source_id),
                    "subject_id": subject,
                    "subject_type": first.get("subject_type", "person"),
                    "subject_ids": sorted(
                        {
                            value
                            for p in members
                            for value in p.get("subject_ids", [subject])
                        }
                    ),
                    "event_label": label,
                    "state": "ended" if ended else "active",
                    "status": "ended" if ended else "active",
                    "finalized": finalized,
                    "start_seconds": first["timestamp"],
                    "last_detected_seconds": last["timestamp"],
                    "end_seconds": segment["end"]
                    if segment["end"] is not None
                    else (session.latest if session.ended else None),
                    "end_reason": reason,
                    "started_at": first.get("observed_at", _utc()),
                    "created_at": previous["created_at"] if previous else _utc(),
                    "updated_at": previous["updated_at"] if previous else _utc(),
                    "observation_count": sum(p.get("weight", 1) for p in members),
                    "confidence": max(p.get("confidence", 0.0) for p in members),
                    "description": last.get("description", ""),
                    "identity_state": attributes.get(
                        "identity_state", identity.get("status", "unknown")
                    ),
                    "person_id": attributes.get("person_id", identity.get("person_id")),
                    "person_name": attributes.get(
                        "person_name",
                        attributes.get(
                            "name", identity.get("name", attributes.get("identity"))
                        ),
                    ),
                    "evidence": evidence,
                }
                if event["subject_type"] == "group":
                    event.update(
                        identity_state="group", person_id=None, person_name=None
                    )
                self._correct_event_identity(event)
                if previous != event:
                    event["revision"] += 1
                    event["updated_at"] = _utc()
                    self._events[event_id] = event
                    dirty = True
                    significant = (
                        "state",
                        "finalized",
                        "start_seconds",
                        "end_seconds",
                        "end_reason",
                        "subject_id",
                        "subject_ids",
                        "identity_state",
                        "person_id",
                        "person_name",
                        "evidence",
                    )
                    if (
                        previous is None
                        or event_id in aliases_before
                        or any(
                            previous.get(key) != event.get(key) for key in significant
                        )
                    ):
                        changed.append(event)
                for point in members:
                    self._links[point["message_id"]].add(event_id)
                rebuilt_event_ids.add(event_id)
        # Identity/association revisions may remove an episode's last positive.
        # Retaining only the old materialized event would resurrect that evidence
        # even though its source observations were correctly replaced.
        for event_id in previous_event_ids - rebuilt_event_ids - self._aliases.keys():
            event = self._events.get(event_id)
            if (
                event is None
                or event["finalized"]
                or event["event_label"] != "stranger"
            ):
                continue
            event.update(
                state="ended",
                status="ended",
                finalized=True,
                end_reason="retracted",
                end_seconds=event["last_detected_seconds"],
            )
            evidence = event["evidence"]
            evidence.update(
                revision=evidence["revision"] + 1,
                status="expired",
                available=None,
                failure_reason="event_retracted",
            )
            self._touch(event)
            dirty = True
            changed.append(event)
        aliases_changed = self._aliases != aliases_before
        if dirty or aliases_changed:
            self._persist()
        if changed or aliases_changed:
            self._change(
                changed,
                [
                    {"event_id": key, "canonical_event_id": value}
                    for key, value in self._aliases.items()
                    if aliases_before.get(key) != value
                ],
            )

    def _persist(self) -> None:
        if self._db is None:
            return
        with self._db:
            for table, values, persisted in (
                ("events", self._events, self._persisted_events),
                ("scene_episodes", self._scene_events, self._persisted_scene_events),
                ("observations", self._observations, self._persisted_observations),
            ):
                for key, value in values.items():
                    payload = json.dumps(value, ensure_ascii=False)
                    if persisted.get(key) != payload:
                        self._db.execute(
                            f"INSERT OR REPLACE INTO {table} VALUES (?,?)",
                            (key, payload),
                        )
                        persisted[key] = payload
            for key, value in self._anchors.items():
                if self._persisted_anchors.get(key) != value:
                    self._db.execute(
                        "INSERT OR REPLACE INTO anchors VALUES (?,?,?)",
                        (key, value[0], value[1]),
                    )
                    self._persisted_anchors[key] = value
            for key in self._persisted_aliases.keys() - self._aliases.keys():
                self._db.execute("DELETE FROM aliases WHERE event_id=?", (key,))
            for key, value in self._aliases.items():
                if self._persisted_aliases.get(key) != value:
                    self._db.execute(
                        "INSERT OR REPLACE INTO aliases VALUES (?,?)", (key, value)
                    )
            self._persisted_aliases = dict(self._aliases)
            for message, events in self._links.items():
                value = frozenset(events)
                if self._persisted_links.get(message) != value:
                    self._db.execute("DELETE FROM links WHERE message_id=?", (message,))
                    self._db.executemany(
                        "INSERT INTO links VALUES (?,?)",
                        ((message, event) for event in events),
                    )
                    self._persisted_links[message] = value

    def _change(
        self, events: list[dict[str, Any]], aliases: list[dict[str, Any]] | None = None
    ) -> None:
        self._cursor += 1
        self._changes.append(
            {
                "type": "event_change",
                "epoch": self._epoch,
                "sequence": self._cursor,
                "cursor": self._cursor,
                "events": copy.deepcopy(events),
                "aliases": aliases or [],
            }
        )

    @staticmethod
    def _touch(event: dict[str, Any]) -> None:
        event["revision"] += 1
        event["updated_at"] = _utc()

    def _canonical_subject(self, session_id: str, subject_id: str) -> str:
        seen: set[str] = set()
        while (session_id, subject_id) in self._subject_aliases:
            if subject_id in seen:
                break
            seen.add(subject_id)
            subject_id = self._subject_aliases[session_id, subject_id]
        return subject_id

    def _reconcile_subjects(
        self,
        session_id: str,
        aliases: dict[str, str],
        identities: dict[str, dict[str, Any]],
    ) -> None:
        for old, new in aliases.items():
            if not old or not new or old == new:
                continue
            canonical = self._canonical_subject(session_id, new)
            if canonical != old:
                self._subject_aliases[session_id, old] = canonical
        for (sid, old), new in list(self._subject_aliases.items()):
            if sid == session_id:
                self._subject_aliases[sid, old] = self._canonical_subject(sid, new)
        facts = {
            subject: identity
            for (sid, subject), identity in self._subject_identities.items()
            if sid == session_id
        }
        for subject, identity in identities.items():
            if identity.get("status") == "known" and identity.get("person_id"):
                fact = {
                    key: identity[key]
                    for key in ("status", "person_id", "name", "identity_source")
                    if key in identity
                }
                fact["confirmed_at"] = identity.get(
                    "confirmed_at", self._session(session_id).latest
                )
                previous = facts.get(subject)
                if previous is None or fact["confirmed_at"] >= previous["confirmed_at"]:
                    facts[subject] = fact
        for subject in facts:
            self._subject_identities.pop((session_id, subject), None)
        for subject, identity in facts.items():
            key = (session_id, self._canonical_subject(session_id, subject))
            previous = self._subject_identities.get(key)
            if previous is None or identity["confirmed_at"] >= previous["confirmed_at"]:
                self._subject_identities[key] = identity
        if self._db is not None:
            with self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO subject_aliases VALUES (?,?,?)",
                    (
                        (sid, old, new)
                        for (sid, old), new in self._subject_aliases.items()
                        if sid == session_id
                    ),
                )
                self._db.execute(
                    "DELETE FROM subject_identities WHERE session_id=?", (session_id,)
                )
                self._db.executemany(
                    "INSERT INTO subject_identities VALUES (?,?,?)",
                    (
                        (sid, subject, json.dumps(identity))
                        for (sid, subject), identity in self._subject_identities.items()
                        if sid == session_id
                    ),
                )

        def correct(record: dict[str, Any]) -> None:
            subject = self._canonical_subject(session_id, record["subject_id"])
            subjects = sorted(
                {
                    self._canonical_subject(session_id, value)
                    for value in record.get("subject_ids", [subject])
                }
            )
            if record.get("subject_type") == "group":
                subject = "group:" + ":".join(subjects)
            record.update(subject_id=subject, subject_ids=subjects)
            identity = self._subject_identities.get((session_id, subject))
            if "attributes" in record:
                record["attributes"].update(subject_id=subject, subject_ids=subjects)
                if identity and (
                    record.get("label") != "stranger"
                    or record["timestamp"] <= identity["confirmed_at"]
                ):
                    record["attributes"]["subject_identity"] = {
                        key: value for key, value in identity.items()
                        if key != "confirmed_at"
                    }
            else:
                self._correct_event_identity(record)

        for point in self._observations.values():
            if point["session_id"] == session_id and point["subject_id"] != "*":
                correct(point)
        changed = []
        for event in self._events.values():
            if event["stream_session_id"] != session_id:
                continue
            before = copy.deepcopy(event)
            correct(event)
            if event != before:
                self._touch(event)
                changed.append(event)
        self._rebuild(session_id)
        self._schedule_evidence(session_id)
        self._persist()
        if changed:
            self._change(changed)

    def _correct_event_identity(self, event: dict[str, Any]) -> None:
        if event.get("subject_type") != "person":
            return
        identity = self._subject_identities.get(
            (event["stream_session_id"], event["subject_id"])
        )
        if not identity:
            return
        stranger = event["event_label"] == "stranger"
        if stranger and event["last_detected_seconds"] > identity["confirmed_at"]:
            return
        event.update(
            identity_state="known", person_id=identity["person_id"],
            person_name=identity.get("name"),
        )
        if not stranger:
            return
        event.update(
            state="ended", status="ended", finalized=True, end_reason="retracted",
            end_seconds=max(
                event["last_detected_seconds"],
                min(
                    event.get("end_seconds") or identity["confirmed_at"],
                    identity["confirmed_at"],
                ),
            ),
        )
        evidence = event["evidence"]
        if evidence.get("failure_reason") != "identity_confirmed":
            evidence["revision"] += 1
        evidence.update(status="expired", available=None,
                        failure_reason="identity_confirmed")

    @staticmethod
    def _evidence_rank(event: dict[str, Any]) -> tuple:
        evidence = event["evidence"]
        available = evidence.get("available")
        if available:
            rank = 1 if available.get("partial") else 0
        else:
            rank = {"encoding": 2, "queued": 2, "collecting": 3}.get(
                evidence["status"], 4
            )
        return rank, event["start_seconds"], event["event_id"]

    def _representative_events(
        self, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep episode history internally, with one stranger clip per person."""
        result = []
        strangers: dict[tuple[str, str, bool], dict[str, Any]] = {}
        for event in events:
            if event["event_label"] != "stranger":
                result.append(event)
                continue
            key = (
                event["stream_session_id"],
                event["subject_id"],
                event["end_reason"] == "retracted",
            )
            previous = strangers.get(key)
            if previous is None or self._evidence_rank(event) < self._evidence_rank(
                previous
            ):
                strangers[key] = event
        return result + list(strangers.values())

    def _schedule_evidence(self, session_id: str | None = None) -> None:
        for event_id, event in list(self._events.items()):
            if event_id in self._aliases or (
                session_id and event["stream_session_id"] != session_id
            ):
                continue
            evidence = event["evidence"]
            if evidence["status"] != "collecting":
                continue
            if event["event_label"] == "stranger":
                candidates = [
                    other
                    for key, other in self._events.items()
                    if key not in self._aliases
                    and other["event_label"] == "stranger"
                    and other["end_reason"] != "retracted"
                    and other["stream_session_id"] == event["stream_session_id"]
                    and other["subject_id"] == event["subject_id"]
                ]
                representative = min(candidates, key=self._evidence_rank)
                if representative["event_id"] != event_id:
                    if representative["evidence"].get("available"):
                        evidence.update(
                            status="expired",
                            failure_reason="duplicate_stranger_evidence",
                        )
                        self._touch(event)
                        self._persist()
                        self._change([event])
                    continue
            session = self._sessions.get(event["stream_session_id"])
            if session is None:
                continue
            window = evidence["requested_window"]
            if session.latest < window["end_seconds"] and not session.ended:
                continue
            retained = self._queue.source_window(
                event["stream_session_id"],
                window["start_seconds"], window["end_seconds"],
            )
            source_ring = FrameRing(
                self.settings.buffer_seconds,
                max(self.settings.buffer_max_bytes, sum(len(f.jpeg) for f in retained)),
            )
            for frame in retained:
                source_ring.append(frame)
            frames, reason = source_ring.select(
                window["start_seconds"], window["end_seconds"],
                event["start_seconds"], self.settings.fps,
            )
            pinned = [
                frame
                for window_id, values in session.pins.items()
                if abs(session.windows[window_id]["timestamp"] - event["start_seconds"])
                < 1e-5
                for frame in values
            ]
            if pinned:
                combined = FrameRing(
                    self.settings.buffer_seconds, self.settings.buffer_max_bytes
                )
                for frame in sorted(
                    {f.timestamp_seconds: f for f in [*frames, *pinned]}.values(),
                    key=lambda f: f.timestamp_seconds,
                ):
                    combined.append(frame)
                frames, reason = combined.select(
                    window["start_seconds"],
                    window["end_seconds"],
                    event["start_seconds"],
                    self.settings.fps,
                )
            if not frames:
                evidence.update(status="failed", failure_reason=reason)
                self._touch(event)
                self._counts["encoding_failures"] += 1
                self._persist()
                self._change([event])
                continue
            revision = evidence["revision"]
            source_bytes = sum(len(frame.jpeg) for frame in frames)
            output_reservation = max(512 * 1024, source_bytes * 2)
            required = source_bytes + output_reservation + 64 * 1024
            reserved = sum(job.get("reserved_bytes", 0) for job in self._jobs.values())
            if (
                self._physical_usage() + reserved + required
                > self.settings.max_storage_bytes
            ):
                self._retention(
                    required_bytes=reserved + required, protected_event_id=event_id
                )
            if (
                self._physical_usage() + reserved + required
                > self.settings.max_storage_bytes
            ):
                evidence.update(
                    status="failed", failure_reason="storage_quota_exceeded"
                )
                self._touch(event)
                self._counts["storage_failures"] += 1
                self._persist()
                self._change([event])
                continue
            metadata = {
                "event_id": event_id,
                "stream_session_id": event["stream_session_id"],
                "revision": revision,
                "fps": self.settings.fps,
                "requested_start_seconds": window["start_seconds"],
                "requested_end_seconds": window["end_seconds"],
                "partial_reason": reason,
                "reserved_bytes": output_reservation,
                "max_output_bytes": output_reservation,
            }
            stage = self.settings.directory / "staging" / f"{event_id}-r{revision}"
            destination = (
                self.settings.directory / "videos" / f"{event_id}-r{revision}.mp4"
            )
            try:
                manifest = stage_clip(stage, frames, metadata)
                self._encode_queue.put_nowait((manifest, destination))
            except queue.Full:
                evidence.update(status="failed", failure_reason="encoding_queue_full")
                self._counts["encoding_failures"] += 1
            except OSError:
                evidence.update(
                    status="failed", failure_reason="evidence_storage_unavailable"
                )
                self._counts["storage_failures"] += 1
            else:
                self._jobs[event_id, revision] = metadata
                evidence.update(status="encoding", failure_reason=None)
            self._touch(event)
            self._persist()
            self._change([event])

    def _encode_loop(self) -> None:
        while True:
            try:
                item = self._encode_queue.get(timeout=0.2)
            except queue.Empty:
                if self._closed.is_set():
                    return
                continue
            try:
                if item is None or self._closed.is_set():
                    return
                manifest, destination = item
                stem, revision_text = destination.stem.rsplit("-r", 1)
                metadata = {"event_id": stem, "revision": int(revision_text)}
                try:
                    metadata = json.loads(manifest.read_text(encoding="utf-8"))
                    available = self._encoder(
                        manifest,
                        destination,
                        ffmpeg_path=self.settings.ffmpeg_path,
                        timeout_seconds=self.settings.encoder_timeout_seconds,
                        max_image_side=self.settings.max_image_side,
                    )
                    error = None
                except Exception as exc:
                    available = None
                    error = (
                        "encoder_unavailable"
                        if isinstance(exc, FileNotFoundError)
                        else "video_encoding_failed"
                    )
                # This background worker may wait for storage admission. Live
                # frames never wait for an encoder or filesystem operation.
                while not self._closed.is_set():
                    try:
                        self._queue.put(
                            ("encoded", (metadata, destination, available, error)),
                            timeout=0.2,
                        )
                        break
                    except queue.Full:
                        continue
            finally:
                self._encode_queue.task_done()

    def _encoded(
        self, metadata: dict[str, Any], path: Path, available: Any, error: str | None
    ) -> None:
        event_id, revision = metadata["event_id"], metadata["revision"]
        self._jobs.pop((event_id, revision), None)
        event = self._events.get(event_id)
        if event is None:
            path.unlink(missing_ok=True)
            return
        evidence = event["evidence"]
        if available is not None:
            self._videos[event_id, revision] = path
            if self._db is not None:
                with self._db:
                    self._db.execute(
                        "INSERT OR REPLACE INTO videos VALUES (?,?,?,?)",
                        (
                            event_id,
                            revision,
                            str(path.relative_to(self.settings.directory)),
                            time.time(),
                        ),
                    )
            if evidence["revision"] == revision:
                evidence.update(
                    status="ready",
                    failure_reason=None,
                    available={
                        **available,
                        "revision": revision,
                        "url": f"/v1/events/{event_id}/video?revision={revision}",
                    },
                )
        else:
            self._counts["encoding_failures"] += 1
            if evidence["revision"] == revision:
                evidence.update(status="failed", failure_reason=error)
        stage = self.settings.directory / "staging" / f"{event_id}-r{revision}"
        if stage.is_dir() and stage.resolve().is_relative_to(
            (self.settings.directory / "staging").resolve()
        ):
            shutil.rmtree(stage)
        self._touch(event)
        self._persist()
        self._change([event])
        self._schedule_evidence(event["stream_session_id"])

    def _physical_usage(self) -> int:
        total = 0
        for path in self.settings.directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                pass
        self._storage_bytes = total
        return total

    def _prune_ingress(self) -> None:
        """Keep revision/clip inputs while reclaiming acknowledged source history."""
        self._persist()
        now = self._clock()
        for run, session in self._sessions.items():
            windows = {
                key: value for key, value in session.windows.items()
                if now - value["admitted"] <= self.settings.revision_seconds
            }
            events = [
                event for event in self._events.values()
                if event["stream_session_id"] == run
                and (not event.get("finalized") or event["evidence"].get("status")
                     in {"collecting", "queued", "encoding"})
            ]
            ranges = {
                (event["evidence"]["requested_window"]["start_seconds"],
                 event["evidence"]["requested_window"]["end_seconds"])
                for event in events
            }
            times = [window["timestamp"] for window in windows.values()]
            times.extend(
                point["timestamp"] for point in self._observations.values()
                if point["session_id"] == run
                and now - point["admitted"] <= self.settings.revision_seconds
            )
            for timestamp in times:
                start = timestamp - self.settings.pre_roll_seconds
                ranges.add((start, start + self.settings.clip_seconds))
            # Recently ended runs need not occupy the raw-source allowance for
            # seven days. Their public metadata/videos have separate retention.
            before = (
                float("inf") if session.ended and not windows and not events
                else session.latest - self.settings.buffer_seconds
            )
            self._queue.prune_processed(
                run, before=before, protected_ranges=sorted(ranges),
                protected_windows=set(windows),
            )

    def _retention(
        self, required_bytes: int = 0, protected_event_id: str | None = None
    ) -> None:
        if self._db is None:
            return
        records = list(
            self._db.execute(
                "SELECT event_id,revision,path,created_at FROM videos "
                "ORDER BY created_at"
            )
        )
        sizes: dict[tuple[str, int], int] = {}
        for event_id, revision, _, _ in records:
            path = self._videos.get((event_id, revision))
            try:
                sizes[event_id, revision] = path.stat().st_size if path else 0
            except OSError:
                sizes[event_id, revision] = 0
        total = self._physical_usage()
        limit = max(0, self.settings.max_storage_bytes - required_bytes)
        self._storage_bytes = total
        cutoff = time.time() - self.settings.retention_days * 86400
        protected_runs = (
            {run for run, session in self._sessions.items() if not session.ended}
            | {
                event["stream_session_id"]
                for event in self._events.values()
                if not event.get("finalized")
                or event["evidence"].get("status")
                in {"collecting", "queued", "encoding"}
            }
            | {
                self._events[event_id]["stream_session_id"]
                for event_id, _revision in (*self._jobs, *self._leases)
                if event_id in self._events
            }
        )
        self._queue.prune_completed_runs(before=cutoff, protected=protected_runs)
        if self._queue.storage_pressure:
            # Prior-process runs have no in-memory session and may have no end
            # fact. Reclaim their acknowledged data, respecting replay/jobs/leases.
            self._queue.prune_completed_runs(
                before=float("inf"),
                protected=protected_runs | set(self._sessions),
                include_interrupted=True,
            )
            self._prune_ingress()
        # Retain one coherent history: age/quota evicts an ended event together
        # with every immutable video revision, aliases and observation links.
        removed = False
        for event in sorted(list(self._events.values()), key=lambda v: v["created_at"]):
            event_id = event["event_id"]
            if event_id == protected_event_id:
                continue
            if event_id in self._aliases or event["state"] != "ended":
                continue
            try:
                created = datetime.fromisoformat(event["created_at"]).timestamp()
            except ValueError:
                continue
            if created >= cutoff and total <= limit:
                continue
            ids = {event_id} | {
                key for key, value in self._aliases.items() if value == event_id
            }
            if any(key[0] in ids for key in self._jobs):
                continue
            if any(key[0] in ids for key in self._leases):
                continue
            blocked = False
            for video_key, video_path in list(self._videos.items()):
                if video_key[0] not in ids:
                    continue
                try:
                    video_path.unlink(missing_ok=True)
                except OSError:
                    blocked = True
                    break
                total -= sizes.get(video_key, 0)
                self._videos.pop(video_key, None)
            if blocked:
                continue
            messages = {
                message for message, linked in self._links.items() if linked & ids
            }
            removed_keys = [
                key
                for key, point in self._observations.items()
                if (
                    point["message_id"] in messages
                    and point["session_id"] == event["stream_session_id"]
                    and point.get("subject_id") == event["subject_id"]
                    and event["start_seconds"]
                    <= point["timestamp"]
                    <= event["last_detected_seconds"]
                )
            ]
            for key in removed_keys:
                self._observations.pop(key, None)
                self._persisted_observations.pop(key, None)
            for message in messages:
                self._links[message].difference_update(ids)
                if not self._links[message]:
                    del self._links[message]
                    self._persisted_links.pop(message, None)
            for key in ids:
                self._events.pop(key, None)
                self._scene_events.pop(key, None)
                self._persisted_scene_events.pop(key, None)
                self._anchors.pop(key, None)
                self._aliases.pop(key, None)
                self._persisted_events.pop(key, None)
                self._persisted_anchors.pop(key, None)
                for stage in (self.settings.directory / "staging").glob(f"{key}-r*"):
                    if stage.is_dir() and stage.resolve().is_relative_to(
                        (self.settings.directory / "staging").resolve()
                    ):
                        shutil.rmtree(stage)
            if not any(
                value["stream_session_id"] == event["stream_session_id"]
                for value in self._events.values()
            ):
                orphan_keys = [
                    key
                    for key, point in self._observations.items()
                    if point["session_id"] == event["stream_session_id"]
                ]
                for key in orphan_keys:
                    self._observations.pop(key, None)
                    self._persisted_observations.pop(key, None)
                removed_keys.extend(orphan_keys)
            with self._db:
                self._db.execute(
                    "DELETE FROM observations WHERE "
                    "json_extract(payload,'$.session_id')=? "
                    "AND json_extract(payload,'$.subject_id')=? "
                    "AND json_extract(payload,'$.timestamp') BETWEEN ? AND ?",
                    (
                        event["stream_session_id"],
                        event["subject_id"],
                        event["start_seconds"],
                        event["last_detected_seconds"],
                    ),
                )
                if not any(
                    value["stream_session_id"] == event["stream_session_id"]
                    for value in self._events.values()
                ):
                    self._db.execute(
                        "DELETE FROM observations WHERE "
                        "json_extract(payload,'$.session_id')=?",
                        (event["stream_session_id"],),
                    )
                for key in ids:
                    self._db.execute("DELETE FROM events WHERE event_id=?", (key,))
                    self._db.execute(
                        "DELETE FROM scene_episodes WHERE event_id=?", (key,)
                    )
                    self._db.execute("DELETE FROM anchors WHERE event_id=?", (key,))
                    self._db.execute(
                        "DELETE FROM aliases WHERE event_id=? OR canonical_event_id=?",
                        (key, key),
                    )
                    self._db.execute("DELETE FROM links WHERE event_id=?", (key,))
                    self._db.execute("DELETE FROM videos WHERE event_id=?", (key,))
                self._db.executemany(
                    "DELETE FROM observations WHERE observation_key=?",
                    ((key,) for key in removed_keys),
                )
            removed = True
            self._db.execute("PRAGMA incremental_vacuum(256)")
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            total = self._physical_usage()
        if removed:
            self._epoch = uuid4().hex
            self._reset_reason = "retention"
            self._cursor = 0
            self._changes.clear()
            self._persist()
            self._db.execute("PRAGMA incremental_vacuum(256)")
        changed = []
        for event_id, revision, _, created_at in records:
            if event_id not in self._events:
                continue
            if event_id == protected_event_id or any(
                key[0] == event_id for key in self._jobs
            ):
                continue
            if (event_id, revision) in self._leases:
                continue
            if created_at >= cutoff and total <= limit:
                continue
            path = self._videos.get((event_id, revision))
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue  # An active Windows download may hold this file.
            total -= sizes[event_id, revision]
            self._videos.pop((event_id, revision), None)
            with self._db:
                self._db.execute(
                    "DELETE FROM videos WHERE event_id=? AND revision=?",
                    (event_id, revision),
                )
            event = self._events.get(event_id)
            evidence = event.get("evidence", {}) if event else {}
            if (evidence.get("available") or {}).get("revision") == revision:
                evidence.update(
                    status="expired", available=None, failure_reason="retention"
                )
                self._touch(event)
                changed.append(event)
        if changed:
            self._persist()
            self._change(changed)
        retained_sessions = (
            {event["stream_session_id"] for event in self._events.values()}
            | {point["session_id"] for point in self._observations.values()}
            | {sid for sid, session in self._sessions.items() if not session.ended}
        )
        retired_aliases = [
            key for key in self._subject_aliases if key[0] not in retained_sessions
        ]
        if retired_aliases:
            with self._db:
                self._db.executemany(
                    "DELETE FROM subject_aliases WHERE session_id=? AND subject_id=?",
                    retired_aliases,
                )
            for key in retired_aliases:
                self._subject_aliases.pop(key, None)
        retired_identities = [
            key for key in self._subject_identities if key[0] not in retained_sessions
        ]
        if retired_identities:
            with self._db:
                self._db.executemany(
                    "DELETE FROM subject_identities "
                    "WHERE session_id=? AND subject_id=?",
                    retired_identities,
                )
            for key in retired_identities:
                self._subject_identities.pop(key, None)
        self._storage_bytes = self._physical_usage()
