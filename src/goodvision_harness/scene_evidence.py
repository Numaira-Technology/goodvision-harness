"""Bounded source-time evidence and change-driven scene inspection scheduling.

The planner proposes work; it never calls a model or treats planned work as evidence.
Callers must report completion and reset both components when a stream restarts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4


def _time(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("source timestamps must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    stream_session_id: str
    frame_sequence: int
    timestamp_seconds: float
    jpeg_bytes: bytes


class SceneEvidenceBuffer:
    """Retain JPEGs within a session, bounded by age, count, and encoded bytes."""

    def __init__(
        self,
        max_frames: int = 48,
        max_bytes: int = 16 * 1024 * 1024,
        history_seconds: float = 8.0,
    ) -> None:
        if max_frames < 1 or max_bytes < 1 or _time(history_seconds) <= 0:
            raise ValueError("evidence limits must be positive")
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.history_seconds = history_seconds
        self.reset()

    def reset(self) -> None:
        self._session: str | None = None
        self._frames: list[EvidenceFrame] = []
        self._latest = -math.inf
        self.byte_count = 0

    def __len__(self) -> int:
        return len(self._frames)

    def add(
        self,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
        jpeg_bytes: bytes,
    ) -> bool:
        timestamp = _time(timestamp_seconds)
        if (
            not isinstance(stream_session_id, str)
            or not stream_session_id.strip()
            or not isinstance(frame_sequence, int)
            or isinstance(frame_sequence, bool)
            or frame_sequence < 0
            or not isinstance(jpeg_bytes, bytes)
            or not jpeg_bytes
            or len(jpeg_bytes) > self.max_bytes
        ):
            return False
        if self._session not in (None, stream_session_id):
            return False
        if timestamp < self._latest - self.history_seconds or any(
            frame.frame_sequence == frame_sequence
            or frame.timestamp_seconds == timestamp
            for frame in self._frames
        ):
            return False
        self._session = stream_session_id
        self._latest = max(self._latest, timestamp)
        frame = EvidenceFrame(stream_session_id, frame_sequence, timestamp, jpeg_bytes)
        self._frames.append(frame)
        self._frames.sort(key=lambda item: item.timestamp_seconds)
        self.byte_count += len(jpeg_bytes)
        while self._frames and (
            self._frames[0].timestamp_seconds < self._latest - self.history_seconds
            or len(self._frames) > self.max_frames
            or self.byte_count > self.max_bytes
        ):
            self.byte_count -= len(self._frames.pop(0).jpeg_bytes)
        return frame in self._frames

    def select(
        self,
        *,
        stream_session_id: str,
        start_seconds: float,
        end_seconds: float,
        limit: int = 4,
    ) -> tuple[EvidenceFrame, ...]:
        start, end = _time(start_seconds), _time(end_seconds)
        if end < start or limit < 1:
            raise ValueError("evidence range must be ordered and limit positive")
        if stream_session_id != self._session:
            return ()
        frames = [f for f in self._frames if start <= f.timestamp_seconds <= end]
        if len(frames) <= limit:
            return tuple(frames)
        if limit == 1:
            return (frames[-1],)
        return tuple(
            frames[round(i * (len(frames) - 1) / (limit - 1))] for i in range(limit)
        )


@dataclass(frozen=True, slots=True)
class InspectionRequest:
    request_id: str
    subject_ids: tuple[str, ...]
    reason: str
    event_labels: tuple[str, ...]
    start_seconds: float
    end_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _Subject:
    observed_at: float
    bbox: tuple[float, ...] | None
    dirty: str = "new_person"


@dataclass(slots=True)
class _Task:
    first_ready: float = 0.0
    last_sent: float = -math.inf
    next_allowed: float = 0.0
    failures: int = 0
    pending: str | None = None
    retry: InspectionRequest | None = None


class InspectionPlanner:
    """Schedule observed people and interactions fairly, with a full-scene fallback.

    Boxes only supply a cheap motion cue. A request is neither an event verdict nor
    a completed inspection. Times are source-video seconds, including completion.
    """

    def __init__(
        self,
        *,
        cooldown_seconds: float = 3.0,
        refresh_seconds: float = 8.0,
        full_scene_interval_seconds: float = 15.0,
        history_seconds: float = 3.0,
        pending_timeout_seconds: float = 20.0,
        retry_backoff_seconds: float = 5.0,
        max_retries: int = 2,
        max_requests_per_plan: int = 2,
        max_pending_requests: int = 4,
        max_subjects: int = 128,
        observation_max_age_seconds: float = 3.0,
        motion_threshold: float = 0.08,
    ) -> None:
        for value in (
            cooldown_seconds,
            refresh_seconds,
            full_scene_interval_seconds,
            history_seconds,
            pending_timeout_seconds,
            retry_backoff_seconds,
            observation_max_age_seconds,
            motion_threshold,
        ):
            if _time(value) <= 0:
                raise ValueError("inspection intervals and thresholds must be positive")
        if min(max_requests_per_plan, max_pending_requests, max_subjects) < 1:
            raise ValueError("inspection capacity must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.cooldown_seconds = cooldown_seconds
        self.refresh_seconds = refresh_seconds
        self.full_scene_interval_seconds = full_scene_interval_seconds
        self.history_seconds = history_seconds
        self.pending_timeout_seconds = pending_timeout_seconds
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retries = max_retries
        self.max_requests_per_plan = max_requests_per_plan
        self.max_pending_requests = max_pending_requests
        self.max_subjects = max_subjects
        self.observation_max_age_seconds = observation_max_age_seconds
        self.motion_threshold = motion_threshold
        self._serial = 0
        self._namespace = uuid4().hex[:12]
        self.reset()

    def reset(self) -> None:
        self._session: str | None = None
        self._latest = -math.inf
        self._subjects: dict[str, _Subject] = {}
        self._tasks: dict[tuple[str, ...], _Task] = {(): _Task()}
        self._pending: dict[str, InspectionRequest] = {}

    @staticmethod
    def _bbox(value: Any) -> tuple[float, ...] | None:
        try:
            box = tuple(float(value[k]) for k in ("x1", "y1", "x2", "y2"))
            if (
                all(math.isfinite(v) for v in box)
                and box[2] > box[0]
                and box[3] > box[1]
            ):
                return box
        except (KeyError, TypeError, ValueError):
            pass
        return None

    def complete(
        self,
        request_id: str,
        *,
        success: bool,
        timestamp_seconds: float,
    ) -> None:
        timestamp = _time(timestamp_seconds)
        request = self._pending.get(request_id)
        if request is None or timestamp < request.end_seconds:
            return
        # Late callbacks cannot move cooldowns backward or resurrect expired work.
        timestamp = max(timestamp, self._latest)
        self._pending.pop(request_id)
        task = self._tasks[request.subject_ids]
        task.pending = None
        task.retry = None if success else request
        task.failures = 0 if success else task.failures + 1
        delay = (
            self.cooldown_seconds
            if success
            else min(self.retry_backoff_seconds * 2 ** min(task.failures - 1, 10), 60.0)
        )
        task.next_allowed = timestamp + delay
        if task.failures > self.max_retries:
            task.retry = None
            task.next_allowed = timestamp + max(
                delay, self.refresh_seconds, self.full_scene_interval_seconds
            )

    def plan(
        self,
        scene_snapshot: Mapping[str, Any],
        *,
        timestamp_seconds: float,
    ) -> tuple[InspectionRequest, ...]:
        timestamp = _time(timestamp_seconds)
        session = str(scene_snapshot.get("stream_session_id", ""))
        if self._session not in (None, session) or timestamp <= self._latest:
            return ()
        self._session, self._latest = session, timestamp
        for request in tuple(self._pending.values()):
            if timestamp - request.end_seconds >= self.pending_timeout_seconds:
                self.complete(
                    request.request_id, success=False, timestamp_seconds=timestamp
                )

        visible: dict[str, Mapping[str, Any]] = {}
        present: set[str] = set()
        for entity in scene_snapshot.get("entities", ()):
            if (
                not isinstance(entity, Mapping)
                or entity.get("label", "person") != "person"
            ):
                continue
            subject_id = str(entity.get("entity_id") or entity.get("subject_id") or "")
            if subject_id:
                present.add(subject_id)
            try:
                observed = _time(entity["last_observed_timestamp_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not subject_id
                or entity.get("observation_state") != "observed"
                or not 0 <= timestamp - observed <= self.observation_max_age_seconds
            ):
                continue
            visible[subject_id] = entity
        # Keep already admitted people first, then admit newcomers as capacity frees.
        admitted = [key for key in self._subjects if key in present]
        admitted += sorted(key for key in visible if key not in self._subjects)
        admitted = admitted[: self.max_subjects]
        self._subjects = {k: v for k, v in self._subjects.items() if k in admitted}
        active = set(admitted) & visible.keys()
        for key in tuple(self._tasks):
            if (
                key
                and not set(key) <= set(admitted)
                and self._tasks[key].pending is None
            ):
                del self._tasks[key]

        candidates: dict[tuple[str, ...], tuple[str, tuple[str, ...]]] = {}
        for subject_id in admitted:
            self._tasks.setdefault((subject_id,), _Task(first_ready=timestamp))
            if subject_id not in active:
                continue
            entity = visible[subject_id]
            observed = float(entity["last_observed_timestamp_seconds"])
            bbox = self._bbox(entity.get("bbox"))
            subject = self._subjects.get(subject_id)
            if subject is None:
                subject = _Subject(observed, bbox)
                self._subjects[subject_id] = subject
            elif observed > subject.observed_at:
                if (
                    bbox
                    and subject.bbox
                    and max(abs(a - b) for a, b in zip(bbox, subject.bbox, strict=True))
                    >= self.motion_threshold
                ):
                    subject.dirty = subject.dirty or "motion_change"
                    subject.bbox = bbox
                subject.observed_at = observed
            if subject.dirty:
                candidates[(subject_id,)] = (subject.dirty, ())
        for event in scene_snapshot.get("events", ()):
            if not isinstance(event, Mapping) or event.get("phase") not in (
                "suspected",
                "confirmed",
            ):
                continue
            ids = tuple(sorted(set(str(v) for v in event.get("subject_ids", ()))))
            label = event.get("event_label")
            if ids and set(ids) <= active and isinstance(label, str) and label:
                reason = (
                    "suspected_event"
                    if event["phase"] == "suspected"
                    else "event_refresh"
                )
                labels = candidates.get(ids, ("", ()))[1]
                candidates[ids] = (reason, tuple(sorted(set((*labels, label)))))
        fallback = self._tasks[()]
        if timestamp - fallback.last_sent >= self.full_scene_interval_seconds:
            candidates[()] = ("full_scene_scan", ())
        for ids, task in self._tasks.items():
            if task.retry and (not ids or set(ids) <= active):
                candidates.setdefault(ids, (task.retry.reason, task.retry.event_labels))
        for ids in tuple(self._tasks):
            if len(ids) > 1 and ids not in candidates and not self._tasks[ids].pending:
                del self._tasks[ids]
        for ids in candidates:
            if ids not in self._tasks and len(self._tasks) < 2 * self.max_subjects + 1:
                self._tasks[ids] = _Task(first_ready=timestamp)

        # Oldest served first: a continuously active person cannot monopolize slots.
        def fairness_key(ids: tuple[str, ...]) -> tuple[bool, float, tuple[str, ...]]:
            task = self._tasks.get(ids, _Task(first_ready=timestamp))
            served = (
                task.last_sent if math.isfinite(task.last_sent) else task.first_ready
            )
            return bool(ids), served, ids

        ordered = sorted(candidates, key=fairness_key)
        busy = {s for request in self._pending.values() for s in request.subject_ids}
        requests: list[InspectionRequest] = []
        capacity = min(
            self.max_requests_per_plan, self.max_pending_requests - len(self._pending)
        )
        for ids in ordered:
            if len(requests) >= capacity:
                break
            task = self._tasks.get(ids)
            if task is None:
                continue
            reason, labels = candidates[ids]
            interval = (
                self.refresh_seconds
                if reason == "event_refresh"
                else self.cooldown_seconds
            )
            if (
                task.pending
                or timestamp < task.next_allowed
                or timestamp - task.last_sent < interval
            ):
                continue
            if set(ids) & busy or any(
                timestamp - self._tasks.get((s,), _Task()).last_sent
                < self.cooldown_seconds
                for s in ids
            ):
                continue
            self._serial += 1
            request = InspectionRequest(
                f"inspection-{self._namespace}-{self._serial}",
                ids,
                reason,
                labels,
                max(0.0, timestamp - self.history_seconds),
                timestamp,
            )
            task.pending, task.last_sent = request.request_id, timestamp
            if task.retry is None:
                task.failures = 0
            task.retry = None
            self._pending[request.request_id] = request
            requests.append(request)
            busy.update(ids)
            for subject_id in ids:
                self._subjects[subject_id].dirty = ""
                self._tasks.setdefault((subject_id,), _Task()).last_sent = timestamp
        return tuple(requests)
