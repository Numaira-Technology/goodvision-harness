"""Bounded, session-local people and event episodes built from source observations.

The registry owns person association. This module never merges people by geometry,
and a tracker prediction is never evidence that a person or event was observed.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from .contracts import NormalizedBBox

_EVENT_LABELS = frozenset({"fight", "smoking", "fall", "weapon", "stranger"})
_IDENTITY_KEYS = ("status", "person_id", "name", "identity_source", "matched_memory_id")


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be nonempty text of at most 256 characters")
    return value.strip()


def _integer(value: int, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def _time(value: float, name: str = "timestamp_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass
class _Person:
    subject_id: str
    bbox: NormalizedBBox
    first_seen: float
    last_seen: float
    frame_sequence: int
    history: deque
    identity: dict = field(default_factory=dict)
    identity_source: tuple[float, int] = (-1.0, -1)


@dataclass
class _Event:
    event_label: str
    subject_ids: tuple[str, ...]
    history: deque
    event_id: str | None = None
    phase: str = "ended"
    first_seen: float = -1.0
    last_seen: float = -1.0
    frame_sequence: int = -1
    positives: int = 0
    negatives: int = 0
    uncertain: bool = False
    confidence: float | None = None


class SceneState:
    """Store real person observations and source-ordered event evidence.

    Methods return False for stale, duplicate, ungrounded or capacity-rejected
    observations; malformed arguments raise ValueError. Snapshot copies are safe
    for callers to mutate. All retained collections have explicit size limits.
    """

    def __init__(
        self,
        stream_session_id: str,
        source_id: str,
        *,
        max_entities: int = 128,
        history_size: int = 64,
        occlusion_seconds: float = 3.0,
        retention_seconds: float = 90.0,
        confirm_observations: int = 2,
        clear_observations: int = 2,
    ) -> None:
        self.stream_session_id = _text(stream_session_id, "stream_session_id")
        self.source_id = _text(source_id, "source_id")
        self.max_entities = _integer(max_entities, "max_entities", 1, 4096)
        self.history_size = _integer(history_size, "history_size", 1, 1024)
        self.occlusion_seconds = _time(occlusion_seconds, "occlusion_seconds")
        self.retention_seconds = _time(retention_seconds, "retention_seconds")
        if (
            self.occlusion_seconds <= 0
            or self.retention_seconds <= self.occlusion_seconds
        ):
            raise ValueError("retention_seconds must exceed positive occlusion_seconds")
        self.confirm_observations = _integer(
            confirm_observations, "confirm_observations", 1
        )
        self.clear_observations = _integer(clear_observations, "clear_observations", 1)
        self.max_events = min(self.max_entities * len(_EVENT_LABELS), 2048)
        self._people: dict[str, _Person] = {}
        self._events: dict[tuple[str, tuple[str, ...]], _Event] = {}
        self._evidence: OrderedDict[tuple[str, str], tuple[str, ...]] = OrderedDict()
        self._transitions: deque = deque(maxlen=self.history_size)
        self._counters: Counter = Counter()
        self._timestamp = 0.0
        self._namespace = uuid4().hex[:12]
        self._episode = 0
        self._lock = RLock()

    def _reject(self, reason: str) -> bool:
        self._counters[reason] += 1
        return False

    def advance(self, timestamp_seconds: float) -> None:
        """Advance source time; occlusion never generates negative evidence."""
        timestamp = _time(timestamp_seconds)
        with self._lock:
            self._timestamp = max(self._timestamp, timestamp)
            expired = [
                key
                for key, person in self._people.items()
                if self._timestamp - person.last_seen > self.retention_seconds
            ]
            for key in expired:
                del self._people[key]
                self._counters["entities_expired"] += 1
            for key, event in list(self._events.items()):
                if self._timestamp - event.last_seen > self.retention_seconds or any(
                    subject not in self._people for subject in event.subject_ids
                ):
                    del self._events[key]
                    self._counters["events_expired"] += 1

    def observe_person(
        self,
        subject_id: str,
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        bbox: NormalizedBBox,
        identity: Mapping | None = None,
        observation_state: str = "observed",
    ) -> bool:
        subject_id = _text(subject_id, "subject_id")
        timestamp = _time(timestamp_seconds)
        sequence = _integer(frame_sequence, "frame_sequence", 0)
        if not isinstance(bbox, NormalizedBBox):
            raise ValueError("bbox must be a NormalizedBBox")
        if not isinstance(observation_state, str) or observation_state not in {
            "observed",
            "predicted",
            "held",
            "occluded",
            "lost",
        }:
            raise ValueError("unsupported observation_state")
        if identity is not None and not isinstance(identity, Mapping):
            raise ValueError("identity must be a mapping")
        safe_identity = {
            key: _text(identity[key], f"identity.{key}")
            for key in _IDENTITY_KEYS
            if identity is not None and key in identity and identity[key] is not None
        }
        with self._lock:
            self.advance(timestamp)
            if observation_state != "observed":
                return self._reject("nonobserved_people_ignored")
            if self._timestamp - timestamp > self.retention_seconds:
                return self._reject("stale_people_ignored")
            person = self._people.get(subject_id)
            if person is not None:
                if timestamp < person.last_seen or sequence < person.frame_sequence:
                    if self._has_source_observation(subject_id, timestamp, sequence):
                        if self._update_identity(
                            person, safe_identity, timestamp, sequence
                        ):
                            self._counters["historical_identity_updates"] += 1
                            return True
                    return self._reject("stale_people_ignored")
                same_frame = sequence == person.frame_sequence
                if same_frame and timestamp != person.last_seen:
                    return self._reject("stale_people_ignored")
            else:
                if len(self._people) >= self.max_entities:
                    return self._reject("entity_capacity_rejected")
                person = _Person(
                    subject_id,
                    bbox,
                    timestamp,
                    timestamp,
                    sequence,
                    deque(maxlen=self.history_size),
                )
                self._people[subject_id] = person
                self._counters["entities_created"] += 1
                same_frame = False
            self._update_identity(person, safe_identity, timestamp, sequence)
            if same_frame:
                self._counters["same_frame_person_updates"] += 1
                return True
            person.bbox, person.last_seen, person.frame_sequence = (
                bbox,
                timestamp,
                sequence,
            )
            person.history.append(
                {
                    "timestamp_seconds": timestamp,
                    "frame_sequence": sequence,
                    "bbox": bbox.as_dict(),
                }
            )
            self._counters["person_observations"] += 1
            return True

    @staticmethod
    def _update_identity(
        person: _Person,
        identity: dict,
        timestamp: float,
        sequence: int,
    ) -> bool:
        # Face inference arrives behind tracking. Its own source watermark protects
        # established identity without rewinding the body's location or visibility.
        if (
            not identity
            or timestamp < person.identity_source[0]
            or sequence < person.identity_source[1]
        ):
            return False
        if identity.get("status") != "known" and person.identity:
            return False
        if any(
            key in identity and identity[key] != person.identity.get(key)
            for key in ("person_id", "matched_memory_id")
        ):
            person.identity.clear()
        person.identity.update(identity)
        person.identity_source = (timestamp, sequence)
        return True

    def forget_identity(self, person_id: str) -> None:
        """Immediately remove a deleted gallery identity from retained entities."""
        person_id = _text(person_id, "person_id")
        with self._lock:
            for person in self._people.values():
                if person_id in (
                    person.identity.get("person_id"),
                    person.identity.get("matched_memory_id"),
                ):
                    person.identity.clear()
                    self._counters["identities_forgotten"] += 1

    def _has_source_observation(
        self, subject: str, timestamp: float, sequence: int
    ) -> bool:
        return any(
            item["frame_sequence"] == sequence
            and abs(item["timestamp_seconds"] - timestamp) <= 0.6
            for item in self._people[subject].history
        )

    def _transition(self, event: _Event, kind: str) -> None:
        self._transitions.append(
            {
                "type": kind,
                "event_id": event.event_id,
                "event_label": event.event_label,
                "subject_ids": list(event.subject_ids),
                "phase": event.phase,
                "timestamp_seconds": event.last_seen,
                "frame_sequence": event.frame_sequence,
            }
        )
        self._counters[f"event_{kind}s"] += 1

    def observe_event(
        self,
        event_label: str,
        subject_ids: tuple[str, ...],
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        detected: bool | None,
        evidence_id: str,
        confidence: float | None = None,
    ) -> bool:
        if not isinstance(event_label, str) or event_label not in _EVENT_LABELS:
            raise ValueError("unsupported event_label")
        if (
            not isinstance(subject_ids, (tuple, list))
            or not 1 <= len(subject_ids) <= self.max_entities
        ):
            raise ValueError("subject_ids must be a nonempty bounded sequence")
        participants = tuple(
            sorted({_text(item, "subject_id") for item in subject_ids})
        )
        timestamp = _time(timestamp_seconds)
        sequence = _integer(frame_sequence, "frame_sequence", 0)
        evidence_id = _text(evidence_id, "evidence_id")
        if detected is not None and type(detected) is not bool:
            raise ValueError("detected must be true, false or None")
        if confidence is not None:
            confidence = _time(confidence, "confidence")
            if confidence > 1:
                raise ValueError("confidence must be between 0 and 1")
        key, evidence_key = (event_label, participants), (event_label, evidence_id)
        with self._lock:
            if evidence_key in self._evidence:
                reason = (
                    "duplicate_evidence_ignored"
                    if self._evidence[evidence_key] == participants
                    else "reassigned_evidence_ignored"
                )
                return self._reject(reason)
            if self._timestamp - timestamp > self.retention_seconds:
                return self._reject("stale_evidence_ignored")
            if any(subject not in self._people for subject in participants):
                return self._reject("unknown_participants_ignored")
            event = self._events.get(key)
            if event is not None and (
                timestamp < event.last_seen or sequence <= event.frame_sequence
            ):
                return self._reject("stale_evidence_ignored")
            if not all(
                self._has_source_observation(subject, timestamp, sequence)
                for subject in participants
            ):
                return self._reject("ungrounded_evidence_ignored")
            # Unassigned or malformed source references cannot advance scene time.
            self.advance(timestamp)
            event = self._events.get(key)
            if event is None:
                if len(self._events) >= self.max_events:
                    # Reclaim completed episodes only; never evict an active incident.
                    ended = next(
                        (
                            item
                            for item, value in self._events.items()
                            if value.phase == "ended"
                        ),
                        None,
                    )
                    if ended is None:
                        return self._reject("event_capacity_rejected")
                    del self._events[ended]
                    self._counters["ended_events_evicted"] += 1
                event = _Event(
                    event_label, participants, deque(maxlen=self.history_size)
                )
                self._events[key] = event
            self._evidence[evidence_key] = participants
            while len(self._evidence) > self.max_events * self.history_size:
                self._evidence.popitem(last=False)
            transition = "update"
            if detected is True and event.phase == "ended":
                self._episode += 1
                event = _Event(
                    event_label,
                    participants,
                    deque(maxlen=self.history_size),
                    event_id=f"event-{self._namespace}-{self._episode}",
                    phase="suspected",
                    first_seen=timestamp,
                )
                self._events[key] = event
                transition = "start"
            event.last_seen, event.frame_sequence = timestamp, sequence
            event.uncertain, event.confidence = detected is None, confidence
            if detected is True:
                event.positives += 1
                event.negatives = 0
                if event.positives >= self.confirm_observations:
                    event.phase = "confirmed"
            elif detected is False:
                event.negatives += 1
                event.positives = 0
                if (
                    event.event_id
                    and event.phase != "ended"
                    and event.negatives >= self.clear_observations
                ):
                    event.phase, transition = "ended", "end"
            event.history.append(
                {
                    "timestamp_seconds": timestamp,
                    "frame_sequence": sequence,
                    "evidence_id": evidence_id,
                    "detected": detected,
                    "confidence": confidence,
                    "phase": event.phase,
                }
            )
            self._counters["event_observations"] += 1
            if event.event_id:
                self._transition(event, transition)
            return True

    def snapshot(self, *, include_history: bool = True) -> dict:
        """Copy scene state; internal scheduling can omit evidence histories."""
        if type(include_history) is not bool:
            raise ValueError("include_history must be a boolean")
        with self._lock:
            visible = {
                key
                for key, person in self._people.items()
                if self._timestamp - person.last_seen <= self.occlusion_seconds
            }
            entities = [
                {
                    "entity_id": person.subject_id,
                    "subject_id": person.subject_id,
                    "label": "person",
                    "bbox": person.bbox.as_dict(),
                    "identity": person.identity,
                    "observation_state": "observed"
                    if person.subject_id in visible
                    else "occluded",
                    "first_observed_timestamp_seconds": person.first_seen,
                    "last_observed_timestamp_seconds": person.last_seen,
                    "frame_sequence": person.frame_sequence,
                    **({"history": list(person.history)} if include_history else {}),
                }
                for person in self._people.values()
            ]
            events = [
                {
                    "event_id": event.event_id,
                    "event_label": event.event_label,
                    "subject_ids": list(event.subject_ids),
                    "phase": event.phase,
                    "observation_state": "uncertain"
                    if event.uncertain
                    or any(subject not in visible for subject in event.subject_ids)
                    else "observed",
                    "first_evidence_timestamp_seconds": event.first_seen,
                    "last_evidence_timestamp_seconds": event.last_seen,
                    "frame_sequence": event.frame_sequence,
                    "confidence": event.confidence,
                    "positive_observations": event.positives,
                    "negative_observations": event.negatives,
                    **({"history": list(event.history)} if include_history else {}),
                }
                for event in self._events.values()
                if event.event_id is not None
            ]
            return deepcopy(
                {
                    "schema_version": "1",
                    "stream_session_id": self.stream_session_id,
                    "source_id": self.source_id,
                    "timestamp_seconds": self._timestamp,
                    "entities": entities,
                    "events": events,
                    "transitions": list(self._transitions) if include_history else [],
                    "counters": dict(self._counters),
                }
            )
