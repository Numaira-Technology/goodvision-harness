"""Rising-edge alarm records keyed by person and event.

The live pipeline can emit many positive windows for one ongoing event.
This logger records the first detection for each (person, event) pair and
clears that edge on an explicit negative so a later recurrence logs again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .contracts import AnalysisResult
from .grounding import SAFETY_EVENT_LABELS

_ALARM_EVENTS = frozenset((*SAFETY_EVENT_LABELS, "stranger"))


@dataclass(frozen=True, slots=True)
class AlarmEvent:
    person_index: str
    person_id: str | None
    event_label: str
    timestamp_seconds: float
    observed_at: str
    confidence: float
    message: str


class AlarmLogger:
    """Session-local rising-edge alarm filter."""

    def __init__(self) -> None:
        self._active: set[tuple[str, str]] = set()
        self._events: deque[AlarmEvent] = deque(maxlen=256)

    def reset(self) -> None:
        self._active.clear()
        self._events.clear()

    def events(self) -> tuple[AlarmEvent, ...]:
        return tuple(self._events)

    def observe(self, result: AnalysisResult) -> tuple[AlarmEvent, ...]:
        objects = {item.object_id: item for item in result.objects}
        emitted: list[AlarmEvent] = []
        positives: set[tuple[str, str]] = set()
        for judgment in result.judgments:
            if judgment.event_label not in _ALARM_EVENTS:
                continue
            people = _people_for_judgment(judgment.object_ids, objects)
            if not judgment.detected:
                self._active = {
                    key for key in self._active if key[1] != judgment.event_label
                }
                continue
            if not people:
                people = (("unassigned", None),)
            for person_index, person_id in people:
                key = (person_index, judgment.event_label)
                positives.add(key)
                if key in self._active:
                    continue
                self._active.add(key)
                identity = person_id or person_index
                message = (
                    f"ALARM person={person_index} event={judgment.event_label} "
                    f"time={result.frame.observed_at} t={result.frame.timestamp_seconds:.3f}s"
                )
                if person_id:
                    message += f" known={person_id}"
                alarm = AlarmEvent(
                    person_index=person_index,
                    person_id=person_id,
                    event_label=judgment.event_label,
                    timestamp_seconds=result.frame.timestamp_seconds,
                    observed_at=result.frame.observed_at,
                    confidence=judgment.confidence,
                    message=message,
                )
                self._events.append(alarm)
                emitted.append(alarm)
                del identity
        return tuple(emitted)


def _people_for_judgment(
    object_ids: tuple[str, ...],
    objects: dict[str, object],
) -> tuple[tuple[str, str | None], ...]:
    people: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for object_id in object_ids:
        item = objects.get(object_id)
        attributes = getattr(item, "attributes", {}) if item is not None else {}
        if not isinstance(attributes, dict):
            attributes = dict(attributes) if attributes else {}
        subject = attributes.get("subject_id")
        if not isinstance(subject, str) or not subject.strip():
            subjects = attributes.get("subject_ids") or ()
            subject = next(
                (
                    value for value in subjects
                    if isinstance(value, str) and value.strip()
                ),
                None,
            )
        if not isinstance(subject, str) or subject in seen:
            continue
        seen.add(subject)
        person_index = attributes.get("person_index")
        if not isinstance(person_index, str) or not person_index.strip():
            person_index = subject
        person_id = attributes.get("person_id")
        people.append(
            (
                person_index,
                person_id if isinstance(person_id, str) and person_id.strip() else None,
            )
        )
    return tuple(people)
