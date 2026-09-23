"""Project gated scene episodes into the existing AnalysisResult contract.

These are state snapshots, not new model observations. They never feed the
scene, identity registry, overlay timeline, or evidence archive back again.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from uuid import uuid4

from PIL import Image

from ..contracts import (
    PUBLIC_MODEL_NAME,
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)

_NAMES = {
    "fight": "打架", "smoking": "吸烟", "fall": "跌倒",
    "weapon": "持械", "stranger": "陌生人",
}
_EVENT_FIELDS = (
    "event_id", "event_label", "subject_ids", "phase", "alert",
    "observation_state", "first_evidence_timestamp_seconds",
    "last_evidence_timestamp_seconds", "last_positive_timestamp_seconds",
    "confirmed_since_timestamp_seconds", "ended_since_timestamp_seconds",
    "frame_sequence", "confidence", "positive_observations",
    "negative_observations", "ended_reason", "alert_reason",
)


class SceneResultPublisher:
    """Session-local projection; caller serializes it with the service lock."""

    def __init__(self) -> None:
        self.reset(None)

    def reset(self, session_id: str | None) -> None:
        self.session_id = session_id
        self.frame: FrameReference | None = None
        self._sequence = 0
        self._signature: str | None = None

    def observe_reference(self, session_id: str, frame: FrameReference) -> None:
        if session_id != self.session_id:
            return
        if self.frame is None or (
            frame.timestamp_seconds, frame.source_frame_index
        ) >= (self.frame.timestamp_seconds, self.frame.source_frame_index):
            self.frame = frame

    def observe_frame(self, session_id: str, jpeg: bytes, timestamp: float,
                      sequence: int, observed_at: str) -> None:
        if session_id != self.session_id:
            return
        # Read the JPEG header only; do not decode another full display image.
        with Image.open(io.BytesIO(jpeg)) as source:
            width, height = source.size
        self.observe_reference(session_id, FrameReference(
            sequence, timestamp, observed_at, width, height,
        ))

    def project(self, snapshot: Mapping, *, labels: tuple[str, ...],
                emitted_at: str) -> AnalysisResult | None:
        scene = snapshot.get("scene")
        if (not scene or scene.get("stream_session_id") != self.session_id
                or self.frame is None):
            return None
        # A raw observation from the future must not lend its geometry/time to
        # an older scene. The next capture/scene callback will try again.
        if self.frame.timestamp_seconds > scene["timestamp_seconds"]:
            return None
        people = {p["subject_id"]: p for p in scene["entities"]}
        events = [
            {key: event[key] for key in _EVENT_FIELDS if key in event}
            for event in scene["events"] if event["event_label"] in labels
        ]
        # Ignore capture-only clock/box changes and periodic scene heartbeats.
        # Evidence changes, identity gates, and lifecycle changes do publish.
        signature = json.dumps({
            "events": sorted(events, key=lambda e: e["event_id"]),
            "identities": {
                sid: people[sid].get("identity", {})
                for sid in sorted({s for e in events for s in e["subject_ids"]})
                if sid in people
            },
            "state": snapshot["state"], "labels": labels,
        }, sort_keys=True)
        if signature == self._signature:
            return None
        judgments = []
        objects = {}
        for label in labels:
            episodes = [event for event in events if event["event_label"] == label]
            if not episodes:
                # An empty scene is not proof that this capability ran/cleared.
                continue
            confirmed = [e for e in episodes if e["phase"] == "confirmed"]
            active_subjects = {s for e in confirmed for s in e["subject_ids"]}
            if not active_subjects <= people.keys():
                # Positives require real associated boxes in the old contract.
                continue
            object_ids = []
            label_objects = {}
            for subject in sorted({s for e in episodes for s in e["subject_ids"]}):
                person = people.get(subject)
                if person is None:
                    continue
                associated = [e for e in events if subject in e["subject_ids"]]
                identity = person.get("identity", {})
                attrs = {
                    **identity,
                    "subject_id": subject,
                    "scene_state": snapshot["state"],
                    "observation_state": person["observation_state"],
                    "last_observed_timestamp_seconds": person[
                        "last_observed_timestamp_seconds"
                    ],
                    "last_observed_frame_sequence": person["frame_sequence"],
                    "event_ids": [e["event_id"] for e in associated],
                    "scene_events": associated,
                }
                box = person["bbox"]
                item = GroundedObject(
                    object_id=subject, label="person",
                    bbox=NormalizedBBox(*(box[k] for k in ("x1", "y1", "x2", "y2"))),
                    attributes=attrs,
                )
                label_objects[subject] = item
                if subject in active_subjects:
                    object_ids.append(subject)
            detected = bool(confirmed)
            if detected:
                description = f"已确认{_NAMES[label]}事件。 / Confirmed {label} event."
            else:
                description = (
                    f"{_NAMES[label]}事件未确认、已结束或已撤回。 / "
                    f"{label} event is suspected, ended or retracted."
                )
            judgments.append(EventJudgment(
                event_label=label, detected=detected,
                confidence=max((e.get("confidence") or 0.0)
                               for e in (confirmed or episodes)),
                description=description, object_ids=tuple(object_ids),
            ))
            objects.update(label_objects)
        if not judgments:
            return None
        self._sequence += 1
        result = AnalysisResult(
            message_id=str(uuid4()), source_id=scene["source_id"],
            stream_session_id=self.session_id, sequence=self._sequence,
            emitted_at=emitted_at, frame=self.frame, judgments=tuple(judgments),
            objects=tuple(objects.values()), model_name=PUBLIC_MODEL_NAME,
            model_latency_seconds=0.0, analysis_kind="scene_state",
            missing_event_labels=tuple(
                label for label in labels
                if label not in {judgment.event_label for judgment in judgments}
            ),
        )
        self._signature = signature
        return result
