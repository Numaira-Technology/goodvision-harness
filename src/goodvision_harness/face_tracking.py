"""Conservative, session-local continuity for face recognition decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from .contracts import NormalizedBBox
from .grounding import ModelObject
from .person_identity import PersonIdentity
from .rich_labeling import RichLabelSnapshot

_MAX_FACE_GAP = 3.0
_MAX_BODY_AGE = 0.75
_STRANGER_HOLD_SECONDS = 3.0


@dataclass(slots=True)
class _Track:
    bbox: NormalizedBBox
    seen_at: float
    identity: PersonIdentity = field(default_factory=PersonIdentity)
    unknown_count: int = 0
    body_id: str | None = None
    body_bbox: NormalizedBBox | None = None
    track_id: str = ""
    unknown_at: float | None = None
    unknown_attributes: dict[str, object] | None = None
    unknown_confidence: float | None = None
    conflict_person_id: str | None = None
    conflict_count: int = 0
    conflict_at: float = 0.0

    @property
    def person_id(self):
        return self.identity.person_id

    @person_id.setter
    def person_id(self, value):
        if value is None:
            self.identity.clear()
        else:
            self.identity.attributes.update(status="known", person_id=value)

    @property
    def known_at(self):
        return self.identity.confirmed_at

    @known_at.setter
    def known_at(self, value):
        self.identity.confirmed_at = value

    @property
    def known_embeddings(self):
        return self.identity.embeddings

    @known_embeddings.setter
    def known_embeddings(self, value):
        self.identity.embeddings = value


class FaceIdentityTracker:
    """Never share this history across stream sessions or camera sources.

    A body track supplies continuity, not identity evidence. Only a successful
    gallery match establishes identity. Poor observations cannot create a
    stranger alert. A usable unknown face is reported on its first observation;
    established known identities retain their trajectory-based protection.
    """

    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._last_timestamp = -1.0
        self._last_sequence = -1
        self._next_track_id = 1

    def rebind_body(self, previous_id: str, current_id: str) -> None:
        for track in self._tracks:
            if track.body_id == previous_id:
                track.body_id = current_id

    def bind_identity(self, face_ids: set[str], identity: PersonIdentity) -> None:
        """Bind even face-only tracks when exact-source association arrives late."""
        for track in self._tracks:
            if track.track_id in face_ids:
                track.identity = identity

    def identity_for(self, face_id: str) -> PersonIdentity | None:
        return next((t.identity for t in self._tracks if t.track_id == face_id), None)

    def forget_bodies(
        self, subject_ids: set[str], *, face_ids: set[str] | None = None
    ) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if track.body_id not in subject_ids
            and track.track_id not in (face_ids or ())
        ]

    def forget_person(self, person_id: str) -> None:
        for track in self._tracks:
            if track.person_id == person_id:
                track.person_id = None
                track.known_at = 0.0
                track.known_embeddings = ()
                track.unknown_count = 0
                track.unknown_at = None
                track.unknown_attributes = None
                track.conflict_person_id = None
                track.conflict_count = 0

    def resolve(
        self,
        objects: Sequence[ModelObject],
        *,
        timestamp: float,
        frame_sequence: int,
        people: Mapping[str, str],
        snapshot: RichLabelSnapshot | None = None,
        identities: Mapping[str, PersonIdentity] | None = None,
        face_subjects: Mapping[str, str] | None = None,
    ) -> tuple[ModelObject, ...]:
        objects = tuple(
            _removed_identity(item)
            if item.attributes.get("status") == "known"
            and item.attributes.get("person_id") not in people
            else item
            for item in objects
        )
        if timestamp <= self._last_timestamp or frame_sequence <= self._last_sequence:
            # A delayed request must neither inherit future identity nor mutate
            # newer history. Direct positive matches remain valid observations.
            return tuple(
                _unconfirmed(item, "out_of_order_frame")
                if item.attributes.get("status") == "unknown"
                else item
                for item in objects
            )
        self._last_timestamp = timestamp
        self._last_sequence = frame_sequence
        fresh_snapshot = (
            snapshot is not None
            and snapshot.frame_sequence <= frame_sequence
            and 0 <= timestamp - snapshot.timestamp_seconds <= _MAX_BODY_AGE
        )
        bodies = (
            {
                item.track_id: item.bbox
                for item in snapshot.objects
                if item.label.casefold() in {"person", "pedestrian"}
                and item.attributes.get("observation_state", "observed") == "observed"
                and timestamp
                - float(
                    item.attributes.get(
                        "last_observed_timestamp_seconds",
                        snapshot.timestamp_seconds,
                    )
                )
                <= _MAX_BODY_AGE
            }
            if fresh_snapshot and snapshot is not None
            else {}
        )

        retained: list[_Track] = []
        for track in self._tracks:
            if track.person_id not in people:
                track.known_embeddings = ()
                track.person_id = None
            if timestamp - track.seen_at > _MAX_FACE_GAP:
                continue
            if track.body_id is not None and fresh_snapshot:
                bbox = bodies.get(track.body_id)
                if bbox is None and _body_for_face(track.bbox, bodies) is not None:
                    # A different/ambiguous observed body occupies this position.
                    continue
                if bbox is not None and (
                    track.body_bbox is not None and _iou(track.body_bbox, bbox) < 0.2
                ):
                    continue
                if bbox is not None and track.body_bbox is not None:
                    track.bbox = _project_face(track.bbox, track.body_bbox, bbox)
                if bbox is not None and _body_for_face(track.bbox, bodies) == "":
                    continue
                if bbox is not None:
                    track.body_bbox = bbox
                    track.seen_at = timestamp
            retained.append(track)
        self._tracks = retained

        body_ids = [_body_for_face(item.bbox, bodies) for item in objects]
        counts = Counter(body_ids)
        body_ids = [key if key is None or counts[key] == 1 else "" for key in body_ids]
        proposals: list[int | None] = []
        for item, body_id in zip(objects, body_ids, strict=True):
            scores = []
            if body_id == "":
                proposals.append(None)
                continue
            for index, track in enumerate(self._tracks):
                if body_id is not None and track.body_id is not None:
                    score = 1.0 if body_id == track.body_id else 0.0
                else:
                    score = _face_geometry_score(item.bbox, track.bbox)
                if score >= 0.3:
                    scores.append((score, index))
            scores.sort(reverse=True)
            proposals.append(
                scores[0][1]
                if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.15)
                else None
            )
        proposal_counts = Counter(proposals)
        resolved: list[ModelObject] = []
        observed: set[int] = set()
        for item, body_id, proposal in zip(objects, body_ids, proposals, strict=True):
            if proposal is None or proposal_counts[proposal] != 1:
                track = _Track(
                    item.bbox, timestamp, track_id=f"face-{self._next_track_id}"
                )
                self._next_track_id += 1
                self._tracks.append(track)
            else:
                track = retained[proposal]
            owner = body_id or (face_subjects or {}).get(track.track_id)
            identity = (identities or {}).get(owner or "")
            if identity is not None and identity.confirmed_at <= timestamp:
                track.identity = identity
            observed.add(id(track))
            track.bbox = item.bbox
            track.seen_at = timestamp
            if body_id:
                track.body_id = body_id
                track.body_bbox = bodies[body_id]
            attributes = dict(item.attributes)
            attributes["face_track_id"] = track.track_id
            status = attributes.get("status")
            attributes["observation_status"] = status
            if status == "known":
                incoming_id = str(attributes["person_id"])
                if track.person_id is not None and incoming_id != track.person_id:
                    if (
                        track.conflict_person_id != incoming_id
                        or timestamp - track.conflict_at > 3.0
                    ):
                        track.conflict_person_id = incoming_id
                        track.conflict_count = 0
                        track.conflict_at = timestamp
                    threshold = float(attributes.get("threshold", 0.4))
                    strong = (
                        float(attributes.get("similarity", 1.0))
                        >= min(
                            1.0,
                            threshold + 0.05,
                        )
                        and float(attributes.get("match_margin", 1.0)) >= 0.10
                    )
                    track.conflict_count = track.conflict_count + 1 if strong else 0
                    if track.conflict_count < 3:
                        attributes = _known_attributes(
                            track, timestamp, attributes, people
                        )
                        resolved.append(
                            replace(item, label="known_person", attributes=attributes)
                        )
                        continue
                    track.track_id = f"face-{self._next_track_id}"
                    self._next_track_id += 1
                    attributes.update(face_track_id=track.track_id, identity_reset=True)
                    track.identity = PersonIdentity()
                track.person_id = str(attributes["person_id"])
                track.identity.attributes.update(
                    name=people[track.person_id], identity_source="face_match"
                )
                track.known_at = timestamp
                track.unknown_count = 0
                track.unknown_at = None
                track.unknown_attributes = None
                track.conflict_count = 0
                track.conflict_person_id = None
                if item.identity_quality and item.identity_embedding:
                    track.known_embeddings = (
                        *track.known_embeddings[-4:],
                        item.identity_embedding,
                    )
                attributes["identity_source"] = "face_match"
            elif status == "unusable":
                track.unknown_count = 0
                track.conflict_count = 0
                track.conflict_person_id = None
                if track.person_id is not None:
                    attributes = _known_attributes(track, timestamp, attributes, people)
                    item = replace(item, label="known_person")
                elif _confirmed_stranger(track, timestamp):
                    attributes = _held_stranger_attributes(track, timestamp, attributes)
                    item = replace(item, label="unknown_person")
            elif status == "unknown":
                if track.person_id is not None:
                    mismatch = _strong_mismatch(track, item)
                    if not mismatch:
                        track.conflict_count = 0
                        track.conflict_person_id = None
                    else:
                        if (
                            track.conflict_person_id != "__unknown__"
                            or timestamp - track.conflict_at > 4.0
                        ):
                            track.conflict_person_id = "__unknown__"
                            track.conflict_count = 0
                            track.conflict_at = timestamp
                        track.conflict_count += 1
                    if track.conflict_count < 5 or timestamp - track.conflict_at < 2.0:
                        attributes = _known_attributes(
                            track, timestamp, attributes, people
                        )
                        resolved.append(
                            replace(item, label="known_person", attributes=attributes)
                        )
                        continue
                    track.identity = PersonIdentity()
                    track.track_id = f"face-{self._next_track_id}"
                    self._next_track_id += 1
                    track.unknown_count = 0
                    attributes.update(face_track_id=track.track_id, identity_reset=True)
                track.unknown_count += 1
                # Diagnostic only: a new stranger does not wait for a streak.
                attributes["unknown_confirmations"] = track.unknown_count
                track.person_id = None
                track.unknown_at = timestamp
                attributes["identity_source"] = "face_match"
                track.unknown_attributes = dict(attributes)
                track.unknown_confidence = item.confidence
            if body_id:
                attributes["person_track_id"] = body_id
            resolved.append(replace(item, attributes=attributes))
        for track in self._tracks:
            if id(track) not in observed:
                track.unknown_count = 0
                track.conflict_count = 0
                track.conflict_person_id = None
                if not _confirmed_stranger(track, timestamp):
                    continue
                # An observed face on this person wins, including ambiguous
                # associations. Never add a second stale stranger to that person.
                if any(
                    _iou(item.bbox, track.bbox) > 0.1
                    or (
                        track.body_id is not None
                        and track.body_bbox is not None
                        and _body_for_face(item.bbox, {track.body_id: track.body_bbox})
                    )
                    for item in objects
                ):
                    continue
                resolved.append(
                    ModelObject(
                        label="unknown_person",
                        bbox=track.bbox,
                        confidence=track.unknown_confidence,
                        attributes=_held_stranger_attributes(track, timestamp, {}),
                    )
                )
        return tuple(resolved)


def _known_attributes(
    track: _Track,
    timestamp: float,
    attributes: dict[str, object],
    people: Mapping[str, str],
) -> dict[str, object]:
    assert track.person_id is not None
    attributes.update(
        status="known",
        person_id=track.person_id,
        name=people[track.person_id],
        identity_source="track_history",
        identity_age_seconds=round(timestamp - track.known_at, 3),
    )
    if track.body_id:
        attributes["person_track_id"] = track.body_id
    return attributes


def _strong_mismatch(track: _Track, item: ModelObject) -> bool:
    if (
        not item.identity_quality
        or not item.identity_embedding
        or not track.known_embeddings
    ):
        return False
    attributes = item.attributes
    similarity = attributes.get("similarity")
    if (
        similarity is None
        or float(similarity) >= float(attributes.get("threshold", 0.4)) - 0.1
    ):
        return False
    current = item.identity_embedding
    scores = [
        sum(left * right for left, right in zip(current, sample, strict=True))
        for sample in track.known_embeddings
        if len(sample) == len(current)
    ]
    return bool(scores) and max(scores) < 0.25


def _confirmed_stranger(track: _Track, timestamp: float) -> bool:
    return (
        track.person_id is None
        and track.unknown_at is not None
        and timestamp - track.unknown_at <= _STRANGER_HOLD_SECONDS
        and track.unknown_attributes is not None
    )


def _held_stranger_attributes(
    track: _Track, timestamp: float, current: Mapping[str, object]
) -> dict[str, object]:
    attributes = {
        **(track.unknown_attributes or {}),
        **current,
        "status": "unknown",
        "face_track_id": track.track_id,
        "identity_source": "track_history",
        "identity_age_seconds": round(timestamp - (track.unknown_at or 0.0), 3),
    }
    if track.body_id is not None:
        attributes["person_track_id"] = track.body_id
    return attributes


def _project_face(
    face: NormalizedBBox, previous: NormalizedBBox, current: NormalizedBBox
) -> NormalizedBBox:
    def x(value: float) -> float:
        relative = (value - previous.x1) / (previous.x2 - previous.x1)
        return max(0.0, min(1.0, current.x1 + relative * (current.x2 - current.x1)))

    def y(value: float) -> float:
        relative = (value - previous.y1) / (previous.y2 - previous.y1)
        return max(0.0, min(1.0, current.y1 + relative * (current.y2 - current.y1)))

    left, top, right, bottom = x(face.x1), y(face.y1), x(face.x2), y(face.y2)
    if right <= left or bottom <= top:
        return face
    return NormalizedBBox(left, top, right, bottom)


def _unconfirmed(item: ModelObject, reason: str) -> ModelObject:
    return replace(
        item,
        label="unusable_face",
        attributes={**item.attributes, "status": "unusable", "reason": reason},
    )


def _removed_identity(item: ModelObject) -> ModelObject:
    attributes = dict(item.attributes)
    for key in ("person_id", "name", "identity_source", "subject_identity"):
        attributes.pop(key, None)
    attributes.update(status="unusable", reason="identity_removed")
    return replace(
        item,
        label="unusable_face",
        attributes=attributes,
        identity_embedding=(),
        identity_quality=False,
    )


def _iou(left: NormalizedBBox, right: NormalizedBBox) -> float:
    intersection = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )
    union = (
        (left.x2 - left.x1) * (left.y2 - left.y1)
        + (right.x2 - right.x1) * (right.y2 - right.y1)
        - intersection
    )
    return intersection / union if union else 0.0


def _face_geometry_score(current: NormalizedBBox, previous: NormalizedBBox) -> float:
    """Face-only overlap is insufficient across abrupt scale/shape changes.

    A reliable shared body track takes precedence in the caller. Without that
    evidence, require plausible adjacent face dimensions before carrying an
    identity. Gradual approach remains valid because each step is compared.
    """
    width_ratio = (current.x2 - current.x1) / (previous.x2 - previous.x1)
    height_ratio = (current.y2 - current.y1) / (previous.y2 - previous.y1)
    area_ratio = width_ratio * height_ratio
    # Equivalent isotropic edge change is limited to 1.5x in either direction;
    # moderate head rotation may change one dimension more than the other.
    if (
        not 1.0 / 2.25 - 1e-9 <= area_ratio <= 2.25 + 1e-9
        or not 0.5 <= width_ratio <= 2.0
        or not 0.5 <= height_ratio <= 2.0
    ):
        return 0.0
    return _iou(current, previous)


def _body_for_face(
    face: NormalizedBBox, bodies: Mapping[str, NormalizedBBox]
) -> str | None:
    candidates = []
    face_area = (face.x2 - face.x1) * (face.y2 - face.y1)
    for track_id, body in bodies.items():
        intersection = max(0.0, min(face.x2, body.x2) - max(face.x1, body.x1)) * max(
            0.0, min(face.y2, body.y2) - max(face.y1, body.y1)
        )
        relative_y = ((face.y1 + face.y2) * 0.5 - body.y1) / (body.y2 - body.y1)
        if intersection / face_area >= 0.9 and 0 <= relative_y <= 0.6:
            candidates.append(track_id)
    if len(candidates) > 1:
        return ""  # Ambiguous body association must also disable face fallback.
    return candidates[0] if candidates else None
