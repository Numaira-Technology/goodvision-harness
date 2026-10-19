"""SAM-clocked display association, isolated from information/archival delivery.

Inference threads only put small, image-free observations in bounded mailboxes.
Only the native SAM consumer reads history or changes display state. SceneState
is reused for its evidence rules; this projection never publishes analysis facts.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, deque
from itertools import count
from threading import Lock

from .contracts import NormalizedBBox
from .events.state_machine import SceneState
from .person_registry import _face_body_score
from .vlm_fusion import _unambiguous_matches


def _box(value):
    return NormalizedBBox(*(value[key] for key in ("x1", "y1", "x2", "y2")))


class DisplaySubjectIDs:
    """One source-session map shared by display projection and fusion."""

    _scopes = count(1)
    _scope_lock = Lock()

    def __init__(self):
        with self._scope_lock:
            self._scope = next(self._scopes)
        self._lock = Lock()
        self._subjects = {}

    def resolve(self, epoch, track_id):
        # Keep full typed identity values locally; never truncate native IDs.
        key = json.dumps([epoch, track_id], separators=(",", ":"))
        with self._lock:
            if key not in self._subjects:
                self._subjects[key] = f"sam-{self._scope}-{len(self._subjects) + 1}"
            return self._subjects[key]


def display_subject_id(epoch, track_id, *, subject_ids):
    return subject_ids.resolve(epoch, track_id)


class SAMDisplayFusion:
    def __init__(self, session_id, source_id, *, stranger_enabled=False):
        self.session_id, self.source_id = session_id, source_id
        self.subject_ids = DisplaySubjectIDs()
        self.stranger_enabled = stranger_enabled
        self._inbox_lock = Lock()
        self._inbox = {kind: deque(maxlen=128) for kind in ("face", "vlm")}
        self._pending = deque(maxlen=256)
        self._history = deque(maxlen=1024)
        self._gallery_version = None
        self._forgotten = set()
        self._legacy = None
        self._face_sources = {}
        self._epoch = None
        self.grounding_required = False
        self._state = SceneState(
            session_id, source_id, person_history_size=1024,
            person_history_seconds=90.0,
        )

    def offer_vlm(self, report):
        if (report.get("stream_session_id") != self.session_id
                or report.get("source_id") != self.source_id
                or report.get("state") != "completed"):
            return
        # No images, service locks, scene callbacks or background association.
        payload = copy.deepcopy({
            key: report[key] for key in ("frame", "objects", "observations")
        })
        with self._inbox_lock:
            self._inbox["vlm"].append(payload)
            if any(o["detected"] is True for o in payload["observations"]):
                self.grounding_required = True

    def offer_face(self, payload):
        if payload.get("error"):
            return
        payload = copy.deepcopy(payload)
        with self._inbox_lock:
            if (self._gallery_version is None
                    or payload["gallery_version"] == self._gallery_version):
                self._inbox["face"].append(payload)

    def require_gallery(self, version):
        with self._inbox_lock:
            self._gallery_version = version
            self._inbox["face"].clear()

    def offer_scene(self, snapshot):
        # Stranger review remains authoritative in the existing identity pipeline.
        # Safety/name display does not depend on that pipeline finishing.
        scene = snapshot.get("scene", {})
        if scene.get("stream_session_id") != self.session_id:
            return
        with self._inbox_lock:
            self._legacy = scene

    def forget_identity(self, person_id):
        self._forgotten.add(person_id)
        self._state.forget_identity(person_id)
        # Forget also invalidates in-flight work if the ID is later re-enrolled.
        with self._inbox_lock:
            self._inbox["face"].clear()
        self._pending = deque(
            (row for row in self._pending if row[0] != "face"), maxlen=256,
        )

    def allow_identity(self, person_id):
        self._forgotten.discard(person_id)

    def _subject(self, epoch, track_id):
        return display_subject_id(epoch, track_id, subject_ids=self.subject_ids)

    def advance(self, frames):
        """Consume a newly accepted SAM batch; never call either inference path."""
        with self._inbox_lock:
            incoming = [
                (kind, payload) for kind, queue in self._inbox.items()
                for payload in queue
            ]
            for queue in self._inbox.values():
                queue.clear()
            gallery_version, legacy = self._gallery_version, self._legacy
        self._pending.extend(incoming)
        for frame in frames:
            self._epoch = frame["epoch"]
            boxes = {}
            for subject in frame["subjects"]:
                if subject["tracking_confirmation"] != "confirmed":
                    continue
                sid = self._subject(frame["epoch"], subject["track_id"])
                points = [point for polygon in subject["polygons"] for point in polygon]
                box = _box(frame["boxes"][subject["track_id"]]) if (
                    subject["track_id"] in frame.get("boxes", {})
                ) else NormalizedBBox(
                    min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points),
                )
                observed = self._state.observe_person(
                    sid, timestamp_seconds=frame["media_timestamp"],
                    frame_sequence=frame["source_sequence"], bbox=box,
                    native_track={"backend": "sam", "epoch": frame["epoch"],
                                  "object_id": subject["track_id"]},
                    tracking_confirmation="confirmed",
                )
                if observed:
                    boxes[sid] = box
            self._history.append((
                frame["source_sequence"], frame["media_timestamp"],
                frame["epoch"], boxes,
            ))
            self._state.advance(frame["media_timestamp"])
            self._state.observe_tracking_presence(
                tuple(boxes), timestamp_seconds=frame["media_timestamp"],
                frame_sequence=frame["source_sequence"],
            )
        now = frames[-1]["media_timestamp"]
        while self._history and self._history[0][1] < now - 90.0:
            self._history.popleft()
        waiting = deque(maxlen=256)
        for kind, payload in sorted(self._pending, key=lambda row: (
            row[1]["frame"]["timestamp_seconds"],
            row[1]["frame"]["source_frame_index"],
        )):
            frame = payload["frame"]
            at, sequence = frame["timestamp_seconds"], frame["source_frame_index"]
            if at < now - 90.0:
                continue
            if (kind == "face" and gallery_version is not None
                    and payload["gallery_version"] != gallery_version):
                continue
            source = min(self._history, key=lambda row: (
                row[0] != sequence, abs(row[1] - at), row[1] > at, row[0],
            ), default=None)
            gap = 0.75 if kind == "face" else 0.55
            if source is None or abs(source[1] - at) > gap or at > now:
                if at >= now - gap:
                    waiting.append((kind, payload))
                continue
            # Select one complete historical frame, including an empty frame.
            # Never assemble candidates from different times or retired epochs.
            if source[2] != self._epoch:
                continue
            if kind == "vlm":
                self._vlm(payload, source[3])
            else:
                self._face(payload, source[3])
        self._pending = waiting
        snapshot = self._state.snapshot(include_history=False)
        active_ids = {person["subject_id"] for person in snapshot["entities"]}
        self._face_sources = {key: value for key, value in self._face_sources.items()
                              if key in active_ids}
        self._merge_stranger_state(snapshot, legacy)
        with self._inbox_lock:
            pending_positive = any(
                any(o["detected"] is True for o in payload["observations"])
                for kind, payload in (
                    list(self._pending)
                    + [("vlm", p) for p in self._inbox["vlm"]]
                )
                if kind == "vlm"
            )
            self.grounding_required = pending_positive or any(
                e["phase"] != "ended" and e["event_label"] != "stranger"
                for e in snapshot["events"]
            )
        return {"scene": snapshot}

    def _vlm(self, payload, candidates):
        at = payload["frame"]["timestamp_seconds"]
        sequence = payload["frame"]["source_frame_index"]
        boxes = {
            obj["object_id"]: _box(obj["bbox"])
            for obj in payload["objects"]
            if obj["attributes"]["source_frame_sequence"] == sequence
        }
        matches = _unambiguous_matches(boxes, candidates)
        for observation in payload["observations"]:
            ids = observation["subject_ids"]
            if (not observation["accepted"] or not ids
                    or not set(ids) <= matches.keys()):
                continue
            subjects = tuple(matches[sid] for sid in ids)
            for sid in ids:
                if not self._state._has_source_observation(matches[sid], at, sequence):
                    self._state.observe_person(
                        matches[sid], timestamp_seconds=at, frame_sequence=sequence,
                        bbox=boxes[sid], allow_historical=True,
                    )
            self._state.observe_event(
                observation["event_label"], subjects, timestamp_seconds=at,
                frame_sequence=sequence, detected=observation["detected"],
                confidence=observation["confidence"],
                evidence_id=observation["evidence_id"], allow_historical=True,
            )

    def _face(self, payload, candidates):
        at = payload["frame"]["timestamp_seconds"]
        sequence = payload["frame"]["source_frame_index"]
        proposals = {}
        counts = Counter()
        for index, face in enumerate(payload["faces"]):
            box = _box(face["bbox"])
            possible = [sid for sid, body in candidates.items()
                        if _face_body_score(box, body) is not None]
            counts.update(possible)
            if len(possible) == 1:
                proposals[index] = possible[0]
        for index, sid in proposals.items():
            face = payload["faces"][index]
            if (counts[sid] != 1 or face.get("status") != "known"
                    or face.get("person_id") in self._forgotten
                    or (at, sequence) <= self._face_sources.get(sid, (-1, -1))):
                continue
            identity = {key: face[key] for key in ("status", "person_id", "name")}
            identity["identity_source"] = "face_match"
            if self._state.observe_person(
                sid, timestamp_seconds=at, frame_sequence=sequence,
                bbox=_box(face["bbox"]), identity=identity,
            ):
                self._face_sources[sid] = (at, sequence)

    def _merge_stranger_state(self, snapshot, legacy):
        if not self.stranger_enabled or not legacy:
            return
        mapping = {}
        people = {item["subject_id"]: item for item in snapshot["entities"]}
        for entity in legacy.get("entities", ()):
            native = entity.get("native_track", {})
            sid = self._subject(native.get("epoch"), native.get("object_id"))
            if sid not in people or native.get("epoch") != self._epoch:
                continue
            mapping[entity["subject_id"]] = sid
            if people[sid].get("identity", {}).get("status") != "known":
                people[sid]["identity_pending"] = entity.get("identity_pending", False)
        for event in legacy.get("events", ()):
            ids = event["subject_ids"]
            if (event["event_label"] != "stranger" or not ids
                    or not set(ids) <= mapping.keys()
                    or any(people[mapping[sid]].get("identity", {}).get("status")
                           == "known" for sid in ids)):
                continue
            snapshot["events"].append({
                **event, "subject_ids": [mapping[sid] for sid in ids],
            })
