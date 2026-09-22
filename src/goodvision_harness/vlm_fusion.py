"""Optional, bounded association of already-published VLM evidence with SAM people."""

from __future__ import annotations

import copy
import time
from collections import Counter, OrderedDict
from threading import Event, Lock, Thread

from .contracts import NormalizedBBox
from .scene_evidence import InspectionRequest
from .scene_runtime import TemporalInspection


def _overlap(a, b):
    intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0.0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return intersection / union if union else 0.0


def _unambiguous_matches(boxes, candidates):
    """Bind one VLM observation to source-aligned tracking, never across reports."""
    scores = {
        key: {sid: _overlap(box, other) for sid, other in candidates.items()}
        for key, box in boxes.items()
    }

    def winner(rows):
        ranked = sorted(rows, reverse=True)
        if (
            ranked
            and ranked[0][0] >= 0.5
            and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.15)
        ):
            return ranked[0][1]
        return None

    matches = {}
    for key, values in scores.items():
        target = winner((value, sid) for sid, value in values.items())
        if (
            target is not None
            and winner((values[target], other) for other, values in scores.items())
            == key
        ):
            matches[key] = target
    return matches


class VLMFusion:
    """Slow scene/archive callbacks run here, never on VLM input/output threads."""

    def __init__(self, scene, on_update=None):
        self.scene, self.on_update = scene, on_update
        self._lock, self._stop, self._wake = Lock(), Event(), Event()
        self._pending, self._links = OrderedDict(), OrderedDict()
        self._stats = Counter()
        self._thread = None
        self._bytes = 0
        self._finishing = Event()

    def start(self):
        if self._thread is None:
            self._thread = Thread(
                target=self._run, name="vlm-background-association", daemon=True
            )
            self._thread.start()

    def offer(self, report, evidence):
        size = sum(len(frame.jpeg_bytes) for frame in evidence)
        with self._lock:
            if self._stop.is_set():
                return
            self._pending[report["message_id"]] = (
                report,
                evidence,
                size,
                set(),
                time.monotonic(),
            )
            self._bytes += size
            while len(self._pending) > 64 or self._bytes > 32 * 1024**2:
                _, removed = self._pending.popitem(last=False)
                self._bytes -= removed[2]
                self._stats["capacity_dropped_reports"] += 1
        self._wake.set()

    def finish(self):
        self._finishing.set()
        self._wake.set()

    def close(self):
        self._stop.set()
        self._wake.set()
        with self._lock:
            self._stats["cancelled_reports"] += len(self._pending)
            self._pending.clear()
            self._bytes = 0

    def stats(self):
        with self._lock:
            return {
                **self._stats,
                "pending_reports": len(self._pending),
                "pending_bytes": self._bytes,
                "retained_links": len(self._links),
            }

    def links(self):
        with self._lock:
            return copy.deepcopy(tuple(self._links.values()))

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                rows = tuple(self._pending.items())
            changed = False
            for message_id, (report, evidence, _size, done, admitted) in rows:
                if self._stop.is_set():
                    return
                expired = time.monotonic() - admitted > 90.0
                try:
                    if not expired:
                        changed = self._associate(report, evidence, done) or changed
                except Exception:
                    with self._lock:
                        self._stats["association_failures"] += 1
                if (
                    expired
                    or self._finishing.is_set()
                    or len(done) == len(report["observations"])
                ):
                    with self._lock:
                        row = self._pending.pop(message_id, None)
                        if row is not None:
                            self._stats["unassociated_observations"] += len(
                                report["observations"]
                            ) - len(done)
                            self._bytes -= row[2]
                            self._stats[
                                "expired_reports" if expired else "completed_reports"
                            ] += 1
            if changed and not self._stop.is_set() and self.on_update is not None:
                try:
                    self.on_update()
                except Exception:
                    with self._lock:
                        self._stats["callback_failures"] += 1
            self._wake.wait(0.1)
            self._wake.clear()

    def _associate(self, report, evidence, done):
        scene = self.scene
        at, sequence = (
            report["frame"]["timestamp_seconds"],
            report["frame"]["source_frame_index"],
        )
        if (
            report["stream_session_id"] != scene.stream_session_id
            or report["source_id"] != scene.source_id
        ):
            return False
        changed = False
        with scene._lock:
            if scene._closed or self._stop.is_set():
                return False
            snapshot = scene.scene.snapshot(include_history=True)
            # Only actual completed tracking sources may lend person association.
            # Nearby source times are explicit association evidence, not substitute
            # timestamps/images for the VLM observation.
            candidates, sources = {}, {}
            for person in snapshot["entities"]:
                history = [
                    item
                    for item in person.get("history", ())
                    if (
                        scene.tracking_mode == "immediate"
                        or scene._tracking_frames.get(item["frame_sequence"])
                        == item["timestamp_seconds"]
                    )
                    and not item.get("identity_only")
                    and abs(item["timestamp_seconds"] - at) <= 0.55
                ]
                if not history:
                    continue
                item = min(
                    history, key=lambda item: abs(item["timestamp_seconds"] - at)
                )
                sid = person["subject_id"]
                candidates[sid] = NormalizedBBox(
                    *(item["bbox"][key] for key in ("x1", "y1", "x2", "y2"))
                )
                sources[sid] = (item["timestamp_seconds"], item["frame_sequence"])
            boxes = {
                obj["object_id"]: NormalizedBBox(
                    *(obj["bbox"][key] for key in ("x1", "y1", "x2", "y2"))
                )
                for obj in report["objects"]
                if obj["attributes"]["source_frame_sequence"] == sequence
            }
            matches = _unambiguous_matches(boxes, candidates)
            for observation in report["observations"]:
                eid = observation["evidence_id"]
                if eid in done or not observation["accepted"]:
                    done.add(eid)
                    continue
                if not set(observation["subject_ids"]) <= matches.keys():
                    continue
                ids = tuple(sorted(matches[sid] for sid in observation["subject_ids"]))
                if len(set(ids)) != len(ids):
                    continue
                for sid in observation["subject_ids"]:
                    target = matches[sid]
                    if not scene.scene._has_source_observation(target, at, sequence):
                        scene.scene.observe_person(
                            target,
                            timestamp_seconds=at,
                            frame_sequence=sequence,
                            bbox=boxes[sid],
                            allow_historical=True,
                        )
                if scene.scene.observe_event(
                    observation["event_label"],
                    ids,
                    timestamp_seconds=at,
                    frame_sequence=sequence,
                    detected=observation["detected"],
                    confidence=observation["confidence"],
                    evidence_id=eid,
                    allow_historical=True,
                ):
                    done.add(eid)
                    changed = True
                    link = {
                        "message_id": report["message_id"],
                        "request_id": report["request_id"],
                        "evidence_id": eid,
                        "event_label": observation["event_label"],
                        "vlm_event_ids": [],
                        "scene_event_ids": [
                            event["event_id"]
                            for event in scene.scene.snapshot(include_history=False)[
                                "events"
                            ]
                            if event["event_label"] == observation["event_label"]
                            and set(event["subject_ids"]) == set(ids)
                        ],
                        "source_frame_sequence": sequence,
                        "source_timestamp_seconds": at,
                        "method": "source_time_box_overlap",
                        "subjects": [
                            {
                                "vlm_subject_id": sid,
                                "subject_id": matches[sid],
                                "tracking_timestamp_seconds": sources[matches[sid]][0],
                                "tracking_frame_sequence": sources[matches[sid]][1],
                                "source_delta_seconds": sources[matches[sid]][0] - at,
                            }
                            for sid in observation["subject_ids"]
                        ],
                    }
                    with self._lock:
                        self._links[eid] = link
                        self._stats["associated_observations"] += 1
                        while len(self._links) > 256:
                            self._links.popitem(last=False)
            if changed and evidence and len(evidence) == len(report["evidence"]):
                request = InspectionRequest(
                    report["request_id"],
                    (),
                    "independent_vlm_association",
                    tuple(scene.event_labels),
                    evidence[0].timestamp_seconds,
                    evidence[-1].timestamp_seconds,
                )
                scene._latest_inspection = TemporalInspection(
                    scene.stream_session_id,
                    scene.source_id,
                    request,
                    evidence,
                    subject_boxes=tuple(
                        (sequence, target, boxes[sid])
                        for sid, target in matches.items()
                    ),
                    event_sensitivities=scene.event_sensitivities,
                )
        return changed
