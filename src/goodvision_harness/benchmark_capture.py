"""Per-video benchmark observations and text-only answer evidence."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .scene_runtime import _PERSON_LABELS


def public_questions(questions, *, video_id):
    if not isinstance(video_id, str) or not video_id or not questions:
        raise ValueError("benchmark requires a video_id and its questions")
    fields = ("qa_id", "question", "response_options", "capability",
              "answer_type", "public_context")
    rows = []
    for question in questions:
        if question.get("video_id", video_id) != video_id:
            raise ValueError("benchmark questions must belong to one video")
        if not question.get("qa_id") or not isinstance(question.get("question"), str):
            raise ValueError("benchmark questions require qa_id and question")
        rows.append(copy.deepcopy({key: question[key] for key in fields if key in question}))
    if len({row["qa_id"] for row in rows}) != len(rows):
        raise ValueError("benchmark qa_id values must be unique")
    return rows


class BenchmarkCapture:
    def __init__(self, path: Path, metadata: dict[str, Any]):
        self.path = path
        self.metadata = copy.deepcopy(metadata)
        self.metadata["started_at"] = datetime.now(timezone.utc).isoformat()
        self.questions = tuple(self.metadata["questions"])
        self._lock = RLock()
        self._state = "running"
        self._error = None
        self._predictions = []
        self._events = {}
        self._merges = {}
        self.answer_ready = False
        self.processing_finished = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new session never overwrites another run's records.
        with self.path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "metadata", **self.metadata}) + "\n")

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False) + "\n")

    def record_frames(self, snapshots):
        for snapshot in snapshots:
            subjects = []
            for person in snapshot.objects:
                if person.label.lower() not in _PERSON_LABELS:
                    continue
                if person.attributes.get("tracking_confirmation", "confirmed") != "confirmed":
                    continue
                if (person.observation_state or person.attributes.get(
                    "observation_state", "observed"
                )) != "observed":
                    continue
                subjects.append({"id": person.track_id, "bbox": person.bbox.as_dict()})
            self.write("frame", {"t": snapshot.timestamp_seconds,
                                 "frame": snapshot.frame_sequence, "subjects": subjects})

    def record_merges(self, merges):
        with self._lock:
            for old, new in dict(merges).items():
                if self._merges.get(old) != new:
                    self._merges[old] = new
                    self.write("subject_merge", {"old": old, "new": new})

    def record_scene(self, snapshot):
        scene = snapshot.get("scene", {})
        with self._lock:
            current = {event["event_id"]: event for event in scene.get("events", ())}
            for transition in scene.get("transitions", ()):
                event_id = transition["event_id"]
                if event_id in current:
                    continue
                if event_id in self._events and transition.get("type") in {"end", "retract"}:
                    event = {**self._events[event_id], "phase": (
                        "retracted" if transition["type"] == "retract" else "ended"
                    )}
                    event["transition_type"] = transition["type"]
                    event["transition_timestamp_seconds"] = transition["timestamp_seconds"]
                    if event != self._events[event_id]:
                        self._events[event_id] = event
                        self.write("event", event)
            for event in scene.get("events", ()):
                event = {key: value for key, value in event.items() if key != "history"}
                if self._events.get(event["event_id"]) != event:
                    self._events[event["event_id"]] = copy.deepcopy(event)
                    self.write("event", event)

    def record_inspection(self, job, result, *, accepted, error=None):
        window = {"request_id": job.request.request_id,
                  "window_index": job.benchmark_window_index,
                  "start_seconds": job.request.start_seconds,
                  "end_seconds": job.request.end_seconds}
        if result is not None:
            self.write("model_response", {**window, "stage": "observation",
                                          "accepted": accepted,
                                          "response": result.raw_response})
        if accepted and result is not None:
            if result.verdict_error:
                self.write("error", {**window, "stage": "verdicts",
                                     "error": result.verdict_error})
            for observation in result.observations:
                self.write("observation", {**observation, **window})
        elif error or not accepted:
            self.write("error", {**window, "error": error or "inspection_not_accepted"})

    def evidence(self):
        with self._lock, self.path.open(encoding="utf-8") as handle:
            return build_answer_evidence([json.loads(line) for line in handle])

    def record_independent(self, report, result):
        window = {key: result[key] for key in (
            "request_id", "window_index", "start_seconds", "end_seconds", "lane",
        )}
        self.write("vlm_report", {**window, "report": report})
        if result.get("response") is not None:
            self.write("model_response", {**window, "stage": "observation",
                                          "accepted": not result.get("error"),
                                          "response": result["response"]})
        for observation in result.get("observations", ()):
            self.write("observation", {**observation, **window})
        for stage, error in (("observation", result.get("error")),
                             ("verdicts", result.get("verdict_error"))):
            if error:
                self.write("error", {**window, "stage": stage, "error": error})

    def set_answering(self):
        with self._lock:
            self._state = "answering"
            self._error = None
            self._predictions = []

    def finish(self, status: str, error: str | None = None, predictions=()) -> None:
        with self._lock:
            if self._state in {"complete", "failed"}:
                return
            answers = {row["qa_id"]: row["prediction"] for row in predictions}
            self._predictions = [
                {"qa_id": row["qa_id"], "prediction": answers.get(row["qa_id"], "")}
                for row in self.questions
            ]
            for row in self._predictions:
                self.write("prediction", row)
            self._state, self._error = status, error
            self.write("end", {"status": status, "error": error,
                               "finished_at": datetime.now(timezone.utc).isoformat()})

    def status(self):
        with self._lock:
            return {"stream_session_id": self.metadata["stream_session_id"],
                    "video_id": self.metadata["video_id"], "status": self._state,
                    "processing_finished": self.processing_finished,
                    "error": self._error, "predictions": list(self._predictions),
                    "metadata": copy.deepcopy(self.metadata)}


def build_answer_evidence(records):
    frames, events, merges, observations, errors = {}, {}, {}, [], []
    reports = []
    for row in records:
        kind = row["kind"]
        payload = {key: value for key, value in row.items() if key != "kind"}
        if kind == "frame":
            frames[row["frame"]] = payload
        elif kind == "event":
            events[row["event_id"]] = payload
        elif kind == "subject_merge":
            merges[row["old"]] = row["new"]
        elif kind == "observation":
            observations.append(payload)
        elif kind == "vlm_report":
            report = row["report"]
            reports.append({"window_index": row["window_index"], "lane": row["lane"],
                            "start_seconds": row["start_seconds"],
                            "end_seconds": row["end_seconds"],
                            "judgments": report["judgments"],
                            "objects": report["objects"],
                            "subject_id_scope": "observation"})
        elif kind == "error" and row.get("stage") != "answer":
            errors.append(payload)

    def canonical(subject):
        visited = set()
        while subject in merges and subject not in visited:
            visited.add(subject)
            subject = merges[subject]
        return subject

    ordered = sorted(frames.values(), key=lambda row: (row["t"], row["frame"]))
    for frame in ordered:
        people = {canonical(person["id"]): person["bbox"] for person in frame["subjects"]}
        frame["subjects"] = [{"id": subject, "bbox": bbox} for subject, bbox in people.items()]
        frame["count"] = len(people)
    for event in events.values():
        event["subject_ids"] = sorted({canonical(subject) for subject in event.get("subject_ids", ())})
    for row in observations:
        row["subject_ids"] = sorted({canonical(subject) for subject in row.get("subject_ids", ())})
    return {"frames": ordered, "events": list(events.values()),
            "subject_id_merges": merges,
            "observations": sorted(observations, key=lambda row: row["start_seconds"]),
            "vlm_reports": reports,
            "errors": errors}
