"""Shared data shapes for the Benchmark V1 runner.

The runner is split into a *capture* step and an *extract* step joined by one
intermediate JSON format (a "capture"). The capture step runs against a live
harness (events over HTTP, per-frame subject boxes over the in-process tracking
snapshot hook); the extract step is a set of pure functions over the capture, so
answer logic is unit-tested without a GPU.

Capture JSON (one object per media, `video_id` joins to media.jsonl):

    {
      "video_id": "crowd_mot17_mot17_02",
      "fps": 30.0,
      "duration_seconds": 20.0,
      "frames": [
        {"t": 0.033, "frame": 1,
         "subjects": [{"id": "s1", "bbox": [x1, y1, x2, y2]}, ...]},
        ...
      ],
      "events": [
        {"event_label": "fight", "subject_ids": ["s1", "s2"],
         "first_evidence": 10.1, "confirmed_since": 12.3, "ended_since": 18.0},
        ...
      ]
    }

All bbox values are source-normalized xyxy in [0, 1] (harness
`BBOX_COORDINATE_SYSTEM = "source_normalized_xyxy"`). Times are seconds from the
start of the source.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Benchmark capability -> the harness event label it maps to. The public
# benchmark target label for physical conflict is the harness "fight" event.
EVENT_TARGET_TO_HARNESS_LABEL = {
    "physical_conflict": "fight",
}


def question_target_label(question: Question) -> str | None:
    """The harness event label this question is scored against, or None.

    Feature selection and answer extraction must agree on this. If a session
    enables one label and the extractor reads another, every episode is missed
    and the question scores zero without anything failing.
    """
    ctx = question.public_context
    if question.capability == "event_recognition":
        raw = ctx.get("queried_event_label") or question.raw.get("target_label")
    elif question.capability == "temporal_questions":
        raw = ctx.get("event_label") or question.raw.get("target_label")
    else:
        return None  # counting and location come from tracking, not events
    if not raw:
        return None
    label = str(raw)
    return EVENT_TARGET_TO_HARNESS_LABEL.get(label, label)


@dataclass(frozen=True, slots=True)
class Subject:
    """One tracked person in a single frame, source-normalized xyxy."""

    id: str
    bbox: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def center_inside_image(self) -> bool:
        """Match the benchmark's `bbox_centers_inside_image` counting rule."""
        cx, cy = self.center
        return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0

    @staticmethod
    def from_json(payload: Any) -> "Subject":
        box = payload["bbox"]
        if isinstance(box, dict):
            values = (box["x1"], box["y1"], box["x2"], box["y2"])
        else:
            values = tuple(box)
        if len(values) != 4:
            raise ValueError("bbox must have four values")
        sid = str(payload.get("id") or payload.get("subject_id") or "")
        if not sid:
            raise ValueError("subject requires an id")
        return Subject(id=sid, bbox=tuple(float(v) for v in values))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Frame:
    """Per-frame tracked subjects at a source timestamp."""

    t: float
    frame: int
    subjects: tuple[Subject, ...]

    @staticmethod
    def from_json(payload: Any) -> "Frame":
        return Frame(
            t=float(payload["t"]),
            frame=int(payload.get("frame", 0)),
            subjects=tuple(Subject.from_json(s) for s in payload.get("subjects", ())),
        )


@dataclass(frozen=True, slots=True)
class SceneEvent:
    """One persistent scene event episode with its lifecycle timestamps."""

    event_label: str
    subject_ids: tuple[str, ...] = ()
    first_evidence: float | None = None
    confirmed_since: float | None = None
    ended_since: float | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_since is not None

    @staticmethod
    def from_json(payload: Any) -> "SceneEvent":
        return SceneEvent(
            event_label=str(payload["event_label"]),
            subject_ids=tuple(str(s) for s in payload.get("subject_ids", ())),
            first_evidence=_opt_float(payload.get("first_evidence")),
            confirmed_since=_opt_float(payload.get("confirmed_since")),
            ended_since=_opt_float(payload.get("ended_since")),
        )


@dataclass(frozen=True, slots=True)
class Capture:
    """All structured output the runner needs for one media item."""

    video_id: str
    fps: float
    duration_seconds: float | None
    frames: tuple[Frame, ...] = field(default_factory=tuple)
    events: tuple[SceneEvent, ...] = field(default_factory=tuple)

    def frames_in_window(self, start_s: float, end_s: float) -> Iterator[Frame]:
        for frame in self.frames:
            if start_s <= frame.t <= end_s:
                yield frame

    def events_for_label(self, label: str) -> list[SceneEvent]:
        return [e for e in self.events if e.event_label == label]

    @staticmethod
    def from_json(payload: Any) -> "Capture":
        return Capture(
            video_id=str(payload["video_id"]),
            fps=float(payload.get("fps", 0.0)),
            duration_seconds=_opt_float(payload.get("duration_seconds")),
            frames=tuple(Frame.from_json(f) for f in payload.get("frames", ())),
            events=tuple(SceneEvent.from_json(e) for e in payload.get("events", ())),
        )

    @staticmethod
    def load(path: str | Path) -> "Capture":
        return Capture.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class Question:
    """One public benchmark question (no answer)."""

    qa_id: str
    capability: str
    question: str
    video_id: str
    answer_type: str
    response_options: tuple[str, ...] | None
    public_context: dict[str, Any]
    raw: dict[str, Any]

    def public_payload(self) -> dict[str, Any]:
        fields = ("qa_id", "video_id", "question", "response_options",
                  "capability", "answer_type", "public_context")
        return {key: self.raw[key] for key in fields if key in self.raw}

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "Question":
        options = payload.get("response_options")
        return Question(
            qa_id=str(payload["qa_id"]),
            capability=str(payload["capability"]),
            question=str(payload["question"]),
            video_id=str(payload["video_id"]),
            answer_type=str(payload.get("answer_type", "")),
            response_options=tuple(options) if options else None,
            public_context=dict(payload.get("public_context") or {}),
            raw=payload,
        )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_questions(path: str | Path) -> list[Question]:
    return [Question.from_json(row) for row in load_jsonl(path)]


def load_media(path: str | Path) -> dict[str, dict[str, Any]]:
    """Index media rows by video_id."""
    return {str(row["video_id"]): row for row in load_jsonl(path)}


def build_prompt(question: Question) -> str:
    """Reproduce the prompt shape from bench_v1_sample/how_to_feed.md.

    Only needed for a generic-VLM path; the structured extractors do not use it.
    """
    lines = ["Watch the full video carefully before answering.", "", f"Question: {question.question}"]
    if question.response_options:
        lines += ["", "Options:", *question.response_options]
    lines += [""]
    if question.response_options:
        lines.append("Answer with only the exact text of one option.")
    else:
        lines.append("Answer with only a single number.")
    return "\n".join(lines)
