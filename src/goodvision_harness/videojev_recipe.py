"""One VideoJev recipe shared by the live workbench and the benchmark.

A recipe is everything between "these frames" and "this decision": which
questions are asked (labels, sensitivities and any injected definitions such
as bag events), how many frames a window holds and how far apart they are,
and an optional per-label gate on the "present" probability.

The live System 1 inspector and the benchmark's current-workbench adapter both
build their request bodies, question templates and decisions through this
module, so a benchmark score describes exactly what the workbench would do
with the same frames. The production recipe reproduces the existing workbench
behaviour; nothing here changes it.
"""

from __future__ import annotations

import base64
import io
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .event_registry import (
    BUILTIN_VOCABULARY,
    EventVocabulary,
    vocabulary_from_json,
)
from .videojev_contract import scene_question_template
from .videojev_core import MODEL, REVISION, choice_indices

# Live System 1 windows (service.py passes these to the independent branch).
PRODUCTION_WINDOW_FRAMES = 2
PRODUCTION_FRAME_STEP_SECONDS = 0.5
MAX_IMAGE_SIDE = 720
_JSON_FIELDS = {
    "name",
    "labels",
    "sensitivities",
    "definitions",
    "window_frames",
    "frame_step_seconds",
    "gates",
}


def encode_frame(jpeg: bytes) -> tuple[bytes, int, int]:
    """JPEG as the workbench sends it: unchanged up to 720 px, else re-encoded."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as image:
        width, height = image.size
    if max(width, height) <= MAX_IMAGE_SIDE:
        return jpeg, width, height
    import cv2
    import numpy as np

    from .video_to_stream import encode_jpeg

    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return encode_jpeg(bgr, max_image_side=MAX_IMAGE_SIDE, quality=90)


@dataclass(frozen=True)
class VideoJevRecipe:
    name: str
    labels: tuple[str, ...]
    sensitivities: tuple[tuple[str, str], ...] = ()
    vocabulary: EventVocabulary = BUILTIN_VOCABULARY
    window_frames: int = PRODUCTION_WINDOW_FRAMES
    frame_step_seconds: float = PRODUCTION_FRAME_STEP_SECONDS
    gates: tuple[tuple[str, float], ...] = ()
    model: str = field(default=MODEL)
    revision: str = field(default=REVISION)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("recipe name must be nonempty text")
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("recipe labels must be unique and nonempty")
        self.vocabulary.resolve(self.labels)  # unknown labels raise here
        if type(self.window_frames) is not int or not 2 <= self.window_frames <= 4:
            raise ValueError("recipe window_frames must be 2, 3 or 4")
        step = self.frame_step_seconds
        if type(step) not in (int, float) or not math.isfinite(step) or step <= 0:
            raise ValueError("recipe frame_step_seconds must be positive")
        for label, gate in self.gates:
            if label not in self.labels:
                raise ValueError("recipe gate refers to an unknown label")
            if type(gate) not in (int, float) or not 0.34 <= gate <= 1:
                raise ValueError("recipe gates must be between 0.34 and 1")

    # -- questions -----------------------------------------------------------
    def question_template(self) -> tuple[dict, ...]:
        return scene_question_template(
            labels=self.labels,
            sensitivities=dict(self.sensitivities),
            vocabulary=self.vocabulary,
        )

    def expected(self) -> tuple[dict, ...]:
        return tuple(
            {"id": row["id"], "options": row["options"]}
            for row in self.question_template()
        )

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_step_seconds

    # -- request -------------------------------------------------------------
    def request_body(
        self,
        frames: Sequence[tuple[float, bytes]],
        *,
        session: str,
        request_id: str,
        include_template: bool = True,
        fps: float | None = None,
    ) -> dict[str, Any]:
        """``frames`` are (source time, JPEG) oldest first, as both paths hold them."""
        if len(frames) != self.window_frames:
            raise ValueError("frame count differs from the recipe window")
        times = [float(t) for t, _ in frames]
        if any(b <= a for a, b in itertools.pairwise(times)):
            raise ValueError("frame times must strictly increase")
        body: dict[str, Any] = {
            "request_id": request_id,
            "session": session,
            "fps": self.fps if fps is None else fps,
            "frame_times": times,
            "images": [
                base64.b64encode(encode_frame(jpeg)[0]).decode("ascii")
                for _, jpeg in frames
            ],
        }
        if include_template:
            body["question_template"] = [
                dict(row, query_parts=list(row["query_parts"]))
                for row in self.question_template()
            ]
        return body

    # -- decision ------------------------------------------------------------
    def decode(self, raw: Any) -> dict[str, dict]:
        """Per label: answer (True/False/None), probabilities, gated detection.

        Answers come from the same parser the live inspector uses (including
        the PPE-derived row). Without a gate, detected equals the answer, as
        in production; with a gate, "present" also needs its probability.
        """
        from .ppe import native_rows, parse_ppe_choices

        expected = self.expected()
        values = parse_ppe_choices(raw, expected)
        rows = {
            row["id"]: row
            for row in native_rows(raw)
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        choice_indices(
            [rows[q["id"]] for q in expected if q["id"] in rows], expected
        )
        gates = dict(self.gates)
        result = {}
        for label in self.labels:
            qid = "scene/" + label
            answer = values[qid]
            probabilities = rows[qid]["probabilities"]
            gate = gates.get(label)
            detected = answer
            if answer is True and gate is not None:
                detected = probabilities[0] >= gate
            result[label] = {
                "answer": answer,
                "probabilities": list(probabilities),
                "gate": gate,
                "detected": detected,
            }
        return result


def production_recipe(
    labels: Iterable[str],
    sensitivities: Mapping[str, str] | None = None,
    vocabulary: EventVocabulary = BUILTIN_VOCABULARY,
) -> VideoJevRecipe:
    """The live System 1 behaviour: 2 frames 0.5 s apart, argmax, no gate."""
    return VideoJevRecipe(
        name="production",
        labels=tuple(labels),
        sensitivities=tuple(sorted((sensitivities or {}).items())),
        vocabulary=vocabulary,
    )


def recipe_from_json(value: Any) -> VideoJevRecipe:
    """Build a recipe from a JSON profile (benchmark) or config entry."""
    if not isinstance(value, Mapping) or set(value) - _JSON_FIELDS:
        raise ValueError("invalid recipe fields")
    vocabulary = vocabulary_from_json(value.get("definitions") or None)
    sensitivities = value.get("sensitivities", {})
    gates = value.get("gates", {})
    if not isinstance(sensitivities, Mapping) or not isinstance(gates, Mapping):
        raise ValueError("recipe sensitivities and gates must be objects")
    labels = value.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("recipe labels must be a list of text")
    return VideoJevRecipe(
        name=value.get("name", "custom"),
        labels=tuple(labels),
        sensitivities=tuple(sorted(sensitivities.items())),
        vocabulary=vocabulary,
        window_frames=value.get("window_frames", PRODUCTION_WINDOW_FRAMES),
        frame_step_seconds=value.get(
            "frame_step_seconds", PRODUCTION_FRAME_STEP_SECONDS
        ),
        gates=tuple(sorted(gates.items())),
    )
