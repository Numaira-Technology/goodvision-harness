"""Versioned short grounding wire format; unchanged internal event semantics."""

from __future__ import annotations

import json
import re

from .grounding import SAFETY_EVENT_LABELS
from .scene_model import _ResponseValidationError
from .vlm_compact_model import raw_reply
from .vlm_event_policy import OUTPUT_FORMAT_MARKER, event_policy_text
from .vlm_independent_model import IndependentVLMInspector

WIRE_FORMAT = "lean-v1"
_ID = r"p[1-9][0-9]{0,2}"
_LIMIT = 64


def response_format(labels):
    """Named short fields and homogeneous arrays, without positional tuples."""
    labels = tuple(labels)
    if (
        not labels
        or len(labels) != len(set(labels))
        or any(label not in SAFETY_EVENT_LABELS for label in labels)
    ):
        raise ValueError("invalid grounding schema labels")
    subject_id = {"type": "string", "pattern": f"^{_ID}$"}
    vote = {
        "type": "object",
        "properties": {
            "p": {"type": "array", "maxItems": _LIMIT, "items": subject_id},
            "d": {"type": ["boolean", "null"]},
            "c": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
        "required": ["p", "d", "c"],
        "additionalProperties": False,
    }
    subject = {
        "type": "object",
        "properties": {
            "i": subject_id,
            "b": {
                "type": ["array", "null"],
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            },
        },
        "required": ["i", "b"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grounding_lean_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "v": {"type": "integer", "enum": [1]},
                    "s": {"type": "array", "maxItems": _LIMIT, "items": subject},
                    "e": {
                        "type": "object",
                        "properties": {
                            label: {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": _LIMIT,
                                "items": vote,
                            }
                            for label in labels
                        },
                        "required": list(labels),
                        "additionalProperties": False,
                    },
                },
                "required": ["v", "s", "e"],
                "additionalProperties": False,
            },
        },
    }


def _reject():
    raise _ResponseValidationError("invalid_lean_grounding", "verdict_schema")


def expand(raw, labels):
    """Validate field structure, then rename without guessing missing judgments.

    Localization failures still pass to the existing bbox validator: malformed
    boxes never become usable objects and never turn true/null events into false.
    """
    if (
        not isinstance(raw, dict)
        or set(raw) != {"v", "s", "e"}
        or type(raw["v"]) is not int
        or raw["v"] != 1
        or not isinstance(raw["s"], list)
        or len(raw["s"]) > _LIMIT
        or not isinstance(raw["e"], dict)
        or set(raw["e"]) != set(labels)
    ):
        _reject()
    subjects, seen = [], set()
    for subject in raw["s"]:
        if (
            not isinstance(subject, dict)
            or set(subject) != {"i", "b"}
            or not isinstance(subject["i"], str)
            or not re.fullmatch(_ID, subject["i"])
            or subject["i"] in seen
        ):
            _reject()
        seen.add(subject["i"])
        box = subject["b"]
        # v1 explicitly specifies integer 0..1000 coordinates. Fractions must
        # not be silently interpreted as tiny, valid 0..1000 boxes.
        if isinstance(box, list) and any(type(x) is not int for x in box):
            box = None
        subjects.append({"subject_id": subject["i"], "bbox": box})
    verdicts = []
    for label in labels:
        votes = raw["e"][label]
        if not isinstance(votes, list) or not votes or len(votes) > _LIMIT:
            _reject()
        for vote in votes:
            if (
                not isinstance(vote, dict)
                or set(vote) != {"p", "d", "c"}
                or not isinstance(vote["p"], list)
                or len(vote["p"]) > _LIMIT
                or any(
                    not isinstance(sid, str) or not re.fullmatch(_ID, sid)
                    for sid in vote["p"]
                )
                or len(set(vote["p"])) != len(vote["p"])
                or (label == "ground_litter" and vote["p"])
            ):
                _reject()
            verdicts.append(
                {
                    "event_label": label,
                    "subject_ids": vote["p"],
                    "detected": vote["d"],
                    "confidence": vote["c"],
                }
            )
            if len(verdicts) > _LIMIT:
                _reject()
    return {
        "bbox_coordinate_system": "normalized_1000",
        "subjects": subjects,
        "verdicts": verdicts,
    }


class LeanGroundingInspector(IndependentVLMInspector):
    def build_messages(self, job, *, encoded_images=None):
        messages = super().build_messages(job, encoded_images=encoded_images)
        messages[0]["content"] = (
            event_policy_text(
                job.request.event_labels,
                dict(job.event_sensitivities),
            )
            + OUTPUT_FORMAT_MARKER
            + (
                "Return minified JSON with exactly v,s,e. v=1 declares this format "
                "and integer normalized 0..1000 coordinates, NEVER pixels. "
                "s is an array of {i,b}: i is a response-local person reference "
                "p1,p2,..., not an identity; b is [x1,y1,x2,y2] on the 0..1000 grid, "
                "x1<x2 and y1<y2. Ground visible evaluated people in the LAST image, "
                "including people doing none of the actions. Use b:null when not "
                "locatable. Never invent hidden people, boxes or earlier positions. "
                "e maps EVERY requested event label to a nonempty array of {p,d,c}: "
                "p is the list of evaluated person references from s, d is "
                "true/false/null, c is confidence 0..1 or null. Include all six "
                "requested labels when six are requested; never omit an absent "
                "or uncertain event. Associate evaluated people for negative and "
                "uncertain judgments too. Group people only when their d and c "
                "are the same; otherwise use separate rows. Do not duplicate an "
                "event/participant group. Use p:[] when participants cannot be "
                "identified; still report the event. Empty s is allowed. "
                "ground_litter always uses p:[] (scene condition). For positive "
                "ppe_violation reference only affected people; negative/uncertain "
                "rows reference evaluated people. At most 64 subjects and 64 "
                "judgment rows in total. No no_events shortcut, explanations, "
                "descriptions, timestamps, extra fields or Markdown."
            )
        )
        if job.grounding_required:
            messages[0]["content"] += (
                " A prior event needs person-specific follow-up. Localize "
                "evaluated people and report all current judgments even if "
                "now negative; do not assume the prior event remains present."
            )
        if (
            self.model == "qwen/qwen3.5-35b-a3b"
            and self._parameters.get("response_format", {}).get("type") == "json_object"
        ):
            # This route rejects multimodal JSON Schema. A shape example keeps
            # JSON Object responses explicit without interpreting missing votes
            # as negative or weakening the strict local parser.
            messages[0]["content"] += (
                " JSON shape only (not evidence or preset answers): "
                + json.dumps(
                    {"v": 1, "s": [], "e": {
                        label: [{"p": [], "d": None, "c": None}]
                        for label in job.request.event_labels
                    }}, separators=(",", ":"),
                )
                + ". Every e value MUST be an array of objects even for a single "
                "judgment. Replace all example judgments using the images and "
                "the rules above."
            )
        return messages

    @staticmethod
    def _parse(job, response):
        raw = expand(raw_reply(response), job.request.event_labels)
        return IndependentVLMInspector._parse(
            job,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(raw, separators=(",", ":"))},
                    }
                ]
            },
        )
