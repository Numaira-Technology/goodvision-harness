"""Bounded source-evidence protocol for the local VLM CPU companion."""

from __future__ import annotations

import base64
import binascii
import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import NormalizedBBox
from .event_sensitivity import normalize_event_sensitivities
from .runtime.scene import TemporalInspection
from .runtime.scheduling import EvidenceFrame, InspectionRequest
from .scene_inspection import _validate_job

MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 23 * 1024 * 1024
MAX_COMPLETION_BYTES = 256 * 1024
INSPECTION_KINDS = frozenset({"temporal", "full", "compact", "events"})
_OPTION_DEFAULTS = {"max_tokens": 1600, "max_image_side": 720, "jpeg_quality": 90}


@dataclass(frozen=True, slots=True)
class InspectionPayload:
    job: TemporalInspection
    kind: str
    model: str
    parameters: dict[str, Any]
    options: dict[str, int]
    deadline_seconds: float


def _text(value: object, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("invalid inspection text")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("invalid inspection number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("invalid inspection number") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid inspection number")
    # Retain JSON's integer/float representation so remote prompt metadata is
    # identical to the existing local builder, including source timestamps.
    return value


def _sequence(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError("invalid inspection sequence")
    return value


def _options(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) - _OPTION_DEFAULTS.keys():
        raise ValueError("invalid inspection options")
    result = {**_OPTION_DEFAULTS, **value}
    for name, maximum in (
        ("max_tokens", 16384),
        ("max_image_side", 2048),
        ("jpeg_quality", 100),
    ):
        if type(result[name]) is not int or not 1 <= result[name] <= maximum:
            raise ValueError("invalid inspection options")
    return result


def _parameters(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError("invalid inspection parameters")
    if {"model", "messages", "endpoint", "api_key", "base_url"} & value.keys():
        raise ValueError("invalid inspection parameters")
    if "stream" in value and value["stream"] is not False:
        raise ValueError("inspection completions cannot stream")
    for name in ("max_tokens", "max_completion_tokens"):
        if name in value and (
            type(value[name]) is not int or not 1 <= value[name] <= 16384
        ):
            raise ValueError("invalid inspection token budget")
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, RecursionError):
        raise ValueError("invalid inspection parameters") from None
    if len(encoded.encode()) > 64 * 1024:
        raise ValueError("inspection parameters exceed their byte limit")
    return copy.deepcopy(dict(value))


def encode_inspection(job, kind, model, parameters, options, deadline_seconds):
    """Serialize original JPEGs without opening, resizing or re-encoding pixels."""
    _validate_job(job)
    if kind not in INSPECTION_KINDS:
        raise ValueError("unsupported inspection kind")
    deadline = _number(deadline_seconds)
    if not 0 < deadline <= 300:
        raise ValueError("invalid inspection deadline")
    request = job.request
    return {
        "version": 1,
        "kind": kind,
        "model": _text(model),
        "parameters": _parameters(parameters),
        "options": _options(options),
        "deadline_seconds": deadline,
        "job": {
            "stream_session_id": job.stream_session_id,
            "request": {
                "request_id": request.request_id,
                "subject_ids": list(request.subject_ids),
                "reason": request.reason,
                "event_labels": list(request.event_labels),
                "start_seconds": request.start_seconds,
                "end_seconds": request.end_seconds,
            },
            "frames": [
                {
                    "frame_sequence": f.frame_sequence,
                    "timestamp_seconds": f.timestamp_seconds,
                    "jpeg_base64": base64.b64encode(f.jpeg_bytes).decode("ascii"),
                }
                for f in job.frames
            ],
            "subject_boxes": [
                {
                    "frame_sequence": sequence,
                    "subject_id": sid,
                    "bbox": [box.x1, box.y1, box.x2, box.y2],
                }
                for sequence, sid, box in job.subject_boxes
            ],
            "event_sensitivities": dict(job.event_sensitivities),
            "evidence_after_frame_sequence": job.evidence_after_frame_sequence,
            "grounding_required": job.grounding_required,
        },
    }


def decode_inspection(payload: object) -> InspectionPayload:
    """Validate untrusted metadata and byte bounds before any pixel decoding."""
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "kind",
        "model",
        "parameters",
        "options",
        "deadline_seconds",
        "job",
    }:
        raise ValueError("invalid inspection envelope")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported inspection version")
    kind = payload["kind"]
    if not isinstance(kind, str) or kind not in INSPECTION_KINDS:
        raise ValueError("unsupported inspection kind")
    deadline = _number(payload["deadline_seconds"])
    if not 0 < deadline <= 300:
        raise ValueError("invalid inspection deadline")
    wire = payload["job"]
    if not isinstance(wire, dict) or set(wire) != {
        "stream_session_id",
        "request",
        "frames",
        "subject_boxes",
        "event_sensitivities",
        "evidence_after_frame_sequence",
        "grounding_required",
    }:
        raise ValueError("invalid inspection job")
    session = _text(wire["stream_session_id"])
    raw = wire["request"]
    if not isinstance(raw, dict) or set(raw) != {
        "request_id",
        "subject_ids",
        "reason",
        "event_labels",
        "start_seconds",
        "end_seconds",
    }:
        raise ValueError("invalid inspection request")
    subjects, labels = raw["subject_ids"], raw["event_labels"]
    if (
        not isinstance(subjects, list)
        or len(subjects) > 128
        or not isinstance(labels, list)
        or not 1 <= len(labels) <= 6
    ):
        raise ValueError("invalid inspection request")
    subjects = tuple(_text(v) for v in subjects)
    labels = tuple(_text(v, maximum=64) for v in labels)
    if len(set(subjects)) != len(subjects) or len(set(labels)) != len(labels):
        raise ValueError("duplicate inspection references")
    request = InspectionRequest(
        _text(raw["request_id"]),
        subjects,
        _text(raw["reason"], maximum=128),
        labels,
        _number(raw["start_seconds"]),
        _number(raw["end_seconds"]),
    )
    raw_frames = wire["frames"]
    if not isinstance(raw_frames, list) or not 2 <= len(raw_frames) <= 4:
        raise ValueError("invalid inspection frame count")
    frames, total_bytes, previous_sequence = [], 0, -1
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict) or set(raw_frame) != {
            "frame_sequence",
            "timestamp_seconds",
            "jpeg_base64",
        }:
            raise ValueError("invalid inspection frame")
        sequence = _sequence(raw_frame["frame_sequence"])
        if sequence <= previous_sequence:
            raise ValueError("inspection frame sequences must increase")
        previous_sequence = sequence
        encoded = raw_frame["jpeg_base64"]
        if not isinstance(encoded, str) or len(encoded) > (
            (MAX_EVIDENCE_BYTES - total_bytes + 2) // 3 * 4
        ):
            raise ValueError("inspection evidence exceeds its byte limit")
        try:
            jpeg = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("invalid inspection image encoding") from None
        total_bytes += len(jpeg)
        if not jpeg or total_bytes > MAX_EVIDENCE_BYTES:
            raise ValueError("inspection evidence exceeds its byte limit")
        frames.append(
            EvidenceFrame(
                session, sequence, _number(raw_frame["timestamp_seconds"]), jpeg
            )
        )
    raw_boxes = wire["subject_boxes"]
    if not isinstance(raw_boxes, list) or len(raw_boxes) > 512:
        raise ValueError("invalid inspection boxes")
    boxes = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, dict) or set(raw_box) != {
            "frame_sequence",
            "subject_id",
            "bbox",
        }:
            raise ValueError("invalid inspection box")
        box = raw_box["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("invalid inspection box")
        boxes.append(
            (
                _sequence(raw_box["frame_sequence"]),
                _text(raw_box["subject_id"]),
                NormalizedBBox(*(_number(v) for v in box)),
            )
        )
    after = wire["evidence_after_frame_sequence"]
    if after is not None:
        after = _sequence(after, minimum=-1)
    if type(wire["grounding_required"]) is not bool:
        raise ValueError("invalid inspection grounding requirement")
    if not isinstance(wire["event_sensitivities"], dict):
        raise ValueError("invalid inspection sensitivities")
    sensitivities = normalize_event_sensitivities(wire["event_sensitivities"], labels)
    job = TemporalInspection(
        session,
        "",
        request,
        tuple(frames),
        tuple(boxes),
        tuple(sensitivities.items()),
        after,
        grounding_required=wire["grounding_required"],
    )
    _validate_job(job)
    return InspectionPayload(
        job,
        kind,
        _text(payload["model"]),
        _parameters(payload["parameters"]),
        _options(payload["options"]),
        deadline,
    )
