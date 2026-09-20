"""Replay saved person and event observations through the scene engine offline.

Each UTF-8 JSONL record is a session header, person observation, attributed event
verdict, or source-time tick. Output is one scene snapshot per input record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.contracts import NormalizedBBox  # noqa: E402
from goodvision_harness.scene_state import SceneState  # noqa: E402

MAX_LINE_BYTES = 1024 * 1024


class ReplayError(ValueError):
    """A concise, line-local input error safe to show without echoing records."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be non-empty text of at most 256 characters")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _probability(value: Any, field: str) -> float:
    number = _number(value, field)
    if number > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON numbers are not allowed")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON fields are not allowed")
        result[key] = value
    return result


def _records(source: BinaryIO) -> Iterator[tuple[int, dict[str, Any]]]:
    line_number = 0
    while raw := source.readline(MAX_LINE_BYTES + 1):
        line_number += 1
        if len(raw) > MAX_LINE_BYTES:
            raise ReplayError(f"line {line_number}: record exceeds 1 MiB")
        try:
            row = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_keys,
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ReplayError(f"line {line_number}: invalid UTF-8 JSON object") from exc
        if not isinstance(row, dict):
            raise ReplayError(f"line {line_number}: record must be an object")
        yield line_number, row


def _fields(row: dict[str, Any], required: set[str], optional: set[str]) -> None:
    if not required <= row.keys():
        raise ValueError("missing required field")
    if row.keys() - required - optional:
        raise ValueError("unknown record field")


def _apply(scene: SceneState, row: dict[str, Any]) -> bool | None:
    kind = row.get("type")
    if kind == "tick":
        _fields(row, {"type", "timestamp_seconds"}, set())
        scene.advance(_number(row["timestamp_seconds"], "timestamp_seconds"))
        return None
    if kind == "person":
        _fields(
            row,
            {"type", "subject_id", "timestamp_seconds", "frame_sequence", "bbox"},
            {"identity", "observation_state"},
        )
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must be [x1,y1,x2,y2] on a normalized 0..1 canvas")
        identity = row.get("identity")
        if identity is not None and not isinstance(identity, dict):
            raise ValueError("identity must be an object")
        if identity is not None:
            _fields(
                identity,
                set(),
                {"status", "person_id", "name", "identity_source", "matched_memory_id"},
            )
        return scene.observe_person(
            _text(row["subject_id"], "subject_id"),
            timestamp_seconds=_number(row["timestamp_seconds"], "timestamp_seconds"),
            frame_sequence=_integer(row["frame_sequence"], "frame_sequence"),
            bbox=NormalizedBBox(
                *(_probability(value, "bbox coordinate") for value in bbox)
            ),
            identity=identity,
            observation_state=_text(
                row.get("observation_state", "observed"), "observation_state"
            ),
        )
    if kind == "event":
        _fields(
            row,
            {
                "type",
                "event_label",
                "subject_ids",
                "detected",
                "evidence_id",
                "timestamp_seconds",
                "frame_sequence",
            },
            {"confidence"},
        )
        subjects = row["subject_ids"]
        if not isinstance(subjects, list) or not 1 <= len(subjects) <= 128:
            raise ValueError("subject_ids must contain 1..128 person IDs")
        if row["detected"] is not None and type(row["detected"]) is not bool:
            raise ValueError("detected must be true, false or null")
        confidence = row.get("confidence")
        return scene.observe_event(
            _text(row["event_label"], "event_label"),
            tuple(_text(subject, "subject_id") for subject in subjects),
            timestamp_seconds=_number(row["timestamp_seconds"], "timestamp_seconds"),
            frame_sequence=_integer(row["frame_sequence"], "frame_sequence"),
            detected=row["detected"],
            evidence_id=_text(row["evidence_id"], "evidence_id"),
            confidence=(
                None if confidence is None else _probability(confidence, "confidence")
            ),
        )
    if kind == "session":
        raise ValueError("only one session header is allowed")
    raise ValueError("type must be person, event or tick after the session header")


def replay(source: BinaryIO) -> Iterator[dict[str, Any]]:
    """Stream records in arrival order; source timestamps govern the scene state.

    A well-formed but stale/unattributed observation emits accepted=false. Invalid
    JSON or fields abort at that line. The original earlier snapshots remain valid.
    """
    scene: SceneState | None = None
    for line_number, row in _records(source):
        try:
            if scene is None:
                if row.get("type") != "session":
                    raise ValueError("first record must be a session header")
                _fields(row, {"type", "stream_session_id", "source_id"}, set())
                scene = SceneState(
                    _text(row["stream_session_id"], "stream_session_id"),
                    _text(row["source_id"], "source_id"),
                )
                accepted: bool | None = True
            else:
                accepted = _apply(scene, row)
        except (ValueError, TypeError, OverflowError) as exc:
            # Validation errors contain field descriptions, never the input data.
            raise ReplayError(f"line {line_number}: {exc}") from exc
        yield {
            "line": line_number,
            "action": row["type"],
            "accepted": accepted,
            "scene": scene.snapshot(),
        }
    if scene is None:
        raise ReplayError("line 1: first record must be a session header")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSONL observations")
    parser.add_argument(
        "--output",
        type=Path,
        help="write a new JSONL file; existing files are preserved",
    )
    args = parser.parse_args(argv)
    output: TextIO = sys.stdout
    owned_output = False
    try:
        # Open input first so a missing input cannot leave an empty output file.
        with args.input.open("rb") as source:
            if args.output is not None:
                output = args.output.open("x", encoding="utf-8", newline="\n")
                owned_output = True
            for row in replay(source):
                output.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
    except ReplayError as exc:
        print(f"scene replay: {exc}", file=sys.stderr)
        return 2
    except FileExistsError:
        print("scene replay: output already exists; choose a new path", file=sys.stderr)
        return 2
    except OSError:
        print("scene replay: could not read input or write output", file=sys.stderr)
        return 2
    finally:
        if owned_output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
