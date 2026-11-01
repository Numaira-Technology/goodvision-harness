#!/usr/bin/env python3
"""Validate an offline A:W benchmark row; never authenticate or write to Sheets.

Usage: python3 -I validate_row.py row.json (or - for stdin).
Payload: {"values": [23 JSON cells], "notes": {"E": "...", ...}}.
This checks syntax and count arithmetic, not provenance truth or write authority.
"""

import json
import math
import re
import sys
from datetime import date
from pathlib import Path

CONTRACT = [
    ("A", "model", "text"),
    ("B", "backend", "backend"),
    ("C", "p50_s", "number"),
    ("D", "p95_s", "number"),
    *[(chr(69 + i), "accuracy", "percent") for i in range(15)],
    ("T", "hardware", "text"),
    ("U", "price", "number"),
    ("V", "spec_ver", "text"),
    ("W", "run_date", "date"),
]
BLOCKS = {
    "50": ("E", ["F", "G", "H", "I"]),
    "100": ("J", ["K", "L", "M", "N"]),
    "120": ("O", ["P", "Q", "R", "S"]),
}
NOTE_COLS = [c for overall, subs in BLOCKS.values() for c in [overall, *subs]]
BACKENDS = {"language", "decision", "vision"}
RATIO_RE = re.compile(r"(?<![\w./-])(\d+)/(\d+)(?![\w./])")


def is_blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def is_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def note_counts(note, *, subtask):
    """Return (correct, valid, errors, attempted, provisional) or raise ValueError."""
    ratios = RATIO_RE.findall(note)
    if len(ratios) != 1:
        raise ValueError("note needs exactly one correct/valid ratio")
    correct, valid = map(int, ratios[0])
    if not 0 <= correct <= valid or valid == 0:
        raise ValueError("note requires 0 <= correct <= valid and valid > 0")

    def count(name):
        found = re.findall(r"\b" + name + r"(\d+)(?![\w./+-])", note)
        if len(found) != 1:
            raise ValueError(f"note needs exactly one {name}N count")
        return int(found[0])

    errors, attempted = count("err"), count("attempted")
    if attempted != valid + errors:
        raise ValueError("attempted must equal valid + err")
    provisional = re.findall(r"\bprovisional=(yes|no)\b", note)
    if len(provisional) != 1:
        raise ValueError("note needs provisional=yes or provisional=no")
    if subtask:
        tp, fn, tn, fp = (count(key) for key in ("TP", "FN", "TN", "FP"))
        if tp + tn != correct or tp + fn + tn + fp != valid:
            raise ValueError("TP/FN/TN/FP do not match correct/valid")
        if tn + fp == 0 and "no negatives" not in note.lower():
            raise ValueError("note must disclose no negatives (specificity n/a)")
    return correct, valid, errors, attempted, provisional[0]


def check(payload):
    errors, warnings = [], []
    if not isinstance(payload, dict):
        return ["payload must be an object with values and notes"], []
    values, notes = payload.get("values"), payload.get("notes", {})
    if not isinstance(values, list) or len(values) != len(CONTRACT):
        return ["values must contain exactly 23 cells A:W, including blanks"], []
    if not isinstance(notes, dict):
        return ["notes must be an object mapping column letters to strings"], []
    allowed = {col for col, _, _ in CONTRACT}
    for col, note in notes.items():
        if col not in allowed or not isinstance(note, str) or not note.strip():
            errors.append(f"invalid note: {col!r} must be an A:W column with text")
    by_col = dict(zip((col for col, _, _ in CONTRACT), values, strict=True))
    for (col, key, kind), value in zip(CONTRACT, values, strict=True):
        tag = f"{col} ({key})"
        if kind == "text":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{tag} must be nonempty text")
        elif kind == "backend":
            if not isinstance(value, str) or value not in BACKENDS:
                errors.append(f"{tag} must be language, decision or vision")
        elif kind in {"number", "percent"} and not is_blank(value):
            if not is_number(value) or value < 0:
                errors.append(f"{tag} must be a finite nonnegative JSON number")
            elif kind == "percent" and value > 100:
                errors.append(f"{tag} must be within 0..100")
        elif kind == "date":
            try:
                if not isinstance(value, str) or not re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", value
                ):
                    raise ValueError
                date.fromisoformat(value)
            except ValueError:
                errors.append(f"{tag} must be a valid YYYY-MM-DD calendar date")

    p50, p95 = by_col["C"], by_col["D"]
    if is_blank(p50) != is_blank(p95):
        errors.append("C/D latency values must both be present or both blank")
    if is_number(p50) and is_number(p95) and p95 < p50:
        errors.append("D (p95) must be >= C (p50)")
    for col in ("C", "D", "U"):
        if not is_blank(by_col[col]) and is_blank(notes.get(col)):
            errors.append(f"{col} needs a timing or price unit/source note")

    parsed = {}
    filled = 0
    for dataset, (overall, subs) in BLOCKS.items():
        present = [col for col in subs if not is_blank(by_col[col])]
        if is_blank(by_col[overall]):
            if present:
                errors.append(f"dataset {dataset} has subtasks without overall")
        else:
            filled += 1
            if len(present) != 4:
                warnings.append(
                    f"dataset {dataset} has missing subtasks; explain before publishing"
                )
        for col in [overall, *subs]:
            if is_blank(by_col[col]):
                continue
            note = notes.get(col)
            if not isinstance(note, str) or not note.strip():
                errors.append(f"{col} needs an accuracy note")
                continue
            try:
                counts = note_counts(note, subtask=col != overall)
                correct, valid, _, attempted, _ = counts
                if (
                    is_number(by_col[col])
                    and abs(by_col[col] - round(100 * correct / valid, 1)) > 1e-8
                ):
                    raise ValueError(
                        "percentage does not match note correct/valid (one decimal)"
                    )
                if col == overall and attempted != int(dataset):
                    raise ValueError(
                        f"overall attempted must match the {dataset}-question cohort"
                    )
                parsed[col] = counts
            except ValueError as exc:
                errors.append(f"{col}: {exc}")
        if overall in parsed:
            for col in subs:
                if col in parsed and parsed[col][3] > parsed[overall][3]:
                    errors.append(f"{col} attempted exceeds overall")
                if (
                    col in parsed
                    and parsed[col][4] == "yes"
                    and parsed[overall][4] != "yes"
                ):
                    errors.append(f"{overall} must disclose provisional subtask labels")
            if any(
                sum(parsed[col][i] for col in subs if col in parsed)
                > parsed[overall][i]
                for i in range(4)
            ):
                errors.append(f"dataset {dataset} subtask counts exceed overall")
            if dataset != "120" and all(col in parsed for col in subs):
                if any(
                    sum(parsed[col][i] for col in subs) != parsed[overall][i]
                    for i in range(4)
                ):
                    errors.append(
                        f"dataset {dataset} subtask counts do not sum to overall"
                    )
    if not filled:
        errors.append("at least one dataset overall is required")
    return errors, warnings


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "-"
    try:
        raw = sys.stdin.read() if arg == "-" else Path(arg).read_text(encoding="utf-8")
        payload = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
        errors, warnings = check(payload)
    except (OSError, ValueError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(
        "OK: offline A:W syntax/count checks passed; "
        "verify provenance and live target before writing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
