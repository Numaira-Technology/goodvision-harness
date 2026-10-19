"""Lossless window tables and shared identifier fragments for final-answer evidence."""
from collections import Counter
import json
import re

ENCODING = "lossless_window_tables_v1"
UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
MARKERS = {"$string", "$join", "$table", "$blocks"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def expand_window_evidence(value):
    if not isinstance(value, dict) or value.get("encoding") != ENCODING:
        return value
    strings = value["strings"]

    def expand(item):
        if isinstance(item, dict):
            return {key: expand(child) for key, child in item.items()}
        if not isinstance(item, list):
            return item
        tag = item[0] if item and isinstance(item[0], str) else None
        if tag == "$string":
            return strings[item[1]]
        if tag == "$join":
            return "".join(expand(part) for part in item[1])
        if tag == "$blocks":
            return [row for block in item[1] for row in expand(block)]
        if tag == "$table":
            columns, fixed, rows = item[1:]
            varying = [key for key in columns if key not in fixed]
            result = []
            for row in rows:
                if len(row) != len(varying):
                    raise ValueError("Window table row width differs from its columns")
                cells = {**fixed, **dict(zip(varying, row))}
                result.append({key: expand(cells[key]) for key in columns})
            return result
        return [expand(child) for child in item]

    return expand(value["evidence"])


def compact_window_evidence(value):
    """Factor storage repetition, never merge observations or infer missing values."""
    counts = Counter()

    def collect(item):
        if isinstance(item, str):
            if len(item) >= 12:
                counts[item] += 1
            for match in UUID.finditer(item):
                if match.group() != item:
                    counts[match.group()] += 1
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            if item and isinstance(item[0], str) and item[0] in MARKERS:
                raise ValueError("Input collides with window table notation")
            for child in item:
                collect(child)

    collect(value)
    strings = [text for text, count in counts.items()
               if count >= 2 and (count - 1) * (len(text) - 18) > 24]
    indexes = {text: index for index, text in enumerate(strings)}

    def table(rows):
        columns = list(rows[0])
        fixed = {key: encode(rows[0][key]) for key in columns
                 if all(_json(row[key]) == _json(rows[0][key]) for row in rows[1:])}
        varying = [key for key in columns if key not in fixed]
        return ["$table", columns, fixed,
                [[encode(row[key]) for key in varying] for row in rows]]

    def signature(row):
        return (tuple(row), _json(row.get("window_id")), _json(row.get("vlm_lane")))

    def encode(item):
        if isinstance(item, str):
            if item in indexes:
                return ["$string", indexes[item]]
            parts, start = [], 0
            for match in UUID.finditer(item):
                if match.group() not in indexes:
                    continue
                parts.extend([item[start:match.start()], ["$string", indexes[match.group()]]])
                start = match.end()
            if parts:
                parts.append(item[start:])
                candidate = ["$join", [part for part in parts if part != ""]]
                if len(_json(candidate)) < len(_json(item)):
                    return candidate
            return item
        if isinstance(item, dict):
            return {key: encode(child) for key, child in item.items()}
        if not isinstance(item, list):
            return item
        if len(item) < 2 or not all(isinstance(row, dict) for row in item):
            return [encode(child) for child in item]
        # Consecutive blocks preserve exact source order and field presence.
        blocks, start = [], 0
        for end in range(1, len(item) + 1):
            if end != len(item) and signature(item[end]) == signature(item[start]):
                continue
            rows = item[start:end]
            blocks.append(table(rows) if len(rows) > 1 else [encode(rows[0])])
            start = end
        return ["$blocks", blocks]

    packed = {"encoding": ENCODING, "strings": strings, "evidence": encode(value),
              "record_format": (
                  'LOSSLESS notation: ["$string",i] is strings[i]; ["$join",parts] '
                  'concatenates expanded string parts without separators. '
                  '["$table",columns,fixed,rows] is a list of objects: fixed values '
                  'apply to EVERY row, and each row supplies the remaining columns '
                  'in listed order. ["$blocks",blocks] concatenates expanded lists '
                  'in order. Expand recursively. Every row remains a separate '
                  'observation at its original time, even if all fields repeat. '
                  'No value, precision, text, ID, null, empty list or record is omitted. '
                  'Shared UUID fragments do not merge identities or events.')}
    if expand_window_evidence(packed) != value:
        raise ValueError("Window tables cannot reconstruct their input")
    return packed
