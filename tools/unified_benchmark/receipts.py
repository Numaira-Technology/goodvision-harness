"""Keep nonfinite raw evidence inside strict JSON receipts.

Only diagnostic fields use this codec; scores, plans, ledgers and hashes retain
strict JSON. The text is a re-encoding of decoded values, not original wire bytes.
Finite raw fields keep their existing structure and need no encoding marker.
"""

import json
from pathlib import Path

from .contract import canonical

JSON_TEXT = "json-text-nonfinite-v1"


def raw_fields(**fields):
    """Return raw fields with explicit encoding only when strict JSON rejects them."""
    result = {}
    for name, value in fields.items():
        try:
            canonical(value)
        except ValueError:
            # NaN/Infinity tokens live inside a quoted string in the outer receipt.
            # Do not silently turn them into null or a valid numeric observation.
            result[name] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            )
            result[name + "_encoding"] = JSON_TEXT
        else:
            result[name] = value
    return result


def read_raw_field(receipt, name):
    """Decode a raw, still-encoded envelope once; never rewrite its evidence."""
    return decode_record(receipt)[name]


def read_receipt(path):
    """Read either historical codec without changing bytes or their hash."""
    try:
        return read_receipt_bytes(Path(path).read_bytes())
    except ValueError as error:
        raise ValueError(f"invalid receipt {path}: {error}") from error


FIELDS = ("response", "generation_metadata", "timings")
GLOBAL_FIELDS = ("response", "generation_metadata")
GLOBAL_ENCODING = "python-json-nonfinite-v1"
FIELD_ENCODING = JSON_TEXT


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate receipt key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite value outside raw JSON text: " + value)


def read_receipt_bytes(data):
    record = json.loads(data, object_pairs_hook=_unique, parse_constant=_constant)
    if not isinstance(record, dict):
        raise ValueError("receipt must be an object")
    # Also reject overflowing finite JSON tokens such as 1e400 in the envelope.
    json.dumps(record, allow_nan=False)
    return decode_record(record)


def decode_record(record):
    """Decode a strict JSON envelope without mutating it or its raw bytes."""
    if not isinstance(record, dict):
        raise ValueError("receipt must be an object")
    json.dumps(record, allow_nan=False)
    markers = {name for name in record if name.endswith("_encoding")}
    known_markers = {"raw_json_encoding"} | {name + "_encoding" for name in FIELDS}
    if markers - known_markers:
        raise ValueError("unknown raw receipt encoding field")
    global_marker = "raw_json_encoding" in record
    tagged_fields = [name for name in FIELDS if name + "_encoding" in record]
    json_fields = [name for name in FIELDS if name + "_json" in record]
    result = dict(record)
    if global_marker:
        if tagged_fields or record["raw_json_encoding"] != GLOBAL_ENCODING:
            raise ValueError("mixed or unsupported raw receipt encoding")
        if not json_fields or any(name not in GLOBAL_FIELDS for name in json_fields):
            raise ValueError("invalid global raw receipt fields")
        for name in json_fields:
            if name not in record or record[name] is not None:
                raise ValueError("ambiguous raw receipt payload")
            text = record[name + "_json"]
            if not isinstance(text, str):
                raise ValueError("raw JSON payload must be text")
            result[name] = json.loads(text, object_pairs_hook=_unique)
    else:
        if json_fields:
            raise ValueError("unmarked or mixed raw JSON payload")
        for name in tagged_fields:
            if record[name + "_encoding"] != FIELD_ENCODING:
                raise ValueError("unsupported raw receipt field encoding")
            if not isinstance(record.get(name), str):
                raise ValueError("raw JSON payload must be text")
            result[name] = json.loads(record[name], object_pairs_hook=_unique)
    return result
