"""Keep nonfinite raw evidence inside strict JSON receipts.

Only diagnostic fields use this codec; scores, plans, ledgers and hashes retain
strict JSON. The text is a re-encoding of decoded values, not original wire bytes.
Finite raw fields keep their existing structure and need no encoding marker.
"""

import json

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
    """Decode an explicitly tagged raw field for replay; reject unknown codecs."""
    encoding = receipt.get(name + "_encoding")
    if encoding is None:
        return receipt[name]
    if encoding != JSON_TEXT or not isinstance(receipt.get(name), str):
        raise ValueError("invalid raw receipt field encoding: " + name)
    return json.loads(receipt[name])
