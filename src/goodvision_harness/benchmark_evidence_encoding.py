"""Lossless shared-value and record-schema encoding from the offline benchmark."""

import copy
from collections import Counter
import json
import re

ENCODING = "lossless_value_dictionary_v1"
REFERENCE = "$value_ref"
SCHEMA_ENCODING = "lossless_schema_dictionary_v1"


def expand_repeated_evidence(value):
    if isinstance(value, dict) and value.get("encoding") == SCHEMA_ENCODING:
        value = _expand_schema_evidence(value)
    if not isinstance(value, dict) or value.get("encoding") != ENCODING:
        return value
    definitions = value["value_definitions"]
    memo, visiting = {}, set()

    def expand(item):
        if isinstance(item, dict):
            if set(item) == {REFERENCE}:
                index = item[REFERENCE]
                if type(index) is not int or not 0 <= index < len(definitions) or index in visiting:
                    raise ValueError("Invalid evidence dictionary reference")
                if index not in memo:
                    visiting.add(index)
                    memo[index] = expand(definitions[index])
                    visiting.remove(index)
                return copy.deepcopy(memo[index])
            return {key: expand(child) for key, child in item.items()}
        if isinstance(item, list):
            return [expand(child) for child in item]
        return item

    return expand(value["encoded_evidence"])


def compact_repeated_evidence(value):
    counts = Counter()

    def key_of(item):
        if not isinstance(item, (str, list, dict)):
            return None
        text = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) >= (32 if isinstance(item, str) else 80) else None

    def collect(item):
        if isinstance(item, dict) and set(item) == {REFERENCE}:
            raise ValueError("Input collides with reserved evidence dictionary reference")
        key = key_of(item)
        if key is not None:
            counts[key] += 1
        children = item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        for child in children:
            collect(child)

    collect(value)
    repeated = {key for key, n in counts.items() if n >= 2 and (n - 1) * (len(key) - 20) > 80}
    indexes, definitions = {}, []

    def encode(item):
        key = key_of(item)
        if key in indexes:
            return {REFERENCE: indexes[key]}
        if isinstance(item, dict):
            encoded = {name: encode(child) for name, child in item.items()}
        elif isinstance(item, list):
            encoded = [encode(child) for child in item]
        else:
            encoded = item
        if key in repeated:
            indexes[key] = len(definitions)
            definitions.append(encoded)
            return {REFERENCE: indexes[key]}
        return encoded

    encoded = encode(value)
    packed = {
        "encoding": ENCODING,
        "record_format": (
            'LOSSLESS shared-value notation: replace every object {"$value_ref": i} with '
            'value_definitions[i], recursively, then read encoded_evidence as the original '
            'evidence. Definitions only refer to earlier definitions. Each reference is a '
            'full occurrence in its own window/time/position, NOT an omitted or merged '
            'observation. After reference expansion, {columns,rows} tables use their listed '
            'column order as usual. No source text, sample, timestamp, numeric precision or '
            'record has been deleted or paraphrased. Questions are outside this encoding.'
        ),
        "value_definitions": definitions,
        "encoded_evidence": encoded,
    }
    # Retain the original codec round-trip check before selecting this representation.
    if expand_repeated_evidence(packed) != value:
        raise ValueError("Evidence dictionary cannot reconstruct its input")
    return packed


def _expand_schema_evidence(value):
    schemas, strings = value["record_schemas"], value["string_definitions"]

    def lookup(items, index):
        if not 0 <= index < len(items):
            raise ValueError("Invalid lossless schema reference")
        return items[index]

    def expand(item):
        if isinstance(item, str) and re.fullmatch(r"\$s[0-9]+", item):
            return lookup(strings, int(item[2:]))
        if isinstance(item, list):
            if item and isinstance(item[0], str) and re.fullmatch(r"\$r[0-9]+", item[0]):
                columns = lookup(schemas, int(item[0][2:]))
                if len(item) - 1 != len(columns) or len(set(columns)) != len(columns):
                    raise ValueError("Invalid lossless schema row")
                return {key: expand(cell) for key, cell in zip(columns, item[1:])}
            return [expand(cell) for cell in item]
        if isinstance(item, dict):
            return {key: expand(cell) for key, cell in item.items()}
        return item

    return expand(value["encoded_evidence"])


def compact_schema_evidence(value):
    """Share record keys and labels while preserving structure, order and values."""
    counts = Counter()

    def collect(item):
        if isinstance(item, str):
            if re.fullmatch(r"\$[rs][0-9]+", item):
                raise ValueError("Input collides with reserved schema notation")
            counts[item] += 1
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    strings = [text for text, count in counts.items()
               if count >= 2 and (count - 1) * (len(text) - 8) > 12]
    string_ids = {text: index for index, text in enumerate(strings)}
    schemas, schema_ids = [], {}

    def encode(item):
        if isinstance(item, str) and item in string_ids:
            return "$s" + str(string_ids[item])
        if isinstance(item, dict):
            columns = tuple(item)
            if columns not in schema_ids:
                schema_ids[columns] = len(schemas)
                schemas.append(list(columns))
            return ["$r" + str(schema_ids[columns]), *[encode(cell) for cell in item.values()]]
        if isinstance(item, list):
            return [encode(cell) for cell in item]
        return item

    encoded = encode(value)
    packed = {
        "encoding": SCHEMA_ENCODING,
        "record_format": (
            'LOSSLESS record notation: an array beginning with "$rN" is an object whose '
            'remaining values follow record_schemas[N]. A string "$sN" stands for '
            'string_definitions[N]. Other arrays/scalars retain their exact meaning. '
            'Expand these recursively to read encoded_evidence, then follow the inner '
            'lossless shared-value definitions and columns/rows tables. Every window, '
            'timestamp, object, description and numeric value is retained, without rounding '
            'or paraphrase. This notation does not join identities or combine observations.'
        ),
        "record_schemas": schemas,
        "string_definitions": strings,
        "encoded_evidence": encoded,
    }
    if _expand_schema_evidence(packed) != value:
        raise ValueError("Evidence schemas cannot reconstruct their input")
    return packed
