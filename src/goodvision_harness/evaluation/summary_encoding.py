"""Lossless window tables and shared identifier fragments for final-answer evidence."""
from collections import Counter
import json
from os.path import commonprefix
import re

ENCODING = "lossless_window_tables_v1"
COLUMN_ENCODING = "lossless_window_columns_v2"
UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
MARKERS = {"$string", "$join", "$table", "$blocks", "$columns"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def expand_window_evidence(value):
    if not isinstance(value, dict) or value.get("encoding") not in {ENCODING, COLUMN_ENCODING}:
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
        if tag == "$columns":
            columns, count, specs = item[1:]
            values = [_expand_column(spec, count, expand) for spec in specs]
            if len(columns) != len(values) or any(len(column) != count for column in values):
                raise ValueError("Window column length differs from its record count")
            return [{key: values[column][row] for column, key in enumerate(columns)}
                    for row in range(count)]
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


def _expand_column(spec, count, expand):
    kind = spec[0]
    if kind == "values":
        return [expand(cell) for cell in spec[1]]
    if kind == "constant":
        return [expand(spec[1]) for _ in range(count)]
    if kind == "dictionary":
        return [expand(spec[1][index]) for index in spec[2]]
    if kind == "default":
        values = [expand(spec[1]) for _ in range(count)]
        for index, cell in spec[2]:
            values[index] = expand(cell)
        return values
    if kind == "affix":
        return [spec[1] + cell + spec[2]
                for cell in _expand_column(spec[3], count, expand)]
    if kind == "integer_strings":
        return [str(spec[1] + index * spec[2]) for index in range(count)]
    raise ValueError("Unknown lossless column notation")


def _compact_column(values, encode, *, affixes=True):
    """Share exact column values and ID templates, never infer observation states."""
    encoded = [encode(value) for value in values]
    keys = [_json(value) for value in values]
    if len(set(keys)) == 1:
        return ["constant", encoded[0]]
    candidates = [["values", encoded]]
    indexes, definitions, codes = {}, [], []
    for key, cell in zip(keys, encoded):
        if key not in indexes:
            indexes[key] = len(definitions)
            definitions.append(cell)
        codes.append(indexes[key])
    candidates.append(["dictionary", definitions, codes])
    default = Counter(keys).most_common(1)[0][0]
    candidates.append(["default", encoded[keys.index(default)],
                       [[i, cell] for i, (key, cell) in enumerate(zip(keys, encoded)) if key != default]])
    if all(isinstance(value, str) for value in values):
        if all(re.fullmatch(r"-?(?:0|[1-9][0-9]{0,15})", value) for value in values):
            start, step = int(values[0]), int(values[1]) - int(values[0])
            if all(str(start + i * step) == value for i, value in enumerate(values)):
                candidates.append(["integer_strings", start, step])
        if affixes:
            prefix = commonprefix(values)
            tails = [value[len(prefix):] for value in values]
            suffix = commonprefix([tail[::-1] for tail in tails])[::-1]
            if prefix or suffix:
                middles = [tail[:-len(suffix)] if suffix else tail for tail in tails]
                candidates.append(["affix", prefix, suffix,
                                   _compact_column(middles, encode, affixes=False)])
    return min(candidates, key=lambda candidate: len(_json(candidate)))


def compact_window_evidence(value, *, columnar=False):
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
        if columnar:
            return ["$columns", columns, len(rows),
                    [_compact_column([row[key] for row in rows], encode) for key in columns]]
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

    packed = {"encoding": COLUMN_ENCODING if columnar else ENCODING,
              "strings": strings, "evidence": encode(value),
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
    if columnar:
        packed["record_format"] += (
            ' ["$columns",names,count,specs] stores count separate objects in source order; '
            'specs has one column specification per name. ["values",cells] lists every value; '
            '["constant",v] repeats v count times; ["dictionary",values,indexes] selects '
            'values[indexes[row]]; ["default",v,overrides] repeats v except at each '
            '[zero_based_row,value] override. ["affix",prefix,suffix,middles] concatenates '
            'prefix + each expanded middle + suffix, without separators. '
            '["integer_strings",start,step] means the exact decimal strings of '
            'start + row*step, not numeric observations. Expand nested encoded values normally. '
            'A stored default is an exact repeated source value, never an inferred missing value. '
            'false, null, empty lists, absent fields and every occurrence remain distinct.'
        )
    if _json(expand_window_evidence(packed)) != _json(value):
        raise ValueError("Window tables cannot reconstruct their input")
    return packed
