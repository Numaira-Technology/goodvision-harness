#!/usr/bin/env python3
"""Offline validator for a single contributor row before it goes on the shared bench sheet.

The sheet is PIVOTED and has NO tracking columns: one row per model, with dataset size as a top
band. A row is 23 values in order A->W (see ../references/column-contract.md). The three dataset
blocks are 50 (E-I), 100 (J-N) and all-120 (O-S); each block is Overall + Fight/Smoking/Fall/Weapon.
A block you did not run is left blank; at least one block must carry an Overall %.

Checks: column count, cell types, percentages in range, required cells, the per-subtask notes, and
per-block consistency (a block with subtask %s must have its Overall %). Pure stdlib, no network,
no credentials, no writes. Run with -I.

Usage:
  python3 -I validate_row.py my_row.json
  cat my_row.json | python3 -I validate_row.py -

my_row.json shape:
  {
    "values": [<A>, <B>, ..., <W>],     # exactly 23, blanks as ""
    "notes":  {"F": "...", "K": "..."}  # one per NON-BLANK subtask cell (F-I, K-N, P-S)
  }
Exit code 0 = OK, 1 = problems found (printed to stderr).
"""
import json
import re
import sys

# col -> (key, kind). kind: text | backend | optnum (blank ok) | pct_opt (blank ok, else 0..100) | date
CONTRACT = [
    ("A", "model",          "text"),
    ("B", "backend",        "backend"),
    ("C", "p50_s",          "optnum"),
    ("D", "p95_s",          "optnum"),
    ("E", "d50_overall",    "pct_opt"),
    ("F", "d50_fight",      "pct_opt"),
    ("G", "d50_smoking",    "pct_opt"),
    ("H", "d50_fall",       "pct_opt"),
    ("I", "d50_weapon",     "pct_opt"),
    ("J", "d100_overall",   "pct_opt"),
    ("K", "d100_fight",     "pct_opt"),
    ("L", "d100_smoking",   "pct_opt"),
    ("M", "d100_fall",      "pct_opt"),
    ("N", "d100_weapon",    "pct_opt"),
    ("O", "d120_overall",   "pct_opt"),
    ("P", "d120_fight",     "pct_opt"),
    ("Q", "d120_smoking",   "pct_opt"),
    ("R", "d120_fall",      "pct_opt"),
    ("S", "d120_weapon",    "pct_opt"),
    ("T", "hardware",       "text"),
    ("U", "price",          "optnum"),
    ("V", "spec_ver",       "text"),
    ("W", "run_date",       "date"),
]
# dataset id -> (overall col, [subtask cols])
BLOCKS = {
    "50":  ("E", ["F", "G", "H", "I"]),
    "100": ("J", ["K", "L", "M", "N"]),
    "120": ("O", ["P", "Q", "R", "S"]),
}
NOTE_COLS = [c for _, subs in BLOCKS.values() for c in subs]
BACKENDS = {"language", "decision", "vision"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NOTE_RE = re.compile(r"\b\d+/\d+\b")  # expects an x/N somewhere in the note


def is_blank(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def as_num(v):
    if isinstance(v, bool):
        raise ValueError("bool, not number")
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).strip())


def check(payload):
    errs, warns = [], []
    if not isinstance(payload, dict):
        return ["payload must be a JSON object with 'values' and 'notes'"], []
    values = payload.get("values")
    notes = payload.get("notes", {}) or {}

    if not isinstance(values, list):
        return ["'values' must be a list"], []
    if len(values) != len(CONTRACT):
        errs.append("expected %d values (A->W), got %d — send blanks as \"\", never drop a column"
                    % (len(CONTRACT), len(values)))
        return errs, warns

    for (col, key, kind), v in zip(CONTRACT, values):
        tag = "%s (%s)" % (col, key)
        if kind == "text":
            if is_blank(v):
                errs.append("%s is required and blank" % tag)
        elif kind == "backend":
            if is_blank(v) or str(v).strip().lower() not in BACKENDS:
                errs.append("%s must be one of %s" % (tag, sorted(BACKENDS)))
        elif kind == "optnum":
            if not is_blank(v):
                try:
                    if as_num(v) < 0:
                        errs.append("%s is negative (%s)" % (tag, v))
                except ValueError:
                    errs.append("%s must be a number or blank, got %r" % (tag, v))
        elif kind == "pct_opt":
            if not is_blank(v):
                try:
                    n = as_num(v)
                    if not (0.0 <= n <= 100.0):
                        errs.append("%s = %s is outside 0..100" % (tag, v))
                except ValueError:
                    errs.append("%s must be a number 0..100 or blank, got %r" % (tag, v))
        elif kind == "date":
            if is_blank(v) or not DATE_RE.match(str(v).strip()):
                errs.append("%s must be YYYY-MM-DD, got %r" % (tag, v))

    by_col = {col: v for (col, _, _), v in zip(CONTRACT, values)}

    # per-block consistency + at least one block filled
    filled_blocks = 0
    for ds, (ov_col, sub_cols) in BLOCKS.items():
        ov_present = not is_blank(by_col.get(ov_col))
        sub_present = [c for c in sub_cols if not is_blank(by_col.get(c))]
        if ov_present:
            filled_blocks += 1
            if not sub_present:
                warns.append("dataset %s has Overall %% but no subtask %%s (ok if intended)" % ds)
        elif sub_present:
            errs.append("dataset %s has subtask %%s (%s) but Overall %% (%s) is blank"
                        % (ds, ",".join(sub_present), ov_col))
    if filled_blocks == 0:
        errs.append("no dataset block filled — at least one of 50/100/120 needs an Overall %")

    # per-subtask notes: required whenever that subtask % is present
    for col in NOTE_COLS:
        if is_blank(by_col.get(col)):
            continue
        note = notes.get(col)
        if is_blank(note):
            errs.append("%s has a %% but no cell note (need x/N + TP/FN/TN/FP)" % col)
        elif not NOTE_RE.search(str(note)):
            warns.append("%s note has no x/N pattern: %r" % (col, note))

    extra = set(notes) - set(NOTE_COLS)
    if extra:
        warns.append("notes for non-subtask columns ignored by the sheet: %s" % sorted(extra))

    return errs, warns


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "-"
    raw = sys.stdin.read() if arg == "-" else open(arg).read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print("invalid JSON: %s" % e, file=sys.stderr)
        sys.exit(1)

    errs, warns = check(payload)
    for w in warns:
        print("WARN: %s" % w, file=sys.stderr)
    if errs:
        for e in errs:
            print("ERROR: %s" % e, file=sys.stderr)
        print("\n%d error(s) — fix before writing to the sheet." % len(errs), file=sys.stderr)
        sys.exit(1)
    print("OK: row matches the contract (%d values%s)."
          % (len(payload["values"]), ", %d warning(s)" % len(warns) if warns else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
