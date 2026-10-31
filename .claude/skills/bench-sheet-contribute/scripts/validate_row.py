#!/usr/bin/env python3
"""Offline validator for a single contributor row before it goes on the shared bench sheet.

The sheet is PIVOTED: one row per model, with dataset size as a top band. A row is 25
values in order A->Y (see ../references/column-contract.md). The three dataset blocks are
50 (G-K), 100 (L-P) and all-120 (Q-U); each block is Overall + Fight/Smoking/Fall/Weapon.
A block you did not run is left blank; at least one block must carry an Overall %.

Checks: column count, cell types, percentages in range, required cells, the per-subtask
notes, and per-block consistency (a block with subtask %s must have its Overall %). Pure
stdlib, no network, no credentials, no writes. Run with -I.

Usage:
  python3 -I validate_row.py my_row.json
  cat my_row.json | python3 -I validate_row.py -

my_row.json shape:
  {
    "values": [<A>, <B>, ..., <Y>],     # exactly 25, blanks as ""
    "notes":  {"H": "...", "M": "..."}  # one per NON-BLANK subtask cell (H-K, M-P, R-U)
  }
Exit code 0 = OK, 1 = problems found (printed to stderr).
"""
import json
import re
import sys

# col -> (key, kind). kind: text | backend | optnum (blank ok) | pct_opt (blank ok,
# else 0..100) | date | status_run | status_val
CONTRACT = [
    ("A", "model",          "text"),
    ("B", "backend",        "backend"),
    ("C", "run_status",     "status_run"),
    ("D", "validation",     "status_val"),
    ("E", "p50_s",          "optnum"),
    ("F", "p95_s",          "optnum"),
    ("G", "d50_overall",    "pct_opt"),
    ("H", "d50_fight",      "pct_opt"),
    ("I", "d50_smoking",    "pct_opt"),
    ("J", "d50_fall",       "pct_opt"),
    ("K", "d50_weapon",     "pct_opt"),
    ("L", "d100_overall",   "pct_opt"),
    ("M", "d100_fight",     "pct_opt"),
    ("N", "d100_smoking",   "pct_opt"),
    ("O", "d100_fall",      "pct_opt"),
    ("P", "d100_weapon",    "pct_opt"),
    ("Q", "d120_overall",   "pct_opt"),
    ("R", "d120_fight",     "pct_opt"),
    ("S", "d120_smoking",   "pct_opt"),
    ("T", "d120_fall",      "pct_opt"),
    ("U", "d120_weapon",    "pct_opt"),
    ("V", "hardware",       "text"),
    ("W", "price_per_mtok", "optnum"),
    ("X", "spec_ver",       "text"),
    ("Y", "run_date",       "date"),
]
# dataset id -> (overall col, [subtask cols])
BLOCKS = {
    "50":  ("G", ["H", "I", "J", "K"]),
    "100": ("L", ["M", "N", "O", "P"]),
    "120": ("Q", ["R", "S", "T", "U"]),
}
NOTE_COLS = [c for _, subs in BLOCKS.values() for c in subs]
BACKENDS = {"language", "decision", "vision"}
RUN_VALUES = {"yes", "running", "no"}
VAL_VALUES = {"yes", "no"}
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
        errs.append("expected %d values (A->Y), got %d — send blanks as \"\", never drop a column"
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
        elif kind == "status_run":
            if is_blank(v) or str(v).strip().lower() not in RUN_VALUES:
                errs.append("%s must be one of %s (lowercase)" % (tag, sorted(RUN_VALUES)))
        elif kind == "status_val":
            if is_blank(v) or str(v).strip().lower() not in VAL_VALUES:
                errs.append("%s must be one of %s (lowercase) — 'yes' only for decision/clip models"
                            % (tag, sorted(VAL_VALUES)))

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
