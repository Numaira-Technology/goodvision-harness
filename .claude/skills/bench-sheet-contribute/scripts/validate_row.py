#!/usr/bin/env python3
"""Offline validator for a single contributor row before it goes on the shared bench sheet.

Checks a row payload against the `results`-tab column contract (see
../references/column-contract.md): 17 values in order A->Q, correct cell types,
percentages in range, required cells present, and the four per-subtask notes.
Pure stdlib, no network, no credentials, no writes. Run with -I.

Usage:
  python3 -I validate_row.py my_row.json
  cat my_row.json | python3 -I validate_row.py -

my_row.json shape:
  {
    "values": [<A>, <B>, ..., <Q>],          # exactly 17, blanks as ""
    "notes":  {"G": "...", "H": "...", "I": "...", "J": "..."}
  }
Exit code 0 = OK, 1 = problems found (printed to stderr).
"""
import json
import re
import sys

# col -> (key, kind). kind: text | num | pct | bool | optnum (blank allowed)
CONTRACT = [
    ("A", "model",          "text"),
    ("B", "backend",        "backend"),
    ("C", "single_inf_s",   "optnum"),
    ("D", "p50_s",          "optnum"),
    ("E", "p95_s",          "optnum"),
    ("F", "overall_pct",    "pct"),
    ("G", "fight_pct",      "pct_opt"),
    ("H", "smoking_pct",    "pct_opt"),
    ("I", "fall_pct",       "pct_opt"),
    ("J", "weapon_pct",     "pct_opt"),
    ("K", "hardware",       "text"),
    ("L", "price_per_mtok", "optnum"),
    ("M", "dataset_n",      "posint"),
    ("N", "spec_ver",       "text"),
    ("O", "run_date",       "date"),
    ("P", "provisional",    "bool"),
    ("Q", "source",         "text"),
    ("R", "run_status",     "status_run"),
    ("S", "validation",     "status_val"),
]
NOTE_COLS = ["G", "H", "I", "J"]
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
        errs.append("expected %d values (A->S), got %d — send blanks as \"\", never drop a column"
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
        elif kind in ("optnum",):
            if not is_blank(v):
                try:
                    n = as_num(v)
                    if n < 0:
                        errs.append("%s is negative (%s)" % (tag, v))
                except ValueError:
                    errs.append("%s must be a number or blank, got %r" % (tag, v))
        elif kind in ("pct", "pct_opt"):
            if is_blank(v):
                if kind == "pct":
                    errs.append("%s (overall accuracy) is required and blank" % tag)
                else:
                    warns.append("%s blank — class not run? (ok if intended)" % tag)
            else:
                try:
                    n = as_num(v)
                    if not (0.0 <= n <= 100.0):
                        errs.append("%s = %s is outside 0..100" % (tag, v))
                except ValueError:
                    errs.append("%s must be a number 0..100, got %r" % (tag, v))
        elif kind == "posint":
            try:
                n = as_num(v)
                if n <= 0 or n != int(n):
                    errs.append("%s must be a positive integer, got %r" % (tag, v))
            except (ValueError, TypeError):
                errs.append("%s (Dataset n) is required as a positive integer, got %r" % (tag, v))
        elif kind == "date":
            if is_blank(v) or not DATE_RE.match(str(v).strip()):
                errs.append("%s must be YYYY-MM-DD, got %r" % (tag, v))
        elif kind == "bool":
            ok = isinstance(v, bool) or str(v).strip().upper() in ("TRUE", "FALSE")
            if not ok:
                errs.append("%s must be TRUE/FALSE, got %r" % (tag, v))
        elif kind == "status_run":
            if is_blank(v) or str(v).strip().lower() not in RUN_VALUES:
                errs.append("%s must be one of %s (lowercase)" % (tag, sorted(RUN_VALUES)))
        elif kind == "status_val":
            if is_blank(v) or str(v).strip().lower() not in VAL_VALUES:
                errs.append("%s must be one of %s (lowercase) — 'yes' only for decision/clip models"
                            % (tag, sorted(VAL_VALUES)))

    # per-subtask notes: required whenever that class % is present
    by_col = {col: v for (col, _, _), v in zip(CONTRACT, values)}
    for col in NOTE_COLS:
        pct_present = not is_blank(by_col.get(col))
        note = notes.get(col)
        if pct_present and is_blank(note):
            errs.append("%s has a %% but no cell note (need x/N + TP/FN/TN/FP)" % col)
        elif pct_present and not NOTE_RE.search(str(note)):
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
