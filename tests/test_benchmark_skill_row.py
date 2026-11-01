"""Offline contributor-row regression fixtures; no Sheets credentials or writes."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".claude/skills/bench-sheet-contribute/scripts/validate_row.py"
)
spec = importlib.util.spec_from_file_location("benchmark_row", SCRIPT)
row = importlib.util.module_from_spec(spec)
spec.loader.exec_module(row)


def fixture():
    values = [""] * 23
    for col, value in {
        "A": "test / v2 / fixed",
        "B": "language",
        "E": 90.0,
        "F": 100.0,
        "G": 100.0,
        "H": 75.0,
        "I": 100.0,
        "T": "synthetic",
        "V": "v2 fixed",
        "W": "2026-11-01",
    }.items():
        values[ord(col) - 65] = value
    notes = {
        "E": "overall 45/50 · err0 · attempted50 · provisional=no · source=synthetic",
        "F": "fight 10/10 · TP5 FN0 TN5 FP0 · err0 · attempted10 · provisional=no",
        "G": "smoking 10/10 · TP10 FN0 TN0 FP0 · err0 · attempted10 "
        "· provisional=no · no negatives",
        "H": "fall 15/20 · TP5 FN3 TN10 FP2 · err0 · attempted20 · provisional=no",
        "I": "weapon 10/10 · TP10 FN0 TN0 FP0 · err0 · attempted10 "
        "· provisional=no · no negatives",
    }
    return {"values": values, "notes": notes}


def test_valid_offline_row_and_metadata_notes():
    payload = fixture()
    payload["values"][2:4] = [0.5, 1.0]
    payload["values"][20] = 0.01
    payload["notes"].update(
        C="complete @12 round-trip", D="same run", U="USD/run synthetic"
    )
    assert row.check(payload) == ([], [])


@pytest.mark.parametrize(
    "col,value",
    [
        ("A", {}),
        ("T", ["GPU"]),
        ("V", 2),
        ("B", "threshold"),
        ("C", float("nan")),
        ("D", float("inf")),
        ("U", -1),
        ("E", "90"),
        ("E", True),
        ("E", 101),
        ("E", float("nan")),
        ("E", []),
        ("W", "2026-99-99"),
        ("W", "2026-02-29"),
        ("W", "20261101"),
    ],
)
def test_bad_types_dates_numbers_are_errors(col, value):
    payload = fixture()
    payload["values"][ord(col) - 65] = value
    assert row.check(payload)[0]


@pytest.mark.parametrize(
    "notes", [[], ["F"], "bad", None, {"Z": "outside"}, {"F": 123}]
)
def test_bad_notes_never_crash(notes):
    payload = fixture()
    payload["notes"] = notes
    assert row.check(payload)[0]


@pytest.mark.parametrize(
    "old,new",
    [
        ("10/10", "100/2"),
        ("10/10", "0/10"),
        ("TP5", "TP4"),
        ("attempted10", "attempted11"),
        ("err0", ""),
        ("provisional=no", ""),
        ("10/10", "10/10 and 1/2"),
    ],
)
def test_subtask_note_math_must_match(old, new):
    payload = fixture()
    payload["notes"]["F"] = payload["notes"]["F"].replace(old, new)
    assert row.check(payload)[0]


def test_errors_stay_outside_valid_but_inside_attempted():
    payload = fixture()
    payload["values"][4] = round(100 * 44 / 49, 1)
    payload["values"][5] = 100.0
    payload["notes"]["E"] = "overall 44/49 · err1 · attempted50 · provisional=no"
    payload["notes"]["F"] = (
        "fight 9/9 · TP4 FN0 TN5 FP0 · err1 · attempted10 · provisional=no"
    )
    assert row.check(payload) == ([], [])


def test_missing_negatives_must_be_disclosed():
    payload = fixture()
    payload["notes"]["G"] = payload["notes"]["G"].replace("no negatives", "")
    assert row.check(payload)[0]


def test_overall_cannot_hide_provisional_subtask():
    payload = fixture()
    payload["notes"]["F"] = payload["notes"]["F"].replace(
        "provisional=no", "provisional=yes"
    )
    assert row.check(payload)[0]


def test_other_dataset_blocks_need_overall_and_matching_cohort():
    payload = fixture()
    payload["values"][10] = 100.0
    payload["notes"]["K"] = payload["notes"]["F"]
    assert row.check(payload)[0]
    payload = fixture()
    payload["notes"]["E"] = "overall 90/100 · err0 · attempted100 · provisional=no"
    assert row.check(payload)[0]


def test_p95_and_missing_timing_notes():
    payload = fixture()
    payload["values"][2:4] = [2.0, 1.0]
    assert row.check(payload)[0]
    payload["values"][2:4] = [1.0, 2.0]
    assert row.check(payload)[0]


@pytest.mark.parametrize(
    "raw",
    ["{", '{"values": [], "values": []}', '{"values": NaN}', '{"values": Infinity}'],
)
def test_cli_invalid_json_exits_cleanly(raw):
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "-"],
        input=raw,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_dry_run_and_no_input_mutation(tmp_path):
    payload = fixture()
    original = copy.deepcopy(payload)
    assert not row.check(payload)[0]
    assert payload == original
    path = tmp_path / "row.json"
    path.write_text(json.dumps(payload))
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "offline" in result.stdout
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "old,new",
    [
        ("TP5", "TP5.9"),
        ("err0", "err0.5"),
        ("attempted10", "attempted10.9"),
        ("TN5", "TN5/6"),
        ("FN0", "FN0e2"),
    ],
)
def test_count_tokens_reject_decimal_or_partial_numbers(old, new):
    payload = fixture()
    payload["notes"]["F"] = payload["notes"]["F"].replace(old, new)
    assert row.check(payload)[0]


def test_all120_subtasks_cannot_exceed_overall():
    payload = fixture()
    payload["values"][14:19] = [100.0] * 5
    payload["notes"]["O"] = "overall 120/120 · err0 · attempted120 · provisional=no"
    for col in "PQRS":
        payload["notes"][col] = (
            "event 100/100 · TP50 FN0 TN50 FP0 · err0 · attempted100 · provisional=no"
        )
    assert any("exceed overall" in e for e in row.check(payload)[0])


def test_huge_numeric_cell_is_rejected_without_exception():
    payload = fixture()
    payload["values"][2] = 10**1000
    assert row.check(payload)[0]
