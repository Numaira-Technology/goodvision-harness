from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dataset_benchmark",
    Path(__file__).parents[1] / "tools" / "vlm_simple_jev_dataset_benchmark.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def sample(sid="v01", group="fall"):
    return {
        "sample_id": sid,
        "group": group,
        "endpoint": 2.0,
        "expected": {label: label == group for label in module.baseline.LABELS},
    }


def test_schedule_has_456_unique_requests_and_balances_categories():
    samples = [
        sample(f"v{i + 1:02d}", group)
        for i, group in enumerate(["fall"] * 10 + ["adl"] * 10 + ["smoking"] * 10)
    ]
    tasks = module.schedule(samples, 3)
    assert len(tasks) == 456
    assert len({(r, s["sample_id"], c) for r, s, c in tasks}) == 456
    assert sum(r == -2 for r, _, _ in tasks) == 1
    assert sum(r == -1 for r, _, _ in tasks) == 5
    assert [tasks[i][1]["group"] for i in (6, 11, 16)] == ["fall", "adl", "smoking"]
    assert len({tasks[i][2] for i in (6, 11, 16, 21, 26)}) == 5


@pytest.mark.parametrize(
    "change,reason",
    [
        ("duplicate", "duplicate_sample_id"),
        ("unfrozen", "dataset_not_frozen"),
        ("bad_reference", "invalid_reference"),
    ],
)
def test_reject_invalid_or_unfrozen_reference(tmp_path, change, reason):
    data = {"annotation_status": "verified_before_inference", "samples": [sample()]}
    if change == "duplicate":
        data["samples"].append(sample())
    elif change == "unfrozen":
        data["annotation_status"] = "provisional"
    else:
        data["samples"][0]["expected"]["fall"] = "true"
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=reason):
        module.read_dataset(path)


def test_same_timestamp_different_videos_do_not_merge_and_unknown_is_not_negative():
    samples = [sample(), sample("v02", "adl")]
    common = {
        "case": "simple_vision",
        "scope": "scene_events_only",
        "endpoint": 2.0,
        "warmup": False,
        "control": False,
        "repeat": 0,
        "total_seconds": 1,
        "complete_labels": True,
    }
    rows = [
        {
            **common,
            "sample_id": "v01",
            "ok": True,
            "decisions": {label: None for label in module.baseline.LABELS},
        },
        {**common, "sample_id": "v02", "ok": False},
        {
            **common,
            "sample_id": "v01",
            "ok": True,
            "control": True,
            "decisions": {label: True for label in module.baseline.LABELS},
        },
    ]
    result = module.summarize(rows, samples)["simple_vision"]
    assert result["attempts"] == 2 and result["successes"] == 1
    assert set(result["by_sample"]) == {"v01", "v02"}
    fall = result["confusion"]["fall"]
    assert fall["unknown_positive"] == 1 and fall["request_error"] == 1
    assert fall["reference_positive"] == fall["reference_negative"] == 1
    assert fall["tp"] == fall["tn"] == fall["fn"] == fall["fp"] == 0
