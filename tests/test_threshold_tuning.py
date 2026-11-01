"""Threshold analysis must preserve v2 eligibility and a held-out test set."""

import copy
import hashlib
import json
import sys

import pytest

from tools.event_benchmark import tune_threshold as tuner


def run(prefix):
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    questions = []
    for event in tuner.EVENTS:
        for gt, score in (("yes", 0.8), ("no", 0.2)):
            name = f"{prefix}-{event}-{gt}.mp4"
            questions.append({"number": len(questions) + 1, "filename": name,
                              "sha256": digest(name), "event": event, "gt": gt,
                              "score": score, "frames_planned": 2,
                              "frames_ok": 2, "frames_failed": 0})
    return {"spec_version": "v2", "backend": "decision", "model": "model-a",
            "samples_json_sha256": digest(prefix + "samples"),
            "manifest_sha256": digest(prefix + "manifest"),
            "runner_sha256": digest("runner"), "spec_sha256": digest("spec"),
            "config": {"gate": 0.5, "interval": 1.0, "frame_rule": "fixed",
                       "frame_width": 720, "temperature": 0, "retries": 3,
                       "timeout": 90, "decision": "max-frame-score >= gate"},
            "overall": {"attempted": 8, "total": 8, "errors": 0},
            "per_video": questions}


def check(data, **kwargs):
    return tuner.validate_run(data["per_video"], data, **kwargs)


@pytest.mark.parametrize("score", [None, True, "0.8", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(score):
    data = run("val")
    data["per_video"][0]["score"] = score
    with pytest.raises(ValueError, match="probabilities"):
        check(data, validation=True)


@pytest.mark.parametrize("change", ["historical", "language", "manifest", "gt", "partial", "missing", "duplicate", "totals"])
def test_ineligible_validation_is_rejected(change):
    data = run("val")
    row = data["per_video"][0]
    if change == "historical":
        data["spec_version"] = "v1"
    elif change == "language":
        data["backend"] = "language"
    elif change == "manifest":
        data.pop("manifest_sha256")
    elif change == "gt":
        row["gt"] = "unknown"
    elif change == "partial":
        row["frames_failed"] = 1
    elif change == "missing":
        row.pop("score")
        row["error"] = "missing file"
    elif change == "duplicate":
        data["per_video"][1] = copy.deepcopy(row)
    else:
        data["overall"]["attempted"] = 9
    with pytest.raises(ValueError):
        check(data, validation=True)


@pytest.mark.parametrize("boundary", ["filename", "renamed_media", "model", "sampling", "runner"])
def test_pair_rejects_overlap_or_incompatible_contract(boundary):
    val, test = run("val"), run("test")
    if boundary == "filename":
        test["per_video"][0]["filename"] = val["per_video"][0]["filename"]
    elif boundary == "renamed_media":
        test["per_video"][0]["sha256"] = val["per_video"][0]["sha256"]
    elif boundary == "model":
        test["model"] = "model-b"
    elif boundary == "sampling":
        test["config"]["interval"] = 4.0
    else:
        test["runner_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        tuner.validate_pair(val["per_video"], val, test["per_video"], test)


def test_each_validation_event_requires_both_ground_truth_classes():
    data = run("val")
    data["per_video"][1]["gt"] = "yes"
    with pytest.raises(ValueError, match="positive and negative"):
        check(data, validation=True)


def test_cli_reports_test_errors_and_keeps_fixed_gate_comparison(tmp_path, monkeypatch):
    val, test = run("val"), run("test")
    test["per_video"][0].pop("score")
    test["per_video"][0]["error"] = "provider failed"
    test["overall"].update(total=7, errors=1)
    paths = [tmp_path / name for name in ("val.json", "test.json", "report.json")]
    paths[0].write_text(json.dumps(val))
    paths[1].write_text(json.dumps(test))
    monkeypatch.setattr(sys, "argv", ["tune_threshold", "--val", str(paths[0]),
                                     "--test", str(paths[1]), "--out", str(paths[2])])
    tuner.main()
    report = json.loads(paths[2].read_text())
    result = report["test"][str(paths[1])]
    assert result["counts"] == {"attempted": 8, "total": 7, "errors": 1}
    assert result["overall_at_tau"]["n"] == result["overall_at_0.5"]["n"] == 7
    assert report["val_counts"]["total"] == 8


def test_cli_rejects_identical_val_and_test_before_writing(tmp_path, monkeypatch):
    source, output = tmp_path / "run.json", tmp_path / "report.json"
    source.write_text(json.dumps(run("shared")))
    monkeypatch.setattr(sys, "argv", ["tune_threshold", "--val", str(source),
                                     "--test", str(source), "--out", str(output)])
    with pytest.raises(SystemExit) as error:
        tuner.main()
    assert error.value.code == 2
    assert not output.exists()
