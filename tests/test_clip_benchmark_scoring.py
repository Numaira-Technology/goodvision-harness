"""Protect the frozen-threshold contract without model dependencies."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def scoring():
    path = Path(__file__).resolve().parents[1] / "tools/clip_benchmark/scoring.py"
    spec = importlib.util.spec_from_file_location("clip_benchmark_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("invalid", [None, {}, [], True, "0.1", float("nan"), float("inf")])
def test_evaluation_rejects_invalid_thresholds_before_writing_reports(scoring, tmp_path, invalid):
    thresholds = dict.fromkeys(scoring.EVENTS, 0.1)
    thresholds["fight"] = invalid
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="frozen threshold"):
        scoring.evaluate(tmp_path / "missing.jsonl", [], thresholds, output, "test/model")
    assert not output.exists()


@pytest.mark.parametrize("thresholds", [{}, {"fight": 0.1}, None, []])
def test_evaluation_rejects_incomplete_threshold_handoff(scoring, tmp_path, thresholds):
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="[Ff]rozen threshold"):
        scoring.evaluate(tmp_path / "missing.jsonl", [], thresholds, output, "test/model")
    assert not output.exists()


def test_evaluation_preserves_frozen_decision_and_explicit_zero_bag_thresholds(scoring, tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "filename": "synthetic.mp4", "quality_reference": True,
        "predictions": {"fight": {"margin": 0.2}, "bag_knife": {"margin": 0.1}},
        "errors": None,
    }) + "\n")
    samples = [{"filename": "synthetic.mp4", "event": event, "proposed_answer": answer}
               for event, answer in (("fight", "no"), ("bag_knife", "yes"))]
    thresholds = dict.fromkeys(scoring.EVENTS, 0)
    thresholds["fight"] = 0.3
    output = tmp_path / "report"
    summary = scoring.evaluate(records, samples, thresholds, output, "test/model")
    questions = json.loads((output / "per-question.json").read_text())["questions"]
    assert summary["thresholds"] == thresholds
    assert [row["prediction"] for row in questions] == ["no", "yes"]
    assert summary["quality"]["accuracy"] == 1.0
    assert summary["zero_threshold_baseline"]["quality"]["accuracy"] == 0.5
    assert thresholds["fight"] == 0.3
