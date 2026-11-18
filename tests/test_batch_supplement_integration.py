"""The batch controller uses the verified optional test cohort."""

import csv
import json
import threading

import pytest
from test_unified_dataset202 import inputs as inputs
from test_unified_hands_off import api_batch_config, payload
from test_unified_test_supplement import bundle_factory as bundle_factory
from test_unified_test_supplement import supplement as supplement

from tools.unified_benchmark import api_run, batch
from tools.unified_benchmark.contract import atomic_json, file_hash


@pytest.mark.parametrize("added", [2, 20])
def test_batch_accepts_and_reserves_frozen_supplement(
    bundle_factory, supplement, tmp_path, added,
):
    supplement["samples"] = supplement["samples"][:added]
    bundle = bundle_factory("batch", bags=True, aliases=("qwen8b",))
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic-not-a-key\n")
    config = api_batch_config(bundle, key)
    checked = batch.validate_config(config)
    assert len(checked["matrix"]) == 11
    assert sum(r["split"] == "test" for r in checked["matrix"]) == 10
    units = {
        split: len(batch.read(bundle / f"plans/qwen8b-{split}.json")["units"])
        for split in ("validation", "test")
    }
    assert checked["reserved_upper_usd"] == pytest.approx(
        (units["validation"] + 10 * units["test"]) * 3 * 0.001,
    )


@pytest.mark.parametrize("added", [2, 20])
def test_batch_summary_uses_plan_before_any_round_starts(
    bundle_factory, supplement, tmp_path, added,
):
    supplement["samples"] = supplement["samples"][:added]
    bundle = bundle_factory("batch", bags=True, aliases=("qwen8b",))
    config = {"models": [{"alias": "qwen8b", "route": "api", "bundle": str(bundle)}]}
    result = batch.summarize(tmp_path, config, {
        "batch_id": "synthetic", "matrix": [], "reserved_upper_usd": 0,
    })
    assert result["validation_questions"] == 44
    assert result["test_questions"] == 202 + added


def test_batch_completed_supplement_verifies_real_receipts_and_plan(
    bundle_factory, tmp_path, monkeypatch,
):
    bundle = bundle_factory("batch", bags=True, aliases=("qwen8b",))
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic-not-a-key\n")
    config = api_batch_config(bundle, key)
    item = config["models"][0]
    calls = []

    def synthetic_call(self, _uri, _body):
        calls.append(self.output)
        return payload(self.config)

    monkeypatch.setattr(api_run.Transport, "call", synthetic_call)
    root = tmp_path / "runs"
    row = {
        "alias": "qwen8b", "split": "test", "round": 1,
        "output": "qwen8b/round-01", "state": "completed", "invocations": 1,
    }
    api_run.execute_model(
        "qwen8b", bundle, root / row["output"], "synthetic-not-a-key",
        threading.Event(), item["bundle_sha256"], phase="test",
    )
    result = batch.verify_finished(root, item, row)
    assert result["planned_questions"] == result["complete_valid_questions"] == 222
    assert len(calls) == len(batch.read(bundle / "plans/qwen8b-test.json")["units"])
    summary = batch.summarize(root, config, {
        "batch_id": "synthetic", "matrix": [row], "reserved_upper_usd": 0,
    })
    assert summary["test_questions"] == 222
    assert summary["models"]["qwen8b"]["rounds"][0]["complete_valid_questions"] == 222
    with (root / "rounds.csv").open(newline="") as handle:
        assert next(csv.DictReader(handle))["complete_valid_questions"] == "222"
    assert batch.verify_finished(root, item, row) == result
    assert len(calls) == 444

    # Even an internally re-sealed report cannot replace the frozen denominator.
    directory = root / row["output"] / "qwen8b"
    path = directory / "test/summary.json"
    report = json.loads(path.read_text())
    report["planned_questions"] = 202
    atomic_json(path, report)
    seal_path = directory / "test/completion.json"
    seal = json.loads(seal_path.read_text())
    seal["files"]["test/summary.json"] = file_hash(path)
    atomic_json(seal_path, seal)
    with pytest.raises(ValueError, match="cohort differs"):
        batch.verify_finished(root, item, row)
    assert len(calls) == 444
