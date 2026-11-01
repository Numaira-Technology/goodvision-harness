"""Exercise handoff failures without importing encoders or using a GPU."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    runtime = ModuleType("runtime")
    runtime.PROMPTS = {event: [] for event in ("fight", "smoking", "fall", "weapon", "bag_knife", "bag_alcohol_bottle")}
    runtime.REVISIONS = {"test/clip": "a", "test/siglip": "b"}
    calls, closed = [], []
    outcomes = {}

    class Encoder:
        def __init__(self, model_id, cache_dir):
            self.model_id = model_id
            self.metadata = {"model": model_id}

        def close(self):
            closed.append(self.model_id)

    def run_streams(media, encoder, thresholds, output, **kwargs):
        calls.append(output.name)
        return outcomes.get(output.name, {"execution_complete": True})

    runtime.ClipEncoder = Encoder
    runtime.run_streams = run_streams
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=lambda value: None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda value: None))
    path = Path(__file__).resolve().parents[1] / "tools/clip_benchmark/run.py"
    spec = importlib.util.spec_from_file_location("clip_benchmark_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"validation": [], "test": []}))
    thresholds = tmp_path / "thresholds.json"
    frozen = {model: dict.fromkeys(runtime.PROMPTS, 0.1) for model in runtime.REVISIONS}
    thresholds.write_text(json.dumps(frozen))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [str(path), "--config", str(config), "--output", str(output),
                                    "--thresholds", str(thresholds), "--cache-dir", str(tmp_path / "cache")])
    return SimpleNamespace(module=module, calls=calls, closed=closed, outcomes=outcomes,
                           thresholds=thresholds, frozen=frozen, output=output)


def test_incomplete_validation_never_enters_test_or_waits_for_thresholds(runner):
    runner.thresholds.unlink()
    runner.outcomes.update({"validation-clip": {"execution_complete": False},
                           "validation-siglip": {"execution_complete": False}})
    assert runner.module.main() == 1
    assert runner.calls == ["validation-clip", "validation-siglip"]
    assert len(runner.closed) == 2


def test_incomplete_test_returns_failure_and_preserves_summary(runner):
    runner.outcomes["test-clip"] = {"execution_complete": False}
    assert runner.module.main() == 1
    assert json.loads((runner.output / "completion.json").read_text())["execution_complete"] is False
    assert len(runner.closed) == 2


@pytest.mark.parametrize("invalid", [{}, {"fight": float("nan")}, {"fight": True}])
def test_missing_or_invalid_thresholds_fail_before_test_inference(runner, invalid):
    if invalid:
        runner.frozen["test/clip"].update(invalid)
    else:
        runner.frozen["test/clip"] = invalid
    runner.thresholds.write_text(json.dumps(runner.frozen))
    with pytest.raises(ValueError, match="frozen"):
        runner.module.main()
    assert runner.calls == ["validation-clip", "validation-siglip"]
    assert len(runner.closed) == 2


def test_complete_groups_return_success(runner):
    assert runner.module.main() == 0
    assert runner.calls == ["validation-clip", "validation-siglip", "test-clip", "test-siglip"]
    assert len(runner.closed) == 2
