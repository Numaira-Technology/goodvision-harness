"""Check runtime preconditions without importing CUDA/model dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_empty_media_cannot_report_success(monkeypatch, tmp_path):
    for name in ("torch", "torch.nn", "torch.nn.functional"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    transformers = ModuleType("transformers")
    transformers.AutoModel = transformers.AutoProcessor = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    timing = ModuleType("base_timing")
    timing.TimedVideoFileStream = object
    monkeypatch.setitem(sys.modules, "base_timing", timing)
    path = Path(__file__).resolve().parents[1] / "tools/clip_benchmark/runtime.py"
    spec = importlib.util.spec_from_file_location("clip_benchmark_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="at least one primary video"):
        runtime.run_streams([], SimpleNamespace(metadata={}), {}, output)
    assert not output.exists()
