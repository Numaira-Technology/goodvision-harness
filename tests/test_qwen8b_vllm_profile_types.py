"""Reject consistently resealed profiles that change JSON types."""

import json

import pytest
from test_qwen8b_vllm_integrity import frozen, seal

from tools import registry
from tools import start_qwen8b_vllm as launcher
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import digest, file_hash
from tools.unified_benchmark.preflight import model_configs
from tools.unified_benchmark.workflow import verify_bundle

__all__ = ["frozen"]


@pytest.mark.parametrize(
    "keys,value",
    [
        (("max_tokens",), 30.0),
        (("temperature",), False),
        (("window_frames",), True),
        (("current_access_verified",), 0),
        (("vllm", "enforce_eager"), 1),
        (("vllm", "max_num_seqs"), 12.0),
        (("env_artifact", "size"), 4252375781.0),
    ],
)
def test_resealed_profile_preserves_json_types(frozen, keys, value):
    path = frozen / "configs/qwen8b_r2_vllm.json"
    config = json.loads(path.read_text())
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    expected = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    assert config == expected  # Python equality conceals these type changes.
    assert digest(config) != digest(expected)
    path.write_text(json.dumps(config))
    report_path = frozen / "preflight.json"
    report = json.loads(report_path.read_text())
    for split in ("validation", "test"):
        plan_path = frozen / f"plans/qwen8b_r2_vllm-{split}.json"
        plan = json.loads(plan_path.read_text())
        plan["config_sha256"] = digest(config)
        for unit in plan["units"]:
            unit["request_sha256"] = digest(request_body(config, unit, frozen)[1])
        plan_path.write_text(json.dumps(plan))
        report["models"]["qwen8b_r2_vllm"][split]["plan_sha256"] = digest(plan)
    report_path.write_text(json.dumps(report))
    seal(frozen)
    assert verify_bundle(frozen) == file_hash(frozen / "bundle-files-sha256.json")
    with pytest.raises(ValueError, match="profile differs"):
        launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm"))


def test_selected_artifact_identity_preserves_json_types(frozen):
    env = registry.env_of("qwen8b-vllm") | {"size": 4252375781.0}
    with pytest.raises(ValueError, match="pack identity differs"):
        launcher._verify_eval_bundle(frozen, "both", env)
