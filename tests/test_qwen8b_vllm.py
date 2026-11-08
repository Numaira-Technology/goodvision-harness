"""Official exam stays 12-slot unified-frame; only the serving engine changes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.qwen8b import eval_local, serve_vllm
from tools.start_qwen8b_vllm import MAX_DPH, _eligible_offers
from tools.unified_benchmark.contract import RULES
from tools.unified_benchmark.preflight import model_configs

ROOT = Path(__file__).resolve().parents[1]


def test_exam_rules_still_require_twelve_slots():
    assert RULES["slots"] == 12
    assert RULES["fps"] == 1
    assert RULES["frame_width"] == 720


def test_eval_local_hardcodes_twelve_slots():
    src = Path(eval_local.__file__).read_text()
    assert '"slots": 12' in src
    assert '"slots_per_model": 12' in src
    assert "--slots" not in src
    assert "Queue(ledger, config[\"kind\"], LocalTransport" in src


def test_vllm_profile_keeps_exam_fields():
    host = model_configs(["qwen8b_r2_host"])["qwen8b_r2_host"]
    vllm = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    for key in ("model", "max_tokens", "temperature", "window_frames",
                "prompt_policy", "answer_parser", "weight_revision", "execution"):
        assert vllm[key] == host[key]
    assert vllm["max_tokens"] == 30
    assert vllm["temperature"] == 0
    assert vllm["engine"] == "vllm"
    assert vllm["engine_version"] == "0.31.0"
    assert vllm["vllm"]["dtype"] == "bfloat16"
    assert vllm["vllm"]["max_num_seqs"] == 12
    assert vllm["vllm"]["limit_mm_per_prompt"] == {"image": 1, "video": 0}


def test_serve_vllm_does_not_shrink_exam_inputs(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    monkeypatch.setattr(serve_vllm, "snapshot", lambda _cache: snap)
    argv, cfg = serve_vllm.serve_argv(tmp_path)
    joined = " ".join(argv)
    assert "--max-num-seqs" in argv and argv[argv.index("--max-num-seqs") + 1] == "12"
    assert "--dtype" in argv and argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "max_pixels" not in joined
    assert "mm-processor-kwargs" not in joined
    assert cfg["max_model_len"] >= 8192
    assert cfg["limit_mm_per_prompt"]["image"] == 1


def test_eligible_offers_reject_expensive_or_multi_gpu():
    with pytest.raises(RuntimeError, match="multi-GPU|more expensive"):
        _eligible_offers([{"num_gpus": 2, "dph_total": 1.0}])
    with pytest.raises(RuntimeError, match="more expensive"):
        _eligible_offers([{"num_gpus": 1, "dph_total": MAX_DPH + 0.5}])
    ok = _eligible_offers([
        {"num_gpus": 1, "dph_total": 1.22, "id": 1},
        {"num_gpus": 1, "dph_total": 3.0, "id": 2},
    ])
    assert [o["id"] for o in ok] == [1]


def test_prepare_vllm_bundle_preserves_request_hashes(tmp_path):
    from tools.qwen8b.prepare_vllm_bundle import prepare
    src = tmp_path / "host"
    (src / "plans").mkdir(parents=True)
    (src / "configs").mkdir()
    (src / "frames").mkdir()
    host_plan = {
        "protocol": "unified-frame-120-v2",
        "split": "validation",
        "config_sha256": "abc",
        "rules_sha256": "def",
        "questions": [{"number": 1, "unit_ids": ["u1"]}],
        "units": [{"id": "u1", "request_sha256": "cafebabedeadbeef" * 2,
                   "frames": [{"sha256": "0" * 64, "path": "frames/x.jpg"}]}],
    }
    for split in ("validation", "test"):
        p = dict(host_plan, split=split)
        (src / "plans" / f"qwen8b_r2_host-{split}.json").write_text(json.dumps(p))
    (src / "bundle-files-sha256.json").write_text("{}")
    dest = tmp_path / "vllm"
    info = prepare(src, dest)
    plan = json.loads((dest / "plans" / "qwen8b_r2_vllm-validation.json").read_text())
    assert plan["units"][0]["request_sha256"] == host_plan["units"][0]["request_sha256"]
    assert plan["units"][0]["frames"][0]["sha256"] == "0" * 64
    assert info["exam_unchanged"] is True
    assert (dest / "configs" / "qwen8b_r2_vllm.json").is_file()
