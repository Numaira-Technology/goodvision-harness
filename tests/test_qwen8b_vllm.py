"""Official exam stays 12-slot unified-frame; only the serving engine changes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.qwen8b import eval_local, serve_vllm
from tools.start_qwen8b_vllm import (
    MAX_DPH,
    HF_ENV_KEY,
    _eligible_offers,
    _packed_python,
    _require_r2_pack,
    _verify_restored_env,
)
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


def test_launcher_requires_r2_pack_and_does_not_pip_on_gpu():
    import tools.start_qwen8b_vllm as launcher
    src = Path(launcher.__file__).read_text()
    assert "python3 -m tools.build_env" not in src
    assert "_build_env" not in src
    assert "restore_env" in src
    with pytest.raises(RuntimeError, match="does not pip-install"):
        _require_r2_pack({"r2_key": "envs/qwen8b-vllm/x.tar.gz", "sha256": ""})
    with pytest.raises(RuntimeError, match="HF env"):
        _require_r2_pack({"r2_key": HF_ENV_KEY, "sha256": "abc"})
    assert _require_r2_pack(
        {"r2_key": "envs/qwen8b-vllm/x.tar.gz", "sha256": "deadbeef"}
    ) == "deadbeef"


def test_packed_python_is_beside_activate():
    assert _packed_python("/opt/env/qwen8b-vllm/bin/activate") == (
        "/opt/env/qwen8b-vllm/bin/python"
    )


def test_verify_restored_env_uses_absolute_python_and_surfaces_output(monkeypatch):
    captured = {}

    def boom(cmd, **kwargs):
        captured["cmd"] = cmd
        raise subprocess.CalledProcessError(
            1, cmd, output="OSError: libcuda.so.1: cannot open shared object file\n"
        )

    monkeypatch.setattr("tools.start_qwen8b_vllm.subprocess.check_output", boom)
    with pytest.raises(RuntimeError, match="libcuda.so.1"):
        _verify_restored_env(["ssh"], "/opt/env/qwen8b-vllm/bin/activate")
    remote = captured["cmd"][-1]
    assert "/opt/env/qwen8b-vllm/bin/python" in remote
    assert "python -c" in remote
    assert "source " not in remote
    assert "bash -lc" not in remote
    assert "set -u" not in remote
    assert "nvidia-smi" in remote


def test_calledprocesserror_str_hides_output_like_59781901():
    """The 59781901 log printed str(CalledProcessError), which has no stdout."""
    err = subprocess.CalledProcessError(
        1, ["ssh", "bash -lc ..."], output="ModuleNotFoundError: vllm\n"
    )
    assert "ModuleNotFoundError" not in str(err)
    assert err.output == "ModuleNotFoundError: vllm\n"


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


def test_diag_import_does_not_eval_or_prefetch():
    from tools import diag_qwen8b_vllm_import as diag
    src = Path(diag.__file__).read_text()
    assert "from tools.qwen8b.eval_local" not in src
    assert "prefetch_models" not in src
    assert "serve_vllm" not in src
    assert "eval_main" not in src
    assert diag.MAX_DPH <= 1.50
    inner = diag._old_verify_remote("/opt/env/qwen8b-vllm/bin/activate")
    assert inner.startswith("bash -lc ")
    assert "set -u" in inner
    new = diag._new_verify_remote("/opt/env/qwen8b-vllm/bin/activate")
    assert "/opt/env/qwen8b-vllm/bin/python" in new
    assert "source " not in new


def test_diag_eligible_refuses_price_jump():
    from tools.diag_qwen8b_vllm_import import _eligible
    with pytest.raises(RuntimeError, match="more expensive|authorization"):
        _eligible([{"num_gpus": 1, "gpu_ram": 24564, "dph_total": 1.80}])
    with pytest.raises(RuntimeError, match="more expensive|authorization"):
        _eligible([{"num_gpus": 1, "gpu_ram": 49140, "dph_total": 1.21}])
    ok = _eligible([{"num_gpus": 1, "gpu_ram": 24564, "dph_total": 1.21, "id": 1}])
    assert ok[0]["id"] == 1


def test_vllm_requirements_pin_user_site_leaks():
    text = (ROOT / "tools/qwen8b_vllm/requirements.txt").read_text()
    assert "typing_extensions==4.16.0" in text
    assert "filelock==4.2.9" in text
    assert "tqdm==4.70.1" in text
    lock = (ROOT / "tools/qwen8b_vllm/requirements.lock").read_text()
    assert "typing_extensions==4.16.0" in lock
    assert "filelock==4.2.9" in lock
    assert "tqdm==4.70.1" in lock
    assert "vllm==0.31.0" in lock
    assert "torch==2.13.0" in lock


def test_build_env_refuses_frozen_vllm_and_hf_keys():
    from tools.build_env import FROZEN_R2_KEYS, _refuse_frozen_key
    assert "envs/qwen8b/cu128-py312-torch280-ada.tar.gz" in FROZEN_R2_KEYS
    assert "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz" in FROZEN_R2_KEYS
    with pytest.raises(SystemExit):
        _refuse_frozen_key("envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz")
