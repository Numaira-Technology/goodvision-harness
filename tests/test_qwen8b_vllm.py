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
    INCOMPLETE_VLLM_KEY,
    _eligible_offers,
    _packed_python,
    _require_r2_pack,
    _smoke,
    _verify_restored_env,
    _vllm_serve_remote,
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
    assert "_patch_setuptools" not in src
    assert "SETUPTOOLS_COMPILERS_ERRORS" not in src
    from tools.start_qwen8b_vllm import EXPECTED_R2_KEY, EXPECTED_SHA256, EXPECTED_SIZE, OLD_VLLM_KEY
    with pytest.raises(RuntimeError, match="does not pip-install"):
        _require_r2_pack({"r2_key": EXPECTED_R2_KEY, "sha256": ""})
    with pytest.raises(RuntimeError, match="HF env"):
        _require_r2_pack({"r2_key": HF_ENV_KEY, "sha256": "abc"})
    with pytest.raises(RuntimeError, match="superseded cu128"):
        _require_r2_pack({"r2_key": OLD_VLLM_KEY, "sha256": "abc"})
    with pytest.raises(RuntimeError, match="incomplete"):
        _require_r2_pack(
            {"r2_key": INCOMPLETE_VLLM_KEY, "sha256": EXPECTED_SHA256,
             "size": EXPECTED_SIZE}
        )
    assert _require_r2_pack(
        {"r2_key": EXPECTED_R2_KEY, "sha256": EXPECTED_SHA256,
         "size": EXPECTED_SIZE}
    ) == EXPECTED_SHA256


def test_packed_python_is_beside_activate():
    assert _packed_python("/opt/env/qwen8b-vllm/bin/activate") == (
        "/opt/env/qwen8b-vllm/bin/python"
    )


def test_vllm_serve_remote_keeps_exam_and_avoids_repo_pythonpath():
    remote = _vllm_serve_remote("/opt/env/qwen8b-vllm/bin/activate", "/weights/snap")
    assert "PYTHONPATH=" not in remote
    assert "PYTHONNOUSERSITE=1" in remote
    assert "/opt/env/qwen8b-vllm/bin/python -s -m vllm.entrypoints.openai.api_server" in remote
    assert "--max-num-seqs 12" in remote
    assert "--dtype bfloat16" in remote
    assert "--enforce-eager" in remote
    assert "pip install" not in remote
    assert "python -m tools.build_env" not in remote


def test_verify_restored_env_uses_absolute_python_and_surfaces_output(monkeypatch):
    captured = {"cmds": []}

    def boom(cmd, **kwargs):
        captured["cmds"].append(cmd)
        raise subprocess.CalledProcessError(
            1, cmd, output="OSError: libcuda.so.1: cannot open shared object file\n"
        )

    monkeypatch.setattr("tools.start_qwen8b_vllm.subprocess.check_output", boom)
    with pytest.raises(RuntimeError, match="libcuda.so.1"):
        _verify_restored_env(["ssh"], "/opt/env/qwen8b-vllm/bin/activate")
    first = captured["cmds"][0][-1]
    assert "/opt/env/qwen8b-vllm/bin/python" in first
    assert "python -c" in first
    assert "source " not in first
    assert "bash -lc" not in first
    assert "set -u" not in first
    assert "nvidia-smi" in first
    assert "torch.cuda.is_available" in first
    assert "vllm.entrypoints.openai.api_server" in first
    assert "torch.utils.cpp_extension" in first


def test_calledprocesserror_str_hides_output_like_59781901():
    """The 59781901 log printed str(CalledProcessError), which has no stdout."""
    err = subprocess.CalledProcessError(
        1, ["ssh", "bash -lc ..."], output="ModuleNotFoundError: vllm\n"
    )
    assert "ModuleNotFoundError" not in str(err)
    assert err.output == "ModuleNotFoundError: vllm\n"


def test_smoke_uses_frozen_jpeg_and_surfaces_http_body(monkeypatch, tmp_path):
    import tools.start_qwen8b_vllm as launcher
    from io import BytesIO
    from urllib.error import HTTPError

    src = Path(launcher.__file__).read_text()
    assert "/9j/4AAQSkZJRgABAQAAAQABAAD" not in src
    assert "_smoke(opener, args.eval_bundle)" in src
    monkeypatch.setattr(launcher, "_smoke_body", lambda bundle: {"model": "x"})

    class Boom:
        def open(self, req, timeout=0):
            raise HTTPError(
                "http://127.0.0.1/v1/chat/completions", 400, "Bad Request",
                hdrs=None, fp=BytesIO(b'{"error":{"message":"image too small"}}'),
            )

    with pytest.raises(RuntimeError, match="image too small"):
        _smoke(Boom(), tmp_path)


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
    from tools.unified_benchmark.workflow import verify_bundle
    from test_qwen8b_review import frozen_host_bundle
    src = tmp_path / "host"
    _, unit = frozen_host_bundle(src)
    original = {str(p.relative_to(src)): p.read_bytes() for p in src.rglob('*') if p.is_file()}
    dest = tmp_path / "vllm"
    info = prepare(src, dest)
    plan = json.loads((dest / "plans/qwen8b_r2_vllm-validation.json").read_text())
    assert plan["units"][0]["request_sha256"] == unit["request_sha256"]
    assert plan["units"][0]["frames"] == unit["frames"]
    assert info["exam_unchanged"] is True
    assert verify_bundle(dest) == info['bundle_sha256']
    assert original == {str(p.relative_to(src)): p.read_bytes() for p in src.rglob('*') if p.is_file()}


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
    assert "envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz" in FROZEN_R2_KEYS
    assert "envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz" in FROZEN_R2_KEYS
    with pytest.raises(SystemExit):
        _refuse_frozen_key("envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz")
    with pytest.raises(SystemExit):
        _refuse_frozen_key("envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz")


def test_cuda_query_uses_selected_pack_compat():
    from tools.start_qwen8b_r2 import _vast_query
    from tools import registry
    assert "cuda_vers>=12.8" in _vast_query(80)
    spec = registry.env_of("qwen8b-vllm")
    assert "cuda_vers>=13.0" in _vast_query(80, cuda_version=spec["compat"]["cuda"])


@pytest.mark.parametrize("model", [None, "wrong", 42])
def test_smoke_requires_exact_model(monkeypatch, tmp_path, model):
    import io
    from tools import start_qwen8b_vllm as launcher
    monkeypatch.setattr(launcher, "_smoke_body", lambda _: {})
    class Opener:
        def open(self, *a, **kw):
            return io.BytesIO(json.dumps({"model": model}).encode())
    with pytest.raises(RuntimeError, match="model mismatch"):
        launcher._smoke(Opener(), tmp_path)


def test_bundle_check_rejects_profile_before_rental(tmp_path, monkeypatch):
    from tools import start_qwen8b_r2 as host
    from tools.unified_benchmark import workflow
    monkeypatch.setattr(workflow, "verify_bundle", lambda _: "verified")
    (tmp_path / "configs").mkdir()
    config = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    path = tmp_path / "configs/qwen8b_r2_vllm.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "output"
    host._verify_eval_bundle(tmp_path, "qwen8b_r2_vllm", "vllm", output)
    path.write_text(json.dumps(config | {"engine": "transformers"}))
    with pytest.raises(ValueError, match="engine/profile"):
        host._verify_eval_bundle(tmp_path, "qwen8b_r2_vllm", "vllm", output)
    path.write_text(json.dumps(config))
    output.mkdir()
    with pytest.raises(ValueError, match="output already exists"):
        host._verify_eval_bundle(tmp_path, "qwen8b_r2_vllm", "vllm", output)


@pytest.mark.parametrize("module,alias", [("tools.start_qwen8b_r2", "qwen8b_r2_host"),
                                         ("tools.start_qwen8b_vllm", "qwen8b_r2_vllm")])
def test_launchers_validate_before_loading_credentials(monkeypatch, tmp_path, module, alias):
    import importlib
    import sys
    from tools import start_qwen8b_r2 as host
    launcher = importlib.import_module(module)
    def stop(bundle, selected, engine, output):
        assert selected == alias
        raise ValueError("offline bundle rejection")
    monkeypatch.setattr(host, "_verify_eval_bundle", stop)
    monkeypatch.setattr(sys, "argv", [module, "--eval-bundle", str(tmp_path),
                                    "--eval-output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="offline bundle rejection"):
        launcher.main()


def test_readiness_does_not_accept_other_models(monkeypatch):
    import io
    from unittest.mock import Mock
    from tools import start_qwen8b_vllm as launcher
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    class Opener:
        def open(self, *a, **kw):
            return io.BytesIO(b'{"data":[{"id":"wrong"}]}')
    monkeypatch.setattr(launcher, "build_opener", lambda _: Opener())
    with pytest.raises(RuntimeError, match="did not become ready"):
        launcher._wait_ready(Mock(poll=lambda: None), timeout=1)
