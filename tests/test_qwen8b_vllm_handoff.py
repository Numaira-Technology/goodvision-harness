"""Packed-interpreter handoff regressions, independent of the artifact generation."""

import subprocess

import pytest

from tools.start_qwen8b_vllm import (
    _packed_python,
    _verify_restored_env,
    _vllm_serve_remote,
)


def test_packed_python_is_beside_activate():
    assert _packed_python("/opt/env/qwen8b-vllm/bin/activate") == (
        "/opt/env/qwen8b-vllm/bin/python"
    )


def test_vllm_serve_remote_keeps_exam_and_avoids_repo_pythonpath():
    remote = _vllm_serve_remote("/opt/env/qwen8b-vllm/bin/activate", "/weights/snap")
    assert "PYTHONPATH=" not in remote
    assert "source " not in remote
    assert "conda activate" not in remote
    assert "PYTHONNOUSERSITE=1" in remote
    assert "/opt/env/qwen8b-vllm/bin/python -s -c" in remote
    assert "sys.prefix" in remote
    assert "vllm.entrypoints.openai.api_server" in remote
    assert "--max-num-seqs" in remote
    assert "bfloat16" in remote
    assert "--enforce-eager" in remote
    assert "pip install" not in remote
    assert "python -m tools.build_env" not in remote
    assert "packed interpreter missing" in remote


def test_verify_restored_env_uses_absolute_python_and_surfaces_output(monkeypatch):
    captured = {"cmds": []}

    def boom(cmd, **kwargs):
        captured["cmds"].append(cmd)
        raise subprocess.CalledProcessError(
            1, cmd, output="OSError: libcuda.so.1: cannot open shared object file\n"
        )

    monkeypatch.setattr("tools.start_qwen8b_vllm.subprocess.check_output", boom)
    with pytest.raises(RuntimeError, match=r"libcuda\.so\.1"):
        _verify_restored_env(["ssh"], "/opt/env/qwen8b-vllm/bin/activate")
    first = captured["cmds"][0][-1]
    assert "/opt/env/qwen8b-vllm/bin/python" in first
    assert " -s -c " in first
    assert "PYTHONNOUSERSITE=1" in first
    assert "sys.prefix" in first
    assert "source " not in first
    assert "bash -lc" not in first
    assert "set -u" not in first
    assert "nvidia-smi" in first
    assert "torch.cuda.is_available" in first
    assert "vllm.entrypoints.openai.api_server" in first
    assert "torch.utils.cpp_extension" in first
    assert len(captured["cmds"]) == 1
