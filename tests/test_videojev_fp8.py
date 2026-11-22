"""The FP8 switch for native VideoJev is opt-in and scoped. CPU only.

A small stand-in model reproduces VideoJev-4B's module names (from the pinned
revision's weight index): 32 decoder layers, a full-attention layer every 4th,
the last layer without its MLP, the vision encoder and the FP32 decision head.
torchao itself is not installed in CI; apply_fp8 is checked against a stub that
records what quantize_ received.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
from torch import nn

from goodvision_harness import videojev_core, videojev_service
from goodvision_harness.videojev_core import (
    FP8_MIN_CAPABILITY,
    QUANTIZATION_MODES,
    NativeVideoJevModel,
    apply_fp8,
    check_quantization,
    fp8_linear_filter,
    fp8_linear_names,
)
from goodvision_harness.videojev_runtime import VideoJevRuntime

H, MLP = 32, 64  # stand-ins for 2560 / 9216; both divisible by 16 like the real ones


def full_attention():
    m = nn.Module()
    m.q_proj, m.k_proj = nn.Linear(H, 2 * H, bias=False), nn.Linear(H, H, bias=False)
    m.v_proj, m.o_proj = nn.Linear(H, H, bias=False), nn.Linear(H, H, bias=False)
    m.q_norm, m.k_norm = nn.LayerNorm(16), nn.LayerNorm(16)
    return m


def linear_attention():
    m = nn.Module()
    m.in_proj_qkv, m.in_proj_z = nn.Linear(H, 3 * H, bias=False), nn.Linear(H, H, bias=False)
    m.in_proj_a, m.in_proj_b = nn.Linear(H, 16, bias=False), nn.Linear(H, 16, bias=False)
    m.out_proj, m.conv1d = nn.Linear(H, H, bias=False), nn.Conv1d(H, H, 4, groups=H)
    return m


def mlp():
    m = nn.Module()
    m.gate_proj, m.up_proj = nn.Linear(H, MLP, bias=False), nn.Linear(H, MLP, bias=False)
    m.down_proj = nn.Linear(MLP, H, bias=False)
    return m


def videojev_like(layers=32, interval=4):
    language = nn.Module()
    language.embed_tokens = nn.Embedding(64, H)
    language.layers = nn.ModuleList()
    for i in range(layers):
        layer = nn.Module()
        if (i + 1) % interval == 0:
            layer.self_attn = full_attention()
        else:
            layer.linear_attn = linear_attention()
        if i != layers - 1:  # VideoJev deletes the last layer's MLP
            layer.mlp = mlp()
        language.layers.append(layer)
    visual = nn.Module()
    visual.blocks = nn.ModuleList([mlp() for _ in range(2)])
    visual.merger = nn.Module()
    visual.merger.linear_fc1, visual.merger.linear_fc2 = nn.Linear(H, H), nn.Linear(H, H)
    backbone = nn.Module()
    backbone.language_model, backbone.visual = language, visual
    head = nn.Module()
    head.q, head.k = nn.Linear(H, H), nn.Linear(H, H)
    model = nn.Module()
    model.backbone, model.head = backbone, head
    return model


def test_scope_is_mlp_and_full_attention_of_the_language_model():
    names = fp8_linear_names(videojev_like())
    # 31 MLPs x 3 + 8 full-attention layers x 4 = 125 at the real layer layout.
    assert len(names) == 125
    assert all(n.startswith("backbone.language_model.layers.") for n in names)
    assert not any(".linear_attn." in n for n in names)
    assert {n.rsplit(".", 1)[1] for n in names} == {
        "gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"}
    assert "backbone.language_model.layers.31.mlp.up_proj" not in names
    assert "backbone.language_model.layers.31.self_attn.q_proj" in names


@pytest.mark.parametrize("fqn", [
    "backbone.visual.blocks.0.up_proj",
    "backbone.visual.merger.linear_fc1",
    "head.q",
    "backbone.language_model.layers.0.linear_attn.in_proj_qkv",
    "backbone.language_model.layers.0.linear_attn.out_proj",
])
def test_left_out_modules(fqn):
    assert not fp8_linear_filter(nn.Linear(H, H), fqn)


def test_filter_needs_plain_linear_with_dims_divisible_by_16():
    name = "backbone.language_model.layers.3.mlp.up_proj"
    assert fp8_linear_filter(nn.Linear(H, MLP), name)
    assert not fp8_linear_filter(nn.Linear(H, 24), name)
    assert not fp8_linear_filter(nn.Embedding(H, H), name)

    class Wrapped(nn.Linear):
        pass

    assert not fp8_linear_filter(Wrapped(H, H), name)


def test_bf16_is_the_default_everywhere():
    assert QUANTIZATION_MODES[0] == "bf16"
    for cls in (NativeVideoJevModel, VideoJevRuntime):
        assert inspect.signature(cls.__init__).parameters["quantization"].default == "bf16"


@pytest.mark.parametrize(("mode", "capability", "error"), [
    ("bf16", (8, 0), None),
    ("bf16", None, None),
    ("fp8", (8, 9), None),
    ("fp8", (9, 0), None),
    ("fp8", (8, 6), RuntimeError),  # RTX 3090 / A10
    ("fp8", (8, 0), RuntimeError),  # A100
    ("int8", None, ValueError),
])
def test_check_quantization(mode, capability, error):
    if error is None:
        check_quantization(mode, capability)
    else:
        with pytest.raises(error):
            check_quantization(mode, capability)
    assert FP8_MIN_CAPABILITY == (8, 9)


def stub_torchao(monkeypatch, version="0.13.0"):
    calls = []
    quantization = types.ModuleType("torchao.quantization")
    granularity = types.ModuleType("torchao.quantization.granularity")

    class PerTensor:
        pass

    class Float8DynamicActivationFloat8WeightConfig:
        def __init__(self, granularity):
            self.granularity = granularity

    quantization.Float8DynamicActivationFloat8WeightConfig = (
        Float8DynamicActivationFloat8WeightConfig)
    quantization.quantize_ = lambda model, config, filter_fn: calls.append(
        (model, config, filter_fn))
    granularity.PerTensor = PerTensor
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)
    monkeypatch.setitem(sys.modules, "torchao.quantization.granularity", granularity)
    real = videojev_core.importlib.metadata.version
    monkeypatch.setattr(videojev_core.importlib.metadata, "version",
                        lambda n: version if n == "torchao" else real(n))
    return calls, PerTensor


def test_apply_fp8_passes_per_tensor_config_and_the_scope_filter(monkeypatch):
    calls, per_tensor = stub_torchao(monkeypatch)
    model = videojev_like()
    info = apply_fp8(model)
    [(seen, config, filter_fn)] = calls
    assert seen is model and filter_fn is fp8_linear_filter
    assert isinstance(config.granularity, per_tensor)
    assert info["mode"] == "fp8" and info["quantized_linears"] == 125


def test_apply_fp8_refuses_other_torchao_versions(monkeypatch):
    stub_torchao(monkeypatch, version="0.14.0")
    with pytest.raises(RuntimeError, match=r"torchao==0\.13\.0"):
        apply_fp8(videojev_like())


def test_apply_fp8_refuses_a_model_with_nothing_to_quantize(monkeypatch):
    stub_torchao(monkeypatch)
    with pytest.raises(RuntimeError, match="no linear"):
        apply_fp8(nn.Sequential(nn.Linear(H, H)))


def test_service_cli_refuses_fp8_for_decision3(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "videojev_service", "--runtime", "decision3", "--quantization", "fp8",
        "--cache-dir", str(tmp_path), "--result-log", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit):
        videojev_service.main()


def test_fp8_requirements_extend_the_bf16_environment():
    from pathlib import Path

    lines = Path("tools/videojev_probe/requirements-fp8.txt").read_text().splitlines()
    pins = [line for line in lines if line and not line.startswith("#")]
    assert pins == ["-r requirements.txt", f"torchao=={videojev_core.TORCHAO_VERSION}"]
