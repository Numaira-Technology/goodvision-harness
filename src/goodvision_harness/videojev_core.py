"""Shared native VideoJev loading, choice decoding and video records."""

import importlib.metadata
import math
import time

MODEL = "ZhuYuhan/VideoJev-4B"
REVISION = "1c1a253bf90dfa66176124d25a48ffb791d517d5"
CONTEXT = 32768
OPTIONS = [
    "A. Present under the specified event policy.",
    "B. Assessed absent under the specified event policy.",
    "C. Insufficient evidence to decide under the specified event policy.",
]
VALUES = (True, False, None)

# Weight precision of the language model. "bf16" is the production default and
# the only mode any benchmark or live result so far was measured with. "fp8"
# is an experiment: torchao dynamic per-tensor FP8 on selected linear layers.
QUANTIZATION_MODES = ("bf16", "fp8")
TORCHAO_VERSION = "0.13.0"  # pairs with torch 2.8.0
# FP8 tensor cores start at Ada (sm_89); A100 (sm_80) has none.
FP8_MIN_CAPABILITY = (8, 9)
_FP8_PREFIX = "backbone.language_model.layers."
# Inside a decoder layer, only the dense MLP and the full-attention projections.
# Left alone on purpose: linear_attn (Gated DeltaNet), the vision encoder
# (prepare_for_inference keeps it FP32), the FP32 decision head, embeddings.
_FP8_PARTS = (".mlp.", ".self_attn.")
# torch.compile scope for latency experiments. "off" is production. "mlp"
# compiles each decoder layer's dense MLP (plain linears + SiLU, dynamic
# sequence length); "layers" compiles whole decoder layers, where dynamo may
# graph-break around the flash-attention / Gated DeltaNet kernels.
COMPILE_MODES = ("off", "mlp", "layers")


def choice_indices(raw, expected):
    """Validate a flat choice group and return each question's selected index."""
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ValueError("incomplete question group")
    rows = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("invalid choice row")
        if item["id"] in rows:
            raise ValueError("duplicate choice ID")
        rows[item["id"]] = item
    if set(rows) != {q["id"] for q in expected}:
        raise ValueError("unknown or missing choice ID")
    indices = {}
    for q in expected:
        row = rows[q["id"]]
        scores, index = row.get("probabilities"), row.get("answer_index")
        if (
            row.get("task") != "choice"
            or type(index) is not int
            or not 0 <= index < 3
            or not isinstance(scores, list)
            or len(scores) != 3
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                for p in scores
            )
            or not math.isclose(sum(scores), 1, abs_tol=1e-4)
            or scores[index] != max(scores)
            or row.get("answer") != q["options"][index]
        ):
            raise ValueError("invalid choice probability/answer contract")
        indices[q["id"]] = index
    return indices


def video_record(processor, grid, frame_count, fps):
    """Build the native video record after the caller checks its input contract."""
    return {
        "frames": frame_count,
        "fps": fps,
        "grid_thw": grid,
        "tokens": math.prod(grid) // 4,
        "score_times": list(
            processor._calculate_timestamps(list(range(frame_count)), fps, 2)
        ),
        "sampled_duration": frame_count / fps,
    }


def record_for_grid(processor, grid, frame_count, fps):
    if (
        type(frame_count) is not int or not 2 <= frame_count <= 4
        or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
        or len(grid) != 3
        or any(type(v) is not int or v <= 0 for v in grid)
        or grid[0] != (frame_count + 1) // 2
    ):
        raise ValueError("native processor resampled frames or changed temporal grid")
    return video_record(processor, grid, frame_count, fps)


def fp8_linear_filter(module, fqn):
    """Whether ``quantize_`` should turn this module into an FP8 linear.

    ``fqn`` is the module's dotted name inside VideoJevForVideoUnderstanding.
    FP8 matmuls need both dimensions divisible by 16. Matches plain
    ``torch.nn.Linear`` only (not subclasses), checked by name so this module
    stays importable without torch.
    """
    kind = type(module)
    return (
        (kind.__module__, kind.__qualname__) == ("torch.nn.modules.linear", "Linear")
        and fqn.startswith(_FP8_PREFIX)
        and any(part in fqn for part in _FP8_PARTS)
        and ".linear_attn." not in fqn
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


def fp8_linear_names(model):
    """Dotted names of the linears that the FP8 mode would quantize."""
    return [name for name, module in model.named_modules()
            if fp8_linear_filter(module, name)]


def check_quantization(mode, capability=None):
    """Refuse an unknown mode, or FP8 on a GPU without FP8 tensor cores."""
    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"quantization must be one of {QUANTIZATION_MODES}")
    if mode == "fp8" and capability is not None and tuple(capability) < FP8_MIN_CAPABILITY:
        major, minor = capability
        raise RuntimeError(f"fp8 needs an sm_89+ GPU (Ada, Hopper); this GPU is sm_{major}{minor}")


def apply_fp8(model):
    """Quantize the selected linears in place; return what was quantized."""
    if importlib.metadata.version("torchao") != TORCHAO_VERSION:
        raise RuntimeError(f"fp8 needs torchao=={TORCHAO_VERSION}")
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerTensor

    names = fp8_linear_names(model)
    if not names:
        raise RuntimeError("fp8 found no linear layers to quantize")
    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        filter_fn=fp8_linear_filter,
    )
    return {
        "mode": "fp8",
        "scheme": "torchao Float8DynamicActivationFloat8Weight, per-tensor",
        "quantized_linears": len(names),
        "scope": "language_model mlp + self_attn linears",
    }


def compile_targets(model, mode):
    """(name, module) pairs that ``mode`` compiles, in layer order."""
    if mode not in COMPILE_MODES:
        raise ValueError(f"compile must be one of {COMPILE_MODES}")
    if mode == "off":
        return []
    layers = model.backbone.language_model.layers
    if mode == "layers":
        return [(f"backbone.language_model.layers.{i}", layer) for i, layer in enumerate(layers)]
    return [(f"backbone.language_model.layers.{i}.mlp", layer.mlp)
            for i, layer in enumerate(layers) if getattr(layer, "mlp", None) is not None]


def apply_compile(model, mode):
    """Compile the selected modules in place (``nn.Module.compile``)."""
    targets = compile_targets(model, mode)
    for _, module in targets:
        module.compile(dynamic=True)
    return {"mode": mode, "modules": len(targets), "dynamic": True} if targets else {"mode": "off"}


class NativeVideoJevModel:
    """Load the native model shared by live inference and probe tools.

    BF16 by default. ``quantization="fp8"`` is an opt-in experiment for latency
    tests; it is never selected implicitly.
    """

    def __init__(self, cache_dir, *, quantization="bf16", compile_mode="off"):
        import torch
        from transformers import AutoModel, AutoProcessor

        check_quantization(quantization)
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile must be one of {COMPILE_MODES}")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("requires a CUDA GPU supporting BF16; no CPU fallback")
        if importlib.metadata.version("transformers") != "5.14.1":
            raise RuntimeError("use isolated environment with transformers==5.14.1")
        # Fail before the 9 GB load, not after it.
        check_quantization(quantization, torch.cuda.get_device_capability(0))
        self.torch = torch
        started = time.perf_counter()
        shared = dict(
            revision=REVISION,
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL, **shared)
        self.model = AutoModel.from_pretrained(
            MODEL, **shared, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="flash_attention_2",
        )
        self.model.prepare_for_inference()
        quantized = {"mode": "bf16"}
        if quantization == "fp8":
            quantized = apply_fp8(self.model)
        compiled = apply_compile(self.model, compile_mode)  # after FP8, so FP8 linears compile too
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        packages = ["torch", "transformers", "flash-attn", "flash-linear-attention",
                    "causal-conv1d", "fla-core", "accelerate"]
        if quantization == "fp8":
            packages.append("torchao")
        self.metadata = {
            "model": MODEL, "revision": REVISION,
            "gpu": props.name, "gpu_total_bytes": props.total_memory,
            "gpu_allocated_bytes_after_load": torch.cuda.memory_allocated(0),
            "model_load_seconds": time.perf_counter() - started,
            "versions": {n: importlib.metadata.version(n) for n in packages},
            "cuda_runtime": torch.version.cuda,
            "attention": "flash_attention_2", "context": CONTEXT,
            "quantization": quantized,
            "compile": compiled,
        }
