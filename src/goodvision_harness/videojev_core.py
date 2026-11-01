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


class NativeVideoJevModel:
    """Load the native BF16 model shared by live inference and probe tools."""

    def __init__(self, cache_dir):
        import torch
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("requires a CUDA GPU supporting BF16; no CPU fallback")
        if importlib.metadata.version("transformers") != "5.14.1":
            raise RuntimeError("use isolated environment with transformers==5.14.1")
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
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        self.metadata = {
            "model": MODEL, "revision": REVISION,
            "gpu": props.name, "gpu_total_bytes": props.total_memory,
            "model_load_seconds": time.perf_counter() - started,
            "versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "torch", "transformers", "flash-attn", "flash-linear-attention",
                    "causal-conv1d", "fla-core", "accelerate",
                )
            },
            "cuda_runtime": torch.version.cuda,
            "attention": "flash_attention_2", "context": CONTEXT,
        }
