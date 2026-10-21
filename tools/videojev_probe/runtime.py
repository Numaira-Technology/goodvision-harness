"""Isolated CUDA runner using the pinned upstream feature/choice API."""

import importlib.metadata
import io
import math
import time

from .contract import CONTEXT, MODEL, REVISION


def record_for_grid(processor, grid, frame_count, fps):
    if (
        type(frame_count) is not int
        or not 2 <= frame_count <= 4
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
        or len(grid) != 3
        or any(type(v) is not int or v <= 0 for v in grid)
        or grid[0] != (frame_count + 1) // 2
    ):
        raise ValueError("native processor resampled frames or changed temporal grid")
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


class VideoJevRuntime:
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
            MODEL,
            **shared,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="flash_attention_2",
        )
        self.model.prepare_for_inference()
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        self.metadata = {
            "model": MODEL,
            "revision": REVISION,
            "gpu": props.name,
            "gpu_total_bytes": props.total_memory,
            "model_load_seconds": time.perf_counter() - started,
            "versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "torch",
                    "transformers",
                    "flash-attn",
                    "flash-linear-attention",
                    "causal-conv1d",
                    "fla-core",
                    "accelerate",
                )
            },
            "cuda_runtime": torch.version.cuda,
            "attention": "flash_attention_2",
            "context": CONTEXT,
        }

    def predict(self, payloads, fps, questions):
        import numpy as np
        from PIL import Image
        from transformers.video_utils import VideoMetadata

        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        frames = []
        for payload in payloads:
            with Image.open(io.BytesIO(payload)) as image:
                frames.append(np.array(image.convert("RGB")))
        if len({x.shape for x in frames}) != 1:
            raise ValueError("original evidence frames must share geometry")
        n = len(frames)
        vp = self.processor.video_processor
        native = vp(
            videos=[np.stack(frames)],
            video_metadata=[
                VideoMetadata(
                    total_num_frames=n, fps=fps, frames_indices=list(range(n))
                )
            ],
            do_sample_frames=False,
            size={
                "shortest_edge": vp.size["shortest_edge"],
                "longest_edge": n * 640 * 640,
            },
            return_tensors="pt",
        )
        grid = native["video_grid_thw"][0].tolist()
        record = record_for_grid(self.processor, grid, n, fps)
        processed = time.perf_counter()
        with torch.inference_mode():
            features = self.model.encode_video(
                native["pixel_values_videos"], torch.tensor([grid])
            )
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            raw = self.model.predict_features(
                self.processor, features, record, questions, context=CONTEXT
            )
            torch.cuda.synchronize()
            done = time.perf_counter()
        return raw, {
            "preprocess_seconds": processed - start,
            "visual_encode_seconds": encoded - processed,
            "decision_seconds": done - encoded,
            "model_pipeline_seconds": done - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "input_frame_count": n,
            "internal_temporal_padding_frames": grid[0] * 2 - n,
            "video_grid_thw": grid,
            "visual_tokens": record["tokens"],
            "native_temporal_fps": fps,
        }
