"""Native CUDA scoring with separate CPU preparation of original JPEG frames."""

import importlib.metadata
import io
import math
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .videojev_contract import CONTEXT, MODEL, REVISION
from .videojev_questions import CompiledQuestions


def record_for_grid(processor, grid, frame_count, fps):
    if (
        type(frame_count) is not int or not 2 <= frame_count <= 4
        or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
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


@dataclass(frozen=True)
class _PreparedVideo:
    pixel_values: object
    record: dict
    context: int
    timings: dict


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


class VideoJevRuntime(NativeVideoJevModel):
    """Prepare windows on CPU and score compiled scene questions on GPU."""

    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        native = vars(sys.modules[type(self.model).__module__])
        self._format_predictions = native["format_predictions"]
        self.decoder = dict(self.model.config.decoder)
        if not 0 < self.decoder["threshold"] < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.metadata.update(
            torch_cpu_threads=self.torch.get_num_threads(),
            torch_interop_threads=self.torch.get_num_interop_threads(),
        )

    def prepare(self, payloads, fps, *, context=CONTEXT, max_pixels=409600):
        """Prepare on a CPU worker; native positions use the supplied uniform FPS.

        Actual source times remain in the questions. No frame is sampled away;
        odd counts receive only the native processor's temporal patch padding.
        """
        import numpy as np
        from PIL import Image
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
        from transformers.video_utils import VideoMetadata

        start = time.perf_counter()
        if (
            not isinstance(payloads, (list, tuple)) or not 2 <= len(payloads) <= 4
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(context) is not int or context < 64
            or type(max_pixels) is not int or max_pixels <= 0
        ):
            raise ValueError("provide 2..4 original JPEGs, positive FPS and pixel budget")
        frames = []
        for payload in payloads:
            if not isinstance(payload, bytes):
                raise ValueError("original JPEG payloads must be bytes")
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG" or image.width * image.height > 16 * 1024**2:
                    raise ValueError("original evidence must be bounded JPEG frames")
                frames.append(np.array(image.convert("RGB")))
        if len({x.shape for x in frames}) != 1:
            raise ValueError("original evidence frames must share geometry")
        n = len(frames)
        height, width = frames[0].shape[:2]
        vp = self.processor.video_processor
        factor = vp.patch_size * vp.merge_size
        budget = context * 4 // 5

        def geometry(cap):
            h, w = smart_resize(
                n, height, width, temporal_factor=vp.temporal_patch_size,
                factor=factor, min_pixels=vp.size["shortest_edge"], max_pixels=n * cap,
            )
            return h, w, (n + 1) // 2 * (h // factor) * (w // factor)

        low, high = max(1024, math.ceil(vp.size["shortest_edge"] / n)), max_pixels
        if high < low or geometry(low)[2] > budget:
            raise ValueError("complete window cannot fit the context at native resolution")
        while low < high:
            middle = (low + high + 1) // 2
            if geometry(middle)[2] <= budget:
                low = middle
            else:
                high = middle - 1
        native = vp(
            videos=[np.stack(frames)],
            video_metadata=[VideoMetadata(
                total_num_frames=n, fps=fps, frames_indices=list(range(n)),
            )],
            do_sample_frames=False,
            size={"shortest_edge": vp.size["shortest_edge"], "longest_edge": n * low},
            return_tensors="pt",
        )
        pixels = native["pixel_values_videos"]
        if pixels.device.type != "cpu" or native["video_grid_thw"].device.type != "cpu":
            raise ValueError("video preparation must produce CPU tensors")
        grid = native["video_grid_thw"][0].tolist()
        record = record_for_grid(self.processor, grid, n, fps)
        if grid[1] % 2 or grid[2] % 2:
            raise ValueError("native processor resampled frames or changed temporal grid")
        if record["tokens"] > budget:
            raise ValueError("native video grid exceeds the requested context budget")
        return _PreparedVideo(pixels, record, context, {
            "preprocess_seconds": time.perf_counter() - start,
            "input_frame_count": n,
            "internal_temporal_padding_frames": grid[0] * 2 - n,
            "video_grid_thw": grid,
            "visual_tokens": record["tokens"],
            "native_temporal_fps": fps,
            "source_height": height, "source_width": width,
            "processed_height": grid[1] * vp.patch_size,
            "processed_width": grid[2] * vp.patch_size,
            "context": context, "max_pixels": max_pixels,
            "processor_max_pixels": low,
        })

    def predict_prepared(self, prepared, *, question_template, frame_times):
        """Score fixed shared questions on the service's single GPU worker."""
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            features = self.model.encode_video(
                prepared.pixel_values, torch.tensor([prepared.record["grid_thw"]]),
            )
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            if question_template.compiled is None:
                question_template.compiled = CompiledQuestions(
                    self.processor, torch, question_template.rows,
                )
            raw, tokens = [], 0
            for batch, rows in question_template.compiled.groups(
                prepared.record, frame_times, prepared.context,
            ):
                tokens += batch["input_ids"].shape[1]
                output = self.model(**batch, visual_features=features)
                cpu = SimpleNamespace(
                    logits=output.logits.float().cpu(),
                    probabilities=output.probabilities.float().cpu(),
                    score_mask=output.score_mask.cpu(),
                )
                raw.extend(self._format_predictions(cpu, rows, prepared.record, self.decoder))
            torch.cuda.synchronize()
            done = time.perf_counter()
        return raw, {
            **prepared.timings,
            "visual_encode_seconds": encoded - start,
            "decision_seconds": done - encoded,
            "gpu_pipeline_seconds": done - start,
            "model_pipeline_seconds": prepared.timings["preprocess_seconds"] + done - start,
            "native_input_tokens": tokens,
        }
