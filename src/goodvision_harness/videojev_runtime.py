"""Native CUDA scoring with separate CPU preparation of original JPEG frames."""

import io
import math
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

from .videojev_core import CONTEXT, MODEL, REVISION, NativeVideoJevModel, record_for_grid
from .videojev_questions import CompiledQuestions


@dataclass(frozen=True)
class _PreparedVideo:
    pixel_values: object
    record: dict
    context: int
    timings: dict


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
