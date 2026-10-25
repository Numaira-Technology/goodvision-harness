"""Isolated CUDA runner using the pinned upstream feature/choice API."""

import io
import time

from goodvision_harness.videojev_core import (
    CONTEXT, MODEL, REVISION, NativeVideoJevModel, record_for_grid,
)


class VideoJevRuntime(NativeVideoJevModel):
    """Keep the probe's original preprocessing and upstream prediction path."""

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
