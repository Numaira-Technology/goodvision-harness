"""Native CUDA scoring with separate CPU preparation of original JPEG frames."""

import importlib.metadata
import io
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

from .contract import CONTEXT, MODEL, REVISION

QUESTION_LAYOUTS = ("repeated", "shared", "prefix", "prefix-cache")


def record_for_grid(processor, grid, frame_count, fps):
    if (
        type(frame_count) is not int or not 2 <= frame_count <= 64
        or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
        or len(grid) != 3
        or any(type(v) is not int or v <= 0 for v in grid)
        or grid[0] != (frame_count + 1) // 2
        or grid[1] % 2 or grid[2] % 2
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


def _shared_questions(rows):
    """Keep an identical query prefix once within one original native group."""
    if len(rows) < 2:
        return rows, False
    parts = [row["query"].rpartition("\nFor ") for row in rows]
    prefix = parts[0][0]
    if not prefix or any(not marker or head != prefix for head, marker, _ in parts):
        return rows, False
    return [
        rows[0],
        *[
            {**row, "query": marker + question}
            for row, (_, marker, question) in zip(rows[1:], parts[1:], strict=True)
        ],
    ], True


class VideoJevRuntime:
    def __init__(self, cache_dir, *, question_layout="repeated"):
        import torch
        from transformers import AutoModel, AutoProcessor

        if question_layout not in QUESTION_LAYOUTS:
            raise ValueError("invalid question_layout")
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
        native = vars(sys.modules[type(self.model).__module__])
        self._prepare_query_groups = native["prepare_query_groups"]
        self._format_predictions = native["format_predictions"]
        self.question_layout = question_layout
        self._prefix_cache_key = None
        self._prefix_cache = None
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
            "question_layout_default": question_layout,
        }

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
            not isinstance(payloads, (list, tuple)) or not 2 <= len(payloads) <= 64
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(context) is not int or context < 64
            or type(max_pixels) is not int or max_pixels <= 0
        ):
            raise ValueError("provide 2..64 original JPEGs, positive FPS and pixel budget")
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

    def _layout_group(self, prepared, batch, rows, question_layout):
        if question_layout == "repeated":
            return batch, "repeated", 0
        shared, changed = _shared_questions(rows)
        if not changed:
            return batch, "repeated", 0
        prefix_ids = None
        if question_layout in ("prefix", "prefix-cache"):
            marker = "\nEvidence metadata: "
            policies = [row["query"].partition(marker) for row in rows]
            policy = policies[0][0]
            if (not policy or any(not separator or head != policy
                                  for head, separator, _ in policies)):
                return batch, "repeated", 0
            # A complete text-only message precedes the original video message.
            # Dynamic source times and person context stay after the video.
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": policy}]}],
                tokenize=False, add_generation_prompt=False,
            )
            prefix_ids = self.torch.tensor([
                self.processor.tokenizer.encode(text, add_special_tokens=False)
            ], dtype=batch["input_ids"].dtype)
            shared = [
                {**shared[0], "query": marker + policies[0][2]},
                *shared[1:],
            ]
        prefix_length = 0 if prefix_ids is None else prefix_ids.shape[1]
        # Original native groups remain the boundary. Each new group retains
        # its own full first-row prefix, even when the original window split.
        groups = iter(self._prepare_query_groups(
            self.processor, prepared.record, shared, prepared.context - prefix_length,
        ))
        try:
            replacement, replacement_rows = next(groups)
            if next(groups, None) is not None or len(replacement_rows) != len(rows):
                return batch, "repeated", 0
        except (StopIteration, ValueError):
            return batch, "repeated", 0
        if prefix_ids is not None:
            replacement = {
                **replacement,
                "input_ids": self.torch.cat([prefix_ids, replacement["input_ids"]], dim=1),
                "mm_token_type_ids": self.torch.cat([
                    self.torch.zeros_like(prefix_ids), replacement["mm_token_type_ids"],
                ], dim=1),
                "pair_indices": replacement["pair_indices"] + prefix_length,
                "query_indices": replacement["query_indices"] + prefix_length,
            }
        return replacement, question_layout, prefix_length

    def _prefix_forward(self, batch, features, prefix_length, *, use_cache):
        """Score a text-first layout, optionally reusing only its pure-text state.

        The pinned native forward disables caching. This opt-in path preserves
        its visual injection, complete-window positions and decision head, but
        lets Transformers build unpadded continuation masks. Packed q/k lengths
        from a full forward must not be reused with a longer cached key prefix.
        """
        torch, model = self.torch, self.model
        device = model.get_input_embeddings().weight.device
        ids = batch["input_ids"].to(device)
        types = batch["mm_token_type_ids"].to(device)
        grid = batch["video_grid_thw"].to(device)
        pairs = batch["pair_indices"].to(device)
        queries = batch["query_indices"].to(device)
        mask = batch["score_mask"].to(device)
        tasks = batch["decision_tasks"].to(device)
        stats = {"prefix_tokens": prefix_length, "prefix_cache_enabled": use_cache,
                 "prefix_cache_hit": False,
                 "prefix_prefill_seconds": 0.0, "prefix_clone_seconds": 0.0}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.get_input_embeddings()(ids)
            slots = ids == model.config.backbone_config.video_token_id
            if int(slots.sum()) != len(features) or slots[:, :prefix_length].any():
                raise ValueError("visual features must belong to this window's suffix")
            embeds = embeds.masked_scatter(
                slots[..., None].expand_as(embeds), features.to(embeds),
            )
            if model.is_moe:
                positions, _ = model.backbone.get_rope_index(
                    ids, types, video_grid_thw=grid,
                )
            else:
                positions = model.backbone.compute_3d_position_ids(
                    input_ids=ids, inputs_embeds=embeds, video_grid_thw=grid,
                    attention_mask=None, past_key_values=None, mm_token_type_ids=types,
                )
            language = model.backbone.language_model
            if use_cache:
                # One immutable base state per loaded model. Deep-copy attention
                # KV, convolution state and DeltaNet recurrent state per window.
                # A suffix-updated state is never retained for another video.
                key = (tuple(ids[0, :prefix_length].tolist()),
                       tuple(positions[..., :prefix_length].flatten().tolist()))
                stats["prefix_cache_hit"] = key == self._prefix_cache_key
                if not stats["prefix_cache_hit"]:
                    self._prefix_cache = None
                    self._prefix_cache_key = None
                    before = time.perf_counter()
                    base = language(
                        inputs_embeds=embeds[:, :prefix_length],
                        position_ids=positions[..., :prefix_length],
                        attention_mask=None, use_cache=True,
                    ).past_key_values
                    torch.cuda.synchronize()
                    stats["prefix_prefill_seconds"] = time.perf_counter() - before
                    self._prefix_cache, self._prefix_cache_key = base, key
                before = time.perf_counter()
                working = deepcopy(self._prefix_cache)
                torch.cuda.synchronize()
                stats["prefix_clone_seconds"] = time.perf_counter() - before
                hidden = language(
                    inputs_embeds=embeds[:, prefix_length:],
                    position_ids=positions[..., prefix_length:],
                    attention_mask=None, past_key_values=working, use_cache=True,
                ).last_hidden_state[0]
                pairs, queries = pairs - prefix_length, queries - prefix_length
                stats["forward_tokens"] = ids.shape[1] - (
                    prefix_length if stats["prefix_cache_hit"] else 0
                )
            else:
                hidden = language(
                    inputs_embeds=embeds, position_ids=positions,
                    attention_mask=None, use_cache=False,
                ).last_hidden_state[0]
                stats["forward_tokens"] = ids.shape[1]
        with torch.autocast(device_type="cuda", enabled=False):
            logits = model.head(hidden[pairs], hidden[queries])
            valid = logits.masked_fill(~mask, -torch.inf)
            probabilities = torch.where(
                (tasks == 1)[:, None], valid.softmax(-1), logits.sigmoid(),
            ) * mask
        return SimpleNamespace(logits=logits, probabilities=probabilities,
                               score_mask=mask), stats

    def predict_prepared(self, prepared, questions, *, question_layout=None):
        """Encode and score on the service's single GPU worker thread."""
        if not isinstance(prepared, _PreparedVideo):
            raise ValueError("provide the result of prepare")
        layout = self.question_layout if question_layout is None else question_layout
        if layout not in QUESTION_LAYOUTS:
            raise ValueError("invalid question_layout")
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            features = self.model.encode_video(
                prepared.pixel_values, torch.tensor([prepared.record["grid_thw"]]),
            )
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            decoder = dict(self.model.config.decoder)
            if not 0 < decoder["threshold"] < 1:
                raise ValueError("threshold must be between 0 and 1")
            raw, layouts, tokens, scoring_events, prefix_stats = [], [], [], [], []
            group_prepare_seconds = 0.0
            groups = iter(self._prepare_query_groups(
                self.processor, prepared.record, questions, prepared.context,
            ))
            while True:
                group_started = time.perf_counter()
                try:
                    batch, rows = next(groups)
                except StopIteration:
                    group_prepare_seconds += time.perf_counter() - group_started
                    break
                batch, actual_layout, prefix_length = self._layout_group(
                    prepared, batch, rows, layout,
                )
                group_prepare_seconds += time.perf_counter() - group_started
                layouts.append(actual_layout)
                tokens.append(int(batch["input_ids"].shape[1]))
                score_start = torch.cuda.Event(enable_timing=True)
                score_end = torch.cuda.Event(enable_timing=True)
                score_start.record()
                if prefix_length:
                    output, cache_stats = self._prefix_forward(
                        batch, features, prefix_length,
                        use_cache=actual_layout == "prefix-cache",
                    )
                    prefix_stats.append(cache_stats)
                else:
                    output = self.model(**batch, visual_features=features)
                score_end.record()
                scoring_events.append((score_start, score_end))
                raw.extend(self._format_predictions(output, rows, prepared.record, decoder))
            torch.cuda.synchronize()
            done = time.perf_counter()
        return raw, {
            **prepared.timings,
            "visual_encode_seconds": encoded - start,
            "decision_seconds": done - encoded,
            "gpu_pipeline_seconds": done - start,
            "model_pipeline_seconds": prepared.timings["preprocess_seconds"] + done - start,
            "question_layout_requested": layout,
            "question_layout": layouts[0] if len(set(layouts)) == 1 else "mixed",
            "question_layout_groups": layouts,
            "native_group_count": len(layouts),
            "native_input_tokens": sum(tokens),
            "native_group_input_tokens": tokens,
            "question_group_prepare_seconds": group_prepare_seconds,
            "prefix_cache_groups": prefix_stats,
            "gpu_scoring_seconds": sum(
                first.elapsed_time(last) / 1000 for first, last in scoring_events
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def predict(self, payloads, fps, questions, *, question_layout=None):
        return self.predict_prepared(
            self.prepare(payloads, fps), questions, question_layout=question_layout,
        )
