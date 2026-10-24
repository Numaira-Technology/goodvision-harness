"""Native CUDA scoring with separate CPU preparation of original JPEG frames."""

import io
import math
import sys
import time
from copy import deepcopy
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

    def __init__(self, cache_dir, *, prefix_cache=True):
        super().__init__(cache_dir)
        self.prefix_cache = prefix_cache
        native = vars(sys.modules[type(self.model).__module__])
        self._format_predictions = native["format_predictions"]
        self.decoder = dict(self.model.config.decoder)
        if not 0 < self.decoder["threshold"] < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.metadata.update(
            torch_cpu_threads=self.torch.get_num_threads(),
            torch_interop_threads=self.torch.get_num_interop_threads(),
            prefix_cache=prefix_cache,
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

    def _prefix_inputs(self, batch, features, prefix):
        """Embed only the suffix while retaining full-window absolute positions."""
        torch, model = self.torch, self.model
        device = model.get_input_embeddings().weight.device
        length = prefix.length
        if prefix.device_ids is None:
            prefix.device_ids = prefix.ids.to(device)
        ids = batch["input_ids"][:, length:].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if prefix.embeddings is None:
                prefix.embeddings = model.get_input_embeddings()(prefix.device_ids)
            embeds = model.get_input_embeddings()(ids)
            slots = ids == model.config.backbone_config.video_token_id
            embeds = embeds.masked_scatter(
                slots[..., None].expand_as(embeds), features.to(embeds),
            )
            if model.is_moe:
                positions, _ = model.backbone.get_rope_index(
                    batch["input_ids"], batch["mm_token_type_ids"],
                    video_grid_thw=batch["video_grid_thw"],
                )
            else:
                positions = model.backbone.compute_3d_position_ids(
                    input_ids=batch["input_ids"], inputs_embeds=None,
                    video_grid_thw=batch["video_grid_thw"], attention_mask=None,
                    past_key_values=None, mm_token_type_ids=batch["mm_token_type_ids"],
                )
                model.backbone.rope_deltas = model.backbone.rope_deltas.to(device)
            positions = positions.to(device)
        return embeds, positions, batch

    def _working_prefix(self, prefix, count, device):
        """Restore an independent hybrid state without reallocating its linear buffers."""
        working = prefix.working_states.get(count)
        if working is None:
            working = deepcopy(prefix.state)
            if count > 1:
                working.reorder_cache(self.torch.zeros(count, device=device, dtype=self.torch.long))
            prefix.working_states[count] = working
        for original, target in zip(prefix.state.layers, working.layers, strict=True):
            # Dynamic attention appends with cat and replaces these read-only views.
            if getattr(original, "keys", None) is not None:
                target.keys = original.keys.expand(count, *original.keys.shape[1:])
                target.values = original.values.expand(count, *original.values.shape[1:])
            # Linear attention updates its state in place, so keep separate buffers.
            for name in ("conv_states", "recurrent_states"):
                states = getattr(original, name, {})
                destination = getattr(target, name, {})
                for index, value in states.items():
                    if value is not None:
                        destination[index].copy_(value.expand(count, *value.shape[1:]))
        return working

    def _score_cached(self, batches, features, prefix):
        """Continue independent batch rows; never retain their video state."""
        torch, model = self.torch, self.model
        parts = [self._prefix_inputs(batch, visual, prefix)
                 for batch, visual in zip(batches, features, strict=True)]
        length = prefix.length
        device = parts[0][0].device
        with torch.autocast("cuda", dtype=torch.bfloat16):
            language = model.backbone.language_model
            hit = prefix.state is not None
            if not hit:
                prefill = language(
                    inputs_embeds=prefix.embeddings,
                    position_ids=parts[0][1][..., :length],
                    attention_mask=None, use_cache=True,
                )
                prefix.state = prefill.past_key_values
                del prefill
            working = self._working_prefix(prefix, len(parts), device)
            lengths = [embeds.shape[1] for embeds, _positions, _batch in parts]
            longest = max(lengths)
            padding_mask = None
            equal = len(set(lengths)) == 1
            if len(parts) > 1 and not equal:
                padding_mask = torch.zeros(
                    (len(parts), length + longest), dtype=torch.bool, device=device,
                )
                for index, size in enumerate(lengths):
                    padding_mask[index, :length + size] = True
            if len(parts) == 1:
                embeds, positions = parts[0][0], parts[0][1][..., length:]
            elif equal:
                embeds = torch.cat([part[0] for part in parts])
                positions = torch.cat([part[1][..., length:] for part in parts], dim=1)
            else:
                embeds = torch.cat([
                    torch.nn.functional.pad(part[0], (0, 0, 0, longest - size))
                    for part, size in zip(parts, lengths, strict=True)
                ])
                positions = torch.cat([
                    torch.nn.functional.pad(part[1][..., length:], (0, longest - size))
                    for part, size in zip(parts, lengths, strict=True)
                ], dim=1)
            # Right padding follows every scored position; discard updated states.
            hidden = language(
                inputs_embeds=embeds, position_ids=positions,
                attention_mask=padding_mask, past_key_values=working, use_cache=True,
            ).last_hidden_state
            pair_hidden = torch.cat([
                hidden[index, part[2]["pair_indices"].to(device) - length]
                for index, part in enumerate(parts)
            ])
            query_hidden = torch.cat([
                hidden[index, part[2]["query_indices"].to(device) - length]
                for index, part in enumerate(parts)
            ])
        with torch.autocast(device_type="cuda", enabled=False):
            mask = torch.cat([batch["score_mask"].to(device) for batch in batches])
            tasks = torch.cat([batch["decision_tasks"].to(device) for batch in batches])
            logits = model.head(pair_hidden, query_hidden)
            valid = logits.masked_fill(~mask, -torch.inf)
            probabilities = torch.where(
                (tasks == 1)[:, None], valid.softmax(-1), logits.sigmoid(),
            ) * mask
        output = SimpleNamespace(logits=logits, probabilities=probabilities, score_mask=mask)
        return output, hit, [
            size + (length if not hit and index == 0 else 0)
            for index, size in enumerate(lengths)
        ]

    def predict_prepared(self, prepared, *, question_template, frame_times):
        """Score native post-video choices on the service's single GPU worker."""
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
            raw, tokens, forwarded, hits, misses, prefix_tokens = [], 0, 0, 0, 0, 0
            for batch, rows, prefix in question_template.compiled.groups(
                prepared.record, frame_times, prepared.context, prefix_cache=self.prefix_cache,
            ):
                tokens += batch["input_ids"].shape[1]
                if prefix is None:
                    output = self.model(**batch, visual_features=features)
                    forwarded += batch["input_ids"].shape[1]
                else:
                    output, hit, counts = self._score_cached([batch], [features], prefix)
                    forwarded += counts[0]
                    hits += int(hit)
                    misses += int(not hit)
                    prefix_tokens = prefix.length
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
            "forward_input_tokens": forwarded,
            "prefix_tokens": prefix_tokens,
            "prefix_cache_hits": hits,
            "prefix_cache_misses": misses,
        }

    def predict_prepared_batch(self, windows):
        """Yield individual outcomes; None keeps the original sequential route."""
        if not self.prefix_cache:
            yield None
            return
        started = time.perf_counter()
        eligible = []
        for index, window in enumerate(windows):
            try:
                template, prepared = window["question_template"], window["prepared"]
                if template.compiled is None:
                    template.compiled = CompiledQuestions(self.processor, self.torch, template.rows)
                groups = list(template.compiled.groups(
                    prepared.record, window["frame_times"], prepared.context, prefix_cache=True,
                ))
                if (len(groups) != 1 or groups[0][2] is None
                        or (eligible and groups[0][2] is not eligible[0][2][2])):
                    yield None
                    return
                eligible.append((index, window, groups[0]))
            except Exception as exc:
                yield index, exc
        if len(eligible) < 2:
            yield None
            return

        torch = self.torch
        torch.cuda.synchronize()
        try:
            before = time.perf_counter()
            videos = [item[1]["prepared"] for item in eligible]
            device = self.model.get_input_embeddings().weight.device
            pixels = torch.cat([video.pixel_values for video in videos]).to(device)
            grids = torch.tensor([video.record["grid_thw"] for video in videos], device=device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                visual = self.model.backbone.visual(
                    pixels, grid_thw=grids, return_dict=True,
                ).pooler_output
                features = list(visual.split([video.record["tokens"] for video in videos]))
                torch.cuda.synchronize()
            encoded_windows = eligible
            # One shared visual forward; do not sum this value per window.
            visual_seconds = [time.perf_counter() - before] * len(eligible)
        except Exception as exc:
            for index, _window, _group in eligible:
                yield index, exc
            return
        try:
            with torch.inference_mode():
                encoded = time.perf_counter()
                prefix = encoded_windows[0][2][2]
                output, hit, counts = self._score_cached(
                    [item[2][0] for item in encoded_windows], features, prefix,
                )
                cpu = SimpleNamespace(
                    logits=output.logits.float().cpu(),
                    probabilities=output.probabilities.float().cpu(),
                    score_mask=output.score_mask.cpu(),
                )
                torch.cuda.synchronize()
                done = time.perf_counter()
            row_offset = 0
            for position, (index, window, group) in enumerate(encoded_windows):
                batch, rows, _prefix = group
                per_window = SimpleNamespace(**{
                    key: getattr(cpu, key)[row_offset:row_offset + len(rows)]
                    for key in ("logits", "probabilities", "score_mask")
                })
                row_offset += len(rows)
                try:
                    raw = self._format_predictions(
                        per_window, rows, window["prepared"].record, self.decoder,
                    )
                    prepared = window["prepared"]
                    cached = hit or position > 0
                    yield index, (raw, {
                        **prepared.timings,
                        "gpu_batch_size": len(encoded_windows),
                        "visual_batch_size": len(encoded_windows),
                        "visual_encode_seconds": visual_seconds[position],
                        "decision_seconds": done - encoded,
                        "gpu_pipeline_seconds": done - started,
                        "model_pipeline_seconds": prepared.timings["preprocess_seconds"] + done - started,
                        "native_input_tokens": batch["input_ids"].shape[1],
                        "forward_input_tokens": counts[position],
                        "prefix_tokens": prefix.length,
                        "prefix_cache_hits": int(cached),
                        "prefix_cache_misses": int(not cached),
                    })
                except Exception as exc:
                    yield index, exc
        except Exception as exc:
            for index, _window, _group in encoded_windows:
                yield index, exc
