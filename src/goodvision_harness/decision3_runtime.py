"""Prepare official Decision 3 video questions on CPU and batch independent rows."""

import importlib.metadata
import io
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from .videojev_contract import CONTEXT


MODEL = "vllm-sr/d3-nano"
REVISION = "462efc366229ea268cf4eed3738b3fc4415e4817"
MAX_PIXELS = 200704
QUESTION_BATCH_SIZE = 8


@dataclass(frozen=True)
class _PreparedVideo:
    frames: object
    native: object
    rows: tuple
    template: object
    frame_times: object
    timings: dict


class _CompiledQuestions:
    """Cache official rendering and two exact, complete CPU tokenizations."""

    def __init__(self, runtime, rows):
        self.runtime = runtime
        self.rows = tuple(rows)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.questions, self.texts, self.errors = {}, {}, {}
        marker = "{source_times}"
        while any(marker in part for row in self.rows for part in (*row["query_parts"], *row["options"])):
            marker = "{" + marker + "}"
        self.marker = marker
        native, helpers = runtime.native, runtime.helpers
        for row in self.rows:
            key = row["id"]
            try:
                question = helpers.normalize_question({
                    "type": "choice",
                    "instructions": marker.join(row["query_parts"]),
                    "criteria": {option: None for option in row["options"]},
                })
                text = native.video_text("", question.rendered, 0, 1)
                if text.count(helpers.IMAGE_TOKEN) or text.count(helpers.VIDEO_TOKEN) != 1:
                    raise ValueError("the state or question contains a literal image or video placeholder token")
                if text.count(marker) != 1:
                    raise ValueError("a question must contain exactly one source-time slot")
                self.questions[key] = question
                self.texts[key] = text
            except ValueError as error:
                self.errors[key] = {
                    "type": "choice", "error": "invalid_question", "message": str(error),
                }

    def bind(self, prepared, times):
        """Tokenize whole official prompts; never concatenate independently encoded BPE spans."""
        key = (times, tuple(prepared.video_texts))
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            native = self.runtime.native
            processor = native.processor
            texts = {name: text.replace(self.marker, times) for name, text in self.texts.items()}
            keys = list(texts)
            lengths, encoded_rows = {}, []
            errors = dict(self.errors)
            if keys:
                planned = processor.tokenizer(
                    [texts[name] for name in keys] + prepared.video_texts,
                    add_special_tokens=False,
                )["input_ids"]
                visual = sum(len(ids) for ids in planned[len(keys):])
                expanded = [processor.get_text_with_replacements(
                    [texts[name]], [], prepared.video_texts,
                )[0][0] for name in keys]
                encoded = processor.tokenizer(
                    expanded, padding=False, return_token_type_ids=False,
                )["input_ids"]
                for name, before, ids in zip(keys, planned, encoded):
                    length = len(before) - 1 + visual
                    if native.max_length is not None and length > native.max_length:
                        errors[name] = {
                            "type": self.questions[name].kind,
                            "error": "max_length_exceeded",
                            "message": f"the question prompt has {length} tokens ({visual} for "
                            f"0 image(s) and 1 video(s)), over the maximum context length of "
                            f"{native.max_length} tokens; nothing was truncated",
                        }
                        continue
                    if len(ids) != length:
                        raise RuntimeError(f"planned {length} input tokens, the processor produced {len(ids)}")
                    tokens = self.runtime.torch.tensor(ids, dtype=self.runtime.torch.long)
                    types = self.runtime.torch.as_tensor(
                        processor.create_mm_token_type_ids(tokens[None]),
                        dtype=self.runtime.torch.long,
                    )[0]
                    lengths[name] = length
                    encoded_rows.append((name, tokens, types, len(self.questions[name].keys)))
            value = (texts, lengths, errors, tuple(encoded_rows))
            self.cache[key] = value
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            return value


class Decision3Runtime:
    """Keep camera evidence independent while sharing visual and question forwards."""

    prefix_cache = False

    def __init__(self, cache_dir):
        import torch
        from transformers import AutoModel

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Decision 3 requires a CUDA GPU supporting BF16")
        self.torch = torch
        self._template_lock = threading.Lock()
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True,
            device="cuda:0", cache_dir=str(cache_dir), batch_size=QUESTION_BATCH_SIZE,
            verify="none",
        )
        self.native = self.model.runtime
        self.helpers = sys.modules[type(self.native).__module__]
        if self.native.permutation_average:
            raise ValueError("this fixed Decision 3 path requires original option order")
        self.native.synchronize()
        props = torch.cuda.get_device_properties(0)
        self.metadata = {
            "model": MODEL, "revision": REVISION,
            "gpu": props.name, "gpu_total_bytes": props.total_memory,
            "model_load_seconds": time.perf_counter() - started,
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "flash-linear-attention", "fla-core",
            )},
            "cuda_runtime": torch.version.cuda,
            "precision": "BF16 backbone, FP32 readout",
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "question_batch_size": QUESTION_BATCH_SIZE,
            "prefix_cache": False, "max_pixels_per_frame": MAX_PIXELS,
            "max_input_tokens": self.model.max_input_tokens,
            "context": self.model.max_input_tokens,
            "api": "official_choice_independent_rows",
        }

    def _compiled(self, template):
        with self._template_lock:
            compiled = template.compiled
            if not isinstance(compiled, _CompiledQuestions) or compiled.runtime is not self:
                compiled = _CompiledQuestions(self, template.rows)
                template.compiled = compiled
            return compiled

    def prepare(self, payloads, fps, *, question_template=None, frame_times=None,
                context=CONTEXT, max_pixels=MAX_PIXELS):
        """Decode, process and tokenize on a CPU worker when the bound questions are supplied.

        Context is accepted for the shared interface; the checkpoint's own input limit and
        media budget remain authoritative. Unbound callers retain a single-window fallback.
        """
        import numpy as np
        from PIL import Image

        started = time.perf_counter()
        if (not isinstance(payloads, (list, tuple)) or len(payloads) != 2
                or type(fps) not in (int, float) or not math.isfinite(fps)
                or not math.isclose(fps, 2.0, rel_tol=0, abs_tol=1e-8)
                or type(max_pixels) is not int or max_pixels != MAX_PIXELS):
            raise ValueError("provide two original JPEGs at 2 FPS with the fixed official pixel budget")
        frames = []
        for payload in payloads:
            if not isinstance(payload, bytes):
                raise ValueError("original JPEG payloads must be bytes")
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG" or image.width * image.height > 16 * 1024**2:
                    raise ValueError("original evidence must be bounded JPEG frames")
                frames.append(np.array(image.convert("RGB")))
        if len({frame.shape for frame in frames}) != 1:
            raise ValueError("original evidence frames must share geometry")
        array = np.ascontiguousarray(np.stack(frames))
        timings = {
            "input_frame_count": len(frames), "internal_temporal_padding_frames": 0,
            "native_temporal_fps": 2.0,
            "source_height": array.shape[1], "source_width": array.shape[2],
        }
        prepared = _PreparedVideo(array, None, (), None, None, timings)
        if question_template is not None and frame_times is not None:
            prepared = self._bind(prepared, question_template, frame_times)
        timings["preprocess_seconds"] = time.perf_counter() - started
        return prepared

    def _bind(self, prepared, template, frame_times):
        times = json.dumps(frame_times, allow_nan=False, separators=(",", ":"))
        if prepared.template is template and prepared.frame_times == times:
            return prepared
        native = self.native
        # Reuse the official media planner and CPU processor without rendering questions twice.
        media = native._prepare_videos("", {}, [], native.load_videos([prepared.frames]))
        if any(value.device.type != "cpu" for value in media.media.values()):
            raise ValueError("video preparation must produce CPU tensors")
        compiled = self._compiled(template)
        texts, lengths, errors, rows = compiled.bind(media, times)
        media.keys = [row["id"] for row in template.rows]
        media.questions = dict(compiled.questions)
        media.texts = {key: text for key, text in texts.items() if key in lengths}
        media.lengths, media.errors = dict(lengths), dict(errors)
        return _PreparedVideo(
            prepared.frames, media, rows, template, times, prepared.timings,
        )

    def _logits(self, rows, prepared, features):
        """Use the official embeddings, positions, attention encoder and masked readout."""
        torch, native = self.torch, self.native
        backbone = native.backbone
        width = max(len(row[2]) for row in rows)
        padded = any(len(row[2]) != width for row in rows)
        if padded:
            ids = torch.full((len(rows), width), native.pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            types = torch.zeros_like(ids)
            for index, (_window, _key, tokens, token_types, _count) in enumerate(rows):
                ids[index, -len(tokens):] = tokens
                mask[index, -len(tokens):] = 1
                types[index, -len(tokens):] = token_types
        else:
            ids = torch.stack([row[2] for row in rows])
            mask = torch.ones_like(ids)
            types = torch.stack([row[3] for row in rows])
        ids, mask, types = (value.to(native.device) for value in (ids, mask, types))
        grids = torch.cat([
            prepared[row[0]].native.media["video_grid_thw"] for row in rows
        ]).to(native.device)
        with torch.inference_mode(), self.helpers.sdpa_backends(native.device):
            embeds = backbone.get_input_embeddings()(ids)
            # The flattened placeholder order is row order; a row gets only its own camera.
            values = torch.cat([features[row[0]] for row in rows]).to(embeds.device, embeds.dtype)
            _, video_mask = backbone.get_placeholder_mask(
                ids, inputs_embeds=embeds, video_features=values,
            )
            embeds = embeds.masked_scatter(video_mask, values)
            positions = backbone.compute_3d_position_ids(
                input_ids=ids, image_grid_thw=None, video_grid_thw=grids,
                inputs_embeds=embeds, attention_mask=mask, past_key_values=None,
                mm_token_type_ids=types,
            )
            fused = native.fast.fused if native.fast is not None else None
            if fused is not None:
                rotary = native.fast.lm.rotary_emb(embeds, positions)
                hidden = fused.forward(
                    embeds, mask.bool()[:, None, None, :],
                    mask.reshape(-1) if padded else None, rotary,
                )[:, -1]
            elif native.encoder is not None:
                hidden = native.encoder(embeds, mask, positions)[:, -1]
            else:
                hidden = backbone.language_model(
                    input_ids=None, position_ids=positions, attention_mask=mask,
                    past_key_values=None, inputs_embeds=embeds, use_cache=False,
                ).last_hidden_state[:, -1]
            if native.readout_dtype == "float32":
                logits = hidden.float() @ native.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, native.readout).float()
            counts = torch.as_tensor([row[4] for row in rows], device=native.device)[:, None]
            invalid = torch.arange(self.helpers.MAX_OPTIONS, device=native.device)[None] >= counts
            return logits.masked_fill(invalid, float("-inf"))

    def predict_prepared(self, prepared, *, question_template, frame_times):
        for _index, result in self.predict_prepared_batch([{
            "prepared": prepared, "question_template": question_template, "frame_times": frame_times,
        }]):
            if isinstance(result, Exception):
                raise result
            return result
        raise RuntimeError("Decision 3 did not return a window result")

    def predict_prepared_batch(self, windows):
        """Yield each completed window once, with independent question rows capped at eight."""
        # Import only at execution time, after both runtime and service modules are loaded.
        from .videojev_service import _shared_cuda_failure

        native = self.native
        started = time.perf_counter()
        prepared, probabilities, pending, decision_seconds = {}, {}, {}, {}
        rows = []
        for index, window in enumerate(windows):
            try:
                before = time.perf_counter()
                video = self._bind(window["prepared"], window["question_template"], window["frame_times"])
                # An inference's row errors must not mutate a reusable prepared window.
                video = replace(video, native=replace(video.native, errors=dict(video.native.errors)))
                prepared[index] = video
                probabilities[index], decision_seconds[index] = {}, 0.0
                pending[index] = len(video.rows)
                rows.extend((index, *row) for row in video.rows)
                if not video.rows:
                    yield index, (native.respond(video.native, {}, 0), {
                        **video.timings, "gpu_batch_size": 0, "visual_batch_size": 0,
                        "gpu_pipeline_seconds": 0.0,
                        "model_pipeline_seconds": video.timings["preprocess_seconds"] + time.perf_counter() - before,
                    })
            except Exception as error:
                yield index, error
        active = [index for index in prepared if pending[index]]
        if not active:
            return
        try:
            native.synchronize()
            visual_start = time.perf_counter()
            media = {key: self.torch.cat([
                prepared[index].native.media[key] for index in active
            ]) for key in ("pixel_values_videos", "video_grid_thw")}
            visual = native.media_features(media)["video"]
            features = dict(zip(active, visual))
            if len(visual) != len(active):
                raise RuntimeError("the vision tower did not return one feature sequence per window")
            native.synchronize()
            visual_seconds = time.perf_counter() - visual_start
        except Exception as error:
            for index in active:
                yield index, error
            return
        for start in range(0, len(rows), QUESTION_BATCH_SIZE):
            chunk = rows[start:start + QUESTION_BATCH_SIZE]
            affected = set(row[0] for row in chunk)
            before = time.perf_counter()
            try:
                scores = (self._logits(chunk, prepared, features) / native.temperature).softmax(-1).cpu().tolist()
                if len(scores) != len(chunk):
                    raise RuntimeError("the readout did not return one probability row per question")
                native.synchronize()
                updates = [
                    (index, key, values[:count])
                    for (index, key, _ids, _types, count), values in zip(chunk, scores)
                ]
            except Exception as error:
                if _shared_cuda_failure(error):
                    for index in active:
                        if pending[index]:
                            yield index, error
                    return
                for index, key, _ids, _types, _count in chunk:
                    prepared[index].native.errors[key] = {
                        "type": prepared[index].native.questions[key].kind,
                        "error": "inference_failed",
                        "message": f"{type(error).__name__}: {error}",
                    }
                    pending[index] -= 1
            else:
                for index, key, values in updates:
                    probabilities[index][key] = values
                    pending[index] -= 1
            elapsed = time.perf_counter() - before
            for index in affected:
                decision_seconds[index] += elapsed
            for index in sorted(affected):
                if pending[index]:
                    continue
                video = prepared[index]
                try:
                    tokens = sum(video.native.lengths.values())
                    raw = native.respond(video.native, probabilities[index], tokens)
                    # Shared batch wall times must not be summed across windows as GPU work.
                    elapsed = time.perf_counter() - started
                    yield index, (raw, {
                        **video.timings, "gpu_batch_size": len(active), "visual_batch_size": len(active),
                        "visual_encode_seconds": visual_seconds,
                        "decision_seconds": decision_seconds[index], "gpu_pipeline_seconds": elapsed,
                        "model_pipeline_seconds": video.timings["preprocess_seconds"] + elapsed,
                        "native_input_tokens": tokens, "forward_input_tokens": tokens,
                    })
                except Exception as error:
                    yield index, error

    def validate_response(self, raw, rows):
        """Accept official per-question errors without discarding valid sibling answers."""
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise ValueError("Decision 3 response must contain the official answers mapping")
