"""Prepared-window scoring through the pinned in-process vLLM engine."""

import importlib.metadata
import sys
import time
from types import SimpleNamespace

from tools.videojev_probe.contract import CONTEXT, MODEL, REVISION
from tools.videojev_probe.question_templates import CompiledQuestions
from tools.videojev_probe.runtime import VideoJevRuntime


def engine_hf_config(config):
    # Type discovery precedes loading the actual VideoJev configuration.
    if config.model_type.startswith("dummy_"):
        return config
    native = config.backbone_config
    native.architectures = ["VideoJevForPooling"]
    native.videojev_head_dim = config.head_dim
    native.videojev_head_logit_scale = config.head_logit_scale
    return native


class EngineVideoJevRuntime(VideoJevRuntime):
    def __init__(self, cache_dir, *, question_layout="repeated", context=CONTEXT,
                 language_fp8=False):
        import torch
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.models import ModelRegistry
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        from .model import native_positions

        started = time.perf_counter()
        self.torch = torch
        self._native_positions = native_positions
        shared = dict(
            revision=REVISION, cache_dir=str(cache_dir),
            trust_remote_code=True, local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL, **shared)
        self.native_config = AutoConfig.from_pretrained(MODEL, **shared)
        native_class = get_class_from_dynamic_module(
            "modeling_videojev.VideoJevForVideoUnderstanding", MODEL,
            revision=REVISION, cache_dir=str(cache_dir), local_files_only=True,
        )
        native = vars(sys.modules[native_class.__module__])
        self._prepare_query_groups = native["prepare_query_groups"]
        self._format_predictions = native["format_predictions"]
        self.decoder = dict(self.native_config.decoder)
        self._policy_prefix = None
        self.question_layout = question_layout
        self.single_sequence = False
        self.packed_choice_supported = True
        ModelRegistry.register_model(
            "VideoJevForPooling", "tools.videojev_vllm.model:VideoJevForPooling",
        )
        mlp_fp8 = language_fp8 == "mlp"
        quantization_config = None
        if mlp_fp8:
            quantization_config = {"targets": {
                "language_model.layers.*.mlp.gate_up_proj": "fp8_per_tensor",
                "language_model.layers.*.mlp.down_proj": "fp8_per_tensor",
            }}
        arguments = EngineArgs(
            model=MODEL, revision=REVISION, code_revision=REVISION,
            trust_remote_code=True, download_dir=str(cache_dir),
            hf_overrides=engine_hf_config, dtype="bfloat16", runner="pooling",
            quantization="online" if mlp_fp8 else "fp8" if language_fp8 else None,
            quantization_config=quantization_config,
            skip_tokenizer_init=True, distributed_executor_backend="uni",
            tensor_parallel_size=1, pipeline_parallel_size=1,
            worker_cls="tools.videojev_vllm.worker.VideoJevWorker",
            max_model_len=context, max_num_seqs=2,
            max_num_batched_tokens=context * 2,
            enable_chunked_prefill=False, enable_prefix_caching=False,
            mm_processor_cache_gb=0, limit_mm_per_prompt={"video": 1},
            enforce_eager=True, kv_cache_memory_bytes=4 * 1024**3,
            mamba_ssm_cache_dtype="float32", mamba_cache_mode="none",
            disable_log_stats=True,
        )
        config = arguments.create_engine_config()
        torch.set_float32_matmul_precision("highest")
        self.engine = EngineCore(config, UniProcExecutor, log_stats=False)
        self.model = self.engine.model_executor.collective_rpc(
            "apply_model", args=(lambda model: model,),
        )[0]
        torch.cuda.synchronize()
        self.metadata = {
            "model": MODEL, "revision": REVISION, "execution_backend": "vllm-0.30",
            "gpu": torch.cuda.get_device_properties(0).name,
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "model_load_seconds": time.perf_counter() - started,
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "vllm", "flash-attn",
            )},
            "cuda_runtime": torch.version.cuda,
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "attention": "original-vision-fa2/vllm-language",
            "context": context, "question_layout_default": question_layout,
            "packed_choice_supported": True, "single_sequence": False,
            "prefix_cache_enabled": False, "synthetic_model_calls": False,
            "engine_state_bytes": 4 * 1024**3, "mamba_ssm_cache_dtype": "float32",
            "language_linear_quantization": (
                "fp8-online-mlp" if mlp_fp8 else "fp8-online" if language_fp8 else "bfloat16"
            ),
        }

    def _groups(self, window):
        prepared = window["prepared"]
        layout = window["question_layout"] or self.question_layout
        if layout not in ("repeated", "shared"):
            raise ValueError("The engine path uses the original non-prefix layout")
        template = window["question_template"]
        if template is not None:
            if template.compiled is None:
                template.compiled = CompiledQuestions(
                    self.processor, self.torch, template.rows, self._prefix_for_policy,
                )
            return list(template.compiled.groups(
                prepared.record, window["frame_times"], prepared.context, layout,
            ))
        result = []
        for batch, rows in self._prepare_query_groups(
            self.processor, prepared.record, window["questions"], prepared.context,
        ):
            batch, actual_layout, prefix = self._layout_group(prepared, batch, rows, layout)
            result.append((batch, rows, actual_layout, prefix))
        return result

    def _request(self, batch, prepared, request_id):
        from vllm.multimodal.inputs import (
            MultiModalBatchedField, MultiModalFeatureSpec, MultiModalFieldElem,
            MultiModalKwargsItem, PlaceholderRange,
        )
        from vllm.pooling_params import PoolingParams
        from vllm.v1.engine import EngineCoreRequest

        torch = self.torch
        grid = batch["video_grid_thw"]
        ids = batch["input_ids"]
        positions, delta = self._native_positions(
            self.native_config.backbone_config, ids, batch["mm_token_type_ids"], grid,
        )
        slots = ids[0] == self.native_config.backbone_config.video_token_id
        indices = slots.nonzero().flatten()
        first, last = int(indices[0]), int(indices[-1])
        cpu = MultiModalBatchedField(keep_on_cpu=True)
        data = MultiModalKwargsItem({
            "video_pixels": MultiModalFieldElem(prepared.pixel_values, cpu),
            "video_grid": MultiModalFieldElem(grid[0], cpu),
            "native_positions": MultiModalFieldElem(positions, cpu),
            "native_delta": MultiModalFieldElem(torch.tensor(delta), cpu),
        })
        feature = MultiModalFeatureSpec(
            data=data, modality="video", identifier=request_id,
            mm_position=PlaceholderRange(
                offset=first, length=last - first + 1, is_embed=slots[first:last + 1],
            ),
        )
        params = PoolingParams(task="token_embed", extra_kwargs={
            "pairs": batch["pair_indices"].tolist(),
            "queries": batch["query_indices"].tolist(),
            "mask": batch["score_mask"].tolist(),
            "tasks": torch.as_tensor(batch["decision_tasks"]).tolist(),
        })
        return EngineCoreRequest(
            request_id=request_id, prompt_token_ids=ids[0].tolist(),
            mm_features=[feature], sampling_params=None, pooling_params=params,
            arrival_time=time.time(), lora_request=None, cache_salt=None,
            data_parallel_rank=None, external_req_id=request_id,
        )

    def _predict_windows(self, windows):
        torch = self.torch
        started = time.perf_counter()
        self.model.visual_events.clear()
        self.model.language_events.clear()
        pending = {}
        raw, tokens, layouts = ([[] for _ in windows] for _ in range(3))
        failed_returns = [[] for _ in windows]
        errors, preparation = [None for _ in windows], [0.0 for _ in windows]
        for index, window in enumerate(windows):
            before = time.perf_counter()
            try:
                groups = self._groups(window)
                tokens[index] = [int(group[0]["input_ids"].shape[1]) for group in groups]
                layouts[index] = [group[2] for group in groups]
                raw[index] = [None for _ in groups]
                for group_index, (batch, rows, _layout, _prefix) in enumerate(groups):
                    request_id = str(time.time_ns())
                    request = self._request(batch, window["prepared"], request_id)
                    owned, wave = self.engine.preprocess_add_request(request)
                    self.engine.add_request(owned, wave)
                    pending[request_id] = index, group_index, rows
            except Exception as exc:
                errors[index] = exc
                owned_ids = [key for key, value in pending.items() if value[0] == index]
                self.engine.abort_requests(owned_ids)
                for key in owned_ids:
                    del pending[key]
            preparation[index] = time.perf_counter() - before
        execution_error = None
        try:
            while self.engine.scheduler.has_requests():
                outputs, executed = self.engine.step_fn()
                if not outputs or 0 not in outputs:
                    self.engine.post_step(executed)
                    continue
                for result in outputs[0].outputs:
                    if not result.finished or result.request_id not in pending:
                        continue
                    index, group_index, rows = pending.pop(result.request_id)
                    output = None
                    try:
                        if result.pooling_output is None:
                            raise RuntimeError(
                                f"Engine window ended without scores: {result.finish_reason}",
                            )
                        output = result.pooling_output.float().cpu()
                        cpu = SimpleNamespace(
                            logits=output[0], probabilities=output[1], score_mask=output[2].bool(),
                        )
                        raw[index][group_index] = list(self._format_predictions(
                            cpu, rows, windows[index]["prepared"].record, self.decoder,
                        ))
                    except Exception as exc:
                        failed_returns[index].append({
                            "group": group_index, "error": f"{type(exc).__name__}: {exc}",
                            "scores": None if output is None else output.tolist(),
                        })
                        errors[index] = exc
                # Pooling has no draft-token state. Retain returned scores before
                # post-step bookkeeping, which must not discard completed work.
                self.engine.post_step(executed)
        except Exception as exc:
            execution_error = exc
            for index, _group_index, _rows in pending.values():
                errors[index] = exc
        finally:
            # A failed shared execution must not leak requests into the next call.
            if pending:
                try:
                    self.engine.abort_requests(list(pending))
                except Exception as exc:
                    if execution_error is None:
                        execution_error = exc
        for index, _group_index, _rows in pending.values():
            if errors[index] is None:
                errors[index] = RuntimeError("Engine ended before returning this window's scores")
        segments = {}
        if execution_error is None:
            try:
                torch.cuda.synchronize()
                segments = {
                    "visual_encode_seconds": sum(
                        first.elapsed_time(last) for first, last in self.model.visual_events
                    ) / 1000,
                    "gpu_scoring_seconds": sum(
                        first.elapsed_time(last) for first, last in self.model.language_events
                    ) / 1000,
                }
            except Exception as exc:
                execution_error = exc
        done = time.perf_counter()
        # Publish completed windows before a later shared failure reaches the server.
        for index in sorted(range(len(windows)), key=lambda i: errors[i] is not None):
            window = windows[index]
            if errors[index] is not None:
                errors[index].raw = {
                    "reports": [report for group in raw[index] if group is not None for report in group],
                    "failed_returns": failed_returns[index],
                }
                yield index, errors[index]
                continue
            prepared = window["prepared"]
            reports = [report for group in raw[index] for report in group]
            timings = {
                **prepared.timings,
                "question_layout_requested": window["question_layout"] or self.question_layout,
                "question_layout": layouts[index][0] if len(set(layouts[index])) == 1 else "mixed",
                "question_layout_groups": layouts[index],
                "native_group_count": len(layouts[index]),
                "native_input_tokens": sum(tokens[index]),
                "native_group_input_tokens": tokens[index],
                "question_group_prepare_seconds": preparation[index],
                "prefix_cache_groups": [], "gpu_batch_size": len(windows),
                **segments,
                "gpu_pipeline_seconds": done - started,
                "model_pipeline_seconds": prepared.timings["preprocess_seconds"] + done - started,
                "timings_scope": "engine-batch" if len(windows) > 1 else "window",
                "execution_backend": "vllm-0.30",
            }
            if segments:
                timings["decision_seconds"] = done - started - segments["visual_encode_seconds"]
            yield index, (reports, timings)
        if execution_error is not None:
            # Preserve the original CUDA/device exception for existing shared-fault
            # handling, including failures after all score tensors were returned.
            raise execution_error

    def predict_prepared(self, prepared, questions, *, frame_times=None,
                         question_layout=None, question_template=None):
        windows = [{
            "prepared": prepared, "questions": questions, "frame_times": frame_times,
            "question_layout": question_layout or self.question_layout,
            "question_template": question_template,
        }]
        outcomes = self._predict_windows(windows)
        result = None
        try:
            _, result = next(outcomes)
            try:
                next(outcomes)
            except StopIteration:
                if isinstance(result, Exception):
                    raise result
                return result
        except Exception as exc:
            if isinstance(result, Exception):
                exc.raw = getattr(result, "raw", None)
            elif result is not None:
                exc.raw = {"reports": result[0]}
            raise
        finally:
            outcomes.close()

    def predict_prepared_batch(self, windows):
        yield from self._predict_windows(windows)

    def close(self):
        self.engine.shutdown()
