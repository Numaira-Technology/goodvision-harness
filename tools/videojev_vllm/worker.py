"""Explicit state allocation without synthetic model calls or graph capture."""

from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import CompilationTimes


class VideoJevWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch

        # The original decision head uses FP32, not TF32 approximations.
        torch.set_float32_matmul_precision("highest")

    def determine_available_memory(self):
        return self.cache_config.kv_cache_memory_bytes

    def compile_or_warm_up_model(self):
        # Eager execution compiles needed kernels on the first real request.
        # Do not call the stock profile, graph-capture, or pooler warmup paths.
        return CompilationTimes(language_model=0.0, encoder=0.0)
