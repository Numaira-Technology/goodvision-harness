"""Bind the spawned native VLM process to the existing paid-request ledger."""
from dataclasses import dataclass
from threading import Lock
from types import MethodType

from ..vlm_independent import _inspector
from .model_ledger import ModelLedger

_lock = Lock()
_ledger = None
_identity = None


@dataclass(frozen=True)
class LedgerInspectorFactory:
    """Picklable factory; one writer in the child, shared by its worker threads."""

    directory: str
    seed_manifest: str
    seed_sha256: str
    owner_lock: str

    def __call__(self, config):
        global _ledger, _identity
        with _lock:
            if _ledger is None:
                _ledger = ModelLedger(self.directory, seed_manifest_path=self.seed_manifest,
                                      seed_manifest_sha256=self.seed_sha256,
                                      owner_lock_path=self.owner_lock)
                _identity = self
            elif _identity != self:
                raise ValueError("native process ledger ownership changed")
        inspector = _ledger.attach(_inspector(config), {
            "stream_session_id": config.stream_session_id, "source_id": config.source_id,
            "lane": config.output_mode, "scheduling_mode": "native_streaming"})
        original = inspector.inspect

        def inspect(instance, job):
            instance.evaluation_context = {
                "request_id": job.request.request_id,
                "source_frame_sequences": [frame.frame_sequence for frame in job.frames]}
            return original(job)

        inspector.inspect = MethodType(inspect, inspector)
        # Formal capture performs only ledgered model requests; skip the native
        # optional unauthenticated HEAD connection probe.
        inspector.warmup = None
        return inspector
