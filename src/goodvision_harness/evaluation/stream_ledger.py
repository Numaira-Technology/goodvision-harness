"""Bind the spawned native VLM process to the existing paid-request ledger."""
from dataclasses import dataclass, replace
import logging
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
        options = dict(config.options)
        transport = options.pop("evaluation_transport", {})
        options["max_tokens"] = transport.get("lane_max_tokens", {}).get(
            config.output_mode, options.get("max_tokens", 1600))
        try:
            inspector = _inspector(replace(config, options=options))
        except Exception as exc:
            # Retain a safe diagnostic when the native worker reduces the error
            # to inspection_failed; never log credentials or exception values.
            logging.getLogger(__name__).error(
                "evaluation_inspector_initialization_failed lane=%s exception_type=%s",
                config.output_mode, type(exc).__name__)
            raise
        inspector = _ledger.attach(inspector, {
            "stream_session_id": config.stream_session_id, "source_id": config.source_id,
            "lane": config.output_mode, "scheduling_mode": "native_streaming"})
        original = inspector.inspect
        default_budget = inspector._parameters["max_tokens"]
        legacy_budgets = transport.get("legacy_window_budgets", {})

        def inspect(instance, job):
            sequences = [frame.frame_sequence for frame in job.frames]
            budget_key = config.output_mode + ":" + ",".join(map(str, sequences))
            budget = legacy_budgets.get(budget_key, default_budget)
            if type(budget) is not int or not 1 <= budget <= 16384:
                raise ValueError("temporal output token budget must be bounded")
            # Each native worker thread owns its inspector. Apply the frozen
            # window budget before payload construction and ledger hashing.
            instance._parameters["max_tokens"] = budget
            instance.evaluation_context = {
                "request_id": job.request.request_id,
                "source_frame_sequences": sequences}
            return original(job)

        inspector.inspect = MethodType(inspect, inspector)
        # Formal capture performs only ledgered model requests; skip the native
        # optional unauthenticated HEAD connection probe.
        inspector.warmup = None
        return inspector
