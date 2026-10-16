"""Bind the spawned native VLM process to the existing paid-request ledger."""
from dataclasses import dataclass, replace
import logging
import math
from threading import Lock
import time
from types import MethodType

from ..model_api import ModelAPIError
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
        inspector.evaluation_stop_event = None
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
            instance.last_rate_limit_backoff_seconds = 0.0
            # Match the offline evaluation's explicit-429 recovery policy. Each
            # send still passes through the cumulative three-attempt ledger.
            # Sleep outside inspect's per-request HTTP deadline, using the same
            # job, prompt and output budget on every attempt.
            for attempt in range(3):
                stop = instance.evaluation_stop_event
                if stop is not None and stop.is_set():
                    raise ModelAPIError("evaluation observation stopped", code="cancelled")
                try:
                    return original(job)
                except ModelAPIError as exc:
                    if exc.status_code != 429 or attempt == 2:
                        raise
                    retry_after = getattr(exc, "evaluation_diagnostic", {}).get(
                        "retry_after_seconds")
                    delay = max(60.0, retry_after or 0.0)
                    if not math.isfinite(delay) or delay > 300.0:
                        raise
                    receipt = instance.last_ledger_receipt or {}
                    if receipt.get("attempt", 0) >= 3:
                        raise
                    row = {"kind": "vlm_rate_limit_backoff", "lane": config.output_mode,
                           "request_id": job.request.request_id,
                           "source_frame_sequences": sequences,
                           "call_id": receipt.get("call_id"),
                           "request_sha256": receipt.get("request_sha256"),
                           "seconds": delay, "started_epoch": time.time(),
                           "same_job_and_parameters": True}
                    with _ledger.lock:
                        _ledger._append("rate-limit-backoff.jsonl", row)
                    waiting = time.monotonic()
                    try:
                        if stop is None:
                            time.sleep(delay)
                        elif stop.wait(delay):
                            raise ModelAPIError("evaluation backoff stopped", code="cancelled")
                    finally:
                        instance.last_rate_limit_backoff_seconds += time.monotonic() - waiting

        inspector.inspect = MethodType(inspect, inspector)
        # Formal capture performs only ledgered model requests; skip the native
        # optional unauthenticated HEAD connection probe.
        inspector.warmup = None
        return inspector
