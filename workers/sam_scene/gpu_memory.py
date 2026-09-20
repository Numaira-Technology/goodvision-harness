"""Same-process CUDA accounting and idle-only allocator cache reclamation."""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time

LOGGER = logging.getLogger(__name__)


class GPUMemory:
    def __init__(self, cuda=None):
        self._cuda_override = cuda
        self._mutex = threading.RLock()
        self._pending = False
        self._reclaims = 0
        self._last_reclaim = None
        self._error = None

    def _cuda(self):
        if self._cuda_override is not None:
            return self._cuda_override
        # Monitoring must not import torch or initialize a second CUDA context.
        torch = sys.modules.get("torch")
        return getattr(torch, "cuda", None)

    def mark_pending(self):
        with self._mutex:
            self._pending = True

    def reclaim_if_idle(self, active):
        """Caller holds the shared GPU admission slot until this returns."""
        with self._mutex:
            if active or not self._pending:
                return False
            cuda = self._cuda()
            if cuda is None or not cuda.is_initialized():
                return False
            try:
                device = cuda.current_device()
                before = cuda.memory_reserved(device)
                # Discarded native tracking state may contain reference cycles.
                gc.collect()
                with cuda.device(device):
                    cuda.synchronize(device)
                    cuda.empty_cache()
                after = cuda.memory_reserved(device)
                self._last_reclaim = {
                    "time": int(time.time()),
                    "released_bytes": max(0, before - after),
                }
                self._reclaims += 1
                self._pending = False
                self._error = None
                LOGGER.info(
                    "Idle CUDA cache reclaimed bytes=%d", max(0, before - after)
                )
                return True
            except Exception:
                self._error = "cache_reclaim_failed"
                LOGGER.exception("Idle CUDA cache reclamation failed")
                return False

    def snapshot(self):
        with self._mutex:
            result = {
                "available": False,
                "reason": "cuda_not_initialized",
                "reclaim_pending": self._pending,
                "reclaim_count": self._reclaims,
                "last_reclaim": self._last_reclaim,
                "error_code": self._error,
            }
            cuda = self._cuda()
            try:
                if cuda is None or not cuda.is_initialized():
                    return result
                device = cuda.current_device()
                allocated = cuda.memory_allocated(device)
                reserved = cuda.memory_reserved(device)
                free, total = cuda.mem_get_info(device)
                return result | {
                    "available": True,
                    "reason": None,
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "unused_reserved_bytes": max(0, reserved - allocated),
                    "peak_allocated_bytes": cuda.max_memory_allocated(device),
                    "device_free_bytes": free,
                    "device_total_bytes": total,
                }
            except Exception:
                return result | {"reason": "telemetry_unavailable"}


def maintain_idle_gpu(sessions, admission, memory):
    """Also covers expiry and abandoned requests, never racing live CUDA work."""
    if not admission.acquire(blocking=False):
        return False
    try:
        sessions.expire()
        return memory.reclaim_if_idle(sessions.has_active_session)
    finally:
        admission.release()
