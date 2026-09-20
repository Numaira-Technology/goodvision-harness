"""Bounded pre-admission work for a live network source.

Durable face admission and inference still run in their original order. The
capture thread only offers candidate windows: a full queue rejects the new
candidate before any identity, evidence admission or model state is changed.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class FaceAdmissionQueue:
    def __init__(self, admit, *, capacity=2, max_bytes=32 * 1024**2):
        self._admit = admit
        self._capacity, self._max_bytes = capacity, max_bytes
        self._lock = Lock()
        self._count = self._bytes = 0
        self._error = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="face-durable-admission",
        )

    def raise_if_failed(self):
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("live face admission failed") from error

    def offer(self, window, *, branch):
        size = sum(
            len(frame.jpeg_bytes) + len(frame.presentation_jpeg_bytes or b"")
            for frame in window.frames
        )
        with self._lock:
            if self._closed or self._error is not None:
                raise RuntimeError("live face admission is unavailable")
            if self._count >= self._capacity or self._bytes + size > self._max_bytes:
                return False
            self._count += 1
            self._bytes += size
            try:
                self._executor.submit(self._run, window, branch, size)
            except BaseException:
                self._count -= 1
                self._bytes -= size
                raise
        return True

    def _run(self, window, branch, size):
        try:
            self.raise_if_failed()
            self._admit(window, branch=branch)
        except Exception as exc:
            with self._lock:
                if self._error is None:
                    self._error = exc
        finally:
            with self._lock:
                self._count -= 1
                self._bytes -= size

    def close(self):
        # Drain offered candidates before the existing session waits for durable
        # face futures. Do not cancel work while its SQLite admission is running.
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.raise_if_failed()
