"""Coalesced, session-owned background identity review."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Condition, Thread


class IdentityReviewWorker:
    def __init__(self, review: Callable[[], None]) -> None:
        self._review = review
        self._condition = Condition()
        self._requested = self._completed = 0
        self._closing = False
        self._failed = False
        self._thread = Thread(
            target=self._run, name="person-identity-review", daemon=True,
        )
        self._thread.start()

    def wake(self) -> None:
        with self._condition:
            if not self._closing:
                self._requested += 1
                self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closing or self._requested > self._completed
                )
                if self._requested == self._completed and self._closing:
                    return
                revision = self._requested
            try:
                self._review()
            except Exception:
                with self._condition:
                    self._failed = True
            finally:
                with self._condition:
                    self._completed = revision
                    self._condition.notify_all()

    def wait_until_idle(self, timeout_seconds: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._completed < self._requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def stats(self) -> dict:
        with self._condition:
            return {
                "state": "failed" if self._failed else (
                    "running" if self._completed < self._requested else "idle"
                ),
                "pending": self._requested > self._completed,
            }

    def close(self, timeout_seconds: float = 0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout_seconds)
