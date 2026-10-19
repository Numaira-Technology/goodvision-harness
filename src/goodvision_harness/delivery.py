"""Optional, non-durable webhook delivery for analysis results."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from .privacy import public_safe_value

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class SignedWebhookDisabledError(ValueError):
    """A public-safe refusal of the retired signed delivery protocol."""


@dataclass(slots=True)
class DeliveryStats:
    submitted: int = 0
    delivered: int = 0
    failed: int = 0
    dropped: int = 0
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class _DeliveryItem:
    body: bytes
    message_id: str


class WebhookDelivery:
    """Deliver result JSON on a bounded background queue.

    This adapter is intentionally non-durable. Clients that need replay should
    consume the SSE endpoint or persist results on the receiving side.
    """

    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        queue_size: int = 100,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if secret:
            raise SignedWebhookDisabledError(
                "Signed webhook delivery is disabled in this branch. "
                "Disable the webhook configuration; no unsigned fallback is allowed."
            )
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("webhook URL must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._transport = transport
        self._queue: queue.Queue[_DeliveryItem] = queue.Queue(maxsize=queue_size)
        self._stats = DeliveryStats()
        self._stats_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("webhook delivery is closed")
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="analysis-webhook",
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Mapping[str, Any]) -> bool:
        if self._closed:
            return False
        safe_payload = public_safe_value(dict(payload))
        message_id = str(safe_payload.get("message_id", "")).strip()
        if not message_id:
            raise ValueError("webhook payload requires message_id")
        body = json.dumps(
            safe_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        item = _DeliveryItem(body=body, message_id=message_id)
        with self._stats_lock:
            self._stats.submitted += 1
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                discarded = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                discarded = None
            if discarded is not None:
                with self._stats_lock:
                    self._stats.dropped += 1
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                with self._stats_lock:
                    self._stats.dropped += 1
                return False

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return asdict(self._stats)

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = time.monotonic() + timeout_seconds
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout_seconds: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        thread = self._thread
        if thread is None:
            self._discard_pending()
            self._stop.set()
            return
        if not self.flush(timeout_seconds):
            self._discard_pending()
        self._stop.set()
        thread.join(timeout_seconds)

    def _discard_pending(self) -> None:
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            dropped += 1
        if dropped:
            with self._stats_lock:
                self._stats.dropped += dropped

    def _run(self) -> None:
        with httpx.Client(
            timeout=self.timeout_seconds,
            transport=self._transport,
        ) as client:
            while True:
                if self._stop.is_set() and self._queue.empty():
                    return
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._deliver(client, item)
                finally:
                    self._queue.task_done()

    def _deliver(self, client: httpx.Client, item: _DeliveryItem) -> None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": item.message_id,
            "User-Agent": "goodvision-harness/1",
        }
        last_error = "delivery failed"
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(self.url, content=item.body, headers=headers)
                if 200 <= response.status_code < 300:
                    with self._stats_lock:
                        self._stats.delivered += 1
                        self._stats.last_error = None
                    return
                last_error = f"webhook returned HTTP {response.status_code}"
                retryable = response.status_code in _RETRYABLE_STATUS
            except httpx.HTTPError:
                last_error = "webhook request failed"
                retryable = True

            if not retryable or attempt >= self.max_retries or self._stop.is_set():
                break
            delay = self.retry_backoff_seconds * (2**attempt)
            if delay and self._stop.wait(delay):
                break

        with self._stats_lock:
            self._stats.failed += 1
            self._stats.last_error = last_error
