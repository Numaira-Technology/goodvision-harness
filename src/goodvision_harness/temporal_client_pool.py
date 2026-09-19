"""Service-owned VLM threads and clients, with independent per-run leases."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread, current_thread
from typing import Any

from .latency_trace import log_timing, trace_id
from .model_api import ModelAPIError
from .scene_inspection import TemporalInspector


def _unavailable() -> ModelAPIError:
    return ModelAPIError(
        "temporal client lease is no longer available", code="cancelled"
    )


@dataclass(frozen=True)
class _Configuration:
    revision: int
    factory: Callable[[], TemporalInspector]


@dataclass
class _Call:
    lease: _Lease
    job: Any
    future: Future
    submitted_at: float


class _Lease:
    def __init__(self, actor, revision, accept):
        self._actor = actor
        self._revision = revision
        self._accept = accept
        self._closed = Event()

    def inspect(self, job):
        return self._actor.submit(self, job).result()

    def close(self):
        # A retiring run must never close a connection borrowed by a new run.
        self._closed.set()
        self._actor.cancel_pending(self)

    def accepts(self, job):
        return not self._closed.is_set() and (
            self._accept is None or self._accept(job)
        )


class _Actor:
    def __init__(self, index):
        self._condition = Condition()
        self._configuration: _Configuration | None = None
        self._pending: _Call | None = None
        self._closed = False
        self._warm_requested = False
        self._last_use = 0.0
        self.thread = Thread(
            target=self._run, name=f"persistent-temporal-model-{index}", daemon=True
        )
        self.thread.start()

    def configure(self, configuration):
        with self._condition:
            changed = self._configuration != configuration
            self._configuration = configuration
            if changed:
                self._cancel_pending_locked()
            # Refresh an idle pool before a later run, but do not add a HEAD
            # request ahead of every frame or every quick run restart.
            self._warm_requested = configuration is not None and (
                changed or time.monotonic() - self._last_use >= 45
            )
            self._condition.notify()

    def submit(self, lease, job):
        with self._condition:
            if self._closed or lease._closed.is_set() or (
                self._configuration is None
                or self._configuration.revision != lease._revision
            ):
                raise _unavailable()
            if self._pending is not None:
                raise ModelAPIError("temporal client queue is full", code="overloaded")
            future = Future()
            self._pending = _Call(lease, job, future, time.monotonic())
            self._condition.notify()
            return future

    def _cancel_pending_locked(self, lease=None):
        if self._pending is not None and (
            lease is None or self._pending.lease is lease
        ):
            pending, self._pending = self._pending, None
            pending.future.set_exception(_unavailable())

    def cancel_pending(self, lease):
        with self._condition:
            self._cancel_pending_locked(lease)

    def stop(self):
        with self._condition:
            self._closed = True
            self._configuration = None
            self._cancel_pending_locked()
            self._condition.notify()

    @staticmethod
    def _close(client):
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _run(self):
        client = configuration = None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda current=configuration: (
                        self._closed or self._pending is not None
                        or self._configuration != current or self._warm_requested
                    ))
                    if self._closed:
                        return
                    desired = self._configuration
                    call, self._pending = self._pending, None
                    warm, self._warm_requested = self._warm_requested, False
                try:
                    if desired != configuration:
                        self._close(client)
                        client = None
                        configuration = desired
                    if desired is None:
                        continue
                    if client is None:
                        client = desired.factory()
                        warm = True
                    if warm:
                        warmup = getattr(client, "warmup", None)
                        if callable(warmup):
                            try:
                                warmup()
                            except Exception:
                                # Prewarming is best effort, never an admission
                                # or model-health check for actual inference.
                                pass
                    if call is not None:
                        with self._condition:
                            current = not self._closed and (
                                self._configuration == desired
                                and desired.revision == call.lease._revision
                            )
                        if not current or not call.lease.accepts(call.job):
                            raise _unavailable()
                        log_timing(
                            "vlm_pool_dispatch_timing",
                            stream_session_id=trace_id(getattr(
                                call.job, "stream_session_id", None
                            )),
                            request_id=trace_id(getattr(
                                getattr(call.job, "request", None), "request_id", None
                            )),
                            pool_queue_wait_seconds=(
                                time.monotonic() - call.submitted_at
                            ),
                        )
                        result = client.inspect(call.job)
                        # A completed old request belongs only to its own future;
                        # it cannot appear in another run's completion queue.
                        if not call.lease.accepts(call.job):
                            raise _unavailable()
                        with self._condition:
                            self._last_use = time.monotonic()
                        call.future.set_result(result)
                except BaseException as error:
                    if call is not None and not call.future.done():
                        with self._condition:
                            self._last_use = time.monotonic()
                        call.future.set_exception(error)
                finally:
                    with self._condition:
                        self._last_use = time.monotonic()
                    # Owner threads outlive runs. Do not pin their last JPEGs,
                    # session predicate or an obsolete client's bound method.
                    call = result = warmup = None
        finally:
            self._close(client)


class TemporalClientPool:
    """At most four persistent owner threads, one bounded queue per owner.

    prepare() is non-blocking: each owner initializes and prewarms its client in
    parallel with run startup. Inspector loops never migrate across threads.
    Configuration changes retire clients on their owner after any active call.
    Raw configuration/credentials must stay in the factory, never in log fields.
    """

    def __init__(self):
        self._lock = RLock()
        self._actors: list[_Actor] = []
        self._key = None
        self._size = 0
        self._revision = 0
        self._configuration = None
        self._closed = False

    def prepare(self, key, factory, size, *, accept=None):
        if type(size) is not int or not 1 <= size <= 4:
            raise ValueError("temporal client count must be within 1..4")
        with self._lock:
            if self._closed:
                raise _unavailable()
            if (key, size) != (self._key, self._size):
                self._revision += 1
                self._key, self._size = key, size
                self._configuration = _Configuration(self._revision, factory)
            while len(self._actors) < size:
                self._actors.append(_Actor(len(self._actors)))
            for index, actor in enumerate(self._actors):
                actor.configure(self._configuration if index < size else None)
            return tuple(
                _Lease(actor, self._revision, accept) for actor in self._actors[:size]
            )

    def invalidate(self):
        with self._lock:
            self._revision += 1
            self._key, self._size, self._configuration = None, 0, None
            for actor in self._actors:
                actor.configure(None)

    def close(self, timeout_seconds=1.0):
        if type(timeout_seconds) not in (int, float) or (
            not math.isfinite(timeout_seconds) or timeout_seconds < 0
        ):
            raise ValueError("close timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout_seconds
        with self._lock:
            self._closed = True
            self._configuration = None
            actors = tuple(self._actors)
            for actor in actors:
                actor.stop()
        for actor in actors:
            if actor.thread is not current_thread():
                actor.thread.join(max(0.0, deadline - time.monotonic()))
