"""Process-isolated WHEP reception with a single bounded shared BGR slot.

Only complete decoded images cross the process boundary. A slow consumer may
miss images but can never block the ICE/SRTP loop or discard compressed packets.
URLs and credentials travel through multiprocessing's private bootstrap pipe.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Iterator

import numpy as np

from .latency_trace import log_timing
from .video_to_stream import LiveFrame

DEFAULT_FRAME_CAPACITY = 32 * 1024 * 1024


class _StopFlag:
    """Single-writer byte flag; reading it never takes a parent-owned lock."""

    def __init__(self, context):
        self._value = context.RawValue("b", 0)

    def set(self):
        self._value.value = 1

    def is_set(self):
        return bool(self._value.value)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.02, remaining))
        return True


class SharedFrameSlot:
    """One reusable shared allocation; the consumer receives an owned copy.

    RawArray uses multiprocessing's cross-platform shared mmap ownership and
    cleanup. No per-frame pickle, pipe payload, named segment, or feeder thread.
    The producer never waits for the consumer's copy lock.
    """

    def __init__(self, context, capacity: int = DEFAULT_FRAME_CAPACITY):
        if type(capacity) is not int or not 0 < capacity <= 128 * 1024 * 1024:
            raise ValueError("WHEP frame capacity must be within 1..128 MiB")
        self.capacity = capacity
        self.pixels = context.RawArray("B", capacity)
        # publication version, consumed version, frame index, height, width
        self.header = context.RawArray("q", 5)
        self.timestamp = context.RawValue("d", 0)
        self.lock = context.Lock()
        # Event.set() takes a condition lock which a descheduled consumer could
        # hold. A single pending semaphore token needs no consumer-owned mutex.
        self.ready = context.Semaphore(0)
        # Single writer (child); parent only reads these diagnostic counters.
        self.overwritten = context.RawValue("q", 0)
        self.busy = context.RawValue("q", 0)

    def publish(self, frame: LiveFrame) -> bool:
        image = frame.bgr
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) <= 0
            or image.nbytes > self.capacity
        ):
            raise ValueError("decoded frame exceeds the shared BGR capacity")
        if not self.lock.acquire(False):
            self.busy.value += 1
            return False
        try:
            pending = self.header[0] != self.header[1]
            if pending:
                self.overwritten.value += 1
            target = np.frombuffer(self.pixels, dtype=np.uint8, count=image.size)
            np.copyto(target.reshape(image.shape), image)
            self.header[2:5] = frame.index, image.shape[0], image.shape[1]
            self.timestamp.value = frame.timestamp_seconds
            self.header[0] += 1
            if not pending:
                self.ready.release()
            return True
        finally:
            self.lock.release()

    def take(self, timeout: float = 0.1) -> LiveFrame | None:
        if not self.ready.acquire(timeout=timeout):
            return None
        if not self.lock.acquire(timeout=timeout):
            self.ready.release()
            return None
        try:
            version, consumed, index, height, width = self.header
            if version == consumed:
                return None
            image = np.frombuffer(self.pixels, dtype=np.uint8, count=height * width * 3)
            image = image.reshape(height, width, 3).copy()
            stamp = self.timestamp.value
            self.header[1] = version
            return LiveFrame(index=index, timestamp_seconds=stamp, bgr=image)
        finally:
            self.lock.release()


async def _receive(options, slot, stop):
    from .stream_adapter import NetworkStreamAdapter

    adapter = NetworkStreamAdapter(**options)
    adapter._stop = stop

    def publish(frame):
        if stop.is_set():
            return
        adapter._diagnostics.enqueued(frame.index)
        if not slot.publish(frame):
            adapter._diagnostics.dequeued(frame.index, dropped=True)

    adapter._put_latest = publish
    last_log = time.monotonic()
    last_version = 0

    def log_handoff(final=False):
        nonlocal last_log, last_version
        now = time.monotonic()
        version = int(slot.header[0])
        log_timing(
            "network_receive_process_timing",
            pid=os.getpid(),
            connection_id=adapter._diagnostics.connection_id,
            final=final,
            shared_frames_published=version,
            shared_latest_consumed_version=int(slot.header[1]),
            shared_frames_overwritten=slot.overwritten.value,
            shared_copy_busy_frames=slot.busy.value,
            window_shared_fps=(version - last_version) / max(now - last_log, 0.001),
        )
        last_log, last_version = now, version

    task = asyncio.create_task(adapter._receive_whep())
    try:
        while not task.done():
            if stop.is_set():
                task.cancel()
                break
            await asyncio.wait({task}, timeout=0.05)
            if time.monotonic() - last_log >= 5:
                log_handoff()
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        adapter._diagnostics.emit(final=True)
        log_handoff(final=True)


def _worker(options, slot, stop, result):
    # spawn does not inherit Uvicorn logging configuration.
    logger = logging.getLogger("uvicorn.error")
    logger.handlers = [logging.StreamHandler()]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    outcome = ("end", None)
    try:
        asyncio.run(_receive(options, slot, stop))
    except asyncio.CancelledError:
        if not stop.is_set():
            outcome = ("error", "connection_failed")
    except BaseException as exc:
        # Never transport/log the exception message: it may contain credentials.
        code = getattr(exc, "code", "connection_failed")
        outcome = (
            "error",
            "missing_support" if code == "missing_support" else "connection_failed",
        )
        log_timing(
            "network_receive_process_error",
            pid=os.getpid(),
            error_type=type(exc).__name__,
            code=outcome[1],
        )
    finally:
        try:
            result.send(outcome)
        except (OSError, EOFError):
            pass
        result.close()


class WhepProcessReceiver:
    """Lifecycle owner. stop is nonblocking; frames/close reap the child."""

    def __init__(
        self,
        options: dict,
        *,
        capacity=DEFAULT_FRAME_CAPACITY,
        worker=_worker,
        shutdown_timeout=2.0,
    ):
        self._context = multiprocessing.get_context("spawn")
        self._slot = SharedFrameSlot(self._context, capacity)
        self._stop = _StopFlag(self._context)
        self._reader, self._writer = self._context.Pipe(duplex=False)
        self._process = self._context.Process(
            target=worker,
            args=(options, self._slot, self._stop, self._writer),
            name="whep-video-receiver",
            daemon=True,
        )
        self._lifecycle = threading.Lock()
        self._started = self._closed = False
        self._shutdown_timeout = shutdown_timeout
        self._connection_id = options.get("connection_id")

    @property
    def dropped_frames(self):
        return self._slot.overwritten.value + self._slot.busy.value

    @property
    def pid(self):
        return self._process.pid if self._started and not self._closed else None

    def stop(self):
        self._stop.set()

    def frames(self) -> Iterator[LiveFrame]:
        from .stream_adapter import StreamAdapterError

        try:
            with self._lifecycle:
                if self._closed or self._stop.is_set():
                    return
                if self._started:
                    raise RuntimeError("WHEP receiver already consumed")
                self._process.start()
                self._started = True
                self._writer.close()
            log_timing(
                "network_receive_process_started",
                pid=self._process.pid,
                connection_id=self._connection_id,
                shared_frame_capacity_bytes=self._slot.capacity,
            )
            while not self._stop.is_set():
                frame = self._slot.take()
                if self._stop.is_set():
                    return
                if frame is not None:
                    yield frame
                    continue
                try:
                    completed = self._reader.poll()
                except (OSError, EOFError):
                    raise StreamAdapterError("WebRTC receiver exited") from None
                if completed:
                    try:
                        outcome, code = self._reader.recv()
                    except (EOFError, OSError):
                        outcome, code = "error", "connection_failed"
                    if outcome == "error":
                        raise StreamAdapterError("WebRTC receiver failed", code=code)
                    return
                if not self._process.is_alive():
                    raise StreamAdapterError("WebRTC receiver exited")
        finally:
            self.close()

    def close(self):
        self.stop()
        with self._lifecycle:
            if self._closed:
                return
            if self._started:
                self._process.join(self._shutdown_timeout)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(1.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(1.0)
                if self._process.is_alive():
                    raise RuntimeError("WebRTC receiver could not be stopped")
                self._process.close()
            self._reader.close()
            self._writer.close()
            self._closed = True
