"""Bounded native-video batches, separate from the single-image label protocol.

Every batch contains only already received frames. SAM memory is scoped to that
batch; a new predictor epoch is mandatory for the next batch. This adapter does
not claim unbounded camera memory or cross-batch re-identification.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from typing import Any
from uuid import uuid4

import httpx

from .latency_trace import log_timing, trace_id
from .rich_labeling import (
    RichLabelProtocolError,
    RichLabelSelection,
    RichLabelSnapshot,
    select_rich_labels,
)

BACKEND = "sam3.1_video"
MAX_BATCH_FRAMES = 16
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_BATCH_JPEG_BYTES = 20 * 1024 * 1024
# A release can include pending source frames plus the current append, each
# with bounded SAM contour polygons for every visible person.
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class TrackingCommitBackpressure(RuntimeError):
    """No live state changed; an ordered retained result may be retried later."""


@dataclass(frozen=True, slots=True)
class SerializedVideoPayload:
    """An immutable request encoded before the single HTTP operation starts."""

    body: bytes
    metadata: Mapping[str, Any]
    preparation_seconds: float
    transport_metrics: Mapping[str, Any]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_sequence: int
    timestamp_seconds: float
    jpeg_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.frame_sequence) is not int or self.frame_sequence < 0:
            raise ValueError("invalid frame_sequence")
        if (
            type(self.timestamp_seconds) not in (int, float)
            or not math.isfinite(self.timestamp_seconds)
            or self.timestamp_seconds < 0
        ):
            raise ValueError("invalid timestamp_seconds")
        if not isinstance(self.jpeg_bytes, bytes) or not self.jpeg_bytes:
            raise ValueError("jpeg_bytes must be nonempty bytes")
        if len(self.jpeg_bytes) > MAX_JPEG_BYTES:
            raise ValueError("JPEG exceeds byte limit")


def _validate_frames(frames: Sequence[VideoFrame]) -> None:
    if not 1 <= len(frames) <= MAX_BATCH_FRAMES:
        raise ValueError("video batches require 1-16 frames")
    if sum(len(frame.jpeg_bytes) for frame in frames) > MAX_BATCH_JPEG_BYTES:
        raise ValueError("video batch exceeds byte limit")
    for previous, current in pairwise(frames):
        if (
            current.frame_sequence <= previous.frame_sequence
            or current.timestamp_seconds <= previous.timestamp_seconds
        ):
            raise ValueError("video batch source frames must increase strictly")


@dataclass(frozen=True, slots=True)
class VideoTrackingBatch:
    stream_session_id: str
    batch_id: str
    predictor_epoch: str
    frames: tuple[RichLabelSnapshot, ...]
    completed_frame_sequence: int
    completed_timestamp_seconds: float

    @classmethod
    def parse(
        cls,
        payload: Mapping[str, Any],
        *,
        stream_session_id: str,
        batch_id: str,
        inputs: Sequence[VideoFrame],
    ) -> VideoTrackingBatch:
        """Validate the whole response before any frame changes application state."""
        _validate_frames(inputs)
        try:
            if set(payload) != {
                "stream_session_id",
                "batch_id",
                "predictor_epoch",
                "backend",
                "memory_scope",
                "frames",
                "completed_frame_sequence",
                "completed_timestamp_seconds",
            }:
                raise ValueError("unexpected video response fields")
            if (
                payload["stream_session_id"] != stream_session_id
                or payload["batch_id"] != batch_id
                or payload["backend"] != BACKEND
                or payload["memory_scope"] != "batch"
            ):
                raise ValueError("video response scope mismatch")
            epoch = _text(payload["predictor_epoch"], "predictor_epoch")
            records = payload["frames"]
            if not isinstance(records, list) or len(records) != len(inputs):
                raise ValueError("video response must cover every requested frame")
            snapshots = tuple(RichLabelSnapshot.from_dict(value) for value in records)
            for frame, snapshot in zip(inputs, snapshots, strict=True):
                if (
                    snapshot.stream_session_id != stream_session_id
                    or snapshot.frame_sequence != frame.frame_sequence
                    or not math.isclose(
                        snapshot.timestamp_seconds,
                        frame.timestamp_seconds,
                        rel_tol=0,
                        abs_tol=1e-6,
                    )
                    or len(snapshot.objects) > 64
                ):
                    raise ValueError("video response source mismatch")
                native_ids: set[str] = set()
                for item in snapshot.objects:
                    native = item.attributes.get("native_track")
                    if (
                        not isinstance(native, Mapping)
                        or set(native) != {"backend", "epoch", "object_id"}
                        or native["backend"] != BACKEND
                        or native["epoch"] != epoch
                        or item.label != "person"
                        or item.observation_state != "observed"
                        or item.last_observed_frame_sequence != frame.frame_sequence
                        or item.last_observed_timestamp_seconds is None
                        or not math.isclose(
                            item.last_observed_timestamp_seconds,
                            frame.timestamp_seconds,
                            rel_tol=0,
                            abs_tol=1e-6,
                        )
                    ):
                        raise ValueError("invalid native observation provenance")
                    native_id = _text(native["object_id"], "native object ID")
                    if native_id in native_ids:
                        raise ValueError("duplicate native object ID")
                    native_ids.add(native_id)
            last = snapshots[-1]
            if (
                type(payload["completed_frame_sequence"]) is not int
                or payload["completed_frame_sequence"] != last.frame_sequence
                or type(payload["completed_timestamp_seconds"]) not in (int, float)
                or not math.isclose(
                    payload["completed_timestamp_seconds"],
                    last.timestamp_seconds,
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("video completion watermark mismatch")
            return cls(
                stream_session_id,
                batch_id,
                epoch,
                snapshots,
                last.frame_sequence,
                last.timestamp_seconds,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RichLabelProtocolError(str(exc)) from exc


class VideoTrackingHTTPClient:
    """Synchronous facade over one bounded, cancellable asynchronous I/O loop.

    A client admits one operation at a time. Its deadline covers upload, response
    headers, bounded raw body reading and protocol parsing. Cancellation keeps
    the slot occupied until the actual task finishes cleanup; retries cannot
    create another thread or accumulate requests behind a stuck operation.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 75.0,
        transport: httpx.AsyncBaseTransport | None = None,
        transport_max_side: int | None = None,
        transport_jpeg_quality: int | None = None,
    ) -> None:
        url = httpx.URL(endpoint)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("video endpoint must be HTTP(S)")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name, value, minimum, maximum in (
            ("transport_max_side", transport_max_side, 128, 1280),
            ("transport_jpeg_quality", transport_jpeg_quality, 1, 100),
        ):
            if value is not None and (
                type(value) is not int or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        self.endpoint = str(url.copy_with(query=None, fragment=None))
        self._health = str(
            url.copy_with(path="/health/video", query=None, fragment=None)
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise ValueError("video transport must support asynchronous I/O")
        self._timeout_seconds = float(timeout_seconds)
        self._transport = transport
        self._request_headers: dict[str, str] = {}
        self._trust_env = True
        self._transport_max_side = transport_max_side
        self._transport_jpeg_quality = transport_jpeg_quality
        self._transport_codec = "jpeg"
        self._transport_stats_lock = threading.Lock()
        self._transport_metrics = {
            "requests": 0,
            "frames": 0,
            "source_jpeg_bytes": 0,
            "transport_jpeg_bytes": 0,
            "transport_bytes": 0,
            "encode_seconds": 0.0,
            "compression_seconds": 0.0,
            "last_frames": 0,
            "last_source_jpeg_bytes": 0,
            "last_transport_jpeg_bytes": 0,
            "last_transport_bytes": 0,
            "last_encode_seconds": 0.0,
            "last_compression_seconds": 0.0,
        }
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._active: concurrent.futures.Future | None = None
        self._task: asyncio.Task | None = None
        self._deadline_task: asyncio.Task | None = None
        self._cancel_requested = False
        self._deadline_expired = False
        self._shutdown_task: asyncio.Task | None = None

    @staticmethod
    def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name, value in pairs:
            if name in fields:
                raise RichLabelProtocolError("duplicate video response field")
            fields[name] = value
        return fields

    @staticmethod
    def _invalid_constant(_value: str) -> None:
        raise RichLabelProtocolError("nonfinite video response value")

    def transport_stats(self) -> dict[str, Any]:
        """Prepared payload totals; transport bytes exclude JSON/base64 overhead."""
        with self._transport_stats_lock:
            return {
                **self._transport_metrics,
                "max_side": self._transport_max_side,
                "jpeg_quality": self._transport_jpeg_quality,
                "codec": self._transport_codec,
            }

    def _record_transport(
        self, frames, wire_bytes, elapsed, *, codec="jpeg", encode_seconds=0.0
    ):
        original_bytes = sum(len(frame.jpeg_bytes) for frame in frames)
        jpeg_bytes = wire_bytes if codec == "jpeg" else 0
        with self._transport_stats_lock:
            metrics = self._transport_metrics
            metrics["requests"] += 1
            metrics["frames"] += len(frames)
            metrics["source_jpeg_bytes"] += original_bytes
            metrics["transport_jpeg_bytes"] += jpeg_bytes
            metrics["transport_bytes"] += wire_bytes
            metrics["compression_seconds"] += elapsed
            metrics["encode_seconds"] += encode_seconds
            metrics.update(
                last_frames=len(frames),
                last_source_jpeg_bytes=original_bytes,
                last_transport_jpeg_bytes=jpeg_bytes,
                last_transport_bytes=wire_bytes,
                last_compression_seconds=elapsed,
                last_encode_seconds=encode_seconds,
            )

    def _transport_frames(self, frames: tuple[VideoFrame, ...], *, deadline: float):
        """Compress once inside the admitted request; never alter source evidence.

        Header checks bound decoded pixels before allocating image buffers. A
        resize preserves the entire view, so normalized source geometry remains
        valid. No transform is performed unless explicitly configured.
        """
        started = time.monotonic()
        transformed = (
            self._transport_max_side is not None
            or self._transport_jpeg_quality is not None
        )
        outputs = []
        pixels = 0
        wire_bytes = 0
        for frame in frames:
            if time.monotonic() >= deadline:
                raise httpx.TimeoutException(
                    "video compression exceeded total deadline"
                )
            output = frame
            if transformed:
                from PIL import Image

                with Image.open(BytesIO(frame.jpeg_bytes)) as source:
                    width, height = source.size
                    pixels += width * height
                    if (
                        source.format != "JPEG"
                        or width <= 0
                        or height <= 0
                        or width * height > 3840 * 2160
                        or pixels > 64 * 1024 * 1024
                    ):
                        raise ValueError("video transport image exceeds decoded limits")
                    scale = (
                        min(1.0, self._transport_max_side / max(width, height))
                        if self._transport_max_side
                        else 1.0
                    )
                    if scale < 1 or self._transport_jpeg_quality is not None:
                        image = source.convert("RGB")
                        if scale < 1:
                            resized = image.resize(
                                (
                                    max(1, round(width * scale)),
                                    max(1, round(height * scale)),
                                ),
                                Image.Resampling.LANCZOS,
                            )
                            image.close()
                            image = resized
                        try:
                            buffer = BytesIO()
                            image.save(
                                buffer,
                                format="JPEG",
                                quality=self._transport_jpeg_quality or 90,
                            )
                            output = VideoFrame(
                                frame.frame_sequence,
                                frame.timestamp_seconds,
                                buffer.getvalue(),
                            )
                        finally:
                            image.close()
            wire_bytes += len(output.jpeg_bytes)
            if wire_bytes > MAX_BATCH_JPEG_BYTES:
                raise ValueError("video transport batch exceeds byte limit")
            outputs.append(output)
        if time.monotonic() >= deadline:
            raise httpx.TimeoutException("video compression exceeded total deadline")
        elapsed = time.monotonic() - started if transformed else 0.0
        self._record_transport(frames, wire_bytes, elapsed)
        return tuple(outputs)

    async def _exchange(
        self,
        method: str,
        url: str,
        payload_factory: Callable[[], dict[str, Any]] | None,
        result_parser: Callable[[dict[str, Any]], Any] | None,
        *,
        deadline: float,
    ) -> Any:
        started = time.monotonic()
        timing: dict[str, Any] = {
            "phase": "client_setup", "outcome": "error",
            "route": getattr(getattr(self, "method", None), "id", "legacy"),
        }
        try:
            result = await self._exchange_timed(
                method, url, payload_factory, result_parser,
                deadline=deadline, timing=timing,
            )
            if time.monotonic() >= deadline:
                # The existing _finish_operation/_request checks will reject
                # this late result. Classify it without changing that path.
                timing.update(outcome="timeout", deadline_exceeded=True)
            else:
                timing.update(outcome="ok", phase="complete")
            return result
        except BaseException as exc:
            timing["error_type"] = type(exc).__name__
            if isinstance(exc, httpx.RemoteProtocolError):
                detail = str(exc).lower()
                timing["protocol_error_reason"] = (
                    "eof_before_headers" if "without sending a response" in detail
                    else "incomplete_body"
                    if "incomplete" in detail and "body" in detail
                    else "malformed_headers"
                    if "illegal header" in detail or "illegal status line" in detail
                    else "other"
                )
                timing["cause_type"] = (
                    type(exc.__cause__).__name__ if exc.__cause__ else None
                )
            if isinstance(exc, httpx.TimeoutException) or (
                isinstance(exc, asyncio.CancelledError)
                and time.monotonic() >= deadline
            ):
                timing["outcome"] = "timeout"
            raise
        finally:
            # A cancellation may interrupt the transport without its .failed
            # callback. Retain elapsed work while distinguishing incomplete IO.
            for metric, phase_started in timing.pop("_transport_starts", {}).values():
                timing[metric] = timing.get(metric, 0.0) + max(
                    0.0, time.monotonic() - phase_started
                )
            phase_started = timing.pop("_phase_started_at", None)
            phase = timing["phase"]
            if phase_started is not None and phase in {
                "http_until_headers", "response_read",
            }:
                timing.setdefault(
                    f"{phase}_seconds", time.monotonic() - phase_started,
                )
            if method == "POST" and url.endswith("/append"):
                timing["total_seconds"] = time.monotonic() - started
                log_timing("sam_client_http_timing", **timing)

    async def _exchange_timed(
        self,
        method: str,
        url: str,
        payload_factory: Callable[[], dict[str, Any]] | None,
        result_parser: Callable[[dict[str, Any]], Any] | None,
        *,
        deadline: float,
        timing: dict[str, Any],
    ) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._timeout_seconds, connect=min(self._timeout_seconds, 5)
                ),
                transport=self._transport,
                follow_redirects=False,
                trust_env=self._trust_env,
                headers={"Accept-Encoding": "identity", **self._request_headers},
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            )
        timing["phase"] = "payload_prepare"
        prepare_started = time.monotonic()
        transport_requests = self.transport_stats()["requests"]
        prepared = None
        try:
            value = payload_factory() if payload_factory is not None else None
            if isinstance(value, SerializedVideoPayload):
                prepared = value
                kwargs = {
                    "content": value.body,
                    "headers": {"Content-Type": "application/json"},
                }
            else:
                kwargs = {"json": value} if payload_factory is not None else {}
        finally:
            timing["payload_prepare_seconds"] = time.monotonic() - prepare_started
        transport = self.transport_stats()
        if prepared is not None:
            # Preparation ran on another stage, outside this request's elapsed
            # time. Do not add it to the HTTP duration or read another batch's
            # concurrently updated transport counters.
            timing["prepared_payload_seconds"] = prepared.preparation_seconds
            timing.update(prepared.transport_metrics)
        elif transport["requests"] > transport_requests:
            # Detail within payload_prepare_seconds, not an extra additive stage.
            timing.update(
                compression_seconds=transport["last_compression_seconds"],
                video_encode_seconds=transport["last_encode_seconds"],
                transport_bytes=transport["last_transport_bytes"],
                transport_codec=transport["codec"],
            )
        payload = (
            dict(prepared.metadata) if prepared is not None else kwargs.get("json", {})
        )
        if isinstance(payload, dict):
            timing["request_id"] = trace_id(payload.get("request_id"))
            timing["stream_session_id"] = trace_id(payload.get("stream_session_id"))
            rows = payload.get("frames", ())
            if isinstance(rows, (list, tuple)):
                timing["input_frames"] = len(rows)
                if rows:
                    for name, row in (("first", rows[0]), ("last", rows[-1])):
                        if not isinstance(row, dict):
                            continue
                        sequence = row.get("frame_sequence")
                        timestamp = row.get("timestamp_seconds")
                        if type(sequence) is int:
                            timing[f"{name}_source_sequence"] = sequence
                        if type(timestamp) in (int, float) and math.isfinite(timestamp):
                            timing[f"{name}_source_timestamp_seconds"] = timestamp
            timing["final"] = payload.get("is_last_batch") is True
        if time.monotonic() >= deadline:
            raise httpx.TimeoutException("video preparation exceeded total deadline")
        # Deliver a queued cancellation before opening the network stream after
        # bounded CPU preparation, which cannot itself yield to the event loop.
        await asyncio.sleep(0)
        # httpcore trace reports client IO boundaries, not remote receipt or
        # pure wire time. In particular, send_request_body can complete before
        # the server has read its body. Never retain trace info (URLs, headers,
        # credentials and body objects can all occur there).
        transport_starts = timing["_transport_starts"] = {}
        timing["transport_trace_observed"] = False
        transport_metrics = {
            "connect_tcp": "tcp_connect_seconds",
            "start_tls": "tls_handshake_seconds",
            "send_request_headers": "send_request_headers_seconds",
            "send_request_body": "send_request_body_seconds",
            "receive_response_headers": "receive_response_headers_seconds",
        }

        async def transport_trace(name, _info):
            parts = name.split(".")
            if (
                len(parts) != 3
                or parts[0] not in {"connection", "proxy", "http11", "http2"}
                or parts[2] not in {"started", "complete", "failed"}
            ):
                return
            metric = transport_metrics.get(parts[1])
            if metric is None:
                return
            timing["transport_trace_observed"] = True
            identity, state = ".".join(parts[:2]), parts[2]
            now = time.monotonic()
            if state == "started":
                transport_starts[identity] = (metric, now)
                if parts[1] == "send_request_body":
                    timing["request_body_completed"] = False
            else:
                started = transport_starts.pop(identity, None)
                if started is not None:
                    timing[metric] = timing.get(metric, 0.0) + max(
                        0.0, now - started[1]
                    )
                if parts[1] == "connect_tcp" and state == "complete":
                    timing["tcp_connections_opened"] = timing.get(
                        "tcp_connections_opened", 0
                    ) + 1
                if parts[1] == "send_request_body" and state == "complete":
                    timing["request_body_completed"] = True

        kwargs["extensions"] = {"trace": transport_trace}
        timing["phase"] = "http_until_headers"
        http_started = time.monotonic()
        timing["_phase_started_at"] = http_started
        async with self._client.stream(method, url, **kwargs) as response:
            # Includes httpx serialization, upload, worker execution, and response
            # headers; this is deliberately NOT labelled pure network time.
            timing["http_until_headers_seconds"] = time.monotonic() - http_started
            timing["http_status"] = response.status_code
            size = response.request.headers.get("content-length", "")
            if size.isascii() and size.isdigit() and len(size) <= 12:
                timing["request_bytes"] = int(size)
            response.raise_for_status()
            timing["phase"] = "response_read"
            read_started = time.monotonic()
            timing["_phase_started_at"] = read_started
            # Reject compressed bodies before any decoder can expand them.
            if response.headers.get(
                "content-encoding", "identity"
            ).strip().lower() not in {
                "",
                "identity",
            }:
                raise RichLabelProtocolError("compressed video response is unsupported")
            declared = response.headers.get("content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError:
                    raise RichLabelProtocolError(
                        "invalid video response length"
                    ) from None
                if not 0 <= length <= MAX_RESPONSE_BYTES:
                    raise RichLabelProtocolError("video response exceeds byte limit")
            data = bytearray()
            if response.is_stream_consumed:
                # Standard MockTransport responses may already hold their body.
                # Real network streaming uses aiter_raw below, never decompression.
                if len(response.content) > MAX_RESPONSE_BYTES:
                    raise RichLabelProtocolError("video response exceeds byte limit")
                data.extend(response.content)
            else:
                async for chunk in response.aiter_raw(chunk_size=16 * 1024):
                    if len(chunk) > MAX_RESPONSE_BYTES - len(data):
                        raise RichLabelProtocolError(
                            "video response exceeds byte limit"
                        )
                    data.extend(chunk)
            timing["response_read_seconds"] = time.monotonic() - read_started
            timing["response_bytes"] = len(data)
            timing["phase"] = "response_json"
            json_started = time.monotonic()
            try:
                value = json.loads(
                    data,
                    object_pairs_hook=self._unique_fields,
                    parse_constant=self._invalid_constant,
                )
            except (ValueError, RecursionError) as exc:
                raise RichLabelProtocolError("invalid video response JSON") from exc
            finally:
                timing["response_json_seconds"] = time.monotonic() - json_started
        if not isinstance(value, dict):
            raise RichLabelProtocolError("video response must be an object")
        timing["phase"] = "result_validation"
        validation_started = time.monotonic()
        try:
            return result_parser(value) if result_parser is not None else value
        finally:
            timing["result_validation_seconds"] = time.monotonic() - validation_started

    async def _watch_deadline(
        self, operation: concurrent.futures.Future, task: asyncio.Task, deadline: float
    ) -> None:
        try:
            await asyncio.wait_for(
                asyncio.shield(task), max(0, deadline - time.monotonic())
            )
        except asyncio.TimeoutError:
            self._cancel_on_loop(operation)
        except (Exception, asyncio.CancelledError):
            # The underlying task's done callback owns result/error delivery.
            pass

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def _start_operation(
        self,
        operation: concurrent.futures.Future,
        deadline: float,
        args: tuple[Any, ...],
    ) -> None:
        assert self._loop is not None
        with self._lock:
            if self._closed or operation.cancelled():
                if self._active is operation:
                    self._active = None
                close = self._closed
            else:
                self._task = self._loop.create_task(
                    self._exchange(*args, deadline=deadline)
                )
                self._task.add_done_callback(
                    lambda task: self._finish_operation(operation, task, deadline)
                )
                self._deadline_task = self._loop.create_task(
                    self._watch_deadline(operation, self._task, deadline)
                )
                return
        if close:
            self._shutdown_on_loop()

    def _finish_operation(
        self, operation: concurrent.futures.Future, task: asyncio.Task, deadline: float
    ) -> None:
        with self._lock:
            try:
                value = task.result()
            except asyncio.CancelledError:
                if not operation.done():
                    if self._deadline_expired or time.monotonic() >= deadline:
                        operation.set_exception(
                            httpx.TimeoutException(
                                "video request exceeded total deadline"
                            )
                        )
                    else:
                        operation.cancel()
            except Exception as exc:
                if not operation.done():
                    operation.set_exception(exc)
            else:
                if not operation.done():
                    if time.monotonic() >= deadline:
                        operation.set_exception(
                            httpx.TimeoutException(
                                "video request exceeded total deadline"
                            )
                        )
                    else:
                        operation.set_result(value)
            finally:
                if self._active is operation:
                    self._active = self._task = None
                close = self._closed
        if close:
            self._shutdown_on_loop()

    def _cancel_on_loop(
        self, operation: concurrent.futures.Future, timed_out: bool = True
    ) -> None:
        with self._lock:
            if self._active is operation and timed_out:
                self._deadline_expired = True
            if (
                self._active is operation
                and self._task is not None
                and not self._cancel_requested
            ):
                self._cancel_requested = True
                self._task.cancel()

    def _request(
        self,
        method: str,
        url: str,
        *,
        deadline: float,
        payload_factory: Callable[[], dict[str, Any]] | None = None,
        result_parser: Callable[[dict[str, Any]], Any] | None = None,
    ) -> Any:
        with self._lock:
            if self._closed:
                raise httpx.RequestError("video client is closed")
            if self._active is not None:
                raise httpx.PoolTimeout("video client already has an active request")
            if self._thread is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._run_loop, name="native-video-http", daemon=True
                )
                self._thread.start()
            operation: concurrent.futures.Future = concurrent.futures.Future()
            self._active = operation
            self._cancel_requested = False
            self._deadline_expired = False
            assert self._loop is not None
            self._loop.call_soon_threadsafe(
                self._start_operation,
                operation,
                deadline,
                (method, url, payload_factory, result_parser),
            )
        try:
            result = operation.result(timeout=max(0, deadline - time.monotonic()))
            if time.monotonic() >= deadline:
                raise httpx.TimeoutException("video request exceeded total deadline")
            return result
        except concurrent.futures.TimeoutError:
            with self._lock:
                operation.cancel()
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._cancel_on_loop, operation)
            raise httpx.TimeoutException(
                "video request exceeded total deadline"
            ) from None
        except concurrent.futures.CancelledError:
            if time.monotonic() >= deadline:
                raise httpx.TimeoutException(
                    "video request exceeded total deadline"
                ) from None
            raise httpx.RequestError("video request was cancelled") from None

    def gpu_ready(self) -> bool:
        try:
            value = self._request(
                "GET",
                self._health,
                deadline=time.monotonic() + min(self._timeout_seconds, 5.0),
            )
            return (
                value.get("ready") is True
                and value.get("gpu_ready") is True
                and value.get("backend") == BACKEND
                and value.get("native_sam_video_tracking") is True
                and value.get("memory_scope") == "batch"
            )
        except (httpx.HTTPError, ValueError, RichLabelProtocolError):
            return False

    def infer_batch(
        self,
        frames: Sequence[VideoFrame],
        *,
        stream_session_id: str,
        batch_id: str | None = None,
    ) -> VideoTrackingBatch:
        deadline = time.monotonic() + self._timeout_seconds
        _text(stream_session_id, "stream_session_id")
        batch_id = _text(batch_id or uuid4().hex, "batch_id")
        _validate_frames(frames)
        frames = tuple(frames)
        _validate_frames(frames)
        return self._request(
            "POST",
            self.endpoint,
            deadline=deadline,
            payload_factory=lambda: {
                "stream_session_id": stream_session_id,
                "batch_id": batch_id,
                "frames": [
                    {
                        "frame_sequence": frame.frame_sequence,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "jpeg_base64": base64.b64encode(frame.jpeg_bytes).decode(
                            "ascii"
                        ),
                    }
                    for frame in self._transport_frames(frames, deadline=deadline)
                ],
            },
            result_parser=lambda payload: VideoTrackingBatch.parse(
                payload,
                stream_session_id=stream_session_id,
                batch_id=batch_id,
                inputs=frames,
            ),
        )

    async def _shutdown(self) -> None:
        try:
            if self._deadline_task is not None:
                await asyncio.gather(self._deadline_task, return_exceptions=True)
            if self._client is not None:
                await asyncio.wait_for(self._client.aclose(), 1.0)
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            assert self._loop is not None
            self._loop.stop()

    def _shutdown_on_loop(self) -> None:
        assert self._loop is not None
        with self._lock:
            operation = self._active
            if operation is None and self._shutdown_task is None:
                self._shutdown_task = self._loop.create_task(self._shutdown())
        if operation is not None:
            self._cancel_on_loop(operation, timed_out=False)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._active is not None:
                self._active.cancel()
            thread, loop = self._thread, self._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(self._shutdown_on_loop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)


class VideoTrackingCoordinator:
    """One in-flight batch plus one bounded, sampled waiting window.

    Backpressure drops oldest waiting frames and reports the count. It never
    merges those frames with a previous model memory. Callbacks run outside the
    scheduling lock; consumers must also verify their current source session.
    """

    def __init__(
        self,
        client: VideoTrackingHTTPClient,
        *,
        max_fps: float = 4,
        batch_frames: int = 16,
        batch_seconds: float = 4,
        history_size: int = 128,
        on_batch: Callable[[VideoTrackingBatch], None] | None = None,
    ) -> None:
        if type(batch_frames) is not int or not 1 <= batch_frames <= MAX_BATCH_FRAMES:
            raise ValueError("batch_frames must be within 1-16")
        if not math.isfinite(max_fps) or not 0 < max_fps <= 30:
            raise ValueError("video max_fps must be within (0,30]")
        if not math.isfinite(batch_seconds) or not 0 < batch_seconds <= 10:
            raise ValueError("batch_seconds must be within (0,10]")
        if type(history_size) is not int or not 2 <= history_size <= 4096:
            raise ValueError("history_size must be within 2-4096")
        self.max_fps, self.history_size = max_fps, history_size
        self._client, self._on_batch = client, on_batch
        self._batch_frames, self._batch_seconds = batch_frames, batch_seconds
        self._condition = threading.Condition()
        self._pending: deque[tuple[VideoFrame, float]] = deque()
        self._pending_bytes = 0
        self._history: deque[RichLabelSnapshot] = deque(maxlen=history_size)
        self._epochs: deque[str] = deque(maxlen=4096)
        self._session: str | None = None
        self._generation = 0
        self._last_input = (-1, -1.0)
        self._last_sample = -math.inf
        self._closed = self._in_flight = self._flush = False
        self._thread: threading.Thread | None = None
        self._stats: dict[str, Any] = {
            "submitted": 0,
            "sampled_out": 0,
            "dropped_backpressure": 0,
            "rejected_order": 0,
            "rejected_session": 0,
            "stale_results": 0,
            "started": 0,
            "succeeded": 0,
            "failed": 0,
            "resets": 0,
            "last_error": None,
            "last_latency_seconds": None,
            "last_frame_sequence": None,
            "completed_timestamp_seconds": None,
            "last_queue_wait_seconds": None,
            "last_request_frames": 0,
            "total_queue_wait_seconds": 0.0,
        }

    def reset(self, stream_session_id: str | None = None) -> None:
        if stream_session_id is not None:
            _text(stream_session_id, "stream_session_id")
        with self._condition:
            self._generation += 1
            self._session = stream_session_id
            self._pending.clear()
            self._pending_bytes = 0
            self._history.clear()
            self._epochs.clear()
            self._last_input = (-1, -1.0)
            self._last_sample = -math.inf
            self._flush = False
            self._stats.update(
                resets=self._stats["resets"] + 1,
                last_error=None,
                last_frame_sequence=None,
                completed_timestamp_seconds=None,
            )
            self._condition.notify_all()

    def submit(
        self,
        jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> bool:
        frame = VideoFrame(frame_sequence, timestamp_seconds, jpeg_bytes)
        _text(stream_session_id, "stream_session_id")
        with self._condition:
            if self._closed:
                return False
            if self._session is None:
                self._session = stream_session_id
            if stream_session_id != self._session:
                self._stats["rejected_session"] += 1
                return False
            if (
                frame_sequence <= self._last_input[0]
                or timestamp_seconds <= self._last_input[1]
            ):
                self._stats["rejected_order"] += 1
                return False
            self._last_input = (frame_sequence, timestamp_seconds)
            if timestamp_seconds - self._last_sample < 1 / self.max_fps - 1e-8:
                self._stats["sampled_out"] += 1
                return False
            self._last_sample = timestamp_seconds
            while self._pending and (
                len(self._pending) >= self._batch_frames
                or self._pending_bytes + len(jpeg_bytes) > MAX_BATCH_JPEG_BYTES
                or timestamp_seconds - self._pending[0][0].timestamp_seconds
                > self._batch_seconds
            ):
                dropped, _ = self._pending.popleft()
                self._pending_bytes -= len(dropped.jpeg_bytes)
                self._stats["dropped_backpressure"] += 1
            self._pending.append((frame, time.monotonic()))
            self._pending_bytes += len(jpeg_bytes)
            self._stats["submitted"] += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="native-video-tracking",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()
            return True

    submit_frame = submit

    def flush(self) -> None:
        """Dispatch a partial final window without waiting for another frame."""
        with self._condition:
            self._flush = True
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._closed:
                        age = (
                            time.monotonic() - self._pending[0][1]
                            if self._pending
                            else 0
                        )
                        if self._pending and (
                            self._flush
                            or len(self._pending) >= self._batch_frames
                            or age >= self._batch_seconds
                        ):
                            break
                        self._condition.wait(
                            max(0.001, self._batch_seconds - age)
                            if self._pending
                            else None,
                        )
                    if self._closed:
                        return
                    self._stats["last_queue_wait_seconds"] = age
                    self._stats["total_queue_wait_seconds"] += age
                    frames = tuple(frame for frame, _ in self._pending)
                    self._stats["last_request_frames"] = len(frames)
                    self._pending.clear()
                    self._pending_bytes = 0
                    generation, session = self._generation, self._session
                    self._in_flight, self._flush = True, False
                    self._stats["started"] += 1
                started = time.monotonic()
                try:
                    result = self._client.infer_batch(frames, stream_session_id=session)
                    with self._condition:
                        stale = self._closed or generation != self._generation
                        if stale:
                            self._stats["stale_results"] += 1
                        elif result.predictor_epoch in self._epochs:
                            raise RichLabelProtocolError("native batch reused an epoch")
                        else:
                            self._epochs.append(result.predictor_epoch)
                    if not stale and self._on_batch is not None:
                        self._on_batch(result)
                    with self._condition:
                        if not self._closed and generation == self._generation:
                            self._history.extend(result.frames)
                            self._stats.update(
                                succeeded=self._stats["succeeded"] + 1,
                                last_error=None,
                                last_frame_sequence=result.completed_frame_sequence,
                                completed_timestamp_seconds=result.completed_timestamp_seconds,
                            )
                except Exception as exc:
                    with self._condition:
                        if generation == self._generation:
                            # Do not put remote exception text or credentials in status.
                            self._stats["failed"] += 1
                            self._stats["last_error"] = type(exc).__name__
                finally:
                    with self._condition:
                        if generation == self._generation:
                            self._stats["last_latency_seconds"] = (
                                time.monotonic() - started
                            )
                        self._in_flight = False
                        self._condition.notify_all()
        finally:
            self._client.close()

    def latest(self, stream_session_id: str | None = None) -> RichLabelSnapshot | None:
        with self._condition:
            if stream_session_id not in (None, self._session):
                return None
            return self._history[-1] if self._history else None

    def history(self) -> tuple[RichLabelSnapshot, ...]:
        with self._condition:
            return tuple(self._history)

    def select(
        self,
        stream_session_id: str,
        timestamp_seconds: float,
        *,
        frame_sequence: int | None = None,
    ) -> RichLabelSelection:
        with self._condition:
            history = tuple(self._history) if stream_session_id == self._session else ()
        return select_rich_labels(
            history,
            timestamp_seconds,
            max_gap_seconds=0.5,
            max_hold_seconds=0.5,
            frame_sequence=frame_sequence,
        )

    def stats(self) -> dict[str, Any]:
        with self._condition:
            completed = self._stats["completed_timestamp_seconds"]
            return {
                **self._stats,
                "backend": BACKEND,
                "memory_scope": "batch",
                "state": "closed" if self._closed else "running",
                "stream_session_id": self._session,
                "max_fps": self.max_fps,
                "pending": bool(self._pending),
                "pending_frames": len(self._pending),
                "pending_bytes": self._pending_bytes,
                "in_flight": self._in_flight,
                "history_size": len(self._history),
                "history_capacity": self.history_size,
                "transport": self._client.transport_stats()
                if hasattr(self._client, "transport_stats")
                else {},
                "tracking_lag_seconds": (
                    None
                    if completed is None
                    else max(0, self._last_input[1] - completed)
                ),
            }

    def wait_until_idle(self, timeout_seconds: float = 2.0) -> bool:
        end = time.monotonic() + timeout_seconds
        with self._condition:
            while self._pending or self._in_flight:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout_seconds: float = 2.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending.clear()
            self._pending_bytes = 0
            thread = self._thread
            self._condition.notify_all()
        if thread is None:
            self._client.close()
        elif thread is not threading.current_thread():
            thread.join(timeout_seconds)

    def close_complete(self) -> bool:
        """A replacement route must wait for the last old batch to finish."""
        with self._condition:
            return self._closed and (
                self._thread is None or not self._thread.is_alive()
            )
