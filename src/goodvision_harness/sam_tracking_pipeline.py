"""Causal SAM preparation, one ordered HTTP request, and ordered live commit.

The remote protocol remains single-flight. Durable mode preserves every
admitted source. Latest-only mode retains two recent in-memory candidates and
overlaps one preparation slot with HTTP. Unsent work expires; sent requests and
results retain their original recovery guarantees.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .latency_trace import log_timing, trace_id
from .sam_tracking_spool import TrackingSpool
from .video_tracking import (
    MAX_BATCH_JPEG_BYTES,
    TrackingCommitBackpressure,
    VideoFrame,
    _text,
)
from .video_tracking_session import (
    WATERMARK_FIELDS,
    PreparedSessionAppend,
    VideoSessionTrackingCoordinator,
)


@dataclass(frozen=True, slots=True)
class _Source:
    sequence: int
    timestamp: float
    size: int
    arrived: float


@dataclass(slots=True)
class _Work:
    generation: int
    session: str
    ordinal: int
    request_id: str
    sources: tuple[_Source, ...]
    frames: tuple[VideoFrame, ...] = ()
    prepared: PreparedSessionAppend | None = None
    final: bool = False
    result: object = None
    result_durable: bool = False
    started: float = 0.0
    received: float = 0.0
    timing: dict = field(default_factory=dict)


class PipelinedVideoSessionTrackingCoordinator(VideoSessionTrackingCoordinator):
    """One preparation slot, one append, at most two unapplied results."""

    def __init__(
        self, client, *, spool_directory=None, spool_max_bytes=256 * 1024**2,
        on_native_batch=None, queue_policy="durable_ordered",
        realtime_pending_frames=2, realtime_max_age_seconds=2.0,
        realtime_append_frames=1,
        result_deadline_seconds=4.0, **kwargs
    ):
        if queue_policy not in {"durable_ordered", "realtime_latest", "latest_only"}:
            raise ValueError("unsupported SAM queue policy")
        if (type(realtime_pending_frames) is not int
                or not 1 <= realtime_pending_frames <= 8):
            raise ValueError("realtime_pending_frames must be within 1..8")
        for name, value in (("realtime_max_age_seconds", realtime_max_age_seconds),
                            ("result_deadline_seconds", result_deadline_seconds)):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 < value <= 30):
                raise ValueError(f"{name} must be within (0, 30]")
        if (type(realtime_append_frames) is not int
                or realtime_append_frames not in {1, 2, 4}):
            raise ValueError("realtime_append_frames must be 1, 2, or 4")
        self._realtime_append_frames = min(
            realtime_append_frames, client.max_append_frames,
        )
        self._queue_policy = queue_policy
        self._realtime = queue_policy == "realtime_latest"
        self._latest_only = queue_policy == "latest_only"
        self._candidate_frames = {}
        self._realtime_pending_frames = realtime_pending_frames
        self._realtime_max_age = float(realtime_max_age_seconds)
        self._result_deadline = float(result_deadline_seconds)
        self._latest_result_arrived = None
        self._skip_counts = {"superseded": 0, "expired": 0}
        if self._realtime:
            kwargs.update(pending_capacity=realtime_pending_frames)
        elif self._latest_only:
            kwargs.update(pending_capacity=2)
        super().__init__(client, **kwargs)
        self.on_native_batch = on_native_batch
        if client.startup_frames != 1 or client.max_append_frames > 4:
            raise ValueError("pipelined tracking requires the causal SAM route")
        # Container images keep application code read-only. Recovery journals
        # belong beside persistent deployment settings, never under /app.
        settings_path = os.environ.get("HARNESS_DEPLOYMENT_SETTINGS_PATH", "").strip()
        default_spool = (
            Path(settings_path).parent / "sam-spool"
            if settings_path else Path(".goodvision/sam-spool")
        )
        self._spool_directory = Path(
            spool_directory
            or os.environ.get("GOODVISION_TRACKING_SPOOL_DIRECTORY")
            or os.environ.get("GOODVISION_SAM_SPOOL_DIRECTORY")
            or default_spool
        )
        self._spool_max_bytes = spool_max_bytes
        self._spool = None
        self._prepared = self._preparing = self._requesting = self._committing = None
        self._commits = deque()
        self._prepare_thread = self._commit_thread = None
        self._native_ready = None
        self._next_ordinal = 0
        self._next_sample_at = 0.0
        self._eof_sent = False
        self._retained_failures = []
        # Recent completed stages, isolated per run. These only predict when to
        # prepare; they never change admission, request deadlines, or FIFO order.
        self._prepare_durations = deque(maxlen=16)
        self._append_durations = deque(maxlen=16)
        self._stats.update(
            prepared_batches=0,
            session_candidate_frames=0,
            received_batches=0,
            results_within_deadline=0,
            results_after_deadline=0,
            rejected_capacity=0,
            recovery_required=False,
        )

    def _journal(self):
        if self._spool is None:
            self._spool = TrackingSpool(
                self._spool_directory / f"{uuid4().hex}.sqlite3",
                max_bytes=self._spool_max_bytes,
                keep_open=True,
            )
        return self._spool

    def reset(self, stream_session_id=None):
        with self._condition:
            if self._spool is not None:
                self._spool.finish(self._generation, "reset")
            super().reset(stream_session_id)
            self._prepared = None
            self._candidate_frames.clear()
            self._commits.clear()
            self._native_ready = None
            self._next_ordinal = 0
            self._next_sample_at = 0.0
            self._latest_result_arrived = None
            self._skip_counts = {"superseded": 0, "expired": 0}
            self._eof_sent = False
            self._prepare_durations.clear()
            self._append_durations.clear()
            self._stats.update(
                recovery_required=False,
                session_candidate_frames=0,
                prepared_batches=0,
                received_batches=0,
                results_within_deadline=0,
                results_after_deadline=0,
                rejected_capacity=0,
            )
            self._condition.notify_all()

    def submit(
        self, jpeg_bytes, *, stream_session_id, frame_sequence, timestamp_seconds
    ):
        frame = VideoFrame(frame_sequence, timestamp_seconds, jpeg_bytes)
        _text(stream_session_id, "stream_session_id")
        with self._condition:
            if (
                self._closed
                or self._failed_session
                or self._finished_session
                or self._flush
            ):
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
            self._last_input = frame_sequence, timestamp_seconds
            arrived = time.monotonic()
            if self._capture_started_at is None:
                self._capture_started_at = arrived
            self._stats["session_source_frames"] += 1
            # Match source-clock sampling slots, rather than requiring a minimum
            # gap since the previous accepted frame. Camera timestamps jitter
            # around each 3fps boundary; a 0.32s gap after a 0.35s gap still
            # represents the next source sample and must not be discarded.
            if timestamp_seconds + 1e-6 < self._next_sample_at:
                self._stats["sampled_out"] += 1
                self._stats["session_sampled_out"] += 1
                return False
            try:
                if self._latest_only:
                    # Candidates are not durable admissions. Replacing one does
                    # no image preparation or database work on the capture path.
                    self._skip_pending(
                        [s for s in self._pending
                         if arrived - s.arrived >= self._realtime_max_age],
                        "expired",
                    )
                    if len(self._pending) >= self._pending_capacity:
                        self._skip_pending(
                            list(self._pending)[:len(self._pending)
                                                - self._pending_capacity + 1],
                            "superseded",
                        )
                    self._candidate_frames[frame_sequence] = frame
                else:
                    journal = self._journal()
                    journal.begin(self._generation, self._session)
                    if self._realtime:
                        self._expire_unsent(arrived)
                        if len(self._pending) >= self._realtime_pending_frames:
                            excess = (
                                len(self._pending) - self._realtime_pending_frames + 1
                            )
                            self._skip_pending(
                                list(self._pending)[:excess],
                                "superseded",
                            )
                    journal.admit(self._generation, self._session, frame, arrived)
            except Exception as exc:
                self._stats["rejected_capacity"] += 1
                self._fail(self._generation, "admission", exc)
                raise
            self._last_sample = timestamp_seconds
            elapsed_slots = math.floor(
                (timestamp_seconds + 1e-6 - self._next_sample_at) * self.max_fps
            ) + 1
            self._next_sample_at += elapsed_slots / self.max_fps
            self._source_arrivals[frame_sequence] = arrived
            self._pending.append(
                _Source(frame_sequence, timestamp_seconds, len(jpeg_bytes), arrived)
            )
            self._pending_bytes += len(jpeg_bytes)
            self._stats["submitted"] += 1
            if not self._latest_only:
                self._stats["session_admitted_frames"] += 1
            else:
                self._stats["session_candidate_frames"] += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="sam-ordered-request", daemon=True
                )
                self._prepare_thread = threading.Thread(
                    target=self._prepare_loop,
                    name="sam-prepare",
                    daemon=True,
                )
                self._commit_thread = threading.Thread(
                    target=self._commit_loop,
                    name="sam-ordered-commit",
                    daemon=True,
                )
                for worker in (self._thread, self._prepare_thread, self._commit_thread):
                    worker.start()
            self._condition.notify_all()
        return True

    submit_frame = submit

    def flush(self):
        with self._condition:
            if self._closed or self._failed_session or self._finished_session:
                return
            if self._capture_finished_at is None:
                self._capture_finished_at = time.monotonic()
            if self._thread is None:
                self._finished_session = True
            else:
                self._flush = True
            self._condition.notify_all()

    def _current(self, work):
        return not self._closed and work.generation == self._generation

    def _new_work(self, sources=(), *, final=False):
        self._next_ordinal += 1
        work = _Work(
            self._generation,
            self._session,
            self._next_ordinal,
            uuid4().hex,
            tuple(sources),
            final=final,
        )
        if not self._latest_only or final:
            self._journal().batch(
                work.request_id,
                work.generation,
                work.ordinal,
                [source.sequence for source in work.sources],
            )
        return work

    def _fail(self, generation, phase, exc, work=None):
        with self._condition:
            if generation == self._generation and not self._closed:
                self._failed_session = True
                self._cleanup_requested = self._server_generation is not None
                self._stats["failed"] += 1
                self._stats.update(
                    last_error=type(exc).__name__, recovery_required=True
                )
                self._native_ready = None
            try:
                if (
                    work is not None
                    and work.result is not None
                    and not work.result_durable
                ):
                    self._retained_failures.append(work)
                if work is not None:
                    self._journal().mark(
                        work.request_id, f"{phase}_failed", type(exc).__name__
                    )
                if self._spool is not None:
                    self._spool.finish(generation, "failed", type(exc).__name__)
            except Exception:
                # A failed journal cannot erase already persisted sources. The
                # explicit failure remains visible even if its final marker fails.
                self._stats["last_error"] = "TrackingJournalFailure"
                if work is not None:
                    self._retained_failures.append(work)
            self._condition.notify_all()
        log_timing(
            "sam_pipeline_failure",
            generation=generation,
            phase=phase,
            request_id=trace_id(work.request_id) if work else None,
            error_type=type(exc).__name__,
            recovery_required=True,
        )

    def _record_skip(self, sources, reason):
        for source in sources:
            self._source_arrivals.pop(source.sequence, None)
        self._skip_counts[reason] += len(sources)
        self._stats["dropped_backpressure"] += len(sources)
        self._stats["session_dropped_backpressure"] += len(sources)
        log_timing(
            "sam_realtime_skip", generation=self._generation,
            stream_session_id=trace_id(self._session), reason=reason,
            input_sequences=[source.sequence for source in sources],
            queue_policy=self._queue_policy,
        )
        self._condition.notify_all()

    def _skip_pending(self, sources, reason):
        if not sources:
            return
        if self._latest_only:
            for source in sources:
                self._candidate_frames.pop(source.sequence, None)
        else:
            self._journal().skip(
                self._generation, [source.sequence for source in sources], reason,
            )
        sequences = {source.sequence for source in sources}
        self._pending = deque(s for s in self._pending if s.sequence not in sequences)
        self._pending_bytes -= sum(source.size for source in sources)
        self._record_skip(sources, reason)

    def _expire_unsent(self, now):
        """Called under the condition; never touches preparing or sent work."""
        if not (self._realtime or self._latest_only) or self._failed_session:
            return
        self._skip_pending(
            [s for s in self._pending if now - s.arrived >= self._realtime_max_age],
            "expired",
        )
        work = self._prepared
        if (work is not None and work.generation == self._generation
                and now - work.sources[0].arrived >= self._realtime_max_age):
            self._journal().skip(
                work.generation, [s.sequence for s in work.sources], "expired",
                request_id=work.request_id,
            )
            self._prepared = None
            self._record_skip(work.sources, "expired")

    def _expiry_wait(self, now):
        sources = list(self._pending)
        if self._prepared is not None:
            sources.extend(self._prepared.sources)
        if not sources or self._failed_session:
            return None
        return max(
            0.001, min(s.arrived for s in sources) + self._realtime_max_age - now,
        )

    def _preparation_schedule(self, now):
        """Under the condition: (seconds until preparation, safe timing facts).

        None waits for a state change. A positive wait wakes without another
        source arriving, so an admitted prefix cannot miss its JIT deadline.
        """
        if (
            self._failed_session
            or self._finished_session
            or self._native_ready != self._generation
            or self._prepared is not None
            or not self._pending
        ):
            return None, {}
        if self._latest_only:
            # Keep a single preparation slot busy while the ordered append runs.
            # A slow request cannot grow a prepared FIFO: later candidates stay
            # replaceable, and the prepared source is rechecked before dispatch.
            if len(self._commits) + bool(self._committing) >= 2:
                return None, {}
            return 0.0, {"prepare_trigger": (
                "latest_request_overlap" if self._requesting is not None
                else "latest_request_idle"
            )}
        if self._realtime:
            # Do not freeze an old image while HTTP or commit is blocked.
            if (self._requesting is not None
                    or len(self._commits) + bool(self._committing) >= 2):
                return None, {}
            return 0.0, {"prepare_trigger": "realtime_request_idle"}
        if self._requesting is None:
            return 0.0, {"prepare_trigger": "request_idle"}
        if len(self._pending) >= self._client.max_append_frames:
            return 0.0, {"prepare_trigger": "full_batch"}
        # During warmup retain the full-batch-or-idle policy. At least three
        # actual stages are needed; a single cold request is not a forecast.
        if len(self._append_durations) < 3 or len(self._prepare_durations) < 3:
            return None, {}
        count, size = 0, 0
        for source in self._pending:
            if count and size + source.size > MAX_BATCH_JPEG_BYTES:
                break
            count += 1
            size += source.size
        # Wait until the measured preparation lead before freezing any prefix.
        # At the live ~1 Hz cadence there may only be one or two pending frames;
        # requiring three serializes preparation with HTTP indefinitely.
        if self._requesting.started <= 0:
            return None, {}
        requests = sorted(self._append_durations)
        middle = len(requests) // 2
        expected = (requests[middle] + requests[(len(requests) - 1) // 2]) / 2
        preparations = sorted(self._prepare_durations)
        lead = min(0.9, preparations[math.ceil(0.95 * len(preparations)) - 1] + 0.05)
        remaining = self._requesting.started + expected - now
        return max(0.0, remaining - lead), {
            "prepare_trigger": "predicted_request_completion",
            "predicted_append_seconds": expected,
            "predicted_remaining_seconds": max(0.0, remaining),
            "prepare_lead_seconds": lead,
            "append_timing_samples": len(requests),
            "prepare_timing_samples": len(preparations),
            "predicted_prefix_frames": count,
        }

    def _prepare_loop(self):
        while True:
            with self._condition:
                while True:
                    if self._closed:
                        return
                    now = time.monotonic()
                    try:
                        self._expire_unsent(now)
                    except Exception as exc:
                        self._fail(self._generation, "skip", exc)
                    wait, preparation_timing = self._preparation_schedule(now)
                    if (self._realtime or self._latest_only) and wait is None:
                        wait = self._expiry_wait(now)
                    if wait == 0.0:
                        break
                    self._condition.wait(wait)
                sources, size = [], 0
                limit = (1 if self._latest_only else
                         self._realtime_append_frames if self._realtime
                         else self._client.max_append_frames)
                while self._pending and len(sources) < limit:
                    candidate = self._pending[0]
                    if sources and size + candidate.size > MAX_BATCH_JPEG_BYTES:
                        break
                    sources.append(self._pending.popleft())
                    size += candidate.size
                self._pending_bytes -= size
                try:
                    work = self._new_work(sources)
                except Exception as exc:
                    self._fail(self._generation, "prepare", exc)
                    continue
                self._preparing = work
                if self._latest_only:
                    work.frames = tuple(
                        self._candidate_frames.pop(source.sequence)
                        for source in work.sources
                    )
            started = time.monotonic()
            try:
                if self._latest_only:
                    # Once selected this is executing work, not another waiting
                    # slot. Persist it on this worker before preparing/sending.
                    journal = self._journal()
                    journal.begin(work.generation, work.session)
                    journal.admit(
                        work.generation, work.session, work.frames[0],
                        work.sources[0].arrived,
                    )
                    journal.batch(
                        work.request_id, work.generation, work.ordinal,
                        [source.sequence for source in work.sources],
                    )
                    with self._condition:
                        if not self._current(work) or self._failed_session:
                            journal.mark(work.request_id, "interrupted")
                            continue
                        self._stats["session_admitted_frames"] += 1
                else:
                    work.frames = self._journal().load(
                        work.generation,
                        [source.sequence for source in work.sources],
                    )
                work.prepared = self._client.prepare_append(
                    work.frames,
                    stream_session_id=work.session,
                    request_id=work.request_id,
                )
                self._journal().prepared(
                    work.request_id,
                    work.prepared.payload.body,
                    work.prepared.payload_hash,
                )
                with self._condition:
                    if self._current(work) and not self._failed_session:
                        self._prepare_durations.append(time.monotonic() - started)
                        self._preparing = None
                        self._prepared = work
                        self._stats["prepared_batches"] += 1
                    else:
                        self._journal().mark(work.request_id, "interrupted")
            except Exception as exc:
                self._fail(work.generation, "prepare", exc, work)
            finally:
                log_timing(
                    "sam_pipeline_prepare",
                    generation=work.generation,
                    request_id=trace_id(work.request_id),
                    input_sequences=[s.sequence for s in work.sources],
                    prepare_seconds=time.monotonic() - started,
                    **preparation_timing,
                )
                with self._condition:
                    self._preparing = None
                    self._condition.notify_all()

    def _run(self):
        try:
            while True:
                with self._condition:
                    while True:
                        if self._closed:
                            return
                        if self._cleanup_requested:
                            operation = "cleanup"
                            self._cleanup_requested = False
                            break
                        if not self._failed_session and not self._finished_session:
                            try:
                                self._expire_unsent(time.monotonic())
                            except Exception as exc:
                                self._fail(self._generation, "skip", exc)
                                continue
                            # Reused workers also need to finish a run that
                            # never received a frame, regardless of queue mode.
                            # No native session exists to receive an EOF append.
                            if (self._flush and not self._pending
                                    and self._preparing is None
                                    and self._prepared is None
                                    and self._stats["received_batches"] == 0):
                                operation = "finish_empty"
                                generation = self._generation
                                break
                            if self._native_ready != self._generation and self._pending:
                                operation = "open"
                                generation, session = self._generation, self._session
                                break
                            capacity = len(self._commits) + bool(self._committing) < 2
                            if capacity and self._prepared is not None:
                                operation, work = "append", self._prepared
                                self._prepared = None
                                self._requesting = work
                                break
                            if (
                                capacity
                                and self._flush
                                and not self._eof_sent
                                and self._native_ready == self._generation
                                and not self._pending
                                and self._preparing is None
                                and self._prepared is None
                            ):
                                operation = "append"
                                try:
                                    work = self._new_work(final=True)
                                except Exception as exc:
                                    self._fail(self._generation, "eof", exc)
                                    continue
                                self._requesting = work
                                self._eof_sent = True
                                break
                        self._condition.wait()
                    self._in_flight = True
                    self._condition.notify_all()
                if operation == "finish_empty":
                    self._discard()
                    with self._condition:
                        try:
                            if generation == self._generation and not self._closed:
                                self._journal().finish(generation, "completed")
                                self._finished_session = True
                                self._flush = False
                        except Exception as exc:
                            self._fail(generation, "eof", exc)
                        self._native_ready = None
                        self._in_flight = False
                        self._condition.notify_all()
                    continue
                if operation == "cleanup":
                    self._discard()
                    with self._condition:
                        self._native_ready = None
                        self._in_flight = False
                        self._condition.notify_all()
                    continue
                if operation == "open":
                    try:
                        self._discard()
                        self._client.open_session(session)
                        with self._condition:
                            self._server_generation = generation
                            if generation == self._generation and not self._closed:
                                self._native_ready = generation
                            else:
                                self._cleanup_requested = True
                    except Exception as exc:
                        self._fail(generation, "session_setup", exc)
                    finally:
                        with self._condition:
                            self._in_flight = False
                            self._condition.notify_all()
                    continue
                work.started = time.monotonic()
                work.timing = {
                    "stream_session_id": trace_id(work.session),
                    "generation": work.generation,
                    "request_id": work.request_id,
                    "input_sequences": [s.sequence for s in work.sources],
                    "input_frames": len(work.sources),
                    "final": work.final,
                    "queue_oldest_seconds": (
                        work.started - work.sources[0].arrived if work.sources else 0.0
                    ),
                    "pipeline_overlap": True,
                }
                with self._condition:
                    self._stats["started"] += 1
                    self._stats["last_request_frames"] = len(work.sources)
                    work.timing.update(
                        {
                            key: self._stats[key]
                            for key in (
                                "session_source_frames",
                                "session_admitted_frames",
                                "session_sampled_out",
                                "session_dropped_backpressure",
                            )
                        }
                    )
                    # _requesting is assigned before its start clock; wake the
                    # preparation timer again once that clock is available.
                    self._condition.notify_all()
                log_timing("sam_coordinator_dispatch", **work.timing, outcome="started")
                try:
                    stage_started = time.monotonic()
                    try:
                        self._journal().mark(work.request_id, "requesting")
                    finally:
                        work.timing["journal_request_mark_seconds"] = (
                            time.monotonic() - stage_started
                        )
                    work.result = self._client.append(
                        work.frames,
                        stream_session_id=work.session,
                        request_id=work.request_id,
                        is_last_batch=work.final,
                        prepared=work.prepared,
                    )
                    work.received = time.monotonic()
                    work.timing["append_seconds"] = work.received - work.started
                    stage_started = time.monotonic()
                    try:
                        self._journal().result(work.request_id, work.result)
                    finally:
                        work.timing["journal_result_seconds"] = (
                            time.monotonic() - stage_started
                        )
                    work.result_durable = True
                    stage_started = time.monotonic()
                    with self._condition:
                        publish_native = (
                            self._current(work) and not self._failed_session
                        )
                    native_outcome = "skipped"
                    if publish_native and self.on_native_batch is not None:
                        try:
                            self.on_native_batch(work.result, work.generation)
                            native_outcome = "completed"
                        except Exception as exc:
                            native_outcome = "error"
                            # Presentation failure cannot discard durable analysis work.
                            log_timing("sam_native_display_failure",
                                       generation=work.generation,
                                       error_type=type(exc).__name__)
                    work.timing["native_publish_seconds"] = (
                        time.monotonic() - stage_started
                    )
                    # The callback only updates the in-memory native metadata.
                    # This elapsed time includes its eligibility lock, not WS
                    # delivery or the later L2/L3 commit. Emit before that commit.
                    log_timing(
                        "sam_native_result_timing",
                        stream_session_id=trace_id(work.session),
                        generation=work.generation,
                        request_id=trace_id(work.request_id),
                        output_frames=len(work.result.frames),
                        native_publish_outcome=native_outcome,
                        **{key: work.timing[key] for key in (
                            "journal_request_mark_seconds", "journal_result_seconds",
                            "native_publish_seconds",
                        )},
                    )
                    with self._condition:
                        self._stats["received_batches"] += 1
                        if self._current(work) and not self._failed_session:
                            if work.sources and not work.final:
                                self._append_durations.append(
                                    work.received - work.started
                                )
                            self._commits.append(work)
                        else:
                            self._journal().mark(work.request_id, "interrupted")
                            self._stats["stale_results"] += 1
                except Exception as exc:
                    self._fail(work.generation, "request", exc, work)
                finally:
                    with self._condition:
                        self._requesting = None
                        self._in_flight = False
                        self._condition.notify_all()
        finally:
            self._discard()
            self._client.close()

    def _commit_loop(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or (self._commits and not self._failed_session)
                )
                if self._closed:
                    return
                work = self._commits.popleft()
                self._committing = work
            started = time.monotonic()
            outcome = "stale"
            try:
                with self._condition:
                    current = self._current(work)
                if current:
                    result = work.result
                    applied = not result.frames or self._on_batch is None
                    if result.frames and self._on_batch is not None:
                        while True:
                            try:
                                self._on_batch(result)
                                applied = True
                                break
                            except TrackingCommitBackpressure:
                                with self._condition:
                                    if not self._current(work) or self._failed_session:
                                        break
                                    self._condition.wait(0.1)
                                    if not self._current(work) or self._failed_session:
                                        break
                    completed = time.monotonic()
                    with self._condition:
                        if self._current(work) and applied and not self._failed_session:
                            self._journal().mark(work.request_id, "committed")
                            self._history.extend(result.frames)
                            work.timing["source_to_result"] = []
                            for snapshot in result.frames:
                                arrival = self._source_arrivals.pop(
                                    snapshot.frame_sequence,
                                    None,
                                )
                                if arrival is not None:
                                    age = completed - arrival
                                    self._stats["source_to_result_seconds"] = age
                                    self._latest_result_arrived = arrival
                                    self._stats[
                                        "results_within_deadline"
                                        if age <= self._result_deadline
                                        else "results_after_deadline"
                                    ] += 1
                                    work.timing["source_to_result"].append(
                                        {
                                            "sequence": snapshot.frame_sequence,
                                            "seconds": age,
                                        }
                                    )
                            work.timing["completion_interval_seconds"] = (
                                completed - self._last_completion_at
                                if self._last_completion_at is not None
                                else None
                            )
                            self._last_completion_at = completed
                            during = self._capture_finished_at is None
                            self._stats[
                                "completed_during_capture"
                                if during
                                else "completed_after_capture"
                            ] += len(result.frames)
                            self._stats["processed_frames"] += len(result.frames)
                            self._stats["succeeded"] += 1
                            self._stats.update(
                                last_error=None,
                                last_latency_seconds=completed - work.started,
                                last_frame_sequence=result.emitted_frame_sequence,
                                completed_timestamp_seconds=result.emitted_timestamp_seconds,
                                **{
                                    key: getattr(result, key)
                                    for key in WATERMARK_FIELDS
                                    | {"pending_frame_count"}
                                },
                            )
                            work.timing.update(
                                during_capture=during,
                                pending_frames_at_completion=len(self._pending),
                            )
                            outcome = "ok"
                            if work.final:
                                self._journal().finish(work.generation, "completed")
                                self._finished_session = True
                                self._flush = False
                if outcome == "stale":
                    self._journal().mark(work.request_id, "interrupted")
            except Exception as exc:
                outcome = "error"
                self._fail(work.generation, "commit", exc, work)
                work.timing["error_type"] = type(exc).__name__
            finally:
                work.timing.update(
                    outcome=outcome,
                    commit_seconds=time.monotonic() - started,
                    commit_queue_seconds=started - work.received,
                    total_seconds=time.monotonic() - work.started,
                    output_sequences=[s.frame_sequence for s in work.result.frames],
                    output_frames=len(work.result.frames),
                )
                log_timing("sam_coordinator_timing", **work.timing)
                with self._condition:
                    self._committing = None
                    self._condition.notify_all()

    def stats(self):
        with self._condition:
            result = super().stats()
            staged = [
                work
                for work in (self._preparing, self._prepared)
                if work is not None and work.generation == self._generation
            ]
            staged_frames = sum(len(work.sources) for work in staged)
            result.update(
                pipeline_overlap=True,
                queue_policy=self._queue_policy,
                effective_max_append_frames=(
                    1 if self._latest_only else
                    self._realtime_append_frames if self._realtime
                    else self._client.max_append_frames
                ),
                realtime_max_age_seconds=(
                    self._realtime_max_age
                    if self._realtime or self._latest_only else None
                ),
                skipped_by_reason=dict(self._skip_counts),
                oldest_pending_seconds=max(
                    (time.monotonic() - source.arrived for source in self._pending),
                    default=0.0,
                ),
                latest_result_age_seconds=(
                    time.monotonic() - self._latest_result_arrived
                    if self._latest_result_arrived is not None else None
                ),
                result_deadline_seconds=self._result_deadline,
                admitted_deadline_coverage=(
                    self._stats["results_within_deadline"]
                    / self._stats["session_admitted_frames"]
                    if self._stats["session_admitted_frames"] else None
                ),
                pending=bool(self._pending) or bool(staged_frames),
                pending_frames=len(self._pending) + staged_frames,
                pending_bytes=self._pending_bytes
                + sum(source.size for work in staged for source in work.sources),
                candidate_frames=len(self._pending) if self._latest_only else 0,
                disk_pending_frames=0 if self._latest_only else len(self._pending),
                journal_capacity_bytes=self._spool_max_bytes,
                preparing_frames=len(self._preparing.sources) if self._preparing else 0,
                prepared_frames=len(self._prepared.sources) if self._prepared else 0,
                pending_commits=len(self._commits) + bool(self._committing),
            )
            return result

    def retained_sources(self, batch):
        """Exact private source JPEGs for an uncommitted current-run response.

        A live consumer may re-pin these into its shorter evidence cache before
        retrying a commit. Source times/indices must remain unchanged.
        """
        with self._condition:
            if (
                self._spool is None
                or self._closed
                or batch.stream_session_id != self._session
            ):
                return ()
            journal = self._spool
        return journal.sources_for_request(batch.request_id, batch.stream_session_id)

    def wait_until_idle(self, timeout_seconds=2.0):
        end = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                active = (
                    self._in_flight
                    or self._preparing
                    or self._committing
                    or self._cleanup_requested
                )
                waiting = (
                    self._pending
                    or self._prepared
                    or self._commits
                    or (self._flush and not self._finished_session)
                )
                if not active and (not waiting or self._failed_session or self._closed):
                    return True
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)

    def close(self, timeout_seconds=2.0):
        try:
            with self._condition:
                self._closed = True
                if self._capture_finished_at is None:
                    self._capture_finished_at = time.monotonic()
                try:
                    if self._spool is not None:
                        self._spool.finish(self._generation, "closed")
                finally:
                    self._condition.notify_all()
            deadline = time.monotonic() + timeout_seconds
            if self._thread is None:
                self._client.close()
            for worker in (self._thread, self._prepare_thread, self._commit_thread):
                if worker is not None and worker is not threading.current_thread():
                    worker.join(max(0.0, deadline - time.monotonic()))
        finally:
            if self._spool is not None:
                # A timeout can leave an append/preparation completing later. The
                # store accepts those durable facts with short connections.
                self._spool.close()

    def close_complete(self):
        workers = (self._thread, self._prepare_thread, self._commit_thread)
        complete = super().close_complete() and all(
            worker is None or not worker.is_alive() for worker in workers
        )
        if complete and self._spool is not None:
            self._spool.close()
        return complete
