"""Continuous source playback, independently of sparse SAM admissions/results."""

from __future__ import annotations

import time
from dataclasses import replace

from .scene_delivery import TrackedSceneDisplayWorker, _people, render_scene_jpeg


class ContinuousTrackedSceneDisplayWorker(TrackedSceneDisplayWorker):
    """Delay complete capture JPEGs, then render whatever annotation is available.

    This presentation buffer is independent of durable SAM evidence. Exact masks
    remain exact-source only. The existing renderer uses dashed boxes for short
    held observations and omits expired geometry; display never invents evidence.
    """

    def __init__(self, *args, max_buffer_bytes=64 * 1024 * 1024, **kwargs):
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("display buffer size must be positive")
        super().__init__(*args, **kwargs)
        self._packets = {}
        self._buffer_bytes = 0
        self._max_buffer_bytes = max_buffer_bytes
        self._displayed_packet = None
        self._last_timing_log = 0.0

    def submit(self, packet, *, retain_source=True):
        # Admission to inference does not determine admission to presentation.
        with self._condition:
            submitted = self._stats["submitted_frames"]
            full = len(self._sources) == self._sources.maxlen
            super().submit(packet, retain_source=not full)
            if self._stats["submitted_frames"] == submitted:
                return
            if full:
                self._stats["discarded_frames"] += 1
                return
            self._packets[packet.frame_sequence] = packet
            self._buffer_bytes += len(packet.jpeg)
            retained = {source.sequence for source in self._sources}
            for sequence in tuple(self._packets):
                if sequence not in retained:
                    self._buffer_bytes -= len(self._packets.pop(sequence).jpeg)
            # Preserve delayed older frames when byte capacity is exhausted;
            # dropping every oldest frame could prevent any reaching its delay.
            while self._buffer_bytes > self._max_buffer_bytes and self._sources:
                removed = self._sources.pop()
                self._buffer_bytes -= len(self._packets.pop(removed.sequence).jpeg)
                self._stats["discarded_frames"] += 1
            self._condition.notify_all()

    def _run(self):
        next_render = 0.0
        while not self._stop.is_set():
            with self._condition:
                now = time.monotonic()
                ready = [
                    s
                    for s in self._sources
                    if s.received_at + self._delay_seconds <= now
                ]
                refresh = (
                    self._displayed_source is not None
                    and self._event_revision > self._displayed_event_revision
                )
                if (not ready and not refresh) or now < next_render:
                    self._condition.wait(0.02 if self._sources or refresh else 0.25)
                    continue
                if ready:
                    stamp = ready[-1]
                    packet = self._packets[stamp.sequence]
                    removed = 0
                    while self._sources and self._sources[0].sequence <= stamp.sequence:
                        old = self._sources.popleft()
                        self._buffer_bytes -= len(self._packets.pop(old.sequence).jpeg)
                        removed += 1
                    self._stats["replaced_frames"] += max(0, removed - 1)
                else:
                    stamp, packet = self._displayed_source, self._displayed_packet
                revision = self._event_revision
                self._stats["active_frames"] = 1
            started = time.monotonic()
            next_render = started + self._interval
            try:
                snapshot = self.scene.display_snapshot(stamp.timestamp)
                if snapshot.get("state") == "stopped":
                    continue
                snapshot_seconds = time.monotonic() - started
                rendering = time.monotonic()
                jpeg = render_scene_jpeg(
                    packet.jpeg, snapshot, stamp.timestamp, quality=self._quality
                )
                render_seconds = time.monotonic() - rendering
                # Delay changes during rendering must also apply to this frame.
                while not self._stop.is_set():
                    with self._condition:
                        remaining = (
                            stamp.received_at + self._delay_seconds - time.monotonic()
                        )
                    if remaining <= 0:
                        break
                    self._stop.wait(min(0.05, remaining))
                if self._stop.is_set():
                    return
                people = _people(snapshot, stamp.timestamp)
                mode = (
                    "exact"
                    if people and all(age == 0 for _, _, age in people)
                    else "held"
                    if people
                    else "none"
                )
                output = replace(
                    packet,
                    jpeg=jpeg,
                    annotation_mode=mode,
                    stream_delay_seconds=time.monotonic() - stamp.received_at,
                )
                publishing = time.monotonic()
                try:
                    self._publish(output)
                except Exception:
                    with self._condition:
                        self._stats["publish_failures"] += 1
                    continue
                now = time.monotonic()
                with self._condition:
                    repaint = stamp.sequence == self._last_published_sequence
                    self._stats[
                        "same_source_repaints" if repaint else "distinct_source_frames"
                    ] += 1
                    self._stats["published_frames"] += 1
                    self._last_published_sequence = stamp.sequence
                    self._displayed_source, self._displayed_packet = stamp, packet
                    self._displayed_event_revision = revision
                    self._display_state = "running"
                    self._stats.update(
                        displayed_source_timestamp_seconds=stamp.timestamp,
                        stream_delay_seconds=now - stamp.received_at,
                        last_snapshot_seconds=snapshot_seconds,
                        last_jpeg_render_seconds=render_seconds,
                        last_publish_seconds=now - publishing,
                        last_source_to_publish_seconds=now - stamp.received_at,
                    )
                    elapsed = self._display_elapsed_seconds(now)
                    timing = dict(
                        event="sam_display_timing",
                        playback_mode="continuous_source",
                        source_frame_sequence=stamp.sequence,
                        source_timestamp_seconds=stamp.timestamp,
                        source_to_publish_seconds=now - stamp.received_at,
                        snapshot_seconds=snapshot_seconds,
                        render_seconds=render_seconds,
                        publish_seconds=now - publishing,
                        late_event_repaint=repaint,
                        distinct_source_frames=self._stats["distinct_source_frames"],
                        same_source_repaints=self._stats["same_source_repaints"],
                        distinct_source_fps=self._stats["distinct_source_frames"]
                        / max(elapsed, 0.001),
                    )
                if now - self._last_timing_log >= 5:
                    self._last_timing_log = now
                    self._log_display_timing(timing)
            except Exception:
                with self._condition:
                    self._stats["render_failures"] += 1
            finally:
                with self._condition:
                    self._stats["active_frames"] = 0
                    self._stats["last_render_seconds"] = time.monotonic() - started
                    self._condition.notify_all()

    def stats(self):
        with self._condition:
            return super().stats() | {
                "mode": "continuous_source",
                "pending_bytes": self._buffer_bytes,
                "max_pending_bytes": self._max_buffer_bytes,
                "retained_display_bytes": len(self._displayed_packet.jpeg)
                if self._displayed_packet is not None
                else 0,
            }

    def close(self, timeout_seconds=1.0):
        # Validate before mutating state, consistently with the base class.
        if not isinstance(
            timeout_seconds, (int, float)
        ) or not 0 <= timeout_seconds < float("inf"):
            raise ValueError("close timeout must be finite and non-negative")
        with self._condition:
            self._stop.set()
            self._packets.clear()
            self._buffer_bytes = 0
            self._displayed_packet = None
        super().close(timeout_seconds)
