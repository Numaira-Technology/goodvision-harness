"""Source-time scene overlays composed outside the live video callback."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import Condition, Event, Thread, current_thread
from typing import Any

from PIL import ImageDraw, ImageFont

from .result_bus import FramePacket
from .rich_labeling import validate_mask_contours
from .scene_evidence import EvidenceFrame
from .scene_inspection import _read_image
from .scene_runtime import SceneRuntime

_MAX_JPEG_BYTES = 16 * 1024 * 1024
_MAX_PEOPLE = 128
_MAX_EVENTS = 2048  # Same hard ceiling as SceneState's event registry.
_MAX_LINES = 8
_HOLD_SECONDS = 1.0
_DISPLAY_LOGGER = logging.getLogger("uvicorn.error")
_SAFE_SESSION_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
_GREEN = (73, 220, 138)
_YELLOW = (255, 214, 64)
_RED = (239, 68, 68)
_GRAY = (190, 198, 211)
_PHASE_COLORS = {
    "suspected": _YELLOW,
    "confirmed": _RED,
    "ended": _GRAY,
    "retracted": _GRAY,
}


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _safe_text(value: Any, *, fallback: str = "", limit: int = 48) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        return fallback
    if not all(character.isprintable() for character in value):
        return fallback
    return value[:limit]


def _scene(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    value = snapshot.get("scene", snapshot)
    return value if isinstance(value, Mapping) else {}


def _people(
    snapshot: Mapping[str, Any], timestamp: float
) -> list[tuple[Mapping[str, Any], tuple[float, ...], float]]:
    entities = _scene(snapshot).get("entities", ())
    if not isinstance(entities, (list, tuple)):
        return []
    people = []
    for entity in entities[:_MAX_PEOPLE]:
        if not isinstance(entity, Mapping):
            continue
        seen = entity.get("last_observed_timestamp_seconds")
        if not _finite(seen) or not 0 <= timestamp - seen <= _HOLD_SECONDS:
            continue
        # Prediction or an externally supplied occlusion location cannot be
        # promoted into a source observation by the display layer.
        if entity.get("observation_state") != "observed":
            continue
        box = entity.get("bbox")
        if not isinstance(box, Mapping):
            continue
        values = tuple(box.get(key) for key in ("x1", "y1", "x2", "y2"))
        if (
            not all(_finite(value) and 0 <= value <= 1 for value in values)
            or values[0] >= values[2]
            or values[1] >= values[3]
        ):
            continue
        people.append((entity, values, timestamp - seen))
    return people


def _dashed_box(draw, bounds: tuple[int, ...], color: tuple[int, ...], width: int):
    x1, y1, x2, y2 = bounds
    step = max(8, width * 5)
    for x in range(x1, x2, step):
        right = min(x + step // 2, x2)
        draw.line((x, y1, right, y1), fill=color, width=width)
        draw.line((x, y2, right, y2), fill=color, width=width)
    for y in range(y1, y2, step):
        bottom = min(y + step // 2, y2)
        draw.line((x1, y, x1, bottom), fill=color, width=width)
        draw.line((x2, y, x2, bottom), fill=color, width=width)


def _display_events(snapshot: Mapping[str, Any], timestamp: float) -> list[Mapping]:
    """Use the current event clock only when explicitly supplied by the runtime."""
    clock = _scene(snapshot).get("event_state_timestamp_seconds", timestamp)
    if not _finite(clock) or clock < 0:
        clock = timestamp
    events = _scene(snapshot).get("events", ())
    if not isinstance(events, (tuple, list)):
        return []
    result = []
    for event in events[:_MAX_EVENTS]:
        if not isinstance(event, Mapping):
            continue
        phase = event.get("phase")
        at = event.get("last_evidence_timestamp_seconds")
        if not isinstance(phase, str) or phase not in _PHASE_COLORS:
            continue
        if not _finite(at) or at < 0 or at > clock:
            continue
        subjects = event.get("subject_ids")
        if (
            event.get("event_label")
            not in ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter", "stranger")
            or not isinstance(subjects, (list, tuple))
            or (not subjects and event.get("event_label") != "ground_litter")
            or len(subjects) > _MAX_PEOPLE
            or not all(isinstance(subject, str) and subject for subject in subjects)
        ):
            continue
        result.append(event)
    return result


def _event_lines(
    snapshot: Mapping[str, Any],
    timestamp: float,
    visible_ids: set[str],
) -> list[tuple[str, tuple[int, ...]]]:
    candidates = []
    for event in _display_events(snapshot, timestamp):
        phase = event["phase"]
        at = event["last_evidence_timestamp_seconds"]
        candidates.append(
            (
                0 if phase in ("confirmed", "suspected") else 1,
                -at,
                event,
            )
        )
    # A creation-ordered history must not hide new alarms behind ended episodes.
    # Stable sorting preserves input order only when urgency and source time tie.
    candidates.sort(key=lambda item: item[:2])
    lines = []
    for _, _, event in candidates:
        phase = event["phase"]
        at = event["last_evidence_timestamp_seconds"]
        label = event.get("event_label")
        if label not in ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter", "stranger"):
            continue
        subjects = event.get("subject_ids")
        if not isinstance(subjects, (list, tuple)) or not all(
            isinstance(subject, str) for subject in subjects[:8]
        ):
            continue
        ids = [
            _safe_text(subject, fallback="person", limit=24) for subject in subjects[:8]
        ]
        uncertain = event.get("observation_state") == "uncertain" or (
            phase in ("suspected", "confirmed")
            and (
                timestamp - at > 3
                or any(subject not in visible_ids for subject in subjects[:8])
            )
        )
        uncertainty = " | uncertain" if uncertain else ""
        status = f"{label}: {phase}{uncertainty} | {', '.join(ids)}"
        lines.append((status[:160], _PHASE_COLORS[phase]))
        if len(lines) == _MAX_LINES:
            break
    return lines


def _source_contours(entity, timestamp, width, height):
    """Only draw masks derived from this exact source, never held geometry."""
    value = entity.get("mask_contours")
    if value is None:
        return ()
    try:
        validate_mask_contours(value)
    except (TypeError, ValueError):
        return ()
    if (
        value["source_timestamp_seconds"] != timestamp
        or value["source_frame_sequence"] != entity.get("frame_sequence")
        or entity.get("last_observed_timestamp_seconds") != timestamp
    ):
        return ()
    return tuple(
        tuple((round(x * (width - 1)), round(y * (height - 1))) for x, y in polygon)
        for polygon in value["polygons"]
    )


def render_scene_jpeg(
    jpeg: bytes,
    snapshot: Mapping[str, Any],
    media_timestamp: float,
    quality: int = 85,
    breakdown: dict[str, float] | None = None,
) -> bytes:
    """Render presentation state only; held geometry is never event evidence.

    SAM contours come from the exact source mask; other observations use boxes. A
    one-second visual hold is dashed and age-labelled; older/future geometry is
    omitted. Event text can persist independently and explicitly shows uncertain
    visibility. Metadata and credentials are not copied into the JPEG.

    When `breakdown` is supplied it is populated with `decode_seconds`,
    `draw_seconds` and `encode_seconds` so the caller can attribute this stage's
    host CPU across JPEG decode, PIL draw and JPEG re-encode (display is stage 7).
    """
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= _MAX_JPEG_BYTES:
        raise ValueError("scene JPEG must be nonempty and at most 16 MiB")
    if not isinstance(snapshot, Mapping):
        raise ValueError("scene snapshot must be a mapping")
    if not _finite(media_timestamp) or media_timestamp < 0:
        raise ValueError("scene display timestamp must be finite and non-negative")
    if type(quality) is not int or not 1 <= quality <= 95:
        raise ValueError("scene JPEG quality must be an integer from 1 through 95")
    _decode_started = time.monotonic()
    with _read_image(jpeg) as image:
        _draw_started = time.monotonic()
        if breakdown is not None:
            breakdown["decode_seconds"] = _draw_started - _decode_started
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        width, height = image.size
        stroke = max(1, min(4, round(min(width, height) / 300)))
        people = _people(snapshot, media_timestamp)
        subject_phases: dict[str, str] = {}
        for event in _display_events(snapshot, media_timestamp):
            phase = event["phase"]
            if phase not in {"suspected", "confirmed"}:
                continue
            for subject in event["subject_ids"]:
                # A suspected event must not override another confirmed event.
                if subject_phases.get(subject) != "confirmed":
                    subject_phases[subject] = phase
        visible_ids: set[str] = set()
        for entity, box, age in people:
            subject = _safe_text(entity.get("subject_id"), fallback="person")
            raw_subject = entity.get("subject_id")
            if isinstance(raw_subject, str):
                visible_ids.add(raw_subject)
            identity = entity.get("identity")
            name = (
                _safe_text(identity.get("name"))
                if isinstance(identity, Mapping)
                else ""
            )
            label = f"{subject} | {name}" if name and name != subject else subject
            held = age > 1e-6
            if held:
                # Place stale-position status first so clipping a long identity
                # label cannot make held geometry look like a current observation.
                label = f"hold {age:.1f}s | {label}"
            color = _PHASE_COLORS.get(
                subject_phases.get(raw_subject),
                _YELLOW if entity.get("identity_pending")
                else _GRAY if held else _GREEN,
            )
            bounds = (
                round(box[0] * (width - 1)),
                round(box[1] * (height - 1)),
                round(box[2] * (width - 1)),
                round(box[3] * (height - 1)),
            )
            contours = _source_contours(entity, media_timestamp, width, height)
            if contours:
                for contour in contours:
                    draw.line((*contour, contour[0]), fill=color, width=stroke)
            elif held:
                _dashed_box(draw, bounds, color, stroke)
            else:
                draw.rectangle(bounds, outline=color, width=stroke)
            x, y = bounds[0], max(0, bounds[1] - 15)
            label = label[:96]
            extent = draw.textbbox((x, y), label, font=font)
            draw.rectangle(
                (x, y, min(width - 1, extent[2] + 4), min(height - 1, y + 14)),
                fill=(16, 23, 35),
            )
            draw.text((x + 2, y), label, font=font, fill=color)
        for index, (text, color) in enumerate(
            _event_lines(snapshot, media_timestamp, visible_ids)
        ):
            y = 5 + index * 16
            if y + 14 >= height or width < 16:
                break
            extent = draw.textbbox((7, y), text, font=font)
            draw.rectangle(
                (5, y - 1, min(width - 1, extent[2] + 5), y + 14),
                fill=(16, 23, 35),
            )
            draw.text((7, y), text, font=font, fill=color)
        _encode_started = time.monotonic()
        if breakdown is not None:
            breakdown["draw_seconds"] = _encode_started - _draw_started
        with io.BytesIO() as encoded:
            image.save(encoded, format="JPEG", quality=quality)
            result = encoded.getvalue()
        if breakdown is not None:
            breakdown["encode_seconds"] = time.monotonic() - _encode_started
        if len(result) > _MAX_JPEG_BYTES:
            raise ValueError("rendered scene JPEG exceeds size limit")
        return result


class SceneDisplayWorker:
    """Session-owned latest-frame rendering; caller generation-guards publish."""

    def __init__(
        self,
        scene: SceneRuntime,
        publish: Callable[[FramePacket], None],
        *,
        quality: int = 85,
        max_fps: float = 10,
    ) -> None:
        if type(quality) is not int or not 1 <= quality <= 95:
            raise ValueError("scene JPEG quality must be an integer from 1 through 95")
        if not _finite(max_fps) or not 0 < max_fps <= 60:
            raise ValueError("scene display FPS must be finite and within (0, 60]")
        self.scene = scene
        self._publish = publish
        self._quality = quality
        self._interval = 1 / max_fps
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._pending: FramePacket | None = None
        self._last_timestamp = -1.0
        self._last_sequence = -1
        self._stats = {
            "submitted_frames": 0,
            "published_frames": 0,
            "replaced_frames": 0,
            "rejected_frames": 0,
            "discarded_frames": 0,
            "render_failures": 0,
            "publish_failures": 0,
            "active_frames": 0,
            "last_render_seconds": None,
        }

    def start(self) -> None:
        with self._condition:
            if self._stop.is_set() or self._thread is not None:
                return
            self._thread = Thread(target=self._run, name="scene-display", daemon=True)
            self._thread.start()

    def submit(self, packet: FramePacket) -> None:
        """Keep one immutable JPEG reference; no decode, snapshot, or rendering."""
        with self._condition:
            if self._stop.is_set():
                return
            if (
                not isinstance(packet, FramePacket)
                or not isinstance(packet.jpeg, bytes)
                or not 0 < len(packet.jpeg) <= _MAX_JPEG_BYTES
                or not _finite(packet.media_timestamp)
                or packet.media_timestamp < 0
                or type(packet.frame_sequence) is not int
                or packet.media_timestamp < self._last_timestamp
                or packet.frame_sequence <= self._last_sequence
            ):
                self._stats["rejected_frames"] += 1
                return
            self._last_timestamp = packet.media_timestamp
            self._last_sequence = packet.frame_sequence
            if self._pending is not None:
                self._stats["replaced_frames"] += 1
            self._pending = packet
            self._stats["submitted_frames"] += 1
            self._condition.notify_all()

    def _run(self) -> None:
        next_render = 0.0
        while not self._stop.is_set():
            with self._condition:
                while not self._stop.is_set():
                    remaining = next_render - time.monotonic()
                    if self._pending is not None and remaining <= 0:
                        break
                    self._condition.wait(
                        timeout=max(0.001, remaining) if self._pending else None
                    )
                if self._stop.is_set():
                    return
                packet, self._pending = self._pending, None
                self._stats["active_frames"] = 1
            assert packet is not None
            started = time.monotonic()
            next_render = started + self._interval
            try:
                snapshot = self.scene.display_snapshot(packet.media_timestamp)
                if self._stop.is_set() or snapshot.get("state") == "stopped":
                    with self._condition:
                        self._stats["discarded_frames"] += 1
                    continue
                rendered = render_scene_jpeg(
                    packet.jpeg, snapshot, packet.media_timestamp, quality=self._quality
                )
                output = replace(
                    packet,
                    jpeg=rendered,
                    annotation_mode="held"
                    if _people(snapshot, packet.media_timestamp)
                    else "none",
                    analysis_message_id=None,
                    next_analysis_message_id=None,
                )
            except Exception:
                with self._condition:
                    self._stats["render_failures"] += 1
                continue
            finally:
                with self._condition:
                    self._stats["active_frames"] = 0
                    self._stats["last_render_seconds"] = round(
                        time.monotonic() - started, 6
                    )
            if self._stop.is_set():
                with self._condition:
                    self._stats["discarded_frames"] += 1
                continue
            try:
                self._publish(output)
            except Exception:
                with self._condition:
                    self._stats["publish_failures"] += 1
            else:
                with self._condition:
                    self._stats["published_frames"] += 1

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                **self._stats,
                "state": (
                    "stopping"
                    if self._stop.is_set() and self._thread and self._thread.is_alive()
                    else "stopped"
                    if self._stop.is_set()
                    else "running"
                    if self._thread
                    else "ready"
                ),
                "pending_frames": int(self._pending is not None),
                "pending_bytes": len(self._pending.jpeg) if self._pending else 0,
                "max_pending_frames": 1,
            }

    def close(self, timeout_seconds: float = 1.0) -> None:
        if not _finite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("close timeout must be finite and non-negative")
        self._stop.set()
        with self._condition:
            if self._pending is not None:
                self._stats["discarded_frames"] += 1
            self._pending = None
            thread = self._thread
            self._condition.notify_all()
        if thread is not None and thread is not current_thread():
            thread.join(timeout_seconds)


@dataclass(frozen=True, slots=True)
class _SourceStamp:
    sequence: int
    timestamp: float
    observed_at: str
    received_at: float


class TrackedSceneDisplayWorker(SceneDisplayWorker):
    """Display completed sampled sources, after their minimum presentation delay.

    Pending JPEGs remain in the scene's bounded evidence cache. This worker keeps
    512 source stamps and the last displayed source JPEG and geometry, so late
    events can repaint it even after evidence eviction. Capture packets update
    freshness but are never substituted for missing source images.
    """

    def __init__(self, scene, publish, *, quality=85, max_fps=4, delay_seconds=0.0):
        super().__init__(scene, publish, quality=quality, max_fps=max_fps)
        if not _finite(delay_seconds) or not 0 <= delay_seconds <= 30:
            raise ValueError(
                "tracked display delay must be between zero and 30 seconds"
            )
        self._delay_seconds = float(delay_seconds)
        self._sources: deque[_SourceStamp] = deque(maxlen=512)
        self._displayed_source: _SourceStamp | None = None
        self._displayed_frame: tuple[EvidenceFrame, list[dict[str, Any]]] | None = None
        self._stopped_at: float | None = None
        self._first_source_received_at: float | None = None
        self._last_published_sequence = -1
        self._event_revision = 0
        self._displayed_event_revision = 0
        self._event_signature: tuple | None = None
        self._display_state = "buffering"
        self._stats.update(
            source_stamps_dropped=0,
            unavailable_sources=0,
            unavailable_source_frames=0,
            displayed_source_timestamp_seconds=None,
            stream_delay_seconds=None,
            distinct_source_frames=0,
            same_source_repaints=0,
            last_snapshot_seconds=None,
            last_jpeg_render_seconds=None,
            last_publish_seconds=None,
            last_source_to_publish_seconds=None,
            timing_log_failures=0,
        )

    def set_delay_seconds(self, delay_seconds: float) -> None:
        if not _finite(delay_seconds) or not 0 <= delay_seconds <= 30:
            raise ValueError(
                "tracked display delay must be between zero and 30 seconds"
            )
        with self._condition:
            self._delay_seconds = float(delay_seconds)
            self._condition.notify_all()

    def submit(self, packet: FramePacket, *, retain_source: bool = True) -> None:
        """Record capture metadata; retain_source marks actual SAM admissions."""
        with self._condition:
            if self._stop.is_set():
                return
            if (
                not isinstance(packet, FramePacket)
                or not isinstance(packet.jpeg, bytes)
                or not 0 < len(packet.jpeg) <= _MAX_JPEG_BYTES
                or not _finite(packet.media_timestamp)
                or packet.media_timestamp < 0
                or type(packet.frame_sequence) is not int
                or packet.media_timestamp < self._last_timestamp
                or packet.frame_sequence <= self._last_sequence
                or type(retain_source) is not bool
            ):
                self._stats["rejected_frames"] += 1
                return
            self._last_timestamp = packet.media_timestamp
            self._last_sequence = packet.frame_sequence
            self._stats["submitted_frames"] += 1
            received_at = time.monotonic()
            if self._first_source_received_at is None:
                self._first_source_received_at = received_at
            if retain_source:
                if len(self._sources) == self._sources.maxlen:
                    self._stats["source_stamps_dropped"] += 1
                self._sources.append(
                    _SourceStamp(
                        packet.frame_sequence,
                        packet.media_timestamp,
                        packet.observed_at,
                        received_at,
                    )
                )
            self._condition.notify_all()

    def notify_tracking(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def notify_events(self, events=None) -> None:
        """Refresh the already displayed source when a late verdict arrives."""
        signature = None if events is None else tuple(
            (
                event.get("event_id"), event.get("phase"),
                event.get("observation_state"),
                event.get("last_evidence_timestamp_seconds"),
                tuple(event.get("subject_ids", ())),
            )
            for event in events[:_MAX_EVENTS]
        )
        with self._condition:
            if signature is not None and signature == self._event_signature:
                return
            self._event_signature = signature
            self._event_revision += 1
            self._condition.notify_all()

    def _display_elapsed_seconds(self, now: float) -> float:
        """Caller holds the condition; include startup and waiting after capture."""
        if self._first_source_received_at is None:
            return 0.0
        end = self._stopped_at if self._stopped_at is not None else now
        return max(0.0, end - self._first_source_received_at)

    def _log_display_timing(self, timing: dict[str, Any]) -> None:
        """Diagnostics must never turn a successful publication into a failure."""
        try:
            timing["recorded_at"] = datetime.now(timezone.utc).isoformat()
            session_id = self.scene.stream_session_id
            if (
                isinstance(session_id, str)
                and _SAFE_SESSION_IDENTIFIER.fullmatch(session_id)
            ):
                timing["stream_session_id"] = session_id
            _DISPLAY_LOGGER.info("%s", json.dumps(timing, separators=(",", ":")))
        except Exception:
            with self._condition:
                self._stats["timing_log_failures"] += 1

    def _run(self) -> None:
        next_render = 0.0
        while not self._stop.is_set():
            with self._condition:
                if self._stop.is_set():
                    return
                refresh = (
                    self._displayed_source
                    if self._event_revision > self._displayed_event_revision else None
                )
                if not self._sources and refresh is None:
                    self._condition.wait()
                    continue
                pending = tuple(
                    (source.sequence, source.timestamp) for source in self._sources
                )
            # Never acquire scene/registry locks while holding the display lock.
            # Reject/eviction is final for these already-arrived source JPEGs.
            unavailable = set(self.scene.unavailable_display_sources(pending))
            with self._condition:
                if self._stop.is_set():
                    return
                if unavailable:
                    previous_count = len(self._sources)
                    self._sources = deque(
                        (source for source in self._sources
                         if source.sequence not in unavailable),
                        maxlen=self._sources.maxlen,
                    )
                    removed = previous_count - len(self._sources)
                    self._stats["unavailable_sources"] += removed
                    self._stats["unavailable_source_frames"] += removed
                    self._stats["discarded_frames"] += removed
                    if removed:
                        self._display_state = "buffering"
                    self._condition.notify_all()
                now = time.monotonic()
                ready = tuple(
                    source for source in self._sources
                    if source.received_at + self._delay_seconds <= now
                )
                refresh = (
                    self._displayed_source
                    if self._event_revision > self._displayed_event_revision else None
                )
                if (not ready and refresh is None) or now < next_render:
                    self._condition.wait(
                        max(0.001, min(0.25, next_render - now))
                        if ready or refresh is not None else 0.25
                    )
                    continue
                after_sequence = self._last_published_sequence
                capture_timestamp = self._last_timestamp
                event_revision = self._event_revision
                displayed_frame = self._displayed_frame
                self._stats["active_frames"] = 1
            started = time.monotonic()
            next_render = started + self._interval
            source = output = selected = None
            try:
                late_event_repaint = False
                snapshot_started = time.monotonic()
                if ready:
                    selected = self.scene.tracked_display_frame(
                        before_timestamp_seconds=ready[-1].timestamp,
                        after_frame_sequence=after_sequence,
                    )
                if (
                    selected is None and refresh is not None
                    and displayed_frame is not None
                ):
                    frame, entities = displayed_frame
                    snapshot = self.scene.display_snapshot(
                        refresh.timestamp, frame_sequence=refresh.sequence,
                        source_entities=entities,
                    )
                    selected = frame, snapshot
                    late_event_repaint = True
                snapshot_seconds = time.monotonic() - snapshot_started
                if selected is None:
                    with self._condition:
                        self._display_state = "buffering"
                        self._stats["unavailable_sources"] += 1
                        self._displayed_event_revision = event_revision
                    continue
                source, snapshot = selected
                stamp = next(
                    (
                        item
                        for item in (*ready, *((refresh,) if refresh else ()))
                        if item.sequence == source.frame_sequence
                        and item.timestamp == source.timestamp_seconds
                    ),
                    None,
                )
                if (
                    stamp is None
                    or self._stop.is_set()
                    or snapshot.get("state") == "stopped"
                ):
                    with self._condition:
                        self._stats["discarded_frames"] += 1
                    continue
                render_started = time.monotonic()
                render_breakdown: dict[str, float] = {}
                rendered = render_scene_jpeg(
                    source.jpeg_bytes,
                    snapshot,
                    source.timestamp_seconds,
                    quality=self._quality,
                    breakdown=render_breakdown,
                )
                render_seconds = time.monotonic() - render_started
                with self._condition:
                    capture_timestamp = max(capture_timestamp, self._last_timestamp)
                    # Settings can change while rendering. Never publish sooner
                    # than a newly increased user minimum delay.
                    if time.monotonic() < stamp.received_at + self._delay_seconds:
                        continue
                elapsed = time.monotonic() - stamp.received_at
                delay = max(0.0, capture_timestamp - stamp.timestamp, elapsed)
                output = FramePacket(
                    jpeg=rendered,
                    media_timestamp=stamp.timestamp,
                    frame_sequence=stamp.sequence,
                    observed_at=stamp.observed_at,
                    annotation_mode="exact"
                    if _people(snapshot, stamp.timestamp)
                    else "none",
                    stream_delay_seconds=delay,
                )
                if self._stop.is_set():
                    with self._condition:
                        self._stats["discarded_frames"] += 1
                    continue
                try:
                    publish_started = time.monotonic()
                    self._publish(output)
                except Exception:
                    with self._condition:
                        self._stats["publish_failures"] += 1
                else:
                    published_at = time.monotonic()
                    publish_seconds = published_at - publish_started
                    source_to_publish_seconds = max(
                        0.0, published_at - stamp.received_at
                    )
                    with self._condition:
                        same_source = stamp.sequence == self._last_published_sequence
                        self._stats[
                            "same_source_repaints" if same_source
                            else "distinct_source_frames"
                        ] += 1
                        self._last_published_sequence = stamp.sequence
                        self._displayed_source = stamp
                        self._displayed_frame = source, snapshot["entities"]
                        self._displayed_event_revision = event_revision
                        while (
                            self._sources
                            and self._sources[0].sequence <= stamp.sequence
                        ):
                            self._sources.popleft()
                        self._display_state = "running"
                        self._stats.update(
                            published_frames=self._stats["published_frames"] + 1,
                            displayed_source_timestamp_seconds=stamp.timestamp,
                            stream_delay_seconds=round(delay, 3),
                            last_snapshot_seconds=round(snapshot_seconds, 6),
                            last_jpeg_render_seconds=round(render_seconds, 6),
                            last_publish_seconds=round(publish_seconds, 6),
                            last_source_to_publish_seconds=round(
                                source_to_publish_seconds, 6
                            ),
                        )
                        elapsed = self._display_elapsed_seconds(published_at)
                        timing = {
                            "event": "sam_display_timing",
                            "source_frame_sequence": stamp.sequence,
                            "source_timestamp_seconds": stamp.timestamp,
                            "source_to_publish_seconds": round(
                                source_to_publish_seconds, 6
                            ),
                            "snapshot_seconds": round(snapshot_seconds, 6),
                            "render_seconds": round(render_seconds, 6),
                            "render_decode_seconds": round(
                                render_breakdown.get("decode_seconds", 0.0), 6
                            ),
                            "render_draw_seconds": round(
                                render_breakdown.get("draw_seconds", 0.0), 6
                            ),
                            "render_encode_seconds": round(
                                render_breakdown.get("encode_seconds", 0.0), 6
                            ),
                            "publish_seconds": round(publish_seconds, 6),
                            "late_event_repaint": late_event_repaint,
                            "distinct_source_frames": self._stats[
                                "distinct_source_frames"
                            ],
                            "same_source_repaints": self._stats["same_source_repaints"],
                            "distinct_source_fps": round(
                                self._stats["distinct_source_frames"] / elapsed, 6
                            ) if elapsed > 0 else 0.0,
                        }
                    self._log_display_timing(timing)
            except Exception:
                with self._condition:
                    self._stats["render_failures"] += 1
            finally:
                source = output = selected = None
                with self._condition:
                    self._stats["active_frames"] = 0
                    self._stats["last_render_seconds"] = round(
                        time.monotonic() - started, 6
                    )
                    self._condition.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._condition:
            result = super().stats()
            now = time.monotonic()
            elapsed = self._display_elapsed_seconds(now)
            result["display_elapsed_seconds"] = round(elapsed, 6)
            result["distinct_source_fps"] = round(
                self._stats["distinct_source_frames"] / elapsed, 6
            ) if elapsed > 0 else 0.0
            if self._thread and not self._stop.is_set():
                result["state"] = self._display_state
            if self._displayed_source is not None:
                now = (
                    self._stopped_at
                    if self._stopped_at is not None else time.monotonic()
                )
                result["stream_delay_seconds"] = round(
                    max(
                        0.0,
                        self._last_timestamp - self._displayed_source.timestamp,
                        now - self._displayed_source.received_at,
                    ), 3,
                )
            return result | {
                "mode": "tracked_source",
                "buffer_delay_seconds": self._delay_seconds,
                "capture_timestamp_seconds": self._last_timestamp
                if self._last_timestamp >= 0
                else None,
                "pending_frames": len(self._sources),
                "max_pending_frames": self._sources.maxlen,
                "pending_bytes": 0,
                "retained_display_bytes": (
                    len(self._displayed_frame[0].jpeg_bytes)
                    if self._displayed_frame is not None else 0
                ),
                "last_displayed_frame_sequence": self._last_published_sequence
                if self._last_published_sequence >= 0
                else None,
            }

    def wait_until_idle(self, timeout_seconds: float = 1.0) -> bool:
        """After native EOF flush, wait boundedly for remaining delayed sources."""
        if not _finite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("display drain timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while (
                self._sources or self._stats["active_frames"]
                or (
                    self._displayed_source is not None
                    and self._event_revision > self._displayed_event_revision
                )
            ):
                if self._stop.is_set():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout_seconds: float = 1.0) -> None:
        if not _finite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("close timeout must be finite and non-negative")
        with self._condition:
            self._stop.set()
            if self._stopped_at is None:
                self._stopped_at = time.monotonic()
            self._stats["discarded_frames"] += len(self._sources)
            self._sources.clear()
            self._displayed_frame = None
        super().close(timeout_seconds)
