"""Source-time scene overlays composed outside the live video callback."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import Condition, Event, Thread, current_thread
from typing import Any

from PIL import ImageDraw, ImageFont

from .result_bus import FramePacket
from .scene_inspection import _read_image
from .scene_runtime import SceneRuntime

_MAX_JPEG_BYTES = 16 * 1024 * 1024
_MAX_PEOPLE = 128
_MAX_EVENTS = 2048  # Same hard ceiling as SceneState's event registry.
_MAX_LINES = 8
_HOLD_SECONDS = 1.0
_GREEN = (73, 220, 138)
_AMBER = (255, 192, 76)
_RED = (250, 104, 99)
_GRAY = (190, 198, 211)
_PHASE_COLORS = {
    "suspected": _AMBER,
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


def _event_lines(
    snapshot: Mapping[str, Any],
    timestamp: float,
    visible_ids: set[str],
) -> list[tuple[str, tuple[int, ...]]]:
    events = _scene(snapshot).get("events", ())
    if not isinstance(events, (tuple, list)):
        return []
    candidates = []
    for event in events[:_MAX_EVENTS]:
        if not isinstance(event, Mapping):
            continue
        phase = event.get("phase")
        at = event.get("last_evidence_timestamp_seconds")
        if not isinstance(phase, str) or phase not in _PHASE_COLORS:
            continue
        if not _finite(at) or at < 0 or at > timestamp:
            continue
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
        if label not in ("fight", "smoking", "fall", "weapon", "stranger"):
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


def render_scene_jpeg(
    jpeg: bytes,
    snapshot: Mapping[str, Any],
    media_timestamp: float,
    quality: int = 85,
) -> bytes:
    """Render presentation state only; held geometry is never event evidence.

    Boxes come from observations at or before the source frame timestamp. A
    one-second visual hold is dashed and age-labelled; older/future geometry is
    omitted. Event text can persist independently and explicitly shows uncertain
    visibility. Metadata and credentials are not copied into the JPEG.
    """
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= _MAX_JPEG_BYTES:
        raise ValueError("scene JPEG must be nonempty and at most 16 MiB")
    if not isinstance(snapshot, Mapping):
        raise ValueError("scene snapshot must be a mapping")
    if not _finite(media_timestamp) or media_timestamp < 0:
        raise ValueError("scene display timestamp must be finite and non-negative")
    if type(quality) is not int or not 1 <= quality <= 95:
        raise ValueError("scene JPEG quality must be an integer from 1 through 95")
    with _read_image(jpeg) as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        width, height = image.size
        stroke = max(1, min(4, round(min(width, height) / 300)))
        people = _people(snapshot, media_timestamp)
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
            color = _AMBER if held else _GREEN
            bounds = (
                round(box[0] * (width - 1)),
                round(box[1] * (height - 1)),
                round(box[2] * (width - 1)),
                round(box[3] * (height - 1)),
            )
            if held:
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
        with io.BytesIO() as encoded:
            image.save(encoded, format="JPEG", quality=quality)
            result = encoded.getvalue()
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
                snapshot = self.scene.snapshot()
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
