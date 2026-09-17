"""Replay local videos in real time through the full live pipeline.

Each video is paced like a camera and passes through the real sampler, the
per-branch windows, the real prompt builder and response parser, the result
organizer and the session diagnostics. Only the remote analysis service is
replaced by a simulator, which can be slowed down, time out, fail
authentication, return unparseable text, or finish out of order.

The simulator behaves as an *oracle*: it reports an event exactly when a frame
it received lies inside a labelled interval. A miss therefore means the action
evidence never reached the analysis request (not sampled, dropped, failed or
too late), not that a model failed to recognise it. Real recognition quality
still needs labelled footage and the real service.

``--speed`` compresses the timeline: a 2 s action replayed at ``--speed 3``
lasts 0.67 s for the sampler, which is how fast events are emulated. The
sampling rate stays the configured rate in compressed time.

Example::

    python tools/replay_stress.py clip.mp4 --event smoking:2.0-3.5 \\
        --speed 2 --latency-ms 900 --slow-rate 0.1 --timeout-rate 0.05 \\
        --output tmp/replay/report.json

With ``--real`` the safety branch calls the configured analysis service with
the same per-event sensitivity policy as the live service (smoking defaults to
high; override with ``--event-sensitivity smoking=standard``). Paid requests
are capped by ``--max-requests``: the run is refused up front when the
estimated count exceeds it, and any request beyond it fails without being
sent.

Manifests use ``{"clips": [{"id", "path", "events": [{"label", "start",
"end"}], "ignore": [{"start", "end"}]}]}``. Clips without ``events`` are
treated as all-negative; positives inside ``ignore`` intervals are not counted
as false alarms. Relative paths resolve against the manifest's directory, then
the working directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.config import HarnessConfig, load_config  # noqa: E402
from goodvision_harness.diagnostics import failure_category  # noqa: E402
from goodvision_harness.event_sensitivity import (  # noqa: E402
    normalize_event_sensitivities,
)
from goodvision_harness.grounding import (  # noqa: E402
    DETECTION_EVENT_LABELS,
    SAFETY_EVENT_LABELS,
    DryRunStrangerGroundingClient,
    build_selected_features_grounding_client,
)
from goodvision_harness.model_api import ModelAPIClient, ModelAPIError  # noqa: E402
from goodvision_harness.pipeline import (  # noqa: E402
    FILE_MODE,
    SessionCallbacks,
    StreamingSession,
    mixed_branch_sampling,
)
from goodvision_harness.video_to_stream import (  # noqa: E402
    LiveFrame,
    VideoFileStream,
    probe_video,
)

_TIMESTAMP_PATTERN = re.compile(r"f\d+=(\d+(?:\.\d+)?)s")
_OBJECT_LABELS = {
    "fight": "person",
    "smoking": "cigarette",
    "fall": "person",
    "weapon": "knife",
}
_ZH_NAMES = {"fight": "打架", "smoking": "吸烟", "fall": "跌倒", "weapon": "持械"}


# ------------------------------------------------------------------ inputs

@dataclass(frozen=True, slots=True)
class LabelledEvent:
    label: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.label not in SAFETY_EVENT_LABELS:
            raise ValueError(f"unsupported event label: {self.label}")
        if not 0 <= self.start <= self.end:
            raise ValueError("event interval must satisfy 0 <= start <= end")

    def scaled(self, speed: float) -> LabelledEvent:
        return LabelledEvent(self.label, self.start / speed, self.end / speed)


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    path: Path
    events: tuple[LabelledEvent, ...] = ()
    # Ambiguous source intervals (e.g. an item half-raised): positives inside
    # them are neither hits nor false alarms.
    ignore: tuple[tuple[float, float], ...] = ()


def parse_event(text: str) -> LabelledEvent:
    """Parse ``label:start-end`` in source seconds."""

    match = re.fullmatch(r"\s*([a-z]+)\s*:\s*([\d.]+)\s*-\s*([\d.]+)\s*", text)
    if match is None:
        raise argparse.ArgumentTypeError("events use label:start-end, e.g. fight:2-3.5")
    try:
        return LabelledEvent(match[1], float(match[2]), float(match[3]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def load_manifest(path: Path) -> list[Clip]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = []
    for item in data.get("clips", []):
        clip_path = Path(item["path"]).expanduser()
        if not clip_path.is_absolute():
            beside = path.parent / clip_path
            clip_path = beside if beside.exists() else Path.cwd() / clip_path
        events = tuple(
            LabelledEvent(str(e["label"]), float(e["start"]), float(e["end"]))
            for e in item.get("events", [])
        )
        ignore = tuple(
            (float(span["start"]), float(span["end"]))
            for span in item.get("ignore", [])
        )
        if any(not 0 <= start <= end for start, end in ignore):
            raise ValueError("ignore intervals must satisfy 0 <= start <= end")
        clips.append(Clip(
            str(item.get("id") or clip_path.stem), clip_path, events, ignore,
        ))
    if not clips:
        raise ValueError("manifest contains no clips")
    return clips


# -------------------------------------------------------------- simulation

def oracle_labels(
    events: Sequence[LabelledEvent],
    labels: Sequence[str],
    timestamps: Sequence[float],
) -> tuple[str, ...]:
    """Labels whose interval contains any of the sent frame timestamps."""

    hits = {
        event.label
        for event in events
        for ts in timestamps
        # Timestamps are printed with millisecond precision.
        if event.start - 0.0005 <= ts <= event.end + 0.0005
    }
    return tuple(label for label in labels if label in hits)


def _timestamps_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    text = ""
    for part in messages[-1].get("content", []) if messages else []:
        if isinstance(part, Mapping) and part.get("type") == "text":
            text += str(part.get("text", ""))
    return tuple(float(v) for v in _TIMESTAMP_PATTERN.findall(text))

@dataclass(slots=True)
class SimulationProfile:
    latency_ms: float = 800.0
    latency_sigma: float = 0.35
    slow_rate: float = 0.0
    slow_ms: float = 4000.0
    timeout_rate: float = 0.0
    timeout_ms: float = 8000.0
    auth_rate: float = 0.0
    parse_error_rate: float = 0.0
    false_positive_rate: float = 0.0
    face_latency_ms: float = 150.0
    seed: int = 7

    def validate(self) -> None:
        for name in (
            "slow_rate", "timeout_rate", "auth_rate", "parse_error_rate",
            "false_positive_rate",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.auth_rate + self.timeout_rate + self.parse_error_rate > 1:
            raise ValueError("failure rates must not add up to more than 1")
        for name in ("latency_ms", "slow_ms", "timeout_ms", "face_latency_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(slots=True)
class CallRecord:
    call_id: int
    started: float
    timestamps: tuple[float, ...]
    outcome: str = "pending"
    finished: float | None = None
    positives: tuple[str, ...] = ()
    oracle_positives: tuple[str, ...] = ()


class SimulatedAnalysisTransport:
    """Stands in for the remote service behind the real multi-event client."""

    def __init__(
        self,
        profile: SimulationProfile,
        events: Sequence[LabelledEvent],
        labels: Sequence[str],
        *,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        profile.validate()
        self.profile = profile
        self.events = tuple(events)
        self.labels = tuple(labels)
        self._sleep = sleep
        self._clock = clock
        self._rng = random.Random(profile.seed)
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []

    def _draw(self) -> tuple[str, float, bool]:
        p = self.profile
        with self._lock:
            roll = self._rng.random()
            latency = p.latency_ms * math.exp(self._rng.gauss(0.0, p.latency_sigma))
            if self._rng.random() < p.slow_rate:
                latency = max(latency, p.slow_ms)
            false_positive = self._rng.random() < p.false_positive_rate
        if roll < p.auth_rate:
            return "auth", min(latency, 150.0), False
        roll -= p.auth_rate
        if roll < p.timeout_rate:
            return "timeout", p.timeout_ms, False
        roll -= p.timeout_rate
        if roll < p.parse_error_rate:
            return "parse", latency, False
        return "ok", latency, false_positive

    def oracle(self, timestamps: Sequence[float]) -> tuple[str, ...]:
        return oracle_labels(self.events, self.labels, timestamps)

    def chat_completion(self, *, messages: Sequence[Mapping[str, Any]], **_: Any):
        timestamps = _timestamps_from_messages(messages)
        with self._lock:
            record = CallRecord(len(self.calls) + 1, self._clock(), timestamps)
            self.calls.append(record)
        outcome, latency_ms, false_positive = self._draw()
        self._sleep(latency_ms / 1000.0)
        record.finished = self._clock()
        record.outcome = outcome
        if outcome == "auth":
            raise ModelAPIError("simulated 401", code="http_error", status_code=401)
        if outcome == "timeout":
            raise ModelAPIError("simulated timeout", code="timeout")
        if outcome == "parse":
            return {"text": "simulated unparseable reply"}
        record.oracle_positives = self.oracle(timestamps)
        positives = set(record.oracle_positives)
        if false_positive and self.labels:
            with self._lock:
                positives.add(self._rng.choice(self.labels))
        record.positives = tuple(label for label in self.labels if label in positives)
        return {"text": self._reply(record.positives)}

    def assistant_text(self, response: Mapping[str, Any]) -> str:
        return str(response["text"])

    def _reply(self, positives: Sequence[str]) -> str:
        if not positives:
            return json.dumps({"no_events": 0.95})
        events = []
        for label in self.labels:
            detected = label in positives
            zh = _ZH_NAMES[label]
            events.append({
                "event_label": label,
                "detected": detected,
                "confidence": 0.9 if detected else 0.95,
                "description": (
                    f"模拟{zh}事件。 / Simulated {label} event."
                    if detected
                    else f"未发现{zh}。 / No simulated {label}."
                ),
                "objects": (
                    [{
                        "label": _OBJECT_LABELS[label],
                        "confidence": 0.9,
                        "bbox_2d": [300, 200, 600, 900],
                    }]
                    if detected else []
                ),
            })
        return json.dumps({"events": events}, ensure_ascii=False)


class RequestBudget:
    """Hard cap on paid requests shared by every replayed clip and speed."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("request budget must be at least 1")
        self.limit = int(limit)
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


class RecordingTransport:
    """The configured real service, with the simulator's per-request records."""

    def __init__(
        self,
        inner: Any,
        events: Sequence[LabelledEvent],
        labels: Sequence[str],
        *,
        budget: RequestBudget | None = None,
        clock=time.monotonic,
    ) -> None:
        self._inner = inner
        self.events = tuple(events)
        self.labels = tuple(labels)
        self._budget = budget
        self._clock = clock
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []

    def chat_completion(self, *, messages: Sequence[Mapping[str, Any]], **params):
        timestamps = _timestamps_from_messages(messages)
        with self._lock:
            record = CallRecord(len(self.calls) + 1, self._clock(), timestamps)
            self.calls.append(record)
        if self._budget is not None and not self._budget.take():
            record.finished = self._clock()
            record.outcome = "budget_exhausted"
            # Never sent; reported as a failed request, not as a negative.
            raise ModelAPIError("replay request budget exhausted")
        try:
            response = self._inner.chat_completion(messages=messages, **params)
        except ModelAPIError as exc:
            record.finished = self._clock()
            record.outcome = failure_category({
                "stage": "request", "code": exc.code, "http_status": exc.status_code,
            })
            raise
        except Exception:
            record.finished = self._clock()
            record.outcome = "internal"
            raise
        record.finished = self._clock()
        record.outcome = "ok"
        record.oracle_positives = oracle_labels(self.events, self.labels, timestamps)
        return response

    def assistant_text(self, response: Any) -> str:
        return self._inner.assistant_text(response)


class SimulatedFaceClient:
    """Local-face stand-in with its own latency; never reports a stranger."""

    def __init__(self, settings, latency_ms: float, *, sleep=time.sleep) -> None:
        self._inner = DryRunStrangerGroundingClient(settings)
        self.settings = self._inner.settings
        self.memory_context = None
        self.model_name = self._inner.model_name
        self._latency = latency_ms / 1000.0
        self._sleep = sleep

    def classify_window(self, window):
        self._sleep(self._latency)
        # Window index 0 is a known-person dry-run observation.
        response = self._inner.classify_window(replace(window, index=0))
        return replace(response, window=window)


class CompressedTimelineStream:
    """A file stream whose timeline runs ``speed`` times faster."""

    def __init__(
        self, path: Path, *, speed: float, display_fps: float, realtime: bool,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self._inner = VideoFileStream(
            path,
            display_fps=display_fps * speed,
            playback_speed=speed,
            realtime=realtime,
        )
        self.info = self._inner.info

    @property
    def realtime(self) -> bool:
        return self._inner.realtime

    @property
    def dropped_frames(self) -> int:
        return self._inner.dropped_frames

    @property
    def stopped(self) -> bool:
        return self._inner.stopped

    def stop(self) -> None:
        self._inner.stop()

    def frames(self) -> Iterator[LiveFrame]:
        for frame in self._inner.frames():
            yield replace(frame, timestamp_seconds=frame.timestamp_seconds / self.speed)


# ------------------------------------------------------------------ session

def session_config(
    config: HarnessConfig,
    features: Sequence[str],
    *,
    event_fps: float | None = None,
    face_fps: float | None = None,
    window_size: int | None = None,
    window_stride: int | None = None,
    safety_side: int | None = None,
    safety_quality: int | None = None,
    concurrency: int | None = None,
    model_canvas: tuple[int, int] | str | None = "config",
) -> HarnessConfig:
    """Mirror the service's multi-feature session configuration.

    ``model_canvas`` keeps the configured fixed model canvas by default; pass
    ``None`` to send safety images at ``safety_side`` as they are, or a
    ``(width, height)`` to try another canvas. A canvas re-encodes at the
    safety JPEG quality.
    """

    canvas = config.inference.image_size if model_canvas == "config" else model_canvas
    inference = replace(
        config.inference,
        image_size=canvas,
        image_jpeg_quality=safety_quality or config.video.jpeg_quality,
    )
    video = replace(
        config.video,
        fps=event_fps or config.video.fps,
        max_image_side=safety_side or config.video.max_image_side,
        jpeg_quality=safety_quality or config.video.jpeg_quality,
        realtime=True,
        playback_speed=1.0,
    )
    window = replace(
        config.sliding_window,
        size_frames=window_size or config.sliding_window.size_frames,
        stride_frames=window_stride or config.sliding_window.stride_frames,
        include_partial_final_window=False,
    )
    base = replace(
        config,
        inference=inference,
        video=video,
        sliding_window=window,
        face=replace(config.face, sample_fps=face_fps or config.face.sample_fps),
        runtime=replace(
            config.runtime,
            max_concurrent_requests=(
                concurrency or config.runtime.max_concurrent_requests
            ),
        ),
    )
    if "stranger" not in features:
        return base
    sampling = mixed_branch_sampling(base, features)
    return replace(
        base,
        video=replace(
            base.video,
            fps=sampling.base_fps if sampling else base.face.sample_fps,
            max_image_side=base.face.max_image_side,
            jpeg_quality=base.face.jpeg_quality,
        ),
        sliding_window=replace(base.sliding_window, size_frames=1, stride_frames=1),
        branch_sampling=sampling,
    )


@dataclass(slots=True)
class ReplayObservations:
    arrivals: list[tuple[float, float]] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def on_display_frame(self, _jpeg: bytes, timestamp: float, _index: int) -> None:
        with self.lock:
            self.arrivals.append((timestamp, time.monotonic()))

    def on_event(self, record: Mapping[str, Any]) -> None:
        if record.get("type") == "window" and record.get("state") in {
            "submitted", "dropped",
        }:
            with self.lock:
                self.windows.append(dict(record))

    def on_analysis_result(self, result) -> None:
        with self.lock:
            self.results.append({
                "emitted": time.monotonic(),
                "timestamp": result.frame.timestamp_seconds,
                "judgments": [
                    (j.event_label, j.detected) for j in result.judgments
                ],
            })

    def arrival_at(self, timestamp: float) -> float | None:
        with self.lock:
            for ts, wall in self.arrivals:
                if ts >= timestamp - 1e-6:
                    return wall
        return None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 4)


def _event_report(
    event: LabelledEvent,
    observations: ReplayObservations,
    calls: Sequence[CallRecord],
    *,
    window_span: float,
    real: bool = False,
) -> dict[str, Any]:
    def overlaps(start: float, end: float) -> bool:
        return start <= event.end + 1e-6 and end >= event.start - 1e-6

    safety_windows = [
        w for w in observations.windows
        if w.get("branch", "safety") != "stranger"
        and overlaps(float(w["start_seconds"]), float(w["end_seconds"]))
    ]
    sent = [
        c for c in calls
        if any(event.start - 0.0005 <= ts <= event.end + 0.0005 for ts in c.timestamps)
    ]
    ok = [c for c in sent if c.outcome == "ok"]
    alarms = [
        r for r in observations.results
        if (event.label, True) in r["judgments"]
        and event.start - 1e-6 <= r["timestamp"] <= event.end + window_span + 1e-6
    ]
    delay = None
    if alarms:
        first = min(alarms, key=lambda r: r["emitted"])
        arrival = observations.arrival_at(event.start)
        if arrival is not None:
            delay = round(first["emitted"] - arrival, 4)
    outcomes = Counter(c.outcome for c in sent)
    if alarms:
        verdict = "detected"
    elif not safety_windows:
        verdict = "not_sampled"
    elif not sent:
        verdict = "dropped"
    elif not ok:
        verdict = "request_failed"
    else:
        # The oracle always reports sent evidence; a real service may not.
        verdict = "not_recognized" if real else "result_not_published"
    return {
        "label": event.label,
        "start_seconds": round(event.start, 4),
        "end_seconds": round(event.end, 4),
        "duration_seconds": round(event.end - event.start, 4),
        "windows_touching_event": len(safety_windows),
        "windows_dropped": sum(1 for w in safety_windows if w["state"] == "dropped"),
        "requests_with_event_frame": len(sent),
        "request_outcomes": dict(outcomes),
        "alarms": len(alarms),
        "first_alarm_delay_seconds": delay,
        "verdict": verdict,
    }


def replay_clip(
    clip: Clip,
    config: HarnessConfig,
    features: Sequence[str],
    profile: SimulationProfile,
    *,
    speed: float = 1.0,
    realtime: bool = True,
    safety_side: int = 360,
    safety_quality: int = 75,
    real_transport: Any | None = None,
    event_sensitivities: Mapping[str, str] | None = None,
    request_budget: RequestBudget | None = None,
) -> dict[str, Any]:
    labels = tuple(f for f in features if f in SAFETY_EVENT_LABELS)
    events = tuple(e.scaled(speed) for e in clip.events if e.label in labels)
    ignored = tuple((start / speed, end / speed) for start, end in clip.ignore)
    # Same policy as the live service: unspecified events use its defaults.
    sensitivities = normalize_event_sensitivities(event_sensitivities, labels)
    real = real_transport is not None
    transport = (
        RecordingTransport(real_transport, events, labels, budget=request_budget)
        if real else SimulatedAnalysisTransport(profile, events, labels)
    )
    face_client = (
        SimulatedFaceClient(config.inference, profile.face_latency_ms)
        if "stranger" in features else None
    )
    client = build_selected_features_grounding_client(
        config.inference,
        config.model_api,
        features=tuple(features),
        stranger_client=face_client,
        safety_max_image_side=safety_side,
        safety_jpeg_quality=safety_quality,
        face_live_scheduler="single_flight",
        transport=transport,  # type: ignore[arg-type]
        event_sensitivities=sensitivities,
    )
    stream = CompressedTimelineStream(
        clip.path,
        speed=speed,
        display_fps=max(config.display.fps, config.video.fps),
        realtime=realtime,
    )
    observations = ReplayObservations()
    ended = threading.Event()
    end_reason: dict[str, Any] = {}

    def on_end(reason: str, error: str | None) -> None:
        end_reason.update(reason=reason, error=error)
        ended.set()

    session = StreamingSession(
        config,
        client,
        SessionCallbacks(
            on_event=observations.on_event,
            on_analysis_result=observations.on_analysis_result,
            on_display_frame=observations.on_display_frame,
            on_end=on_end,
        ),
        mode=FILE_MODE,
        video_stream=stream,  # type: ignore[arg-type]
        source_id=f"replay-{clip.clip_id}",
    )
    wall_started = time.monotonic()
    session.start()
    request_timeout = (
        config.model_api.timeout_seconds if real else profile.timeout_ms / 1000
    )
    timeout = stream.info.duration_seconds / speed + request_timeout + 30
    if not ended.wait(timeout):
        session.stop()
        ended.wait(10)
    session.join(10)
    wall = time.monotonic() - wall_started

    sampling = config.branch_sampling
    cadence = sampling.safety if sampling else None
    fps = cadence.fps if cadence else config.video.fps
    size = cadence.size_frames if cadence else config.sliding_window.size_frames
    window_span = (size - 1) / fps
    event_reports = [
        _event_report(
            e, observations, transport.calls, window_span=window_span, real=real,
        )
        for e in events
    ]
    false_alarms = [
        r for r in observations.results
        for label, detected in r["judgments"]
        if detected and label in labels
        and not any(
            e.label == label
            and e.start - 1e-6 <= r["timestamp"] <= e.end + window_span + 1e-6
            for e in events
        )
        and not any(
            start - 1e-6 <= r["timestamp"] <= end + window_span + 1e-6
            for start, end in ignored
        )
    ]
    order_violations = 0
    newest = -math.inf
    for result in sorted(observations.results, key=lambda r: r["emitted"]):
        if not any(label in labels for label, _ in result["judgments"]):
            continue
        if result["timestamp"] < newest - 1e-9:
            order_violations += 1
        newest = max(newest, result["timestamp"])
    detected = sum(1 for r in event_reports if r["verdict"] == "detected")
    delays = [
        r["first_alarm_delay_seconds"] for r in event_reports
        if r["first_alarm_delay_seconds"] is not None
    ]
    return {
        "clip": clip.clip_id,
        "path": str(clip.path),
        "analysis_service": "real" if real else "simulated",
        "event_sensitivities": dict(sensitivities),
        "speed": speed,
        "source_duration_seconds": round(stream.info.duration_seconds, 3),
        "replay_wall_seconds": round(wall, 3),
        "end_reason": end_reason.get("reason"),
        "source_frames_dropped_by_pacing": stream.dropped_frames,
        "summary": {
            "events": len(event_reports),
            "detected": detected,
            "missed": len(event_reports) - detected,
            "miss_reasons": dict(Counter(
                r["verdict"] for r in event_reports if r["verdict"] != "detected"
            )),
            "first_alarm_delay_p50": _percentile(delays, 0.5),
            "first_alarm_delay_p95": _percentile(delays, 0.95),
            "false_alarms": len(false_alarms),
            "requests": len(transport.calls),
            "request_outcomes": dict(Counter(c.outcome for c in transport.calls)),
            "late_results_out_of_order": order_violations,
        },
        "events": event_reports,
        "diagnostics": {
            key: value
            for key, value in session.diagnostics_snapshot().items()
            if key != "model_inputs"
        },
    }


def estimate_safety_requests(
    clips: Sequence[Clip],
    config: HarnessConfig,
    speeds: Sequence[float],
) -> int:
    """Upper bound on safety requests: one per sampled safety window."""

    sampling = config.branch_sampling
    fps = sampling.safety.fps if sampling else config.video.fps
    size = sampling.safety.size_frames if sampling else (
        config.sliding_window.size_frames
    )
    stride = sampling.safety.stride_frames if sampling else (
        config.sliding_window.stride_frames
    )
    total = 0
    for speed in speeds:
        for clip in clips:
            duration = probe_video(clip.path).duration_seconds / speed
            frames = math.floor(duration * fps) + 1
            if frames >= size:
                total += (frames - size) // stride + 1
    return total


def _event_sensitivity(text: str) -> tuple[str, str]:
    label, _, level = text.partition("=")
    label, level = label.strip(), level.strip()
    if label not in SAFETY_EVENT_LABELS or level not in {"standard", "high"}:
        raise argparse.ArgumentTypeError(
            "use label=standard|high with fight, smoking, fall or weapon"
        )
    return label, level


# ---------------------------------------------------------------------- cli

def _model_canvas(text: str) -> tuple[int, int] | str | None:
    value = text.strip().lower()
    if value in {"config", "none"}:
        return None if value == "none" else "config"
    width, sep, height = value.partition("x")
    if not sep or not width.isdigit() or not height.isdigit() or (
        int(width) < 1 or int(height) < 1
    ):
        raise argparse.ArgumentTypeError("use config, none or WIDTHxHEIGHT")
    return int(width), int(height)


def _features(text: str) -> tuple[str, ...]:
    selected = {item.strip() for item in text.split(",") if item.strip()}
    unknown = selected - set(DETECTION_EVENT_LABELS)
    if unknown or not selected:
        raise argparse.ArgumentTypeError(
            "features must be a non-empty subset of " + ",".join(DETECTION_EVENT_LABELS)
        )
    return tuple(label for label in DETECTION_EVENT_LABELS if label in selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--event", action="append", type=parse_event, default=[],
        help="label:start-end in source seconds (single video only)",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--features", type=_features, default=("fight", "smoking", "fall", "weapon"),
    )
    parser.add_argument("--speed", type=float, action="append", default=None,
                        help="timeline compression; repeat to sweep")
    parser.add_argument("--no-realtime", action="store_true",
                        help="decode as fast as possible (no pacing, no drops)")
    parser.add_argument("--event-fps", type=float)
    parser.add_argument("--face-fps", type=float)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--window-stride", type=int)
    parser.add_argument("--safety-side", type=int)
    parser.add_argument("--safety-quality", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--model-canvas", type=_model_canvas, default="config",
        help=(
            "fixed safety-model canvas: config (default, as the service), none "
            "(send --safety-side images as they are) or WIDTHxHEIGHT"
        ),
    )
    sim = parser.add_argument_group("simulated analysis service")
    for name, default in (
        ("latency-ms", 800.0), ("latency-sigma", 0.35), ("slow-rate", 0.0),
        ("slow-ms", 4000.0), ("timeout-rate", 0.0), ("timeout-ms", 8000.0),
        ("auth-rate", 0.0), ("parse-error-rate", 0.0),
        ("false-positive-rate", 0.0), ("face-latency-ms", 150.0),
    ):
        sim.add_argument(f"--{name}", type=float, default=default)
    sim.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--real", action="store_true",
        help=(
            "send safety requests to the configured analysis service (uses the "
            "API key from the environment/.env and incurs cost); stranger stays "
            "simulated"
        ),
    )
    parser.add_argument(
        "--event-sensitivity", action="append", type=_event_sensitivity,
        default=[], metavar="LABEL=LEVEL",
        help="per-event policy; unspecified events use the service defaults",
    )
    parser.add_argument(
        "--max-requests", type=int, default=300,
        help="upper bound on paid requests with --real (default 300)",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.manifest:
        clips = load_manifest(args.manifest)
    elif args.videos:
        if args.event and len(args.videos) != 1:
            raise SystemExit("--event applies to exactly one video; use --manifest")
        clips = [
            Clip(path.stem, path, tuple(args.event) if args.event else ())
            for path in args.videos
        ]
    else:
        raise SystemExit("give at least one video or --manifest")
    for clip in clips:
        if not clip.path.is_file():
            raise SystemExit(f"video not found: {clip.path}")

    profile = SimulationProfile(
        latency_ms=args.latency_ms, latency_sigma=args.latency_sigma,
        slow_rate=args.slow_rate, slow_ms=args.slow_ms,
        timeout_rate=args.timeout_rate, timeout_ms=args.timeout_ms,
        auth_rate=args.auth_rate, parse_error_rate=args.parse_error_rate,
        false_positive_rate=args.false_positive_rate,
        face_latency_ms=args.face_latency_ms, seed=args.seed,
    )
    profile.validate()
    base = load_config(args.config)
    config = session_config(
        base, args.features,
        event_fps=args.event_fps, face_fps=args.face_fps,
        window_size=args.window_size, window_stride=args.window_stride,
        safety_side=args.safety_side, safety_quality=args.safety_quality,
        concurrency=args.concurrency,
        model_canvas=args.model_canvas,
    )
    safety_side = args.safety_side or base.video.max_image_side
    safety_quality = args.safety_quality or base.video.jpeg_quality
    canvas = config.inference.image_size
    if canvas and args.safety_side and args.safety_side != max(canvas):
        print(
            f"note: every safety image is fitted into the {canvas[0]}x{canvas[1]} "
            "model canvas; add --model-canvas none to send "
            f"{args.safety_side} px images as they are",
            flush=True,
        )
    safety_labels = tuple(f for f in args.features if f in SAFETY_EVENT_LABELS)
    sensitivities = normalize_event_sensitivities(
        dict(args.event_sensitivity), safety_labels,
    )
    speeds = args.speed or [1.0]
    real_transport = None
    budget = None
    if args.real:
        if args.max_requests < 1:
            raise SystemExit("--max-requests must be at least 1")
        estimate = estimate_safety_requests(clips, config, speeds)
        if estimate > args.max_requests:
            raise SystemExit(
                f"about {estimate} paid requests exceed --max-requests "
                f"{args.max_requests}; raise the limit or replay fewer clips"
            )
        print(f"real service: up to {estimate} requests", flush=True)
        # Created only after the bound check: construction needs the API key.
        real_transport = ModelAPIClient(base.model_api)
        budget = RequestBudget(args.max_requests)
    runs = []
    for speed in speeds:
        for clip in clips:
            report = replay_clip(
                clip, config, args.features, profile,
                speed=speed, realtime=not args.no_realtime,
                safety_side=safety_side, safety_quality=safety_quality,
                real_transport=real_transport,
                event_sensitivities=sensitivities,
                request_budget=budget,
            )
            runs.append(report)
            summary = report["summary"]
            print(
                f"{clip.clip_id} x{speed:g}: {summary['detected']}/{summary['events']}"
                f" detected, misses={summary['miss_reasons']},"
                f" false_alarms={summary['false_alarms']},"
                f" requests={summary['request_outcomes']},"
                f" delay_p50={summary['first_alarm_delay_p50']}",
                flush=True,
            )
    output = {
        "tool": "replay_stress",
        "features": list(args.features),
        "analysis_service": "real" if args.real else "simulated",
        "simulation": None if args.real else asdict(profile),
        "event_sensitivities": sensitivities,
        "paid_requests": budget.used if budget is not None else 0,
        "settings": {
            "event_fps": config.branch_sampling.safety.fps
            if config.branch_sampling else config.video.fps,
            "window_size": config.branch_sampling.safety.size_frames
            if config.branch_sampling else config.sliding_window.size_frames,
            "safety_side": safety_side,
            "safety_quality": safety_quality,
            "model_canvas": list(canvas) if canvas else None,
            "concurrency": config.runtime.max_concurrent_requests,
        },
        "runs": runs,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
