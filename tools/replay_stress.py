"""Paced replay through SceneRuntime -> SceneInspectionWorker -> TemporalVLMInspector.

Uses recorded source-aligned SAM observations, or explicitly synthetic tracks
for offline plumbing tests. It does not run SAM, Face, the browser, or encoding
of final evidence. Reports must not be presented as full live-system acceptance.
Simulation replaces only HTTP; temporal prompts, image encoding, response
validation, scheduling and event lifecycle remain the production implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import uuid4

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.config import load_config  # noqa: E402
from goodvision_harness.contracts import NormalizedBBox  # noqa: E402
from goodvision_harness.event_sensitivity import (  # noqa: E402
    normalize_event_sensitivities,
)
from goodvision_harness.model_api import (  # noqa: E402
    ModelAPIError,
    environment_api_key,
)
from goodvision_harness.rich_labeling import (  # noqa: E402
    RichLabelObject,
    RichLabelSnapshot,
)
from goodvision_harness.scene_model import TemporalVLMInspector  # noqa: E402
from goodvision_harness.scene_runtime import SceneRuntime  # noqa: E402
from goodvision_harness.scene_worker import SceneInspectionWorker  # noqa: E402

LABELS = ("fight", "smoking", "fall", "weapon")


@dataclass(frozen=True)
class LabelledEvent:
    label: str
    start: float
    end: float

    def __post_init__(self):
        if self.label not in LABELS or not (
            math.isfinite(self.start)
            and math.isfinite(self.end)
            and 0 <= self.start <= self.end
        ):
            raise ValueError(
                "events need a supported label and finite 0 <= start <= end"
            )


@dataclass(frozen=True)
class Clip:
    clip_id: str
    path: Path
    events: tuple[LabelledEvent, ...] = ()
    tracks: Path | None = None
    ignore: tuple[tuple[float, float], ...] = ()


def load_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = []
    for row in data.get("clips", []):
        if "events" not in row:
            raise ValueError(
                "each clip needs reviewed events; use [] for a negative clip"
            )

        def resolve(value):
            candidate = Path(value)
            return candidate if candidate.is_absolute() else path.parent / candidate

        ignore = tuple(
            (float(v["start"]), float(v["end"])) for v in row.get("ignore", [])
        )
        if any(
            not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b)
            for a, b in ignore
        ):
            raise ValueError("ignore intervals must be finite and ordered")
        clips.append(
            Clip(
                row.get("id", Path(row["path"]).stem),
                resolve(row["path"]),
                tuple(
                    LabelledEvent(v["label"], float(v["start"]), float(v["end"]))
                    for v in row.get("events", [])
                ),
                resolve(row["tracks"]) if row.get("tracks") else None,
                ignore,
            )
        )
    if not clips or len({c.clip_id for c in clips}) != len(clips):
        raise ValueError("manifest needs clips with unique IDs")
    return clips


def parse_sensitivities(values, labels=LABELS):
    supplied = {}
    for value in values:
        parts = value.split("=")
        if len(parts) != 2 or parts[0] in supplied:
            raise ValueError("use unique --event-sensitivity label=standard|high")
        supplied[parts[0]] = parts[1]
    return normalize_event_sensitivities(supplied, labels)


class RequestBudget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("request budget must be a positive integer")
        self.limit, self.used = limit, 0
        self._lock = threading.Lock()

    def take(self):
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


@dataclass(frozen=True)
class SimulationProfile:
    latency_ms: float = 800
    slow_rate: float = 0
    slow_ms: float = 4000
    timeout_rate: float = 0
    auth_rate: float = 0
    parse_error_rate: float = 0
    seed: int = 7

    def __post_init__(self):
        rates = (
            self.slow_rate,
            self.timeout_rate,
            self.auth_rate,
            self.parse_error_rate,
        )
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rates):
            raise ValueError("fault rates must be finite probabilities")
        if sum(rates[1:]) > 1:
            raise ValueError("failure probabilities exceed one")
        if any(not math.isfinite(v) or v < 0 for v in (self.latency_ms, self.slow_ms)):
            raise ValueError("latencies must be finite and nonnegative")


def temporal_payload(messages):
    """Read timestamps from the real temporal JSON metadata, not legacy f0 text."""
    values = [
        json.loads(part["text"])
        for part in messages[-1]["content"]
        if part["type"] == "text"
    ]
    context = values[0]
    return context, [v for v in values[1:] if "frame_sequence" in v]


class ReplayInspector(TemporalVLMInspector):
    def __init__(
        self, *args, budget, calls, call_lock, profile, events, real, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.budget, self.calls, self.call_lock = budget, calls, call_lock
        self.profile, self.events, self.real = profile, events, real
        self._replay_job = None

    def inspect(self, job):
        self._replay_job = job
        try:
            return super().inspect(job)
        finally:
            self._replay_job = None

    def _ensure_transport(self):
        if self.real:
            return super()._ensure_transport()
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = httpx.AsyncClient(
                transport=httpx.MockTransport(self._simulate)
            )

    async def _simulate(self, request):
        payload = json.loads(request.content)
        context, frames = temporal_payload(payload["messages"])
        job = self._replay_job
        # Each endpoint contributes new evidence at most once. Context overlap
        # alone must not turn a past transient into a second positive vote.
        previous = job.evidence_after_frame_sequence
        novel = [
            f for f in frames if previous is None or f["frame_sequence"] > previous
        ]
        rng = random.Random(f"{self.profile.seed}:{job.frames[-1].frame_sequence}")
        latency = self.profile.latency_ms
        if rng.random() < self.profile.slow_rate:
            latency = max(latency, self.profile.slow_ms)
        await asyncio.sleep(latency / 1000)
        roll = rng.random()
        if roll < self.profile.auth_rate:
            return httpx.Response(401)
        if roll < self.profile.auth_rate + self.profile.timeout_rate:
            raise httpx.ReadTimeout("injected timeout", request=request)
        if roll < (
            self.profile.auth_rate
            + self.profile.timeout_rate
            + self.profile.parse_error_rate
        ):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "{"}}]}
            )
        subjects = context["allowed_subject_ids"]
        verdicts = [
            {
                "event_label": label,
                "subject_ids": subjects[:2] if label == "fight" else subjects[:1],
                "detected": any(
                    e.label == label
                    and e.start <= f["source_timestamp_seconds"] <= e.end
                    for e in self.events
                    for f in novel
                ),
                "confidence": 0.9,
            }
            for label in context["allowed_event_labels"]
        ]
        if payload.get("response_format", {}).get("type") == "json_object":
            verdicts = {"verdicts": verdicts}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(verdicts),
                        }
                    }
                ]
            },
        )

    async def _request(self, payload, headers, *, timing):
        if not self.budget.take():
            raise ModelAPIError("replay budget exhausted", code="budget_exhausted")
        _, frames = temporal_payload(payload["messages"])
        record = {
            "request_id": self._replay_job.request.request_id,
            "source_timestamps": [f["source_timestamp_seconds"] for f in frames],
            "started_monotonic": time.monotonic(),
        }
        with self.call_lock:
            self.calls.append(record)
        try:
            return await super()._request(payload, headers, timing=timing)
        finally:
            record["finished_monotonic"] = time.monotonic()


def read_tracks(path):
    if path is None:
        return {}
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            snapshot = RichLabelSnapshot.from_dict(json.loads(line))
            if snapshot.frame_sequence in records:
                raise ValueError("duplicate recorded source sequence")
            records[snapshot.frame_sequence] = snapshot
    return records


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_clip(
    clip,
    config,
    *,
    speed=1,
    real=False,
    synthetic_tracks=False,
    profile=None,
    budget=None,
    sensitivities=None,
    realtime=True,
    sample_fps=None,
    drain_seconds=60,
):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive and finite")
    if not math.isfinite(drain_seconds) or drain_seconds <= 0:
        raise ValueError("drain timeout must be positive and finite")
    if real and (synthetic_tracks or not realtime):
        raise ValueError("real replay requires recorded tracks and real-time pacing")
    if synthetic_tracks == (clip.tracks is not None):
        raise ValueError("choose recorded tracks or explicit synthetic tracks")
    profile = profile or SimulationProfile()
    budget = budget or RequestBudget(100)
    sample_fps = config.rich_labeling.max_fps if sample_fps is None else sample_fps
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError("sample fps must be in (0,30]")
    tracks = read_tracks(clip.tracks)
    if not synthetic_tracks and not tracks:
        raise ValueError("recorded tracks are empty")
    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise ValueError("video could not be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("video fps must be finite and positive")
    events = tuple(
        LabelledEvent(e.label, e.start / speed, e.end / speed) for e in clip.events
    )
    scene = SceneRuntime(
        uuid4().hex,
        "local-replay",
        event_labels=LABELS,
        event_sensitivities=sensitivities,
        tracking_mode="batch",
        native_memory_scope="session",
        temporal_observation_fps=config.runtime.scene_observation_fps,
        max_pending_inspections=8,
    )
    scene.automatic_temporal_inference = True
    calls, observations = [], []
    lock = threading.Lock()
    settings = replace(
        config.model_api,
        max_retries=0,
        timeout_seconds=config.runtime.scene_timeout_seconds,
    )
    key = environment_api_key(settings) if real else "simulation"
    if real and not key and settings.provider != "local":
        capture.release()
        raise ValueError("configured model credential is unavailable")

    def inspector():
        return ReplayInspector(
            settings,
            model=config.inference.model,
            api_key=key,
            request_parameters=config.inference.request_parameters,
            max_image_side=config.runtime.scene_max_image_side,
            jpeg_quality=config.runtime.scene_jpeg_quality,
            max_tokens=config.runtime.scene_max_tokens,
            budget=budget,
            calls=calls,
            call_lock=lock,
            profile=profile,
            events=events,
            real=real,
        )

    def on_update():
        state = scene.snapshot(include_history=False)
        with lock:
            observations.append(
                {"at": time.monotonic(), "events": state["scene"]["events"]}
            )

    worker = SceneInspectionWorker(
        scene,
        inspector(),
        max_in_flight=config.runtime.scene_max_in_flight,
        inspector_factory=inspector,
        on_update=on_update,
        poll_seconds=0.01,
    )
    started = time.monotonic()
    sequence = samples = missing_tracks = 0
    next_sample = 0
    max_arrival_lag = 0
    completed = False
    worker.start()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index, original_at = sequence, sequence / fps
            sequence += 1
            at = original_at / speed
            if at + 1e-9 < next_sample:
                continue
            next_sample = (math.floor(at * sample_fps + 1e-9) + 1) / sample_fps
            if realtime:
                time.sleep(max(0, started + at - time.monotonic()))
            max_arrival_lag = max(max_arrival_lag, time.monotonic() - started - at)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("source JPEG encoding failed")
            scene.observe_frame(encoded.tobytes(), at, index, force=True)
            samples += 1
            if synthetic_tracks:
                snapshot = RichLabelSnapshot(
                    scene.stream_session_id,
                    index,
                    at,
                    tuple(
                        RichLabelObject(
                            sid, "person", NormalizedBBox(x, 0.1, x + 0.3, 0.9)
                        )
                        for sid, x in (("synthetic-a", 0.05), ("synthetic-b", 0.6))
                    ),
                )
            else:
                snapshot = tracks.get(index)
                if snapshot is None:
                    missing_tracks += 1
                    continue  # Missing observations are never empty-person evidence.
                if not math.isclose(
                    snapshot.timestamp_seconds, original_at, abs_tol=1e-6
                ):
                    raise ValueError("recorded source timestamp does not match video")
                # Masks are not used for this replay; their embedded original
                # timestamps must not be misrepresented after timeline compression.
                snapshot = replace(
                    snapshot,
                    stream_session_id=scene.stream_session_id,
                    timestamp_seconds=at,
                    objects=tuple(
                        replace(
                            item,
                            attributes={
                                k: v
                                for k, v in item.attributes.items()
                                if k != "mask_contours"
                            },
                            last_observed_timestamp_seconds=(
                                item.last_observed_timestamp_seconds / speed
                                if item.last_observed_timestamp_seconds is not None
                                else None
                            ),
                        )
                        for item in snapshot.objects
                    ),
                )
            deadline = time.monotonic() + drain_seconds
            while not scene.observe_tracking_batch(
                (snapshot,),
                completed_timestamp_seconds=at,
                completed_frame_sequence=index,
            ):
                if scene.tracking_last_rejection != "inspection_capacity":
                    raise ValueError(
                        "recorded tracking batch rejected: "
                        + str(scene.tracking_last_rejection)
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("replay backpressure did not drain")
                worker.wake()
                time.sleep(0.01)
            worker.wake()
        scene.finish_inspection_sources()
        worker.wake()
        deadline = time.monotonic() + drain_seconds
        while time.monotonic() < deadline:
            state = scene.snapshot(include_history=False)
            worker_state = worker.stats()
            if (
                not state["inspections"]
                and not state["pending_tracking_intervals"]
                and worker_state["active_requests"] == 0
                and worker_state["pending_completions"] == 0
            ):
                completed = True
                break
            time.sleep(0.01)
        # The callback follows completion, so give its lock a final consistent view.
        worker.close(timeout_seconds=drain_seconds)
        report = {
            "clip_id": clip.clip_id,
            "speed": speed,
            "realtime": realtime,
            "entrypoint": "SceneRuntime/SceneInspectionWorker/TemporalVLMInspector",
            "perception_source": "synthetic_tracks"
            if synthetic_tracks
            else "recorded_tracks",
            "sam_validated": False,
            "face_validated": False,
            "display_validated": False,
            "model_mode": "real" if real else "oracle_http",
            "simulation_profile": None if real else asdict(profile),
            "accuracy_measured": False,
            "drain_complete": completed,
            "sampled_frames": samples,
            "missing_track_frames": missing_tracks,
            "max_arrival_lag_seconds": max_arrival_lag,
            "event_sensitivities": dict(scene.event_sensitivities),
            "source_sha256": _sha256(clip.path),
            "tracks_sha256": _sha256(clip.tracks) if clip.tracks else None,
            "configuration": {
                "sample_fps": sample_fps,
                "observation_fps": config.runtime.scene_observation_fps,
                "max_in_flight": config.runtime.scene_max_in_flight,
                "max_image_side": config.runtime.scene_max_image_side,
                "jpeg_quality": config.runtime.scene_jpeg_quality,
                "timeout_seconds": settings.timeout_seconds,
                "max_tokens": config.runtime.scene_max_tokens,
                "model": config.inference.model,
                "provider": settings.provider,
            },
            "diagnostics": scene.diagnostics.snapshot(LABELS),
            "worker": worker.stats(),
            "calls": calls,
            "final_scene": scene.snapshot(include_history=False),
            "request_budget_used": budget.used,
        }
        summaries = []
        for event in events:
            hits = [
                (item, alarm)
                for item in observations
                for alarm in item["events"]
                if alarm["event_label"] == event.label
                and alarm["last_positive_timestamp_seconds"] is not None
                and event.start <= alarm["last_positive_timestamp_seconds"] <= event.end
            ]
            confirmed = [item for item, alarm in hits if alarm["phase"] == "confirmed"]
            summaries.append(
                {
                    "label": event.label,
                    "start": event.start,
                    "end": event.end,
                    "first_alert_delay_seconds": (
                        max(0, hits[0][0]["at"] - started - event.start)
                        if hits and realtime
                        else None
                    ),
                    "confirmed_delay_seconds": (
                        max(0, confirmed[0]["at"] - started - event.start)
                        if confirmed and realtime
                        else None
                    ),
                    "observed": bool(hits),
                    "confirmed": bool(confirmed),
                }
            )
        report["events"] = summaries
        # Count unique positive event instances, not persistent redraws.
        false_alarms = set()
        for item in observations:
            for alarm in item["events"]:
                at = alarm["last_positive_timestamp_seconds"]
                if at is None:
                    continue
                if (
                    alarm["phase"] in {"suspected", "confirmed"}
                    and not any(
                        e.label == alarm["event_label"] and e.start <= at <= e.end
                        for e in events
                    )
                    and not any(a / speed <= at <= b / speed for a, b in clip.ignore)
                ):
                    false_alarms.add(alarm["event_id"])
        report["false_alarm_instances"] = len(false_alarms)
        return report
    finally:
        worker.close(timeout_seconds=1)
        scene.close()
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--synthetic-tracks", action="store_true")
    parser.add_argument(
        "--no-realtime", action="store_true", help="offline simulation only"
    )
    parser.add_argument("--speeds", default="1")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--event-sensitivity", action="append", default=[])
    for name, default in (
        ("latency-ms", 800),
        ("slow-ms", 4000),
        ("slow-rate", 0),
        ("timeout-rate", 0),
        ("auth-rate", 0),
        ("parse-error-rate", 0),
    ):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        clips = load_manifest(args.manifest)
        speeds = [float(v) for v in args.speeds.split(",")]
        if any(not math.isfinite(s) or s <= 0 for s in speeds):
            raise ValueError("speeds must be positive and finite")
        sensitivities = parse_sensitivities(args.event_sensitivity)
        budget = RequestBudget(args.max_requests)
        profile = SimulationProfile(
            **{k: getattr(args, k) for k in SimulationProfile.__dataclass_fields__}
        )
        if args.real and (args.synthetic_tracks or args.no_realtime):
            raise ValueError("real mode requires recorded tracks and paced replay")
        if args.real and any(
            (
                profile.slow_rate,
                profile.timeout_rate,
                profile.auth_rate,
                profile.parse_error_rate,
            )
        ):
            raise ValueError("fault injection is simulation-only")
        estimate = 0
        for clip in clips:
            if args.synthetic_tracks == (clip.tracks is not None):
                raise ValueError("each clip needs tracks, or choose --synthetic-tracks")
            if clip.tracks is not None and not read_tracks(clip.tracks):
                raise ValueError("recorded tracks are empty")
            cap = cv2.VideoCapture(str(clip.path))
            try:
                fps, frames = (
                    cap.get(cv2.CAP_PROP_FPS),
                    cap.get(cv2.CAP_PROP_FRAME_COUNT),
                )
                if not cap.isOpened() or fps <= 0 or frames <= 0:
                    raise ValueError("video duration is unavailable")
                estimate += sum(
                    math.ceil(
                        frames / fps / speed * config.runtime.scene_observation_fps
                    )
                    + 2
                    for speed in speeds
                )
            finally:
                cap.release()
        if args.real and estimate > budget.limit:
            raise ValueError(
                f"estimated {estimate} requests exceed budget {budget.limit}"
            )
    except ValueError as error:
        parser.error(str(error))
    # Reserve the output before any external request; never overwrite an old run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        report = {
            "schema_version": 1,
            "estimated_requests": estimate,
            "manifest_sha256": _sha256(args.manifest),
            "config_sha256": _sha256(args.config),
            "replay_tool_sha256": _sha256(Path(__file__)),
            "runs": [],
        }
        for clip in clips:
            for speed in speeds:
                report["runs"].append(
                    replay_clip(
                        clip,
                        config,
                        speed=speed,
                        real=args.real,
                        synthetic_tracks=args.synthetic_tracks,
                        profile=profile,
                        budget=budget,
                        sensitivities=sensitivities,
                        realtime=not args.no_realtime,
                        sample_fps=args.sample_fps,
                    )
                )
        report["request_budget"] = {"limit": budget.limit, "used": budget.used}
        json.dump(report, output, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
