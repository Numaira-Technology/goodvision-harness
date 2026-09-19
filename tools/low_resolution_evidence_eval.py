"""Local synthetic replay evaluation of event continuity and H.264 evidence.

No model or network is used. Deterministic timestamp-based judgments exercise
the real file decoder, StreamingSession, person registry, archive, and encoder.
This measures pipeline behavior and playable evidence, NOT recognition accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from goodvision_harness.config import (
    DisplaySettings,
    FaceSettings,
    HarnessConfig,
    InferenceSettings,
    RichLabelSettings,
    RuntimeSettings,
    ServerSettings,
    ServiceSettings,
    SlidingWindowSettings,
    StreamSettings,
    VideoSettings,
)
from goodvision_harness.contracts import AnalysisResult, NormalizedBBox
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundingResponse,
    ModelActor,
    ModelObject,
)
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.sliding_window import FrameWindow
from goodvision_harness.video_to_stream import LiveFrame, VideoFileStream, probe_video

SIZES = ((64, 36), (96, 54), (160, 90), (320, 180), (161, 91))
SOURCE_FPS = (1, 5)
SCENARIOS = ("bridge", "split", "short", "gap")
LIMITATION = (
    "Deterministic synthetic judgments validate only temporal grouping, capture, "
    "storage, and playable video. These results do not measure model accuracy, "
    "face recognition, or smoking recognition at these resolutions."
)


@dataclass(frozen=True, slots=True)
class Variant:
    width: int
    height: int
    source_fps: int
    scenario: str
    buffer_max_bytes: int = 64 * 1024 * 1024

    @property
    def name(self) -> str:
        return f"{self.width}x{self.height}-{self.source_fps}fps-{self.scenario}"

    @property
    def duration(self) -> float:
        return {
            "bridge": 9.0,
            "split": 11.0,
            "short": 0.6,
            "gap": 7.0,
            "cache_pressure": 9.0,
        }[self.scenario]


def synthetic_frame(width: int, height: int, frame_index: int) -> np.ndarray:
    """Bright textured moving content that exposes black/frozen/wrong-size clips."""
    y, x = np.indices((height, width))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 40 + ((x * 3 + frame_index * 17) % 170)
    image[:, :, 1] = 55 + ((y * 7 + frame_index * 11) % 150)
    image[:, :, 2] = np.where((x // 5 + y // 5) % 2, 220, 65)
    side = max(4, min(width, height) // 5)
    left = (frame_index * 3) % max(1, width - side)
    cv2.rectangle(
        image,
        (left, height // 3),
        (left + side, height // 3 + side),
        (245, 245, 245),
        -1,
    )
    cv2.rectangle(
        image,
        (width // 5, height // 7),
        (width * 4 // 5, height * 6 // 7),
        (30, 230, 180),
        1,
    )
    return image


def create_source(
    path: Path, variant: Variant, ffmpeg_path: str = "ffmpeg"
) -> dict[str, Any]:
    """FFV1 preserves odd source dimensions that H.264 yuv420p cannot represent."""
    frame_count = max(1, math.ceil(variant.duration * variant.source_fps))
    payload = b"".join(
        synthetic_frame(variant.width, variant.height, index).tobytes()
        for index in range(frame_count)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{variant.width}x{variant.height}",
            "-framerate",
            str(variant.source_fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-threads",
            "1",
            "-pix_fmt",
            "bgr0",
            str(path),
        ],
        input=payload,
        capture_output=True,
        timeout=20,
        check=False,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    if completed.returncode:
        raise RuntimeError("synthetic_source_encoding_failed")
    info = probe_video(path)
    if (info.width, info.height) != (variant.width, variant.height):
        raise AssertionError("lossless source dimensions changed")
    return {
        "path": str(path),
        "width": info.width,
        "height": info.height,
        "fps": info.source_fps,
        "frame_count": info.frame_count,
        "duration_seconds": info.duration_seconds,
    }


def positive_at(timestamp: float, scenario: str) -> bool:
    if scenario in {"bridge", "cache_pressure"}:
        return 1 <= timestamp < 4 or 5 <= timestamp < 8
    if scenario == "split":
        # The negative evidence spans source t=4 through t=6 inclusive.
        return 1 <= timestamp < 4 or 6 + 1e-6 < timestamp < 10
    if scenario == "short":
        return True
    if scenario == "gap":
        return 1 <= timestamp < 6
    raise ValueError(f"unknown scenario: {scenario}")


class DeterministicClient:
    event_labels = ("smoking",)

    def __init__(self, settings: InferenceSettings, scenario: str) -> None:
        self.settings = settings
        self.scenario = scenario

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        positive = positive_at(window.end_timestamp, self.scenario)
        return GroundingResponse(
            window=window,
            classification=GroundedClassification(
                detected=positive,
                confidence=1.0,
                description="Synthetic positive fixture"
                if positive
                else "Synthetic negative fixture",
                objects=(
                    ModelObject(
                        "cigarette",
                        NormalizedBBox(0.4, 0.4, 0.48, 0.5),
                        confidence=1.0,
                        actor_ids=("synthetic-actor",),
                    ),
                )
                if positive
                else (),
                actors=(
                    ModelActor("synthetic-actor", NormalizedBBox(0.2, 0.1, 0.8, 0.95)),
                ),
            ),
            latency_seconds=0.0,
        )


class GappedFileStream(VideoFileStream):
    """A real decoded file with an explicit simulated transport outage."""

    def frames(self) -> Iterator[LiveFrame]:
        for frame in super().frames():
            if 1.4 - 1e-6 <= frame.timestamp_seconds < 3.8 - 1e-6:
                continue
            yield frame


def make_config(output: Path, variant: Variant, ffmpeg_path: str) -> HarnessConfig:
    # Construct settings directly so this local evaluation never reads .env,
    # credentials, enrolled faces, or the user's deployment configuration.
    inference = InferenceSettings("deterministic-fixture", "smoking", "", "")
    return HarnessConfig(
        path=output / "synthetic-config",
        default_video=None,
        stream=StreamSettings(),
        video=VideoSettings(fps=variant.source_fps, realtime=False),
        display=DisplaySettings(fps=5),
        rich_labeling=RichLabelSettings(enabled=False),
        sliding_window=SlidingWindowSettings(size_frames=1, stride_frames=1),
        aiclub=ModelAPISettings(
            endpoint="http://127.0.0.1/unused", api_key_env="UNUSED_EVAL_KEY"
        ),
        inference=inference,
        memory=None,
        face=FaceSettings(),
        runtime=RuntimeSettings(max_concurrent_requests=1),
        server=ServerSettings(),
        service=ServiceSettings(),
        events=EventArchiveSettings(
            directory=output / "archive",
            buffer_max_bytes=variant.buffer_max_bytes,
            ffmpeg_path=ffmpeg_path,
        ),
    )


def decode_evidence(
    path: Path, available: dict[str, Any], variant: Variant
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    decoded = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        advertised_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded.append(image)
    finally:
        capture.release()
    if not decoded:
        raise AssertionError("evidence has no decodable frames")
    seek_capture = cv2.VideoCapture(str(path))
    try:
        seek_ok = seek_capture.set(cv2.CAP_PROP_POS_FRAMES, len(decoded) - 1)
        seek_ok = bool(seek_ok and seek_capture.read()[0])
    finally:
        seek_capture.release()
    scale = min(1.0, 720 / max(variant.width, variant.height))
    expected_size = (
        max(2, round(variant.width * scale) // 2 * 2),
        max(2, round(variant.height * scale) // 2 * 2),
    )
    maes = []
    for index, frame in enumerate(decoded):
        timestamp = float(available["start_seconds"]) + index / fps
        source_index = math.floor(timestamp * variant.source_fps + 1e-5)
        reference = synthetic_frame(variant.width, variant.height, source_index)
        if (reference.shape[1], reference.shape[0]) != expected_size:
            reference = cv2.resize(
                reference, expected_size, interpolation=cv2.INTER_AREA
            )
        maes.append(float(np.abs(frame.astype(np.float32) - reference).mean()))
    means = [float(frame.mean()) for frame in decoded]
    deviations = [float(frame.std()) for frame in decoded]
    return {
        "path": str(path),
        "width": decoded[0].shape[1],
        "height": decoded[0].shape[0],
        "fps": fps,
        "decoded_frames": len(decoded),
        "advertised_frames": advertised_count,
        "duration_seconds": len(decoded) / fps,
        "seek_last_frame_ok": seek_ok,
        "all_dimensions_correct": all(
            (frame.shape[1], frame.shape[0]) == expected_size for frame in decoded
        ),
        "min_pixel_mean": round(min(means), 3),
        "min_pixel_std": round(min(deviations), 3),
        "max_source_pixel_mae": round(max(maes), 3),
        "distinct_decoded_frames": len({frame.tobytes() for frame in decoded}),
        "partial": available.get("partial"),
        "partial_reason": available.get("partial_reason"),
    }


def evaluate_variant(
    output: Path,
    variant: Variant,
    *,
    ffmpeg_path: str = "ffmpeg",
    shared_archive: EventArchive | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    source_path = output / "source.mkv"
    source = create_source(source_path, variant, ffmpeg_path)
    config = make_config(output, variant, ffmpeg_path)
    archive = shared_archive or EventArchive(config.events)
    observed: list[AnalysisResult] = []
    endings = []
    memory_samples = []
    callbacks = SessionCallbacks(
        on_analysis_result=observed.append,
        on_display_frame=lambda *_: memory_samples.append(archive.status()),
        on_end=lambda reason, error: endings.append((reason, error)),
    )
    client = DeterministicClient(config.inference, variant.scenario)
    if variant.scenario == "gap":
        session = StreamingSession(
            config,
            client,
            callbacks,
            mode="file",
            video_stream=GappedFileStream(source_path, display_fps=5, realtime=False),
            source_id="synthetic-low-resolution",
            archive=archive,
        )
    else:
        session = StreamingSession.for_file(
            config,
            client,
            callbacks,
            video_path=source_path,
            realtime=False,
            source_id="synthetic-low-resolution",
            archive=archive,
        )
    started = time.monotonic()
    try:
        session.start()
        session.join(20)
        if session.running:
            session.stop()
            raise TimeoutError("synthetic replay did not finish")
        deadline = time.monotonic() + 20
        while True:
            if not archive.flush(5):
                raise TimeoutError("archive did not drain")
            events = archive.list_events(
                limit=500,
                stream_session_id=session.stream_session_id,
            )["events"]
            if events and all(
                event["evidence"]["status"] not in {"collecting", "queued", "encoding"}
                for event in events
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("evidence did not finish")
            time.sleep(0.01)
        memory_samples.append(archive.status())
        clips = []
        for event in events:
            available = event["evidence"].get("available")
            if available:
                path = archive.video_path(event["event_id"], available["revision"])
                assert path is not None
                clips.append(decode_evidence(path, available, variant))
        checks = {
            "session_completed": endings == [("eof", None)],
            "event_count": len(events) == (2 if variant.scenario == "split" else 1),
            "every_event_has_video": len(clips) == len(events),
            "decode_count_matches_header": all(
                c["decoded_frames"] == c["advertised_frames"] for c in clips
            ),
            "five_fps": all(abs(c["fps"] - 5) < 1e-6 for c in clips),
            "dimensions_preserved_without_upscaling": all(
                c["all_dimensions_correct"] for c in clips
            ),
            "content_visible": all(
                c["min_pixel_mean"] > 20 and c["min_pixel_std"] > 10 for c in clips
            ),
            "content_matches_source": all(
                c["max_source_pixel_mae"] < 35 for c in clips
            ),
            "seekable": all(c["seek_last_frame_ok"] for c in clips),
            "cache_within_limit": all(
                sample.get("buffer_bytes", 0) <= variant.buffer_max_bytes
                for sample in memory_samples
            ),
        }
        if variant.scenario in {"bridge", "split", "cache_pressure"}:
            checks["full_three_second_evidence"] = all(
                not clip["partial"] and abs(clip["duration_seconds"] - 3) <= 0.201
                for clip in clips
            )
        elif variant.scenario in {"short", "gap"}:
            checks["partial_is_explicit"] = all(clip["partial"] for clip in clips)
            checks["no_fake_full_clip"] = all(
                clip["duration_seconds"] < 3 for clip in clips
            )
        if variant.scenario == "cache_pressure":
            # Only the hot source ring is under this tiny RAM-cache budget.
            # Ingress now owns its queued JPEGs in SQLite, not an in-memory FIFO;
            # a short cache must not make a durably admitted clip disappear.
            checks["no_source_admission_loss"] = all(
                sample.get("dropped", 0) == 0
                and sample.get("storage_failures", 0) == 0
                for sample in memory_samples
            )
            checks["every_evidence_ready"] = all(
                event["evidence"]["status"] == "ready" for event in events
            )
        return {
            "variant": variant.name,
            "scenario": variant.scenario,
            "source": source,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "observations": len(observed),
            "positive_observations": sum(r.judgments[0].detected for r in observed),
            "event_count": len(events),
            "video_count": len(clips),
            "clips": clips,
            "event_summaries": [
                {
                    key: event[key]
                    for key in (
                        "event_id",
                        "subject_id",
                        "start_seconds",
                        "last_detected_seconds",
                        "end_reason",
                        "evidence",
                    )
                }
                for event in events
            ],
            "buffer_max_bytes": variant.buffer_max_bytes,
            "cache_measurement_scope": "in_memory_source_ring_only",
            "max_observed_cache_bytes": max(
                (s.get("buffer_bytes", 0) for s in memory_samples),
                default=0,
            ),
            # This is queued source payload size, not SQLite physical usage or
            # process RSS. Disk and encoding allocations have separate budgets.
            "max_observed_durable_ingress_payload_bytes": max(
                (s.get("ingress_bytes", 0) for s in memory_samples),
                default=0,
            ),
            "max_observed_storage_bytes": max(
                (s.get("storage_bytes", 0) for s in memory_samples),
                default=0,
            ),
            "archive_status": archive.status(),
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        session.stop()
        session.join(2)
        if shared_archive is None:
            archive.close()


def write_report(output: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "test_scope": LIMITATION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": len(records),
        "passed": sum(record.get("passed", False) for record in records),
        "failed": sum(not record.get("passed", False) for record in records),
        "event_count": sum(record.get("event_count", 0) for record in records),
        "video_count": sum(record.get("video_count", 0) for record in records),
        "records": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Low-resolution event-evidence replay",
        "",
        LIMITATION,
        "",
        f"Variants: {report['variants']}; passed: {report['passed']}; "
        f"failed: {report['failed']}; "
        f"events: {report['event_count']}; playable videos: {report['video_count']}.",
        "",
        "| Variant | Events | Videos | Duration(s) | Size(s) | Result |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for record in records:
        clips = record.get("clips", [])
        durations = (
            ", ".join(f"{clip['duration_seconds']:.1f}" for clip in clips) or "—"
        )
        sizes = ", ".join(f"{clip['width']}x{clip['height']}" for clip in clips) or "—"
        failed = [key for key, value in record.get("checks", {}).items() if not value]
        result = (
            "PASS"
            if record.get("passed")
            else "FAIL: " + ", ".join(failed or [record.get("error", "unknown")])
        )
        lines.append(
            f"| {record['variant']} | {record.get('event_count', 0)} | "
            f"{record.get('video_count', 0)} | {durations} | {sizes} | {result} |"
        )
    lines.extend(
        [
            "",
            "At 1 source fps, repeated output frames preserve real elapsed time "
            "at 5 fps. This adds no new visual information. Short input and "
            "simulated source outages must remain explicitly partial.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--quick", action="store_true", help="Run 64x36 only plus cache pressure."
    )
    args = parser.parse_args(argv)
    if shutil.which(args.ffmpeg) is None:
        parser.error("a local FFmpeg executable is required")
    output = args.output or Path("tmp") / datetime.now().strftime(
        "low-resolution-evidence-%Y%m%d-%H%M%S"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sizes = SIZES[:1] if args.quick else SIZES
    variants = [
        Variant(width, height, fps, scenario)
        for width, height in sizes
        for fps in SOURCE_FPS
        for scenario in SCENARIOS
    ]
    variants.extend(
        Variant(96, 54, fps, "cache_pressure", 8 * 1024) for fps in SOURCE_FPS
    )
    if not args.quick:
        variants.extend(
            [
                Variant(1920, 1080, 5, "short"),
                Variant(17, 9, 1, "bridge"),
                Variant(17, 9, 5, "bridge"),
            ]
        )
    records = []
    for variant in variants:
        try:
            records.append(
                evaluate_variant(
                    output / variant.name, variant, ffmpeg_path=args.ffmpeg
                )
            )
        except Exception as exc:
            records.append(
                {
                    "variant": variant.name,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report = write_report(output, records)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "variants",
                    "passed",
                    "failed",
                    "event_count",
                    "video_count",
                )
            },
            ensure_ascii=False,
        )
    )
    print(str(output / "report.md"))
    return int(report["failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())
