"""Benchmark the real rich-label worker against a local video.

This runner intentionally uses the same raw-JPEG HTTP client, JPEG encoder,
and rich overlay renderer as the Harness.  It writes one record per decoded
source frame, a compact aggregate metrics document, and a full-cadence
annotated MP4.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import cv2
import httpx

# Keep the checked-out runner usable before an editable install.
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from goodvision_harness.overlay import draw_rich_overlay  # noqa: E402
from goodvision_harness.rich_labeling import (  # noqa: E402
    RichLabelHTTPClient,
    RichLabelObject,
    RichLabelSnapshot,
)
from goodvision_harness.video_to_stream import encode_jpeg, probe_video  # noqa: E402

METRICS_FILENAME = "metrics.json"
FRAMES_FILENAME = "frames.jsonl"
VIDEO_FILENAME = "annotated.mp4"


@dataclass(frozen=True, slots=True)
class OutputPaths:
    metrics: Path
    frames: Path
    video: Path


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    status: str
    latency_ms: float
    snapshot: RichLabelSnapshot | None = None
    http_status: int | None = None
    error_type: str | None = None


@dataclass(slots=True)
class TrackMetric:
    label: str
    duration_frames: int = 0
    detector_observations: int = 0
    first_frame_sequence: int | None = None
    last_frame_sequence: int | None = None
    first_timestamp_seconds: float | None = None
    last_timestamp_seconds: float | None = None
    longest_contiguous_frames: int = 0
    _current_contiguous_frames: int = 0
    _last_rendered_frame: int | None = None

    def observe_detection(self, label: str) -> None:
        self.label = label
        self.detector_observations += 1

    def observe_rendered(
        self,
        *,
        label: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> None:
        self.label = label
        self.duration_frames += 1
        if self.first_frame_sequence is None:
            self.first_frame_sequence = frame_sequence
            self.first_timestamp_seconds = timestamp_seconds
        if (
            self._last_rendered_frame is not None
            and frame_sequence == self._last_rendered_frame + 1
        ):
            self._current_contiguous_frames += 1
        else:
            self._current_contiguous_frames = 1
        self.longest_contiguous_frames = max(
            self.longest_contiguous_frames,
            self._current_contiguous_frames,
        )
        self._last_rendered_frame = frame_sequence
        self.last_frame_sequence = frame_sequence
        self.last_timestamp_seconds = timestamp_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "duration_frames": self.duration_frames,
            "longest_contiguous_frames": self.longest_contiguous_frames,
            "detector_observations": self.detector_observations,
            "first_frame_sequence": self.first_frame_sequence,
            "last_frame_sequence": self.last_frame_sequence,
            "first_timestamp_seconds": self.first_timestamp_seconds,
            "last_timestamp_seconds": self.last_timestamp_seconds,
        }


class TrackMetrics:
    """Accumulate detector observations and rendered-frame track durations."""

    def __init__(self) -> None:
        self._tracks: dict[str, TrackMetric] = {}

    def observe_detection(self, objects: Sequence[RichLabelObject]) -> None:
        for item in objects:
            metric = self._tracks.setdefault(
                item.track_id,
                TrackMetric(label=item.label),
            )
            metric.observe_detection(item.label)

    def observe_rendered_frame(
        self,
        frame_sequence: int,
        timestamp_seconds: float,
        objects: Sequence[RichLabelObject],
    ) -> None:
        for item in objects:
            metric = self._tracks.setdefault(
                item.track_id,
                TrackMetric(label=item.label),
            )
            metric.observe_rendered(
                label=item.label,
                frame_sequence=frame_sequence,
                timestamp_seconds=timestamp_seconds,
            )

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {
            track_id: self._tracks[track_id].as_dict()
            for track_id in sorted(self._tracks)
        }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile (the common R-7 method)."""

    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    rendered = [float(value) for value in values]
    if any(not math.isfinite(value) for value in rendered):
        raise ValueError("percentile values must be finite")
    if not rendered:
        return None
    ordered = sorted(rendered)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * alpha)


def summarize_latencies(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize millisecond samples without requiring NumPy."""

    rendered = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in rendered):
        raise ValueError("latencies must be finite and non-negative")
    if not rendered:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(rendered),
        "min": _round_metric(min(rendered)),
        "mean": _round_metric(sum(rendered) / len(rendered)),
        "p50": _round_metric(percentile(rendered, 0.50)),
        "p95": _round_metric(percentile(rendered, 0.95)),
        "p99": _round_metric(percentile(rendered, 0.99)),
        "max": _round_metric(max(rendered)),
    }


def advance_sample_deadline(
    timestamp_seconds: float,
    next_timestamp_seconds: float,
    sample_fps: float,
) -> tuple[bool, float]:
    """Decide whether this source frame crosses the next sample deadline."""

    values = (timestamp_seconds, next_timestamp_seconds, sample_fps)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sample timing values must be finite")
    if timestamp_seconds < 0 or next_timestamp_seconds < 0 or sample_fps <= 0:
        raise ValueError("sample timing values must be non-negative and fps positive")
    if timestamp_seconds + 1e-9 < next_timestamp_seconds:
        return False, next_timestamp_seconds
    period = 1.0 / sample_fps
    deadlines_crossed = max(
        1,
        math.floor((timestamp_seconds - next_timestamp_seconds) / period) + 1,
    )
    return True, next_timestamp_seconds + (deadlines_crossed * period)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real rich-label HTTP worker over a local video and write "
            "metrics.json, frames.jsonl, and annotated.mp4."
        )
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        type=_http_url,
        default="http://127.0.0.1:8091/v1/labels",
    )
    parser.add_argument("--sample-fps", type=_positive_float, default=7.5)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--max-image-side", type=_positive_integer, default=720)
    parser.add_argument("--jpeg-quality", type=_jpeg_quality, default=70)
    parser.add_argument("--max-frames", type=_positive_integer)
    parser.add_argument("--session-id", type=_non_empty_text)
    parser.add_argument("--fourcc", type=_fourcc, default="mp4v")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace benchmark files already present in --output-dir",
    )
    return parser


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(arguments.input_video).resolve()
    if not input_path.is_file():
        raise ValueError(f"input video does not exist: {input_path}")
    output_paths = _prepare_outputs(
        input_path,
        Path(arguments.output_dir),
        overwrite=bool(arguments.overwrite),
    )
    video_info = probe_video(input_path)
    session_id = arguments.session_id or f"benchmark-{uuid4().hex}"
    started_at = _utc_now()
    wall_started = time.perf_counter()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {input_path}")
    writer = cv2.VideoWriter(
        str(output_paths.video),
        cv2.VideoWriter_fourcc(*arguments.fourcc),
        video_info.source_fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("OpenCV could not open the annotated MP4 writer")

    client = RichLabelHTTPClient(
        arguments.endpoint,
        timeout_seconds=arguments.timeout_seconds,
    )
    status_counts: Counter[str] = Counter()
    latencies: list[float] = []
    successful_latencies: list[float] = []
    successful_object_counts: list[int] = []
    labels: Counter[str] = Counter()
    track_metrics = TrackMetrics()
    current_snapshot: RichLabelSnapshot | None = None
    processed_frames = 0
    next_sample_timestamp = 0.0

    try:
        with output_paths.frames.open("w", encoding="utf-8", newline="\n") as log:
            while (
                arguments.max_frames is None
                or processed_frames < arguments.max_frames
            ):
                ok, frame = capture.read()
                if not ok:
                    break
                frame_sequence = processed_frames
                timestamp_seconds = frame_sequence / video_info.source_fps
                should_request, next_sample_timestamp = advance_sample_deadline(
                    timestamp_seconds,
                    next_sample_timestamp,
                    arguments.sample_fps,
                )
                outcome: RequestOutcome | None = None
                if should_request:
                    jpeg, _width, _height = encode_jpeg(
                        frame,
                        max_image_side=arguments.max_image_side,
                        quality=arguments.jpeg_quality,
                    )
                    outcome = _request_frame(
                        client,
                        jpeg,
                        stream_session_id=session_id,
                        frame_sequence=frame_sequence,
                        timestamp_seconds=timestamp_seconds,
                    )
                    status_counts[outcome.status] += 1
                    latencies.append(outcome.latency_ms)
                    if outcome.status == "success":
                        assert outcome.snapshot is not None
                        current_snapshot = outcome.snapshot
                        successful_latencies.append(outcome.latency_ms)
                        count = len(current_snapshot.objects)
                        successful_object_counts.append(count)
                        labels.update(item.label for item in current_snapshot.objects)
                        track_metrics.observe_detection(current_snapshot.objects)

                objects = (
                    current_snapshot.objects
                    if current_snapshot is not None
                    else ()
                )
                track_metrics.observe_rendered_frame(
                    frame_sequence,
                    timestamp_seconds,
                    objects,
                )
                rendered = draw_rich_overlay(frame, objects)
                writer.write(rendered)
                record = _frame_record(
                    frame_sequence=frame_sequence,
                    timestamp_seconds=timestamp_seconds,
                    outcome=outcome,
                    annotation_snapshot=current_snapshot,
                )
                log.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
                log.write("\n")
                processed_frames += 1
    finally:
        client.close()
        writer.release()
        capture.release()

    if processed_frames < 1:
        raise ValueError("the video yielded no decodable frames")
    elapsed_seconds = max(0.0, time.perf_counter() - wall_started)
    metrics = _build_metrics(
        input_path=input_path,
        output_paths=output_paths,
        endpoint=arguments.endpoint,
        session_id=session_id,
        started_at=started_at,
        completed_at=_utc_now(),
        elapsed_seconds=elapsed_seconds,
        video_info=video_info,
        processed_frames=processed_frames,
        settings=arguments,
        status_counts=status_counts,
        latencies=latencies,
        successful_latencies=successful_latencies,
        successful_object_counts=successful_object_counts,
        labels=labels,
        track_metrics=track_metrics,
    )
    output_paths.metrics.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _request_frame(
    client: RichLabelHTTPClient,
    jpeg: bytes,
    *,
    stream_session_id: str,
    frame_sequence: int,
    timestamp_seconds: float,
) -> RequestOutcome:
    started = time.perf_counter()
    try:
        snapshot = client.infer(
            jpeg,
            stream_session_id=stream_session_id,
            frame_sequence=frame_sequence,
            timestamp_seconds=timestamp_seconds,
        )
    except httpx.HTTPStatusError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        status_code = exc.response.status_code
        return RequestOutcome(
            status="http_429" if status_code == 429 else "failure",
            latency_ms=latency_ms,
            http_status=status_code,
            error_type=type(exc).__name__,
        )
    except Exception as exc:
        return RequestOutcome(
            status="failure",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_type=type(exc).__name__,
        )
    return RequestOutcome(
        status="success",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        snapshot=snapshot,
        http_status=200,
    )


def _frame_record(
    *,
    frame_sequence: int,
    timestamp_seconds: float,
    outcome: RequestOutcome | None,
    annotation_snapshot: RichLabelSnapshot | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "attempted": outcome is not None,
        "status": "not_sampled" if outcome is None else outcome.status,
        "latency_ms": (
            None if outcome is None else _round_metric(outcome.latency_ms)
        ),
        "http_status": None if outcome is None else outcome.http_status,
        "error_type": None if outcome is None else outcome.error_type,
    }
    objects = (
        () if annotation_snapshot is None else annotation_snapshot.objects
    )
    exact = (
        outcome is not None
        and outcome.status == "success"
        and annotation_snapshot is outcome.snapshot
    )
    annotation_mode = "exact" if exact else ("held" if objects else "none")
    return {
        "type": "rich_label_benchmark_frame",
        "frame_sequence": frame_sequence,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "request": request,
        "annotation": {
            "mode": annotation_mode,
            "source_frame_sequence": (
                None
                if annotation_snapshot is None
                else annotation_snapshot.frame_sequence
            ),
            "object_count": len(objects),
            "objects": [item.as_dict() for item in objects],
        },
    }


def _build_metrics(
    *,
    input_path: Path,
    output_paths: OutputPaths,
    endpoint: str,
    session_id: str,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    video_info: Any,
    processed_frames: int,
    settings: argparse.Namespace,
    status_counts: Mapping[str, int],
    latencies: Sequence[float],
    successful_latencies: Sequence[float],
    successful_object_counts: Sequence[int],
    labels: Mapping[str, int],
    track_metrics: TrackMetrics,
) -> dict[str, Any]:
    attempted = sum(status_counts.values())
    succeeded = int(status_counts.get("success", 0))
    failed = int(status_counts.get("failure", 0))
    http_429 = int(status_counts.get("http_429", 0))
    media_seconds = processed_frames / video_info.source_fps
    object_total = sum(successful_object_counts)
    return {
        "type": "rich_label_benchmark_metrics",
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_wall_seconds": _round_metric(elapsed_seconds),
        "session_id": session_id,
        "worker_endpoint": _safe_endpoint(endpoint),
        "input": {
            "path": str(input_path),
            "reported_frame_count": video_info.frame_count,
            "processed_frame_count": processed_frames,
            "source_fps": _round_metric(video_info.source_fps),
            "processed_media_seconds": _round_metric(media_seconds),
            "width": video_info.width,
            "height": video_info.height,
        },
        "settings": {
            "sample_fps": settings.sample_fps,
            "timeout_seconds": settings.timeout_seconds,
            "max_image_side": settings.max_image_side,
            "jpeg_quality": settings.jpeg_quality,
            "max_frames": settings.max_frames,
            "fourcc": settings.fourcc,
        },
        "requests": {
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "http_429": http_429,
            "failure_total_including_429": failed + http_429,
            "success_rate": _safe_ratio(succeeded, attempted),
            "effective_fps": _safe_ratio(succeeded, elapsed_seconds),
            "attempted_fps": _safe_ratio(attempted, elapsed_seconds),
            "successful_requests_per_media_second": _safe_ratio(
                succeeded,
                media_seconds,
            ),
        },
        "request_latency_ms": summarize_latencies(latencies),
        "successful_request_latency_ms": summarize_latencies(
            successful_latencies
        ),
        "objects": {
            "total_across_successful_requests": object_total,
            "mean_per_successful_request": _safe_ratio(
                object_total,
                len(successful_object_counts),
            ),
            "min_per_successful_request": (
                min(successful_object_counts) if successful_object_counts else None
            ),
            "max_per_successful_request": (
                max(successful_object_counts) if successful_object_counts else None
            ),
            "by_label": dict(sorted(labels.items())),
        },
        "tracks": track_metrics.as_dict(),
        "outputs": {
            "metrics_json": str(output_paths.metrics.resolve()),
            "frames_jsonl": str(output_paths.frames.resolve()),
            "annotated_mp4": str(output_paths.video.resolve()),
        },
    }


def _prepare_outputs(
    input_path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> OutputPaths:
    resolved_dir = output_dir.resolve()
    paths = OutputPaths(
        metrics=resolved_dir / METRICS_FILENAME,
        frames=resolved_dir / FRAMES_FILENAME,
        video=resolved_dir / VIDEO_FILENAME,
    )
    if input_path.resolve() in {
        paths.metrics.resolve(),
        paths.frames.resolve(),
        paths.video.resolve(),
    }:
        raise ValueError("output files must not overwrite the input video")
    existing = [
        path
        for path in (paths.metrics, paths.frames, paths.video)
        if path.exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise ValueError(f"output files already exist ({names}); pass --overwrite")
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _safe_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return _round_metric(numerator / denominator)


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _positive_float(value: str) -> float:
    try:
        rendered = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(rendered) or rendered <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return rendered


def _positive_integer(value: str) -> int:
    try:
        rendered = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if rendered < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return rendered


def _jpeg_quality(value: str) -> int:
    rendered = _positive_integer(value)
    if rendered > 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return rendered


def _http_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise argparse.ArgumentTypeError("URL credentials are not supported")
    return value


def _non_empty_text(value: str) -> str:
    if not value.strip() or value != value.strip():
        raise argparse.ArgumentTypeError("must be non-empty without outer whitespace")
    return value


def _fourcc(value: str) -> str:
    if len(value) != 4 or not value.isascii() or not value.isalnum():
        raise argparse.ArgumentTypeError("must be four ASCII letters or digits")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        metrics = run_benchmark(arguments)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, ensure_ascii=True, sort_keys=True))
    return 0 if metrics["requests"]["succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
