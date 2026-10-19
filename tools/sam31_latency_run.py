"""Replay retained video at capture speed through the production SAM client.

JPEG transport preparation is inside the measured client, never pre-encoded.
This bounded diagnostic does not call an event VLM or enable a camera. It reports
live lag separately from EOF drain and saves exact-source tracking observations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from goodvision_harness.config import RichLabelSettings
from goodvision_harness.perception import (
    METHODS,
    create_tracking_client,
    create_tracking_coordinator,
)
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.video_tracking import VideoFrame
from goodvision_harness.video_tracking_session import (
    continuous_source_history_seconds,
)


def load_source(video, *, fps, max_seconds):
    """Explicit nearest-frame resampling, bounded by frames and original bytes."""
    if (
        type(fps) not in (int, float)
        or not math.isfinite(fps)
        or type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0.5 <= fps <= 15
        or not 0 < max_seconds <= 60
    ):
        raise ValueError("replay requires fps 0.5..15 and duration 0..60 seconds")
    capture = cv2.VideoCapture(str(Path(video).resolve()))
    frames, mapping = [], []
    total_bytes = 0
    try:
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if (
            not capture.isOpened()
            or not math.isfinite(native_fps)
            or native_fps < 0.5
            or not math.isfinite(total)
            or total < 1
        ):
            raise ValueError("recording has no usable source frame rate")
        total = int(total)
        fps = min(fps, native_fps)
        count = math.ceil(min(max_seconds, total / native_fps) * fps - 1e-8)
        if not 1 <= count <= 512:
            raise ValueError("replay requires 1..512 source samples")
        for sequence in range(count):
            timestamp = sequence / fps
            original = min(total - 1, math.floor(timestamp * native_fps + 0.5))
            capture.set(cv2.CAP_PROP_POS_FRAMES, original)
            ok, decoded = capture.read()
            if not ok:
                raise ValueError("recording frame unavailable")
            height, width = decoded.shape[:2]
            if max(height, width) > 1280:
                scale = 1280 / max(height, width)
                decoded = cv2.resize(
                    decoded,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                )
            ok, encoded = cv2.imencode(".jpg", decoded, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("source JPEG unavailable")
            jpeg = encoded.tobytes()
            total_bytes += len(jpeg)
            if total_bytes > 64 * 1024 * 1024:
                raise ValueError("replay source exceeds 64 MiB")
            frames.append(VideoFrame(sequence, timestamp, jpeg))
            mapping.append(
                {
                    "frame_sequence": sequence,
                    "timestamp_seconds": timestamp,
                    "original_frame_sequence": original,
                    "original_timestamp_seconds": original / native_fps,
                    "source_jpeg_file": f"source-frames/frame-{sequence:06d}.jpg",
                    "source_jpeg_bytes": len(jpeg),
                }
            )
        return frames, mapping, fps
    finally:
        capture.release()


class MeasuredClient:
    """Measure the selected production client without changing its protocol."""

    def __init__(self, client):
        self._wrapped = client
        self.measurements = []
        self.measurement_lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def append(
        self, frames, *, stream_session_id, is_last_batch=False, request_id=None,
        prepared=None, retry_transport=True,
    ):
        before = time.perf_counter()
        result = None
        error_type = None
        try:
            result = self._wrapped.append(
                frames,
                stream_session_id=stream_session_id,
                is_last_batch=is_last_batch,
                request_id=request_id,
                prepared=prepared,
                retry_transport=retry_transport,
            )
            return result
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            row = {
                "request_id": result.request_id if result else request_id,
                "succeeded": result is not None,
                "error_type": error_type,
                "source_sequences": [frame.frame_sequence for frame in frames],
                "source_jpeg_bytes": sum(len(frame.jpeg_bytes) for frame in frames),
                "elapsed_seconds": time.perf_counter() - before,
                "prepared_payload_seconds": (
                    prepared.payload.preparation_seconds
                    if prepared is not None else None
                ),
                "emitted_sequences": [frame.frame_sequence for frame in result.frames]
                if result
                else [],
                "pending_frame_count": result.pending_frame_count if result else None,
                "final": is_last_batch,
            }
            with self.measurement_lock:
                self.measurements.append(row)

    def measurement_snapshot(self):
        with self.measurement_lock:
            return [dict(row) for row in self.measurements]


def distribution(values):
    values = list(values)
    return {
        "samples": len(values),
        "peak": max(values) if values else None,
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def summarize_observations(rows, tracks, capture_count):
    # Capture and completion rows occur at configurable rates. Mixing them into
    # percentiles would bias comparisons toward the faster append configuration.
    clocks = [row for row in rows if row["kind"] == "clock" and row["during_capture"]]
    live = [
        row["tracking_lag_seconds"]
        for row in clocks
        if row["tracking_lag_seconds"] is not None
    ]
    sequences = [row["native"]["frame_sequence"] for row in tracks]
    if len(set(sequences)) != len(sequences) or len(sequences) > capture_count:
        raise ValueError("duplicate or excess source tracking observations")
    return {
        "live_tracking_lag_seconds": distribution(live),
        "live_lag_sampling": {
            "population": "periodic_clock_during_capture",
            "target_interval_seconds": 0.1,
            "samples_without_output": sum(
                row["tracking_lag_seconds"] is None for row in clocks
            ),
        },
        "tracking_coverage": {
            "processed_source_frames": len(sequences),
            "unprocessed_source_frames": capture_count - len(sequences),
            "fraction": len(sequences) / capture_count,
            "completed_during_capture": sum(row["during_capture"] for row in tracks),
            "completed_after_capture": sum(not row["during_capture"] for row in tracks),
            "all_capture_frames_processed": len(sequences) == capture_count,
        },
        "source_to_result_seconds": {
            "during_capture": distribution(
                row["source_to_result_seconds"]
                for row in tracks
                if row["during_capture"]
            ),
            "after_capture": distribution(
                row["source_to_result_seconds"]
                for row in tracks
                if not row["during_capture"]
            ),
        },
    }


def source_jpeg_bytes(output, source):
    """Read retained source values; legacy metadata alone cannot prove equality."""
    name = source.get("source_jpeg_file")
    if not isinstance(name, str) or not name:
        raise ValueError("saved source lacks retained JPEG values; migration is required")
    root = Path(output).resolve()
    relative = Path(name)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
        or path.stat().st_size > 64 * 1024 * 1024
    ):
        raise ValueError("invalid retained source JPEG path or size")
    with path.open("rb") as source_file:
        data = source_file.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("retained source JPEG exceeds byte budget")
    return data


def render_annotations(frames, mapping, tracks, *, output, fps):
    """Offline exact-source review; a missing result never reuses an old box."""
    output = Path(output)
    video_path, map_path = output / "annotated.mp4", output / "annotation-map.jsonl"
    if video_path.exists() or map_path.exists():
        raise FileExistsError("annotation output must be new")
    spec = importlib.util.spec_from_file_location(
        "sam31_latency_renderer", Path(__file__).with_name("sam31_session_run.py")
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    originals = {frame.frame_sequence: frame for frame in frames}
    available = {}
    for row in tracks:
        raw = RichLabelSnapshot.from_dict(row.get("native", row.get("raw")))
        canonical = RichLabelSnapshot.from_dict(row["canonical"])
        source = originals.get(raw.frame_sequence)
        if (
            source is None
            or source.timestamp_seconds != raw.timestamp_seconds
            or (canonical.frame_sequence, canonical.timestamp_seconds)
            != (raw.frame_sequence, raw.timestamp_seconds)
            or raw.frame_sequence in available
        ):
            raise ValueError("annotation result has no unique exact original source")
        available[raw.frame_sequence] = (canonical, row)
    writer = None
    records = []
    try:
        for index, (frame, source) in enumerate(zip(frames, mapping, strict=True)):
            if (source["frame_sequence"], source["timestamp_seconds"]) != (
                frame.frame_sequence,
                frame.timestamp_seconds,
            ) or frame.jpeg_bytes != source_jpeg_bytes(output, source):
                raise ValueError(
                    "annotation original differs from retained source JPEG values"
                )
            match = available.get(frame.frame_sequence)
            if match:
                snapshot, row = match
                epochs = {
                    item.attributes["native_track"]["epoch"]
                    for item in snapshot.objects
                }
                if len(epochs) > 1:
                    raise ValueError("annotation frame contains mixed native epochs")
                epoch = next(iter(epochs), row.get("epoch", "empty-observation"))
            else:
                snapshot = RichLabelSnapshot(
                    "unprocessed", frame.frame_sequence, frame.timestamp_seconds, ()
                )
                row, epoch = {}, "unprocessed"
            image = renderer.render_session_source_frame(
                frame,
                snapshot,
                update_number=index + 1,
                predictor_epoch=epoch,
                first_frame=index == 0,
            )
            if not match:
                image[:76] = 24
                for line, text in enumerate(
                    (
                        f"Source {frame.timestamp_seconds:.3f}s "
                        f"| frame {frame.frame_sequence}",
                        "NOT PROCESSED - no tracking result for this source frame",
                        "No boxes copied from earlier frames.",
                    )
                ):
                    cv2.putText(
                        image,
                        text,
                        (8, 20 + line * 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (80, 190, 255),
                        1,
                        cv2.LINE_AA,
                    )
            size = image.shape[1], image.shape[0]
            if writer is None:
                annotation_size = size
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                )
                if not writer.isOpened():
                    raise ValueError("annotation video writer unavailable")
            elif size != annotation_size:
                raise ValueError("annotation source dimensions changed")
            writer.write(image)
            records.append(
                {
                    "output_frame_sequence": index,
                    "output_timestamp_seconds": index / fps,
                    "source_frame_sequence": frame.frame_sequence,
                    "source_timestamp_seconds": frame.timestamp_seconds,
                    "source_jpeg_file": source["source_jpeg_file"],
                    "state": "observed" if match else "not_processed",
                    "subject_ids": [item.track_id for item in snapshot.objects],
                    "source_to_result_seconds": row.get("source_to_result_seconds"),
                }
            )
        map_path.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
    finally:
        if writer is not None:
            writer.release()
    return {
        "path": video_path.name,
        "source_mapping": map_path.name,
        "frames": len(records),
        "fps": fps,
        "not_processed_frames": sum(row["state"] == "not_processed" for row in records),
        "rendered_after_measurement": True,
        "presentation": "all_original_samples_with_exact_source_boxes",
    }


def render_saved_preview(output, *, video=None):
    """Render a saved run without contacting SAM or modifying its benchmark report."""
    output = Path(output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    mapping = [
        json.loads(line)
        for line in (output / "source-map.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    tracks = [
        json.loads(line)
        for line in (output / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    fps = report["capture_fps"]
    frames, _, actual_fps = load_source(
        video or report["source_video"],
        fps=fps,
        max_seconds=min(60, len(mapping) / fps),
    )
    if actual_fps != fps or len(frames) != len(mapping):
        raise ValueError("saved source sampling does not match recording")
    result = render_annotations(frames, mapping, tracks, output=output, fps=fps)
    (output / "annotation-report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _replay(args, frames, mapping, fps, client, output):
    runtime = SceneRuntime(
        uuid4().hex,
        "latency-recording",
        mode="active",
        tracking_mode="batch",
        native_memory_scope="session",
        tracking_history_seconds=continuous_source_history_seconds(fps),
    )
    runtime.automatic_temporal_inference = False
    registry = PersonRegistry(runtime.stream_session_id, runtime.source_id)
    lock = threading.RLock()
    rows, tracks, commits = [], [], []
    stop_monitor = threading.Event()
    capture_complete = False
    measurement_complete = False
    source_by_sequence = {frame.frame_sequence: frame for frame in frames}

    def sample(kind):
        state = runtime.snapshot()
        stats = coordinator.stats()
        row = {
            "kind": kind,
            "elapsed_seconds": time.perf_counter() - started,
            "during_capture": not capture_complete,
            "capture_timestamp_seconds": state["capture_timestamp_seconds"],
            "tracking_lag_seconds": state["tracking_lag_seconds"],
            "tracking_completed_timestamp_seconds": state[
                "tracking_completed_timestamp_seconds"
            ],
            "pending_frames": stats["pending_frames"],
            "pending_bytes": stats["pending_bytes"],
            "entities": len(state["scene"]["entities"]),
        }
        rows.append(row)
        return row

    def on_batch(batch):
        with lock:
            if measurement_complete:
                return
            for snapshot in batch.frames:
                source = source_by_sequence.get(snapshot.frame_sequence)
                if (
                    source is None
                    or source.timestamp_seconds != snapshot.timestamp_seconds
                ):
                    raise ValueError("native result does not match an original sample")
            canonical = []

            def canonicalize(snapshots):
                if not registry.begin_native_epoch(
                    batch.backend, batch.predictor_epoch
                ):
                    raise ValueError("native epoch rejected")
                for snapshot in snapshots:
                    registry.observe_snapshot(snapshot)
                    mapped = registry.snapshot_at(
                        snapshot.timestamp_seconds, snapshot.frame_sequence
                    )
                    if mapped is None or (
                        mapped.frame_sequence,
                        mapped.timestamp_seconds,
                    ) != (snapshot.frame_sequence, snapshot.timestamp_seconds):
                        raise ValueError("source mapping unavailable")
                    canonical.append(mapped)
                runtime.apply_subject_merges(registry.drain_subject_merges())
                return tuple(canonical)

            accepted = runtime.commit_native_tracking_batch(
                batch.frames,
                completed_timestamp_seconds=batch.completed_timestamp_seconds,
                completed_frame_sequence=batch.completed_frame_sequence,
                canonicalize=canonicalize,
            )
            row = sample("commit")
            row.update(
                accepted=accepted,
                epoch=batch.predictor_epoch,
                emitted_sequences=[frame.frame_sequence for frame in batch.frames],
            )
            commits.append(row)
            if not accepted:
                raise ValueError("scene rejected source evidence")
            for native, mapped in zip(batch.frames, canonical, strict=True):
                tracks.append(
                    {
                        "native": native.as_dict(),
                        "canonical": mapped.as_dict(),
                        "arrival_elapsed_seconds": row["elapsed_seconds"],
                        "source_to_result_seconds": row["elapsed_seconds"]
                        - native.timestamp_seconds,
                        "during_capture": row["during_capture"],
                        "epoch": batch.predictor_epoch,
                    }
                )

    coordinator = create_tracking_coordinator(
        client, replay_settings(args, fps), on_batch
    )
    coordinator.reset(runtime.stream_session_id)
    # Spool/coordinator initialization is preparation, not replay time. Starting
    # earlier makes slow hosts submit the first frames as an overdue burst.
    started = time.perf_counter()

    def monitor():
        while not stop_monitor.wait(0.1):
            with lock:
                if measurement_complete:
                    return
                sample("clock")

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    drained = False
    try:
        for frame, source_row in zip(frames, mapping, strict=True):
            while (
                remaining := started + frame.timestamp_seconds - time.perf_counter()
            ) > 0:
                time.sleep(min(remaining, 0.05))
            with lock:
                arrival = time.perf_counter() - started
                ingest_started = time.perf_counter()
                admitted = coordinator.submit(
                    frame.jpeg_bytes,
                    stream_session_id=runtime.stream_session_id,
                    frame_sequence=frame.frame_sequence,
                    timestamp_seconds=frame.timestamp_seconds,
                )
                runtime.observe_frame(
                    frame.jpeg_bytes,
                    frame.timestamp_seconds,
                    frame.frame_sequence,
                    retain_source=admitted,
                )
                source_row.update(
                    arrival_elapsed_seconds=arrival,
                    arrival_lateness_seconds=max(
                        0.0, arrival - frame.timestamp_seconds
                    ),
                    ingest_seconds=time.perf_counter() - ingest_started,
                    admitted_to_queue=admitted,
                )
                sample("capture")
        with lock:
            capture_complete = True
            capture_finished_at = time.perf_counter()
        coordinator.flush()
        deadline = time.perf_counter() + 90
        while time.perf_counter() < deadline:
            if coordinator.wait_until_idle(timeout_seconds=0.2):
                drained = True
                break
        with lock:
            measurement_complete = True
            measured_until = time.perf_counter()
            final = runtime.snapshot()
            stats = coordinator.stats()
            summary = summarize_observations(rows, tracks, len(frames))
            report = {
                "ok": drained and stats["failed"] == 0 and bool(commits),
                "source_video": str(Path(args.video).resolve()),
                "processing_method": getattr(args, "processing_method", None),
                "backend": client.backend,
                "capture_frames": len(frames),
                "capture_fps": fps,
                "elapsed_seconds": measured_until - started,
                "capture_done_wall_seconds": capture_finished_at - started,
                "eof_drain_seconds": measured_until - capture_finished_at,
                "first_output_seconds": commits[0]["elapsed_seconds"]
                if commits
                else None,
                **summary,
                "arrival_lateness_seconds": distribution(
                    [row["arrival_lateness_seconds"] for row in mapping]
                ),
                "ingest_seconds": distribution(
                    [row["ingest_seconds"] for row in mapping]
                ),
                "source_evidence_unchanged": all(
                    frame.jpeg_bytes == source_jpeg_bytes(output, row)
                    for frame, row in zip(frames, mapping, strict=True)
                ),
                "online_transport_preparation": True,
                "transport_max_side": args.transport_max_side,
                "transport_jpeg_quality": args.transport_jpeg_quality,
                "transport_codec": getattr(args, "transport_codec", "jpeg"),
                "transport_crf": getattr(args, "transport_crf", 23),
                "batch_frames": coordinator._batch_frames,
                "batch_seconds": coordinator._batch_seconds,
                "pending_capacity": stats["pending_capacity"],
                "startup_frames": client.startup_frames,
                "max_append_frames": client.max_append_frames,
                "requested_batch_frames": args.batch_frames,
                "requested_batch_seconds": args.batch_seconds,
                "finished_idle": drained,
                "coordinator": stats,
                "requests": client.measurement_snapshot(),
                "final_runtime": final,
                "tracking_rejections": final["tracking_rejections"],
                "epochs": sorted({row["epoch"] for row in commits}),
                "accuracy_measured": False,
                "event_model_enabled": False,
                "caveat": "EOF drain and zero final lag do not represent live latency. "
                "Native IDs are not ground truth person identities.",
            }
            if hasattr(client, "transport_stats"):
                report["transport"] = client.transport_stats()
            for name, records in (
                ("observations", rows),
                ("tracks", tracks),
                ("source-map", mapping),
                ("commits", commits),
            ):
                (output / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row, allow_nan=False) + "\n" for row in records),
                    encoding="utf-8",
                )
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
            return report
    finally:
        stop_monitor.set()
        thread.join(timeout=1)
        coordinator.close(timeout_seconds=2)
        runtime.close()


def replay_settings(args, fps):
    """Use the console's factory and dispatch policy, including route overrides."""
    return RichLabelSettings(
        enabled=True,
        backend="sam31_stream",
        endpoint=args.sam_endpoint,
        processing_method=getattr(args, "processing_method", None),
        max_fps=fps,
        history_size=512,
        video_batch_frames=args.batch_frames,
        video_batch_seconds=args.batch_seconds,
        video_timeout_seconds=75,
        video_transport_max_side=args.transport_max_side,
        video_transport_jpeg_quality=args.transport_jpeg_quality,
        video_transport_codec=getattr(args, "transport_codec", "jpeg"),
        video_transport_crf=getattr(args, "transport_crf", 23),
    )


def run(args):
    frames, mapping, fps = load_source(
        args.video, fps=args.fps, max_seconds=args.max_seconds
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-frames").mkdir()
    for frame, source in zip(frames, mapping, strict=True):
        (output / source["source_jpeg_file"]).write_bytes(frame.jpeg_bytes)
    client = None
    try:
        client = MeasuredClient(create_tracking_client(replay_settings(args, fps)))
        ready = getattr(client, "ready_for_new_session", client.gpu_ready)
        if not ready():
            raise RuntimeError(
                "selected GPU worker unavailable or session still occupied"
            )
        report = _replay(args, frames, mapping, fps, client, output)
        if getattr(args, "annotation", False):
            tracks = [
                json.loads(line)
                for line in (output / "tracks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            report["annotation"] = render_annotations(
                frames, mapping, tracks, output=output, fps=fps
            )
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
        return report
    except Exception as error:
        (output / "failure.json").write_text(
            json.dumps({"ok": False, "error_type": type(error).__name__}),
            encoding="utf-8",
        )
        raise
    finally:
        if client is not None:
            client.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--sam-endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--processing-method",
        choices=tuple(METHODS),
        help="use the console route profile (dispatch 1 frame / 0.1s, queue 4); "
        "omit for the legacy continuous SAM endpoint",
    )
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--max-seconds", type=float, default=48)
    parser.add_argument("--batch-frames", type=int, default=4)
    parser.add_argument("--batch-seconds", type=float, default=0.5)
    parser.add_argument("--transport-max-side", type=int)
    parser.add_argument("--transport-jpeg-quality", type=int)
    parser.add_argument("--transport-codec", choices=("jpeg", "h264"), default="jpeg")
    parser.add_argument("--transport-crf", type=int, default=23)
    parser.add_argument(
        "--annotation",
        action="store_true",
        help="render all sampled originals after the timed replay",
    )
    return parser


def main():
    args = argument_parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "processing_method",
                    "backend",
                    "batch_frames",
                    "batch_seconds",
                    "pending_capacity",
                    "capture_frames",
                    "elapsed_seconds",
                    "first_output_seconds",
                    "live_tracking_lag_seconds",
                    "epochs",
                    "tracking_rejections",
                    "coordinator",
                )
            }
        )
    )
    return int(not report["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
