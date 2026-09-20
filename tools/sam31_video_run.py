"""Validate SAM 3.1 native video batches against a local recording.

Memory and native IDs belong to one batch. This does not measure tracking accuracy
or establish continuous camera memory across batches. VLM calls require the explicit
--with-inference flag; the default only contacts the configured SAM video worker.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from goodvision_harness.config import load_config
from goodvision_harness.model_api import environment_api_key
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelSnapshot
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.video_tracking import (
    BACKEND,
    MAX_BATCH_JPEG_BYTES,
    VideoFrame,
    VideoTrackingHTTPClient,
)


def _validate(args) -> tuple[str, ...]:
    for value, maximum, label in (
        (args.fps, 15, "fps"),
        (args.max_seconds, 600, "max_seconds"),
        (args.timeout_seconds, 300, "timeout_seconds"),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= maximum
        ):
            raise ValueError(f"{label} must be within (0,{maximum}]")
    if type(args.batch_frames) is not int or not 1 <= args.batch_frames <= 16:
        raise ValueError("batch_frames must be within 1..16")
    if type(args.max_requests) is not int or not 0 <= args.max_requests <= 100:
        raise ValueError("max_requests must be within 0..100")
    labels = tuple(dict.fromkeys(args.events.split(",")))
    if not labels or not set(labels) <= {"fight", "smoking", "fall", "weapon"}:
        raise ValueError("events must be supported temporal safety labels")
    return labels


def _state(runtime: SceneRuntime) -> dict:
    state = runtime.snapshot()
    for entity in state["scene"]["entities"]:
        entity.pop("history", None)
    for event in state["scene"]["events"]:
        event.pop("history", None)
    return state


def _write(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _caption(value: str, limit: int = 43) -> str:
    text = str(value).encode("ascii", "replace").decode("ascii")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_native_source_frame(
    frame: VideoFrame,
    snapshot: RichLabelSnapshot,
    *,
    batch_number: int,
    predictor_epoch: str,
    batch_start: bool,
) -> np.ndarray:
    """Annotate only observed tracks belonging to this exact immutable JPEG."""
    if (snapshot.frame_sequence, snapshot.timestamp_seconds) != (
        frame.frame_sequence,
        frame.timestamp_seconds,
    ):
        raise ValueError("annotation snapshot does not match source JPEG")
    decoded = cv2.imdecode(
        np.frombuffer(frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if decoded is None:
        raise ValueError("could not decode annotation source JPEG")
    height, width = decoded.shape[:2]
    header, panel = 76, 360
    canvas = np.full((height + header, width + panel, 3), 24, dtype=np.uint8)
    canvas[header:, :width] = decoded
    observed = [
        item
        for item in snapshot.objects
        if (
            item.observation_state
            or item.attributes.get("observation_state", "observed")
        )
        == "observed"
    ]
    title = (
        f"Source {frame.timestamp_seconds:.3f}s | frame {frame.frame_sequence} "
        f"| people {len(observed)}"
    )
    phase = "NEW BATCH" if batch_start else "BATCH"
    lines = (
        title,
        f"{phase} {batch_number} | epoch {_caption(predictor_epoch, 36)}",
        "Memory is batch-scoped; cross-batch IDs are separate.",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (8, 20 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 210, 255) if batch_start and index == 1 else (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    palette = ((80, 220, 100), (255, 170, 80), (150, 110, 255), (80, 230, 230))
    rows = max(0, (height - 25) // 31)
    for index, item in enumerate(observed[:128]):
        native = item.attributes.get("native_track", {})
        if native.get("epoch") != predictor_epoch:
            raise ValueError("annotation contains a different native epoch")
        native_id = str(native.get("object_id", ""))
        subject_id = item.track_id
        suffix = subject_id.rsplit("-", 1)[-1]
        color = palette[
            int(suffix) % len(palette) if suffix.isdigit() else index % len(palette)
        ]
        x1, y1, x2, y2 = item.bbox.to_pixels(width, height)
        cv2.rectangle(canvas, (x1, y1 + header), (x2, y2 + header), color, 2)
        short = f"S{_caption(suffix, 12)} N{_caption(native_id, 12)}"
        cv2.putText(
            canvas,
            short,
            (x1, max(header + 12, y1 + header - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
        if index < rows:
            top = header + 15 + index * 31
            cv2.putText(
                canvas,
                _caption(subject_id),
                (width + 8, top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Native {_caption(native_id, 36)}",
                (width + 8, top + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
    note = (
        "No people observed"
        if not observed
        else f"Exact IDs and boxes: tracks.jsonl ({len(observed)} people)"
    )
    cv2.putText(
        canvas,
        note,
        (width + 8, canvas.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _make_inspector(args):
    if not args.with_inference:
        return None
    config = load_config(args.config)
    key = environment_api_key(config.model_api)
    if not key:
        raise ValueError("configured model API credential is unavailable")
    return TemporalVLMInspector(
        replace(
            config.model_api,
            timeout_seconds=config.runtime.scene_timeout_seconds,
            max_retries=0,
        ),
        model=config.inference.model,
        api_key=key,
        request_parameters=config.inference.request_parameters,
        max_image_side=config.runtime.scene_max_image_side,
        max_tokens=config.runtime.scene_max_tokens,
    )


def run(args) -> dict:
    labels = _validate(args)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("output directory must be new")
    runtime = SceneRuntime(
        uuid4().hex,
        "recording",
        mode="active",
        tracking_mode="batch",
        event_labels=labels,
    )
    registry = PersonRegistry(runtime.stream_session_id, runtime.source_id)
    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    detector = inspector = annotation_writer = None
    annotation_size = None
    annotated_frames = 0
    started = time.monotonic()
    samples = batches = requests = model_failures = 0
    accepted_verdicts = rejected_verdicts = native_instances = pts_fallbacks = 0
    processing_total = processing_max = 0.0
    epochs: set[str] = set()
    try:
        if not capture.isOpened():
            raise ValueError("could not open local recording")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(native_fps) or native_fps <= 0:
            raise ValueError("recording has no usable source frame rate")
        fps = min(args.fps, native_fps)
        detector = VideoTrackingHTTPClient(
            args.sam_endpoint, timeout_seconds=args.timeout_seconds
        )
        if not detector.gpu_ready():
            raise RuntimeError("SAM 3.1 native-video worker is not GPU-ready")
        inspector = _make_inspector(args)
        runtime.automatic_temporal_inference = inspector is not None
        output.mkdir(parents=True, exist_ok=False)
        pending: list[VideoFrame] = []
        pending_bytes = 0
        sequence, next_at, previous_at = 0, 0.0, -1.0
        with (
            (output / "tracks.jsonl").open("x", encoding="utf-8") as tracks,
            (output / "scene.jsonl").open("x", encoding="utf-8") as states,
            (output / "batches.jsonl").open("x", encoding="utf-8") as batch_log,
            (output / "annotations.jsonl").open("x", encoding="utf-8") as annotations,
        ):

            def flush() -> None:
                nonlocal batches, requests, model_failures, pending_bytes
                nonlocal accepted_verdicts, rejected_verdicts, native_instances
                nonlocal processing_total, processing_max
                nonlocal annotation_writer, annotation_size, annotated_frames
                if not pending:
                    return
                frames = tuple(pending)
                before = time.monotonic()
                batch = detector.infer_batch(
                    frames, stream_session_id=runtime.stream_session_id
                )
                elapsed = time.monotonic() - before
                if batch.predictor_epoch in epochs:
                    raise ValueError("worker reused native memory epoch across batches")
                epochs.add(batch.predictor_epoch)
                canonical = []

                def canonicalize(raw_snapshots):
                    if not registry.begin_native_epoch(BACKEND, batch.predictor_epoch):
                        raise ValueError("native registry rejected predictor epoch")
                    for raw in raw_snapshots:
                        registry.observe_snapshot(raw)
                        observed = registry.snapshot_at(
                            raw.timestamp_seconds, raw.frame_sequence
                        )
                        if observed is None or (
                            observed.timestamp_seconds,
                            observed.frame_sequence,
                        ) != (raw.timestamp_seconds, raw.frame_sequence):
                            raise ValueError(
                                "native source frame was not canonicalized"
                            )
                        canonical.append(observed)
                    return tuple(canonical)

                if not runtime.commit_native_tracking_batch(
                    batch.frames,
                    completed_timestamp_seconds=batch.completed_timestamp_seconds,
                    completed_frame_sequence=batch.completed_frame_sequence,
                    canonicalize=canonicalize,
                ):
                    raise ValueError("scene rejected completed native video evidence")
                batches += 1
                processing_total += elapsed
                processing_max = max(processing_max, elapsed)
                native_ids = {
                    item.attributes["native_track"]["object_id"]
                    for raw in batch.frames
                    for item in raw.objects
                }
                native_instances += len(native_ids)
                for frame_index, (frame, raw, observed) in enumerate(
                    zip(frames, batch.frames, canonical, strict=True)
                ):
                    _write(
                        tracks,
                        {
                            "batch_id": batch.batch_id,
                            "predictor_epoch": batch.predictor_epoch,
                            "memory_scope": "batch",
                            "raw": raw.as_dict(),
                            "canonical": observed.as_dict(),
                        },
                    )
                    annotated = render_native_source_frame(
                        frame,
                        observed,
                        batch_number=batches,
                        predictor_epoch=batch.predictor_epoch,
                        batch_start=frame_index == 0,
                    )
                    size = (annotated.shape[1], annotated.shape[0])
                    if annotation_writer is None:
                        annotation_size = size
                        annotation_writer = cv2.VideoWriter(
                            str(output / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            size,
                        )
                        if not annotation_writer.isOpened():
                            raise ValueError("could not create annotated video")
                    elif size != annotation_size:
                        raise ValueError(
                            "sampled source dimensions changed during recording"
                        )
                    annotation_writer.write(annotated)
                    _write(
                        annotations,
                        {
                            "output_frame_sequence": annotated_frames,
                            "output_timestamp_seconds": annotated_frames / fps,
                            "source_frame_sequence": frame.frame_sequence,
                            "source_timestamp_seconds": frame.timestamp_seconds,
                            "batch_number": batches,
                            "batch_id": batch.batch_id,
                            "predictor_epoch": batch.predictor_epoch,
                            "batch_start": frame_index == 0,
                            "memory_scope": "batch",
                            "subject_ids": [item.track_id for item in observed.objects],
                            "native_object_ids": [
                                item.attributes["native_track"]["object_id"]
                                for item in observed.objects
                            ],
                        },
                    )
                    annotated_frames += 1
                while inspector is not None and requests < args.max_requests:
                    jobs = runtime.take_inspections(limit=1)
                    if not jobs:
                        break
                    job = jobs[0]
                    requests += 1
                    counts = {}
                    try:
                        verdicts = inspector.inspect(job)
                        runtime.complete_inspection(
                            job.request.request_id, verdicts, completion_counts=counts
                        )
                    except Exception:
                        model_failures += 1
                        runtime.complete_inspection(
                            job.request.request_id, (), success=False
                        )
                    accepted_verdicts += counts.get("accepted_verdicts", 0)
                    rejected_verdicts += counts.get("rejected_verdicts", 0)
                metadata = {
                    "batch_id": batch.batch_id,
                    "predictor_epoch": batch.predictor_epoch,
                    "memory_scope": "batch",
                    "frame_count": len(frames),
                    "frame_sequences": [frame.frame_sequence for frame in frames],
                    "start_timestamp_seconds": frames[0].timestamp_seconds,
                    "completed_timestamp_seconds": batch.completed_timestamp_seconds,
                    "processing_latency_seconds": elapsed,
                    "native_object_instances": len(native_ids),
                }
                _write(batch_log, metadata)
                _write(
                    states,
                    {"kind": "batch_completed", **metadata, "runtime": _state(runtime)},
                )
                pending.clear()
                pending_bytes = 0

            while True:
                ok, image = capture.read()
                if not ok:
                    break
                at = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(at) or at < 0 or at <= previous_at:
                    at = sequence / native_fps
                    pts_fallbacks += 1
                if at <= previous_at:
                    raise ValueError("recording has inconsistent source timestamps")
                previous_at = at
                index = sequence
                sequence += 1
                if at > args.max_seconds:
                    break
                if at + 1e-8 < next_at:
                    continue
                while next_at <= at + 1e-8:
                    next_at += 1 / fps
                height, width = image.shape[:2]
                if max(height, width) > 1280:
                    scale = 1280 / max(height, width)
                    image = cv2.resize(
                        image,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                    )
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    raise ValueError("could not encode source JPEG")
                frame = VideoFrame(index, at, encoded.tobytes())
                if pending and (
                    pending_bytes + len(frame.jpeg_bytes) > MAX_BATCH_JPEG_BYTES
                    or at - pending[0].timestamp_seconds > 4.0
                ):
                    flush()
                runtime.observe_frame(frame.jpeg_bytes, at, index, force=True)
                _write(
                    states,
                    {
                        "kind": "source_frame",
                        "frame_sequence": index,
                        "timestamp_seconds": at,
                        "runtime": _state(runtime),
                    },
                )
                pending.append(frame)
                pending_bytes += len(frame.jpeg_bytes)
                samples += 1
                if len(pending) >= args.batch_frames:
                    flush()
            flush()
        if samples == 0 or batches == 0:
            raise ValueError(
                "recording produced no sampled frames or completed batches"
            )
        if annotation_writer is not None:
            annotation_writer.release()
            annotation_writer = None
        report = {
            "ok": model_failures == 0,
            "backend": BACKEND,
            "memory_scope": "batch",
            "continuous_camera_memory": False,
            "cross_batch_identity_continuity": False,
            "samples": samples,
            "batches": batches,
            "source_fps": native_fps,
            "sample_fps": fps,
            "annotated_video": {
                "path": "annotated.mp4",
                "frames": annotated_frames,
                "fps": fps,
                "source_mapping": "annotations.jsonl",
                "boxes": "exact_observed_source_frame",
                "cross_batch_identity_continuity": False,
            },
            "timestamp_source": "decoder_pts_with_fps_fallback",
            "timestamp_fallback_frames": pts_fallbacks,
            "native_object_instances_across_batches": native_instances,
            "processing_latency_seconds": {
                "total": processing_total,
                "mean": processing_total / batches if batches else None,
                "max": processing_max if batches else None,
            },
            "elapsed_seconds": time.monotonic() - started,
            "model_enabled": inspector is not None,
            "model_requests": requests,
            "model_failures": model_failures,
            "request_budget": args.max_requests,
            "budget_exhausted": inspector is not None and requests >= args.max_requests,
            "accepted_verdicts": accepted_verdicts,
            "rejected_verdicts": rejected_verdicts,
            "model": inspector.stats() if inspector else None,
            "accuracy_measured": False,
            "identity_switches": None,
            "ground_truth": "not_supplied",
            "final_runtime": _state(runtime),
        }
        with (output / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        return report
    finally:
        runtime.close()
        capture.release()
        if annotation_writer is not None:
            annotation_writer.release()
        if inspector is not None:
            inspector.close()
        if detector is not None:
            detector.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--sam-endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--events", default="smoking,fight,fall,weapon")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--timeout-seconds", type=float, default=75)
    parser.add_argument("--max-requests", type=int, default=12)
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--with-inference", action="store_true")
    inference.add_argument(
        "--no-inference", dest="with_inference", action="store_false"
    )
    parser.set_defaults(with_inference=False)
    args = parser.parse_args()
    try:
        _validate(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        report = run(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": (
                        "Run failed; check local input, worker readiness, "
                        "and new output directory."
                    ),
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "ok",
                    "backend",
                    "memory_scope",
                    "continuous_camera_memory",
                    "samples",
                    "batches",
                    "model_requests",
                    "model_failures",
                    "accuracy_measured",
                )
            }
        )
    )
    return int(not report["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
