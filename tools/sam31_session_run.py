"""Review continuous SAM video sessions on a finite local recording.

This is an offline, ordered validation run, not a live latency benchmark. Native
tracking uses one session; delayed outputs are drawn only on their original JPEG.
Event VLM calls remain disabled unless --with-inference is explicitly supplied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.video_tracking import BACKEND, MAX_BATCH_JPEG_BYTES, VideoFrame
from goodvision_harness.video_tracking_session import (
    VideoSessionTrackingHTTPClient,
    continuous_source_history_seconds,
)

# Reuse the reviewed exact-source box renderer and explicitly gated VLM setup.
_spec = importlib.util.spec_from_file_location(
    "sam31_batch_helpers", Path(__file__).with_name("sam31_video_run.py")
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_make_inspector = _helpers._make_inspector


def render_session_source_frame(
    frame, snapshot, *, update_number, predictor_epoch, first_frame=False
):
    image = _helpers.render_native_source_frame(
        frame,
        snapshot,
        batch_number=update_number,
        predictor_epoch=predictor_epoch,
        batch_start=False,
    )
    image[:76] = 24
    lines = (
        f"Source {frame.timestamp_seconds:.3f}s | frame {frame.frame_sequence} "
        f"| people {len(snapshot.objects)}",
        f"{'SESSION START' if first_frame else 'CONTINUOUS SESSION'} "
        f"| update {update_number}",
        f"Stable native epoch {_helpers._caption(predictor_epoch, 42)}",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (8, 20 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return image


def _transport(frame, max_side, quality):
    if max_side is None and quality is None:
        return frame
    image = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("transport source JPEG is invalid")
    height, width = image.shape[:2]
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality or 90])
    if not ok:
        raise ValueError("could not encode transport JPEG")
    return VideoFrame(frame.frame_sequence, frame.timestamp_seconds, encoded.tobytes())


def run(args):
    labels = _helpers._validate(args)
    if args.batch_frames != 16 or args.fps < 0.5:
        raise ValueError(
            "continuous session runner requires batch_frames=16 and fps>=0.5"
        )
    max_side = getattr(args, "transport_max_side", None)
    quality = getattr(args, "transport_jpeg_quality", None)
    if max_side is not None and (
        type(max_side) is not int or not 128 <= max_side <= 1280
    ):
        raise ValueError("transport max side must be within 128..1280")
    if quality is not None and (type(quality) is not int or not 1 <= quality <= 100):
        raise ValueError("transport JPEG quality must be within 1..100")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("output directory must be new")
    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    detector = inspector = runtime = writer = None
    created = False
    started = time.perf_counter()
    sample_count = annotated_count = requests = model_failures = 0
    accepted_verdicts = rejected_verdicts = pts_fallbacks = 0
    original_bytes = transport_bytes = pending_bytes = peak_pending_bytes = 0
    peak_pending_frames = 0
    updates = []
    pending = OrderedDict()
    chunk = []
    native_ids = set()
    annotation_size = source_size = transport_size = None
    try:
        if not capture.isOpened():
            raise ValueError("could not open local recording")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(native_fps) or native_fps <= 0:
            raise ValueError("recording has no usable source frame rate")
        fps = min(args.fps, native_fps)
        if fps < 0.5:
            raise ValueError(
                "source rate is below bounded continuous retention capacity"
            )
        runtime = SceneRuntime(
            uuid4().hex,
            "recording",
            mode="active",
            tracking_mode="batch",
            native_memory_scope="session",
            tracking_history_seconds=continuous_source_history_seconds(fps),
            event_labels=labels,
        )
        registry = PersonRegistry(runtime.stream_session_id, runtime.source_id)
        detector = VideoSessionTrackingHTTPClient(
            args.sam_endpoint, timeout_seconds=args.timeout_seconds
        )
        if not detector.gpu_ready():
            raise RuntimeError("continuous native-video worker is not GPU-ready")
        inspector = _make_inspector(args)
        runtime.automatic_temporal_inference = inspector is not None
        output.mkdir(parents=True, exist_ok=False)
        created = True
        session = detector.open_session(runtime.stream_session_id)
        with (
            (output / "tracks.jsonl").open("x", encoding="utf-8") as tracks,
            (output / "scene.jsonl").open("x", encoding="utf-8") as states,
            (output / "appends.jsonl").open("x", encoding="utf-8") as appends,
            (output / "annotations.jsonl").open("x", encoding="utf-8") as annotations,
        ):

            def append(*, final=False):
                nonlocal pending_bytes, writer, annotation_size, annotated_count
                nonlocal requests, model_failures, accepted_verdicts, rejected_verdicts
                inputs = tuple(chunk)
                before = time.perf_counter()
                result = detector.append(
                    inputs,
                    stream_session_id=runtime.stream_session_id,
                    is_last_batch=final,
                )
                elapsed = time.perf_counter() - before
                if result.predictor_epoch != session.predictor_epoch:
                    raise ValueError("worker changed continuous session epoch")
                originals = []
                for observed in result.frames:
                    frame = pending.get(observed.frame_sequence)
                    if frame is None or (
                        frame.frame_sequence,
                        frame.timestamp_seconds,
                    ) != (observed.frame_sequence, observed.timestamp_seconds):
                        raise ValueError(
                            "delayed observation has no matching retained original"
                        )
                    originals.append(frame)
                canonical = []

                def canonicalize(snapshots):
                    if not registry.begin_native_epoch(BACKEND, result.predictor_epoch):
                        raise ValueError("registry rejected continuous native epoch")
                    for raw in snapshots:
                        registry.observe_snapshot(raw)
                        mapped = registry.snapshot_at(
                            raw.timestamp_seconds, raw.frame_sequence
                        )
                        if mapped is None or (
                            mapped.frame_sequence,
                            mapped.timestamp_seconds,
                        ) != (raw.frame_sequence, raw.timestamp_seconds):
                            raise ValueError(
                                "continuous source could not be canonicalized"
                            )
                        canonical.append(mapped)
                    return tuple(canonical)

                if result.frames and not runtime.commit_native_tracking_batch(
                    result.frames,
                    completed_timestamp_seconds=result.completed_timestamp_seconds,
                    completed_frame_sequence=result.completed_frame_sequence,
                    canonicalize=canonicalize,
                ):
                    raise ValueError("scene rejected delayed continuous evidence")
                update_number = len(updates) + 1
                for index, (frame, raw, mapped) in enumerate(
                    zip(originals, result.frames, canonical, strict=True)
                ):
                    _helpers._write(
                        tracks,
                        {
                            "request_id": result.request_id,
                            "predictor_epoch": result.predictor_epoch,
                            "memory_scope": "session",
                            "raw": raw.as_dict(),
                            "canonical": mapped.as_dict(),
                        },
                    )
                    image = render_session_source_frame(
                        frame,
                        mapped,
                        update_number=update_number,
                        predictor_epoch=result.predictor_epoch,
                        first_frame=annotated_count == 0,
                    )
                    size = image.shape[1], image.shape[0]
                    if writer is None:
                        annotation_size = size
                        writer = cv2.VideoWriter(
                            str(output / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            size,
                        )
                        if not writer.isOpened():
                            raise ValueError("could not create annotated video")
                    elif size != annotation_size:
                        raise ValueError("annotation source dimensions changed")
                    writer.write(image)
                    _helpers._write(
                        annotations,
                        {
                            "output_frame_sequence": annotated_count,
                            "output_timestamp_seconds": annotated_count / fps,
                            "source_frame_sequence": frame.frame_sequence,
                            "source_timestamp_seconds": frame.timestamp_seconds,
                            "request_id": result.request_id,
                            "update_number": update_number,
                            "response_start": index == 0,
                            "session_start": annotated_count == 0,
                            "predictor_epoch": result.predictor_epoch,
                            "memory_scope": "session",
                            "subject_ids": [obj.track_id for obj in mapped.objects],
                            "native_object_ids": [
                                obj.attributes["native_track"]["object_id"]
                                for obj in mapped.objects
                            ],
                        },
                    )
                    native_ids.update(
                        obj.attributes["native_track"]["object_id"]
                        for obj in raw.objects
                    )
                    pending_bytes -= len(pending.pop(frame.frame_sequence).jpeg_bytes)
                    annotated_count += 1
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
                    "request_id": result.request_id,
                    "predictor_epoch": result.predictor_epoch,
                    "memory_scope": "session",
                    "state": result.state,
                    "is_last_batch": final,
                    "input_frame_sequences": [frame.frame_sequence for frame in inputs],
                    "emitted_frame_sequences": [
                        frame.frame_sequence for frame in result.frames
                    ],
                    "processing_latency_seconds": elapsed,
                    "pending_frame_count": result.pending_frame_count,
                    "transport_jpeg_bytes": sum(
                        len(frame.jpeg_bytes) for frame in inputs
                    ),
                }
                for prefix in ("accepted", "processed", "emitted"):
                    for suffix in ("frame_sequence", "timestamp_seconds"):
                        key = f"{prefix}_{suffix}"
                        metadata[key] = getattr(result, key)
                updates.append(metadata)
                _helpers._write(appends, metadata)
                _helpers._write(
                    states,
                    {
                        "kind": "append_completed",
                        **metadata,
                        "runtime": _helpers._state(runtime),
                    },
                )
                chunk.clear()

            sequence, next_at, previous_at = 0, 0.0, -1.0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                at = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(at) or at < 0 or at <= previous_at:
                    at = sequence / native_fps
                    pts_fallbacks += 1
                if at <= previous_at:
                    raise ValueError("recording source timestamps are inconsistent")
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
                shape = image.shape[1], image.shape[0]
                if source_size is not None and shape != source_size:
                    raise ValueError("source dimensions changed")
                source_size = shape
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    raise ValueError("could not encode source JPEG")
                frame = VideoFrame(index, at, encoded.tobytes())
                transported = _transport(frame, max_side, quality)
                decoded = cv2.imdecode(
                    np.frombuffer(transported.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                shape = decoded.shape[1], decoded.shape[0]
                if transport_size is not None and shape != transport_size:
                    raise ValueError("transport dimensions changed")
                transport_size = shape
                if (
                    len(pending) >= 32
                    or pending_bytes + len(frame.jpeg_bytes) > 64 * 1024 * 1024
                ):
                    raise ValueError(
                        "pending original evidence exceeds bounded retention"
                    )
                if (
                    sum(len(item.jpeg_bytes) for item in chunk)
                    + len(transported.jpeg_bytes)
                    > MAX_BATCH_JPEG_BYTES
                ):
                    raise ValueError("continuous append exceeds byte bound")
                pending[index] = frame
                pending_bytes += len(frame.jpeg_bytes)
                peak_pending_bytes = max(peak_pending_bytes, pending_bytes)
                peak_pending_frames = max(peak_pending_frames, len(pending))
                runtime.observe_frame(frame.jpeg_bytes, at, index, force=True)
                chunk.append(transported)
                original_bytes += len(frame.jpeg_bytes)
                transport_bytes += len(transported.jpeg_bytes)
                sample_count += 1
                _helpers._write(
                    states,
                    {
                        "kind": "source_frame",
                        "frame_sequence": index,
                        "timestamp_seconds": at,
                        "runtime": _helpers._state(runtime),
                    },
                )
                if len(chunk) == 16:
                    append()
            if not sample_count:
                raise ValueError("recording produced no sampled frames")
            # Even an exact multiple of sixteen requires an empty final request
            # to drain native lookahead; no duplicate or synthetic frame is added.
            append(final=True)
        if pending or annotated_count != sample_count:
            raise ValueError("continuous output did not cover every original source")
        if writer is not None:
            writer.release()
            writer = None
        final = updates[-1]
        report = {
            "ok": model_failures == 0,
            "backend": BACKEND,
            "memory_scope": "session",
            "continuous_camera_memory": len(updates) > 1,
            "cross_append_identity_continuity": len(updates) > 1,
            "offline_ordered_validation": True,
            "live_latency_measured": False,
            "samples": sample_count,
            "appends": len(updates),
            "predictor_epoch": session.predictor_epoch,
            "source_fps": native_fps,
            "sample_fps": fps,
            "annotated_video": {
                "path": "annotated.mp4",
                "frames": annotated_count,
                "fps": fps,
                "source_mapping": "annotations.jsonl",
                "boxes": "exact_observed_source_frame",
                "cross_append_identity_continuity": len(updates) > 1,
            },
            "timestamp_source": "decoder_pts_with_fps_fallback",
            "timestamp_fallback_frames": pts_fallbacks,
            "transport": {
                "enabled": max_side is not None or quality is not None,
                "max_side": max_side,
                "jpeg_quality": (quality or 90)
                if max_side is not None or quality is not None
                else None,
                "source_dimensions": source_size,
                "transport_dimensions": transport_size,
                "source_evidence_unchanged": True,
                "original_jpeg_bytes": original_bytes,
                "transport_jpeg_bytes": transport_bytes,
            },
            "pending_originals": {
                "peak_frames": peak_pending_frames,
                "peak_bytes": peak_pending_bytes,
                "remaining_frames": len(pending),
                "remaining_bytes": pending_bytes,
            },
            "final_watermarks": {
                key: value
                for key, value in final.items()
                if key.startswith(("accepted_", "processed_", "emitted_"))
            },
            "native_object_ids": len(native_ids),
            "pending_frame_count": final["pending_frame_count"],
            "processing_latency_seconds": {
                "total": sum(row["processing_latency_seconds"] for row in updates),
                "mean": sum(row["processing_latency_seconds"] for row in updates)
                / len(updates),
                "max": max(row["processing_latency_seconds"] for row in updates),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "model_enabled": inspector is not None,
            "model_requests": requests,
            "model_failures": model_failures,
            "request_budget": args.max_requests,
            "budget_exhausted": inspector is not None and requests >= args.max_requests,
            "accepted_verdicts": accepted_verdicts,
            "rejected_verdicts": rejected_verdicts,
            "model": inspector.stats() if inspector else None,
            "accuracy_measured": False,
            "mask_accuracy_measured": False,
            "identity_switches": None,
            "ground_truth": "not_supplied",
            "final_runtime": _helpers._state(runtime),
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        return report
    except Exception as error:
        if created:
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error_type": type(error).__name__,
                        "continuous_session_aborted": True,
                        "samples": sample_count,
                        "annotated_frames": annotated_count,
                    }
                ),
                encoding="utf-8",
            )
        raise
    finally:
        if runtime is not None:
            runtime.close()
        capture.release()
        if writer is not None:
            writer.release()
        if inspector is not None:
            inspector.close()
        if detector is not None:
            detector.close()


def main():
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
    parser.add_argument("--transport-max-side", type=int, default=None)
    parser.add_argument("--transport-jpeg-quality", type=int, default=None)
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--with-inference", action="store_true")
    inference.add_argument(
        "--no-inference", dest="with_inference", action="store_false"
    )
    parser.set_defaults(with_inference=False)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": "Continuous session aborted; check local recording, "
                    "worker readiness and new output directory.",
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
                    "memory_scope",
                    "samples",
                    "appends",
                    "pending_frame_count",
                    "model_requests",
                    "accuracy_measured",
                    "transport",
                )
            }
        )
    )
    return int(not report["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
