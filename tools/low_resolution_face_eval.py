"""Offline face evaluation using local media, local models and a disposable gallery.

This measures observed behavior, not a benchmark accuracy score: source identity
and visibility need human verification. A control input must depict a different
person. No images, embeddings or enrollment names are included in the report.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A borrowed Python environment may have another checkout installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognizerConfig,
)
from goodvision_harness.grounding import ModelObject
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.sliding_window import (
    FrameWindow,
    SampledFrame,
)


@dataclass
class MediaFrame:
    image: Any
    timestamp: float
    source_index: int


def degrade(image: Any, long_side: int | None) -> Any:
    """Remove source detail without later enlarging it for the evaluator."""
    if long_side is None or max(image.shape[:2]) <= long_side:
        return image.copy()
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _encode(image: Any) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("could not encode evaluation frame")
    return encoded.tobytes()


def read_media(
    path: Path, *, start: float, sample_fps: float, max_frames: int
) -> list[MediaFrame]:
    if not path.is_file():
        raise ValueError(f"media file does not exist: {path}")
    if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif"}:
        image = cv2.imdecode(
            np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("input image could not be decoded")
        return [MediaFrame(image, 0.0, 0)]
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("input video could not be opened")
    frames = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video needs a valid source frame rate")
        source_index = max(0, int(start * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        next_sample = start
        while len(frames) < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            timestamp = source_index / fps
            if timestamp + 1e-8 >= next_sample:
                frames.append(MediaFrame(image, timestamp, source_index))
                next_sample = timestamp + 1.0 / sample_fps
            source_index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("input video has no frames in the selected interval")
    return frames


def _infer(
    runtime: FaceRecognitionRuntime, image: Any, index: int, timestamp: float
) -> tuple[tuple[ModelObject, ...], float]:
    height, width = image.shape[:2]
    window = FrameWindow(
        index=index,
        frames=(
            SampledFrame(
                sample_index=index,
                source_frame_index=index,
                timestamp_seconds=timestamp,
                jpeg_bytes=_encode(image),
                width=width,
                height=height,
            ),
        ),
    )
    response = runtime.classify_window(window)
    if response.classification is None:
        raise RuntimeError(
            response.error or "face inference returned no classification"
        )
    return tuple(response.classification.objects), response.latency_seconds * 1000


def _public_faces(
    objects: tuple[ModelObject, ...], identities: dict[str, str], expected_id: str
) -> list[dict[str, Any]]:
    faces = []
    for item in objects:
        attrs = item.attributes
        person_id = str(attrs.get("person_id", ""))
        faces.append(
            {
                "status": attrs.get("status"),
                "identity": identities.get(person_id),
                "matches_enrollment": person_id == expected_id,
                "reason": attrs.get("reason"),
                "observation_status": attrs.get("observation_status"),
                "identity_source": attrs.get("identity_source"),
                "face_width_input_px": attrs.get("face_width"),
                "face_height_input_px": attrs.get("face_height"),
                "detection_score": attrs.get("detection_score"),
                "similarity": attrs.get("similarity"),
                "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
            }
        )
    return faces


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    present = 0
    output_present = 0
    expected_frames = 0
    adjacent_pairs = state_switches = known_unknown_switches = identity_switches = 0
    previous = None
    for row in rows:
        faces = row[mode]
        present += bool(row["raw"])
        output_present += bool(faces)
        expected_frames += any(face["matches_enrollment"] for face in faces)
        counts.update(face["status"] for face in faces)
        # Restrict this metric to adjacent single-face observations. Missing or
        # multi-face frames break pairs; they must never imply stable identity.
        current = (
            (faces[0]["status"], faces[0]["identity"])
            if len(faces) == len(row["raw"]) == 1
            else None
        )
        if previous is not None and current is not None:
            adjacent_pairs += 1
            state_switches += current != previous
            known_unknown_switches += {current[0], previous[0]} == {"known", "unknown"}
            identity_switches += (
                current[0] == previous[0] == "known" and current[1] != previous[1]
            )
        previous = current
    latencies = [row["inference_ms"] for row in rows]
    return {
        "frames": len(rows),
        "frames_with_face_detection": present,
        "frames_without_face_detection": len(rows) - present,
        "frames_with_face_output": output_present,
        "face_detection_frame_rate": present / len(rows) if rows else None,
        "face_observation_counts": dict(counts),
        "frames_matching_enrollment": expected_frames,
        "matching_enrollment_frame_rate": expected_frames / len(rows) if rows else None,
        "adjacent_single_face_pairs": adjacent_pairs,
        "adjacent_state_or_identity_switches": state_switches,
        "adjacent_known_unknown_switches": known_unknown_switches,
        "adjacent_known_identity_switches": identity_switches,
        "inference_ms_median": statistics.median(latencies) if latencies else None,
        "inference_ms_max": max(latencies) if latencies else None,
    }


def evaluate_source(
    runtime: FaceRecognitionRuntime,
    frames: list[MediaFrame],
    *,
    seed: tuple[ModelObject, ...],
    people: dict[str, str],
    expected_id: str,
    controls: bool,
) -> dict[str, Any]:
    identities = {person_id: f"identity-{i + 1}" for i, person_id in enumerate(people)}
    raw_cache: dict[tuple[int | None, int], tuple[tuple[ModelObject, ...], float]] = {}
    for index, frame in enumerate(frames):
        raw_cache[None, index] = _infer(runtime, frame.image, index, frame.timestamp)
    levels = []
    for long_side in (None, 320, 160, 96, 64, 32):
        print(
            f"Evaluating {'control' if controls else 'subject'} "
            f"at {long_side or 'original'}",
            flush=True,
        )
        cold = PersonRegistry("cold", "evaluation")
        seeded = PersonRegistry("seeded", "evaluation")
        seeded.resolve_faces(seed, timestamp=0, frame_sequence=0, people=people)
        rows = []
        for index, frame in enumerate(frames):
            image = degrade(frame.image, long_side)
            if (long_side, index) not in raw_cache:
                raw_cache[long_side, index] = _infer(
                    runtime, image, index, frame.timestamp
                )
            raw, latency = raw_cache[long_side, index]
            timeline = 1.0 + frame.timestamp - frames[0].timestamp
            started = time.perf_counter()
            cold_objects = cold.resolve_faces(
                raw,
                timestamp=timeline,
                frame_sequence=index + 1,
                people=people,
            )
            seeded_objects = seeded.resolve_faces(
                raw,
                timestamp=timeline,
                frame_sequence=index + 1,
                people=people,
            )
            rows.append(
                {
                    "source_frame_index": frame.source_index,
                    "source_timestamp_seconds": frame.timestamp,
                    "input_width": image.shape[1],
                    "input_height": image.shape[0],
                    "original_width": frame.image.shape[1],
                    "original_height": frame.image.shape[0],
                    "inference_ms": latency,
                    "two_registry_updates_ms": (time.perf_counter() - started) * 1000,
                    "raw": _public_faces(raw, identities, expected_id),
                    "cold_start": _public_faces(cold_objects, identities, expected_id),
                    "clear_seed": _public_faces(
                        seeded_objects, identities, expected_id
                    ),
                    "original_faces": _public_faces(
                        raw_cache[None, index][0], identities, expected_id
                    ),
                }
            )
        recovery_raw = raw_cache[None, len(frames) - 1][0]
        recovery = seeded.resolve_faces(
            recovery_raw,
            timestamp=2.0 + frames[-1].timestamp - frames[0].timestamp,
            frame_sequence=len(frames) + 1,
            people=people,
        )
        summaries = {
            mode: summarize(rows, mode) for mode in ("raw", "cold_start", "clear_seed")
        }
        if controls:
            for summary in summaries.values():
                summary["false_enrollment_match_frames"] = summary[
                    "frames_matching_enrollment"
                ]
        levels.append(
            {
                "target_long_side": long_side or "original",
                "summary": summaries,
                "clear_recovery": _public_faces(recovery, identities, expected_id),
                "frames": rows,
            }
        )
    return {
        "role": "different_person_control" if controls else "subject",
        "levels": levels,
    }


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Local image or video"
    )
    parser.add_argument(
        "--control-input",
        type=Path,
        help="Local media verified to depict a different person",
    )
    enrollment = parser.add_mutually_exclusive_group(required=True)
    enrollment.add_argument(
        "--enroll-image",
        type=Path,
        help="Single clear face used only in the temporary gallery",
    )
    enrollment.add_argument(
        "--enroll-frame-seconds",
        type=float,
        help="Enroll a frame from --input at this source time",
    )
    parser.add_argument(
        "--gallery-input",
        type=Path,
        help="Optional gallery copied read-only into a temporary directory",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Existing InsightFace model root; downloads disabled",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        choices=(
            "CPUExecutionProvider",
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON report; refuses to overwrite existing files",
    )
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument(
        "--enrollment-min-face-size",
        type=int,
        default=80,
        help="Evaluation-only enrollment cutoff; production default is 80 pixels",
    )
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--control-start-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not 0 < args.sample_fps <= 30 or not 1 <= args.max_frames <= 240:
        parser.error("sample-fps must be in (0,30] and max-frames in [1,240]")
    if not 20 <= args.enrollment_min_face_size <= 640:
        parser.error("enrollment-min-face-size must be in [20,640]")
    times = (
        args.start_seconds,
        args.control_start_seconds,
        args.enroll_frame_seconds or 0,
    )
    if any(not np.isfinite(value) or value < 0 for value in times):
        parser.error("source timestamps must be finite and non-negative")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    model_root = args.model_root.expanduser().resolve()
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        if not (model_root / "models" / "buffalo_l" / name).is_file():
            raise ValueError(
                f"required local model is absent: {name}; no download attempted"
            )
    frames = read_media(
        args.input,
        start=args.start_seconds,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
    )
    enrollment = read_media(
        args.enroll_image or args.input,
        start=args.enroll_frame_seconds or 0,
        sample_fps=1,
        max_frames=1,
    )[0]
    original_gallery = args.gallery_input.read_bytes() if args.gallery_input else None
    with tempfile.TemporaryDirectory(prefix="goodvision-lowres-eval-") as temporary:
        gallery_path = Path(temporary) / "evaluation-gallery.json"
        if original_gallery is not None:
            gallery_path.write_bytes(original_gallery)
        runtime = FaceRecognitionRuntime(
            FaceRecognizerConfig(
                model_root=model_root,
                provider=args.provider,
                allow_model_download=False,
                gallery_path=gallery_path,
                enrollment_min_face_size=args.enrollment_min_face_size,
            )
        )
        warmup_seconds = runtime.warmup()
        expected_id = "low-resolution-evaluation-subject"
        if any(person.person_id == expected_id for person in runtime.list_people()):
            raise ValueError(
                "temporary evaluation identity collides with input gallery"
            )
        runtime.enroll(expected_id, "Evaluation subject", _encode(enrollment.image))
        enrollment_faces, _ = _infer(runtime, enrollment.image, 0, 0)
        people = {person.person_id: person.name for person in runtime.list_people()}
        seed, _ = _infer(runtime, frames[0].image, 0, 0)
        if not any(item.attributes.get("person_id") == expected_id for item in seed):
            raise ValueError(
                "first evaluation frame is not recognized as enrollment; "
                "choose a clear start frame"
            )
        report = {
            "schema_version": "1.0",
            "purpose": "local_low_resolution_face_evaluation",
            "source": str(args.input.resolve()),
            "provider": args.provider,
            "model": "buffalo_l",
            "detector_input_size": 640,
            "protocol": {
                "sample_fps": args.sample_fps,
                "max_frames": args.max_frames,
                "start_seconds": args.start_seconds,
                "control_start_seconds": args.control_start_seconds,
                "jpeg_quality": 95,
                "recognition_threshold": runtime.config.threshold,
                "minimum_face_size": runtime.config.min_face_size,
            },
            "enrollment": {
                "input_width": enrollment.image.shape[1],
                "input_height": enrollment.image.shape[0],
                "minimum_face_size": args.enrollment_min_face_size,
                "production_minimum_face_size": 80,
                "uses_production_cutoff": args.enrollment_min_face_size == 80,
                "faces": _public_faces(
                    enrollment_faces,
                    {expected_id: "evaluation-subject"},
                    expected_id,
                ),
            },
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("insightface", "numpy")
            },
            "temporary_gallery_people": len(people),
            "temporary_gallery_samples": sum(
                person.sample_count for person in runtime.list_people()
            ),
            "source_gallery": str(args.gallery_input.resolve())
            if args.gallery_input else None,
            "warmup_seconds": warmup_seconds,
            "notes": [
                "Input dimensions and face_width_input_px/face_height_input_px "
                "describe actual decoded pixels.",
                "The fixed 640 detector input can resize pixels but cannot "
                "recover source details.",
                "Detection frame rate is observed presence, not recall against "
                "human face annotations.",
                "Single-image inputs measure degradation only, not angle changes "
                "or tracking accuracy.",
                "Adjacent switch counts exclude missing and multi-face frames; "
                "zero detections do not prove stability.",
                "Clear seed uses the first subject frame; geometry is not forced "
                "to match later/control frames.",
                "Clear recovery uses the final original source frame after "
                "each degraded sequence.",
                "Raw, cold and seeded variants reuse the same actual model "
                "observations; tracking adds no inference calls.",
                "Gallery is temporary and deleted at exit; original input "
                "gallery is read-only.",
            ],
            "evaluations": [
                evaluate_source(
                    runtime,
                    frames,
                    seed=seed,
                    people=people,
                    expected_id=expected_id,
                    controls=False,
                )
            ],
        }
        if args.control_input:
            controls = read_media(
                args.control_input,
                start=args.control_start_seconds,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
            )
            report["control_source"] = str(args.control_input.resolve())
            report["evaluations"].append(
                evaluate_source(
                    runtime,
                    controls,
                    seed=seed,
                    people=people,
                    expected_id=expected_id,
                    controls=True,
                )
            )
        if args.gallery_input and args.gallery_input.read_bytes() != original_gallery:
            raise RuntimeError("input gallery changed concurrently; report discarded")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(f"Report saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
