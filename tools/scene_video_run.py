"""Run continuous person/event analysis over a local recording, with explicit inputs.

SAM mode requires a real GPU-ready worker. Recorded-track mode never claims to
validate SAM. Output directories are created exclusively and are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from goodvision_harness.config import load_config
from goodvision_harness.model_api import environment_api_key
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelHTTPClient, RichLabelSnapshot
from goodvision_harness.scene_delivery import render_scene_jpeg
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import SceneRuntime


def read_tracks(path: Path) -> dict[int, dict]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("recorded tracks exceed 16 MiB")
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        sequence = value["frame_sequence"]
        if type(sequence) is not int or sequence < 0 or sequence in records:
            raise ValueError(
                "recorded tracks require unique nonnegative frame sequences"
            )
        records[sequence] = value
    return records


def run(args) -> dict:
    config = load_config(args.config)
    output = Path(args.output)
    # Validate expensive/external prerequisites before creating any output.
    labels = tuple(args.events.split(","))
    scene = SceneRuntime(uuid4().hex, "recording", mode="active", event_labels=labels)
    tracks = read_tracks(Path(args.recorded_tracks)) if args.recorded_tracks else None
    detector = None
    inspector = None
    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError("could not open recording")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(native_fps) or native_fps <= 0:
            raise ValueError("recording has no usable source frame rate")
        fps = min(args.fps, native_fps)
        if args.sam_endpoint:
            detector = RichLabelHTTPClient(
                args.sam_endpoint,
                mode="tracks",
                include_attributes=False,
                timeout_seconds=20,
            )
            if not detector.gpu_ready():
                raise RuntimeError(
                    "SAM worker is not GPU-ready; weights and CUDA warmup must complete"
                )
        if not args.no_inference:
            key = environment_api_key(config.model_api)
            if not key:
                raise ValueError("configured model API credential is unavailable")
            inspector = TemporalVLMInspector(
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
            scene.automatic_temporal_inference = True
        output.mkdir(parents=True, exist_ok=False)
        registry = PersonRegistry(
            stream_session_id=scene.stream_session_id, source_id=scene.source_id
        )
        sequence = samples = requests = failures = 0
        next_at = 0.0
        with (
            (output / "scene.jsonl").open("x", encoding="utf-8") as states,
            (output / "tracks.jsonl").open("x", encoding="utf-8") as observed,
        ):
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                at = sequence / native_fps
                index = sequence
                sequence += 1
                if at > args.max_seconds:
                    break
                if at + 1e-8 < next_at:
                    continue
                next_at += 1 / fps
                height, width = frame.shape[:2]
                if max(height, width) > 1280:
                    frame = cv2.resize(
                        frame,
                        (
                            round(width * 1280 / max(height, width)),
                            round(height * 1280 / max(height, width)),
                        ),
                    )
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    raise ValueError("could not encode source frame")
                jpeg = encoded.tobytes()
                snapshot = None
                if detector is not None:
                    snapshot = detector.infer(
                        jpeg,
                        stream_session_id=scene.stream_session_id,
                        frame_sequence=index,
                        timestamp_seconds=at,
                    )
                elif index in tracks:
                    value = {
                        **tracks[index],
                        "stream_session_id": scene.stream_session_id,
                    }
                    snapshot = RichLabelSnapshot.from_dict(value)
                    if not math.isclose(snapshot.timestamp_seconds, at, abs_tol=1e-6):
                        raise ValueError(
                            "recorded track source timestamp does not match video"
                        )
                if snapshot is not None:
                    observed.write(
                        json.dumps(snapshot.as_dict(), ensure_ascii=False) + "\n"
                    )
                    registry.observe_snapshot(snapshot)
                    canonical = registry.snapshot_at(at, index)
                    if canonical is not None and (
                        canonical.timestamp_seconds,
                        canonical.frame_sequence,
                    ) == (at, index):
                        scene.observe_tracks(canonical)
                scene.observe_frame(jpeg, at, index, force=True)
                if inspector is not None and requests < args.max_requests:
                    for job in scene.take_inspections(limit=1):
                        requests += 1
                        try:
                            verdicts = inspector.inspect(job)
                            scene.complete_inspection(job.request.request_id, verdicts)
                        except Exception:
                            failures += 1
                            scene.complete_inspection(
                                job.request.request_id, (), success=False
                            )
                state = scene.snapshot()
                states.write(json.dumps(state, ensure_ascii=False) + "\n")
                rendered = render_scene_jpeg(jpeg, state, at)
                image = cv2.imdecode(
                    np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (image.shape[1], image.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("annotated video encoder is unavailable")
                writer.write(image)
                samples += 1
        report = {
            "perception_source": "sam_gpu" if detector else "recorded_tracks",
            "sam_validated": detector is not None,
            "model_enabled": inspector is not None,
            "samples": samples,
            "source_fps": native_fps,
            "output_fps": fps,
            "model_requests": requests,
            "model_failures": failures,
            "request_budget": args.max_requests,
            "budget_exhausted": requests >= args.max_requests and inspector is not None,
            "model": inspector.stats() if inspector else None,
            "final_scene": scene.snapshot(),
            "accuracy_measured": False,
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    finally:
        scene.close()
        capture.release()
        if writer is not None:
            writer.release()
        if inspector is not None:
            inspector.close()
        if detector is not None:
            detector.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--config", default="config.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sam-endpoint")
    source.add_argument("--recorded-tracks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", default="smoking,fight,fall,weapon")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--max-requests", type=int, default=12)
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="validate tracking/rendering without model API calls",
    )
    args = parser.parse_args()
    if (
        not 0 < args.fps <= 15
        or not 0 < args.max_seconds <= 600
        or not 1 <= args.max_requests <= 100
    ):
        parser.error(
            "fps must be within (0,15], duration within (0,600], "
            "and requests within 1..100"
        )
    try:
        report = run(args)
    except Exception as error:
        # Do not print HTTP errors, credentials, responses, or private video names.
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "message": (
                        "Run failed; check local input, configured credentials, "
                        "worker readiness, and the output directory."
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
                    "perception_source",
                    "samples",
                    "model_requests",
                    "model_failures",
                    "accuracy_measured",
                )
            }
        )
    )
    return int(report["model_failures"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
