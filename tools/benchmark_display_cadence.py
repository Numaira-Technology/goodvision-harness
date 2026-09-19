"""Benchmark synthetic source frames with independent face/event overlays.

Run from the repository with its Python environment:
  python tools/benchmark_display_cadence.py --duration 4 --delay 2.5

No real inference, network transport, browser, or video decoding is included.
The measured excess over the configured delay includes source JPEG encoding,
compositor scheduling, annotation selection, rendering and output encoding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.contracts import (  # noqa: E402
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delayed_overlay import DelayedOverlayComposer  # noqa: E402
from goodvision_harness.result_bus import FramePacket  # noqa: E402
from goodvision_harness.service import (  # noqa: E402
    _analysis_overlay_arguments,
    _render_analysis_jpeg,
)
from goodvision_harness.video_to_stream import encode_jpeg  # noqa: E402

STAMP = "2026-09-10T00:00:00Z"
DISPLAY_RATES = (15, 30)


def _duration_seconds(value: str | float) -> float:
    duration = float(value)
    minimum = 2 / min(DISPLAY_RATES)
    if not math.isfinite(duration) or duration < minimum:
        raise argparse.ArgumentTypeError(
            f"duration must be finite and at least {minimum:.6f} seconds "
            "to measure two or more frames at each display rate"
        )
    return duration


def _delay_seconds(value: str | float) -> float:
    delay = float(value)
    if not math.isfinite(delay) or delay < 0:
        raise argparse.ArgumentTypeError("delay must be finite and non-negative")
    return delay


def result(sequence, timestamp, width, height, branch):
    if branch == "face":
        objects = tuple(
            GroundedObject(
                name, "known_person", NormalizedBBox(x, 0.3, x + 0.12, 0.55),
                0.95, {
                    "subject_id": name, "subject_ids": [name],
                    "person_id": name, "name": name, "status": "known",
                    "face_observation_state": "observed",
                },
            )
            for name, x in (("Alice", 0.15), ("Bob", 0.65))
        )
        judgment = EventJudgment("stranger", False, 0.95, "Known people", ())
    else:
        objects = (GroundedObject(
            "cigarette", "cigarette", NormalizedBBox(0.2, 0.5, 0.22, 0.53),
            0.95, {"subject_id": "Alice", "subject_ids": ["Alice"]},
        ),)
        judgment = EventJudgment("smoking", True, 0.95, "Smoking", ("cigarette",))
    return AnalysisResult(
        message_id=f"{branch}-{sequence}", source_id="synthetic",
        stream_session_id="benchmark", sequence=sequence + 1, emitted_at=STAMP,
        frame=FrameReference(sequence, timestamp, STAMP, width, height),
        judgments=(judgment,), objects=objects, model_name="synthetic",
        model_latency_seconds=0, analysis_kind="selected_features",
        missing_event_labels=("smoking",) if branch == "face" else ("stranger",),
    )


def run(fps, duration, delay, width, height):
    duration = _duration_seconds(duration)
    delay = _delay_seconds(delay)
    rng = np.random.default_rng(123)
    texture = rng.integers(0, 256, (height // 4, width // 4, 3), dtype=np.uint8)
    background = cv2.resize(texture, (width, height), interpolation=cv2.INTER_LINEAR)
    source_started, encodes, published = {}, [], []
    last_done = threading.Event()
    frame_count = round(fps * duration)
    rendered_person_frames = 0
    render_errors = 0

    def render(jpeg, analysis):
        nonlocal rendered_person_frames, render_errors
        try:
            arguments = _analysis_overlay_arguments(analysis)
            if len(arguments["face_objects"]) != 2 or len(arguments["objects"]) != 1:
                raise ValueError(
                    "expected two face boxes and one independent event box"
                )
            rendered = _render_analysis_jpeg(jpeg, analysis, quality=90)
        except Exception:
            render_errors += 1
            raise
        rendered_person_frames += 1
        return rendered

    def publish(packet):
        now = time.monotonic()
        published.append((
            packet.frame_sequence, now,
            now - source_started[packet.frame_sequence] - delay,
            packet.annotation_mode,
        ))
        if packet.frame_sequence == frame_count - 1:
            last_done.set()

    composer = DelayedOverlayComposer(
        delay_seconds=delay, max_gap_seconds=0.8, output_fps=fps,
        render=render, publish=publish,
    )
    # Warm up OpenCV/fonts outside the paced measurement.
    warm_face = result(0, 0, width, height, "face")
    jpeg, _, _ = encode_jpeg(background, max_image_side=None, quality=90)
    for _ in range(5):
        _render_analysis_jpeg(
            jpeg, warm_face, quality=90,
        )
    began = time.monotonic()
    next_model_sample = 0
    sample_count = 0
    try:
        for sequence in range(frame_count):
            timestamp = sequence / fps
            remaining = began + timestamp - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            source_started[sequence] = time.monotonic()
            # Distinct moving pixels for every native source presentation.
            image = background.copy()
            x = (sequence * 13) % (width - 80)
            image[height - 100:height - 20, x:x + 80] = sequence % 256
            encode_started = time.monotonic()
            jpeg, _, _ = encode_jpeg(image, max_image_side=None, quality=90)
            encodes.append(time.monotonic() - encode_started)
            if timestamp + 1e-9 >= next_model_sample:
                while next_model_sample <= timestamp + 1e-9:
                    next_model_sample += 1 / 2.5
                for branch in ("face", "safety"):
                    composer.submit_result(result(
                        sequence, timestamp, width, height, branch,
                    ))
                sample_count += 1
            composer.submit_frame(FramePacket(jpeg, timestamp, sequence, STAMP))
        if not last_done.wait(delay + 5):
            raise RuntimeError("final frame did not arrive within the watchdog")
        statistics = composer.stats()
    finally:
        composer.close()
    if len(published) < 2:
        raise RuntimeError("fewer than two frames arrived; cannot measure output FPS")
    intervals = published[-1][1] - published[0][1]
    excess_ms = np.array([entry[2] for entry in published]) * 1000
    modes = Counter(entry[3] for entry in published)
    rendering_verified = (
        render_errors == 0
        and rendered_person_frames == len(published)
        and set(modes).issubset({"exact", "held"})
    )
    return {
        "size": f"{width}x{height}", "input_fps": fps,
        "paced_source_seconds": duration, "configured_delay_seconds": delay,
        "source_frames": frame_count, "published_frames": len(published),
        "person_rendered_frames": rendered_person_frames,
        "dropped_frames": statistics["frames_dropped"],
        "delivered_fps": round((len(published) - 1) / intervals, 2),
        "beyond_delay_median_ms": round(float(np.median(excess_ms)), 2),
        "beyond_delay_p95_ms": round(float(np.percentile(excess_ms, 95)), 2),
        "source_encode_median_ms": round(float(np.median(encodes)) * 1000, 2),
        "synthetic_face_samples": sample_count,
        "synthetic_safety_samples": sample_count,
        "render_errors": render_errors,
        "annotation_modes": dict(modes),
        "rendering_verified": rendering_verified,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=_duration_seconds, default=4,
        help="paced source duration per rate, in seconds (default: 4)",
    )
    parser.add_argument(
        "--delay", type=_delay_seconds, default=2.5,
        help="unchanged compositor buffer delay, in seconds (default: 2.5)",
    )
    args = parser.parse_args()
    for rate in DISPLAY_RATES:
        report = run(rate, args.duration, args.delay, 1920, 1080)
        print(json.dumps(report), flush=True)
        if not report["rendering_verified"]:
            raise SystemExit(
                "Benchmark failed: some frames lacked independent face/event boxes"
            )
