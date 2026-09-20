"""Validate real GPU readiness and a finite SAM3.1 batch of chronological JPEGs."""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8092")
    parser.add_argument("--frame-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=75)
    args = parser.parse_args()
    if not 1 <= len(args.images) <= 16:
        parser.error("provide 1 to 16 chronological JPEGs")
    if not math.isfinite(args.frame_interval) or args.frame_interval <= 0:
        parser.error("frame-interval must be positive and finite")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 300:
        parser.error("timeout must be in (0, 300]")
    endpoint = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        parser.error("endpoint must be HTTP(S) without embedded credentials")
    try:
        with urllib.request.urlopen(
            endpoint + "/health/video", timeout=args.timeout
        ) as response:
            health = json.load(response)
        if not (
            health.get("ready") is True
            and health.get("gpu_ready") is True
            and health.get("backend") == "sam3.1_video"
            and health.get("memory_scope") == "batch"
        ):
            raise ValueError("real SAM3.1 GPU readiness was not verified")
        session = "sam31-smoke-" + uuid.uuid4().hex
        frames = []
        total = 0
        for index, path in enumerate(args.images):
            if not 0 < path.stat().st_size <= 8 * 1024 * 1024:
                raise ValueError("JPEG size outside worker budget")
            image = path.read_bytes()
            total += len(image)
            if total > 24 * 1024 * 1024:
                raise ValueError("JPEG batch exceeds worker budget")
            frames.append(
                {
                    "frame_sequence": index,
                    "timestamp_seconds": index * args.frame_interval,
                    "jpeg_base64": base64.b64encode(image).decode("ascii"),
                }
            )
        batch_id = uuid.uuid4().hex
        payload = json.dumps(
            {"stream_session_id": session, "batch_id": batch_id, "frames": frames}
        ).encode()
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError("JSON batch exceeds worker budget")
        request = urllib.request.Request(
            endpoint + "/v1/video/batches",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        if (
            result.get("stream_session_id") != session
            or result.get("batch_id") != batch_id
            or result.get("backend") != "sam3.1_video"
            or result.get("memory_scope") != "batch"
            or len(result.get("frames", [])) != len(frames)
            or not result.get("predictor_epoch")
        ):
            raise ValueError("worker response does not match submitted video batch")
        visible = 0
        for source, snapshot in zip(frames, result["frames"], strict=True):
            if (
                snapshot.get("stream_session_id") != session
                or snapshot.get("frame_sequence") != source["frame_sequence"]
                or snapshot.get("timestamp_seconds") != source["timestamp_seconds"]
            ):
                raise ValueError("worker returned a different source frame")
            for obj in snapshot["objects"]:
                native = obj.get("attributes", {}).get("native_track", {})
                if (
                    native.get("epoch") != result["predictor_epoch"]
                    or native.get("backend") != "sam3.1_video"
                    or not isinstance(native.get("object_id"), str)
                    or obj.get("last_observed_frame_sequence")
                    != source["frame_sequence"]
                    or obj.get("last_observed_timestamp_seconds")
                    != source["timestamp_seconds"]
                ):
                    raise ValueError("worker returned invalid native source provenance")
                visible += 1
        print(
            json.dumps(
                {
                    "gpu_ready": True,
                    "backend": "sam3.1_video",
                    "memory_scope": "batch",
                    "frames": len(frames),
                    "visible_observations": visible,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "predictor_epoch": result["predictor_epoch"],
                    "accuracy_measured": False,
                }
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        print(json.dumps({"gpu_ready": False, "error": "native_video_smoke_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
