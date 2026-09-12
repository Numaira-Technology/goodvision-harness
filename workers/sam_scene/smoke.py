"""Readiness and protocol check against a real SAM scene worker.

Use a chronological list of local JPEGs containing a visible person; the first
frame must contain at least one detected person. It prints measurements, not
images, and fails when the remote service cannot attest real GPU readiness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8092")
    parser.add_argument("--frame-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.frame_interval) or args.frame_interval <= 0:
        parser.error("frame-interval must be finite and positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    endpoint = args.endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        parser.error(
            "endpoint must be an HTTP(S) worker URL without embedded credentials"
        )
    try:
        with urllib.request.urlopen(
            endpoint + "/health/tracking", timeout=args.timeout
        ) as response:
            health = json.load(response)
        if health.get("ready") is not True or health.get("gpu_ready") is not True:
            raise ValueError("worker has not verified real GPU inference")
        session = "sam-smoke-" + uuid.uuid4().hex
        rows = []
        for sequence, path in enumerate(args.images):
            payload = path.read_bytes()
            timestamp = sequence * args.frame_interval
            request = urllib.request.Request(
                endpoint + "/v1/tracks",
                data=payload,
                headers={
                    "Content-Type": "image/jpeg",
                    "X-GoodVision-Stream-Session-Id": session,
                    "X-GoodVision-Frame-Sequence": str(sequence),
                    "X-GoodVision-Timestamp-Seconds": str(timestamp),
                },
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                snapshot = json.load(response)
            latency = time.perf_counter() - started
            if (
                snapshot.get("stream_session_id") != session
                or snapshot.get("frame_sequence") != sequence
                or snapshot.get("timestamp_seconds") != timestamp
            ):
                raise ValueError("worker returned a different source frame")
            objects = snapshot["objects"]
            ids = [item["track_id"] for item in objects]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate track IDs")
            for item in objects:
                box = item["bbox"]
                values = [box[key] for key in ("x1", "y1", "x2", "y2")]
                if any(
                    not math.isfinite(value) or not 0 <= value <= 1 for value in values
                ):
                    raise ValueError("invalid normalized bbox")
                if values[0] >= values[2] or values[1] >= values[3]:
                    raise ValueError("empty bbox")
                state = item["observation_state"]
                if state not in {"observed", "predicted"}:
                    raise ValueError("missing observation provenance")
                if (
                    item["last_observed_frame_sequence"] > sequence
                    or item["last_observed_timestamp_seconds"] > timestamp
                ):
                    raise ValueError("observation references a future frame")
                if state == "observed" and (
                    item["last_observed_frame_sequence"] != sequence
                    or item["last_observed_timestamp_seconds"] != timestamp
                ):
                    raise ValueError("observed object references a different frame")
            if sequence == 0 and not any(item["label"] == "person" for item in objects):
                raise ValueError(
                    "first test frame must produce a real person detection"
                )
            rows.append(
                {
                    "frame_sequence": sequence,
                    "latency_seconds": round(latency, 4),
                    "track_ids": ids,
                    "observed_count": sum(
                        item["observation_state"] == "observed" for item in objects
                    ),
                }
            )
        latencies = sorted(row["latency_seconds"] for row in rows)
        print(
            json.dumps(
                {
                    "gpu_ready": True,
                    "backend": health.get("backend"),
                    "sam3_commit": health.get("sam3_commit"),
                    "frames": rows,
                    "median_latency_seconds": statistics.median(latencies),
                    "p95_latency_seconds": latencies[
                        max(0, math.ceil(0.95 * len(latencies)) - 1)
                    ],
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {"passed": False, "error": type(exc).__name__, "detail": str(exc)}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
