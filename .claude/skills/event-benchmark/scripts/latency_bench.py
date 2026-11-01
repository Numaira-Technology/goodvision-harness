#!/usr/bin/env python3
"""Measure one request's latency with the canonical event benchmark envelopes.

Only valid model answers contribute to latency percentiles. Every attempted call
is retained, and an incomplete run exits unsuccessfully. No calls are retried.
The timer includes HTTP transport, response reading and answer validation, but
excludes image preparation, model loading and the product pipeline.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO / "tools/event_benchmark/run_eval.py"
_spec = importlib.util.spec_from_file_location("event_benchmark_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

# Reuse the current runner's registry and request settings, rather than keeping
# a separate roster of model IDs that may not belong to the accuracy exam.
CANDIDATES = runner.CANDIDATES
CHAT_URL = runner.CHAT_URL
LUNA_URL = runner.LUNA_URL
FRAME_WIDTH = runner.FRAME_WIDTH
TIMEOUT = runner.TIMEOUT
MAX_RESPONSE_BYTES = 1024 * 1024
QCORE = "Does any fighting occur in this image?"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_key(env_file):
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(env_file, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    if key:
                        return key
    except OSError:
        pass
    raise ValueError(
        "OPENROUTER_API_KEY not found in the environment or configured env file"
    )


def prepare_frame(path, *, video=False):
    """Normalize either input type to a verified 720px-wide, even-height JPEG."""
    command = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if video:
        command += ["-ss", "0.5"]
    command += [
        "-i",
        str(path),
        "-vframes",
        "1",
        "-vf",
        f"scale={FRAME_WIDTH}:-2",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    raw = subprocess.run(
        command, capture_output=True, check=True, timeout=TIMEOUT
    ).stdout
    if not raw or not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise ValueError("frame preparation did not produce a complete JPEG")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "json",
            "-i",
            "pipe:0",
        ],
        input=raw,
        capture_output=True,
        check=True,
        timeout=TIMEOUT,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("prepared frame must contain exactly one image stream")
    stream = streams[0]
    height = stream.get("height")
    if (
        stream.get("codec_name") != "mjpeg"
        or stream.get("width") != FRAME_WIDTH
        or type(height) is not int
        or height <= 0
        or height % 2
    ):
        raise ValueError(
            "prepared frame must be a 720px JPEG with positive even height"
        )
    return raw


def frame_uri(jpeg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def build_body(backend, model, uri):
    if backend == "language":
        prompt = (
            f"{QCORE} Answer only with compact JSON: "
            '{"answer": true} or {"answer": false}.'
        )
        return CHAT_URL, {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                }
            ],
            "max_tokens": runner.CHAT_MAX_TOKENS,
            "temperature": runner.TEMPERATURE,
        }
    if backend == "decision":
        return LUNA_URL, {
            "model": model,
            "state": [
                "Surveillance video frame.",
                {"type": "image_url", "image_url": {"url": uri}},
            ],
            "questions": {
                "answer": {
                    "type": "noul",
                    "instructions": QCORE,
                    "criteria": {"true": "Yes — " + QCORE, "false": "No — " + QCORE},
                }
            },
        }
    raise ValueError("unknown backend")


def invalid_constant(value):
    raise ValueError("non-finite JSON constant")


def strict_json(raw):
    return json.loads(
        raw, object_pairs_hook=runner._unique, parse_constant=invalid_constant
    )


def validate_answer(raw, backend):
    """Match run_eval v2 answer eligibility, with duplicate/non-finite rejection.

    Kept separate from runner.frame_score because that function retries and
    sleeps; doing so here would change the single-request latency measurement.
    """
    payload = strict_json(raw)
    if not isinstance(payload, dict) or payload.get("error") is not None:
        raise ValueError("invalid provider response")
    if backend == "language":
        choice = payload["choices"][0]
        if choice.get("finish_reason") != "stop" or choice.get("error") is not None:
            raise ValueError("incomplete language answer")
        content = choice["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        answer = strict_json(match.group(0)) if match else None
        if (
            not isinstance(answer, dict)
            or set(answer) != {"answer"}
            or type(answer["answer"]) is not bool
        ):
            raise ValueError("language answer must be a JSON boolean")
        return float(answer["answer"])
    if backend == "decision":
        answer = payload["answers"]["answer"]
        score = answer.get("noul")
        if (
            answer.get("type") != "noul"
            or answer.get("error") is not None
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("decision answer must be a finite probability in [0, 1]")
        return score
    raise ValueError("unknown backend")


def one_call(url, body, key, backend):
    """Record one attempt, including failures; never expose provider bodies/keys."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("oversized response")
        validate_answer(raw, backend)
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error": "http_" + str(error.code),
        }
    except (urllib.error.URLError, OSError):
        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error": "transport_error",
        }
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
        return {
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error": "invalid_response",
        }
    return {"ok": True, "elapsed_s": time.perf_counter() - started}


def pct(values, percentile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * percentile / 100
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def summarize(single, concurrent):
    single_ok = [attempt["elapsed_s"] for attempt in single if attempt["ok"]]
    concurrent_ok = [attempt["elapsed_s"] for attempt in concurrent if attempt["ok"]]
    return {
        "complete": bool(single and concurrent)
        and all(a["ok"] for a in single + concurrent),
        "statistics_basis": (
            "successful validated responses only; inspect attempted and failed counts"
        ),
        "single_inference_s": statistics.median(single_ok) if single_ok else None,
        "single_mean_s": statistics.mean(single_ok) if single_ok else None,
        "single_n_ok": len(single_ok),
        "single_n": len(single),
        "single_n_failed": len(single) - len(single_ok),
        "conc_n_ok": len(concurrent_ok),
        "conc_n": len(concurrent),
        "conc_n_failed": len(concurrent) - len(concurrent_ok),
        "p50_s": pct(concurrent_ok, 50),
        "p95_s": pct(concurrent_ok, 95),
        "single_raw": single_ok,
        "conc_raw": concurrent_ok,
        "single_attempts": single,
        "concurrent_attempts": concurrent,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frame", type=Path, help="image normalized to a 720px JPEG")
    inputs.add_argument(
        "--video", type=Path, help="clip sampled at 0.5s and normalized to a 720px JPEG"
    )
    parser.add_argument("--single-n", type=positive_int, default=20)
    parser.add_argument("--concurrency", type=positive_int, default=12)
    parser.add_argument("--rounds", type=positive_int, default=6)
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument(
        "--out",
        type=Path,
        help="write provenance, attempts and conditional summary JSON",
    )
    args = parser.parse_args(argv)
    model, backend = CANDIDATES[args.candidate]
    try:
        jpeg = prepare_frame(args.frame or args.video, video=args.video is not None)
        key = load_key(args.env_file)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error("input/credential preparation failed: " + type(error).__name__)
    url, body = build_body(backend, model, frame_uri(jpeg))
    run_at = datetime.now(timezone.utc).isoformat()
    # Freeze provenance before the first request, not after a potentially long run.
    result = {
        "schema_version": "event-latency-v2",
        "spec_version": runner.SPEC_VERSION,
        "run_at": run_at,
        "candidate": args.candidate,
        "model": model,
        "backend": backend,
        "frame_sha256": sha256_bytes(jpeg),
        "request_sha256": sha256_bytes(
            json.dumps(
                {"url": url, "body": body},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "runner_sha256": sha256_bytes(RUNNER_PATH.read_bytes()),
        "spec_sha256": sha256_bytes(RUNNER_PATH.with_name("SPEC.md").read_bytes()),
        "timing_boundary": (
            "client HTTP through response validation; excludes image preparation"
        ),
        "config": {
            "question": QCORE,
            "frame_width": FRAME_WIDTH,
            "timeout_s": TIMEOUT,
            "request_retries": 0,
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "video_timestamp_s": 0.5 if args.video else None,
        },
        "concurrency": args.concurrency,
        "rounds": args.rounds,
    }
    single = [one_call(url, body, key, backend) for _ in range(args.single_n)]
    concurrent = []
    for _ in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            concurrent.extend(
                executor.map(
                    lambda _: one_call(url, body, key, backend), range(args.concurrency)
                )
            )
    result.update(summarize(single, concurrent))
    print(f"{args.candidate}: {'complete' if result['complete'] else 'INCOMPLETE'}")
    print(
        f"Single: {result['single_n_ok']}/{result['single_n']} valid; "
        f"median {result['single_inference_s']}s"
    )
    print(
        f"@{args.concurrency}: {result['conc_n_ok']}/{result['conc_n']} valid; "
        f"P50 {result['p50_s']}s; P95 {result['p95_s']}s (valid responses only)"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"-> {args.out}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
