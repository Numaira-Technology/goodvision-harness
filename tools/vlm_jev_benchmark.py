"""Opt-in paired benchmark: existing JSON vs text observation + Jev.

Serial alternating calls, persistent transport, identical four-frame inputs,
excluded warmup pair, no retries. Credentials and images are never saved.
"""

# ruff: noqa: E402
# Local source must precede any installed editable checkout.
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from goodvision_harness.jev_observation import (
    DESCRIPTION_PROMPT,
    TEMPORAL_DESCRIPTION_PROMPT,
    assemble_response,
    build_jev_questions,
    observation_subjects,
)
from goodvision_harness.model_api import ModelAPISettings, nitro_provider_preferences
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_model import _ResponseValidationError
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.vlm_independent_model import IndependentVLMInspector
from goodvision_harness.vlm_transport import create_vlm_http_client

LABELS = ("fight", "smoking", "fall", "weapon")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
TYPESAFE = "https://api.typesafe.ai/v1/systemone"


def make_job(video, endpoint, side, quality, smoking):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("cannot_open_video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        for offset in (1.2, 0.8, 0.4, 0):
            at = round(endpoint - offset, 3)
            if at < 0:
                raise ValueError("window_before_video")
            cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("cannot_decode_frame")
            scale = min(1, side / max(frame.shape[:2]))
            if scale < 1:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError("cannot_encode_frame")
            frames.append(
                EvidenceFrame("jev-benchmark", round(at * fps), at, jpeg.tobytes())
            )
    finally:
        cap.release()
    request = InspectionRequest(
        f"window-{endpoint}",
        (),
        "independent_source",
        LABELS,
        frames[0].timestamp_seconds,
        endpoint,
    )
    return TemporalInspection(
        "jev-benchmark",
        "offline-video",
        request,
        tuple(frames),
        event_sensitivities=tuple(
            (label, smoking if label == "smoking" else "standard") for label in LABELS
        ),
    )


async def post(client, url, key, payload):
    started = time.perf_counter()

    async def perform():
        response = await client.post(
            url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=45
        )
        response.raise_for_status()
        return response.json()

    result = await asyncio.wait_for(perform(), timeout=45)
    return result, time.perf_counter() - started


def completion_text(response):
    if response["choices"][0].get("finish_reason") != "stop":
        raise ValueError("incomplete_completion")
    value = response["choices"][0]["message"]["content"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty_completion")
    return value


def quality(verdicts):
    decisions = {}
    for label in LABELS:
        values = [v.detected for v in verdicts if v.event_label == label]
        decisions[label] = (
            True if True in values else None if not values or None in values else False
        )
    return {
        "complete_labels": {v.event_label for v in verdicts} == set(LABELS),
        "localization_complete": bool(verdicts)
        and all(not v.grounding_issues for v in verdicts),
        "verdict_count": len(verdicts),
        "subject_count": len({sid for v in verdicts for sid in v.subject_ids}),
        "decisions": decisions,
    }


async def trial(client, inspector, job, mode, keys, args):
    started = time.perf_counter()
    record = {"mode": mode, "endpoint": job.request.end_seconds, "ok": False}
    try:
        messages = inspector.build_messages(job)
        if mode == "jev":
            messages[0]["content"] = (
                TEMPORAL_DESCRIPTION_PROMPT
                if args.description_style == "temporal"
                else DESCRIPTION_PROMPT
            )
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
            "provider": nitro_provider_preferences(),
        }
        response, seconds = await post(client, OPENROUTER, keys[0], payload)
        record.update(
            vlm_seconds=seconds,
            provider=response.get("provider"),
            response_model=response.get("model"),
            vlm_usage=response.get("usage", {}),
        )
        text = completion_text(response)
        record["vlm_text"] = text
        if mode == "jev":
            subjects = observation_subjects(text)
            questions = build_jev_questions(
                LABELS, subjects, dict(job.event_sensitivities)
            )
            jev, seconds = await post(
                client,
                TYPESAFE,
                keys[1],
                {"model": "jev-latest", "state": text, "questions": questions},
            )
            record.update(
                jev_seconds=seconds,
                jev_model=jev.get("model"),
                jev_usage=jev.get("usage"),
                jev_answers=jev.get("answers"),
                question_count=len(questions),
            )
            assembled = assemble_response(subjects, LABELS, jev["answers"])
            record["assembled_json"] = assembled
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(assembled)},
                    }
                ]
            }
        verdicts = inspector._parse(job, response)
        record.update(quality(verdicts), ok=True)
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        if isinstance(exc, _ResponseValidationError):
            record["error_code"] = exc.reason
        if isinstance(exc, httpx.HTTPStatusError):
            record["http_status"] = exc.response.status_code
        safe = {
            "missing_observation_subjects",
            "malformed_observation_subject",
            "invalid_observation_box",
            "duplicate_observation_subject",
            "incomplete_completion",
            "empty_completion",
            "jev_answer_coverage",
            "invalid_jev_answer",
        }
        if isinstance(exc, ValueError) and str(exc) in safe:
            record["error_code"] = str(exc)
    record["total_seconds"] = time.perf_counter() - started
    return record


def summarize(records):
    data = [r for r in records if not r["warmup"]]
    result = {}
    for mode in ("json", "jev"):
        rows = [r for r in data if r["mode"] == mode]
        good = [r for r in rows if r["ok"]]
        values = sorted(r["total_seconds"] for r in good)
        result[mode] = {
            "attempts": len(rows),
            "all_attempt_median_seconds": statistics.median(
                r["total_seconds"] for r in rows
            )
            if rows
            else None,
            "all_attempt_p95_seconds": float(
                np.percentile([r["total_seconds"] for r in rows], 95)
            )
            if rows
            else None,
            "successes": len(good),
            "complete_labels": sum(r.get("complete_labels", False) for r in rows),
            "localization_complete": sum(
                r.get("localization_complete", False) for r in rows
            ),
            "median_seconds": statistics.median(values) if values else None,
            "p95_seconds": float(np.percentile(values, 95)) if values else None,
            "median_vlm_seconds": statistics.median(r["vlm_seconds"] for r in good)
            if good
            else None,
            "median_jev_seconds": statistics.median(r["jev_seconds"] for r in good)
            if good and mode == "jev"
            else None,
            "median_output_tokens": statistics.median(
                r["vlm_usage"]["completion_tokens"] for r in good
            )
            if good
            else None,
            "openrouter_cost": sum(
                r.get("vlm_usage", {}).get("cost", 0) or 0 for r in rows
            ),
        }
    pairs = []
    for left in data:
        if left["mode"] != "json" or not left["ok"]:
            continue
        right = next(
            (
                r
                for r in data
                if r["mode"] == "jev"
                and r["repeat"] == left["repeat"]
                and r["endpoint"] == left["endpoint"]
                and r["ok"]
            ),
            None,
        )
        if right:
            pairs.append(
                {
                    "repeat": left["repeat"],
                    "endpoint": left["endpoint"],
                    "saved_seconds": left["total_seconds"] - right["total_seconds"],
                    "same_provider": left["provider"] == right["provider"],
                    "matching_event_decisions": sum(
                        left["decisions"][k] == right["decisions"][k] for k in LABELS
                    ),
                }
            )
    result["pairs"] = pairs
    result["jev_faster_pairs"] = sum(p["saved_seconds"] > 0 for p in pairs)
    result["median_paired_saved_seconds"] = (
        statistics.median(p["saved_seconds"] for p in pairs) if pairs else None
    )
    return result


async def run(args):
    env = dotenv_values(args.env_file, interpolate=False)
    keys = tuple(
        (os.environ.get(name) or env.get(name) or "").strip()
        for name in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY")
    )
    if not all(keys):
        raise ValueError("missing_required_api_key")
    endpoints = [float(v) for v in args.endpoints.split(",")]
    if (
        args.repeats < 1
        or not 1 <= args.side <= 2048
        or not 1 <= args.quality <= 100
        or not 1 <= args.max_tokens <= 16384
        or not endpoints
        or len(set(endpoints)) != len(endpoints)
        or any(not math.isfinite(value) or value < 1.2 for value in endpoints)
    ):
        raise ValueError("invalid_experiment_parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [
        make_job(args.video, end, args.side, args.quality, args.smoking)
        for end in endpoints
    ]
    inspector = IndependentVLMInspector(
        ModelAPISettings(provider="openrouter"),
        model=args.model,
        max_image_side=args.side,
        jpeg_quality=args.quality,
    )
    with args.video.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    manifest = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "side": args.side,
        "jpeg_quality": args.quality,
        "smoking_sensitivity": args.smoking,
        "repeats": args.repeats,
        "endpoints": endpoints,
        "video_sha256": digest,
        "frames": [
            {
                "endpoint": j.request.end_seconds,
                "sha256": [hashlib.sha256(f.jpeg_bytes).hexdigest() for f in j.frames],
            }
            for j in jobs
        ],
        "description_style": args.description_style,
        "description_prompt": (
            TEMPORAL_DESCRIPTION_PROMPT
            if args.description_style == "temporal"
            else DESCRIPTION_PROMPT
        ),
        "baseline_prompt": inspector.build_messages(jobs[0])[0]["content"],
        "protocol": (
            "nonstream both paths; persistent shared connection; "
            "sequential alternating pairs; one excluded warmup pair; no retries; "
            "total includes message preparation, both requests, JSON assembly "
            "and production validation; excludes video decode and later "
            "tracking/event confirmation"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    records = []
    async with create_vlm_http_client() as client:
        with (args.output / "trials.jsonl").open("w", encoding="utf-8") as stream:
            schedule = [(-1, jobs[0])] + [
                (repeat, job) for repeat in range(args.repeats) for job in jobs
            ]
            for index, (repeat, job) in enumerate(schedule):
                modes = (
                    ("json", "jev")
                    if (index + max(0, repeat)) % 2 == 0
                    else ("jev", "json")
                )
                for mode in modes:
                    record = await trial(client, inspector, job, mode, keys, args)
                    record.update(repeat=repeat, warmup=repeat == -1)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                k: record[k]
                                for k in (
                                    "mode",
                                    "endpoint",
                                    "repeat",
                                    "ok",
                                    "total_seconds",
                                    "decisions",
                                    "error_type",
                                    "http_status",
                                )
                                if k in record
                            }
                        ),
                        flush=True,
                    )
                (args.output / "summary.json").write_text(
                    json.dumps(summarize(records), indent=2), encoding="utf-8"
                )
                if repeat == -1 and any(not r["ok"] for r in records):
                    raise RuntimeError(
                        "warmup_failed_inspect_safe_results_before_spending_more"
                    )
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-vl-30b-a3b-instruct:nitro")
    parser.add_argument("--endpoints", default="4.0,9.6,12.8,20.0,24.0,42.4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--side", type=int, default=720)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--smoking", choices=("standard", "high"), default="high")
    parser.add_argument(
        "--description-style", choices=("brief", "temporal"), default="brief"
    )
    asyncio.run(run(parser.parse_args()))
