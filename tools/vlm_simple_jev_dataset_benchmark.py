"""Run the unchanged five-route vision comparison on frozen video windows."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import vlm_simple_jev_benchmark as baseline  # noqa: E402


def read_dataset(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("annotation_status") != "verified_before_inference":
        raise ValueError("dataset_not_frozen")
    samples = document["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("empty_dataset")
    seen = set()
    for sample in samples:
        sid = sample["sample_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"v[0-9]{2,4}", sid):
            raise ValueError("invalid_sample_id")
        if sid in seen:
            raise ValueError("duplicate_sample_id")
        seen.add(sid)
        end = sample["endpoint"]
        if type(end) not in (int, float) or not math.isfinite(end) or end < 1.2:
            raise ValueError("invalid_endpoint")
        expected = sample["expected"]
        if set(expected) != set(baseline.LABELS) or any(
            type(value) is not bool for value in expected.values()
        ):
            raise ValueError("invalid_reference")
        if sample["group"] not in {"fall", "adl", "smoking"}:
            raise ValueError("invalid_group")
    return document


def schedule(samples, repeats):
    """Interleave source groups and rotate route order, without concurrency."""
    groups = [
        [s for s in samples if s["group"] == name]
        for name in ("fall", "adl", "smoking")
    ]
    ordered = [
        group[i]
        for i in range(max(map(len, groups)))
        for group in groups
        if i < len(group)
    ]
    cases = list(baseline.CASES)
    result = [(-2, ordered[0], "simple_vision")]
    result.extend((-1, ordered[0], case) for case in cases)
    for repeat in range(repeats):
        for index, sample in enumerate(ordered):
            shift = (1 + repeat * len(ordered) + index + repeat) % len(cases)
            for case in cases[shift:] + cases[:shift]:
                result.append((repeat, sample, case))
    return result


def summarize(rows, samples):
    reference = {s["sample_id"]: s for s in samples}
    result = baseline.summarize(rows)
    for case, summary in result.items():
        summary.pop("by_window")
        records = [
            r
            for r in rows
            if r["case"] == case and not r["warmup"] and not r["control"]
        ]
        confusion = {}
        for label in baseline.LABELS:
            counts = Counter()
            for row in records:
                actual = reference[row["sample_id"]]["expected"][label]
                counts["reference_positive" if actual else "reference_negative"] += 1
                if not row["ok"]:
                    counts["request_error"] += 1
                    continue
                predicted = row["decisions"][label]
                if predicted is None:
                    counts["unknown_positive" if actual else "unknown_negative"] += 1
                else:
                    counts[
                        "tp"
                        if actual and predicted
                        else "fn"
                        if actual
                        else "fp"
                        if predicted
                        else "tn"
                    ] += 1
            confusion[label] = {
                name: counts[name]
                for name in (
                    "reference_positive",
                    "reference_negative",
                    "tp",
                    "fn",
                    "fp",
                    "tn",
                    "unknown_positive",
                    "unknown_negative",
                    "request_error",
                )
            }
        summary["confusion"] = confusion
        summary["by_sample"] = {}
        for sid in dict.fromkeys(r["sample_id"] for r in records):
            rr = [r for r in records if r["sample_id"] == sid]
            summary["by_sample"][sid] = {
                "group": reference[sid]["group"],
                "expected": reference[sid]["expected"],
                "trials": [
                    {
                        key: r.get(key)
                        for key in (
                            "repeat",
                            "ok",
                            "decisions",
                            "total_seconds",
                            "localization_complete",
                            "subject_count",
                            "error_type",
                            "http_status",
                        )
                    }
                    for r in rr
                ],
            }
    return result


def write_summary(output, rows, samples):
    target = output / "summary.json"
    temporary = output / "summary.tmp"
    temporary.write_text(
        json.dumps(summarize(rows, samples), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


async def run(args):
    if args.repeats < 1:
        raise ValueError("invalid_repeats")
    document = read_dataset(args.dataset)
    samples = document["samples"]
    tasks = schedule(samples, args.repeats)
    if len(tasks) > args.max_requests:
        raise ValueError("request_budget_exceeded")
    env = dotenv_values(args.env_file, interpolate=False)
    key = (
        os.environ.get("OPENROUTER_API_KEY") or env.get("OPENROUTER_API_KEY") or ""
    ).strip()
    if not key:
        raise ValueError("missing_openrouter_key")
    jobs = {}
    for sample in samples:
        video = Path(sample["path"])
        if not video.is_file():
            raise ValueError("source_video_missing")
        job = baseline.make_job(video, sample["endpoint"], 720, 90, "high")
        if len(job.frames) != 4:
            raise ValueError("invalid_frame_count")
        times = [frame.timestamp_seconds for frame in job.frames]
        if (
            any(b <= a for a, b in zip(times, times[1:]))
            or times[-1] != round(sample["endpoint"], 3)
        ):
            raise ValueError("invalid_frame_order_or_endpoint")
        # make_job uses neutral source/window identifiers, never filenames,
        # dataset categories, reference labels or annotations in model messages.
        jobs[sample["sample_id"]] = job
    inspectors = {
        case: baseline.IndependentVLMInspector(
            baseline.ModelAPISettings(provider="openrouter"),
            model=model,
            max_image_side=720,
            jpeg_quality=90,
        )
        for case, (model, _kind) in baseline.CASES.items()
    }
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": "3651916",
        "baseline_script": "tools/vlm_simple_jev_benchmark.py",
        "dataset": str(args.dataset),
        "cases": baseline.CASES,
        "repeats": args.repeats,
        "request_count": len(tasks),
        "samples": samples,
        "context": baseline.CONTEXT,
        "questions": baseline.questions(),
        "source_folder_id": document["source_folder_id"],
        "selection_method": document["selection_method"],
        "annotation_status": document["annotation_status"],
        "annotation_frozen_at": document["annotation_frozen_at"],
        "protocol": {
            "max_image_side": 720,
            "upscale": False,
            "jpeg_quality": 90,
            "frame_offsets_seconds": [-1.2, -0.8, -0.4, 0],
            "smoking_sensitivity": "high",
            "other_sensitivities": "standard",
            "serial": True,
            "group_order": ["fall", "adl", "smoking"],
            "rotating_case_order": True,
            "retries": 0,
            "timeout_seconds": 45,
            "excluded_warmups": 5,
            "excluded_blank_controls": 1,
            "filenames_and_reference_labels_sent": False,
            "event_only_has_localization": False,
            "timing": (
                "Same baseline trial: message preparation, HTTP and parse; "
                "excludes video decode, tracking and downstream confirmation."
            ),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(json.dumps({"prepared": len(samples), "planned_requests": len(tasks)}))
        return
    rows = []
    async with httpx.AsyncClient(
        verify=baseline.verified_ssl_context(),
        limits=httpx.Limits(
            max_connections=8, max_keepalive_connections=8, keepalive_expiry=60
        ),
        follow_redirects=False,
    ) as client:
        with (args.output / "trials.jsonl").open("w", encoding="utf-8") as stream:
            for completed, (repeat, sample, case) in enumerate(tasks, 1):
                row = await baseline.trial(
                    client,
                    case,
                    inspectors[case],
                    jobs[sample["sample_id"]],
                    key,
                    blank=repeat == -2,
                )
                row.update(
                    sample_id=sample["sample_id"],
                    group=sample["group"],
                    repeat=repeat,
                    warmup=repeat == -1,
                    control=repeat == -2,
                )
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if completed % 5 == 1 or completed == len(tasks):
                    write_summary(args.output, rows, samples)
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "planned": len(tasks),
                            "sample": sample["sample_id"],
                            "case": case,
                            "repeat": repeat,
                            "ok": row["ok"],
                            "seconds": row["total_seconds"],
                            "decisions": row.get("decisions"),
                            "error": row.get("error_type"),
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            {
                "completed": len(rows),
                "successes": sum(r["ok"] for r in rows),
                "openrouter_reported_cost_usd": sum(
                    r.get("usage", {}).get("cost", 0) or 0 for r in rows
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=456)
    parser.add_argument("--prepare-only", action="store_true")
    asyncio.run(run(parser.parse_args()))
