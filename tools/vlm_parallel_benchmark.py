"""Fixed-window paired concurrency benchmark using production experimental adapters."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
# ruff: noqa: E402
import vlm_simple_jev_benchmark as old
from vlm_simple_jev_dataset_benchmark import read_dataset

from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)

MODEL = "qwen/qwen3-vl-8b-instruct"
KINDS = {
    "full": old.IndependentVLMInspector,
    "compact": CompactGroundingInspector,
    "events": EventJSONInspector,
}


def request_body(kind, inspector, job):
    return {
        "model": MODEL,
        "messages": inspector.build_messages(job),
        "temperature": 0,
        "max_tokens": 160 if kind == "events" else 1600,
        "stream": False,
        "provider": {"only": ["alibaba"], "allow_fallbacks": False},
    }


async def trial(client, kind, inspector, job, key, pair_start, emit):
    start = time.perf_counter()
    row = {"kind": kind, "ok": False, "start_offset_seconds": start - pair_start}
    try:
        body = request_body(kind, inspector, job)
        row["request_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()
        row["dispatch_offset_seconds"] = time.perf_counter() - pair_start
        r = await client.post(
            old.OPENROUTER,
            json=body,
            headers={"Authorization": "Bearer " + key},
            timeout=45,
        )
        row["http_status"] = r.status_code
        r.raise_for_status()
        response = r.json()
        row.update(
            choices=response.get("choices"),
            usage=response.get("usage", {}),
            provider=response.get("provider"),
            model=response.get("model"),
        )
        votes = inspector._parse(job, response)
        row.update(old.quality(votes), ok=True)
        raw = json.loads(response["choices"][0]["message"]["content"])
        row["compact_negative"] = "no_events" in raw
        row["grounded_labels"] = [
            v.event_label
            for v in votes
            if v.detected is True
            and v.subject_ids
            and len(v.subject_boxes) == len(v.subject_ids)
        ]
    except Exception as e:
        row["error_type"] = type(e).__name__
    row["total_seconds"] = time.perf_counter() - start
    row["completion_offset_seconds"] = time.perf_counter() - pair_start
    emit(row)  # Publish as soon as THIS lane finishes, before gathering its sibling.
    return row


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["repeat"] >= 0:
            groups[r["case"]].append(r)
    return {
        case: {
            "attempts": len(rr),
            "successes": sum(r["ok"] for r in rr),
            "p50_seconds": float(
                old.np.percentile([r["total_seconds"] for r in rr], 50)
            ),
            "p95_seconds": float(
                old.np.percentile([r["total_seconds"] for r in rr], 95)
            ),
            "compact_negatives": sum(r.get("compact_negative", False) for r in rr),
        }
        for case, rr in groups.items()
    }


async def run(args):
    doc = read_dataset(args.dataset)
    samples = doc["samples"]
    if args.limit:
        samples = samples[: args.limit]
    plan = 5 * len(samples) * args.repeats + 5
    if plan > args.max_requests:
        raise ValueError("request_budget_exceeded")
    key = (
        os.environ.get("OPENROUTER_API_KEY")
        or dotenv_values(args.env_file, interpolate=False).get("OPENROUTER_API_KEY")
        or ""
    ).strip()
    assert key
    jobs = {}
    for s in samples:
        video = Path(s["path"])
        assert hashlib.sha256(video.read_bytes()).hexdigest() == s["sha256"]
        job = old.make_job(video, s["endpoint"], 720, 90, "high")
        assert [hashlib.sha256(f.jpeg_bytes).hexdigest() for f in job.frames] == s[
            "frame_sha256"
        ]
        jobs[s["sample_id"]] = job
    inspectors = {
        k: cls(old.ModelAPISettings(provider="openrouter"), model=MODEL)
        for k, cls in KINDS.items()
    }
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "planned_requests": plan,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "samples": samples,
        "repeats": args.repeats,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/goodvision_harness/vlm_compact_model.py",
                ROOT / "src/goodvision_harness/vlm_independent_model.py",
                ROOT / "src/goodvision_harness/event_sensitivity.py",
            ]
        },
        "prompts": {
            k: ins.build_messages(next(iter(jobs.values())))[0]["content"]
            for k, ins in inspectors.items()
        },
        "settings": {
            "side": 720,
            "jpeg_quality": 90,
            "offsets": [-1.2, -0.8, -0.4, 0],
            "provider": "alibaba",
            "fallback": False,
            "temperature": 0,
            "retries": 0,
            "timeout": 45,
            "warmup_requests": 5,
        },
        "timing": (
            "Includes request construction and parse, excludes decode. "
            "Dual lanes are published independently; "
            "pairs never overlap other trials."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.prepare_only:
        print(json.dumps({"prepared": len(samples), "requests": plan}))
        return
    grouped = [
        [s for s in samples if s["group"] == g] for g in ("fall", "adl", "smoking")
    ]
    ordered = [
        ss[i] for i in range(max(map(len, grouped))) for ss in grouped if i < len(ss)
    ]
    rows = []
    async with httpx.AsyncClient(
        verify=old.verified_ssl_context(),
        limits=httpx.Limits(
            max_connections=8, max_keepalive_connections=8, keepalive_expiry=60
        ),
        follow_redirects=False,
    ) as client:
        with (args.output / "trials.jsonl").open("w", encoding="utf-8") as f:
            for repeat in range(-1, args.repeats):
                for index, s in enumerate(ordered[:1] if repeat < 0 else ordered):
                    cases = ["full", "compact", "events", "dual"]
                    shift = (index + max(0, repeat)) % len(cases)
                    for case in cases[shift:] + cases[:shift]:
                        pair_start = time.perf_counter()

                        def emit(row, *, case=case, s=s, repeat=repeat):
                            row.update(
                                case=("dual_" + row["kind"])
                                if case == "dual"
                                else case,
                                sample_id=s["sample_id"],
                                group=s["group"],
                                repeat=repeat,
                                window_id=f"{repeat}-{s['sample_id']}-{case}",
                                expected=s["expected"],
                            )
                            rows.append(row)
                            f.write(json.dumps(row) + "\n")
                            f.flush()
                            if len(rows) % 5 == 0:
                                (args.output / "summary.json").write_text(
                                    json.dumps(summary(rows), indent=2)
                                )
                                print(
                                    json.dumps(
                                        {
                                            "completed": len(rows),
                                            "planned": plan,
                                            "case": row["case"],
                                            "sample": s["sample_id"],
                                            "ok": row["ok"],
                                            "seconds": round(row["total_seconds"], 3),
                                        }
                                    ),
                                    flush=True,
                                )

                        if case == "dual":
                            await asyncio.gather(
                                *(
                                    trial(
                                        client,
                                        k,
                                        inspectors[k],
                                        jobs[s["sample_id"]],
                                        key,
                                        pair_start,
                                        emit,
                                    )
                                    for k in ("events", "compact")
                                )
                            )
                        else:
                            await trial(
                                client,
                                case,
                                inspectors[case],
                                jobs[s["sample_id"]],
                                key,
                                pair_start,
                                emit,
                            )
    (args.output / "summary.json").write_text(json.dumps(summary(rows), indent=2))
    print(
        json.dumps(
            {
                "completed": len(rows),
                "successes": sum(r["ok"] for r in rows),
                "cost": sum(r.get("usage", {}).get("cost", 0) or 0 for r in rows),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-requests", type=int, default=455)
    p.add_argument("--limit", type=int)
    p.add_argument("--prepare-only", action="store_true")
    asyncio.run(run(p.parse_args()))
