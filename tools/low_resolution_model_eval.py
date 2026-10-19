"""Replay explicitly listed local public samples through the configured real model.

This tool sends only the manifest's images to the configured inference endpoint.
It never connects to the camera URL, uses a webhook, or modifies a face gallery.
Frame truth describes visible evidence at original resolution, not guessed motion.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.config import load_config  # noqa: E402
from goodvision_harness.event_sensitivity import (  # noqa: E402
    normalize_event_sensitivities,
)
from goodvision_harness.grounding import (  # noqa: E402
    ACTOR_ASSOCIATION_RULE,
    SAFETY_EVENT_LABELS,
    AIClubMultiGroundingClient,
    parse_multi_grounded_classification,
)
from goodvision_harness.model_api import ModelAPIClient  # noqa: E402
from goodvision_harness.sliding_window import FrameWindow, SampledFrame  # noqa: E402


class MeasuredTransport(ModelAPIClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.local = threading.local()

    def chat_completion(self, **kwargs):
        self.local.usage = None
        self.local.finish_reason = None
        self.local.raw_text = None
        if not getattr(self.local, "include_actors", True):
            kwargs = copy.deepcopy(kwargs)
            removed = False
            for message in kwargs["messages"]:
                for part in (
                    message["content"] if isinstance(message["content"], list) else []
                ):
                    if (
                        part.get("type") == "text"
                        and ACTOR_ASSOCIATION_RULE in part["text"]
                    ):
                        part["text"] = part["text"].replace(ACTOR_ASSOCIATION_RULE, "")
                        removed = True
            if not removed:
                raise ValueError(
                    "actor ablation requires the exact optional actor rule"
                )
        result = super().chat_completion(**kwargs)
        self.local.usage = result.get("usage")
        choices = result.get("choices") or [{}]
        self.local.finish_reason = choices[0].get("finish_reason")
        self.local.raw_text = self.assistant_text(result)
        return result


def prepare_frame(path: Path, timestamp: float, long_side: int, quality: int):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(path))
        actual_timestamp, actual_index = timestamp, 0
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            actual_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            actual_index = max(0, round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            if not ok:
                frame = None
        finally:
            capture.release()
    if frame is None:
        raise ValueError(f"cannot read specified sample: {path.name}@{timestamp}")
    original = (frame.shape[1], frame.shape[0])
    if long_side and max(original) > long_side:
        scale = long_side / max(original)
        frame = cv2.resize(
            frame,
            (max(1, round(original[0] * scale)), max(1, round(original[1] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cannot encode sample")
    return (
        encoded.tobytes(),
        frame.shape[1],
        frame.shape[0],
        original,
        actual_timestamp,
        actual_index,
    )


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[str(row["long_side"])].append(row)
    result = {}
    for size, rows in groups.items():
        labels = {}
        for label in SAFETY_EVENT_LABELS:
            matrix = dict(tp=0, fp=0, tn=0, fn=0, unscored=0, missing=0)
            for row in rows:
                expected = row.get("expected", {}).get(label)
                predicted = next(
                    (
                        v["detected"]
                        for v in row["judgments"]
                        if v["event_label"] == label
                    ),
                    None,
                )
                if expected is None:
                    matrix["unscored"] += 1
                elif predicted is None:
                    matrix["missing"] += 1
                else:
                    key = (
                        ("tp" if predicted else "fn")
                        if expected
                        else ("fp" if predicted else "tn")
                    )
                    matrix[key] += 1
            labels[label] = matrix
        latencies = [row["latency_seconds"] for row in rows]
        result[size] = {
            "frames": len(rows),
            "valid": sum(not r["error"] for r in rows),
            "first_completed_latency_seconds": rows[0]["latency_seconds"],
            "p50_latency_seconds": float(np.percentile(latencies, 50)),
            "p95_latency_seconds": float(np.percentile(latencies, 95)),
            "frames_with_actors": sum(bool(r["actors"]) for r in rows),
            "mean_actor_count": float(np.mean([len(r["actors"]) for r in rows])),
            "labels": labels,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="0,320,160,96,64,32")
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=35)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--event-sensitivity", action="append", default=[],
                        help="label=standard|high; defaults match the service")
    parser.add_argument(
        "--actor-mode", choices=("enabled", "disabled", "both"), default="enabled"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.quality <= 100:
        parser.error("invalid concurrency or JPEG quality")
    sizes = [int(value) for value in args.sizes.split(",")]
    if any(v < 0 for v in sizes) or len(set(sizes)) != len(sizes):
        parser.error("sizes must be unique, nonnegative long-side dimensions")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = load_config(ROOT / "config.yaml")
    try:
        overrides = dict(value.split("=", 1) for value in args.event_sensitivity)
        if len(overrides) != len(args.event_sensitivity):
            raise ValueError("event sensitivity overrides must be unique")
        sensitivities = normalize_event_sensitivities(overrides, SAFETY_EVENT_LABELS)
    except ValueError as error:
        parser.error(str(error))
    transport = MeasuredTransport(
        replace(
            config.model_api,
            timeout_seconds=args.timeout,
            max_retries=0,
        )
    )
    client = AIClubMultiGroundingClient(
        config.inference, transport, event_sensitivities=sensitivities,
    )
    jobs = []
    for clip in manifest["clips"]:
        if not clip.get("public_source_url"):
            parser.error("each clip must have a documented public source")
        for sample in clip["samples"]:
            for size in sample.get("sizes", sizes):
                if size not in sizes:
                    parser.error("per-sample sizes must be included in --sizes")
                for include_actors in {
                    "enabled": (True,),
                    "disabled": (False,),
                    "both": (True, False),
                }[args.actor_mode]:
                    jobs.append((clip, sample, size, include_actors))
    if len(jobs) > args.max_requests:
        parser.error(f"{len(jobs)} requests exceed explicit bound {args.max_requests}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix(".jsonl")
    route = urlsplit(config.model_api.endpoint)
    run_metadata = {
        "schema_version": 2,
        "entrypoint": "legacy_single_frame",
        "event_sensitivities": sensitivities,
        "manifest_file": str(args.manifest.resolve()),
        "sample_inventory": [
            {
                "clip_id": clip["id"], "source_file": clip["path"],
                "timestamp": float(sample["timestamp"]), "long_side": size,
                "include_actors": include_actors,
                "expected": sample.get("expected", {}),
            }
            for clip, sample, size, include_actors in jobs
        ],
        "config_file": str((ROOT / "config.yaml").resolve()),
        "inference_settings": {
            "event_label": config.inference.event_label,
            "system_prompt": config.inference.system_prompt,
            "prompt": config.inference.prompt,
            "multi_event_prompt": config.inference.multi_event_prompt,
            "request_parameters": dict(config.inference.request_parameters),
        },
        "transport_origin": {
            "scheme": route.scheme, "host": route.hostname, "port": route.port,
        },
        "model": config.inference.model,
        "provider": config.model_api.provider,
        "sizes": sizes,
        "quality": args.quality,
        "actor_mode": args.actor_mode,
        "timeout": args.timeout,
        "workers": args.workers,
        "source_files": [
            name for name in (
                "tools/low_resolution_model_eval.py",
                "src/goodvision_harness/grounding.py",
                "src/goodvision_harness/model_api.py",
            )
        ],
        "sources": [
            {"clip_id": clip["id"], "path": clip["path"]}
            for clip in manifest["clips"]
        ],
    }
    metadata_path = args.output.with_suffix(".run.json")
    if not args.resume and any(
        path.exists() for path in (args.output, journal, metadata_path)
    ):
        parser.error("output artifacts already exist; preserve them and use --resume")
    if args.resume and not journal.exists() and any(
        path.exists() for path in (args.output, metadata_path)
    ):
        parser.error("resume journal is absent; prior request state requires review")
    if args.resume and journal.exists():
        if not metadata_path.exists():
            parser.error("resume metadata is absent; preserve request state for migration")
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous_metadata.get("schema_version") != 2:
            parser.error(
                "legacy resume metadata lacks explicit settings; "
                "preserve the journal and migrate from original evidence"
            )
        if previous_metadata != run_metadata:
            parser.error("resume settings differ; preserve the journal for review")
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    records = []
    if args.resume and journal.exists():
        records = [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
            if line
        ]
    completed = {row["key"] for row in records}
    random.Random(20260910).shuffle(jobs)

    def run(job):
        clip, sample, size, include_actors = job
        timestamp = float(sample["timestamp"])
        key = f"{clip['id']}:{timestamp:g}:{size}:q{args.quality}"
        if not include_actors:
            key += ":noactors"
        jpeg, width, height, original, actual_timestamp, actual_index = prepare_frame(
            Path(clip["path"]),
            timestamp,
            size,
            args.quality,
        )
        frame = SampledFrame(0, actual_index, actual_timestamp, jpeg, width, height)
        transport.local.include_actors = include_actors
        response = client.classify_window(FrameWindow(0, (frame,)))
        parse_error = None
        if response.error and getattr(transport.local, "raw_text", None):
            try:
                parse_multi_grounded_classification(
                    transport.local.raw_text,
                    event_labels=SAFETY_EVENT_LABELS,
                    require_bilingual=True,
                )
            except (ValueError, TypeError) as exc:
                parse_error = str(exc)
        classification = response.classification
        judgments = [
            {
                "event_label": event.event_label,
                "detected": event.detected,
                "confidence": event.confidence,
                "description": event.description,
                "objects": [
                    {
                        "label": obj.label,
                        "bbox": obj.bbox.as_dict(),
                        "actor_ids": list(obj.actor_ids),
                    }
                    for obj in event.objects
                ],
            }
            for event in getattr(classification, "judgments", ())
        ]
        return {
            "key": key,
            "clip_id": clip["id"],
            "timestamp": timestamp,
            "actual_timestamp": actual_timestamp,
            "actual_source_frame_index": actual_index,
            "long_side": size,
            "quality": args.quality,
            "include_actors": include_actors,
            "width": width,
            "height": height,
            "original_dimensions": list(original),
            "jpeg_bytes": len(jpeg),
            "latency_seconds": response.latency_seconds,
            "error": response.error,
            "branch_errors": list(response.branch_errors),
            "parse_error": parse_error,
            "invalid_model_text": (
                getattr(transport.local, "raw_text", None) if response.error else None
            ),
            "finish_reason": getattr(transport.local, "finish_reason", None),
            "usage": getattr(transport.local, "usage", None),
            "expected": sample.get("expected", {}),
            "judgments": judgments,
            "actors": [
                {
                    "actor_id": a.actor_id,
                    "bbox": a.bbox.as_dict(),
                    "confidence": a.confidence,
                }
                for a in getattr(classification, "actors", ())
            ],
        }

    jobs = [
        job
        for job in jobs
        if (
            f"{job[0]['id']}:{float(job[1]['timestamp']):g}:{job[2]}:q{args.quality}"
            + ("" if job[3] else ":noactors")
            not in completed
        )
    ]
    mode = "a" if args.resume else "w"
    started = time.monotonic()
    with (
        journal.open(mode, encoding="utf-8") as log,
        ThreadPoolExecutor(
            max_workers=args.workers,
        ) as executor,
    ):
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if len(records) % 10 == 0 or len(records) == len(completed) + len(jobs):
                print(
                    json.dumps(
                        {
                            "completed": len(records),
                            "total": len(completed) + len(jobs),
                            "failures": sum(bool(r["error"]) for r in records),
                        }
                    ),
                    flush=True,
                )
    report = {
        "scope": (
            "Real configured visual-event model on listed public samples; "
            "no local face inference"
        ),
        "manifest_file": str(args.manifest.resolve()),
        "config_model": config.inference.model,
        "configured_provider": config.model_api.provider,
        "config_file": str((ROOT / "config.yaml").resolve()),
        "request_parameters": dict(config.inference.request_parameters),
        "elapsed_seconds_this_run": time.monotonic() - started,
        "workers": args.workers,
        "actor_mode": args.actor_mode,
        "run_metadata": run_metadata,
        "source_crop": "none; explicit public sample frames",
        "records": records,
        "summary": summarize(records),
        "summary_scope": (
            "pooled across requested actor modes; "
            "use summary_by_actor_mode for ablations"
        ),
        "summary_by_actor_mode": {
            mode: summarize(
                [r for r in records if r.get("include_actors", True) == enabled]
            )
            for mode, enabled in (("enabled", True), ("disabled", False))
        },
        "limitations": [
            "Small, manually reviewed sample; no population accuracy claim",
            "Truth is original-frame visible evidence; low-resolution FN "
            "can be information loss",
            "Missing/invalid responses are counted separately from negatives",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
