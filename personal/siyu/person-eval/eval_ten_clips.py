from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from goodvision_harness.config import load_config
from goodvision_harness.grounding import AIClubMultiGroundingClient, SAFETY_EVENT_LABELS
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from low_resolution_model_eval import MeasuredTransport, prepare_frame

VIDEOS = ROOT / "tmp" / "person-eval-videos"
OUTPUT = HERE / "person-eval-10clips.json"
TRIALS_OUTPUT = HERE / "person-eval-trials.json"

# Same channel as config/INFERENCE_PROVIDER. Only the model ID changes.
# Claude: Fable 5.1 (strongest overall) + Opus 5 (strongest single-model).
TRIALS = (
    {"id": "gpt", "model": "gpt-5.5"},
    {"id": "gemini", "model": "gemini-3.8-flash"},
    {"id": "fable", "model": "ClaudeFable5.1"},
    {"id": "opus", "model": "ClaudeOpus5"},
    {"id": "deepseek", "model": "DeepSeek-V4.1-Flash"},
)

# Annotation-balanced set: 3 NTU fights + CAVIAR fall/bag/crowd/walk.
# RWF NonFight (cpFWuHXF) is not required; Walk1/Meet_Crowd are the negatives.
CLIPS = [
    {
        "id": "ntu:fight_0472",
        "file_glob": "*fight_0472*",
        "source": "fixed_conflict50 / ntu_cctv_fights",
        "scene": "Fight throughout the clip",
        "samples": [
            {"timestamp": 8.0, "expected": {"fight": True, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 25.0, "expected": {"fight": True, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "ntu:fight_0822",
        "file_glob": "*fight_0822*",
        "source": "fixed_conflict50 / ntu_cctv_fights",
        "scene": "Quiet then fight",
        "samples": [
            {"timestamp": 1.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 6.0, "expected": {"fight": True, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "ntu:fight_0912",
        "file_glob": "*fight_0912*",
        "source": "fixed_conflict50 / ntu_cctv_fights",
        "scene": "CCTV fight after a delay",
        "samples": [
            {"timestamp": 5.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 16.0, "expected": {"fight": True, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "urfall:fall-01",
        "file": "fall-01-cam0-rgb.mp4",
        "source": "UR Fall Detection Dataset (University of Rzeszow)",
        "scene": "Staged indoor fall, side camera RGB half",
        "samples": [
            {"timestamp": 1.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 4.0, "expected": {"fight": False, "fall": True, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "urfall:fall-02",
        "file": "fall-02-cam0-rgb.mp4",
        "source": "UR Fall Detection Dataset (University of Rzeszow)",
        "scene": "Second staged indoor fall",
        "samples": [
            {"timestamp": 0.8, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 2.5, "expected": {"fight": False, "fall": True, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "caviar:Rest_FallOnFloor",
        "file": "Rest_FallOnFloor.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "Walk then collapse",
        "samples": [
            {"timestamp": 8.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 27.6, "expected": {"fight": False, "fall": True, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "caviar:Rest_SlumpOnFloor",
        "file": "Rest_SlumpOnFloor.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "Person slumps onto the floor",
        "samples": [
            {"timestamp": 10.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 30.0, "expected": {"fight": False, "fall": True, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "caviar:LeftBag",
        "file": "LeftBag.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "Leaves a bag, not a fight",
        "samples": [
            {"timestamp": 8.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 16.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "caviar:Meet_Split_3rdGuy",
        "file": "Meet_Split_3rdGuy.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "Three people meet then split",
        "samples": [
            {"timestamp": 6.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 18.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "caviar:Meet_Crowd",
        "file": "Meet_Crowd.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "Several people walking, no fight",
        "samples": [
            {"timestamp": 4.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 10.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "caviar:Walk1",
        "file": "Walk1.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "One person walking (fight negative)",
        "samples": [
            {"timestamp": 4.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 10.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
    {
        "id": "pexels:smoking-man",
        "file": "smoking-man-pexels.mp4",
        "source": "Pexels 9644233 (ordinary camera, not CCTV)",
        "scene": "Man smoking a cigarette outdoors",
        "samples": [
            {"timestamp": 2.0, "expected": {"fight": False, "fall": False, "smoking": True, "weapon": False}},
            {"timestamp": 6.0, "expected": {"fight": False, "fall": False, "smoking": True, "weapon": False}},
        ],
    },
    {
        "id": "pexels:smoking-night",
        "file": "smoking-night-pexels.mp4",
        "source": "Pexels 9897062 (ordinary camera, not CCTV)",
        "scene": "Close-up of a man smoking at night",
        "samples": [
            {"timestamp": 3.0, "expected": {"fight": False, "fall": False, "smoking": True, "weapon": False}},
        ],
    },
    {
        "id": "commons:feldschiessen",
        "file": "weapon-feldschiessen.ogv",
        "source": "Wikimedia Commons Feldschiessen.ogv (sport shooting)",
        "scene": "People on a range holding and firing rifles",
        "samples": [
            {"timestamp": 15.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": True}},
            {"timestamp": 45.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": True}},
        ],
    },
    {
        "id": "pexels:rifle-reenactment",
        "file": "weapon-reenactment-pexels.mp4",
        "source": "Pexels 9466302 (ordinary camera, not CCTV)",
        "scene": "Men with rifles in a historical reenactment",
        "samples": [
            {"timestamp": 4.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": True}},
            {"timestamp": 10.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": True}},
        ],
    },
    {
        "id": "caviar:Browse1",
        "file": "Browse1.mpg",
        "source": "goodvision CAVIAR annotation",
        "scene": "Browsing a window (hard fall negative)",
        "samples": [
            {"timestamp": 6.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
            {"timestamp": 14.0, "expected": {"fight": False, "fall": False, "smoking": False, "weapon": False}},
        ],
    },
]


def resolve_clip_path(clip: dict) -> Path:
    if clip.get("file"):
        path = VIDEOS / clip["file"]
        if path.exists():
            return path
    pattern = clip.get("file_glob")
    if pattern:
        matches = sorted(VIDEOS.glob(pattern))
        if matches:
            return matches[0]
    raise SystemExit(
        f"missing video for {clip['id']}; put it in {VIDEOS} "
        f"(name can contain {clip.get('file_glob') or clip.get('file')})"
    )


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frames / fps if fps else 0
        return {"fps": fps, "frames": frames, "width": width, "height": height, "duration": duration}
    finally:
        cap.release()


def score_row(row: dict) -> dict:
    predicted = {item["event_label"]: item["detected"] for item in row["judgments"]}
    cells = {}
    correct = 0
    scored = 0
    for label in SAFETY_EVENT_LABELS:
        expected = row["expected"].get(label)
        pred = predicted.get(label)
        if expected is None:
            cells[label] = "unscored"
            continue
        if pred is None or row["error"]:
            cells[label] = "missing"
            continue
        scored += 1
        hit = bool(pred) == bool(expected)
        correct += int(hit)
        if expected and pred:
            cells[label] = "tp"
        elif expected and not pred:
            cells[label] = "fn"
        elif (not expected) and pred:
            cells[label] = "fp"
        else:
            cells[label] = "tn"
    return {"cells": cells, "scored": scored, "correct": correct}


def _rate(matrix: dict, kind: str) -> float | None:
    if kind == "precision":
        denom = matrix["tp"] + matrix["fp"]
        return matrix["tp"] / denom if denom else None
    denom = matrix["tp"] + matrix["fn"]
    return matrix["tp"] / denom if denom else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def summarize_records(config, records: list[dict], clip_meta: dict, wall_seconds: float) -> dict:
    labels = {label: dict(tp=0, fp=0, tn=0, fn=0, unscored=0, missing=0) for label in SAFETY_EVENT_LABELS}
    for row in records:
        for label, cell in row["score"]["cells"].items():
            labels[label][cell] += 1
    for matrix in labels.values():
        matrix["precision"] = _rate(matrix, "precision")
        matrix["recall"] = _rate(matrix, "recall")
    latencies = [row["latency_seconds"] for row in records if not row["error"]]
    per_clip = defaultdict(list)
    for row in records:
        per_clip[row["clip_id"]].append(row)
    clip_rows = []
    for clip_id, rows in per_clip.items():
        scored = sum(r["score"]["scored"] for r in rows)
        correct = sum(r["score"]["correct"] for r in rows)
        clip_rows.append({
            "clip_id": clip_id,
            **clip_meta[clip_id],
            "frames": len(rows),
            "valid": sum(not r["error"] for r in rows),
            "errors": sum(bool(r["error"]) for r in rows),
            "mean_latency": float(np.mean([r["latency_seconds"] for r in rows])),
            "max_latency": float(np.max([r["latency_seconds"] for r in rows])),
            "mean_actors": float(np.mean([r["actor_count"] for r in rows])),
            "label_accuracy": (correct / scored) if scored else None,
            "correct_labels": correct,
            "scored_labels": scored,
        })
    clip_rows.sort(key=lambda item: item["clip_id"])
    scored = sum(r["score"]["scored"] for r in records)
    correct = sum(r["score"]["correct"] for r in records)
    return {
        "trial_id": f"{config.model_api.provider}:{config.inference.model}",
        "model": config.inference.model,
        "provider": config.model_api.provider,
        "requests": len(records),
        "valid": sum(not r["error"] for r in records),
        "wall_seconds": wall_seconds,
        "p50_latency": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_latency": float(np.percentile(latencies, 95)) if latencies else None,
        "mean_latency": float(np.mean(latencies)) if latencies else None,
        "mean_actors": float(np.mean([r["actor_count"] for r in records])),
        "label_accuracy": (correct / scored) if scored else None,
        "correct_labels": correct,
        "scored_labels": scored,
        "labels": labels,
        "clips": clip_rows,
        "frames": records,
    }


def run_trial(config, jobs: list, clip_meta: dict) -> dict:
    from goodvision_harness.model_api import environment_api_key, MissingAPIKeyError

    if not environment_api_key(config.model_api):
        raise MissingAPIKeyError(
            f"missing key for {config.model_api.provider} ({config.model_api.api_key_env})"
        )
    transport = MeasuredTransport(replace(config.model_api, max_retries=0))
    client = AIClubMultiGroundingClient(config.inference, transport)
    records = []
    started = time.perf_counter()
    print(
        f"\n=== TRIAL {config.model_api.provider} / {config.inference.model} ===",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for clip, sample, path in jobs:
            def run(clip=clip, sample=sample, path=path):
                filename = path.name
                try:
                    jpeg, width, height, original, actual_ts, actual_index = prepare_frame(
                        path, sample["timestamp"], 0, 75
                    )
                    frame = SampledFrame(0, actual_index, actual_ts, jpeg, width, height)
                    transport.local.include_actors = True
                    response = client.classify_window(FrameWindow(0, (frame,)))
                    classification = response.classification
                    return {
                        "clip_id": clip["id"],
                        "file": filename,
                        "scene": clip["scene"],
                        "timestamp": sample["timestamp"],
                        "actual_timestamp": actual_ts,
                        "width": width,
                        "height": height,
                        "latency_seconds": response.latency_seconds,
                        "error": response.error,
                        "expected": sample["expected"],
                        "judgments": [
                            {
                                "event_label": event.event_label,
                                "detected": event.detected,
                                "confidence": event.confidence,
                            }
                            for event in getattr(classification, "judgments", ())
                        ],
                        "actor_count": len(getattr(classification, "actors", ()) or ()),
                        "scene_summary": getattr(classification, "scene_summary", "") or "",
                    }
                except Exception as exc:
                    return {
                        "clip_id": clip["id"],
                        "file": filename,
                        "scene": clip["scene"],
                        "timestamp": sample["timestamp"],
                        "actual_timestamp": sample["timestamp"],
                        "width": 0,
                        "height": 0,
                        "latency_seconds": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "expected": sample["expected"],
                        "judgments": [],
                        "actor_count": 0,
                        "scene_summary": "",
                    }
            futures[pool.submit(run)] = f"{clip['id']}@{sample['timestamp']}"
        for future in as_completed(futures):
            row = future.result()
            row["score"] = score_row(row)
            records.append(row)
            print(
                f"{row['clip_id']} t={row['timestamp']} "
                f"lat={row['latency_seconds']:.2f}s err={row['error']} "
                f"{row['score']['cells']} scene={row.get('scene_summary') or '-'}",
                flush=True,
            )
    return summarize_records(config, records, clip_meta, time.perf_counter() - started)


def print_comparison(trials: list[dict]) -> None:
    print("\n======== 模型对比 ========", flush=True)
    header = (
        f"{'trial':<28} {'valid':>8} {'acc':>6} "
        f"{'fight-R':>8} {'fall-R':>8} {'smoke-R':>8} {'weapon-R':>9} {'p50s':>6}"
    )
    print(header, flush=True)
    for trial in trials:
        labels = trial["labels"]
        name = trial["model"]
        if len(name) > 28:
            name = name[:27] + "…"
        print(
            f"{name:<28} {trial['valid']}/{trial['requests']:>3} "
            f"{_fmt(trial['label_accuracy']):>6} "
            f"{_fmt(labels['fight']['recall']):>8} "
            f"{_fmt(labels['fall']['recall']):>8} "
            f"{_fmt(labels['smoking']['recall']):>8} "
            f"{_fmt(labels['weapon']['recall']):>9} "
            f"{_fmt(trial['p50_latency']):>6}",
            flush=True,
        )

    keys = []
    seen = set()
    for trial in trials:
        for row in trial["frames"]:
            key = (row["clip_id"], row["timestamp"])
            if key not in seen:
                seen.add(key)
                keys.append(key)
    print("\n======== 画面说明 ========", flush=True)
    names = [item["model"].split("/")[-1] for item in trials]
    print(f"{'frame':<32} " + " | ".join(f"{name:<24}" for name in names), flush=True)
    by_trial = []
    for trial in trials:
        lookup = {(row["clip_id"], row["timestamp"]): row for row in trial["frames"]}
        by_trial.append(lookup)
    for key in keys:
        cells = []
        for lookup in by_trial:
            row = lookup.get(key) or {}
            text = (row.get("scene_summary") or row.get("error") or "-").replace("\n", " ")
            if len(text) > 24:
                text = text[:23] + "…"
            cells.append(f"{text:<24}")
        print(f"{key[0]} t={key[1]:<6} " + " | ".join(cells), flush=True)


def _select_trials(names: list[str] | None) -> list[dict]:
    wanted = {name.strip().lower() for name in names or []}
    selected = [
        dict(trial)
        for trial in TRIALS
        if not wanted or trial["id"] in wanted or trial["model"].lower() in wanted
    ]
    if names and not selected:
        raise SystemExit("unknown trial; use gpt, gemini, fable, opus, deepseek")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare safety clips across models on one channel")
    parser.add_argument(
        "--trials",
        default="",
        help="comma list: gpt,gemini,fable,opus,deepseek (default: all five)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON path for this run (default: this folder's person-eval-trials.json)",
    )
    args = parser.parse_args()
    names = [item for item in args.trials.split(",") if item.strip()] or None
    selected = _select_trials(names)

    jobs = []
    clip_meta = {}
    for clip in CLIPS:
        path = resolve_clip_path(clip)
        clip["file"] = path.name
        clip_meta[clip["id"]] = {
            **probe(path),
            "file": path.name,
            "source": clip["source"],
            "scene": clip["scene"],
        }
        for sample in clip["samples"]:
            jobs.append((clip, sample, path))

    results = []
    skipped = []
    for trial in selected:
        os.environ.pop("INFERENCE_PROVIDER", None)
        os.environ["INFERENCE_MODEL"] = trial["model"]
        config = load_config(ROOT / "config.yaml")
        try:
            summary = run_trial(config, jobs, clip_meta)
        except Exception as exc:
            skipped.append({"trial": trial, "error": f"{type(exc).__name__}: {exc}"})
            print(f"skip {trial['id']}: {exc}", flush=True)
            continue
        results.append(summary)
        OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not results:
        raise SystemExit("no trial completed")
    payload = {"trials": results, "skipped": skipped}
    dest = Path(args.output).expanduser() if args.output else TRIALS_OUTPUT
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print_comparison(results)
    print(f"\nwrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
