"""Offline path: turn `goodvision-harness-cli` safety output into a capture.

Scheme B (offline, no streaming infra, no GPU) for **video A only**:

    goodvision-harness-cli --video fight.mpeg --no-realtime \
        --output rawA.jsonl

The CLI runs the safety pipeline (not the scene engine), emitting one
`VisionResult` record per window:
    {"type":"vision_result","event_label":"fight","detected":true,
     "confidence":0.9,"window_start_seconds":..,"window_end_seconds":..}

We fold those per-window judgments into capture `events` so the same
`extract`/`run_bench` answer Q1 (event_recognition) and Q4 (temporal end).

This covers only the event/temporal capabilities: the safety pipeline has no
person tracking, so counting/location (video B) still need the scene-engine
streaming scheme. The temporal "end" here is the end of the first contiguous
positive run, an approximation of the scene engine's confirmed `ended_since`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def episodes_from_vision_results(
    records: Iterable[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    """Collapse per-window judgments for `label` into contiguous positive runs.

    Each returned run becomes a capture event with first_evidence/confirmed_since
    at the run's first positive window and ended_since at the window after the run
    ends (or the last positive window's end if the run reaches EOF).
    """
    windows = [
        r for r in records
        if r.get("type", "vision_result") == "vision_result"
        and r.get("event_label") == label
    ]
    windows.sort(key=lambda r: (r.get("window_index", 0), r.get("window_start_seconds", 0.0)))

    episodes: list[dict[str, Any]] = []
    run_start: float | None = None
    last_positive_end: float | None = None
    for record in windows:
        detected = bool(record.get("detected"))
        start = float(record.get("window_start_seconds", 0.0))
        end = float(record.get("window_end_seconds", start))
        if detected:
            if run_start is None:
                run_start = start
            last_positive_end = end
        elif run_start is not None:
            episodes.append(_episode(label, run_start, last_positive_end, end))
            run_start = last_positive_end = None
    if run_start is not None:  # ran to EOF while still positive
        episodes.append(_episode(label, run_start, last_positive_end, last_positive_end))
    return episodes


def _episode(
    label: str, start: float, last_positive_end: float | None, ended: float | None
) -> dict[str, Any]:
    return {
        "event_label": label,
        "subject_ids": [],
        "first_evidence": round(start, 3),
        "confirmed_since": round(start, 3),
        "ended_since": None if ended is None else round(ended, 3),
    }


def capture_from_cli(
    video_id: str, records: Iterable[dict[str, Any]], labels: tuple[str, ...]
) -> dict[str, Any]:
    records = list(records)
    events: list[dict[str, Any]] = []
    for label in labels:
        events.extend(episodes_from_vision_results(records, label))
    return {
        "video_id": video_id,
        "fps": 0.0,
        "duration_seconds": None,
        "frames": [],  # safety pipeline has no per-frame tracks
        "events": events,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-output", type=Path, required=True, help="VisionResult JSONL from the CLI")
    parser.add_argument("--video-id", required=True, help="joins to media.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="<video_id>.json capture")
    parser.add_argument("--labels", default="fight", help="comma-separated event labels to fold")
    args = parser.parse_args(argv)

    records = [
        json.loads(line)
        for line in args.cli_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    capture = capture_from_cli(
        args.video_id, records, tuple(l.strip() for l in args.labels.split(",") if l.strip())
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(capture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[from_cli] {args.video_id}: {len(capture['events'])} events -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
