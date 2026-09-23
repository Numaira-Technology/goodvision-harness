"""Evaluate event episode JSON files with explicit camera-hour exposure."""

import argparse
import json
import sys
from pathlib import Path

from .event_metrics import EventEpisode, evaluate_events


def _episodes(path: Path) -> list[EventEpisode]:
    with path.open("rb") as handle:
        raw = handle.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("episode file exceeds 8 MiB")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) > 4096:
        raise ValueError("episode file must be an array of at most 4096 entries")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("subject_ids"), list):
            raise ValueError("each episode needs an object and actor list")
        result.append(EventEpisode(**{**row, "subject_ids": tuple(row["subject_ids"])}))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("--camera-hours", type=float, required=True)
    parser.add_argument("--min-temporal-iou", type=float, default=0.3)
    parser.add_argument("--ignore-actors", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_events(
            _episodes(args.predictions),
            _episodes(args.targets),
            camera_hours=args.camera_hours,
            min_temporal_iou=args.min_temporal_iou,
            match_actors=not args.ignore_actors,
        )
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
    except (OSError, ValueError, TypeError, OverflowError):
        print(
            "event evaluation: invalid episodes, exposure, or output path",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
