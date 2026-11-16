"""Command line for the latency investigation. It does not rent a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.latency_investigation.export_replay import export_subset
from tools.latency_investigation.metrics_diff import diff_snapshots, snapshot_from_text
from tools.latency_investigation.results_table import (
    render_csv,
    render_markdown,
    rows_from_aiperf_jsonl,
    rows_from_vllm_bench,
)
from tools.latency_investigation.summarize_attempts import summarize_tree


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summarize", help="P50/P95 table from a saved eval tree")
    summary.add_argument("root", type=Path)
    export = sub.add_parser("export", help="write a stratified replay dataset")
    export.add_argument("records", type=Path, help="JSON list of recorded requests")
    export.add_argument("destination", type=Path)
    export.add_argument("--size", type=int, default=250)
    export.add_argument("--seed", type=int, default=20261112)
    diff = sub.add_parser("metrics-diff", help="diff two saved /metrics scrapes")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    report = sub.add_parser("report", help="per-bucket table from a bench result")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--vllm-result", type=Path)
    report.add_argument("--aiperf-jsonl", type=Path)
    report.add_argument(
        "--host", type=Path, help="JSON object of run date, location, gpu, cpu_model"
    )
    report.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        found = summarize_tree(args.root)
        print(json.dumps(found, indent=2))
        return 0 if found else 1
    if args.command == "export":
        records = json.loads(args.records.read_text(encoding="utf-8"))
        manifest = export_subset(
            records, args.destination, size=args.size, seed=args.seed
        )
        print(
            json.dumps(
                {
                    "timed_count": manifest["timed_count"],
                    "warmup_count": manifest["warmup_count"],
                    "timed_500_999": manifest["timed_500_999"],
                    "manifest_sha256": (args.destination / "manifest.sha256")
                    .read_text()
                    .strip(),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "metrics-diff":
        before = snapshot_from_text(args.before.read_text(encoding="utf-8"))
        after = snapshot_from_text(args.after.read_text(encoding="utf-8"))
        print(json.dumps(diff_snapshots(before, after), indent=2))
        return 0
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    host = json.loads(args.host.read_text(encoding="utf-8")) if args.host else {}
    if args.vllm_result:
        result = json.loads(args.vllm_result.read_text(encoding="utf-8"))
        rows = rows_from_vllm_bench(result, manifest, host=host)
    elif args.aiperf_jsonl:
        rows = rows_from_aiperf_jsonl(args.aiperf_jsonl, manifest, host=host)
    else:
        parser.error("report needs --vllm-result or --aiperf-jsonl")
    text = render_markdown(rows)
    sys.stdout.write(text)
    if args.csv:
        args.csv.write_text(render_csv(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
