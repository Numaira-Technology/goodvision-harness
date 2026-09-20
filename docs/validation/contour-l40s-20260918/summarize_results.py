"""Summarize retained real-weight evidence without hiding negative comparisons."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--results",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "tmp/l40s-20260918/results",
)
args = parser.parse_args()
root = args.results
reports = {}
for filename in (
    "baseline-b1-r1",
    "native-off-b1-r1",
    "native-on-b1-r1",
    "native-on-b4-r1",
    "final-on-b1-r1",
):
    path = root / (filename + ".json")
    if path.exists():
        reports[filename] = json.loads(path.read_text())

summary = {
    "scope": "official_checkpoint_native_and_paced_gateway_tests",
    "accuracy_ground_truth_available": False,
    "shared_gpu": True,
    "two_gpus_tested_together": False,
    "native": {},
    "comparisons": {},
    "paced": {},
}
for name, report in reports.items():
    summary["native"][name] = {
        "ok": report["ok"],
        "batch_frames": report["batch_frames"],
        "fixed_person_text": report["fixed_person_text"],
        "worker_sources": report.get("worker_sources"),
        "cases": [
            {
                "lanes": case["lanes"],
                "steady_total_fps": case["steady_saturated_native_fps"],
                "steady_fps_per_lane": case["steady_native_fps_per_lane"],
                "append_p50_seconds": case["append_seconds_p50"],
                "append_p95_seconds": case["append_seconds_p95"],
                "memory": case["memory_before_release"],
                "completed_frames": case["completed_frames"],
                "max_objects_per_frame": max(
                    len(frame["ids"]) for lane in case["outputs"] for frame in lane
                ),
            }
            for case in report["cases"]
        ],
    }

for left, right, label in (
    ("native-off-b1-r1", "native-on-b1-r1", "experimental_cache_toggle"),
    ("baseline-b1-r1", "native-off-b1-r1", "rejected_video_mode_change"),
    ("baseline-b1-r1", "final-on-b1-r1", "final_patch_vs_original"),
):
    if left not in reports or right not in reports:
        continue
    rows = []
    for a, b in zip(reports[left]["cases"], reports[right]["cases"], strict=True):
        assert a["lanes"] == b["lanes"]
        pairs = [
            (x, y)
            for c, d in zip(a["outputs"], b["outputs"], strict=True)
            for x, y in zip(c, d, strict=True)
        ]
        rows.append(
            {
                "lanes": a["lanes"],
                "frames": len(pairs),
                "id_mask_score_equal": all(x == y for x, y in pairs),
                "different_frames": sum(x != y for x, y in pairs),
            }
        )
    summary["comparisons"][label] = rows

for fps in (1, 3):
    path = root / f"paced-final-{fps}fps.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    rows = report["results"]
    delays = sorted(
        x["source_to_result_seconds"] for row in rows for x in row["completed"]
    )
    summary["paced"][str(fps)] = {
        "ok": report["ok"],
        "seconds": report["seconds"],
        "lanes": report["lanes"],
        "eof_drain_seconds": report.get("eof_drain_seconds"),
        "captured": sum(row.get("captured_frames", 0) for row in rows),
        "completed": sum(len(row["completed"]) for row in rows),
        "dropped": sum(len(row["dropped_sequences"]) for row in rows),
        "per_lane": [
            {
                k: row.get(k)
                for k in (
                    "lane",
                    "during_capture_fps",
                    "source_to_result",
                    "captured_frames",
                )
            }
            | {
                "completed": len(row["completed"]),
                "dropped": len(row["dropped_sequences"]),
            }
            for row in rows
        ],
        "all_completion_delays_seconds": delays,
    }

target = Path(__file__).with_name("real-summary.json")
target.write_text(json.dumps(summary, indent=2) + "\n")
print(target)
