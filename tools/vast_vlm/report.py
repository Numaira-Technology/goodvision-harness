"""Offline local deployment accounting; missing/failed/late windows stay losses."""

import json
import math
from pathlib import Path

from tools.vlm_benchmark.accounting import complete, delivery_counts
from tools.vlm_benchmark.evaluation_contract import evaluation_contract
from tools.vlm_benchmark.model_comparison import quality, safe_path
from tools.vlm_benchmark.vision_timeliness import assess_timeliness
from tools.vlm_multistream_benchmark import LABELS, report_lane

from . import profile as p
from .metrics import distribution, token_metrics
from .runtime import write_json


def astra_rows(raw, reference):
    rows, unaligned, seen = [], 0, set()
    for item in raw:
        report = item["report"]
        if report["event_policy"]["sha256"] != p.POLICY_SHA256:
            raise ValueError("result event policy differs")
        if not complete(report):
            continue
        evidence = report["evidence"]
        ticks = tuple(e["frame_sequence"] for e in evidence)
        wid = reference["lookup"].get(ticks)
        if wid is None or any(
            type(e["frame_sequence"]) is not int
            or not math.isfinite(e["timestamp_seconds"])
            or abs(e["timestamp_seconds"] - e["frame_sequence"] / 2.5) > 1e-6
            for e in evidence
        ):
            unaligned += 1
            continue
        identity = (item["camera"], report_lane(report), wid)
        if identity in seen:
            raise ValueError("duplicate aligned judgment")
        seen.add(identity)
        predictions = {
            j["event_label"]: j["observed_detected"] for j in report["judgments"]
        }
        if len(report["judgments"]) != 6 or set(predictions) != set(LABELS):
            raise ValueError("incomplete six-label result")
        for label, pred in predictions.items():
            if type(pred) not in (bool, type(None)):
                raise ValueError("invalid event judgment")
            truth = reference["refs"][wid]["judgments"][label]["detected"]
            outcome = (
                "reference_uncertain"
                if truth is None
                else "model_uncertain"
                if pred is None
                else "tp"
                if truth and pred
                else "fn"
                if truth
                else "fp"
                if pred
                else "tn"
            )
            rows.append({"outcome": outcome})
    return {**quality(rows), "unaligned_complete_windows": unaligned}


def analyze(root, reference):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if any(manifest["settings"].get(k) != v for k, v in p.FROZEN.items()):
        raise ValueError("result is not the frozen self-hosted baseline")
    if (
        manifest["settings"]["real"] is not True
        or manifest["event_policy"]["sha256"] != p.POLICY_SHA256
        or manifest["decoded_evidence"] != [reference["decoded_evidence"]]
        or [v["sha256"] for v in manifest["videos"]] != [p.VIDEO_SHA256]
        or reference["policy"]["sha256"] != p.POLICY_SHA256
        or manifest.get("http_timeouts") != {"events": 20, "compact": 20}
        or manifest.get("transport") != "sse-v1"
    ):
        raise ValueError("simulation, policy or decoded evidence mismatch")
    trials, seen = [], set()
    for job in summary["jobs"]:
        if job["phase"] != "measurement":
            continue
        key = (job["mode"], job["repeat"])
        if key in seen or key not in {
            (m, r) for m in ("events", "compact", "parallel") for r in (1, 2)
        }:
            raise ValueError("unexpected/duplicate trial")
        seen.add(key)
        raw = [
            json.loads(line)
            for line in safe_path(root, job["raw_reports"]).read_text().splitlines()
        ]
        lanes = (
            {"events", "grounding"}
            if job["mode"] == "parallel"
            else {"grounding" if job["mode"] == "compact" else "events"}
        )
        stage = job["result"]
        if job["streams"] != 12 or set(stage["lanes"]) != lanes:
            raise ValueError("wrong stream count or branches")
        timed = assess_timeliness(raw, lanes)
        valid = bool(
            stage["all_drained"]
            and stage["input_timing_valid"]
            and stage["result_records_missing"] == 0
            and stage.get("result_retention", {}).get("valid") is True
            and stage["shared_admission"]["active"] == 0
            and stage["shared_admission"]["waiting"] == 0
            and job.get("recording", {}).get("complete")
        )
        for lane in sorted(lanes):
            selected = [i for i in raw if report_lane(i["report"]) == lane]
            records = [i["report"] for i in selected]
            good = [r for r in records if complete(r)]
            if (
                stage["lanes"][lane]["expected_observations"] != 828
                or stage["lanes"][lane]["reports"] != len(records)
                or stage["lanes"][lane]["complete_valid_results"] != len(good)
            ):
                raise ValueError("raw counts differ from summary")
            delivery = delivery_counts(records, 828)
            trials.append(
                {
                    "mode": job["mode"],
                    "repeat": job["repeat"],
                    "lane": lane,
                    "valid": valid,
                    "planned": 828,
                    "delivery": delivery,
                    "sent_rate": delivery["http_attempted"] / 828,
                    "complete_rate": len(good) / 828,
                    "timeliness": timed,
                    "latency": distribution(
                        [r["timing"].get("source_to_publish_seconds") for r in good]
                    ),
                    "tokens": token_metrics(
                        records,
                        stage["input_schedule_span_seconds"]
                        + stage["drain_elapsed_seconds"],
                    ),
                    "astra": astra_rows(selected, reference),
                    "server_telemetry": job["telemetry"],
                }
            )
    result = {
        "version": p.VERSION,
        "evaluation_contract": evaluation_contract(),
        "reference_sha256": reference["reference_sha256"],
        "all_six_trials": len(seen) == 6,
        "trials": trials,
        "state": summary["state"],
        "limitations": [
            "Astra agreement is not human-confirmed accuracy or box accuracy.",
            "Server metrics are aggregates, not per-request pure GPU inference.",
            "Historical Alibaba routing, precision and timeouts differ.",
        ],
    }
    write_json(root / "self-hosted-comparison.json", result)
    return result
