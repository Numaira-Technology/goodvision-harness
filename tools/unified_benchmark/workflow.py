"""Offline validation/selection/test reporting over a frozen durable ledger.

This command does not send requests. An inference caller supplies normalized
records through Queue; transport smoke and deployment must pass separately.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .contract import EVENTS, RULES, digest, file_hash, score, select_thresholds, write_new
from .retry import Ledger


def verify_bundle(bundle):
    bundle = Path(bundle)
    hashes = json.loads((bundle / "bundle-files-sha256.json").read_text())
    for name, expected in hashes.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_hash(path) != expected:
            raise ValueError("bundle file differs: " + name)
    if json.loads((bundle / "rules.json").read_text()) != RULES:
        raise ValueError("bundle protocol differs; use its retained source snapshot for historical replay")
    report = json.loads((bundle / "preflight.json").read_text())
    if not report.get("offline_passed"):
        raise ValueError("offline preflight not passed")
    return file_hash(bundle / "bundle-files-sha256.json")


def context_for(plan, config, bundle_digest):
    return {"plan_sha256": digest(plan), "config_sha256": digest(config), "rules_sha256": digest(RULES),
            "bundle_sha256": bundle_digest}


def collect(plan, config, ledger):
    expected = {u["id"] for u in plan["units"]}
    if {row["unit"] for row in ledger.rows} - expected:
        raise ValueError("ledger includes unexpected request units")
    return {unit["id"]: ledger.state(unit, config["kind"])[1] for unit in plan["units"]}


def timings(rows):
    # Attempt latency is separate from queue/backoff and does not measure camera latency.
    good = sorted(r["elapsed_seconds"] for r in rows if r["type"] == "result" and r.get("complete")
                  and type(r.get("elapsed_seconds")) in (int, float) and math.isfinite(r["elapsed_seconds"]))
    return {"source": "this run; successful complete transport attempts including parsing/metadata lookup; excludes queue/backoff",
            "n": len(good), "p50_seconds": good[math.ceil(.5 * len(good)) - 1] if good else None,
            "p95_seconds": good[math.ceil(.95 * len(good)) - 1] if good else None,
            "not_12_camera_throughput": True}


def report(plan, config, ledger, thresholds):
    observed = collect(plan, config, ledger)
    result = score(plan, observed, config["kind"], thresholds)
    sent = {r["unit"] for r in ledger.rows if r["type"] == "intent"}
    result.update(sent_unique_units=len(sent), send_coverage=len(sent) / len(plan["units"]),
                  send_semantics="durable transport handoff intent; server receipt not proven after interrupted call",
                  unattempted_unique_units=len(plan["units"]) - len(sent),
                  test_attempt_latency=timings(ledger.rows),
                  thresholds=thresholds, model=config["model"], requested_conditions=config,
                  observed_providers=sorted({r.get("metadata", {}).get("actual_provider") for r in ledger.rows
                      if r.get("metadata", {}).get("actual_provider")}),
                  run_date_utc=datetime.now(timezone.utc).isoformat(),
                  failed_attempt_history="retained only in attempts.jsonl; not final missing count")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("select", "report"))
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--ledger", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--thresholds", type=Path)
    p.add_argument("--thresholds-sha256")
    args = p.parse_args(argv)
    bundle_digest = verify_bundle(args.bundle)
    split = "validation" if args.phase == "select" else "test"
    plan = json.loads((args.bundle / "plans" / f"{args.model}-{split}.json").read_text())
    config = json.loads((args.bundle / "configs" / f"{args.model}.json").read_text())
    if not args.ledger.is_file():
        raise ValueError("existing inference ledger required; no synthetic or empty result")
    ledger = Ledger(args.ledger, context_for(plan, config, bundle_digest))
    try:
        observed = collect(plan, config, ledger)
        if args.phase == "select":
            result = select_thresholds(plan, observed, config["kind"], config["model"])
            result.update(calibration="selected_on_complete_validation24", config_sha256=digest(config), rules_sha256=digest(RULES), bundle_sha256=bundle_digest,
                          model=config["model"], selected_at=datetime.now(timezone.utc).isoformat())
        else:
            if config["kind"] in ("margin", "probability"):
                if not args.thresholds or not args.thresholds_sha256 or file_hash(args.thresholds) != args.thresholds_sha256:
                    raise ValueError("continuous model needs a verified frozen threshold artifact")
                artifact = json.loads(args.thresholds.read_text())
                if any(artifact.get(k) != v for k, v in {"config_sha256": digest(config), "rules_sha256": digest(RULES),
                                                        "bundle_sha256": bundle_digest, "model": config["model"]}.items()):
                    raise ValueError("validation/test configuration mismatch")
                thresholds = artifact["thresholds"]
            else:
                thresholds = {e: .5 for e in EVENTS}  # unused for native argmax / hard JSON answers
            result = report(plan, config, ledger, thresholds)
            result["threshold_artifact_sha256"] = args.thresholds_sha256
        write_new(args.output, result)
    finally:
        ledger.close()
    print(json.dumps({"output": str(args.output), "sha256": file_hash(args.output)}))


if __name__ == "__main__":
    main()
