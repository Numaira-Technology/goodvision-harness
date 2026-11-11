"""Offline validation/selection/test reporting over a frozen durable ledger.

This command does not send requests. An inference caller supplies normalized
records through Queue; transport smoke and deployment must pass separately.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .contract import CURRENT_VERSION, EVENTS, plan_rules, rules_for, digest, file_hash, score, select_thresholds, source_files, validate_threshold_values, write_new
from .retry import Ledger


def verify_bundle(bundle):
    bundle = Path(bundle)
    hashes = json.loads((bundle / "bundle-files-sha256.json").read_text())
    for name, expected in hashes.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_hash(path) != expected:
            raise ValueError("bundle file differs: " + name)
    frozen_rules = json.loads((bundle / "rules.json").read_text())
    try:
        expected_rules = rules_for(frozen_rules.get("version"))
    except (ValueError, AttributeError) as error:
        raise ValueError("bundle protocol differs; use its retained source snapshot for historical replay") from error
    if frozen_rules != expected_rules:
        raise ValueError("bundle protocol differs; use its retained source snapshot for historical replay")
    report = json.loads((bundle / "preflight.json").read_text())
    if not report.get("offline_passed"):
        raise ValueError("offline preflight not passed")
    if not {"rules.json", "preflight.json", "source-hashes.json"} <= set(hashes):
        raise ValueError("bundle is missing required integrity records")
    for name in hashes:
        if not name.startswith("plans/") or not name.endswith(".json"):
            continue
        plan = json.loads((bundle / name).read_text())
        if plan_rules(plan) != frozen_rules:
            raise ValueError("plan protocol differs from bundle")
        if frozen_rules["version"] == CURRENT_VERSION:
            count = 24 if plan["split"] == "validation" else 202
            if {q["number"] for q in plan["questions"]} != set(range(1, count + 1)) or len(plan["questions"]) != count:
                raise ValueError("plan does not cover the frozen question cohort")
    sources = json.loads((bundle / "source-hashes.json").read_text())
    active = source_files()
    if set(sources) != set(active) or report.get("sources_sha256") != digest(sources):
        raise ValueError("frozen source inventory differs; rebuild preflight with the intended code")
    for name, path in active.items():
        snapshot = "source-snapshot/" + name
        if hashes.get(snapshot) != sources[name]:
            raise ValueError("source snapshot fingerprint differs: " + name)
        if file_hash(path) != sources[name]:
            raise ValueError("active source differs from frozen bundle: " + name
                             + "; use the retained code for replay or build a new bundle")
    models = report.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("bundle is missing preflight model/plan records")
    expected_configs, expected_plans = set(), set()
    for alias, phases in models.items():
        if Path(alias).name != alias or set(phases) != {"validation", "test"}:
            raise ValueError("invalid preflight model/plan records")
        config_name = f"configs/{alias}.json"
        expected_configs.add(config_name)
        if config_name not in hashes:
            raise ValueError("unsealed bundle config: " + config_name)
        config = json.loads((bundle / config_name).read_text())
        for split, recorded in phases.items():
            name = f"plans/{alias}-{split}.json"
            expected_plans.add(name)
            if name not in hashes:
                raise ValueError("unsealed bundle plan: " + name)
            plan = json.loads((bundle / name).read_text())
            if (recorded.get("plan_sha256") != digest(plan) or plan.get("split") != split
                    or plan.get("config_sha256") != digest(config)
                    or plan.get("rules_sha256") != digest(frozen_rules)):
                raise ValueError("preflight plan identity differs: " + name)
            for unit in plan["units"]:
                for frame in unit.get("frames", []):
                    if hashes.get(frame["path"]) != frame["sha256"]:
                        raise ValueError("unsealed bundle frame: " + frame["path"])
    for directory, expected in (("configs", expected_configs), ("plans", expected_plans)):
        actual = {str(p.relative_to(bundle)) for p in (bundle / directory).glob("*.json")}
        if actual != expected:
            raise ValueError("bundle has unrecorded or missing " + directory)
    return file_hash(bundle / "bundle-files-sha256.json")


def load_threshold_artifact(path, expected_hash, *, bundle, alias, config, bundle_digest):
    """Reject failed calibration before execution or offline test reporting."""
    if not path or not expected_hash or file_hash(path) != expected_hash:
        raise ValueError("test requires frozen validation thresholds and their SHA256")
    artifact = json.loads(Path(path).read_text())
    if (not isinstance(artifact, dict) or artifact.get("validation_only") is not True
            or artifact.get("calibration") != "selected_on_complete_validation24"):
        raise ValueError("successful validation-only calibration required; test blocked")
    for field, expected in {"config_sha256": digest(config), "bundle_sha256": bundle_digest,
                            "rules_sha256": digest(json.loads((Path(bundle) / "rules.json").read_text())), "model": config["model"]}.items():
        if artifact.get(field) != expected:
            raise ValueError("threshold configuration mismatch: " + field)
    kind = config["kind"]
    validate_threshold_values(artifact.get("thresholds"), kind)
    expected_summary = {"planned_questions": 24, "complete_valid_questions": 24,
                        "incomplete_questions": 0, "final_missing_unique_units": 0}
    if artifact.get("validation_summary") != expected_summary:
        raise ValueError("complete validation24 evidence required; test blocked")
    plan = json.loads((Path(bundle) / "plans" / f"{alias}-validation.json").read_text())
    evidence_hash = artifact.get("validation_evidence_sha256")
    if (plan.get("split") != "validation" or len(plan.get("questions", [])) != 24
            or artifact.get("validation_plan_sha256") != digest(plan)
            or not isinstance(evidence_hash, str) or len(evidence_hash) != 64
            or any(c not in "0123456789abcdef" for c in evidence_hash)):
        raise ValueError("validation plan/evidence identity differs")
    if (artifact.get("comparator") != (">" if kind == "margin" else ">=")
            or artifact.get("bag_calibration") != "uncalibrated_no_bag_validation"
            or any(artifact["thresholds"][e] != (0 if kind == "margin" else .5) for e in EVENTS[4:])):
        raise ValueError("threshold decision rule or uncalibrated bag baseline differs")
    return artifact


def context_for(plan, config, bundle_digest):
    if plan.get("config_sha256") != digest(config):
        raise ValueError("plan configuration differs")
    return {"plan_sha256": digest(plan), "config_sha256": digest(config), "rules_sha256": digest(plan_rules(plan)),
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
    # Offline re-scoring must retain the inference date, including across resumes.
    attempt_times = [r["at_unix"] for r in ledger.rows if r["type"] == "intent"
                     and type(r.get("at_unix")) in (int, float) and math.isfinite(r["at_unix"])]
    result.update(sent_unique_units=len(sent), send_coverage=len(sent) / len(plan["units"]),
                  send_semantics="durable transport handoff intent; server receipt not proven after interrupted call",
                  unattempted_unique_units=len(plan["units"]) - len(sent),
                  test_attempt_latency=timings(ledger.rows),
                  thresholds=thresholds, model=config["model"], requested_conditions=config,
                  observed_providers=sorted({r.get("metadata", {}).get("actual_provider") for r in ledger.rows
                      if r.get("metadata", {}).get("actual_provider")}),
                  run_date_utc=datetime.fromtimestamp(min(attempt_times), timezone.utc).isoformat() if attempt_times else None,
                  run_date_source="earliest timestamped durable request intent; null if unavailable",
                  report_generated_at_utc=datetime.now(timezone.utc).isoformat(),
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
    thresholds = {e: .5 for e in EVENTS}  # unused for native argmax / hard JSON answers
    if args.phase == "report" and config["kind"] in ("margin", "probability"):
        artifact = load_threshold_artifact(args.thresholds, args.thresholds_sha256,
            bundle=args.bundle, alias=args.model, config=config, bundle_digest=bundle_digest)
        thresholds = artifact["thresholds"]
    if not args.ledger.is_file():
        raise ValueError("existing inference ledger required; no synthetic or empty result")
    ledger = Ledger(args.ledger, context_for(plan, config, bundle_digest))
    try:
        observed = collect(plan, config, ledger)
        if args.phase == "select":
            result = select_thresholds(plan, observed, config["kind"], config["model"])
            result.update(calibration="selected_on_complete_validation24", config_sha256=digest(config), rules_sha256=digest(plan_rules(plan)), bundle_sha256=bundle_digest,
                          model=config["model"], selected_at=datetime.now(timezone.utc).isoformat())
        else:
            result = report(plan, config, ledger, thresholds)
            result["threshold_artifact_sha256"] = args.thresholds_sha256
        write_new(args.output, result)
    finally:
        ledger.close()
    print(json.dumps({"output": str(args.output), "sha256": file_hash(args.output)}))


if __name__ == "__main__":
    main()
