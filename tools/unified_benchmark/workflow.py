"""Offline validation/selection/test reporting over a frozen durable ledger.

This command does not send requests. An inference caller supplies normalized
records through Queue; transport smoke and deployment must pass separately.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .contract import (
    BAG_VALIDATION_SCHEMA,
    CURRENT_VERSION,
    EVENTS,
    has_bag_validation,
    digest,
    file_hash,
    plan_rules,
    rules_for,
    score,
    select_thresholds,
    source_files,
    validate_threshold_values,
    write_new,
)
from .retry import Ledger
from . import latency
from .supplement import bundle_test_supplement, verify_supplement_plan


def _model_records(bundle, report, hashes):
    models = report.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("bundle is missing preflight model/plan records")
    models = dict(models)
    # The existing HF->vLLM converter preserves every source byte, including
    # preflight.json. Its sealed lineage can add exactly the derived profile;
    # reconstruct its plan bindings from the original HF plans, not loose files.
    lineage_name = "provenance/vllm-bundle-conversion.json"
    if lineage_name not in hashes:
        return models
    lineage = json.loads((bundle / lineage_name).read_text(encoding="utf-8"))
    source_seal = "provenance/source-bundle-files-sha256.json"
    host, target = "qwen8b_r2_host", "qwen8b_r2_vllm"
    if (
        not isinstance(lineage, dict)
        or source_seal not in hashes
        or lineage.get("source_seal") != source_seal
        or lineage.get("source_bundle_sha256") != file_hash(bundle / source_seal)
        or host not in models
        or target in models
    ):
        raise ValueError("invalid derived preflight lineage")
    inherited = json.loads((bundle / source_seal).read_text(encoding="utf-8"))
    additions = {lineage_name, source_seal, f"configs/{target}.json"} | {
        f"plans/{target}-{split}.json" for split in ("validation", "test")
    }
    if (
        not isinstance(inherited, dict)
        or set(inherited) != set(hashes) - additions
        or any(hashes.get(k) != v for k, v in inherited.items())
    ):
        raise ValueError("derived bundle changed inherited evidence")
    if inherited.get("preflight.json") != hashes["preflight.json"]:
        raise ValueError("derived preflight differs from source")
    source_cfg = json.loads((bundle / f"configs/{host}.json").read_text(encoding="utf-8"))
    target_cfg = json.loads((bundle / f"configs/{target}.json").read_text(encoding="utf-8"))
    if (
        lineage.get("source_profile_sha256") != digest(source_cfg)
        or lineage.get("derived_profile_sha256") != digest(target_cfg)
        or lineage.get("exam_unchanged") is not True
    ):
        raise ValueError("derived profile identity differs")
    models[target] = {}
    for split in ("validation", "test"):
        original = json.loads((bundle / f"plans/{host}-{split}.json").read_text(encoding="utf-8"))
        derived = {**original, "config_sha256": digest(target_cfg)}
        models[target][split] = {"plan_sha256": digest(derived)}
    return models


def _verify_plan_bindings(bundle, report, hashes, frozen_rules):
    rules_digest = digest(frozen_rules)
    models = _model_records(bundle, report, hashes)
    splits = report.get("prepared_splits", ["validation", "test"])
    if splits not in (["validation"], ["validation", "test"]):
        raise ValueError("invalid prepared split scope")
    test_rows = bundle_test_supplement(bundle, hashes, report, frozen_rules["version"])
    overlay = (report.get("dataset_context") or {}).get("bag_validation")
    if overlay is not None:
        if (not isinstance(overlay, dict)
                or not hashes.get("inputs/bag-validation.json")
                or hashes["inputs/bag-validation.json"] != overlay.get("sha256")
                or frozen_rules["version"] != CURRENT_VERSION):
            raise ValueError("bag validation manifest or model scope differs")
    expected_configs, expected_plans = set(), set()
    for alias, phases in models.items():
        if (
            not isinstance(alias, str)
            or not alias
            or Path(alias).name != alias
            or not isinstance(phases, dict)
            or set(phases) != set(splits)
        ):
            raise ValueError("invalid preflight model/plan records")
        config_name = f"configs/{alias}.json"
        expected_configs.add(config_name)
        if config_name not in hashes:
            raise ValueError("unsealed bundle config: " + config_name)
        config = json.loads((bundle / config_name).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("alias") != alias:
            raise ValueError("preflight config identity differs: " + config_name)
        for split, recorded in phases.items():
            name = f"plans/{alias}-{split}.json"
            expected_plans.add(name)
            if name not in hashes:
                raise ValueError("unsealed bundle plan: " + name)
            plan = json.loads((bundle / name).read_text(encoding="utf-8"))
            if not isinstance(plan, dict):
                raise ValueError("preflight plan must be an object: " + name)
            if plan_rules(plan) != frozen_rules:
                raise ValueError("plan protocol differs from bundle")
            if (plan.get("dataset_context") or {}).get("bag_validation") != overlay:
                raise ValueError("plan bag validation fingerprint differs from preflight")
            verify_supplement_plan(plan, test_rows, report)
            if frozen_rules["version"] == CURRENT_VERSION:
                count = (44 if has_bag_validation(plan) else 24) if split == "validation" else 202 + len(test_rows)
                questions = plan.get("questions", [])
                if ({q["number"] for q in questions} != set(range(1, count + 1))
                        or len(questions) != count):
                    raise ValueError("plan does not cover the frozen question cohort")
            if (
                not isinstance(recorded, dict)
                or not isinstance(plan, dict)
                or recorded.get("plan_sha256") != digest(plan)
                or plan.get("split") != split
                or plan.get("config_sha256") != digest(config)
                or plan.get("rules_sha256") != rules_digest
            ):
                raise ValueError("preflight plan identity differs: " + name)
            for unit in plan["units"]:
                for frame in unit.get("frames", []):
                    if hashes.get(frame["path"]) != frame["sha256"]:
                        raise ValueError("unsealed bundle frame: " + frame["path"])
    for directory, expected in (
        ("configs", expected_configs),
        ("plans", expected_plans),
    ):
        actual = {
            p.relative_to(bundle).as_posix()
            for p in (bundle / directory).glob("*.json")
        }
        if actual != expected:
            raise ValueError("bundle has unrecorded or missing " + directory)


def verify_bundle(bundle):
    bundle = Path(bundle)
    hashes = json.loads((bundle / "bundle-files-sha256.json").read_text(encoding="utf-8"))
    if not isinstance(hashes, dict):
        raise ValueError("bundle integrity record must be a mapping")
    inventory = set()
    for path in bundle.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(
                "bundle requires regular files, not links or special files"
            )
        if path.is_file() and path != bundle / "bundle-files-sha256.json":
            inventory.add(path.relative_to(bundle).as_posix())
    if inventory != set(hashes):
        raise ValueError("bundle inventory contains unsealed or missing files")
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("invalid bundle path: " + name)
        path = bundle / name
        if file_hash(path) != expected:
            raise ValueError("bundle file differs: " + name)
    frozen_rules = json.loads((bundle / "rules.json").read_text(encoding="utf-8"))
    try:
        expected_rules = rules_for(frozen_rules.get("version"))
    except (ValueError, AttributeError) as error:
        raise ValueError(
            "bundle protocol differs; use its retained source snapshot "
            "for historical replay"
        ) from error
    if frozen_rules != expected_rules:
        raise ValueError(
            "bundle protocol differs; use its retained source snapshot "
            "for historical replay"
        )
    report = json.loads((bundle / "preflight.json").read_text(encoding="utf-8"))
    if not report.get("offline_passed"):
        raise ValueError("offline preflight not passed")
    if not {"rules.json", "preflight.json", "source-hashes.json"} <= set(hashes):
        raise ValueError("bundle is missing required integrity records")
    sources = json.loads((bundle / "source-hashes.json").read_text(encoding="utf-8"))
    active = source_files()
    if set(sources) != set(active) or report.get("sources_sha256") != digest(sources):
        raise ValueError(
            "frozen source inventory differs; rebuild preflight with the intended code"
        )
    for name, path in active.items():
        snapshot = "source-snapshot/" + name
        if hashes.get(snapshot) != sources[name]:
            raise ValueError("source snapshot fingerprint differs: " + name)
        if file_hash(path) != sources[name]:
            raise ValueError(
                "active source differs from frozen bundle: "
                + name
                + "; use the retained code for replay or build a new bundle"
            )
    _verify_plan_bindings(bundle, report, hashes, frozen_rules)
    return file_hash(bundle / "bundle-files-sha256.json")


def load_threshold_artifact(path, expected_hash, *, bundle, alias, config, bundle_digest):
    """Reject failed calibration before execution or offline test reporting."""
    if not path or not expected_hash or file_hash(path) != expected_hash:
        raise ValueError("test requires frozen validation thresholds and their SHA256")
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen_rules = json.loads((Path(bundle) / "rules.json").read_text(encoding="utf-8"))
    plan = json.loads((Path(bundle) / "plans" / f"{alias}-validation.json").read_text(encoding="utf-8"))
    include_bags = has_bag_validation(plan)
    count = 44 if include_bags else 24
    if (not isinstance(artifact, dict) or artifact.get("validation_only") is not True
            or artifact.get("calibration") != f"selected_on_complete_validation{count}"):
        raise ValueError("successful validation-only calibration required; test blocked")
    for field, expected in {"config_sha256": digest(config), "bundle_sha256": bundle_digest,
                            "rules_sha256": digest(frozen_rules), "model": config["model"]}.items():
        if artifact.get(field) != expected:
            raise ValueError("threshold configuration mismatch: " + field)
    kind = config["kind"]
    validate_threshold_values(artifact.get("thresholds"), kind)
    expected_summary = {"planned_questions": count, "complete_valid_questions": count,
                        "incomplete_questions": 0, "final_missing_unique_units": 0}
    if artifact.get("validation_summary") != expected_summary:
        raise ValueError(f"complete validation{count} evidence required; test blocked")
    evidence_hash = artifact.get("validation_evidence_sha256")
    if (plan.get("split") != "validation" or len(plan.get("questions", [])) != count
            or artifact.get("validation_plan_sha256") != digest(plan)
            or not isinstance(evidence_hash, str) or len(evidence_hash) != 64
            or any(c not in "0123456789abcdef" for c in evidence_hash)):
        raise ValueError("validation plan/evidence identity differs")
    if artifact.get("comparator") != (">" if kind == "margin" else ">="):
        raise ValueError("threshold decision rule differs")
    if include_bags:
        if (kind not in {"margin", "probability"}
                or artifact.get("bag_calibration") != BAG_VALIDATION_SCHEMA or set(artifact.get("details", {})) != set(EVENTS)):
            raise ValueError("complete six-event threshold selection required")
    elif (artifact.get("bag_calibration") != "uncalibrated_no_bag_validation"
            or any(artifact["thresholds"][e] != (0 if kind == "margin" else .5) for e in EVENTS[4:])):
        raise ValueError("threshold decision rule or uncalibrated bag baseline differs")
    return artifact


def context_for(plan, config, bundle_digest):
    return {"plan_sha256": digest(plan), "config_sha256": digest(config), "rules_sha256": digest(plan_rules(plan)),
            "bundle_sha256": bundle_digest}


def collect(plan, config, ledger):
    expected = {u["id"] for u in plan["units"]}
    if {row["unit"] for row in ledger.rows} - expected:
        raise ValueError("ledger includes unexpected request units")
    return {unit["id"]: ledger.state(unit, config["kind"])[1] for unit in plan["units"]}


def timings(rows, *, split=None):
    """Request latency for one phase; unavailable outcomes are not successes."""
    rows = list(rows)
    results = [r for r in rows if r["type"] == "result"]
    completed = {(r.get("unit"), r.get("attempt")) for r in results}
    unresolved = sum(
        r["type"] == "intent" and (r.get("unit"), r.get("attempt")) not in completed
        for r in rows
    )
    good, failed, timeouts, fallback, missing = [], 0, 0, 0, 0
    for row in results:
        if not row.get("complete"):
            failed += 1
            timeouts += latency.is_timeout(row.get("error"))
            continue
        metadata = row.get("metadata") or {}
        legacy = "transport_seconds" not in metadata
        measured = row.get("elapsed_seconds") if legacy else metadata["transport_seconds"]
        if latency.finite(measured) and measured >= 0:
            good.append(measured)
            fallback += legacy
        else:
            missing += 1
    stats = latency.summary(good, failed, timeouts)
    # Unknown outcomes are not proven failures. Do not silently remove them (or
    # successful attempts without valid timing) from an all-outcome percentile.
    if unresolved or missing:
        stats["p95_with_failures_seconds"] = None
    return {
        "source": (
            "this run; one sample per recorded request attempt; transport and answer "
            "parsing, excluding request-body preparation, receipt writes, metadata "
            "lookup, worker queue and retry backoff; legacy fallback uses whole-attempt "
            "time; p95_with_failures counts recorded failures at the request timeout "
            "penalty and is unavailable if outcomes or successful timings are missing"
        ),
        **stats,
        "split": split,
        "attempts": len(results) + unresolved,
        "attempts_without_result": unresolved,
        "successful_attempts_without_timing": missing,
        "p95_with_failures_n": len(good) + failed,
        "attempt_elapsed_fallback": fallback,
        "not_12_camera_throughput": True,
    }


def report(plan, config, ledger, thresholds):
    observed = collect(plan, config, ledger)
    result = score(plan, observed, config["kind"], thresholds)
    if has_bag_validation(plan) and config["kind"] in ("margin", "probability"):
        baseline = dict.fromkeys(EVENTS, 0.0 if config["kind"] == "margin" else .5)
        label = "zero_margin_baseline" if config["kind"] == "margin" else "fixed_probability_baseline"
        result[label] = score(plan, observed, config["kind"], baseline) | {"thresholds": baseline}
    sent = {r["unit"] for r in ledger.rows if r["type"] == "intent"}
    # Offline re-scoring keeps the inference date, including across resumes.
    attempt_times = [r["at_unix"] for r in ledger.rows if r["type"] == "intent"
                     and type(r.get("at_unix")) in (int, float) and math.isfinite(r["at_unix"])]
    result.update(sent_unique_units=len(sent), send_coverage=len(sent) / len(plan["units"]),
                  send_semantics="durable transport handoff intent; server receipt not proven after interrupted call",
                  unattempted_unique_units=len(plan["units"]) - len(sent),
                  test_attempt_latency=timings(ledger.rows, split=plan.get("split")),
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
    plan = json.loads((args.bundle / "plans" / f"{args.model}-{split}.json").read_text(encoding="utf-8"))
    config = json.loads((args.bundle / "configs" / f"{args.model}.json").read_text(encoding="utf-8"))
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
            result.update(config_sha256=digest(config), rules_sha256=digest(plan_rules(plan)), bundle_sha256=bundle_digest,
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
