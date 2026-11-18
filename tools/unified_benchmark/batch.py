"""Validation44 only, or validation44 then ten rounds of the frozen test cohort.

No renting, environment rebuilding, provider changes or publication. GPU routes
require an already prepared, independently managed host and an identity probe.
"""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid

from .contract import (CURRENT_VERSION, atomic_json, digest, file_hash, has_bag_validation,
                       source_files, write_new)
from .locking import lock_exclusive
from .workflow import load_threshold_artifact, verify_bundle

SCHEMA = "event-benchmark-batch-v1"
VALIDATION_SCHEMA = "event-benchmark-validation-v1"
TERMINAL = {"blocked", "failed", "completed_incomplete", "completed", "not_started"}
LIMITS = ("phase_timeout_seconds", "stall_timeout_seconds", "batch_timeout_seconds", "cleanup_timeout_seconds")
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value == 0:
        raise ValueError("finite " + ("positive " if positive else "nonnegative ") + name + " required")
    return value


def cohort(plan):
    questions = [{k: q[k] for k in ("number", "filename", "event", "reference", "source_video_group")}
                 for q in plan["questions"]]
    frames = sorted({(f["path"], f["sha256"]) for u in plan["units"] for f in u["frames"]})
    return digest({"questions": questions, "frames": frames})


def validate_config(config):
    validation_only = config.get("schema") == VALIDATION_SCHEMA
    if (config.get("schema") not in (SCHEMA, VALIDATION_SCHEMA)
            or type(config.get("repeats")) is not int
            or config["repeats"] != (0 if validation_only else 10)):
        if validation_only:
            raise ValueError("validation-only requires zero planned test rounds")
        raise ValueError("batch-v1 requires exactly ten planned test rounds")
    for key in LIMITS:
        number(config.get(key), key)
    if (type(config.get("max_process_restarts")) is not int or not 0 <= config["max_process_restarts"] <= 10):
        raise ValueError("max_process_restarts must be an integer in 0..10")
    if (type(config.get("max_controller_restarts")) is not int
            or not 0 <= config["max_controller_restarts"] <= 10):
        raise ValueError("max_controller_restarts must be an integer in 0..10")
    number(config.get("budget_usd"), "budget_usd", positive=False)
    budget_mode = config.get("budget_mode", "full_reservation")
    if budget_mode not in ("full_reservation", "rolling_reservation"):
        raise ValueError("unknown budget mode")
    if budget_mode == "rolling_reservation" and not validation_only:
        raise ValueError("rolling reservation is restricted to validation-only API batches")
    wait = number(config.get("budget_wait_seconds", 0), "budget_wait_seconds", positive=False)
    if wait > 120 or wait >= config["stall_timeout_seconds"] or (wait and budget_mode != "rolling_reservation"):
        raise ValueError("budget wait must fit rolling validation and the stall deadline")
    initial_spent = number(config.get("initial_spent_usd", 0), "initial_spent_usd", positive=False)
    if initial_spent > config["budget_usd"]:
        raise ValueError("initial spend exceeds budget")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("explicit selected model list required")
    names, cohorts, reserved = set(), {}, 0.0
    plan = []
    for item in models:
        alias = item.get("alias")
        if not isinstance(alias, str) or not alias or alias in {".", ".."} or Path(alias).name != alias or alias in names:
            raise ValueError("unique model alias required")
        names.add(alias)
        bundle = Path(item["bundle"])
        if not bundle.is_absolute() or verify_bundle(bundle) != item.get("bundle_sha256"):
            raise ValueError("bundle identity differs: " + alias)
        cfg = read(bundle / f"configs/{alias}.json")
        route = item.get("route")
        expected = ("qwen" if cfg.get("execution") == "self_hosted_r2" else
                    "api" if cfg["backend"] in ("language", "decision") else "native")
        if route != expected:
            raise ValueError("model execution route differs")
        if budget_mode == "rolling_reservation" and route != "api":
            raise ValueError("rolling reservation requires API routes")
        py = Path(item.get("python", sys.executable))
        if not py.is_absolute() or not py.is_file() or not os.access(py, os.X_OK):
            raise ValueError("absolute executable Python path required")
        if route == "api":
            from .api_run import load_explicit_key
            load_explicit_key(item["env_file"])
            if file_hash(item["env_file"]) != item.get("credential_file_sha256"):
                raise ValueError("credential file identity differs")
            unit_cost = number(item.get("attempt_cost_upper_usd"), "attempt_cost_upper_usd", positive=False)
            if budget_mode == "rolling_reservation" and unit_cost <= 0:
                raise ValueError("rolling reservation requires a positive per-attempt bound")
            if budget_mode == "rolling_reservation" and initial_spent + 12 * unit_cost > config["budget_usd"]:
                raise ValueError("budget cannot reserve the initial twelve request slots")
            if not isinstance(item.get("cost_upper_bound_basis"), str) or not item["cost_upper_bound_basis"].strip():
                raise ValueError("verified per-attempt cost upper-bound basis required")
        else:
            unit_cost = 0
            if item.get("resource_management") != "external":
                raise ValueError("GPU batch only supports a pre-existing, externally managed host")
            if not isinstance(item.get("resource_lifetime_note"), str) or not item["resource_lifetime_note"].strip():
                raise ValueError("record host ownership or independent lease expiry/cleanup")
            number(item.get("hourly_cost_upper_usd"), "hourly_cost_upper_usd", positive=False)
            reserved += item["hourly_cost_upper_usd"] * (
                config["batch_timeout_seconds"] + config["cleanup_timeout_seconds"]) / 3600
            probe = item.get("hardware_probe")
            if (not isinstance(probe, list) or not probe
                    or any(not isinstance(a, str) or not a for a in probe)
                    or not item.get("hardware_sha256")):
                raise ValueError("read-only GPU-host probe command and expected identity hash required")
            if route == "native":
                from .native_deploy import verify_deployment
                verify_deployment(Path(item["deployment"]), item["deployment_sha256"], alias)
            elif not isinstance(item.get("endpoint"), str):
                raise ValueError("Qwen endpoint required")
        phases = ("validation",) if validation_only else ("validation", "test")
        for split in phases:
            data = read(bundle / f"plans/{alias}-{split}.json")
            supplement = (data.get("dataset_context") or {}).get("test_supplement") or {}
            count, videos = 44, 34
            if split == "test":
                count = 202 + len(supplement.get("question_numbers", []))
                videos = 192 + len({
                    q["filename"] for q in data["questions"] if q["number"] > 202
                })
            if (data.get("protocol") != CURRENT_VERSION or not has_bag_validation(data)
                    or len(data["questions"]) != count
                    or len({q["filename"] for q in data["questions"]}) != videos):
                raise ValueError(
                    f"{alias}: requires validation44/34 and the frozen test cohort"
                )
            identity = cohort(data)
            if split in cohorts and cohorts[split] != identity:
                raise ValueError("selected models do not share the same frozen cohort/frames")
            cohorts[split] = identity
            repetitions = 1 if split == "validation" else config["repeats"]
            reserved += len(data["units"]) * 3 * repetitions * unit_cost
            for index in range(repetitions):
                label = "validation" if split == "validation" else f"round-{index + 1:02}"
                plan.append({"id": alias + "/" + label, "alias": alias, "split": split,
                             "round": None if split == "validation" else index + 1,
                             "output": alias + "/" + label, "state": "pending", "invocations": 0})
    reserved += initial_spent
    if budget_mode == "full_reservation" and reserved > config["budget_usd"]:
        raise ValueError(f"worst-case reserved cost {reserved:.6f} exceeds budget")
    return {"matrix": plan, "cohorts": cohorts,
            "reserved_upper_usd": config["budget_usd"] if budget_mode == "rolling_reservation" else reserved,
            "all_attempts_upper_usd": reserved, "budget_mode": budget_mode,
            "pricing_limit": "conditional on supplied reliable bounds; no promise of provider billing correctness",
            "source_sha256": digest({name: file_hash(path) for name, path in source_files().items()})}


def hardware_check(item, output):
    if item["route"] == "api":
        return {"provider_hardware": None}
    raw = subprocess.check_output(item["hardware_probe"], timeout=30)
    data = json.loads(raw)
    if not isinstance(data, dict) or digest(data) != item["hardware_sha256"]:
        raise ValueError("GPU host identity changed")
    write_new(output, data)
    return data


def model_output(root, item, row):
    directory = root / row["output"]
    return directory / ("native" if item["route"] == "native" else item["alias"])


def phase_report(root, item, row):
    base = model_output(root, item, row)
    name = "report.json" if item["route"] == "native" else "summary.json"
    path = base / row["split"] / name
    if path.exists():
        return read(path)
    partial = base / row["split"] / "partial-summary.json"
    if partial.exists():
        return read(partial)
    if item["route"] == "native":
        paths = list((base / "invocations").glob("*/" + row["split"] + "-partial-report.json"))
        if paths:
            return read(max(paths, key=lambda p: p.stat().st_mtime_ns))
    ledger_path = base / row["split"] / "attempts.jsonl"
    if ledger_path.exists():
        from .contract import EVENTS
        from .retry import Ledger
        from .workflow import context_for, report
        bundle = Path(item["bundle"])
        cfg = read(bundle / f"configs/{item['alias']}.json")
        plan = read(bundle / f"plans/{item['alias']}-{row['split']}.json")
        thresholds = dict.fromkeys(EVENTS, 0 if cfg["kind"] == "margin" else .5)
        if row["split"] == "test" and cfg["kind"] in ("margin", "probability"):
            validation = {"output": item["alias"] + "/validation"}
            artifact = model_output(root, item, validation) / "thresholds.json"
            thresholds = read(artifact)["thresholds"]
        ledger = Ledger(ledger_path, context_for(plan, cfg, item["bundle_sha256"]), read_only=True)
        try:
            if any(r["type"] == "intent" for r in ledger.rows):
                result = report(plan, cfg, ledger, thresholds)
                if item["route"] == "api":
                    from .api_run import accounting
                    result["accounting"] = accounting(ledger)
                return result
        finally:
            ledger.close()
    return None


def verify_finished(root, item, row):
    """A saved batch state never substitutes for sealed runner evidence."""
    base = model_output(root, item, row)
    completion = base / row["split"] / "completion.json"
    seal = read(completion)
    report_name = "report.json" if item["route"] == "native" else "summary.json"
    required = {f"{row['split']}/attempts.jsonl", f"{row['split']}/{report_name}"}
    required.add(f"{row['split']}/run.json" if item["route"] == "native" else "run.json")
    if not isinstance(seal.get("files"), dict) or not required <= set(seal["files"]):
        raise ValueError("completed phase evidence anchors missing")
    identity = read(base / "run.json")
    if (identity.get("bundle_sha256") != item["bundle_sha256"]
            or identity.get("phase") != row["split"]
            or identity.get("alias", identity.get("model")) != item["alias"]):
        raise ValueError("completed round identity differs")
    for name, sha in seal["files"].items():
        p = base / name
        if p.is_symlink() or not p.resolve().is_relative_to(base.resolve()) or file_hash(p) != sha:
            raise ValueError("completed round evidence changed")
    if item["route"] == "native":
        # Native seals anchor receipts through the ledger rather than listing
        # each raw file. Reuse the runner's read-only evidence checks before
        # accepting a finished phase or allowing it to authorize test requests.
        from .native_run import preflight_resume

        bundle = Path(item["bundle"])
        cfg = read(bundle / f"configs/{item['alias']}.json")
        preflight_resume(
            item["alias"], bundle, base, cfg, item["bundle_sha256"],
            item["deployment_sha256"], row["split"],
            identity.get("initial_threshold_sha256"),
        )
    result = phase_report(root, item, row)
    plan = read(Path(item["bundle"]) / f"plans/{item['alias']}-{row['split']}.json")
    expected = len(plan["questions"])
    if result is None or result.get("planned_questions") != expected:
        raise ValueError("completed round report cohort differs")
    return result


def validate_state(state, checked, config):
    if (state.get("config_sha256") != digest(config)
            or state.get("source_sha256") != checked["source_sha256"]
            or state.get("cohorts") != checked["cohorts"]
            or state.get("reserved_upper_usd") != checked["reserved_upper_usd"]
            or len(state.get("matrix", [])) != len(checked["matrix"])):
        raise ValueError("batch identity or experiment matrix changed")
    for row, original in zip(state["matrix"], checked["matrix"]):
        if any(row.get(k) != original[k] for k in ("id", "alias", "split", "round", "output")):
            raise ValueError("batch experiment matrix changed")
        if (row.get("state") not in TERMINAL | {"pending", "running", "recovering"}
                or type(row.get("invocations")) is not int
                or not 0 <= row["invocations"] <= config["max_process_restarts"] + 1):
            raise ValueError("invalid saved batch state")
    number(state.get("started_at_unix"), "started_at_unix")


def summarize(root, config, state):
    per_model = {}
    rows = []
    validation_only = config.get("schema") == VALIDATION_SCHEMA
    selected_split = "validation" if validation_only else "test"
    for item in config["models"]:
        rounds = []
        for row in state["matrix"]:
            if row["alias"] != item["alias"] or row["split"] != selected_split:
                continue
            result = (phase_report(root, item, row)
                      if row["invocations"] and row["state"] not in {"running", "recovering"} else None)
            latency = result.get("test_attempt_latency", {}) if result else {}
            r = {"model": item["alias"], "split": row["split"], "round": row["round"], "state": row["state"],
                 "process_restarts": max(0, row["invocations"] - 1),
                 "accuracy": result.get("accuracy") if result else None,
                 "complete_valid_questions": result.get("complete_valid_questions") if result else None,
                 "missing_units": result.get("final_missing_unique_units") if result else None,
                 "p50_seconds": latency.get("p50_seconds"), "p95_seconds": latency.get("p95_seconds"),
                 "latency_n": latency.get("n"),
                 "latency_source": latency.get("source"),
                 "error": row.get("error"),
                 "wall_seconds_including_recovery": (
                     row["finished_at_unix"] - row["started_at_unix"]
                     if row.get("finished_at_unix") is not None and row.get("started_at_unix") is not None else None),
                 "known_receipt_cost_usd": (result.get("accounting", {}).get("known_receipt_cost_usd")
                                            if result and item["route"] == "api" else None),
                 "attempts_without_cost_evidence": (result.get("accounting", {}).get("attempts_without_cost_evidence")
                                                    if result and item["route"] == "api" else None)}
            r.update({name: latency.get(name) for name in (
                "failed_attempts", "timeout_attempts", "p95_with_failures_seconds",
                "failure_penalty_seconds", "percentile_method", "attempts",
                "attempts_without_result", "successful_attempts_without_timing",
                "p95_with_failures_n", "attempt_elapsed_fallback",
            )})
            rounds.append(r)
            rows.append(r)
        metrics = {}
        for key in ("accuracy", "p50_seconds", "p95_seconds"):
            values = [r[key] for r in rounds if type(r[key]) in (int, float) and math.isfinite(r[key])]
            metrics[key] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values) if values else None, "max": max(values) if values else None,
                "excluded_rounds": [r["round"] for r in rounds if r[key] is None]}
        per_model[item["alias"]] = {"rounds": rounds, "metrics": metrics,
            "states": dict(Counter(r["state"] for r in rounds)),
            "started_rounds": sum(r["invocations"] > 0 for r in state["matrix"]
                                  if r["alias"] == item["alias"] and r["split"] == selected_split),
            "valid_complete_rounds": sum(r["state"] == "completed" for r in rounds),
            "latency_aggregation": "statistics of per-round quantiles, not pooled quantiles"}
    first = config["models"][0]
    test_questions = 0 if validation_only else len(read(
        Path(first["bundle"]) / f"plans/{first['alias']}-test.json",
    )["questions"])
    result = {"batch_id": state["batch_id"], "models": per_model,
              "planned_repeats": config.get("repeats", 10), "validation_questions": 44,
              "test_questions": test_questions, "validation_only": validation_only,
              "reserved_cost_upper_usd": state["reserved_upper_usd"],
              "cost_note": "reservation is not measured cost; missing receipts are not zero cost",
              "not_12_camera_throughput": True}
    atomic_json(root / "summary.json", result)
    if rows:
        csv_name = "validation.csv" if validation_only else "rounds.csv"
        temporary = root / (csv_name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(root / csv_name)
    return result


def worker(root, task_id, authorization):
    config, state = read(root / "config.json"), read(root / "state.json")
    row = next(r for r in state["matrix"] if r["id"] == task_id)
    item = next(m for m in config["models"] if m["alias"] == row["alias"])
    directory, bundle = root / row["output"], Path(item["bundle"])
    alias, phase = item["alias"], row["split"]
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / f"worker-{row['invocations']:02}.json"
    try:
        if verify_bundle(bundle) != item["bundle_sha256"]:
            raise ValueError("worker bundle identity differs")
        cfg = read(bundle / f"configs/{alias}.json")
        validation_row = next(r for r in state["matrix"] if r["alias"] == alias and r["split"] == "validation")
        threshold, threshold_sha = None, None
        if phase == "test":
            v = verify_finished(root, item, validation_row)
            if v["complete_valid_questions"] != 44 or v["final_missing_unique_units"]:
                raise ValueError("complete validation44 required before test")
            if cfg["kind"] in ("probability", "margin"):
                threshold = model_output(root, item, validation_row) / "thresholds.json"
                threshold_sha = file_hash(threshold)
                load_threshold_artifact(threshold, threshold_sha, bundle=bundle, alias=alias,
                    config=cfg, bundle_digest=item["bundle_sha256"])
        hardware_check(item, directory / f"hardware-{row['invocations']:02}.json")
        base = model_output(root, item, row)
        resume = base.exists()
        if item["route"] == "api":
            from .api_run import execute_model, load_explicit_key
            if file_hash(item["env_file"]) != item["credential_file_sha256"]:
                raise ValueError("credential file identity changed")
            guard = None
            if config.get("budget_mode") == "rolling_reservation":
                from .budget import BudgetGuard
                guard = BudgetGuard(root, config, row["id"], base / phase, item["attempt_cost_upper_usd"])
            execute_model(alias, bundle, directory, load_explicit_key(item["env_file"]),
                threading.Event(), item["bundle_sha256"], phase=phase, resume=resume,
                threshold_path=threshold, threshold_sha256=threshold_sha, before_intent=guard)
        elif item["route"] == "qwen":
            from tools.qwen8b.eval_local import execute
            execute(alias, bundle, directory, item["endpoint"], item["bundle_sha256"], phase, resume=resume)
        else:
            from .native_run import execute
            deployment = Path(item["deployment"])
            if base.exists() and not any(base.iterdir()):
                base.rmdir()
                resume = False
            with (deployment / "evaluation.lock").open("a") as lock:
                lock_exclusive(lock)
                if resume:
                    from .retry import Ledger, PausedRun
                    from .workflow import context_for

                    plan = read(bundle / f"plans/{alias}-{phase}.json")
                    ledger = Ledger(
                        base / phase / "attempts.jsonl",
                        context_for(plan, cfg, item["bundle_sha256"]),
                        read_only=True,
                    )
                    try:
                        if any(record.get("fatal") for record in ledger.rows):
                            raise PausedRun(
                                "fatal native response retained; "
                                "automatic retries are blocked"
                            )
                    finally:
                        ledger.close()
                # The batch row directory holds supervisor logs; native_run owns its run subdirectory.
                execute(alias, bundle, deployment, base, item["deployment_sha256"], phase=phase,
                        threshold_path=threshold, threshold_hash=threshold_sha,
                        authorization=authorization, resume=resume)
        report = verify_finished(root, item, row)
        status = "completed" if report["incomplete_questions"] == 0 else "completed_incomplete"
        atomic_json(result_file, {"state": status})
        return 0
    except BaseException as error:
        from .retry import PausedRun
        fatal = isinstance(error, (ValueError, PausedRun, FileNotFoundError, PermissionError))
        atomic_json(result_file, {"state": "blocked" if fatal else "recovering",
                                  "error": type(error).__name__ + ": " + str(error)})
        return 2 if fatal else 1


def run(config_path, root, authorization, *, resume=False):
    if os.name != "posix":
        raise ValueError("supervised batch currently requires POSIX")
    config = read(config_path)
    checked = validate_config(config)
    root.mkdir(parents=True, exist_ok=resume)
    with (root / "batch.lock").open("a+") as lock:
        lock_exclusive(lock)
        config_file = root / "config.json"
        if config_file.exists():
            if not resume or read(config_file) != config:
                raise ValueError("batch configuration changed")
        else:
            write_new(config_file, config)
        state_file = root / "state.json"
        if state_file.exists():
            state = read(state_file)
            validate_state(state, checked, config)
        else:
            if any(p.name not in {"batch.lock", "config.json"} for p in root.iterdir()):
                raise ValueError("batch state missing from nonempty output; do not reset attempts")
            state = checked | {"batch_id": uuid.uuid4().hex, "config_sha256": digest(config),
                               "started_at_unix": time.time(), "authorization": authorization}
            atomic_json(state_file, state)
        def save():
            atomic_json(state_file, state)
            summarize(root, config, state)
        deadline = state["started_at_unix"] + config["batch_timeout_seconds"]
        for row in state["matrix"]:
            item = next(m for m in config["models"] if m["alias"] == row["alias"])
            if row["state"] in {"running", "recovering"} and row["invocations"]:
                saved = root / row["output"] / f"worker-{row['invocations']:02}.json"
                completion = model_output(root, item, row) / row["split"] / "completion.json"
                if completion.exists():
                    result = verify_finished(root, item, row)
                    row["state"] = "completed" if result["incomplete_questions"] == 0 else "completed_incomplete"
                    save()
                elif saved.exists() and read(saved).get("state") == "blocked":
                    row.update(read(saved))
                    save()
            if row["state"] in TERMINAL:
                if row["state"] in ("completed", "completed_incomplete"):
                    verify_finished(root, item, row)
                continue
            validation = next(r for r in state["matrix"] if r["alias"] == row["alias"] and r["split"] == "validation")
            if row["split"] == "test" and validation["state"] != "completed":
                row.update(state="blocked", error="validation44 incomplete or failed")
                save()
                continue
            if time.time() >= deadline:
                row.update(state="not_started" if not row["invocations"] else "failed", error="batch deadline")
                save()
                continue
            while row["invocations"] <= config["max_process_restarts"]:
                if time.time() >= deadline:
                    break
                row["invocations"] += 1
                row.setdefault("started_at_unix", time.time())
                row["state"] = "running" if row["invocations"] == 1 else "recovering"
                save()
                folder = root / row["output"]
                folder.mkdir(parents=True, exist_ok=True)
                read_fd, write_fd = os.pipe()
                timeout = min(config["phase_timeout_seconds"], deadline - time.time())
                command = [item.get("python", sys.executable), "-m", "tools.unified_benchmark.batch",
                           "worker", "--output", str(root), "--task-id", row["id"],
                           "--authorization-note", authorization]
                supervisor = [sys.executable, "-m", "tools.unified_benchmark.supervise",
                              "--lifetime-fd", str(read_fd), "--lock-fd", str(lock.fileno()),
                              "--timeout", str(timeout), "--cleanup-timeout", str(config["cleanup_timeout_seconds"]),
                              *command]
                try:
                    with (folder / f"process-{row['invocations']:02}.log").open("xb") as logfile:
                        child = subprocess.Popen(supervisor, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(read_fd, lock.fileno()))
                        os.close(read_fd)
                        read_fd = None
                        progress, changed = None, time.monotonic()
                        while child.poll() is None:
                            files = list(folder.rglob("attempts.jsonl")) + list(folder.rglob("*.json"))
                            current = max((p.stat().st_mtime_ns for p in files), default=0)
                            if current != progress:
                                progress, changed = current, time.monotonic()
                            if time.monotonic() - changed > config["stall_timeout_seconds"]:
                                child.terminate()
                                break
                            time.sleep(.25)
                        child.wait(timeout=config["cleanup_timeout_seconds"] + 10)
                finally:
                    if read_fd is not None:
                        os.close(read_fd)
                    os.close(write_fd)
                result_file = folder / f"worker-{row['invocations']:02}.json"
                outcome = read(result_file) if result_file.exists() else {"state": "recovering", "error": "worker interrupted"}
                row.update(outcome)
                if row["state"] in TERMINAL:
                    row["finished_at_unix"] = time.time()
                    break
                save()
            if row["state"] not in TERMINAL:
                row.update(state="failed", error=row.get("error", "process restart limit"))
                row["finished_at_unix"] = time.time()
            if row["state"] == "blocked":
                # Fatal identity/auth/evidence failures are never acknowledged by an automatic resume.
                for later in state["matrix"]:
                    if later["alias"] == row["alias"] and later["state"] == "pending":
                        later.update(state="blocked", error="model blocked by fatal failure")
            save()
        save()
        return int(any(r["state"] != "completed" for r in state["matrix"]))


def watch(config_path, root, authorization, *, resume=False):
    """Restart a crashed controller, never a normally failed/blocked batch."""
    if os.name != "posix":
        raise ValueError("supervised batch currently requires POSIX")
    config = read(config_path)
    validate_config(config)
    if root.exists() and not resume:
        raise ValueError("existing batch requires --resume")
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.with_name(root.name + "-supervisor.json")
    with path.with_suffix(".lock").open("a+") as lock:
        lock_exclusive(lock)
        if path.exists():
            state = read(path)
            if not resume or state.get("config_sha256") != digest(config):
                raise ValueError("watcher identity changed or --resume missing")
            if (type(state.get("invocations")) is not int
                    or not 0 <= state["invocations"] <= config["max_controller_restarts"] + 1):
                raise ValueError("invalid controller invocation count")
            number(state.get("started_at_unix"), "watcher start time")
        else:
            state = {"config_sha256": digest(config), "started_at_unix": time.time(), "invocations": 0}
            atomic_json(path, state)
        deadline = state["started_at_unix"] + config["batch_timeout_seconds"]
        while state["invocations"] <= config["max_controller_restarts"]:
            if time.time() >= deadline:
                break
            # A crashed controller's worker guardian retains the batch lock until
            # its owned process group has been stopped. Do not overlap controllers.
            batch_lock = root / "batch.lock"
            wait_until = time.monotonic() + config["cleanup_timeout_seconds"] * 2 + 15
            while batch_lock.exists():
                try:
                    with batch_lock.open("a+") as running:
                        lock_exclusive(running)
                    break
                except OSError:
                    if time.monotonic() >= wait_until:
                        raise RuntimeError("previous worker cleanup has not released the batch lock")
                    time.sleep(.25)
            state["invocations"] += 1
            atomic_json(path, state)
            command = [sys.executable, "-m", "tools.unified_benchmark.batch", "run",
                       "--config", str(config_path), "--output", str(root), "--execute",
                       "--authorization-note", authorization]
            if resume or root.exists():
                command.append("--resume")
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
            state.update(controller_pid=child.pid, state="running")
            atomic_json(path, state)
            try:
                code = child.wait(timeout=max(.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=config["cleanup_timeout_seconds"] + 10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                state.update(state="failed", reason="batch watchdog deadline")
                atomic_json(path, state)
                return 1
            if code >= 0:
                # A normal nonzero result preserves failed/incomplete rounds.
                state.update(state="completed" if code == 0 else "failed", exit_code=code)
                atomic_json(path, state)
                return code
            state.update(state="recovering", exit_code=code)
            atomic_json(path, state)
        state.update(state="failed", reason="controller restart/deadline limit")
        atomic_json(path, state)
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "run", "report", "worker", "watch"))
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--authorization-note")
    p.add_argument("--task-id")
    a = p.parse_args(argv)
    if a.action in {"check", "run", "watch"} and a.config is None:
        p.error("--config is required")
    if a.action != "check" and a.output is None:
        p.error("--output is required")
    if a.action == "check":
        checked = validate_config(read(a.config))
        print(json.dumps({"offline_passed": True, "inference_calls": 0,
                          "planned_phases": len(checked["matrix"]), "reserved_upper_usd": checked["reserved_upper_usd"]}))
        return 0
    if a.action == "report":
        with (a.output / "batch.lock").open("a+") as lock:
            lock_exclusive(lock)
            config, state = read(a.output / "config.json"), read(a.output / "state.json")
            validate_state(state, validate_config(config), config)
            for row in state["matrix"]:
                if row["state"] in {"completed", "completed_incomplete"}:
                    item = next(
                        m for m in config["models"] if m["alias"] == row["alias"]
                    )
                    verify_finished(a.output, item, row)
            summarize(a.output, config, state)
        return 0
    if a.action == "worker":
        if os.environ.get("GOODVISION_SUPERVISED_WORKER") != "1":
            p.error("workers must be launched by the supervised batch")
        return worker(a.output, a.task_id, a.authorization_note)
    if not a.execute or not a.authorization_note:
        p.error("run requires --execute and --authorization-note")
    if a.action == "watch":
        return watch(a.config.resolve(), a.output.resolve(), a.authorization_note, resume=a.resume)
    if a.detach:
        validate_config(read(a.config))
        if a.output.exists() and not a.resume:
            p.error("existing batch requires --resume")
        a.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "tools.unified_benchmark.batch", "watch", "--config",
                   str(a.config.resolve()), "--output", str(a.output.resolve()), "--execute",
                   "--authorization-note", a.authorization_note]
        if a.resume:
            command.append("--resume")
        log = a.output.resolve().with_name(a.output.name + "-controller.log")
        with log.open("ab") as handle:
            proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        print(json.dumps({"started_pid": proc.pid, "controller_log": str(log),
                          "output": str(a.output.resolve()), "completed": False}))
        return 0
    return run(a.config.resolve(), a.output.resolve(), a.authorization_note, resume=a.resume)


if __name__ == "__main__":
    raise SystemExit(main())
