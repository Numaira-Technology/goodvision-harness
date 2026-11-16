"""Shared durable API/self-hosted execution. Recovery never sends a request."""
import json
import math
import time

from tools.event_benchmark.evidence import response_metadata
from .adapters import parse_response, request_body
from .contract import (EVENTS, atomic_json, digest, file_hash, plan_rules,
                       read_receipt, select_thresholds, valid_value, write_new)
from .locking import lock_exclusive
from .retry import Ledger, PausedRun, Queue, RequestFailure
from .workflow import collect, context_for, load_threshold_artifact, report


def receipt_path(directory, unit, attempt):
    return directory / "receipts" / (digest(unit["id"]) + f"-attempt-{attempt}.json")


def outcome_path(directory, unit, attempt):
    return directory / "outcomes" / (digest(unit["id"]) + f"-attempt-{attempt}.json")


def decode(config, unit, record, *, local=False):
    """Re-parse the retained response and verify provider/model, without a GET."""
    status = record.get("http_status")
    if record.get("transport_error") or (status is not None and status != 200):
        from .api_run import retry_after
        raise RequestFailure(record.get("transport_error") or f"HTTP {status}",
            status=status, retry_after=retry_after(record.get("retry_after")),
            fatal=status in (400, 401, 402, 403, 404))
    payload = record.get("response")
    if not isinstance(payload, dict):
        raise RequestFailure("invalid response envelope")
    metadata = response_metadata(payload)
    generation = record.get("generation_metadata")
    if generation is not None:
        if not isinstance(generation, dict) or generation.get("id") != metadata.get("generation_id"):
            raise ValueError("saved generation metadata identity differs")
        for key, value in response_metadata(generation).items():
            metadata.setdefault(key, value)
    if local:
        metadata["actual_provider"] = "self-hosted-r2-4090"
    expected_provider = ("self-hosted-r2-4090" if local else
                         config["provider_context"]["config"]["expected_provider_name"])
    actual = metadata.get("actual_model")
    if (metadata.get("actual_provider") != expected_provider or not isinstance(actual, str)
            or not (actual == config["model"] or
                    config["backend"] == "decision" and actual.startswith(config["model"] + "-"))):
        raise RequestFailure("provider/model receipt identity differs or unavailable",
                             fatal=True, metadata=metadata)
    try:
        answers = parse_response(config, unit, payload)
    except (ValueError, KeyError, TypeError, IndexError):
        answers = {}
    return answers, metadata


class DurableTransport:
    def __init__(self, inner, directory, ledger, config, *, local=False):
        self.inner, self.directory, self.ledger = inner, directory, ledger
        self.config, self.local = config, local

    def __call__(self, unit):
        attempt = self.ledger.state(unit, self.config["kind"])[0]
        started = time.monotonic()
        failure = None
        try:
            answers, metadata = self.inner(unit)
        except RequestFailure as error:
            answers, metadata, failure = {}, error.metadata or {}, error
        raw = receipt_path(self.directory, unit, attempt)
        if raw.is_file():
            metadata = dict(metadata, receipt=raw.relative_to(self.directory).as_posix(),
                            receipt_sha256=file_hash(raw))
        elif failure is None or not failure.fatal:
            raise ValueError("transport did not retain its raw receipt")
        row = {"unit": unit["id"], "attempt": attempt, "request_sha256": unit["request_sha256"],
               "context": self.ledger.context, "answers": answers, "metadata": metadata,
               "elapsed_seconds": time.monotonic() - started, "completed_at_unix": time.time(),
               "error": failure.reason if failure else None,
               "http_status": failure.status if failure else 200,
               "fatal": failure.fatal if failure else False,
               "retry_after": failure.retry_after if failure else 0}
        path = outcome_path(self.directory, unit, attempt)
        write_new(path, row)
        metadata = dict(metadata, outcome=path.relative_to(self.directory).as_posix(),
                        outcome_sha256=file_hash(path))
        if failure:
            raise RequestFailure(failure.reason, status=failure.status, fatal=failure.fatal,
                                 retry_after=failure.retry_after, metadata=metadata)
        return answers, metadata


def reconcile(directory, ledger, plan, config, *, local=False, sealed=False):
    """Check every recorded answer; recover receipt/outcome gaps before new calls."""
    collect(plan, config, ledger)
    units = {u["id"]: u for u in plan["units"]}
    results = {(r["unit"], r["attempt"]): r for r in ledger.rows if r["type"] == "result"}
    expected_raw, expected_outcomes = set(), set()
    for intent in list(ledger.rows):
        if intent["type"] != "intent":
            continue
        unit, attempt = units[intent["unit"]], intent["attempt"]
        raw, outcome = receipt_path(directory, unit, attempt), outcome_path(directory, unit, attempt)
        expected_raw.add(raw)
        expected_outcomes.add(outcome)
        result = results.get((unit["id"], attempt))
        for path in (raw, outcome):
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("receipt path escapes run")
        saved = json.loads(outcome.read_text()) if outcome.exists() else None
        if result:
            meta = result.get("metadata", {})
            if (not saved or meta.get("outcome") != outcome.relative_to(directory).as_posix()
                    or meta.get("outcome_sha256") != file_hash(outcome)):
                raise ValueError("saved outcome missing or changed")
        if saved:
            expected = {k: intent[k] for k in ("unit", "attempt", "request_sha256", "context")}
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError("saved outcome identity differs")
            if (not isinstance(saved.get("answers"), dict)
                    or type(saved.get("fatal")) is not bool):
                raise ValueError("invalid saved outcome")
            for field in ("elapsed_seconds", "completed_at_unix", "retry_after"):
                value = saved.get(field)
                if field == "elapsed_seconds" and value is None and saved.get("timing_unknown_after_crash") is True:
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid saved outcome " + field)
        if raw.exists():
            record = read_receipt(raw)
            if any(record.get(k) != unit[k2] for k, k2 in
                   (("unit", "id"), ("request_sha256", "request_sha256"))):
                raise ValueError("raw receipt request identity differs")
            if saved:
                meta = saved.get("metadata", {})
                if (meta.get("receipt") != raw.relative_to(directory).as_posix()
                        or meta.get("receipt_sha256") != file_hash(raw)):
                    raise ValueError("raw receipt missing integrity anchor or changed")
            try:
                parsed, metadata = decode(config, unit, record, local=local)
                failure = None
            except RequestFailure as error:
                parsed, metadata, failure = {}, error.metadata or {}, error
            if saved and parsed != saved["answers"]:
                raise ValueError("saved answers differ from raw receipt")
            if saved and failure and failure.fatal and not saved["fatal"]:
                raise ValueError("saved fatal response was downgraded")
        elif saved:
            if saved["answers"] or not saved["fatal"] or saved["metadata"].get("receipt"):
                raise ValueError("raw receipt missing")
            parsed, metadata, failure = {}, saved["metadata"], None
        else:
            if result:
                raise ValueError("recorded answer has no receipt")
            continue  # An unresolved intent still consumes one of three attempts.
        if result:
            if result["answers"] != saved["answers"] or result.get("fatal", False) != saved["fatal"]:
                raise ValueError("ledger and outcome differ")
            continue
        if sealed:
            raise ValueError("completed phase has an unrecorded outcome")
        if saved:
            answers, metadata = saved["answers"], saved["metadata"]
            elapsed, completed = saved["elapsed_seconds"], saved["completed_at_unix"]
            status, fatal, delay = saved["http_status"], saved["fatal"], saved["retry_after"]
            error = saved["error"]
        else:
            # Raw response survived, outcome/ledger did not. Keep the first answer;
            # a missing timing sample is unknown, never zero or a reason to resend.
            answers, elapsed = parsed, None
            completed = record.get("recorded_at_unix")
            status, fatal, delay = (failure.status, failure.fatal, failure.retry_after) if failure else (200, False, 0)
            error = failure.reason if failure else None
            metadata = dict(metadata, receipt=raw.relative_to(directory).as_posix(), receipt_sha256=file_hash(raw))
            saved = {"unit": unit["id"], "attempt": attempt, "request_sha256": unit["request_sha256"],
                     "context": ledger.context, "answers": answers, "metadata": metadata,
                     "elapsed_seconds": None, "completed_at_unix": completed,
                     "http_status": status, "fatal": fatal, "retry_after": delay, "error": error,
                     "timing_unknown_after_crash": True}
            if not isinstance(completed, (int, float)) or not math.isfinite(completed):
                raise ValueError("raw orphan has no durable completion timestamp")
            write_new(outcome, saved)
        metadata = dict(metadata, outcome=outcome.relative_to(directory).as_posix(),
                        outcome_sha256=file_hash(outcome), recovered_receipt=True)
        prior = {}
        for row in sorted((r for r in ledger.rows if r["type"] == "result"
                           and r["unit"] == unit["id"] and r["attempt"] < attempt),
                          key=lambda r: r["attempt"]):
            for event, value in row["answers"].items():
                prior.setdefault(event, value)
        valid = {e: v for e, v in answers.items() if e in unit["targets"] and valid_value(v, config["kind"])}
        complete = set(prior | valid) == set(unit["targets"])
        recovered = {k: intent[k] for k in ("unit", "attempt", "request_sha256")}
        recovered.update(type="result", answers=valid, metadata=metadata, complete=complete,
                         elapsed_seconds=elapsed, http_status=status, fatal=fatal,
                         retry_after=delay, error=error)
        if not complete and not fatal and (attempt < 3 or status == 429):
            recovered["cooldown_until"] = completed + max(delay, 30 if attempt == 1 else 60)
        ledger.append(recovered)
    for folder, expected in (("receipts", expected_raw), ("outcomes", expected_outcomes)):
        for path in (directory / folder).glob("*.json"):
            if path not in expected:
                raise ValueError("unrecorded receipt/outcome without a durable intent")
    if any(r.get("fatal") for r in ledger.rows):
        raise PausedRun("fatal response retained; automatic retries are blocked")


def execute_model(alias, bundle, root, bundle_digest, transport_factory, *, phase,
                  connection, resume=False, threshold_path=None, threshold_sha256=None,
                  local=False, queue_factory=Queue, before_intent=None):
    """Execute once or resume the same identity, then seal phase evidence."""
    output = root / alias
    cfg = json.loads((bundle / "configs" / f"{alias}.json").read_text())
    if phase not in ("validation", "test", "both"):
        raise ValueError("invalid execution phase")
    splits = ("validation", "test") if phase == "both" else (phase,)
    if any(not (bundle / "plans" / f"{alias}-{split}.json").is_file() for split in splits):
        raise ValueError("requested phase was not prepared in this bundle")
    identity = {"alias": alias, "phase": phase, "bundle_sha256": bundle_digest,
                "config_sha256": digest(cfg), "connection": connection,
                "threshold_sha256": threshold_sha256}
    output.mkdir(parents=True, exist_ok=resume)
    with (output / "execution.lock").open("a") as lock:
        lock_exclusive(lock)
        manifest = output / "run.json"
        if manifest.exists():
            if not resume or json.loads(manifest.read_text()) != identity:
                raise ValueError("run identity differs")
        else:
            if any(p.name != "execution.lock" for p in output.iterdir()):
                raise ValueError("run identity missing from nonempty output")
            write_new(manifest, identity)
        thresholds, calibration = dict.fromkeys(EVENTS, .5), "not_applicable_hard_boolean"
        if phase == "test" and cfg["kind"] == "probability":
            artifact = load_threshold_artifact(threshold_path, threshold_sha256, bundle=bundle,
                        alias=alias, config=cfg, bundle_digest=bundle_digest)
            thresholds, calibration = artifact["thresholds"], artifact["calibration"]
        results = {}
        for split in splits:
            plan = json.loads((bundle / "plans" / f"{alias}-{split}.json").read_text())
            for unit in plan["units"]:
                if digest(request_body(cfg, unit, bundle)[1]) != unit["request_sha256"]:
                    raise ValueError("frozen request differs")
            directory = output / split
            directory.mkdir(exist_ok=True)
            context = context_for(plan, cfg, bundle_digest)
            ledger = Ledger(directory / "attempts.jsonl", context)
            ledger.before_intent = before_intent
            started = time.time()
            atomic_json(output / "status.json", {"state": "running", "split": split})
            try:
                completion = directory / "completion.json"
                if completion.exists():
                    seal = json.loads(completion.read_text())
                    required = {"run.json", f"{split}/attempts.jsonl", f"{split}/summary.json"}
                    if split == "validation" and cfg["kind"] == "probability":
                        required.add("thresholds.json")
                    if not isinstance(seal.get("files"), dict) or not required <= set(seal["files"]):
                        raise ValueError("completed phase evidence anchors missing")
                    for name, sha in seal["files"].items():
                        path = output / name
                        if not path.resolve().is_relative_to(output.resolve()):
                            raise ValueError("completed phase changed")
                        # Windows byte locks forbid reopening the owned ledger.
                        actual = ledger.file_hash() if path.resolve() == ledger.path.resolve() else file_hash(path)
                        if actual != sha:
                            raise ValueError("completed phase changed")
                reconcile(directory, ledger, plan, cfg, local=local, sealed=completion.exists())
                if not completion.exists():
                    transport = DurableTransport(transport_factory(cfg, directory, ledger),
                                                 directory, ledger, cfg, local=local)
                    queue_factory(ledger, cfg["kind"], transport).run(plan["units"])
                if split == "validation" and cfg["kind"] == "probability":
                    selected = select_thresholds(plan, collect(plan, cfg, ledger), cfg["kind"], cfg["model"])
                    selected.update(config_sha256=digest(cfg), bundle_sha256=bundle_digest,
                                    rules_sha256=digest(plan_rules(plan)), model=cfg["model"])
                    threshold_path = output / "thresholds.json"
                    if threshold_path.exists():
                        if digest(json.loads(threshold_path.read_text())) != digest(selected):
                            raise ValueError("saved thresholds differ from validation evidence")
                    else:
                        write_new(threshold_path, selected)
                    thresholds, calibration = selected["thresholds"], selected["calibration"]
                if completion.exists():
                    result = json.loads((directory / "summary.json").read_text())
                else:
                    result = report(plan, cfg, ledger, thresholds)
                    if split == "test":
                        from .contract import score
                        observed = collect(plan, cfg, ledger)
                        for label, events in (("base100", EVENTS[:4]), ("bag20", EVENTS[4:])):
                            questions = [q for q in plan["questions"] if q["event"] in events and q["number"] <= 120]
                            wanted = {uid for q in questions for uid in q["unit_ids"]}
                            subset = plan | {"questions": questions, "units": [u for u in plan["units"] if u["id"] in wanted]}
                            result[label] = score(subset, {uid: observed[uid] for uid in wanted}, cfg["kind"], thresholds)
                    from .api_run import accounting
                    result.update(split=split, calibration=calibration, failure=None,
                                  started_at_unix=started, finished_at_unix=time.time(),
                                  accounting=accounting(ledger))
                    summary_path = directory / "summary.json"
                    if summary_path.exists():
                        previous = json.loads(summary_path.read_text())
                        volatile = {"started_at_unix", "finished_at_unix", "report_generated_at_utc"}
                        if {k: v for k, v in previous.items() if k not in volatile} != {
                                k: v for k, v in result.items() if k not in volatile}:
                            raise ValueError("unsealed summary differs from ledger")
                        result = previous
                    else:
                        write_new(summary_path, result)
                    files = [directory / "summary.json", ledger.path, manifest]
                    files += list((directory / "receipts").glob("*.json"))
                    files += list((directory / "outcomes").glob("*.json"))
                    if split == "validation" and cfg["kind"] == "probability":
                        files.append(threshold_path)
                    write_new(completion, {"files": {p.relative_to(output).as_posix():
                              ledger.file_hash() if p == ledger.path else file_hash(p) for p in files}})
                results[split] = result
                if result["incomplete_questions"]:
                    atomic_json(output / "status.json", {"state": "completed_incomplete", "split": split})
                    return results
            except Exception as error:
                # The snapshot is replaceable; receipts, ledger and successful summaries are not.
                atomic_json(directory / "partial-summary.json", report(plan, cfg, ledger, thresholds))
                atomic_json(output / "status.json", {"state": "failed", "split": split,
                                                    "error": type(error).__name__ + ": " + str(error)})
                raise
            finally:
                ledger.close()
        atomic_json(output / "status.json", {"state": "completed", "phases": list(results)})
        return results
