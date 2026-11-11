"""Explicitly authorized OpenRouter execution of the frozen offline bundle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import certifi
from tools.event_benchmark.evidence import response_metadata
from .adapters import parse_response, request_body
from .contract import EVENTS, RULES, canonical, digest, file_hash, score, select_thresholds, write_new
from .receipts import raw_fields
from .retry import Ledger, PausedRun, Queue, RequestFailure
from .workflow import collect, context_for, load_threshold_artifact, report, verify_bundle

ALIASES = ("qwen8b", "gemini25", "gemini38", "qwen235b", "luna", "maverick")


def load_explicit_key(path):
    # A dotenv value may be quoted. Do not silently prefer a different process key.
    values = [line.split("=", 1)[1].strip().strip('"\'') for line in Path(path).read_text().splitlines()
              if line.startswith("OPENROUTER_API_KEY=")]
    if len(values) != 1 or not values[0]:
        raise ValueError("one nonempty OPENROUTER_API_KEY required in the selected file")
    return values[0]


def retry_after(value, now=None):
    if not value:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, seconds) if math.isfinite(seconds) else 0


def atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical(data) + b"\n")
    temp.replace(path)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "OpenRouter redirects are refused", headers, fp
        )


class Transport:
    def __init__(self, bundle, config, key, output, ledger, billing_stop):
        self.bundle, self.config, self.key, self.output = bundle, config, key, output
        self.ledger, self.billing_stop = ledger, billing_stop
        self.ssl = ssl.create_default_context(cafile=certifi.where())
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self.ssl), _NoRedirect()
        )

    def call(self, url, body=None):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc != "openrouter.ai"
                or parsed.fragment or parsed.path not in {
                    "/api/v1/chat/completions", "/api/alpha/decisions",
                    "/api/v1/generation",
                }):
            raise ValueError("OpenRouter transport refuses an unapproved endpoint")
        req = urllib.request.Request(url, data=canonical(body) if body is not None else None,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        with self.opener.open(req, timeout=90) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("oversized response")
        return json.loads(raw)

    def __call__(self, unit):
        if self.billing_stop.is_set():
            raise RequestFailure("account billing pause", fatal=True)
        attempt = self.ledger.state(unit, self.config["kind"])[0]
        prefix = digest(unit["id"]) + f"-attempt-{attempt}"
        uri, body = request_body(self.config, unit, self.bundle)
        if digest(body) != unit["request_sha256"]:
            raise RequestFailure("request fingerprint mismatch", fatal=True)
        try:
            payload = self.call(uri, body)
        except urllib.error.HTTPError as error:
            raw = error.read(2 * 1024 * 1024).decode(errors="replace").replace(self.key, "[REDACTED]")
            write_new(self.output / "receipts" / (prefix + ".json"), {
                "http_status": error.code, "body": raw, "retry_after": error.headers.get("Retry-After"),
                "unit": unit["id"], "request_sha256": unit["request_sha256"]})
            if error.code == 402:
                self.billing_stop.set()
            raise RequestFailure("HTTP " + str(error.code), status=error.code,
                retry_after=retry_after(error.headers.get("Retry-After")), fatal=error.code in (400, 401, 402, 403, 404)) from error
        except (OSError, TimeoutError, ValueError) as error:
            write_new(self.output / "receipts" / (prefix + ".json"), {
                "transport_error": type(error).__name__, "detail": str(error).replace(self.key, "[REDACTED]"),
                "unit": unit["id"], "request_sha256": unit["request_sha256"]})
            raise RequestFailure(type(error).__name__) from error
        metadata = response_metadata(payload)
        # Provider metadata GET is read-only and never replaces a model answer.
        generation = metadata.get("generation_id")
        generation_data = None
        if generation and (not metadata.get("actual_provider") or not metadata.get("actual_model")):
            try:
                generation_payload = self.call(
                    "https://openrouter.ai/api/v1/generation?id="
                    + urllib.parse.quote(generation, safe="")
                )
                generation_data = (
                    generation_payload.get("data")
                    if isinstance(generation_payload, dict) else None
                )
                if (not isinstance(generation_data, dict)
                        or generation_data.get("id") != generation):
                    raise ValueError("generation identity mismatch")
                for k, v in response_metadata(generation_data).items():
                    metadata.setdefault(k, v)
            except (OSError, ValueError, KeyError, TypeError):
                metadata["metadata_lookup_error"] = "unavailable"
        write_new(self.output / "receipts" / (prefix + ".json"), {
            **raw_fields(response=payload, generation_metadata=generation_data),
            "unit": unit["id"], "request_sha256": unit["request_sha256"]})
        expected = self.config["provider_context"]["config"]["expected_provider_name"]
        if metadata.get("actual_provider") and metadata["actual_provider"] != expected:
            raise RequestFailure("actual provider mismatch: " + metadata["actual_provider"], fatal=True)
        if not metadata.get("actual_provider") or not metadata.get("actual_model"):
            raise RequestFailure("provider/model receipt identity unavailable", fatal=True)
        actual_model = metadata.get("actual_model")
        if actual_model and actual_model != self.config["model"] and not (
                self.config["backend"] == "decision" and actual_model.startswith(self.config["model"] + "-")):
            raise RequestFailure("actual model mismatch: " + actual_model, fatal=True)
        try:
            if self.config["backend"] == "language" and payload.get("choices", [{}])[0].get("message", {}).get("refusal"):
                raise ValueError("explicit refusal")
            answers = parse_response(self.config, unit, payload)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            answers = {}
            metadata["answer_error"] = str(error)[:500]
        metadata["receipt"] = "receipts/" + prefix + ".json"
        return answers, metadata


def accounting(ledger):
    results = [r for r in ledger.rows if r["type"] == "result"]
    costs, unknown = [], 0
    for row in results:
        meta = row.get("metadata", {})
        cost = meta.get("usage", {}).get("cost", meta.get("total_cost"))
        if type(cost) in (int, float) and math.isfinite(cost):
            costs.append(cost)
        else:
            unknown += 1
    return {"known_receipt_cost_usd": sum(costs), "cost_known_attempts": len(costs),
            "attempts_without_cost_evidence": unknown, "unknown_cost_is_not_zero": True,
            "actual_providers": sorted({r.get("metadata", {}).get("actual_provider") for r in results if r.get("metadata", {}).get("actual_provider")}),
            "actual_models": sorted({r.get("metadata", {}).get("actual_model") for r in results if r.get("metadata", {}).get("actual_model")}),
            "valid_results_missing_provider_metadata": sum(bool(r.get("answers")) and not r.get("metadata", {}).get("actual_provider") for r in results)}


def execute_model(alias, bundle, root, key, billing_stop, bundle_digest, *, phase="both", threshold_path=None, threshold_sha256=None):
    output = root / alias
    config = json.loads((bundle / "configs" / f"{alias}.json").read_text())
    stage_results, thresholds, calibration = {}, {e: .5 for e in EVENTS}, "not_applicable_hard_boolean"
    if phase == "test" and config["kind"] == "probability":
        artifact = load_threshold_artifact(threshold_path, threshold_sha256,
            bundle=bundle, alias=alias, config=config, bundle_digest=bundle_digest)
        thresholds, calibration = artifact["thresholds"], artifact["calibration"]
    output.mkdir(exist_ok=False)
    for split in (("validation", "test") if phase == "both" else (phase,)):
        plan = json.loads((bundle / "plans" / f"{alias}-{split}.json").read_text())
        directory = output / split
        directory.mkdir()
        ledger = Ledger(directory / "attempts.jsonl", context_for(plan, config, bundle_digest))
        started = time.time()
        atomic(output / "status.json", {"state": "running", "split": split, "started_at": started,
                                       "planned_units": len(plan["units"]), "slots": 12})
        failed = None
        try:
            q = Queue(ledger, config["kind"], Transport(bundle, config, key, directory, ledger, billing_stop))
            q.run(plan["units"])
        except Exception as error:
            failed = type(error).__name__ + ": " + str(error)
        observed = collect(plan, config, ledger)
        if split == "validation" and config["kind"] == "probability":
            try:
                chosen = select_thresholds(plan, observed, "probability", config["model"])
                thresholds, calibration = chosen["thresholds"], "selected_on_complete_validation24"
            except ValueError as error:
                failed = "threshold selection: " + str(error)
                chosen = {"thresholds": None, "reason": str(error)}
                calibration = "not_calibrated; test blocked"
            if chosen.get("thresholds") is not None:
                chosen["validation_only"] = True
            write_new(output / "thresholds.json", chosen | {"calibration": calibration,
                "config_sha256": digest(config), "bundle_sha256": bundle_digest,
                "rules_sha256": digest(RULES), "model": config["model"], "selected_before_test_at": time.time()})
        result = report(plan, config, ledger, thresholds)
        result.update(split=split, calibration=calibration, started_at_unix=started, finished_at_unix=time.time(),
                      elapsed_seconds=time.time() - started, failure=failed, accounting=accounting(ledger))
        if split == "test":
            for label, events in (("base100", EVENTS[:4]), ("bag20", EVENTS[4:])):
                subset_q = [q for q in plan["questions"] if q["event"] in events]
                wanted = {uid for q in subset_q for uid in q["unit_ids"]}
                subset = plan | {"questions": subset_q, "units": [u for u in plan["units"] if u["id"] in wanted]}
                result[label] = score(subset, {uid: v for uid, v in observed.items() if uid in wanted}, config["kind"], thresholds)
                sent = {r["unit"] for r in ledger.rows if r["type"] == "intent" and r["unit"] in wanted}
                result[label].update(sent_unique_units=len(sent), send_coverage=len(sent) / len(wanted))
        write_new(directory / "summary.json", result)
        stage_results[split] = {k: result[k] for k in ("correct", "complete_valid_questions", "accuracy", "incomplete_questions", "final_missing_unique_units", "accounting")}
        ledger.close()
        if failed:
            atomic(output / "status.json", {"state": "stopped", "split": split, "error": failed, "results": stage_results})
            return
    atomic(output / "status.json", {"state": "completed", "results": stage_results})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    keys = parser.add_mutually_exclusive_group(required=True)
    keys.add_argument("--env-file", type=Path)
    keys.add_argument("--credential-map", type=Path, help="private JSON alias -> dotenv path; no keys")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--phase", choices=("validation", "test", "both"), default="validation")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--thresholds-sha256")
    parser.add_argument("--authorization-note", required=True, help="record the approved scope/cost decision")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("no network inference without explicit --execute")
    bundle_digest = verify_bundle(args.bundle)
    if len(set(args.models)) != len(args.models):
        parser.error("duplicate model aliases")
    groups = {}
    for alias in args.models:
        if Path(alias).name != alias:
            parser.error("model alias cannot contain directories")
        config = json.loads((args.bundle / "configs" / (alias + ".json")).read_text())
        if config["backend"] not in ("language", "decision"):
            parser.error("this command executes API profiles only")
        # Validate every selected calibrated model before any model can spend money.
        if args.phase == "test" and config["kind"] == "probability":
            load_threshold_artifact(args.thresholds, args.thresholds_sha256,
                bundle=args.bundle, alias=alias, config=config, bundle_digest=bundle_digest)
        provider = config["provider_context"]["config"]["expected_provider_name"]
        groups.setdefault(provider, []).append(alias)
    credentials = (json.loads(args.credential_map.read_text()) if args.credential_map
                   else {alias: str(args.env_file) for alias in args.models})
    if not set(args.models) <= set(credentials):
        parser.error("credential map must cover each selected model")
    keys_by_alias = {alias: load_explicit_key(credentials[alias]) for alias in args.models}
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "run-manifest.json", {
        "bundle_sha256": bundle_digest, "runner_sha256": file_hash(__file__), "aliases": args.models,
        "started_at": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
        "authorization": args.authorization_note, "provider_groups": groups,
        "same_provider_models_do_not_overlap": True, "slots_per_model": 12,
        "no_extra_recovery_after_three_attempts": True})
    billing_stop = threading.Event()
    def group(aliases):
        for alias in aliases:
            if billing_stop.is_set():
                break
            execute_model(alias, args.bundle, args.output, keys_by_alias[alias], billing_stop,
                          bundle_digest, phase=args.phase, threshold_path=args.thresholds,
                          threshold_sha256=args.thresholds_sha256)
    failures = []
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(group, aliases) for aliases in groups.values()]
        for future in futures:
            try:
                future.result()
            except Exception as error:
                message = type(error).__name__ + ": " + str(error)
                for key in keys_by_alias.values():
                    message = message.replace(key, "[REDACTED]")
                failures.append(message)
    statuses = {alias: json.loads((args.output / alias / "status.json").read_text())
                if (args.output / alias / "status.json").exists() else {"state": "not_started"}
                for alias in args.models}
    write_new(args.output / "completion.json", {"finished_at": datetime.now(timezone.utc).isoformat(),
        "billing_paused": billing_stop.is_set(), "orchestrator_errors": failures, "model_status": statuses})
    return int(bool(failures) or any(s["state"] != "completed" for s in statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())
