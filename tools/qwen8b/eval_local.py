#!/usr/bin/env python3
"""Run a frozen unified bundle against a self-hosted OpenAI-compatible Qwen8B.

No OpenRouter key, no generation lookup, no provider fallback. The local server
must identify as qwen/qwen3-vl-8b-instruct. Headline scoring stays correct/all.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from tools.unified_benchmark.adapters import parse_response, request_body
from tools.unified_benchmark.api_run import InvalidResponse, MAX_RESPONSE_BYTES, accounting, atomic, retry_after
from tools.unified_benchmark.contract import EVENTS, canonical, digest, file_hash, score, write_new, write_receipt
from tools.unified_benchmark.retry import Ledger, Queue, RequestFailure
from tools.unified_benchmark.workflow import collect, context_for, report, verify_bundle


class LocalTransport:
    def __init__(self, bundle, config, endpoint, output, ledger):
        self.bundle, self.config, self.endpoint = bundle, config, endpoint.rstrip("/")
        self.output, self.ledger = output, ledger
        self.expected_model = config["model"]

    def call(self, body):
        req = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=canonical(body),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InvalidResponse("oversized response", status, raw[:MAX_RESPONSE_BYTES], truncated=True)
        try:
            payload = json.loads(raw)
            canonical(payload)  # Reject nonfinite JSON without discarding its raw evidence.
        except ValueError as error:
            raise InvalidResponse("invalid JSON response", status, raw) from error
        return payload

    def __call__(self, unit):
        attempt = self.ledger.state(unit, self.config["kind"])[0]
        prefix = digest(unit["id"]) + f"-attempt-{attempt}"
        receipt = self.output / "receipts" / (prefix + ".json")
        def save(record):
            write_receipt(receipt, record | {"unit": unit["id"], "request_sha256": unit["request_sha256"]})
            return {"receipt": str(receipt.relative_to(self.output)), "receipt_sha256": file_hash(receipt)}
        _url, body = request_body(self.config, unit, self.bundle)
        if digest(body) != unit["request_sha256"]:
            raise RequestFailure("request fingerprint mismatch", fatal=True)
        try:
            payload = self.call(body)
        except InvalidResponse as error:
            metadata = save({"http_status": error.status, "transport_error": type(error).__name__,
                "detail": str(error), "body_base64": base64.b64encode(error.body).decode(),
                "body_truncated": error.truncated, "captured_body_bytes": len(error.body)})
            raise RequestFailure(str(error), status=error.status, metadata=metadata) from error
        except urllib.error.HTTPError as error:
            raw = error.read(2 * 1024 * 1024).decode(errors="replace")
            metadata = save({"http_status": error.code, "body": raw})
            raise RequestFailure("HTTP " + str(error.code), status=error.code,
                                 retry_after=retry_after(error.headers.get("Retry-After")),
                                 fatal=error.code in (400, 401, 403, 404), metadata=metadata) from error
        except (OSError, TimeoutError, ValueError) as error:
            metadata = save({"transport_error": type(error).__name__, "detail": str(error)[:500]})
            raise RequestFailure(type(error).__name__, metadata=metadata) from error
        metadata = save({"http_status": 200, "response": payload})
        if not isinstance(payload, dict):
            raise RequestFailure("invalid response envelope", status=200, metadata=metadata)
        try:
            canonical(payload)
        except ValueError as error:
            raise RequestFailure("nonfinite response", status=200, metadata=metadata) from error
        actual_model = payload.get("model")
        if not isinstance(actual_model, str) or not actual_model:
            raise RequestFailure("actual model identity unavailable", fatal=True, metadata=metadata)
        metadata.update({
            "actual_provider": "self-hosted-r2-4090",
            "actual_model": actual_model,
        })
        if metadata["actual_model"] != self.expected_model:
            raise RequestFailure("actual model mismatch: " + actual_model, fatal=True, metadata=metadata)
        try:
            answers = parse_response(self.config, unit, payload)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            answers = {}
            metadata["answer_error"] = str(error)[:500]
        return answers, metadata


def execute(alias, bundle, root, endpoint, bundle_digest, phase):
    output = root / alias
    config = json.loads((bundle / "configs" / f"{alias}.json").read_text())
    if (config.get("execution") != "self_hosted_r2" or config.get("backend") != "language"
            or config.get("kind") != "choice"):
        raise ValueError("eval_local requires a self_hosted_r2 hard-answer language profile")
    thresholds, calibration = {e: .5 for e in EVENTS}, "not_applicable_hard_boolean"
    output.mkdir(exist_ok=False)
    stage_results = {}
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
            Queue(ledger, config["kind"], LocalTransport(bundle, config, endpoint, directory, ledger)).run(plan["units"])
        except Exception as error:
            failed = type(error).__name__ + ": " + str(error)
        observed = collect(plan, config, ledger)
        result = report(plan, config, ledger, thresholds)
        result.update(split=split, calibration=calibration, started_at_unix=started,
                      finished_at_unix=time.time(), elapsed_seconds=time.time() - started,
                      failure=failed, accounting=accounting(ledger),
                      execution="self_hosted_r2", endpoint=endpoint)
        if split == "test":
            for label, events in (("base100", EVENTS[:4]), ("bag20", EVENTS[4:])):
                subset_q = [q for q in plan["questions"] if q["event"] in events and q["number"] <= 120]
                wanted = {uid for q in subset_q for uid in q["unit_ids"]}
                subset = plan | {"questions": subset_q, "units": [u for u in plan["units"] if u["id"] in wanted]}
                result[label] = score(subset, {uid: v for uid, v in observed.items() if uid in wanted},
                                      config["kind"], thresholds)
        write_new(directory / "summary.json", result)
        stage_results[split] = {k: result.get(k) for k in (
            "correct", "complete_valid_questions", "accuracy", "incomplete_questions",
            "final_missing_unique_units")}
        ledger.close()
        if failed:
            atomic(output / "status.json", {"state": "stopped", "split": split, "error": failed,
                                            "results": stage_results})
            return
    atomic(output / "status.json", {"state": "completed", "results": stage_results})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible origin, no trailing path")
    parser.add_argument("--model", default="qwen8b_r2_host")
    parser.add_argument("--phase", choices=("validation", "test", "both"), default="both")
    parser.add_argument("--authorization-note", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("no inference without explicit --execute")
    bundle_digest = verify_bundle(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.bundle / "configs" / (args.model + ".json")).read_text())
    write_new(args.output / "run-manifest.json", {
        "bundle_sha256": bundle_digest, "runner_sha256": file_hash(__file__),
        "alias": args.model, "endpoint": args.endpoint, "phase": args.phase,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "authorization": args.authorization_note, "execution": "self_hosted_r2",
        "engine": config.get("engine", "transformers"),
        "engine_version": config.get("engine_version"),
        "slots_per_model": 12})
    execute(args.model, args.bundle, args.output, args.endpoint, bundle_digest, args.phase)
    status = json.loads((args.output / args.model / "status.json").read_text())
    write_new(args.output / "completion.json", {
        "finished_at": datetime.now(timezone.utc).isoformat(), "model_status": status})
    return 0 if status.get("state") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
