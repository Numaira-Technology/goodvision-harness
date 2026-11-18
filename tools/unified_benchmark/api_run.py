"""Explicitly authorized OpenRouter execution of the frozen offline bundle."""
import argparse
import base64
import http.client
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
from .contract import canonical, digest, file_hash, write_new, write_receipt, atomic_json
from .latency import REQUEST_TIMEOUT_SECONDS, raised_timeout
from .retry import RequestFailure
from .workflow import load_threshold_artifact, verify_bundle

ALIASES = ("qwen8b", "gemini25", "gemini38", "qwen235b", "luna", "maverick")

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class InvalidResponse(ValueError):
    """Keep bounded response bytes when the HTTP envelope cannot be decoded."""
    def __init__(self, reason, status, body, *, truncated=False):
        super().__init__(reason)
        self.status, self.body, self.truncated = status, body, truncated


def http_error_body(error, *, secret=None):
    """Retain bounded error bytes even when reading the error stream fails."""
    read_error = None
    try:
        raw = error.read(MAX_RESPONSE_BYTES + 1)
    except (http.client.HTTPException, OSError, ValueError) as failure:
        read_error = type(failure).__name__
        raw = getattr(failure, "partial", b"")
        if not isinstance(raw, bytes):
            raw = b""
    truncated = len(raw) > MAX_RESPONSE_BYTES
    captured = raw[:MAX_RESPONSE_BYTES]
    redacted = captured.replace(secret.encode(), b"[REDACTED]") if secret else captured
    return {"body": redacted.decode(errors="replace"),
            "body_base64": base64.b64encode(redacted).decode(),
            "body_truncated": truncated, "body_incomplete": read_error is not None,
            "body_read_error": read_error, "body_redacted": redacted != captured,
            "captured_body_bytes": len(captured)}


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
    atomic_json(path, data)


class Transport:
    def __init__(self, bundle, config, key, output, ledger, billing_stop):
        self.bundle, self.config, self.key, self.output = bundle, config, key, output
        self.ledger, self.billing_stop = ledger, billing_stop
        self.ssl = ssl.create_default_context(cafile=certifi.where())

    def call(self, url, body=None):
        req = urllib.request.Request(url, data=canonical(body) if body is not None else None,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        status = None
        try:
            with urllib.request.urlopen(
                    req, context=self.ssl, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = response.status
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except http.client.HTTPException as error:
            partial = getattr(error, "partial", b"")
            if not isinstance(partial, bytes):
                partial = b""
            raise InvalidResponse("invalid or incomplete HTTP response", status,
                partial[:MAX_RESPONSE_BYTES], truncated=len(partial) > MAX_RESPONSE_BYTES) from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InvalidResponse("oversized response", status, raw[:MAX_RESPONSE_BYTES], truncated=True)
        try:
            return json.loads(raw)
        except ValueError as error:
            raise InvalidResponse("invalid JSON response", status, raw) from error

    def __call__(self, unit):
        if self.billing_stop.is_set():
            raise RequestFailure("account billing pause", fatal=True)
        attempt = self.ledger.state(unit, self.config["kind"])[0]
        prefix = digest(unit["id"]) + f"-attempt-{attempt}"
        def save(record):
            path = self.output / "receipts" / (prefix + ".json")
            write_receipt(path, record | {"recorded_at_unix": time.time()})
        uri, body = request_body(self.config, unit, self.bundle)
        if digest(body) != unit["request_sha256"]:
            raise RequestFailure("request fingerprint mismatch", fatal=True)
        sent = time.monotonic()

        def failed():
            return {"transport_seconds": time.monotonic() - sent}

        try:
            payload = self.call(uri, body)
        except InvalidResponse as error:
            elapsed = failed()
            raw = error.body.replace(self.key.encode(), b"[REDACTED]")
            save({
                "http_status": error.status, "transport_error": type(error).__name__, "detail": str(error),
                "body_base64": base64.b64encode(raw).decode(), "body_truncated": error.truncated,
                "body_redacted": raw != error.body, "captured_body_bytes": len(error.body),
                "unit": unit["id"], "request_sha256": unit["request_sha256"]})
            raise RequestFailure(
                str(error), status=error.status, metadata=elapsed) from error
        except urllib.error.HTTPError as error:
            elapsed = failed()
            if error.code == 402:
                self.billing_stop.set()
            save({
                **http_error_body(error, secret=self.key),
                "http_status": error.code, "retry_after": error.headers.get("Retry-After"),
                "unit": unit["id"], "request_sha256": unit["request_sha256"]})
            raise RequestFailure(
                "HTTP " + str(error.code), status=error.code, metadata=elapsed,
                retry_after=retry_after(error.headers.get("Retry-After")), fatal=error.code in (400, 401, 402, 403, 404)) from error
        except (OSError, TimeoutError, ValueError) as error:
            elapsed = failed()
            save({
                "transport_error": type(error).__name__, "detail": str(error).replace(self.key, "[REDACTED]"),
                "unit": unit["id"], "request_sha256": unit["request_sha256"]})
            reason = "timeout" if raised_timeout(error) else type(error).__name__
            raise RequestFailure(reason, metadata=elapsed) from error
        transport_seconds = time.monotonic() - sent
        metadata = response_metadata(payload)
        metadata["transport_seconds"] = transport_seconds
        # Provider metadata GET is read-only and never replaces a model answer.
        generation = metadata.get("generation_id")
        generation_data = None
        if generation and (not metadata.get("actual_provider") or not metadata.get("actual_model")):
            lookup_started = time.monotonic()
            try:
                generation_data = self.call("https://openrouter.ai/api/v1/generation?id=" + urllib.parse.quote(generation, safe=""))["data"]
                if not isinstance(generation_data, dict):
                    raise ValueError("invalid generation metadata envelope")
                if generation_data.get("id") != generation:
                    raise ValueError("generation identity mismatch")
                for k, v in response_metadata(generation_data).items():
                    metadata.setdefault(k, v)
            except (OSError, ValueError, KeyError, TypeError):
                metadata["metadata_lookup_error"] = "unavailable"
            # Recorded separately; outside transport_seconds.
            metadata["metadata_lookup_seconds"] = time.monotonic() - lookup_started
        save({
            "response": payload, "generation_metadata": generation_data,
            "unit": unit["id"], "request_sha256": unit["request_sha256"]})
        metadata["receipt"] = "receipts/" + prefix + ".json"
        metadata["receipt_sha256"] = file_hash(self.output / metadata["receipt"])
        expected = self.config["provider_context"]["config"]["expected_provider_name"]
        if metadata.get("actual_provider") and metadata["actual_provider"] != expected:
            raise RequestFailure("actual provider mismatch: " + metadata["actual_provider"], fatal=True, metadata=metadata)
        if not metadata.get("actual_provider") or not metadata.get("actual_model"):
            raise RequestFailure("provider/model receipt identity unavailable", fatal=True, metadata=metadata)
        actual_model = metadata.get("actual_model")
        if actual_model and actual_model != self.config["model"] and not (
                self.config["backend"] == "decision" and actual_model.startswith(self.config["model"] + "-")):
            raise RequestFailure("actual model mismatch: " + actual_model, fatal=True, metadata=metadata)
        parse_started = time.monotonic()
        try:
            answers = parse_response(self.config, unit, payload)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            answers = {}
            metadata["answer_error"] = str(error)[:500]
        # Request sent to response parsed; excludes metadata lookup and receipt writes.
        parse_seconds = time.monotonic() - parse_started
        metadata["transport_seconds"] = transport_seconds + parse_seconds
        return answers, metadata


def accounting(ledger):
    results = [r for r in ledger.rows if r["type"] == "result"]
    completed = {(r["unit"], r["attempt"]) for r in results}
    unresolved = sum(r["type"] == "intent" and (r["unit"], r["attempt"]) not in completed for r in ledger.rows)
    costs, unknown = [], 0
    for row in results:
        meta = row.get("metadata", {})
        cost = meta.get("usage", {}).get("cost", meta.get("total_cost"))
        if type(cost) in (int, float) and math.isfinite(cost):
            costs.append(cost)
        else:
            unknown += 1
    return {"known_receipt_cost_usd": sum(costs), "cost_known_attempts": len(costs),
            "attempts_without_cost_evidence": unknown + unresolved, "unresolved_intents": unresolved,
            "unknown_cost_is_not_zero": True,
            "actual_providers": sorted({r.get("metadata", {}).get("actual_provider") for r in results if r.get("metadata", {}).get("actual_provider")}),
            "actual_models": sorted({r.get("metadata", {}).get("actual_model") for r in results if r.get("metadata", {}).get("actual_model")}),
            "valid_results_missing_provider_metadata": sum(bool(r.get("answers")) and not r.get("metadata", {}).get("actual_provider") for r in results)}


def execute_model(alias, bundle, root, key, billing_stop, bundle_digest, *, phase="both",
                  threshold_path=None, threshold_sha256=None, resume=False, before_intent=None):
    from .http_execution import execute_model as execute_http
    return execute_http(alias, bundle, root, bundle_digest,
        lambda config, directory, ledger: Transport(bundle, config, key, directory, ledger, billing_stop),
        phase=phase, connection={"route": "openrouter", "credential_sha256": digest(key)},
        threshold_path=threshold_path, threshold_sha256=threshold_sha256, resume=resume,
        before_intent=before_intent)


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
    parser.add_argument("--resume", action="store_true")
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
        if config["backend"] not in ("language", "decision") or config.get("execution") == "self_hosted_r2":
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
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {
        "bundle_sha256": bundle_digest, "runner_sha256": file_hash(__file__), "aliases": args.models,
        "started_at": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
        "authorization": args.authorization_note, "provider_groups": groups,
        "same_provider_models_do_not_overlap": True, "slots_per_model": 12,
        "no_extra_recovery_after_three_attempts": True}
    manifest_path = args.output / "run-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume or any(previous.get(k) != manifest[k] for k in
                ("bundle_sha256", "runner_sha256", "aliases", "phase", "provider_groups")):
            raise ValueError("resume API run identity differs")
    else:
        write_new(manifest_path, manifest)
    billing_stop = threading.Event()
    def group(aliases):
        for alias in aliases:
            if billing_stop.is_set():
                break
            execute_model(alias, args.bundle, args.output, keys_by_alias[alias], billing_stop,
                          bundle_digest, phase=args.phase, threshold_path=args.thresholds,
                          threshold_sha256=args.thresholds_sha256, resume=args.resume)
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
    atomic(args.output / "completion.json", {"finished_at": datetime.now(timezone.utc).isoformat(),
        "billing_paused": billing_stop.is_set(), "orchestrator_errors": failures, "model_status": statuses})
    return int(bool(failures) or any(s["state"] != "completed" for s in statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())
