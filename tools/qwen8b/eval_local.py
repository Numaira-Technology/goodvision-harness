#!/usr/bin/env python3
"""Run unified-frame-120-v2 against a self-hosted OpenAI-compatible Qwen8B.

No OpenRouter key, no generation lookup, no provider fallback. The local server
must identify as qwen/qwen3-vl-8b-instruct. Headline scoring stays correct/all.
"""
from __future__ import annotations

import argparse
import base64
import http.client
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from tools.unified_benchmark.adapters import parse_response, request_body
from tools.unified_benchmark.api_run import InvalidResponse, MAX_RESPONSE_BYTES, atomic, http_error_body, retry_after
from tools.unified_benchmark.contract import canonical, digest, file_hash, write_new, write_receipt
from tools.unified_benchmark.retry import RequestFailure
from tools.unified_benchmark.workflow import verify_bundle
from tools.unified_benchmark.receipts import raw_fields


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
        status = None
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                status = getattr(response, "status", None)
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except http.client.HTTPException as error:
            partial = getattr(error, "partial", b"")
            if not isinstance(partial, bytes):
                partial = b""
            raise InvalidResponse(
                "invalid or incomplete HTTP response",
                status,
                partial[:MAX_RESPONSE_BYTES],
                truncated=len(partial) > MAX_RESPONSE_BYTES,
            ) from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InvalidResponse(
                "oversized response", status, raw[:MAX_RESPONSE_BYTES], truncated=True
            )
        try:
            payload = json.loads(raw)
        except ValueError as error:
            raise InvalidResponse("invalid JSON response", status, raw) from error
        if not isinstance(payload, dict):
            raise InvalidResponse("invalid response envelope", status, raw)
        return payload

    def __call__(self, unit):
        attempt = self.ledger.state(unit, self.config["kind"])[0]
        prefix = digest(unit["id"]) + f"-attempt-{attempt}"
        receipt = self.output / "receipts" / (prefix + ".json")

        def save(record):
            write_receipt(
                receipt,
                record | {"unit": unit["id"], "request_sha256": unit["request_sha256"], "recorded_at_unix": time.time()},
            )
            return {
                "receipt": str(receipt.relative_to(self.output)),
                "receipt_sha256": file_hash(receipt),
            }

        _url, body = request_body(self.config, unit, self.bundle)
        if digest(body) != unit["request_sha256"]:
            raise RequestFailure("request fingerprint mismatch", fatal=True)
        try:
            payload = self.call(body)
        except InvalidResponse as error:
            metadata = save({
                "http_status": error.status,
                "transport_error": type(error).__name__,
                "detail": str(error),
                "body_base64": base64.b64encode(error.body).decode(),
                "body_truncated": error.truncated,
                "body_incomplete": True,
                "captured_body_bytes": len(error.body),
            })
            raise RequestFailure(str(error), status=error.status, metadata=metadata) from error
        except urllib.error.HTTPError as error:
            metadata = save({
                **http_error_body(error),
                "http_status": error.code,
                "retry_after": error.headers.get("Retry-After"),
            })
            raise RequestFailure(
                "HTTP " + str(error.code),
                status=error.code,
                retry_after=retry_after(error.headers.get("Retry-After")),
                fatal=error.code in (400, 401, 403, 404),
                metadata=metadata,
            ) from error
        except (OSError, TimeoutError, ValueError) as error:
            metadata = save({
                "transport_error": type(error).__name__,
                "detail": str(error)[:500],
                "body_incomplete": isinstance(error, TimeoutError),
            })
            raise RequestFailure(type(error).__name__, metadata=metadata) from error
        metadata = save({"http_status": 200, **raw_fields(response=payload)})
        if not isinstance(payload, dict):
            raise RequestFailure(
                "actual model identity missing or invalid",
                fatal=True,
                metadata=metadata,
            )
        actual_model = payload.get("model")
        if isinstance(actual_model, str) and actual_model.strip():
            metadata["actual_model"] = actual_model
        metadata["actual_provider"] = "self-hosted-r2-4090"
        if not isinstance(actual_model, str) or not actual_model.strip():
            raise RequestFailure(
                "actual model identity missing or invalid",
                fatal=True,
                metadata=metadata,
            )
        if actual_model != self.expected_model:
            raise RequestFailure(
                "actual model mismatch: " + repr(actual_model)[:512],
                fatal=True,
                metadata=metadata,
            )
        try:
            answers = parse_response(self.config, unit, payload)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            answers = {}
            metadata["answer_error"] = str(error)[:500]
        return answers, metadata


def execute(alias, bundle, root, endpoint, bundle_digest, phase, *, resume=False, before_intent=None):
    from tools.unified_benchmark.http_execution import execute_model
    config = json.loads((bundle / "configs" / f"{alias}.json").read_text(encoding="utf-8"))
    if (config.get("execution") != "self_hosted_r2" or config.get("backend") != "language"
            or config.get("kind") != "choice"):
        raise ValueError("eval_local requires a self_hosted_r2 hard-answer language profile")
    return execute_model(alias, bundle, root, bundle_digest,
        lambda cfg, directory, ledger: LocalTransport(bundle, cfg, endpoint, directory, ledger),
        phase=phase, connection={"endpoint": endpoint}, resume=resume, local=True,
        before_intent=before_intent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible origin, no trailing path")
    parser.add_argument("--model", default="qwen8b_r2_host")
    parser.add_argument("--phase", choices=("validation", "test", "both"), default="both")
    parser.add_argument("--authorization-note", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("no inference without explicit --execute")
    bundle_digest = verify_bundle(args.bundle)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config = json.loads((args.bundle / "configs" / (args.model + ".json")).read_text(encoding="utf-8"))
    manifest = {
        "bundle_sha256": bundle_digest, "runner_sha256": file_hash(__file__),
        "alias": args.model, "endpoint": args.endpoint, "phase": args.phase,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "authorization": args.authorization_note, "execution": "self_hosted_r2",
        "engine": config.get("engine", "transformers"),
        "engine_version": config.get("engine_version"),
        "slots_per_model": 12}
    path = args.output / "run-manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if not args.resume or any(previous.get(k) != manifest[k] for k in
                ("bundle_sha256", "runner_sha256", "alias", "endpoint", "phase", "engine", "engine_version")):
            raise ValueError("resume Qwen run identity differs")
    else:
        write_new(path, manifest)
    execute(args.model, args.bundle, args.output, args.endpoint, bundle_digest, args.phase, resume=args.resume)
    status = json.loads((args.output / args.model / "status.json").read_text(encoding="utf-8"))
    atomic(args.output / "completion.json", {
        "finished_at": datetime.now(timezone.utc).isoformat(), "model_status": status})
    return 0 if status.get("state") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
