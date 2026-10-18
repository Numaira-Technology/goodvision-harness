"""Cumulative model ledger with full-input hashes, unknown-outcome blocks, and receipt reuse.

Use at the model request boundary without probes, retries, or response changes.
Request identity and unknown-payment safeguards apply independently of resource budgets.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import math
import re
from pathlib import Path
from threading import RLock
import time
from types import MethodType
from uuid import uuid4


class LedgerBlocked(RuntimeError):
    def __init__(self, reason, *, scope="shared"):
        super().__init__(reason)
        self.code, self.scope = reason, scope


def request_without_images(value):
    if isinstance(value, dict):
        return {key: request_without_images(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [request_without_images(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_data_url_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "data_url_characters": len(value)}
    return value


def request_digest(request):
    # Preserve the ledger's hash encoding and the IDs in the original prompt.
    raw = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class ModelLedger:
    def __init__(self, root, *, seed_manifest_path, seed_manifest_sha256, owner_lock_path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._handles = []
        self.lock, self.poisoned, self.closed = RLock(), False, False
        self.in_flight, self.blocked, self.counts, self.cache = set(), set(), {}, {}
        self.balance_blocked = False
        try:
            # Lock the task's model writer and this run directory against concurrent paid dispatch.
            for path in (Path(owner_lock_path).resolve(), self.root / "writer.lock"):
                handle = path.open("a+")
                self._handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = Path(seed_manifest_path).resolve()
            raw = manifest_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != seed_manifest_sha256:
                raise LedgerBlocked("seed_manifest_hash_mismatch")
            self.seed_manifest_sha256 = seed_manifest_sha256
            manifest = json.loads(raw)
            if manifest.get("schema_version") != "unified-model-ledger-seed-v1":
                raise LedgerBlocked("seed_manifest_schema_mismatch")
            paths = {}
            for name in ("model-requests.jsonl", "model-responses.jsonl", "attempt-ledger.jsonl"):
                entry = manifest["files"][name]
                path = (manifest_path.parent / entry["path"]).resolve()
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise LedgerBlocked("seed_file_hash_mismatch")
                paths[name] = [path]
                current = self.root / name
                if current.exists():
                    paths[name].append(current)
            self._load(paths)
        except BaseException:
            self.close()
            raise

    def _load(self, paths):
        requests, responses, terminal = {}, {}, {}
        for path in paths["model-requests.jsonl"]:
            for row in _rows(path):
                call, digest = row["call_id"], request_digest(row["request"])
                if row["request_sha256"] != digest:
                    raise LedgerBlocked("request_hash_mismatch")
                if call in requests:
                    if requests[call]["request"] != row["request"]:
                        raise LedgerBlocked("duplicate_call_identity_mismatch")
                    continue
                requests[call] = row
                self.counts[digest] = self.counts.get(digest, 0) + 1
        for path in paths["model-responses.jsonl"]:
            for row in _rows(path):
                call = row["call_id"]
                sent = requests.get(call)
                if (not sent or sent["request"] != row["request"]
                        or sent["request_sha256"] != row["request_sha256"]):
                    raise LedgerBlocked("receipt_request_mismatch")
                if call in responses and responses[call] != row:
                    raise LedgerBlocked("duplicate_receipt_mismatch")
                responses[call] = row
        dispatched = set()
        for path in paths["attempt-ledger.jsonl"]:
            for row in _rows(path):
                call = row["call_id"]
                if (call not in requests or row["request_sha256"] != requests[call]["request_sha256"]
                        or row["state"] not in {"dispatching", "recorded", "outcome_unknown"}):
                    raise LedgerBlocked("attempt_identity_mismatch")
                if row["state"] == "dispatching":
                    dispatched.add(call)
                else:
                    if call in terminal and terminal[call] != row["state"]:
                        raise LedgerBlocked("attempt_terminal_conflict")
                    terminal[call] = row["state"]
        if dispatched != requests.keys():
            raise LedgerBlocked("attempt_dispatch_missing")
        for call, sent in requests.items():
            digest, receipt = sent["request_sha256"], responses.get(call)
            state = terminal.get(call)
            response = receipt.get("response") if receipt else None
            response_id = response.get("id") if isinstance(response, dict) else None
            known_rejection = receipt and receipt.get("error", {}).get("status_code") in (402, 429)
            if state == "recorded" and not (response_id or known_rejection):
                raise LedgerBlocked("recorded_attempt_missing_receipt")
            if state != "recorded":
                self.blocked.add(digest)
            elif response_id:
                # Cache empty or malformed answers too; replay them unchanged for the native parser.
                self.cache[digest] = (copy.deepcopy(response), call)
        # Balance blocks belong to this run; historical 402s do not permanently block a recovered account.
        self.balance_blocked = (self.root / "balance-blocked.jsonl").exists()

    def _append(self, name, value):
        try:
            with (self.root / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.poisoned = True
            raise

    def begin(self, payload, context):
        request = request_without_images(payload)
        digest = request_digest(request)
        with self.lock:
            if self.closed or self.poisoned or self.balance_blocked:
                raise LedgerBlocked("model_ledger_unavailable")
            if digest in self.blocked or digest in self.in_flight:
                raise LedgerBlocked("unresolved_same_input_request", scope="source")
            if digest in self.cache:
                response, call = self.cache[digest]
                receipt = {"call_id": call, "request_sha256": digest,
                           "response_id": response["id"], "cached": True,
                           "paid_dispatch": False, "context": context, "state": "recorded"}
                self._append("cache-hits.jsonl", receipt)
                return None, copy.deepcopy(response), receipt
            if self.counts.get(digest, 0) >= 3:
                raise LedgerBlocked("same_input_attempts_exhausted", scope="source")
            attempt = self.counts.get(digest, 0) + 1
            row = {"call_id": str(uuid4()), "request_sha256": digest, "request": request,
                   "attempt": attempt, "context": context, "started_epoch": time.time()}
            self._append("model-requests.jsonl", row)
            self._append("attempt-ledger.jsonl", {key: row[key] for key in
                         ("call_id", "request_sha256", "attempt", "context")} | {"state": "dispatching"})
            self.counts[digest] = attempt
            self.in_flight.add(digest)
            return row, None, None

    def finish(self, row, *, response=None, error=None):
        with self.lock:
            saved = copy.deepcopy(row)
            if response is not None:
                saved["response"] = response
            if error is not None:
                saved["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None),
                                  "status_code": getattr(error, "status_code", None)}
                retry_after = getattr(error, "retry_after_seconds", None)
                if retry_after is not None:
                    saved["error"]["retry_after_seconds"] = retry_after
                diagnostic = getattr(error, "evaluation_diagnostic", None)
                if diagnostic is not None:
                    saved["error"]["diagnostic"] = diagnostic
            response_id = response.get("id") if isinstance(response, dict) else None
            known_rejection = saved.get("error", {}).get("status_code") in (402, 429)
            state = "recorded" if response_id or known_rejection else "outcome_unknown"
            terminal = {key: row[key] for key in ("call_id", "request_sha256", "attempt")}
            terminal.update(state=state, response_id=response_id)
            self._append("model-responses.jsonl", saved)
            self._append("attempt-ledger.jsonl", terminal)
            digest = row["request_sha256"]
            self.in_flight.discard(digest)
            if state == "outcome_unknown":
                self.blocked.add(digest)
            elif response_id:
                self.cache[digest] = (copy.deepcopy(response), row["call_id"])
            if saved.get("error", {}).get("status_code") == 402:
                self.balance_blocked = True
                self._append("balance-blocked.jsonl", terminal)
            return {**terminal, "cached": False, "paid_dispatch": True}

    def attach(self, inspector, context):
        """Wrap one inspector's transport without changing prompts, parsing, or timeouts."""
        original, ledger = inspector._request, self

        async def error_observer(response):
            body = bytearray()
            truncated = False
            async for chunk in response.aiter_bytes():
                remaining = 4096 - len(body)
                body.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated = True
                    break
            text = bytes(body).decode("utf-8", errors="replace")
            secret = inspector._api_key
            if isinstance(secret, str) and secret:
                text = text.replace(secret, "[REDACTED]")
            text = re.sub(r"(?i)(bearer\s+|sk-[a-z0-9-]*)([a-z0-9_-]{10,})", "[REDACTED]", text)
            try:
                delay = float(response.headers.get("retry-after", "nan"))
            except ValueError:
                delay = float("nan")
            return {"http_status": response.status_code, "body_excerpt_redacted": text,
                    "body_truncated": truncated, "body_sha256": hashlib.sha256(body).hexdigest(),
                    "retry_after_seconds": delay if math.isfinite(delay) and delay >= 0 else None}

        inspector.evaluation_error_observer = error_observer

        async def recorded(instance, payload, headers, *, timing):
            current_context = {**context, **getattr(instance, "evaluation_context", {})}
            row, cached, receipt = ledger.begin(payload, current_context)
            if row is None:
                instance.last_ledger_receipt = receipt
                return cached
            response, error = None, None
            try:
                response = await original(payload, headers, timing=timing)
                return response
            except BaseException as exc:
                error = exc
                raise
            finally:
                instance.last_ledger_receipt = ledger.finish(row, response=response, error=error)

        inspector._request = MethodType(recorded, inspector)
        inspector.last_ledger_receipt = None
        return inspector

    def has_prior_request(self, *, model, messages, parameters):
        """Read existing send history before changing an answer's representation."""
        payload = {key: value for key, value in parameters.items() if value is not None}
        payload.update(model=model, messages=messages)
        digest = request_digest(request_without_images(payload))
        with self.lock:
            return digest in self.counts

    def complete(self, client, *, model, messages, parameters, context):
        """Synchronous same-evidence summary boundary; disable the client's own retries."""
        if client.settings.max_retries != 0:
            raise LedgerBlocked("hidden_model_retry_forbidden")
        payload = {key: value for key, value in parameters.items() if value is not None}
        payload.update(model=model, messages=messages)
        row, cached, receipt = self.begin(payload, context)
        if row is None:
            return cached, receipt
        response, error = None, None
        try:
            response = client.chat_completion(model=model, messages=messages, **parameters)
            return response, self.finish(row, response=response)
        except BaseException as exc:
            error = exc
            # Do not retry a failed finish write and risk a conflicting terminal record.
            if not self.poisoned:
                receipt = self.finish(row, response=response, error=error)
                exc.evaluation_ledger_receipt = receipt
            raise

    def close(self):
        self.closed = True
        for handle in reversed(self._handles):
            handle.close()
        self._handles.clear()
