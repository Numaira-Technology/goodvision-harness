"""Cumulative model ledger with explicit operation scopes and receipt reuse.

Use at the model request boundary without probes, retries, or response changes.
Request identity and unknown-payment safeguards apply independently of resource budgets.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import math
import re
from pathlib import Path
from threading import RLock
import time
from types import MethodType

from .model_operation import operation_for_request, operation_key, validate_operation
from .summary_recovery import recovery_calls


class LedgerBlocked(RuntimeError):
    def __init__(self, reason, *, scope="shared"):
        super().__init__(reason)
        self.code, self.scope = reason, scope


def request_without_images(value):
    if isinstance(value, dict):
        if set(value) == {"image_data_url_sha256", "data_url_characters"}:
            # Read the old archive without computing or trusting its image digest.
            return {"image_omitted": True, "data_url_characters": value["data_url_characters"]}
        return {key: request_without_images(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [request_without_images(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_omitted": True, "data_url_characters": len(value)}
    return value


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class ModelLedger:
    def __init__(self, root, *, seed_manifest_path, owner_lock_path, seed_manifest_sha256=None,
                 summary_recovery=None, window_replay=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._handles = []
        self.lock, self.poisoned, self.closed = RLock(), False, False
        self.in_flight, self.blocked, self.counts, self.cache = set(), set(), {}, {}
        self.summary_requests = {}
        self.requests_by_operation = {}
        self.unresolved_legacy_contexts = []
        self.conflicting_operations = set()
        self._call_sequence = 0
        self.balance_blocked = False
        self.summary_recovery = summary_recovery
        self.window_replay = copy.deepcopy(window_replay)
        self.recovery_selections = {}
        self.response_errors = set()
        try:
            # Lock the task's model writer and this run directory against concurrent paid dispatch.
            for path in (Path(owner_lock_path).resolve(), self.root / "writer.lock"):
                handle = path.open("a+")
                self._handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = Path(seed_manifest_path).resolve()
            raw = manifest_path.read_bytes()
            self.seed_manifest_path = manifest_path
            # Temporary input compatibility for old launchers. This value is not
            # an identity or a check; new launchers must omit the legacy keyword.
            manifest = json.loads(raw)
            if manifest.get("schema_version") not in {"unified-model-ledger-seed-v1",
                                                       "unified-model-ledger-seed-v2"}:
                raise LedgerBlocked("seed_manifest_schema_mismatch")
            paths = {}
            for name in ("model-requests.jsonl", "model-responses.jsonl", "attempt-ledger.jsonl"):
                entry = manifest["files"][name]
                path = (manifest_path.parent / entry["path"]).resolve()
                paths[name] = [path]
                current = self.root / name
                if current.exists():
                    paths[name].append(current)
            self._load(paths)
        except BaseException:
            self.close()
            raise

    def _load(self, paths):
        requests, responses, terminal, operations = {}, {}, {}, {}
        for path in paths["model-requests.jsonl"]:
            for row in _rows(path):
                call = row["call_id"]
                if call in requests:
                    if (requests[call]["request"] != row["request"]
                            or requests[call].get("context") != row.get("context")):
                        raise LedgerBlocked("duplicate_call_identity_mismatch")
                    continue
                requests[call] = row
                try:
                    operation = operation_for_request(row.get("context", {}), row["request"])
                    if "operation" in row and operation != validate_operation(row["operation"]):
                        raise ValueError("Saved operation differs from its context")
                except (ValueError, KeyError, TypeError, IndexError):
                    if "operation" in row or "operation" in row.get("context", {}):
                        raise LedgerBlocked("saved_operation_invalid") from None
                    self.unresolved_legacy_contexts.append(row.get("context", {}))
                    operations[call] = None
                    continue
                operations[call] = operation
                key = operation_key(operation)
                self.counts[key] = self.counts.get(key, 0) + 1
        for path in paths["model-responses.jsonl"]:
            for row in _rows(path):
                call = row["call_id"]
                sent = requests.get(call)
                if (not sent or sent["request"] != row["request"]
                        or ("operation" in row and row["operation"] != operations[call])):
                    raise LedgerBlocked("receipt_request_mismatch")
                if call in responses and responses[call] != row:
                    raise LedgerBlocked("duplicate_receipt_mismatch")
                responses[call] = row
        dispatched = set()
        for path in paths["attempt-ledger.jsonl"]:
            for row in _rows(path):
                call = row["call_id"]
                if (call not in requests
                        or ("operation" in row and row["operation"] != operations[call])
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
        try:
            superseded, self.recovery_selections = recovery_calls(self.summary_recovery,
                requests=requests, responses=responses, terminal=terminal, operations=operations,
                root=self.root)
        except (ValueError, KeyError, TypeError) as exc:
            raise LedgerBlocked("invalid_explicit_summary_recovery") from exc
        for call, sent in requests.items():
            operation, receipt = operations[call], responses.get(call)
            state = terminal.get(call)
            response = receipt.get("response") if receipt else None
            response_id = response.get("id") if isinstance(response, dict) else None
            status = receipt.get("error", {}).get("status_code") if receipt else None
            http_response = type(status) is int and 400 <= status < 600
            if state == "recorded" and not (response_id or http_response):
                raise LedgerBlocked("recorded_attempt_missing_receipt")
            if operation is None:
                continue
            key = operation_key(operation)
            if call in superseded:
                continue
            self._remember_request(key, operation, sent["request"])
            if state != "recorded":
                self.blocked.add(key)
            elif response_id:
                # Cache empty or malformed answers too; replay them unchanged for the native parser.
                self.cache[key] = (copy.deepcopy(response), call)
            elif http_response and status not in {402, 429}:
                self.response_errors.add(key)
        prefix = str(self.root / "call-")
        prior_sequences = [int(call[len(prefix):]) for call in requests
                           if call.startswith(prefix) and call[len(prefix):].isdigit()]
        self._call_sequence = max([len(requests), *prior_sequences])
        # Balance blocks belong to this run; historical 402s do not permanently block a recovered account.
        self.balance_blocked = (self.root / "balance-blocked.jsonl").exists()
        self._apply_window_replay(operations)

    def _apply_window_replay(self, operations):
        """Carry unique prior sends into one explicitly authorized window replay.

        The authorization is local audit text. Source and session fields narrow
        the budget transfer; lane and frame sequences keep their original scope.
        Receipts, cached responses and unknown-outcome blocks stay on old keys.
        """
        replay = self.window_replay
        if replay is None:
            return
        fields = {"authorization", "source_id", "prior_session", "new_session"}
        if (not isinstance(replay, dict) or set(replay) != fields
                or any(not isinstance(replay[field], str) or not replay[field].strip()
                       for field in fields)
                or replay["prior_session"] == replay["new_session"]):
            raise LedgerBlocked("invalid_explicit_window_replay")
        matched = False
        # Operations are indexed by unique call ID after seed/current deduplication.
        for operation in operations.values():
            if (operation is None or operation["kind"] != "window"
                    or operation["source"] != replay["source_id"]
                    or operation["session"] != replay["prior_session"]):
                continue
            matched = True
            key = operation_key({**operation, "session": replay["new_session"]})
            self.counts[key] = self.counts.get(key, 0) + 1
        if not matched:
            raise LedgerBlocked("window_replay_prior_requests_missing")

    def _append(self, name, value):
        try:
            with (self.root / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.poisoned = True
            raise

    def _remember_request(self, key, operation, request):
        normalized = request_without_images(request)
        previous = self.requests_by_operation.get(key)
        if previous is not None and previous != normalized:
            self.conflicting_operations.add(key)
        self.requests_by_operation.setdefault(key, normalized)
        if operation["kind"] == "summary":
            self.summary_requests.setdefault(key, request)

    def _check_legacy_scope(self, operation):
        for context in self.unresolved_legacy_contexts:
            if context.get("phase") == "same_evidence_summary":
                if operation["kind"] != "summary":
                    continue
                known = [(context.get(field), operation[target]) for field, target in
                         (("qa_id", "question"), ("video_id", "video"))
                         if isinstance(context.get(field), str) and context[field]]
                # Missing scope never authorizes another paid answer. Use any
                # surviving question/video fields to keep the refusal narrow.
                if not known or all(saved == current for saved, current in known):
                    raise LedgerBlocked("unresolved_legacy_answer_scope", scope="source")
            elif context.get("stream_session_id") == operation["session"]:
                raise LedgerBlocked("unresolved_legacy_window_scope", scope="source")
            elif not context.get("stream_session_id"):
                # There is no trustworthy narrow scope for this old dispatch.
                raise LedgerBlocked("legacy_dispatch_scope_missing")

    def prior_summary_request(self, *, operation, model, parameters):
        """Keep paid answer identity stable across later lossless input encodings."""
        with self.lock:
            operation = validate_operation(operation)
            self._check_legacy_scope(operation)
            key = operation_key(operation)
            if key in self.conflicting_operations:
                raise LedgerBlocked("conflicting_operation_payloads", scope="source")
            request = self.summary_requests.get(key)
            if request is None:
                return None
            expected = {key: value for key, value in parameters.items() if value is not None}
            expected["model"] = model
            if {key: value for key, value in request.items() if key != "messages"} != expected:
                raise LedgerBlocked("summary_parameters_changed_after_dispatch", scope="source")
            return copy.deepcopy(request)

    def begin(self, payload, context):
        request = request_without_images(payload)
        try:
            operation = operation_for_request(context, request)
        except (ValueError, KeyError, TypeError, IndexError):
            raise LedgerBlocked("model_operation_scope_missing", scope="source") from None
        replay = self.window_replay
        if (replay is not None and operation["kind"] == "window"
                and operation["source"] == replay["source_id"]
                and operation["session"] == replay["new_session"]):
            context = {**context, "window_replay": copy.deepcopy(replay)}
        key = operation_key(operation)
        with self.lock:
            if self.closed or self.poisoned or self.balance_blocked:
                raise LedgerBlocked("model_ledger_unavailable")
            self._check_legacy_scope(operation)
            if key in self.blocked or key in self.in_flight:
                raise LedgerBlocked("unresolved_same_input_request", scope="source")
            if key in self.response_errors:
                raise LedgerBlocked("http_response_requires_explicit_recovery", scope="source")
            if (key in self.conflicting_operations or
                    (key in self.requests_by_operation and self.requests_by_operation[key] != request)):
                raise LedgerBlocked("operation_payload_changed_after_dispatch", scope="source")
            if key in self.cache:
                response, call = self.cache[key]
                receipt = {"call_id": call, "operation": operation, "attempt": self.counts[key],
                           "response_id": response["id"], "cached": True,
                           "paid_dispatch": False, "context": context, "state": "recorded"}
                self._append("cache-hits.jsonl", receipt)
                return None, copy.deepcopy(response), receipt
            if self.counts.get(key, 0) >= 3:
                raise LedgerBlocked("same_input_attempts_exhausted", scope="source")
            attempt = self.counts.get(key, 0) + 1
            self._call_sequence += 1
            call = str(self.root / ("call-" + str(self._call_sequence)))
            row = {"call_id": call, "operation": operation, "request": request,
                   "attempt": attempt, "context": {**context, "operation": operation},
                   "started_epoch": time.time()}
            recovery = self.recovery_selections.get(operation.get("question"))
            if recovery is not None:
                row["context"].update(recovery_authorization=recovery["authorization"],
                    prior_call_ids=recovery["prior_call_ids"])
            self._append("model-requests.jsonl", row)
            self._append("attempt-ledger.jsonl", {key: row[key] for key in
                         ("call_id", "operation", "attempt", "context")} | {"state": "dispatching"})
            self.counts[key] = attempt
            self._remember_request(key, operation, request)
            self.in_flight.add(key)
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
            status = saved.get("error", {}).get("status_code")
            http_response = type(status) is int and 400 <= status < 600
            state = "recorded" if response_id or http_response else "outcome_unknown"
            usage = response.get("usage") if isinstance(response, dict) else None
            cost = usage.get("cost") if isinstance(usage, dict) else None
            cost_known = type(cost) in {int, float} and math.isfinite(cost) and cost >= 0
            saved.update(request_state="response_received" if response_id or http_response else "outcome_unknown",
                         response_kind="http_error" if http_response else "completion" if response_id else None,
                         cost_status="reported" if cost_known else "unknown",
                         reported_cost_usd=cost if cost_known else None)
            terminal = {key: row[key] for key in ("call_id", "operation", "attempt")}
            terminal.update(state=state, response_id=response_id)
            self._append("model-responses.jsonl", saved)
            self._append("attempt-ledger.jsonl", terminal)
            key = operation_key(row["operation"])
            self.in_flight.discard(key)
            if state == "outcome_unknown":
                self.blocked.add(key)
            elif response_id:
                self.cache[key] = (copy.deepcopy(response), row["call_id"])
            elif http_response and status not in {402, 429}:
                self.response_errors.add(key)
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
                    "body_truncated": truncated,
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
