"""Execute final answers from saved observations, preserving replies and ledger identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from ..model_api import ModelAPIClient, ModelAPIError
from ..scene_model import _json, _strip_json_fence, _ResponseValidationError
from .failures import failure_policy
from .model_ledger import LedgerBlocked
from .privacy import reject_private
from .recording import publish_once
from .answer_artifact import _canonical, ANSWER_CONTRACT, ANSWER_INSTRUCTION, short_answer_messages
from .evidence import NATIVE_SKIP_POLICY
from .summary_input import message_bytes, prepare_summary_messages
from .rate_limit import wait_after_429
from .native_gaps import is_local_gap, disclose_skipped_observations
from .summary_context import prepare_context_messages

STATES = frozenset({"answered", "insufficient_evidence", "incomplete_video",
                    "request_failure", "unsupported"})


def _native_window_gaps_only(evidence):
    """Separate saved window quality from the source's completed lifecycle."""
    provenance = evidence.get("provenance", {})
    manifest = evidence.get("observation_manifest", {})
    failures = evidence.get("failures", [])
    if (provenance.get("coverage_policy") != NATIVE_SKIP_POLICY
            or provenance.get("scheduling_mode") != "native_streaming"
            or not failures or evidence.get("request_failure") is not True
            or not all(evidence.get(key) is True for key in
                       ("source_eof", "tracking_drained", "events_drained", "fusion_drained"))
            or not (manifest.get("person_frames_complete") is True
                    or manifest.get("person_frames_accounted") is True)
            or not (evidence.get("frames") or evidence.get("independent_events")
                    or evidence.get("fused_episodes"))):
        return False
    return all(is_local_gap(failure, evidence) for failure in failures)


def execute_summary(artifact_path, *, artifact_sha256, output, client, ledger,
                    model, parameters, max_input_bytes, context_budget=None, prepare_only=False):
    raw = Path(artifact_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact_sha256:
        raise ValueError("summary artifact file hash mismatch")
    artifact = json.loads(raw)
    request = artifact["summary_request"]
    reject_private(request)
    if (request.get("state") != "not_called"
            or request.get("method") != "unified_same_evidence_summary_v2"
            or request["evidence_sha256"] != artifact["evidence_sha256"]):
        raise ValueError("summary must read an unchanged same-evidence request artifact")
    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("frozen summary input size limit required")
    if {"model", "messages", "stream"} & parameters.keys():
        raise ValueError("summary parameters cannot replace the frozen request")
    messages = request["messages"]
    if len(messages) != 2 or [row.get("role") for row in messages] != ["system", "user"]:
        raise ValueError("expected the saved two-message answer request")
    payload = json.loads(messages[1]["content"])
    evidence, question = payload["evidence"], payload["question"]
    evidence_hash = hashlib.sha256(_canonical({k: v for k, v in evidence.items()
                                               if k != "evidence_sha256"}).encode()).hexdigest()
    if (evidence_hash != artifact["evidence_sha256"]
            or evidence.get("evidence_sha256") != evidence_hash
            or question["qa_id"] != artifact["qa_id"]
            or question["video_id"] != artifact["video_id"]
            or evidence["video_id"] != artifact["video_id"]):
        raise ValueError("final answer request evidence or question identity mismatch")
    complete = (evidence.get("request_failure") is False and not evidence.get("failures")
                and evidence.get("observation_manifest", {}).get("complete") is True
                and all(evidence.get(key) is True for key in
                        ("source_eof", "tracking_drained", "events_drained", "fusion_drained")))
    output = Path(output)
    previous_request = (json.loads((output / "request.json").read_bytes())
                        if output.exists() else None)
    preparation = None
    context_preparation = None
    prior_identity_error = None
    prior_dispatched = False
    skip_disclosure = None
    # A rule-level grounding gap is not a failed native capture. The final
    # model can assess the actual saved evidence and return insufficient evidence.
    native_grounding_gap = (
        evidence.get("provenance", {}).get("coverage_policy") == NATIVE_SKIP_POLICY
        and artifact["parser"]["status"] == "incomplete_video"
        and artifact["parser"].get("reason") == "grounding_lane_not_complete")
    eligible = complete and (native_grounding_gap or artifact["parser"]["status"] not in {
        "request_failure", "incomplete_video", "not_started", "interrupted"})
    native_gaps = _native_window_gaps_only(evidence)
    if native_gaps and artifact["parser"]["status"] not in {
            "not_started", "interrupted"}:
        eligible = True
    if eligible and native_gaps and (previous_request is None or "skip_disclosure" in previous_request):
        # Never use the new note to bypass a paid original or compressed request,
        # including an unknown receipt. Existing outputs keep their representation.
        legacy, legacy_preparation = messages, None
        if len(message_bytes(messages)) > max_input_bytes:
            legacy, legacy_preparation = prepare_summary_messages(messages, max_input_bytes=max_input_bytes)
        if previous_request is not None or not ledger.has_prior_request(
                model=model, messages=messages, parameters=parameters):
            if previous_request is not None or not ledger.has_prior_request(
                    model=model, messages=legacy, parameters=parameters):
                messages = disclose_skipped_observations(messages, evidence)
                skip_disclosure = "native_skipped_observation_times_v1"
            else:
                messages, preparation = legacy, legacy_preparation
    if (eligible and len(message_bytes(messages)) > max_input_bytes
            and (previous_request is None or "input_preparation" in previous_request)):
        # Existing terminal outputs retain their representation. Paid original requests
        # retain their ledger identity, including cached replies and unknown outcomes.
        if previous_request is not None or not ledger.has_prior_request(
                model=model, messages=messages, parameters=parameters):
            messages, preparation = prepare_summary_messages(
                messages, max_input_bytes=max_input_bytes)
    if eligible:
        if previous_request is not None and context_budget is not None:
            if "context_preparation" in previous_request:
                context_preparation = previous_request["context_preparation"]
                if context_preparation["budget"] != context_budget:
                    raise ValueError("Existing final-answer token budget changed")
                messages = previous_request["messages"]
        elif previous_request is None:
            try:
                prior = ledger.prior_summary_request(qa_id=artifact["qa_id"],
                    evidence_sha256=artifact["evidence_sha256"], model=model, parameters=parameters)
            except LedgerBlocked as exc:
                if exc.scope != "source":
                    raise
                prior, prior_identity_error = None, exc
            if prior is not None:
                prior_dispatched = True
                messages, preparation, skip_disclosure = prior["messages"], None, None
            elif prior_identity_error is None:
                # Upgrade only never-dispatched answers, including saved old artifacts.
                # A new prompt must never bypass a successful, invalid or unknown receipt.
                messages = short_answer_messages(question, evidence)
                preparation, skip_disclosure = None, None
                if native_gaps:
                    messages = disclose_skipped_observations(messages, evidence, short_answer=True)
                    skip_disclosure = "native_skipped_observation_times_v1"
                if len(message_bytes(messages)) > max_input_bytes:
                    messages, preparation = prepare_summary_messages(messages, max_input_bytes=max_input_bytes)
                if context_budget is not None:
                    messages, context_preparation = prepare_context_messages(messages,
                        model=model, parameters=parameters, max_input_bytes=max_input_bytes,
                        budget=context_budget)
    encoded = message_bytes(messages)
    identity = {"qa_id": artifact["qa_id"], "video_id": artifact["video_id"],
                "method": request["method"], "evidence_sha256": artifact["evidence_sha256"],
                "artifact_sha256": artifact_sha256, "model": model,
                "messages_sha256": hashlib.sha256(encoded).hexdigest(),
                "same_evidence_not_historical_question_conditioned_summary": True}
    short_contract = ANSWER_INSTRUCTION in messages[0]["content"]
    if short_contract:
        identity["answer_contract"] = ANSWER_CONTRACT
    saved_request = {**identity, "messages": messages, "parameters": parameters,
                     "max_input_bytes": max_input_bytes}
    if preparation is not None:
        saved_request["input_preparation"] = preparation
    if skip_disclosure is not None:
        saved_request["skip_disclosure"] = skip_disclosure
    if context_preparation is not None:
        saved_request["context_preparation"] = context_preparation
    if prepare_only:
        reason = ("prior_summary_identity_blocked" if prior_identity_error is not None else
                  "existing_answer_output_preserved" if previous_request is not None else
                  "previously_dispatched_answer_preserved" if prior_dispatched else
                  "three_lane_observation_failed_or_incomplete" if not eligible else
                  "frozen_input_size_limit" if len(encoded) > max_input_bytes else
                  "model_context_limit" if context_preparation is not None
                  and not context_preparation["usage"]["fits"] else None)
        return {"prepared_request": saved_request, "ready_for_dispatch": reason is None,
                "reason": reason, "input_bytes": len(encoded), "paid_dispatch": False}
    if output.exists():
        previous = json.loads((output / "outcome.json").read_bytes())
        if previous_request != saved_request or any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("existing final-answer identity changed")
        if (previous.get("status") == "interrupted"
                or (previous.get("failure_policy") or {}).get("scope") == "shared"):
            raise RuntimeError("final answer requires independent recovery with the cumulative ledger")
        if previous.get("status") not in STATES | {"invalid_answer", "not_called"}:
            raise ValueError("unknown final-answer terminal state")
        return previous
    output.mkdir(parents=True, exist_ok=False)
    publish_once(output / "request.json", saved_request)
    if prior_identity_error is not None:
        result = {**identity, "status": "request_failure", "final_answer": None,
                  "failure_policy": failure_policy(prior_identity_error), "paid_dispatch": False}
        publish_once(output / "outcome.json", result)
        return result
    if not eligible:
        result = {**identity, "status": "not_called", "final_answer": None,
                  "reason": "three_lane_observation_failed_or_incomplete",
                  "observation_status": artifact["parser"]["status"], "paid_dispatch": False}
        publish_once(output / "outcome.json", result)
        return result
    if len(encoded) > max_input_bytes:
        result = {**identity, "status": "not_called", "final_answer": None,
                  "reason": "frozen_input_size_limit", "input_bytes": len(encoded),
                  "max_input_bytes": max_input_bytes, "paid_dispatch": False}
        publish_once(output / "outcome.json", result)
        return result
    if context_preparation is not None and not context_preparation["usage"]["fits"]:
        result = {**identity, "status": "not_called", "final_answer": None,
                  "reason": "model_context_limit", "paid_dispatch": False,
                  "context_usage": context_preparation["usage"]}
        publish_once(output / "outcome.json", result)
        return result
    started, receipt, text = time.monotonic(), None, None
    backoff_seconds = 0.0
    context = {"phase": "same_evidence_summary", "qa_id": artifact["qa_id"],
               "video_id": artifact["video_id"], "evidence_sha256": artifact["evidence_sha256"]}
    try:
        for attempt in range(3):
            try:
                response, receipt = ledger.complete(client, model=model, messages=messages,
                    parameters=parameters, context=context)
                break
            except ModelAPIError as exc:
                receipt = getattr(exc, "evaluation_ledger_receipt", None)
                if exc.status_code != 429:
                    raise
                elapsed, retry_allowed = wait_after_429(exc, ledger=ledger, context=context,
                                                        receipt=receipt, clock=time)
                backoff_seconds += elapsed
                if attempt == 2 or (receipt or {}).get("attempt", 0) >= 3 or not retry_allowed:
                    raise
        publish_once(output / "response.json", response)
        text = ModelAPIClient.assistant_text(response)
        if response.get("choices", [{}])[0].get("finish_reason") == "length":
            raise _ResponseValidationError("output_truncated", "summary_answer")
        parsed = _json(_strip_json_fence(text), stage="summary_answer")
        if short_contract:
            predictions = parsed.get("predictions") if isinstance(parsed, dict) else None
            if (not isinstance(parsed, dict) or set(parsed) != {"predictions"}
                    or not isinstance(predictions, list) or len(predictions) != 1
                    or not isinstance(predictions[0], dict)
                    or set(predictions[0]) != {"qa_id", "prediction"}
                    or predictions[0]["qa_id"] != artifact["qa_id"]
                    or not isinstance(predictions[0]["prediction"], str)):
                raise _ResponseValidationError("short_answer_contract", "summary_answer")
            answer = predictions[0]["prediction"]
            parsed = {"status": "answered" if answer.strip() else "insufficient_evidence",
                      "final_answer": answer if answer.strip() else None, "evidence_refs": []}
        elif (not isinstance(parsed, dict) or set(parsed) != {"status", "final_answer", "evidence_refs"}
                or not isinstance(parsed["status"], str) or parsed["status"] not in STATES
                or not isinstance(parsed["evidence_refs"], list)
                or (parsed["final_answer"] is not None and not isinstance(parsed["final_answer"], str))
                or (parsed["status"] == "answered" and not parsed["final_answer"])):
            raise _ResponseValidationError("summary_answer_contract", "summary_answer")
        # Preserve original references; do not judge correctness or retry semantic errors.
        result = {**identity, **parsed, "raw_text": text, "ledger_receipt": receipt,
                  "client_wall_seconds": time.monotonic() - started,
                  "rate_limit_backoff_seconds": backoff_seconds,
                  "model_seconds": time.monotonic() - started - backoff_seconds,
                  "references_validated": False}
    except (ModelAPIError, _ResponseValidationError, LedgerBlocked) as exc:
        policy = failure_policy(exc)
        if isinstance(exc, _ResponseValidationError):
            policy = {"scope": "source", "action": "preserve_invalid_answer_no_retry", "code": exc.reason}
        result = {**identity,
                  "status": "invalid_answer" if isinstance(exc, _ResponseValidationError) else "request_failure",
                  "final_answer": None, "raw_text": text, "failure_policy": policy,
                  "ledger_receipt": receipt, "client_wall_seconds": time.monotonic() - started,
                  "rate_limit_backoff_seconds": backoff_seconds,
                  "model_seconds": time.monotonic() - started - backoff_seconds}
        publish_once(output / "outcome.json", result)
        if policy["scope"] == "shared":
            raise
        return result
    except Exception as exc:
        publish_once(output / "outcome.json", {**identity, "status": "interrupted",
                     "final_answer": None, "raw_text": text, "failure_policy": failure_policy(exc)})
        raise
    publish_once(output / "outcome.json", result)
    return result
