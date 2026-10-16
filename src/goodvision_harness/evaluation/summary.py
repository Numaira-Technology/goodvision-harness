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
from .answer_artifact import _canonical

STATES = frozenset({"answered", "insufficient_evidence", "incomplete_video",
                    "request_failure", "unsupported"})


def execute_summary(artifact_path, *, artifact_sha256, output, client, ledger,
                    model, parameters, max_input_bytes):
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
    encoded = json.dumps(request["messages"], ensure_ascii=False, separators=(",", ":")).encode()
    identity = {"qa_id": artifact["qa_id"], "video_id": artifact["video_id"],
                "method": request["method"], "evidence_sha256": artifact["evidence_sha256"],
                "artifact_sha256": artifact_sha256, "model": model,
                "messages_sha256": hashlib.sha256(encoded).hexdigest(),
                "same_evidence_not_historical_question_conditioned_summary": True}
    saved_request = {**identity, "messages": request["messages"], "parameters": parameters,
                     "max_input_bytes": max_input_bytes}
    if output.exists():
        previous_request = json.loads((output / "request.json").read_bytes())
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
    if not complete or artifact["parser"]["status"] in {"request_failure", "incomplete_video", "not_started", "interrupted"}:
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
    started, receipt, text = time.monotonic(), None, None
    try:
        response, receipt = ledger.complete(
            client, model=model, messages=request["messages"], parameters=parameters,
            context={"phase": "same_evidence_summary", "qa_id": artifact["qa_id"],
                     "video_id": artifact["video_id"], "evidence_sha256": artifact["evidence_sha256"]})
        publish_once(output / "response.json", response)
        text = ModelAPIClient.assistant_text(response)
        parsed = _json(_strip_json_fence(text), stage="summary_answer")
        if (not isinstance(parsed, dict) or set(parsed) != {"status", "final_answer", "evidence_refs"}
                or not isinstance(parsed["status"], str) or parsed["status"] not in STATES
                or not isinstance(parsed["evidence_refs"], list)
                or (parsed["final_answer"] is not None and not isinstance(parsed["final_answer"], str))
                or (parsed["status"] == "answered" and not parsed["final_answer"])):
            raise _ResponseValidationError("summary_answer_contract", "summary_answer")
        # Preserve original references; do not judge correctness or retry semantic errors.
        result = {**identity, **parsed, "raw_text": text, "ledger_receipt": receipt,
                  "client_wall_seconds": time.monotonic() - started, "references_validated": False}
    except (ModelAPIError, _ResponseValidationError, LedgerBlocked) as exc:
        policy = failure_policy(exc)
        if isinstance(exc, _ResponseValidationError):
            policy = {"scope": "source", "action": "preserve_invalid_answer_no_retry", "code": exc.reason}
        result = {**identity,
                  "status": "invalid_answer" if isinstance(exc, _ResponseValidationError) else "request_failure",
                  "final_answer": None, "raw_text": text, "failure_policy": policy,
                  "ledger_receipt": receipt, "client_wall_seconds": time.monotonic() - started}
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
