"""Native Decisions transport with bounded responses and operator-only audit."""

import asyncio
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

import httpx

from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.vlm_transport import create_vlm_http_client
from tools.benchmark_runtime.scene_model import _json, _ResponseValidationError
from tools.benchmark_runtime.vlm_compact_model import EventJSONInspector
from tools.benchmark_runtime.vlm_event_policy import (
    event_policy_snapshot,
    event_policy_text,
)
from tools.benchmark_runtime.vlm_independent_model import IndependentVerdict

from . import profile as p

CHOICES = [
    {
        "value": "positive",
        "description": "true: evidence meets the selected event policy.",
    },
    {"value": "negative", "description": "false: evaluated absent under that policy."},
    {
        "value": "uncertain",
        "description": "null: insufficient evidence under that policy.",
    },
]
VALUES = {"positive": True, "negative": False, "uncertain": None}


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def parse_answers(body, labels=p.LABELS):
    if not isinstance(body, dict) or body.get("model") != p.MODEL:
        raise _ResponseValidationError("invalid_verdict", "decisions_model")
    answers = body.get("answers")
    if (
        not isinstance(answers, list)
        or len(answers) != len(labels)
        or any(not isinstance(a, dict) for a in answers)
        or any(not isinstance(a.get("name"), str) for a in answers)
        or {a["name"] for a in answers} != set(labels)
    ):
        raise _ResponseValidationError("invalid_verdict", "decisions_answer_names")
    by_name = {a["name"]: a for a in answers}
    answers = [by_name[label] for label in labels]
    for answer in answers:
        if (
            answer.get("type") != "choice"
            or not isinstance(answer.get("choice"), str)
            or answer["choice"] not in VALUES
        ):
            raise _ResponseValidationError(
                "invalid_verdict", "decisions_refusal_or_type"
            )
        probs = answer.get("probabilities")
        if (
            not probability(answer.get("confidence"))
            or not isinstance(probs, list)
            or len(probs) != 3
            or any(not isinstance(x, dict) for x in probs)
            or any(not isinstance(x.get("value"), str) for x in probs)
            or {x.get("value") for x in probs} != set(VALUES)
            or any(not probability(x.get("probability")) for x in probs)
            or abs(sum(x["probability"] for x in probs) - 1) > 0.02
        ):
            raise _ResponseValidationError("invalid_verdict", "decisions_probabilities")
    # Confidence is recorded, never used as an extra detection threshold.
    return tuple(
        IndependentVerdict(
            a["name"], (), VALUES[a["choice"]], a["confidence"], (), ("scene_only",)
        )
        for a in answers
    )


def request_body(messages):
    content = [{"type": "input_text", "text": messages[0]["content"]}]
    for part in messages[1]["content"]:
        if part["type"] == "text":
            content.append({"type": "input_text", "text": part["text"]})
        elif part["type"] == "image_url":
            content.append(
                {"type": "input_image", "image_url": part["image_url"]["url"]}
            )
        else:
            raise ValueError("unexpected Decisions input part")
    return {
        "model": p.MODEL,
        "input": [{"role": "user", "content": content}],
        "questions": [
            {
                "type": "choice",
                "name": label,
                "choices": CHOICES,
                "instructions": f"Evaluate {label} independently for the whole visible "
                "scene at the LAST image, using the supplied event policy and scene "
                "aggregation rules. Earlier images provide temporal context.",
            }
            for label in p.LABELS
        ],
    }


def safe_headers(headers):
    """Allow only request IDs, bounded counts and reset durations, never raw text."""
    result = {}
    rid = headers.get("x-request-id", "")
    if re.fullmatch(r"req[-_][A-Za-z0-9_-]{1,160}", rid):
        result["openai_request_id"] = rid
    for key in (
        "openai-processing-ms",
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
    ):
        try:
            value = float(headers[key])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value) and 0 <= value <= 10**12:
            result[key] = value
    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        value = headers.get(key, "")
        if re.fullmatch(r"(?:[0-9]{1,9}(?:\.[0-9]{1,6})?(?:ms|s|m|h))+", value):
            result[key] = value
    return result


class DecisionInspector(EventJSONInspector):
    audit_lane = "events"

    def __init__(self, *, api_key, audit_dir=None):
        super().__init__(
            p.DecisionSettings(),
            model=p.MODEL,
            api_key=api_key,
            max_image_side=720,
            jpeg_quality=90,
        )
        self.audit_path = None
        if audit_dir is not None:
            directory = Path(audit_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.audit_path = directory / f"{os.getpid()}-{uuid.uuid4().hex}.jsonl"

    def _ensure_transport(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = create_vlm_http_client(trust_env=False)

    def build_messages(self, job, *, encoded_images=None):
        if (
            tuple(job.request.event_labels) != p.LABELS
            or event_policy_snapshot(p.LABELS, dict(job.event_sensitivities))["sha256"]
            != p.POLICY_SHA256
        ):
            raise ValueError("Decisions requires the frozen six-event policy")
        messages = super().build_messages(job, encoded_images=encoded_images)
        messages[0]["content"] = event_policy_text(
            p.LABELS, dict(job.event_sensitivities)
        )
        return messages

    @staticmethod
    def _parse(job, response):
        return parse_answers(response, job.request.event_labels)

    def _request_body(self, messages):
        return request_body(messages)

    def _validate_response(self, result):
        return parse_answers(result)

    async def _request(self, payload, headers, *, timing):
        started = time.monotonic()
        timing.update(http_request_attempted=False)
        audit = {
            "request_id": timing.get("request_id"),
            "stream_session_id": timing.get("stream_session_id"),
            "vlm_lane": self.audit_lane,
            "provider": "OpenAI",
            "model": p.MODEL,
            "complete": False,
        }
        response = None
        try:
            body = self._request_body(payload["messages"])
            # Drop every inherited parameter/header. No chat/SSE/provider settings.
            request = httpx.Request(
                "POST",
                p.ENDPOINT,
                json=body,
                headers={
                    "Authorization": headers["Authorization"],
                    "Content-Type": "application/json",
                },
                extensions={"timeout": httpx.Timeout(20).as_dict()},
            )
            timing["decision_question_count"] = len(body["questions"])
            audit["question_count"] = len(body["questions"])
            timing["request_bytes"] = len(request.content)
            timing["request_build_seconds"] = time.monotonic() - started
            timing["http_request_attempted"] = True
            response = await self._http.send(
                request, stream=True, auth=None, follow_redirects=False
            )
            timing["http_status"] = response.status_code
            timing["http_until_headers_seconds"] = time.monotonic() - started
            audit.update(
                http_status=response.status_code, headers=safe_headers(response.headers)
            )
            if "retry-after" in audit["headers"]:
                timing["retry_after_seconds"] = audit["headers"]["retry-after"]
            chunks = bytearray()
            read_started = time.monotonic()
            limit = 256 * 1024 if len(body["questions"]) > 6 else 64 * 1024
            async for chunk in response.aiter_bytes():
                if len(chunks) + len(chunk) > limit:
                    if response.status_code != 200:
                        break
                    raise _ResponseValidationError(
                        "response_size_limit", "decisions_body"
                    )
                chunks.extend(chunk)
            timing["response_read_seconds"] = time.monotonic() - read_started
            timing["response_bytes"] = len(chunks)
            if response.status_code != 200:
                try:
                    error = _json(bytes(chunks), stage="decisions_error").get(
                        "error", {}
                    )
                    code = error.get("code")
                    if code in {
                        "rate_limit_exceeded",
                        "insufficient_quota",
                        "invalid_api_key",
                        "model_not_found",
                        "permission_denied",
                    }:
                        audit["error_code"] = code
                except (ValueError, AttributeError, TypeError):
                    pass
                raise ModelAPIError(
                    "Decisions request failed",
                    code="http_error",
                    status_code=response.status_code,
                )
            result = _json(bytes(chunks), stage="decisions_response")
            if not isinstance(result, dict):
                raise _ResponseValidationError("invalid_verdict", "decisions_response")
            usage = result.get("usage")
            if isinstance(usage, dict):
                for native, common in (
                    ("input_tokens", "prompt_tokens"),
                    ("output_tokens", "completion_tokens"),
                ):
                    value = usage.get(native)
                    if type(value) is int and value >= 0:
                        timing[common] = value
                        audit[native] = value
                result = {
                    **result,
                    "usage": {
                        **usage,
                        **{
                            k: timing[k]
                            for k in ("prompt_tokens", "completion_tokens")
                            if k in timing
                        },
                    },
                }
            # This validation also keeps refusals out of the audit's complete count.
            verdicts = self._validate_response(result)
            audit["complete"] = True
            audit["judgments"] = {v.event_label: v.detected for v in verdicts}
            return result
        except (httpx.RequestError, TimeoutError, asyncio.CancelledError):
            # An incomplete optional error body cannot erase a received status.
            # In particular, the suite must still stop on 401/403 and count 429.
            if response is not None and response.status_code != 200:
                raise ModelAPIError(
                    "Decisions request failed",
                    code="http_error",
                    status_code=response.status_code,
                ) from None
            raise
        finally:
            try:
                if response is not None:
                    await response.aclose()
            except (httpx.RequestError, TimeoutError, asyncio.CancelledError):
                audit["complete"] = False
                if response is not None and response.status_code != 200:
                    raise ModelAPIError(
                        "Decisions request failed",
                        code="http_error",
                        status_code=response.status_code,
                    ) from None
                raise
            finally:
                timing["total_seconds"] = time.monotonic() - started
                self.last_http_timing = dict(timing)
                audit.update(
                    http_request_attempted=timing["http_request_attempted"],
                    http_elapsed_seconds=timing["total_seconds"],
                )
                if self.audit_path is not None:
                    with self.audit_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(audit, allow_nan=False) + "\n")


def inspector(config, *, audit_dir=None):
    if (
        not isinstance(config.model_settings, p.DecisionSettings)
        or config.model != p.MODEL
        or config.output_mode != "events"
        or tuple(config.labels) != p.LABELS
    ):
        raise ValueError(
            "Decisions adapter supports only its frozen scene-events route"
        )
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY is unavailable")
    return DecisionInspector(api_key=config.api_key, audit_dir=audit_dir)
