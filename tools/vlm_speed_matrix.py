"""Benchmark the native rich client; JSONL validity means protocol, not accuracy.

Each case gets one persistent client and event loop. Trials interleave cases and
rotate input images, alternating original/mirrored pixels for each image. The
first trial includes lazy connection setup; later trials reuse the same pool.
Example: --env-file .env --image frame.jpg --cases tmp/cases.json
         --output tmp/speed.jsonl --repeats 4 --trace-transport
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from dotenv import dotenv_values
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goodvision_harness import vlm_rich_labeling as vlm
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.streaming_model_api import _safe_usage
from goodvision_harness.video_to_stream import encode_jpeg
from goodvision_harness.vlm_validation import validate_complete_compact_annotation


def load_key(env_file: Path, key_env: str = "OPENROUTER_API_KEY") -> str:
    """Read exactly the requested credential; never import route/proxy settings."""
    if not env_file.is_file():
        raise ValueError("environment file does not exist")
    values = dotenv_values(env_file, interpolate=False)
    key = (os.environ.get(key_env) or values.get(key_env) or "").strip()
    if not key:
        raise ValueError("requested API key is missing")
    return key


def load_cases(path: Path) -> list[dict[str, Any]]:
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, list) or not values:
        raise ValueError("cases must be a non-empty JSON list")
    defaults = {
        "execution_mode": "single",
        "structured_output": False,
        "side": 720,
        "http2": False,
        "trust_env": True,
        "keepalive_expiry": 60.0,
        "stream": True,
        "max_tokens": 1800,
        "prompt_cache": False,
        "request_parameters": {},
        "tls_cadata": False,
        "prompt_suffix": "",
        "wire_format": "compact",
        "require_complete": False,
        "omit_example": False,
        "parallel_strategy": "crops",
    }
    cases = []
    identifiers = set()
    for raw in values:
        if not isinstance(raw, dict) or set(raw) - (set(defaults) | {"id", "model"}):
            raise ValueError("case has unknown fields or is not an object")
        case = {**defaults, **raw}
        for key in ("id", "model"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError("case requires non-empty id and model")
        if case["id"] in identifiers:
            raise ValueError("case ids must be unique")
        identifiers.add(case["id"])
        for key in (
            "structured_output",
            "http2",
            "trust_env",
            "stream",
            "prompt_cache",
            "tls_cadata",
            "require_complete",
            "omit_example",
        ):
            if not isinstance(case[key], bool):
                raise ValueError("case switches must be JSON booleans")
        for key in ("side", "max_tokens"):
            if (
                isinstance(case[key], bool)
                or not isinstance(case[key], int)
                or case[key] < 1
            ):
                raise ValueError("side and max_tokens must be positive integers")
        expiry = case["keepalive_expiry"]
        if (
            isinstance(expiry, bool)
            or not isinstance(expiry, (int, float))
            or not math.isfinite(expiry)
            or expiry < 0
        ):
            raise ValueError("keepalive_expiry must be finite and non-negative")
        if case["execution_mode"] not in ("single", "parallel"):
            raise ValueError("execution_mode must be single or parallel")
        if case["wire_format"] not in ("compact", "named"):
            raise ValueError("wire_format must be compact or named")
        if case["parallel_strategy"] not in ("crops", "full_frame"):
            raise ValueError("parallel_strategy must be crops or full_frame")
        if (
            case["parallel_strategy"] == "full_frame"
            and case["execution_mode"] != "parallel"
        ):
            raise ValueError("full_frame requires parallel execution")
        if not isinstance(case["request_parameters"], dict):
            raise ValueError("request_parameters must be an object")
        if not isinstance(case["prompt_suffix"], str):
            raise ValueError("prompt_suffix must be a string")
        if {"model", "messages", "max_tokens", "stream", "stream_options"} & set(
            case["request_parameters"]
        ):
            raise ValueError("request_parameters overrides a fixed field")
        json.dumps(case, allow_nan=False)
        cases.append(case)
    return cases


class MatrixClient(vlm.VLMRichLabelClient):
    """Change pool options while keeping the production client's loop lifecycle."""

    def __init__(
        self,
        *args: Any,
        connection_options: Mapping[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.connection_options = dict(connection_options)
        self.transport = transport
        self.pool_creation_seconds = 0.0

    def _connection_pool(self) -> httpx.AsyncClient:
        with self._state_lock:
            if self._closed:
                raise ModelAPIError("rich annotation client is closed")
            if self._http_client is None:
                started = time.perf_counter()
                verify: Any = True
                if self.connection_options.get("tls_cadata", False):
                    from goodvision_harness.vlm_transport import verified_ssl_context

                    verify = verified_ssl_context(
                        trust_env=self.connection_options["trust_env"]
                    )
                self._http_client = httpx.AsyncClient(
                    http2=self.connection_options["http2"],
                    trust_env=self.connection_options["trust_env"],
                    limits=httpx.Limits(
                        max_connections=3,
                        max_keepalive_connections=3,
                        keepalive_expiry=self.connection_options["keepalive_expiry"],
                    ),
                    follow_redirects=False,
                    transport=self.transport,
                    verify=verify,
                )
                self.pool_creation_seconds = time.perf_counter() - started
            return self._http_client


async def complete_nonstream(
    client: httpx.AsyncClient,
    *,
    settings: ModelAPISettings,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    deadline_seconds: float,
    request_parameters: Mapping[str, Any] | None = None,
    trace_transport: bool = False,
) -> dict[str, Any]:
    """One bounded JSON POST, no retries, with safe errors and metadata only."""
    if (
        isinstance(deadline_seconds, bool)
        or not isinstance(deadline_seconds, (int, float))
        or not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
    ):
        raise ModelAPIError("model JSON deadline must be finite and positive")
    parameters = dict(request_parameters or {})
    if {"model", "messages", "stream", "stream_options"} & parameters.keys():
        raise ModelAPIError("model JSON parameters override a fixed field")
    if not api_key.strip() or not model.strip() or not messages:
        raise ModelAPIError("model JSON request is incomplete")
    started = time.perf_counter()
    timing: dict[str, Any] = {}
    transport: dict[str, Any] = {"trace_enabled": trace_transport}
    phase_starts: dict[str, float] = {}
    phases = {
        "connect_tcp": "connect_tcp_seconds",
        "start_tls": "tls_seconds",
        "send_request_headers": "request_headers_seconds",
        "send_request_body": "request_body_seconds",
        "receive_response_headers": "wait_response_headers_seconds",
    }
    if trace_transport:
        transport.update(tcp_connect_count=0, trace_events_observed=False)

    async def trace(name: str, _info: dict[str, Any]) -> None:
        parts = name.split(".")
        if (
            len(parts) != 3
            or parts[0] not in {"connection", "proxy", "http11", "http2"}
            or parts[1] not in phases
            or parts[2] not in {"started", "complete", "failed"}
        ):
            return
        transport["trace_events_observed"] = True
        identity = ".".join(parts[:2])
        if parts[2] == "started":
            phase_starts[identity] = time.perf_counter()
        else:
            phase_started = phase_starts.pop(identity, None)
            if phase_started is not None:
                metric = phases[parts[1]]
                timing[metric] = (
                    timing.get(metric, 0.0) + time.perf_counter() - phase_started
                )
            if parts[1:] == ["connect_tcp", "complete"]:
                transport["tcp_connect_count"] += 1

    async def request() -> dict[str, Any]:
        headers = httpx.Headers(settings.extra_headers)
        headers.update(
            {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        async with client.stream(
            "POST",
            settings.endpoint,
            json={**parameters, "model": model, "messages": messages, "stream": False},
            headers=headers,
            timeout=httpx.Timeout(deadline_seconds),
            follow_redirects=False,
            extensions={"trace": trace} if trace_transport else None,
        ) as response:
            timing["response_headers_seconds"] = time.perf_counter() - started
            version = response.extensions.get("http_version")
            if version in (b"HTTP/1.0", b"HTTP/1.1", b"HTTP/2", b"HTTP/3"):
                transport["http_version"] = version.decode("ascii")
            if response.status_code != 200:
                raise ModelAPIError(f"model JSON returned HTTP {response.status_code}")
            media_type = response.headers.get("content-type", "").split(";", 1)[0]
            if media_type.strip().lower() != "application/json":
                raise ModelAPIError("model JSON has unexpected content type")
            await response.aread()
            timing["response_eof_seconds"] = time.perf_counter() - started
            data = response.json()
        if not isinstance(data, dict) or "error" in data:
            raise ModelAPIError("model JSON response is invalid")
        choices = data.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise ModelAPIError("model JSON has no choice")
        if choices[0].get("finish_reason") != "stop":
            raise ModelAPIError("model JSON did not finish normally")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelAPIError("model JSON has no assistant content")
        timing["elapsed_seconds"] = time.perf_counter() - started
        result = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ],
            "usage": _safe_usage(data.get("usage")),
            "_timing": timing,
            "_transport": transport,
        }
        for field in ("id", "provider", "service_tier"):
            value = data.get(field)
            if isinstance(value, str) and 0 < len(value) <= 256 and value.isprintable():
                result[field] = value
        return result

    try:
        return await asyncio.wait_for(request(), timeout=deadline_seconds)
    except (asyncio.TimeoutError, TimeoutError):
        raise ModelAPIError("model JSON exceeded its wall deadline") from None
    except ModelAPIError:
        raise
    except Exception:
        raise ModelAPIError("model JSON request failed") from None


def image_metadata(encoded: bytes) -> dict[str, Any]:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image cannot be decoded")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "dimensions": [image.shape[1], image.shape[0]],
        "bytes": len(encoded),
    }


def validate_response(response: Mapping[str, Any], scope: str) -> dict[str, Any]:
    """Validate raw JSON strictly, independently of the forgiving domain parser."""
    result: dict[str, Any] = {
        "schema_valid": False,
        "parse_valid": False,
        "valid": False,
        "protocol_valid": False,
    }
    try:
        content = response["choices"][0]["message"]["content"]
        raw = json.loads(content)
        schema = vlm.build_compact_rich_annotation_response_format(scope)[
            "json_schema"
        ]["schema"]
        errors = list(Draft202012Validator(schema).iter_errors(raw))
        result["schema_valid"] = not errors
        result["schema_error_count"] = len(errors)
        # Names of validator rules are fixed; error messages can quote remote text.
        result["schema_error_rules"] = sorted(
            {str(error.validator) for error in errors}
        )
    except (KeyError, IndexError, TypeError, ValueError):
        result["json_error"] = True
        return result
    try:
        objects = vlm.parse_rich_annotation(content)
        result.update(parse_valid=True, objects=[item.as_dict() for item in objects])
        result["people"] = sum(item.label == "person" for item in objects)
        result["items"] = len(objects) - result["people"]
        if scope == "people" and result["items"]:
            result["scope_valid"] = False
        elif scope == "items" and result["people"]:
            result["scope_valid"] = False
        else:
            result["scope_valid"] = True
    except Exception as exc:
        result["parse_error_type"] = type(exc).__name__
    # The production contract also rejects duplicate keys, disguised person
    # labels and more than eight items, beyond checks in the current wire schema.
    try:
        validate_complete_compact_annotation(content, scope)
        result["complete_contract_valid"] = True
    except Exception as exc:
        result["complete_contract_valid"] = False
        result["complete_contract_error_type"] = type(exc).__name__
    result["valid"] = (
        all(
            result.get(key, False)
            for key in (
                "schema_valid",
                "parse_valid",
                "scope_valid",
                "complete_contract_valid",
            )
        )
        and response["choices"][0].get("finish_reason") == "stop"
    )
    result["protocol_valid"] = result["valid"]
    return result


@contextmanager
def capture_calls(
    case: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    trace_transport: bool = False,
):
    """Process-local diagnostic override; always restore the native callable."""
    original = vlm.complete_stream
    names = (
        ["people_left", "people_right", "items"]
        if case["execution_mode"] == "parallel"
        else ["all"]
    )

    async def complete(client: httpx.AsyncClient, **kwargs: Any) -> dict[str, Any]:
        name = names[len(records)]
        scope = "people" if name.startswith("people") else name
        record: dict[str, Any] = {"name": name, "scope": scope}
        records.append(record)
        encoded = None
        for message in kwargs["messages"]:
            for block in (
                message["content"] if isinstance(message["content"], list) else []
            ):
                if block.get("type") == "image_url":
                    encoded = base64.b64decode(
                        block["image_url"]["url"].split(",", 1)[1]
                    )
                    record["image"] = image_metadata(encoded)
        named_wire = case.get("wire_format", "compact") == "named"
        if named_wire:
            from goodvision_harness.vlm_object_wire import (
                build_object_wire_messages,
                build_object_wire_response_format,
                object_wire_to_compact,
            )

            if encoded is None:
                raise ValueError("named wire requires an image")
            kwargs["messages"] = build_object_wire_messages(encoded, scope)
            if case.get("parallel_strategy") == "full_frame" and scope == "people":
                low, high = (0, 500) if name == "people_left" else (500, 1000)
                region_instruction = (
                    f" Include a person only if {low} <= (x1+x2)/2 < {high}. "
                    "Assign by the center x of their visible box: left boundary "
                    "included, right boundary excluded. Keep FULL-image coordinates "
                    "and the whole visible person box, even across the boundary."
                )
                kwargs["messages"][1]["content"][0]["text"] += region_instruction
            if case.get("prompt_cache", False):
                kwargs["messages"] = vlm._cacheable_annotation_messages(
                    kwargs["messages"]
                )
            if case.get("structured_output", False):
                kwargs["request_parameters"] = {
                    **kwargs.get("request_parameters", {}),
                    "response_format": build_object_wire_response_format(scope),
                }
        suffix = case.get("prompt_suffix", "")
        record["prompt_example_removed"] = False
        if suffix or case.get("omit_example", False):
            # Copy only changed containers; do not mutate native/cacheable prompts.
            messages = list(kwargs["messages"])
            for index, message in enumerate(messages):
                if message.get("role") != "user" or not isinstance(
                    message.get("content"), list
                ):
                    continue
                blocks = list(message["content"])
                text_index = next(
                    (
                        i
                        for i, block in enumerate(blocks)
                        if block.get("type") == "text"
                    ),
                    None,
                )
                if text_index is not None:
                    prompt = blocks[text_index]["text"]
                    if case.get("omit_example", False):
                        start = prompt.find("Complete valid example (")
                        end = prompt.find("Each row=", start)
                        if start >= 0 and end > start:
                            prompt = (
                                prompt[:start]
                                + "Top-level JSON has only key o, an array of rows. "
                                + prompt[end:]
                            )
                            record["prompt_example_removed"] = True
                    blocks[text_index] = {
                        **blocks[text_index],
                        "text": prompt + (" " + suffix if suffix else ""),
                    }
                    messages[index] = {**message, "content": blocks}
                    break
            kwargs["messages"] = messages
        started = time.perf_counter()
        try:
            call = original if case["stream"] else complete_nonstream
            response = await call(client, **kwargs, trace_transport=trace_transport)
            if named_wire:
                # Retain the original reply even when strict conversion rejects it.
                record["wire_response"] = response
                record["wire_conversion_valid"] = False
                conversion_started = time.perf_counter()
                try:
                    compact = object_wire_to_compact(
                        response["choices"][0]["message"]["content"],
                        scope,
                    )
                finally:
                    record["wire_conversion_seconds"] = (
                        time.perf_counter() - conversion_started
                    )
                first_choice = response["choices"][0]
                response = {
                    **response,
                    "choices": [
                        {
                            **first_choice,
                            "message": {
                                **first_choice["message"],
                                "content": compact,
                            },
                        },
                        *response["choices"][1:],
                    ],
                }
                record["wire_conversion_valid"] = True
            record["response"] = response
            return response
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            if isinstance(exc, ModelAPIError):
                status = re.fullmatch(
                    r"model (?:stream|JSON) returned HTTP ([1-5][0-9]{2})", str(exc)
                )
                if status:
                    record["http_status"] = int(status[1])
                    record["error_code"] = "http_" + status[1]
            raise
        finally:
            record["wall_seconds"] = time.perf_counter() - started

    vlm.complete_stream = complete
    try:
        yield
    finally:
        vlm.complete_stream = original


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = sorted(row["wall_seconds"] for row in rows)
    valid = sum(bool(row["valid"]) for row in rows)
    under = sum(bool(row["valid"]) and row["wall_seconds"] < 3 for row in rows)
    return {
        "requests": len(rows),
        "valid": valid,
        "protocol_valid": valid,
        "visual_quality_evaluated": False,
        "validity_definition": "protocol_and_schema_only",
        "failed": len(rows) - valid,
        "valid_under_3_seconds": under,
        "valid_under_3_rate": under / len(rows) if rows else None,
        "protocol_valid_under_3_rate": under / len(rows) if rows else None,
        "median_wall_seconds": statistics.median(times) if times else None,
        "p95_wall_seconds": times[math.ceil(len(times) * 0.95) - 1] if times else None,
        "max_wall_seconds": max(times) if times else None,
        "failure_types": dict(
            Counter(
                row.get("error_type", "protocol_validation")
                for row in rows
                if not row["valid"]
            )
        ),
        "api_requests": sum(len(row["parts"]) for row in rows),
        "usage": aggregate_usage([part for row in rows for part in row["parts"]]),
        "reported_cost": sum(row["usage"].get("cost", 0) for row in rows),
        "cost_reporting_complete": bool(rows)
        and all(
            bool(row["parts"])
            and all(
                "cost"
                in part.get("wire_response", part.get("response", {})).get("usage", {})
                for part in row["parts"]
            )
            for row in rows
        ),
    }


def aggregate_usage(parts: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for part in parts:
        response = part.get("wire_response", part.get("response", {}))
        for key, value in _safe_usage(response.get("usage")).items():
            if isinstance(value, dict):
                details = total.setdefault(key, {})
                for name, count in value.items():
                    details[name] = details.get(name, 0) + count
            else:
                total[key] = total.get(key, 0) + value
    return total


def run_trial(
    client: MatrixClient,
    case: Mapping[str, Any],
    encoded: bytes,
    trial: int,
    *,
    trace_transport: bool = False,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}
    client._response_observer = lambda response: observed.update(response)
    client._part_response_observer = lambda name, response: observed.setdefault(
        "parts", {}
    ).update({name: response})
    row: dict[str, Any] = {
        "type": "trial",
        "case_id": case["id"],
        "trial": trial,
        "cold_client": trial == 0,
        "image": image_metadata(encoded),
        "parts": parts,
        "valid": False,
    }
    started = time.perf_counter()
    try:
        with capture_calls(case, parts, trace_transport=trace_transport):
            snapshot = client.infer(
                encoded,
                stream_session_id="speed-matrix",
                frame_sequence=trial,
                timestamp_seconds=trial * 15.0,
            )
        row["objects"] = [item.as_dict() for item in snapshot.objects]
        row["people"] = sum(item.label == "person" for item in snapshot.objects)
        row["items"] = len(snapshot.objects) - row["people"]
        row["native_success"] = True
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["native_success"] = False
    row["wall_seconds"] = time.perf_counter() - started
    # Schema QA is deliberately outside the native inference wall measurement.
    for part in parts:
        if "wire_response" in part:
            from goodvision_harness.vlm_object_wire import (
                build_object_wire_response_format,
            )

            wire_validation: dict[str, Any] = {"schema_valid": False}
            try:
                wire_content = part["wire_response"]["choices"][0]["message"]["content"]
                wire_schema = build_object_wire_response_format(part["scope"])[
                    "json_schema"
                ]["schema"]
                errors = list(
                    Draft202012Validator(wire_schema).iter_errors(
                        json.loads(wire_content)
                    )
                )
                wire_validation.update(
                    schema_valid=not errors,
                    schema_error_count=len(errors),
                    schema_error_rules=sorted(
                        {str(error.validator) for error in errors}
                    ),
                )
            except (KeyError, IndexError, TypeError, ValueError):
                wire_validation["json_error"] = True
            part["wire_validation"] = wire_validation
        if "response" in part:
            part["validation"] = validate_response(part["response"], part["scope"])
    expected_parts = 3 if case["execution_mode"] == "parallel" else 1
    row["valid"] = (
        row["native_success"]
        and len(parts) == expected_parts
        and all(part.get("validation", {}).get("valid", False) for part in parts)
        and all(
            part.get("wire_validation", {}).get("schema_valid", True) for part in parts
        )
    )
    row["protocol_valid"] = row["valid"]
    row["visual_quality_evaluated"] = False
    row["usage"] = aggregate_usage(parts)
    row["pool_creation_seconds"] = client.pool_creation_seconds if trial == 0 else 0.0
    if observed.get("_timing"):
        row["native_timing"] = observed["_timing"]
    return row


async def preflight_key(key: str, *, trust_env: bool = True) -> None:
    """Unpaid authentication check; any failure stops before completions."""
    try:
        async with httpx.AsyncClient(
            trust_env=trust_env, follow_redirects=False
        ) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
        if response.status_code != 200:
            raise ModelAPIError(f"key preflight returned HTTP {response.status_code}")
    except ModelAPIError:
        raise
    except Exception:
        raise ModelAPIError("key preflight request failed") from None


def run(args: argparse.Namespace) -> None:
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("repeats and timeout must be positive")
    cases = load_cases(args.cases)
    key = load_key(args.env_file, args.key_env)
    images = []
    for path in args.image:
        image = cv2.imdecode(
            np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("input image cannot be decoded")
        images.append(image)
    if args.preflight:
        asyncio.run(preflight_key(key, trust_env=cases[0]["trust_env"]))
    settings = ModelAPISettings(
        provider="openrouter",
        timeout_seconds=args.timeout,
        max_retries=0,
        extra_headers={"X-OpenRouter-Cache": "false"},
    )
    run_id = uuid.uuid4().hex
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clients = []
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            started = time.perf_counter()
            client = MatrixClient(
                settings,
                model=case["model"],
                api_key=key,
                wire_format="compact",
                require_complete=case["require_complete"],
                max_tokens=case["max_tokens"],
                max_image_side=case["side"],
                request_parameters=case["request_parameters"],
                execution_mode=case["execution_mode"],
                parallel_strategy=case["parallel_strategy"],
                structured_output=case["structured_output"],
                prompt_cache=case["prompt_cache"],
                connection_options=case,
            )
            clients.append((case, client, time.perf_counter() - started))
        with args.output.open("a", encoding="utf-8") as output:
            for trial in range(args.repeats):
                image_index = (trial // 2) % len(images)
                frame = (
                    cv2.flip(images[image_index], 1)
                    if trial % 2
                    else images[image_index]
                )
                for case, client, creation_seconds in clients:
                    started = time.perf_counter()
                    encoded, _, _ = encode_jpeg(
                        frame, max_image_side=max(frame.shape[:2]), quality=95
                    )
                    preparation_seconds = time.perf_counter() - started
                    row = run_trial(
                        client,
                        case,
                        encoded,
                        trial,
                        trace_transport=args.trace_transport,
                    )
                    row.update(
                        run_id=run_id,
                        case=case,
                        image_index=image_index,
                        mirrored=bool(trial % 2),
                        input_prepare_seconds=preparation_seconds,
                        client_creation_seconds=creation_seconds if trial == 0 else 0.0,
                    )
                    rows.append(row)
                    output.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    output.flush()
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "case_id",
                                    "trial",
                                    "valid",
                                    "protocol_valid",
                                    "visual_quality_evaluated",
                                    "wall_seconds",
                                    "usage",
                                )
                            }
                        ),
                        flush=True,
                    )
            for case in cases:
                summary = {
                    "type": "summary",
                    "run_id": run_id,
                    "case_id": case["id"],
                    **summarize([row for row in rows if row["case_id"] == case["id"]]),
                }
                output.write(json.dumps(summary) + "\n")
                print(json.dumps(summary), flush=True)
    finally:
        for _, client, _ in clients:
            client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--trace-transport", action="store_true")
    parser.add_argument(
        "--preflight", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        # Input errors may contain arbitrary caller or remote text; keep stdout clean.
        print(
            json.dumps({"type": "fatal", "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
