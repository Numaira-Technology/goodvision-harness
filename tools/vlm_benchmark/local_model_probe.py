"""Four bounded local image requests: vision transport and production JSON parsers."""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

from goodvision_harness.grounding import SAFETY_EVENT_LABELS
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
    event_response_format,
    raw_reply,
)
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.telemetry import parse_metrics


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or not parsed.port
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("health probe only accepts http://127.0.0.1:PORT")


def jpeg(color):
    with Image.new("RGB", (720, 720), color) as picture, io.BytesIO() as buffer:
        picture.save(buffer, "JPEG", quality=90)
        return buffer.getvalue()


def image_part(data):
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode()
        },
    }


def vision_payload(color):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    *[image_part(jpeg(c)) for c in ("white", "black", "green", color)],
                    {
                        "type": "text",
                        "text": "Inspect the LAST image. Return only a JSON object "
                        "with key color and its dominant color: "
                        "red, blue, green or white.",
                    },
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "visual_health",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "enum": ["red", "blue", "green", "white"],
                        }
                    },
                    "required": ["color"],
                    "additionalProperties": False,
                },
            },
        },
        "max_tokens": 32,
    }


def production_probe(model, endpoint, mode):
    frames = tuple(EvidenceFrame("health", i, i * 0.4, jpeg("gray")) for i in range(4))
    job = TemporalInspection(
        "health",
        "health-camera",
        InspectionRequest(
            "health-window",
            (),
            "independent_source",
            SAFETY_EVENT_LABELS,
            frames[0].timestamp_seconds,
            frames[-1].timestamp_seconds,
        ),
        frames,
        grounding_required=(mode == "compact"),
    )
    cls = EventJSONInspector if mode == "events" else CompactGroundingInspector
    inspector = cls(
        ModelAPISettings(
            provider="local", endpoint=endpoint + "/v1/chat/completions", max_retries=0
        ),
        model=model,
        api_key="",
    )
    try:
        messages = inspector.build_messages(job)
    finally:
        inspector.close()
    return (
        job,
        cls,
        {
            "messages": messages,
            "response_format": event_response_format(SAFETY_EVENT_LABELS)
            if mode == "events"
            else {"type": "json_object"},
            "max_tokens": 160 if mode == "events" else 1600,
        },
    )


def probe(endpoint, model, output):
    validate_endpoint(endpoint)
    document = {
        "state": "checking",
        "model": model,
        "checks": [],
        "scope": "local vision/schema smoke test; not event accuracy/capacity",
    }
    write_json(output, document)
    current = "model_identity"
    try:
        with httpx.Client(
            base_url=endpoint, timeout=60, trust_env=False, follow_redirects=False
        ) as client:
            response = client.get("/v1/models")
            response.raise_for_status()
            if model not in {r.get("id") for r in response.json().get("data", [])}:
                raise RuntimeError("served model does not match deployment")

            def send(name, payload, check):
                nonlocal current
                current = name
                began = time.monotonic()
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "temperature": 0,
                        "stream": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                        **payload,
                    },
                )
                response.raise_for_status()
                raw = response.json()
                check(raw)
                document["checks"].append(
                    {
                        "name": name,
                        "passed": True,
                        "seconds": time.monotonic() - began,
                        "usage": {
                            k: raw.get("usage", {}).get(k)
                            for k in ("prompt_tokens", "completion_tokens")
                        },
                    }
                )
                write_json(output, document)

            for color in ("red", "blue"):

                def check_color(raw, expected=color):
                    if raw_reply(raw) != {"color": expected}:
                        raise RuntimeError("four-image vision sanity check failed")

                send("vision_" + color, vision_payload(color), check_color)
            for mode in ("events", "compact"):
                current = mode + "_production_schema"
                job, cls, payload = production_probe(model, endpoint, mode)

                def check_production(raw, job=job, cls=cls):
                    # A syntactically valid but partial result must not pass health.
                    verdicts = cls._parse(job, raw)
                    if {v.event_label for v in verdicts} != set(SAFETY_EVENT_LABELS):
                        raise RuntimeError("health result omitted event labels")
                    issues = {issue for v in verdicts for issue in v.grounding_issues}
                    if issues - {"scene_only", "unlocalized_event"}:
                        raise RuntimeError("invalid grounding output structure")

                send(mode + "_production_schema", payload, check_production)
            try:
                metrics = client.get("/metrics", timeout=2)
                document["metrics_exposed"] = metrics.status_code == 200 and bool(
                    parse_metrics(metrics.text)
                )
            except httpx.HTTPError:
                document["metrics_exposed"] = False
        document["state"] = "ready"
    except BaseException as exc:
        document.update(
            state="failed", failed_check=current, error_type=type(exc).__name__
        )
        if isinstance(exc, httpx.HTTPStatusError):
            document["http_status"] = exc.response.status_code
        # No provider body, prompt, response text or credentials in failure messages.
        raise
    finally:
        write_json(output, document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        probe(args.endpoint, args.model, args.output)
    except Exception as exc:
        print(f"Model health failed: {type(exc).__name__}; see {args.output}")
        return 2
    print(f"Image and production-schema checks passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
