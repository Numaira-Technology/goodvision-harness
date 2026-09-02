"""Bounded SSE timing probe; requests require an explicit execute opt-in."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, aclosing
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goodvision_harness.config import load_config  # noqa: E402
from goodvision_harness.model_api import (  # noqa: E402
    DEFAULT_OPENROUTER_ENDPOINT,
    ModelAPISettings,
    environment_api_key,
)
from goodvision_harness.video_to_stream import encode_jpeg  # noqa: E402
from goodvision_harness.vlm_rich_labeling import (  # noqa: E402
    build_compact_rich_annotation_messages,
    build_rich_annotation_messages,
)


class _StreamFailure(Exception):
    def __init__(self, category: str) -> None:
        self.category = category


def _safe_name(value: Any) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= 256 and value.isprintable():
        return value
    return None


def _safe_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        number = value.get(key)
        if (
            isinstance(number, (int, float)) and not isinstance(number, bool)
            and math.isfinite(number) and number >= 0
        ):
            result[key] = number
    for key in ("prompt_tokens_details", "completion_tokens_details"):
        details = value.get(key)
        if isinstance(details, Mapping):
            result[key] = {
                name: number for name, number in details.items()
                if isinstance(name, str) and name.endswith("_tokens")
                and isinstance(number, int) and not isinstance(number, bool)
                and number >= 0
            }
    return result


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    data: list[str] = []
    async with aclosing(response.aiter_lines()) as lines:
        async for line in lines:
            if not line:
                if data:
                    yield "\n".join(data)
                    data.clear()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def _http_client(deadline_seconds: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(deadline_seconds), follow_redirects=False,
    )


async def _collect(
    *, settings: ModelAPISettings, payload: dict[str, Any], key: str,
    deadline_seconds: float, metrics: dict[str, Any], response: dict[str, Any],
    started: float, http_client: httpx.AsyncClient | None,
) -> None:
    phase_starts: dict[str, float] = {}
    trace_seen = False

    def elapsed() -> float:
        return round(time.perf_counter() - started, 6)

    async def trace(name: str, _info: Mapping[str, Any]) -> None:
        nonlocal trace_seen
        trace_seen = True
        # Never retain trace info: it can contain sockets, URLs, or exceptions.
        for phase, metric in (
            ("connect_tcp", "tcp_connect_seconds"),
            ("start_tls", "tls_handshake_seconds"),
        ):
            if name.endswith(f".{phase}.started"):
                phase_starts[phase] = time.perf_counter()
            elif name.endswith(f".{phase}.complete") and phase in phase_starts:
                metrics[metric] = round(time.perf_counter() - phase_starts[phase], 6)
                metrics["connection_ready_seconds"] = elapsed()

    headers = dict(settings.extra_headers)
    headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    message = response["choices"][0]["message"]
    async with AsyncExitStack() as stack:
        if http_client is None:
            setup_started = time.perf_counter()
            client = await stack.enter_async_context(_http_client(deadline_seconds))
            metrics["client_setup_seconds"] = round(
                time.perf_counter() - setup_started, 6,
            )
        else:
            client = http_client
            metrics["client_setup_seconds"] = 0.0
        async with client.stream(
            "POST", settings.endpoint, json=payload, headers=headers,
            extensions={"trace": trace},
        ) as stream:
            metrics["response_headers_seconds"] = elapsed()
            metrics["connection_reused"] = (
                "connect_tcp" not in phase_starts if trace_seen else None
            )
            metrics["http_status"] = stream.status_code
            if stream.status_code != 200:
                raise _StreamFailure("http_error")
            media_type = stream.headers.get("content-type", "").split(";", 1)[0]
            if media_type.strip().lower() != "text/event-stream":
                raise _StreamFailure("unexpected_content_type")
            async with aclosing(_sse_data(stream)) as events:
                async for data in events:
                    if not data.strip():
                        continue  # Empty events and comment-only keepalives are harmless.
                    if metrics["done_received"]:
                        raise _StreamFailure("data_after_done")
                    if data.strip() == "[DONE]":
                        metrics["done_received"] = True
                        metrics["complete_seconds"] = elapsed()
                        continue  # Drain EOF so HTTPX can return the connection to its pool.
                    try:
                        chunk = json.loads(data)
                    except (TypeError, ValueError):
                        raise _StreamFailure("invalid_sse_json") from None
                    if not isinstance(chunk, Mapping):
                        raise _StreamFailure("invalid_sse_shape")
                    if "error" in chunk:
                        raise _StreamFailure("remote_stream_error")
                    for field in ("id", "provider"):
                        value = _safe_name(chunk.get(field))
                        if value is not None:
                            response[field] = value
                    if "usage" in chunk:
                        response["usage"].update(_safe_usage(chunk["usage"]))
                    choices = chunk.get("choices", [])
                    if not isinstance(choices, list):
                        raise _StreamFailure("invalid_sse_shape")
                    for choice in choices:
                        if not isinstance(choice, Mapping):
                            raise _StreamFailure("invalid_sse_shape")
                        if choice.get("index", 0) != 0:
                            continue
                        delta = choice.get("delta", {})
                        if not isinstance(delta, Mapping):
                            raise _StreamFailure("invalid_sse_shape")
                        content = delta.get("content")
                        if content is not None and not isinstance(content, str):
                            raise _StreamFailure("invalid_content_delta")
                        if content:
                            metrics.setdefault("first_content_token_seconds", elapsed())
                            message["content"] += content
                        finish = choice.get("finish_reason")
                        if finish is not None:
                            if not isinstance(finish, str):
                                raise _StreamFailure("invalid_finish_reason")
                            response["choices"][0]["finish_reason"] = finish
            metrics["response_eof_seconds"] = elapsed()
            if not metrics["done_received"]:
                raise _StreamFailure("missing_done")
            finish = response["choices"][0]["finish_reason"]
            if finish == "length":
                raise _StreamFailure("truncated")
            if finish != "stop":
                raise _StreamFailure("incomplete_finish_reason")
            if not message["content"]:
                raise _StreamFailure("empty_content")


async def probe_rich_latency_async(
    *, settings: ModelAPISettings, model: str,
    messages: Sequence[Mapping[str, Any]],
    request_parameters: Mapping[str, Any] | None = None,
    expected_endpoint: str = DEFAULT_OPENROUTER_ENDPOINT,
    deadline_seconds: float = 20.0, api_key: str | None = None,
    execute: bool = False, http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Measure one request; success means completed SSE, not validated rich JSON.

    A supplied http_client stays open and excludes its earlier setup from request
    timing. Otherwise client construction is included and separately measured.
    """
    if settings.endpoint != expected_endpoint:
        raise ValueError("endpoint does not exactly match expected_endpoint")
    if not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 120:
        raise ValueError("deadline_seconds must be greater than 0 and at most 120")
    if not isinstance(model, str) or not model.strip() or not messages:
        raise ValueError("model and messages are required")
    parameters = dict(request_parameters or {})
    if {"model", "messages", "stream", "stream_options"} & parameters.keys():
        raise ValueError("request_parameters must not override fixed stream fields")
    metrics: dict[str, Any] = {
        "status": "not_run", "selected_provider": settings.provider,
        "endpoint": settings.endpoint, "model": model,
        "deadline_seconds": deadline_seconds, "done_received": False,
        "api_requests": 0,
        "client_reused": http_client is not None,
        "client_setup_included": http_client is None,
        "connection_reused": None,
    }
    response: dict[str, Any] = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                     "finish_reason": None}],
        "usage": {},
    }
    result = {"metrics": metrics, "response": response, "assistant_text": ""}
    if not execute:
        return result
    key = api_key or environment_api_key(settings)
    if not key:
        metrics.update(status="failure", error_category="missing_api_key")
        return result
    payload = {
        **parameters, "model": model, "messages": list(messages), "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    metrics["api_requests"] = 1
    try:
        await asyncio.wait_for(
            _collect(
                settings=settings, payload=payload, key=key,
                deadline_seconds=deadline_seconds, metrics=metrics,
                response=response, started=started, http_client=http_client,
            ),
            timeout=deadline_seconds,
        )
        metrics["status"] = "success"
    except (asyncio.TimeoutError, TimeoutError):
        metrics.update(status="failure", error_category="wall_deadline")
    except _StreamFailure as exc:
        metrics.update(status="failure", error_category=exc.category)
    except httpx.TimeoutException:
        metrics.update(status="failure", error_category="socket_timeout")
    except httpx.HTTPError:
        metrics.update(status="failure", error_category="network_error")
    except Exception as exc:
        metrics.update(status="failure", error_category="local_error",
                       error_type=type(exc).__name__)
    finally:
        metrics["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        metrics["finish_reason"] = response["choices"][0]["finish_reason"]
        if "provider" in response:
            metrics["response_provider"] = response["provider"]
        if "id" in response:
            metrics["response_id"] = response["id"]
        result["assistant_text"] = response["choices"][0]["message"]["content"]
    return result


def probe_rich_latency(**kwargs: Any) -> dict[str, Any]:
    """Synchronous entry point; use probe_rich_latency_async inside an event loop."""
    return asyncio.run(probe_rich_latency_async(**kwargs))


async def _run_cli_samples(
    args: argparse.Namespace, config: Any, messages: Sequence[Mapping[str, Any]],
) -> int:
    async with AsyncExitStack() as stack:
        client = None
        initial_setup_seconds = 0.0
        if args.execute:
            setup_started = time.perf_counter()
            client = await stack.enter_async_context(_http_client(args.deadline_seconds))
            initial_setup_seconds = time.perf_counter() - setup_started
        for index in range(args.samples):
            result = await probe_rich_latency_async(
                settings=config.aiclub,
                model=config.rich_labeling.model or config.inference.model,
                messages=messages,
                request_parameters={
                    **config.rich_labeling.vlm_request_parameters,
                    "max_tokens": config.rich_labeling.vlm_max_tokens,
                },
                expected_endpoint=args.expected_endpoint,
                deadline_seconds=args.deadline_seconds, execute=args.execute,
                http_client=client,
            )
            metrics = result["metrics"]
            if args.execute:
                setup = initial_setup_seconds if index == 0 else 0.0
                metrics["initial_client_setup_seconds"] = round(setup, 6)
                metrics["elapsed_with_initial_client_setup_seconds"] = round(
                    metrics.get("elapsed_seconds", 0.0) + setup, 6,
                )
            (args.output_dir / f"sample-{index + 1:03d}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            print(json.dumps(metrics, ensure_ascii=False))
            if metrics["status"] == "failure":
                return 1
    return 0


def _build_cli_messages(image_path: Path, config: Any) -> list[dict[str, Any]]:
    image = cv2.imdecode(
        np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("input image could not be decoded")
    jpeg, _width, _height = encode_jpeg(
        image, max_image_side=config.rich_labeling.vlm_max_image_side, quality=85,
    )
    if config.rich_labeling.vlm_wire_format == "compact":
        return build_compact_rich_annotation_messages(jpeg)
    return build_rich_annotation_messages(jpeg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.vlm-rich.yaml")
    parser.add_argument("--expected-endpoint", default=DEFAULT_OPENROUTER_ENDPOINT)
    parser.add_argument("--deadline-seconds", type=float, default=20.0)
    parser.add_argument("--samples", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/vlm-rich-latency"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if config.aiclub.endpoint != args.expected_endpoint:
            raise ValueError("endpoint mismatch")
        if config.rich_labeling.vlm_execution_mode == "parallel":
            print(
                "This is a single-request latency diagnostic; parallel execution "
                "is not supported here. Use tools/vlm_rich_smoke.py to test the "
                "complete parallel rich annotation result.",
                file=sys.stderr,
            )
            return 1
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("output directory must be empty")
        messages = _build_cli_messages(args.image, config)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return asyncio.run(_run_cli_samples(args, config, messages))
    except Exception as exc:
        print(f"Latency probe setup failed ({type(exc).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
