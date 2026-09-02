"""Run a bounded VLM rich-label smoke check on local images or video samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from uuid import uuid4

import cv2
import numpy as np

# Use this checkout even when another worktree is installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goodvision_harness.aiclub import (  # noqa: E402
    DEFAULT_AICLUB_ENDPOINT,
    AIClubClient,
    AIClubError,
    api_key_present,
)
from goodvision_harness.config import HarnessConfig, load_config  # noqa: E402
from goodvision_harness.overlay import draw_rich_overlay  # noqa: E402
from goodvision_harness.rich_labeling import RichLabelProtocolError  # noqa: E402
from goodvision_harness.video_to_stream import (  # noqa: E402
    crop_frame,
    encode_jpeg,
    probe_video,
)
from goodvision_harness.vlm_rich_labeling import VLMRichLabelClient  # noqa: E402


class EndpointMismatchError(ValueError):
    """The loaded endpoint differs from the explicitly expected destination."""


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def _positive_float(value: str) -> float:
    number = _nonnegative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="an already-cropped image")
    source.add_argument(
        "--video", type=Path, help="video cropped per stream.frame_crop"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config.vlm-rich.yaml"
    )
    parser.add_argument(
        "--expected-endpoint", default=DEFAULT_AICLUB_ENDPOINT,
        help="require this exact API destination before reading media or making calls",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/vlm-rich-smoke"),
        help="an empty output directory (default: output/vlm-rich-smoke)",
    )
    parser.add_argument("--samples", type=_positive_integer, default=3)
    parser.add_argument("--interval-seconds", type=_positive_float, default=2.0)
    parser.add_argument("--start-seconds", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="exercise decoding and output files without making API requests",
    )
    parser.add_argument(
        "--save-model-response", action="store_true",
        help="save assistant text, finish reason, and token usage for parser diagnosis",
    )
    return parser


def _sample_frames(
    arguments: argparse.Namespace, config: HarnessConfig
) -> Iterator[tuple[int, float, Any]]:
    if arguments.image is not None:
        # imdecode supports Unicode Windows paths through Python's file reader.
        encoded = np.frombuffer(arguments.image.read_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("the image could not be decoded")
        yield 0, 0.0, frame
        return

    info = probe_video(arguments.video)
    if not math.isfinite(info.source_fps) or info.source_fps <= 0:
        raise ValueError("the video must report a finite positive frame rate")
    capture = cv2.VideoCapture(str(arguments.video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("the video could not be opened")
    try:
        previous_sequence = -1
        for index in range(arguments.samples):
            target_seconds = (
                arguments.start_seconds + index * arguments.interval_seconds
            )
            sequence = round(target_seconds * info.source_fps)
            if sequence >= info.frame_count:
                break
            if sequence == previous_sequence:
                continue
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, sequence):
                raise ValueError("the video does not support frame seeking")
            ok, frame = capture.read()
            if not ok:
                raise ValueError("a requested video frame could not be decoded")
            previous_sequence = sequence
            yield (
                sequence,
                sequence / info.source_fps,
                crop_frame(frame, mode=config.stream.frame_crop),
            )
    finally:
        capture.release()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _provider_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or len(name) > 128 or not name.isprintable() or "://" in name:
        return None
    return name


def _request_route(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Select upstream routing preferences without copying arbitrary parameters."""
    provider = parameters.get("provider")
    route: dict[str, Any] = {}
    if isinstance(provider, Mapping):
        for key in ("only", "order"):
            values = provider.get(key)
            if isinstance(values, (list, tuple)):
                route[key] = [
                    name for value in values
                    if (name := _provider_name(value)) is not None
                ]
        for key in ("allow_fallbacks", "require_parameters"):
            if isinstance(provider.get(key), bool):
                route[key] = provider[key]
    return route


def _response_diagnostic(
    response: Mapping[str, Any], *, include_parts: bool = True,
) -> dict[str, Any]:
    """Select response content without copying transport or request metadata."""
    try:
        assistant_text = AIClubClient.assistant_text(response)
    except (AIClubError, AttributeError, IndexError, KeyError, TypeError):
        assistant_text = None
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    raw_usage = response.get("usage")
    usage: dict[str, Any] = {}
    if isinstance(raw_usage, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = value
        for key in ("prompt_tokens_details", "completion_tokens_details"):
            details = raw_usage.get(key)
            if isinstance(details, Mapping):
                usage[key] = {
                    name: value for name, value in details.items()
                    if isinstance(name, str) and name.endswith("_tokens")
                    and isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0
                }
    diagnostic = {
        "assistant_text": assistant_text,
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "usage": usage,
    }
    provider = _provider_name(response.get("provider"))
    if provider is not None:
        diagnostic["provider"] = provider
    response_id = _provider_name(response.get("id"))
    if response_id is not None:
        diagnostic["id"] = response_id
    timing = response.get("_timing")
    if isinstance(timing, Mapping):
        diagnostic["timing"] = {
            name: value for name in (
                "response_headers_seconds", "first_content_token_seconds",
                "complete_seconds", "response_eof_seconds", "elapsed_seconds",
                "max_part_seconds",
            )
            if isinstance(value := timing.get(name), (int, float))
            and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        }
    parts = response.get("parts")
    if include_parts and isinstance(parts, Mapping):
        diagnostic["parts"] = {
            name: _response_diagnostic(part, include_parts=False)
            for name in ("people_left", "people_right", "items")
            if isinstance(part := parts.get(name), Mapping)
        }
    return diagnostic


def _failure_diagnostic(exc: Exception) -> dict[str, Any]:
    """Keep local validation reasons and safe transport categories only."""
    diagnostic: dict[str, Any] = {
        "status": "failure", "error_type": type(exc).__name__,
    }
    if isinstance(exc, RichLabelProtocolError):
        diagnostic["error_reason"] = str(exc)
    cause = exc.__cause__ or exc
    if isinstance(cause, HTTPError):
        if isinstance(cause.code, int) and not isinstance(cause.code, bool):
            diagnostic["http_status"] = cause.code
    elif isinstance(cause, TimeoutError):
        diagnostic["error_category"] = "timeout"
    elif isinstance(cause, URLError):
        diagnostic["error_category"] = "network_error"
        diagnostic["network_reason_type"] = type(cause.reason).__name__
    return diagnostic


def run_smoke(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    if config.aiclub.endpoint != arguments.expected_endpoint:
        raise EndpointMismatchError(
            "Configured API endpoint does not exactly match --expected-endpoint. "
            "Stopped before reading media or making API requests."
        )
    if config.rich_labeling.backend != "vlm":
        raise ValueError("the smoke config must select rich_labeling.backend: vlm")
    if not arguments.dry_run and not api_key_present(config.aiclub):
        raise ValueError("set the configured inference API-key environment variable")
    input_path = arguments.image or arguments.video
    if not input_path.is_file():
        raise ValueError("the input file does not exist")
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("choose an empty output directory to preserve previous runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Never retry a paid request in this smoke tool. --samples is an upper bound.
    settings = replace(
        config.aiclub,
        timeout_seconds=config.rich_labeling.timeout_seconds,
        max_retries=0,
    )
    requests: list[dict[str, Any]] = []

    def observe_response(response: Mapping[str, Any]) -> None:
        provider = _provider_name(response.get("provider"))
        if provider is not None and requests:
            requests[-1]["response_provider"] = provider
        if arguments.save_model_response:
            _write_json(
                output_dir / f"response-{len(requests):03d}.json",
                _response_diagnostic(response),
            )

    def observe_part_response(part_name: str, response: Mapping[str, Any]) -> None:
        if part_name not in ("people_left", "people_right", "items") or not requests:
            return
        filename = f"response-{len(requests):03d}-{part_name}.json"
        _write_json(
            output_dir / filename, _response_diagnostic(response, include_parts=False),
        )
        requests[-1].setdefault("part_response_files", {})[part_name] = filename

    client = VLMRichLabelClient(
        settings,
        model=config.rich_labeling.model or config.inference.model,
        max_tokens=config.rich_labeling.vlm_max_tokens,
        max_image_side=config.rich_labeling.vlm_max_image_side,
        wire_format=config.rich_labeling.vlm_wire_format,
        execution_mode=config.rich_labeling.vlm_execution_mode,
        structured_output=config.rich_labeling.vlm_structured_output,
        request_parameters=config.rich_labeling.vlm_request_parameters,
        dry_run=arguments.dry_run,
        response_observer=observe_response,
        part_response_observer=(
            observe_part_response if arguments.save_model_response else None
        ),
    )
    frames = _sample_frames(arguments, config)
    snapshots: list[dict[str, object]] = []
    labels: Counter[str] = Counter()
    session_id = f"vlm-rich-smoke-{uuid4().hex}"
    metrics: dict[str, Any] = {
        "type": "vlm_rich_smoke",
        "dry_run": bool(arguments.dry_run),
        "endpoint": settings.endpoint,
        "model": config.rich_labeling.model or config.inference.model,
        "selected_provider": settings.provider,
        "wire_format": config.rich_labeling.vlm_wire_format,
        "execution_mode": config.rich_labeling.vlm_execution_mode,
        "structured_output": config.rich_labeling.vlm_structured_output,
        "request_route": _request_route(config.rich_labeling.vlm_request_parameters),
        "status": "success",
        "requested_samples": 1 if arguments.image else arguments.samples,
        "frame_crop": (
            "none (image already cropped)"
            if arguments.image else config.stream.frame_crop
        ),
        "requests": requests,
    }
    started = time.perf_counter()
    try:
        for sequence, timestamp, frame in frames:
            jpeg, width, height = encode_jpeg(
                frame,
                max_image_side=config.rich_labeling.vlm_max_image_side,
                quality=config.video.jpeg_quality,
            )
            request: dict[str, Any] = {
                "frame_sequence": sequence,
                "timestamp_seconds": round(timestamp, 6),
                "image_width": width,
                "image_height": height,
            }
            requests.append(request)
            request_started = time.perf_counter()
            calls_before = client.api_requests
            try:
                snapshot = client.infer(
                    jpeg,
                    stream_session_id=session_id,
                    frame_sequence=sequence,
                    timestamp_seconds=timestamp,
                )
            except Exception as exc:
                # Do not persist raw provider exceptions, URLs, or credentials.
                failure = _failure_diagnostic(exc)
                request.update(failure)
                metrics.update(failure)
                break
            finally:
                request["api_requests"] = client.api_requests - calls_before
                request["latency_ms"] = round(
                    (time.perf_counter() - request_started) * 1000, 2
                )
            request.update(status="success", object_count=len(snapshot.objects))
            request["within_2_5_seconds"] = request["latency_ms"] <= 2500
            snapshots.append(snapshot.as_dict())
            labels.update(item.label for item in snapshot.objects)
            rendered = draw_rich_overlay(frame, snapshot.objects)
            annotated, _, _ = encode_jpeg(
                rendered, max_image_side=max(rendered.shape[:2]), quality=90
            )
            (output_dir / f"annotated-{len(snapshots):03d}.jpg").write_bytes(annotated)
        if not requests:
            metrics.update(status="failure", error_type="NoSampledFrames")
    except Exception as exc:
        metrics.update(_failure_diagnostic(exc))
    finally:
        frames.close()
        client.close()
        latencies = [request["latency_ms"] for request in requests]
        metrics.update(
            attempted_samples=len(requests),
            successful_samples=len(snapshots),
            api_requests=sum(request.get("api_requests", 0) for request in requests),
            elapsed_seconds=round(time.perf_counter() - started, 3),
            mean_latency_ms=(
                round(sum(latencies) / len(latencies), 2) if latencies else None
            ),
            labels=dict(sorted(labels.items())),
            response_providers=sorted({
                request["response_provider"] for request in requests
                if "response_provider" in request
            }),
        )
        _write_json(output_dir / "snapshots.json", snapshots)
        _write_json(output_dir / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        metrics = run_smoke(arguments)
    except EndpointMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Smoke setup failed ({type(exc).__name__}). Check the input file, config, "
            "API-key environment, and empty output directory.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Saved {metrics['successful_samples']} sample(s) "
        f"to {arguments.output_dir.resolve()} "
        f"({metrics['status']}; API requests: {metrics['api_requests']})."
    )
    if metrics["status"] != "success":
        print(
            f"Failure type: {metrics['error_type']}; see metrics.json.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
