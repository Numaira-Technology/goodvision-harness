"""Operator route checks. Vendor identifiers never enter the console contract."""

from __future__ import annotations

import copy
import math
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from .grounding import (
    MIN_MULTI_EVENT_MAX_TOKENS,
    SAFETY_EVENT_LABELS,
    build_multi_grounding_messages,
    parse_multi_grounded_classification,
)
from .inference import InferenceSettings
from .model_api import (
    OPENROUTER_EXCLUDED_PROVIDERS,
    ModelAPIClient,
    ModelAPIError,
    nitro_provider_preferences,
)
from .sliding_window import FrameWindow, SampledFrame

PREFERRED_THRESHOLD_SECONDS = 2.5
PROBE_TRIALS = 3
MAX_PROBE_CONCURRENCY = 4
EXCLUDED_PROVIDERS = OPENROUTER_EXCLUDED_PROVIDERS


@dataclass(frozen=True)
class ModelRoute:
    id: str
    model: str
    provider: str
    preferred: bool = False
    reasoning: str | None = None

    @property
    def request_model(self) -> str:
        # Keep the historical identity intact for persisted selections.
        return f"{self.model.removesuffix(':nitro')}:nitro"

    def parameters(self, original: dict[str, Any]) -> dict[str, Any]:
        if self.provider.split("/", 1)[0].lower() in EXCLUDED_PROVIDERS:
            raise ValueError("route is unavailable")
        parameters = copy.deepcopy(original)
        for key in (
            "model", "models", "route", "endpoint", "messages", "api_key_env",
            "enable_thinking", "reasoning", "provider", "stream",
        ):
            parameters.pop(key, None)
        parameters["provider"] = nitro_provider_preferences()
        if self.reasoning is not None:
            parameters["reasoning"] = {"effort": self.reasoning}
        return parameters

    def inference(self, settings: InferenceSettings) -> InferenceSettings:
        return replace(
            settings, model=self.request_model,
            request_parameters=self.parameters(dict(settings.request_parameters)),
        )


# Stable IDs are persisted. Never reassign an ID to a different combination.
# Region screening and endpoint provenance: docs/internal/route-catalog.md.
# Hong Kong network acceptance remains pending until tested from that egress.
ROUTE_CATALOG = (
    ModelRoute("A", "qwen/qwen3.5-35b-a3b", "deepinfra", True, "none"),
    ModelRoute("B", "qwen/qwen3.5-35b-a3b", "parasail/fp8", True, "none"),
    ModelRoute("C", "qwen/qwen3-vl-8b-instruct", "alibaba"),
    ModelRoute("D", "qwen/qwen3.5-35b-a3b", "alibaba", True, "none"),
    ModelRoute("E", "qwen/qwen3.5-flash-02-23", "alibaba", False, "none"),
    ModelRoute("F", "qwen/qwen3.6-35b-a3b", "deepinfra/fp8", True, "none"),
    ModelRoute("G", "qwen/qwen3.6-35b-a3b", "parasail/fp8", True, "none"),
    ModelRoute("H", "qwen/qwen3.6-35b-a3b", "coreweave/fp8", True, "none"),
    ModelRoute("I", "qwen/qwen3.5-9b", "parasail/bf16", False, "none"),
    ModelRoute("J", "qwen/qwen3.5-27b", "deepinfra/fp8", False, "none"),
    ModelRoute("K", "qwen/qwen3.5-27b", "alibaba", False, "none"),
    ModelRoute("L", "qwen/qwen3.5-27b", "novita/bf16", False, "none"),
    ModelRoute("M", "qwen/qwen3.6-27b", "deepinfra/fp8", False, "none"),
    ModelRoute("N", "qwen/qwen3.6-27b", "alibaba", False, "none"),
    ModelRoute("O", "qwen/qwen3.6-27b", "chutes/fp8", False, "none"),
    ModelRoute("P", "qwen/qwen3.8-27b", "alibaba", False, "none"),
    ModelRoute("Q", "qwen/qwen3.8-27b", "parasail/fp8", False, "none"),
    ModelRoute("R", "qwen/qwen3.8-27b", "coreweave/fp8", False, "none"),
    ModelRoute("S", "qwen/qwen3.8-27b", "novita", False, "none"),
    ModelRoute("T", "qwen/qwen3.8-27b", "chutes/fp8", False, "none"),
    ModelRoute("U", "qwen/qwen3.8-27b", "reka/fp8", False, "none"),
    ModelRoute("V", "qwen/qwen3.8-flash", "alibaba", False, "none"),
    ModelRoute("W", "qwen/qwen3.7-flash", "alibaba", False, "none"),
    ModelRoute("X", "qwen/qwen3.6-flash", "alibaba", False, "none"),
    ModelRoute("Y", "qwen/qwen3-vl-8b-instruct", "parasail/bf16"),
    ModelRoute("Z", "qwen/qwen3-vl-30b-a3b-instruct", "deepinfra/fp8"),
    ModelRoute("AA", "qwen/qwen3-vl-30b-a3b-instruct", "alibaba"),
    ModelRoute("AB", "qwen/qwen3-vl-30b-a3b-instruct", "novita/bf16"),
    ModelRoute("AC", "qwen/qwen3-vl-32b-instruct", "alibaba"),
    ModelRoute("AD", "qwen/qwen2.5-vl-72b-instruct", "parasail/fp8"),
    ModelRoute("AE", "qwen/qwen3.5-122b-a10b", "deepinfra/fp4", False, "none"),
    ModelRoute("AF", "qwen/qwen3.5-122b-a10b", "alibaba", False, "none"),
    ModelRoute("AG", "qwen/qwen3.5-122b-a10b", "novita/bf16", False, "none"),
    ModelRoute("AH", "qwen/qwen3.5-397b-a17b", "deepinfra/fp8", False, "none"),
    ModelRoute("AI", "qwen/qwen3.5-397b-a17b", "alibaba", False, "none"),
    ModelRoute("AJ", "qwen/qwen3.5-397b-a17b", "parasail/fp8", False, "none"),
    ModelRoute("AK", "qwen/qwen3.5-397b-a17b", "novita", False, "none"),
    ModelRoute("AL", "qwen/qwen3-vl-235b-a22b-instruct", "alibaba"),
    ModelRoute("AM", "qwen/qwen3-vl-235b-a22b-instruct", "parasail/fp8"),
    ModelRoute("AN", "qwen/qwen3.5-plus-02-15", "alibaba", False, "none"),
    ModelRoute("AO", "qwen/qwen3.5-plus-20260420", "alibaba", False, "none"),
    ModelRoute("AP", "qwen/qwen3.6-plus", "alibaba", False, "none"),
    ModelRoute("AQ", "qwen/qwen3.7-plus", "alibaba", False, "none"),
    ModelRoute("AR", "qwen/qwen3.8-max-0902", "alibaba", False, "none"),
)
ROUTES_BY_ID = {
    route.id: route for route in ROUTE_CATALOG
    if route.provider.split("/", 1)[0].lower() not in EXCLUDED_PROVIDERS
}
# Keep the fastest eight stable routes from the 2026-09-08 sweep and recheck.
# Allowed historical IDs remain restorable; removed providers cannot be restored.
# Only these eight routes are probed/shown.
# Nitro removes provider pinning; duplicate base models retain their old IDs
# and public positions so saved selections and A-H numbering stay compatible.
ROUTES = tuple(ROUTES_BY_ID[key] for key in (
    "C", "D", "AA", "AE", "G", "K", "B", "F",
))
# Public numbering follows the current shortlist without reassigning persisted IDs.
PUBLIC_ROUTE_IDS = {
    route.id: chr(ord("A") + index) for index, route in enumerate(ROUTES)
}


def empty_results() -> list[dict[str, Any]]:
    return [
        {
            "id": route.id, "label": f"线路 {route.id}", "state": "pending",
            "samples": [], "median_seconds": None, "eligible": False,
        }
        for route in ROUTES
    ]


def choose_route(results: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Prefer the fastest 35B route strictly below 2.5s, else fastest valid route."""
    eligible = [
        row for row in results
        if row["id"] in ROUTES_BY_ID and row.get("eligible")
        and isinstance(row.get("median_seconds"), (int, float))
        and math.isfinite(row["median_seconds"]) and row["median_seconds"] > 0
    ]
    if not eligible:
        return None, "no_available_route"
    preferred = [
        row for row in eligible
        if ROUTES_BY_ID[row["id"]].preferred
        and row["median_seconds"] < PREFERRED_THRESHOLD_SECONDS
    ]
    fastest = min(
        preferred or eligible, key=lambda row: (row["median_seconds"], row["id"]),
    )
    return fastest["id"], (
        "preferred_below_threshold" if preferred else "fastest_available"
    )


def probe_window(index: int = 0) -> FrameWindow:
    """Anonymous synthetic input, never a camera frame or a captured face."""
    image = np.full((180, 320, 3), (232, 226, 214), dtype=np.uint8)
    cv2.rectangle(image, (0, 130), (319, 179), (105, 116, 92), -1)
    for x, y in ((105, 55), (205, 62)):
        cv2.circle(image, (x, y), 15, (70, 105, 165), -1)
        cv2.line(image, (x, y + 15), (x, 128), (45, 65, 95), 10)
        cv2.line(image, (x, y + 30), (x - 35, 103), (45, 65, 95), 8)
        cv2.line(image, (x, y + 30), (x + 34, 96), (45, 65, 95), 8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise RuntimeError("could not prepare route test image")
    frame = SampledFrame(index, index, float(index), encoded.tobytes(), 320, 180)
    return FrameWindow(index, (frame,))


def probe_sample(
    transport: ModelAPIClient, settings: InferenceSettings,
    route: ModelRoute, index: int,
) -> dict[str, Any]:
    fixed = bool(settings.multi_event_prompt.strip())
    messages = build_multi_grounding_messages(
        probe_window(index), system_prompt=settings.system_prompt,
        prompt=settings.multi_event_prompt if fixed else settings.prompt,
        event_labels=SAFETY_EVENT_LABELS, append_contract=not fixed,
    )
    parameters = route.parameters(dict(settings.request_parameters))
    parameters["max_tokens"] = max(
        int(parameters.get("max_tokens") or MIN_MULTI_EVENT_MAX_TOKENS),
        MIN_MULTI_EVENT_MAX_TOKENS,
    )
    started = time.monotonic()
    error = None
    try:
        response = transport.chat_completion(
            model=route.request_model, messages=messages, **parameters,
        )
        parse_multi_grounded_classification(
            transport.assistant_text(response), event_labels=SAFETY_EVENT_LABELS,
        )
    except ModelAPIError:
        error = "request_failed"
    except (TypeError, ValueError):
        error = "invalid_result"
    return {"seconds": time.monotonic() - started, "ok": error is None,
            "error": error}


def finish_result(row: dict[str, Any]) -> None:
    row["median_seconds"] = None
    row["eligible"] = len(row["samples"]) == PROBE_TRIALS and all(
        sample["ok"] for sample in row["samples"]
    )
    row["state"] = "complete" if row["eligible"] else "unavailable"
    if row["eligible"]:
        row["median_seconds"] = statistics.median(
            sample["seconds"] for sample in row["samples"]
        )
