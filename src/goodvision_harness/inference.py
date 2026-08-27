"""Qwen request/response handling for a single frame window.

This module is intentionally free of scheduling, pacing, and UI concerns: it
turns one ``FrameWindow`` into one chat-completion request and one parsed
``WindowResponse``. Streaming, window construction, and result ordering live
in ``video_to_stream``, ``pipeline``, and ``organizer`` respectively.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .config import InferenceSettings
from .openrouter import OpenRouterClient, OpenRouterError, OpenRouterSettings
from .prompt import (
    BinaryClassification,
    build_window_messages,
    parse_binary_classification,
)
from .reference_memory import MemoryContext
from .sliding_window import FrameWindow


@dataclass(frozen=True, slots=True)
class WindowResponse:
    """Outcome of classifying one frame window."""

    window: FrameWindow
    classification: BinaryClassification | None
    raw_text: str | None
    latency_seconds: float
    response_id: str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.classification is not None


class WindowClient(Protocol):
    """Anything that can turn a frame window into a ``WindowResponse``."""

    settings: InferenceSettings
    memory_context: MemoryContext | None

    def classify_window(self, window: FrameWindow) -> WindowResponse: ...


class QwenWindowClient:
    """Send one frame window to a Qwen VL chat endpoint and parse the reply.

    Transport and parse failures are folded into the returned
    ``WindowResponse`` instead of raised, so worker threads never leak
    exceptions into the scheduler.
    """

    def __init__(
        self,
        settings: InferenceSettings,
        transport: OpenRouterClient,
        memory_context: MemoryContext | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self.memory_context = memory_context

    def classify_window(self, window: FrameWindow) -> WindowResponse:
        started = time.monotonic()
        try:
            response = self._transport.chat_completion(
                model=self.settings.model,
                messages=build_window_messages(
                    window,
                    event_label=self.settings.event_label,
                    system_prompt=self.settings.system_prompt,
                    prompt=self.settings.prompt,
                    memory_context=self.memory_context,
                ),
                **dict(self.settings.request_parameters),
            )
            text = self._transport.assistant_text(response)
        except (OpenRouterError, ValueError) as exc:
            return WindowResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )
        latency = time.monotonic() - started
        response_id = (
            str(response["id"]) if response.get("id") is not None else None
        )
        usage = response.get("usage")
        try:
            classification = parse_binary_classification(
                text,
                allowed_memory_ids=(
                    self.memory_context.memory_ids
                    if self.memory_context is not None
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            return WindowResponse(
                window=window,
                classification=None,
                raw_text=text,
                latency_seconds=latency,
                response_id=response_id,
                usage=usage if isinstance(usage, Mapping) else None,
                error=f"unparseable model reply: {exc}",
            )
        return WindowResponse(
            window=window,
            classification=classification,
            raw_text=text,
            latency_seconds=latency,
            response_id=response_id,
            usage=usage if isinstance(usage, Mapping) else None,
        )


class DryRunWindowClient:
    """Deterministic stand-in that never touches the network.

    Windows alternate between stretches of "clear" and "detected" so the demo
    UI exercises both states, and a small index-derived delay simulates model
    latency without randomness.
    """

    def __init__(
        self,
        settings: InferenceSettings,
        memory_context: MemoryContext | None = None,
    ) -> None:
        self.settings = settings
        self.memory_context = memory_context

    def classify_window(self, window: FrameWindow) -> WindowResponse:
        started = time.monotonic()
        time.sleep(0.35 + (window.index * 13 % 40) / 100.0)
        detected = window.index % 6 in {2, 3, 4}
        confidence = round(0.6 + (window.index * 37 % 100) / 250.0, 3)
        memory_ids = (
            self.memory_context.memory_ids
            if self.memory_context is not None
            else ()
        )
        if self.memory_context is not None and not memory_ids:
            detected = False
        state = "visibly active" if detected else "not visible"
        classification = BinaryClassification(
            detected=detected,
            confidence=confidence,
            description=(
                f"Dry run: {self.settings.event_label} {state} in {window.label}."
            ),
            matched_memory_ids=memory_ids[:1] if detected else (),
        )
        return WindowResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=time.monotonic() - started,
        )


def build_client(
    settings: InferenceSettings,
    openrouter: OpenRouterSettings,
    *,
    memory_context: MemoryContext | None = None,
    dry_run: bool = False,
) -> WindowClient:
    """Build the live Qwen client, or the offline stand-in for dry runs."""

    if dry_run:
        return DryRunWindowClient(settings, memory_context)
    return QwenWindowClient(
        settings,
        OpenRouterClient(openrouter),
        memory_context,
    )
