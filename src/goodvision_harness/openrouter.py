"""Small dependency-free client for OpenRouter's chat-completions endpoint."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class OpenRouterError(RuntimeError):
    """Raised when the configured model API cannot return a usable response."""


@dataclass(frozen=True, slots=True)
class OpenRouterSettings:
    """Connection settings that do not depend on a specific model or prompt."""

    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "MODEL_API_KEY"
    timeout_seconds: float = 90.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    http_referer: str | None = None
    app_title: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("openrouter.endpoint must be an absolute HTTP(S) URL")
        if not self.api_key_env:
            raise ValueError("openrouter.api_key_env must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("openrouter.timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("openrouter.max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("openrouter.retry_backoff_seconds must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OpenRouterSettings:
        extra_headers = value.get("extra_headers") or {}
        if not isinstance(extra_headers, Mapping):
            raise TypeError("openrouter.extra_headers must be a mapping")
        return cls(
            endpoint=str(
                value.get("endpoint", "https://openrouter.ai/api/v1/chat/completions")
            ),
            api_key_env=str(value.get("api_key_env", "MODEL_API_KEY")),
            timeout_seconds=float(value.get("timeout_seconds", 90.0)),
            max_retries=int(value.get("max_retries", 3)),
            retry_backoff_seconds=float(value.get("retry_backoff_seconds", 1.0)),
            http_referer=_optional_string(value.get("http_referer")),
            app_title=_optional_string(value.get("app_title")),
            extra_headers={str(key): str(item) for key, item in extra_headers.items()},
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def api_key_present(settings: OpenRouterSettings) -> bool:
    """True when the configured API-key environment variable is set."""

    return bool(os.environ.get(settings.api_key_env, "").strip())


class OpenRouterClient:
    """Generic JSON client used by the video inference layer."""

    _RETRYABLE_STATUSES: ClassVar[set[int]] = {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings
        self._api_key = (api_key or os.environ.get(settings.api_key_env, "")).strip()
        if not self._api_key:
            raise OpenRouterError("model API key is not configured")

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        """POST a model-agnostic chat completion and return its JSON object."""

        if not model.strip():
            raise ValueError("model must not be empty")
        if not messages:
            raise ValueError("messages must not be empty")

        payload = {key: value for key, value in parameters.items() if value is not None}
        payload.update({"model": model, "messages": list(messages)})
        return self._post_json(payload)

    @staticmethod
    def assistant_text(response: Mapping[str, Any]) -> str:
        """Extract text from either string or structured assistant content."""

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(
                "response did not contain an assistant message"
            ) from exc

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Sequence):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, Mapping) and block.get("type") == "text"
            ]
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
        raise OpenRouterError("assistant message did not contain text")

    def _headers(self) -> dict[str, str]:
        headers = {
            str(key): str(value) for key, value in self.settings.extra_headers.items()
        }
        headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "goodvision-streaming-harness/1",
            }
        )
        if self.settings.http_referer:
            headers["HTTP-Referer"] = self.settings.http_referer
        if self.settings.app_title:
            headers["X-Title"] = self.settings.app_title
        return headers

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.settings.endpoint,
            data=encoded,
            headers=self._headers(),
            method="POST",
        )

        for attempt in range(self.settings.max_retries + 1):
            try:
                with urlopen(
                    request, timeout=self.settings.timeout_seconds
                ) as response:
                    body = response.read()
                try:
                    decoded = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise OpenRouterError("model API returned invalid JSON") from exc
                if not isinstance(decoded, dict):
                    raise OpenRouterError("model API returned an invalid response")
                return decoded
            except HTTPError as exc:
                # Consume the response so the connection can be released, but
                # never copy the vendor error body into application records.
                exc.read()
                if (
                    exc.code in self._RETRYABLE_STATUSES
                    and attempt < self.settings.max_retries
                ):
                    self._sleep_before_retry(attempt, exc.headers.get("Retry-After"))
                    continue
                raise OpenRouterError(
                    f"model API returned HTTP {int(exc.code)}"
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.settings.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise OpenRouterError("model API request failed") from exc

        raise AssertionError("retry loop terminated unexpectedly")

    def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        delay: float | None = None
        if retry_after:
            try:
                delay = max(0.0, float(retry_after))
            except ValueError:
                delay = None
        if delay is None:
            delay = self.settings.retry_backoff_seconds * (2**attempt)
        if delay:
            time.sleep(delay)
