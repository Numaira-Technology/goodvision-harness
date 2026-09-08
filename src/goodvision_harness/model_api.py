"""Shared client for selectable OpenAI-compatible chat-completions services."""

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

import httpx


DEFAULT_AICLUB_ENDPOINT = (
    "https://aiclub.zimaocloud.com/model/openApi/v1/chat/completions"
)
DEFAULT_AICLUB_API_KEY_ENV = "AICLUB_API_KEY"
DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DASHSCOPE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
PROVIDER_DEFAULTS = {
    "aiclub": (DEFAULT_AICLUB_ENDPOINT, DEFAULT_AICLUB_API_KEY_ENV, "qwen3.5-plus"),
    "openrouter": (
        DEFAULT_OPENROUTER_ENDPOINT, "OPENROUTER_API_KEY", "qwen/qwen3.5-35b-a3b"
    ),
    "dashscope": (DEFAULT_DASHSCOPE_ENDPOINT, "DASHSCOPE_API_KEY", "qwen3.5-plus"),
}


class ModelAPIError(RuntimeError):
    """Raised when the selected service cannot return a usable chat completion."""


@dataclass(frozen=True, slots=True)
class ModelAPISettings:
    """Connection settings that do not depend on a specific model or prompt."""

    endpoint: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 90.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    provider: str = "aiclub"

    def __post_init__(self) -> None:
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                "inference_service.provider must be aiclub, openrouter, or dashscope"
            )
        endpoint, api_key_env, _model = PROVIDER_DEFAULTS[self.provider]
        if not self.endpoint:
            object.__setattr__(self, "endpoint", endpoint)
        if not self.api_key_env:
            object.__setattr__(self, "api_key_env", api_key_env)
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("inference_service.endpoint must be an absolute HTTP(S) URL")
        if not self.api_key_env.strip():
            raise ValueError("inference_service.api_key_env must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("inference_service.timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("inference_service.max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "inference_service.retry_backoff_seconds must be non-negative"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelAPISettings:
        extra_headers = value.get("extra_headers") or {}
        if not isinstance(extra_headers, Mapping):
            raise TypeError("inference_service.extra_headers must be a mapping")
        return cls(
            endpoint=str(value.get("endpoint", "")),
            api_key_env=str(value.get("api_key_env", "")),
            provider=str(value.get("provider", "aiclub")).strip().lower(),
            timeout_seconds=float(value.get("timeout_seconds", 90.0)),
            max_retries=int(value.get("max_retries", 3)),
            retry_backoff_seconds=float(value.get("retry_backoff_seconds", 1.0)),
            extra_headers={str(key): str(item) for key, item in extra_headers.items()},
        )


def api_key_present(settings: ModelAPISettings) -> bool:
    """True when the configured API-key environment variable is set."""

    return bool(environment_api_key(settings))


def environment_api_key(settings: ModelAPISettings) -> str:
    # Generic overrides are explicit; never guess using another route's key.
    names = ["INFERENCE_API_KEY", settings.api_key_env]
    if settings.api_key_env in {"INFERENCE_API_KEY", "MODEL_API_KEY"}:
        names.append(PROVIDER_DEFAULTS[settings.provider][1])
    names.append("MODEL_API_KEY")
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


class ModelAPIClient:
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
        settings: ModelAPISettings,
        *,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings
        self._api_key = (api_key or environment_api_key(settings)).strip()
        if not self._api_key:
            raise ModelAPIError(f"missing model API key; set {settings.api_key_env}")

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
            raise ModelAPIError(
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
        raise ModelAPIError("assistant message did not contain text")

    def _headers(self) -> dict[str, str]:
        headers = {
            str(key): str(value) for key, value in self.settings.extra_headers.items()
        }
        headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "goodvision-harness/1",
            }
        )
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
                    raise ModelAPIError("model API returned invalid JSON") from exc
                if not isinstance(decoded, dict):
                    raise ModelAPIError("model API returned an invalid response")
                return decoded
            except HTTPError as exc:
                # Consume the provider body so the connection can be released,
                # but never copy vendor details into logs or public records.
                exc.read()
                if (
                    exc.code in self._RETRYABLE_STATUSES
                    and attempt < self.settings.max_retries
                ):
                    retry_after = None
                    if exc.headers is not None:
                        retry_after = exc.headers.get("Retry-After")
                    self._sleep_before_retry(attempt, retry_after)
                    continue
                raise ModelAPIError(
                    f"model API returned HTTP {int(exc.code)}"
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.settings.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise ModelAPIError("model API request failed") from exc

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


class PooledModelAPIClient(ModelAPIClient):
    """Shared, thread-safe verified HTTPS pool for route probes and live runs."""

    def __init__(self, settings: ModelAPISettings, *, api_key: str | None = None):
        super().__init__(settings, api_key=api_key)
        self._http = httpx.Client(
            http2=False, follow_redirects=False,
            limits=httpx.Limits(
                max_connections=6, max_keepalive_connections=6,
                keepalive_expiry=120,
            ),
        )

    def close(self) -> None:
        self._http.close()

    def _post_json(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self._http.post(
                    self.settings.endpoint, headers=self._headers(), json=payload,
                    timeout=self.settings.timeout_seconds,
                )
            except httpx.RequestError as exc:
                if attempt < self.settings.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise ModelAPIError("model API request failed") from exc
            if not 200 <= response.status_code < 300:
                if (response.status_code in self._RETRYABLE_STATUSES
                        and attempt < self.settings.max_retries):
                    self._sleep_before_retry(attempt, response.headers.get("Retry-After"))
                    continue
                raise ModelAPIError(f"model API returned HTTP {response.status_code}")
            try:
                decoded = response.json()
            except (ValueError, UnicodeDecodeError) as exc:
                raise ModelAPIError("model API returned invalid JSON") from exc
            if not isinstance(decoded, dict):
                raise ModelAPIError("model API returned an invalid response")
            return decoded
        raise AssertionError("retry loop terminated unexpectedly")
