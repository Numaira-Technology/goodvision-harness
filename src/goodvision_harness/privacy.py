"""Public-output privacy helpers.

The service deliberately keeps implementation/vendor identifiers internal.  All
records that can leave the process should pass through these helpers, while raw
third-party failures are reduced to stable, operator-safe messages at the point
where they are caught.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

PUBLIC_ANALYSIS_ERROR = "analysis request failed"
PUBLIC_FACE_ERROR = "local face analysis failed"

_PRIVATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Remove the whole vendor URL, not just its recognizable substring.
        r"https?://[^\s\"'<>]*(?:paddle|bcebos|huggingface|hf\.co|modelscope|"
        r"nvidia|openai|anthropic|deepseek|qwen|aliyun|dashscope|jsdelivr|"
        r"googleapis|fastapi|onnx|opencv|insightface)[^\s\"'<>]*",
        r"https?://aiclub\.zimaocloud\.com(?:/[^\s\"']*)?",
        r"(?<![a-z0-9])ai[\s_-]*club[\w.+:/-]*",
        r"https?://(?:www\.)?openrouter\.ai(?:/[^\s\"']*)?",
        r"(?<![a-z0-9])open[\s_-]*router[\w.+:/-]*",
        r"(?<![a-z0-9])qwen[\w./:+-]*",
        r"(?<![a-z0-9])(?:coreweave|deepinfra)[\w.+:/-]*",
        r"(?<![a-z0-9])insight[\s_-]*face[\w.+:/-]*",
        r"\bbuffalo(?:[_-][\w.-]+)?\b",
        r"\bantelope(?:v\d+)?\b",
        r"\bonnx(?:runtime)?(?:-gpu)?\b",
        r"\b(?:cpu|cuda|dml)executionprovider\b",
        r"\bfaceanalysis\b",
        # Include suffixed exception/module identifiers and model variants.
        r"(?<![a-z0-9])paddle[\w.+:/-]*",
        r"(?<![a-z0-9])(?:pp[\s_-]*)?(?:yolo[e]?|lcnet|picodet)[\w.+:/-]*",
        r"(?<![a-z0-9])(?:nvidia|tensorrt|cudnn|cublas|cufft|curand|"
        r"nvjitlink|cuda|directml|onednn|mkl)[\w.+:/-]*",
        r"(?<![a-z0-9])(?:openai|anthropic|claude|deepseek|gemini|"
        r"hugging[\s_-]*face|modelscope|ollama|dashscope)[\w.+:/-]*",
        r"(?<![a-z0-9])(?:opencv|ffmpeg|libav|fastapi|starlette|pydantic|"
        r"uvicorn|swagger|redoc|jsdelivr|httpx|aiohttp|aiortc|pyyaml|"
        r"numpy|scipy|scikit[\s_-]*\w*|"
        r"pandas|matplotlib|faiss|pytorch|torch|tensorflow|timm|"
        r"onnxruntime|onnx)[\w.+:/-]*",
        r"(?<![a-z0-9])(?:zimaocloud|bcebos|baidu|alibaba|aliyun|"
        r"tencent|huawei|microsoft|google|azure|amazon|aws)[\w.+:/-]*",
        r"飞桨|通义千问|通义|千问|英伟达|百度|阿里云|腾讯云|华为云",
    )
)


def contains_private_implementation_detail(value: object) -> bool:
    """Return whether text contains a known private engine/provider marker."""

    rendered = str(value)
    return any(pattern.search(rendered) for pattern in _PRIVATE_PATTERNS)


def redact_public_text(value: object, *, fallback: str = "analysis engine") -> str:
    """Redact known implementation identifiers from otherwise useful text."""

    rendered = str(value)
    for pattern in _PRIVATE_PATTERNS:
        rendered = pattern.sub(fallback, rendered)
    return rendered


def public_error_message(
    _error: object | None = None,
    *,
    fallback: str = PUBLIC_ANALYSIS_ERROR,
) -> str:
    """Return a stable public failure without echoing an underlying exception."""

    return fallback


def public_safe_value(value: Any) -> Any:
    """Recursively redact strings in JSON-shaped public records."""

    if isinstance(value, str):
        return redact_public_text(value)
    if isinstance(value, Mapping):
        return {
            redact_public_text(key): public_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(public_safe_value(item) for item in value)
    if isinstance(value, list):
        return [public_safe_value(item) for item in value]
    return value
