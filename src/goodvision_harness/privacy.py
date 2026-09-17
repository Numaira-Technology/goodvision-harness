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
PUBLIC_TRACKING_NAME = "人员追踪模块"
PUBLIC_TRACKING_METHOD = "person_tracking"
PUBLIC_TRACKING_ERROR = "人物跟踪服务连接或处理失败。请重新开始检测。"
PUBLIC_INSPECTION_ERROR = "事件分析服务连接或处理失败。请重新开始检测。"
PUBLIC_FACE_ERROR = "local face analysis failed"
PUBLIC_FACE_UNAVAILABLE_ERROR = (
    "本地人脸识别组件不可用。请安装或修复服务端人脸组件后重试。 / "
    "Local face recognition is unavailable; install or repair the server's "
    "face components, then retry."
)
PUBLIC_STREAM_ERROR = "video input failed; check the video source and retry"
PUBLIC_STREAM_SUPPORT_ERROR = "video input support is missing on this server"
PUBLIC_MISSING_API_KEY_ERROR = (
    "请先在控制台保存 API key，再启动正式检测；无需创建 .env。 / "  # noqa: RUF001
    "Save an API key in the console before starting live detection; "
    "no .env file is required."
)


class LocalFaceUnavailableError(RuntimeError):
    """Known missing local components, safe to map to an actionable public error."""


# These values belong to the application's correlation/CRUD contract, not to
# its implementation metadata. Customer-supplied identifiers can legitimately
# contain any text; changing them would break lookups or merge distinct records.
_IDENTIFIER_FIELDS = frozenset({
    "message_id", "observation_id", "source_id", "stream_session_id",
    "object_id", "track_id", "person_id", "stranger_id", "memory_id",
    "analysis_message_id", "next_analysis_message_id", "association_message_id",
    "subject_id", "event_id", "canonical_event_id", "requested_event_id",
    "person_track_id", "face_track_id", "actor_id", "epoch", "predictor_epoch",
})
_IDENTIFIER_LIST_FIELDS = frozenset({
    "object_ids", "memory_ids", "matched_memory_ids",
    "subject_ids", "event_ids", "person_track_ids", "actor_ids",
})
_IMPLEMENTATION_FIELDS = frozenset({
    "engine", "provider", "backend", "model", "model_name",
    "execution_provider", "execution_providers",
})

_TRACKING_PATTERN = re.compile(
    r"https?://[^\s\"'<>]*(?:segment-anything|facebookresearch/sam|"
    r"facebook/sam)[^\s\"'<>]*|"
    r"(?<![a-z0-9])(?:(?:facebook(?:research)?/)?sam(?:[\s_-]*v?[23](?:[._]?1)?)?"
    r"(?:[_./:+-][a-z0-9_.:/+-]+|(?:Runtime)?Error)?|"
    r"segment[\s_-]*anything(?:[\s_-]*[23](?:\.1)?)?)(?![a-z0-9])",
    re.IGNORECASE,
)
_TRACKING_ALIASES = {
    "sam_live": PUBLIC_TRACKING_METHOD,
    "sam31": "person_tracking_video",
    "sam31_stream": "person_tracking_stream",
    "sam3.1_video": "person_tracking_video",
    "native_sam_video_tracking": "native_person_tracking",
}

_PRIVATE_PATTERNS = tuple(
    _TRACKING_PATTERN if pattern == _TRACKING_PATTERN.pattern
    else re.compile(pattern, re.IGNORECASE)
    for pattern in (
        _TRACKING_PATTERN.pattern,
        # Remove the whole vendor URL, not just its recognizable substring.
        r"https?://[^\s\"'<>]*(?:paddle|bcebos|huggingface|hf\.co|modelscope|"
        r"nvidia|openai|anthropic|deepseek|qwen|aliyun|dashscope|jsdelivr|"
        r"googleapis|fastapi|onnx|opencv|insightface)[^\s\"'<>]*",
        r"https?://aiclub\.zimaocloud\.com(?:/[^\s\"']*)?",
        r"(?<![a-z0-9])ai[\s_-]*club[\w.+:/-]*",
        r"https?://(?:www\.)?openrouter\.ai(?:/[^\s\"']*)?",
        r"(?<![a-z0-9])open[\s_-]*router[\w.+:/-]*",
        r"(?<![a-z0-9])qwen[\w./:+-]*",
        r"(?<![a-z0-9])(?:coreweave|deepinfra|reka|parasail|novita|chutes|ultralytics|"
        r"bytetrack|botsort|xformers|vllm)[\w.+:/-]*",
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

# These tokens can also be names, physical objects, animals, or customer IDs.
# Keep the strict vocabulary above for diagnostic and release-file audits;
# ordinary result data needs evidence that a token denotes an implementation.
_DOMAIN_TERM = re.compile(
    r"^(torch|pandas|timm|claude|gemini|google|amazon|azure|aws|baidu|"
    r"alibaba|tencent|huawei|microsoft|buffalo|antelope)(.*)$",
    re.IGNORECASE,
)
_IMPLEMENTATION_FORM = re.compile(
    r"[./:](?=[a-z0-9_])|(?:error|exception|warning)(?:[:_-].*)?$|"
    r"(?:claude|gemini)[-_/]?(?:\d|(?:sonnet|opus|haiku|flash|pro)(?:[-_/]|$))|"
    r"(?:torch|pandas|timm)[-_v]?\d|"
    r"torch(?:vision|audio|text|data|metrics|script|_dynamo|_inductor)|"
    r"(?:googleapis|amazonaws|microsoftonline)|"
    r"[-_](?:api|sdk|client|backend|runtime|provider|engine|model|key|version)"
    r"(?:[-_]|$)|(?:api|sdk)(?:key|client|error|exception)$|"
    r"^buffalo[_-]|^antelopev\d",
    re.IGNORECASE,
)
_IMPLEMENTATION_CONTEXT = re.compile(
    r"(?<![a-z0-9])(?:"
    r"(?:from|import)\s+(?:torch|pandas|timm)(?:\.[\w.]+)?|"
    r"(?:torch|pandas|timm)\s+(?:backend|framework|library|module|runtime)|"
    r"(?:claude|gemini)\s+(?:model|api|sdk)|"
    r"(?:google|amazon|azure|aws|baidu|alibaba|tencent|huawei|microsoft)"
    r"\s+(?:cloud|api|sdk))\b",
    re.IGNORECASE,
)


def _redact_data_match(match: re.Match[str], fallback: str) -> str:
    token = match.group()
    if _TRACKING_PATTERN.fullmatch(token):
        # Sam may be an enrolled person's name; uppercase/model forms are metadata.
        if token.lower() == "sam" and token != "SAM":
            return token
        return _TRACKING_ALIASES.get(token.lower(), PUBLIC_TRACKING_NAME)
    if _DOMAIN_TERM.fullmatch(token) and not _IMPLEMENTATION_FORM.search(token):
        return token
    return fallback


def contains_private_implementation_detail(value: object) -> bool:
    """Return whether text contains a known private engine/provider marker."""

    rendered = str(value)
    return any(pattern.search(rendered) for pattern in _PRIVATE_PATTERNS)


def redact_public_text(value: object, *, fallback: str = "analysis engine") -> str:
    """Redact implementation details while preserving ordinary domain values."""

    rendered = str(value)
    if rendered in _TRACKING_ALIASES:
        return _TRACKING_ALIASES[rendered]
    rendered = re.sub(
        r"(?i)(?<![a-z0-9])sam\s+(?:model|backend|tracker|module|模型|追踪)",
        lambda _: PUBLIC_TRACKING_NAME, rendered,
    )
    rendered = _IMPLEMENTATION_CONTEXT.sub(lambda _: fallback, rendered)
    for pattern in _PRIVATE_PATTERNS:
        rendered = pattern.sub(
            lambda match: _redact_data_match(match, fallback), rendered,
        )
    return rendered


def public_error_message(
    _error: object | None = None,
    *,
    fallback: str = PUBLIC_ANALYSIS_ERROR,
) -> str:
    """Preserve known public failures without echoing an underlying exception."""

    if isinstance(_error, LocalFaceUnavailableError):
        return PUBLIC_FACE_UNAVAILABLE_ERROR
    if isinstance(_error, str) and _error in {
        PUBLIC_STREAM_ERROR, PUBLIC_STREAM_SUPPORT_ERROR,
        PUBLIC_TRACKING_ERROR, PUBLIC_INSPECTION_ERROR,
    }:
        return _error
    return fallback


def _public_implementation_value(value: Any) -> Any:
    """Strictly project explicit engine metadata, not ordinary result data."""

    if isinstance(value, str):
        if value in _TRACKING_ALIASES:
            return _TRACKING_ALIASES[value]
        for pattern in _PRIVATE_PATTERNS:
            value = pattern.sub(
                PUBLIC_TRACKING_NAME if pattern is _TRACKING_PATTERN
                else "analysis engine", value,
            )
        return value
    if isinstance(value, Mapping):
        return {
            _public_implementation_value(str(key)): _public_implementation_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_public_implementation_value(item) for item in value)
    if isinstance(value, list):
        return [_public_implementation_value(item) for item in value]
    return value


def _public_field_value(key: object, value: Any) -> Any:
    if key in _IDENTIFIER_FIELDS and isinstance(value, str):
        return value
    if (
        key in _IDENTIFIER_LIST_FIELDS
        and isinstance(value, (list, tuple))
        and all(isinstance(item, str) for item in value)
    ):
        return tuple(value) if isinstance(value, tuple) else list(value)
    if key in _IMPLEMENTATION_FIELDS:
        return _public_implementation_value(value)
    return public_safe_value(value)


def public_safe_value(value: Any) -> Any:
    """Redact implementation text without rewriting business identifiers."""

    if isinstance(value, str):
        return redact_public_text(value)
    if isinstance(value, Mapping):
        return {
            redact_public_text(key): _public_field_value(key, item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(public_safe_value(item) for item in value)
    if isinstance(value, list):
        return [public_safe_value(item) for item in value]
    return value
