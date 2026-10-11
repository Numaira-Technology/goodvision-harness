"""Explicit benchmark routes; no credential or provider fallback between APIs."""

import os
import re
from urllib.parse import urlsplit

OPENROUTER_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
FIXED_OPENROUTER_PROVIDERS = (
    "deepinfra", "siliconflow", "novita", "parasail", "venice", "alibaba",
)
DASHSCOPE_MODEL = "qwen3-vl-30b-a3b-instruct"
KEY_ENV = {"openrouter": "OPENROUTER_API_KEY", "dashscope": "DASHSCOPE_API_KEY"}
_DASHSCOPE_HOSTS = {
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "dashscope-us.aliyuncs.com",
    "cn-hongkong.dashscope.aliyuncs.com",
}
_WORKSPACE_HOST = re.compile(
    r"[a-z0-9][a-z0-9-]*\."
    r"(?:cn-beijing|ap-southeast-1|ap-northeast-1|eu-central-1|us-east-1|cn-hongkong)"
    r"\.maas\.aliyuncs\.com\Z"
)


def model_name(backend, model):
    # Translate only this known model pair; never silently substitute a model.
    if backend == "dashscope" and model in ("", OPENROUTER_MODEL):
        return DASHSCOPE_MODEL
    return model or OPENROUTER_MODEL


def validate_route(backend, endpoint="", provider_only=""):
    if backend not in KEY_ENV:
        raise ValueError("external backend must be openrouter or dashscope")
    if backend == "openrouter":
        if endpoint:
            raise ValueError("OpenRouter uses its fixed endpoint")
        return
    if provider_only:
        raise ValueError("--provider-only is an OpenRouter routing option")
    if not endpoint:  # Existing ModelAPISettings default: Beijing shared domain.
        return
    url = urlsplit(endpoint)
    # A full Chat Completions URL, including workspace-specific domains. Fail
    # before uploading a bundle or reading credentials on base-URL mistakes.
    if (
        url.scheme != "https"
        or not url.hostname
        or not (
            url.hostname in _DASHSCOPE_HOSTS
            or _WORKSPACE_HOST.fullmatch(url.hostname)
        )
        or url.port not in (None, 443)
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path != "/compatible-mode/v1/chat/completions"
        or any(c.isspace() for c in endpoint)
    ):
        raise ValueError(
            "DashScope requires an Alibaba HTTPS Chat Completions endpoint"
        )


def read_api_key(backend, env_file=None, *, required=True):
    name = KEY_ENV[backend]
    key = os.environ.get(name, "").strip()
    if not key and env_file and env_file.is_file():
        from dotenv import dotenv_values

        key = (dotenv_values(env_file).get(name) or "").strip()
    if not key and required:
        raise ValueError(f"{name} is unavailable")
    return key


def mode_api_key(settings):
    if settings.provider in KEY_ENV:
        return read_api_key(settings.provider, required=False)
    # The optional local endpoint retains its dedicated local credential rules.
    from goodvision_harness.model_api import environment_api_key

    return environment_api_key(settings)
