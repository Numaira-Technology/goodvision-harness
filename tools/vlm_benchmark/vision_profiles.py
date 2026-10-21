"""Frozen visual comparison arms; no credentials or IO at import time."""

PROFILES = {
    "qwen27": {
        "model": "qwen/qwen3.8-27b",
        "provider": "cerebras/fp16",
        "provider_name": "Cerebras",
        "prefer": False,
    },
    "gemini": {
        "model": "google/gemini-3.1-flash-lite",
        "provider": "google-ai-studio",
        "provider_name": "Google AI Studio",
        "prefer": False,
    },
    "glm": {
        "model": "z-ai/glm-5.3-flash",
        "provider": "coreweave/nvfp4",
        "provider_name": "CoreWeave",
        "prefer": False,
    },
    "qwen35": {
        "model": "qwen/qwen3.5-35b-a3b",
        "provider": "alibaba",
        "provider_name": "Alibaba",
        "prefer": True,
    },
    "qwen36_flash": {
        "model": "qwen/qwen3.6-flash",
        "provider": "alibaba",
        "provider_name": "Alibaba",
        "prefer": False,
    },
}
# Gemini and 35B remain in the catalog for explicit, bounded diagnostics only.
MATRIX_ARMS = ("qwen27", "glm")
PROFILES.update(
    qwen27_paced={**PROFILES["qwen27"], "base_profile": "qwen27"},
    glm_low={**PROFILES["glm"], "base_profile": "glm"},
)
OPTIMIZATION_ARMS = ("qwen27", "qwen27_paced", "glm", "glm_low")
OPTIMIZATION_PAIRS = (("qwen27", "qwen27_paced"), ("glm", "glm_low"))
OPTIMIZATION_VERSION = "reasoning-and-dispatch-ab-v1"
QWEN36_FLASH_VERSION = "qwen36-flash-alibaba-v1"
QWEN36_FLASH_ARMS = ("qwen36_flash",)
FORMAL_ARMS = (*OPTIMIZATION_ARMS, *QWEN36_FLASH_ARMS)
PROFILE_VERSION = "vision-two-models-3s-v7"


def experiment_arms(experiment=None):
    groups = {
        None: MATRIX_ARMS,
        OPTIMIZATION_VERSION: OPTIMIZATION_ARMS,
        QWEN36_FLASH_VERSION: QWEN36_FLASH_ARMS,
    }
    if experiment not in groups:
        raise ValueError("unknown comparison experiment")
    return groups[experiment]


def base_profile(profile):
    return PROFILES[profile].get("base_profile", profile)


def dispatch_contract(profile):
    return {
        "policy": "smooth-http-start-v1" if profile == "qwen27_paced" else "original",
        "rate": "streams * observation_fps * branches"
        if profile == "qwen27_paced"
        else None,
        "max_wait_seconds": 1.0 if profile == "qwen27_paced" else 0.0,
        "source_arrival": "staggered",
        "wait_included_in_deadline": True,
    }


def evaluation_contract():
    """A scoring deadline, independent of the HTTP cancellation timeout."""
    return {
        "deadline_seconds": 3,
        "minimum_on_time_coverage": 0.95,
        "cameras": 12,
        "planned_windows_per_camera": 69,
        "required_timely_per_camera": 66,
        "latency_field": "source_to_publish_seconds",
        "scope": "latest evidence enters VLM branch through result publication",
        "denominator": "all planned windows, including unsent/failed/missing/late",
        "parallel": "same camera/window_id; both complete within deadline",
        "acceptance": "every camera in each mode and each repeat independently",
    }


def runtime_contract(profile):
    """Reasoning is mandatory on GLM; the cap includes thinking and JSON.

    Use one HTTP deadline across the new matrix. Historical five-second events
    runs must not be pooled with this protocol. Load and judgment rules stay fixed.
    """
    if profile not in PROFILES:
        raise ValueError("unknown vision comparison profile")
    return {
        "reasoning": {"enabled": True, "effort": "low"}
        if profile == "glm_low"
        else {"enabled": base_profile(profile) == "glm"},
        "max_tokens": (
            {"events": 8192, "compact": 8192}
            if base_profile(profile) == "glm"
            else {"events": 160, "compact": 1600}
        ),
        "http_timeouts": {"events": 20, "compact": 20},
        "max_retries": 0,
        "preflight_429_policy": "continue_partial_verified_after_bounded_cooldown",
        "temperature": 0,
        "dispatch": dispatch_contract(profile),
    }


def request_parameters(profile):
    contract = runtime_contract(profile)
    return {
        "provider": routing(profile),
        "reasoning": contract["reasoning"],
        "temperature": contract["temperature"],
    }


def routing(profile):
    arm = PROFILES[profile]
    return {
        "order" if arm["prefer"] else "only": [arm["provider"]],
        "allow_fallbacks": arm["prefer"],
        "require_parameters": True,
        "sort": "latency",
    }


def validate_profile(args):
    if args.vision_profile not in FORMAL_ARMS and getattr(args, "real", False):
        raise ValueError("profile excluded from formal matrix; use bounded diagnostics")
    arm = PROFILES[args.vision_profile]
    expected = {
        "backend": "openrouter",
        "endpoint": "",
        "model": arm["model"],
        "provider_only": "" if arm["prefer"] else arm["provider"],
        "provider_prefer": arm["provider"] if arm["prefer"] else "",
        "require_route_match": True,
        "streams": [12],
        "modes": ["events", "compact", "parallel"],
        "grounding_format": "lean-v1",
        "parallel_scheduling": "paired",
        "global_limit": 48,
        "per_stream_limit": 4,
        "local_concurrency": 4,
        "request_rate": 40,
        "observation_fps": 1.25,
        "duration": 55.798021,
        "repeats": 2,
        "queue_timeout": 1,
        "http_timeout": 20,
        "warmup_seconds": 10,
        "soak_seconds": 0,
        "arrival_mode": "staggered",
        "monitor_layout": "single-video",
        "record": True,
        "admission_storage": "memory",
        "admission_diagnostics": True,
        "rate_limit_cooldown": 120,
        "allow_partial_rate_limited_preflight": True,
        "grounding_concurrency_sweep": [],
        "parallel_concurrency_sweep": [],
        "measurement_start_index": 0,
    }
    drift = [key for key, value in expected.items() if getattr(args, key) != value]
    if drift:
        raise ValueError("visual comparison settings differ: " + ", ".join(drift))
