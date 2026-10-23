"""Frozen scene-event comparison; import and planning never read credentials."""

from dataclasses import dataclass

from goodvision_harness.model_api import ModelAPISettings
from tools.benchmark_runtime.vlm_event_policy import event_policy_snapshot

MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/decisions"
VERSION = "openai-decisions-events-v1"
LABELS = ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter")
POLICY_SHA256 = "d9c487c37404df2f6d1f4d1f3da39559abffe28f59c76e7fef85f4a9f9e48a7b"
VIDEO_SHA256 = "2ac7a0f3271c4c9937b1796bdc43526617d9b1715de9bdf15713d8d3d80dacb9"
BASE_INPUT_PRICE_PER_MILLION = 0.10
FROZEN = {
    "backend": "openai_decisions",
    "model": MODEL,
    "endpoint": ENDPOINT,
    "streams": [12],
    "modes": ["events"],
    "duration": 55.798021,
    "repeats": 2,
    "observation_fps": 1.25,
    "global_limit": 48,
    "per_stream_limit": 4,
    "local_concurrency": 4,
    "request_rate": 40,
    "queue_timeout": 1,
    "http_timeout": 20,
    "warmup_seconds": 10,
    "soak_seconds": 0,
    "arrival_mode": "staggered",
    "monitor_layout": "single-video",
    "record": True,
    "admission_storage": "memory",
    "admission_diagnostics": True,
    "provider_only": "",
    "provider_prefer": "",
    "vision_profile": "",
    "self_hosted_qwen35": False,
    "cpu_offload": False,
    "metrics_url": "",
    "grounding_concurrency_sweep": [],
    "parallel_concurrency_sweep": [],
    "measurement_start_index": 0,
    "rate_limit_cooldown": 60,
    "allow_partial_rate_limited_preflight": False,
}


def check_policy():
    policy = event_policy_snapshot(LABELS)
    if policy["sha256"] != POLICY_SHA256:
        raise ValueError("six-event policy changed; create a separate comparison")
    return policy


def validate(args):
    contract = FROZEN
    if getattr(args, "decisions_person_tracking", False):
        from tools.tracked_decisions.profile import FROZEN as contract
    changed = [k for k, v in contract.items() if getattr(args, k, None) != v]
    if changed:
        raise ValueError("Decisions contract changed: " + ", ".join(changed))
    check_policy()


@dataclass(frozen=True, slots=True)
class DecisionSettings(ModelAPISettings):
    """Benchmark-only settings; never register Decisions as a chat provider."""

    provider: str = "openai_decisions"
    endpoint: str = ENDPOINT
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 20
    max_retries: int = 0

    def __post_init__(self):
        if (
            self.provider != "openai_decisions"
            or self.endpoint != ENDPOINT
            or self.api_key_env != "OPENAI_API_KEY"
            or self.timeout_seconds != 20
            or self.max_retries != 0
            or self.extra_headers
            or self.inspection_endpoint
        ):
            raise ValueError("Decisions requires its fixed direct route and credential")
