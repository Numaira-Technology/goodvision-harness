"""Frozen deployment and load contract; no network or secret access."""

MODEL_REPO = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
# Keep the historical alias so the existing grounding shape hint is identical.
SERVED_MODEL = "qwen/qwen3.5-35b-a3b"
ENGINE_VERSION = "0.31.0"
IMAGE = (
    "vllm/vllm-openai@sha256:"
    "c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
)
VIDEO_SHA256 = "2ac7a0f3271c4c9937b1796bdc43526617d9b1715de9bdf15713d8d3d80dacb9"
POLICY_SHA256 = "d9c487c37404df2f6d1f4d1f3da39559abffe28f59c76e7fef85f4a9f9e48a7b"
VERSION = "vast-qwen35-bf16-v1"
BASE_URL = "http://127.0.0.1:8000"
CACHE_POLICY = "prefix=off;multimodal_processor=off;warmup=10s"
FROZEN = {
    "backend": "local",
    "model": SERVED_MODEL,
    "endpoint": BASE_URL + "/v1/chat/completions",
    "metrics_url": BASE_URL + "/metrics",
    "model_revision": MODEL_REVISION,
    "quantization": "BF16",
    "engine_version": ENGINE_VERSION,
    "cache_policy": CACHE_POLICY,
    "streams": [12],
    "modes": ["events", "compact", "parallel"],
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
    "parallel_scheduling": "paired",
    "grounding_format": "lean-v1",
    "monitor_layout": "single-video",
    "record": True,
    "admission_storage": "memory",
    "admission_diagnostics": True,
    "provider_only": "",
    "provider_prefer": "",
    "vision_profile": "",
    "cpu_offload": False,
    "self_hosted_qwen35": True,
    "grounding_concurrency_sweep": [],
    "parallel_concurrency_sweep": [],
    "measurement_start_index": 0,
}


def policy_check():
    from tools.benchmark_runtime.vlm_event_policy import event_policy_snapshot
    from tools.vlm_multistream_benchmark import LABELS

    policy = event_policy_snapshot(LABELS)
    if policy["sha256"] != POLICY_SHA256:
        raise ValueError("six-event policy changed; do not run this baseline")
    return policy


def validate(args):
    changed = [key for key, value in FROZEN.items() if getattr(args, key) != value]
    if changed:
        raise ValueError("self-hosted contract changed: " + ", ".join(changed))
    policy_check()
