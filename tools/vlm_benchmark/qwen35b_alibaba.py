"""Historical tracked Qwen comparison configuration; no execution entry point."""

MODEL = "qwen/qwen3.5-35b-a3b"


def fixed_arguments(*, allow_provider_fallback=False):
    # No model, provider, layout or load overrides in this entry point.
    return [
        "--model",
        MODEL,
        "--provider-prefer" if allow_provider_fallback else "--provider-only",
        "alibaba",
        "--modes",
        "events",
        "grounding",
        "parallel",
        "--streams",
        "12",
        "--grounding-format",
        "lean-v1",
        "--duration",
        "55.798021",
        "--repeats",
        "2",
        "--warmup-seconds",
        "10",
        "--soak-seconds",
        "0",
        "--global-limit",
        "48",
        "--per-stream-limit",
        "4",
        "--request-rate",
        "40",
        "--http-timeout",
        "20",
        "--arrival-mode",
        "staggered",
        "--record",
        "--monitor-layout",
        "single-video",
    ]
