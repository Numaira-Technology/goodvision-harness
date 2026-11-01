# ruff: noqa: RUF001
"""48GB 4090 FP8 + YOLO26-S. Offline by default; never rents an instance."""

import copy
import json
from pathlib import Path

from tools.benchmark_runtime.person_grounding import TrackedPersonInspector
from tools.tracked_grounding import suite
from tools.vlm_benchmark import modes

from . import benchmark, runtime
from . import fp8_profile as p


class LocalTracked(benchmark.LocalTransport, TrackedPersonInspector):
    build_messages = TrackedPersonInspector.build_messages


def inspector(config, *, broker_url):
    return benchmark.inspector(
        config,
        grounding_class=LocalTracked,
        extra={"broker_url": broker_url} if config.output_mode != "events" else None,
    )


def arguments(*, video=None, real=False):
    args = modes.parser().parse_args([])
    for key, value in p.FROZEN.items():
        setattr(args, key, copy.deepcopy(value))
    args.video = [Path(video)] if video else []
    args.real = real
    modes.validate(args)
    p.policy_check()
    return args


def request_budget(args):
    jobs = list(modes.jobs(args))

    def calls(duration, streams, mode):
        return (
            modes.planned_observations(duration, args.observation_fps)
            * streams
            * (2 if mode == "parallel" else 1)
        )

    formal = sum(calls(j["duration"], j["streams"], j["mode"]) for j in jobs)
    warmup = sum(calls(10, j["streams"], j["mode"]) for j in jobs)
    preflight = 2 * calls(2.4, 1, "events")
    return {
        "measurement_upper": formal,
        "warmup_upper": warmup,
        "preflight_upper": preflight,
        "total_upper": formal + warmup + preflight,
    }


def plan():
    args = arguments()
    return {
        **suite.plan(),
        "version": p.VERSION,
        "model": p.MODEL_REPO,
        "provider": "local vLLM; no cloud fallback",
        "revision": p.MODEL_REVISION,
        "image": p.IMAGE,
        "quantization": "FP8",
        "dtype": "bfloat16 (unquantized components)",
        "transport": "sse-v1",
        "report_title": "YOLO26-S 人物 ID + Qwen3.5 35B",
        "http_timeouts": {"events": 20, "grounding": 20},
        "hardware": "one modified 48GB RTX 4090 shared with YOLO26-S; unvalidated",
        "server_sequences": p.SERVER_SEQUENCES,
        "gpu_memory_utilization": p.GPU_UTILIZATION,
        "yolo_weights_sha256": p.YOLO_SHA256,
        "metrics_url": args.metrics_url,
        "request_budget": request_budget(args),
        "extra_calls": "two 2.4s single-camera preflights plus 10s warmup per round",
        "comparison_notes": [
            "本地官方 FP8 checkpoint；历史云 35B 的实际精度和硬件未知。",
            "本轮 events/grounding 调用超时均 20 秒；历史 YOLO 云 events 为 5 秒。",
            "六类标准、人物协议、原始证据、十二路负载不变；三秒为评分期限。",
            "API TPS 含等待；流式估计及 vLLM 聚合耗时都不是逐请求纯 GPU 时间。",
        ],
        "readiness": (
            "GPU smoke test and capacity validation required; no 3-second guarantee"
        ),
    }


def run(video, output, packet, volume, *, preflight_only=False, person_display="ids"):
    from . import fp8_adapter as adapter

    if Path(output).exists():
        raise ValueError("choose a new output directory")
    active = runtime.verify_active(volume, p=p)
    vlm = json.loads(Path(active["deployment"]).read_text())
    if vlm.get("experiment") is not None:
        raise ValueError("use fp8_sweep with the matching server experiment")
    from .yolo_runtime import verify_active

    detector = verify_active(volume)
    benchmark.readiness()
    deployment = {
        "vlm": vlm,
        "detector": detector,
    }
    try:
        return suite.run(
            video,
            output,
            p.DETECTOR_URL,
            packet,
            model_adapter=adapter,
            preflight_only=preflight_only,
            person_display=person_display,
        )
    finally:
        if Path(output).is_dir():
            runtime.write_json(Path(output) / "deployment.json", deployment)


def stage_metrics(records, stage):
    from .metrics import token_metrics

    return token_metrics(
        records, stage["input_schedule_span_seconds"] + stage["drain_elapsed_seconds"]
    )
