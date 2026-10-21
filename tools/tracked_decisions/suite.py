"""Twelve-camera tracked Decisions runner; no credentials or requests in plans."""

import copy
from functools import partial
from pathlib import Path

from tools.decisions_benchmark import profile as decisions
from tools.decisions_benchmark.suite import check_stage as check_stage
from tools.tracked_grounding import suite as tracked
from tools.vlm_benchmark import modes

from . import profile as p
from .adapter import inspector


def credential(env_file=None):
    from tools.decisions_benchmark.suite import credential as read_openai_key

    return read_openai_key(env_file)


def arguments(*, video=None, real=False):
    args = modes.parser().parse_args([])
    for key, value in p.FROZEN.items():
        setattr(args, key, copy.deepcopy(value))
    args.real = real
    args.video = [Path(video)] if video else []
    modes.validate(args)
    return args


def plan():
    args = arguments()
    jobs = list(modes.jobs(args))
    warmup = sum(
        j["streams"]
        * (2 if j["mode"] == "parallel" else 1)
        * modes.planned_observations(10, 1.25)
        for j in jobs
    )
    preflight = 2 * modes.planned_observations(2.4, 1.25)
    return {
        **tracked.plan(),
        "version": p.VERSION,
        "model": decisions.MODEL,
        "report_title": "YOLO26-S 人物 ID + Decisions API",
        "provider": "OpenAI direct; no fallback",
        "endpoint": decisions.ENDPOINT,
        "http_timeouts": {"events": 20, "grounding": 20},
        "question_protocol": (
            "positive/negative/uncertain; 5 per person + 6 scene/unassigned"
        ),
        "max_last_frame_people": p.MAX_PEOPLE,
        "question_limit_behavior": (
            "record local failure; never truncate people or split calls"
        ),
        "extra_calls": (
            "two single-camera 2.4s preflights plus 10s twelve-camera warmup per round"
        ),
        "request_budget": {
            "formal": 6624,
            "warmup": warmup,
            "preflight": preflight,
            "total_upper": 6624 + warmup + preflight,
        },
        "comparison_notes": [
            "Same YOLO flow/full-scene images/cadence/six-class policy "
            "as tracked Qwen.",
            "Historical tracked Qwen used events timeout 5s; Decisions uses 20s. "
            "Compare the same 3s denominator; eventual completion is not strictly "
            "matched. Longer in-flight calls may also affect admission.",
            "Decisions uses fixed choice questions; Qwen generated person JSON. "
            "Astra scores scene judgments, not person association.",
        ],
        "tps": (
            "record input tokens and complete windows/s; "
            "output decode TPS is not comparable"
        ),
        "execute": False,
    }


def factory(broker_url, location):
    return partial(inspector, broker_url=broker_url, audit_dir=location / "api-audit")


def run(video, output, detector_url, packet, *, env_file=None, preflight_only=False):
    from . import suite as adapter

    return tracked.run(
        video,
        output,
        detector_url,
        packet,
        env_file=env_file,
        model_adapter=adapter,
        preflight_only=preflight_only,
    )
