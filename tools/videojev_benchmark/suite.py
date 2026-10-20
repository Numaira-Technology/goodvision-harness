# ruff: noqa: RUF001
"""Explicit six-round native-model experiment; never rents GPUs."""

import argparse
import copy
import json
from pathlib import Path

from tools.tracked_grounding import suite as tracked
from tools.vast_vlm.profile import FROZEN
from tools.videojev_probe.contract import MODEL, REVISION
from tools.vlm_benchmark import modes

from .adapter import ENDPOINT
from .adapter import factory as factory
from .adapter import inspector as inspector


def arguments(*, video=None, real=False):
    args = modes.parser().parse_args([])
    for k, v in FROZEN.items():
        setattr(args, k, copy.deepcopy(v))
    for k, v in dict(
        self_hosted_qwen35=False,
        videojev_native=True,
        model=MODEL,
        model_revision=REVISION,
        endpoint=ENDPOINT,
        metrics_url="",
        quantization="none",
        engine_version="transformers-5.14.1-native",
        cache_policy="no cross-window evidence cache; shared kernel cache; warmup 10s",
        person_ids_only=True,
        backend_person_tracks=True,
        monitor=True,
    ).items():
        setattr(args, k, v)
    args.real = real
    args.video = [Path(video)] if video else []
    modes.validate(args)
    return args


def plan():
    return {
        **tracked.plan(),
        "version": "videojev4b-focused-twelve-v1",
        "model": MODEL,
        "revision": REVISION,
        "provider": "local native Transformers; no cloud fallback",
        "report_title": "YOLO26-S + VideoJev-4B 精简 PPE",
        "http_timeouts": {"events": 20, "grounding": 20},
        "server": (
            "one serialized GPU worker; bounded FIFO"
            " 48; skip expired requests; no batching"
        ),
        "ppe_variant": "focused",
        "question_protocol": (
            "6 scene/unassigned + 5 per last-frame person; native choice scoring"
        ),
        "hardware": "one RTX 4090 24GB shared with live YOLO26-S",
        "comparison_notes": [
            "VideoJev 原生单 GPU 串行执行，48 个客户端并发槽不等于 GPU 批处理；"
            "这是当前服务实现的容量测试。",
            "实时 YOLO；只显示人物 ID；六类标准及原负载不变。",
            "PPE 使用精简版。grounding 的场景问题覆盖未分配人物，"
            "和人物问题按原协议合并。",
            "请求超时20秒，三秒为评分期限；Astra仅评场景，不评价人物ID绑定。",
            "不输出生成 token，因此不报告生成 TPS。",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("video", "packet", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--detector-url", default="http://127.0.0.1:8092")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return
    if not all((args.video, args.packet, args.output)):
        p.error("video packet output required")
    from . import suite

    tracked.run(
        args.video,
        args.output,
        args.detector_url,
        args.packet,
        model_adapter=suite,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()
