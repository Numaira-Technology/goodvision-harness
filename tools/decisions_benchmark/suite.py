"""Reuse the experimental twelve-camera runner, source sampling and recordings."""

import copy
import json
import os
from pathlib import Path

from tools.vlm_benchmark import modes
from tools.vlm_benchmark.model_comparison import load_reference, sha
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.routes import read_api_key

from . import profile as p


def arguments(video=None, output=None, *, real=False):
    args = modes.parser().parse_args([])
    for key, value in p.FROZEN.items():
        setattr(args, key, copy.deepcopy(value))
    args.video = [Path(video)] if video else []
    args.output = Path(output) if output else None
    args.real = real
    modes.validate(args)
    return args


def plan():
    args = arguments()
    return {
        "profile": p.VERSION,
        "model": p.MODEL,
        "endpoint": p.ENDPOINT,
        "frozen": p.FROZEN,
        "event_policy": p.check_policy(),
        "source_video_sha256": p.VIDEO_SHA256,
        "execute": False,
        "request_budget": modes.request_budget(args),
        "recordings": 2,
        "preflight_requests": 3,
        "automatic_retries": 0,
        "input_layout": "original separate frames; rolling context up to four",
        "output": "six positive/negative/uncertain choices; no boxes or person IDs",
        "deadline": "3 seconds; each camera needs 66/69 timely complete windows",
        "cost": {
            "input_usd_per_million": p.BASE_INPUT_PRICE_PER_MILLION,
            "price_checked": "2026-10-17",
            "billed_cost_verified": False,
        },
        "reference": "Astra matched by exact evidence; not human-confirmed accuracy",
        "tps": "No output TPS; record input tokens/s and complete windows/s",
    }


def credential(env_file=None):
    key = read_api_key("openai_decisions", env_file, required=True)
    # Only this one selected credential is inherited by spawned benchmark workers.
    os.environ["OPENAI_API_KEY"] = key
    return key


def check_video(video):
    if sha(video) != p.VIDEO_SHA256:
        raise ValueError("video differs from the original output.mp4")


def preflight(video, output, *, env_file=None):
    """Three visual calls only; stops at the first rejected/incomplete answer."""
    from goodvision_harness.runtime.scene import TemporalInspection
    from goodvision_harness.runtime.scheduling import EvidenceFrame, InspectionRequest
    from tools.vlm_benchmark.lock import acquire
    from tools.vlm_multistream_benchmark import decode

    from .adapter import DecisionInspector

    p.check_policy()
    check_video(video)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    key = credential(env_file)
    frames = decode(Path(video), 2.5, 720, 90)
    lock_root = modes.ROOT / "outputs" / "vlm-modes" / "run"
    lock_root.parent.parent.mkdir(parents=True, exist_ok=True)
    lock = acquire(lock_root)
    client = None
    results = []
    try:
        client = DecisionInspector(api_key=key, audit_dir=output / "api-audit")
        for number, start in enumerate((18, 66, 114), 1):
            evidence = tuple(
                EvidenceFrame(
                    "decisions-preflight", i, frames.media_seconds[i], frames[i]
                )
                for i in range(start, start + 4)
            )
            request = InspectionRequest(
                f"decisions-preflight-{number}",
                (),
                "independent_source",
                p.LABELS,
                evidence[0].timestamp_seconds,
                evidence[-1].timestamp_seconds,
            )
            job = TemporalInspection(
                "decisions-preflight", "camera01", request, evidence
            )
            verdicts = client.inspect(job)
            results.append(
                {
                    "case": number,
                    "complete": len(verdicts) == 6,
                    "http_timing": client.last_http_timing,
                }
            )
        return {"state": "passed_preflight", "requests": results}
    finally:
        cleaned = False
        try:
            if client is not None:
                client.close()
            cleaned = True
        finally:
            os.close(lock)
            write_json(
                output / "preflight.json",
                {
                    "profile": p.VERSION,
                    "passed": cleaned and len(results) == 3,
                    "requests": results,
                    "formal_test_started": False,
                },
            )


def run(video, output, packet, *, env_file=None):
    from tools.vlm_benchmark.admission_storage import memory_root

    from .report import analyze

    check_video(video)
    memory_root()  # Same tmpfs admission as the latest cloud comparison.
    reference = load_reference(packet, video)
    if reference["policy"]["sha256"] != p.POLICY_SHA256:
        raise ValueError("Astra reference policy differs")
    output = Path(output)
    if output.exists():
        raise ValueError("choose a new result directory")
    credential(env_file)
    args = arguments(video, output, real=True)
    try:
        # Shared runner includes a one-camera preflight and warmups under its lock.
        modes.run(args)
    finally:
        if output.exists():
            write_json(output / "decisions-plan.json", plan())
    return analyze(output, reference)


def check_stage(job, result, location):
    statuses = {
        str(k)
        for lane in result["lanes"].values()
        for k, value in lane["http_status_counts"].items()
        if value
    }
    if statuses & {"401", "402", "403"}:
        raise RuntimeError(
            "Decisions credentials/billing/access rejected; stop remaining trials"
        )
    for path in (location / "api-audit").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if json.loads(line).get("error_code") in {
                "insufficient_quota",
                "model_not_found",
            }:
                raise RuntimeError(
                    "Decisions quota/model unavailable; stop remaining trials"
                )
