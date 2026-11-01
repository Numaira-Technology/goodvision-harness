"""Frozen twelve-camera suite. Planning is offline; execution is explicit."""

import argparse
import copy
import json
import os
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from tools.benchmark_runtime.vlm_event_policy import event_policy_snapshot
from tools.vlm_benchmark import modes
from tools.vlm_benchmark.lock import acquire
from tools.vlm_benchmark.model_comparison import load_reference, sha
from tools.vlm_benchmark.qwen35b_alibaba import fixed_arguments
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.routes import read_api_key
from tools.vlm_benchmark.telemetry import Telemetry
from tools.vlm_multistream_benchmark import (
    LABELS,
    decode,
    planned_observations,
    run_stage,
)

from .detector import DetectorBroker
from .inspector import inspector
from .monitor import PersonMonitor

POLICY_SHA = "d9c487c37404df2f6d1f4d1f3da39559abffe28f59c76e7fef85f4a9f9e48a7b"
VIDEO_SHA = "2ac7a0f3271c4c9937b1796bdc43526617d9b1715de9bdf15713d8d3d80dacb9"


def arguments(*, real=False, baseline=False, video=None, load_profile="historical"):
    args = modes.parser().parse_args(fixed_arguments(allow_provider_fallback=True))
    args.real = real
    args.video = [Path(video)] if video else []
    args.admission_storage = "memory"
    args.admission_diagnostics = True
    args.person_ids_only = not baseline
    args.backend_person_tracks = not baseline
    args.monitor = True
    args.parallel_scheduling = "paired"
    if load_profile not in {"historical", "four-second"}:
        raise ValueError("invalid tracked load profile")
    if load_profile == "four-second":
        args.observation_fps = 0.25
        args.global_limit = 4
        # Keep the evidence and detector load unchanged when reducing VLM calls.
        args.source_sample_fps = 2.5
    modes.validate(args)
    if event_policy_snapshot(LABELS)["sha256"] != POLICY_SHA:
        raise ValueError("six-event policy changed")
    return args


def plan(*, baseline=False, load_profile="historical"):
    args = arguments(baseline=baseline, load_profile=load_profile)
    windows = 12 * planned_observations(args.duration, args.observation_fps, 2.5)
    return {
        "version": "yolo26s-qwen35-tracked-v3-isolated",
        "integration": "main-grounding-isolated-benchmarks-v3",
        "load_profile": load_profile,
        "grounding_source": "vlm" if baseline else "detector_ids",
        "person_display": "off" if baseline else "ids",
        "execute": False,
        "variant": "qwen-own-grounding" if baseline else "detector-person-judgments",
        "model": args.model,
        "provider": "OpenRouter; Alibaba preferred; fallback allowed",
        "detector": None if baseline else "yolo26-s; existing /v1/tracks IoU tracker",
        "streams": 12,
        "modes": ["events", "grounding", "parallel"],
        "repeats": 2,
        "duration_seconds": args.duration,
        "sample_fps": 2.5,
        "observation_fps": args.observation_fps,
        "frames_per_window": "2-4 original full-scene frames",
        "planned_windows_per_branch_per_round": windows,
        "formal_vlm_requests": windows * 8,
        "extra_calls": (
            "one 2.4s single-camera grounding preflight; 10s warmup before each round"
        ),
        "detector_shared": (
            "once per source frame per camera, reused by overlapping windows"
        ),
        "detector_sampling_fps": 2.5,
        "global_limit": args.global_limit,
        "per_stream_limit": 4,
        "parallel_scheduling": "paired",
        "arrival_mode": "staggered",
        "http_timeouts": {"events": 5, "grounding": 20},
        "detector_timeout_seconds": 2,
        "detector_queue_capacity": 48,
        "deadline_seconds": 3,
        "target_per_camera": 0.95,
        "latency_scope": (
            "nonstream model API send through complete response body; "
            "excludes local admission, detector, request construction and parsing"
        ),
        "diagnostic_latency_scope": (
            "source ingress before detector enqueue through final publication"
        ),
        "preview": (
            "15 FPS single video; no overlays or evidence pane"
            if baseline
            else (
                "15 FPS single video; person_display controls annotations "
                "(default: IDs only); no evidence pane"
            )
        ),
        "event_policy": event_policy_snapshot(LABELS),
        "source_sha256": VIDEO_SHA,
        "quality_scope": (
            "Astra scene judgments only; person association needs separate annotation"
        ),
        "automatic_retries": 0,
    }


def run(
    video,
    output,
    detector_url,
    packet,
    *,
    env_file=None,
    baseline=False,
    model_adapter=None,
    preflight_only=False,
    person_display="ids",
    load_profile="historical",
):
    from .report import analyze

    video, output = Path(video), Path(output)
    if model_adapter is not None and baseline:
        raise ValueError("local tracked adapter cannot run the cloud baseline")
    if model_adapter is not None and load_profile != "historical":
        raise ValueError("load profile override is only supported for cloud Qwen")
    args = (
        model_adapter.arguments(video=video, real=True)
        if model_adapter
        else arguments(
            real=True, baseline=baseline, video=video, load_profile=load_profile
        )
    )
    run_plan = (
        model_adapter.plan()
        if model_adapter
        else plan(baseline=baseline, load_profile=load_profile)
    )
    if person_display not in {"ids", "boxes", "off"}:
        raise ValueError("invalid person display")
    args.person_display = "off" if baseline else person_display
    run_plan["person_display"] = args.person_display
    if sha(video) != VIDEO_SHA:
        raise ValueError("use the original output.mp4")
    reference = load_reference(packet, video)
    if reference["policy"]["sha256"] != POLICY_SHA:
        raise ValueError("reference policy mismatch")
    if output.exists():
        raise ValueError("choose a new output directory")
    if not baseline and not detector_url:
        raise ValueError("detector URL required")
    from tools.vlm_benchmark.admission_storage import memory_root

    memory_root()  # Fail before paid requests on hosts without the agreed tmpfs.
    detector_ready = (
        None if baseline else DetectorBroker(detector_url, output).check_ready()
    )
    if model_adapter is None:
        os.environ["OPENROUTER_API_KEY"] = read_api_key(
            "openrouter", env_file, required=True
        )
    elif hasattr(model_adapter, "credential"):
        model_adapter.credential(env_file)
    base = (
        model_adapter.configure(args)
        if model_adapter and hasattr(model_adapter, "configure")
        else modes.configure(args)
    )
    framesets = [decode(video, 2.5, 720, 90)]
    output.mkdir(parents=True)
    document = {
        **run_plan,
        "execute": True,
        "state": "running",
        "jobs": [],
    }
    from tools.vlm_benchmark.provenance import source_identity

    write_json(
        output / "manifest.json",
        {
            **run_plan,
            "execute": True,
            "preflight_only": preflight_only,
            "git_revision": modes.checkout_revision(),
            "source_identity": source_identity(modes.ROOT),
            "detector_readiness": detector_ready,
            "source_hashes": {
                str(path.relative_to(modes.ROOT)): sha(path)
                for path in (
                    *sorted((modes.ROOT / "tools/tracked_grounding").glob("*.py")),
                    *sorted((modes.ROOT / "tools/decisions_benchmark").glob("*.py")),
                    *sorted((modes.ROOT / "tools/tracked_decisions").glob("*.py")),
                    *sorted((modes.ROOT / "tools/vast_vlm").glob("*")),
                    *sorted((modes.ROOT / "tools/videojev_probe").glob("*.py")),
                    *sorted((modes.ROOT / "tools/videojev_benchmark").glob("*.py")),
                    *sorted((modes.ROOT / "tools/vlm_benchmark").glob("monitor.*")),
                    modes.ROOT / "tools/vlm_benchmark/source_feeder.py",
                    modes.ROOT / "tools/vlm_multistream_benchmark.py",
                    *sorted((modes.ROOT / "src/goodvision_harness").rglob("*.py")),
                    *sorted((modes.ROOT / "tools/benchmark_runtime").glob("*.py")),
                    *sorted((modes.ROOT / "tools/vlm_benchmark").glob("*.py")),
                )
                if path.is_file()
            },
            "settings": {
                k: v for k, v in vars(args).items() if k not in {"output", "video"}
            },
            "request_parameters": base.inference.request_parameters,
            "decoded_evidence": reference["decoded_evidence"],
        },
    )
    lock_root = modes.ROOT / "outputs" / "vlm-modes" / "run"
    lock_root.parent.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire(lock_root)
    try:
        with PersonMonitor(output, framesets, args) as monitor:

            def trial(job):
                location = output / job["id"]
                location.mkdir()
                options = SimpleNamespace(**copy.deepcopy(vars(args)))
                options.duration, options.output_mode, options.deadline = (
                    job["duration"],
                    job["mode"],
                    3,
                )
                with ExitStack() as resources:
                    broker = (
                        None
                        if baseline
                        else resources.enter_context(
                            DetectorBroker(detector_url, location)
                        )
                    )
                    monitor.broker = broker
                    options.grounding_source = "vlm" if baseline else "detector_ids"
                    options.detector_broker_url = broker.endpoint if broker else None
                    factory = (
                        modes.mode_inspector
                        if baseline
                        else partial(
                            model_adapter.inspector if model_adapter else inspector,
                            broker_url=broker.endpoint,
                        )
                    )
                    if model_adapter and hasattr(model_adapter, "factory_for_job"):
                        factory = model_adapter.factory_for_job(broker.endpoint, job)
                    elif model_adapter and hasattr(model_adapter, "factory"):
                        factory = model_adapter.factory(broker.endpoint, location)
                    telemetry = resources.enter_context(
                        Telemetry(
                            location / "telemetry.jsonl",
                            metrics_url=getattr(args, "metrics_url", None),
                        )
                    )
                    monitor.begin(job, location)
                    try:
                        result = run_stage(
                            options,
                            base,
                            job["streams"],
                            framesets,
                            location,
                            job["repeat"],
                            inspector_factory=factory,
                            observer=monitor,
                            source_hook=broker.submit if broker else None,
                        )
                        detector_counts = broker.drain() if broker else None
                        recording = monitor.finish(result)
                    except BaseException as exc:
                        monitor.finish(error=exc)
                        raise
                    entry = {
                        **job,
                        "result": result,
                        "recording": recording,
                        "detector_counts": detector_counts,
                        "raw_reports": str(
                            (
                                location
                                / f"streams-{job['streams']}-repeat-{job['repeat']}"
                                / "reports.jsonl"
                            ).relative_to(output)
                        ),
                    }
                    if model_adapter and hasattr(model_adapter, "stage_metrics"):
                        # Close telemetry first so its final boundary is included.
                        resources.close()
                        entry["telemetry"] = telemetry.summary()
                    document["jobs"].append(entry)
                    write_json(output / "summary.json", document)
                    if (
                        not result["all_drained"]
                        or not result["input_timing_valid"]
                        or result["result_records_missing"]
                        or result["result_retention"].get("valid") is not True
                        or (
                            job["phase"] == "measurement"
                            and not (recording or {}).get("complete")
                        )
                    ):
                        raise RuntimeError("input/delivery/recording integrity failed")
                    if job["phase"] == "preflight" and not all(
                        lane["complete_valid_results"] > 0
                        for lane in result["lanes"].values()
                    ):
                        raise RuntimeError("no valid preflight result for every branch")
                    if any(
                        lane["http_status_counts"].get("402")
                        for lane in result["lanes"].values()
                    ):
                        raise RuntimeError("billing rejection; stop remaining trials")
                    if model_adapter and hasattr(model_adapter, "check_stage"):
                        model_adapter.check_stage(job, result, location)
                    return result

            trial(
                {
                    "id": "preflight",
                    "mode": "compact",
                    "streams": 1,
                    "repeat": 1,
                    "duration": 2.4,
                    "phase": "preflight",
                }
            )
            if model_adapter:
                trial(
                    {
                        "id": "preflight-events",
                        "mode": "events",
                        "streams": 1,
                        "repeat": 1,
                        "duration": 2.4,
                        "phase": "preflight",
                    }
                )
            for job in [] if preflight_only else modes.jobs(args):
                time.sleep(60)  # Same load, explicit cooldown; no window retries.
                trial(
                    {
                        **job,
                        "id": job["id"] + "-warmup",
                        "duration": 10,
                        "phase": "warmup",
                    }
                )
                trial(job)
        document["state"] = "completed"
    except BaseException:
        document["state"] = "interrupted"
        raise
    finally:
        os.close(fd)
        write_json(output / "summary.json", document)
        analyze(output, reference)
        if model_adapter is None:
            from .api_report import analyze as analyze_api

            analyze_api(output)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="original Qwen-owned grounding on the same load",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-url")
    parser.add_argument(
        "--load-profile",
        choices=["historical", "four-second"],
        default="historical",
        help="four-second: 0.25 windows/s/camera, four shared VLM slots; same evidence",
    )
    parser.add_argument(
        "--person-display", choices=["ids", "boxes", "off"], default="ids"
    )
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--env-file", type=Path, default=modes.ROOT / ".env")
    args = parser.parse_args(argv)
    if not args.real:
        preview = plan(baseline=args.baseline, load_profile=args.load_profile)
        preview["person_display"] = "off" if args.baseline else args.person_display
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if not all((args.video, args.output, args.packet)) or not (
        args.baseline or args.detector_url
    ):
        parser.error(
            "--real requires --video --output --packet and --detector-url "
            "(unless --baseline)"
        )
    run(
        args.video,
        args.output,
        args.detector_url,
        args.packet,
        env_file=args.env_file,
        baseline=args.baseline,
        person_display=args.person_display,
        load_profile=args.load_profile,
    )
    return 0
