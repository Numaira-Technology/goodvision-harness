"""Bounded events/compact/parallel comparison, against a running model endpoint.

Does not provision GPUs, download weights, or silently run paid API calls.
OpenRouter is the default backend; requests remain simulated unless --real is set.
Local GPU telemetry is meaningful only for an explicitly selected local deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ruff: noqa: E402
from PIL import Image

from goodvision_harness.config import load_config
from goodvision_harness.model_api import ModelAPISettings, environment_api_key
from goodvision_harness.vlm_compact_model import event_response_format
from goodvision_harness.vlm_independent import _inspector
from tools.vlm_benchmark.lock import acquire
from tools.vlm_benchmark.mode_report import render_modes
from tools.vlm_benchmark.recording import save_recording
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.telemetry import Telemetry
from tools.vlm_multistream_benchmark import (
    LABELS,
    decode,
    planned_observations,
    run_stage,
)

MODES = ("events", "compact", "parallel")
DEFAULT_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"


def checkout_revision():
    # Uploaded bundles have source hashes but need neither a .git directory nor git.
    if not shutil.which("git"):
        return ""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def mode_inspector(config):
    # Do not pass an events-only JSON Schema to the compact grounding lane.
    parameters = dict(config.parameters)
    parameters["response_format"] = (
        event_response_format(config.labels)
        if config.output_mode == "events"
        else {"type": "json_object"}
    )
    return _inspector(replace(config, parameters=parameters))


def validate(args):
    if not 0 <= args.monitor_port <= 65535 or (args.headed and not args.record):
        raise ValueError("invalid monitor port or --headed without --record")
    if not args.model.strip() or len(args.model) > 200:
        raise ValueError("invalid model name")
    for name, low, high in (
        ("duration", 3, 300),
        ("warmup_seconds", 3, 30),
        ("soak_seconds", 0, 1800),
        ("observation_fps", 0.25, 3),
        ("request_rate", 0.1, 200),
        ("queue_timeout", 0.01, 3),
        ("http_timeout", 1, 20),
        ("simulated_latency", 0, 20),
        ("simulated_grounding_latency", 0, 20),
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"invalid {name}")
    if not 1 <= args.repeats <= 3 or not 1 <= args.global_limit <= 128:
        raise ValueError("invalid repeats/global_limit")
    if not 1 <= args.per_stream_limit <= min(4, args.global_limit):
        raise ValueError("invalid shared per-stream concurrency")
    if not 1 <= args.local_concurrency <= 4:
        raise ValueError("invalid local concurrency")
    if (
        not args.streams
        or sorted(set(args.streams)) != args.streams
        or any(not 1 <= n <= 12 for n in args.streams)
    ):
        raise ValueError("streams must be an increasing unique subset of 1..12")
    if not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("select each mode at most once")
    if any(mode not in MODES for mode in args.modes):
        raise ValueError(
            "parallel already includes events; use events, compact, parallel"
        )
    if args.soak_seconds and (args.soak_seconds < 30 or 12 not in args.streams):
        raise ValueError("soak requires 12 streams and at least 30 seconds")
    for url in (args.endpoint, args.metrics_url):
        if url:
            parsed = urlsplit(url)
            # Accessing port also validates its syntax/range, before any requests.
            _ = parsed.port
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "endpoint URLs must be HTTP(S), without credentials or query"
                )
    if args.backend == "openrouter" and (args.endpoint or args.metrics_url):
        raise ValueError("external API mode cannot claim local inference metrics")
    endpoint = args.endpoint or "http://127.0.0.1:8000/v1/chat/completions"
    if args.metrics_url:

        def origin(u):
            return urlsplit(u).scheme, urlsplit(u).netloc

        if origin(endpoint) != origin(args.metrics_url):
            raise ValueError("metrics must share the model endpoint origin")
    nominal = (
        max(args.streams)
        * args.observation_fps
        * (2 if "parallel" in args.modes else 1)
    )
    if args.request_rate < nominal and not args.allow_rate_limited:
        raise ValueError(
            f"planned load {nominal:g}/s exceeds request cap {args.request_rate:g}/s; "
            "raise the cap or explicitly use --allow-rate-limited "
            "for a limiter experiment"
        )
    if args.real and not args.video:
        raise ValueError("real tests require --video")
    if (
        args.real
        and args.backend == "local"
        and not all(
            (
                args.model_revision,
                args.quantization,
                args.engine_version,
                args.cache_policy,
            )
        )
    ):
        raise ValueError(
            "record --model-revision, --quantization, "
            "--engine-version and --cache-policy"
        )


def jobs(args):
    for count in args.streams:
        for repetition in range(1, args.repeats + 1):
            order = args.modes if repetition % 2 else list(reversed(args.modes))
            for mode in order:
                yield {
                    "id": f"{count}-streams-{mode}-repeat-{repetition}",
                    "mode": mode,
                    "streams": count,
                    "repeat": repetition,
                    "phase": "measurement",
                    "duration": args.duration,
                }


def request_budget(args):
    lanes = sum(2 if mode == "parallel" else 1 for mode in args.modes)
    measurements = (
        sum(args.streams)
        * args.repeats
        * lanes
        * planned_observations(args.duration, args.observation_fps)
    )
    warmups = (
        (1 + len(args.streams) * args.repeats)
        * lanes
        * planned_observations(args.warmup_seconds, args.observation_fps)
    )
    soaks = (
        12 * lanes * planned_observations(args.soak_seconds, args.observation_fps)
        if args.soak_seconds
        else 0
    )
    return {
        "measurement_upper": measurements,
        "preflight_and_warmup_upper": warmups,
        "optional_soak_upper": soaks,
        "total_upper": measurements + warmups + soaks,
    }


def configure(args):
    config_path = ROOT / "config.yaml"
    if not config_path.exists():  # Portable bundle contains the packaged defaults.
        config_path = ROOT / "src/goodvision_harness/config.yaml"
    base = load_config(config_path)
    parameters = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if args.backend == "local"
        else {
            "reasoning": {"enabled": False},
            "provider": {
                "sort": "latency",
                "require_parameters": True,
                "allow_fallbacks": True,
                "ignore": ["coreweave", "novita"],
            },
        }
    )
    if args.backend == "openrouter" and args.model == DEFAULT_MODEL:
        # This Instruct endpoint does not advertise reasoning support. With
        # require_parameters=True even a disabled reasoning flag can filter routes.
        parameters.pop("reasoning", None)
    settings = ModelAPISettings(
        provider=args.backend,
        endpoint=args.endpoint or "",
        max_retries=0,
        timeout_seconds=args.http_timeout,
    )
    return replace(
        base,
        aiclub=settings,
        inference=replace(
            base.inference,
            model=args.model,
            request_parameters={**parameters, "temperature": 0},
        ),
        runtime=replace(
            base.runtime,
            scene_max_image_side=720,
            scene_jpeg_quality=90,
            scene_max_tokens=1600,
            scene_timeout_seconds=args.http_timeout,
        ),
    )


def usable(stage):
    return (
        stage["input_timing_valid"]
        and stage["all_drained"]
        and not stage["result_records_missing"]
        and all(row["complete_valid_results"] for row in stage["lanes"].values())
    )


def run(args):
    validate(args)
    # A second suite would contaminate both capacity and server counter deltas.
    # Simulation is offline and may run in isolated unit-test processes.
    lock = None
    if args.real:
        lock_root = ROOT / "outputs" / "vlm-modes" / "run"
        lock_root.parent.parent.mkdir(parents=True, exist_ok=True)
        lock = acquire(lock_root)
    try:
        return _run(args)
    finally:
        if lock is not None:
            os.close(lock)


def _run(args):
    base = configure(args)
    if (
        args.real
        and args.backend != "local"
        and not environment_api_key(base.model_api)
    ):
        raise ValueError("configured API credential is unavailable")
    output = args.output or ROOT / "outputs/vlm-modes" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        k: [str(x) for x in v]
        if isinstance(v, list) and k == "video"
        else str(v)
        if isinstance(v, Path)
        else v
        for k, v in vars(args).items()
    }
    document = {
        "state": "preparing",
        "settings": settings,
        "event_labels": list(LABELS),
        "jobs": [],
        "request_budget": request_budget(args),
    }
    render_modes(output, document)
    monitor = None
    resources = ExitStack()

    def sha(p):
        digest = hashlib.sha256()
        with Path(p).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def trial(job):
        location = output / job["id"]
        location.mkdir()
        document["active_stage"] = job["id"]
        render_modes(output, document)
        options = SimpleNamespace(**vars(args))
        options.duration, options.output_mode, options.deadline = (
            job["duration"],
            job["mode"],
            3,
        )
        telemetry_file = location / "telemetry.jsonl"
        recording = None
        monitor_begun = False
        try:
            if monitor:
                monitor.begin(job, location)
                monitor_begun = True
            with Telemetry(
                telemetry_file,
                args.metrics_url if args.real else None,
                environment_api_key(base.model_api) if args.real else "",
            ) as telemetry:
                result = run_stage(
                    options,
                    base,
                    job["streams"],
                    framesets,
                    location,
                    job["repeat"],
                    inspector_factory=mode_inspector,
                    **({"observer": monitor} if monitor else {}),
                )
            if monitor:
                recording = monitor.finish(result)
        except BaseException as exc:
            if monitor_begun:
                recording = monitor.finish(error=exc)
            if args.record and job["phase"] in {"measurement", "soak"}:
                # Keep interrupted captures discoverable without inventing a
                # completed benchmark result for this partial stage.
                recording = {
                    **(recording or {}),
                    "state": "interrupted",
                    "complete": False,
                    "benchmark_completed": False,
                    "error_type": type(exc).__name__,
                }
                folder = location / "recording"
                folder.mkdir(exist_ok=True)
                save_recording(folder / "recording.json", recording)
            document.setdefault("interrupted_jobs", []).append(
                {**job, "recording": recording, "error_type": type(exc).__name__}
            )
            raise
        saved = {
            **job,
            "result": result,
            "telemetry": telemetry.summary(),
            "raw_reports": str(
                (
                    location
                    / f"streams-{job['streams']}-repeat-{job['repeat']}"
                    / "reports.jsonl"
                ).relative_to(output)
            ),
            "telemetry_file": str(telemetry_file.relative_to(output)),
            "recording": recording,
        }
        document["jobs"].append(saved)
        document.pop("active_stage", None)
        render_modes(output, document)
        if recording and not recording.get("complete"):
            raise RuntimeError(
                "incomplete browser recording; stopped subsequent stages"
            )
        print(
            f"{job['phase']}: {job['mode']} / {job['streams']} streams "
            f"/ repeat {job['repeat']}",
            flush=True,
        )
        if any(r["http_status_counts"].get("429") for r in result["lanes"].values()):
            raise RuntimeError("HTTP 429; stopped subsequent stages")
        if (
            not result["all_drained"]
            or not result["input_timing_valid"]
            or result["result_records_missing"]
        ):
            raise RuntimeError(
                "invalid input/drain/retention; stopped subsequent stages"
            )
        return result

    try:
        # Replace inherited cloud defaults; no headers or keys are saved.
        revision = checkout_revision()
        manifest = {
            "settings": settings,
            "model": args.model,
            "git_revision": revision,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "event_labels": list(LABELS),
            "request_parameters": base.inference.request_parameters,
            "image_side": 720,
            "jpeg_quality": 90,
            "output_limits": {"events": 160, "compact": 1600},
            "http_timeouts": {
                "events": min(5, args.http_timeout),
                "compact": args.http_timeout,
            },
            "source_expiry_seconds": 3,
            "admission": (
                "per_stream_limit shared across lanes, local_concurrency per lane"
            ),
            "deployment_metadata": (
                "operator-declared; endpoint does not attest GPU/quantization"
            ),
            "source_hashes": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    *sorted((ROOT / "src/goodvision_harness").glob("*.py")),
                    *sorted((ROOT / "tools/vlm_benchmark").glob("*.py")),
                    *sorted((ROOT / "tools/vlm_benchmark").glob("monitor.*")),
                    ROOT / "tools/vlm_multistream_benchmark.py",
                ]
            },
            "videos": [{"path": str(p), "sha256": sha(p)} for p in args.video],
        }
        write_json(output / "manifest.json", manifest)

        framesets = [
            decode(p, max(2, args.observation_fps * 2), 720, 90) for p in args.video
        ]
        if not framesets:
            buffer = io.BytesIO()
            Image.new("RGB", (64, 48), "gray").save(buffer, "JPEG")
            framesets = [[buffer.getvalue()]]
        if args.monitor or args.record:
            from tools.vlm_benchmark.monitor import LiveMonitor

            monitor = resources.enter_context(LiveMonitor(output, framesets, args))
        document["state"] = "running"
        for mode in args.modes:
            probe = trial(
                {
                    "id": f"preflight-{mode}",
                    "mode": mode,
                    "streams": 1,
                    "repeat": 1,
                    "phase": "preflight",
                    "duration": args.warmup_seconds,
                }
            )
            if not usable(probe):
                raise RuntimeError(
                    f"{mode} preflight: incomplete result or invalid input/drain"
                )
        for job in jobs(args):
            warm = trial(
                {
                    **job,
                    "id": job["id"] + "-warmup",
                    "streams": 1,
                    "phase": "warmup",
                    "duration": args.warmup_seconds,
                }
            )
            if not usable(warm):
                raise RuntimeError("warmup unavailable; stopped subsequent stages")
            trial(job)
        if args.soak_seconds:
            for mode in args.modes:
                trials = [
                    j["result"]
                    for j in document["jobs"]
                    if j["phase"] == "measurement"
                    and j["streams"] == 12
                    and j["mode"] == mode
                ]
                # Explicit completeness screen, not an unstated latency SLA.
                if len(trials) != args.repeats or not all(
                    r["complete_result_fraction"] >= 0.99
                    for t in trials
                    for camera in t["per_stream"]
                    for r in camera["lanes"].values()
                ):
                    document.setdefault("skipped_soaks", []).append(mode)
                    continue
                trial(
                    {
                        "id": f"soak-{mode}",
                        "mode": mode,
                        "streams": 12,
                        "repeat": 1,
                        "phase": "soak",
                        "duration": args.soak_seconds,
                    }
                )
        document["state"] = "completed"
        document["reason"] = (
            "Measurements completed; inspect each lane and missing windows. "
            "Not an accuracy or business-SLA acceptance."
        )
    except (Exception, KeyboardInterrupt) as exc:
        document["state"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        document["reason"] = (
            str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        )
        render_modes(output, document)
        raise
    finally:
        resources.close()
    render_modes(output, document)
    print(f"Report: {output / 'report.html'}", flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--backend", choices=["openrouter", "local"], default="openrouter")
    p.add_argument(
        "--model", default=DEFAULT_MODEL, help="exact served model name/alias"
    )
    p.add_argument("--endpoint", default="")
    p.add_argument("--metrics-url", default="")
    p.add_argument("--model-revision", default="")
    p.add_argument("--quantization", default="")
    p.add_argument("--engine-version", default="")
    p.add_argument("--cache-policy", default="")
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="record this as a different deployment",
    )
    p.add_argument("--video", type=Path, action="append", default=[])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--streams", nargs="+", type=int, default=[1, 4, 8, 12])
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-seconds", type=float, default=3)
    p.add_argument("--soak-seconds", type=int, default=0)
    p.add_argument("--observation-fps", type=float, default=1.25)
    p.add_argument("--global-limit", type=int, default=16)
    p.add_argument("--per-stream-limit", type=int, default=4)
    p.add_argument("--local-concurrency", type=int, default=4)
    p.add_argument("--request-rate", type=float, default=40)
    p.add_argument("--allow-rate-limited", action="store_true")
    p.add_argument("--queue-timeout", type=float, default=1)
    p.add_argument("--http-timeout", type=float, default=15)
    p.add_argument(
        "--arrival-mode", choices=["aligned", "staggered"], default="staggered"
    )
    p.add_argument("--simulated-latency", type=float, default=0.1)
    p.add_argument("--simulated-grounding-latency", type=float, default=0.3)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--monitor", action="store_true", help="serve a loopback live monitor"
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="record each measurement/soak browser viewport (implies --monitor)",
    )
    p.add_argument(
        "--headed", action="store_true", help="show the recording Chromium window"
    )
    p.add_argument(
        "--monitor-port",
        type=int,
        default=0,
        help="loopback port; 0 selects a free port",
    )
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        validate(args)
    except ValueError as exc:
        p.error(str(exc))
    print(
        json.dumps(
            {
                "request_budget": request_budget(args),
                "real_model": args.real,
                "backend": args.backend,
                "model": args.model,
                "modes": args.modes,
                "streams": args.streams,
            }
        ),
        flush=True,
    )
    if not args.plan_only:
        run(args)


if __name__ == "__main__":
    main()
