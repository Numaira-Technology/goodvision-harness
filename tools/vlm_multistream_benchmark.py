"""Real-time, per-stream capacity probe through the production VLM branch.

Simulation validates scheduling only. --real uses the configured model and
credentials; predecoded videos isolate VLM work from decoding and networking.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from goodvision_harness.config import load_config  # noqa: E402
from goodvision_harness.event_sensitivity import (  # noqa: E402
    EVENT_SENSITIVITY_DEFAULTS,
)
from goodvision_harness.grounding import SAFETY_EVENT_LABELS  # noqa: E402
from goodvision_harness.model_api import environment_api_key  # noqa: E402
from goodvision_harness.vlm_admission import (  # noqa: E402
    AdmissionSettings,
    SharedAdmission,
)
from goodvision_harness.vlm_independent import (  # noqa: E402
    IndependentVLMBranch,
    VLMProcessConfig,
    _inspector,
)
from goodvision_harness.vlm_independent_model import IndependentVerdict  # noqa: E402

# Follow the production safety catalog; stranger uses a separate identity lane.
LABELS = SAFETY_EVENT_LABELS


class SimulatedInspector:
    def __init__(self, config):
        self.seconds = config.parameters["simulation_seconds"]

    def inspect(self, job):
        time.sleep(self.seconds)
        # Complete schema, but explicitly not evidence of model recognition.
        return tuple(
            IndependentVerdict(label, (), False, None, (), ("scene_only",))
            for label in job.request.event_labels
        )

    def close(self):
        pass


def simulated_factory(config):
    return SimulatedInspector(config)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def timing(values):
    return {
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def release_schedule(count, duration, sample_fps, observation_fps, mode):
    """Shift each source clock, never its frame indices, rate or total duration."""
    if mode not in {"aligned", "staggered"}:
        raise ValueError("invalid arrival mode")
    offsets = [
        i / (count * observation_fps) if mode == "staggered" else 0.0
        for i in range(count)
    ]
    groups = {}
    for tick in range(math.ceil(duration * sample_fps)):
        # Keep the previous rotating offer order for simultaneous frames.
        for order in range(count):
            camera = (tick + order) % count
            ns = round((tick / sample_fps + offsets[camera]) * 1_000_000_000)
            groups.setdefault(ns, []).append((camera, tick))
    return offsets, [
        (ns / 1_000_000_000, actions) for ns, actions in sorted(groups.items())
    ]


def decode(path, sample_fps, max_side, quality):
    import av

    frames, due = [], 0.0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.time is None or frame.time + 1e-9 < due:
                continue
            image = frame.to_image()
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            frames.append(buffer.getvalue())
            due += 1 / sample_fps
    if len(frames) < 4:
        raise ValueError("video has fewer than four sample frames")
    return frames


def summarize(reports, stats, duration, expected, deadline):
    reports = [r for r in reports if r.get("vlm_lane") != "grounding"]
    good = [r for r in reports if r["state"] == "completed" and r["analysis_complete"]]
    on_time = [r for r in good if r["timing"]["source_to_publish_seconds"] <= deadline]
    errors = {}
    http_statuses = {}
    validation_reasons = {}
    response_diagnostics = {}
    for row in reports:
        if row["error_code"]:
            errors[row["error_code"]] = errors.get(row["error_code"], 0) + 1
        status = row["timing"].get("http_status")
        reason = row["timing"].get("validation_reason")
        if reason:
            validation_reasons[reason] = validation_reasons.get(reason, 0) + 1
        shape = row["timing"].get("response_diagnostics")
        if shape:
            key = json.dumps(shape, sort_keys=True)
            response_diagnostics[key] = response_diagnostics.get(key, 0) + 1
        if status is not None:
            http_statuses[str(status)] = http_statuses.get(str(status), 0) + 1
    valid_rows = len(good)
    judgments = [j for r in good for j in r["judgments"]]
    format_counts = {}
    for row in good:
        wrapper = row["timing"].get("response_diagnostics", {}).get("content_wrapper")
        if wrapper:
            format_counts[wrapper] = format_counts.get(wrapper, 0) + 1
    phase_metrics = {}
    for section in ("http_timing", "inspection_timing"):
        names = {name for row in reports for name in row["timing"].get(section, {})}
        for name in sorted(names):
            values = [
                row["timing"][section][name]
                for row in reports
                if name in row["timing"].get(section, {})
            ]
            phase_metrics[f"{section}.{name}"] = {
                "count": len(values),
                **timing(values),
            }
    return {
        "expected_observations": expected,
        "scheduled_observations": stats.get("scheduled_observation_windows"),
        "reports": len(reports),
        "complete_valid_results": valid_rows,
        "on_time_valid_results": len(on_time),
        "on_time_coverage": len(on_time) / expected if expected else None,
        "effective_results_per_second": valid_rows / duration,
        "uncertain_judgment_fraction": (
            sum(j["observed_detected"] is None for j in judgments) / len(judgments)
            if judgments
            else None
        ),
        "source_to_publish_seconds": timing(
            [r["timing"]["source_to_publish_seconds"] for r in good]
        ),
        "model_seconds": timing([r["timing"]["model_seconds"] for r in good]),
        "admission_wait_seconds": timing(
            [r["timing"].get("admission_wait_seconds", 0) for r in reports]
        ),
        "errors": errors,
        "http_status_counts": http_statuses,
        "validation_reason_counts": validation_reasons,
        "response_diagnostic_counts": response_diagnostics,
        "accepted_wrapper_counts": format_counts,
        "request_phase_metrics": phase_metrics,
        "process_stats": stats,
    }


def run_stage(args, base, count, framesets, output, repetition):
    stage = output / f"streams-{count}-repeat-{repetition}"
    stage.mkdir()
    gate_settings = (
        AdmissionSettings(
            str((stage / "admission.sqlite").resolve()),
            args.global_limit,
            args.per_stream_limit,
            args.request_rate,
            args.queue_timeout,
        )
        if args.global_limit
        else None
    )
    gate = SharedAdmission(gate_settings) if gate_settings else None
    branches, reports, cursors = (
        [],
        [[] for _ in range(count)],
        [0] * count,
    )
    missing_records = 0
    raw = (stage / "reports.jsonl").open("w")
    status_log = (stage / "status.jsonl").open("w")

    def collect():
        nonlocal missing_records
        for i, branch in enumerate(branches):
            for record in branch.results.records_after(cursors[i]):
                missing_records += max(0, record.cursor - cursors[i] - 1)
                cursors[i] = record.cursor
                reports[i].append(record.payload)
                raw.write(
                    json.dumps({"camera": i + 1, "report": record.payload}) + "\n"
                )
        raw.flush()

    try:
        for i in range(count):
            cfg = VLMProcessConfig(
                f"load-{output.name}-{count}-{repetition}-{i}",
                f"camera{i + 1:02d}",
                1,
                LABELS,
                replace(
                    base.model_api,
                    max_retries=0,
                    timeout_seconds=base.runtime.scene_timeout_seconds,
                ),
                base.inference.model,
                api_key=environment_api_key(base.model_api) if args.real else "",
                parameters=(
                    dict(base.inference.request_parameters)
                    if args.real
                    else {"simulation_seconds": args.simulated_latency}
                ),
                options={
                    "max_tokens": base.runtime.scene_max_tokens,
                    "max_image_side": base.runtime.scene_max_image_side,
                    "jpeg_quality": base.runtime.scene_jpeg_quality,
                },
                sensitivities=tuple(EVENT_SENSITIVITY_DEFAULTS.items()),
                observation_fps=args.observation_fps,
                max_in_flight=args.local_concurrency,
                output_mode=args.output_mode,
                admission=gate_settings,
            )
            branch = IndependentVLMBranch(
                cfg, inspector_factory=_inspector if args.real else simulated_factory
            )
            branches.append(branch)
            branch.start()
        ready_deadline = time.monotonic() + 15
        while any(b.health()["state"] != "running" for b in branches):
            if time.monotonic() >= ready_deadline or any(
                b.health()["state"] == "failed" for b in branches
            ):
                raise RuntimeError("VLM branch startup failed")
            time.sleep(0.02)
        sample_fps = max(2.0, args.observation_fps * 2)
        ticks = math.ceil(args.duration * sample_fps)
        arrival_mode = getattr(args, "arrival_mode", "aligned")
        offsets, releases = release_schedule(
            count, args.duration, sample_fps, args.observation_fps, arrival_mode
        )
        release_lags, sender_missed, offer_rejected = [], 0, [0] * count
        missed_by_stream = [0] * count
        start = time.monotonic()
        next_status = start
        for release_at, actions in releases:
            due = start + release_at
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            lag = time.monotonic() - due
            release_lags.append(lag)
            if lag > 1 / sample_fps:
                sender_missed += 1
                for camera, _tick in actions:
                    missed_by_stream[camera] += 1
                # Do not hide a stalled sender with accelerated catchup.
                continue
            for i, tick in actions:
                frames = framesets[i % len(framesets)]
                if not branches[i].offer(
                    frames[tick % len(frames)], tick / sample_fps, tick
                ):
                    offer_rejected[i] += 1
            collect()
            if time.monotonic() >= next_status:
                status_log.write(
                    json.dumps(
                        {
                            "elapsed": time.monotonic() - start,
                            "streams": [b.stats() for b in branches],
                        }
                    )
                    + "\n"
                )
                status_log.flush()
                next_status += 1
        remaining = start + args.duration + max(offsets) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        for branch in branches:
            branch.finish_source()
        drain_deadline = time.monotonic() + 15
        while time.monotonic() < drain_deadline:
            collect()
            if all(b.stats()["state"] == "drained" for b in branches):
                break
            time.sleep(0.02)
        collect()
        first_endpoint = math.ceil(0.75 * sample_fps) / sample_fps
        last_endpoint = (ticks - 1) / sample_fps
        expected = max(
            0,
            1
            + math.floor(
                (last_endpoint - first_endpoint) * args.observation_fps + 1e-9
            ),
        )
        stats = [b.stats() for b in branches]
        result = {
            "streams": count,
            "event_labels": list(LABELS),
            "repeat": repetition,
            "measurement_kind": "real_model_capacity"
            if args.real
            else "simulated_scheduler_only",
            "duration_seconds": args.duration,
            "arrival_mode": arrival_mode,
            "source_phase_offsets_seconds": offsets,
            "input_schedule_span_seconds": args.duration + max(offsets),
            "observation_fps": args.observation_fps,
            "deadline_seconds": args.deadline,
            "sender_missed_samples": sender_missed,
            "sender_missed_samples_by_stream": missed_by_stream,
            "offer_rejected_by_stream": offer_rejected,
            "input_release_lag_seconds": timing(release_lags),
            "all_drained": all(s["state"] == "drained" for s in stats),
            "result_records_missing": missing_records,
            "input_timing_valid": sender_missed == 0
            and max(release_lags, default=0) < 0.25,
            "shared_admission": gate.snapshot() if gate else None,
            "per_stream": [
                summarize(rr, ss, args.duration, expected, args.deadline)
                for rr, ss in zip(reports, stats, strict=True)
            ],
        }
        (stage / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        raw.close()
        status_log.close()
        for branch in branches:
            branch.close()
        for branch in branches:
            branch.close(timeout_seconds=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--simulated-latency", type=float, default=1)
    parser.add_argument(
        "--output-mode", choices=["events", "full", "parallel"], default="events"
    )
    parser.add_argument(
        "--global-limit",
        type=int,
        default=16,
        help="0 disables shared admission for comparison",
    )
    parser.add_argument("--per-stream-limit", type=int, default=2)
    parser.add_argument("--local-concurrency", type=int, default=4)
    parser.add_argument("--request-rate", type=float, default=20)
    parser.add_argument("--queue-timeout", type=float, default=1)
    parser.add_argument("--observation-fps", type=float, default=1.25)
    parser.add_argument("--deadline", type=float, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (
        not 3 <= args.duration <= 3600
        or not 1 <= args.repeats <= 10
        or any(n < 1 or n > 12 for n in args.streams)
        or len(args.streams) != len(set(args.streams))
        or not 0 < args.observation_fps <= 3
        or not 1 <= args.local_concurrency <= 4
        or not 0 <= args.simulated_latency <= 20
        or not 0 < args.deadline <= 30
        or not 0 <= args.global_limit <= 128
    ):
        parser.error("invalid load-test bounds")
    planned = math.ceil(
        sum(args.streams)
        * args.duration
        * args.observation_fps
        * args.repeats
        * (2 if args.output_mode == "parallel" else 1)
    )
    print(
        f"Planned request upper estimate: {planned}; real model: {args.real}",
        flush=True,
    )
    if args.plan_only:
        return
    base = load_config(args.config)
    if args.real and (not args.video or not environment_api_key(base.model_api)):
        parser.error("--real requires --video and the configured provider API key")
    framesets = [
        decode(
            p,
            max(2.0, args.observation_fps * 2),
            base.runtime.scene_max_image_side,
            base.runtime.scene_jpeg_quality,
        )
        for p in args.video
    ]
    if not framesets:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), "gray").save(buffer, format="JPEG")
        framesets = [[buffer.getvalue()]]
    output = args.output or ROOT / "outputs/vlm-multistream" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "event_labels": list(LABELS),
        "model": base.inference.model if args.real else "SIMULATED",
        "provider": base.model_api.provider if args.real else None,
        "settings": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "video"
        },
        "videos": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in args.video
        ],
        "image_side": base.runtime.scene_max_image_side,
        "jpeg_quality": base.runtime.scene_jpeg_quality,
        "limits": [
            "Predecoded realtime replay excludes decoding, camera upload and SAM.",
            "Repeated sources test load, not distinct-camera recognition accuracy.",
            "Long runs loop videos; no event accuracy is computed.",
            "Shared SQLite is for one host/shared local container volume only.",
        ],
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/goodvision_harness/vlm_admission.py",
                ROOT / "src/goodvision_harness/vlm_independent.py",
                ROOT / "src/goodvision_harness/vlm_compact_model.py",
                ROOT / "src/goodvision_harness/event_sensitivity.py",
                ROOT / "src/goodvision_harness/config.py",
            ]
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for n in args.streams:
        for repetition in range(1, args.repeats + 1):
            print(f"Running {n} streams, repeat {repetition}", flush=True)
            result = run_stage(args, base, n, framesets, output, repetition)
            results.append(result)
            (output / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            coverage = [row["on_time_coverage"] for row in result["per_stream"]]
            print(
                f"Per-stream on-time coverage: {min(coverage):.1%}.."
                f"{max(coverage):.1%}; drained={result['all_drained']}; "
                f"input_timing_valid={result['input_timing_valid']}",
                flush=True,
            )
    print(output, flush=True)
    if any(not r["all_drained"] or not r["input_timing_valid"] for r in results):
        raise SystemExit("Invalid timing/drain: inspect reports before using this run")


if __name__ == "__main__":
    main()
