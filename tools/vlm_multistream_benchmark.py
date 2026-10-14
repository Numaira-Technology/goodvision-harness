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
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from goodvision_harness.config import load_config  # noqa: E402
from goodvision_harness.event_sensitivity import (  # noqa: E402
    EVENT_SENSITIVITY_DEFAULTS,
)
from goodvision_harness.grounding import SAFETY_EVENT_LABELS  # noqa: E402
from goodvision_harness.model_api import environment_api_key  # noqa: E402
from goodvision_harness.vlm_admission import (  # noqa: E402
    MAX_PER_STREAM_CONCURRENCY,
    AdmissionSettings,
    SharedAdmission,
)
from goodvision_harness.vlm_independent import (  # noqa: E402
    INFORMATION_HTTP_DEADLINE_SECONDS,
    IndependentVLMBranch,
    VLMProcessConfig,
    _inspector,
)
from goodvision_harness.vlm_independent_model import IndependentVerdict  # noqa: E402
from tools.vlm_benchmark.accounting import (  # noqa: E402
    delivery_counts,
    reconcile_retention,
)
from tools.vlm_benchmark.source_feeder import TimedSourceFeeder  # noqa: E402

# Follow the production safety catalog; stranger uses a separate identity lane.
LABELS = SAFETY_EVENT_LABELS
DEMO_PERIOD_SECONDS = 14.0
DEMO_POSITIVE_WINDOW_SECONDS = 5.0


def preview_demo_requested():
    """Keep the upstream environment opt-in; ordinary simulation stays negative."""
    return os.environ.get("PREVIEW_DEMO_EVENTS", "").strip().lower() not in {
        "", "0", "false", "no", "off",
    }


def preview_demo_offset(source_id):
    # Stable across processes, PYTHONHASHSEED values, machines and reruns.
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1000 / 1000 * DEMO_PERIOD_SECONDS


def preview_demo_settings(real, requested=None, streams=0):
    requested = preview_demo_requested() if requested is None else requested
    if type(requested) is not bool:
        raise ValueError("preview demo selection must be boolean")
    return {
        "requested": requested,
        "enabled": requested and not real,
        "algorithm": "source-time-sha256-v1",
        "period_seconds": DEMO_PERIOD_SECONDS,
        "positive_window_seconds": DEMO_POSITIVE_WINDOW_SECONDS,
        "clock": "last_evidence_frame_source_seconds",
        "phase_hash": "SHA-256(source_id UTF-8), first 8 bytes big-endian modulo 1000",
        "source_phase_offsets_seconds": {
            f"camera{i + 1:02d}": preview_demo_offset(f"camera{i + 1:02d}")
            for i in range(streams)
        },
        "label_order": list(LABELS),
        "meaning": (
            "Scripted synthetic preview verdicts; no model recognition or inference."
        ),
    }


class SimulatedInspector:
    def __init__(self, config):
        self.last_http_timing = {"http_request_attempted": False}
        self.seconds = (
            config.parameters.get(
                "simulation_grounding_seconds", config.parameters["simulation_seconds"]
            )
            if config.output_mode == "compact"
            else config.parameters["simulation_seconds"]
        )
        # Only this simulator implements the opt-in preview demo. The stage
        # freezes its flag into config so an inherited environment change cannot
        # make the effective experiment differ from the saved manifest.
        self.demo_events = config.parameters.get(
            "simulation_demo_events", preview_demo_requested(),
        )
        if type(self.demo_events) is not bool:
            raise ValueError("preview demo selection must be boolean")
        self.demo_offset = preview_demo_offset(config.source_id)

    def _demo_observed_label(self, labels, source_seconds):
        if not (self.demo_events and labels):
            return None
        clock = source_seconds + self.demo_offset
        if clock % DEMO_PERIOD_SECONDS >= DEMO_POSITIVE_WINDOW_SECONDS:
            return None
        return labels[int(clock // DEMO_PERIOD_SECONDS) % len(labels)]

    def inspect(self, job):
        time.sleep(self.seconds)
        labels = list(job.request.event_labels)
        observed = self._demo_observed_label(
            labels, job.frames[-1].timestamp_seconds,
        )
        # Complete schema. Without PREVIEW_DEMO_EVENTS this is explicitly not
        # evidence of model recognition (every verdict False), as before.
        return tuple(
            IndependentVerdict(
                label, (),
                True if label == observed else False,
                0.9 if label == observed else None,
                (), ("scene_only",),
            )
            for label in labels
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


def planned_observations(duration, observation_fps):
    """Count integer observation slots on the same source clock as run_stage."""
    sample_fps = max(2.0, observation_fps * 2)
    ticks = math.ceil(duration * sample_fps)
    first = math.ceil(0.75 * sample_fps) / sample_fps
    last = (ticks - 1) / sample_fps
    return max(0, 1 + math.floor((last - first) * observation_fps + 1e-9))


class DecodedFrames(list):
    """JPEG samples plus exact media PTS for monitoring; list API is unchanged."""

    def __init__(self, frames, path, timestamps, duration):
        super().__init__(frames)
        self.source_path = Path(path).resolve()
        self.media_seconds = timestamps
        self.media_duration = duration


def decode(path, sample_fps, max_side, quality):
    import av

    frames, timestamps, due = [], [], 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0
        for frame in container.decode(video=0):
            if frame.time is None:
                continue
            at = max(0.0, frame.time - origin)
            if at + 1e-9 < due:
                continue
            image = frame.to_image()
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            frames.append(buffer.getvalue())
            timestamps.append(at)
            due += 1 / sample_fps
    if len(frames) < 4:
        raise ValueError("video has fewer than four sample frames")
    return DecodedFrames(
        frames, path, timestamps, duration or timestamps[-1] + 1 / sample_fps
    )


def report_lane(report):
    """Standalone compact/full reports use the production 'combined' label."""
    return (
        "grounding" if report.get("vlm_lane") in {"grounding", "combined"} else "events"
    )


def summarize(reports, stats, duration, expected, deadline, *, lane=None):
    # Preserve the historical primary-lane view for existing API reports.
    reports = [
        r
        for r in reports
        if (report_lane(r) == lane if lane else r.get("vlm_lane") != "grounding")
    ]
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
    for name in (
        "source_to_request_seconds",
        "dispatch_wait_seconds",
        "reorder_wait_seconds",
    ):
        values = [r["timing"][name] for r in reports if name in r["timing"]]
        phase_metrics[name] = {"count": len(values), **timing(values)}
    # Disjoint residual outside inspect(), including publication and IPC;
    # it is not pure JSON parsing or pure GPU time.
    residuals = [
        max(
            0,
            r["timing"]["source_to_publish_seconds"]
            - r["timing"]["source_to_request_seconds"]
            - r["timing"]["model_seconds"],
        )
        for r in reports
        if "source_to_request_seconds" in r["timing"]
    ]
    phase_metrics["after_inspect_to_publish_seconds"] = {
        "count": len(residuals),
        **timing(residuals),
    }
    latencies = [r["timing"]["source_to_publish_seconds"] for r in good]
    return {
        "expected_observations": expected,
        "scheduled_observations": stats.get("scheduled_observation_windows"),
        "reports": len(reports),
        "complete_valid_results": valid_rows,
        "complete_result_fraction": valid_rows / expected if expected else None,
        "without_complete_result": max(0, expected - valid_rows),
        "late_complete_results": valid_rows - len(on_time),
        "within_seconds": {
            str(limit): {
                "count": sum(v <= limit for v in latencies),
                "fraction_of_planned": (
                    sum(v <= limit for v in latencies) / expected if expected else None
                ),
            }
            for limit in (1, 2, 3, 5)
        },
        "localization_complete_results": sum(
            r.get("localization_complete") is True for r in good
        ),
        "all_negative_results": sum(
            bool(r["judgments"])
            and all(j["observed_detected"] is False for j in r["judgments"])
            for r in good
        ),
        "historical_only_results": sum(r.get("historical_only", False) for r in good),
        "delivery_accounting": delivery_counts(reports, expected),
        "source_to_completion_seconds": timing(
            [r["timing"]["source_to_completion_seconds"] for r in good
             if "source_to_completion_seconds" in r["timing"]]
        ),
        "completion_to_publish_seconds": timing(
            [r["timing"]["completion_to_publish_seconds"] for r in good
             if "completion_to_publish_seconds" in r["timing"]]
        ),
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
            [
                r["timing"]["admission_wait_seconds"]
                for r in reports
                if "admission_wait_seconds" in r["timing"]
            ]
        ),
        "errors": errors,
        "http_status_counts": http_statuses,
        "validation_reason_counts": validation_reasons,
        "response_diagnostic_counts": response_diagnostics,
        "accepted_wrapper_counts": format_counts,
        "request_phase_metrics": phase_metrics,
        "process_stats": stats,
    }


def paired_summary(reports_by_camera, expected):
    """Pair only the exact same production window, never nearest timestamps."""
    pairs = {}
    unpaired_records = duplicates = unknown_http = 0
    for camera, reports in enumerate(reports_by_camera):
        for r in reports:
            unknown_http += (
                type(r.get("timing", {}).get("http_request_attempted")) is not bool
            )
            window = r.get("window_id")
            if window is not None:
                pair = pairs.setdefault((camera, window), {})
                lane = report_lane(r)
                duplicates += lane in pair
                pair[lane] = r
            else:
                unpaired_records += 1

    def complete(r):
        return r and r["state"] == "completed" and r["analysis_complete"]

    def attempted(row, key="http_request_attempted"):
        return (row or {}).get("timing", {}).get(key) is True

    both, event_only, grounding_only, differences, full_latency = 0, 0, 0, [], []
    both_sent = one_sent = both_inspected = 0
    both_sent_complete = 0
    complete_peer_unsent = complete_peer_failed = 0
    for pair in pairs.values():
        event, ground = pair.get("events"), pair.get("grounding")
        sent = int(attempted(event)) + int(attempted(ground))
        both_sent += sent == 2
        one_sent += sent == 1
        both_inspected += attempted(event, "inspection_attempted") and attempted(
            ground, "inspection_attempted"
        )
        if complete(event) and complete(ground):
            both += 1
            both_sent_complete += sent == 2
            e = event["timing"]["source_to_publish_seconds"]
            g = ground["timing"]["source_to_publish_seconds"]
            differences.append(g - e)
            full_latency.append(max(g, e))
        elif complete(event):
            event_only += 1
        elif complete(ground):
            grounding_only += 1
        if bool(complete(event)) != bool(complete(ground)):
            peer = ground if complete(event) else event
            complete_peer_failed += attempted(peer)
            complete_peer_unsent += peer is None or (
                peer.get("timing", {}).get("http_request_attempted") is False
            )
    known = not (
        unknown_http or unpaired_records or duplicates or len(pairs) > expected
    )
    return {
        "expected_source_windows": expected,
        "both_complete": both,
        "both_complete_fraction": both / expected if expected else None,
        "without_both_complete": max(0, expected - both),
        "events_complete_without_grounding": event_only,
        "grounding_complete_without_events": grounding_only,
        "http_pair_accounting_known": known,
        "http_attempt_unknown_records": unknown_http,
        "records_without_window_id": unpaired_records,
        "duplicate_window_lane_records": duplicates,
        "both_http_attempted": both_sent,
        "both_http_attempted_fraction": (
            both_sent / expected if expected and known else None
        ),
        "only_one_http_attempted": one_sent,
        "neither_http_attempted": expected - both_sent - one_sent if known else None,
        "both_inspection_attempted": both_inspected,
        "both_complete_fraction_of_both_http_attempted": (
            both_sent_complete / both_sent if both_sent and known else None
        ),
        "one_complete_peer_http_not_attempted": complete_peer_unsent,
        "one_complete_peer_http_attempted_without_complete": complete_peer_failed,
        "paired_admission_superseded": sum(
            "events" in pair
            and "grounding" in pair
            and any(
                r.get("error_code") == "admission_superseded" for r in pair.values()
            )
            for pair in pairs.values()
        ),
        "grounding_minus_events_seconds": timing(differences),
        "both_published_seconds": timing(full_latency),
        "pairing": "same camera and exact window_id; unmatched windows remain missing",
    }


def drain_budget_seconds(args, base):
    """Allow the final fresh request to expire and publish, without extending it."""
    if args.real:
        request_timeout = base.runtime.scene_timeout_seconds
        if args.output_mode == "events":
            request_timeout = min(request_timeout, INFORMATION_HTTP_DEADLINE_SECONDS)
    else:
        request_timeout = args.simulated_latency
        if args.output_mode in {"compact", "parallel"}:
            request_timeout = max(
                request_timeout,
                getattr(args, "simulated_grounding_latency", request_timeout),
            )
    # Production admission can start a request until its evidence is 3 s old.
    # Then allow its existing HTTP deadline plus 2 s for result/status delivery.
    return max(15.0, request_timeout + 3.0 + 2.0)


def run_stage(
    args,
    base,
    count,
    framesets,
    output,
    repetition,
    *,
    inspector_factory=None,
    observer=None,
    source_wait=None,
):
    api_key = ""
    if args.real:
        if getattr(args, "backend", "") in {"openrouter", "dashscope"}:
            # Match the modes preflight and remote launcher: a generic override
            # may belong to another API. Never pass an empty key to inspectors,
            # whose legacy fallback would otherwise read that generic override.
            api_key = os.environ.get(base.model_api.api_key_env, "").strip()
            if not api_key:
                raise ValueError("selected benchmark API credential is unavailable")
        else:
            api_key = environment_api_key(base.model_api)
    demo = preview_demo_settings(
        args.real, getattr(args, "preview_demo_events", None), count,
    )
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
        [[0, 0] for _ in range(count)],
    )
    missing_records = 0
    raw = (stage / "reports.jsonl").open("w")
    status_log = (stage / "status.jsonl").open("w")

    def collect():
        nonlocal missing_records
        for i, branch in enumerate(branches):
            buses = [branch.results]
            if args.output_mode == "parallel":
                buses.append(branch.grounding_results)
            for bus_index, bus in enumerate(buses):
                for record in bus.records_after(cursors[i][bus_index]):
                    missing_records += max(0, record.cursor - cursors[i][bus_index] - 1)
                    cursors[i][bus_index] = record.cursor
                    payload = dict(record.payload)
                    if not args.real:
                        payload["synthetic"] = True
                        payload["measurement_kind"] = "simulated_scheduler_only"
                        payload["preview_demo_events"] = demo
                        payload["model"] = {
                            **payload.get("model", {}),
                            "name": "SIMULATED", "is_simulated": True,
                        }
                    if args.output_mode == "full":
                        payload.setdefault("vlm_lane", "combined")
                    reports[i].append(payload)
                    raw.write(json.dumps({"camera": i + 1, "report": payload}) + "\n")
                    if observer is not None:
                        observer.result(i, payload)
        raw.flush()

    feeder = None
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
                api_key=api_key,
                parameters=(
                    dict(base.inference.request_parameters)
                    if args.real
                    else {
                        "simulation_seconds": args.simulated_latency,
                        "simulation_demo_events": demo["enabled"],
                        "simulation_grounding_seconds": getattr(
                            args, "simulated_grounding_latency", args.simulated_latency
                        ),
                    }
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
                grounding_format=getattr(args, "grounding_format", "legacy"),
                parallel_scheduling=getattr(args, "parallel_scheduling", "independent"),
                event_policy=getattr(args, "event_policy", "main-baseline"),
                retain_historical_results=True,
                admission=gate_settings,
            )
            branch = IndependentVLMBranch(
                cfg,
                inspector_factory=(inspector_factory or _inspector)
                if args.real
                else simulated_factory,
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
        arrival_mode = getattr(args, "arrival_mode", "aligned")
        offsets, releases = release_schedule(
            count, args.duration, sample_fps, args.observation_fps, arrival_mode
        )
        # Record startup before a short/virtual source can finish. Status is
        # collected separately and never enters the paced source release loop.
        status_log.write(json.dumps({
            "elapsed": 0.0, "streams": [b.stats() for b in branches],
        }) + "\n")
        status_log.flush()
        start = time.monotonic()
        if observer is not None:
            observer.source_started(start, offsets)
        feeder = TimedSourceFeeder(
            branches, framesets, releases, start, sample_fps,
            monotonic=time.monotonic, wait=source_wait,
        )
        feeder.start()
        next_status = start + 1
        while feeder.thread.is_alive():
            feeder.notify(observer)
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
            feeder.thread.join(timeout=0.01)
        feeder.check()
        feeder.notify(observer)
        release_lags = feeder.release_lags
        sender_missed = feeder.missed
        missed_by_stream = feeder.missed_by_stream
        offer_rejected = feeder.offer_rejected
        remaining = start + args.duration + max(offsets) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        for branch in branches:
            branch.finish_source()
        if observer is not None:
            observer.draining()
        drain_started = time.monotonic()
        drain_timeout = drain_budget_seconds(args, base)
        drain_deadline = drain_started + drain_timeout
        while time.monotonic() < drain_deadline:
            collect()
            if all(b.stats()["state"] == "drained" for b in branches):
                break
            time.sleep(0.02)
        collect()
        expected = planned_observations(args.duration, args.observation_fps)
        stats = [b.stats() for b in branches]
        lanes = (
            ("events", "grounding")
            if args.output_mode == "parallel"
            else ("events" if args.output_mode == "events" else "grounding",)
        )
        per_stream = []
        for rr, ss in zip(reports, stats, strict=True):
            row = summarize(rr, ss, args.duration, expected, args.deadline)
            row["lanes"] = {
                lane: summarize(
                    rr, ss, args.duration, expected, args.deadline, lane=lane
                )
                for lane in lanes
            }
            per_stream.append(row)
        result = {
            "streams": count,
            "event_labels": list(LABELS),
            "repeat": repetition,
            "output_mode": args.output_mode,
            "nominal_requests_per_second": count * args.observation_fps * len(lanes),
            "rate_limit_below_offered_load": bool(
                gate and args.request_rate < count * args.observation_fps * len(lanes)
            ),
            "admission_scope": (
                "per-stream limit shared by lanes; global limit/rate cover all lanes"
            ),
            "parallel_scheduling": getattr(args, "parallel_scheduling", "independent"),
            "measurement_kind": "real_model_capacity"
            if args.real
            else "simulated_scheduler_only",
            "preview_demo_events": demo,
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
            "drain_timeout_seconds": drain_timeout,
            "drain_elapsed_seconds": time.monotonic() - drain_started,
            "result_records_missing": missing_records,
            "result_retention": reconcile_retention(reports, stats, lanes, report_lane),
            "input_timing_valid": sender_missed == 0
            and max(release_lags, default=0) < 0.25,
            "shared_admission": gate.snapshot() if gate else None,
            "per_stream": per_stream,
            "lanes": {
                lane: summarize(
                    [r for rr in reports for r in rr],
                    {},
                    args.duration,
                    count * expected,
                    args.deadline,
                    lane=lane,
                )
                for lane in lanes
            },
            "paired_windows": paired_summary(reports, count * expected)
            if args.output_mode == "parallel"
            else None,
        }
        (stage / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        try:
            if feeder is not None:
                feeder.close()
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
        "--output-mode",
        choices=["events", "compact", "full", "parallel"],
        default="events",
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
    parser.add_argument("--event-policy", choices=["main-baseline", "research-v1"],
                        default="main-baseline")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (
        not 3 <= args.duration <= 3600
        or not 1 <= args.repeats <= 10
        or any(n < 1 or n > 12 for n in args.streams)
        or len(args.streams) != len(set(args.streams))
        or not 0 < args.observation_fps <= 3
        or not 1 <= args.local_concurrency <= MAX_PER_STREAM_CONCURRENCY
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
    args.preview_demo_events = preview_demo_requested()
    base = load_config(args.config)
    if args.real and (
        not args.video
        or (
            base.model_api.provider != "local"
            and not environment_api_key(base.model_api)
        )
    ):
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
        "preview_demo_events": preview_demo_settings(
            args.real, args.preview_demo_events, streams=max(args.streams),
        ),
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
        "retain_historical_results": True,
        "source_hashes": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *sorted((ROOT / "src/goodvision_harness").rglob("*.py")),
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
