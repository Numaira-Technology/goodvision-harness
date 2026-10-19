"""Twelve-stream fixed-evidence quality and queue comparison; plan-only by default.

Use production inspectors, raw scene aggregation and shared admission.
Pace twelve fixed-evidence sources without decoding or tracking. Never send
references to an inspector. A fresh output directory is required;
no implicit paid retries.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import time
from bisect import bisect_right
from datetime import datetime, timezone
from itertools import groupby, pairwise
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

from goodvision_harness.grounding import SAFETY_EVENT_LABELS
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.vlm_event_policy import annotation_policy
from goodvision_harness.vlm_independent import (
    IndependentEventState,
    SourceSample,
    VLMProcessConfig,
    utc_now,
)
from tools.vlm_benchmark.behavior_queue import run_stage
from tools.vlm_benchmark.behavior_report import render
from tools.vlm_benchmark.lock import acquire
from tools.vlm_benchmark.modes import ROOT, mode_inspector
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.routes import (
    DASHSCOPE_MODEL,
    model_name,
    read_api_key,
    validate_route,
)

DEFAULT_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
MODES = ("events", "grounding")


def read_json(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def packet_file(root, name):
    if not isinstance(name, str):
        raise ValueError("invalid packet path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in relative.parts
        or "\\" in name
    ):
        raise ValueError("packet paths must be portable relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("packet files must exist inside packet")
    return path


def load_packet(packet, annotations):
    """Require explicit provenance, policy values and ordered evidence windows."""
    packet = Path(packet).resolve()
    manifest = read_json(packet / "input-manifest.json")
    policy = read_json(packet / "event-policy.json")
    reference = read_json(annotations)
    if set(policy["event_labels"]) != set(SAFETY_EVENT_LABELS):
        raise ValueError("this comparison requires all six event labels")
    expected = annotation_policy(
        policy["event_labels"], policy["sensitivities"],
        profile="research-v1",
    )
    policy_values = {key: policy.get(key) for key in expected}
    if policy_values != expected:
        raise ValueError("packet policy differs from current production policy")
    source_video = manifest.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError("packet requires an explicit source video reference")
    if (
        reference.get("source_video") != source_video
        or reference.get("event_policy") != expected
        or manifest.get("policy_version") != expected["version"]
    ):
        raise ValueError("reference requires matching source and explicit policy values")
    policy = policy_values
    if (
        reference.get("human_reviewed") is not False
        or manifest.get("prediction_data_included") is not False
        or manifest.get("previous_annotations_included") is not False
        or not reference.get("annotator_model")
    ):
        raise ValueError("independent AI reference provenance is required")
    if not 1 <= len(manifest["windows"]) <= 1000:
        raise ValueError("packet must have 1..1000 windows")
    if not finite(manifest.get("source_duration_seconds")):
        raise ValueError("invalid source duration")
    for key, low, high in (("image_side", 1, 2048), ("jpeg_quality", 1, 100)):
        if type(manifest.get(key)) is not int or not low <= manifest[key] <= high:
            raise ValueError("invalid image settings")
    windows, images = {}, {}
    evidence_times = set()
    for w in manifest["windows"]:
        wid = w["window_id"]
        if not isinstance(wid, str) or not re.fullmatch(r"W[0-9]{3,6}", wid):
            raise ValueError("invalid window id")
        if wid in windows or not 2 <= len(w["frames"]) <= 4:
            raise ValueError("duplicate window or invalid frame count")
        times, payloads = [], []
        for frame in w["frames"]:
            at = frame["media_seconds"]
            if not finite(at) or not 0 <= at < manifest["source_duration_seconds"]:
                raise ValueError("invalid evidence timestamp")
            path = packet_file(packet, frame["path"])
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("frame exceeds memory limit")
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                if im.format != "JPEG" or max(im.size) > manifest["image_side"]:
                    raise ValueError("invalid evidence image format or dimensions")
                im.verify()
            times.append(at)
            payloads.append(data)
        if (
            any(b <= a for a, b in pairwise(times))
            or not finite(w["judgment_time_seconds"])
            or w["judgment_time_seconds"] != times[-1]
        ):
            raise ValueError("window must be continuous and judged at last frame")
        if tuple(times) in evidence_times:
            raise ValueError("deduplicate evidence windows before comparison")
        evidence_times.add(tuple(times))
        windows[wid], images[wid] = w, payloads
    references = {}
    for row in reference["rows"]:
        wid = row["window_id"]
        if wid not in windows or wid in references:
            raise ValueError("reference has unknown or duplicate window")
        w = windows[wid]
        reference_times = row.get("evidence_timestamps_seconds")
        if (
            not finite(row["judgment_time_seconds"])
            or not isinstance(reference_times, list)
            or not all(finite(t) for t in reference_times)
            or row["judgment_time_seconds"] != w["judgment_time_seconds"]
            or reference_times != [f["media_seconds"] for f in w["frames"]]
            or set(row["judgments"]) != set(policy["event_labels"])
        ):
            raise ValueError("reference labels or timestamps do not match evidence")
        for verdict in row["judgments"].values():
            if type(verdict["detected"]) not in (bool, type(None)):
                raise ValueError("reference labels must be boolean or null")
            if not isinstance(verdict["reason"], str) or not verdict["reason"].strip():
                raise ValueError("reference requires visible-evidence reasons")
        references[wid] = row
    if (
        set(references) != set(windows)
        or set(reference.get("inspected_window_ids", [])) != set(windows)
        or reference.get("uninspected_window_ids") != []
    ):
        raise ValueError("reference does not cover every inspected window")
    return manifest, policy, reference, images


def schedule(manifest, repeats, streams=12, observation_fps=1.25, grounding_interval=0):
    # Sparse localization samples the newest available frozen observation at
    # each tick. Source time keeps advancing at 1x; never stretch 69 observations
    # into a longer video by merely changing the sender's sleep interval.
    windows = manifest["windows"]
    times = [w["judgment_time_seconds"] for w in windows] if grounding_interval else []
    selected = {}
    if grounding_interval:
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("source-clock sampling requires ordered windows")
        tick = 0
        while times[0] + tick * grounding_interval <= times[-1]:
            at = times[0] + tick * grounding_interval
            index = bisect_right(times, at) - 1
            selected.setdefault(windows[index]["window_id"], at)
            tick += 1
    planned = []
    for repeat in range(1, repeats + 1):
        for mode in MODES if repeat % 2 else tuple(reversed(MODES)):
            stage = []
            for i, w in enumerate(windows):
                sampling_selected = (
                    mode == "events"
                    or not grounding_interval
                    or w["window_id"] in selected
                )
                source_at = (
                    selected.get(w["window_id"], times[i])
                    if grounding_interval and mode == "grounding"
                    else times[i]
                    if grounding_interval
                    else i / observation_fps
                )
                offset = source_at - times[0] if grounding_interval else source_at
                for stream in range(1, streams + 1):
                    stage.append(
                        {
                            "request_id": (
                                f"r{repeat}-{mode}-s{stream:02}-{w['window_id']}"
                            ),
                            "repeat": repeat,
                            "mode": mode,
                            "stream_id": stream,
                            "window_id": w["window_id"],
                            "sampling_selected": sampling_selected,
                            "evidence_age_at_release_seconds": max(
                                0, source_at - times[i]
                            )
                            if grounding_interval
                            else 0,
                            "release_offset_seconds": offset
                            + (stream - 1) / streams / observation_fps,
                        }
                    )
            planned.extend(sorted(stage, key=lambda p: p["release_offset_seconds"]))
    return planned


def make_job(item, window, payloads, policy):
    session = f"quality-{item['repeat']}-{item['mode']}-s{item.get('stream_id', 1)}"
    frames = tuple(
        EvidenceFrame(session, i, f["media_seconds"], data)
        for i, (f, data) in enumerate(zip(window["frames"], payloads, strict=True))
    )
    request = InspectionRequest(
        item["request_id"],
        (),
        "fixed_evidence_quality",
        tuple(policy["event_labels"]),
        frames[0].timestamp_seconds,
        frames[-1].timestamp_seconds,
    )
    return TemporalInspection(
        session,
        "frozen-video",
        request,
        frames,
        event_sensitivities=tuple(policy["sensitivities"].items()),
    )


def make_config(item, manifest, policy, args, key=""):
    provider = {
        "sort": "latency",
        "require_parameters": True,
        "allow_fallbacks": not bool(args.provider_only),
    }
    if args.provider_only:
        provider["only"] = [args.provider_only]
    return VLMProcessConfig(
        f"quality-{item['repeat']}-{item['mode']}-s{item.get('stream_id', 1)}",
        "frozen-video",
        1,
        tuple(policy["event_labels"]),
        ModelAPISettings(
            provider=args.backend,
            endpoint=args.endpoint,
            max_retries=0,
            timeout_seconds=args.http_timeout,
        ),
        model_name(args.backend, args.model),
        api_key=key,
        parameters={
            "temperature": 0,
            **({"provider": provider} if args.backend == "openrouter" else {}),
        },
        options={
            "max_tokens": 1600,
            "max_image_side": manifest["image_side"],
            "jpeg_quality": manifest["jpeg_quality"],
        },
        sensitivities=tuple(policy["sensitivities"].items()),
        max_in_flight=args.per_stream_limit,
        output_mode="events" if item["mode"] == "events" else "compact",
        grounding_format=args.grounding_format,
        event_policy="research-v1",
        retain_historical_results=True,
    )


class SimulatedInspector:
    """Exercise the real parser; synthetic output never reads reference labels."""

    def __init__(self, config, latency=0):
        self.adapter = mode_inspector(config)
        self.latency = latency
        self.last_response_diagnostics = {"provider_category": "simulation"}

    def inspect(self, job):
        time.sleep(self.latency)
        payload = (
            dict.fromkeys(job.request.event_labels, False)
            if self.adapter.__class__.__name__ == "EventJSONInspector"
            else {"no_events": 0.5}
        )
        if self.adapter.__class__.__name__ == "LeanGroundingInspector":
            payload = {
                "v": 1,
                "s": [],
                "e": {
                    label: [{"p": [], "d": False, "c": 0.5}]
                    for label in job.request.event_labels
                },
            }
        return self.adapter._parse(
            job,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(payload)},
                    }
                ]
            },
        )

    def close(self):
        self.adapter.close()


def evaluate(item, job, config, inspector):
    start = time.monotonic()
    error, status, verdicts = None, None, ()
    try:
        verdicts = inspector.inspect(job)
    except KeyboardInterrupt:
        error = "interrupted"
    except ModelAPIError as exc:
        error, status = exc.code, exc.status_code
    except Exception:
        # Exception strings may contain provider secrets; preserve safe codes only.
        error = "client_error"
    elapsed = time.monotonic() - start
    report = IndependentEventState(config).report(
        job,
        SourceSample(job.frames[-1], utc_now(), start),
        verdicts,
        {"model_seconds": elapsed},
        error=error,
    )
    return {
        **item,
        "requested_model": config.model,
        "state": "failed"
        if error
        else "complete"
        if report["analysis_complete"]
        else "incomplete",
        "error_code": error,
        "http_status": status,
        "inspect_seconds": elapsed,
        "judgments": {
            v["event_label"]: v["observed_detected"] for v in report["judgments"]
        },
        "report": report,
        "response_diagnostics": getattr(inspector, "last_response_diagnostics", {}),
        "http_timing": getattr(inspector, "last_http_timing", {}),
        "inspection_timing": getattr(inspector, "last_inspection_timing", {}),
    }


def key_from_environment(env_file, backend="openrouter"):
    return read_api_key(backend, env_file)


def run(args, *, inspector_factory=None):
    validate_route(args.backend, args.endpoint, args.provider_only)
    args.model = model_name(args.backend, args.model)
    if not 1 <= args.repeats <= 3 or not 1 <= args.max_requests <= 6000:
        raise ValueError("invalid repeats or request budget")
    if not finite(args.http_timeout) or not 1 <= args.http_timeout <= 20:
        raise ValueError("HTTP timeout must be within 1..20 seconds")
    if args.model != (
        DASHSCOPE_MODEL if args.backend == "dashscope" else DEFAULT_MODEL
    ):
        raise ValueError("this comparison requires Qwen3-VL-30B-A3B-Instruct")
    if args.real and args.simulate:
        raise ValueError("real and simulated execution are mutually exclusive")
    if args.streams != 12:
        raise ValueError("formal comparison must use twelve streams")
    if not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("select unique benchmark modes")
    if args.grounding_interval not in (0, 4, 5):
        raise ValueError("grounding interval must be 0, 4 or 5 seconds")
    if args.grounding_interval and args.observation_fps != 1.25:
        raise ValueError("sparse grounding keeps the existing events source cadence")
    for name, low, high in (
        ("observation_fps", 0.25, 3),
        ("request_rate", 1, 200),
        ("queue_timeout", 0.01, 3),
        ("simulation_time_scale", 0.1, 1),
        ("simulated_latency", 0, 20),
        ("input_lag_tolerance", 0.01, 0.5),
    ):
        if not finite(getattr(args, name)) or not low <= getattr(args, name) <= high:
            raise ValueError(f"invalid {name}")
    if not 1 <= args.global_limit <= 48 or not 1 <= args.per_stream_limit <= min(
        4, args.global_limit
    ):
        raise ValueError("invalid twelve-stream concurrency")
    if args.real and args.simulation_time_scale != 1:
        raise ValueError("real execution cannot accelerate the source clock")
    if args.request_rate / (args.simulation_time_scale if args.simulate else 1) > 200:
        raise ValueError("scaled admission rate exceeds production limit")
    annotations = args.annotations or args.packet / "annotations.json"
    manifest, policy, reference, images = load_packet(args.packet, annotations)
    planned = schedule(
        manifest,
        args.repeats,
        args.streams,
        args.observation_fps,
        args.grounding_interval,
    )
    planned = [item for item in planned if item["mode"] in args.modes]
    selected_requests = sum(p["sampling_selected"] for p in planned)
    if selected_requests > args.max_requests:
        raise ValueError("planned calls exceed explicit maximum request budget")
    key = key_from_environment(args.env_file, args.backend) if args.real else ""
    output = args.output or ROOT / "outputs/vlm-behavior-compare" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    # Freeze only validated input files, never copy neighboring files or credentials.
    inputs = output / "inputs"
    inputs.mkdir()
    write_json(inputs / "input-manifest.json", manifest)
    write_json(inputs / "event-policy.json", policy)
    write_json(inputs / "annotations.json", reference)
    for w in manifest["windows"]:
        for frame, data in zip(w["frames"], images[w["window_id"]], strict=True):
            target = inputs / frame["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    mode_settings = {}
    for mode in args.modes:
        cfg = make_config({"mode": mode, "repeat": 1}, manifest, policy, args)
        adapter = mode_inspector(cfg)
        try:
            mode_settings[mode] = {
                "adapter": type(adapter).__name__,
                "timeout_seconds": adapter.settings.timeout_seconds,
                "parameters": dict(adapter._parameters),
            }
        finally:
            adapter.close()
    doc = {
        "schema_version": 1,
        "measurement_kind": "real_model_reference_agreement"
        if args.real
        else "simulation_not_model_quality"
        if args.simulate
        else "plan_only",
        "state": "prepared",
        "model": args.model,
        "provider": args.backend,
        "endpoint": ModelAPISettings(
            provider=args.backend, endpoint=args.endpoint
        ).endpoint,
        "streams": 12,
        "modes": list(args.modes),
        "single_stream": False,
        "global_limit": args.global_limit,
        "per_stream_limit": args.per_stream_limit,
        "request_rate": args.request_rate,
        "observation_fps": args.observation_fps,
        "grounding_interval_seconds": args.grounding_interval or None,
        "grounding_format": args.grounding_format,
        "source_clock_mode": "original_timestamps_1x"
        if args.grounding_interval
        else "fixed_cadence",
        "queue_timeout_seconds": args.queue_timeout,
        "input_lag_tolerance_seconds": args.input_lag_tolerance,
        "simulation_time_scale": args.simulation_time_scale if args.simulate else 1,
        "provider_only": args.provider_only,
        "provider_routing": "direct" if args.backend == "dashscope" else "latency",
        "provider_fallbacks": args.backend == "openrouter"
        and not bool(args.provider_only),
        "model_fallbacks": False,
        "retries": 0,
        "continue_on_timeout": args.continue_on_timeout,
        "publication_collection": "incremental_including_stop_drain",
        "dns_transport": "tcp"
        if "use-vc" in os.environ.get("RES_OPTIONS", "").split()
        else "system_default",
        "queue_mode": "paced_sources_latest_pending_production_shared_admission",
        "latency_scope": (
            "source-window ingress through queue, inspect and publication; "
            "excludes decoding and display"
        ),
        "source_video": manifest["source_video"],
        "event_policy": policy,
        "retain_historical_results": True,
        "reference_file": "inputs/annotations.json",
        "original_reference_file": str(annotations),
        "reference_model": reference["annotator_model"],
        "reference_human_reviewed": False,
        "unique_windows": len(manifest["windows"]),
        "repeats": args.repeats,
        "planned_requests": selected_requests,
        "reference_window_slots": len(planned),
        "mode_settings": mode_settings,
        "source_files": [
            p.relative_to(ROOT).as_posix()
            for p in [
                Path(__file__),
                *sorted((ROOT / "src/goodvision_harness").rglob("*.py")),
                ROOT / "tools/vlm_benchmark/behavior_report.py",
                ROOT / "tools/vlm_benchmark/behavior_queue.py",
                ROOT / "tools/vlm_benchmark/routes.py",
                ROOT / "src/goodvision_harness/vlm_admission.py",
                ROOT / "src/goodvision_harness/vlm_event_policy.py",
                ROOT / "src/goodvision_harness/vlm_independent.py",
                ROOT / "src/goodvision_harness/vlm_independent_model.py",
                ROOT / "src/goodvision_harness/vlm_compact_model.py",
                ROOT / "src/goodvision_harness/vlm_lean_grounding.py",
                ROOT / "src/goodvision_harness/scene_model.py",
                ROOT / "src/goodvision_harness/scene_inspection.py",
            ]
        ],
        "planned": planned,
    }
    records = []
    write_json(output / "manifest.json", doc)
    if not (args.real or args.simulate):
        render(output, doc, manifest, reference, records)
        return output
    factory = inspector_factory or (
        mode_inspector
        if args.real
        else lambda cfg: SimulatedInspector(
            cfg, args.simulated_latency * args.simulation_time_scale
        )
    )
    lock, stop_reason = None, None
    doc["stages"] = []
    try:
        if args.real:
            (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
            lock = acquire(ROOT / "outputs/vlm-modes/run")
        doc["state"] = "running"
        windows = {w["window_id"]: w for w in manifest["windows"]}
        with (output / "results.jsonl").open("x", encoding="utf-8") as journal:

            def collect(result):
                records.append(result)
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()

            for stage_key, items in groupby(
                planned, key=lambda p: (p["repeat"], p["mode"])
            ):
                if stop_reason:
                    break
                database = output / f"admission-r{stage_key[0]}-{stage_key[1]}.sqlite3"
                result = run_stage(
                    list(items),
                    args,
                    database,
                    lambda item: make_config(item, manifest, policy, args, key),
                    lambda item: make_job(
                        item,
                        windows[item["window_id"]],
                        images[item["window_id"]],
                        policy,
                    ),
                    factory,
                    evaluate,
                    collect,
                )
                doc["stages"].append(result)
                stop_reason = result["stop_reason"]
                write_json(output / "manifest.json", doc)
                print(
                    f"{stage_key[1]} repeat {stage_key[0]}: "
                    f"{len(records)}/{len(planned)} window outcomes recorded",
                    flush=True,
                )
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception:
        stop_reason = "runner_error"
    finally:
        if lock is not None:
            os.close(lock)
        doc["state"] = "stopped" if stop_reason else "completed"
        doc["stop_reason"] = stop_reason
        doc["recorded_windows"] = len(records)
        doc["recorded_requests"] = sum(r.get("attempted", False) for r in records)
        write_json(output / "manifest.json", doc)
        render(output, doc, manifest, reference, records)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--backend", choices=["openrouter", "dashscope"], default="openrouter"
    )
    p.add_argument("--endpoint", default="", help="DashScope full Chat Completions URL")
    p.add_argument(
        "--model", choices=[DEFAULT_MODEL, DASHSCOPE_MODEL], default=DEFAULT_MODEL
    )
    p.add_argument(
        "--provider-only",
        choices=["deepinfra", "novita", "siliconflow"],
        help="Pin an OpenRouter provider with fallbacks disabled; default is automatic",
    )
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-requests", type=int, default=3500)
    p.add_argument("--streams", type=int, choices=[12], default=12)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument(
        "--grounding-format",
        choices=["legacy", "lean-v1"],
        default="legacy",
        help="Short grounding output with event and localization checks",
    )
    p.add_argument("--global-limit", type=int, default=48)
    p.add_argument("--per-stream-limit", type=int, default=4)
    p.add_argument("--request-rate", type=float, default=20)
    p.add_argument("--observation-fps", type=float, default=1.25)
    p.add_argument(
        "--grounding-interval",
        type=int,
        choices=[0, 4, 5],
        default=0,
        help=(
            "Sample grounding every 4 or 5 source seconds at 1x; "
            "0 keeps every observation"
        ),
    )
    p.add_argument("--queue-timeout", type=float, default=1)
    p.add_argument("--input-lag-tolerance", type=float, default=0.1)
    p.add_argument("--simulated-latency", type=float, default=1.5)
    p.add_argument("--simulation-time-scale", type=float, default=1)
    p.add_argument("--http-timeout", type=float, default=20)
    p.add_argument(
        "--continue-on-timeout",
        action="store_true",
        help=(
            "Record request timeouts as failures and continue the bounded plan; "
            "authentication, billing, rate-limit and other stop rules still apply"
        ),
    )
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--real", action="store_true")
    execution.add_argument("--simulate", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        output = run(args)
    except (ValueError, KeyError, TypeError, OSError):
        p.exit(
            2,
            "Input validation failed; check packet, reference, budget, "
            "credential availability and unused output directory.\n",
        )
    print(f"Report: {output / 'report.html'}")
    manifest = read_json(output / "manifest.json")
    if manifest["state"] == "stopped":
        p.exit(1, f"Run stopped: {manifest['stop_reason']}\n")
