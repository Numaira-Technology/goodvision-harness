"""Bounded sequential probe with scene reference accounting and optional YOLO IDs."""

import json
import time
from collections import Counter
from pathlib import Path

from .contract import LABELS, OPTIONS, parse_choices, plan, questions
from .packet import digest, validate, write_json
from .ppe import experiment, merge_equipment, values_for_question


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(windows, rows, reference):
    refs = {r["window_id"]: r for r in reference["rows"]}
    complete = [r for r in rows if r["complete"]]
    counts = {label: Counter() for label in LABELS}
    for row in complete:
        for label in LABELS:
            truth = refs[row["window_id"]]["judgments"][label]["detected"]
            actual = row["decisions"][f"scene/{label}"]
            counts[label]["aligned"] += 1
            if truth is None:
                counts[label]["reference_uncertain"] += 1
            elif actual is None:
                counts[label]["model_uncertain_on_reference_certain"] += 1
            else:
                counts[label]["comparable"] += 1
                counts[label]["agree"] += actual == truth
                counts[label]["false_positive"] += actual and not truth
                counts[label]["false_negative"] += truth and not actual
    comparable = sum(c["comparable"] for c in counts.values())
    certain = sum(
        r["judgments"][label]["detected"] is not None
        for r in refs.values()
        for label in LABELS
    )
    latency = [r["probe_window_seconds"] for r in complete]
    coverage = {
        label: {
            str(value): sum(
                r["judgments"][label]["detected"] is value for r in refs.values()
            )
            for value in (True, False, None)
        }
        for label in LABELS
    }
    return {
        "planned_windows": len(windows),
        "attempted_windows": len(rows),
        "complete_windows": len(complete),
        "not_attempted_windows": len(windows) - len(rows),
        "complete_rate": len(complete) / len(windows),
        "within_3s_rate": sum(t <= 3 for t in latency) / len(windows),
        "latency_p50_seconds": percentile(latency, 0.5),
        "latency_p95_seconds": percentile(latency, 0.95),
        "astra_binary_comparable": comparable,
        "astra_agreement": (
            sum(c["agree"] for c in counts.values()) / comparable
            if comparable
            else None
        ),
        "decided_coverage_of_all_reference_certain": comparable / certain
        if certain
        else None,
        "per_label": {k: dict(v) for k, v in counts.items()},
        "reference_class_coverage": coverage,
        "person_accuracy": None,
        "errors": dict(Counter(r["error"] for r in rows if not r["complete"])),
        "scope": (
            "Small sequential probe, NOT twelve-stream throughput or per-camera SLA. "
            "Timer includes optional YOLO, preprocessing, model and validation; "
            "excludes model loading and initial packet validation. No warmup excluded. "
            "Astra uses scene-only binary comparable judgments, not person binding. "
            "Zero positive examples cannot establish recall for that class."
        ),
    }


def detect(window, images, detector_url):
    from tools.benchmark_runtime.person_detector import DetectorBroker

    # Sparse windows cannot establish continuity across omitted video frames.
    # Start a new bounded session per window, using the existing tracker unchanged.
    with DetectorBroker(detector_url) as broker:
        for i, (frame, image) in enumerate(zip(window["frames"], images, strict=True)):
            broker.submit(0, window["window_id"], i, frame["media_seconds"], image)
        return broker.lookup(window["window_id"], list(range(len(images))))


def execute(
    packet,
    output,
    cache_dir,
    *,
    scope="scene",
    detector_url=None,
    runtime_factory=None,
    ppe_variant="baseline",
    ppe_option_order="canonical",
    person_run=None,
):
    config = experiment(ppe_variant, ppe_option_order)
    if (
        scope not in {"scene", "person"}
        or (scope == "person" and bool(detector_url) == bool(person_run))
        or (scope == "scene" and (detector_url or person_run))
    ):
        raise ValueError(
            "person probe requires exactly one YOLO26-S URL or saved person run"
        )
    packet, output = Path(packet), Path(output)
    manifest, _policy, reference, images = validate(packet)
    saved, replay = {}, None
    if person_run:
        from .preview import saved_people

        saved, replay = saved_people(packet, person_run, manifest)
    if scope == "person" and detector_url:
        from tools.benchmark_runtime.person_detector import DetectorBroker

        DetectorBroker(detector_url).check_ready()
    windows = manifest["windows"]
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[2]
    from tools.vlm_benchmark.provenance import source_identity

    sources = [
        *Path(__file__).parent.glob("*.py"),
        source_root / "tools/benchmark_runtime/vlm_event_policy.py",
        source_root / "tools/benchmark_runtime/event_sensitivity.py",
        source_root / "tools/benchmark_runtime/person_detector.py",
        source_root / "tools/vlm_benchmark/behavior_reference.py",
    ]
    run_manifest = {
        **plan(),
        "execute": True,
        "state": "initializing",
        "probe_scope": scope,
        "ppe_experiment": config,
        "detector_replay": replay,
        "scene_questions_per_window": 7 if ppe_variant == "equipment" else 6,
        "person_questions": "five event judgments; "
        "PPE uses two observations in equipment variant",
        "source_identity": source_identity(source_root),
        "source_hashes": {str(p.relative_to(source_root)): digest(p) for p in sources},
        "input_hashes": {
            n: digest(packet / n)
            for n in ("input-manifest.json", "event-policy.json", "annotations.json")
        },
        "planned_window_ids": [w["window_id"] for w in windows],
        "warmup_windows": 0,
    }
    write_json(output / "manifest.json", run_manifest)
    rows = []

    def summary():
        value = summarize(windows, rows, reference)
        value.update(ppe_experiment=config, detector_replay=replay)
        if replay:
            value["scope"] += (
                " SAVED YOLO SNAPSHOTS: excludes live detection, "
                "not full pipeline latency."
            )
        return value

    try:
        if runtime_factory is None:
            from .runtime import VideoJevRuntime

            runtime_factory = VideoJevRuntime
        runtime = runtime_factory(cache_dir)
        run_manifest.update(state="running", runtime=runtime.metadata)
        write_json(output / "manifest.json", run_manifest)
        for window in windows:
            start = time.perf_counter()
            row = {
                "window_id": window["window_id"],
                "complete": False,
                "source_times": [f["media_seconds"] for f in window["frames"]],
            }
            try:
                snapshots = (
                    saved.get(window["window_id"])
                    if person_run
                    else (
                        detect(window, images[window["window_id"]], detector_url)
                        if scope == "person"
                        else None
                    )
                )
                row["detector_seconds"] = time.perf_counter() - start
                if snapshots is not None:
                    row["detector_snapshots"] = snapshots
                qs = questions(
                    window,
                    snapshots,
                    ppe_variant=ppe_variant,
                    ppe_option_order=ppe_option_order,
                )
                row["question_count"] = len(qs)
                row["option_semantics"] = {
                    q["id"]: values_for_question(q, OPTIONS) for q in qs
                }
                write_json(output / f"{window['window_id']}-request.json", qs)
                raw, timings = runtime.predict(
                    images[window["window_id"]], manifest["sampling_fps"], qs
                )
                try:
                    json.dumps({"raw": raw, "timings": timings}, allow_nan=False)
                except (ValueError, TypeError):
                    row["invalid_output_repr"] = repr(raw)[:12000]
                    raise ValueError("nonfinite or non-JSON model output") from None
                row.update(raw=raw, timings=timings)
                row["observations"] = parse_choices(raw, qs)
                row["decisions"] = merge_equipment(row["observations"])
                row["complete"] = True
            except Exception as exc:
                row["error"] = type(exc).__name__
                row["error_detail"] = str(exc)[:1200]
            row["probe_window_seconds"] = time.perf_counter() - start
            rows.append(row)
            write_json(output / f"{window['window_id']}-result.json", row)
            write_json(output / "summary.json", summary())
            if row.get("error") in {"OutOfMemoryError", "RuntimeError"}:
                raise RuntimeError("GPU/model failure; remaining windows not attempted")
        run_manifest["state"] = "finished"
    except BaseException as exc:
        run_manifest.update(
            state="failed",
            failure_type=type(exc).__name__,
            failure_detail=str(exc)[:1200],
        )
        raise
    finally:
        write_json(output / "manifest.json", run_manifest)
        write_json(output / "summary.json", summary())
    return summary()
