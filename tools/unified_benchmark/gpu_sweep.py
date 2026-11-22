"""GPU / precision sweep for the CURRENT VideoJev service, always against a baseline.

The frozen native benchmark (native_run.py) replays a historical service on one
RTX 4090 and must stay that way. This tool answers a different question: on the
current ``goodvision_harness.videojev_service``, how do the GPU model and the
weight precision (``--quantization``) change latency and answers?

``run`` sends the frozen bundle's own request bodies (fingerprints checked) to
an already started service: twelve video workers, at most one request in flight
per source, one attempt per unit, no retries. Failures stay in the denominator.

``compare`` puts a candidate run next to a baseline run of the same bundle,
split and service settings. The baseline is 4090 + BF16 unless stated
otherwise; a comparison without a baseline is not produced.

Outputs contain reference answers and per-question correctness: keep them on
the authorized evaluation machine, like the bundle itself.
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .adapters import parse_response, request_body
from .contract import digest, file_hash, score, write_new
from .latency import PERCENTILE_METHOD, nearest_rank

TIMEOUT_SECONDS = 90
STAGES = (
    "server_elapsed_seconds", "cpu_queue_seconds", "server_preprocess_seconds",
    "gpu_queue_seconds", "visual_encode_seconds", "decision_seconds", "ppe_seconds",
    "server_gpu_seconds",
)
# Service settings that must match between baseline and candidate. GPU and
# quantization are what a sweep varies; everything else is held fixed.
FIXED_SETTINGS = ("gpu_batch_size", "max_pixels", "queue_policy", "context")
FIXED_RUNTIME = ("model", "revision", "attention", "prefix_cache", "torch_cpu_threads")
BASELINE = {"gpu": "4090", "quantization": "bf16"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def http_post(url, body, timeout=TIMEOUT_SECONDS):
    request = urllib.request.Request(
        url.rstrip("/") + "/predict", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode(errors="replace")}


def http_health(url):
    with urllib.request.urlopen(url.rstrip("/") + "/", timeout=5) as response:
        return json.load(response)


def describe(health):
    """The parts of /health that identify the GPU, precision and fixed settings."""
    runtime = health.get("runtime") or {}
    quantization = (runtime.get("quantization") or {}).get("mode", "bf16")
    return {
        "gpu": runtime.get("gpu"),
        "gpu_total_bytes": runtime.get("gpu_total_bytes"),
        "quantization": quantization,
        "settings": {k: health.get(k) for k in FIXED_SETTINGS},
        "runtime": {k: runtime.get(k) for k in FIXED_RUNTIME},
        "versions": runtime.get("versions"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "gpu_allocated_bytes_after_load": runtime.get("gpu_allocated_bytes_after_load"),
    }


def run_units(plan, config, bundle, post, *, workers=12, clock=time.perf_counter):
    """One attempt per unit; ordered within a source, sources in parallel."""
    if config.get("backend") != "videojev":
        raise ValueError("gpu_sweep measures the VideoJev service only")
    groups = defaultdict(list)
    for unit in plan["units"]:
        groups[unit["source_id"]].append(unit)
    records = {}

    def one(unit):
        body = request_body(config, unit, bundle)[1]
        if digest(body) != unit["request_sha256"]:
            raise ValueError("frozen request differs: " + unit["id"])
        started = clock()
        try:
            status, response = post(body)
            error = None
        except Exception as exc:  # a failed send is a recorded outcome
            status, response, error = None, None, f"{type(exc).__name__}: {exc}"
        seconds = clock() - started
        answers = {}
        if error is None and status == 200:
            try:
                answers = parse_response(config, unit, response)
            except ValueError as exc:
                error = f"invalid response: {exc}"
        elif error is None:
            error = f"http {status}"
        timings = (response or {}).get("timings") if isinstance(response, dict) else None
        records[unit["id"]] = {
            "unit": unit["id"], "source_id": unit["source_id"], "http_status": status,
            "error": error, "client_seconds": seconds, "answers": answers,
            "complete": set(answers) == set(unit["targets"]),
            "timings": {k: v for k, v in (timings or {}).items() if k in STAGES},
        }

    def run(group):
        for unit in group:
            one(unit)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, groups.values()))
    return [records[u["id"]] for u in plan["units"]]


def percentiles(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    return {"n": len(values), "p50": nearest_rank(values, .5),
            "p95": nearest_rank(values, .95), "mean": statistics.fmean(values)}


def summarize(plan, config, records, wall_seconds):
    evidence = {r["unit"]: r["answers"] for r in records}
    scored = score(plan, evidence, config["kind"])
    good = [r for r in records if r["complete"]]
    return {
        "accuracy": scored["accuracy"], "correct": scored["correct"],
        "planned_questions": scored["planned_questions"],
        "incomplete_questions": scored["incomplete_questions"],
        "unknown_questions": scored["final_unknown_questions"],
        "per_event": {e: {k: m[k] for k in ("accuracy", "correct", "planned_questions")}
                      for e, m in scored["per_event"].items()},
        "units": len(records), "failed_units": len(records) - len(good),
        "latency_method": PERCENTILE_METHOD + "; successful units only",
        "client_seconds": percentiles([r["client_seconds"] for r in good]),
        "stages": {s: percentiles([r["timings"].get(s) for r in good]) for s in STAGES},
        "wall_seconds": wall_seconds,
        "units_per_second": len(good) / wall_seconds if wall_seconds else None,
        "questions": scored["questions"],
    }


def run(args):
    from .workflow import verify_bundle

    bundle = Path(args.bundle)
    bundle_digest = verify_bundle(bundle)
    plan = json.loads((bundle / "plans" / f"videojev-{args.split}.json").read_text())
    config = json.loads((bundle / "configs" / "videojev.json").read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    health = http_health(args.url)
    if health.get("ready") is not True:
        raise ValueError("service is not ready")
    started = time.perf_counter()
    records = run_units(plan, config, bundle, lambda body: http_post(args.url, body))
    wall = time.perf_counter() - started
    with (output / "units.jsonl").open("x") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    result = {
        "kind": "videojev-gpu-sweep", "label": args.label, "at": stamp(),
        "bundle_sha256": bundle_digest, "split": args.split,
        "plan_sha256": digest(plan), "config_sha256": digest(config),
        "service": describe(health), "health": health,
        "summary": summarize(plan, config, records, wall),
        "units_sha256": file_hash(output / "units.jsonl"),
    }
    write_new(output / "run.json", result)
    print(json.dumps({"output": str(output), "accuracy": result["summary"]["accuracy"],
                      "client_p50": result["summary"]["client_seconds"]["p50"],
                      "client_p95": result["summary"]["client_seconds"]["p95"]}))


def is_baseline(service):
    return BASELINE["gpu"] in (service.get("gpu") or "") and (
        service.get("quantization") == BASELINE["quantization"])


def compare_runs(baseline, candidate, base_units, cand_units, *, allow_other_baseline=False):
    for key in ("bundle_sha256", "split", "plan_sha256", "config_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"baseline and candidate differ in {key}")
    b, c = baseline["service"], candidate["service"]
    for key in ("settings", "runtime"):
        if b[key] != c[key]:
            raise ValueError(f"baseline and candidate service {key} differ")
    if not is_baseline(b) and not allow_other_baseline:
        raise ValueError("baseline must be RTX 4090 + bf16 (pass --allow-other-baseline to override)")
    bs, cs = baseline["summary"], candidate["summary"]
    base_q = {q["number"]: q for q in bs["questions"]}
    flipped = [
        {"number": q["number"], "event": q["event"], "baseline": base_q[q["number"]]["prediction"],
         "candidate": q["prediction"]}
        for q in cs["questions"] if q["prediction"] != base_q[q["number"]]["prediction"]
    ]
    base_by_unit = {r["unit"]: r for r in base_units}
    drift, answer_flips, close_calls = [], 0, 0
    for record in cand_units:
        other = base_by_unit.get(record["unit"])
        if other is None:
            continue
        for event, value in record["answers"].items():
            ref = other["answers"].get(event)
            if not ref:
                continue
            p, q = value["probabilities"], ref["probabilities"]
            drift.append(max(abs(x - y) for x, y in zip(p, q, strict=True)))
            answer_flips += value["answer"] != ref["answer"]
            top = sorted(q, reverse=True)
            close_calls += top[0] - top[1] < 0.1

    def ratio(stage_b, stage_c, key):
        x, y = stage_b.get(key), stage_c.get(key)
        return y / x if x and y is not None else None

    stages = {}
    for stage in ("client_seconds", *STAGES):
        sb = bs["client_seconds"] if stage == "client_seconds" else bs["stages"][stage]
        sc = cs["client_seconds"] if stage == "client_seconds" else cs["stages"][stage]
        stages[stage] = {"baseline": {k: sb[k] for k in ("p50", "p95")},
                         "candidate": {k: sc[k] for k in ("p50", "p95")},
                         "p50_ratio": ratio(sb, sc, "p50"), "p95_ratio": ratio(sb, sc, "p95")}
    return {
        "baseline": {"label": baseline["label"], "gpu": b["gpu"], "quantization": b["quantization"],
                     "is_default_baseline": is_baseline(b)},
        "candidate": {"label": candidate["label"], "gpu": c["gpu"], "quantization": c["quantization"]},
        "accuracy": {"baseline": bs["accuracy"], "candidate": cs["accuracy"],
                     "correct_delta": cs["correct"] - bs["correct"],
                     "questions_with_changed_prediction": len(flipped), "changed": flipped},
        "answers": {"unit_answers_changed": answer_flips,
                    "baseline_close_calls_top2_gap_lt_0.1": close_calls,
                    "max_probability_drift": percentiles(drift)},
        "failed_units": {"baseline": bs["failed_units"], "candidate": cs["failed_units"]},
        "throughput_units_per_second": {"baseline": bs["units_per_second"],
                                        "candidate": cs["units_per_second"]},
        "memory_after_load_bytes": {"baseline": b.get("gpu_allocated_bytes_after_load"),
                                    "candidate": c.get("gpu_allocated_bytes_after_load")},
        "latency": stages,
        "ratio_meaning": "candidate / baseline; below 1 is faster",
    }


def read_run(path):
    path = Path(path)
    result = json.loads((path / "run.json").read_text())
    if file_hash(path / "units.jsonl") != result["units_sha256"]:
        raise ValueError(f"units.jsonl differs from its run.json: {path}")
    units = [json.loads(line) for line in (path / "units.jsonl").read_text().splitlines()]
    return result, units


def compare(args):
    baseline, base_units = read_run(args.baseline)
    candidate, cand_units = read_run(args.candidate)
    result = compare_runs(baseline, candidate, base_units, cand_units,
                          allow_other_baseline=args.allow_other_baseline)
    write_new(Path(args.output), result)
    print(json.dumps({k: result[k] for k in ("baseline", "candidate", "accuracy")}, default=str)[:2000])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="measure one service configuration")
    r.add_argument("--bundle", required=True)
    r.add_argument("--split", choices=("validation", "test"), default="test")
    r.add_argument("--url", default="http://127.0.0.1:8093")
    r.add_argument("--label", required=True, help="e.g. 4090-bf16 (the baseline), l40s-fp8")
    r.add_argument("--output", required=True, help="new directory")
    c = sub.add_parser("compare", help="candidate against a baseline run")
    c.add_argument("--baseline", required=True, help="run directory, normally 4090-bf16")
    c.add_argument("--candidate", required=True)
    c.add_argument("--output", required=True, help="new JSON file")
    c.add_argument("--allow-other-baseline", action="store_true")
    args = parser.parse_args(argv)
    (run if args.command == "run" else compare)(args)


if __name__ == "__main__":
    main()
